# SPDX-License-Identifier: AGPL-3.0-only

"""按 user_id 隔离的记忆空间。

ALM 以 user_id 为唯一检索作用域，且 `MemoryGraph.to_dict()` 的 YAML 序列化
**不保留节点 metadata**，因此这里采用「每个 user_id 一份独立图谱 + 独立向量索引」
的隔离方式：不做任何核心改动即可保证「禁止跨 user_id 返回记忆」。

隔离代价是空间数量随 user_id 增长，由上层 ALMEngine 用 LRU 控制内存驻留数。
"""

import hashlib
import logging
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.core.path_tracker import PathTracker
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph

from alm.config import ALMConfig
from alm.contract import AddMessage
from alm.judge import RelevanceJudge

try:
    from dba_pipeline.embedding.store import VectorStore
    from dba_pipeline.extraction.dba import MemoryDBA
    from dba_pipeline.extraction.graph_builder import GraphBuilder
    from dba_pipeline.llm.inference import InferenceEngine
    from dba_pipeline.retrieval.retriever import PEAK_NODES_AUTO, PurposeDrivenRetriever

    HAS_DEPS = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确报错
    HAS_DEPS = False
    _IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

# 兜底节点的单块字符预算。**必须 ≤ embedding 实际可容纳量**：实测
# SiliconFlow / BAAI-bge-large-zh-v1.5 在 550 字符返回 200、600 字符返回 400
# （见 dba_pipeline/embedding/store.py 的 _MAX_CHARS 标定），故取 500 留余量。
#
# 超长消息不再截断丢弃，而是切成多块、各成一个节点——否则一条 8000 字符的消息
# 会变成"只有前 500 字符可被向量检索到"的节点（可检索比例 6%）：内容确实落库了，
# 却检索不到，与 ALM 契约「已完整存储且可被检索」不符。
_FALLBACK_CHUNK_CHARS = 500

# 切块时的句末标点：优先在这类字符之后断开，避免把一句话从中间劈开
_SENTENCE_ENDINGS = "。！？!?；;\n"

# 「全局 / 总体视角」的本地关键词。**仅用于观测，不参与任何控制流。**
#
# 为什么要单独记一个「宽判定」：F1「摘要与压缩 / 长历史全局综合」的题干措辞未知。
# 它可能显式要求压缩（LLM 的 condense 标签会命中），也可能只是隐含的总体视角问法
# （如「这一路是怎么走过来的」），后者**不会**被严格要求显式措辞的 condense 捕获。
# 两种判定同时记录，一轮流量后即可判定：
#   宽判定命中、严判定不命中 → 题干是隐含全局型，该放宽触发条件；
#   两者都不命中             → F1 与压缩/全局意图无关，本方向不成立。
#
# 只对 query 做子串匹配并记录布尔值，**不落 query 正文**（合规口径同其它日志）。
_GLOBAL_VIEW_HINTS = (
    "总结", "概括", "汇总", "压缩", "回顾", "总的", "整体", "总体",
    "一路", "这些日子", "这段时间", "都经历", "变化", "综合", "一直以来",
)


def _looks_like_global_query(query: str) -> bool:
    """本地粗判： query 是否带「全局 / 总体视角」措辞（仅观测用）"""
    return any(h in (query or "") for h in _GLOBAL_VIEW_HINTS)


def _clip_for_log(text: str, limit: int) -> str:
    """留痕截断：把**单条**日志记录钉在上限以内。``limit <= 0`` 表示不截断。

    为什么必须有界（0930 Full 实测）：`trace_io=1` 时单个日志文件 **84 MB**、约 60 秒
    写满一次轮转（100m×10）；平台上确有把整份文档当 query 送进来的轨道，单条 `[IO]`
    可达数 MB——一条记录就能顶掉一次轮转，把前面的证据冲掉。

    截断保留**头部**：形态判定（是长会话还是整段作文、query 里有没有内联选项）看的都是
    开头。尾部标注原长，便于区分"本来就短"与"被截了"。
    """
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…（留痕截断：原文共 {len(text)} 字）"


# ---- 平台 QA 反馈护栏 ----
#
# 平台在**每次 search 之后**会回写一条 1 消息的 Add，内容是评测元数据（0926 实测，
# 全轮 4/4 命中同一形态，另 130 次 Add 均无此形态）：
#
#   [user] Streaming online QA feedback:
#   {"method": "locomo_llm_judge", "predicted_answer": "Yesterday.",
#    "qa_id": "locomo1:q0000:step01", "question": "When Jon has lost his job as a banker?",
#    "schema_version": "streaming-online-feedback-v1", "score": 0.0}
#
# **为什么必须拦在入图之前**：它含「题目 + 我们上一轮的答案 + 得分」。抽成节点后，
# 它与后续对同一题的查询几乎同文，余弦接近 1.0 —— 会抢走种子位并把扩展引偏；
# 更糟的是它把**上一次的错误答案**（`predicted_answer`）也写进记忆，可能反过来
# 误导下一轮的作答。实测污染样本：`u_44***7417` 的节点 `n524`
# （`(thing) The QA question asks what a courtroom demonstration of unfastening a chain…`）。
#
# **但绝不能静默丢弃**：`qa_id`（如 `locomo1:q0000:step01`）与 `method`
# （如 `locomo_llm_judge`）是平台**唯一**主动告知「这道题属于哪个数据集、怎么判分」
# 的信道——此前我们完全无法把 query 对应到能力项，B2/D1/F1 的归因一直缺这一环。
# 故：内容不进图，字段单独留痕（`[IO] qa_feedback`）。
_QA_FEEDBACK_HINTS = ("streaming online qa feedback", "streaming-online-feedback")

_QA_FIELD_RES = (
    ("qa_id", re.compile(r'"qa_id"\s*:\s*"([^"]*)"')),
    ("method", re.compile(r'"method"\s*:\s*"([^"]*)"')),
    ("score", re.compile(r'"score"\s*:\s*(-?[0-9.]+)')),
    ("question", re.compile(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"')),
    ("predicted", re.compile(r'"predicted_answer"\s*:\s*"((?:[^"\\]|\\.)*)"')),
)

# 只含平台自己给出的题号/方法/分数——**不含题面与正文**。合规上可常开，故承载
# 「唯一能把 query 归到能力项」的信道（见 0927 审查 C6：题面留给 trace_io 控制的
# `[IO] qa_feedback`，但 qa_id 不能跟着一起关，否则 Full 评测期 B2/D1/F1 归因断掉）。
_QA_ID_KEYS = ("qa_id", "method", "score")


def _is_qa_feedback(text: str) -> bool:
    """是否为平台的 QA 反馈回写（评测元数据，不是用户记忆）。

    判据用**平台自己的 schema 标记**而非启发式形态：
    首行 `Streaming online QA feedback:` 或 `schema_version` 里的
    `streaming-online-feedback`。二者都是平台逐字给出的，不存在误伤真实记忆的可能
    ——实测 1495 条真实记忆消息全部不命中。
    """
    low = (text or "").lower()
    return any(h in low for h in _QA_FEEDBACK_HINTS)


def _qa_feedback_summary(text: str) -> str:
    """从 QA 反馈里抽出可留痕的字段（截断，只用于日志）"""
    parts = []
    for key, rx in _QA_FIELD_RES:
        m = rx.search(text or "")
        if not m:
            continue
        value = m.group(1)
        if key in ("question", "predicted"):
            value = " ".join(value.split())[:120]
        parts.append(f"{key}={value}")
    return "  ".join(parts) or "（字段未解析出）"


def _qa_feedback_ids(text: str) -> str:
    """只抽题号/方法/分数，**不含题面与正文**（可不受 trace_io 控制，见审查 C6）"""
    parts = []
    for key, rx in _QA_FIELD_RES:
        if key not in _QA_ID_KEYS:
            continue
        m = rx.search(text or "")
        if m:
            parts.append(f"{key}={m.group(1)}")
    return "  ".join(parts) or "（字段未解析出）"


# 平台把选择题的**选项与作答指令内联在 query 字符串里**（实测 `scriptMem` 那题：
# 全长 1147 字 = 题干 168 + 选项块 876 + 作答指令 103）。选项与作答指令是留给平台侧
# answer 模型的，检索侧只该消费题干——这一整坨进向量会把读取方向带偏，实测同题：
#   ① 种子排序被「六种立场描述」的质心接管：答案节点 n385（"foremost man"）由第 1
#      掉到第 20（余弦 0.609→0.514），而 top-12 被"oath/Billing/Stockmanns"这类表面词占满；
#   ② 目的推断退化成 `identify the correct interpretation` / `select the best-supported
#      option` 这类**元目的**（只给题干时是 `infer the underlying shift in Billing's stance`），
#      于是目的回归对真正的证据不敏感（n385 的 purpose 余弦仅 0.343）。
_OPTION_MARK_RE = re.compile(r"(?:^|[\s（(])([A-H])[.、．)）]\s")
_OPTION_MIN_RUN = 3


def _strip_option_block(query: str) -> str:
    """剥离内联在 query 尾部的「选项块 + 作答指令」，只留题干。

    判据用**连续递增的选项标号**（A、B、C… 至少 `_OPTION_MIN_RUN` 个），而不是
    "出现 A."：题干自身也可能含 A./B. 字样，而连续三个同序标号只可能来自选项块。
    命中则从**最后一个**成立的 A 处截断（选项块在尾部；其后的作答指令一并去掉）。
    未命中就原样返回——普通问答题、以及选项走 `options` 契约字段的请求都不受影响。
    """
    text = query or ""
    hits = [(m.start(1), m.group(1)) for m in _OPTION_MARK_RE.finditer(text)]
    start = None
    for i, (pos, label) in enumerate(hits):
        if label != "A":
            continue
        run, expect = 1, ord("B")
        for _pos, label2 in hits[i + 1:]:
            if label2 == chr(expect):
                run += 1
                expect += 1
        if run >= _OPTION_MIN_RUN:
            start = pos
    if start is None:
        return text.strip()
    return text[:start].strip()


# 「要唯一/精确答案」的提问形态：平台把选择题的选项与作答指令内联在 `query` 里，这本身
# 就是**最强的类型信号**（结构判定，不是关键词表）。刻意只收这几类表述：宁可漏判，
# 不可误判——误判会把"丰富联想"类的问题也压成一条。
_ANSWER_INSTRUCTION_RE = re.compile(
    r"enclosed in parentheses"
    r"|only the correct answer"
    r"|only one correct"
    r"|single correct"
    r"|choose the correct"
    r"|e\.?\s?g\.,?\s*\(X\)"
    r"|只(?:需|要)?回答(?:唯一)?(?:正确)?(?:的)?(?:选项|答案)"
    r"|仅(?:需|要)?回答(?:唯一)?(?:正确)?(?:的)?(?:选项|答案)"
    r"|选择(?:正确|最合适|最佳)的?(?:选项|答案)"
    r"|正确答案是",
    re.I,
)


def _has_answer_instruction(query: str) -> bool:
    """query 里有没有**明确要求作答**的指令（"enclosed in parentheses" / "请只回答选项" …）。

    这是"该不该动这条 query"的**唯一开关**：只有题面明说要一个答案时，才允许剥离选项块
    并收拢读出。判据落在"指令"而不是"有没有 A./B./C. 标号"，是为了不误伤普通列举句
    （"Compare version A. alpha, B. beta, and C. gamma" 同样有连续标号）——见
    config.precision_route 的"宁可漏判，不可误判"口径：**缺指令的内联选项题只会漏判**
    （不剥离、不收拢），代价是退回本轮修复前的行为，不会把正常问题改坏。
    """
    return bool(_ANSWER_INSTRUCTION_RE.search(query or ""))


def _is_precision_query(query: str) -> bool:
    """这条 query 是否在要一个**唯一/精确答案**（→ 剥离选项块 + 收拢读出）。

    与 `_has_answer_instruction` 同判据，故"剥离"与"收拢"两个动作永远同步开关——不会出现
    "剥了没收"或"收了没剥"。契约里选项走 `options` 字段的情形由调用方一并判
    （见 `_search_locked`）。
    """
    return _has_answer_instruction(query)


def _retrieval_text(query: str) -> str:
    """检索用文本：题面**明确要求作答**时剥离内联选项块，否则原样返回。

    两个都必须满足才是"剥离"，缺一不可：
    ① 有作答指令（`_has_answer_instruction`）——否则普通列举句会被整体截断误伤；
    ② 剥离后非空——若整条 query 就是选项块（`start == 0`），退回原文，避免空串进 embedding。
    """
    if not _has_answer_instruction(query):
        return query
    return _strip_option_block(query) or query


def _split_by_chars(text: str, budget: int) -> List[str]:
    """按字符预算切块，尽量让每块以完整句子结尾。

    纯硬切（每 budget 字符一刀）会把一句话劈成两半，两块各自都表达不完整语义，
    向量质量都会下降。这里在窗口内**向前回看**找最后一个句末标点，找到就在那里断。

    回看有下限（budget // 2）：否则标点稀疏的长句会退化成极短的块，块数膨胀。
    窗口内找不到标点时**退而切在最近的空白处**——见下方说明。

    入口守卫 `budget <= 0`：否则 `end = start + 0 = start`、`floor = start`，
    两个回看 range 均为空 → 追加空块、`start` 不变 → **死循环**（占住线程，
    直到进程被杀）。触发条件只是一个配错的环境变量（见 0927 审查 B4）。
    """
    if budget <= 0 or len(text) <= budget:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + budget, len(text))
        if end < len(text):
            floor = start + budget // 2
            for i in range(end - 1, floor - 1, -1):
                if text[i] in _SENTENCE_ENDINGS:
                    end = i + 1
                    break
            else:
                # 回看窗口内没有句末标点 → 退而切在最近的空白处。
                # 为什么必须有这一步：英文长句经常超过 budget//2 字符（口语化文本、
                # 剧本、论文摘要都是），此时按预算硬切会把单词劈成两半——线上实测
                # 出现过以半个词开头的节点块（`[n2] n. You must include units…`），
                # 两块各自都表达不完整语义，向量质量都下降。空白是**所有语言都有的
                # 词边界**，比硬切严格更优；向后找不到就在整个已用预算里向前找，
                # 都没有（无空格语言的长串）才按预算硬切——保证进展，绝不返回空块。
                for i in range(end - 1, start, -1):
                    if text[i].isspace():
                        end = i + 1
                        break
        chunks.append(text[start:end])
        start = end
    return chunks


def _cosine(a, b) -> float:
    """余弦相似度；任一输入缺失或为零向量时返回 0（仅服务于弃权判定）"""
    if a is None or b is None:
        return 0.0
    vec_a = np.asarray(a, dtype=float)
    vec_b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class RetrievalStats:
    """检索可观测性：累计每次 Search 的「候选最高余弦」与弃权次数。

    存在的理由是**标定弃权阈值**：ALM 真实数据不可离线观测，但平台每次 smoke 都会打
    我们的 `/search`，把这些请求的 best_cos 记下来（只记数值，**不记 query 正文**，
    符合平台的合规要求），跑完一轮就能看到真实查询实际落在什么区间：

    - 若 p10 已经高于阈值 → 阈值偏低，漏弃权会伤 H（无记忆时没返回空）；
    - 若 p90 低于阈值 → 阈值偏高，过度弃权会成片伤 A（有记忆也返回空）；
    - 若阈值正好落在分布中间 → 必须重标定，此时任何一个方向的偏移都在扣分。

    Add / Search 在线程池中并发执行，故计数加锁。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.searches = 0
        self.abstained = 0
        self._cosines: List[float] = []

    def record_search(self, best_cos: float) -> None:
        """每次检索都调用（含最终未弃权的），保证余弦分布完整"""
        with self._lock:
            self.searches += 1
            self._cosines.append(round(float(best_cos), 4))

    def record_abstain(self) -> None:
        """确认弃权时调用。

        与 record_search 分开，是因为两段式判定的弃权点落在检索**之后**
        （模糊带要等 top-N 排好才能交给 LLM 判定），无法在记余弦时一并确定。
        """
        with self._lock:
            self.abstained += 1

    def last_cosine(self) -> Optional[float]:
        """最近一次检索的候选最高余弦；尚未记录过则返回 None。

        供标定脚本按查询取值用（串行发起检索即可一一对应）。
        """
        with self._lock:
            return self._cosines[-1] if self._cosines else None

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = int(round(q * (len(ordered) - 1)))
        return ordered[max(0, min(pos, len(ordered) - 1))]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            searches = self.searches
            abstained = self.abstained
            cosines = list(self._cosines)
        return {
            "searches": searches,
            "abstained": abstained,
            "abstain_rate": round(abstained / searches, 4) if searches else 0.0,
            "min": round(min(cosines), 4) if cosines else 0.0,
            "p10": round(self._quantile(cosines, 0.10), 4),
            "p25": round(self._quantile(cosines, 0.25), 4),
            "p50": round(self._quantile(cosines, 0.50), 4),
            "p75": round(self._quantile(cosines, 0.75), 4),
            "p90": round(self._quantile(cosines, 0.90), 4),
            "max": round(max(cosines), 4) if cosines else 0.0,
        }

    def report(self) -> str:
        """人类可读的分布汇总（进程退出时打印到 stderr）"""
        data = self.summary()
        lines = [
            "=" * 60,
            "[ALM] 检索余弦分布（标定 ALM_ABSTAIN_COS 用）",
            "=" * 60,
            f"  检索次数 : {data['searches']}",
            f"  弃权次数 : {data['abstained']}（{data['abstain_rate'] * 100:.1f}%）",
            f"  best_cos : min={data['min']}  p10={data['p10']}  p25={data['p25']}  "
            f"p50={data['p50']}",
            f"             p75={data['p75']}  p90={data['p90']}  max={data['max']}",
            "=" * 60,
        ]
        return "\n".join(lines)


class MemorySpace:
    """单个 user_id 的完整记忆栈：图谱 + 向量 + DBA 维护 + PAR 检索"""

    def __init__(
        self,
        user_id: str,
        config: ALMConfig,
        embeddings,
        llm,
        yaml_path: Path,
        stats: Optional[RetrievalStats] = None,
    ):
        if not HAS_DEPS:
            raise RuntimeError(
                f"ALM 记忆链路依赖未安装（{_IMPORT_ERROR}）；请先执行 pip install -e ."
            )

        self.user_id = user_id
        self.config = config
        self.yaml_path = Path(yaml_path)
        self.stats = stats
        # 串行化单个用户空间内的图谱写入与 YAML 落盘
        self._lock = threading.RLock()
        # 自上次落盘以来图谱是否被改动过。add() 置位、save() 成功后清位。
        # 作用不只是省一次写盘：留存储存期清理（alm/cleanup.py）按 mtime 判定
        # 空间最后一次被写入的时间，而 engine.close() 会对**全部驻留空间**调 save()。
        # 若无条件重写，停服这一步就会把每个空间的 mtime 刷成停服时刻，
        # 使「先停服再清理」的既定流程永远判定不到过期空间（静默空转）。
        self._dirty = False

        self.graph = (
            load_graph(str(self.yaml_path)) if self.yaml_path.exists() else MemoryGraph()
        )
        self.vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
        if not self._load_index():
            self._reindex()

        # 文档通道（异构输入路由的目标区）：与图谱**完全隔离**。
        #
        # 为什么不能塞进图：文档块之间没有记忆意义上的关联，一旦进图就会参与
        # 种子选择 / 图扩展 / 峰值检测。线上图文件实测正是这条路径把 289 个孤立
        # 文档块变成了种子层——峰值落在 hop0、`跳=1`、多跳彻底失效（32 个平台图
        # 节点全是兜底块且边数恰好为 0）。隔离后它们不进图算法，只在读侧由
        # 专门的 document_lookup 取回。
        self.docs: List[Dict[str, Any]] = []
        self._doc_index: Dict[str, int] = {}
        self._doc_seq = 0
        self.doc_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
        self._load_documents()

        self.builder = GraphBuilder(
            graph=self.graph,
            vector_store=self.vector_store,
            orphan_threshold=self.config.orphan_threshold,
            dedup_threshold=self.config.graph_dedup_threshold,
            lock=self._lock,
        )
        self.dba = MemoryDBA(
            llm=llm, graph=self.graph, vector_store=self.vector_store, graph_builder=self.builder,
            edge_wide=self.config.edge_wide,
            triage_enabled=self.config.triage_enabled,
        )
        self.retriever = PurposeDrivenRetriever(
            llm=llm,
            embeddings=embeddings,
            graph=self.graph,
            vector_store=self.vector_store,
            inference=InferenceEngine(llm),
            path_tracker=PathTracker(),
            purpose_filter_threshold=self.config.purpose_filter_threshold,
            # 0930：把容忍带带宽接出来（此前从未传入 → 一直吃 PeakFinder 库默认 0.10）。
            # ratio 默认 0 = 走绝对模式，行为与改造前逐字一致。
            peak_tolerance=self.config.peak_tolerance,
            peak_tolerance_ratio=self.config.peak_tolerance_ratio,
        )
        # 弃权模糊带的二次判定（仅当 config.abstain_verify_hi > 0 时被调用）
        self.judge = RelevanceJudge(
            llm,
            top_n=config.abstain_judge_top_n,
            max_chars=config.abstain_judge_max_chars,
        )

    # ---- 初始化 ----

    def _reindex(self):
        """从图谱内容重建向量索引。

        进程重启后不依赖 FAISS 落盘，避免索引与图谱不同步；
        代价是冷启动需重新嵌入一次，空间在内存中以 LRU 缓存摊薄。
        """
        node_ids = [
            nid for nid in self.graph.graph.nodes()
            if self.graph.graph.nodes[nid].get("content")
        ]
        if not node_ids:
            return
        contents = [self.graph.graph.nodes[nid].get("content", "") for nid in node_ids]
        self.vector_store.add_memories(node_ids, contents)
        logger.info("空间 %s 重建向量索引: %d 条", self._masked_id(), len(node_ids))

    # ---- 向量索引持久化 ----
    # 无索引落盘时，每次空间载入（LRU 换出后再访问、或进程重启）都要把整图重新嵌入
    # 一遍——API embedding 下这是成百上千次外部调用。改为随 YAML 一起落盘索引，
    # 并用「节点集合指纹」校验一致性，不一致才回退重建。

    def _index_paths(self):
        """索引与其指纹文件的路径。

        必须让每个空间独占一个子目录：VectorStore.save 会把向量缓存写到索引路径的
        同级目录，若直接放在 data_dir 下，多个空间会互相覆盖。
        """
        base = self.yaml_path.parent / (self.yaml_path.stem + ".index")
        return base / "faiss", base / "stamp.yaml"

    def _index_digest(self) -> Dict[str, Any]:
        """索引指纹：节点集合 + **embedding 模型名**

        为什么必须把模型名纳进来：落盘索引里存的是**某个模型产出的向量**，而指纹原先只认
        节点集合。换 embedding 模型后，旧索引会被判定为"与图谱一致"而照常加载，于是
        **旧模型的节点向量与新模型的查询向量混用**——余弦分布完全错位，检索质量崩掉，
        且全过程没有任何报错或日志（最坏的一种失效）。

        纳入模型名后，换模型即指纹不匹配 → `_load_index` 返回 False → 自动走 `_reindex()`
        全量重建。旧索引文件不需要人工清理。
        """
        node_ids = sorted(str(n) for n in self.graph.graph.nodes())
        return {
            "count": len(node_ids),
            "digest": hashlib.sha1("|".join(node_ids).encode("utf-8")).hexdigest(),
            "embedding": self.config.embedding_model or "",
            # 同时纳入 embedding 的单条字符预算：它决定长文本的向量是"全文"还是"仅前缀"。
            # 改这个值会让**已落盘的向量**与新查询向量语义不一致（老向量只覆盖前缀），
            # 而指纹若不含它，这类改变会被判定为"索引一致"而静默沿用——与模型名同一个
            # 坑（见上）。故一并纳入，改值即自动重建。
            "max_chars": getattr(self.vector_store.embeddings, "_MAX_CHARS", None),
        }

    def _load_index(self) -> bool:
        """复用落盘索引；与当前图谱不一致时返回 False 以触发重建"""
        index_path, stamp_path = self._index_paths()
        try:
            if not index_path.exists() or not stamp_path.exists():
                return False
            with open(stamp_path, "r", encoding="utf-8") as f:
                stamp = yaml.safe_load(f) or {}
            if stamp != self._index_digest():
                logger.info("空间 %s 索引与图谱不一致，改为重建", self._masked_id())
                return False
            self.vector_store.load(str(index_path), embeddings=self.vector_store.embeddings)
            logger.info("空间 %s 复用落盘索引: %d 条", self._masked_id(), stamp.get("count"))
            return True
        except Exception as exc:
            logger.warning("空间 %s 索引加载失败，改为重建: %s", self._masked_id(), exc)
            return False

    def _save_index(self):
        """落盘 FAISS 索引与向量缓存（失败不影响主流程，仅降级为下次重建）"""
        index_path, stamp_path = self._index_paths()
        try:
            self.vector_store.save(str(index_path))
            stamp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stamp_path, "w", encoding="utf-8") as f:
                yaml.dump(self._index_digest(), f, allow_unicode=True)
        except Exception as exc:
            logger.warning("空间 %s 索引保存失败: %s", self._masked_id(), exc)

    # ---- 文档通道（异构输入的隔离区）----
    #
    # 背景：平台的输入混了两类形态完全不同的数据——**对话记忆**（本系统的设计目标）
    # 与**知识库式长文档**（整部剧本、论文+系统提示词、NTSB 摘录、任务轨迹）。
    # 后者交给节点/边抽取是范畴错误。线上实测其后果：
    #   · 抽取产出为空 → 走兜底 → 原文被切成几百个孤立节点灌进图；
    #   · 32 个平台图「节点全是兜底块、边数恰好为 0」，最大一个 289 节点 0 边
    #     （原文约 14.5 万字符，是一整部易卜生剧本）；
    #   · 这些孤立块被当成记忆种子 → 峰值落在 hop0 → `跳=1` → 图扩展零次。
    # 故文档走**独立存储 + 独立检索**：切块、向量化、按块序存，不抽节点、不建边、
    # 不进图。文档唯一的"边"就是块序，读侧用相邻块扩展来利用它。

    def _doc_paths(self):
        """文档区路径：条文 YAML + 独立向量索引目录（含指纹文件）。

        必须与图谱的索引分开放：两者的 id 空间、失效条件、生命周期都不同，
        混在一起会让图谱的「索引指纹」把文档块也算进去，换一次文档就重建一次图索引。
        """
        stem = self.yaml_path.stem
        base = self.yaml_path.parent
        return (
            base / f"{stem}.docs.yaml",
            base / f"{stem}.docs.index" / "faiss",
            base / f"{stem}.docs.index" / "stamp.yaml",
        )

    def _load_documents(self) -> None:
        """载入文档区：条文 YAML 提供内容与块序，落盘索引提供向量。

        有索引就直接复用（**不重新 embedding**）；索引缺失或损坏才回退全量重嵌。
        与图谱索引不同，文档区不需要指纹校验——条文与向量总是一起重写的，
        不存在「图谱变了而索引没跟上」这种中间态。
        """
        docs_yaml, docs_index, _ = self._doc_paths()
        try:
            if docs_yaml.exists():
                data = yaml.safe_load(docs_yaml.read_text(encoding="utf-8")) or {}
                self.docs = list(data.get("docs") or [])
                self._doc_seq = int(data.get("seq") or 0)
                self._doc_index = {d["id"]: i for i, d in enumerate(self.docs)}
        except Exception as exc:
            logger.warning("空间 %s 文档区读取失败，按空处理: %s", self._masked_id(), exc)
            self.docs, self._doc_index, self._doc_seq = [], {}, 0
        if not self.docs:
            return

        if docs_index.exists():
            try:
                self.doc_store.load(str(docs_index), embeddings=self.doc_store.embeddings)
                ntotal = getattr(getattr(self.doc_store.store, "index", None), "ntotal", None)
                if ntotal == len(self.docs):
                    logger.info("空间 %s 文档区载入: %d 块（复用落盘索引）",
                                self._masked_id(), len(self.docs))
                    return
                # 条文与索引条数不一致时**必须重建**：_save_documents 是先写条文、再存索引，
                # 中途中断会让条文比索引多；多出来的块没有向量、永远检索不到，而且因为
                # "索引文件存在"这条判断，它**永远不会**触发重建（静默失效）。
                logger.warning(
                    "空间 %s 文档区索引条数 %s 与条文 %d 不一致，改为重建",
                    self._masked_id(), ntotal, len(self.docs),
                )
            except Exception as exc:
                logger.warning("空间 %s 文档索引加载失败，改为重建: %s", self._masked_id(), exc)
        # 索引缺失 / 损坏 / 条数不符 → 全量重嵌（正常启动不会走到这里）。
        # 必须换一个**全新的** VectorStore：加载失败可能留下半加载的 FAISS，
        # 直接对旧实例 add_memories 会往已有索引上追加，产出重复条目。
        self.doc_store = VectorStore(
            embeddings=self.doc_store.embeddings, backend="faiss", persist_dir=None
        )
        self.doc_store.add_memories(
            [d["id"] for d in self.docs], [d["content"] for d in self.docs]
        )
        logger.info("空间 %s 文档区重建向量: %d 块", self._masked_id(), len(self.docs))

    def _save_documents(self) -> None:
        """落盘文档条文与索引（失败只降级为下次重嵌，不影响主流程）"""
        if not self.docs:
            return
        docs_yaml, docs_index, _ = self._doc_paths()
        try:
            docs_yaml.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = docs_yaml.with_name(docs_yaml.name + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump({"seq": self._doc_seq, "docs": self.docs}, f, allow_unicode=True)
            os.replace(tmp_path, docs_yaml)
            self.doc_store.save(str(docs_index))
        except Exception as exc:
            logger.warning("空间 %s 文档区保存失败: %s", self._masked_id(), exc)

    def _classify_batch(self, messages: List[AddMessage]) -> str:
        """按**形态**给一批输入分类，决定走哪条通道。

        纯计数、不调 LLM：线上实测两类形态差 1~2 个数量级（见 config.doc_chars_per_msg
        的标定数据），再加一次 LLM 分类会把每次 Add 的输入成本翻倍，不划算。

        返回值：
            "document"     —— 长文档（走文档通道）
            "conversation" —— 对话记忆（走现有 DBA 链路）
            "noise"        —— 空内容（不落库）
        """
        texts = [(m.content or "").strip() for m in messages]
        texts = [t for t in texts if t]
        if not texts:
            return "noise"
        total = sum(len(t) for t in texts)
        if total == 0:
            return "noise"
        if (len(texts) <= self.config.doc_max_messages
                and total / len(texts) >= self.config.doc_chars_per_msg):
            return "document"
        return "conversation"

    def _store_documents(
        self, messages: List[AddMessage], batch_ts: Optional[int] = None
    ) -> List[str]:
        """把原文切块存入文档区：不抽节点、不建边、不进图。

        它是 `_divert_raw()` 的落点，承接三类内容（见该方法）：极端长输入、
        抽取判定无记忆价值的原文、超量溢出。**注意它接的不一定是"文档"**——
        名字来自最初的场景（平台把整部剧本/论文当一次 Add 灌进来），但凡是没能
        变成图节点的原文都从这里落，语义是"非图内容区"。

        为什么不做抽取：这类内容的节点抽取产出恒为空（`create=0`），旧路径因此走兜底，
        把原文切成几百个孤立节点灌进图 → 被当成记忆种子 → 峰值落在 hop0 → `跳=1`。
        这里的正确表示就是「有序的块 + 向量」，块之间唯一的语义关系是**块序**
        （读侧用邻块扩展利用它）。
        """
        chunks: List[str] = []
        for msg in messages:
            text = (msg.content or "").strip()
            if not text:
                continue
            # 每个消息单独切块：块序在单篇文档内部才有意义，跨消息连续编号会让
            # 邻块扩展把两篇不同文档的末尾与开头拼在一起。
            chunks.extend(_split_by_chars(text, self.config.doc_chunk_chars))
        if not chunks:
            return []

        doc_id = f"d{self._doc_seq}"
        self._doc_seq += 1
        # id 前缀刻意用 d（图节点是 n*），保证两个 id 空间永不重叠——
        # 读侧要靠前缀区分「这是文档块」还是「这是图节点」。
        new_docs = [
            {
                "id": f"{doc_id}c{i:04d}",
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "content": c,
                "timestamp": batch_ts,
            }
            for i, c in enumerate(chunks)
        ]
        self.doc_store.add_memories(
            [d["id"] for d in new_docs], [d["content"] for d in new_docs]
        )
        self.docs.extend(new_docs)
        self._doc_index = {d["id"]: i for i, d in enumerate(self.docs)}
        return [d["id"] for d in new_docs]

    def _document_lookup(self, query_vec, k: int) -> List[Dict[str, Any]]:
        """文档通道检索：向量 top-k + 相邻块扩展。

        邻块扩展是文档唯一的"图扩展"：命中块往往只截到半句，上下文在邻块里，
        纯 top-k 会把答案切掉。不设上限——文档块不参与图算法，条数不构成「跳=1」
        那种结构性风险，只受 k 与邻域半径控制。
        """
        if not self.docs or self.doc_store.store is None:
            return []
        try:
            hits = self.doc_store.search_by_vector(np.asarray(query_vec, dtype=float), k=k)
        except Exception as exc:
            logger.warning("空间 %s 文档通道检索失败: %s", self._masked_id(), exc)
            return []

        picked: List[Dict[str, Any]] = []
        seen: set = set()
        radius = max(0, self.config.doc_neighbor)
        for mid, score in hits:
            pos = self._doc_index.get(str(mid))
            if pos is None:
                continue
            for off in range(-radius, radius + 1):
                j = pos + off
                if not (0 <= j < len(self.docs)):
                    continue
                doc = self.docs[j]
                if doc["id"] in seen:
                    continue
                seen.add(doc["id"])
                picked.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "score": float(score) if off == 0 else None,
                    "hit": off == 0,
                    "node_type": "原文摘录",
                    "timestamp": doc.get("timestamp"),
                })
        return picked

    # ---- 写入 ----

    def add(self, messages: List[AddMessage]) -> Dict[str, int]:
        """同步写入一批消息：维护完成后才返回（对齐 ALM 契约）

        入口先按**形态**路由（`_classify_batch`，纯计数、不调 LLM）：
          · 文档型 → 文档通道（切块 + 向量，不抽节点、不建边，见 `_store_documents`）
          · 会话型 → DBA 链路，并按 `add_batch_size` **分批**逐组维护
          · 空内容 → 丢弃
        """
        limit = self.config.max_messages_per_add

        # 平台 QA 反馈护栏（判据与依据见 _is_qa_feedback）。必须在 payload/overflow 切分
        # **之前**对整批过滤：否则落在第 limit 条起的反馈会经 overflow → _divert_raw
        # 作为文档块入库（0927 审查 C3）。这类消息既不是记忆内容，又会以近乎 1.0 的
        # 余弦抢走同题的种子位。
        # 过滤后整批为空时，下面的形态判定会判 "noise" 并正常返回，不会因为
        # "这次 Add 什么也没建"而报错。
        feedback = [m.content for m in messages if _is_qa_feedback(m.content)]
        if feedback:
            logger.warning(
                "空间 %s 丢弃平台 QA 反馈 %d/%d 条（评测元数据，不入图）",
                self._masked_id(), len(feedback), len(messages),
            )
            # qa_id / method / score 是平台唯一主动给出的"题目归属"信道，必须留痕。
            # 独立信道**不受 trace_io 控制**：只含平台自己的题号/方法/分数，不含题面与
            # 正文，合规上可常开。若与 [IO] 一起被关掉，Full 评测期 B2/D1/F1 的归因就断了
            # （0927 审查 C6）。
            for content in feedback:
                logger.info(
                    "[QA] qa_feedback 空间=%s %s",
                    self._masked_id(), _qa_feedback_ids(content),
                )
            # 含题面的完整留痕仍只在冒烟期开启（与 [IO] search 入参 同一合规口径）。
            if self.config.trace_io:
                for content in feedback:
                    logger.info(
                        "[IO] qa_feedback 空间=%s %s",
                        self._masked_id(), _qa_feedback_summary(content),
                    )
            messages = [m for m in messages if not _is_qa_feedback(m.content)]

        payload = messages[:limit]
        overflow = messages[limit:]

        batch_ts = self._batch_timestamp(payload)

        # 形态判定不受 trace 开关影响：路由本身是行为，不是观测。
        # doc_route=False 时全部按对话处理（A/B 对照用）。
        kind = self._classify_batch(payload) if self.config.doc_route else "conversation"

        # 冒烟期留痕：只记长度时看不出平台送进来的文本**是什么形态**（长会话占比、
        # 是否整段作文、triage 该不该放行），而这些正是兜底率与建图质量的判据。
        # 默认关闭，且仅在 ALM 冒烟的公开测试数据上开启（见 config.trace_io）。
        if self.config.trace_io:
            logger.info(
                "[IO] add 入参 空间=%s 形态=%s 消息=%d 首条时间戳=%s 原文:\n%s",
                self._masked_id(), kind, len(payload), batch_ts,
                _clip_for_log(self._format_conversation(payload), self.config.trace_max_chars),
            )

        if kind == "noise":
            logger.info(
                "空间 %s 输入为空（%d 条消息），按噪音丢弃", self._masked_id(), len(payload)
            )
            return {"messages": len(messages), "nodes": 0}

        acts = {"create": 0, "update": 0, "deprecate": 0}
        fallback_reasons: List[str] = []

        if kind == "document":
            with self._lock:
                # 先置位再写：中途抛错时保持 dirty，让退出路径仍会落盘。
                self._dirty = True
                created = self._store_documents(payload, batch_ts)
            logger.info(
                "空间 %s 文档通道: messages=%d 字符=%d 块=%d（不建节点、不建边）",
                self._masked_id(), len(payload),
                sum(len(m.content or "") for m in payload), len(created),
            )
            self._trace_created(created)
            return {"messages": len(messages), "nodes": len(created)}

        # ---- 会话通道：按 add_batch_size 分批，逐组维护 ----
        # 为什么必须分批：整批一次性交给 maintain 等价于 0828 §5.5 里的「M=全部」。
        # 该文实测 M=9 已让已有图上的耗时翻倍、误 deprecate 率升高；§五另测
        # 「整段一次灌入 → 压缩更深、联想路径断裂」。线上图文件侧的证据更硬：
        # 值变更（update/deprecate）在 >6 条的 73 个批次里**一次都没出现**，
        # 而 8 次 update 全落在 3~6 条批次、4 次 deprecate 全落在 1~2 条批次。
        groups = self._split_add_batches(payload)
        created = []
        with self._lock:
            # 先置位再改图：maintain 中途抛错时图谱可能已被部分修改，
            # 此时保持 dirty 让退出路径仍会落盘，宁多写一次也不丢内容。
            self._dirty = True
            for group in groups:
                conversation = self._format_conversation(group)
                grp_ts = self._batch_timestamp(group)
                result = self.dba.maintain(conversation, timestamp=grp_ts)
                grp_created = (result.get("result") or {}).get("created_ids") or []
                reason = self._fallback_reason(result, grp_created)
                if reason:
                    # 只记原因与条数，不记正文。用 WARNING 而非 INFO：兜底意味着本批
                    # **没有建图**（只留下孤立原文节点），是系统退化的信号，不该随着
                    # 日志级别被压掉。reason 的三取值见 _fallback_reason。
                    logger.warning(
                        "空间 %s 兜底触发: reason=%s messages=%d",
                        self._masked_id(), reason, len(group),
                    )
                    fallback_reasons.append(reason)
                    grp_created = self._divert_raw(group, grp_ts)
                created.extend(grp_created)
                for op in ((result or {}).get("ops") or {}).get("node_ops") or []:
                    action = (op or {}).get("action")
                    if action in acts:
                        acts[action] += 1
            if overflow:
                # 条数上限只用于控制 LLM 成本，**不等于可以丢弃内容**：契约要求每次
                # Add 的消息已完整存储。超出部分改走无需 LLM 的兜底落库。
                logger.warning(
                    "空间 %s 单次 Add %d 条超过维护上限 %d，超出 %d 条改走兜底落库",
                    self._masked_id(), len(messages), limit, len(overflow),
                )
                created = created + self._divert_raw(overflow, batch_ts)

        # 节点操作计数与兜底原因已在上面的分组循环里聚合（acts / fallback_reasons）。
        #
        # 为什么要记 ops_upd / ops_dep：D1「新值覆盖与当前状态」连续多轮接近 0 分，
        # 而写侧的值变更处理已离线验证是正确的（真实模型 4/5 会发 deprecate+create，
        # 且顺序正确）。线上实测把疑点钉在了**批次规模**上——值变更只在 ≤6 条的批次里
        # 出现，>6 条一次都没有——故本行同时记「分批数」，便于直接对照。
        logger.info(
            "空间 %s 写入完成: messages=%d chars=%d max_msg=%d 分批=%d 兜底=%s "
            "nodes=%d ops_create=%d ops_upd=%d ops_dep=%d",
            self._masked_id(), len(messages),
            # 只记长度，不记正文：用于观测平台单次 Add 的**形态**（长会话占比、
            # 单条长度分布），以判断 triage 的可信上限等设计参数是否标定得当。
            sum(len(m.content or "") for m in messages),
            max((len(m.content or "") for m in messages), default=0),
            len(groups),
            ",".join(fallback_reasons) or "无",
            len(created),
            acts["create"], acts["update"], acts["deprecate"],
        )
        self._trace_created(created)
        return {"messages": len(messages), "nodes": len(created)}

    def _trace_created(self, created: List[str]) -> None:
        """[IO] 落库留痕：落库节点的原文才是真正"记住了什么"。

        判断兜底是否把原文切碎、抽取是否丢掉了关键实体，都必须看这一行，光看
        nodes 计数做不到。文档块不在图里，故按 id 前缀分派取内容。
        """
        if not self.config.trace_io:
            return
        lines = []
        for nid in created:
            if nid in self._doc_index:
                lines.append(f"  [{nid}](原文摘录) {self.docs[self._doc_index[nid]]['content']}")
                continue
            node = self.graph.get_node(nid) or {}
            ntype = node.get("type") or node.get("node_type")
            ntype = getattr(ntype, "value", None) or ntype or "-"
            lines.append(f"  [{nid}]({ntype}) {node.get('content') or ''}")
        logger.info(
            "[IO] add 落库 空间=%s 节点=%d:\n%s",
            self._masked_id(), len(created),
            _clip_for_log("\n".join(lines), self.config.trace_max_chars) or "  （无）",
        )

    def _split_add_batches(self, messages: List[AddMessage]) -> List[List[AddMessage]]:
        """把一次 Add 的消息切成若干组，交给 DBA **逐组**维护。

        **两重约束，缺一不可**：
          · 条数 `add_batch_size`（默认 6）——对齐 0828 §5.5 的 M=6。注意该文只扫了
            M=3/6/9，且实验语料是**本地短对话**，M 的语义是"多少轮对话"；
          · 字符数 `add_max_chars`（默认 1500）——因为**条数不约束长度**。平台实测
            "3~6 条一批、单条均长 505 字"这类长消息对话，6 条就有 3000 字。只按条数
            切时线上仍有 9% 的组超过 1500 字（最大 2934），这些组 triage 被跳过、
            整段直送抽取；输入一长，抽取 LLM 就会"少切、切粗"（省 token、怕遗漏），
            节点内容从单一事实退化成概述 → 意图模糊 → 连边时判不出关系类型
            （线上边型 scenario 占 65.6%、causal 仅 21.9%，与"判不出就退回同场景"一致）。
          两重约束后实测 0 组超过 1500 字，代价是组数 292 → 307（+5%）。

        单条消息本身就超预算时，在**该消息内部**按句末标点切（纯规则，不调 LLM）。
        切分单元刻意保持在"段落 / 句子"粒度而非孤立小句：抽取原则 3 要求消解指代
        （"他"→"用户"），切成孤立小句会把这个上下文切掉。
        切出的片段沿用原消息的 role 与 timestamp。

        两个旋钮都 ≤0 时完全不切（A/B 对照用）。
        """
        size = self.config.add_batch_size or 0
        budget = self.config.add_max_chars or 0
        if size <= 0 and budget <= 0:
            return [list(messages)]

        # ① 单条超预算 → 先在消息内按句末标点切（仅此时才切分，否则整条进贪心累计）
        units: List[AddMessage] = []
        for m in messages:
            text = m.content or ""
            if budget > 0 and len(text) > budget:
                units.extend(
                    AddMessage(role=m.role, content=piece, timestamp=m.timestamp)
                    for piece in _split_by_chars(text, budget)
                )
            else:
                units.append(m)

        # ② 贪心成组：条数与字符双上限
        groups: List[List[AddMessage]] = []
        cur: List[AddMessage] = []
        cur_len = 0
        for u in units:
            n = len(u.content or "")
            if cur and ((size > 0 and len(cur) >= size)
                        or (budget > 0 and cur_len + n > budget)):
                groups.append(cur)
                cur, cur_len = [], 0
            cur.append(u)
            cur_len += n
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def _batch_timestamp(messages: List[AddMessage]) -> Optional[int]:
        """取本批消息的代表时间（最后一个携带的时间戳，Unix 毫秒）。

        ALM 的 Add 以批次为单位，节点抽取不区分单条消息，因此时间戳按批次注入；
        取最后一个非空时间戳，等价于「本批最新事件时间」，与时间序上升的会话一致。
        """
        for msg in reversed(messages):
            if msg.timestamp is not None:
                return msg.timestamp
        return None

    def _fallback_reason(self, result: Dict[str, Any], created: List[str]) -> Optional[str]:
        """需要兜底落库时返回原因标识，否则返回 None。

        ALM 要求每次 Add 的消息「已完整存储且可被检索」，而 Ariadne 的
        triage 会在判定无维护价值时整段跳过、LLM 也可能抽不出节点——
        这两种情况都必须兜底，否则该会话记忆永久缺失。

        但「LLM 发起了 create 却一个都没落地」不属于这两种情况：那说明这些内容
        命中了语义去重，即图中已有等价记忆。此时再兜底是有害的——兜底刻意把去重
        阈值提到 2.0（相当于关闭去重），会把同一批内容以原文形式重复写入。平台重试
        同一次 Add 时正好走这条链：去重命中 → created 为空 → 兜底关去重 → 必然重复。

        返回值是可观测性的一部分：平台流量下兜底率偏高时，必须能区分是
        triage 判 SKIP（调用链上游决策）还是抽取产出为空（调用链下游问题），
        否则两条完全不同的处置路径无从选择。三种取值互补：
            "triage"    —— 维护判断前置直接跳过，压根没跑抽取
            "no_ops"    —— 跑了抽取但一个 node_op 都没有
            "no_create" —— 有 ops 却没有 create，且没有任何 update/deprecate/fix_type
        """
        if result.get("skipped"):
            return "triage"
        if created:
            return None
        node_ops = (result.get("ops") or {}).get("node_ops") or []
        if not node_ops:
            return "no_ops"
        if any(op.get("action") == "create" for op in node_ops):
            return None
        if not any(op.get("action") in ("update", "deprecate", "fix_type") for op in node_ops):
            return "no_create"
        return None

    def _divert_raw(
        self, messages: List[AddMessage], batch_ts: Optional[int] = None
    ) -> List[str]:
        """存放**未经抽取**的原文内容（契约要求「已完整存储且可被检索」）。

        调用点有三处，语义相同——内容没能变成图节点：
          ① 形态判定为文档型（极端长输入，先去保护 LLM）
          ② 抽取判定无记忆价值：triage SKIP / no_ops / no_create
          ③ 超过 `max_messages_per_add` 的溢出部分

        为什么默认送**文档通道**而不是像原来那样切成孤立节点灌进图：实测那些孤立块
        会被当成记忆种子 → 峰值落在 hop0 → `跳=1` → 多跳彻底失效；线上 32 个平台图
        「节点全是兜底块、边恰好为 0」正是由此而来。文档通道同样可检索（向量 + 邻块
        扩展），但不参与种子 / 扩展 / 峰值。

        `doc_route=false` 时退回旧行为（图内兜底块），用于 A/B 对照。
        """
        if self.config.doc_route:
            return self._store_documents(messages, batch_ts)
        return self._store_raw_fallback(messages, batch_ts)

    def _store_raw_fallback(
        self, messages: List[AddMessage], batch_ts: Optional[int] = None
    ) -> List[str]:
        """兜底落库：把消息原文逐条存为可检索节点

        原文兜底必须绕过语义去重，否则与既有记忆语义相近的消息会被判为重复而整条
        丢弃——而 add() 仍返回 200，形成静默丢数据（实测：同批「随便聊聊第N段」
        只有第 1 条落库）。调用方 add() 已持有 space 锁，builder 共享同一把锁，
        因此这里的阈值临时调整不会被并发观察到。

        长消息按 `_FALLBACK_CHUNK_CHARS` **切块**而非截断：每块单独成节点，保证
        每块都能被完整嵌入、可被向量检索到。块边界就近取句子结束符，避免把一句话
        从中间劈开——劈开会让两块的向量都表达不完整语义，检索质量下降。
        """
        limit = self.config.fallback_max_nodes

        # 先把消息格式化成原文，再按字符预算切块
        chunks: List[str] = []
        for msg in messages:
            text = msg.content.strip()
            if not text:
                continue
            chunks.extend(_split_by_chars(f"{msg.role}: {text}", _FALLBACK_CHUNK_CHARS))

        if len(chunks) > limit:
            logger.warning(
                "空间 %s 兜底落库需 %d 块，超过节点上限 %d，尾部 %d 块未存储",
                self._masked_id(), len(chunks), limit, len(chunks) - limit,
            )
            chunks = chunks[:limit]
        if not chunks:
            return []

        ops = [
            {"action": "create", "content": c, "node_type": NodeType.THING.value}
            for c in chunks
        ]

        saved_threshold = self.builder.dedup_threshold
        self.builder.dedup_threshold = self.config.fallback_dedup_threshold
        try:
            result = self.builder.apply_ops(ops, [], batch_timestamp=batch_ts)
        finally:
            self.builder.dedup_threshold = saved_threshold

        created = result.get("created_ids") or []
        logger.info(
            "空间 %s 兜底落库: %d 块原文 → %d 个节点",
            self._masked_id(), len(chunks), len(created),
        )
        return created

    @staticmethod
    def _format_conversation(messages: List[AddMessage]) -> str:
        """把结构化消息拼成 DBA 维护所需的对话文本"""
        lines = []
        for msg in messages:
            prefix = f"[{msg.role}]"
            if msg.timestamp is not None:
                prefix = f"[{msg.role}@{msg.timestamp}]"
            lines.append(f"{prefix} {msg.content}")
        return "\n".join(lines)

    # ---- 检索 ----

    def _effective_seed_k(self) -> int:
        """按图规模自适应确定 seed 数量。

        单 user 空间的图往往只有几十到几百个节点，固定 seed_k 会造成两种失真：
        - 过大（如 40）在百节点级图上会把整张图召回为种子，PAR 的图扩展无事可做，
          T2/T3 梯度全部为空；
        - 过小在孤立节点占比高时直接限制召回——孤立节点没有任何边，无法通过
          图扩展被找回，只能作为种子被召回。

        规则：图规模不超过 seed_k 时全量取种子；否则按 1/10 取值并夹在
        [seed_k_min, seed_k]。下界不低于 2，因为底层检索器的 purpose_seeds
        取 k = seed_k - seed_k//2，seed_k=1 时 k=0 会触发 FAISS 断言崩溃。
        """
        total = self.graph.node_count
        if total <= 0:
            return max(2, self.config.seed_k_min)
        if total <= self.config.seed_k:
            return max(2, total)
        return max(2, self.config.seed_k_min, min(self.config.seed_k, total // 10))

    @staticmethod
    def _story_id_from_text(text: str) -> str:
        """由叙述正文派生 item id。

        0927 前这里还有一个"对采纳节点集合取哈希"的版本（id 更稳定）；StoryRank 停止
        回 id 列表后它失去了输入，只剩这一个。稳定性由正文保证：同一批记忆 + 同一问题
        在同一作用域下正文一致，id 即一致。
        """
        return "story:" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]

    def _render_content(self, node_id: str, content: str) -> str:
        """为输出内容附上节点类型，保留实体/属性的类型信息。

        content 是唯一能传给平台回答模型的信道，节点类型（person / emotion /
        preference 等）不带出去就彻底丢失，而它正是 A（显式事实：实体与属性）与
        B（关系组合）判断「这条是事实、偏好还是情绪」的依据。
        """
        try:
            node_type = self.graph.get_node_type(node_id)
        except Exception:
            return content
        label = getattr(node_type, "value", None) or str(node_type or "")
        return f"[{label}] {content}" if label else content

    def _node_created_at(self, node_id: str) -> Optional[str]:
        """节点的记录时间，渲染成 `YYYY-MM-DD`；无时间戳则返回 None（不输出该字段）。

        契约里 `created_at` 是可选字段，但平台侧确实会用：CLBench 的
        `format_selected_memories()` 把每条记忆渲染成 `- [<created_at>] 正文`，缺失即整条
        记忆丢掉时间线索；官方 Answer 模板第 7 条还要求把 yesterday / last month 这类相对
        时间换算成日期，前提同样是记忆里带时间。

        注意语义：节点 `timestamp` 是**记录时间**（写入图谱的时刻），不是事件发生时间
        （见 dba_pipeline/llm/inference.py 顶部说明），因此这里只提供日期，不做任何换算。
        """
        try:
            ts = self.graph.graph.nodes[node_id].get("timestamp")
        except Exception:
            return None
        return self._format_created_at(ts)

    @staticmethod
    def _format_created_at(ts: Any) -> Optional[str]:
        """Unix 毫秒 → `YYYY-MM-DD`；非法/缺失返回 None（`created_at` 是可选字段）"""
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts <= 0:
            return None
        try:
            return datetime.fromtimestamp(ts / 1000.0).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None

    def _story_created_at(self, result: Dict[str, Any], doc_nodes: List[Dict[str, Any]]) -> Optional[str]:
        """叙事条目的 `created_at`：本次叙事**主证据里最新的记录时点**。

        0927（locomo1 时间锚）：此前叙事条目刻意不带 `created_at`（理由见下方调用处的注释：
        怕"多条记忆跨大时间跨度、给单一日期会误导"），但这与 `alm/config.py` 里声明的
        "`created_at` 由 `_node_created_at()` **无条件输出给平台**"相矛盾，也让相对时间类问题
        失去唯一可用的锚点——官方 Answer 模板第 7 条要求把 yesterday / last month 换算成
        日期，**前提是记忆里带时间**。locomo1（"Lost my job as a banker yesterday"，平台已带
        首条时间戳）恒 0 就是这条链路断了的后果。

        取**最大**记录时间（= 本批最新事件的时点），只做日期格式化、**不做任何时间换算**；
        语义是"这条记忆的时点"，不是"叙述里每件事都发生在这一天"。
        """
        latest = None
        for node_id, _content in (result.get("peak_memories") or []):
            try:
                ts = self.graph.graph.nodes[node_id].get("timestamp")
            except Exception:
                continue
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts <= 0:
                continue
            if latest is None or ts > latest:
                latest = ts
        # 文档块（不在图里）也参与本次叙事，其 timestamp 是入库批次时点
        for d in doc_nodes or []:
            ts = d.get("timestamp")
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts <= 0:
                continue
            if latest is None or ts > latest:
                latest = ts
        return self._format_created_at(latest)

    def _best_cosine(self, query_vec, candidates: List[Dict[str, Any]]) -> float:
        """候选与查询的最大余弦相似度（弃权判定用）"""
        if query_vec is None or not candidates:
            return 0.0
        vectors = self.vector_store.get_content_vectors([c["id"] for c in candidates])
        return max((_cosine(query_vec, v) for v in vectors), default=0.0)

    def _abstain(self, best_cos: float, reason: str) -> List[Dict[str, Any]]:
        """按契约返回空数组；同时记账与打日志"""
        if self.stats is not None:
            self.stats.record_abstain()
        logger.info(
            "空间 %s 弃权: best_cos=%.4f 硬下限=%.4f 原因=%s",
            self._masked_id(), best_cos, self.config.abstain_cosine, reason,
        )
        return []

    def search(
        self, query: str, top_k: int, options: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """检索并按 ALM 契约组装 data 数组。

        返回内容 = MCP `query_memory` 对外给出的记忆内容（即叙事），检索直接使用共享检索器，
        不做候选扩召、重排或形态切换。`options` 只是契约字段（选择题才下发，见 contract.py），
        不并入检索文本；平台若把选项**内联进 query**（实测 `scriptMem` 即如此），由
        `_strip_option_block` 在进检索前剥离——选项与作答指令属于下游 answer 模型。

        全程持有空间锁：add() 改图时持的是同一把锁，检索若不持锁就可能读到「字典
        边迭代边被改」的中间态；代价是同 user_id 的并发检索串行（跨 user_id 互不
        影响，而评测里同一 user_id 的并发度远低于跨 user_id 的并发度）。
        """
        with self._lock:
            return self._search_locked(query, top_k, options)

    def _route_purposes(self, query: str) -> Optional[Tuple[List[str], bool, bool]]:
        """时序路由：一次目的判定 →（仅当判定为时序）追加锚点/事实文本。

        返回最终 purpose 列表（供 retrieve_with_story 注入）。任何失败/退化都
        **静默退回**：抛出异常时返回 None（由检索器内部自行推断）、工具无产出时返回
        原始 purposes。

        为什么必须由 ALM 显式调用 infer_purpose：`Retriever.retrieve()` 的 `purpose`
        语义是「提供时**跳过**内部独立推断」（见 retriever.py 的 `if purpose is not None`
        分支），故显式传 purpose 就等于放弃内部推断，必须自己先把 purposes 推断出来，
        否则会完全丢失「目的驱动种子」。`seed_ids` 是内部计算的，外部无法注入种子，
        所以用时序产出驱动主通道的唯一途径就是把产出并入 purpose。

        四条约束（Plan §3.2）：
          ① 判定只看 query、一次定死：本方法只接收 query，infer_purpose 只调用一次，
             且在任何检索结果产生之前完成；
          ② 时序产出只**追加**：merged = purposes + 追加文本，不改写既有目的；
          ③ 不碰控制流：本方法不决定是否调用主通道（由调用方无条件执行）；
          ④ 全过程打日志：判定结果 / match_type / 锚点 id / 追加条数，只记数值与 ID，
             **不记 query 正文**（合规要求）。
        """
        # ① 目的判定（时序标签与 purposes 合并在同一次 LLM 调用里，不新增调用）
        try:
            info = self.retriever.inference.infer_purpose(query)
            # infer_purpose 直接返回 json.loads 的结果：合法 JSON 也可能是数组或标量，
            # 那种情况下 .get 会抛 AttributeError。放进同一个 try，按退化处理。
            purposes = list(info.get("purposes") or []) if isinstance(info, dict) else []
            is_temporal = bool(info.get("temporal")) if isinstance(info, dict) else False
            # 压缩标签与 temporal 同源：同一次 infer_purpose 调用里产出，不新增 LLM 调用
            condense = bool(info.get("condense")) if isinstance(info, dict) else False
            # 状态标签同源：问「某个属性的当前/最新取值」时为真
            state = bool(info.get("state")) if isinstance(info, dict) else False
        except Exception as exc:
            logger.warning(
                "空间 %s 目的推断失败，改由检索器内部推断: %s", self._masked_id(), exc
            )
            return None

        if not purposes:
            # 退化（LLM 返回了合法 JSON 但无 purposes）：退回内部推断，避免空目的向量
            logger.warning("空间 %s 目的推断无结果，改由检索器内部推断", self._masked_id())
            return None

        # 压缩观测锚点：`压缩` 是 LLM 严格判定（生效），`宽压缩` 是本地关键词粗判
        # （仅观测，见 _GLOBAL_VIEW_HINTS）。两者都不命中即说明 F1 题干与压缩/全局
        # 意图无关，本方向不成立。只记布尔与长度，不记正文。
        logger.info(
            "空间 %s 时序路由: 判定=%s 目的数=%d 压缩=%s 宽压缩=%s 状态=%s 检索文本=%d字",
            self._masked_id(), "时序" if is_temporal else "非时序", len(purposes),
            "是" if condense else "否",
            "是" if _looks_like_global_query(query) else "否",
            "是" if state else "否", len(query or ""),
        )
        # 时序产出只在 temporal_route 开启时才追加：只开 condense_route 时必须**不**
        # 隐式启用时序改写（temporal_route 默认 False 是"拿到 A/B 净收益证据前不改变
        # 线上行为"的冻结决定，不能被另一个开关绕过）。
        if not (is_temporal and self.config.temporal_route):
            return purposes, condense, state

        # ② 时序工具：时间锚点 → 共时事实（仅向量检索 + 图单跳，无 LLM 调用）
        try:
            res = self.retriever.temporal_lookup(
                query,
                k_seed=self.config.temporal_k_seed,
                max_facts=self.config.temporal_max_facts,
                max_anchors=self.config.temporal_max_anchors,
            )
        except Exception as exc:
            logger.warning(
                "空间 %s 时序查询失败，静默退回仅目的通道: %s", self._masked_id(), exc
            )
            return purposes, condense, state

        extra: List[str] = []
        seen_text = set()
        anchor_ids: List[str] = []
        fact_count = 0
        for group in res.get("matches") or []:
            anchor = group.get("time_anchor") or {}
            if anchor.get("id"):
                anchor_ids.append(anchor["id"])
            # 锚点自身也并入目的：查询问的往往是「何时」，锚点文本直接给出时间线索
            texts = [anchor.get("content") or ""]
            for fact in group.get("facts") or []:
                # time_consistent 为 False 表示事实自带的时期词与该锚点完全不相交
                # （疑似 TEMPORAL 边错配）。宁漏不误：这类事实不并入目的，避免引偏。
                if fact.get("time_consistent") is False:
                    continue
                texts.append(fact.get("content") or "")
            for text in texts:
                text = text.strip()
                if text and text not in seen_text:
                    seen_text.add(text)
                    extra.append(text)
            fact_count += len(group.get("facts") or [])

        logger.info(
            "空间 %s 时序查询: match_type=%s 锚点=%s 事实数=%d 可并入=%d",
            self._masked_id(), res.get("match_type"), anchor_ids, fact_count, len(extra),
        )
        if not extra:
            logger.info("空间 %s 时序路由无可用锚点，退回仅目的通道", self._masked_id())
            return purposes, condense, state

        # ③ 只追加不覆盖
        merged = purposes + extra
        logger.info(
            "空间 %s 时序路由追加: 目的 %d → %d（追加 %d 条锚点/事实文本）",
            self._masked_id(), len(purposes), len(merged), len(extra),
        )
        return merged, condense, state

    def _search_locked(self, query: str, top_k: int, options: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.retriever.path_tracker is not None:
            self.retriever.path_tracker.start_session()

        # 检索侧只消费题干：平台会把选择题的选项与作答指令**内联进 query**（见
        # _strip_option_block）。那一整坨属于下游 answer 模型；下面所有消费方——目的推断、
        # 种子/文档向量、StoryRank、模糊带判定——统一改用剥离后的检索文本，避免
        # ①种子被「六种立场描述的质心」带偏、②目的退化成"选出正确选项"这类元目的。
        # 原始 query 仍原样进日志，用于观察平台的入参形态。
        # 精确证据路由：本系统的联想设计对召回有利，但在"要一个唯一答案"的提问上，多抓的
        # 那部分就是噪声——判题模型的表现只跟**证据密度**走（实测见 config.precision_route）。
        # 判型看**整条 query**（作答指令与契约的 options 字段是最强的类型信号），但**目的向量
        # 与检索文本仍只用题干**：选项参与判型、不参与嵌入。
        #
        # 剥离只在**题面明确要求作答**时进行。不能用"有连续 A./B./C. 标号"当判据：普通列举句
        # （"Compare version A. alpha, B. beta, and C. gamma"）同样有标号，会被整体截断误伤。
        # 兜底 `or query`：若整条 query 就是选项块（剥离后为空），退回原文——否则空串进
        # embedding 会 400（`_embed` 的 400 重试到 250 字符预算后直接抛错 → /search 5xx）。
        instructed = _is_precision_query(query)
        precise = bool(options) or instructed
        retrieval_query = _retrieval_text(query)
        use_precision = self.config.precision_route and precise
        max_hops = self.config.precision_max_hops if use_precision else self.config.max_hops
        # 只收跳数不够：seed_k × expand_k 仍会顶出 150 条左右，叙事照样超预算被硬截断
        # （见 config.precision_max_peak_nodes 的标定）。故同时按 combined_score 取前 N；
        # 配 0 = 自适应（由检索层按篇幅预算反推条数，见 retriever._auto_peak_cap）。
        max_peak = ((self.config.precision_max_peak_nodes or PEAK_NODES_AUTO) if use_precision
                    else self.config.max_peak_nodes)

        # 冒烟期留痕：题面必须看到。离线复现不了平台查询，而 B2/D1/F1 的失败是否
        # 与题干形态（隐含全局视角、选项绑定的知识更新题）相关，只能从原题判断。
        # 同时记「检索用」字符数：内联选项的题会从 ~1150 掉到 ~170，是稳定可判的形态特征。
        if self.config.trace_io:
            logger.info(
                "[IO] search 入参 空间=%s top_k=%d 检索用=%d字(原=%d字) 判型=%s 精检=%s "
                "跳上限=%d query=%s\n选项:\n%s",
                self._masked_id(), top_k, len(retrieval_query), len(query or ""),
                "是" if precise else "否", "是" if use_precision else "否", max_hops,
                _clip_for_log(query, self.config.trace_max_chars),
                _clip_for_log(
                    "\n".join(f"  {o}" for o in options) if options else "  （无）",
                    self.config.trace_max_chars,
                ),
            )

        # 时序路由（Plan §3.2）+ 压缩路由：两者共用**同一次** infer_purpose 调用
        # （零新增 LLM 调用）。时序路把锚点/事实文本**追加**进 purpose 驱动主通道；
        # 压缩路只携带一个布尔标签，由 retrieve_with_story 决定 StoryRank 用哪个
        # prompt 变体——照抄 temporal_route 的形状：只改输出形态，不碰检索控制流。
        # 任一开关开启就必须由 ALM 显式推断：retrieve() 的 purpose 语义是「提供时
        # 跳过内部推断」，不先推出来会丢掉「目的驱动种子」。
        injected_purpose, condense, state = None, False, False
        if (self.config.temporal_route or self.config.condense_route
                or self.config.state_route):
            routed = self._route_purposes(retrieval_query)
            if routed is not None:
                injected_purpose, condense, state = routed
        if not self.config.condense_route:
            condense = False
        if not self.config.state_route:
            state = False

        # with_response=False：ALM 只消费 stories，GenerateResponse 的产物
        # 从不被读取，却要多打一次 LLM（约 2.7s；Search 并发可达 256）。同时主办方把
        # 「Search 阶段生成最终答案」划为红线，即便结果被丢弃也不应触发该链路。
        # 主通道无条件执行（约束 ③）：无论路由是否命中、是否失败，这里都必须走到。
        # 查询向量在检索**之前**取：文档通道要用它做向量召回，弃权判定也要用。
        # 原实现在检索之后才取，文档通道需要在检索前拿到它，故前移（同一个调用，
        # 不多花一次 embedding）。**不允许静默降级**：拿不到查询向量就返回空数组，
        # 与「库里确实没有相关记忆」在平台侧完全无法区分，等于把 embedding 故障
        # 变成一条不重试的错误答案。这里直接抛出，由 /search 转成 5xx 交给平台重试
        # （embedding 层内部已先做过有界退避重试）。
        try:
            query_vec = self.vector_store._embed(retrieval_query)
        except Exception as exc:
            logger.error(
                "空间 %s 查询向量计算失败（embedding 不可用）: %s", self._masked_id(), exc
            )
            raise RuntimeError(
                "查询向量计算失败：embedding 服务不可用，请检查 EMBEDDING_MODEL / "
                "EMBEDDING_API_BASE / EMBEDDING_API_KEY"
            ) from exc

        # 文档通道：与主通道**并行**的另一条路，只捞文档块。
        # 它不参与种子选择 / 图扩展 / 峰值检测——这正是隔离的意义：线上实测
        # 文档块一旦进图，就会把峰值拉到 hop0、`跳=1`，多跳彻底失效。
        doc_nodes = (
            self._document_lookup(query_vec, self.config.doc_lookup_k)
            if self.config.doc_route else []
        )
        if doc_nodes:
            logger.info(
                "空间 %s 文档通道: 取回 %d 块（直命中 %d）",
                self._masked_id(), len(doc_nodes),
                sum(1 for d in doc_nodes if d.get("hit")),
            )

        result = self.retriever.retrieve_with_story(
            retrieval_query,
            with_response=False,
            render_timestamps=self.config.render_timestamps,
            seed_k=self._effective_seed_k(),
            max_hops=max_hops,
            expand_k=self.config.expand_k,
            purpose=injected_purpose,
            condense=condense,
            state=state,
            # 0930：按场景/时段分段（规则 12），默认关闭；开启后跨场景不再挤进同一自然段
            scene=self.config.story_scene_paragraphs,
            max_peak_nodes=max_peak,
            # 文档块作为「原文摘录」并入 StoryRank 的输入：一次 LLM 调用统合，
            # 不让叙事与原文证据分成两段互不相干的内容。
            extra_nodes=doc_nodes,
        )
        # 约束 ④：主通道是否执行必须可观测（失效模式是静默的）
        logger.info(
            "空间 %s 主通道已执行: 峰值=%d 叙事=%d",
            self._masked_id(),
            len(result.get("peak_memories") or []),
            len(result.get("stories") or []),
        )

        items: List[Dict[str, Any]] = []

        # 只返回叙事一条。MCP 的 `query_memory` 对外给出的记忆内容就是 `stories`
        # （0927 起连 id 列表都不再产出，只余一个 used 计数），所以这是对
        # MCP 最严格的忠实映射。实测（Plan §7.10.6）：叙事首条已覆盖全部命中的 gold fact，
        # 其后的原子节点边际贡献为 0（top-1 累计命中 == top-all），而平台也没有任何检索侧
        # 指标会因条数变化——故节点条目是纯冗余，收拢掉。
        stories = result.get("stories") or []
        if stories:
            story = stories[0]
            # 0927：StoryRank 不再回 id 列表（只回 used 计数，见 story_rank 的 Returns），
            # 故 id 一律由正文派生。副作用是同一批记忆在不同轮次正文不同 → id 不稳定；
            # 但平台只用 `content` 判分，`id` 仅作标识，这个代价可以接受。
            story_id = self._story_id_from_text(story)
            entry = {"id": story_id, "content": story, "score": 1.0}
            # 0927（L2a）：叙事条目是否也带 `created_at`（本叙事主证据里最新的记录时点）。
            # **默认关闭**（`story_created_at`）：平台实测为无效——开启后 created_at 确实从
            # 46/46 None 变成 24/46 带日期，但 locomo1 的回写仍是 predicted=Yesterday./0 分，
            # 即锚点给了下游没用。按"保系统健康"（无效改动不占默认位）改由开关控制。
            if self.config.story_created_at:
                created_at = self._story_created_at(result, doc_nodes)
                if created_at:
                    entry["created_at"] = created_at
            items.append(entry)
        else:
            # 叙事缺失（LLM 调用失败等）时**不能静默返回空数组**——那与「弃权」无法区分，
            # 会把「检索到了但没整理成文」误报成「库里没有相关记忆」。
            #
            # 退回**峰值节点原文**（原来退的是"采纳节点"，而采纳 id 已随 story_rank 的
            # 改动消失；峰值节点是同一批候选里按 combined_score 排在最前的，语义等价）。
            #
            # 0927 审查 B3：**必须并上本次文档通道取回的块**。此前这里只遍历
            # `peak_memories`（其元素只由图节点构成，文档块 id `d*` 永不出现），
            # 于是文档型查询一旦叙事生成失败就只剩空数组——恰是注释声称要避免的失效。
            logger.warning("空间 %s 叙事缺失，退回峰值节点与文档块原文", self._masked_id())
            emitted: set = set()
            for node_id, _peak_content in (result.get("peak_memories") or []):
                if node_id in emitted:
                    continue
                node = self.graph.get_node(node_id)
                if not node or node.get("deprecated") or node.get("forgotten"):
                    continue
                content = node.get("content")
                if not content:
                    continue
                emitted.add(node_id)
                entry = {
                    "id": node_id,
                    "content": self._render_content(node_id, content),
                    # 占位分：最终分数由 _normalize_scores 按返回顺序折算（首条恒 1.0）
                    "score": 1.0,
                }
                created_at = self._node_created_at(node_id)
                if created_at:
                    entry["created_at"] = created_at
                items.append(entry)
            # 文档块不在图里：叙事缺失时同样要能退回它的原文，否则文档型查询
            # 一旦叙事生成失败就只剩空数组（与「弃权」不可区分）。
            for d in doc_nodes:
                did = d.get("id")
                if not did or did in emitted or not d.get("content"):
                    continue
                emitted.add(did)
                items.append({
                    "id": did,
                    "content": d["content"],
                    # 占位分：与图节点一致，最终由 _normalize_scores 折算
                    "score": 1.0,
                })

        # 契约要求返回条数不超过 top_k
        items = items[:top_k]

        # 弃权（唯一保留的 ALM 自造能力）：best_cos = max(图候选池最大余弦, 文档通道直命中最高分)。
        #
        # 为什么必须并上文档通道：文档型内容（长文档 / 抽取判空的原文）现在都住在文档
        # 通道里、不参与图检索，`_collect_par_candidates` 拿不到它们。只看图侧会把这类
        # 空间的 best_cos **系统性低估**，线上实测（2026-09-26 冒烟）后果有两个：
        #   ① 文档密集空间可能因低分触发硬下限直接弃权，文档通道找到的内容被整批丢掉；
        #   ② 更多 query 落进 [abstain_cosine, abstain_verify_hi) 模糊带 → 判定调用与
        #      弃权都变多。该轮弃权率 9/46 = 20%（改造前 1/46），其中 6 个空间同时出现
        #      「文档通道取回 + 弃权」，如 u_40***7f55 取回 23 块后 best_cos=0.5386 弃权。
        # 只取**直命中**的分数：邻块扩展项的 score 是 None，它们是上下文补充、不是命中证据。
        # 0927 起文档分已是**真余弦**（见 `VectorStore._relevance`），与图余弦同一把尺，
        # 故这里的 max() 是合法比较（此前是 1/(3−2cos)，与余弦在 cos≈0.5 处交叉，见审查 B2）。
        candidates = self._collect_par_candidates(result)
        graph_best = self._best_cosine(query_vec, candidates)
        doc_best = max((d["score"] for d in doc_nodes
                        if d.get("score") is not None), default=0.0)
        best_cos = max(graph_best, doc_best)
        if self.stats is not None:
            self.stats.record_search(best_cos)
        # 结构化检索指标。**只记数值与计数**，不记 query 正文，也不记叙事正文——
        # 与上面 record_search 同一条合规口径（见 0919 §五）。
        #
        # 平台的真实查询无法离线复现，出问题时只能靠这一行反推失败环节：
        #   候选/峰值都小  → 种子没召回（embedding 或阈值问题）
        #   跳=0           → 图扩展完全没参与，退化成纯向量
        #   路径=story     → 叙事由 LLM 正常合成
        #   路径=拼接      → **StoryRank 输出不可解析，已按相关度降级拼接前 N 条节点**。
        #                    这是最需要警惕的一态：0926 线上 7 次降级的正文都是
        #                    1.2 万~2.3 万字符的原文流水账（节点边界/边/方向全被抹平），
        #                    而聚合指标（采纳/丢弃）与正常叙事**完全无法区分**——
        #                    此前只能靠 `grep StoryRank JSON 解析失败` 计数才能发现。
        #   路径=raw       → 叙事整理失败，已退回采纳节点原文（见下方分支）
        #   返回=0         → 弃权路径，最终返回空数组
        #   喂入点/喂入边  → 实际交给 StoryRank 的点与边数量。**边数为 0 或远小于点数**
        #                    说明 StoryRank 收到的是一堆孤立节点，因果链无从表达——
        #                    这是 B2 的判决依据（对应线上 B2 连续多轮 0 分）。
        #   正文           → 返回条目的正文字符数。**0927 起用它取代 `采纳=`**：后者是
        #                    后端按二元组覆盖率算的"覆盖节点数"，但在**句式重复**的数据集上会
        #                    **饱和**——实测 529/529 全过、恒等于喂入点（审查文档 §11.4），
        #                    不可信；正文长度是可信的，且不依赖 trace_io。
        #   带比/带宽/带模式 → 寻峰容忍带的读数（0930 接入）：`带比` = 带内候选 / 候选总数，
        #                    线上此前只能事后反推，实测中位 0.81（几乎不筛）。`带模式`只在
        #                    `ALM_PEAK_TOLERANCE_RATIO > 0` 时为「相对」，即带宽随场景自适应。
        #   目的过         → 目的过滤的逐跳通过率合计（out/in）。该读数此前完全不可见，
        #                    是判断阈值 0.30 是否在窄分布上退化的唯一依据。
        _band = result.get("peak_band") or {}
        _pp = result.get("purpose_pass") or []
        _pp_in = sum(int(x.get("in", 0)) for x in _pp)
        _pp_out = sum(int(x.get("out", 0)) for x in _pp)
        logger.info(
            "空间 %s 检索: best_cos=%.4f 候选=%d 峰值=%d 跳=%d 判型=%s 精检=%s 喂入上限=%s "
            "正文=%d 返回=%d "
            "路径=%s 喂入点=%d 喂入边=%d 图cos=%.4f 文cos=%.4f 文档块=%d "
            "带比=%s 带宽=%s 带模式=%s 目的过=%s",
            self._masked_id(), best_cos, len(candidates),
            len(result.get("peak_memories") or []),
            len(result.get("hop_history") or []),
            "是" if precise else "否", "是" if use_precision else "否",
            ("不限" if not max_peak else ("自适应" if max_peak < 0 else max_peak)),
            len(items[0]["content"]) if items else 0,
            len(items),
            ("拼接" if result.get("story_degraded") else ("story" if stories else "raw")),
            result.get("path_nodes", 0), result.get("path_edges", 0),
            graph_best, doc_best, len(doc_nodes),
            _band.get("band_ratio", "—"), _band.get("delta", "—"), _band.get("mode", "—"),
            (f"{_pp_out}/{_pp_in}" if _pp_in else "—"),
        )

        # 弃权第一段（硬下限）：低于它一律弃权，连判定调用都不必发。
        if self.config.abstain_cosine > 0.0 and best_cos < self.config.abstain_cosine:
            return self._abstain(best_cos, "低于硬下限")

        # 弃权第二段（模糊带判定）：余弦落在 [abstain_cosine, abstain_verify_hi) 时，
        # 单看数值分不清「库里确实没有」与「有、但提问措辞与节点原文距离远」——实测
        # 两类样本的余弦区间重叠（见 alm/judge.py 顶部注释）。故把已排好序的 top-N
        # 交给 LLM 判一次；判否才弃权，判定失败按放行处理。
        if self.config.abstain_verify_hi > 0.0 and best_cos < self.config.abstain_verify_hi:
            verdict = self.judge.is_relevant(retrieval_query, items)
            if verdict is False:
                return self._abstain(best_cos, "模糊带判定为无相关信息")
            logger.info(
                "空间 %s 模糊带放行: best_cos=%.4f verdict=%s",
                self._masked_id(), best_cos, verdict,
            )

        # 冒烟期留痕：这一条就是回答模型真正读到的"记忆"（ALM 只返回叙事一条），
        # 也是 B2/D1/F1 三个 0 的最终证人——叙事是否已退化成"全图低分辨率概览"，
        # 只能从正文判断。同时记采纳数与喂入边型，区分「没喂因果边」与「喂了没写」。
        if self.config.trace_io:
            body = "\n".join(
                f"  [{it.get('id')}] score={it.get('score')} "
                f"created_at={it.get('created_at')} {it.get('content')}"
                for it in items
            )
            logger.info(
                # 0927 起不再打 `采纳=`（覆盖率度量在句式重复的数据集上饱和，不可信，见上一处注释）
                "[IO] search 返回 空间=%s 条数=%d 喂入点=%d 喂入边=%d 边型=%s:\n%s",
                self._masked_id(), len(items),
                result.get("path_nodes", 0), result.get("path_edges", 0),
                result.get("path_edge_types") or {}, body or "  （空）",
            )

        return self._normalize_scores(items)

    def _collect_par_candidates(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """T1 种子 + T2 图扩展的候选并集，仅供弃权判定取 best_cos 用。

        直接取 hop_history（它比 peak 容忍带更全）。返回内容不再参与排序，故这里的
        排序只影响可读性。
        """
        par_scores = result.get("peak_scores") or {}
        candidates: Dict[str, Dict[str, Any]] = {}

        for entry in result.get("hop_history") or []:
            hop = int(entry.get("hop", 0))
            for cand in entry.get("candidates") or []:
                node_id = cand.get("id")
                if not node_id or node_id in candidates:
                    continue
                content = cand.get("content") or self.graph.get_content(node_id)
                if not content:
                    continue
                candidates[node_id] = {
                    "id": node_id,
                    "content": content,
                    "par_score": float(par_scores.get(node_id, 0.0)),
                    "tier": 1 if hop == 0 else 2,
                }

        ordered = list(candidates.values())
        ordered.sort(key=lambda item: (item["tier"], -item["par_score"]))
        return ordered

    @staticmethod
    def _normalize_scores(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把重排得分归一化到 (0, 1]，保证「数值越大越相关」

        注意：此处按**单次查询**的最大值归一化，因此每次查询的首条恒为 1.0，
        分数**不跨查询可比**。若平台要求跨查询同尺度，需改用绝对分而非归一化。
        """
        if not items:
            return items
        top = max(item["score"] for item in items)
        if top <= 0:
            # 极端情况下没有可用分数，则按返回顺序给出单调递减的占位分
            total = len(items)
            for rank, item in enumerate(items):
                item["score"] = round(1.0 - rank / total, 6)
            return items
        for item in items:
            item["score"] = round(item["score"] / top, 6)
        return items

    # ---- 持久化 ----

    def save(self, force: bool = False):
        """原子写回 YAML，避免进程中断损坏主文件

        仅在图谱确有改动时落盘（`force=True` 可强制）。这不只是省 I/O：cleanup.py
        按 mtime 判断空间最后一次被写入的时间，无条件重写会让「停服」本身刷新 mtime，
        使 30 天留存清理永远判定不到过期空间。详见 __init__ 中 _dirty 的说明。
        """
        with self._lock:
            if not self._dirty and not force:
                return
            data = self.graph.to_dict()
            self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(self.yaml_path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                os.replace(tmp_path, self.yaml_path)
                self._save_index()
                # 文档区与图谱一起落盘：两者共享 _dirty，故只在本次确有写入时写。
                self._save_documents()
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            # 写盘成功后才清位：中途失败时保持 dirty，下次仍会重试落盘
            self._dirty = False

    # ---- 内部工具 ----

    def _masked_id(self) -> str:
        """日志中不输出完整 user_id（评测数据与标识不宜落入日志）"""
        if len(self.user_id) <= 8:
            return "***"
        return f"{self.user_id[:4]}***{self.user_id[-4:]}"
