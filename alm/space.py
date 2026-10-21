# SPDX-License-Identifier: AGPL-3.0-only

"""按 user_id 隔离的记忆空间。

ALM 以 user_id 为唯一检索作用域，且 `MemoryGraph.to_dict()` 的 YAML 序列化
**不保留节点 metadata**，因此这里采用「每个 user_id 一份独立图谱 + 独立向量索引」
的隔离方式：不做任何核心改动即可保证「禁止跨 user_id 返回记忆」。

隔离代价是空间数量随 user_id 增长，由上层 ALMEngine 用 LRU 控制内存驻留数。
落盘上**一个 user_id 独占一个目录**，`self.yaml_path` 是该目录内的 `graph.yaml`，
索引与文档区都从它的父目录派生——这样删除一个空间就是删除一个目录，
不会在 data_dir 下留下需要按后缀去认领的零散产物。
"""

import bisect
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
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
from alm.router import AssociationJudge

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

# evidence 档随返回附带的「类型图例」（作 `data[0]`）。
#
# 为什么必须发：evidence 把证据渲染成 `[id]（person）…` / `[from] --causal--> [to]`，
# 而 `person/causal/...` 是我们的**私有词表**；调用方的 answer 模型看不到我们的 prompt，
# 没有任何地方解释这些词的含义。
# 词表取自 dba_pipeline/core/jump_axis.py 的 NodeType(6) / RelationType(8)，渲染一律小写。
_EVIDENCE_LEGEND = (
    "[类型图例/legend] 以下为记忆证据块。节点块格式 `[id]（类型）内容`，类型："
    "person=人物、emotion=情绪、action=行为、status=状态、reason=原因、thing=实体；"
    "关系块格式 `[from] --关系--> [to]`，关系："
    "causal=因果(前者导致后者)、sequence=时序(前者先于后者)、scenario=同场景、"
    "preference=偏好指向、social=社交关系、attribute=属性归属、temporal=时间定位、"
    "taxonomic=分类归属。"
)

# 文档增量日志超过这个字节数就把日志并入条文快照（摊薄一次全量重写）。
# 文档区只增不改，故日志是权威来源、快照只是加速回放的检查点。
_DOCS_LOG_MAX_BYTES = 4 * 1024 * 1024

# 单次 rerank 请求的候选条数上限（防误配 `ALM_RERANK_CANDIDATES` 把整库一次性送上去）。
_RERANK_MAX_CANDIDATES = 200

# 「全局 / 总体视角」的本地关键词。**仅用于观测，不参与任何控制流。**
#
# 为什么要单独记一个「宽判定」：「摘要与压缩 / 长历史全局综合」的题干措辞未知。
# 它可能显式要求压缩（LLM 的 condense 标签会命中），也可能只是隐含的总体视角问法
# （如「这一路是怎么走过来的」），后者**不会**被严格要求显式措辞的 condense 捕获。
# 两种判定同时记录，一轮流量后即可判定：
#   宽判定命中、严判定不命中 → 题干是隐含全局型，该放宽触发条件；
#   两者都不命中             → 题干与压缩/全局意图无关，本方向不成立。
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

    为什么必须有界（运行实测）：`trace_io=1` 时单个日志文件 **84 MB**、约 60 秒
    写满一次轮转（100m×10）；调用方确有把整份文档当 query 送进来的轨道，单条 `[IO]`
    可达数 MB——一条记录就能顶掉一次轮转，把前面的证据冲掉。

    截断保留**头部**：形态判定（是长会话还是整段作文、query 里有没有内联选项）看的都是
    开头。尾部标注原长，便于区分"本来就短"与"被截了"。
    """
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…（留痕截断：原文共 {len(text)} 字）"


# ---- 调用方 QA 反馈护栏 ----
#
# 调用方在**每次 search 之后**会回写一条 1 消息的 Add，内容是运行元数据（实测，
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
# （如 `locomo_llm_judge`）是调用方**唯一**主动告知「这道题属于哪个数据集、怎么判分」
# 的信道——此前我们完全无法把 query 对应到能力项，这些能力项的归因一直缺这一环。
# 故：内容不进图，字段单独留痕（`[IO] qa_feedback`）。
_QA_FEEDBACK_HINTS = ("streaming online qa feedback", "streaming-online-feedback")

_QA_FIELD_RES = (
    ("qa_id", re.compile(r'"qa_id"\s*:\s*"([^"]*)"')),
    ("method", re.compile(r'"method"\s*:\s*"([^"]*)"')),
    ("score", re.compile(r'"score"\s*:\s*(-?[0-9.]+)')),
    ("question", re.compile(r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"')),
    ("predicted", re.compile(r'"predicted_answer"\s*:\s*"((?:[^"\\]|\\.)*)"')),
)

# 只含调用方自己给出的题号/方法/分数——**不含题面与正文**。合规上可常开，故承载
# 「唯一能把 query 归到能力项」的信道（题面留给 trace_io 控制的
# `[IO] qa_feedback`，但 qa_id 不能跟着一起关，否则生产运行期这些能力项的归因断掉）。
_QA_ID_KEYS = ("qa_id", "method", "score")


def _is_qa_feedback(text: str) -> bool:
    """是否为调用方的 QA 反馈回写（运行元数据，不是用户记忆）。

    判据用**调用方自己的 schema 标记**而非启发式形态：
    首行 `Streaming online QA feedback:` 或 `schema_version` 里的
    `streaming-online-feedback`。二者都是调用方逐字给出的，不存在误伤真实记忆的可能
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
    """只抽题号/方法/分数，**不含题面与正文**（可不受 trace_io 控制）"""
    parts = []
    for key, rx in _QA_FIELD_RES:
        if key not in _QA_ID_KEYS:
            continue
        m = rx.search(text or "")
        if m:
            parts.append(f"{key}={m.group(1)}")
    return "  ".join(parts) or "（字段未解析出）"


# 调用方把选择题的**选项与作答指令内联在 query 字符串里**（实测 `scriptMem` 那题：
# 全长 1147 字 = 题干 168 + 选项块 876 + 作答指令 103）。选项与作答指令是留给调用方侧
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


# 「要唯一/精确答案」的提问形态：调用方把选择题的选项与作答指令内联在 `query` 里，这本身
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
    直到进程被杀）。触发条件只是一个配错的环境变量。
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

    存在的理由是**标定弃权阈值**：ALM 真实数据不可离线观测，但调用方每次 smoke 都会打
    我们的 `/search`，把这些请求的 best_cos 记下来（只记数值，**不记 query 正文**，
    符合调用方的合规要求），跑完一轮就能看到真实查询实际落在什么区间：

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


class _BackfillGate:
    """回补槽位闸门：**按优先级**放行，而不是按到达顺序。

    为什么不用 Semaphore：Semaphore 先到先得，但回补队列里最该先跑的，是
    「刚刚有检索在等它的那个空间」——否则那次检索要白等到 `graph_wait_seconds`
    超时才落空，等图建好时已经没人要了。

    优先级由 `MemorySpace._backfill_priority`（最近一次检索的时间戳）给出，且**动态**
    生效：等待中的线程每次被唤醒都重新排序，所以「后来被检索」能立即插队。
    """

    def __init__(self, slots: int):
        self._cv = threading.Condition()
        self._free = max(1, int(slots))
        self._waiters = []       # [(seq, space)]
        self._seq = 0

    def acquire(self, space) -> None:
        with self._cv:
            self._seq += 1
            ticket = (self._seq, space)
            self._waiters.append(ticket)
            while True:
                self._waiters.sort(
                    key=lambda t: (-getattr(t[1], "_backfill_priority", 0.0), t[0])
                )
                if self._free > 0 and self._waiters[0] is ticket:
                    self._waiters.pop(0)
                    self._free -= 1
                    return
                self._cv.wait()

    def release(self) -> None:
        with self._cv:
            self._free += 1
            self._cv.notify_all()


# 直接构造 MemorySpace（不经过 Engine）时的默认闸门，例如单元测试。
_DEFAULT_BACKFILL_GATE = _BackfillGate(8)


class MemorySpace:
    """单个 user_id 的完整记忆栈：图谱 + 向量 + DBA 维护 + PAR 检索"""

    # 已处理 request_id 的**持久化**容量（与 Engine.MAX_REQUEST_LOG 同量级）。
    # 超出后按插入顺序淘汰最旧的：重试通常紧跟在失败之后，这个窗口足够覆盖。
    MAX_RID_LOG = 8192

    def __init__(
        self,
        user_id: str,
        config: ALMConfig,
        embeddings,
        llm,
        yaml_path: Path,
        stats: Optional[RetrievalStats] = None,
        reranker=None,
        backfill_gate: Optional[_BackfillGate] = None,
    ):
        if not HAS_DEPS:
            raise RuntimeError(
                f"ALM 记忆链路依赖未安装（{_IMPORT_ERROR}）；请先执行 pip install -e ."
            )

        self.user_id = user_id
        self.config = config
        self.yaml_path = Path(yaml_path)
        self.stats = stats
        # Cross-encoder 重排器（仅纯向量通道用）。None = 不重排；由 engine 按
        # `ALM_RERANK` 决定是否注入（见 dba_pipeline/rerank.py）。它失败会**静默降级**
        # 回余弦原序，故这里不做任何可用性校验。
        self.reranker = reranker
        # 串行化单个用户空间内的图谱写入与 YAML 落盘
        self._lock = threading.RLock()
        # 自上次落盘以来图谱是否被改动过。add() 置位、save() 成功后清位。
        # 作用不只是省一次写盘：留存储存期清理（alm/cleanup.py）按 mtime 判定
        # 空间最后一次被写入的时间，而 engine.close() 会对**全部驻留空间**调 save()。
        # 若无条件重写，停服这一步就会把每个空间的 mtime 刷成停服时刻，
        # 使「先停服再清理」的既定流程永远判定不到过期空间（静默空转）。
        self._dirty = False

        # 图必须先载入：归属的**兼容兜底**要按"图里有没有节点"判定（见 `_load_space_mode`）。
        self.graph = (
            load_graph(str(self.yaml_path)) if self.yaml_path.exists() else MemoryGraph()
        )

        # 按空间 mode：归属决定存在空间目录的 `mode.yaml` 里，判定后**冻结**。
        # 缺省（无该文件）时的取法见 `_load_space_mode`：新空间（无图）= 初态 `vector`
        # 零 LLM 起步；**已有图**的空间 = 兼容兜底按 `graph` 认领（防静默降级）。
        # 优先级与来源见 config.force_mode。
        self._space_mode: Optional[str] = self._load_space_mode()
        self.vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
        if not self._load_index():
            self._reindex()

        # 文档通道（异构输入路由的目标区）：与图谱**完全隔离**。
        #
        # 为什么不能塞进图：文档块之间没有记忆意义上的关联，一旦进图就会参与
        # 种子选择 / 图扩展 / 峰值检测。线上图文件实测正是这条路径把 289 个孤立
        # 文档块变成了种子层——峰值落在 hop0、`跳=1`、多跳彻底失效（32 个调用方图
        # 节点全是兜底块且边数恰好为 0）。隔离后它们不进图算法，只在读侧由
        # 专门的 document_lookup 取回。
        self.docs: List[Dict[str, Any]] = []
        self._doc_index: Dict[str, int] = {}
        self._doc_seq = 0
        # 原始消息：文档区的**权威内容**。`self.docs` 里的切片是由它**派生**的
        # 视图（见 `_chunks_from_msgs`），不作为持久化内容——切片抹平了消息边界与角色，
        # 而"重新抽取"（回补建图）与"局部建边"都要求能还原出原始消息序列。
        self.doc_msgs: List[Dict[str, Any]] = []
        # 是否见过**会话型**内容。仅当它为真时才值得跑归属判定 / 回补：
        # 纯文档型空间（长文档、抽取兜底）无论如何都该走向量通道。
        # 旧数据没有 channel 字段 → 载入时按 conv 处理（保持旧行为）。
        self._conv_seen = False
        # 已持久化到盘上的**消息**条数（快照 + 增量日志合计）。落盘只写
        # `self.doc_msgs` 中**超出这一位**的新增部分——文档区只增不改，故不需要全量重写。
        self._msgs_logged = 0
        self.doc_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
        self._load_documents()
        # 已成功处理的 request_id（内存 + `requests.log` 落盘），用于**跨重启**幂等。
        # 必须放在 `_load_documents()` 之后：旧数据的 rid 只存在于原始消息里。
        self._rids: "OrderedDict[str, None]" = self._load_rids()

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
            # 把容忍带带宽接出来（此前从未传入 → 一直吃 PeakFinder 库默认 0.10）。
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
        # 空间归属判定器：前 N 题到达时**判一次**，判完冻结。
        self.router = AssociationJudge(llm, sample_chars=config.route_sample_chars)
        # 判定用的观察窗：已到达的 query。判完即清（冻结后不再收集）。
        self._route_probe: List[str] = []
        # 归属判定**在途**标记。判定拆成"短锁取快照 → 锁外跑 LLM"两段，期间
        # `_space_mode` 仍为 None；这个标记防止并发查询在判定返回前又各自触发一次判定。
        self._route_deciding = False
        # 后台任务在跑（回补建图）。`Engine._evict_locked` 会跳过这些空间——
        # 否则空间被 LRU 换出后重建出**第二个实例**，而后台线程仍在第一个实例上
        # `save()`（整图覆盖写 YAML），把重建期间写入的内容抹掉。
        # 与 `_lease` 防的是同一类丢数据，区别只是"占用者"是后台线程而非请求线程。
        self._bg_active = False
        # 回补的**取消**标志：向量兜底后置位，循环在下一批之前退出（见 abandon_backfill）。
        self._bg_cancel = False
        # 回补闸门的排序键：最近一次检索的时间戳。越大越优先（见 _BackfillGate）。
        self._backfill_priority = 0.0
        # 连续"等满一轮仍落空"的次数。达到 `abandon_after_misses` 才真正放弃回补：
        # 等待单位是单次检索，同一空间后面的题还能再等一轮，大空间需要多轮才建得完。
        self._fallback_misses = 0
        # 全进程共享的回补槽位闸门（由 Engine 注入；直接构造时用默认值）。
        self._backfill_gate = backfill_gate or _DEFAULT_BACKFILL_GATE
        # 串行化落盘。`save()` 会被**请求线程**（engine.add 之后）与**回补线程**
        # （run_backfill 每批）**同时**调用，而它是整图 / 整文档区的覆盖写——
        # 不加锁时两线程交错写同一个 YAML 会把文件写坏（48h / 16 并发下高频）。
        # 用独立锁而非 `_lock`：落盘耗时远长于临界区，且调用点多在 `_lock` 之外，
        # 独立锁能串行化落盘而不扩大 `_lock` 的持有时长。
        self._save_lock = threading.Lock()


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

    # ---- 按空间 mode ----
    # 归属决定（vector / graph）存在空间目录的 `mode.yaml` 里，由判断器在**前 N 题**写入后
    # **冻结**——之后不再路由。为什么不能放全局 config：写侧 90% 的成本是**按空间**付的，
    # 而 `engine` 把同一个 config 实例传给每个空间，写上去会让一个空间的决定泄漏到所有空间。

    def _mode_path(self) -> Path:
        """归属文件的路径：与 `graph.yaml` / `graph.index/` / `docs.index/` 同目录"""
        return self.yaml_path.parent / "mode.yaml"

    def _load_space_mode(self) -> Optional[str]:
        """读取归属。无有效 `mode.yaml` 时按"图里有没有节点"兜底。

        - `mode.yaml` 存在且取值合法 → 以它为准（**显式决定优先**）；
        - 缺失 / 取值非法 / 读取失败 → **兼容兜底**：图里已有节点则按 `graph` 认领，
          否则返回 None（= 初态 `vector`）。

        为什么要有这条兜底：本特性上线**之前**建的图空间没有 `mode.yaml`（也包含
        "`set_space_mode` 落盘失败"这一情形），若一律按 `vector` 处理，就会把已经建好的
        图**静默降级**为纯向量——图被弃用且不再增长，全程无告警。判据用"**图非空**"
        而不是"`graph.yaml` 存在"：空图不算，新空间（无图）仍走初态 `vector`，
        所以不会把新空间误判成 graph（判断器该判的还是判）。

        ⚠️ **不落盘**：这里只推导、不写 `mode.yaml`。写回会刷新空间目录的 mtime，
        而 [alm/cleanup.py](cleanup.py) 按"目录最新 mtime"判定 30 天留存期——
        一次加载就把老空间刷成"刚刚写过"，会导致**过期空间漏删**。
        """
        path = self._mode_path()
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                mode = str(data.get("mode") or "").strip().lower()
                if mode in ("vector", "graph"):
                    return mode
                logger.warning(
                    "空间 %s mode.yaml 取值非法（%r），改按兼容兜底判定", self._masked_id(), mode
                )
        except Exception as exc:
            logger.warning(
                "空间 %s mode.yaml 读取失败，改按兼容兜底判定: %s", self._masked_id(), exc
            )
        # 兼容兜底：图非空 = 该空间确实建过图 → 认 graph，避免静默降级。
        if self.graph.node_count > 0:
            logger.warning(
                "空间 %s 无有效 mode.yaml 但图已有 %d 个节点，按 graph 认领（兼容兜底，不落盘）",
                self._masked_id(), self.graph.node_count,
            )
            return "graph"
        return None

    @property
    def space_mode(self) -> str:
        """**归属**（`mode.yaml` 里判完冻结的那个值）。无有效文件时按兼容兜底取：
        图非空 → `graph`，否则初态 `vector`（零 LLM，见 `_load_space_mode`）。

        ⚠️ 它**不等于**实际生效的模式：强制臂（`force_mode="vector"|"graph"`）下
        不判定归属，本值会停在初态 `vector`，而写侧实际可能在走 graph。
        要观测"实际在跑什么"请用 :attr:`effective_mode`。
        """
        return self._space_mode or "vector"

    @property
    def effective_mode(self) -> str:
        """**实际生效**的模式（写侧口径，与 `/add` 的判定一致）。

        与 `space_mode`（归属）刻意分开，因为两者在强制臂下会不同——
        巡检/日志只看归属会把"强制 graph 臂"误读成"没建图"。
        """
        return self._effective_write_mode()

    @property
    def bg_active(self) -> bool:
        """是否有**后台回补线程**正在该空间上运行。

        供 `ALMEngine._evict_locked` 判断"该空间能否被 LRU 换出"：回补线程不持
        `_lease`（见 `start_backfill`），若空间被换出，它会继续在旧实例上逐批整图
        覆盖写 YAML，把换出后重建实例期间写入的内容抹掉。

        ⚠️ 必须是**属性**，不能让 engine 直接猜名字：此前 engine 读 `bg_active`、
        这里存 `_bg_active`，`getattr(sp, "bg_active", False)` 恒取默认 `False`
        → 保护**静默失效**（回补中的空间照样被换出）。
        """
        return self._bg_active

    def _effective_write_mode(self) -> str:
        """本次 `/add` 实际生效的写侧模式。

        只有两个来源：**唯一开关 `force_mode`** + **按空间归属**。
        旧模式（全局默认 `write_mode`）已删除——它会让归属判定被静默盖过，见 config.force_mode。
        """
        if self.config.force_mode == "vector":
            return "documents"                      # 纯向量臂：整站不建图
        if self.config.force_mode == "graph":
            return "graph"                          # 强制图臂：复现改动前行为
        return "documents" if self.space_mode == "vector" else "graph"

    def _effective_read_mode(self) -> str:
        """本次 `/search` 实际生效的读侧模式（同上，只有两个来源）"""
        if self.config.force_mode == "vector":
            return "vector"
        if self.config.force_mode == "graph":
            return "graph"
        return "vector" if self.space_mode == "vector" else "graph"

    def set_space_mode(self, mode: str, reason: str = "") -> None:
        """写入并**冻结**归属（写入后不再路由）。

        落盘失败**不阻断**本次请求——本次仍按新 mode 执行，只是下次启动会退回初态
        `vector`。那是"少建图"，方向安全。
        """
        mode = str(mode).strip().lower()
        if mode not in ("vector", "graph"):
            raise ValueError(f"mode 非法：{mode!r}，可选 vector / graph")
        self._space_mode = mode
        payload = {
            "mode": mode,
            "reason": reason,
            "decided_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            path = self._mode_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(payload, f, allow_unicode=True)
            logger.info(
                "空间 %s 归属已定并冻结: mode=%s reason=%s",
                self._masked_id(), mode, reason or "-",
            )
        except Exception as exc:
            logger.warning(
                "空间 %s mode.yaml 落盘失败（本次仍按 %s 执行）: %s",
                self._masked_id(), mode, exc,
            )

    def _corpus_sample(self) -> str:
        """归属判定的**语料侧**输入：文档区最前面的若干字符。

        取"最前面"而不是随机采样：实测四版判定用的都是该空间开头约 700 字，
        且生产日志里调用方是**持续追加**写入的（前 700 字在该空间开始提问时已经存在）。
        用**文档区**而不是图：归属未定时该空间还没建图（初态 vector），图里是空的。
        """
        limit = self.config.route_sample_chars
        buf: List[str] = []
        total = 0
        for doc in self.docs:
            content = (doc or {}).get("content") or ""
            if not content:
                continue
            buf.append(content)
            total += len(content)
            if total >= limit:
                break
        return "\n".join(buf)[:limit]

    def _maybe_decide_space_mode(self, retrieval_query: str) -> None:
        """前 N 题到达时判一次该空间该不该建图，然后**冻结**归属。

        ⚠️ 判定 LLM 必须在**锁外**：`association_p` 是网络调用（最坏 `llm_timeout`），
        持着空间锁跑会把同 user_id 的并发 `/add` 整段堵住。故拆成两段：
          ① **短锁**：把本题计入观察窗；满则取出「语料样本 + 前 N 题」并清空窗口，
             同时置 `_route_deciding`（只消费一次，防并发重复判定）；
          ② **锁外**：`_apply_space_mode` 跑 LLM、落 `mode.yaml`、触发回补。

        判定失败与"判不准"都按 `vector` 兜底——这是**成本优先**下的显式选择
        （误判成 graph 要白付一整空间的建图钱；误判成 vector 只是可能的静默丢分）。
        """
        if self.config.force_mode != "auto" or self._space_mode is not None:
            return
        # 纯文档型空间（长文档 / 抽取兜底）**不判定、不建图**：
        # 文档通道本就是为这类语料设计的，让它们进入图链路等于绕过形态分流，
        # 既白花一次判定 LLM，又会在回补时把整篇文档重抽成孤立节点。
        # 空空间也不判——等有内容再说（否则会按空语料冻结成 vector）。
        if not self._conv_seen:
            if self.doc_msgs:
                self.set_space_mode("vector", reason="仅文档型语料（文档通道）")
            return
        with self._lock:
            # 双检：并发下可能已被前一题判掉（或已冻结），此时不再收窗口。
            if self._space_mode is not None or self._route_deciding:
                return
            self._route_probe.append(retrieval_query)
            if len(self._route_probe) < self.config.route_probe_questions:
                return
            sample = self._corpus_sample()
            questions = list(self._route_probe)
            # 判完即清：归属冻结后不再收集，避免观察窗在长会话里无限增长。
            self._route_probe.clear()
            self._route_deciding = True
        try:
            self._apply_space_mode(sample, questions)
        finally:
            self._route_deciding = False

    def _apply_space_mode(self, corpus: str, questions: List[str]) -> None:
        """跑归属判定并落盘（**锁外**，见 `_maybe_decide_space_mode`）。

        强制臂下**不判定归属**：force_mode 是实验对照，必须盖过一切，否则 A/B 时会出现
        "设了 graph 臂却被判成 vector"的静默错配。
        """
        if self.config.force_mode != "auto":
            return
        try:
            score = self.router.association_p(corpus, questions)
        except Exception as exc:      # 兜底：任何异常都不该让 /search 失败
            logger.warning("空间 %s 归属判定异常，按 vector 兜底: %s", self._masked_id(), exc)
            self.set_space_mode("vector", reason="判定异常")
            return
        if score is None:
            self.set_space_mode("vector", reason="判定失败")
            return
        # 预算闸门：与语义判定并列的第二道判据——语义回答"该不该建图"，
        # 这里回答"来不来得及建完"。超预算的空间若判成 graph，只会在检索结束后
        # 才建好（对成绩零贡献），却把整段回补的 token 照付一遍。
        pred_min = self._predict_backfill_minutes()
        if (self.config.graph_max_pred_min > 0
                and pred_min > self.config.graph_max_pred_min):
            self.set_space_mode(
                "vector",
                reason=(f"p={score:.2f} 题数={len(questions)} "
                        f"预算超限(pred={pred_min:.0f}min)"),
            )
            return
        mode = "graph" if score >= self.config.route_min_p else "vector"
        self.set_space_mode(
            mode, reason=f"p={score:.2f} 题数={len(questions)}"
        )
        if mode == "graph":
            # 判成 graph 后，把判定前落在 L1 的语料**后台**补建进图。
            # 放后台的理由见 config.backfill_enabled——同步跑会把这次 /search 拖到超时。
            self.start_backfill()

    def _graph_ready(self) -> bool:
        """图侧当前是否可用（#3 建图等待窗的判据）。

        · 回补 **running 且确有线程在跑**（`_bg_active`）：图正被逐步建起来，尚不完整
          → 不可用（调用方先等，见 `_wait_for_graph`）；
        · 回补 **done**：已建完 → 可用；
        · 其余（pending / skipped / 无进度文件 / **running 但无活动线程**）：没有正在
          进行的建图——若图里已有节点（历史已建 / 兜底认定）则可用，否则退回向量。

        ⚠️ `running 但无活动线程` 是必须单独处理的现场：回补某批异常中止、或进程重启后
        磁盘上遗留 `running`，此时**没有线程会再推进它**。若仅凭磁盘状态就判"未就绪"，
        等待窗会永远等不到 done → 每次 /search 白等满 `graph_wait_seconds`。

        `force_mode != auto` 的实验臂不走这里（见 `search()`），保持强制语义。
        """
        try:
            status = self.backfill_progress().get("status")
            if status == "done":
                return True
            if status == "running" and self._bg_active:
                return False
            return self.graph.graph.number_of_nodes() > 0
        except Exception:
            return False

    def _wait_for_graph(self, seconds: float) -> bool:
        """建图未完成时最多等 `seconds` 秒，返回等待后图是否就绪。

        **只在"回补在跑且有活动线程"时才等**：没有线程在推进，"等"只会白白拖长本次
        /search（陈旧 running 的现场，见 `_graph_ready`）。轮询间隔 0.5s，一旦回补状态
        离开 running 或线程退出就立即返回。到点仍未就绪则返回 False，由调用方退向量兜底。
        """
        if seconds <= 0:
            return self._graph_ready()
        if not (self.backfill_progress().get("status") == "running" and self._bg_active):
            return self._graph_ready()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if (self.backfill_progress().get("status") != "running"
                    or not self._bg_active):
                break
            time.sleep(0.5)
        return self._graph_ready()

    # ---- 回补建图 ----
    # 归属判定在前 N 题完成，此前该空间走 vector（零 LLM），语料全落在 L1 的原始消息里。
    # 判成 graph 之后这段语料要补建进图；形态由三个约束决定（后台 / 逐批持锁 / 可续建，
    # 见 config.backfill_enabled）。

    def _backfill_path(self) -> Path:
        return self.yaml_path.parent / "backfill.yaml"

    def backfill_progress(self) -> Dict[str, Any]:
        """读取回补进度。**任何异常都按"未补"处理**——重补一遍只是多花钱，
        漏补会让图侧永久缺内容（不可逆）。所以方向选安全的那一侧。"""
        try:
            path = self._backfill_path()
            if not path.exists():
                return {"done": 0, "status": "pending"}
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {
                "done": max(0, int(data.get("done") or 0)),
                "status": str(data.get("status") or "pending"),
            }
        except Exception as exc:
            logger.warning("空间 %s 回补进度读取失败，按未补处理: %s", self._masked_id(), exc)
            return {"done": 0, "status": "pending"}

    def _save_backfill_progress(self, done: int, status: str) -> None:
        try:
            path = self._backfill_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            # 原子写（先写 .tmp 再 os.replace）：`_wait_for_graph` 会**并发轮询**这个文件，
            # 非原子写会让轮询撞上「被截断的空文件」→ 解析成 pending → 误判图不就绪、
            # 白白退向量兜底。与 `_compact_docs` 同款两步写。
            tmp_path = path.with_name(path.name + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    {"done": int(done), "status": status,
                     "updated_at": datetime.now().isoformat(timespec="seconds")},
                    f, allow_unicode=True,
                )
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning("空间 %s 回补进度落盘失败: %s", self._masked_id(), exc)

    def backfill_pending(self) -> List[Dict[str, Any]]:
        """待补的原始消息（断点之后的部分）"""
        return self.doc_msgs[self.backfill_progress()["done"]:]

    def estimate_backfill_groups(self, pending: Optional[List[Dict[str, Any]]] = None) -> int:
        """估算要跑多少组（供启动前止损）。

        直接复用 add 的 `_split_add_batches`，所以这是**批次数本身**，不是近似——
        止损判断与实际执行不会出现口径差（两处都传同一组 `backfill_*` 覆盖值）。
        """
        records = self.doc_msgs if pending is None else pending
        # 只统计**会话型**记录：doc 记录（长文档 / 抽取兜底）永远不参与回补，
        # 否则等于绕回文档通道把整篇文档重抽成孤立节点。
        records = [r for r in records if r.get("channel") != "doc"]
        if not records:
            return 0
        return len(self._split_add_batches(
            [self._record_to_message(r) for r in records],
            size=self.config.backfill_batch_size,
            budget=self.config.backfill_max_chars,
        ))

    def _predict_backfill_minutes(
        self, pending: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """预测把待补语料建完需要多少分钟（判定时刻 / 续建时刻的预算闸门用）。

        组数直接复用 `estimate_backfill_groups`（与真实分批同一函数，不是近似），
        单组耗时取配置（实测 4~5s，含两次 LLM 调用与落盘）。

        `pending` 为 None 时按**全部原文**估算（判定时刻：图还是空的）；
        续建场景应传 `backfill_pending()`，否则会把已补部分重复计入。

        异常一律按 0 处理：宁可建图，不可因估算失败把该建的空间错判成 vector。
        """
        try:
            groups = self.estimate_backfill_groups(pending)
        except Exception:
            return 0.0
        return groups * float(self.config.backfill_group_seconds) / 60.0

    @staticmethod
    def _record_to_message(record: Dict[str, Any]) -> AddMessage:
        """把落盘的原始消息记录还原成 `AddMessage`（补建图与重新抽取的入口）"""
        return AddMessage(
            role=str(record.get("role") or "user"),
            content=str(record.get("content") or ""),
            timestamp=record.get("timestamp"),
        )

    def run_backfill(self) -> int:
        """把 L1 的待补消息**同步**补建进图，返回本次补入的消息条数。

        同步版同时是线程体与测试入口：测试直接调它（确定性），生产走 `start_backfill`。
        """
        progress = self.backfill_progress()
        if progress["status"] == "done":
            return 0
        pending = self.doc_msgs[progress["done"]:]
        if not pending:
            self._save_backfill_progress(progress["done"], "done")
            return 0

        # 只有 conv 记录需要建图。doc 记录（长文档 / 抽取兜底）**直接跳过**并计入
        # 已消费：否则等于绕过文档通道，把整篇文档重抽成孤立节点。
        conv_items = [
            (i, r) for i, r in enumerate(pending) if r.get("channel") != "doc"
        ]
        skipped_docs = len(pending) - len(conv_items)
        if not conv_items:
            self._save_backfill_progress(len(self.doc_msgs), "done")
            return 0

        # 启动前止损（与判定时刻同一把预算闸门，改用"分钟"口径）：
        # 排队期间可能又收到新语料，判定时刻的估算会过期，这里按**当前待补**复核一次。
        groups = self.estimate_backfill_groups(pending)
        pred_min = self._predict_backfill_minutes(pending)
        if (self.config.graph_max_pred_min > 0
                and pred_min > self.config.graph_max_pred_min):
            logger.warning(
                "空间 %s 回补中止（超预算）: 估算 %d 组 ≈ %.0f 分钟 > 上限 %.0f 分钟",
                self._masked_id(), groups, pred_min, self.config.graph_max_pred_min,
            )
            self._save_backfill_progress(progress["done"], "skipped")
            return 0

        logger.info("空间 %s 回补开始: 待补 %d 条会话消息 / %d 组（跳过 %d 条文档型）",
                    self._masked_id(), len(conv_items), groups, skipped_docs)
        msgs = [self._record_to_message(r) for _, r in conv_items]
        done = progress["done"]
        extracted = 0
        cursor = 0
        self._save_backfill_progress(done, "running")
        for group, n_msgs in self._split_add_batches_span(
            msgs,
            size=self.config.backfill_batch_size,
            budget=self.config.backfill_max_chars,
        ):
            # 取消检查放在**每批之前**：粒度 ≤ 1 组（约 4~5 秒），
            # 既能让槽位尽快让给下一项，又不会把正在写的一批截断成半张图。
            if self._bg_cancel:
                break
            try:
                # ② 逐批持锁：临界区 1~3 秒，与正常 add 同量级——不阻塞并发检索。
                with self._lock:
                    self._dirty = True
                    self.dba.maintain(self._format_conversation(group),
                                      timestamp=self._batch_timestamp(group))
            except Exception as exc:
                # 已补部分**保留**（进度已按批落盘），下次从断点续建。
                logger.warning("空间 %s 回补某批失败，保留已补部分: %s", self._masked_id(), exc)
                break
            # ③ 按**原始消息条数**推进（不是单元数）：长消息被切成多片时，只有整条建完
            #    才计入 extracted，跨组的半条消息本组记 0（下次整条重做，宁重不漏）。
            #    done 是 doc_msgs 的下标，需映射回"本组最后一条记录之后"，
            #    这样夹在中间的 doc 记录也会被一并标记为已消费。
            cursor += n_msgs
            extracted += n_msgs
            # n_msgs == 0：本组只消费了某条长消息的**中间片段**，该消息尚未整条建完。
            # 此时绝不能推进断点——cursor 仍是 0 时 `conv_items[cursor - 1]` 会取到
            # 列表末尾（-1），把 done 直接推到队尾，剩余内容会在重启/中止后永久跳过。
            if n_msgs:
                done = progress["done"] + conv_items[cursor - 1][0] + 1
                self._save_backfill_progress(done, "running")   # 每批落进度：重启可续
            self.save()
        if self._bg_cancel:
            # 竞态：取消信号可能与"最后一批刚写完"同时到达。此时语料其实已经补全，
            # 不能记成 skipped（那会让一张完整的图被永久判死）。
            if cursor >= len(conv_items):
                done = len(self.doc_msgs)
                self._save_backfill_progress(done, "done")
                logger.info("空间 %s 回补完成（取消信号到达时已补全）: %d 条会话消息",
                            self._masked_id(), extracted)
            else:
                # 检索已经用向量答完，继续建图只会烧钱；且半张图没有消费者，
                # 归属已被 `abandon_backfill` 翻回 vector，这里只负责收尾。
                self._save_backfill_progress(done, "skipped")
                logger.info("空间 %s 回补放弃: 已补 %d / %d 条会话消息（向量已兜底）",
                            self._masked_id(), extracted, len(conv_items))
            return extracted
        if cursor >= len(conv_items):
            done = len(self.doc_msgs)      # 收尾：尾部 doc 记录一并标记已消费
        finished = done >= len(self.doc_msgs)
        self._save_backfill_progress(done, "done" if finished else "running")
        logger.info("空间 %s 回补结束: 累计 %d / %d 条会话消息",
                    self._masked_id(), extracted, len(conv_items))
        return extracted

    def start_backfill(self) -> Optional[threading.Thread]:
        """在**后台线程**里跑回补。返回线程对象；未启动时返回 None。

        幂等：开关关闭 / 已补完 / **本实例已有线程在跑** → 直接返回，不重复起线程
        （重复起线程会让同一批消息被建两次图——宁可漏跑一次，不可重跑）。
        """
        if not self.config.backfill_enabled:
            return None
        with self._lock:
            progress = self.backfill_progress()
            # done = 已建完；skipped = 已判定放弃（止损或向量兜底），重跑只会重复撞墙。
            if progress["status"] in ("done", "skipped"):
                return None
            # 用**内存态** `_bg_active` 做"已在跑"判据，而不是磁盘状态：磁盘上的
            # `running` 可能是上次中断 / 进程重启遗留的陈旧值（那时并没有线程在推进），
            # 那种情况必须允许**续建**，否则图永久不完整（见 `_graph_ready` 的说明）。
            if self._bg_active:
                return None
            if not self.backfill_pending():
                self._save_backfill_progress(progress["done"], "done")
                return None
            # 在锁内一次定死"占用 + 落 running"，避免 /search 并发触发起出两个线程。
            # 必须在 `thread.start()` **之前**置位：否则 start() 与 worker 首次执行之间
            # 存在窗口，空间可能正好在这个窗口里被换出 → 丢数据。
            self._bg_active = True
            self._bg_cancel = False
            self._save_backfill_progress(progress["done"], "running")

        def _worker():
            # 槽位由 `_BackfillGate` 按优先级发放（最近被检索的空间先拿）。
            # 等待期间 `_bg_active` 仍为 True（该空间不可换出）——有意如此：宁可少换出，
            # 也不要回补作业被换出后继续写旧实例而丢数据。
            gate = self._backfill_gate
            # 排队期间可能已被放弃：先看一眼取消标志，避免白占一次槽位。
            if self._bg_cancel:
                self._bg_active = False
                self._save_backfill_progress(
                    self.backfill_progress().get("done", 0), "skipped"
                )
                return
            gate.acquire(self)
            try:
                self.run_backfill()
            except Exception as exc:   # 线程内异常不能冒泡（daemon 线程里冒泡只进 stderr）
                logger.warning("空间 %s 回补线程异常: %s", self._masked_id(), exc)
            finally:
                gate.release()
                self._bg_active = False   # 释放"后台在跑"标记，让该空间重新可被换出

        thread = threading.Thread(
            target=_worker, name=f"backfill-{self._masked_id()}", daemon=True
        )
        thread.start()
        return thread

    def _note_fallback(self) -> None:
        """记一次「等满等待窗仍落空」。

        只在**达到阈值**时才放弃回补：等待单位是单次检索请求，同一空间后面的题
        还能再等一轮；大空间需要 2~3 轮才建得完，第一次落空就放弃等于让它永远建不成。
        阈值 0 = 永不放弃（仅受预算闸门与止损上限约束）。
        """
        self._fallback_misses += 1
        limit = int(self.config.abandon_after_misses)
        if limit > 0 and self._fallback_misses >= limit:
            self.abandon_backfill(reason=f"连续{self._fallback_misses}次未就绪")

    def abandon_backfill(self, reason: str = "向量已兜底") -> None:
        """放弃回补，并把归属**翻回 vector**。

        为什么必须翻 mode：`_graph_ready()` 只看"图上有没有节点"，半成品图会被
        当成可用图——那比纯向量更差（图残缺、扩展路径断裂）。翻成 vector 后，
        后续检索稳定走向量通道，行为可预测。

        线程侧只置取消标志：正在跑的那一批会写完（≤1 组），再在下一批之前退出，
        这样既不会写出半个批次，也能让槽位尽快让给下一个空间。
        """
        if not self._bg_active:
            return
        # 竞态保护：线程可能刚好补完最后一批（此时 `_bg_active` 尚未复位）。
        # 这种情况**不能**降级——那张图是完整的，翻成 vector 等于白建。
        try:
            if self.backfill_progress().get("done", 0) >= len(self.doc_msgs):
                return
        except Exception:
            pass
        self._bg_cancel = True
        try:
            self.set_space_mode("vector", reason=f"回补放弃({reason})")
        except Exception as exc:
            logger.warning("空间 %s 回补放弃时改写 mode 失败: %s", self._masked_id(), exc)
        logger.info("空间 %s 回补放弃信号已发出: %s", self._masked_id(), reason)

    # ---- 向量索引持久化 ----
    # 无索引落盘时，每次空间载入（LRU 换出后再访问、或进程重启）都要把整图重新嵌入
    # 一遍——API embedding 下这是成百上千次外部调用。改为随 YAML 一起落盘索引，
    # 并用「节点集合指纹」校验一致性，不一致才回退重建。

    def _index_paths(self):
        """图谱索引与其指纹文件的路径（都在空间目录内）。

        空间目录 = `yaml_path.parent`（见 `engine._graph_path`），一个 user_id 独占一个，
        故这里只需在同一目录内区分"图谱索引"与"文档索引"，不会再与别的空间撞名。
        名字保留 `graph.index` 前缀而不是裸 `index/`：同一目录下还有 `docs.index/`，
        成对命名读起来才分得清谁是谁。
        """
        base = self.yaml_path.parent / "graph.index"
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
        """落盘 FAISS 索引与向量缓存（失败不影响主流程，仅降级为下次重建）

        ⚠️ **不要在这里加"指纹未变就跳过写盘"的优化**（2026-10-01 试过并已移除）：
        `_index_digest()` 只覆盖**节点 id 集合** + embedding 模型名 + 字符预算，
        **不含节点正文**。而 `VectorStore.save()` 写的是**本次变更的向量**
        （`_vec_log`）——当节点内容被更新而 id 不变时，digest 不变，跳过写盘会把
        **刚重新算出的新向量丢掉**，落盘索引从此保留旧文本的向量，重启后语义错位，
        且不报错。收益也接近零：`VectorStore.save()` 本身已是**增量追加分片**，
        无变更时不写任何文件，不需要上层再挡一次。
        """
        index_path, stamp_path = self._index_paths()
        digest = self._index_digest()
        try:
            self.vector_store.save(str(index_path))
            stamp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stamp_path, "w", encoding="utf-8") as f:
                yaml.dump(digest, f, allow_unicode=True)
        except Exception as exc:
            logger.warning("空间 %s 索引保存失败: %s", self._masked_id(), exc)

    # ---- 文档通道（异构输入的隔离区）----
    #
    # 背景：调用方的输入混了两类形态完全不同的数据——**对话记忆**（本系统的设计目标）
    # 与**知识库式长文档**（整部剧本、论文+系统提示词、NTSB 摘录、任务轨迹）。
    # 后者交给节点/边抽取是范畴错误。线上实测其后果：
    #   · 抽取产出为空 → 走兜底 → 原文被切成几百个孤立节点灌进图；
    #   · 32 个调用方图「节点全是兜底块、边数恰好为 0」，最大一个 289 节点 0 边
    #     （原文约 14.5 万字符，是一整部易卜生剧本）；
    #   · 这些孤立块被当成记忆种子 → 峰值落在 hop0 → `跳=1` → 图扩展零次。
    # 故文档走**独立存储 + 独立检索**：切块、向量化、按块序存，不抽节点、不建边、
    # 不进图。文档唯一的"边"就是块序，读侧用相邻块扩展来利用它。

    def _doc_paths(self):
        """文档区路径：条文 YAML + 独立向量索引目录（含指纹文件），都在空间目录内。

        必须与图谱的索引分开放：两者的 id 空间、失效条件、生命周期都不同，
        混在一起会让图谱的「索引指纹」把文档块也算进去，换一次文档就重建一次图索引。
        """
        base = self.yaml_path.parent
        return (
            base / "docs.yaml",
            base / "docs.index" / "faiss",
            base / "docs.index" / "stamp.yaml",
        )

    def _docs_log_path(self) -> Path:
        """文档区增量日志（append-only JSONL，每行一批新增条文）。

        为什么需要它：`_save_documents` 原先每次 Add 都 `yaml.dump` **整份** docs，
        而 documents 写下**全部语料**都走这条路 → 第 k 次 Add 要重写前 k 次累计的
        全部条文，总量 O(n²)。线上实测一轮运行 4,378 次 Add、终态 ~10 MB，写放大
        接近 46 GB，与已修的「索引全量重写」同源。

        文档区只增不改（`_store_documents` 只 append，没有删除 / 更新路径），故改为
        「快照 + 增量日志」：每次 Add 只追加一行，条文快照仅在日志超过
        `_DOCS_LOG_MAX_BYTES` 时重写一次（见 `_compact_docs`）。

        ⚠️ 新增落盘产物必须同步 `alm/cleanup.py`：留存清理按**空间目录**整目录删除，
        产物只要落在空间目录内就自动被覆盖；落在目录外的才会漏删。
        """
        return self.yaml_path.parent / "docs.log"

    def _replay_docs_log(self) -> None:
        """回放文档增量日志（JSONL，每行一批新增条文）。

        两条"幂等 / 自愈"设计：
          ① **按 id 去重**：合并是"先写快照、后清日志"，两步之间崩溃会让快照与日志
             内容重叠；去重后重复回放不会产生重复条文（`chunk_total` 这类计数字段才
             不会被污染）。
          ② 末尾**半行**（写入途中崩溃）解析不了 → 就地截断到最后一个完整换行，
             否则下次追加会接在残行之后，变成一条永远解析不了的脏行（此后所有条文
             都读不出来）。
        """
        log_path = self._docs_log_path()
        if not log_path.exists():
            return
        raw = log_path.read_bytes()
        known = {
            (f"{d.get('doc_id')}#{d.get('msg_index')}"
             if d.get("msg_index") is not None and d.get("doc_id") else str(d.get("id") or ""))
            for d in self.doc_msgs
        }
        # 已见过的 request_id（从快照里读回）。重启后调用方重试同一 request_id 时，
        # 该批消息会带着同一个 rid 再回放一次——整行跳过，避免重复建图/重复付费。
        # 注意不能只靠上面的 `known`（按内容去重）：重试会生成**新的** doc_id，
        # `(doc_id, msg_index)` 因此不同，内容去重挡不住。
        seen_rids = {d.get("rid") for d in self.doc_msgs if d.get("rid")}
        good = 0
        for line in raw.splitlines(keepends=True):
            if not line.strip():
                good += len(line)
                continue
            if not line.endswith(b"\n"):
                break  # 半行：丢弃
            try:
                rec = json.loads(line)
            except Exception:
                break
            rec_msgs = rec.get("msgs") or rec.get("docs") or []
            # 整批同 rid（`_store_documents` 写时逐条挂同一个值），取首条即可。
            rid = rec_msgs[0].get("rid") if rec_msgs else None
            if rid and rid in seen_rids:
                good += len(line)
                continue        # 该 Add 已回放过 → 幂等跳过
            if rid:
                seen_rids.add(rid)
            for doc in rec_msgs:
                key = (
                    f"{doc.get('doc_id')}#{doc.get('msg_index')}"
                    if doc.get("msg_index") is not None and doc.get("doc_id")
                    else str(doc.get("id") or "")
                )
                if key in known:
                    continue
                known.add(key)
                self.doc_msgs.append(doc)
            self._doc_seq = max(self._doc_seq, int(rec.get("seq") or 0))
            good += len(line)
        if good != len(raw):
            try:
                with open(log_path, "wb") as f:
                    f.write(raw[:good])
            except OSError as exc:  # 截断失败不致命：下次仍会重试
                logger.warning("空间 %s 文档日志残行截断失败: %s", self._masked_id(), exc)
            else:
                logger.warning(
                    "空间 %s 文档日志尾部残行已截断（%d → %d 字节）",
                    self._masked_id(), len(raw), good,
                )

    # ---- request_id 幂等（跨重启）----
    # 进程内的 `Engine._request_log` 重启即失；而调用方对 5xx 的有限重试经常跨重启，
    # 若不持久化，重试会被当成新请求 → graph 模式重复抽取/重复付费，vector 模式
    # 重复落 L1。这里把「已成功处理」的 request_id 落到空间目录内的小日志。

    def _rid_log_path(self) -> Path:
        """已处理 request_id 的落盘日志（空间目录内，随目录一起被留存清理）"""
        return self.yaml_path.parent / "requests.log"

    def _load_rids(self) -> "OrderedDict[str, None]":
        """载入已处理的 request_id：旧数据里挂在消息上的 + 落盘日志。

        日志按追加顺序保存，后写的更新；超出容量时淘汰最旧的，与
        `Engine.MAX_REQUEST_LOG` 同一口径（重试通常紧跟失败之后，窗口足够）。
        """
        rids: "OrderedDict[str, None]" = OrderedDict()
        for rec in self.doc_msgs:
            rid = rec.get("rid")
            if rid:
                rids[str(rid)] = None
        lines = 0
        try:
            for line in self._rid_log_path().read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rid = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rid, str) or not rid:
                    continue
                lines += 1
                if rid in rids:
                    rids.move_to_end(rid)
                else:
                    rids[rid] = None
        except OSError:
            pass
        while len(rids) > self.MAX_RID_LOG:
            rids.popitem(last=False)
        self._rid_lines = lines
        return rids

    def seen_request(self, request_id: Optional[str]) -> bool:
        """该 request_id 是否已成功处理过（跨进程重启仍有效）"""
        if not request_id:
            return False
        with self._lock:
            return request_id in self._rids

    def remember_request(self, request_id: Optional[str]) -> None:
        """在**成功落盘之后**登记 request_id，供重启后的重试幂等回放。

        必须后置：先登记再处理会在中途崩溃时把「没写成功」的请求也标成已完成，
        重试被跳过 → 静默丢内容。反过来的代价只是极端情况下重复一次（方向安全）。
        """
        if not request_id:
            return
        with self._lock:
            if request_id in self._rids:
                self._rids.move_to_end(request_id)
                return
            self._rids[request_id] = None
            while len(self._rids) > self.MAX_RID_LOG:
                self._rids.popitem(last=False)
            self._rid_lines = getattr(self, "_rid_lines", 0) + 1
            try:
                path = self._rid_log_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    # 上次写到一半崩溃会留下没有换行的残行；直接追加会把两条 rid
                    # 拼成一条无法解析的脏行（两条都失效）。先补换行再写。
                    if self._rid_log_needs_separator(path):
                        f.write("\n")
                    f.write(json.dumps(request_id, ensure_ascii=False) + "\n")
                if self._rid_lines > self.MAX_RID_LOG * 2:
                    self._compact_rids()
            except OSError as exc:
                logger.warning("空间 %s request_id 落盘失败: %s", self._masked_id(), exc)

    @staticmethod
    def _rid_log_needs_separator(path: Path) -> bool:
        """日志非空且末尾不是换行 → 追加前必须先补一个换行"""
        try:
            if not path.exists() or path.stat().st_size <= 0:
                return False
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                return f.read(1) != b"\n"
        except OSError:
            return False

    def _compact_rids(self) -> None:
        """把 request_id 日志重写为当前保留集（原子替换，容量有界）"""
        path = self._rid_log_path()
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for rid in self._rids:
                    f.write(json.dumps(rid, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
            self._rid_lines = len(self._rids)
        except OSError as exc:
            logger.warning("空间 %s request_id 日志压缩失败: %s", self._masked_id(), exc)

    def _doc_index_stamp(self) -> Dict[str, Any]:
        """文档索引的**向量语义**指纹：embedding 模型 + 单条字符预算。

        为什么不纳入块集合：块是只增不改的确定性追加序，条数差异由
        `_load_documents` 的 `ntotal` 校验（补齐尾部）处理；若把块集合也算进指纹，
        任何一次新增都会让"只补尾部"退化成全量重嵌。
        这里只负责识别「向量空间已变」——换模型或改 `_MAX_CHARS` 后，旧索引里的
        向量与新查询向量不可比，必须整体重建（与图谱索引的同一类失效）。
        """
        return {
            "version": 1,
            "embedding": self.config.embedding_model or "",
            "max_chars": getattr(self.doc_store.embeddings, "_MAX_CHARS", None),
        }

    def _load_documents(self) -> None:
        """载入文档区：条文快照 + 增量日志提供内容与块序，落盘索引提供向量。

        索引**与当前 embedding 配置一致**时直接复用（不重新 embedding）；
        索引缺失、损坏、指纹不符或无法校验时才回退全量重嵌。
        指纹只管"向量是否同一语义空间"，块数差异仍走下面的补齐/重建分支。
        """
        docs_yaml, docs_index, docs_stamp = self._doc_paths()
        try:
            if docs_yaml.exists():
                data = yaml.safe_load(docs_yaml.read_text(encoding="utf-8")) or {}
                # 兼容旧快照：此前落盘的是 `docs`（切片），之后是 `msgs`（原始消息）。
                # 旧数据按"一条切片=一条消息"读入——内容不丢，只是丢掉了 role / 边界。
                self.doc_msgs = list(data.get("msgs") or data.get("docs") or [])
                self._doc_seq = int(data.get("seq") or 0)
        except Exception as exc:
            logger.warning("空间 %s 文档区读取失败，按空处理: %s", self._masked_id(), exc)
            self.doc_msgs, self._doc_seq = [], 0
        # 回放增量日志（快照之后新追加的消息都在这里）。
        self._replay_docs_log()
        # 是否包含会话型内容：决定该空间要不要参与归属判定与回补。
        # 旧数据没有 channel 字段 → 按 conv 处理（保持旧行为）。
        self._conv_seen = any(
            str(r.get("channel") or "conv") != "doc" for r in self.doc_msgs
        )
        # 切片是**派生视图**：由原始消息在载入时重建。id 由 (doc_id, 批内块序)
        # 决定 → 重建结果与落盘索引**逐条对齐**，故条数校验照旧成立。
        self.docs = self._chunks_from_msgs(self.doc_msgs)
        self._doc_index = {d["id"]: i for i, d in enumerate(self.docs)}
        self._msgs_logged = len(self.doc_msgs)
        if not self.docs:
            return

        if docs_index.exists():
            # 先用指纹确认落盘向量与当前 embedding 处于**同一语义空间**。
            # 缺失指纹（旧数据）也按不一致处理：宁可重建一次，也不要把旧模型的向量
            # 与新模型的查询向量混用（同维度时全程静默，检索质量崩掉且无告警）。
            stamp_ok = False
            try:
                stamp = (
                    yaml.safe_load(docs_stamp.read_text(encoding="utf-8")) or {}
                    if docs_stamp.exists() else None
                )
                current = self._doc_index_stamp()
                stamp_ok = stamp == current
                if not stamp_ok:
                    logger.warning(
                        "空间 %s 文档索引指纹不符或缺失（落盘=%s 当前=%s），改为重建",
                        self._masked_id(), stamp, current,
                    )
            except Exception as exc:
                logger.warning("空间 %s 文档索引指纹读取失败，改为重建: %s",
                               self._masked_id(), exc)

            if stamp_ok:
                try:
                    self.doc_store.load(str(docs_index), embeddings=self.doc_store.embeddings)
                    ntotal = getattr(getattr(self.doc_store.store, "index", None), "ntotal", None)
                    if ntotal == len(self.docs):
                        logger.info("空间 %s 文档区载入: %d 块（复用落盘索引）",
                                    self._masked_id(), len(self.docs))
                        return
                    # 索引**落后**于条文（条 > 索引）：典型成因是某次写入的 embedding 失败——
                    # 消息进了条文、向量没进索引（见 `_store_documents` 的顺序约束）。
                    # 切片是**确定性的追加序**（id = `{doc_id}c{块序}`，见 `_chunks_from_msgs`），
                    # 所以索引里装的就是 `self.docs` 的前 ntotal 块 → **只补缺的尾部**即可。
                    # 旧实现这里直接重建整个文档区：4 千块的空间等于数千次 embedding，
                    # 既烧 token，又会把瞬时并发顶上去、反过来诱发超时/限流。
                    if isinstance(ntotal, int) and 0 <= ntotal < len(self.docs):
                        missing = self.docs[ntotal:]
                        self.doc_store.add_memories(
                            [d["id"] for d in missing], [d["content"] for d in missing]
                        )
                        logger.warning(
                            "空间 %s 文档区索引落后，已补齐尾部: 索引 %d → %d（补 %d 块）",
                            self._masked_id(), ntotal, len(self.docs), len(missing),
                        )
                        return
                    # 索引**超前**于条文（条 < 索引）或读不到 ntotal：追加无法修正（FAISS 删条代价高），
                    # 仍回退全量重建。
                    logger.warning(
                        "空间 %s 文档区索引条数 %s 与条文 %d 不匹配，改为重建",
                        self._masked_id(), ntotal, len(self.docs),
                    )
                except Exception as exc:
                    logger.warning("空间 %s 文档索引加载失败，改为重建: %s",
                                   self._masked_id(), exc)
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
        """落盘文档条文与索引（失败只降级为下次重嵌，不影响主流程）

        条文**只追加新增部分**（`self.doc_msgs[self._msgs_logged:]`）：原先每次 Add 都
        `yaml.dump` 整份 docs，而 documents 写下全部语料都走这条路 → 写放大 O(n²)
        （见 `_docs_log_path`）。快照只在日志涨到 `_DOCS_LOG_MAX_BYTES` 时才重写一次。
        """
        if not self.doc_msgs:
            return
        docs_yaml, docs_index, docs_stamp = self._doc_paths()
        try:
            docs_yaml.parent.mkdir(parents=True, exist_ok=True)
            new_msgs = self.doc_msgs[self._msgs_logged:]
            if new_msgs:
                self._append_docs_log(new_msgs)
            self.doc_store.save(str(docs_index))
            # 指纹必须在索引**写成功之后**落：中途失败时保持旧指纹，下次载入会发现
            # 不一致并重建（宁可重嵌，也不复用半截索引）。
            docs_stamp.parent.mkdir(parents=True, exist_ok=True)
            with open(docs_stamp, "w", encoding="utf-8") as f:
                yaml.dump(self._doc_index_stamp(), f, allow_unicode=True)
        except Exception as exc:
            logger.warning("空间 %s 文档区保存失败: %s", self._masked_id(), exc)

    def _append_docs_log(self, new_msgs: List[Dict[str, Any]]) -> None:
        """把新增原文**消息**追加成日志的一行（一条 Add 一批，写量与本次新增成正比）"""
        log_path = self._docs_log_path()
        line = json.dumps({"seq": self._doc_seq, "msgs": new_msgs}, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._msgs_logged = len(self.doc_msgs)
        try:
            if log_path.stat().st_size > _DOCS_LOG_MAX_BYTES:
                self._compact_docs()
        except OSError:
            pass

    def _compact_docs(self) -> None:
        """把增量日志并入条文快照，然后清空日志（摊薄一次全量重写）。

        **先原子写快照，后删日志**：两步之间崩溃时快照与日志内容重叠，由
        `_replay_docs_log` 的按 id 去重兜住（幂等），不会产生重复条文。
        """
        docs_yaml, _, _ = self._doc_paths()
        tmp_path = docs_yaml.with_name(docs_yaml.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump({"seq": self._doc_seq, "msgs": self.doc_msgs}, f, allow_unicode=True)
        os.replace(tmp_path, docs_yaml)
        try:
            self._docs_log_path().unlink()
        except OSError:
            pass
        logger.info("空间 %s 文档区日志已并入快照: %d 条消息", self._masked_id(), len(self.doc_msgs))

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

    def _chunks_from_msgs(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """由原始消息**派生**切片。

        为什么切片不再是持久化内容：
          · 它抹平了**消息边界与角色**（role / timestamp）——而"回补建图"要在归属判成
            graph 后重新抽取这段语料，抽取的指代消解正依赖这两样；
          · 「局部建边」也只能从**原始消息**取上下文，从切片取会复现块图（0 边、峰落 hop0）。
        故切片降级为**向量索引的派生视图**，载入时重建。

        重建必须**确定性**：id = `{doc_id}c{批内块序:04d}` —— 与落盘时的算法逐字相同，
        所以 `_load_documents` 里的"条数必须等于文档数"校验照旧成立（不一致才重建索引）。
        块序在**单条消息内部**连续编号（不跨消息），与 `_store_documents` 的原语义一致：
        跨消息连续编号会让邻块扩展把两篇不同文档的末尾与开头拼在一起。
        """
        by_doc: Dict[str, List[tuple]] = {}
        order: List[str] = []
        for msg in msgs:
            did = str(msg.get("doc_id") or "")
            if not did:
                continue
            if did not in by_doc:
                by_doc[did] = []
                order.append(did)
            text = (msg.get("content") or "").strip()
            if not text:
                continue
            for part in _split_by_chars(text, self.config.doc_chunk_chars):
                by_doc[did].append((part, msg.get("timestamp")))
        chunks: List[Dict[str, Any]] = []
        for did in order:
            parts = by_doc[did]
            for i, (content, ts) in enumerate(parts):
                chunks.append({
                    "id": f"{did}c{i:04d}",
                    "doc_id": did,
                    "chunk_index": i,
                    "chunk_total": len(parts),
                    "content": content,
                    "timestamp": ts,
                })
        return chunks

    def _store_documents(
        self,
        messages: List[AddMessage],
        batch_ts: Optional[int] = None,
        rid: Optional[str] = None,
        channel: str = "conv",
    ) -> List[str]:
        """把原文切块存入文档区：不抽节点、不建边、不进图。

        它是 `_divert_raw()` 的落点，承接三类内容（见该方法）：极端长输入、
        抽取判定无记忆价值的原文、超量溢出。**注意它接的不一定是"文档"**——
        名字来自最初的场景（调用方把整部剧本/论文当一次 Add 灌进来），但凡是没能
        变成图节点的原文都从这里落，语义是"非图内容区"。

        为什么不做抽取：这类内容的节点抽取产出恒为空（`create=0`），旧路径因此走兜底，
        把原文切成几百个孤立节点灌进图 → 被当成记忆种子 → 峰值落在 hop0 → `跳=1`。
        这里的正确表示就是「有序的块 + 向量」，块之间唯一的语义关系是**块序**
        （读侧用邻块扩展利用它）。
        """
        # 原始消息：role / 时间戳 / 消息边界**原样保留**，这是 L1 的权威内容。
        # id 前缀刻意用 d（图节点是 n*），保证两个 id 空间永不重叠——读侧靠前缀区分
        # 「这是文档块」还是「这是图节点」。
        doc_id = f"d{self._doc_seq}"
        self._doc_seq += 1
        msgs = [
            {
                "doc_id": doc_id,
                "msg_index": i,
                "role": getattr(msg, "role", None),
                "content": text,
                "timestamp": msg.timestamp if msg.timestamp is not None else batch_ts,
                # 调用方 request_id 落到**消息**上（该批每条相同）。挂消息而非挂
                # 日志行，是为了让 `_append_docs_log` 与 `_compact_docs` 零改动——
                # rid 随 msgs 自动进日志、自动进快照，compact 后幂等信息不会丢。
                # rid 由调用方**显式传入**（不读 self 上的状态），见 `add()` 的说明。
                "rid": rid,
                # 内容通道：conv = 会话型（判定前暂存 L1，判成 graph 后要被回补建图）；
                # doc = 文档型 / 抽取失败兜底（**永不回补**，否则等于绕过文档通道）。
                "channel": channel,
            }
            for i, (msg, text) in enumerate(
                ((m, (m.content or "").strip()) for m in messages)
            )
            if text
        ]
        if not msgs:
            return []
        # 切片是派生视图：内存里照旧维护 `self.docs`（读侧两条检索路径都只读它），
        # 但**落盘的是 `msgs`**。
        new_docs = self._chunks_from_msgs(msgs)
        # **先写向量、后登记条文**，顺序不可颠倒：`VectorStore.add_memories` 是
        # 「先批量 embedding，再改内部状态」（见 store.py），嵌入失败时它不留半条状态；
        # 若反过来先 `extend` 条文再嵌入，一次失败就会把"没有向量的消息"永久留在条文里
        # → 条文 > 索引 → 该空间下次载入必须补/重建（见 `_load_documents`）。
        self.doc_store.add_memories(
            [d["id"] for d in new_docs], [d["content"] for d in new_docs]
        )
        self.doc_msgs.extend(msgs)
        if channel != "doc":
            self._conv_seen = True
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
            # 邻块扩展必须限制在**同一份文档**内：`self.docs` 是所有文档块扁平拼接的，
            # 只做 pos±radius 会让某文档末块把**下一份文档首块**当成上下文取回。
            hit_doc_id = self.docs[pos].get("doc_id")
            for off in range(-radius, radius + 1):
                j = pos + off
                if not (0 <= j < len(self.docs)):
                    continue
                doc = self.docs[j]
                if doc.get("doc_id") != hit_doc_id:
                    continue
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

    def _pure_vector_search(self, query_vec, top_k: int) -> List[Dict[str, Any]]:
        """纯向量检索：图节点区 + 文档区各取 top-k，按分数合并。

        这是 `force_mode="vector"` 的实现，也是 `force_mode="graph"` 之外**唯一**的读侧路径。

        为什么必须把三层一起关掉：目的推断、图扩展、StoryRank 合计只占 5.3% 的 LLM 调用
        （见 config.force_mode），但它们正是"图到底帮没帮上忙"的全部来源。
        做对照实验时若只关其中一层，就分不清增益来自哪一层。

        与 `_document_lookup` 的差别：那条是**文档通道**（带邻块扩展，且只捞文档区），
        这条是**全库直查**——图节点与文档块合在一个排序里，谁分高谁上。

        **只做召回、不做重排**：开了重排时这里多召回 `rerank_candidates` 条候选，重排由
        调用方在**锁外**执行（见 `search()`）——重排是网络调用，持锁会阻塞同 user_id 的 /add。
        返回条数因此可能是 `rerank_candidates`（> top_k），由重排那一步收敛到 top_k。
        """
        # 召回深度：不重排时 = top_k；重排时先多取（重排器再收敛到 top_k）。
        fetch_k = top_k
        if self.reranker is not None:
            # 上限兜底：防止误配把整库一次性送进 rerank（单请求体积与耗时都不可控）。
            wanted = min(int(self.config.rerank_candidates or top_k), _RERANK_MAX_CANDIDATES)
            fetch_k = max(top_k, wanted)

        merged: List[tuple] = []
        for store in (self.vector_store, self.doc_store):
            if store is None or getattr(store, "store", None) is None:
                continue
            try:
                hits = store.search_by_vector(np.asarray(query_vec, dtype=float), k=fetch_k)
            except Exception as exc:
                logger.warning("空间 %s 纯向量检索失败: %s", self._masked_id(), exc)
                continue
            for mid, score in hits:
                merged.append((float(score), str(mid)))

        merged.sort(key=lambda x: -x[0])

        items: List[Dict[str, Any]] = []
        emitted: set = set()
        for score, mid in merged:
            if len(items) >= fetch_k:
                break
            if mid in emitted:
                continue
            emitted.add(mid)

            pos = self._doc_index.get(mid)
            if pos is not None:
                doc = self.docs[pos]
                content = doc.get("content")
                ntype = "原文摘录"
            else:
                node = self.graph.get_node(mid)
                if not node or node.get("deprecated") or node.get("forgotten"):
                    continue
                content = node.get("content")
                ntype = str(node.get("node_type") or "")
            if not content:
                continue

            # `id` 直接用节点/块自身的 id（不是从正文派生的 hash）：纯向量路径下
            # 同一条记忆每轮返回的 id 都相同，比叙事路径的 `_story_id_from_text`
            # 稳定——这也是这条路线的一个附带好处。
            items.append({"id": mid, "content": content, "score": score, "node_type": ntype})
        return items

    def _rerank_items(
        self, query: str, items: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """用 cross-encoder 重排候选并按 `top_k` 截断（**必须在空间锁外**调用）。

        **静默降级**：任何失败（网络 / 非 200 / 结构不可解析）都退回余弦原序，只记
        warning——重排是排序优化，不能让它把 /search 拖垮或变成 5xx。
        `score` 仍保留**余弦原值**（跨轮稳定、可与其他路径比较）；列表顺序才是重排结果。

        无论走哪条分支都收敛到 `top_k`：调用方（`_pure_vector_search`）在开启重排时会
        多召回候选，契约的"返回 ≤ top_k"由这里保证。
        """
        if len(items) <= 1:
            return items[:top_k]
        try:
            order = self.reranker.rerank(query, [it.get("content") or "" for it in items])
        except Exception as exc:
            logger.warning("空间 %s rerank 异常，保留余弦原序: %s", self._masked_id(), exc)
            return items[:top_k]
        if not order:
            return items[:top_k]

        ranked: List[Dict[str, Any]] = []
        taken: set = set()
        for idx, _score in order:
            if 0 <= idx < len(items) and idx not in taken:
                taken.add(idx)
                ranked.append(items[idx])
        # 重排器只返回它打过分的那些；其余按原序补在后面（保证不丢条）。
        for i, it in enumerate(items):
            if i not in taken:
                ranked.append(it)
        logger.info(
            "空间 %s rerank: 候选=%d → 返回=%d（cross-encoder 重排，score 仍为余弦）",
            self._masked_id(), len(items), min(len(ranked), top_k),
        )
        return ranked[:top_k]

    # ---- 写入 ----

    def add(self, messages: List[AddMessage], request_id: Optional[str] = None) -> Dict[str, int]:
        """同步写入一批消息：维护完成后才返回（对齐 ALM 契约）

        入口先按**形态**路由（`_classify_batch`，纯计数、不调 LLM）：
          · 文档型 → 文档通道（切块 + 向量，不抽节点、不建边，见 `_store_documents`）
          · 会话型 → DBA 链路，并按 `add_batch_size` **分批**逐组维护
          · 空内容 → 丢弃
        """
        limit = self.config.max_messages_per_add

        # 跨重启幂等：同一 request_id 已成功处理过 → 直接回放"已存储"的结果。
        # 进程内的 `Engine._request_log` 只覆盖同一次运行，重启后的重试靠这里拦下，
        # 避免 graph 模式重复抽取/重复付费、vector 模式重复落 L1。
        if self.seen_request(request_id):
            logger.info(
                "空间 %s 重复 request_id，跳过维护（跨重启幂等）", self._masked_id()
            )
            return {"messages": len(messages), "nodes": 0}

        # 调用方 request_id（幂等持久化）：落进 `_store_documents` 的**每条消息**，
        # 从而随 `docs.log` / 快照一起持久化——重启后回放仍能按 rid 去重。
        # 为什么需要：`Engine._request_log` 是**内存态**，重启即失；而一次运行里
        # 重启过 2 次，调用方重试同一 request_id 会被当成新请求 → **重复建图、重复付费**。
        #
        # ⚠️ rid 必须是**局部变量显式下传**，不能挂到 `self` 上：`engine._lease` 只做
        # 引用计数、不串行化同一 user_id 的 Add，两批 Add 会在**锁外**交错；挂实例属性
        # 时后到的请求会覆盖先到者的值 → 两批都打成同一个 rid → 重启回放时
        # `_replay_docs_log` 按 rid **整行跳过** → 先到那批 L1 原文静默丢失。
        rid = request_id

        # 调用方 QA 反馈护栏（判据与依据见 _is_qa_feedback）。必须在 payload/overflow 切分
        # **之前**对整批过滤：否则落在第 limit 条起的反馈会经 overflow → _divert_raw
        # 作为文档块入库。这类消息既不是记忆内容，又会以近乎 1.0 的
        # 余弦抢走同题的种子位。
        # 过滤后整批为空时，下面的形态判定会判 "noise" 并正常返回，不会因为
        # "这次 Add 什么也没建"而报错。
        feedback = [m.content for m in messages if _is_qa_feedback(m.content)]
        if feedback:
            logger.warning(
                "空间 %s 丢弃调用方 QA 反馈 %d/%d 条（运行元数据，不入图）",
                self._masked_id(), len(feedback), len(messages),
            )
            # qa_id / method / score 是调用方唯一主动给出的"题目归属"信道，必须留痕。
            # 独立信道**不受 trace_io 控制**：只含调用方自己的题号/方法/分数，不含题面与
            # 正文，合规上可常开。若与 [IO] 一起被关掉，生产运行期的题目归因就断了。
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
        # 形态判定要在被 `documents` 模式覆盖**之前**存一份：覆盖只表示"这一批暂不进图"，
        # 不代表它天生是文档。回补建图与归属判定都依赖这个原始形态。
        raw_kind = kind
        # 写侧模式：`documents` 表示**整个空间不建图**。
        # 它盖过形态判定——因为这一档的意义正是"这个语料到底值不值得建图"，
        # 此时不该再让形态判据把任何一批放回图链路。噪音仍照常丢弃。
        # 见 config.force_mode：写侧占 LLM 调用 94.7%，不建图 = 零 LLM 调用。
        if self._effective_write_mode() == "documents" and kind != "noise":
            kind = "document"

        # 冒烟期留痕：只记长度时看不出调用方送进来的文本**是什么形态**（长会话占比、
        # 是否整段作文、triage 该不该放行），而这些正是兜底率与建图质量的判据。
        # 默认关闭，且仅在 ALM 冒烟的公开测试数据上开启（见 config.trace_io）。
        if self.config.trace_io:
            logger.info(
                "[IO] add 入参 空间=%s 模式=%s 形态=%s 消息=%d 首条时间戳=%s 原文:\n%s",
                self._masked_id(), self._effective_write_mode(), kind, len(payload), batch_ts,
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
            # 文档通道**零 LLM**，条数上限（`max_messages_per_add`）对它不适用——上限
            # 只为控制抽取/连边的 LLM 成本。故必须落**整批**：否则写侧 `documents`
            # 模式（含 `force_mode=vector`、以及新空间的初态）下，每条 Add 的第 `limit`
            # 条起会被静默丢弃（既不进图也不进文档区），而契约要求"每次 Add 的消息已
            # 完整存储且可被检索"。会话分支的 overflow 走 `_divert_raw`，语义一致。
            if overflow:
                logger.warning(
                    "空间 %s 单次 Add %d 条超过维护上限 %d，超出 %d 条一并落文档通道"
                    "（文档通道零 LLM，不丢内容）",
                    self._masked_id(), len(messages), limit, len(overflow),
                )
            with self._lock:
                # 先置位再写：中途抛错时保持 dirty，让退出路径仍会落盘。
                self._dirty = True
                created = self._store_documents(
                    messages, batch_ts, rid=rid,
                    channel="doc" if raw_kind == "document" else "conv",
                )
            logger.info(
                "空间 %s 文档通道: messages=%d 字符=%d 块=%d（不建节点、不建边）",
                self._masked_id(), len(messages),
                sum(len(m.content or "") for m in messages), len(created),
            )
            self._trace_created(created)
            return {"messages": len(messages), "nodes": len(created)}

        # ---- 会话通道：按 add_batch_size 分批，逐组维护 ----
        # 为什么必须分批：整批一次性交给 maintain 等价于「M=全部」。
        # 实测 M=9 已让已有图上的耗时翻倍、误 deprecate 率升高；另测
        # 「整段一次灌入 → 压缩更深、联想路径断裂」。线上图文件侧的证据更硬：
        # 值变更（update/deprecate）在 >6 条的 73 个批次里**一次都没出现**，
        # 而 8 次 update 全落在 3~6 条批次、4 次 deprecate 全落在 1~2 条批次。
        groups = self._split_add_batches(payload)
        created = []
        with self._lock:
            # 走到会话通道即证明该空间有对话型内容——归属判定与回补只对这类空间有意义。
            self._conv_seen = True
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
                    grp_created = self._divert_raw(group, grp_ts, rid=rid)
                created.extend(grp_created)
                for op in ((result or {}).get("ops") or {}).get("node_ops") or []:
                    # `action` 省略即 create——
                    # 不比这一下的话 `acts` 会**恒为全 0**，而这正是上面那条日志的内容。
                    action = (op or {}).get("action") or "create"
                    if action in acts:
                        acts[action] += 1
            if overflow:
                # 条数上限只用于控制 LLM 成本，**不等于可以丢弃内容**：契约要求每次
                # Add 的消息已完整存储。超出部分改走无需 LLM 的兜底落库。
                logger.warning(
                    "空间 %s 单次 Add %d 条超过维护上限 %d，超出 %d 条改走兜底落库",
                    self._masked_id(), len(messages), limit, len(overflow),
                )
                # 溢出的是**会话内容**（本该进图，只是超过单次维护上限），标 conv：
                # 若后续被回补，仍应建图；与"抽取失败兜底"的 doc 区分开。
                created = created + self._divert_raw(
                    overflow, batch_ts, rid=rid, channel="conv"
                )

        # 节点操作计数与兜底原因已在上面的分组循环里聚合（acts / fallback_reasons）。
        #
        # 为什么要记 ops_upd / ops_dep：「新值覆盖与当前状态」这类查询连续多轮接近 0 分，
        # 而写侧的值变更处理已离线验证是正确的（真实模型 4/5 会发 deprecate+create，
        # 且顺序正确）。线上实测把疑点钉在了**批次规模**上——值变更只在 ≤6 条的批次里
        # 出现，>6 条一次都没有——故本行同时记「分批数」，便于直接对照。
        logger.info(
            "空间 %s 写入完成: messages=%d chars=%d max_msg=%d 分批=%d 兜底=%s "
            "nodes=%d ops_create=%d ops_upd=%d ops_dep=%d",
            self._masked_id(), len(messages),
            # 只记长度，不记正文：用于观测调用方单次 Add 的**形态**（长会话占比、
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

    def _split_add_batches(
        self,
        messages: List[AddMessage],
        size: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> List[List[AddMessage]]:
        """把一次 Add 的消息切成若干组，交给 DBA **逐组**维护。

        **两重约束，缺一不可**：
          · 条数 `add_batch_size`（默认 6）——对齐 M=6。注意只扫了
            M=3/6/9，且实验语料是**本地短对话**，M 的语义是"多少轮对话"；
          · 字符数 `add_max_chars`（默认 1500）——因为**条数不约束长度**。调用方实测
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

        `size` / `budget` 为**显式覆盖**（#4）：回补建图与在线 Add 的批次口径解耦——
        在线 Add 用 `add_batch_size/add_max_chars`（对齐 M=6，管的是对话
        轮次、切细以求抽取质量），而回补是**一次性把 L1 存量**灌进图（调用方单请求上限
        30 分钟），用更粗的 `backfill_batch_size/backfill_max_chars` 摊薄每批的固定开销
        （每批一次 LLM 抽取、一次落盘），否则 400 条按每 6 条切要跑 60+ 批。None 时沿用
        在线口径（调用方显式传值，见 `run_backfill` / `estimate_backfill_groups`）。
        """
        return [g for g, _ in self._split_add_batches_span(messages, size, budget)]

    def _split_add_batches_span(
        self,
        messages: List[AddMessage],
        size: Optional[int] = None,
        budget: Optional[int] = None,
    ) -> List[Tuple[List[AddMessage], int]]:
        """同 `_split_add_batches`，但每组附带**该组覆盖的原始消息条数**。

        为什么需要：切分单元可能是整条消息，也可能是某条长消息被 `budget` 切出的片段。
        `run_backfill` 的断点 `done` 是 `doc_msgs` 的**消息下标**，只能按消息条数推进；
        若按单元数推进，长消息被切碎后 `done` 会越过真实下标，重启续建就会**跳过**尚未
        建图的消息（丢内容）。

        组边界落在某条消息中间时，该消息**不计入本组**（其最后一片落在后面的组里）——
        于是重启会把它整条重做，方向是"宁重不漏"（重复内容由语义去重兜住）。
        """
        size = (self.config.add_batch_size if size is None else size) or 0
        budget = (self.config.add_max_chars if budget is None else budget) or 0
        if size <= 0 and budget <= 0:
            return [(list(messages), len(messages))]
        if not messages:
            return []

        # ① 单条超预算 → 先在消息内按句末标点切（仅此时才切分，否则整条进贪心累计）
        # `origin[i]` = 第 i 个单元来自第几条原始消息（非递减）。
        units: List[AddMessage] = []
        origin: List[int] = []
        for idx, m in enumerate(messages):
            text = m.content or ""
            if budget > 0 and len(text) > budget:
                pieces = list(_split_by_chars(text, budget))
                units.extend(
                    AddMessage(role=m.role, content=p, timestamp=m.timestamp) for p in pieces
                )
                origin.extend([idx] * len(pieces))
            else:
                units.append(m)
                origin.append(idx)

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

        # 每条原始消息的**最后一片**在 units 中的下标（origin 非递减，顺序赋值即末次）
        msg_end = [0] * len(messages)
        for i, o in enumerate(origin):
            msg_end[o] = i

        # 逐组换算"本组新增了多少条**整条完成**的消息"：消费到第 cum 个单元时，
        # 末片下标 < cum 的消息即已整条建完。各组的增量之和 = 总消息数。
        spanned: List[Tuple[List[AddMessage], int]] = []
        cum = 0
        prev_full = 0
        for g in groups:
            cum += len(g)
            full = bisect.bisect_left(msg_end, cum)
            spanned.append((g, full - prev_full))
            prev_full = full
        return spanned

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
        阈值提到 2.0（相当于关闭去重），会把同一批内容以原文形式重复写入。调用方重试
        同一次 Add 时正好走这条链：去重命中 → created 为空 → 兜底关去重 → 必然重复。

        返回值是可观测性的一部分：调用方流量下兜底率偏高时，必须能区分是
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
        # ⚠️ `action` 省略即 create（输出瘦身）。
        # 这里**必须同口径**，否则会误判并触发一次错误的兜底：
        #   "有 create 但全被去重合并了"（created_ids 为空，`created` 为假）
        #   → `op.get("action") == "create"` 现在恒为假 → 掉到下面判成 "no_create"
        #   → `_divert_raw` 把**已经成功合并**的原文又抄一份进文档区
        #   → 重复内容 + 无效 embedding + 每次触发一条 WARNING。
        # 也就是说：这是"语义从必填变可省略"时最容易漏的那类读取点——它不在
        # graph_builder 里，而在调用方的判定逻辑里。
        def _act(op: Dict[str, Any]) -> str:
            return (op or {}).get("action") or "create"

        if any(_act(op) == "create" for op in node_ops):
            return None
        if not any(_act(op) in ("update", "deprecate", "fix_type") for op in node_ops):
            return "no_create"
        return None

    def _divert_raw(
        self,
        messages: List[AddMessage],
        batch_ts: Optional[int] = None,
        rid: Optional[str] = None,
        channel: str = "doc",
    ) -> List[str]:
        """存放**未经抽取**的原文内容（契约要求「已完整存储且可被检索」）。

        调用点有三处，语义相同——内容没能变成图节点：
          ① 形态判定为文档型（极端长输入，先去保护 LLM）
          ② 抽取判定无记忆价值：triage SKIP / no_ops / no_create
          ③ 超过 `max_messages_per_add` 的溢出部分

        为什么默认送**文档通道**而不是像原来那样切成孤立节点灌进图：实测那些孤立块
        会被当成记忆种子 → 峰值落在 hop0 → `跳=1` → 多跳彻底失效；线上 32 个调用方图
        「节点全是兜底块、边恰好为 0」正是由此而来。文档通道同样可检索（向量 + 邻块
        扩展），但不参与种子 / 扩展 / 峰值。

        `doc_route=false` 时退回旧行为（图内兜底块），用于 A/B 对照。
        """
        if self.config.doc_route:
            # 兜底原文默认标 doc：**永不回补建图**，否则等于绕过文档通道。
            # 只有"超量溢出"这类本应进图的会话内容才显式传 conv（见调用点）。
            return self._store_documents(messages, batch_ts, rid=rid, channel=channel)
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
          梯度全部为空；
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

        此前这里还有一个"对采纳节点集合取哈希"的版本（id 更稳定）；StoryRank 停止
        回 id 列表后它失去了输入，只剩这一个。稳定性由正文保证：同一批记忆 + 同一问题
        在同一作用域下正文一致，id 即一致。
        """
        return "story:" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]

    def _render_evidence_node(self, node_id: str, content: str) -> str:
        """证据块的节点行：`[id]（node_type）正文`。

        为什么带 `id`（而不只是类型）：evidence 档会**单独回传关系边**
        （`[from] --rel--> [to]`），边引用的是节点 id；节点行不带 id，边就成了悬空
        引用、下游无法把"关系"对回"事实"。格式与既有的 raw 块逐字一致。
        """
        try:
            node_type = self.graph.get_node_type(node_id)
            label = getattr(node_type, "value", None) or str(node_type or "")
        except Exception:
            label = ""
        return f"[{node_id}]（{label}）{content}" if label else f"[{node_id}] {content}"

    def _evidence_items(
        self, result: Dict[str, Any], doc_nodes: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """evidence 档的证据块组装：峰值节点原文 + **关系边** + 文档块。

        - 节点行带 `[id]（类型）` 前缀（见 `_render_evidence_node`）；
        - **关系边**取自 `result["path"]["edges"]`（检索器回传的图内路径），渲染成
          `[from] --rel--> [to]`（`is_reverse` 时两端互换）；只保留**两端都在已返回
          节点集合内**的边，避免悬空引用。这是 raw 块的信息面（节点+边）——
          移除 StoryRank 后若只剩裸节点，`路径与因果`就无从表达。
        - 顶部预算：给边预留 `min(边数, top_k//2)` 个位置，否则节点一多（线上峰值常
          上百）会把边全部挤掉，等于没传。
        """
        nodes: List[Dict[str, Any]] = []
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
                "content": self._render_evidence_node(node_id, content),
            }
            created_at = self._node_created_at(node_id)
            if created_at:
                entry["created_at"] = created_at
            nodes.append(entry)
        # 文档块不在图里：evidence 档 / 叙事缺失兜底都要能回传它的原文，
        # 否则文档型查询会只剩空数组（与「弃权」不可区分）。
        for d in doc_nodes or []:
            did = d.get("id")
            if not did or did in emitted or not d.get("content"):
                continue
            emitted.add(did)
            nodes.append({"id": did, "content": d["content"]})

        # 先把边按「两端都在候选集合内」收集起来（不落地），**等节点按预算截断后**
        # 再用最终保留的节点集过滤。顺序不能反：此前先用截断前的 emitted 过滤，
        # 节点随后被 top_k 截掉，于是返回的边引用了未返回的节点 id（悬空引用）。
        edge_tuples: List[tuple] = []
        for e in (result.get("path") or {}).get("edges") or []:
            e = e or {}
            frm, to = e.get("from"), e.get("to")
            if not frm or not to or frm not in emitted or to not in emitted:
                continue
            rel = e.get("rel_type") or "?"
            if e.get("is_reverse"):
                frm, to = to, frm
            edge_tuples.append((frm, to, rel))

        # 图例作 `data[0]`：显式给出 `[id]（类型）` / `--关系-->` 的词表含义。
        # 只在 top_k 有余量时发——top_k≤2 时图例会挤掉证据，宁可不发。
        head: List[Dict[str, Any]] = []
        if self.config.evidence_legend and top_k >= 3:
            head.append({"id": "rules", "content": _EVIDENCE_LEGEND})

        all_nodes = list(nodes)
        reserve = min(len(edge_tuples), max(0, top_k // 2))
        nodes = all_nodes[: max(0, top_k - len(head) - reserve)]

        kept_ids = {n["id"] for n in nodes}
        edges: List[Dict[str, Any]] = []
        seen_edge: set = set()
        for frm, to, rel in edge_tuples:
            if frm not in kept_ids or to not in kept_ids:
                continue
            key = (frm, to, rel)
            if key in seen_edge:
                continue
            seen_edge.add(key)
            edges.append({
                "id": f"e:{frm}->{to}:{rel}",
                "content": f"[{frm}] --{rel}--> [{to}]",
            })
        # 边因端点被 top_k 截掉而丢弃时，会留下空位：把原本为边预留、但最终没用上
        # 的节点预算补回来。只**追加节点**不会让已有边悬空（边已按 kept_ids 校验），
        # 也保证返回条数尽量贴近 top_k（此前可能明显少于 top_k）。
        spare = top_k - len(head) - len(nodes) - len(edges)
        if spare > 0:
            nodes.extend(all_nodes[len(nodes): len(nodes) + spare])
        return (head + nodes + edges)[:top_k]

    def _node_created_at(self, node_id: str) -> Optional[str]:
        """节点的记录时间，渲染成 `YYYY-MM-DD`；无时间戳则返回 None（不输出该字段）。

        契约里 `created_at` 是可选字段，但调用方侧确实会用：CLBench 的
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

        此前叙事条目刻意不带 `created_at`（理由见下方调用处的注释：
        怕"多条记忆跨大时间跨度、给单一日期会误导"），但这与 `alm/config.py` 里声明的
        "`created_at` 由 `_node_created_at()` **无条件输出给调用方**"相矛盾，也让相对时间类问题
        失去唯一可用的锚点——官方 Answer 模板第 7 条要求把 yesterday / last month 换算成
        日期，**前提是记忆里带时间**。locomo1（"Lost my job as a banker yesterday"，调用方已带
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
        不并入检索文本；调用方若把选项**内联进 query**（实测 `scriptMem` 即如此），由
        `_strip_option_block` 在进检索前剥离——选项与作答指令属于下游 answer 模型。

        检索主体仍全程持有空间锁（add() 改图时持的是同一把锁，不持锁就可能读到
        「字典边迭代边被改」的中间态；代价是同 user_id 的并发检索串行）。但
        **归属判定与目的推断这两次 LLM 在锁外**（见下），只有真正的检索遍历落在锁内。
        """
        # 读侧模式的**归属判定**必须在目的推断之前、且**都在锁外**：
        #   · 判定前该空间读侧是 `vector`（不推断目的，零 LLM）；
        #   · 判定当场切 `graph` 后才需要目的 → 若把目的推断留在锁内（旧写法），
        #     判定那一题会持锁连做两次 LLM（association + infer_purpose），把同
        #     user_id 的并发 /add 堵住（回归）。
        # 故顺序固定为：**定类型（锁外）→ 读一次模式 → 目的推断（锁外）→ 进锁检索**。
        # 只读一次模式，也顺带消除了"search() 与 _search_locked 各读一次、中途被判定
        # 改写导致两次不一致"的隐患。
        # TopK 压制（#2）：调用方契约给到 100，但没有任何场景需要 100 条——它只会把
        # 叙事撑爆预算、把 LLM（目的推断/建图）的输入 token 抬高。这里按
        # `search_max_items`（默认 25）硬封顶，早于任何 embedding / LLM 调用。
        requested_top_k = top_k
        top_k = max(1, min(int(top_k), self.config.search_max_items))

        retrieval_query = _retrieval_text(query)
        # 回补闸门的优先级：刚被检索的空间最该先建图（见 _BackfillGate）。
        self._backfill_priority = time.time()
        self._maybe_decide_space_mode(retrieval_query)

        # 重启恢复：磁盘上遗留 `running`、但本进程**没有线程在推进**的空间，
        # 若此刻正好被检索到，就地在后台续建。惰性恢复而不是启动横扫，
        # 免得启动时把几百个空间一次性读进内存。
        # 预算复核由 `run_backfill` 在启动时统一做（唯一一处闸门），这里不再重复。
        if (self.config.backfill_enabled
                and self._effective_read_mode() == "graph"
                and not self._bg_active
                and self.backfill_progress().get("status") == "running"):
            self.start_backfill()

        # 建图等待窗（#3）：判成 graph 后，图是后台回补（backfill）**逐步**建起来的，
        # 期间直接检索会拿到残缺的图。正确语义是——**等图建完**：回补在跑就轮询等待，
        # 到 `graph_wait_seconds` 仍未就绪才退回纯向量兜底；回补不在跑（没有在建图）
        # 时不等，仅按当前图是否可用决定。force_mode 非 auto 是实验对照臂，不介入。
        force_vector = False
        if (self.config.force_mode == "auto"
                and self._effective_read_mode() != "vector"
                and not self._graph_ready()):
            force_vector = not self._wait_for_graph(self.config.graph_wait_seconds)

        # 兜底止损：第一次落空只计数（后面的题还能再等一轮），达到阈值才放弃回补。
        if force_vector:
            self._note_fallback()

        purpose_info: Optional[Dict[str, Any]] = None
        if self._effective_read_mode() != "vector" and not force_vector:
            purpose_info = self._infer_purpose_raw(retrieval_query)
        if force_vector:
            logger.info(
                "空间 %s 建图未就绪，超时兜底纯向量: 等待上限=%.1fs",
                self._masked_id(), self.config.graph_wait_seconds,
            )
        if top_k != requested_top_k:
            logger.info(
                "空间 %s TopK 压制: 请求=%d 生效=%d 上限=%d",
                self._masked_id(), requested_top_k, top_k, self.config.search_max_items,
            )
        with self._lock:
            items, finalize = self._search_locked(
                query, top_k, options, purpose_info=purpose_info, force_vector=force_vector
            )

        # 弃权模糊带的二次判定（一次 LLM 调用）必须在空间锁**外**：与归属判定 /
        # 目的推断同一口径——持锁跑网络调用会把同 user_id 的并发 /add 整段堵住
        # （最坏 `llm_timeout` × 重试）。检索阶段只负责把判定所需的上下文带出来。
        if finalize is not None:
            if finalize.get("needs_judge"):
                verdict = self.judge.is_relevant(finalize["query"], items)
                if verdict is False:
                    items = self._abstain(finalize["best_cos"], "模糊带判定为无相关信息")
                else:
                    logger.info(
                        "空间 %s 模糊带放行: best_cos=%.4f verdict=%s",
                        self._masked_id(), finalize["best_cos"], verdict,
                    )
            # 冒烟期留痕：这一条就是回答模型真正读到的"记忆"（ALM 只返回叙事一条），
            # 也是这些能力项连续 0 分的最终证人——叙事是否已退化成"全图低分辨率概览"，
            # 只能从正文判断。放在弃权判定**之后**，保证日志与实际返回一致。
            if self.config.trace_io:
                body = "\n".join(
                    f"  [{it.get('id')}] score={it.get('score')} "
                    f"created_at={it.get('created_at')} {it.get('content')}"
                    for it in items
                )
                logger.info(
                    # 不再打 `采纳=`（覆盖率度量在句式重复的数据集上饱和，不可信，见上一处注释）
                    "[IO] search 返回 空间=%s 条数=%d 喂入点=%d 喂入边=%d 边型=%s:\n%s",
                    self._masked_id(), len(items),
                    finalize.get("path_nodes", 0), finalize.get("path_edges", 0),
                    finalize.get("edge_types") or {}, body or "  （空）",
                )

        # Cross-encoder 重排（#5）必须在空间锁**外**：它是网络调用（被节流时最坏等
        # `throttle_max_wait`），持锁会阻塞同 user_id 的并发 /add——与「把目的推断
        # 移出锁」同一类问题。只作用于**纯向量通道**（图/证据档的候选已由种子+图扩展+
        # 峰值检测筛过，不叠加）。它自带静默降级，失败即保留余弦序。
        if self.reranker is not None and (
            force_vector or self._effective_read_mode() == "vector"
        ):
            items = self._rerank_items(retrieval_query, items, top_k)
        return items

    def _infer_purpose_raw(self, query: str) -> Optional[Dict[str, Any]]:
        """目的推断的**裸调用**：只依赖 `query`，不含任何锁内状态 → 可在锁外调。

        单独拆出来的理由见 `search()`：它是读侧唯一必做的 LLM，必须离开空间锁。
        失败返回 None（调用方按"退化"处理，退回检索器内部推断）。
        """
        try:
            return self.retriever.inference.infer_purpose(query)
        except Exception as exc:
            logger.warning(
                "空间 %s 目的推断失败，改由检索器内部推断: %s", self._masked_id(), exc
            )
            return None

    def _route_purposes(
        self, query: str, purpose_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[List[str], bool, bool]]:
        """时序路由：一次目的判定 →（仅当判定为时序）追加锚点/事实文本。

        返回最终 purpose 列表（供 retrieve_with_story 注入）。任何失败/退化都
        **静默退回**：抛出异常时返回 None（由检索器内部自行推断）、工具无产出时返回
        原始 purposes。

        为什么必须由 ALM 显式调用 infer_purpose：`Retriever.retrieve()` 的 `purpose`
        语义是「提供时**跳过**内部独立推断」（见 retriever.py 的 `if purpose is not None`
        分支），故显式传 purpose 就等于放弃内部推断，必须自己先把 purposes 推断出来，
        否则会完全丢失「目的驱动种子」。`seed_ids` 是内部计算的，外部无法注入种子，
        所以用时序产出驱动主通道的唯一途径就是把产出并入 purpose。

        四条约束：
          ① 判定只看 query、一次定死：本方法只接收 query，infer_purpose 只调用一次，
             且在任何检索结果产生之前完成；
          ② 时序产出只**追加**：merged = purposes + 追加文本，不改写既有目的；
          ③ 不碰控制流：本方法不决定是否调用主通道（由调用方无条件执行）；
          ④ 全过程打日志：判定结果 / match_type / 锚点 id / 追加条数，只记数值与 ID，
             **不记 query 正文**（合规要求）。
        """
        # ① 目的判定（时序标签与 purposes 合并在同一次 LLM 调用里，不新增调用）
        # 结果通常已由 `search()` 在**锁外**算好并传入；未传（直接调用本方法）时就地
        # 补算，保持原语义。`_infer_purpose_raw` 自己不抛异常，故下面的 try 只负责兜住
        # "返回了非 dict 的合法 JSON" 那一类。
        info = purpose_info if purpose_info is not None else self._infer_purpose_raw(query)
        try:
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
        # （仅观测，见 _GLOBAL_VIEW_HINTS）。两者都不命中即说明题干与压缩/全局
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

    def _search_locked(
        self,
        query: str,
        top_k: int,
        options: Optional[List[str]] = None,
        purpose_info: Optional[Dict[str, Any]] = None,
        force_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.retriever.path_tracker is not None:
            self.retriever.path_tracker.start_session()

        # 检索侧只消费题干：调用方会把选择题的选项与作答指令**内联进 query**（见
        # _strip_option_block）。那一整坨属于下游 answer 模型；下面所有消费方——目的推断、
        # 种子/文档向量、StoryRank、模糊带判定——统一改用剥离后的检索文本，避免
        # ①种子被「六种立场描述的质心」带偏、②目的退化成"选出正确选项"这类元目的。
        # 原始 query 仍原样进日志，用于观察调用方的入参形态。
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

        # 冒烟期留痕：题面必须看到。离线复现不了调用方查询，而这些能力项的失败是否
        # 与题干形态（隐含全局视角、选项绑定的知识更新题）相关，只能从原题判断。
        # 同时记「检索用」字符数：内联选项的题会从 ~1150 掉到 ~170，是稳定可判的形态特征。
        if self.config.trace_io:
            logger.info(
                "[IO] search 入参 空间=%s 模式=%s top_k=%d 检索用=%d字(原=%d字) 判型=%s 精检=%s "
                "跳上限=%d query=%s\n选项:\n%s",
                self._masked_id(), self._effective_read_mode(), top_k,
                len(retrieval_query), len(query or ""),
                "是" if precise else "否", "是" if use_precision else "否", max_hops,
                _clip_for_log(query, self.config.trace_max_chars),
                _clip_for_log(
                    "\n".join(f"  {o}" for o in options) if options else "  （无）",
                    self.config.trace_max_chars,
                ),
            )

        # 时序路由 + 压缩路由：两者共用**同一次** infer_purpose 调用
        # （零新增 LLM 调用）。时序路把锚点/事实文本**追加**进 purpose 驱动主通道；
        # 压缩路只携带一个布尔标签，由 retrieve_with_story 决定 StoryRank 用哪个
        # prompt 变体——照抄 temporal_route 的形状：只改输出形态，不碰检索控制流。
        # 任一开关开启就必须由 ALM 显式推断：retrieve() 的 purpose 语义是「提供时
        # 跳过内部推断」，不先推出来会丢掉「目的驱动种子」。
        injected_purpose, condense, state = None, False, False
        if (self.config.temporal_route or self.config.condense_route
                or self.config.state_route):
            routed = self._route_purposes(retrieval_query, purpose_info)
            if routed is not None:
                injected_purpose, condense, state = routed
        else:
            # 三个 route 开关全关时，本方法不再需要 purposes 去带时序/压缩标签，但主通道
            # **仍要** purposes 驱动种子。不注入的话 `retrieve()` 会在**锁内**自行推断
            # （retriever.py:523）——那正是要消除的持锁 LLM。这里把锁外已算好的
            # purposes 注入，令 `retrieve()` 走「提供即跳过内部推断」分支。注入列表与
            # 内部推断同源、同文本，检索行为不变。
            if isinstance(purpose_info, dict):
                injected_purpose = list(purpose_info.get("purposes") or []) or None
        if not self.config.condense_route:
            condense = False
        if not self.config.state_route:
            state = False

        # with_response=False：ALM 只消费 stories，GenerateResponse 的产物
        # 从不被读取，却要多打一次 LLM（约 2.7s；Search 并发可达 256）。同时规范要求
        # 「Search 阶段不生成最终答案」，即便结果被丢弃也不应触发该链路。
        # 主通道无条件执行（约束 ③）：无论路由是否命中、是否失败，这里都必须走到。
        # 查询向量在检索**之前**取：文档通道要用它做向量召回，弃权判定也要用。
        # 原实现在检索之后才取，文档通道需要在检索前拿到它，故前移（同一个调用，
        # 不多花一次 embedding）。**不允许静默降级**：拿不到查询向量就返回空数组，
        # 与「库里确实没有相关记忆」在调用方侧完全无法区分，等于把 embedding 故障
        # 变成一条不重试的错误答案。这里直接抛出，由 /search 转成 5xx 交给调用方重试
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

        # 归属判定已在 `search()` 里、**进锁之前**完成（`_maybe_decide_space_mode`）：
        # 判定 LLM 与它触发的目的推断都不该持锁（见 search() 的说明）。到这里 `_space_mode`
        # 要么已冻结、要么仍是未定（本轮按初态 vector 走），本方法只需按当前模式分派。

        # 读侧模式：`vector` = 纯向量直查，**跳过目的推断 / 图扩展 / StoryRank**。
        # 放在这个位置是有意的：query 向量已经算好，而三条 LLM 路径都还没进——
        # 所以这一档下 /search 的 LLM 调用数为 0。
        # 它同时是 `force_mode="vector"` 下唯一可用的检索路径（那时图里没有节点）。
        # `force_vector`（#3）：归属是 graph 但图尚未建完且等超时 → 同样退到这条路。
        if force_vector or self._effective_read_mode() == "vector":
            items = self._pure_vector_search(query_vec, top_k)
            logger.info(
                "空间 %s 纯向量通道: top_k=%d 候选=%d（无目的推断、无图扩展、无叙事）%s",
                self._masked_id(), top_k, len(items),
                "（建图未就绪兜底）" if force_vector else "",
            )
            return items, None

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
            # 按场景/时段分段（规则 12），默认关闭；开启后跨场景不再挤进同一自然段
            scene=self.config.story_scene_paragraphs,
            max_peak_nodes=max_peak,
            # 文档块作为「原文摘录」并入 StoryRank 的输入：一次 LLM 调用统合，
            # 不让叙事与原文证据分成两段互不相干的内容。
            extra_nodes=doc_nodes,
            # `search_shape="evidence"`：**不调 StoryRank**——图只负责召回，
            # 表述交给 answer 模型。此时 `stories` 恒为空，组装自然落到下方"证据块"
            # 分支（峰值节点原文 + 文档块）。
            story=self.config.search_shape != "evidence",
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
        # （连 id 列表都不再产出，只余一个 used 计数），所以这是对
        # MCP 最严格的忠实映射。实测：叙事首条已覆盖全部命中的 gold fact，
        # 其后的原子节点边际贡献为 0（top-1 累计命中 == top-all），而调用方也没有任何检索侧
        # 指标会因条数变化——故节点条目是纯冗余，收拢掉。
        #
        # `search_shape="evidence"` 时**不走这一支**——`stories` 恒为空（StoryRank
        # 根本没被调用），直接落到下方的"证据块"分支。那条分支此前只是"叙事失败"的
        # 兜底，现在同时是新形态的正式实现：峰值节点原文 + 文档块，按 combined_score
        # 排序、确定性去重、带 `[node_type]` 前缀、受 top_k 与 max_peak 约束。
        stories = result.get("stories") or []
        if stories:
            story = stories[0]
            # StoryRank 不再回 id 列表（只回 used 计数，见 story_rank 的 Returns），
            # 故 id 一律由正文派生。副作用是同一批记忆在不同轮次正文不同 → id 不稳定；
            # 但调用方只用 `content` 判分，`id` 仅作标识，这个代价可以接受。
            story_id = self._story_id_from_text(story)
            entry = {"id": story_id, "content": story}
            # 叙事条目是否也带 `created_at`（本叙事主证据里最新的记录时点）。
            # **默认关闭**（`story_created_at`）：调用方实测为无效——开启后 created_at 确实从
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
            # **必须并上本次文档通道取回的块**。此前这里只遍历
            # `peak_memories`（其元素只由图节点构成，文档块 id `d*` 永不出现），
            # 于是文档型查询一旦叙事生成失败就只剩空数组——恰是注释声称要避免的失效。
            if self.config.search_shape != "evidence":
                logger.warning("空间 %s 叙事缺失，退回峰值节点与文档块原文", self._masked_id())
            # 证据块 = 峰值节点原文（带 `[id]（类型）` 前缀）+ 关系边 + 文档块。
            # **关系边必须单独回传**——移除 StoryRank 后若只剩裸节点，"因果/时序/路径"
            # 这类关系信息就丢了（正是 raw 块要保留的信息面）。
            items = self._evidence_items(result, doc_nodes, top_k)

        # 新形态下的正常路径不是失效——所以**不打 WARNING**（失效告警必须保持可计数）。
        if self.config.search_shape == "evidence":
            n_edges = sum(1 for it in items if str(it.get("id", "")).startswith("e:"))
            n_rules = sum(1 for it in items if it.get("id") == "rules")
            logger.info(
                "空间 %s 输出形态=evidence：直出证据块 %d 条（节点 %d + 边 %d + 图例 %d，未调 StoryRank）",
                self._masked_id(), len(items),
                len(items) - n_edges - n_rules, n_edges, n_rules,
            )

        # 契约要求返回条数不超过 top_k
        items = items[:top_k]

        # 弃权（唯一保留的 ALM 自造能力）：best_cos = max(图候选池最大余弦, 文档通道直命中最高分)。
        #
        # 为什么必须并上文档通道：文档型内容（长文档 / 抽取判空的原文）现在都住在文档
        # 通道里、不参与图检索，`_collect_par_candidates` 拿不到它们。只看图侧会把这类
        # 空间的 best_cos **系统性低估**，线上实测（冒烟）后果有两个：
        #   ① 文档密集空间可能因低分触发硬下限直接弃权，文档通道找到的内容被整批丢掉；
        #   ② 更多 query 落进 [abstain_cosine, abstain_verify_hi) 模糊带 → 判定调用与
        #      弃权都变多。该轮弃权率 9/46 = 20%（改造前 1/46），其中 6 个空间同时出现
        #      「文档通道取回 + 弃权」，如 u_40***7f55 取回 23 块后 best_cos=0.5386 弃权。
        # 只取**直命中**的分数：邻块扩展项的 score 是 None，它们是上下文补充、不是命中证据。
        # 文档分已是**真余弦**（见 `VectorStore._relevance`），与图余弦同一把尺，
        # 故这里的 max() 是合法比较（此前是 1/(3−2cos)，与余弦在 cos≈0.5 处交叉）。
        candidates = self._collect_par_candidates(result)
        graph_best = self._best_cosine(query_vec, candidates)
        doc_best = max((d["score"] for d in doc_nodes
                        if d.get("score") is not None), default=0.0)
        best_cos = max(graph_best, doc_best)
        if self.stats is not None:
            self.stats.record_search(best_cos)
        # 结构化检索指标。**只记数值与计数**，不记 query 正文，也不记叙事正文——
        # 与上面 record_search 同一条合规口径。
        #
        # 调用方的真实查询无法离线复现，出问题时只能靠这一行反推失败环节：
        #   候选/峰值都小  → 种子没召回（embedding 或阈值问题）
        #   跳=0           → 图扩展完全没参与，退化成纯向量
        #   路径=story     → 叙事由 LLM 正常合成
        #   路径=拼接      → **StoryRank 输出不可解析，已按相关度降级拼接前 N 条节点**。
        #                    这是最需要警惕的一态：线上 7 次降级的正文都是
        #                    1.2 万~2.3 万字符的原文流水账（节点边界/边/方向全被抹平），
        #                    而聚合指标（采纳/丢弃）与正常叙事**完全无法区分**——
        #                    此前只能靠 `grep StoryRank JSON 解析失败` 计数才能发现。
        #   路径=raw       → 叙事整理失败，已退回采纳节点原文（见下方分支）
        #   返回=0         → 弃权路径，最终返回空数组
        #   喂入点/喂入边  → 实际交给 StoryRank 的点与边数量。**边数为 0 或远小于点数**
        #                    说明 StoryRank 收到的是一堆孤立节点，因果链无从表达——
        #                    这是该能力项的判决依据（对应线上该项连续多轮 0 分）。
        #   正文           → 返回条目的正文字符数。**用它取代 `采纳=`**：后者是
        #                    后端按二元组覆盖率算的"覆盖节点数"，但在**句式重复**的数据集上会
        #                    **饱和**——实测 529/529 全过、恒等于喂入点，
        #                    不可信；正文长度是可信的，且不依赖 trace_io。
        #   带比/带宽/带模式 → 寻峰容忍带的读数：`带比` = 带内候选 / 候选总数，
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
            ("证据" if self.config.search_shape == "evidence" else
             ("拼接" if result.get("story_degraded") else ("story" if stories else "raw"))),
            result.get("path_nodes", 0), result.get("path_edges", 0),
            graph_best, doc_best, len(doc_nodes),
            _band.get("band_ratio", "—"), _band.get("delta", "—"), _band.get("mode", "—"),
            (f"{_pp_out}/{_pp_in}" if _pp_in else "—"),
        )

        # 弃权第一段（硬下限）：低于它一律弃权，连判定调用都不必发。
        if self.config.abstain_cosine > 0.0 and best_cos < self.config.abstain_cosine:
            return self._abstain(best_cos, "低于硬下限"), None

        # 弃权第二段（模糊带判定）**不在锁内执行**：它是一次 LLM 调用（最坏
        # `llm_timeout` × 重试），持锁会把同 user_id 的并发 /add 堵住。这里只把
        # 判定所需的上下文带出锁，由 `search()` 在锁外调用 `judge.is_relevant()`；
        # 判否才弃权，判定失败按放行处理。
        finalize: Optional[Dict[str, Any]] = {
            "best_cos": best_cos,
            "query": retrieval_query,
            "needs_judge": bool(
                self.config.abstain_verify_hi > 0.0
                and best_cos < self.config.abstain_verify_hi
            ),
            # 供 `search()` 在**弃权判定之后**打 [IO] 留痕（保证日志与实际返回一致）
            "path_nodes": result.get("path_nodes", 0),
            "path_edges": result.get("path_edges", 0),
            "edge_types": result.get("path_edge_types") or {},
        }

        # 图链路**不返回 `score`**：这一档每条都是占位分（叙事 / 证据块 / 文档块一律 1.0），
        # 再归一化仍是 1.0，零信息量——而契约里 `score` 是**可选**，不如省掉。
        # 排序信息由**返回顺序**承载（PAR 已按 combined_score 降序给出）。
        # 纯向量链路不受影响：那条路返回**真实余弦**，在它自己的分支里直接 return。
        return items, finalize

    def _collect_par_candidates(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """种子 + 图扩展的候选并集，仅供弃权判定取 best_cos 用。

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

    # ---- 持久化 ----

    def save(self, force: bool = False):
        """落盘的**唯一入口**：串行化后转交 `_save_locked`。

        ⚠️ 不要绕过本方法直接调 `_save_locked`——绕过就失去并发保护
        （请求线程与回补线程会交错覆盖写同一个 YAML）。
        """
        with self._save_lock:
            self._save_locked(force)

    def _save_locked(self, force: bool = False):
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
        """日志中不输出完整 user_id（运行数据与标识不宜落入日志）"""
        if len(self.user_id) <= 8:
            return "***"
        return f"{self.user_id[:4]}***{self.user_id[-4:]}"
