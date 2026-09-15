# SPDX-License-Identifier: AGPL-3.0-only

"""
完整检索入口：串联跳转轴 + 目的回归 + 寻峰终止

论文第六章 Algorithm: Purpose-Driven Associative Retrieval
"""

import logging
import re
import numpy as np
from typing import Any, Callable, List, Tuple, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

from dba_pipeline.core.jump_axis import get_jump_weight, NodeType, RelationType
from dba_pipeline.core.purpose import PurposeModel
from dba_pipeline.core.peak_find import PeakFinder
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.llm.inference import InferenceEngine
from dba_pipeline.embedding.store import VectorStore

# 阶段观测回调：on_stage(stage, info)，供面板「调用测试」把链路每一段展示出来。
StageHook = Optional[Callable[[str, Dict[str, Any]], None]]


def emit_stage(on_stage: StageHook, stage: str, **info) -> None:
    """把阶段事件交给调用方。

    观测是旁路：回调抛错只记 debug 日志，绝不打断检索本身。
    """
    if on_stage is None:
        return
    try:
        on_stage(stage, info)
    except Exception:
        logging.getLogger(__name__).debug("阶段回调失败（忽略）", exc_info=True)


def _rel_value(rel_type: Any) -> Any:
    """关系类型可能是枚举，转成可序列化的值"""
    return getattr(rel_type, "value", rel_type)


def _hop_digest(combined: Dict[str, Dict], limit: int = 8) -> List[Dict[str, Any]]:
    """每跳候选摘要（按组合得分降序取前 limit 条），供面板展示扩展轨迹"""
    ordered = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
    return [
        {
            "id": c["id"],
            "from": c["from"],
            "rel_type": _rel_value(c["rel_type"]),
            "is_reverse": c["is_reverse"],
            "combined_score": round(float(c["combined_score"]), 4),
            "purpose_score": round(float(c["purpose_score"]), 4),
            "jump_weight": round(float(c["jump_weight"]), 4),
        }
        for c in ordered[:limit]
    ]

# ── 时期词表（单一来源）──────────────────────────────────────────────────
# 历史上「锚点白名单」(TOOL_TIME_ANCHOR_RE) 与「查询校验词表」是两份独立正则，
# 结果出现一批"锚点能定位、查询却不校验"的词（今天 / 上个月 / 大二上学期 …），
# 这些查询会静默走 fallback。这里把时期词集中定义成四类：
#
#   (1) 绝对时期词 → 查询出现时**字面硬校验**      _explicit_periods
#   (2) 模糊时期词 → 查询出现时按**映射集合软校验** _fuzzy_periods
#   (3) 人生阶段词 → 查询出现时按**同族阶段软校验** _era_periods
#   (4) 仅锚点词   → 只用于识别锚点、**不做查询校验**（日常语境里太常见，
#                    例如「晚上」「工作」，硬校验会把正常查询误判为拒答）
#
# 覆盖情况由 eval/test_timestamp_chain.py 的「时期词覆盖检查」兜底，防止再次漂移。

# (1) 绝对时期词。注意「周X」必须排在裸「上周/这周」之前：否则「上周三」会被拆成
#     「上周」，与库中的「上周五」锚点误判为同一时期（实测跨期漂移的来源之一）。
_EXPLICIT_TOKENS = (
    r"\d{1,2}月\d{1,2}[日号]?", r"\d{1,2}[日号]", r"\d{1,2}月",
    r"[这上本下前那]?周[一二三四五六日天]",
    "昨天", "明天", "前天", "今天",
    "上周", "这周", "本周", "下周",
    "上个月", "这个月", "下个月",
    "去年", "今年", "明年", "前年",
    "寒假", "暑假", "上学期", "下学期", "上半年", "下半年", "年底", "年初",
)
_EXPLICIT_PERIOD_RE = re.compile("|".join(_EXPLICIT_TOKENS))


def _explicit_periods(text: str) -> set:
    """抽取文本中的显式日期/时期词，供时序命中的字面校验使用。"""
    return set(m.group(0) for m in _EXPLICIT_PERIOD_RE.finditer(text or ""))


# (2) 模糊时期词——没有字面日期可校验，但可映射到「合理的锚点字面集合」做软校验。
# 动机：查询「前几天」时，若库中根本没有任何近期锚点，旧逻辑会静默退回语义最近邻
# （实测会捞出「这周二」这类仅语义相近的锚点），调用方拿不到任何"这是回退"的信号。
# 映射策略：只放行语义上确实覆盖该模糊词的时间表述，其余一律拒答。
# 边界按"该词的时间跨度"划线——「前几天」不含一周前的「上周三」，「这两天」不含上周；
# 而「最近」「那天」本身跨度就大，允许覆盖到上周。
_FUZZY_PERIOD_RULES = (
    # 回溯性的"前几天"：仅 3~6 天前的范围，不含"上周"
    (("前几天", "前两天", "前些天", "头几天"),
     ("前天", "大前天", "前几天", "前两天", "前些天")),
    # "这两天 / 这几天"：只覆盖当下前后一两天
    (("这两天", "这几天"),
     ("这两天", "这几天", "昨天", "今天", "前天")),
    # 注：「最近 / 近来 / 这段时间」刻意**不做查询校验**——跨度过大且在日常查询里极常见，
    # 一旦库中无该类锚点就会被判成"无该时段记录"，而实际上库里可能有大量普通节点，
    # 这个提示是错的。它们仍可作锚点（见 _ANCHOR_ONLY_TOKENS），查询走 fallback 并带信号。
    # "那天 / 当天 / 当时"：特指某个已被提及的过去时点，可能落在一周前
    (("那会儿", "那时候", "那天", "当天", "当时"),
     ("那天", "当天", "当时", "那天晚上", "上周", "这周")),
    # "后来 / 之后"：承接一个紧接着的后续时点，库中通常需要显式"后来/第二天"
    (("后来", "之后", "随后"),
     ("后来", "之后", "随后", "第二天", "次日")),
)

_FUZZY_TOKENS = tuple(
    w for words, _ in _FUZZY_PERIOD_RULES for w in words
)


def _fuzzy_periods(text: str):
    """抽取查询中的模糊时期词，返回 [(命中的查询词, 可接受锚点子串元组), ...]"""
    t = text or ""
    out = []
    for words, anchors in _FUZZY_PERIOD_RULES:
        hit = [w for w in words if w in t]
        if hit:
            out.append((hit[0], anchors))
    return out


# (3) 人生阶段词——查询出现时，锚点须是**同族阶段词**（「高中」不匹配「大学」）。
# 按学段分族：只用一张"全部阶段词"的平表会让「大二」匹配到「高一」（实测误配）。
_ERA_FAMILY = {
    "高中": ("高中", "高一", "高二", "高三"),
    "高一": ("高中", "高一", "高二", "高三"),
    "高二": ("高中", "高一", "高二", "高三"),
    "高三": ("高中", "高一", "高二", "高三"),
    "大学": ("大学", "大一", "大二", "大三", "大四"),
    "大一": ("大学", "大一", "大二", "大三", "大四"),
    "大二": ("大学", "大一", "大二", "大三", "大四"),
    "大三": ("大学", "大一", "大二", "大三", "大四"),
    "大四": ("大学", "大一", "大二", "大三", "大四"),
    "大专": ("大专",),
}
_ERA_TOKENS = tuple(_ERA_FAMILY)


def _era_periods(text: str):
    """抽取查询中的人生阶段词，返回命中的词列表"""
    t = text or ""
    return [w for w in _ERA_TOKENS if w in t]


def _era_acceptable(words) -> tuple:
    """查询命中的阶段词 → 同族阶段词集合（用于软校验）"""
    out = []
    for w in words:
        for a in _ERA_FAMILY.get(w, (w,)):
            if a not in out:
                out.append(a)
    return tuple(out)


def _period_literals(text: str) -> set:
    """文本自带的时期字面集合（显式 + 模糊词 + 人生阶段词）"""
    t = text or ""
    lit = set(_explicit_periods(t))
    lit |= {w for w, _ in _fuzzy_periods(t)}
    lit |= set(_era_periods(t))
    return lit


def _time_consistency(anchor_acceptable, fact_content: str, fact_periods: set):
    """判断事实自带的时期词与锚点是否相容。

    Returns True（相容）/ False（显式冲突）/ None（无法判定）。

    三层判断，顺序不能颠倒：
    1. 事实的时期字面与锚点可接受集有交集 → 相容；
    2. 锚点的可接受串**出现在事实文本里** → 相容（如锚点「那天」的可接受集含「上周」，
       而事实写的是「上周三」——集合不交但字面包含，属同义）；
    3. 事实自带时期词、且以上都不成立 → 显式冲突；
    4. 事实没有时期词、锚点也无映射 → 无法判定。
    """
    if not anchor_acceptable:
        return None
    if fact_periods & anchor_acceptable:
        return True
    if any(a in fact_content for a in anchor_acceptable):
        return True
    if fact_periods:
        return False
    return None


def _anchor_acceptable_periods(anchor_text: str):
    """锚点可覆盖的时期字面集合；锚点本身不含时期词时返回 None（无法判定）。

    用于兜底 TEMPORAL 边的错配：事实自带的时期词若与该集合完全不相交，说明它很可能
    不属于这个时段。注意它**只能抓显式冲突**（如把「今天」挂到「上周三」上），
    对"同区间内挂错事件"（如把别的「前天」事件挂到「前几天」上）无能为力——
    那需要 source 层才能发现。
    """
    t = anchor_text or ""
    lit = _period_literals(t)
    for _, anchors in _fuzzy_periods(t):
        lit |= set(anchors)
    return lit or None


# (4) 仅作锚点、不做查询校验的词
_ANCHOR_ONLY_TOKENS = (
    r"\d{1,4}年", r"\d{1,2}号", r"\d{1,2}日",
    r"^[这上本下前那]?周",
    "凌晨", "早上", "上午", "中午", "下午", "晚上",
    "那段时间", "近年来", "工作", "毕业", "入职", "开学",
    # 跨度过大、日常语境太常见，只作锚点、不做查询校验（见 _FUZZY_PERIOD_RULES 注释）
    "最近", "近来", "这段时间",
    # 模糊词的"承接型"锚点：可作为锚点被定位，但查询里出现时不单独触发校验
    "第二天", "次日",
)


# 独立时序工具 temporal_lookup 用的时间锚点判定：覆盖绝对时期 + 指示词 + 人生阶段。
# 模糊时期词同样要列入：LLM 会把「前几天/这两天/后来」抽成 THING 锚点节点，
# 白名单缺了它们就会出现"锚点存在、却定位不到"（实测「前几天」节点被漏掉）。
TOOL_TIME_ANCHOR_RE = re.compile("|".join(
    _EXPLICIT_TOKENS + _FUZZY_TOKENS + _ERA_TOKENS + _ANCHOR_ONLY_TOKENS
))


class PurposeDrivenRetriever:
    """目的驱动的联想记忆检索器（完整链路）"""

    def __init__(
        self,
        llm: ChatOpenAI,
        embeddings: Embeddings,
        graph: MemoryGraph,
        vector_store: VectorStore,
        inference: InferenceEngine,
        distance_decay: float = 0.85,
        jump_weight_coef: float = 0.5,
        purpose_weight_coef: float = 0.5,
        purpose_filter_threshold: float = 0.2,
        purpose_filter_decay: float = 0.0,
        path_tracker=None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.graph = graph
        self.vector_store = vector_store
        self.inference = inference

        self.purpose_model = PurposeModel(llm, self._embed)
        self.peak_finder = PeakFinder(patience=2, min_delta=0.015)

        self.distance_decay = distance_decay
        self.jump_weight_coef = jump_weight_coef
        self.purpose_weight_coef = purpose_weight_coef
        self.purpose_filter_threshold = purpose_filter_threshold
        self.purpose_filter_decay = purpose_filter_decay
        self.path_tracker = path_tracker  # 可选：PathTracker 实例

    def _get_hop_threshold(self, hop: int) -> float:
        """计算当前跳数的有效过滤阈值

        若 purpose_filter_decay > 0，使用递减公式：
            threshold_hop = purpose_filter_threshold × purpose_filter_decay^hop
        否则使用固定阈值。
        """
        if self.purpose_filter_decay > 0.0:
            return self.purpose_filter_threshold * (self.purpose_filter_decay ** hop)
        return self.purpose_filter_threshold

    def _embed(self, text: str) -> np.ndarray:
        return self.vector_store._embed(text)

    def _is_active(self, node_id: str) -> bool:
        """节点是否仍参与检索：必须存在，且未被废弃/遗忘。

        检索链路需与 DBA 维护（deprecate/forget）保持一致，避免过期或孤立记忆进入结果。
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return False
        if node.get("deprecated") or node.get("forgotten"):
            return False
        return True

    # ---- 独立单跳时序工具 temporal_lookup ----
    # 定位：独立查询工具，不进入主检索流程。按"时间→事件"反向单跳取共时事实，返回多锚点供上层 LLM 挑选。

    def _tool_is_time_node(self, memory_id: str) -> bool:
        """工具判定时间锚点：th/T+数字 前缀（数据集约定），或 THING 且内容含时间/时期词。"""
        if memory_id.startswith("th"):
            return True
        if len(memory_id) > 1 and memory_id[0] == "T" and memory_id[1:].isdigit():
            return True
        try:
            if self.graph.get_node_type(memory_id) != NodeType.THING:
                return False
        except Exception:
            return False
        content = self.graph.graph.nodes[memory_id].get("content", "")
        return bool(TOOL_TIME_ANCHOR_RE.search(content))

    def _has_reverse_temporal(self, memory_id: str) -> bool:
        """是否为"真时间锚点"：存在反向 TEMPORAL 邻居（有事件锚定到它）。"""
        try:
            for nb, rel_type, is_reverse in self.graph.get_neighbors(memory_id):
                if rel_type == RelationType.TEMPORAL and is_reverse:
                    return True
        except Exception:
            return False
        return False

    def temporal_lookup(
        self,
        query: str,
        k_seed: int = 8,
        max_facts: int = 8,
        max_anchors: int = 3,
    ) -> Dict:
        """独立单跳时序工具——**仅做"时间→事件"反向**（主检索器反向权重为 0、做不到的部分）。

        语义匹配到时间锚点，沿 TEMPORAL 边**反向**取共时事件。
        · 不做正向（事件→时间，那是主检索器的活）、不链式扩展、不做目的打分挤占、不跨期漂移。
        · 查询可能命中多个时间锚点（如"三年"可指大专三年/工作三年），**返回全部候选锚点的事实组，
          交由上层 LLM 判断最相关的一个/多个**，而不是硬性消歧。
        · 只返回"真锚点"（带反向 TEMPORAL 邻居）的事实组；无可用锚点时返回空 matches。
        · 查询带时间词时做锚点校验：显式词（上周三）字面校验，模糊词（前几天）与
          人生阶段词（高中/大二）按可接受集合软校验；不匹配即 rejected。
          无时间词的查询不做校验，match_type=fallback。

        Returns:
            {"query": str,
             "matches": [{"time_anchor": {"id","content"},
                          "facts": [{"id","content","from","time_consistent"}...]}...],
             "count": int,
             "match_type": "exact" | "fuzzy" | "fallback" | "rejected",
             # 仅 rejected 时附加
             "rejected": True, "reason": str}

            facts[].time_consistent: True/False/None。None 表示事实里没有时期词、
            无法判定；False 表示事实自带的时期词与锚点完全不相交（疑似 TEMPORAL 边错配）。
        """
        try:
            hits = self.vector_store.search(query, k=k_seed)  # [(id, score, meta)]
        except Exception:
            hits = []
        matched = [(h[0], h[1]) if len(h) >= 2 else (h[0], 1.0) for h in hits]

        # 只定位"时间锚点"（时间推事件型查询的锚必须是时间节点）；跳过废弃/遗忘节点
        time_cands = [m for m in matched if self._is_active(m[0]) and self._tool_is_time_node(m[0])]
        # 过滤真锚点：只有带反向 TEMPORAL 邻居的才是时间锚点（排除误判的"含时间词事件节点"）。
        usable = [m for m in time_cands if self._has_reverse_temporal(m[0])]

        # 拒答与信号：
        #  · 显式日期/时期词（「8月30日」「上周三」）→ 硬校验，锚点必须字面命中；
        #  · 模糊时期词（「前几天」「后来」）→ 软校验，锚点须落在该词的合理时间集合内；
        #  · 人生阶段词（「高中」「大二」）→ 软校验，锚点须为同族阶段词；
        #  · 都没有 → 不校验，退语义最近邻（match_type=fallback）。
        # match_type 显式告知调用方本次是 exact / fuzzy / fallback，避免把回退当成命中。
        rejected = False
        match_type = "fallback"
        reason = None

        q_periods = _explicit_periods(query)
        q_fuzzy = _fuzzy_periods(query)
        q_era = _era_periods(query)
        if q_periods:
            usable = [
                m for m in usable
                if q_periods & _explicit_periods(self.graph.get_content(m[0]) or "")
            ]
            rejected = not usable
            match_type = "rejected" if rejected else "exact"
            if rejected:
                reason = (
                    f"查询含显式时间词 {sorted(q_periods)}，但库中无字面匹配的时间锚点；"
                    "为避免跨期漂移返回空"
                )
        elif q_fuzzy:
            acceptable = tuple({a for _, anchors in q_fuzzy for a in anchors})
            usable = [
                m for m in usable
                if any(a in (self.graph.get_content(m[0]) or "") for a in acceptable)
            ]
            rejected = not usable
            match_type = "rejected" if rejected else "fuzzy"
            if rejected:
                reason = (
                    f"查询含模糊时间词 {[q for q, _ in q_fuzzy]}，其可接受锚点为 "
                    f"{list(acceptable)}，但库中无匹配；为避免漂移返回空"
                )
        elif q_era:
            acceptable_era = _era_acceptable(q_era)
            usable = [
                m for m in usable
                if any(a in (self.graph.get_content(m[0]) or "") for a in acceptable_era)
            ]
            rejected = not usable
            match_type = "rejected" if rejected else "fuzzy"
            if rejected:
                reason = (
                    f"查询含人生阶段词 {q_era}，但库中无同族阶段锚点；为避免漂移返回空"
                )

        matches: List[Dict] = []
        for m in usable[:max_anchors]:
            anchor_id = m[0]
            anchor_content = self.graph.get_content(anchor_id) or ""
            # 时间 → 共时事件（反向，单跳）
            facts: List[Dict] = []
            seen = set([anchor_id])
            # 兜底 TEMPORAL 边错配：事实自带时期词且与锚点完全不相交时打标（不删除，
            # 保留召回、把判断权交给上层）——同区间内挂错事件的情况它抓不到。
            acceptable = _anchor_acceptable_periods(anchor_content)
            for nb, rel_type, is_reverse in self.graph.get_neighbors(anchor_id):
                if rel_type == RelationType.TEMPORAL and is_reverse and nb not in seen:
                    if not self._is_active(nb):
                        continue
                    seen.add(nb)
                    fact_content = self.graph.get_content(nb) or ""
                    fact_periods = _period_literals(fact_content)
                    consistent = _time_consistency(acceptable, fact_content, fact_periods)
                    facts.append({
                        "id": nb, "content": fact_content, "from": anchor_id,
                        "time_consistent": consistent,
                    })
            if max_facts:
                facts = facts[:max_facts]
            matches.append({"time_anchor": {"id": anchor_id, "content": anchor_content}, "facts": facts})

        result = {"query": query, "matches": matches, "count": len(matches),
                  "match_type": match_type}
        if rejected:
            result["rejected"] = True
            result["reason"] = reason
        return result

    # ---- 算法主流程 ----

    def retrieve(
        self,
        query: str,
        seed_k: int = 5,
        max_hops: int = 5,
        expand_k: int = None,
        purpose: Optional[List[str]] = None,
        on_stage: StageHook = None,
    ) -> Dict:
        """执行完整的目的驱动联想检索

        Args:
            query: 用户查询
            seed_k: 种子数量
            max_hops: 最大跳数
            expand_k: 每轮扩展数（默认=seed_k）
            purpose: 可选，调用方（主聊天 LLM/agent）注入的目的列表。
                     提供时跳过独立的目的推断（infer_purpose），检索意图由主 LLM
                     在对话上下文理解中给出；None 时退化到系统内部推断。
            on_stage: 可选，阶段观测回调 on_stage(stage, info)。面板「调用测试」靠它把
                      意图识别 / 种子 / 每一跳 / 寻峰逐段展示；None 时零开销。

        Returns:
            {
                "peak_memories": [(memory_id, content), ...],
                "peak_scores": {memory_id: combined_score, ...},
                "purpose": {"status": str, "purposes": [str]},
                "hop_history": [{"hop": int, "candidates": [...], "mean_score": float}, ...],
                "total_candidates": int,
            }
        """
        # Step 1: 推断状态和目的（可注入：主 LLM 提供则跳过独立推断）
        emit_stage(on_stage, "intent_start", injected=purpose is not None, query=query)
        if purpose is not None:
            purpose_info = {"status": "injected", "purposes": list(purpose)}
        else:
            purpose_info = self.inference.infer_purpose(query)
        purposes = purpose_info.get("purposes", [])
        emit_stage(on_stage, "intent_done", purpose=purpose_info)
        purpose_vec = self.purpose_model.get_purpose_vector(purposes)

        # Step 2: 向量搜索种子记忆（混合 query text + 目的向量）
        half_k = max(1, seed_k // 2)
        text_seeds = self.vector_store.search(query, k=half_k)
        purpose_seeds = self.vector_store.search_by_vector(purpose_vec, k=seed_k - half_k)

        # 合并去重，保持 text 种子优先顺序，purpose 种子补充；跳过废弃/遗忘节点
        seen = set()
        seed_ids = []
        for r in text_seeds:
            if r[0] not in seen and self._is_active(r[0]):
                seen.add(r[0])
                seed_ids.append(r[0])
        for r in purpose_seeds:
            if r[0] not in seen and len(seed_ids) < seed_k and self._is_active(r[0]):
                seen.add(r[0])
                seed_ids.append(r[0])
        # 不足时用更多 text 种子补齐
        if len(seed_ids) < seed_k:
            extra = self.vector_store.search(query, k=seed_k * 2)
            for r in extra:
                if r[0] not in seen and len(seed_ids) < seed_k and self._is_active(r[0]):
                    seen.add(r[0])
                    seed_ids.append(r[0])
        # 计算种子轮目的关联度（使用预缓存向量，无 API 调用）
        seed_vectors = self.vector_store.get_content_vectors(seed_ids)
        seed_scores = self.purpose_model.compute_purpose_score_from_vectors(
            seed_vectors, purpose_vec
        )
        # 种子轮没有跳转轴扩展（jw=0），使用与后续轮次一致的得分公式
        mu_0 = float(np.mean(seed_scores)) * self.purpose_weight_coef
        emit_stage(on_stage, "seeds", ids=seed_ids, mean_score=round(mu_0, 4),
                   text_hits=len(text_seeds), purpose_hits=len(purpose_seeds))

        self.peak_finder.reset()
        decision = self.peak_finder.add_round(mu_0)
        if decision == "peak_found":
            emit_stage(on_stage, "peak_found", hop=0, mean_score=round(mu_0, 4),
                       reason="种子轮即达峰值")
            return self._build_result(seed_ids, purpose_info, [])

        # Step 3-6: 循环扩展
        current_ids = list(seed_ids)
        seed_contents = self.graph.get_contents(seed_ids)
        hop_history: List[Dict] = [{
            "hop": 0,
            "candidates": [
                {"id": mid, "content": content, "purpose_score": float(s),
                 "combined_score": float(s) * self.purpose_weight_coef}
                for mid, content, s in zip(seed_ids, seed_contents, seed_scores)
            ],
            "mean_score": mu_0,
        }]

        for hop in range(1, max_hops + 1):
            # 跳转轴扩展（含可选 PathTracker 动态权重 + 边来源追踪）
            expanded = self.graph.expand_with_trace(current_ids, path_tracker=self.path_tracker)

            # 记录路径激活（供 PathTracker 分析）
            if self.path_tracker is not None:
                for seed_id in current_ids:
                    source_type = self.graph.get_node_type(seed_id)
                    if source_type is None:
                        continue
                    for neighbor_id, rel_type, is_reverse in self.graph.get_neighbors(seed_id):
                        if neighbor_id in seed_ids:
                            continue
                        w = get_jump_weight(source_type, rel_type, is_reverse)
                        if w > 0 and neighbor_id in expanded:
                            pk = self.path_tracker.make_path_key(
                                seed_id, neighbor_id, rel_type.value
                            )
                            self.path_tracker.record_activation(pk)
            if not expanded:
                emit_stage(on_stage, "hop", hop=hop, expanded=0, kept=0, stop="无可扩展邻居")
                result_ids = current_ids
                break

            # 目的过滤（使用预缓存向量，无 API 调用）；跳过废弃/遗忘节点
            candidate_ids = [mid for mid in expanded.keys() if self._is_active(mid)]
            candidate_contents = self.graph.get_contents(candidate_ids)
            candidate_vectors = self.vector_store.get_content_vectors(candidate_ids)
            candidate_scores = self.purpose_model.compute_purpose_score_from_vectors(
                candidate_vectors, purpose_vec
            )

            filtered: Dict[str, Dict] = {}
            # {mem_id: {content, jump_weight, from, rel_type, is_reverse, purpose_score}}
            hop_threshold = self._get_hop_threshold(hop)
            for i, mid in enumerate(candidate_ids):
                ps = float(candidate_scores[i])
                trace = expanded[mid]
                rel_type = trace["rel_type"]
                is_rev = trace["is_reverse"]
                if ps < hop_threshold:
                    continue
                jw = trace["weight"]
                filtered[mid] = {
                    "content": candidate_contents[i],
                    "jump_weight": jw,
                    "from": trace["from"],
                    "rel_type": rel_type,
                    "is_reverse": is_rev,
                    "purpose_score": ps,
                }

            if not filtered:
                emit_stage(on_stage, "hop", hop=hop, expanded=len(expanded), kept=0,
                           stop="目的回归过滤后为空")
                result_ids = current_ids
                break

            # 距离衰减 + 组合得分
            decay = self.distance_decay ** hop
            combined = {}
            for mid, info in filtered.items():
                jw = info["jump_weight"]
                ps = info["purpose_score"]
                combined[mid] = {
                    "id": mid,
                    "content": info["content"],
                    "jump_weight": jw,
                    "purpose_score": ps,
                    "from": info["from"],
                    "rel_type": info["rel_type"],
                    "is_reverse": info["is_reverse"],
                    "combined_score": jw * decay * self.jump_weight_coef
                                      + ps * self.purpose_weight_coef,
                }

            # 当前轮综合得分均值（跳转轴 + 目的，用于寻峰）
            mu_hop = float(np.mean([v["combined_score"] for v in combined.values()]))

            # 寻峰判断
            decision = self.peak_finder.add_round(mu_hop)
            emit_stage(on_stage, "hop", hop=hop, expanded=len(expanded), kept=len(combined),
                       mean_score=round(mu_hop, 4), decision=decision,
                       top=_hop_digest(combined))

            hop_history.append({
                "hop": hop,
                "candidates": sorted(
                    combined.values(),
                    key=lambda x: x["combined_score"],
                    reverse=True,
                ),
                "mean_score": mu_hop,
            })

            if decision == "peak_found":
                emit_stage(on_stage, "peak_found", hop=hop, mean_score=round(mu_hop, 4),
                           peak_index=self.peak_finder.peak_index,
                           tolerance=self.peak_finder.peak_tolerance)
                result_ids = self._collect_peak_tolerance(hop_history)
                break

            # 当前轮候选用作下一轮扩展
            top_k = expand_k if expand_k is not None else seed_k
            current_ids = [
                c["id"]
                for c in sorted(
                    combined.values(),
                    key=lambda x: x["combined_score"],
                    reverse=True,
                )[:top_k]
            ]

        else:
            # 达到 max_hops 仍未找到峰值，用峰值容忍带
            emit_stage(on_stage, "hop", hop=max_hops, stop="达到最大跳数")
            result_ids = self._collect_peak_tolerance(hop_history)

        emit_stage(on_stage, "peak_collected", count=len(result_ids), ids=result_ids)
        return self._build_result(result_ids, purpose_info, hop_history)

    def _collect_peak_tolerance(self, hop_history: List[Dict]) -> List[str]:
        """峰值容忍带：取 μ ≥ peak_mean - tolerance 的所有轮候选"""
        peak_mean = self.peak_finder.history[self.peak_finder.peak_index]
        threshold = peak_mean - self.peak_finder.peak_tolerance
        result_ids = []
        seen = set()
        for h in hop_history:
            mu = h.get("mean_score", 0)
            if mu >= threshold:
                for c in h.get("candidates", []):
                    cid = c.get("id", "")
                    if cid not in seen:
                        seen.add(cid)
                        result_ids.append(cid)
        return result_ids

    def _build_result(
        self,
        result_ids: List[str],
        purpose_info: dict,
        hop_history: List[Dict],
    ) -> Dict:
        # 从 hop_history 收集每条的 combined_score
        id_score = {}
        for h in hop_history:
            for c in h.get("candidates", []):
                cid = c.get("id", "")
                cs = c.get("combined_score", 0)
                if cid not in id_score or cs > id_score[cid]:
                    id_score[cid] = cs

        memories = []
        for mid in result_ids:
            if not self._is_active(mid):
                continue
            content = self.graph.get_content(mid)
            if content:
                memories.append((mid, content))

        # 按 combined_score 降序排列（作为 LLM rerank 前的默认排序）
        memories.sort(key=lambda x: id_score.get(x[0], 0), reverse=True)
        return {
            "peak_memories": memories,  # [(id, content), ...] 保持向后兼容
            "peak_scores": id_score,    # {id: combined_score} 供 rerank 使用
            "purpose": purpose_info,
            "hop_history": hop_history,
            "total_candidates": len(memories),
        }

    # ---- StoryRank：链路理解 → 故事片段 ----

    @staticmethod
    def _collect_trace_nodes(hop_history: List[Dict]) -> Dict:
        """从 hop_history 提取节点信息与父子关系（树结构）

        Returns:
            {"parent": {child: parent}, "score": {id: combined_score},
             "content": {id: content}, "edge": {child: (rel_type, is_reverse)}}
        """
        parent = {}
        score = {}
        content = {}
        edge = {}
        seen = set()
        for h in hop_history:
            for c in h.get("candidates", []):
                cid = c.get("id")
                if not cid:
                    continue
                score[cid] = c.get("combined_score", 0)
                content[cid] = c.get("content", "")
                # 节点首次出现时才记录父边；回访（如双向边折返）不更新，避免 parent 成环
                if cid in seen:
                    continue
                seen.add(cid)
                frm = c.get("from")
                if frm is not None:
                    parent[cid] = frm
                    edge[cid] = (c.get("rel_type"), c.get("is_reverse", False))
        return {"parent": parent, "score": score, "content": content, "edge": edge}

    def select_core_nodes(
        self,
        hop_history: List[Dict],
        result_ids: List[str],
    ) -> List[List[str]]:
        """按连通性拆分核心节点组（粗筛）

        每个结果节点沿 from 回溯到种子（根），共享根的结果归为一组；
        组内节点按 hop 顺序（根 → 叶）排列。

        Returns:
            List[List[str]]：每个元素是一个片段的核心节点 id（从根到叶）
        """
        trace = self._collect_trace_nodes(hop_history)
        parent = trace["parent"]
        score = trace["score"]

        def find_root(nid: str) -> str:
            seen = set()
            while nid in parent and nid not in seen:
                seen.add(nid)
                nid = parent[nid]
            return nid

        # 按根分组结果节点
        groups: Dict[str, List[str]] = {}
        for rid in result_ids:
            if rid not in score:
                continue
            root = find_root(rid)
            groups.setdefault(root, []).append(rid)

        # 回溯收集每个分组的路径节点（结果 → 根），再按 hop 顺序排列（根 → 叶）
        core_groups: List[List[str]] = []
        for _, rids in groups.items():
            core = set()
            for rid in rids:
                nid = rid
                seen_loop = set()
                while nid is not None and nid not in seen_loop:
                    seen_loop.add(nid)
                    core.add(nid)
                    nid = parent.get(nid)
            ordered = []
            seen = set()
            for h in hop_history:
                for c in h.get("candidates", []):
                    cid = c.get("id")
                    if cid in core and cid not in seen:
                        seen.add(cid)
                        ordered.append(cid)
            if ordered:
                core_groups.append(ordered)

        return core_groups

    def _build_path(
        self,
        group: List[str],
        hop_history: List[Dict],
    ) -> Dict:
        """把核心节点组构建成 StoryRank 的 path 结构（nodes + edges）"""
        trace = self._collect_trace_nodes(hop_history)
        parent = trace["parent"]
        content = trace["content"]
        edge = trace["edge"]
        group_set = set(group)

        nodes = []
        for nid in group:
            nt = self.graph.get_node_type(nid)
            nt_val = nt.value if hasattr(nt, "value") else str(nt)
            nodes.append({
                "id": nid,
                "content": content.get(nid) or self.graph.get_content(nid) or "",
                "node_type": nt_val,
                # 记录时间（Unix 毫秒）。仅在调用方开启 render_timestamps 时进入 prompt，
                # 默认不渲染，保证 ALM 侧的叙事文本与既有行为完全一致。
                "timestamp": self.graph.graph.nodes[nid].get("timestamp"),
            })

        edges = []
        for nid in group:
            p = parent.get(nid)
            if p is not None and p in group_set:
                rel_type, is_reverse = edge.get(nid, (None, False))
                rel_val = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
                edges.append({
                    "from": p,
                    "to": nid,
                    "rel_type": rel_val,
                    "is_reverse": is_reverse,
                })

        return {"nodes": nodes, "edges": edges}

    def retrieve_with_story(
        self,
        query: str,
        seed_k: int = 5,
        with_response: bool = True,
        max_hops: Optional[int] = None,
        expand_k: Optional[int] = None,
        render_timestamps: bool = False,
        on_stage: StageHook = None,
    ) -> Dict:
        """检索 → 连通性粗筛 → StoryRank 故事化 →（可选）生成回复

        把检索得到的记忆因果链路理解成故事片段，替代原 rerank 的扁平重排序。
        max_hops / expand_k 为 None 时沿用 retrieve() 的默认值。
        render_timestamps 为 True 时把节点记录时间一并交给 StoryRank（MCP 侧使用），
        默认 False 以保持 ALM 侧输出不变。
        on_stage: 可选阶段观测回调，透传给 retrieve() 并补充故事化两段（面板「调用测试」）。
        """
        retrieve_kwargs: Dict[str, Any] = {"seed_k": seed_k, "on_stage": on_stage}
        if max_hops is not None:
            retrieve_kwargs["max_hops"] = max_hops
        if expand_k is not None:
            retrieve_kwargs["expand_k"] = expand_k
        result = self.retrieve(query, **retrieve_kwargs)
        peak_memories = result.get("peak_memories", [])
        hop_history = result.get("hop_history", [])
        result_ids = [mid for mid, _ in peak_memories]

        # 粗筛：按连通性拆分核心节点组
        core_groups = self.select_core_nodes(hop_history, result_ids)

        # 合并所有核心节点（去重、保持 hop 顺序），一次性生成一段完整故事
        all_nodes = []
        seen = set()
        for group in core_groups:
            for nid in group:
                if nid not in seen:
                    seen.add(nid)
                    all_nodes.append(nid)

        emit_stage(on_stage, "core_nodes", groups=len(core_groups), nodes=len(all_nodes))

        stories = []
        story_nodes = []
        discarded_nodes = []
        if all_nodes:
            path = self._build_path(all_nodes, hop_history)
            emit_stage(on_stage, "storyrank_start", nodes=len(path.get("nodes", [])),
                       edges=len(path.get("edges", [])))
            out = self.inference.story_rank(query, path, render_timestamps=render_timestamps)
            story = out.get("story", "")
            adopted = out.get("adopted_ids", [])
            if story:
                stories.append(story)
            story_nodes = adopted
            discarded_nodes = [nid for nid in all_nodes if nid not in adopted]
            emit_stage(on_stage, "storyrank_done", adopted=len(story_nodes),
                       discarded=len(discarded_nodes), story=story,
                       adopted_ids=list(story_nodes))

        result["stories"] = stories
        result["story_nodes"] = story_nodes
        result["discarded_nodes"] = discarded_nodes

        # 生成回复：聊天 LLM 只接收干净故事，替代 [id] content 列举
        if with_response:
            context_items = [f"[记忆片段] {s}" for s in stories]
            result["response"] = self.inference.generate_response(query, context_items)
        return result
