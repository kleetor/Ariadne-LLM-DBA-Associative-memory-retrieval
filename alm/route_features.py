# SPDX-License-Identifier: AGPL-3.0-only

"""路由特征管道：把一条 query 抽成"该走哪条检索路线"的可用特征。

**这是特征层，不是判定层。** 本模块只把可观测的东西抽出来、按稳定顺序拼成向量；
判定留给后续的探针（线性模型）或路线 V 的分流阈值。分开的理由：判定规则改一次就要
重训一次，而特征定义是长期契约——两者混在一起就没法各自演进。

三条约束：

1. **零 LLM、零网络。** 全部是规则与统计，可单测、可在离线日志上复算。
   查询向量是唯一例外，由调用方经 `embed` 注入——生产里 `_search_locked` 已经算过
   一次 `query_vec`，这里再算一次就是重复 embedding。
2. **判型判据复用 `alm.space`。** `_has_answer_instruction` / `_retrieval_text` 是
   "该不该剥离选项块"的唯一真源，路由特征必须与它同判据；否则会出现"判型说精检、
   探针说普通"这种自相矛盾的状态。
   ⚠️ 因此本模块依赖 `alm.space`。若将来要在 `alm.space` 内部使用本模块，
   请在**函数级**导入，避免模块级循环导入。
3. **`FEATURE_NAMES` 是落盘契约。** 顺序即向量列序，只增不改——改名或调序会让
   已训练的探针与旧特征文件**静默失配**。

用法::

    from alm.route_features import extract, vectorize
    feats = extract(query, options)                 # 纯规则，零依赖
    vec = vectorize(feats, embedding=query_vec)     # 供探针消费
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from alm.space import _has_answer_instruction, _retrieval_text

# ---------------------------------------------------------------------------
# 标量特征：名字 → 取值函数。顺序即向量列序，只增不改。
# ---------------------------------------------------------------------------

# 时间型问法：缺料式里 39% 是这一类（版本变更记录 §4.18.2），故单列强特征
_TEMPORAL_RE = re.compile(
    r"^\s*(when|how long|how many (days?|weeks?|months?|years?|hours?|minutes?|times?)\b"
    r"|what time|what date|how much time|since when|as of when|when did)\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DATE_WORD_RE = re.compile(
    r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|yesterday|today|tomorrow|last (week|month|year)|next (week|month|year))\b"
)
_NEGATION_RE = re.compile(r"(?i)\b(not|never|no|none|without|cannot|can't|don't|doesn't|didn't)\b")
_WH_RE = re.compile(r"^\s*(what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|should|would|will)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def _log1p(n: int) -> float:
    """计数量取 log1p：线性模型/距离度量下，原始计数会被长尾带偏"""
    import math
    return math.log1p(max(0, int(n)))


def extract(
    query: str,
    options: Optional[Sequence[str]] = None,
    *,
    embed=None,
    embed_text: Optional[str] = None,
) -> Dict[str, Any]:
    """抽一条 query 的特征。

    Args:
        query: 平台下发的原始 query（**不要**先剥离，剥离在本函数内做）
        options: 契约 `options` 字段（选择题才有）
        embed: 可选的 `text -> vector` 函数；给了才产 `embedding` 键
        embed_text: 可选，覆盖送去 embedding 的文本（缺省用剥离后的检索文本）

    Returns:
        dict：标量特征（键 ∈ `FEATURE_NAMES`）+ 可选 `embedding`
    """
    query = query or ""
    opts = list(options or [])
    retrieval = _retrieval_text(query)
    toks = _tokens(retrieval)
    m = _WH_RE.search(retrieval)
    head_word = m.group(1).lower() if m else ""

    upper_starts = sum(1 for t in toks if t[:1].isupper())

    feats: Dict[str, Any] = {
        # ---- 结构：这道题在契约上是什么形态 ----
        "has_options": 1.0 if opts else 0.0,
        "n_options": _log1p(len(opts)),
        "n_options_raw": float(len(opts)),
        "has_answer_instruction": 1.0 if _has_answer_instruction(query) else 0.0,
        # 剥离比例：内联选项题会从 ~1150 字掉到 ~170 字，是稳定可判的形态特征
        "stripped_ratio": 1.0 - (len(retrieval) / len(query)) if query else 0.0,
        "query_chars": _log1p(len(query)),
        "retrieval_chars": _log1p(len(retrieval)),
        "n_tokens": _log1p(len(toks)),
        "avg_token_len": (sum(len(t) for t in toks) / len(toks)) if toks else 0.0,
        # ---- 精确线索：数字/年份/引号/专名 ----
        "has_digit": 1.0 if any(ch.isdigit() for ch in retrieval) else 0.0,
        "n_digits": _log1p(sum(ch.isdigit() for ch in retrieval)),
        "has_year": 1.0 if _YEAR_RE.search(retrieval) else 0.0,
        "has_date_word": 1.0 if _DATE_WORD_RE.search(retrieval) else 0.0,
        "has_quote": 1.0 if ('"' in retrieval or "'" in retrieval) else 0.0,
        "proper_ratio": (upper_starts / len(toks)) if toks else 0.0,
        "has_negation": 1.0 if _NEGATION_RE.search(retrieval) else 0.0,
        # ---- 问句形态 ----
        "is_temporal": 1.0 if _TEMPORAL_RE.search(retrieval) else 0.0,
        "is_when": 1.0 if head_word == "when" else 0.0,
        "is_how": 1.0 if head_word == "how" else 0.0,
        "is_what": 1.0 if head_word == "what" else 0.0,
        "is_which": 1.0 if head_word == "which" else 0.0,
        "is_who": 1.0 if head_word in ("who", "whom", "whose") else 0.0,
        "is_where": 1.0 if head_word == "where" else 0.0,
        "is_why": 1.0 if head_word == "why" else 0.0,
        "is_yesno": 1.0 if head_word in (
            "is", "are", "was", "were", "do", "does", "did",
            "can", "could", "should", "would", "will",
        ) else 0.0,
        # ---- 组合 ----
        # "精检形态"：契约选项 或 明确作答指令。与 `_is_precision_query` 同判据。
        "is_precision_shape": 1.0 if (opts or _has_answer_instruction(query)) else 0.0,
    }
    if embed is not None:
        text = embed_text if embed_text is not None else retrieval
        feats["embedding"] = embed(text)
    return feats


# 向量列序 = 标量特征顺序（`embedding` 若存在则拼在最前，由 `vectorize` 决定）
FEATURE_NAMES = tuple(
    k for k in extract("placeholder").keys() if k != "embedding"
)


def numeric_row(feats: Dict[str, Any]) -> List[float]:
    """按 `FEATURE_NAMES` 顺序把标量特征拍平成一行（缺失记 0）"""
    return [float(feats.get(k, 0.0) or 0.0) for k in FEATURE_NAMES]


def vectorize(feats: Dict[str, Any], embedding: Optional[Sequence[float]] = None):
    """把特征转成探针可消费的一维向量：**embedding 在前，标量在后**。

    顺序固定（embedding 优先）是为了让"换特征集"与"换编码器"两件事互不干扰：
    标量列永远在尾部，加特征只是往尾部追加，不移动 embedding 段。

    返回 numpy 数组；未装 numpy 的环境下退回 list（离线复算用得上）。
    """
    row = numeric_row(feats)
    emb = embedding if embedding is not None else feats.get("embedding")
    if emb is None:
        return _as_array(row)
    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    return _as_array([float(x) for x in emb_list] + row)


def _as_array(values: List[float]):
    try:
        import numpy as np
        return np.asarray(values, dtype=np.float32)
    except ImportError:  # pragma: no cover - numpy 是硬依赖，仅离线脚本可能缺
        return list(values)
