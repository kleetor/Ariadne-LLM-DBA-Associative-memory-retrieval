# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 检索重排：把候选整理成常规的相关性降序数组。

AML 的评分只消费 Search 返回的排序数组（`data`），不看 StoryRank 的故事产物，
因此这里对候选做一次统一重排，并保证输出的 score 与顺序严格单调一致。

候选按证据来源分三个梯度（tier）：
- T1 种子：查询直接召回的种子节点
- T2 图扩展：沿关系边（跳转轴）1 跳及以上的扩展节点
- T3 语义救援：被目的回归过滤掉、但语义上可能高度相关的邻居

四种模式（`ALM_RERANK_MODE`）：
- `off`       沿用 PAR 组合得分排序（纯基线）
- `embedding` 只做 PAR 组合分与语义相似度的融合（无梯度先验的对照）
- `tiered`    融合分 × 梯度先验乘子（默认；软加权，高相关的低梯度节点仍可越过弱相关的高梯度节点）
- `llm`       用 gpt-4o-mini 打分后同样乘以梯度先验（失败时回退 `tiered`）
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

LLM_RERANK_PROMPT = """你是记忆检索的相关性评分员。给定一个查询和一组候选记忆，判断每条记忆与查询的相关程度。

评分标准（0-10 的整数）：
- 10：直接回答了查询
- 7-9：高度相关，是回答所需的关键事实
- 4-6：部分相关，提供背景信息
- 1-3：弱相关
- 0：完全无关

只输出 JSON，不要任何解释：
{{"scores": [{{"id": "节点id", "score": 0}}]}}

查询：
{query}

候选记忆：
{listing}"""

# LLM 打分与语义相似度的融合比例（LLM 为主，embedding 作同分/缺失时的稳定兜底）
_LLM_SCORE_WEIGHT = 0.9
_EMBED_TIEBREAK_WEIGHT = 0.1


def cosine(a, b) -> float:
    """余弦相似度；任一输入缺失或为零向量时返回 0"""
    if a is None or b is None:
        return 0.0
    vec_a = np.asarray(a, dtype=float)
    vec_b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def embedding_scores(query_vec, content_vecs: List[Any]) -> List[float]:
    """query 与各候选的余弦相似度，映射到 [0, 1]"""
    return [(cosine(query_vec, vec) + 1.0) / 2.0 for vec in content_vecs]


def _normalize_by_max(values: List[float]) -> List[float]:
    top = max(values) if values else 0.0
    if top <= 0.0:
        return [0.0 for _ in values]
    return [value / top for value in values]


def tier_weight(tier, tier_weights: Optional[List[float]]) -> float:
    """梯度先验乘子；tier 未标注或越界时按最高梯度（乘子最大）处理"""
    if not tier_weights:
        return 1.0
    try:
        index = int(tier) - 1
    except (TypeError, ValueError):
        index = 0
    index = max(0, min(index, len(tier_weights) - 1))
    return float(tier_weights[index])


def parse_llm_scores(text: str, valid_ids) -> Dict[str, float]:
    """从 LLM 输出中解析 {id: 0~10 分}；无法解析时返回空 dict"""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    raw = payload.get("scores")
    if not isinstance(raw, list):
        return {}

    scores: Dict[str, float] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        node_id = str(entry.get("id", ""))
        if node_id not in valid_ids:
            continue
        try:
            scores[node_id] = max(0.0, min(10.0, float(entry.get("score", 0))))
        except (TypeError, ValueError):
            continue
    return scores


def rerank(
    items: List[Dict[str, Any]],
    mode: str,
    query_vec=None,
    content_vecs: Optional[List[Any]] = None,
    weight_par: float = 0.3,
    weight_embedding: float = 0.7,
    tier_weights: Optional[List[float]] = None,
    llm=None,
    query: str = "",
    pool: int = 0,
) -> List[Dict[str, Any]]:
    """对候选重排，就地写入 item["score"]，返回按 score 降序的列表。

    Args:
        items: 候选列表，每项至少含 "id" / "content" / "par_score"，
               `tiered` / `llm` 模式下还需 "tier"（1 种子 / 2 图扩展 / 3 语义救援）
        mode: off | embedding | tiered | llm
        query_vec: 查询向量（off 以外的模式必需）
        content_vecs: 与 items 等长的候选内容向量
        tier_weights: 梯度先验乘子，下标 0/1/2 对应 tier 1/2/3
        llm: LangChain LLM 实例（仅 llm 模式使用）
        pool: 参与 LLM 打分的最大候选数（0 表示全部）
    """
    if not items:
        return items

    mode = (mode or "off").lower()
    if mode == "off":
        for item in items:
            item["score"] = float(item.get("par_score", 0.0))
        return sorted(items, key=lambda x: x["score"], reverse=True)

    vecs = content_vecs if content_vecs is not None else [None] * len(items)
    embeddings = embedding_scores(query_vec, vecs)
    par_norm = _normalize_by_max([float(item.get("par_score", 0.0)) for item in items])

    if mode == "llm" and llm is not None:
        ordered = _rerank_by_llm(
            items, embeddings, llm, query, pool, weight_embedding, tier_weights
        )
        if ordered is not None:
            return ordered
        logger.warning("LLM 重排未生效，回退 tiered 重排")

    inner = [
        weight_par * par_score + weight_embedding * emb_score
        for par_score, emb_score in zip(par_norm, embeddings)
    ]
    if mode == "tiered":
        for item, value in zip(items, inner):
            item["score"] = tier_weight(item.get("tier"), tier_weights) * value
    else:  # embedding：不做梯度先验，作为对照
        for item, value in zip(items, inner):
            item["score"] = value
    return sorted(items, key=lambda x: x["score"], reverse=True)


def _rerank_by_llm(
    items: List[Dict[str, Any]],
    embeddings: List[float],
    llm,
    query: str,
    pool: int,
    weight_embedding: float,
    tier_weights: Optional[List[float]],
) -> Optional[List[Dict[str, Any]]]:
    """用 LLM 打分重排；调用或解析失败时返回 None 交由上层回退"""
    targets = items[:pool] if pool and pool > 0 else items
    listing = "\n".join(f'- id={item["id"]}: {item["content"]}' for item in targets)
    prompt = LLM_RERANK_PROMPT.format(query=query, listing=listing)

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        logger.warning("LLM 重排调用失败: %s", exc)
        return None

    scores = parse_llm_scores(
        getattr(response, "content", "") or "", {item["id"] for item in targets}
    )
    if not scores:
        return None

    for item, emb_score in zip(items, embeddings):
        llm_score = scores.get(item["id"])
        if llm_score is None:
            # 未被 LLM 覆盖的候选排到已评分候选之后
            base = _EMBED_TIEBREAK_WEIGHT * weight_embedding * emb_score
        else:
            base = _LLM_SCORE_WEIGHT * (llm_score / 10.0) + _EMBED_TIEBREAK_WEIGHT * emb_score
        item["score"] = tier_weight(item.get("tier"), tier_weights) * base
    return sorted(items, key=lambda x: x["score"], reverse=True)
