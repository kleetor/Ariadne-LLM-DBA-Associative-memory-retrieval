# SPDX-License-Identifier: AGPL-3.0-only

"""
图谱体检：**确定性**候选发现（不修改图谱、不调用 LLM）

定位：巡检的第一步。只负责"找出可疑"，不负责"下判断"——
是否真的同一条事实、要不要合并，交给上层 agent 看过报告后用 `dba_intervene` 落盘。
这样巡检工具本身是只读的：无写入、无 LLM、也就没有与检索的并发冲突。

本模块只做纯计算（输入是快照数据），不接触 graph / vector_store，便于离线测试。
"""

from typing import Dict, Iterable, List, Sequence

import numpy as np

# 语义去重对"事实陈述"才有意义。锚点 / 实体名这类短内容在向量空间里彼此高度相似，
# 不排除会把候选表淹没——实测「2026年9月7日」与「2026年8月31日」的余弦极高，
# 但它们显然不是"同一条事实"。主防线是**按类型排除 thing**（抽取侧的锚点都是 thing），
# 长度门槛只用来挡退化内容：中文没有词边界，8 字以上会把「用户喜欢喝咖啡」这种
# 正常事实也误挡（实测），所以这里取 4。
DEFAULT_EXCLUDE_TYPES = ("thing",)
DEFAULT_MIN_CHARS = 4
# 单次矩阵乘法的行数上限，避免大图一次性生成 n×n 相似度矩阵占用过多内存
_CHUNK_ROWS = 512


def find_duplicate_candidates(
    nodes: Sequence[Dict],
    vectors: Dict[str, np.ndarray],
    threshold: float = 0.85,
    min_chars: int = DEFAULT_MIN_CHARS,
    exclude_types: Iterable[str] = DEFAULT_EXCLUDE_TYPES,
    max_pairs: int = 50,
) -> List[Dict]:
    """找语义高度相似的节点对（潜在重复）。

    Args:
        nodes: [{"id", "content", "node_type"}]，调用方需自行剔除已废弃/已遗忘节点
        vectors: {node_id: 已归一化或未归一化的向量}；缺失向量的节点自动跳过
        threshold: 余弦相似度阈值
        min_chars: 内容长度下限（去空白后），低于此值不参与比对
        exclude_types: 不参与比对的节点类型（小写比较）
        max_pairs: 最多返回多少对（按相似度降序）

    Returns:
        [{"a": node_id, "b": node_id, "similarity": float}, ...] 按相似度降序。

    复杂度 O(n²·d)：按行分块计算，避免大图一次性分配 n×n 矩阵。
    """
    excluded = {t.lower() for t in exclude_types}
    ids: List[str] = []
    texts: List[str] = []
    mats: List[np.ndarray] = []
    for nd in nodes:
        nid = nd.get("id")
        if not nid or nid not in vectors:
            continue
        ntype = str(nd.get("node_type") or "").split(".")[-1].lower()
        if ntype in excluded:
            continue
        content = (nd.get("content") or "").strip()
        if len(content.replace(" ", "")) < min_chars:
            continue
        vec = np.asarray(vectors[nid], dtype=np.float32).ravel()
        if vec.size == 0:
            continue
        ids.append(nid)
        texts.append(content)
        mats.append(vec)

    if len(ids) < 2:
        return []

    matrix = np.vstack(mats)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    matrix = matrix / norms

    pairs: List[Dict] = []
    for start in range(0, len(ids), _CHUNK_ROWS):
        chunk = matrix[start:start + _CHUNK_ROWS]
        sims = chunk @ matrix.T                      # (rows, n)
        for r in range(sims.shape[0]):
            i = start + r
            # 只取对角线右侧，避免重复与自比
            row = sims[r, i + 1:]
            for offset in np.nonzero(row >= threshold)[0]:
                j = i + 1 + int(offset)
                pairs.append({
                    "a": ids[i], "b": ids[j],
                    "similarity": round(float(row[offset]), 4),
                })

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs[:max_pairs] if max_pairs and max_pairs > 0 else pairs
