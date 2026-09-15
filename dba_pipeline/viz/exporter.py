# SPDX-License-Identifier: AGPL-3.0-only

"""
DBA 3D 可视化 — 图谱导出器

将 MemoryGraph 转换为 3d-force-graph 可消费的 JSON 格式。
"""

import json
from dba_pipeline.graph.memory_graph import MemoryGraph

# 节点类型 → 半径映射
_RADIUS_MAP = {
    "STATUS": 5.5,
    "REASON": 5.5,
    "ACTION": 5.5,
    "THING": 4.5,
    "PERSON": 4.5,
    "EMOTION": 4.5,
}


def to_3dforcegraph(graph: MemoryGraph) -> dict:
    """将 MemoryGraph 导出为 3d-force-graph JSON

    节点格式:
        {id, label, node_type, content,
         deprecated, forgotten, in_degree, out_degree, radius}

    边格式:
        {source, target, rel_type, bidirectional,
         deprecated_src, deprecated_dst}

    Returns:
        {"nodes": [...], "links": [...], "stats": {...}}
    """
    precompute_degrees(graph)
    nodes = _export_nodes(graph)
    links = _export_links(graph)
    stats = _export_stats(graph)

    return {"nodes": nodes, "links": links, "stats": stats}


def to_3dforcegraph_json(graph: MemoryGraph, indent: int = 2) -> str:
    """导出为 JSON 字符串"""
    return json.dumps(to_3dforcegraph(graph), indent=indent, ensure_ascii=False)


# ---- 内部 ----

def precompute_degrees(graph: MemoryGraph):
    """预计算每个节点的入度和出度，写入 metadata"""
    for nid in graph.graph.nodes():
        node = graph.graph.nodes[nid]
        node.setdefault("metadata", {})
        node["metadata"]["_in_degree"] = graph.graph.in_degree(nid)
        node["metadata"]["_out_degree"] = graph.graph.out_degree(nid)


def _export_nodes(graph: MemoryGraph) -> list:
    nodes = []
    for nid, data in graph.graph.nodes(data=True):
        meta = data.get("metadata", {})
        raw_type = data["node_type"].value if hasattr(data["node_type"], "value") else str(data["node_type"])
        nodes.append({
            "id": nid,
            "label": (data.get("content") or "")[:50],
            "node_type": raw_type.upper(),
            "content": data.get("content", ""),
            "deprecated": data.get("deprecated", False),
            "forgotten": data.get("forgotten", False),
            "in_degree": meta.get("_in_degree", 0),
            "out_degree": meta.get("_out_degree", 0),
            "radius": _RADIUS_MAP.get(raw_type.upper(), 2.5),
        })
    return nodes


def _export_links(graph: MemoryGraph) -> list:
    bidirectional_types = {"SCENARIO", "SOCIAL", "ATTRIBUTE"}
    seen = set()
    links = []

    for u, v, data in graph.graph.edges(data=True):
        rel_type = data["rel_type"].value if hasattr(data["rel_type"], "value") else str(data["rel_type"])
        # 双向边的反向边跳过（前端自己画双向箭头）
        if rel_type in bidirectional_types:
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)

        src_dep = graph.graph.nodes[u].get("deprecated", False)
        dst_dep = graph.graph.nodes[v].get("deprecated", False)

        links.append({
            "source": u,
            "target": v,
            "rel_type": rel_type.lower(),
            "bidirectional": rel_type in bidirectional_types,
            "deprecated_src": src_dep,
            "deprecated_dst": dst_dep,
        })

    return links


def _export_stats(graph: MemoryGraph) -> dict:
    deprecated = sum(1 for _, d in graph.graph.nodes(data=True) if d.get("deprecated"))
    forgotten = sum(1 for _, d in graph.graph.nodes(data=True) if d.get("forgotten"))
    orphaned = sum(
        1 for _, d in graph.graph.nodes(data=True)
        if not d.get("deprecated") and not d.get("forgotten")
        and d.get("metadata", {}).get("_in_degree", 0) == 0
        and d.get("metadata", {}).get("_out_degree", 0) == 0
    )
    return {
        "total_nodes": graph.node_count,
        "total_edges": graph.edge_count,
        "deprecated": deprecated,
        "forgotten": forgotten,
        "orphans": orphaned,
    }
