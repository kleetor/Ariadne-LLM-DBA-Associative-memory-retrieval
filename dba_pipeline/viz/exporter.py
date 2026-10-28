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
    degrees = precompute_degrees(graph)
    nodes = _export_nodes(graph, degrees)
    links = _export_links(graph)
    stats = _export_stats(graph, degrees)

    return {"nodes": nodes, "links": links, "stats": stats}


def to_3dforcegraph_json(graph: MemoryGraph, indent: int = 2) -> str:
    """导出为 JSON 字符串"""
    return json.dumps(to_3dforcegraph(graph), indent=indent, ensure_ascii=False)


# ---- 内部 ----

def precompute_degrees(graph: MemoryGraph) -> dict:
    """计算每个节点的入/出度并**返回**，不修改图本身。

    早期实现把这些度写进 `node["metadata"]`，等于在"只读导出"里改动了共享的
    权威图对象，与并发读写存在竞态；改为返回值传递。
    """
    return {
        nid: (graph.graph.in_degree(nid), graph.graph.out_degree(nid))
        for nid in graph.graph.nodes()
    }


# 需要从 graph 节点 metadata 里透传到前端的字段
_METADATA_KEYS = ("channel", "title", "tags", "observed_at", "created_at",
                  "fidelity", "source_hint", "time_inferred")


def _export_nodes(graph: MemoryGraph, degrees: dict = None) -> list:
    if degrees is None:
        degrees = precompute_degrees(graph)
    nodes = []
    for nid, data in graph.graph.nodes(data=True):
        raw_type = data["node_type"].value if hasattr(data["node_type"], "value") else str(data["node_type"])
        in_deg, out_deg = degrees.get(nid, (0, 0))

        # 只挑约定字段透传，避免前端拿到内部 dump 的杂项 key
        raw_meta = data.get("metadata") or {}
        meta = {k: raw_meta[k] for k in _METADATA_KEYS if k in raw_meta}

        nodes.append({
            "id": nid,
            "label": (data.get("content") or "")[:50],
            "node_type": raw_type.upper(),
            "content": data.get("content", ""),
            "deprecated": data.get("deprecated", False),
            "forgotten": data.get("forgotten", False),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "radius": _RADIUS_MAP.get(raw_type.upper(), 2.5),
            "metadata": meta,
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


def _export_stats(graph: MemoryGraph, degrees: dict) -> dict:
    deprecated = 0
    forgotten = 0
    orphaned = 0
    for nid, d in graph.graph.nodes(data=True):
        if d.get("deprecated"):
            deprecated += 1
        if d.get("forgotten"):
            forgotten += 1
        if not d.get("deprecated") and not d.get("forgotten"):
            in_deg, out_deg = degrees.get(nid, (0, 0))
            if in_deg == 0 and out_deg == 0:
                orphaned += 1
    return {
        "total_nodes": graph.node_count,
        "total_edges": graph.edge_count,
        "deprecated": deprecated,
        "forgotten": forgotten,
        "orphans": orphaned,
    }
