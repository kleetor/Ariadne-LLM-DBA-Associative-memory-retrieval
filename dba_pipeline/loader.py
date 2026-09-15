# SPDX-License-Identifier: AGPL-3.0-only

"""
YAML 数据加载器

从 YAML 文件构建 MemoryGraph + 测试查询。
替代 data/sample_data.py 的硬编码方式。
"""

import logging
import os
import yaml
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType, RelationType


# 字符串 → 枚举映射
_NODE_TYPE_MAP = {
    "status": NodeType.STATUS,
    "reason": NodeType.REASON,
    "action": NodeType.ACTION,
    "thing": NodeType.THING,
    "person": NodeType.PERSON,
    "emotion": NodeType.EMOTION,
}

_REL_TYPE_MAP = {
    "causal": RelationType.CAUSAL,
    "scenario": RelationType.SCENARIO,
    "sequence": RelationType.SEQUENCE,
    "preference": RelationType.PREFERENCE,
    "social": RelationType.SOCIAL,
    "attribute": RelationType.ATTRIBUTE,
    "temporal": RelationType.TEMPORAL,
    "taxonomic": RelationType.TAXONOMIC,
}


def load_graph(yaml_path: str) -> MemoryGraph:
    """从 YAML 文件加载知识图谱

    YAML 格式见 data/formal/stress.yaml
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    g = MemoryGraph()

    # 加载节点（补齐 deprecated / forgotten 状态，避免重启后丢失）
    for node in data["nodes"]:
        # 未知类型只跳过并告警：直接下标会让整个 MCP 进程在启动时崩掉。
        # 归一化大小写：映射表键是小写，但手写/外部 YAML 可能用枚举名（"STATUS"）。
        node_type = _NODE_TYPE_MAP.get(str(node.get("type") or "").lower())
        if node_type is None:
            logger.warning(f"跳过未知节点类型 {node.get('type')!r}（节点 {node.get('id')}）")
            continue
        g.add_memory(
            memory_id=node["id"],
            content=node["content"],
            node_type=node_type,
            deprecated=node.get("deprecated", False),
            forgotten=node.get("forgotten", False),
        )
        # 扩展字段回填（登记见 MemoryGraph.PERSISTED_NODE_FIELDS）
        for field in MemoryGraph.PERSISTED_NODE_FIELDS:
            if field in node:
                g.graph.nodes[node["id"]][field] = node[field]

    # 加载边
    for edge in data.get("edges", []):
        rel_type = _REL_TYPE_MAP.get(str(edge.get("type") or "").lower())
        if rel_type is None:
            logger.warning(
                f"跳过未知边类型 {edge.get('type')!r}"
                f"（{edge.get('from')} -> {edge.get('to')}）"
            )
            continue
        if edge.get("bidirectional"):
            g.add_edge_bidirectional(edge["from"], edge["to"], rel_type)
        else:
            g.add_edge(edge["from"], edge["to"], rel_type)

    return g


def load_into(graph: MemoryGraph, yaml_path: str) -> MemoryGraph:
    """把 YAML **就地**加载进已有的 MemoryGraph 实例。

    与 :func:`load_graph` 的区别：复用同一个 MemoryGraph 实例及其底层 nx 图对象，
    因此其它持有该实例引用的组件（GraphBuilder / Retriever / VectorStore 调用方等）
    会同步看到新数据——这是 MCP 在跨进程写入前「重载外部改动」所必需的
    （若是新建实例再赋值，别的组件仍指向旧对象，重载等于没做）。
    """
    fresh = load_graph(yaml_path)
    graph.graph.clear()  # 保留同一 nx 图对象，仅替换其内容
    graph.graph.add_nodes_from(fresh.graph.nodes(data=True))
    graph.graph.add_edges_from(fresh.graph.edges(data=True))
    return graph


def load_queries(yaml_path: str) -> Dict[str, Dict]:
    """从 YAML 文件加载标注查询

    Returns:
        {query_text: {"expected": [...], "unexpected": [...]}}
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    queries = {}
    for q in data.get("queries", []):
        queries[q["text"]] = {
            "expected": q.get("expected", []),
            "unexpected": q.get("unexpected", []),
        }
    return queries


def load_all(
    yaml_path: str,
) -> Tuple[MemoryGraph, Dict[str, Dict]]:
    """一次性加载图谱 + 查询"""
    graph = load_graph(yaml_path)
    queries = load_queries(yaml_path)
    return graph, queries


def load_multi_graph(
    yaml_paths: List[str],
) -> Tuple[MemoryGraph, Dict[str, Dict]]:
    """从多个 YAML 文件合并加载

    Args:
        yaml_paths: YAML 文件路径列表

    Returns:
        合并后的图谱和查询
    """
    graph = MemoryGraph()
    all_queries = {}

    for path in yaml_paths:
        g = load_graph(path)
        # 合并节点和边到同一个 graph（networkx 自动去重，保留节点标志）
        for nid in g.graph.nodes():
            if nid not in graph.graph:
                node_data = g.get_node(nid)
                graph.add_memory(
                    memory_id=nid,
                    content=node_data["content"],
                    node_type=node_data["node_type"],
                    deprecated=node_data.get("deprecated", False),
                    forgotten=node_data.get("forgotten", False),
                )
                # 扩展字段回填（登记见 MemoryGraph.PERSISTED_NODE_FIELDS）
                for field in MemoryGraph.PERSISTED_NODE_FIELDS:
                    if field in node_data:
                        graph.graph.nodes[nid][field] = node_data[field]
        for u, v, edge_data in g.graph.edges(data=True):
            if not graph.graph.has_edge(u, v):
                graph.add_edge(u, v, edge_data["rel_type"])
            else:
                existing = graph.graph.edges[u, v].get("rel_type")
                if existing != edge_data["rel_type"]:
                    logger.warning(
                        f"边 {u}-->{v} 已存在（{existing}），跳过类型不同的边 {edge_data['rel_type']}"
                    )

        qs = load_queries(path)
        all_queries.update(qs)

    return graph, all_queries
