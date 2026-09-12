# SPDX-License-Identifier: AGPL-3.0-only

"""
知识图谱模块：有向带类型的记忆图谱

基于 networkx 实现，支持：
- 节点类型标注（STATUS, ACTION, THING 等）
- 边类型标注（8 种 RelationType）
- 跳转轴扩展查询
"""

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx

from dba_pipeline.core.jump_axis import (
    NodeType, RelationType,
    get_jump_weight, expand_weights,
)


class MemoryGraph:
    """有向带类型的记忆图谱"""

    def __init__(self):
        self.graph = nx.DiGraph()

    # ---- 构建 ----

    def add_memory(
        self,
        memory_id: str,
        content: str,
        node_type: NodeType,
        metadata: Optional[Dict] = None,
        deprecated: bool = False,
        forgotten: bool = False,
    ):
        """添加记忆节点"""
        self.graph.add_node(
            memory_id,
            content=content,
            node_type=node_type,
            metadata=metadata or {},
            deprecated=deprecated,
            forgotten=forgotten,
        )

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: RelationType,
        weight: float = 1.0,
    ):
        """添加有向带类型边
        
        Args:
            from_id: 源节点
            to_id: 目标节点
            rel_type: 关系类型
            weight: 边权重
        """
        self.graph.add_edge(
            from_id, to_id,
            rel_type=rel_type,
            weight=weight,
        )

    def add_edge_bidirectional(
        self,
        id_a: str,
        id_b: str,
        rel_type: RelationType,
        weight: float = 1.0,
    ):
        """添加双向边（用于 SCENARIO, SOCIAL, ATTRIBUTE 等双向关系）"""
        self.add_edge(id_a, id_b, rel_type, weight)
        self.add_edge(id_b, id_a, rel_type, weight)

    # ---- 查询 ----

    def get_node(self, node_id: str) -> Optional[Dict]:
        """获取节点数据"""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_node_type(self, node_id: str) -> Optional[NodeType]:
        """获取节点类型"""
        node = self.get_node(node_id)
        return node["node_type"] if node else None

    def get_neighbors(
        self,
        node_id: str,
    ) -> List[Tuple[str, RelationType, bool]]:
        """获取节点的所有邻居
        
        Returns:
            [(neighbor_id, relation_type, is_reverse), ...]
            is_reverse=True 表示是反向边（目标→源）
        """
        neighbors = []

        # 出边（正向）
        for _, to_id, edge_data in self.graph.out_edges(node_id, data=True):
            neighbors.append((to_id, edge_data["rel_type"], False))

        # 入边（反向）
        for from_id, _, edge_data in self.graph.in_edges(node_id, data=True):
            neighbors.append((from_id, edge_data["rel_type"], True))

        return neighbors

    def expand(
        self,
        seed_ids: List[str],
        path_tracker=None,
    ) -> Dict[str, float]:
        """从种子节点集按跳转轴规则扩展一轮

        Args:
            seed_ids: 当前轮的记忆节点 ID 列表
            path_tracker: 可选的 PathTracker，注入动态权重

        Returns:
            {memory_id: weighted_score} 扩展候选及跳转轴权重（含动态乘数）
        """
        scores: Dict[str, float] = {}

        for seed_id in seed_ids:
            source_type = self.get_node_type(seed_id)
            if source_type is None:
                continue

            for neighbor_id, rel_type, is_reverse in self.get_neighbors(seed_id):
                if neighbor_id in seed_ids:
                    continue  # 不回访
                w = get_jump_weight(source_type, rel_type, is_reverse)
                if w <= 0:
                    continue

                # 注入 PathTracker 动态权重
                if path_tracker is not None:
                    dyn_mult = path_tracker.get_edge_weight_multiplier(
                        seed_id, neighbor_id, rel_type.value
                    )
                    w *= dyn_mult

                if neighbor_id not in scores:
                    scores[neighbor_id] = w
                else:
                    scores[neighbor_id] = max(scores[neighbor_id], w)

        return scores

    def expand_with_trace(
        self,
        seed_ids: List[str],
        path_tracker=None,
    ) -> Dict[str, Dict]:
        """扩展一轮，并记录每条候选边的来源（父节点 + 关系类型 + 方向）

        与 expand 语义一致：同一 neighbor 保留权重最高的来源边（树结构）。

        Returns:
            {neighbor_id: {"weight": float, "from": seed_id,
                           "rel_type": RelationType, "is_reverse": bool}}
        """
        result: Dict[str, Dict] = {}

        for seed_id in seed_ids:
            source_type = self.get_node_type(seed_id)
            if source_type is None:
                continue

            for neighbor_id, rel_type, is_reverse in self.get_neighbors(seed_id):
                if neighbor_id in seed_ids:
                    continue  # 不回访
                w = get_jump_weight(source_type, rel_type, is_reverse)
                if w <= 0:
                    continue

                # 注入 PathTracker 动态权重
                if path_tracker is not None:
                    dyn_mult = path_tracker.get_edge_weight_multiplier(
                        seed_id, neighbor_id, rel_type.value
                    )
                    w *= dyn_mult

                if neighbor_id not in result or w > result[neighbor_id]["weight"]:
                    result[neighbor_id] = {
                        "weight": w,
                        "from": seed_id,
                        "rel_type": rel_type,
                        "is_reverse": is_reverse,
                    }

        return result

    def get_content(self, node_id: str) -> Optional[str]:
        """获取节点内容"""
        node = self.get_node(node_id)
        return node["content"] if node else None

    def get_contents(self, node_ids: List[str]) -> List[Optional[str]]:
        """批量获取节点内容，保留 None 占位以维持与 node_ids 一一对应"""
        return [self.get_content(nid) for nid in node_ids]

    # ---- 统计 ----

    @property
    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    # ---- 序列化 ----

    # 需要随节点持久化的扩展字段：新增节点字段时在此登记，
    # to_dict / from_dict 会自动透传（不登记则落盘时静默丢失）。
    # evidence：原文支撑片段（opt-in，见 MemoryDBA.enable_evidence）
    # source_ref：该节点最近一次内容/证据来自哪个批次（opt-in，见 enable_source_store）
    PERSISTED_NODE_FIELDS = ("speaker", "timestamp", "evidence", "source_ref")

    def to_dict(self) -> dict:
        """导出为可持久化的字典格式（兼容 YAML checkpoint）

        Returns:
            {"nodes": [...], "edges": [...]}
            节点包含: id, content, type (node_type.value), deprecated, forgotten
            边包含:   from, to, type (rel_type.value)
            双向边仅存一条（方向无关），标记 bidirectional: true
        """
        nodes = []
        for nid, ndata in self.graph.nodes(data=True):
            node = {
                "id": nid,
                "content": ndata.get("content", ""),
                "type": ndata["node_type"].value if hasattr(ndata["node_type"], "value") else str(ndata["node_type"]),
                "deprecated": ndata.get("deprecated", False),
                "forgotten": ndata.get("forgotten", False),
            }
            # 扩展字段透传（登记见 PERSISTED_NODE_FIELDS）
            for field in self.PERSISTED_NODE_FIELDS:
                if field in ndata:
                    node[field] = ndata[field]
            nodes.append(node)

        edges = []
        seen = set()
        for u, v, edata in self.graph.edges(data=True):
            # 双向边去重：如果 v→u 也存在且类型相同，只存一条
            rel_type_val = edata["rel_type"].value if hasattr(edata["rel_type"], "value") else str(edata["rel_type"])
            edge_key = tuple(sorted([u, v]) + [rel_type_val])
            if edge_key in seen:
                continue
            seen.add(edge_key)

            is_bidir = self.graph.has_edge(v, u) and self.graph.edges[v, u].get("rel_type") == edata["rel_type"]
            edge = {"from": u, "to": v, "type": rel_type_val}
            if is_bidir:
                edge["bidirectional"] = True
            edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGraph":
        """从字典还原 MemoryGraph 实例

        Args:
            data: {"nodes": [...], "edges": [...]}
        """
        g = cls()
        for nd in data.get("nodes", []):
            g.add_memory(
                memory_id=nd["id"],
                content=nd.get("content", ""),
                node_type=NodeType(nd["type"]) if nd.get("type") else NodeType.THING,
                deprecated=nd.get("deprecated", False),
                forgotten=nd.get("forgotten", False),
            )
            # 扩展字段回填（登记见 PERSISTED_NODE_FIELDS）
            for field in cls.PERSISTED_NODE_FIELDS:
                if field in nd:
                    g.graph.nodes[nd["id"]][field] = nd[field]

        for ed in data.get("edges", []):
            rt = RelationType(ed["type"]) if ed.get("type") else RelationType.CAUSAL
            if ed.get("bidirectional"):
                g.add_edge_bidirectional(ed["from"], ed["to"], rt)
            else:
                g.add_edge(ed["from"], ed["to"], rt)

        return g

    def to_mermaid(self, max_nodes: int = 50) -> str:
        """导出 Mermaid 流程图语法（用于文档和调试）

        Args:
            max_nodes: 最多包含的节点数（防止大图输出过长）

        Returns:
            Mermaid 流程图字符串，可直接粘贴到 ```mermaid 代码块中
        """
        # 颜色映射
        type_colors = {
            "status": "#5DADE2", "reason": "#E74C3C", "action": "#F39C12",
            "thing": "#27AE60", "person": "#A569BD", "emotion": "#E91E63",
        }
        lines = ["graph LR"]

        # 采样节点（按度数排序取 top-N）
        nodes_data = []
        for nid in self.graph.nodes():
            deg = self.graph.in_degree(nid) + self.graph.out_degree(nid)
            ndata = self.graph.nodes[nid]
            nodes_data.append((nid, deg, ndata))

        nodes_data.sort(key=lambda x: -x[1])
        sampled = nodes_data[:max_nodes]
        sampled_ids = {n[0] for n in sampled}

        # 节点定义
        for nid, _, ndata in sampled:
            nt_val = ndata["node_type"].value if hasattr(ndata["node_type"], "value") else str(ndata["node_type"])
            color = type_colors.get(nt_val, "#888")
            label = (ndata.get("content", "") or nid)[:30].replace('"', "")
            tag = ""
            if ndata.get("deprecated"):
                tag = " [废弃]"
            elif ndata.get("forgotten"):
                tag = " [遗忘]"
            lines.append(f'    {nid}["{label}{tag}"]')

        # 边定义（只包含采样节点间的边）
        seen = set()
        for u, v, edata in self.graph.edges(data=True):
            if u not in sampled_ids or v not in sampled_ids:
                continue
            rt_val = edata["rel_type"].value if hasattr(edata["rel_type"], "value") else str(edata["rel_type"])
            edge_key = tuple(sorted([u, v]) + [rt_val])
            if edge_key in seen:
                continue
            seen.add(edge_key)

            is_bidir = self.graph.has_edge(v, u) and self.graph.edges[v, u].get("rel_type") == edata["rel_type"]
            if is_bidir:
                lines.append(f'    {u} <-->|{rt_val}| {v}')
            else:
                lines.append(f'    {u} -->|{rt_val}| {v}')

        return "\n".join(lines)
