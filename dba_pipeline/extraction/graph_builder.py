# SPDX-License-Identifier: AGPL-3.0-only

"""
图谱构建器：执行 LLM DBA 的操作指令，维护 MemoryGraph

纯工程逻辑——不调用 LLM。负责：
- 节点 CRUD：create / update / fix_type / deprecate
- 边 CRUD：create / delete
- 语义去重（基于 embedding 余弦相似度）
- 校验（类型合法性、边引用完整性、权重矩阵兼容性）
- 孤立节点检测（遗忘机制）
"""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType, RelationType, JUMP_AXIS_RULES, get_jump_weight
from dba_pipeline.embedding.store import VectorStore

logger = logging.getLogger(__name__)


class _WriteTransaction:
    """轻量写入事务：追踪 graph 变更，支持出错时回滚"""

    def __init__(self):
        # 新增的节点 ID（用于回滚时删除）
        self._created_nodes: List[str] = []
        # 创建过的边 (from, to, rel_type)
        self._created_edges: List[Tuple[str, str]] = []
        # 被更新节点的旧内容 {node_id: old_content}
        self._updated_nodes: Dict[str, str] = {}
        # 被修正类型节点的旧类型 {node_id: old_node_type}
        self._fixed_types: Dict[str, NodeType] = {}
        # 被废弃节点的旧废弃标记 {node_id: old_deprecated}
        self._deprecated_nodes: Dict[str, bool] = {}
        # 被删除的边 (from, to, rel_type)（回滚时恢复）
        self._deleted_edges: List[Tuple[str, str, RelationType]] = []
        # 统计快照（回滚时恢复）
        self._stats_snapshot: Dict = {}

    def track_create_node(self, node_id: str):
        self._created_nodes.append(node_id)

    def track_create_edge(self, from_id: str, to_id: str):
        self._created_edges.append((from_id, to_id))

    def track_update_node(self, node_id: str, old_content: str):
        if node_id not in self._updated_nodes:
            self._updated_nodes[node_id] = old_content

    def track_fix_type(self, node_id: str, old_type: NodeType):
        if node_id not in self._fixed_types:
            self._fixed_types[node_id] = old_type

    def track_deprecate_node(self, node_id: str, old_deprecated: bool):
        if node_id not in self._deprecated_nodes:
            self._deprecated_nodes[node_id] = old_deprecated

    def track_delete_edge(self, from_id: str, to_id: str, rel_type: RelationType):
        self._deleted_edges.append((from_id, to_id, rel_type))

    def snapshot_stats(self, stats: Dict):
        self._stats_snapshot = dict(stats)

    def rollback(self, builder: "GraphBuilder"):
        """回滚 graph 变更"""
        for from_id, to_id in self._created_edges:
            if builder.graph.graph.has_edge(from_id, to_id):
                builder.graph.graph.remove_edge(from_id, to_id)
        for node_id in self._created_nodes:
            if node_id in builder.graph.graph:
                builder.graph.graph.remove_node(node_id)
        for node_id, old_content in self._updated_nodes.items():
            if node_id in builder.graph.graph:
                builder.graph.graph.nodes[node_id]["content"] = old_content
        for node_id, old_type in self._fixed_types.items():
            if node_id in builder.graph.graph:
                builder.graph.graph.nodes[node_id]["node_type"] = old_type
        for node_id, old_deprecated in self._deprecated_nodes.items():
            if node_id in builder.graph.graph:
                builder.graph.graph.nodes[node_id]["deprecated"] = old_deprecated
        for from_id, to_id, rel_type in self._deleted_edges:
            if (from_id in builder.graph.graph and to_id in builder.graph.graph
                    and not builder.graph.graph.has_edge(from_id, to_id)):
                builder.graph.graph.add_edge(from_id, to_id, rel_type=rel_type)
        if self._stats_snapshot:
            for key, val in self._stats_snapshot.items():
                builder.stats[key] = val
        logger.warning(
            f"事务回滚: 撤销 {len(self._created_nodes)} 节点, "
            f"{len(self._created_edges)} 边, {len(self._updated_nodes)} 更新, "
            f"{len(self._fixed_types)} 类型修正, {len(self._deprecated_nodes)} 废弃, "
            f"{len(self._deleted_edges)} 删边"
        )

    def commit(self):
        """提交事务（写入成功，清空追踪）"""
        self._created_nodes.clear()
        self._created_edges.clear()
        self._updated_nodes.clear()
        self._fixed_types.clear()
        self._deprecated_nodes.clear()
        self._deleted_edges.clear()
        self._stats_snapshot.clear()


class GraphBuilder:
    """图谱构建与维护器"""

    def __init__(
        self,
        graph: MemoryGraph,
        vector_store: VectorStore,
        dedup_threshold: float = 0.85,
        orphan_threshold: int = 3,
        lock=None,
    ):
        """
        Args:
            graph: 目标 MemoryGraph
            vector_store: 向量存储（用于语义去重）
            dedup_threshold: 语义去重余弦阈值（默认 0.85，生产系统已验证）
            orphan_threshold: 连续多少次维护后仍为孤立节点则遗忘
            lock: 可选的共享锁（RLock），用于串行化 graph 修改（与 DBAServer 共享）
        """
        self.graph = graph
        self.vector_store = vector_store
        self.dedup_threshold = dedup_threshold
        self.orphan_threshold = orphan_threshold
        self._lock = lock

        # 孤立节点追踪: {node_id: consecutive_orphan_count}
        self._orphan_tracker: Dict[str, int] = {}

        # 统计
        self.stats = {
            "nodes_created": 0,
            "nodes_updated": 0,
            "nodes_fixed": 0,
            "nodes_deprecated": 0,
            "nodes_forgotten": 0,
            "edges_created": 0,
            "edges_deleted": 0,
            "ops_skipped": 0,
        }

    # ---- 主入口 ----

    def apply_ops(
        self,
        node_ops: List[Dict],
        edge_ops: List[Dict],
        batch_timestamp=None,
        batch_id: str = None,
    ) -> Dict:
        """执行 LLM DBA 输出的操作指令（带事务保护 + 锁保护）

        batch_timestamp: 本批对话的发生时间（Unix 毫秒）。非 None 时写入本批
        新建节点的 timestamp 字段——时间戳不依赖 LLM 产出，由调用方按对话
        元数据确定性注入，避免幻觉与门控遗漏。
        batch_id: 本批对话的批次标识（opt-in，见 MemoryDBA.enable_source_store），
        写入节点的 source_ref，供溯源巡检反查原文。
        """
        prev_ts = getattr(self, "_batch_timestamp", None)
        prev_id = getattr(self, "_batch_id", None)
        self._batch_timestamp = batch_timestamp
        self._batch_id = batch_id
        try:
            if self._lock is None:
                return self._apply_ops_impl(node_ops, edge_ops)
            with self._lock:
                return self._apply_ops_impl(node_ops, edge_ops)
        finally:
            self._batch_timestamp = prev_ts
            self._batch_id = prev_id

    def _apply_ops_impl(
        self,
        node_ops: List[Dict],
        edge_ops: List[Dict],
    ) -> Dict:
        """apply_ops 实际实现（在锁内调用）

        Args:
            node_ops: 节点操作列表 [{"action": "create", ...}, ...]
            edge_ops: 边操作列表 [{"action": "create", ...}, ...]

        Returns:
            {"created_ids": [...], "skipped": [...], "errors": [...]}

        事务保护：跟踪写入操作，如 graph 写入成功但 vector_store 写入失败，
        回滚 graph 变更并记录错误。
        """
        created_ids = []
        skipped = []
        errors = []

        # 事务追踪
        txn = _WriteTransaction()

        # 节点操作
        id_mapping = {}  # LLM 临时 ID → 图谱全局 ID
        for op in node_ops or []:
            try:
                result = self._apply_node_op(op, id_mapping, txn)
                if result:
                    created_ids.append(result)
            except ValueError as e:
                skipped.append({"op": op, "reason": str(e)})
                self.stats["ops_skipped"] += 1
            except Exception as e:
                errors.append({"op": op, "error": str(e)})
                logger.error(f"节点操作失败: {op}, {e}")
                txn.rollback(self)
                break

        # 边操作（仅当节点操作无致命错误时执行）
        if not errors:
            resolved_edge_ops = self._resolve_edge_ids(edge_ops or [], id_mapping)
            for op in resolved_edge_ops:
                try:
                    self._apply_edge_op(op, txn)
                except ValueError as e:
                    skipped.append({"op": op, "reason": str(e)})
                    self.stats["ops_skipped"] += 1
                except Exception as e:
                    errors.append({"op": op, "error": str(e)})
                    logger.error(f"边操作失败: {op}, {e}")
                    txn.rollback(self)
                    break

        # 孤立节点检查（仅在无致命错误时执行）
        if not errors:
            self._check_orphans()

        txn.commit()

        return {
            "created_ids": created_ids,
            "skipped": skipped,
            "errors": errors,
        }

    # ---- 节点操作 ----

    def _apply_node_op(self, op: Dict, id_mapping: Dict[str, str], txn: _WriteTransaction) -> Optional[str]:
        """执行单个节点操作，返回新创建的节点 ID（或 None）"""
        action = op.get("action")
        txn.snapshot_stats(self.stats)

        if action == "create":
            return self._create_node(op, id_mapping, txn)
        elif action == "update":
            self._update_node(op, txn)
        elif action == "fix_type":
            self._fix_node_type(op, txn)
        elif action == "deprecate":
            self._deprecate_node(op, txn)
        else:
            raise ValueError(f"未知节点操作: {action}")

        return None

    def _create_node(self, op: Dict, id_mapping: Dict[str, str], txn: _WriteTransaction) -> Optional[str]:
        """创建新节点（带去重 + 事务追踪）"""
        content = op["content"]
        node_type_str = op["node_type"]

        node_type = self._parse_node_type(node_type_str)

        dup_id = self._find_duplicate(content)
        if dup_id:
            id_mapping[op.get("temp_id", "")] = dup_id
            logger.info(f"节点去重: '{content[:40]}...' → 已有节点 {dup_id}")
            return None

        node_id = self._next_node_id()
        self.graph.add_memory(node_id, content, node_type)

        # 批次级时间戳注入（见 apply_ops 说明）；持久化由 PERSISTED_NODE_FIELDS 负责
        batch_ts = getattr(self, "_batch_timestamp", None)
        if batch_ts is not None:
            self.graph.graph.nodes[node_id]["timestamp"] = batch_ts

        # 原文支撑片段（opt-in，见 MemoryDBA.enable_evidence）；落盘同样由 PERSISTED_NODE_FIELDS 负责
        evidence = op.get("evidence")
        if evidence:
            self.graph.graph.nodes[node_id]["evidence"] = evidence

        # 溯源引用（opt-in，见 MemoryDBA.enable_source_store）：该节点出自哪个批次的对话
        batch_id = getattr(self, "_batch_id", None)
        if batch_id:
            self.graph.graph.nodes[node_id]["source_ref"] = batch_id

        # 写入向量库（失败时回滚 graph）
        try:
            self.vector_store.add_memories([node_id], [content])
        except Exception:
            self.graph.graph.remove_node(node_id)
            raise

        id_mapping[op.get("temp_id", "")] = node_id
        txn.track_create_node(node_id)

        self.stats["nodes_created"] += 1
        logger.info(f"创建节点 [{node_id}] {node_type_str}: {content[:50]}...")
        return node_id

    def _update_node(self, op: Dict, txn: _WriteTransaction):
        """更新已有节点内容（带事务追踪）"""
        target_id = op["target_id"]
        new_content = op["content"]
        reason = op.get("reason", "")

        node = self.graph.get_node(target_id)
        if node is None:
            raise ValueError(f"节点 {target_id} 不存在，无法更新")

        old_content = node["content"]
        txn.track_update_node(target_id, old_content)
        self.graph.graph.nodes[target_id]["content"] = new_content

        # 更新向量库（失败时回滚 content）
        try:
            self.vector_store.update_memories([target_id], [new_content])
        except Exception:
            self.graph.graph.nodes[target_id]["content"] = old_content
            raise

        # 原文支撑片段随内容一起刷新（写在向量更新成功之后，无需回滚）
        evidence = op.get("evidence")
        if evidence:
            self.graph.graph.nodes[target_id]["evidence"] = evidence

        # source_ref 跟着 evidence 走：两者必须同源，否则巡检会拿错批次的原文去核对
        batch_id = getattr(self, "_batch_id", None)
        if batch_id:
            self.graph.graph.nodes[target_id]["source_ref"] = batch_id

        self.stats["nodes_updated"] += 1
        logger.info(f"更新节点 [{target_id}]: '{old_content[:40]}...' → '{new_content[:40]}...' ({reason})")

    def _fix_node_type(self, op: Dict, txn: _WriteTransaction):
        """修正节点类型"""
        target_id = op["target_id"]
        new_type_str = op["node_type"]
        reason = op.get("reason", "")

        node = self.graph.get_node(target_id)
        if node is None:
            raise ValueError(f"节点 {target_id} 不存在，无法修正类型")

        new_type = self._parse_node_type(new_type_str)
        old_type = node["node_type"]
        txn.track_fix_type(target_id, old_type)
        self.graph.graph.nodes[target_id]["node_type"] = new_type

        self.stats["nodes_fixed"] += 1
        logger.info(f"修正节点类型 [{target_id}]: {old_type.value} → {new_type.value} ({reason})")

    def _deprecate_node(self, op: Dict, txn: _WriteTransaction):
        """标记节点失效"""
        target_id = op["target_id"]
        reason = op.get("reason", "")

        node = self.graph.get_node(target_id)
        if node is None:
            raise ValueError(f"节点 {target_id} 不存在，无法 deprecate")

        txn.track_deprecate_node(target_id, node.get("deprecated", False))
        self.graph.graph.nodes[target_id]["deprecated"] = True

        self.stats["nodes_deprecated"] += 1
        logger.info(f"Deprecate 节点 [{target_id}]: {reason}")

    # ---- 边操作 ----

    def _apply_edge_op(self, op: Dict, txn: _WriteTransaction):
        """执行单个边操作"""
        action = op.get("action")

        if action == "create":
            self._create_edge(op, txn)
        elif action == "delete":
            self._delete_edge(op, txn)
        else:
            raise ValueError(f"未知边操作: {action}")

    def _create_edge(self, op: Dict, txn: _WriteTransaction):
        """创建边（带校验 + 去重 + 事务追踪）"""
        from_id = op["from"]
        to_id = op["to"]
        rel_type_str = op["rel_type"]

        # 校验节点存在
        if self.graph.get_node(from_id) is None:
            raise ValueError(f"from 节点 {from_id} 不存在")
        if self.graph.get_node(to_id) is None:
            raise ValueError(f"to 节点 {to_id} 不存在")

        # 校验 rel_type
        rel_type = self._parse_rel_type(rel_type_str)

        # 校验权重矩阵兼容性
        from_type = self.graph.get_node_type(from_id)
        if from_type:
            fwd_weight = get_jump_weight(from_type, rel_type, is_reverse=False)
            if fwd_weight <= 0:
                logger.warning(
                    f"边 {from_id}--[{rel_type_str}]-->{to_id} "
                    f"权重为 0（{from_type.value} 节点不沿 {rel_type_str} 正向扩展），跳过"
                )
                self.stats["ops_skipped"] += 1
                return

        # 边去重：同类型边跳过；不同类型边为避免静默改写，记录并跳过
        if self.graph.graph.has_edge(from_id, to_id):
            existing = self.graph.graph.edges[from_id, to_id]
            if existing.get("rel_type") == rel_type:
                logger.debug(f"边 {from_id}--[{rel_type_str}]-->{to_id} 已存在，跳过")
                return
            logger.warning(
                f"边 {from_id}-->{to_id} 已存在（{existing.get('rel_type')}），"
                f"跳过不同 rel_type 的创建指令 {rel_type_str}"
            )
            return

        # 创建
        self.graph.add_edge(from_id, to_id, rel_type)
        txn.track_create_edge(from_id, to_id)

        # 双向边自动补全（反向边已存在时保留原类型，不覆盖）
        if rel_type in (RelationType.SCENARIO, RelationType.SOCIAL, RelationType.ATTRIBUTE):
            if self.graph.graph.has_edge(to_id, from_id):
                rev_existing = self.graph.graph.edges[to_id, from_id].get("rel_type")
                if rev_existing != rel_type:
                    logger.warning(
                        f"反向边 {to_id}-->{from_id} 已存在（{rev_existing}），跳过补边"
                    )
            else:
                self.graph.add_edge(to_id, from_id, rel_type)
                txn.track_create_edge(to_id, from_id)

        self.stats["edges_created"] += 1
        logger.info(f"创建边: {from_id} --[{rel_type_str}]--> {to_id}")

    def _delete_edge(self, op: Dict, txn: _WriteTransaction):
        """删除边"""
        from_id = op["from"]
        to_id = op["to"]
        reason = op.get("reason", "")

        if not self.graph.graph.has_edge(from_id, to_id):
            logger.debug(f"边 {from_id}-->{to_id} 不存在，跳过删除")
            return

        rel_type = self.graph.graph.edges[from_id, to_id].get("rel_type")
        txn.track_delete_edge(from_id, to_id, rel_type)
        self.graph.graph.remove_edge(from_id, to_id)

        # 双向边同步删除
        if rel_type in (RelationType.SCENARIO, RelationType.SOCIAL, RelationType.ATTRIBUTE):
            if self.graph.graph.has_edge(to_id, from_id):
                rev_rel = self.graph.graph.edges[to_id, from_id].get("rel_type")
                txn.track_delete_edge(to_id, from_id, rev_rel)
                self.graph.graph.remove_edge(to_id, from_id)

        self.stats["edges_deleted"] += 1
        logger.info(f"删除边: {from_id}-->{to_id} ({reason})")

    # ---- 去重 ----

    def _find_duplicate(self, content: str) -> Optional[str]:
        """语义去重：检查 content 是否与已有节点高度相似

        - 无向量能力（store 为 None）时回退到精确字符串匹配，避免静默失效
        - 有向量能力时用 FAISS 初筛 top-k 候选，再用缓存向量算精确余弦相似度
        """
        if self.graph.node_count == 0:
            return None

        # 无向量能力：回退精确字符串匹配
        if self.vector_store.store is None:
            for nid in self.graph.graph.nodes():
                node = self.graph.get_node(nid)
                if (node and not node.get("deprecated")
                        and node.get("content") == content):
                    logger.info(
                        f"节点精确去重: '{content[:40]}...' → 已有节点 {nid}"
                    )
                    return nid
            return None

        # 有向量能力：FAISS 初筛 top-k，再用缓存向量算精确余弦
        query_vec = self.vector_store._embed(content)
        candidates = self.vector_store.search_by_vector(query_vec, k=3)

        best_id, best_sim = None, 0.0
        for mid, _ in candidates:
            node = self.graph.get_node(mid)
            if node is None or node.get("deprecated"):
                continue
            cand_vec = self.vector_store._content_vectors.get(mid)
            if cand_vec is None:
                continue
            sim = float(
                np.dot(query_vec, cand_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(cand_vec) + 1e-8)
            )
            if sim > best_sim:
                best_sim, best_id = sim, mid

        if best_id is not None and best_sim > self.dedup_threshold:
            return best_id
        return None

    # ---- 孤立节点检测 ----

    def _check_orphans(self):
        """检查并处理孤立节点（入边=0 且 出边=0）"""
        for node_id in list(self.graph.graph.nodes()):
            node = self.graph.get_node(node_id)
            if node is None or node.get("deprecated") or node.get("forgotten"):
                continue

            in_degree = self.graph.graph.in_degree(node_id)
            out_degree = self.graph.graph.out_degree(node_id)

            if in_degree == 0 and out_degree == 0:
                self._orphan_tracker[node_id] = self._orphan_tracker.get(node_id, 0) + 1

                if self._orphan_tracker[node_id] >= self.orphan_threshold:
                    self.graph.graph.nodes[node_id]["forgotten"] = True
                    self.stats["nodes_forgotten"] += 1
                    logger.info(f"遗忘节点 [{node_id}]: 连续 {self.orphan_threshold} 次维护为孤立节点")
            else:
                # 重新获得连接，重置计数
                if node_id in self._orphan_tracker:
                    del self._orphan_tracker[node_id]

    # ---- 工具方法 ----

    def _parse_node_type(self, type_str: str) -> NodeType:
        """解析节点类型字符串（支持大写枚举名或小写 value）"""
        type_str = type_str.strip()
        # 先按枚举名匹配（如 "ACTION"）
        try:
            return NodeType[type_str.upper()]
        except KeyError:
            pass
        # 再按枚举值匹配（如 "action"）
        try:
            return NodeType(type_str.lower())
        except ValueError:
            raise ValueError(f"无效的节点类型: '{type_str}'，有效值: {[t.name for t in NodeType]}")

    def _parse_rel_type(self, type_str: str) -> RelationType:
        """解析边类型字符串（支持大写枚举名或小写 value）"""
        type_str = type_str.strip()
        try:
            return RelationType[type_str.upper()]
        except KeyError:
            pass
        try:
            return RelationType(type_str.lower())
        except ValueError:
            raise ValueError(f"无效的边类型: '{type_str}'，有效值: {[t.name for t in RelationType]}")

    def _next_node_id(self) -> str:
        """生成下一个节点 ID"""
        existing = set(self.graph.graph.nodes())
        idx = 1
        while f"n{idx}" in existing:
            idx += 1
        return f"n{idx}"

    def _resolve_edge_ids(
        self, edge_ops: List[Dict], id_mapping: Dict[str, str]
    ) -> List[Dict]:
        """将边操作中的临时 ID 替换为图谱全局 ID"""
        resolved = []
        for op in edge_ops:
            op = dict(op)  # 不修改原对象
            op["from"] = id_mapping.get(op["from"], op["from"])
            op["to"] = id_mapping.get(op["to"], op["to"])
            resolved.append(op)
        return resolved

    def reset_stats(self):
        """重置统计计数器"""
        for key in self.stats:
            self.stats[key] = 0

    def save_state(self) -> dict:
        """导出构建器运行时状态"""
        return {
            "_orphan_tracker": dict(self._orphan_tracker),
            "stats": dict(self.stats),
        }

    def load_state(self, state: dict):
        """恢复构建器运行时状态"""
        self._orphan_tracker = dict(state.get("_orphan_tracker", {}))
        if "stats" in state:
            for k in self.stats:
                self.stats[k] = state["stats"].get(k, self.stats[k])
