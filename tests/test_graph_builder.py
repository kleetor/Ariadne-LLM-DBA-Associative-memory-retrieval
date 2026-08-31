# SPDX-License-Identifier: AGPL-3.0-only

"""图谱构建器单元测试：CRUD / 去重 / 事务 / 孤立遗忘"""
import numpy as np

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType, RelationType
from dba_pipeline.extraction.graph_builder import GraphBuilder


class FakeVectorStore:
    """GraphBuilder 单元测试用的向量存储桩。"""

    def __init__(self, store=None, vectors=None, embed_fn=None, fail_on_add=False):
        self.store = store  # None 表示无向量能力
        self._content_vectors = dict(vectors or {})
        self._embed_fn = embed_fn or (lambda text: np.array([1.0, 0.0]))
        self.fail_on_add = fail_on_add
        self.added = []

    def _embed(self, text):
        return np.asarray(self._embed_fn(text), dtype=float)

    def search_by_vector(self, vector, k=5):
        return [(mid, 0.0) for mid in list(self._content_vectors.keys())[:k]]

    def search(self, query, k=5):
        return []

    def add_memories(self, memory_ids, contents):
        if self.fail_on_add:
            raise RuntimeError("add_memories failed")
        self.added.extend(memory_ids)

    def update_memories(self, memory_ids, contents):
        for mid, content in zip(memory_ids, contents):
            self._content_vectors[mid] = self._embed(content)


def _builder(store=None, vectors=None, orphan_threshold=3):
    graph = MemoryGraph()
    graph.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    vs = FakeVectorStore(store=store, vectors=vectors)
    builder = GraphBuilder(graph, vs, orphan_threshold=orphan_threshold)
    return graph, builder, vs


def test_create_node_allocates_id():
    graph, builder, _ = _builder(store=object())
    result = builder.apply_ops(
        [{"action": "create", "content": "用户养了一只猫", "node_type": "STATUS"}], []
    )
    assert result["created_ids"] == ["n2"]
    assert graph.node_count == 2


def test_exact_match_dedup_without_vectors():
    # P2-4：无向量能力时精确匹配去重，不再静默失效
    graph, builder, _ = _builder(store=None)
    result = builder.apply_ops(
        [{"action": "create", "content": "用户喜欢喝咖啡", "node_type": "STATUS"}], []
    )
    assert result["created_ids"] == []
    assert graph.node_count == 1


def test_cosine_dedup_with_vectors():
    # P2-5：有向量能力时用精确余弦相似度去重
    vectors = {"n1": np.array([1.0, 0.0])}
    embed_fn = lambda text: np.array([1.0, 0.0])  # 与 n1 完全同向

    graph = MemoryGraph()
    graph.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    vs = FakeVectorStore(store=object(), vectors=vectors, embed_fn=embed_fn)
    builder = GraphBuilder(graph, vs)

    result = builder.apply_ops(
        [{"action": "create", "content": "用户喜欢喝咖啡", "node_type": "STATUS"}], []
    )
    assert result["created_ids"] == []
    assert graph.node_count == 1


def test_update_and_deprecate():
    graph, builder, _ = _builder(store=object())
    builder.apply_ops(
        [{"action": "update", "target_id": "n1", "content": "用户改喝茶了", "reason": "改变"}], []
    )
    assert graph.get_node("n1")["content"] == "用户改喝茶了"

    builder.apply_ops(
        [{"action": "deprecate", "target_id": "n1", "reason": "过时"}], []
    )
    assert graph.get_node("n1")["deprecated"] is True


def test_create_edge_and_delete():
    graph, builder, _ = _builder(store=object())
    graph.add_memory("n2", "用户加班", NodeType.REASON)

    builder.apply_ops(
        [], [{"action": "create", "from": "n1", "to": "n2", "rel_type": "CAUSAL"}]
    )
    assert graph.graph.has_edge("n1", "n2")

    builder.apply_ops(
        [], [{"action": "delete", "from": "n1", "to": "n2", "reason": "错误"}]
    )
    assert not graph.graph.has_edge("n1", "n2")


def test_transaction_rollback_on_add_failure():
    graph = MemoryGraph()
    graph.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    vs = FakeVectorStore(store=object(), fail_on_add=True)
    builder = GraphBuilder(graph, vs)

    result = builder.apply_ops(
        [{"action": "create", "content": "新节点", "node_type": "ACTION"}], []
    )
    assert result["errors"] != []
    assert graph.node_count == 1  # 新节点已被回滚


def test_orphan_forgotten_after_threshold():
    graph = MemoryGraph()
    graph.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    graph.add_memory("n2", "用户加班", NodeType.REASON)
    graph.add_edge("n1", "n2", RelationType.CAUSAL)  # n1、n2 非孤立

    vs = FakeVectorStore(store=None)
    builder = GraphBuilder(graph, vs, orphan_threshold=3)

    builder.apply_ops([{"action": "create", "content": "孤立节点", "node_type": "ACTION"}], [])
    builder.apply_ops([], [])
    builder.apply_ops([], [])

    assert graph.get_node("n3")["forgotten"] is True
    assert graph.get_node("n1")["forgotten"] is False
