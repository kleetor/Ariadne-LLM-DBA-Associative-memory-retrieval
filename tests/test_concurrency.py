# SPDX-License-Identifier: AGPL-3.0-only

"""并发锁（共享 RLock）单元测试"""
import threading

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.mcp_server import DBAServer
from dba_pipeline.extraction.graph_builder import GraphBuilder


class FakeVectorStore:
    store = None  # 无向量能力，去重走精确匹配回退

    def add_memories(self, memory_ids, contents):
        pass

    def update_memories(self, memory_ids, contents):
        pass


def test_db_server_default_lock_is_reentrant():
    server = DBAServer(MemoryGraph())
    lock = server._lock
    lock.acquire()
    try:
        # RLock 可重入：同一线程再次 acquire 立即成功；普通 Lock 会返回 False
        assert lock.acquire(timeout=0) is True
        lock.release()
    finally:
        lock.release()


def test_intervene_reentrant_save_no_deadlock(tmp_path):
    server = DBAServer(MemoryGraph(), yaml_path=str(tmp_path / "g.yaml"),
                       vector_store=FakeVectorStore())
    # intervene 内部调用 _save，二者共享同一把 RLock，可重入不应死锁
    result = server.intervene("create_node", {"node_type": "STATUS", "content": "hello"})
    assert result.get("success") is True


def test_graph_builder_and_db_server_share_lock():
    graph = MemoryGraph()
    lock = threading.RLock()
    builder = GraphBuilder(graph=graph, vector_store=FakeVectorStore(), lock=lock)
    server = DBAServer(graph, vector_store=FakeVectorStore(), lock=lock)
    # 两个修改入口共享同一把锁，从而串行化 graph 修改
    assert builder._lock is server._lock


def test_graph_builder_shared_lock_applies_ops():
    graph = MemoryGraph()
    builder = GraphBuilder(graph=graph, vector_store=FakeVectorStore(),
                           lock=threading.RLock())
    result = builder.apply_ops(
        [{"action": "create", "node_type": "STATUS", "content": "x", "temp_id": "t1"}],
        [],
    )
    assert len(result["created_ids"]) == 1
