# SPDX-License-Identifier: AGPL-3.0-only

"""MCP 关键词检索与巡检口径测试。

`dba_add_note` 已移除（项目定位收敛为个人对话记忆），节点直接构造；
这里只验证关键词检索与 note/raw 在统计、巡检中的处理方式。
"""
import threading

from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.extraction.maintenance_scheduler import (
    MaintenanceScheduler, ScheduleConfig, TriggerReason,
)
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.mcp_server import DBAServer, NOTE_TEXT_LIMIT, TOOL_SCHEMAS


class FakeVectorStore:
    store = None

    def __init__(self):
        self.added = []
        self._content_vectors = {}

    def add_memories(self, memory_ids, contents):
        self.added.extend(memory_ids)
        for mid, content in zip(memory_ids, contents):
            self._content_vectors[mid] = content

    def update_memories(self, memory_ids, contents):
        pass

    def remove_memories(self, memory_ids):
        pass

    def reconcile(self, desired):
        return {"added": 0, "updated": 0, "removed": 0}


class FakeReranker:
    def __init__(self, order=None, fail=False):
        self.order = order
        self.fail = fail

    def rerank(self, query, documents):
        if self.fail:
            raise RuntimeError("rerank down")
        if self.order is not None:
            return [(i, 1.0 - i * 0.1) for i in self.order if i < len(documents)]
        return [(i, 0.5) for i in range(len(documents))]


def _server(tmp_path, vector_store=None, reranker=None):
    return DBAServer(
        MemoryGraph(),
        yaml_path=str(tmp_path / "g.yaml"),
        vector_store=vector_store if vector_store is not None else FakeVectorStore(),
        lock=threading.RLock(),
        reranker=reranker,
    )


def _node(server, nid, content, channel=None):
    """直接造一个节点（替代已移除的 add_note）。"""
    meta = {"channel": channel} if channel else {}
    server.graph.add_memory(nid, content, NodeType.THING, metadata=meta)
    return nid


def test_tool_schemas_exclude_add_note():
    names = [t["name"] for t in TOOL_SCHEMAS]
    assert "dba_add_note" not in names
    assert "dba_query_knowledge" in names
    assert len(names) == 10


def test_query_knowledge_returns_text_and_score(tmp_path):
    server = _server(tmp_path)
    _node(server, "n1", "接口超时统一为 30s，重试 2 次")
    out = server.query_knowledge("接口超时")
    assert out["returned"] == 1
    item = out["notes"][0]
    assert 0 < item["score"] <= 1.0
    assert "edge" not in item and "edges" not in item


def test_query_knowledge_covers_ordinary_nodes(tmp_path):
    """关键词检索覆盖全部节点，而不仅是历史 note。"""
    server = _server(tmp_path)
    _node(server, "n1", "小林在准备考研")
    out = server.query_knowledge("考研")
    assert [n["node_id"] for n in out["notes"]] == ["n1"]


def test_query_knowledge_truncates_long_text(tmp_path):
    server = _server(tmp_path)
    _node(server, "n1", "截断测试" + "a" * 1400)
    item = server.query_knowledge("截断测试")["notes"][0]
    assert item["truncated"] is True
    assert len(item["text"]) == NOTE_TEXT_LIMIT


def test_query_knowledge_no_match_and_empty_query(tmp_path):
    server = _server(tmp_path)
    _node(server, "n1", "接口超时统一为 30s")
    assert server.query_knowledge("完全无关") == {
        "notes": [], "returned": 0, "query": "完全无关", "method": "keyword",
    }
    assert "error" in server.query_knowledge("  ")


def test_query_knowledge_rerank_order_and_fallback(tmp_path):
    server = _server(tmp_path, reranker=FakeReranker(order=[1, 0]))
    _node(server, "n1", "接口超时方案 A")
    _node(server, "n2", "接口超时方案 B")
    out = server.query_knowledge("接口超时")
    assert out["method"] == "keyword+rerank"
    assert out["returned"] == 2

    (tmp_path / "sub").mkdir()
    failing = _server(tmp_path / "sub", reranker=FakeReranker(fail=True))
    _node(failing, "n1", "接口超时方案")
    out = failing.query_knowledge("接口超时")
    assert out["method"] == "keyword"
    assert out["returned"] == 1


def test_query_knowledge_works_without_vector_store(tmp_path):
    server = _server(tmp_path, vector_store=None)
    _node(server, "n1", "接口超时统一为 30s")
    out = server.query_knowledge("接口超时")
    assert out["returned"] == 1


def test_query_knowledge_single_term_is_enough(tmp_path):
    """最低命中口径：至少 1 个 term 即返回，不设覆盖率阈值。"""
    server = _server(tmp_path)
    _node(server, "n1", "接口超时统一为 30s")
    out = server.query_knowledge("接口 完全无关")
    assert out["returned"] == 1
    assert 0 < out["notes"][0]["score"] < 0.5


def test_stats_and_review_handle_special_channels(tmp_path):
    server = _server(tmp_path)
    _node(server, "n1", "普通记忆")
    _node(server, "n2", "历史项目笔记", channel="note")
    _node(server, "n3", "抽取失败时的兜底原文", channel="raw")

    stats = server.get_stats()
    assert stats["notes"] == 1
    assert stats["raw_fallback"] == 1

    report = server.review_graph(include_isolated=True)
    isolated_ids = [n["id"] for n in report["isolated_nodes"]]
    # note / raw 天生无边，不应被报成孤立
    assert "n2" not in isolated_ids and "n3" not in isolated_ids
    assert "n1" in isolated_ids
    # raw 要单独浮出来提醒处理
    assert report["raw_total"] == 1
    assert [x["id"] for x in report["raw_backlog"]] == ["n3"]


def test_scheduler_auto_skip_notifies_persistence():
    class FakeDBA:
        def __init__(self):
            self.raw = []

        def store_raw(self, text, timestamp=None):
            self.raw.append((timestamp, text))
            return ["n1"]

    dba = FakeDBA()
    scheduler = MaintenanceScheduler(
        dba=dba, config=ScheduleConfig(auto_skip_threshold=0),
    )
    called = []
    scheduler.on_maintenance_done = lambda result: called.append(result)
    scheduler._consecutive_skips = 0
    scheduler._do_flush_locked([(123, "hello")], TriggerReason.IDLE_TIMEOUT)
    assert dba.raw[0][0] == 123
    assert dba.raw[0][1].endswith("hello")
    assert called == [{"auto_skipped": True}]
