# SPDX-License-Identifier: AGPL-3.0-only

"""MCP 笔记与知识层测试：写入、检索、持久化、rerank 降级与巡检口径。"""
import threading

from dba_pipeline.extraction.maintenance_scheduler import (
    MaintenanceScheduler, ScheduleConfig, TriggerReason,
)
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph
from dba_pipeline.mcp_server import (
    DBAServer, MAX_NOTE_LENGTH, NOTE_TEXT_LIMIT, TOOL_SCHEMAS,
)


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


def test_tool_schemas_include_note_tools():
    names = [t["name"] for t in TOOL_SCHEMAS]
    assert "dba_add_note" in names
    assert "dba_query_knowledge" in names
    assert len(names) == 11


def test_add_note_validates_input(tmp_path):
    server = _server(tmp_path)
    assert "error" in server.add_note("", "正文")
    assert "error" in server.add_note("标题", "")
    assert "error" in server.add_note("标题", "x" * (MAX_NOTE_LENGTH + 1))
    assert "error" in server.add_note("标题", "正文", tags=[1])
    assert "error" in server.add_note("标题", "正文", observed_at="昨天")
    assert "error" in server.add_note("标题", "正文", fidelity="unknown")
    assert server.graph.node_count == 0


def test_add_note_writes_node_metadata_and_vector(tmp_path):
    vs = FakeVectorStore()
    server = _server(tmp_path, vector_store=vs)
    out = server.add_note(
        "接口超时约定", "接口超时统一为 30s，重试 2 次",
        observed_at=1761300000000, tags=["project:alpha"], fidelity="summary",
    )
    assert out["note_id"] == "n1"
    assert out["time_inferred"] is False
    nd = server.graph.graph.nodes["n1"]
    assert nd["content"].startswith("接口超时")
    assert nd["metadata"]["channel"] == "note"
    assert nd["metadata"]["title"] == "接口超时约定"
    assert nd["metadata"]["tags"] == ["project:alpha"]
    assert nd["metadata"]["fidelity"] == "summary"
    assert nd["timestamp"] is not None
    assert "n1" in vs.added


def test_add_note_without_observed_at_marks_inferred(tmp_path):
    server = _server(tmp_path)
    out = server.add_note("标题", "正文")
    assert out["time_inferred"] is True
    meta = server.graph.graph.nodes[out["note_id"]]["metadata"]
    assert meta["time_inferred"] is True
    assert meta["observed_at"] == meta["created_at"]


def test_add_note_persists_and_reloads(tmp_path):
    server = _server(tmp_path)
    note_id = server.add_note("接口超时约定", "接口超时统一为 30s")["note_id"]

    reloaded = load_graph(str(tmp_path / "g.yaml"))
    nd = reloaded.graph.nodes[note_id]
    assert nd["metadata"]["channel"] == "note"
    assert nd["metadata"]["title"] == "接口超时约定"
    assert nd["content"] == "接口超时统一为 30s"

    fresh = DBAServer(reloaded, yaml_path=str(tmp_path / "g.yaml"),
                      lock=threading.RLock())
    out = fresh.query_knowledge("接口超时")
    assert [n["node_id"] for n in out["notes"]] == [note_id]


def test_add_note_exact_duplicate_reuses_node(tmp_path):
    server = _server(tmp_path)
    first = server.add_note("标题", "正文", tags=["a"])
    second = server.add_note("标题", "正文", tags=["a"])
    assert second.get("duplicate") is True
    assert second["note_id"] == first["note_id"]
    assert server.graph.node_count == 1


def test_query_knowledge_returns_text_and_score(tmp_path):
    server = _server(tmp_path)
    server.add_note("接口超时约定", "接口超时统一为 30s，重试 2 次",
                    tags=["project:alpha"], source_hint="docs/api.md §3")
    out = server.query_knowledge("接口超时")
    assert out["returned"] == 1
    item = out["notes"][0]
    assert item["title"] == "接口超时约定"
    assert item["source_hint"] == "docs/api.md §3"
    assert item["score"] == 1.0
    assert "edge" not in item and "edges" not in item


def test_query_knowledge_truncates_long_text(tmp_path):
    server = _server(tmp_path)
    text = "截断测试" + "a" * 1400
    server.add_note("长笔记", text)
    item = server.query_knowledge("截断测试")["notes"][0]
    assert item["truncated"] is True
    assert len(item["text"]) == NOTE_TEXT_LIMIT


def test_query_knowledge_no_match_and_empty_query(tmp_path):
    server = _server(tmp_path)
    server.add_note("接口超时约定", "接口超时统一为 30s")
    assert server.query_knowledge("完全无关") == {
        "notes": [], "returned": 0, "query": "完全无关", "method": "keyword",
    }
    assert "error" in server.query_knowledge("  ")


def test_query_knowledge_rerank_order_and_fallback(tmp_path):
    server = _server(tmp_path, reranker=FakeReranker(order=[1, 0]))
    server.add_note("接口超时 A", "接口超时方案 A", observed_at=1000)
    server.add_note("接口超时 B", "接口超时方案 B", observed_at=2000)
    out = server.query_knowledge("接口超时")
    assert out["method"] == "keyword+rerank"
    assert out["notes"][0]["title"] == "接口超时 A"

    (tmp_path / "sub").mkdir()
    failing = _server(tmp_path / "sub", reranker=FakeReranker(fail=True))
    failing.add_note("接口超时", "接口超时方案")
    out = failing.query_knowledge("接口超时")
    assert out["method"] == "keyword"
    assert out["returned"] == 1


def test_query_knowledge_works_without_vector_store(tmp_path):
    server = _server(tmp_path, vector_store=None)
    server.add_note("接口超时约定", "接口超时统一为 30s")
    out = server.query_knowledge("接口超时")
    assert out["returned"] == 1


def test_query_knowledge_single_term_is_enough(tmp_path):
    """最低命中口径：至少 1 个 term 即返回，不设覆盖率阈值。"""
    server = _server(tmp_path)
    server.add_note("接口超时约定", "接口超时统一为 30s")
    out = server.query_knowledge("接口 完全无关")
    assert out["returned"] == 1
    assert 0 < out["notes"][0]["score"] < 0.5


def test_stats_and_review_skip_notes(tmp_path):
    server = _server(tmp_path)
    server.add_note("接口超时约定", "接口超时统一为 30s")
    assert server.get_stats()["notes"] == 1
    report = server.review_graph(include_isolated=True)
    assert report["isolated_nodes"] == []
    assert report["duplicate_candidates"] == []


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
