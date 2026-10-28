# SPDX-License-Identifier: AGPL-3.0-only

"""审查中确认的关键缺陷的回归测试：

1. ALM 文档日志回放：单行损坏不能连坐它之后的完整行（旧实现会截断删除）。
2. 维护调度器：写临界区**进入失败**时 `_flushing` 必须复位，否则维护永久停摆。
3. MCP checkpoint：`save_dir` 不允许等于图谱目录本身（会覆盖活图谱）。
4. apply_ops 回滚后 `created_ids` 必须清空；去重需单独计数以区分「去重」与「被拒」。
 5. 空间目录名：Windows 保留设备名转义；大小写不敏感文件系统上拒绝仅大小写不同的 id。
 6. Basic 凭据失败限速：超过阈值后直接 429（不再逐次做 PBKDF2 校验）。
"""
import asyncio
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from alm.engine import ALMEngine, space_dirname
from alm.space import MemorySpace
from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.extraction.dba import MemoryDBA
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.extraction.maintenance_scheduler import (
    MaintenanceScheduler,
    TriggerReason,
)
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.mcp_server import DBAServer
from dba_pipeline.webauth import AuthMiddleware, LoginRateLimiter


# ---------- 1. docs.log 回放 ----------

def test_replay_docs_log_skips_corrupt_line_but_keeps_later_lines(tmp_path):
    log = tmp_path / "docs.log"
    rec1 = json.dumps({"seq": 1, "msgs": [
        {"doc_id": "d1", "msg_index": 0, "content": "第一条"}]})
    rec2 = json.dumps({"seq": 2, "msgs": [
        {"doc_id": "d2", "msg_index": 0, "content": "第三条"}]})
    # 中间插一条损坏行（完整换行，只是 JSON 坏了）
    log.write_bytes((rec1 + "\n" + "{broken json\n" + rec2 + "\n").encode("utf-8"))

    stub = SimpleNamespace(
        doc_msgs=[],
        _doc_seq=0,
        _docs_log_path=lambda: log,
        _masked_id=lambda: "test",
    )
    MemorySpace._replay_docs_log(stub)

    # 坏行之后的完整行仍然被回放（旧实现会因 break 全部丢失）
    assert [d["doc_id"] for d in stub.doc_msgs] == ["d1", "d2"]
    # 且磁盘上不能删掉任何完整行
    assert log.read_bytes().count(b"\n") == 3


def test_replay_docs_log_truncates_only_partial_tail(tmp_path):
    log = tmp_path / "docs.log"
    rec = json.dumps({"seq": 1, "msgs": [
        {"doc_id": "d1", "msg_index": 0, "content": "完整行"}]})
    log.write_bytes((rec + "\n" + '{"seq": 2, "msgs": [{"doc_id": "d2"').encode("utf-8"))

    stub = SimpleNamespace(
        doc_msgs=[], _doc_seq=0,
        _docs_log_path=lambda: log, _masked_id=lambda: "test",
    )
    MemorySpace._replay_docs_log(stub)

    assert [d["doc_id"] for d in stub.doc_msgs] == ["d1"]
    # 末尾半行被裁掉，只保留第一条完整行
    assert log.read_bytes().endswith(b"\n")
    assert b"d2" not in log.read_bytes()


# ---------- 2. 维护调度器兜底复位 ----------

def test_flush_resets_flag_when_guard_entry_fails():
    @contextmanager
    def failing_guard():
        raise TimeoutError("无法获取跨进程写锁")
        yield  # pragma: no cover

    sched = MaintenanceScheduler(
        dba=SimpleNamespace(),
        maintain_context=lambda: failing_guard(),
    )
    sched._flushing = True

    sched._do_flush([], TriggerReason.MANUAL)

    assert sched._flushing is False, "临界区进入失败后必须复位，否则维护永久停摆"


# ---------- 3. checkpoint 目录约束 ----------

def test_checkpoint_rejects_graph_dir_itself(tmp_path):
    graph_dir = tmp_path / "data"
    graph_dir.mkdir()
    stub = SimpleNamespace(yaml_path=str(graph_dir / "memory_graph.yaml"))

    assert DBAServer._is_safe_save_dir(stub, "snapshots/latest") is True
    assert DBAServer._is_safe_save_dir(stub, str(graph_dir / "snapshots" / "s1")) is True
    # 图谱目录本身会让 save_checkpoint 原地覆盖活图谱
    assert DBAServer._is_safe_save_dir(stub, ".") is False
    assert DBAServer._is_safe_save_dir(stub, str(graph_dir)) is False
    # 仍然拒绝越界
    assert DBAServer._is_safe_save_dir(stub, "../outside") is False
    assert DBAServer._is_safe_save_dir(stub, str(tmp_path / "outside")) is False


# ---------- 4. apply_ops 回滚后不得漏出悬空 created_ids ----------

class _FailSecondAddVectorStore:
    """无向量能力（走精确去重），但第 2 次 add_memories 抛错，模拟批中途致命失败。"""

    store = None

    def __init__(self):
        self.calls = 0

    def add_memories(self, memory_ids, contents):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")

    def remove_memories(self, memory_ids):
        pass

    def update_memories(self, memory_ids, contents):
        pass


def test_apply_ops_deduped_counted():
    graph = MemoryGraph()
    graph.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    builder = GraphBuilder(graph, _FailSecondAddVectorStore())

    result = builder.apply_ops(
        [{"action": "create", "content": "用户喜欢喝咖啡", "node_type": "STATUS"}], []
    )

    assert result["created_ids"] == []
    assert result["deduped"] == 1, "命中去重必须计数，否则调用方无法区分去重与被拒"
    assert result["errors"] == []


def test_apply_ops_rollback_clears_created_ids():
    graph = MemoryGraph()
    graph.add_memory("n1", "用户喜欢喝咖啡", NodeType.STATUS)
    builder = GraphBuilder(graph, _FailSecondAddVectorStore())

    result = builder.apply_ops(
        [
            {"action": "create", "content": "用户养了一只猫", "node_type": "STATUS"},
            {"action": "create", "content": "用户在学吉他", "node_type": "STATUS"},
        ],
        [],
        check_orphans=False,
    )

    assert result["errors"], "第二批 create 应对应一条致命错误"
    # 回滚后节点已从 graph 删除，created_ids 不能再把它们交出去（会写出悬空 ID）
    assert result["created_ids"] == []
    assert graph.node_count == 1


def test_needs_raw_fallback_distinguishes_rejected_from_deduped():
    result_rejected = {
        "ops": {"node_ops": [{"content": "x", "node_type": "STATUS"}]},
        "result": {"created_ids": [], "deduped": 0, "errors": [{"error": "boom"}]},
    }
    assert MemoryDBA.needs_raw_fallback(None, result_rejected) == "create_rejected"

    result_deduped = {
        "ops": {"node_ops": [{"content": "x", "node_type": "STATUS"}]},
        "result": {"created_ids": [], "deduped": 1},
    }
    assert MemoryDBA.needs_raw_fallback(None, result_deduped) is None


# ---------- 5. 空间目录名：Windows 保留名 & 大小写冲突 ----------

def test_space_dirname_escapes_windows_reserved_names():
    # CON/NUL/COM1 作为目录名会被 Windows 当成设备，首字节必须转义
    assert space_dirname("con") == "%63on"
    assert space_dirname("NUL") == "%4EUL"
    assert space_dirname("com1") == "%63om1"
    assert space_dirname("LPT9") == "%4CPT9"
    # 普通名字不受影响
    assert space_dirname("eval:run_abc:locomo:conv-0") == (
        "eval%3Arun_abc%3Alocomo%3Aconv-0"
    )


def _engine_stub(case_insensitive: bool) -> ALMEngine:
    engine = object.__new__(ALMEngine)
    engine._case_insensitive = case_insensitive
    engine._dirname_owners = {}
    engine._registry_lock = threading.RLock()
    return engine


def test_claim_dirname_rejects_case_only_collision():
    engine = _engine_stub(case_insensitive=True)
    engine._claim_dirname("Alice")
    engine._claim_dirname("Alice")  # 同一 user_id 反复访问正常
    with pytest.raises(ValueError):
        engine._claim_dirname("alice")


def test_claim_dirname_allows_case_variants_on_sensitive_fs():
    engine = _engine_stub(case_insensitive=False)
    engine._claim_dirname("Alice")
    engine._claim_dirname("alice")  # Linux 上是两个不同目录，应放行


# ---------- 6. Basic 凭据失败限速 ----------

class _RejectingStore:
    user = "admin"
    must_change = False

    def refresh(self):
        pass

    def verify(self, user, password):
        return False


def _run_asgi(middleware, scope):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    asyncio.run(middleware(scope, receive, send))
    return messages


def _basic_scope(ip="10.0.0.9"):
    return {
        "type": "http",
        "path": "/api/state",
        "method": "GET",
        "headers": [(b"authorization", b"Basic YWRtaW46d3Jvbmc=")],
        "client": (ip, 5000),
    }


def test_auth_middleware_rate_limits_failed_basic():
    async def app(scope, receive, send):
        raise AssertionError("凭据错误时不该进入应用")

    mw = AuthMiddleware(
        app, store=_RejectingStore(),
        rate_limiter=LoginRateLimiter(max_failures=2, window=300),
    )

    assert _run_asgi(mw, _basic_scope())[0]["status"] == 401
    assert _run_asgi(mw, _basic_scope())[0]["status"] == 401
    # 超过阈值后直接 429：不再做 PBKDF2 校验，也不再给爆破机会
    assert _run_asgi(mw, _basic_scope())[0]["status"] == 429
