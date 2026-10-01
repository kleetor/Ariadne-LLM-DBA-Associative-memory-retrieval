# SPDX-License-Identifier: AGPL-3.0-only

"""边连接拆批的单元测试（`MemoryDBA._link_edges`）。

拆批要同时守住两件事，本文件各测一半：

1. **异常批次被拆开**：新节点多于 `EDGE_BATCH_SIZE` 时按批调用，边不因单次输出撞上限而丢；
2. **正常批次一字不改**：`len(new_ids) <= EDGE_BATCH_SIZE`（含 0）时只调用一次，
   与改造前行为逐字一致——否则每次 Add 都会多花几倍输入 token 而不自知。

用最小替身而不是构造完整 `MemoryDBA`：后者要图、向量库、GraphBuilder 三件套，而
拆批逻辑只依赖 `_build_edge_context` 与 `_parse_response` 两处。
"""

from types import SimpleNamespace

from dba_pipeline.extraction.dba import EDGE_BATCH_SIZE, MemoryDBA


def _chain(payloads):
    """按调用次数依次返回预设响应（超出则重复最后一条）；返回 (chain, state)"""
    state = {"n": 0}

    class _Chain:
        def invoke(self, ctx):
            i = min(state["n"], len(payloads) - 1)
            state["n"] += 1
            return SimpleNamespace(content=payloads[i])

    return _Chain(), state


class _EdgeStub:
    """只为跑 `MemoryDBA._link_edges` 提供它依赖的两处"""

    _parse_response = MemoryDBA._parse_response  # 复用真实解析（含 JSON 兜底）

    def __init__(self):
        self.seen = []

    def _build_edge_context(self, conversation, new_ids):
        self.seen.append(list(new_ids))
        return {"new_nodes": list(new_ids)}

    def _link_edges(self, chain, conversation, new_ids):
        return MemoryDBA._link_edges(self, chain, conversation, new_ids)


def _ops_payload(n: int) -> str:
    ops = ', '.join('{"action": "create", "from": "a%d", "to": "b%d"}' % (i, i) for i in range(n))
    return '{"edge_ops": [%s]}' % ops


def test_small_batch_is_single_call():
    """正常批次：只调一次，行为与改造前一致"""
    stub = _EdgeStub()
    chain, state = _chain([_ops_payload(3)])

    out = stub._link_edges(chain, "conversation", ["n1", "n2", "n3"])

    assert state["n"] == 1
    assert stub.seen == [["n1", "n2", "n3"]]
    assert len(out) == 3


def test_empty_new_ids_still_calls_once():
    """新节点为空时仍照常调一次（改造前就是这么做的，不能顺手改成 0 次）"""
    stub = _EdgeStub()
    chain, state = _chain([_ops_payload(0)])

    out = stub._link_edges(chain, "conversation", [])

    assert state["n"] == 1
    assert stub.seen == [[]]
    assert out == []


def test_at_threshold_is_single_call():
    stub = _EdgeStub()
    chain, state = _chain([_ops_payload(1)])
    ids = ["n%d" % i for i in range(EDGE_BATCH_SIZE)]

    stub._link_edges(chain, "conversation", ids)

    assert state["n"] == 1
    assert stub.seen == [ids]


def test_large_batch_is_split_and_merged():
    """超过阈值 → 拆成 ceil(n/EDGE_BATCH_SIZE) 批，各批结果合并、无遗漏"""
    stub = _EdgeStub()
    n = EDGE_BATCH_SIZE * 2 + 3
    ids = ["n%d" % i for i in range(n)]

    chain, state = _chain([_ops_payload(2), _ops_payload(4), _ops_payload(1)])
    out = stub._link_edges(chain, "conversation", ids)

    assert state["n"] == 3
    # 每批的 id 段：前 EDGE_BATCH_SIZE / 次 EDGE_BATCH_SIZE / 余下的 3
    assert [len(c) for c in stub.seen] == [EDGE_BATCH_SIZE, EDGE_BATCH_SIZE, 3]
    # 三批的 id 拼起来必须等于原序列——分批不能丢节点、也不能重复
    assert [i for c in stub.seen for i in c] == ids
    assert len(out) == 2 + 4 + 1


def test_split_every_batch_is_actually_invoked():
    """批量数必须等于调用次数（防"算了批数却没循环"这类实现错误）"""
    stub = _EdgeStub()
    n = EDGE_BATCH_SIZE * 5 + 1
    ids = ["n%d" % i for i in range(n)]
    chain, state = _chain([_ops_payload(0)])

    stub._link_edges(chain, "conversation", ids)

    assert state["n"] == 6
