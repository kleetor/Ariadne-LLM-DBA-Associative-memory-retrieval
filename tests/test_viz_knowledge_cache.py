# SPDX-License-Identifier: AGPL-3.0-only

"""面板知识检索索引缓存：签名命中复用、签名变化重建、显式失效。"""
from dba_pipeline.viz.api_server import KnowledgeIndexCache


def _nodes(tag):
    return [{
        "id": "n1",
        "content": "接口超时统一 30s " + tag,
        "metadata": {"channel": "note", "title": "接口超时", "tags": [], "observed_at": 1},
    }]


def test_same_signature_reuses_index():
    cache = KnowledgeIndexCache()
    calls = []

    def loader():
        calls.append(1)
        return _nodes("v1")

    first = cache.get(("g.yaml", 100, 1), loader)
    second = cache.get(("g.yaml", 100, 1), loader)

    assert first is second
    assert len(calls) == 1          # 第二次没有再遍历图
    assert len(first) == 1


def test_changed_signature_rebuilds():
    cache = KnowledgeIndexCache()
    seq = iter([_nodes("v1"), _nodes("v2")])

    first = cache.get(("g.yaml", 100, 1), lambda: next(seq))
    second = cache.get(("g.yaml", 200, 1), lambda: next(seq))

    assert first is not second
    assert second.search("v2", top_k=5)      # 新内容进入了新索引


def test_invalidate_forces_rebuild():
    cache = KnowledgeIndexCache()
    calls = []

    def loader():
        calls.append(1)
        return _nodes("v1")

    cache.get(("g.yaml", 100, 1), loader)
    cache.invalidate()
    cache.get(("g.yaml", 100, 1), loader)

    assert len(calls) == 2
