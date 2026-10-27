# SPDX-License-Identifier: AGPL-3.0-only

"""关键词倒排索引单元测试：分词、字段权重、过滤与排序。"""
from dba_pipeline.kwindex import KeywordIndex, tokenize


def _note(nid, title, text, tags=None, ts=None, deprecated=False, forgotten=False):
    return {
        "id": nid,
        "content": text,
        "metadata": {
            "channel": "note",
            "title": title,
            "tags": tags or [],
            "observed_at": ts,
        },
        "timestamp": ts,
        "deprecated": deprecated,
        "forgotten": forgotten,
    }


def test_tokenize_ascii_and_cjk_bigrams():
    tokens = tokenize("API 接口超时 30s")
    assert "30s" in tokens
    assert {"接口", "口超", "超时"} <= tokens
    assert tokenize("API").issuperset({"api"})


def test_tokenize_stopwords_dropped():
    assert tokenize("我们") == set()


def test_title_beats_tags_beats_text():
    idx = KeywordIndex()
    idx.build([
        _note("n1", "接口超时约定", "无关正文", ["其它"]),
        _note("n2", "无关标题", "无关正文", ["接口超时"]),
        _note("n3", "无关标题", "接口超时统一 30s", []),
    ])
    out = idx.search("接口超时", top_k=3)
    assert [h["node_id"] for h in out] == ["n1", "n2", "n3"]
    assert out[0]["score"] == 1.0
    assert 0.6 < out[1]["score"] < 1.0
    assert 0.3 < out[2]["score"] < 0.6


def test_combined_fields_capped_at_one():
    idx = KeywordIndex()
    idx.build([_note("n1", "接口超时", "接口超时统一 30s", ["接口超时"])])
    assert idx.search("接口超时")[0]["score"] == 1.0


def test_deprecated_and_forgotten_skipped():
    idx = KeywordIndex()
    idx.build([
        _note("n1", "接口超时", "x", deprecated=True),
        _note("n2", "接口超时", "x", forgotten=True),
        _note("n3", "接口超时", "x"),
    ])
    assert [h["node_id"] for h in idx.search("接口超时")] == ["n3"]


def test_tie_break_by_observed_at_desc():
    idx = KeywordIndex()
    idx.build([
        _note("n1", "接口超时", "x", ts=1000),
        _note("n2", "接口超时", "x", ts=2000),
    ])
    assert [h["node_id"] for h in idx.search("接口超时")] == ["n2", "n1"]


def test_empty_and_no_match():
    idx = KeywordIndex()
    idx.build([_note("n1", "接口超时", "x")])
    assert idx.search("") == []
    assert idx.search("完全无关的词汇") == []
