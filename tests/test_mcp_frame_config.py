# -*- coding: utf-8 -*-
"""MCP 目的帧开关的配置层测试。

只验证「环境变量 → 检索参数」这一层：不调用 LLM / embedding / 向量库，
用桩替换 retriever，检查 `_story_kwargs` 实际外发的参数。
"""

import pytest

import dba_pipeline.mcp_server as mcp_server
from dba_pipeline.core.purpose_frame import PurposeFrame
from dba_pipeline.graph.memory_graph import MemoryGraph

ENV_KEYS = (
    "ARIADNE_MCP_PURPOSE_FRAME",
    "ARIADNE_MCP_FRAME_WEIGHTS",
    "ARIADNE_MCP_FRAME_ANCHOR",
)


class _FakeInference:
    def infer_purpose(self, query):
        return {"purposes": ["最近很累想倾诉", "理解原因"]}


class _FakeRetriever:
    def __init__(self):
        self.inference = _FakeInference()


def _make_server(monkeypatch, **env):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    server = mcp_server.DBAServer(MemoryGraph())
    server.retriever = _FakeRetriever()
    return server


def test_frame_disabled_keeps_default_path(monkeypatch):
    server = _make_server(monkeypatch)
    kwargs = server._story_kwargs("今天有点累")

    assert "purpose" not in kwargs
    assert "purpose_frame" not in kwargs
    assert kwargs["max_peak_nodes"] is None
    assert kwargs["seed_cos_min"] is None
    assert kwargs["seed_k"] == 5
    assert kwargs["max_hops"] == 5


def test_frame_weights_and_anchor_forwarded(monkeypatch):
    server = _make_server(
        monkeypatch,
        ARIADNE_MCP_PURPOSE_FRAME="1",
        ARIADNE_MCP_FRAME_WEIGHTS="0.6,0.3,0.1",
        ARIADNE_MCP_FRAME_ANCHOR="0.4",
    )
    kwargs = server._story_kwargs("今天有点累")

    assert kwargs["purpose"] == ["最近很累想倾诉", "理解原因"]
    assert isinstance(kwargs["purpose_frame"], PurposeFrame)
    # 「倾诉」命中 vent 预设，主类型应为 emotion
    assert kwargs["purpose_frame"].dominant_type == "emotion"
    assert kwargs["frame_weights"] == (0.6, 0.3, 0.1)
    assert kwargs["frame_anchor"] == 0.4


def test_invalid_env_falls_back_to_defaults(monkeypatch):
    server = _make_server(
        monkeypatch,
        ARIADNE_MCP_PURPOSE_FRAME="1",
        ARIADNE_MCP_FRAME_WEIGHTS="1,2",
        ARIADNE_MCP_FRAME_ANCHOR="9",
    )
    kwargs = server._story_kwargs("今天有点累")

    assert kwargs["frame_weights"] == mcp_server.DEFAULT_FRAME_WEIGHTS
    assert "frame_anchor" not in kwargs  # 0 = 关闭，不透传


def test_anchor_zero_does_not_forward(monkeypatch):
    server = _make_server(
        monkeypatch,
        ARIADNE_MCP_PURPOSE_FRAME="1",
        ARIADNE_MCP_FRAME_ANCHOR="0",
    )
    kwargs = server._story_kwargs("今天有点累")

    assert "purpose_frame" in kwargs
    assert "frame_anchor" not in kwargs


@pytest.mark.parametrize("raw,expected", [
    (None, 0.0),
    ("0", 0.0),
    ("0.4", 0.4),
    ("1", 1.0),
    ("1.5", 0.0),
    ("-0.1", 0.0),
    ("nan", 0.0),
    ("abc", 0.0),
])
def test_parse_frame_anchor(raw, expected):
    assert mcp_server._parse_frame_anchor(raw) == expected
