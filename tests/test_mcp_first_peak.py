# -*- coding: utf-8 -*-
"""首峰即停（ARIADNE_MCP_FIRST_PEAK）的配置与行为测试。"""

import dba_pipeline.mcp_server as mcp_server
from dba_pipeline.core.peak_find import PeakFinder
from dba_pipeline.graph.memory_graph import MemoryGraph


class _StubRetriever:
    def __init__(self):
        self.peak_finder = PeakFinder()


def test_first_peak_off_by_default(monkeypatch):
    monkeypatch.delenv("ARIADNE_MCP_FIRST_PEAK", raising=False)
    retriever = _StubRetriever()
    mcp_server.DBAServer(MemoryGraph(), retriever=retriever)
    assert retriever.peak_finder.first_peak is False


def test_first_peak_env_enables(monkeypatch):
    monkeypatch.setenv("ARIADNE_MCP_FIRST_PEAK", "1")
    retriever = _StubRetriever()
    mcp_server.DBAServer(MemoryGraph(), retriever=retriever)
    assert retriever.peak_finder.first_peak is True


def test_peak_finder_stops_on_first_decline():
    finder = PeakFinder(first_peak=True, min_delta=0.015)
    assert finder.add_round(0.50) == "continue"
    assert finder.add_round(0.40) == "peak_found"


def test_peak_finder_patience_allows_rebound():
    finder = PeakFinder(first_peak=False, min_delta=0.015, patience=2)
    assert finder.add_round(0.50) == "continue"
    assert finder.add_round(0.40) == "continue"   # 首次显著下降，开始计时
    assert finder.add_round(0.45) == "continue"   # 显著回升，计时清零
    assert finder.add_round(0.43) == "continue"   # 再次下降，开始计时
    assert finder.add_round(0.42) == "continue"   # 额外探索第 1 轮
    assert finder.add_round(0.41) == "peak_found"  # 额外探索第 2 轮仍未回升
