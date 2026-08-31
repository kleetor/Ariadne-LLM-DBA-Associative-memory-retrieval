# SPDX-License-Identifier: AGPL-3.0-only

"""StoryRank 单元测试：轨迹记录、连通性筛选、故事化"""
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType, RelationType
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever
from dba_pipeline.llm.inference import InferenceEngine


def _graph():
    g = MemoryGraph()
    g.add_memory("n1", "用户压力很大", NodeType.STATUS)
    g.add_memory("n2", "经常加班到10点", NodeType.REASON)
    g.add_memory("n3", "喝咖啡提神", NodeType.ACTION)
    g.add_edge("n1", "n2", RelationType.CAUSAL)
    g.add_edge("n2", "n3", RelationType.CAUSAL)
    return g


def _retriever(graph):
    r = PurposeDrivenRetriever.__new__(PurposeDrivenRetriever)
    r.graph = graph
    return r


def test_expand_with_trace_records_edge_source():
    graph = _graph()
    trace = graph.expand_with_trace(["n1"])
    assert "n2" in trace
    assert trace["n2"]["from"] == "n1"
    assert trace["n2"]["rel_type"] == RelationType.CAUSAL
    assert trace["n2"]["is_reverse"] is False
    assert trace["n2"]["weight"] > 0


def test_select_core_nodes_splits_components():
    graph = _graph()
    graph.add_memory("n6", "独立的旅行计划", NodeType.THING)
    r = _retriever(graph)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
            {"id": "n6", "content": "独立的旅行计划", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
        ]},
    ]

    groups = r.select_core_nodes(hop_history, ["n2", "n6"])
    # n2 回溯到 n1（一组），n6 是独立种子（另一组）
    assert len(groups) == 2
    flat = {nid for grp in groups for nid in grp}
    assert flat == {"n1", "n2", "n6"}


def test_select_core_nodes_drops_isolated_branch():
    graph = _graph()
    graph.add_memory("n7", "失眠", NodeType.STATUS)
    graph.add_edge("n1", "n7", RelationType.SEQUENCE)
    r = _retriever(graph)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
            {"id": "n7", "content": "失眠", "combined_score": 0.3,
             "from": "n1", "rel_type": RelationType.SEQUENCE, "is_reverse": False},
        ]},
    ]

    # 只有 n2 是结果，n7 不在结果回溯路径上，应被剔除
    groups = r.select_core_nodes(hop_history, ["n2"])
    assert len(groups) == 1
    assert set(groups[0]) == {"n1", "n2"}


def test_build_path_constructs_nodes_and_edges():
    graph = _graph()
    r = _retriever(graph)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
        ]},
    ]

    path = r._build_path(["n1", "n2"], hop_history)
    assert len(path["nodes"]) == 2
    assert path["nodes"][0]["node_type"] == "status"
    assert len(path["edges"]) == 1
    assert path["edges"][0] == {"from": "n1", "to": "n2",
                                "rel_type": "causal", "is_reverse": False}


def test_story_rank_parses_story_and_adopted_ids():
    llm = FakeListChatModel(responses=[
        '{"story": "用户压力大，因为加班，常喝咖啡提神", "adopted_ids": ["n1", "n2", "n3"]}'
    ])
    engine = InferenceEngine(llm)
    path = {
        "nodes": [
            {"id": "n1", "content": "用户压力很大", "node_type": "status"},
            {"id": "n2", "content": "经常加班到10点", "node_type": "reason"},
            {"id": "n3", "content": "喝咖啡提神", "node_type": "action"},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "rel_type": "causal", "is_reverse": False},
            {"from": "n2", "to": "n3", "rel_type": "causal", "is_reverse": False},
        ],
    }
    out = engine.story_rank("最近怎么样", path)
    assert out["story"] == "用户压力大，因为加班，常喝咖啡提神"
    assert out["adopted_ids"] == ["n1", "n2", "n3"]


def test_select_core_nodes_no_infinite_loop_on_revisit():
    """双向边折返（回访）时 parent 不应成环，回溯不死循环"""
    graph = _graph()
    r = _retriever(graph)

    # 模拟双向边折返：n1 → n2 → n1
    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
        ]},
        {"hop": 2, "candidates": [
            {"id": "n1", "content": "用户压力很大", "combined_score": 0.4,
             "from": "n2", "rel_type": RelationType.SCENARIO, "is_reverse": True},
        ]},
    ]

    groups = r.select_core_nodes(hop_history, ["n1", "n2"])
    assert len(groups) == 1
    assert set(groups[0]) == {"n1", "n2"}
