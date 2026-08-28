"""检索过滤单元测试：deprecated/forgotten 节点不应进入检索结果"""
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever


def _graph():
    g = MemoryGraph()
    g.add_memory("n1", "正常记忆A", NodeType.STATUS)
    g.add_memory("n2", "废弃记忆", NodeType.STATUS, deprecated=True)
    g.add_memory("n3", "正常记忆B", NodeType.ACTION)
    g.add_memory("n4", "遗忘记忆", NodeType.REASON, forgotten=True)
    return g


def _retriever(graph):
    r = PurposeDrivenRetriever.__new__(PurposeDrivenRetriever)
    r.graph = graph
    return r


def test_is_active_skips_deprecated_forgotten_and_missing():
    graph = _graph()
    r = _retriever(graph)
    assert r._is_active("n1") is True
    assert r._is_active("n2") is False  # deprecated
    assert r._is_active("n4") is False  # forgotten
    assert r._is_active("nX") is False  # 不存在


def test_build_result_filters_deprecated_and_forgotten():
    graph = _graph()
    r = _retriever(graph)
    memories = r._build_result(["n1", "n2", "n3", "n4"], {}, [])
    ids = [mid for mid, _ in memories["peak_memories"]]
    assert ids == ["n1", "n3"]
