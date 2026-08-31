# SPDX-License-Identifier: AGPL-3.0-only

"""记忆图谱序列化往返单元测试"""
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType, RelationType


def test_roundtrip_preserves_flags():
    g = MemoryGraph()
    g.add_memory("n1", "旧记忆", NodeType.STATUS, deprecated=True, forgotten=True)
    g2 = MemoryGraph.from_dict(g.to_dict())
    node = g2.get_node("n1")
    assert node["deprecated"] is True
    assert node["forgotten"] is True


def test_roundtrip_preserves_edges():
    g = MemoryGraph()
    g.add_memory("n1", "A", NodeType.STATUS)
    g.add_memory("n2", "B", NodeType.REASON)
    g.add_edge("n1", "n2", RelationType.CAUSAL)

    g2 = MemoryGraph.from_dict(g.to_dict())
    assert g2.graph.has_edge("n1", "n2")
    assert g2.get_node_type("n2") == NodeType.REASON
