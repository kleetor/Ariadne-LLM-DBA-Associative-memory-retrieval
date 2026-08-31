# SPDX-License-Identifier: AGPL-3.0-only

"""YAML 加载器单元测试（含 P0-1 修复验证）"""
from dba_pipeline.loader import load_graph


def test_load_graph_preserves_deprecated_and_forgotten(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "nodes:\n"
        "  - id: n1\n"
        "    content: 旧记忆\n"
        "    type: status\n"
        "    deprecated: true\n"
        "    forgotten: true\n"
        "edges: []\n",
        encoding="utf-8",
    )
    g = load_graph(str(p))
    node = g.get_node("n1")
    assert node["deprecated"] is True
    assert node["forgotten"] is True


def test_load_graph_defaults_flags_false(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "nodes:\n"
        "  - id: n1\n"
        "    content: 普通记忆\n"
        "    type: action\n"
        "edges: []\n",
        encoding="utf-8",
    )
    g = load_graph(str(p))
    node = g.get_node("n1")
    assert node["deprecated"] is False
    assert node["forgotten"] is False
