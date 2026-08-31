# SPDX-License-Identifier: AGPL-3.0-only

"""跳转轴权重矩阵单元测试"""
from dba_pipeline.core.jump_axis import (
    NodeType,
    RelationType,
    JUMP_AXIS_RULES,
    get_jump_weight,
    expand_weights,
)


def test_enum_counts():
    assert len(NodeType) == 6
    assert len(RelationType) == 8


def test_weight_matrix_covers_all_types():
    for nt in NodeType:
        assert nt in JUMP_AXIS_RULES
        for rt in RelationType:
            assert rt in JUMP_AXIS_RULES[nt]


def test_taxonomic_blocked_for_status():
    # STATUS 节点不沿 TAXONOMIC 扩展（权重为 0，方向阻断）
    assert get_jump_weight(NodeType.STATUS, RelationType.TAXONOMIC) == 0.0
    assert get_jump_weight(NodeType.STATUS, RelationType.TAXONOMIC, is_reverse=True) == 0.0


def test_reverse_weight_differs():
    # ACTION 的 CAUSAL：正向 0.9，反向 0.7
    assert get_jump_weight(NodeType.ACTION, RelationType.CAUSAL) == 0.9
    assert get_jump_weight(NodeType.ACTION, RelationType.CAUSAL, is_reverse=True) == 0.7


def test_expand_weights_filters_zero():
    weights = expand_weights(
        NodeType.STATUS,
        {
            "e1": (RelationType.CAUSAL, False),
            "e2": (RelationType.TAXONOMIC, False),  # 权重 0，应被过滤
        },
    )
    assert "e1" in weights
    assert "e2" not in weights
