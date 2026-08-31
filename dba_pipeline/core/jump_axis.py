# SPDX-License-Identifier: AGPL-3.0-only

"""
跳转轴模型：有向边 + 类型权重 + 节点类型规则

论文第三章：将知识图谱边类型化为 8 种关系类型，
不同节点类型在不同方向上有不同的扩展权重。
"""

from enum import Enum
from typing import Dict, Tuple


class RelationType(Enum):
    """8 种基本关系类型"""
    CAUSAL = "causal"           # 因果关系，指向结果
    SCENARIO = "scenario"       # 场景/环境归属，双向
    SEQUENCE = "sequence"       # 时序先后，指向后序
    PREFERENCE = "preference"   # 态度/偏好，指向对象
    SOCIAL = "social"           # 社交关系，双向
    ATTRIBUTE = "attribute"     # 属性归属，双向
    TEMPORAL = "temporal"       # 时间定位，指向时间
    TAXONOMIC = "taxonomic"     # 分类/本体，指向父类


class NodeType(Enum):
    """节点的语义角色类型"""
    STATUS = "status"
    REASON = "reason"
    ACTION = "action"
    THING = "thing"
    PERSON = "person"
    EMOTION = "emotion"


# 跳转轴规则矩阵：
# 从某个 NodeType 出发，沿各 RelationType 扩展时，正向/反向的权重
# 权重 0 表示该方向不扩展

JumpAxisWeights = Dict[NodeType, Dict[RelationType, Tuple[float, float]]]
# 格式: {node_type: {relation: (forward_weight, reverse_weight)}}

JUMP_AXIS_RULES: JumpAxisWeights = {
    NodeType.STATUS: {
        RelationType.CAUSAL:     (0.9, 1.0),   # 导致了什么 / 是什么导致的
        RelationType.SCENARIO:   (0.8, 0.8),   # 发生在什么情境
        RelationType.SEQUENCE:   (0.7, 0.0),   # 之后做了什么（不回看）
        RelationType.PREFERENCE: (0.4, 0.0),   # 状态下偏好什么
        RelationType.SOCIAL:     (0.5, 0.5),
        RelationType.ATTRIBUTE:  (0.3, 0.3),
        RelationType.TEMPORAL:   (0.3, 0.0),
        RelationType.TAXONOMIC:  (0.0, 0.0),   # 不往分类链走
    },
    NodeType.REASON: {
        RelationType.CAUSAL:     (1.0, 0.9),   # 原因导致的结果 / 原因的更深层原因
        RelationType.SCENARIO:   (0.6, 0.6),
        RelationType.SEQUENCE:   (0.8, 0.0),
        RelationType.PREFERENCE: (0.2, 0.0),
        RelationType.SOCIAL:     (0.4, 0.4),
        RelationType.ATTRIBUTE:  (0.3, 0.3),
        RelationType.TEMPORAL:   (0.3, 0.0),
        RelationType.TAXONOMIC:  (0.0, 0.0),
    },
    NodeType.ACTION: {
        RelationType.CAUSAL:     (0.9, 0.7),   # 行为导致的结果 / 什么导致该行为
        RelationType.SCENARIO:   (0.8, 0.8),   # 在什么场景下做
        RelationType.SEQUENCE:   (0.9, 0.6),   # 之后做什么 / 之前做什么
        RelationType.PREFERENCE: (0.5, 0.0),
        RelationType.SOCIAL:     (0.6, 0.6),
        RelationType.ATTRIBUTE:  (0.4, 0.4),
        RelationType.TEMPORAL:   (0.5, 0.0),
        RelationType.TAXONOMIC:  (0.0, 0.3),   # 允许向下看子类
    },
    NodeType.THING: {
        RelationType.CAUSAL:     (0.3, 0.3),
        RelationType.SCENARIO:   (1.0, 1.0),   # 用在什么环境
        RelationType.SEQUENCE:   (0.4, 0.4),
        RelationType.PREFERENCE: (0.8, 0.0),   # 用户怎么看待它
        RelationType.SOCIAL:     (0.3, 0.3),
        RelationType.ATTRIBUTE:  (0.7, 0.7),
        RelationType.TEMPORAL:   (0.3, 0.0),
        RelationType.TAXONOMIC:  (0.0, 0.3),   # 有哪些替代（子类） / 不往上抽象（父类=0）
    },
    NodeType.PERSON: {
        RelationType.CAUSAL:     (0.5, 0.5),
        RelationType.SCENARIO:   (0.6, 0.6),
        RelationType.SEQUENCE:   (0.5, 0.5),
        RelationType.PREFERENCE: (0.8, 0.0),   # 喜欢什么
        RelationType.SOCIAL:     (1.0, 1.0),   # 社交关系最重要
        RelationType.ATTRIBUTE:  (0.7, 0.7),
        RelationType.TEMPORAL:   (0.4, 0.0),
        RelationType.TAXONOMIC:  (0.0, 0.0),
    },
    NodeType.EMOTION: {
        RelationType.CAUSAL:     (0.9, 1.0),   # 情绪导致什么 / 什么导致情绪
        RelationType.SCENARIO:   (0.8, 0.8),
        RelationType.SEQUENCE:   (0.6, 0.0),
        RelationType.PREFERENCE: (0.5, 0.0),
        RelationType.SOCIAL:     (0.5, 0.5),
        RelationType.ATTRIBUTE:  (0.4, 0.4),
        RelationType.TEMPORAL:   (0.3, 0.0),
        RelationType.TAXONOMIC:  (0.0, 0.0),
    },
}


def get_jump_weight(
    source_type: NodeType,
    relation: RelationType,
    is_reverse: bool = False,
) -> float:
    """获取从源节点类型沿关系类型扩展的权重"""
    rule = JUMP_AXIS_RULES.get(source_type, {})
    if relation not in rule:
        return 0.0
    fwd, rev = rule[relation]
    return rev if is_reverse else fwd


def expand_weights(
    source_type: NodeType,
    relations: Dict[RelationType, bool],  # {relation_type: is_reverse}
) -> Dict[str, float]:
    """计算一次扩展中各候选边的权重
    
    Args:
        source_type: 当前节点的类型
        relations: 各条出边的类型和方向
        
    Returns:
        {edge_id: weight}
    """
    weights = {}
    for edge_id, (rel_type, is_reverse) in relations.items():
        w = get_jump_weight(source_type, rel_type, is_reverse)
        if w > 0:
            weights[edge_id] = w
    return weights
