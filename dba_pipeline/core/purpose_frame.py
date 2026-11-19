# SPDX-License-Identifier: AGPL-3.0-only

"""目的帧（PurposeFrame）：把自由文本目的编译成可匹配图属性的结构化约束。

为什么要做（E5 定位，2026-11-18）
--------------------------------
目的此前被表示成「把三条目的短语拼接、取一个向量」。这种表示只保留了**话题**，
丢掉了**检索意图**：同一话题下「我想倾诉」与「我要排期」在嵌入空间里几乎同义——
E5 实测三种意图召回的节点类型分布 TVD 仅 0.042–0.066，倾诉只召回 10.6% 的情绪节点。
问题不在检索机制，而在这个表示的**信息量**：一个 1024 维的池化向量装不下
「要哪一类节点、沿哪一类关系找」。

帧的形态
--------
把目的拆成两维可匹配到图属性的约束：

    types      : 6 类节点上的目标分布（STATUS/REASON/ACTION/THING/PERSON/EMOTION）
    relations  : 8 类关系上的期望分布（CAUSAL/SCENARIO/SEQUENCE/...）

检索侧按 `w_emb·余弦 + w_type·类型亲和 + w_rel·关系亲和` 打分（见 retriever），
于是「同一句话 + 不同目的」在**种子选取、逐跳过滤、排序**三处都能分开。

关于亲和度的 floor
------------------
不把非目标类型直接判 0。倾诉的 gold 里也有 action（「每天加班到十一点」是情绪的
事实依据），硬砍会连带砍掉证据。`floor` 把非目标类型压到 0.25/0.15，保留可达性，
让类型只影响**排序与门限**，不做硬筛。

关于 compile_frame 的局限
------------------------
`compile_frame` 是按目的文本里的**语义线索词**做映射，不看数据集里的 intent 标签，
因此不构成标签泄漏；但它仍是手工词典，不是通用的帧推断器。论文里对应的验收项是
「帧编译」（自由目的 → 帧），当前只用于验证**表示本身**是否能把检索分开。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dba_pipeline.core.jump_axis import NodeType, RelationType

# 非目标类型的亲和下限：保留「跨类型证据」（情绪的成因常是 action）
TYPE_FLOOR = 0.25
REL_FLOOR = 0.15


def _key(x) -> str:
    """把 NodeType/RelationType 枚举或落盘字符串统一成小写值。"""
    if isinstance(x, (NodeType, RelationType)):
        return x.value
    return str(x or "").split(".")[-1].strip().lower()


@dataclass(frozen=True)
class PurposeFrame:
    """一次检索的目的帧：目标节点类型 × 期望关系类型。"""

    name: str
    types: Dict[str, float]
    relations: Dict[str, float] = field(default_factory=dict)
    type_floor: float = TYPE_FLOOR
    rel_floor: float = REL_FLOOR

    def type_affinity(self, node_type) -> float:
        """节点类型对目的的亲和度 ∈ [type_floor, 1]。空帧视为无偏好（1.0）。"""
        top = max(self.types.values()) if self.types else 0.0
        if top <= 0:
            return 1.0
        return self.type_floor + (1 - self.type_floor) * (
            self.types.get(_key(node_type), 0.0) / top
        )

    def rel_affinity(self, rel_type) -> float:
        """边类型对目的的亲和度 ∈ [rel_floor, 1]；无关系偏好时取中性 0.5。"""
        top = max(self.relations.values()) if self.relations else 0.0
        if top <= 0:
            return 0.5
        return self.rel_floor + (1 - self.rel_floor) * (
            self.relations.get(_key(rel_type), 0.0) / top
        )

    @property
    def dominant_type(self) -> str:
        return max(self.types, key=self.types.get) if self.types else ""

    def describe(self) -> str:
        t = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.types.items(), key=lambda p: -p[1]))
        r = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.relations.items(), key=lambda p: -p[1]))
        return f"[{self.name}] types({t}) relations({r})"


# 三种典型交互目的的帧。数值是**先验**，不是拟合结果；用于验证表示形态。
# 倾诉：用户要的是「被理解的状态」→ 情绪/状态为主，因果解释其次，动作最低。
# 规划：用户要的是「接下来做什么」→ 动作/时间物为主，当前状态作背景。
# 推荐：用户要的是「可选项」→ 执行动作（助手做了哪些对比/整理）与物（具体选项）为主。
# 注意 THING 不能压过 ACTION：图里的 THING 大量是时间锚（"这周"/"高考前"），
# 它们不是选项；把 THING 置顶会把时间锚当推荐结果（a01 实测 recall 0.33 → 0.00）。
PRESETS: Dict[str, PurposeFrame] = {
    "vent": PurposeFrame(
        name="vent",
        types={NodeType.EMOTION.value: 1.0, NodeType.STATUS.value: 0.55,
               NodeType.REASON.value: 0.35, NodeType.ACTION.value: 0.15,
               NodeType.THING.value: 0.05, NodeType.PERSON.value: 0.05},
        relations={RelationType.CAUSAL.value: 1.0, RelationType.ATTRIBUTE.value: 0.7,
                   RelationType.SOCIAL.value: 0.3, RelationType.SCENARIO.value: 0.2},
    ),
    "plan": PurposeFrame(
        name="plan",
        types={NodeType.ACTION.value: 1.0, NodeType.THING.value: 0.45,
               NodeType.STATUS.value: 0.25, NodeType.REASON.value: 0.20,
               NodeType.EMOTION.value: 0.05, NodeType.PERSON.value: 0.05},
        relations={RelationType.SEQUENCE.value: 1.0, RelationType.TEMPORAL.value: 0.8,
                   RelationType.SCENARIO.value: 0.4, RelationType.CAUSAL.value: 0.3},
    ),
    "recommend": PurposeFrame(
        name="recommend",
        types={NodeType.ACTION.value: 1.0, NodeType.THING.value: 0.55,
               NodeType.STATUS.value: 0.35, NodeType.PERSON.value: 0.25,
               NodeType.EMOTION.value: 0.05, NodeType.REASON.value: 0.10},
        relations={RelationType.TAXONOMIC.value: 1.0, RelationType.ATTRIBUTE.value: 0.8,
                   RelationType.PREFERENCE.value: 0.5, RelationType.SCENARIO.value: 0.3},
    ),
}

# 线索词只从**目的文本**里取，不看数据集标签。顺序即优先级（同分时靠前者胜）。
_CUES: List[tuple] = [
    ("vent", re.compile(r"倾诉|表达|寻求|共情|理解|安慰|鼓励|共鸣|怀念|担心|难受|不满|焦虑|感慨")),
    ("plan", re.compile(r"规划|安排|梳理|列出|设定|设置|确定|准备|标出|预留|步骤|节奏|拆分|进度")),
    ("recommend", re.compile(r"推荐|比较|获取|建议|选择|方向|选项")),
]

# 无任何线索命中时的空帧：types 全空 → 亲和恒为 1，等价于「不做类型约束」。
NEUTRAL = PurposeFrame(name="neutral", types={}, relations={})


def compile_frame(purposes: List[str]) -> PurposeFrame:
    """从目的文本编译帧（线索词计数取最大者）。

    这是**占位实现**：真正的帧编译（自由目的 → 类型/关系分布）应由模型完成，
    见论文 §7.6。这里保持确定性，好让 E11 的差异只来自表示形态。
    """
    text = " ".join(p or "" for p in purposes)
    if not text.strip():
        return NEUTRAL
    counts = {name: len(pat.findall(text)) for name, pat in _CUES}
    best = max(counts, key=lambda k: counts[k])
    if counts[best] == 0:
        return NEUTRAL
    return PRESETS[best]


def frame_for_name(name: Optional[str]) -> PurposeFrame:
    return PRESETS.get(str(name or "").lower(), NEUTRAL)
