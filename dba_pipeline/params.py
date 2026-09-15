# SPDX-License-Identifier: AGPL-3.0-only

"""运行时检索参数：权重矩阵 / 检索规模（种子）/ 目的回归系数。

这些参数原本写死在代码里——权重矩阵是 `core/jump_axis.py` 的模块级常量，
其余是 `retrieval/retriever.py` 的构造函数默认值——进程启动后无法调整。
本模块把它们抽成一份**跨进程共享的 YAML**（默认与图谱同目录：
`retrieval_params.yaml`）：

    WebUI 面板  ──写入──▶  retrieval_params.yaml  ◀──按 mtime 检测并就地生效── MCP

与图谱 YAML 一样，写操作走跨进程文件锁，读操作靠 mtime 判断是否需要重载，
因此面板调参后 MCP 在下一次读写时会自动生效，无需重启。
"""

from __future__ import annotations

import copy
import os
import tempfile
from typing import Dict, Optional

import yaml

from dba_pipeline import filelock
from dba_pipeline.core.jump_axis import JUMP_AXIS_RULES, NodeType, RelationType

PARAMS_FILENAME = "retrieval_params.yaml"

# 标量参数规格：(key, 分组, 标签, 类型, 最小, 最大, 步长, 说明)
# 既供面板自动渲染控件，也作为后端校验范围。
PARAM_SPEC = [
    ("seed_k", "retrieval", "种子数量", "int", 1, 50, 1, "向量检索进入联想扩张的初始种子数"),
    ("max_hops", "retrieval", "最大跳数", "int", 1, 10, 1, "联想扩张的最大跳数"),
    ("expand_k", "retrieval", "每轮扩展数", "int", 0, 50, 1, "0 表示与「种子数量」相同"),
    ("temporal_k_seed", "retrieval", "时序工具·种子数", "int", 1, 30, 1, "定位时间锚点的候选数"),
    ("temporal_max_facts", "retrieval", "时序工具·共时事实上限", "int", 0, 50, 1, "0 表示不截断"),
    ("temporal_max_anchors", "retrieval", "时序工具·锚点上限", "int", 1, 10, 1, "最多返回的时间锚点候选数"),
    ("distance_decay", "scoring", "距离衰减", "float", 0.0, 1.0, 0.01, "每跳距离衰减底数：decay^跳数"),
    ("jump_weight_coef", "scoring", "跳转轴权重系数", "float", 0.0, 1.0, 0.05, "组合得分中跳转轴权重占比"),
    ("purpose_weight_coef", "scoring", "目的关联权重系数", "float", 0.0, 1.0, 0.05, "组合得分中目的关联度占比"),
    ("purpose_filter_threshold", "scoring", "目的回归阈值", "float", 0.0, 1.0, 0.05, "低于该目的关联度的候选被过滤"),
    ("purpose_filter_decay", "scoring", "目的阈值逐跳衰减", "float", 0.0, 1.0, 0.05, "0 表示固定阈值不衰减"),
]

# 分组标题（面板分区展示用）
PARAM_GROUPS = [
    ("retrieval", "检索规模（种子）"),
    ("scoring", "目的回归与打分"),
]

# 关系类型的「方向含义」，用于面板标注权重矩阵里每个数字代表哪一侧：
#   正向 = 沿**出边**（顺箭头）扩展：当前节点 → 边指向的目标
#   反向 = 沿**入边**（逆箭头）回溯：指向当前节点的来源 → 当前节点
#
# 末位是「是否双向」：SCENARIO / SOCIAL / ATTRIBUTE 在建边时会自动补一条反向边
# （见 extraction/graph_builder.py），两侧取到的是同一批邻居，因此只描述对端对象，
# 不再强行区分正/反；其余 5 种是有向的，需分别说明两侧取到什么。
RELATION_DIRECTIONS = {
    "causal": ("导致的结果", "起因", False),
    "scenario": ("场景 / 情境", "场景 / 情境", True),
    "sequence": ("之后发生的", "之前发生的", False),
    "preference": ("偏好的对象", "偏好它的主体", False),
    "social": ("社交对象", "社交对象", True),
    "attribute": ("属性 / 特征", "属性 / 特征", True),
    "temporal": ("时间锚点", "该时间点的记忆", False),
    "taxonomic": ("父类（抽象）", "子类（具体）", False),
}

# 节点类型的中文名（与可视化面板的过滤项保持一致）
NODE_TYPE_LABELS = {
    "status": "状态",
    "reason": "原因",
    "action": "行为",
    "thing": "事物",
    "person": "人物",
    "emotion": "情绪",
}

# 标量参数的代码默认值（与 retrieval/retriever.py:206-210 的构造函数默认值一致）
_SCALAR_DEFAULTS = {
    "seed_k": 5,                      # retriever.py:412 retrieve(seed_k=5)
    "max_hops": 5,                    # retriever.py:413
    "expand_k": 0,                    # retriever.py:414（None → 用 seed_k），此处 0 表示同 seed_k
    "temporal_k_seed": 8,             # retriever.py:285
    "temporal_max_facts": 8,          # retriever.py:286
    "temporal_max_anchors": 3,        # retriever.py:287
    "distance_decay": 0.85,           # retriever.py:206
    "jump_weight_coef": 0.5,          # retriever.py:207
    "purpose_weight_coef": 0.5,       # retriever.py:208
    "purpose_filter_threshold": 0.2,  # retriever.py:209
    "purpose_filter_decay": 0.0,      # retriever.py:210
}


def params_path_for(yaml_path: str) -> str:
    """与图谱同目录的参数文件路径（统一绝对路径，跨进程一致）"""
    directory = os.path.dirname(os.path.abspath(yaml_path)) if yaml_path else "."
    return os.path.join(directory, PARAMS_FILENAME)


def default_weights() -> Dict[str, Dict[str, list]]:
    """默认权重矩阵：直接取自 jump_axis.JUMP_AXIS_RULES（单一事实来源）"""
    return {
        node_type.value: {
            rel.value: [float(pair[0]), float(pair[1])]
            for rel, pair in rules.items()
        }
        for node_type, rules in JUMP_AXIS_RULES.items()
    }


def default_params() -> dict:
    """全部参数的默认值"""
    retrieval = {k: _SCALAR_DEFAULTS[k] for k, g, *_ in PARAM_SPEC if g == "retrieval"}
    scoring = {k: _SCALAR_DEFAULTS[k] for k, g, *_ in PARAM_SPEC if g == "scoring"}
    return {"retrieval": retrieval, "scoring": scoring, "weights": default_weights()}


def merge(base: dict, patch: dict) -> dict:
    """深合并（patch 覆盖 base），用于面板/MCP 提交的部分参数"""
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def normalize(params: dict) -> dict:
    """校验并规整参数；非法值抛 ValueError（面板会转成 400 并给出中文提示）"""
    merged = merge(default_params(), params or {})
    spec = {
        k: (group, kind, lo, hi)
        for k, group, _label, kind, lo, hi, _step, _desc in PARAM_SPEC
    }

    unknown_groups = [key for key in merged if key not in ("retrieval", "scoring", "weights")]
    if unknown_groups:
        raise ValueError(f"未知的参数分组: {', '.join(sorted(unknown_groups))}")

    for group in ("retrieval", "scoring"):
        section = merged.get(group)
        if not isinstance(section, dict):
            raise ValueError(f"参数分组 {group} 必须是对象")
        # 只认登记过的键：否则参数名写错会被静默写进文件，运行时又读不到
        unknown_keys = [key for key in section if key not in spec]
        if unknown_keys:
            raise ValueError(
                f"参数分组 {group} 中存在未定义的参数: {', '.join(sorted(unknown_keys))}"
            )
        for key, (_group, kind, lo, hi) in spec.items():
            if key not in section:
                continue
            raw = section[key]
            if isinstance(raw, bool):  # bool 是 int 子类，不拦会被当成 0/1 放过
                raise ValueError(f"参数 {key} 必须是{'整数' if kind == 'int' else '数值'}")
            try:
                value = int(raw) if kind == "int" else float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"参数 {key} 必须是{'整数' if kind == 'int' else '数值'}")
            if kind == "int" and float(raw) != value:
                raise ValueError(f"参数 {key} 必须是整数")
            if value < lo or value > hi:
                raise ValueError(f"参数 {key} 超出允许范围 [{lo}, {hi}]")
            section[key] = value

    weights = merged.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("权重矩阵必须是对象")
    for node_type, rels in list(weights.items()):
        try:
            NodeType(node_type)
        except ValueError:
            raise ValueError(f"未知节点类型: {node_type}")
        if not isinstance(rels, dict):
            raise ValueError(f"节点类型 {node_type} 的权重必须是对象")
        for rel, pair in list(rels.items()):
            try:
                RelationType(rel)
            except ValueError:
                raise ValueError(f"未知关系类型: {rel}")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"{node_type}.{rel} 需要 [正向, 反向] 两个数值")
            try:
                fwd, rev = float(pair[0]), float(pair[1])
            except (TypeError, ValueError):
                raise ValueError(f"{node_type}.{rel} 的权重必须是数值")
            for name, value in (("正向", fwd), ("反向", rev)):
                if value < 0.0 or value > 1.0:
                    raise ValueError(f"{node_type}.{rel} 的{name}权重需在 [0, 1] 之间")
            rels[rel] = [fwd, rev]
    return merged


def load(path: str) -> dict:
    """读取参数文件（缺失/损坏时回退默认值，不抛异常）。不加锁。

    若需要「读-改-写」请改用 :func:`update`，否则并发调参会互相覆盖。
    """
    if not path or not os.path.exists(path):
        return default_params()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return normalize(raw)
    except Exception:
        return default_params()


def _write_atomic(path: str, clean: dict) -> None:
    """把已规整的参数原子落盘（调用方必须已持有参数文件锁）。"""
    data = yaml.safe_dump(clean, allow_unicode=True, default_flow_style=False, sort_keys=False)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save(path: str, params: dict) -> dict:
    """校验后**整体**写入参数文件（跨进程文件锁内），返回落盘后的参数。

    用于「恢复默认」这类整份覆盖；部分修改请用 :func:`update`。
    """
    clean = normalize(params)
    with filelock.FileLock(filelock.lock_path_for(path)).hold():
        _write_atomic(path, clean)
    return clean


def update(path: str, patch: dict) -> dict:
    """在跨进程文件锁内完成「读取 → 深合并补丁 → 校验 → 原子写」。

    面板与 MCP 都走这里落盘：读和写处于同一个临界区，因此两边即使同时调参，
    也各自基于**锁内读到的最新版本**合并，不会互相覆盖（避免丢失更新）。
    """
    with filelock.FileLock(filelock.lock_path_for(path)).hold():
        merged = normalize(merge(load(path), patch or {}))
        _write_atomic(path, merged)
    return merged


def apply_weights(weights: dict) -> int:
    """把权重矩阵**就地**写入 JUMP_AXIS_RULES（所有读取方立即可见），返回改动条数"""
    changed = 0
    for node_type_value, rels in (weights or {}).items():
        try:
            node_type = NodeType(node_type_value)
        except ValueError:
            continue
        rule = JUMP_AXIS_RULES.setdefault(node_type, {})
        for rel_value, pair in (rels or {}).items():
            try:
                relation = RelationType(rel_value)
            except ValueError:
                continue
            value = (float(pair[0]), float(pair[1]))
            if rule.get(relation) != value:
                changed += 1
            rule[relation] = value
    return changed


def apply(params: dict, retriever=None) -> dict:
    """把参数应用到运行时对象：权重矩阵就地更新，打分系数写入 retriever 实例。

    Returns:
        {"weights": 改动条数, "retriever": 改动条数}
    """
    clean = normalize(params)
    weights_changed = apply_weights(clean["weights"])
    retriever_changed = 0
    if retriever is not None:
        for key, value in clean["scoring"].items():
            if getattr(retriever, key, None) != value:
                setattr(retriever, key, value)
                retriever_changed += 1
    return {"weights": weights_changed, "retriever": retriever_changed}


def _relation_spec(rel: RelationType) -> dict:
    """单个关系类型的面板元信息（含正/反向含义）"""
    forward, reverse, bidirectional = RELATION_DIRECTIONS.get(rel.value, ("", "", False))
    return {
        "key": rel.value,
        "label": rel.name,
        "bidirectional": bidirectional,
        "forward": forward,
        "reverse": reverse,
    }


def spec_payload() -> dict:
    """给面板的元信息：分组、字段规格、默认值"""
    return {
        "groups": [{"key": k, "label": label} for k, label in PARAM_GROUPS],
        "fields": [
            {
                "key": k, "group": g, "label": label, "kind": kind,
                "min": lo, "max": hi, "step": step, "desc": desc,
                "default": _SCALAR_DEFAULTS[k],
            }
            for k, g, label, kind, lo, hi, step, desc in PARAM_SPEC
        ],
        "node_types": [
            {"key": nt.value, "label": nt.name, "cn": NODE_TYPE_LABELS.get(nt.value, "")}
            for nt in NodeType
        ],
        "relations": [_relation_spec(rt) for rt in RelationType],
        "defaults": default_params(),
    }
