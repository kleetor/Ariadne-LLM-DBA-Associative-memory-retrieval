# -*- coding: utf-8 -*-
"""
时序动态更新测试：模拟 dba 增量维护向图谱新增"时间锚点 + 事件 + TEMPORAL 边"，
验证 temporal_lookup 能否正确反映动态更新后的时序结构，并检验：
  ① 新时间锚点（id 为 nX、非 th/T 前缀）能否被 _tool_is_time_node 用**内容正则**识别；
  ② 新增的共时事件能否被反向 TEMPORAL 边取到；
  ③ 移除一条 TEMPORAL 边后，对应事件是否即时消失（动态反映）。

用法:
    python -m eval.test_temporal_dynamic
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever
from dba_pipeline.core.jump_axis import NodeType, RelationType

BASE = "data/dba_eval/lifemem_crossperiod.yaml"


def main() -> None:
    _load_dotenv(ROOT / ".env")
    graph, r = build_retriever(BASE)
    vs = r.vector_store

    # 动态新增：时间锚点 n15（内容含时间词"这周"，id 非 th/T 前缀）+ 两个事件 n16/n17
    print("== 动态新增（n15=这周, n16=准备周报, n17=组织评审）==")
    graph.graph.add_node("n15", content="这周", node_type=NodeType.THING,
                         metadata={}, deprecated=False, forgotten=False)
    graph.graph.add_node("n16", content="用户正在准备周报", node_type=NodeType.ACTION,
                         metadata={}, deprecated=False, forgotten=False)
    graph.graph.add_node("n17", content="用户组织了一次评审", node_type=NodeType.ACTION,
                         metadata={}, deprecated=False, forgotten=False)
    graph.graph.add_edge("n16", "n15", rel_type=RelationType.TEMPORAL)
    graph.graph.add_edge("n17", "n15", rel_type=RelationType.TEMPORAL)
    vs.add_memories(["n15", "n16", "n17"],
                    ["这周", "用户正在准备周报", "用户组织了一次评审"])

    # ① 新时间锚点能否被内容正则识别
    print(f"  _tool_is_time_node(n15) = {r._tool_is_time_node('n15')}")
    print(f"  _has_reverse_temporal(n15) = {r._has_reverse_temporal('n15')}")

    # ② 动态新增的共时事件能否被取到
    res = r.temporal_lookup("这周用户都做了什么？")
    print("\n[temporal_lookup] 这周用户都做了什么？")
    for g in res.get("matches", []):
        ta = g["time_anchor"]
        print(f"  锚点: {ta['id']} = {ta['content']!r}")
        for f in g["facts"]:
            print(f"     - {f['id']} → {f['content']!r}")
    got = {f["id"] for g in res.get("matches", []) for f in g["facts"]}
    anchored = any(g["time_anchor"]["id"] == "n15" for g in res.get("matches", []))
    ok_add = anchored and {"n16", "n17"} <= got
    print(f"  判定①: 锚定新时间锚点n15={anchored}; 取到动态新增事件(n16/n17)={ok_add}")

    # ③ 移除一条 TEMPORAL 边后，对应事件即时消失
    print("\n== 动态删除：移除 n17 -> n15 的 TEMPORAL 边 ==")
    graph.graph.remove_edge("n17", "n15")
    # 同步移除向量（否则 n17 仍可能作为无关节点被检索到，但不应再作为 n15 的共时事件）
    vs.remove_memories(["n17"])
    res2 = r.temporal_lookup("这周用户都做了什么？")
    got2 = {f["id"] for g in res2.get("matches", []) for f in g["facts"]}
    ok_del = "n17" not in got2 and "n16" in got2
    print(f"  移除后取到: {got2}")
    print(f"  判定③: n17 消失且 n16 仍在 = {ok_del}")

    print("\n" + "=" * 60)
    if ok_add and ok_del:
        print("时序动态更新测试通过 [OK]")
    else:
        print("时序动态更新测试未通过")
        sys.exit(1)


if __name__ == "__main__":
    main()
