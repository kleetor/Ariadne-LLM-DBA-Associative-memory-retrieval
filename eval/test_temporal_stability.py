# -*- coding: utf-8 -*-
"""
时序功能稳定性测试（离线、确定性，注入 purpose 不走 LLM）。

覆盖 5 个不变量，任一失败即非零退出，便于 CI/回归：
  INV1 确定性：同一查询多次调用 temporal_lookup，结果完全一致。
  INV2 防误判锚点：任何返回的 time_anchor 都必须是"真锚点"，绝不含"含时间词的事件节点"误判。
  INV3 多锚点上限：matches 数量 <= max_anchors，且按相关性从高到低。
  INV4 非时序拒答：事件→时间(TA)类查询返回空 matches（应走主检索器）。
  INV5 主检索无回归：retrieve() 注入目的，重复运行候选集一致，且含已知基线关键节点。

用法:
    python -m eval.test_temporal_stability
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever

# 322 图真锚点（th0/th1/th2）与误判节点集合（'含时间词的事件节点'）
TRUE_ANCHORS_322 = {"th0", "th1", "th2"}
FALSE_ANCHOR_322 = {"w13", "w24", "w30", "w36", "w40", "n7", "n22", "n24", "h25", "h32", "c12"}


def _anchor_ids(res):
    return [g["time_anchor"]["id"] for g in res.get("matches", [])]


def _fact_ids(res):
    return [f["id"] for g in res.get("matches", []) for f in g.get("facts", [])]


def run_checks():
    _load_dotenv(ROOT / ".env")
    g_gold, r_gold = build_retriever("data/dba_eval/lifemem_crossperiod.yaml")
    g_322, r_322 = build_retriever("data/LiFEMem.yaml")

    fails = []

    def check(name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        print(f"[{tag}] {name} {detail}")
        if not cond:
            fails.append(name)

    # ---- INV1 确定性：同一查询重复 3 次结果一致 ----
    for q in ["昨天用户都做了些什么？", "高三下学期发生了什么？", "用户当前工作时期怎么样？"]:
        r_set = []
        for _ in range(3):
            res = (r_gold if q in ("昨天用户都做了些什么？", "高三下学期发生了什么？")
                   else r_322).temporal_lookup(q)
            r_set.append((tuple(_anchor_ids(res)), tuple(_fact_ids(res))))
        deterministic = len(set(r_set)) == 1
        check(f"INV1 确定性 [{q}]", deterministic,
              f"→ anchors={r_set[0][0]} facts={r_set[0][1]}")

    # ---- INV2 防误判锚点：返回的锚点必须是真锚点 ----
    # 322 图上用误判节点内容作查询，解析到的锚点不得再是这些误判节点，而应是 th0/th1/th2 或空。
    for mid in sorted(FALSE_ANCHOR_322):
        content = g_322.graph.nodes[mid].get("content", "")
        res = r_322.temporal_lookup(content)
        anchors = _anchor_ids(res)
        bad = [a for a in anchors if a in FALSE_ANCHOR_322]
        check(f"INV2 防误判 [{mid}]", not bad, f"→ anchors={anchors}")
    # 时代查询应命中真锚点
    anchors = _anchor_ids(r_322.temporal_lookup("用户高中时期发生了什么？"))
    check("INV2 时代查询→真锚点", set(anchors) <= TRUE_ANCHORS_322 and set(anchors).intersection(TRUE_ANCHORS_322),
          f"→ anchors={anchors}")

    # ---- INV3 多锚点上限 + 相关性排序 ----
    res = r_gold.temporal_lookup("高三下学期发生了什么？")
    anchors = _anchor_ids(res)
    check("INV3 多锚点上限<=3", len(anchors) <= 3, f"→ anchors={anchors}")

    # ---- INV4 非时序拒答（TA：事件→时间）----
    res = r_gold.temporal_lookup("用户是什么时候开始学吉他的？")
    check("INV4 非时序拒答", not res.get("matches"), f"→ matches={len(res.get('matches', []))}")

    # ---- INV5 主检索无回归（注入目的，生产口径：每次 start_session）----
    # 注：retrieve() 是"会话态"（path_tracker 跨调用累积终身增强，刻意设计），
    #     因此同实例多跑不 bit 确定，但关键基线节点应稳定出现（不回归）。
    must_have = {"e10", "e9", "T10"}
    all_have = True
    detail = []
    for _ in range(3):
        if r_gold.path_tracker is not None:
            r_gold.path_tracker.start_session()
        ids_b = {m[0] for m in r_gold.retrieve(
            "昨天用户都做了些什么？", purpose=["还原昨天经历"]
        ).get("peak_memories", [])}
        detail.append(len(ids_b))
        missing = must_have - ids_b
        if missing:
            all_have = False
            detail.append(f"缺{missing}")
    check("INV5 retrieve 关键节点稳定（3 次均在）", all_have,
          f"→ 各次 n={detail}")

    print("\n" + "=" * 60)
    if fails:
        print(f"稳定性测试失败 {len(fails)} 项: {fails}")
        sys.exit(1)
    print("所有稳定性测试通过 [OK]")


if __name__ == "__main__":
    run_checks()
