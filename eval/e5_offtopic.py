# -*- coding: utf-8 -*-
"""E10-D 预览：输出里混入了多少「跨场景无关节点」（填充位效应）。

用户提出的纯向量失效模式之二：
当某个意图真正相关的记忆不够多时，top-k 会把**凑数**的无关节点一起塞进来；
PAR 有绝对阈值挡板，理应混入更少。

指标（不依赖 judge、只读已有产物）：
  对每条 query，定义 `allowed` = 该意图 gold 节点覆盖的场景集合；
  输出节点中场景不在 allowed 的记为「跨场景无关」，算占比。
对比 par 与 vector_purpose 的占比，以及它随输出规模的变化（规模越大越可能凑数）。

用法：python -m eval.e5_offtopic
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CTX = ROOT / "eval/outputs/e5/context_e5_intent.json"
QY = ROOT / "data/dba_eval/intent_queries.yaml"


def _scene(nid: str) -> str:
    return nid[1] if len(nid) >= 2 else "?"


def main() -> None:
    import yaml
    with open(QY, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    allowed = {}
    for sc in cfg.get("scenarios") or []:
        prefix = str(sc["scenario"]).split("_")[0]
        for it in sc.get("intents") or []:
            allowed[f"{prefix}_{it['intent']}"] = {_scene(n) for n in (it.get("gold") or [])}

    data = json.load(open(CTX, encoding="utf-8"))["queries"]
    rows = []
    for r in data:
        st = r["stats"]
        allow = allowed.get(r["query_id"], set())
        if not allow:
            continue
        for arm, key in (("par", "par_ids"), ("vector_purpose", "vp_ids")):
            ids = st[key]
            off = [n for n in ids if _scene(n) not in allow]
            rows.append({
                "qid": r["query_id"], "arm": arm,
                "n": len(ids), "off": len(off),
                "ratio": len(off) / len(ids) if ids else 0.0,
            })

    print(f"{'arm':15s} {'题数':>5s} {'平均节点':>9s} {'无关节点':>9s} {'无关占比':>9s}")
    for arm in ("par", "vector_purpose"):
        sel = [x for x in rows if x["arm"] == arm]
        if not sel:
            continue
        print(f"{arm:15s} {len(sel):5d} {statistics.mean(x['n'] for x in sel):9.1f} "
              f"{statistics.mean(x['off'] for x in sel):9.1f} "
              f"{statistics.mean(x['ratio'] for x in sel):9.1%}")

    # 按输出规模分档：看"规模大 → 凑数多"是否成立
    print("\n按输出规模分档（无关占比）")
    for arm in ("par", "vector_purpose"):
        sel = [x for x in rows if x["arm"] == arm]
        print(f"  {arm}")
        for lo, hi, name in ((0, 15, "≤15"), (16, 30, "16–30"), (31, 999, ">30")):
            band = [x for x in sel if lo <= x["n"] <= hi]
            if band:
                print(f"    {name:6s} n={len(band):3d}  节点={statistics.mean(x['n'] for x in band):5.1f}"
                      f"  无关占比={statistics.mean(x['ratio'] for x in band):.1%}")

    # 逐题配对：par 比 vp 混入更少的比例
    by_q = {}
    for x in rows:
        by_q.setdefault(x["qid"], {})[x["arm"]] = x["ratio"]
    pairs = [(v["par"], v["vector_purpose"]) for v in by_q.values()
             if "par" in v and "vector_purpose" in v]
    win = sum(1 for a, b in pairs if a < b)
    tie = sum(1 for a, b in pairs if abs(a - b) < 1e-9)
    print(f"\n逐题配对：par 混入更少 {win}/{len(pairs)}，持平 {tie}，更多 {len(pairs)-win-tie}")


if __name__ == "__main__":
    main()
