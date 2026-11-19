# -*- coding: utf-8 -*-
"""E5 产物调研：三种意图是否真的"各就各位"，还是都漂向同一类节点。

E4 已经证明：抽象 purposes 会把检索拽向"计划/建议类"节点，把具体事实挤掉。
E5 用同一句话换三种目的，正好可以检验这条解释是否在**意图维度**上同样成立：

  若目的表示有效 → 倾诉该多召回 emotion/status、规划该多 action/时间、推荐该多偏好/选项；
  若目的表示过抽象 → 三种意图的**节点类型分布会几乎一样**（都偏 action），
  那"同一句话换目的结果不同"就只是噪声性的换位，不是"理解变了"。

只读已有产物 `context_e5_intent.json`（含每条的 par_ids / vp_ids），不重跑检索。
指标：每个意图下的节点类型占比、gold recall、以及三意图分布之间的差异度。

用法：python -m eval.inspect_e5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml  # noqa: E402

CTX = "eval/outputs/e5/context_e5_intent.json"
GOLD = "data/dba_eval/comprehensive_gold.yaml"
INTENT_CN = {"vent": "倾诉", "plan": "规划", "recommend": "推荐"}


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 产物调研：意图 × 节点类型")
    ap.add_argument("--ctx", default=CTX)
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--out", default="eval/outputs/e5/inspect.json")
    args = ap.parse_args()

    with open(ROOT / args.gold, encoding="utf-8") as f:
        ntype = {n["id"]: str(n.get("type") or "?")
                 for n in (yaml.safe_load(f) or {}).get("nodes", [])}
        ncontent = {n["id"]: n.get("content", "")
                    for n in (yaml.safe_load(f) or {}).get("nodes", [])}
    data = json.load(open(ROOT / args.ctx, encoding="utf-8"))["queries"]

    # 逐条：类型分布 + gold recall
    rows: List[Dict[str, Any]] = []
    for r in data:
        st = r["stats"]
        gold = set()
        # gold 不在 ctx 里，按 scenario+intent 从数据集读
        rows.append({
            "qid": r["query_id"], "scenario": r["scenario"], "intent": r["intent"],
            "par_ids": st["par_ids"], "vp_ids": st["vp_ids"],
            "par_types": Counter(ntype.get(i, "?") for i in st["par_ids"]),
            "vp_types": Counter(ntype.get(i, "?") for i in st["vp_ids"]),
        })

    # 从数据集补 gold
    with open(ROOT / "data/dba_eval/intent_queries.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    goldmap = {}
    for sc in cfg.get("scenarios") or []:
        prefix = str(sc["scenario"]).split("_")[0]
        for it in sc.get("intents") or []:
            goldmap[f"{prefix}_{it['intent']}"] = set(it.get("gold") or [])
    for r in rows:
        g = goldmap.get(r["qid"], set())
        for arm, key in (("par", "par_ids"), ("vp", "vp_ids")):
            ids = r[key]
            r[f"{arm}_recall"] = (len(set(ids) & g) / len(g)) if g else None

    # 按意图聚合类型分布
    print(f"{'意图':6s} {'n':>3s} " + " ".join(f"{t:>8s}" for t in
                                          ("status", "action", "emotion", "thing", "person", "reason")))
    dist: Dict[str, Dict[str, float]] = {}
    for intent in ("vent", "plan", "recommend"):
        sel = [r for r in rows if r["intent"] == intent]
        total = Counter()
        for r in sel:
            total.update(r["par_types"])
        n = sum(total.values()) or 1
        dist[intent] = {t: total.get(t, 0) / n for t in
                        ("status", "action", "emotion", "thing", "person", "reason")}
        print(f"{INTENT_CN[intent]:6s} {len(sel):3d} " +
              " ".join(f"{dist[intent][t]:8.1%}" for t in dist[intent]))

    # 三意图之间的分布差异（L1 距离 / 2 = 总变差）
    print("\n=== 意图之间节点类型分布的差异（0=完全相同，1=完全不相交）===")
    keys = ("vent", "plan", "recommend")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = dist[keys[i]], dist[keys[j]]
            tvd = sum(abs(a[t] - b[t]) for t in a) / 2
            print(f"  {INTENT_CN[keys[i]]} vs {INTENT_CN[keys[j]]}: TVD={tvd:.3f}")

    print("\n=== gold recall ===")
    for intent in keys:
        sel = [r for r in rows if r["intent"] == intent]
        p = [r["par_recall"] for r in sel if r["par_recall"] is not None]
        v = [r["vp_recall"] for r in sel if r["vp_recall"] is not None]
        print(f"  {INTENT_CN[intent]:6s} par={statistics.mean(p):.3f}  vp={statistics.mean(v):.3f}")

    # 抽样：看同一 scenario 三种意图各自的 top 节点内容
    print("\n=== 抽样：同一情景下三种意图召回的前 6 个节点 ===")
    for sc in ("a01_工作强度", "a05_学习进度"):
        print(f"\n--- {sc}")
        for intent in keys:
            r = next((x for x in rows if x["scenario"] == sc and x["intent"] == intent), None)
            if not r:
                continue
            ids = r["par_ids"][:6]
            print(f"  [{INTENT_CN[intent]}] " +
                  " | ".join(f"{ntype.get(i,'?')[:3]}:{ncontent.get(i,'')[:16]}" for i in ids))

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"experiment": "E5 inspect"},
        "type_dist": dist,
        "per_query": [{k: v for k, v in r.items() if k not in ("par_types", "vp_types")}
                      | {"par_types": dict(r["par_types"]), "vp_types": dict(r["vp_types"])}
                      for r in rows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[inspect] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
