# -*- coding: utf-8 -*-
"""E10 修复拟合：针对 D 项暴露的三个短板逐一打补丁，看能否把跨主体混入压下去。

诊断出的三个可修点（来自 E10-D / E5）：
1. **种子通道没有门槛**——全局 top-k 会把别的主体直接选成起点，后续几十个节点围着它扩展。
   修法：`seed_cos_min`（种子余弦下限，不足不补齐）。
2. **绝对阈值不可移植**——bge-m3 的无关基线 0.40–0.55，任何固定 τ 都很脆弱。
   修法：`purpose_filter_sigma`（每跳用 μ+kσ，只依赖本跳候选分布）。
3. **purposes 太抽象**——抽象短语把检索拉向"概括型节点"，反而更贴其他主体。
   修法：把 query 原话拼进 purposes（`concrete`），让目的向量落在具体语域。

六组配置，全部在 M / H 图上跑主体 A 的 30 条意图查询：
  base         τ=0.30（现状）
  seed.55      种子下限 0.55
  sigma0.5     相对阈值 μ+0.5σ
  sigma1.0     相对阈值 μ+1.0σ
  seed+sigma   seed.55 + sigma0.5（组合）
  concrete     purposes 拼上 query 原话

指标：跨主体混入率（越低越好）、gold facet recall（越高越好）、平均节点数。
两者要一起看——把混入压到 0 但 recall 也归零，等于什么都没做。

用法：python -m eval.e10_repair
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

GRAPHS = {"M": "data/dba_eval/e10_M.yaml", "H": "data/dba_eval/e10_H.yaml"}

CONFIGS = [
    ("base", {}),
    ("seed.55", {"seed_cos_min": 0.55}),
    ("sigma0.5", {"sigma": 0.5}),
    ("sigma1.0", {"sigma": 1.0}),
    ("seed+sigma", {"seed_cos_min": 0.55, "sigma": 0.5}),
    ("concrete", {"concrete": True}),
]


def _subject(nid: str) -> str:
    c = nid[0]
    return "A" if c in ("u", "a", "t") else c.upper()


def _flatten(path: Path) -> List[Dict[str, Any]]:
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    out = []
    for sc in cfg.get("scenarios") or []:
        prefix = str(sc["scenario"]).split("_")[0]
        for it in sc.get("intents") or []:
            out.append({"id": f"{prefix}_{it['intent']}", "text": sc["text"],
                        "intent": it["intent"],
                        "purposes": list(it.get("purposes") or []),
                        "gold": list(it.get("gold") or [])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="E10 修复拟合")
    ap.add_argument("--queries", default="data/dba_eval/intent_queries.yaml")
    ap.add_argument("--out", default="eval/outputs/e10/repair.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    queries = _flatten(ROOT / args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print(f"[E10-repair] queries={len(queries)} emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)

    report: Dict[str, Any] = {}
    for gname, gpath in GRAPHS.items():
        graph, retriever = build_retriever(str(ROOT / gpath), vector_cache_key=f"e10_{gname}")
        report[gname] = {}
        print(f"\n[{gname}] nodes={graph.node_count}", flush=True)
        for cname, cfg in CONFIGS:
            # 每次显式复位，避免上一组配置残留
            retriever.purpose_filter_sigma = float(cfg.get("sigma", 0.0))
            retriever.purpose_filter_threshold = 0.30
            rows = []
            for q in queries:
                purposes = list(q["purposes"])
                if cfg.get("concrete"):
                    purposes = [q["text"]] + purposes
                out = retriever.retrieve_with_story(
                    q["text"], seed_k=5, story=False, with_response=False,
                    purpose=purposes, seed_cos_min=cfg.get("seed_cos_min"))
                ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
                gold = set(q["gold"])
                off = [i for i in ids if _subject(i) != "A"]
                rows.append({
                    "n": len(ids),
                    "off_ratio": (len(off) / len(ids)) if ids else 0.0,
                    "recall": (len(set(ids) & gold) / len(gold)) if gold else None,
                })
            recs = [r["recall"] for r in rows if r["recall"] is not None]
            report[gname][cname] = {
                "mean_nodes": round(statistics.mean(r["n"] for r in rows), 1),
                "off_ratio": round(statistics.mean(r["off_ratio"] for r in rows), 4),
                "recall": round(statistics.mean(recs), 4) if recs else None,
            }
            r = report[gname][cname]
            print(f"  {cname:11s} 节点={r['mean_nodes']:5.1f}  混入={r['off_ratio']:6.1%}  "
                  f"recall={r['recall']:.3f}", flush=True)

    # 相对 base 的改善汇总
    print("\n=== 相对 base 的改善（Δ混入 为负更好；Δrecall 为正更好）===")
    for gname, g in report.items():
        b = g["base"]
        print(f"\n[{gname}] base: 混入 {b['off_ratio']:.1%} / recall {b['recall']:.3f} "
              f"/ 节点 {b['mean_nodes']}")
        for cname, r in g.items():
            if cname == "base":
                continue
            print(f"  {cname:11s} Δ混入={r['off_ratio']-b['off_ratio']:+.1%}  "
                  f"Δrecall={r['recall']-b['recall']:+.3f}  Δ节点={r['mean_nodes']-b['mean_nodes']:+.1f}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"experiment": "E10-repair",
                                      "queries": len(queries),
                                      "embedding": os.environ.get("EMBEDDING_MODEL"),
                                      "note": "混入=输出中非主体A占比；recall=gold facet 覆盖"},
                             "graphs": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E10-repair] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
