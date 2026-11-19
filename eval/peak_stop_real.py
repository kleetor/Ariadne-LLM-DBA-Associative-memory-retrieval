# -*- coding: utf-8 -*-
"""真实图对比：**首峰即停** vs 现行 patience2（走到底），看规模与覆盖的取舍。

背景：受控小图（17 节点）里两种模式没差别——扩展很快耗尽，patience 来不及重置。
真实图（163 节点、5 个种子、链更长）才看得出"多走几跳"的代价。

指标：输出节点数、gold facet recall、实际跳数、无关项（跨场景）占比。
判据：若首峰即停显著减少节点而 recall 掉幅很小（<3pp），则"宁可浅"的取舍成立。

用法：python -m eval.peak_stop_real
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


def _scene(nid: str) -> str:
    return nid[1] if len(nid) >= 2 else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description="首峰即停 vs patience2（真实图）")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--queries", default="data/dba_eval/comprehensive_queries.yaml")
    ap.add_argument("--emb-model", default=None, help="None=沿用 .env（m3）")
    ap.add_argument("--cache-key", default="comprehensive")
    ap.add_argument("--out", default="eval/outputs/mechanism/peak_stop_real.json")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    if args.emb_model:
        os.environ["EMBEDDING_MODEL"] = args.emb_model
    with open(ROOT / args.queries, encoding="utf-8") as f:
        qs = [q for q in (yaml.safe_load(f) or {}).get("queries", [])
              if str(q.get("category") or "") != "abstention"]

    _, r = build_retriever(str(ROOT / args.yaml), vector_cache_key=args.cache_key)
    print(f"[peak-real] queries={len(qs)} emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)

    report: Dict[str, Any] = {}
    for mode, fp in (("patience2", False), ("first_peak", True)):
        r.peak_finder.first_peak = fp
        rows: List[Dict[str, Any]] = []
        for q in qs:
            out = r.retrieve_with_story(
                str(q["text"]), seed_k=5, story=False, with_response=False,
                purpose=list(q.get("purposes") or []) or None)
            ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
            gold = set(q.get("gold_expected") or [])
            allowed = {_scene(g) for g in gold} or None
            rows.append({
                "n": len(ids),
                "recall": (len(set(ids) & gold) / len(gold)) if gold else None,
                "hops": len(out.get("hop_history") or []),
                "off": (sum(1 for i in ids if _scene(i) not in allowed) / len(ids))
                if (allowed and ids) else None,
            })
        recs = [x["recall"] for x in rows if x["recall"] is not None]
        offs = [x["off"] for x in rows if x["off"] is not None]
        report[mode] = {
            "nodes": round(statistics.mean(x["n"] for x in rows), 1),
            "recall": round(statistics.mean(recs), 4) if recs else None,
            "hops": round(statistics.mean(x["hops"] for x in rows), 2),
            "off_ratio": round(statistics.mean(offs), 4) if offs else None,
        }
        m = report[mode]
        print(f"  {mode:11s} 节点={m['nodes']:5.1f}  recall={m['recall']:.4f}  "
              f"跳数={m['hops']:.2f}  " +
              (f"跨场景={m['off_ratio']:.1%}" if m["off_ratio"] is not None else "跨场景=-"),
              flush=True)

    a, b = report["patience2"], report["first_peak"]
    print("\n=== 取舍 ===")
    print(f"  Δ节点={b['nodes']-a['nodes']:+.1f}  "
          f"Δrecall={b['recall']-a['recall']:+.4f}  Δ跳数={b['hops']-a['hops']:+.2f}")
    print("  判据：节点显著减少 且 recall 掉幅 <0.03 → 『宁可浅』成立。")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"embedding": os.environ.get("EMBEDDING_MODEL"),
                                      "n": len(qs)}, **report},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
