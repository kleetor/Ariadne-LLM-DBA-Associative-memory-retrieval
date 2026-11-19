# -*- coding: utf-8 -*-
"""专项：**目的过滤这一环**到底有没有效应（区分"未触发 / 零和 / 负面"）。

问题：现有证据互相矛盾——
- E4：`no_purpose`（整条目的通道关闭）的 gold recall 反而更高（0.894 vs 0.857）；
- E10-D：`par` 的跨主体混入比**无阈值**的向量臂低 7–9pp。
两者不是同一变量：前者关的是"种子+过滤+加权+寻峰判据"，后者对比的是"有过滤 vs 无过滤"，
但后者的对照组还差一个"图扩展局部性"。所以必须单独隔离**过滤**这一个环节。

四个条件（全部保留目的种子、目的加权、寻峰判据不变——只动过滤）：
  full        τ = 0.30（现行默认）
  no_filter   τ = 0.0   —— 过滤恒不触发（**只关过滤**）
  tight       τ = 0.55  —— 让过滤真正生效（现行阈值下它从未触发）
  no_purpose  purpose_blind（整条目的通道关闭，作上界参照）

两张图：
  L（163 节点，单主体）  → 看 gold facet recall 与规模
  H（343 节点，三主体）  → 看**跨主体混入率**（这才是过滤该发挥作用的地方）

同时记录每跳的"进/出"（`_purpose_stats`），直接给出过滤的实际触发率。

判读标准：
- 若 `full ≈ no_filter` → 过滤**未被检验**（不是无效）；
- 若 `tight` 降混入但等量掉 recall → 过滤**零和**（中性，非负）；
- 若 `tight` 同时改善两者 → 过滤有效；
- 若 `no_filter > full` 在 recall 上成立 → 过滤**有负面**。

用法：python -m eval.inspect_purpose_filter
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

CONFIGS = [
    ("full", {"tau": 0.30}),
    ("no_filter", {"tau": 0.0}),
    ("tight", {"tau": 0.55}),
    ("no_purpose", {"blind": True}),
]


def _subject(nid: str) -> str:
    c = nid[0]
    return "A" if c in ("u", "a", "t") else c.upper()


def _flatten_intent(path: Path) -> List[Dict[str, Any]]:
    cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
    out = []
    for sc in cfg.get("scenarios") or []:
        prefix = str(sc["scenario"]).split("_")[0]
        for it in sc.get("intents") or []:
            out.append({"id": f"{prefix}_{it['intent']}", "text": sc["text"],
                        "purposes": list(it.get("purposes") or []),
                        "gold": list(it.get("gold") or [])})
    return out


def _run(retriever, q: Dict[str, Any], cfg: Dict[str, Any], multi_subject: bool) -> Dict[str, Any]:
    retriever.purpose_filter_sigma = 0.0          # 只用绝对 τ，隔离变量
    retriever.purpose_filter_threshold = float(cfg.get("tau", 0.30))
    out = retriever.retrieve_with_story(
        q["text"], seed_k=5, story=False, with_response=False,
        purpose=None if cfg.get("blind") else (list(q["purposes"]) or None),
        purpose_blind=bool(cfg.get("blind")))
    ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
    gold = set(q.get("gold") or [])
    stats = getattr(retriever, "_purpose_stats", []) or []
    ins = sum(s["in"] for s in stats if s["hop"] >= 1)
    outs = sum(s["out"] for s in stats if s["hop"] >= 1)
    row = {
        "n": len(ids),
        "recall": (len(set(ids) & gold) / len(gold)) if gold else None,
        "filter_in": ins, "filter_out": outs,
        "filter_pass": (outs / ins) if ins else None,
    }
    if multi_subject:
        off = [i for i in ids if _subject(i) != "A"]
        row["off_ratio"] = (len(off) / len(ids)) if ids else 0.0
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="目的过滤专项分析")
    ap.add_argument("--out", default="eval/outputs/inspect_purpose_filter.json")
    args = ap.parse_args()
    _load_dotenv(ROOT / ".env")

    # ---- L 图：comprehensive 70 条 ----
    with open(ROOT / "data/dba_eval/comprehensive_queries.yaml", encoding="utf-8") as f:
        qs = [q for q in (yaml.safe_load(f) or {}).get("queries", [])
              if str(q.get("category") or "") != "abstention"]
    comp = [{"id": str(q["id"]), "text": str(q["text"]),
             "purposes": list(q.get("purposes") or []),
             "gold": list(q.get("gold_expected") or [])} for q in qs]

    report: Dict[str, Any] = {}
    graph, retriever = build_retriever(str(ROOT / "data/dba_eval/e10_L.yaml"),
                                       vector_cache_key="e10_L")
    print(f"[L] nodes={graph.node_count} queries={len(comp)}", flush=True)
    report["L"] = {}
    for name, cfg in CONFIGS:
        rows = [_run(retriever, q, cfg, multi_subject=False) for q in comp]
        recs = [r["recall"] for r in rows if r["recall"] is not None]
        passes = [r["filter_pass"] for r in rows if r["filter_pass"] is not None]
        report["L"][name] = {
            "nodes": round(statistics.mean(r["n"] for r in rows), 1),
            "recall": round(statistics.mean(recs), 4) if recs else None,
            "filter_pass": round(statistics.mean(passes), 4) if passes else None,
            "queries_with_filtering": sum(1 for r in rows
                                          if r["filter_pass"] is not None and r["filter_pass"] < 0.999),
        }
        r = report["L"][name]
        print(f"  {name:11s} 节点={r['nodes']:5.1f} recall={r['recall']:.3f} "
              f"过滤通过率={'-' if r['filter_pass'] is None else format(r['filter_pass'], '.1%')} "
              f"真的筛过的题={r['queries_with_filtering']}/{len(comp)}", flush=True)

    # ---- H 图：意图查询 30 条（有跨主体干扰）----
    intent = _flatten_intent(ROOT / "data/dba_eval/intent_queries.yaml")
    graph, retriever = build_retriever(str(ROOT / "data/dba_eval/e10_H.yaml"),
                                       vector_cache_key="e10_H")
    print(f"\n[H] nodes={graph.node_count} queries={len(intent)}", flush=True)
    report["H"] = {}
    for name, cfg in CONFIGS:
        rows = [_run(retriever, q, cfg, multi_subject=True) for q in intent]
        recs = [r["recall"] for r in rows if r["recall"] is not None]
        report["H"][name] = {
            "nodes": round(statistics.mean(r["n"] for r in rows), 1),
            "recall": round(statistics.mean(recs), 4) if recs else None,
            "off_ratio": round(statistics.mean(r["off_ratio"] for r in rows), 4),
            "filter_pass": round(statistics.mean(r["filter_pass"] for r in rows
                                                 if r["filter_pass"] is not None), 4)
            if any(r["filter_pass"] is not None for r in rows) else None,
        }
        r = report["H"][name]
        print(f"  {name:11s} 节点={r['nodes']:5.1f} recall={r['recall']:.3f} "
              f"跨主体混入={r['off_ratio']:6.1%} "
              f"过滤通过率={'-' if r['filter_pass'] is None else format(r['filter_pass'], '.1%')}",
              flush=True)

    # ---- 判读 ----
    print("\n=== 判读 ===")
    for gname in ("L", "H"):
        b, nf = report[gname]["full"], report[gname]["no_filter"]
        t = report[gname]["tight"]
        print(f"\n[{gname}]")
        print(f"  full vs no_filter（只关过滤）：Δrecall={b['recall']-nf['recall']:+.3f}"
              + (f"  Δ混入={b['off_ratio']-nf['off_ratio']:+.1%}" if "off_ratio" in b else ""))
        print(f"  tight vs full（让过滤真正生效）：Δrecall={t['recall']-b['recall']:+.3f}"
              + (f"  Δ混入={t['off_ratio']-b['off_ratio']:+.1%}" if "off_ratio" in b else ""))
    print("\n结论口径：full≈no_filter → 过滤未被检验；tight 两者同向改善 → 有效；"
          "tight 降混入但等量掉 recall → 零和；no_filter 的 recall 更高 → 有负面。")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"experiment": "purpose filter focus"},
                             "graphs": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
