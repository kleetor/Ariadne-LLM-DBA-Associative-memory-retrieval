# -*- coding: utf-8 -*-
"""探测 2：按本数据集的分化程度扫 τ，看"调到多少才有区分力"。

背景：
- 本图分化宽度只有 0.068（同场景 0.585 / 跨场景 0.518）；
- bge-m3 的**无关基线**在 0.40–0.55（`eval/probe_cosine_semantics.py`），
  所以 `τ=0.30` 在语义上等于"什么都不过滤"。
本脚本把 τ 从 0.30 扫到 0.60，在 E5 的 30 条（同句 × 3 目的）上只跑**检索**（不做叙事），
看四件事随 τ 怎么变：
  1. 目的过滤的**通过率**（真的开始筛了吗）
  2. 输出**节点数**（规模崩不崩）
  3. 同情景三意图的 **Jaccard**（意图分化是否变强）
  4. **gold facet recall**（分化是否对方向）

预期：τ 一提高，通过率与规模同时下降，而 recall 会在某处见顶后转跌——
那个峰值就是"本图能给的最好阈值"，同时它的绝对水平也说明这张图的上限在哪。

只跑检索（`story=False`），无 LLM 叙事开销。

用法：python -m eval.e5_tau_sweep
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from eval.e5_intent import _flatten, _jaccard  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

TAUS = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60)


def _run_tau(retriever, queries: List[Dict[str, Any]], tau: float) -> Dict[str, Any]:
    retriever.purpose_filter_threshold = tau
    per_q: Dict[str, Dict[str, Any]] = {}
    pass_rates: List[float] = []
    for q in queries:
        out = retriever.retrieve_with_story(
            q["text"], seed_k=5, story=False, with_response=False,
            purpose=list(q["purposes"]))
        ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
        gold = set(q["gold"])
        per_q[q["id"]] = {
            "ids": ids,
            "recall": (len(set(ids) & gold) / len(gold)) if gold else None,
            "filter": getattr(retriever, "_purpose_stats", []) or [],
        }
        # 通过率：各跳 in/out 的汇总（只看扩展轮，hop≥1）
        ins = sum(s["in"] for s in per_q[q["id"]]["filter"] if s["hop"] >= 1)
        outs = sum(s["out"] for s in per_q[q["id"]]["filter"] if s["hop"] >= 1)
        if ins:
            pass_rates.append(outs / ins)

    # 按 scenario 聚合 Jaccard
    groups: Dict[str, Dict[str, set]] = {}
    recs: List[float] = []
    sizes: List[int] = []
    for q in queries:
        r = per_q[q["id"]]
        groups.setdefault(q["scenario"], {})[q["intent"]] = set(r["ids"])
        sizes.append(len(r["ids"]))
        if r["recall"] is not None:
            recs.append(r["recall"])
    jacs = []
    for sc, g in groups.items():
        intents = sorted(g)
        for i, a in enumerate(intents):
            for b in intents[i + 1:]:
                jacs.append(_jaccard(g[a], g[b]))

    return {
        "tau": tau,
        "mean_nodes": round(float(np.mean(sizes)), 2),
        "nodes_min_max": [min(sizes), max(sizes)],
        "filter_pass_rate": round(float(np.mean(pass_rates)), 4) if pass_rates else None,
        "jaccard": round(float(np.mean(jacs)), 4) if jacs else None,
        "recall": round(float(np.mean(recs)), 4) if recs else None,
    }


def main() -> None:
    _load_dotenv(ROOT / ".env")
    with open(ROOT / "data/dba_eval/intent_queries.yaml", encoding="utf-8") as f:
        queries = _flatten(yaml.safe_load(f) or {})
    graph, retriever = build_retriever(
        str(ROOT / "data/dba_eval/comprehensive_gold.yaml"),
        vector_cache_key="comprehensive")
    base_tau = retriever.purpose_filter_threshold
    print(f"[sweep] queries={len(queries)} 默认 tau={base_tau} "
          f"emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)

    rows = []
    for tau in TAUS:
        r = _run_tau(retriever, queries, tau)
        rows.append(r)
        print(f"  τ={tau:.2f}  节点={r['mean_nodes']:5.1f}({r['nodes_min_max'][0]}-{r['nodes_min_max'][1]})"
              f"  过滤通过率={'-' if r['filter_pass_rate'] is None else format(r['filter_pass_rate'], '.1%')}"
              f"  Jaccard={r['jaccard']:.3f}  recall={r['recall']:.3f}", flush=True)

    retriever.purpose_filter_threshold = base_tau

    best = max((x for x in rows if x["recall"] is not None), key=lambda x: x["recall"])
    print(f"\n[sweep] recall 最高的 τ = {best['tau']:.2f}（recall={best['recall']:.3f}, "
          f"Jaccard={best['jaccard']:.3f}, 节点={best['mean_nodes']}）")
    base = next(x for x in rows if abs(x["tau"] - 0.30) < 1e-9)
    print(f"[sweep] 与 τ=0.30 比：recall {best['recall']-base['recall']:+.3f}，"
          f"Jaccard {best['jaccard']-base['jaccard']:+.3f}")

    p = ROOT / "eval/outputs/e5/tau_sweep.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"experiment": "E5 tau sweep", "queries": len(queries),
                 "taus": TAUS, "embedding": os.environ.get("EMBEDDING_MODEL"),
                 "note": "只检索不做叙事；通过率按扩展轮(hop>=1)的 in/out 汇总"},
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {p}")


if __name__ == "__main__":
    main()
