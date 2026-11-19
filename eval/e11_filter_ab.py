# -*- coding: utf-8 -*-
"""E11-AB：同一目的帧下，「无过滤」vs「加过滤」的净效应。

背景：τ=0.30 在主图上等于不过滤（见计划 §8.8），因此 E11 在 τ=0.30 的收益来自
frame 打分，而不是余弦过滤。本脚本把 frame 固定，只改 τ，回答：

1. 过滤到底删掉了什么：是无关节点，还是 gold 事实？
2. 在**同等输出条数**下，过滤后的集合是否比「不过滤 + 按分数截断」更好？
   （如果只是把尾部截掉，过滤就是预算，不是选择。）
3. 过滤是否改善类型分化（TVD）与跨主体混入（M/H 图）？

口径：
- 两臂都是 frame：`purpose_frame` + `frame_weights` 不变，只改 `purpose_filter_threshold`；
- τ=0 为「无过滤」，`max_peak_nodes=0` 不截断；τ>0 为「frame+filter」；
- 等规模对照：取 τ>0 的节点数 k，用 τ=0 的前 k 个节点（peak_memories 已按分数排序）作对照。

用法：
    python -m eval.e11_filter_ab --yaml data/dba_eval/comprehensive_gold.yaml --cache comprehensive
    python -m eval.e11_filter_ab --yaml data/dba_eval/e10_M.yaml --cache e10_M --out eval/outputs/e11/filter_ab_M.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from dba_pipeline.core.purpose_frame import compile_frame  # noqa: E402
from eval.e5_intent import _flatten  # noqa: E402
from eval.e11_purpose_frame import TYPES, _dist, _scenario_tvd, _tvd, _type_of  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

SUBJECT_A = ("u", "a", "t")


def _subject(nid: str) -> str:
    c = str(nid)[0]
    return "A" if c in SUBJECT_A else c.upper()


def _run_frame(retriever, graph, q: Dict[str, Any], args, tau: float) -> Dict[str, Any]:
    retriever.purpose_filter_threshold = float(tau)
    kw: Dict[str, Any] = {
        "seed_k": args.seed_k,
        "max_peak_nodes": args.max_nodes,  # 0=不截断；>0 用于与 E11 的 max_nodes=10 对齐
        "purpose_frame": compile_frame(q["purposes"]),
        "frame_weights": tuple(args.weights),
    }
    if args.anchor:
        kw["frame_anchor"] = float(args.anchor)
    res = retriever.retrieve(q["text"], purpose=list(q["purposes"]), **kw)
    ids = [m[0] for m in (res.get("peak_memories") or [])]
    passes = res.get("purpose_pass") or []
    return {
        "ids": ids,
        "n": len(ids),
        "types": Counter(_type_of(graph, i) for i in ids),
        "recall": (len(set(ids) & set(q["gold"])) / len(q["gold"])) if q.get("gold") else None,
        "contam": (sum(1 for i in ids if _subject(i) != "A") / len(ids)) if ids else 0.0,
        "filter_in": sum(p.get("in", 0) for p in passes),
        "filter_out": sum(p.get("out", 0) for p in passes),
    }


def _pooled_tvd(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pool: Dict[str, Counter] = {}
    for r in rows:
        pool.setdefault(r["intent"], Counter()).update(r["types"])
    dists = {k: _dist(v) for k, v in pool.items()}
    keys = sorted(dists)
    pairs = {f"{a}|{b}": round(_tvd(dists[a], dists[b]), 4)
             for i, a in enumerate(keys) for b in keys[i + 1:]}
    return {"mean": round(float(np.mean(list(pairs.values()))), 4) if pairs else None,
            "pairs": pairs,
            "dist": {k: {t: round(v, 3) for t, v in d.items()} for k, d in dists.items()}}


def _scenario_mean(rows: List[Dict[str, Any]]) -> float | None:
    sc_rows = [{"scenario": r["scenario"], "intent": r["intent"],
                "frame": {"types": r["types"]}} for r in rows]
    d = _scenario_tvd(sc_rows, "frame")
    vals = [v["mean"] for v in d.values()]
    return round(float(np.mean(vals)), 4) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description="E11-AB：frame-only vs frame+filter")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--queries", default="data/dba_eval/intent_queries.yaml")
    ap.add_argument("--cache", default="comprehensive", help="向量缓存 key")
    ap.add_argument("--out", default="eval/outputs/e11/filter_ab_L.json")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.0, 0.30, 0.45, 0.55])
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="峰值带输出上限；0=不截断（默认，用于等规模分析）")
    ap.add_argument("--weights", nargs=3, type=float, default=[0.55, 0.35, 0.10])
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / args.queries, encoding="utf-8") as f:
        queries = _flatten(yaml.safe_load(f) or {})
    if not queries:
        raise SystemExit("[E11-AB] 没有可跑的条目")

    graph, retriever = build_retriever(
        str(ROOT / args.yaml), llm_model=args.llm_model, vector_cache_key=args.cache)
    print(f"[E11-AB] graph={args.yaml} nodes={graph.node_count} queries={len(queries)} "
          f"cache={args.cache} emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)

    # 目的文本缓存，避免反复打嵌入 API
    orig_embed = retriever.vector_store._embed
    memo: Dict[str, Any] = {}

    def _cached(text: str):
        if text not in memo:
            memo[text] = orig_embed(text)
        return memo[text]

    retriever.vector_store._embed = _cached

    # 1) 各 τ 的原始输出
    per_tau: Dict[str, List[Dict[str, Any]]] = {str(t): [] for t in args.taus}
    for q in queries:
        for tau in args.taus:
            rec = {"id": q["id"], "scenario": q["scenario"], "intent": q["intent"]}
            rec.update(_run_frame(retriever, graph, q, args, tau))
            per_tau[str(tau)].append(rec)

    base = per_tau[str(0.0)]
    gold_by_id = {q["id"]: set(q.get("gold") or []) for q in queries}
    summary: Dict[str, Any] = {}
    for tau in args.taus:
        key = str(tau)
        rows = per_tau[key]
        recalls = [r["recall"] for r in rows if r["recall"] is not None]
        contams = [r["contam"] for r in rows]
        ns = [r["n"] for r in rows]
        fin = sum(r["filter_in"] for r in rows)
        fout = sum(r["filter_out"] for r in rows)

        # 等规模对照：τ=0 的前 k 个（peak_memories 已按分数排序）
        eq_recall, eq_contam, jac, gold_removed, non_gold_removed = [], [], [], [], []
        trunc_rows = []
        for r0, rt in zip(base, rows):
            k = rt["n"]
            trunc = r0["ids"][:k] if k else []
            gold = gold_by_id.get(rt["id"], set())
            if gold:
                eq_recall.append(len(set(trunc) & gold) / len(gold))
            eq_contam.append((sum(1 for i in trunc if _subject(i) != "A") / len(trunc))
                             if trunc else 0.0)
            a, b = set(trunc), set(rt["ids"])
            jac.append(len(a & b) / len(a | b) if (a | b) else 1.0)
            removed = set(r0["ids"]) - set(rt["ids"])
            if gold:
                gold_removed.append(len(removed & gold) / len(gold))
            non_gold_removed.append(len(removed - gold))
            trunc_rows.append({"scenario": rt["scenario"], "intent": rt["intent"],
                               "types": Counter(_type_of(graph, i) for i in trunc)})

        tv = _pooled_tvd(rows)
        tv_eq = _pooled_tvd(trunc_rows)
        summary[key] = {
            "n_mean": round(float(np.mean(ns)), 2),
            "keep_rate": round(fout / fin, 4) if fin else None,
            "filter_events": fin,
            "recall_natural": round(float(np.mean(recalls)), 4) if recalls else None,
            "recall_equal_size": round(float(np.mean(eq_recall)), 4) if eq_recall else None,
            "contam_natural": round(float(np.mean(contams)), 4),
            "contam_equal_size": round(float(np.mean(eq_contam)), 4),
            "tvd_pooled": tv["mean"],
            "tvd_equal_size": tv_eq["mean"],
            "tvd_scenario": _scenario_mean(rows),
            "tvd_scenario_equal_size": _scenario_mean(trunc_rows),
            "jaccard_vs_trunc": round(float(np.mean(jac)), 4) if jac else None,
            "gold_removed_rate": round(float(np.mean(gold_removed)), 4) if gold_removed else None,
            "non_gold_removed_mean": round(float(np.mean(non_gold_removed)), 2),
        }

    print(f"\n{'τ':>5s} {'n':>6s} {'keep':>6s} {'R自然':>7s} {'R等规模':>8s} "
          f"{'混入自然':>8s} {'混入等规模':>9s} {'TVD自然':>7s} {'TVD等规模':>8s} "
          f"{'gold被删':>8s} {'非gold删':>8s}")
    for tau in args.taus:
        s = summary[str(tau)]
        print(f"{tau:5.2f} {s['n_mean']:6.1f} {str(s['keep_rate']):>6s} "
              f"{str(s['recall_natural']):>7s} {str(s['recall_equal_size']):>8s} "
              f"{s['contam_natural']:8.3f} {s['contam_equal_size']:9.3f} "
              f"{str(s['tvd_pooled']):>7s} {str(s['tvd_equal_size']):>8s} "
              f"{str(s['gold_removed_rate']):>8s} {s['non_gold_removed_mean']:8.2f}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {
            "experiment": "E11-AB",
            "yaml": args.yaml,
            "cache": args.cache,
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "weights": list(args.weights),
            "anchor": args.anchor,
            "seed_k": args.seed_k,
            "taus": list(args.taus),
            "note": ("frame 不变，只改 purpose_filter_threshold；等规模 = τ=0 的前 k 个节点。"
                     "gold_removed_rate = 被过滤集合里属于 gold 的比例；非 gold 删 = 被删的无关节点数。"),
        },
        "summary": summary,
        "rows": per_tau,
    }, ensure_ascii=False, indent=2, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)),
        encoding="utf-8")
    print(f"[E11-AB] wrote {p}")


if __name__ == "__main__":
    main()
