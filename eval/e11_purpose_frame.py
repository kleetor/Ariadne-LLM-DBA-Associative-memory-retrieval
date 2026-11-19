# -*- coding: utf-8 -*-
"""E11：目的表示优化——结构化目的帧（PurposeFrame）能否让「同一句话 × 不同目的」真正分开。

E5 的结论是**目的表示失败**，不是检索机制失败：三种意图召回的节点类型分布 TVD 仅
0.042–0.066（倾诉只召回 10.6% 的情绪节点）。原因在表示——把三条目的短语拼成一个
向量，只留下话题、丢了检索意图。§7.6 给的出路是结构化：`{节点类型} × {时间窗} × {主体}`。

本脚本只验证**表示形态**这一步：把目的编译成帧（类型分布 + 关系分布），在检索的
种子重排 / 逐跳过滤 / 组合排序三处生效，其余机制（寻峰、跳转轴、StoryRank）完全不动。

对照（同预算、同 τ、同 seed_k）：
| 臂 | 目的表示 |
|---|---|
| `flat`  | 现状：三条目的拼接后嵌入（E5 的 par 臂） |
| `frame` | 目的帧：`w_emb·余弦 + w_type·类型亲和 + w_rel·关系亲和` |

指标：
1. **同情景三意图的节点类型分布 TVD**（主指标；E5 基线 0.042–0.066，验收线 0.2）；
2. gold facet recall（分化不能以砍掉正确证据为代价）；
3. 目的过滤的实际通过率（E5 里 τ=0.30 从未真正筛过任何候选）。

用法：
    python -m eval.e11_purpose_frame --scenarios a01 --taus 0.30 0.45
    python -m eval.e11_purpose_frame
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
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

TYPES = ("status", "reason", "action", "thing", "person", "emotion")
ARMS = ("flat", "frame")


def _type_of(graph, nid: str) -> str:
    nt = graph.graph.nodes[nid].get("node_type") if nid in graph.graph else None
    return str(getattr(nt, "value", nt) or "?").lower()


def _tvd(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in TYPES) / 2


def _dist(counter: Counter) -> Dict[str, float]:
    n = sum(counter.values()) or 1
    return {t: counter.get(t, 0) / n for t in TYPES}


def _run(retriever, graph, q: Dict[str, Any], arm: str, args) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"seed_k": args.seed_k, "max_peak_nodes": args.max_nodes}
    if arm == "frame":
        kw["purpose_frame"] = compile_frame(q["purposes"])
        kw["frame_weights"] = tuple(args.weights)
        if args.anchor:
            kw["frame_anchor"] = float(args.anchor)
    res = retriever.retrieve(q["text"], purpose=list(q["purposes"]), **kw)
    ids = [m[0] for m in res.get("peak_memories", [])]
    gold = set(q["gold"])
    passes = res.get("purpose_pass") or []
    n_in = sum(p["in"] for p in passes)
    n_out = sum(p["out"] for p in passes)
    frame = compile_frame(q["purposes"]) if arm == "frame" else None
    return {
        "ids": ids,
        "types": Counter(_type_of(graph, i) for i in ids),
        "recall": (len(set(ids) & gold) / len(gold)) if gold else None,
        "n_nodes": len(ids),
        "filter_in": n_in,
        "filter_out": n_out,
        "frame": frame.name if frame else "flat",
        "frame_desc": frame.describe() if frame else "",
    }


def _scenario_tvd(rows: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    by_sc: Dict[str, Dict[str, Counter]] = {}
    for r in rows:
        by_sc.setdefault(r["scenario"], {})[r["intent"]] = r[arm]["types"]
    out = {}
    for sc, intents in by_sc.items():
        keys = sorted(intents)
        if len(keys) < 2:
            continue
        dists = {k: _dist(intents[k]) for k in keys}
        pairs = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                pairs[f"{a}|{b}"] = round(_tvd(dists[a], dists[b]), 4)
        out[sc] = {"mean": round(float(np.mean(list(pairs.values()))), 4), "pairs": pairs,
                   "dist": {k: {t: round(v, 3) for t, v in d.items()} for k, d in dists.items()}}
    return out


def _pooled_tvd(rows: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    """与 inspect_e5 同口径：跨场景把同意图的节点**汇总**后比三意图的类型分布。

    这个口径比逐情景 TVD 宽松（汇总会抹平单情景波动），E5 报告的 0.042–0.066 用的就是它；
    两个口径都报，避免用口径差异夸大收益。
    """
    pool: Dict[str, Counter] = {}
    for r in rows:
        pool.setdefault(r["intent"], Counter()).update(r[arm]["types"])
    dists = {k: _dist(v) for k, v in pool.items()}
    keys = sorted(dists)
    pairs = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs[f"{a}|{b}"] = round(_tvd(dists[a], dists[b]), 4)
    return {"mean": round(float(np.mean(list(pairs.values()))), 4) if pairs else None,
            "pairs": pairs,
            "dist": {k: {t: round(v, 3) for t, v in d.items()} for k, d in dists.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description="E11 目的帧（结构化目的表示）")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--queries", default="data/dba_eval/intent_queries.yaml")
    ap.add_argument("--out", default="eval/outputs/e11/purpose_frame.json")
    ap.add_argument("--scenarios", nargs="*", default=None, help="只跑这些场景前缀，如 a01")
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--max-nodes", type=int, default=10, help="两臂同预算（节点上限）")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.30, 0.45])
    ap.add_argument("--weights", nargs=3, type=float, default=[0.55, 0.35, 0.10],
                    help="w_emb w_type w_rel")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="帧的 {主体} 维权重（0=关；0.4=话题相似度 60%% 目的 + 40%% query）")
    ap.add_argument("--vector-cache-key", default="comprehensive")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / args.queries, encoding="utf-8") as f:
        queries = _flatten(yaml.safe_load(f) or {})
    if args.scenarios:
        pre = tuple(args.scenarios)
        queries = [q for q in queries if str(q["scenario"]).startswith(pre)]
    if not queries:
        print("[E11] 没有可跑的条目")
        return 1

    graph, retriever = build_retriever(
        str(ROOT / args.yaml), llm_model=args.llm_model,
        vector_cache_key=args.vector_cache_key)

    # 目的向量缓存：同一目的文本在两臂、多 τ 下重复出现，避免反复打嵌入 API
    orig_embed = retriever.vector_store._embed
    memo: Dict[str, Any] = {}

    def _cached(text: str):
        if text not in memo:
            memo[text] = orig_embed(text)
        return memo[text]

    retriever.vector_store._embed = _cached

    print(f"[E11] queries={len(queries)} nodes={graph.node_count} "
          f"emb={os.environ.get('EMBEDDING_MODEL')} max_nodes={args.max_nodes} "
          f"weights={tuple(args.weights)}", flush=True)

    runs: Dict[str, Any] = {}
    for tau in args.taus:
        retriever.purpose_filter_threshold = float(tau)
        rows: List[Dict[str, Any]] = []
        for q in queries:
            rec = {"id": q["id"], "scenario": q["scenario"], "intent": q["intent"], "text": q["text"]}
            for arm in ARMS:
                try:
                    rec[arm] = _run(retriever, graph, q, arm, args)
                except Exception as exc:  # 单条失败不拖垮整批
                    rec[arm] = {"ids": [], "types": Counter(), "recall": None, "n_nodes": 0,
                                "filter_in": 0, "filter_out": 0,
                                "frame": arm, "frame_desc": ""}
                    print(f"  !! {q['id']}/{arm} {type(exc).__name__}: {exc}", flush=True)
            rows.append(rec)

        tvd = {arm: _scenario_tvd(rows, arm) for arm in ARMS}
        pooled = {arm: _pooled_tvd(rows, arm) for arm in ARMS}
        summary = {}
        for arm in ARMS:
            rec = [r[arm]["recall"] for r in rows if r[arm]["recall"] is not None]
            tv = [v["mean"] for v in tvd[arm].values()]
            fin = sum(r[arm]["filter_in"] for r in rows)
            fout = sum(r[arm]["filter_out"] for r in rows)
            summary[arm] = {
                "tvd_mean": round(float(np.mean(tv)), 4) if tv else None,
                "tvd_pooled": pooled[arm]["mean"],
                "recall_mean": round(float(np.mean(rec)), 4) if rec else None,
                "n_nodes_mean": round(float(np.mean([r[arm]["n_nodes"] for r in rows])), 2),
                "filter_keep_rate": round(fout / fin, 4) if fin else None,
                "filter_events": fin,
            }
        runs[str(tau)] = {"rows": rows, "tvd": tvd, "pooled": pooled, "summary": summary}

        print(f"\n=== τ={tau} ===")
        print(f"{'scenario':16s} {'TVD_flat':>9s} {'TVD_frame':>10s} "
              f"{'R_flat':>7s} {'R_frame':>8s}")
        for sc in sorted(tvd["flat"]):
            rf = [r["flat"]["recall"] for r in rows if r["scenario"] == sc and r["flat"]["recall"] is not None]
            rm = [r["frame"]["recall"] for r in rows if r["scenario"] == sc and r["frame"]["recall"] is not None]
            print(f"{sc:16s} {tvd['flat'][sc]['mean']:9.3f} {tvd['frame'][sc]['mean']:10.3f} "
                  f"{np.mean(rf) if rf else float('nan'):7.3f} "
                  f"{np.mean(rm) if rm else float('nan'):8.3f}")
        s = summary
        print(f"{'平均':16s} {s['flat']['tvd_mean']:9.3f} {s['frame']['tvd_mean']:10.3f} "
              f"{s['flat']['recall_mean']:7.3f} {s['frame']['recall_mean']:8.3f}")
        print(f"  通过率 flat={s['flat']['filter_keep_rate']} frame={s['frame']['filter_keep_rate']}"
              f"  节点数 flat={s['flat']['n_nodes_mean']} frame={s['frame']['n_nodes_mean']}")
        print(f"  汇总口径 TVD（同 inspect_e5）：flat={s['flat']['tvd_pooled']} "
              f"frame={s['frame']['tvd_pooled']}（E5 基线 0.042–0.066）")
        print("  帧编译：" + " | ".join(
            f"{r['intent']}={r['frame']['frame_desc'] or r['flat']['frame_desc']}"
            for r in rows[:3]))

    # 逐条明细（保留 artifacts，便于抽样复核）
    payload = {
        "meta": {
            "experiment": "E11",
            "design": "struct purpose frame vs flat purpose embedding; same budget/tau/seed_k",
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "weights": list(args.weights),
            "anchor": args.anchor,
            "max_nodes": args.max_nodes,
            "seed_k": args.seed_k,
            "taus": list(args.taus),
            "baseline_e5_tvd": [0.042, 0.066],
            "acceptance": "TVD > 0.2",
        },
        "runs": {
            tau: {
                "summary": v["summary"],
                "tvd": v["tvd"],
                "pooled": v["pooled"],
                "rows": [{**{k: r[k] for k in ("id", "scenario", "intent", "text")},
                          **{arm: {kk: (dict(vv) if isinstance(vv, Counter) else vv)
                                   for kk, vv in r[arm].items()} for arm in ARMS}}
                         for r in v["rows"]],
            } for tau, v in runs.items()
        },
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E11] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
