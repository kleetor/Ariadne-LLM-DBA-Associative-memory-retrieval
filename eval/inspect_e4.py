# -*- coding: utf-8 -*-
"""产物调研（E4）：把「节点集合差异」与「judge 分数差」并置，判断到底哪一环没生效。

E4 的五条件分数打平（差 0.02–0.15，方向还不稳），有两种完全不同的解释：
  a) 机制真的没改变检索结果 → 分数打平是**正确**的；
  b) 机制显著改变了结果，但三维 rubric 分辨不出 → 分数打平是**指标不敏感**。
二者对论文的结论完全不同，必须用产物本身来区分。

做法：对 70 条 query 重跑 `full` 与 `no_purpose`（story=False，只取节点），
1) 算节点集合 Jaccard 与规模差；2) 与三家 judge 的逐题分数差做关联；
3) 按 Jaccard 高低分组，看"差异大的题"是否分数差也大。
4) 抽 3 条把 only-in-full 的节点正文打出来，人工可核。

用法：python -m eval.inspect_e4
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

JUDGES = {
    "qwen": "eval/outputs/e4/judge_purpose_qwen.json",
    "deepseek": "eval/outputs/e4/judge_purpose_deepseek.json",
    "gpt": "eval/outputs/e4/judge_purpose_gpt.json",
}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 产物调研：full vs no_purpose")
    ap.add_argument("--queries", default="data/dba_eval/comprehensive_queries.yaml")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--samples", type=int, default=3, help="打印几条差异样本")
    ap.add_argument("--out", default="eval/outputs/e4/inspect.json")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / args.queries, encoding="utf-8") as f:
        qs = [q for q in (yaml.safe_load(f) or {}).get("queries", [])
              if str(q.get("category") or "") != "abstention"]
    with open(ROOT / args.yaml, encoding="utf-8") as f:
        contents = {n["id"]: n.get("content", "")
                    for n in (yaml.safe_load(f) or {}).get("nodes", [])}

    graph, retriever = build_retriever(str(ROOT / args.yaml),
                                       vector_cache_key="comprehensive")

    # 逐题跑两个条件，记录节点 id
    per_q: Dict[str, Dict[str, Any]] = {}
    for q in qs:
        qid, text = str(q["id"]), str(q["text"])
        purposes = list(q.get("purposes") or [])
        full = retriever.retrieve_with_story(
            text, seed_k=5, story=False, with_response=False, purpose=purposes or None)
        blind = retriever.retrieve_with_story(
            text, seed_k=5, story=False, with_response=False, purpose_blind=True)
        f_ids = [n["id"] for n in (full.get("path") or {}).get("nodes") or []]
        b_ids = [n["id"] for n in (blind.get("path") or {}).get("nodes") or []]
        per_q[qid] = {
            "full": f_ids, "blind": b_ids,
            "jaccard": round(_jaccard(set(f_ids), set(b_ids)), 4),
            "n_full": len(f_ids), "n_blind": len(b_ids),
            "only_full": [i for i in f_ids if i not in set(b_ids)],
            "only_blind": [i for i in b_ids if i not in set(f_ids)],
        }
        print(f"  {qid} full={len(f_ids):3d} blind={len(b_ids):3d} "
              f"J={per_q[qid]['jaccard']:.3f}", flush=True)

    # 关联 judge 分数差。
    # 注意：`judge_comprehensive` 落盘时 label_map 是 None（它的 load_context 只保留
    # query/blocks），真正的「条件→匿名称」映射在 **ctx 文件**里，必须从那里读。
    ctx_map: Dict[str, Dict[str, str]] = {}
    ctx_path = ROOT / "eval/outputs/e4/context_e4_purpose.json"
    if ctx_path.exists():
        for row in json.load(open(ctx_path, encoding="utf-8"))["queries"]:
            if row.get("label_map"):
                ctx_map[row["query_id"]] = row["label_map"]

    scores: Dict[str, Dict[str, float]] = {}
    for name, p in JUDGES.items():
        if not (ROOT / p).exists():
            continue
        d = json.load(open(ROOT / p, encoding="utf-8"))
        for r in d["results"]:
            lm = r.get("label_map") or ctx_map.get(r["query_id"]) or {}
            inv = {v: k for k, v in lm.items()}
            sc = {inv.get(k, k): v["total"] for k, v in r["scores"].items()}
            if "full" in sc and "no_purpose" in sc:
                scores.setdefault(r["query_id"], {})[name] = sc["full"] - sc["no_purpose"]

    common = [qid for qid in per_q if qid in scores]
    print(f"\n=== 概览（n={len(common)}）===")
    jacs = [per_q[q]["jaccard"] for q in common]
    print(f"  节点 Jaccard: 均值={statistics.mean(jacs):.3f}  "
          f"中位={statistics.median(jacs):.3f}  "
          f"<0.5 的题={sum(1 for x in jacs if x < 0.5)}/{len(jacs)}")
    for name in ("qwen", "deepseek", "gpt"):
        ds = [scores[q][name] for q in common if name in scores[q]]
        if ds:
            print(f"  {name:9s} full−no_purpose: 均值={statistics.mean(ds):+.3f}  "
                  f"|差|>0.3 的题={sum(1 for x in ds if abs(x) > 0.3)}/{len(ds)}")

    # 按 Jaccard 分组：差异大的题，分数差是否也更大？
    print("\n=== 按节点差异分组（判断 judge 是否对机制差异不敏感）===")
    lo = [q for q in common if per_q[q]["jaccard"] < 0.5]
    hi = [q for q in common if per_q[q]["jaccard"] >= 0.5]
    for label, grp in (("差异大 J<0.5", lo), ("差异小 J>=0.5", hi)):
        if not grp:
            continue
        print(f"  {label}（n={len(grp)}）")
        for name in ("qwen", "deepseek", "gpt"):
            ds = [scores[q][name] for q in grp if name in scores[q]]
            if ds:
                print(f"    {name:9s} Δ分均值={statistics.mean(ds):+.3f}  "
                      f"|Δ|均值={statistics.mean(abs(x) for x in ds):.3f}")

    # 抽样：把 only-in-full 的正文打出来，人工可核
    print(f"\n=== 抽样 {args.samples} 条（差异最大的）===")
    top = sorted(common, key=lambda q: per_q[q]["jaccard"])[: args.samples]
    for qid in top:
        d = per_q[qid]
        qtext = next((str(q["text"]) for q in qs if str(q["id"]) == qid), "")
        print(f"\n--- {qid}  J={d['jaccard']:.3f}  full={d['n_full']} blind={d['n_blind']}")
        print(f"    query: {qtext}")
        print(f"    仅 full 有（{len(d['only_full'])} 条）:")
        for i in d["only_full"][:8]:
            print(f"      + [{i}] {contents.get(i, '')[:60]}")
        if d["only_blind"]:
            print(f"    仅 no_purpose 有（{len(d['only_blind'])} 条）:")
            for i in d["only_blind"][:4]:
                print(f"      - [{i}] {contents.get(i, '')[:60]}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"experiment": "E4 inspect", "n": len(common)},
        "per_query": {k: {kk: vv for kk, vv in v.items()} for k, v in per_q.items()},
        "score_diff": scores,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[inspect] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
