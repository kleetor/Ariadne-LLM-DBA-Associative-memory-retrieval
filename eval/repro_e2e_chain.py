# -*- coding: utf-8 -*-
"""复现核对：存档 e2e 的 Ariadne 链路产物 vs 当前代码重跑。

用途：判断 2026-09-01 存档结果（Ariadne 4.74 / Graphiti 3.63 / Mem0 2.99）与当前
开发分支的差异，是"链路被改坏了"还是"判分/数据本身稳定"。

对比项（逐 query）：
  - archived path 节点数 vs 当前 path 节点数
  - 节点集合 Jaccard、是否子集/超集
  - 目的推断结果（存档 purpose vs 当前 purpose）
  - total_candidates（峰值候选数）

用法:
    python -m eval.repro_e2e_chain --out eval/outputs/e3/repro_chain.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

GOLD = "data/dba_eval/comprehensive_gold.yaml"
QUERIES = "data/dba_eval/comprehensive_queries.yaml"
ARCHIVE = "eval/outputs/p_candidates_comprehensive.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="复现核对：存档 Ariadne 链路 vs 当前代码")
    parser.add_argument("--out", default="eval/outputs/e3/repro_chain.json")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / QUERIES, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    if args.limit:
        queries = queries[: args.limit]
    with open(ROOT / ARCHIVE, encoding="utf-8") as f:
        arch = {r["query_id"]: r for r in json.load(f)["results"]}

    _, retriever = build_retriever(str(ROOT / GOLD), vector_cache_key="comprehensive")
    print(f"[repro] queries={len(queries)}", flush=True)

    rows = []
    for q in queries:
        qid = str(q["id"])
        a = arch.get(qid) or {}
        a_ids = [n["id"] for n in (a.get("path") or {}).get("nodes", [])]
        a_purpose = str((a.get("purpose") or {}).get("purposes"))

        out = retriever.retrieve_with_story(
            str(q["text"]), story=False, with_response=False)
        c_ids = [n["id"] for n in (out.get("path") or {}).get("nodes", [])]
        c_purpose = str((out.get("purpose") or {}).get("purposes"))

        sa, sc = set(a_ids), set(c_ids)
        jac = (len(sa & sc) / len(sa | sc)) if (sa | sc) else 1.0
        rows.append({
            "query_id": qid,
            "category": q.get("category", ""),
            "arch_nodes": len(a_ids),
            "cur_nodes": len(c_ids),
            "jaccard": round(jac, 3),
            "cur_equals_arch": sa == sc,
            "arch_subset_cur": sa.issubset(sc),
            "cur_subset_arch": sc.issubset(sa),
            "arch_candidates": a.get("total_candidates"),
            "cur_candidates": out.get("total_candidates"),
            "arch_purposes": a_purpose,
            "cur_purposes": c_purpose,
            "same_purposes": a_purpose == c_purpose,
        })
        print(f"  {qid} arch={len(a_ids)} cur={len(c_ids)} jac={jac:.2f} "
              f"cand={a.get('total_candidates')}->{out.get('total_candidates')} "
              f"same_purpose={a_purpose == c_purpose}", flush=True)

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n if n else 0.0
    summary = {
        "n": n,
        "mean_arch_nodes": round(mean("arch_nodes"), 1),
        "mean_cur_nodes": round(mean("cur_nodes"), 1),
        "mean_jaccard": round(mean("jaccard"), 3),
        "identical_sets": sum(1 for r in rows if r["cur_equals_arch"]),
        "arch_subset_cur": sum(1 for r in rows if r["arch_subset_cur"]),
        "cur_subset_arch": sum(1 for r in rows if r["cur_subset_arch"]),
        "same_purposes": sum(1 for r in rows if r["same_purposes"]),
    }
    print("\n[summary]", json.dumps(summary, ensure_ascii=False))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[repro] wrote {out_path}")


if __name__ == "__main__":
    main()
