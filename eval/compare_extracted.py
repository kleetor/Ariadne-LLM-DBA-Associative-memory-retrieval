# -*- coding: utf-8 -*-
"""前置管理产物 vs 原文 检索对比（英文子集）：expected 覆盖、集合、排序质量"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "outputs"


def load(rel):
    return json.loads((OUT / rel).read_text(encoding="utf-8"))["results"]


def main() -> None:
    raw_q = {r["query_id"]: r for r in load("mem0_candidates_en_qdrant.json")}    # 原文+BM25
    raw_f = {r["query_id"]: r for r in load("mem0_candidates_en_faiss.json")}     # 原文纯向量
    ex = {r["query_id"]: r for r in load("mem0_candidates_en_extracted.json")}    # 抽取产物+BM25

    print(f"{'qid':<6}{'exp':<4}{'原文qdrant':<10}{'原文faiss':<10}{'抽取产物':<10}")
    agg = {"raw_q": 0, "raw_f": 0, "ex": 0, "total": 0}
    for qid in ex:
        exp = set(ex[qid]["expected"])
        agg["total"] += len(exp)
        ids = {
            "raw_q": {c["id"] for c in raw_q[qid]["candidates"]},
            "raw_f": {c["id"] for c in raw_f[qid]["candidates"]},
            "ex": {c["id"] for c in ex[qid]["candidates"]},
        }
        hits = {k: len(exp & v) for k, v in ids.items()}
        for k in agg:
            if k != "total":
                agg[k] += hits[k]
        print(f"{qid:<6}{len(exp):<4}{hits['raw_q']:<10}{hits['raw_f']:<10}{hits['ex']:<10}")

    print(f"\nexpected 覆盖汇总:")
    print(f"  原文 qdrant: {agg['raw_q']}/{agg['total']} = {agg['raw_q']/agg['total']:.3f}")
    print(f"  原文 faiss : {agg['raw_f']}/{agg['total']} = {agg['raw_f']/agg['total']:.3f}")
    print(f"  抽取产物   : {agg['ex']}/{agg['total']} = {agg['ex']/agg['total']:.3f}")

    # 排序质量（相似度相关性）—— 与 infer=False 版同基准
    # 简化：直接对比集合差异
    n_diff_set = 0
    for qid in ex:
        if {c["id"] for c in raw_q[qid]["candidates"]} != {c["id"] for c in ex[qid]["candidates"]}:
            n_diff_set += 1
    print(f"\n抽取产物 vs 原文qdrant 的 top-20 集合不同: {n_diff_set}/10")


if __name__ == "__main__":
    main()
