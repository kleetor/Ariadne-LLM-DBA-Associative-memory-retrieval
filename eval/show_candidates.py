# -*- coding: utf-8 -*-
"""查看 P vs Naive RAG 候选对比（按查询）"""
import json
import sys

p = json.load(open("eval/outputs/p_candidates.json", encoding="utf-8"))["results"]
r = json.load(open("eval/outputs/naive_rag_candidates.json", encoding="utf-8"))["results"]
p_by = {x["query_id"]: x for x in p}
r_by = {x["query_id"]: x for x in r}

QIDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["q01", "q07", "q13", "q17", "q25", "q31"]

for qid in QIDS:
    pv, rv = p_by[qid], r_by[qid]
    print("=" * 80)
    print(f"[{qid}] ({pv['category']}) {pv['query']}")
    print(f"  P目的: {pv['purpose']['purposes']}")
    print(f"  P候选({pv['total_candidates']}):")
    for c in pv["candidates"][:8]:
        print(f"    {c['id']:<6} {c['content'][:42]}  ({c['score']:.3f})")
    print(f"  NaiveRAG候选({rv['total_candidates']}):")
    for c in rv["candidates"][:8]:
        print(f"    {c['id']:<6} {c['content'][:42]}  ({c['score']:.3f})")
