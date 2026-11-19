# -*- coding: utf-8 -*-
"""v3 CROSS_OUT 跨场景率计算"""
import json
import yaml

p = {r["query_id"]: r for r in json.load(open("eval/outputs/p_candidates_v3.json", encoding="utf-8"))["results"]}
r = {x["query_id"]: x for x in json.load(open("eval/outputs/naive_rag_candidates_v3.json", encoding="utf-8"))["results"]}
qs = {q["id"]: q for q in yaml.safe_load(open("data/LiFEMem_queries_v3.yaml", encoding="utf-8"))["queries"]}

dom = {"work_pressure": {"w"}, "health": {"he"}, "emotion": {"e"}, "nostalgia": {"h", "c"},
       "relationship": {"h"}, "life": {"w", "e", "f", "sh", "t"}, "learning": {"ln"},
       "social": {"so"}, "future": {"f", "w", "ln"}, "abstention": set()}

PFS = ["he", "ln", "sh", "so", "w", "e", "h", "c", "f", "t", "n"]


def pre(nid):
    for pf in PFS:
        if nid.startswith(pf):
            return pf
    return "?"


def cross_out(rows, scen):
    ids = [c["id"] for c in rows["candidates"]]
    home = dom.get(scen, set())
    if not home:
        return None, len(ids)
    c = sum(1 for i in ids if pre(i) not in home)
    return c / len(ids), len(ids)


print("查询 | 场景 | P跨场率 | RAG跨场率 | P-N | RAG-N")
sums = {"p": [0, 0], "r": [0, 0]}
for qid in p:
    sc = qs[qid]["scenario"]
    po, pn = cross_out(p[qid], sc)
    ro, rn = cross_out(r[qid], sc)
    if po is None:
        continue
    sums["p"][0] += po * pn
    sums["p"][1] += pn
    sums["r"][0] += ro * rn
    sums["r"][1] += rn
    print(f"{qid} | {sc} | {po:.3f} | {ro:.3f} | {pn} | {rn}")
print(f"\n加权均值: P={sums['p'][0]/sums['p'][1]:.3f}  RAG={sums['r'][0]/sums['r'][1]:.3f}")
