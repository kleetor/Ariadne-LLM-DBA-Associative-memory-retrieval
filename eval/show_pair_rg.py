# -*- coding: utf-8 -*-
"""查看 RAG vs Graphiti 纯选理由 + 汇总三组纯选结果"""
import json

rg = json.load(open("eval/outputs/story_pair_rg.json", encoding="utf-8"))
print("===== RAG vs Graphiti 汇总 =====")
print(rg["summary"])
print("\n===== Graphiti 获胜查询 =====")
for r in rg["results"]:
    if r["winner"] == "graphiti":
        print(f"  [{r['query_id']}] {r['reason']}")
print("\n===== tie =====")
for r in rg["results"]:
    if r["winner"] == "tie":
        print(f"  [{r['query_id']}] {r['reason']}")

print("\n===== 三组纯选胜者矩阵 =====")
pg = json.load(open("eval/outputs/story_pair_pg.json", encoding="utf-8"))
v3 = json.load(open("eval/outputs/story_pair_v3.json", encoding="utf-8"))
by_q = {}
for name, data in [("pg", pg), ("v3", v3), ("rg", rg)]:
    for r in data["results"]:
        by_q.setdefault(r["query_id"], {})[name] = r["winner"]
print(f"{'qid':<5}{'PvsRAG':<8}{'PvsG':<8}{'RvsG':<8}")
for qid in sorted(by_q.keys()):
    m = by_q[qid]
    w = lambda k: ("P" if m.get(k) == "p" or m.get(k) == "P" else
                   ("RAG" if "RAG" in str(m.get(k)) or m.get(k) == "rag" else
                    ("G" if "graphiti" in str(m.get(k)) else ("tie" if m.get(k) == "tie" else "-"))))
    print(f"{qid:<5}{w('v3'):<8}{w('pg'):<8}{w('rg'):<8}")
