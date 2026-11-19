# -*- coding: utf-8 -*-
"""T6 结果汇总：P vs Naive RAG 首版对照表"""
import json

d = json.load(open("eval/outputs/story_scores_full.json", encoding="utf-8"))
print("systems:", d["meta"]["systems"], "| queries:", d["meta"]["queries"], "| judge:", d["meta"]["judge"])

rows = d["results"]
# 非弃权类
non_abs = [r for r in rows if r["category"] != "abstention"]
abs_rows = [r for r in rows if r["category"] == "abstention"]

print(f"\n===== 故事质量五维均值（{len(non_abs)} 条非弃权查询，总分=五维平均）=====")
dims = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter", "total"]
sys_names = d["meta"]["systems"]
agg = {s: {x: [] for x in dims} for s in sys_names}
for r in non_abs:
    for s in sys_names:
        sc = r["systems"].get(s, {}).get("scores")
        if sc:
            for x in dims:
                agg[s][x].append(sc[x])
print(f"{'系统':<10}" + "".join(f"{x:>12}" for x in dims))
for s in sys_names:
    line = f"{s:<10}"
    for x in dims:
        v = agg[s][x]
        line += f"{sum(v)/len(v):>12.2f}" if v else f"{'-':>12}"
    print(line)

print(f"\n===== 分能力（类别）=====")
cats = {}
for r in non_abs:
    cats.setdefault(r["category"], []).append(r)
for cat, rs in sorted(cats.items()):
    print(f"  [{cat}] ({len(rs)}条)")
    for s in sys_names:
        totals = [r["systems"][s]["scores"]["total"] for r in rs if s in r["systems"] and r["systems"][s].get("scores")]
        if totals:
            print(f"    {s:<10} total均值={sum(totals)/len(totals):.2f}")

print(f"\n===== 弃权类（{len(abs_rows)} 条，判定：refused=true 为正确拒答）=====")
for r in abs_rows:
    line = f"  {r['query_id']} {r['query'][:25]:<28}"
    for s in sys_names:
        v = r["systems"].get(s, {}).get("abstention", {})
        line += f" | {s}: {'OK-拒答' if v.get('refused') else 'XX-过度检索'}({v.get('score', '-')})"
    print(line)
