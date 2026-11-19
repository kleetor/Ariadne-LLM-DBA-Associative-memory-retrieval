# -*- coding: utf-8 -*-
"""v3 全量数据报告：逐查询候选 + 故事全文 + 评分"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import yaml

scores = json.load(open("eval/outputs/story_scores_v3.json", encoding="utf-8"))["results"]
p_cand = {r["query_id"]: r for r in json.load(open("eval/outputs/p_candidates_v3.json", encoding="utf-8"))["results"]}
rag_cand = {r["query_id"]: r for r in json.load(open("eval/outputs/naive_rag_candidates_v3.json", encoding="utf-8"))["results"]}
stories = json.load(open("eval/outputs/story_texts_v3.json", encoding="utf-8"))
qs = {q["id"]: q for q in yaml.safe_load(open("data/LiFEMem_queries_v3.yaml", encoding="utf-8"))["queries"]}
nodes = {n["id"]: n["content"] for n in yaml.safe_load(open("data/LiFEMem.yaml", encoding="utf-8"))["nodes"]}

CAT_NAMES = {"work_pressure": "工作压力", "health": "健康", "emotion": "情绪", "nostalgia": "怀旧",
             "relationship": "感情", "life": "生活", "learning": "学习", "social": "社交",
             "future": "未来", "abstention": "弃权"}
DIMS = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter"]

L = []
A = L.append
A("# LiFEMem v3 全量数据报告（陈述触发式）")
A("")
A("> 35 条 = 30 陈述 + 5 弃权 ｜ 逐查询：目的推断 / P&RAG 候选 / 故事全文 / 评分")
A("")

for sc in ["work_pressure", "health", "emotion", "nostalgia", "relationship", "life",
           "learning", "social", "future", "abstention"]:
    sub = [r for r in scores if r["category"] == sc]
    if not sub:
        continue
    A(f"## {CAT_NAMES[sc]}（{len(sub)}）")
    A("")
    for row in sub:
        qid = row["query_id"]
        q = qs[qid]
        exp = q.get("expected", [])
        A(f"### {qid} · {row['query']}")
        A("")
        A(f"- **目的推断**：{row['purpose']}")
        A(f"- **expected**：{exp if exp else '（无固定答案）'}")
        for e in exp:
            A(f"  - `{e}`：{nodes.get(e, '?')}")
        pc = p_cand[qid]["candidates"]
        rc = rag_cand[qid]["candidates"]
        for tag, cands in (("P 候选 top10", pc[:10]), ("RAG 候选 top10", rc[:10])):
            A("")
            A(f"**{tag}**（{len(pc)}/{len(rc)} 条）")
            A("")
            for c in cands:
                mark = "←expected" if c["id"] in exp else ""
                A(f"- `{c['id']}` {c['content']}{('  **' + mark + '**') if mark else ''}")
        for tag, key in (("P 故事", "P"), ("RAG 故事", "Naive RAG")):
            st = stories[qid]["stories"][key]["story"]
            sysd = row["systems"]["p" if tag == "P 故事" else "naive_rag"]
            if "scores" in sysd:
                sc5 = sysd["scores"]
                cv = sysd["cross_valid"]
                A("")
                A(f"**{tag}（{len(st)}字）** · " + " ".join(f"{d}={sc5[d]}" for d in DIMS) + f" total={sc5['total']} · CV={cv}")
                A("")
                A(f"> {st}")
            else:
                A("")
                A(f"**{tag}（{len(st)}字）** · 弃权判定 refused={sysd['abstention']['refused']}")
                A("")
                A(f"> {st}")
        A("")
        A("---")
        A("")

out = ROOT / "Plan/0818——LiFEMem_v3全量数据报告.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"[ok] {out}（{len(L)} 行）")
