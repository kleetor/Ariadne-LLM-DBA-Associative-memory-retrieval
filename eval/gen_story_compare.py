# -*- coding: utf-8 -*-
"""生成典型样例的三系统故事全文对比（附带纯选择 LLM 评判）"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 典型样例：覆盖两种指标胜者与最大分歧
QUERIES = ["s02", "s08", "s11", "s14", "s22"]
LABELS = {
    "s02": "P 纯选胜 · 五维三系统均高分（work_pressure）",
    "s08": "Graphiti 纯选胜 · 五维 Graphiti 最高（emotion）",
    "s11": "Graphiti 五维最大胜（5.0 vs P 3.6）· 纯选 P 胜（emotion）",
    "s14": "Graphiti 纯选胜 · 五维 Graphiti 偏低（nostalgia）",
    "s22": "五维最大分歧（P 4.4 / RAG 2.4 / Graphiti 3.2）· 纯选 P 胜（life）",
}


def load(rel):
    return json.load(open(ROOT / rel, encoding="utf-8"))


def main() -> None:
    st_v3 = load("eval/outputs/story_texts_v3.json")
    cache_g = load("eval/outputs/story_texts_graphiti.json")
    sc_v3 = load("eval/outputs/story_scores_v3.json")
    sc_g = load("eval/outputs/story_scores_graphiti.json")
    pair_pg = {r["query_id"]: r for r in load("eval/outputs/story_pair_pg.json")["results"]}
    pair_v3 = {r["query_id"]: r for r in load("eval/outputs/story_pair_v3.json")["results"]}
    scores = {}
    for ds in (sc_v3, sc_g):
        for row in ds["results"]:
            scores.setdefault(row["query_id"], {}).setdefault(row["category"], {}).update(row["systems"])

    out = []
    for qid in QUERIES:
        cat = next(iter(scores[qid]))
        syss = scores[qid][cat]
        st_p = st_v3[qid]["stories"]["P"]["story"]
        st_r = st_v3[qid]["stories"]["Naive RAG"]["story"]
        st_gti = cache_g[qid]["stories"]["graphiti"]["story"]
        out.append(f"### 样例 {qid}（{cat}）—— {LABELS[qid]}")
        out.append(f"**查询**：{st_v3[qid]['query']}\n")
        # 五维
        out.append("**五维盲评**：")
        out.append("| 系统 | total | 相关性 | 连贯 | 完整 | 真实 | 噪音 |")
        out.append("|------|:---:|:---:|:---:|:---:|:---:|:---:|")
        for name, disp in [("p", "P"), ("naive_rag", "Naive RAG"), ("graphiti", "Graphiti")]:
            s = syss.get(name, {}).get("scores", {})
            f = lambda k: f"{s[k]:.1f}" if s.get(k) is not None else "-"
            out.append(f"| {disp} | {f('total')} | {f('relevance')} | "
                       f"{f('coherence')} | {f('completeness')} | "
                       f"{f('faithfulness')} | {f('noise_filter')} |")
        out.append("")
        for name, disp, st in [("P", "P", st_p), ("Naive RAG", "Naive RAG", st_r), ("Graphiti", "Graphiti", st_gti)]:
            out.append(f"**{disp} 故事**（{len(st)} 字）：\n```\n{st}\n```")
        # 纯选择评判
        out.append("**LLM 纯选择评判**：")
        pg = pair_pg.get(qid, {})
        v3 = pair_v3.get(qid, {})
        if pg:
            w = "P" if pg["winner"] == "P" else ("Graphiti" if pg["winner"] == "graphiti" else "tie")
            out.append(f"- **P vs Graphiti**：判 {w} 胜（换位={pg['display_swap']}）。{pg['reason']}")
        if v3:
            w = "P" if v3["winner"] == "p" else ("Naive RAG" if v3["winner"] == "rag" else "tie")
            out.append(f"- **P vs Naive RAG**：判 {w} 胜（换位={v3['display_swap']}）。{v3['reason']}")
        out.append("")

    text = "\n".join(out)
    (ROOT / "eval" / "outputs" / "story_compare_3sys.md").write_text(text, encoding="utf-8")
    print(f"已生成 eval/outputs/story_compare_3sys.md（{len(text)} 字符）")


if __name__ == "__main__":
    main()
