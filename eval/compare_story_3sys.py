# -*- coding: utf-8 -*-
"""
三系统故事质量对比（P / Naive RAG / Graphiti）

数据来源：
- story_scores_v3.json：P 与 RAG 的盲评（2026-08-19，同一 judge 协议）
- story_scores_graphiti.json：Graphiti 的盲评（2026-08-21，同一 judge 协议）

输出：逐条五维分 + 均值汇总（总体 / 按类别）+ 弃权判定 + 联想合理性。

用法:
    python -m eval.compare_story_3sys
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

P_RAG = "eval/outputs/story_scores_v3.json"
GRAPHITI = "eval/outputs/story_scores_graphiti.json"
DIMS = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter"]


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def merge(*datasets):
    """按 query_id 合并多个 story_scores 文件，返回 {qid: {system: entry}}"""
    merged = {}
    for ds in datasets:
        for row in ds["results"]:
            qid = row["query_id"]
            merged.setdefault(qid, {"query": row["query"], "category": row["category"],
                                    "systems": {}})
            merged[qid]["systems"].update(row["systems"])
    return merged


def fmt_score(v):
    if not v:
        return "  -  "
    return f"{v:.1f}"


def main() -> None:
    p_rag = load(P_RAG)
    g = load(GRAPHITI)
    data = merge(p_rag, g)

    systems = ["p", "naive_rag", "graphiti"]
    DISPLAY = {"p": "P", "naive_rag": "Naive RAG", "graphiti": "Graphiti"}

    # ---- 逐条表 ----
    header = f"{'qid':<5}{'cat':<15}{'query':<40}" + \
             "".join(f"{DISPLAY[s]:>26}" for s in systems)
    print(header)
    print("-" * len(header))
    for qid in sorted(data.keys()):
        row = data[qid]
        cat = row["category"]
        qtext = row["query"][:38]
        line = f"{qid:<5}{cat:<15}{qtext:<40}"
        for s in systems:
            sys_e = row["systems"].get(s, {})
            if cat == "abstention":
                line += f"{'ABS:' + str(sys_e.get('abstention','-')):>26}"
            else:
                sc = sys_e.get("scores", {})
                line += f"{fmt_score(sc.get('total')) + '|cv' + fmt_score(sys_e.get('cross_valid',{}).get('cross_valid')):>26}"
        print(line)

    # ---- 汇总 ----
    print("\n===== 均值汇总 =====")
    print(f"{'系统':<12}{'n':<4}{'total':<7}{'rel':<7}{'coh':<7}{'com':<7}{'fai':<7}{'noi':<7}{'cv':<7}")
    for s in systems:
        totals, dims, cvs = [], {d: [] for d in DIMS}, []
        for qid, row in data.items():
            if row["category"] == "abstention":
                continue
            sys_e = row["systems"].get(s, {})
            sc = sys_e.get("scores")
            if not sc:
                continue
            totals.append(sc["total"])
            for d in DIMS:
                dims[d].append(sc.get(d, 0))
            if sys_e.get("cross_valid", {}).get("cross_valid") is not None:
                cvs.append(sys_e["cross_valid"]["cross_valid"])
        n = len(totals)
        if n == 0:
            continue
        avg = lambda xs: sum(xs) / len(xs)
        print(f"{DISPLAY[s]:<12}{n:<4}{avg(totals):<7.2f}"
              + "".join(f"{avg(dims[d]):<7.2f}" for d in DIMS)
              + f"{avg(cvs):<7.2f}")

    # ---- 按类别汇总 ----
    print("\n===== 按类别 total 均值 =====")
    cats = []
    for row in data.values():
        if row["category"] not in cats:
            cats.append(row["category"])
    print(f"{'cat':<15}" + "".join(f"{DISPLAY[s]:>10}" for s in systems))
    for cat in cats:
        if cat == "abstention":
            continue
        line = f"{cat:<15}"
        for s in systems:
            vals = [row["systems"][s]["scores"]["total"]
                    for qid, row in data.items()
                    if row["category"] == cat
                    and s in row["systems"] and row["systems"][s].get("scores")]
            line += f"{sum(vals)/len(vals):>10.2f}" if vals else f"{'-':>10}"
        print(line)

    # ---- 弃权判定 ----
    print("\n===== 弃权类（abstention 判定） =====")
    for qid in sorted(data.keys()):
        row = data[qid]
        if row["category"] != "abstention":
            continue
        vals = []
        for s in systems:
            sys_e = row["systems"].get(s, {})
            vals.append(f"{DISPLAY[s]}={sys_e.get('abstention','-')}")
        print(f"{qid}: " + "  ".join(vals))

    # ---- 胜出统计 ----
    print("\n===== 两两对比（total 均值差） =====")
    import itertools
    for a, b in itertools.combinations(systems, 2):
        pairs = [(row["systems"][a]["scores"]["total"], row["systems"][b]["scores"]["total"])
                 for qid, row in data.items()
                 if row["category"] != "abstention"
                 and a in row["systems"] and b in row["systems"]
                 and row["systems"][a].get("scores") and row["systems"][b].get("scores")]
        wins_a = sum(1 for x, y in pairs if x > y)
        wins_b = sum(1 for x, y in pairs if x < y)
        draws = len(pairs) - wins_a - wins_b
        d = sum(x - y for x, y in pairs) / len(pairs)
        print(f"  {DISPLAY[a]} vs {DISPLAY[b]}: 均值差 {d:+.2f} "
              f"（{DISPLAY[a]}胜{wins_a} / 平{draws} / {DISPLAY[b]}胜{wins_b}，共{len(pairs)}条）")


if __name__ == "__main__":
    main()
