# -*- coding: utf-8 -*-
"""Graphiti 故事质量最低的归因诊断（StoryRank 前后两环节）

StoryRank 前（候选层）：
- 候选数量/去重后节点数
- 候选的"人生维度覆盖"（按节点 id 前缀分域）
- 事实(fact)数 vs 映射出的节点数（映射损耗）
StoryRank 后（故事层）：
- 故事长度分布（P / RAG / Graphiti）
- 故事中出现的"实体/细节量"粗估（数字、课程名、人名等）
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PFS = ["he", "ln", "sh", "so", "w", "e", "h", "c", "f", "t", "n"]
DIM_NAME = {"w": "工作", "he": "健康", "e": "情绪", "h": "学生时代", "c": "学生时代",
            "f": "家庭/未来", "ln": "学习", "so": "社交", "t": "旅行/消费", "n": "噪声"}


def prefix(nid):
    for pf in PFS:
        if nid.startswith(pf):
            return pf
    return "?"


def main() -> None:
    # ---- 候选层 ----
    cands = {
        "P": {r["query_id"]: r for r in json.load(open(ROOT / "eval/outputs/p_candidates_v3.json", encoding="utf-8"))["results"]},
        "Naive RAG": {r["query_id"]: r for r in json.load(open(ROOT / "eval/outputs/naive_rag_candidates_v3.json", encoding="utf-8"))["results"]},
        "Graphiti": {r["query_id"]: r for r in json.load(open(ROOT / "eval/outputs/graphiti_candidates.json", encoding="utf-8"))["results"]},
    }
    # ---- 故事层 ----
    st_v3 = json.load(open(ROOT / "eval/outputs/story_texts_v3.json", encoding="utf-8"))
    st_g = json.load(open(ROOT / "eval/outputs/story_texts_graphiti.json", encoding="utf-8"))

    qids = sorted(st_v3.keys())

    print("===== A. 候选层统计（StoryRank 前） =====")
    print(f"{'系统':<12}{'平均候选数':<10}{'候选>38查询数':<12}{'场景覆盖均值':<12}{'≤2场景查询数'}")
    for name, by_id in cands.items():
        counts, spread = [], []
        for qid in qids:
            r = by_id.get(qid)
            ids = [c["id"] for c in r.get("candidates", [])] if r else []
            counts.append(len(ids))
            dims = {prefix(i) for i in ids}
            spread.append(len(dims))
        over = sum(1 for c in counts if c > 38)
        low = sum(1 for s in spread if s <= 2)
        print(f"{name:<12}{sum(counts)/len(counts):<10.1f}{over:<12}{sum(spread)/len(spread):<12.1f}{low}")

    # 每查询候选场景覆盖（Graphiti vs RAG）
    print("\n每查询候选场景数（G vs R）：")
    print(f"{'qid':<5}{'G':<4}{'R':<4}")
    g_ids = cands["Graphiti"]; r_ids = cands["Naive RAG"]
    g_low = 0
    for qid in qids:
        gs = len({prefix(c["id"]) for c in g_ids[qid].get("candidates", [])})
        rs = len({prefix(c["id"]) for c in r_ids[qid].get("candidates", [])})
        if gs < rs:
            g_low += 1
        print(f"{qid:<5}{gs:<4}{rs:<4}")
    print(f"Graphiti 场景覆盖 < RAG 的查询数: {g_low}/{len(qids)}")

    # 映射损耗：fact 数 vs 节点数（Graphiti）
    print("\nGraphiti 映射损耗（前 6 查询：fact数 → 去重节点数）：")
    for qid in qids[:6]:
        r = g_ids[qid]
        n_fact = len({c.get("fact", "") for c in r.get("candidates", []) if c.get("fact")})
        n_nodes = len(r.get("candidates", []))
        print(f"  {qid}: {n_fact} facts -> {n_nodes} 节点（候选）")

    # ---- 故事层 ----
    print("\n===== B. 故事层统计（StoryRank 后） =====")
    lens = {"P": [], "Naive RAG": [], "Graphiti": []}
    for qid in qids:
        sts = st_v3[qid]["stories"]
        if "P" in sts:
            lens["P"].append(len(sts["P"]["story"]))
        if "Naive RAG" in sts:
            lens["Naive RAG"].append(len(sts["Naive RAG"]["story"]))
        if qid in st_g and "graphiti" in st_g[qid]["stories"]:
            lens["Graphiti"].append(len(st_g[qid]["stories"]["graphiti"]["story"]))
    print(f"{'系统':<12}{'故事长度均值':<12}{'中位数':<10}{'最短':<8}{'最长'}")
    for name, ls in lens.items():
        ls_sorted = sorted(ls)
        print(f"{name:<12}{sum(ls)/len(ls):<12.1f}{ls_sorted[len(ls)//2]:<10}{min(ls):<8}{max(ls)}")

    # 细节量粗估：故事中出现的人名/课程/数字等专有信息
    detail_pat = re.compile(r"吴恩达|李沐|Kaggle|LeetCode|Transformer|PyTorch|CUDA|RXT|豆豆|小陈|陈浩|林晓雨|苏婉婷|刘建国|王秀兰|刘雨桐|高考|大专|表|证")
    print("\n故事细节量（专有信息命中次数均值，样本数=3 系统 × 30 查询）：")
    for name, sts in [("P", st_v3), ("Naive RAG", st_v3), ("Graphiti", st_g)]:
        key = "P" if name == "P" else ("Naive RAG" if name == "Naive RAG" else "graphiti")
        hits = []
        for qid in qids:
            if key in sts[qid]["stories"]:
                hits.append(len(detail_pat.findall(sts[qid]["stories"][key]["story"])))
        print(f"  {name:<12}{sum(hits)/len(hits):.1f}")


if __name__ == "__main__":
    main()
