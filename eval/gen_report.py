# -*- coding: utf-8 -*-
"""
LiFEMem 全量实验报告生成器

1. 对每条查询重新跑 StoryRank，生成 P / Naive RAG 故事文本（story_texts.json 缓存）
2. 组装 Markdown 报告（含逐查询候选 + 故事 + 评分）

用法:
    python -m eval.gen_report            # 全流程
    python -m eval.gen_report --skip-story  # 跳过故事生成（复用缓存）
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from eval.run_p_baseline import _load_dotenv, build_retriever
from eval.story_rank_adapter import story_rank_for_path, story_rank_for_texts

CAND_FILES = {
    "p": "eval/outputs/p_candidates.json",
    "naive_rag": "eval/outputs/naive_rag_candidates.json",
}
SYSTEM_DISPLAY = {"p": "P", "naive_rag": "Naive RAG"}
STORY_CACHE = "eval/outputs/story_texts.json"
SCORE_FILE = "eval/outputs/story_scores_full.json"


def load_json(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def generate_stories(retriever, p_by, r_by, queries, only=None):
    """对每条查询生成 P（带边）与 RAG（文本）故事，缓存"""
    inference = retriever.inference
    cache = {}
    if Path(ROOT / STORY_CACHE).exists():
        cache = load_json(STORY_CACHE)

    todo = [q for q in queries if only is None or q["id"] in only]
    for q in todo:
        qid = q["id"]
        if qid in cache:
            continue
        entry = {"query": q["text"], "category": q.get("category", ""), "stories": {}}
        for name, by_id in (("P", p_by), ("Naive RAG", r_by)):
            r = by_id.get(qid)
            if not r:
                continue
            if name == "P":
                out = story_rank_for_path(inference, q["text"], r.get("path", {"nodes": [], "edges": []}))
            else:
                texts = [c["content"] for c in r.get("candidates", [])]
                out = story_rank_for_texts(inference, q["text"], texts)
            # 0927 审查 C1 消费者同步：story_rank 只返回 {story, used, degraded}，
            # 不再有 adopted_ids；used 即后端按正文覆盖统计的采纳节点数。
            entry["stories"][name] = {"story": out.get("story", ""),
                                      "adopted": out.get("used", 0)}
            print(f"[story] {qid} [{name}] story_len={len(out.get('story', ''))}", flush=True)
        cache[qid] = entry
        with open(ROOT / STORY_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    return cache


def build_report(scores, stories, p_by, r_by, queries) -> str:
    lines = []
    A = lines.append
    A("# LiFEMem 全量实验报告")
    A("")
    A(f"> 数据集：LiFEMem（322 节点 / 410 边） ｜ 查询集：35 条（7 类） ｜ 评测日期：2026-08-18")
    A(f"> 系统：P（目的驱动联想检索）vs Naive RAG（纯向量 top-N） ｜ 盲评：DeepSeek-v4-flash（temperature=0）")
    A("")
    A("## 一、汇总结果")
    A("")
    A("### 1.1 故事质量五维（30 条非弃权查询均值，总分=五维平均）")
    A("")
    A("| 系统 | relevance | coherence | completeness | faithfulness | noise_filter | total |")
    A("|------|:---:|:---:|:---:|:---:|:---:|:---:|")
    dims = ["relevance", "coherence", "completeness", "faithfulness", "noise_filter", "total"]
    sys_keys = list(CAND_FILES.keys())
    agg = {s: {x: [] for x in dims} for s in sys_keys}
    for row in scores["results"]:
        if row["category"] == "abstention":
            continue
        for s in sys_keys:
            sc = row["systems"].get(s, {}).get("scores")
            if sc:
                for x in dims:
                    agg[s][x].append(sc[x])
    for s in sys_keys:
        vals = [f"{sum(agg[s][x])/len(agg[s][x]):.2f}" for x in dims]
        A(f"| {SYSTEM_DISPLAY[s]} | " + " | ".join(vals) + " |")
    A("")
    A("### 1.2 检索层辅助指标")
    A("")
    A("| 系统 | R@all | CROSS_OUT | 平均输出量 | 弃权拒答(正确/总数) |")
    A("|------|:---:|:---:|:---:|:---:|")
    # R@all / CROSS_OUT 从 cross_metric 逻辑重算（内联简化）
    from eval.cross_metric import domain_of
    stats = {}
    for s, fname in CAND_FILES.items():
        cand = {r["query_id"]: r for r in load_json(fname)["results"]}
        co, ra, no, abst_ok, abst_n = [], [], [], 0, 0
        score_by_q = {row["query_id"]: row for row in scores["results"]}
        for q in queries:
            qid = q["id"]
            r = cand.get(qid)
            if not r:
                continue
            out_ids = [c["id"] for c in r.get("candidates", [])]
            expected = set(q.get("expected", []))
            no.append(len(out_ids))
            if expected:
                ra.append(len(set(out_ids) & expected) / len(expected))
                doms = [domain_of(e) for e in expected]
                q_dom = max(set(doms), key=doms.count)
                cross = [i for i in out_ids if domain_of(i) != q_dom]
                co.append(len(cross) / len(out_ids) if out_ids else 0)
            else:
                abst = score_by_q.get(qid, {}).get("systems", {}).get(s, {}).get("abstention", {})
                abst_ok += 1 if abst.get("refused") else 0
                abst_n += 1
        A(f"| {SYSTEM_DISPLAY[s]} | {sum(ra)/len(ra):.3f} | {sum(co)/len(co):.3f} | {sum(no)/len(no):.0f} | {abst_ok}/{abst_n} |")
    A("")
    A("### 1.3 分能力故事质量（total 均值）")
    A("")
    cats = {}
    for row in scores["results"]:
        if row["category"] == "abstention":
            continue
        cats.setdefault(row["category"], []).append(row)
    A("| 类别 | " + " | ".join(SYSTEM_DISPLAY[s] for s in sys_keys) + " |")
    A("|------|" + "---:|" * len(sys_keys))
    for cat in ["single_fact", "preference", "multi_hop", "knowledge_update", "temporal", "cross_scene"]:
        rows = cats.get(cat, [])
        cell = []
        for s in sys_keys:
            vals = [r["systems"][s]["scores"]["total"] for r in rows
                    if s in r["systems"] and r["systems"][s].get("scores")]
            cell.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
        A(f"| {cat} | " + " | ".join(cell) + " |")
    A("")
    A("### 1.4 CROSS_VALID（扩展内容合理联想率，非弃权类）")
    A("")
    A("> 对故事中扩展内容（非查询字面相关部分）标注合理联想（reasonable）与纯噪音（noise）占比，合理联想率 = reasonable/(reasonable+noise)。")
    A("")
    A("| 类别 | " + " | ".join(SYSTEM_DISPLAY[s] for s in sys_keys) + " |")
    A("|------|" + "---:|" * len(sys_keys))
    A("| **总体均值** | " + " | ".join(
        f"{sum(r['systems'][s].get('cross_valid', {}).get('cross_valid', 0) for r in rows if r['systems'][s].get('cross_valid'))/max(1, sum(1 for r in rows if r['systems'][s].get('cross_valid'))):.3f}"
        for s in sys_keys) + " |")
    for cat in ["single_fact", "preference", "multi_hop", "knowledge_update", "temporal", "cross_scene"]:
        rows = cats.get(cat, [])
        cell = []
        for s in sys_keys:
            vals = [r["systems"][s]["cross_valid"]["cross_valid"] for r in rows
                    if r["systems"][s].get("cross_valid")]
            cell.append(f"{sum(vals)/len(vals):.3f}" if vals else "-")
        A(f"| {cat} | " + " | ".join(cell) + " |")
    A("")
    A("## 二、逐查询详情")
    A("")
    for q in queries:
        qid = q["id"]
        A(f"### {qid} [{q['category']}] {q['text']}")
        A("")
        A(f"- **expected**：{q.get('expected', [])}")
        st = stories.get(qid, {})
        for s in sys_keys:
            by_id = p_by if s == "p" else r_by
            r = by_id.get(qid)
            if not r:
                continue
            disp = SYSTEM_DISPLAY[s]
            A(f"- **{disp} 候选（{r['total_candidates']} 条，top8）**：")
            for c in r["candidates"][:8]:
                A(f"  - `{c['id']}` {c['content']}（score={c['score']:.3f}）")
            story = st.get("stories", {}).get(disp, {}).get("story", "")
            score = ""
            for row in scores["results"]:
                if row["query_id"] == qid:
                    sv = row["systems"].get(s, {})
                    score = sv.get("scores", sv.get("abstention", {}))
                    break
            A(f"- **{disp} 故事**（{len(story)}字，评分 {score}）：")
            A(f"  > {story}")
            A("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="LiFEMem 全量报告")
    parser.add_argument("--skip-story", action="store_true", help="跳过故事生成（复用缓存）")
    parser.add_argument("--out", default="Plan/0818——LiFEMem全量实验报告.md")
    parser.add_argument("--score-file", default=SCORE_FILE, help="评分文件（默认 story_scores_full.json）")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    queries = yaml.safe_load(open(ROOT / "data/LiFEMem_queries.yaml", encoding="utf-8"))["queries"]
    p_by = {r["query_id"]: r for r in load_json(CAND_FILES["p"])["results"]}
    r_by = {r["query_id"]: r for r in load_json(CAND_FILES["naive_rag"])["results"]}

    if not args.skip_story:
        graph, retriever = build_retriever("data/LiFEMem.yaml")
        generate_stories(retriever, p_by, r_by, queries)
    stories = load_json(STORY_CACHE)

    scores = load_json(args.score_file)
    report = build_report(scores, stories, p_by, r_by, queries)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[report] 已生成: {out}（{len(report)} 字符）")


if __name__ == "__main__":
    main()
