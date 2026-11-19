# -*- coding: utf-8 -*-
"""
T6: 首版对照运行（故事化 + 盲评编排）

读取各记忆系统的候选文件，统一：
1. 故事化：P 用带边 path；外部框架用候选文本列表（无边）
2. 盲评：五维评分（匿名，不标注来源）；弃权类单独判定
3. 汇总：按系统 × 查询输出评分，生成对照表

用法:
    python -m eval.run_pipeline --systems p,naive_rag
    python -m eval.run_pipeline --systems p,naive_rag,mem0
    python -m eval.run_pipeline --systems p,naive_rag,mem0 --out eval/outputs/story_scores.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 import huggingface_hub（经 langchain_openai 间接导入）之前强制离线
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_OFFLINE"] = "1"

import yaml
from langchain_openai import ChatOpenAI

from eval.run_p_baseline import _load_dotenv, build_retriever
from eval.story_rank_adapter import story_rank_for_path, story_rank_for_texts
from eval.judge_story import judge_story, judge_abstention, judge_cross_valid

# 各系统的候选文件路径
SYSTEM_FILES = {
    "p": "eval/outputs/p_candidates.json",
    "naive_rag": "eval/outputs/naive_rag_candidates.json",
    "mem0": "eval/outputs/mem0_candidates.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="T6: 故事化 + 盲评编排")
    parser.add_argument("--systems", default="p,naive_rag",
                        help="逗号分隔的系统名，取自 SYSTEM_FILES；或 name:file 形式指定候选文件")
    parser.add_argument("--queries", default="data/LiFEMem_queries.yaml", help="查询集 YAML")
    parser.add_argument("--out", default="eval/outputs/story_scores.json")
    parser.add_argument("--story-cache", default="eval/outputs/story_texts.json",
                        help="故事缓存文件（v3 用独立缓存避免 key 冲突）")
    parser.add_argument("--only-queries", default="",
                        help="逗号分隔的查询 id 子集（调试用，默认全部）")
    parser.add_argument("--use-story-cache", action="store_true",
                        help="从 story_texts.json 复用故事文本，跳过 StoryRank（仅重评评分）")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    only = [q.strip() for q in args.only_queries.split(",") if q.strip()]

    story_cache = {}
    if args.use_story_cache:
        with open(ROOT / args.story_cache, encoding="utf-8") as f:
            story_cache = json.load(f)
        print(f"[T6] 复用故事缓存 {len(story_cache)} 条（跳过 StoryRank）")

    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever("data/LiFEMem.yaml")
    inference = retriever.inference

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    # 加载各系统候选（支持 name:file 形式）
    cands = {}
    for s in systems:
        if ":" in s:
            name, fname = s.split(":", 1)
        else:
            name, fname = s, SYSTEM_FILES.get(s, "")
        p = ROOT / fname
        if not p.exists():
            print(f"[T6] 跳过 {name}: 候选文件不存在 {p}")
            continue
        with open(p, encoding="utf-8") as f:
            cands[name] = {r["query_id"]: r for r in json.load(f)["results"]}
    print(f"[T6] 加载候选: {list(cands.keys())}")

    # 查询集
    with open(ROOT / args.queries, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    if only:
        queries = [q for q in queries if q["id"] in only]

    all_scores = {"meta": {"systems": list(cands.keys()), "queries": len(queries),
                           "judge": "deepseek-v4-flash(temperature=0)", "blind": True},
                  "results": []}

    DISPLAY = {"p": "P", "naive_rag": "Naive RAG"}
    for q in queries:
        qid = q["id"]
        q_cat = q.get("category", q.get("scenario", ""))
        is_abstention = (q_cat == "abstention")
        # 查询目的：在查询内容上扩展的检索意图（judge 的噪音判定参照，非系统标识）
        purpose = ", ".join(inference.infer_purpose(q["text"]).get("purposes", []))
        row = {"query_id": qid, "category": q_cat, "query": q["text"],
               "purpose": purpose, "systems": {}}
        story_entry = story_cache.setdefault(qid, {"query": q["text"], "category": q_cat, "stories": {}})
        for s, by_id in cands.items():
            r = by_id.get(qid)
            if not r:
                continue
            disp = DISPLAY.get(s, s)
            # 1) 故事化（优先复用缓存）
            story = story_entry.get("stories", {}).get(disp, {}).get("story", "")
            if not story:
                if s == "p":
                    out = story_rank_for_path(inference, q["text"], r.get("path", {"nodes": [], "edges": []}))
                else:
                    texts = [c["content"] for c in r.get("candidates", [])]
                    out = story_rank_for_texts(inference, q["text"], texts)
                story = out.get("story", "")
                story_entry["stories"][disp] = {"story": story}
                # 定期落盘（新生成的故事缓存）
                with open(ROOT / args.story_cache, "w", encoding="utf-8") as f:
                    json.dump(story_cache, f, ensure_ascii=False, indent=1)
            # 2) 盲评（匿名：仅 query + purpose + story，无系统标识）
            if is_abstention:
                sc = judge_abstention(llm, q["text"], story)
                row["systems"][s] = {"abstention": sc, "story_len": len(story)}
            else:
                sc = judge_story(llm, q["text"], story, purpose=purpose)
                cv = judge_cross_valid(llm, q["text"], story, purpose=purpose)
                row["systems"][s] = {"scores": sc, "cross_valid": cv, "story_len": len(story)}
            print(f"[T6] {qid} [{s}] story_len={len(story)}", flush=True)
        all_scores["results"].append(row)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[T6] 输出: {out_path}", flush=True)

    # 简要汇总
    print("\n[T6] 汇总（非弃权类 total 均值，按系统）:")
    agg = {s: [] for s in cands}
    for row in all_scores["results"]:
        if row["category"] == "abstention":
            continue
        for s, v in row["systems"].items():
            agg[s].append(v["scores"]["total"])
    for s, totals in agg.items():
        if totals:
            print(f"  {s}: avg_total={sum(totals)/len(totals):.2f} ({len(totals)}条)")


if __name__ == "__main__":
    main()
