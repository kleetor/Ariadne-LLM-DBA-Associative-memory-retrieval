# -*- coding: utf-8 -*-
"""查看 P vs Naive RAG 故事片段对比（对指定查询重新跑 StoryRank）"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever
from eval.story_rank_adapter import story_rank_for_path, story_rank_for_texts

QIDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["q07", "q25"]

_load_dotenv(ROOT / ".env")
graph, retriever = build_retriever("data/LiFEMem.yaml")
inference = retriever.inference

p = {x["query_id"]: x for x in json.load(open("eval/outputs/p_candidates.json", encoding="utf-8"))["results"]}
r = {x["query_id"]: x for x in json.load(open("eval/outputs/naive_rag_candidates.json", encoding="utf-8"))["results"]}

for qid in QIDS:
    pv, rv = p[qid], r[qid]
    query = pv["query"]
    print("=" * 90)
    print(f"[{qid}] ({pv['category']}) {query}")
    out_p = story_rank_for_path(inference, query, pv["path"])
    print(f"\n--- P 故事（{len(out_p.get('story', ''))}字，输入 {len(pv['path']['nodes'])} 节点 {len(pv['path']['edges'])} 边）---")
    print(out_p.get("story", ""))
    texts = [c["content"] for c in rv["candidates"][:12]]
    out_r = story_rank_for_texts(inference, query, texts)
    print(f"\n--- Naive RAG 故事（{len(out_r.get('story', ''))}字，输入 {len(texts)} 条文本）---")
    print(out_r.get("story", ""))
    print()
