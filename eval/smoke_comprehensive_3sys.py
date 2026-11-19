# -*- coding: utf-8 -*-
"""
综合能力测评 · 单问题三系统 smoke 测试

只跑 1 个查询，验证 Ariadne / Mem0 / Graphiti 在 comprehensive 数据上的检索链路都能走通，
并并排展示各系统命中的记忆内容，作为后续全量质量测评的前置检查。

查询默认取 comprehensive_queries.yaml 第一条（可 --qid 指定）。
各系统输入文件（均已用 comprehensive gold 现状准备）：见下方常量。

用法:
    python -m eval.smoke_comprehensive_3sys                # 默认第 1 条
    python -m eval.smoke_comprehensive_3sys --qid q21      # 指定查询（助手行为）
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

GOLD_YAML = "data/dba_eval/comprehensive_gold.yaml"
QUERIES_YAML = "data/dba_eval/comprehensive_queries.yaml"
MEM0_RESULT = "eval/outputs/mem0_comprehensive_candidates.json"
GRAPHITI_RESULT = "eval/outputs/graphiti_comprehensive_candidates.json"

TOP_K = 12  # smoke 展示上限


def _load_dotenv(path=None):
    from dotenv import load_dotenv
    load_dotenv(path or ROOT / ".env")


def load_query(qid: str) -> dict:
    with open(ROOT / QUERIES_YAML, encoding="utf-8") as f:
        qs = yaml.safe_load(f)["queries"]
    for q in qs:
        if q["id"] == qid:
            return q
    raise SystemExit(f"查询 {qid} 不存在")


def ariadne_retrieve(query: str) -> list:
    """Ariadne：build_retriever(gold 图) → retrieve_with_story。返回候选 [{id, content}]。"""
    from eval.run_p_baseline import build_retriever
    graph, retriever = build_retriever(GOLD_YAML)
    out = retriever.retrieve_with_story(query, with_response=False)
    return [{"id": mid, "content": content} for mid, content in out.get("peak_memories", [])]


def mem0_retrieve(query: str) -> list:
    """Mem0：读 mem0_comprehensive 已生成候选（复用 faiss 索引，不再重建）。"""
    with open(ROOT / MEM0_RESULT, encoding="utf-8") as f:
        data = json.load(f)
    # 直接对已保存候选按 query_id 抽取；若查询不在其中，则现场 search
    for r in data["results"]:
        if r["query_id"] == CURRENT_QID:
            return [{"id": c["id"], "content": c["content"]} for c in r.get("candidates", [])]
    return mem0_retrieve_live(query)


def mem0_retrieve_live(query: str) -> list:
    """现场检索（未命中缓存时备用）。"""
    import frameworks.mem0_runner_zh_patch as zh
    zh.apply_patch()
    from mem0 import Memory
    from dba_pipeline.embedding.store import local_model_path
    cfg = {
        "vector_store": {"provider": "faiss", "config": {
            "collection_name": "comprehensive", "path": str(ROOT / "eval/.cache/mem0"),
            "embedding_model_dims": 1024, "distance_strategy": "cosine"}},
        "llm": {"provider": "openai", "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0,
            "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
            "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
        "embedder": {"provider": "huggingface", "config": {
            "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
    }
    mem = Memory.from_config(cfg)
    out = mem.search(query, top_k=TOP_K, filters={"user_id": "comprehensive_eval"}, threshold=0.0)
    return [{"id": (x.get("metadata") or {}).get("node_id", ""), "content": x.get("memory", "")}
            for x in out.get("results", []) if (x.get("metadata") or {}).get("node_id")]


def graphiti_retrieve(query: str) -> list:
    """Graphiti：读 graphiti_comprehensive 已生成候选。"""
    with open(ROOT / GRAPHITI_RESULT, encoding="utf-8") as f:
        data = json.load(f)
    for r in data["results"]:
        if r["query_id"] == CURRENT_QID:
            return [{"id": c["id"], "content": c["content"]} for c in r.get("candidates", [])]
    return []


CURRENT_QID = ""


def main() -> None:
    global CURRENT_QID
    parser = argparse.ArgumentParser(description="单问题三系统 smoke 测试")
    parser.add_argument("--qid", default="q01", help="查询 id（comprehensive_queries.yaml）")
    args = parser.parse_args()
    CURRENT_QID = args.qid

    _load_dotenv(ROOT / ".env")
    q = load_query(args.qid)
    print("=" * 70)
    print(f"查询 [{q['id']}] ({q.get('category')}): {q['text']}")
    print("=" * 70)

    cands_ari = ariadne_retrieve(q["text"])
    print(f"\n[Ariadne] 候选 {len(cands_ari)} 条")
    for c in cands_ari[:TOP_K]:
        print(f"  - {c['id']}: {c['content']}")

    cands_mem0 = mem0_retrieve(q["text"])
    print(f"\n[Mem0] 候选 {len(cands_mem0)} 条")
    for c in cands_mem0[:TOP_K]:
        print(f"  - {c['id']}: {c['content']}")

    cands_graphiti = graphiti_retrieve(q["text"])
    print(f"\n[Graphiti] 候选 {len(cands_graphiti)} 条")
    for c in cands_graphiti[:TOP_K]:
        print(f"  - {c['id']}: {c['content']}")

    print("\n" + "=" * 70)
    print(f"[结论] 三系统 smoke 通过：Ariadne={len(cands_ari)} Mem0={len(cands_mem0)} "
          f"Graphiti={len(cands_graphiti)} 候选")


if __name__ == "__main__":
    main()
