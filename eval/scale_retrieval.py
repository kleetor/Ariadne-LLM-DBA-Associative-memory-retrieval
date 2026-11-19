# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
系统「检索侧」注入量随记忆库/图谱规模增长趋势（主环境：Ariadne + Mem0）。

目的：回答"图规模 ↑ → 检索到更多内容 / 检索注入量是否增长"。
- Ariadne：search 是真 LLM（infer_purpose + StoryRank），其 LLM 成本上一轮已测（平坦）。
           这里补「检索到的候选数 + story 长度 + story 注入 token」。
- Mem0：search = 向量+BM25 非 LLM（=0 token），返回条数由 top_k 封顶；这里测返回条数
        与记忆内容注入 token（cl100k）。

注入量用 cl100k_base 统一计数（与 smoke_system_compare.py 同标尺）。

  python eval/scale_retrieval.py
"""

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ["HF_HUB_OFFLINE"] = "1"

SCALES = [50, 100, 200]
QUERIES = [
    "最近项目排得太满了，天天加班到十一二点，感觉整个人都被掏空了……",
    "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？",
    "感觉压力好大，不知道怎么熬过这段日子……",
]
TOP_K = 8

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _tok(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(text)


def _load_dotenv(path=None):
    from dotenv import load_dotenv
    load_dotenv(path or ROOT / ".env")


def measure_ariadne(n_scale: int) -> dict:
    from eval.run_p_baseline import build_retriever
    graph_path = f"data/dba_eval/scale_{n_scale}.yaml"
    graph, retriever = build_retriever(graph_path)
    out = retriever.retrieve_with_story(QUERIES[0], with_response=False)
    peak = out.get("peak_memories", [])
    stories = out.get("stories", [])
    story = stories[0] if stories else ""
    return {
        "n_candidates": len(peak),           # 检索命中的候选数
        "story_chars": len(story),           # story 字符数
        "story_tokens": _tok(story),          # story 注入 token（cl100k）
    }


def measure_mem0(n_scale: int) -> dict:
    import frameworks.mem0_runner_zh_patch as zh
    zh.apply_patch()
    from mem0 import Memory
    from dba_pipeline.embedding.store import local_model_path

    mem_dir = ROOT / f"eval/.cache/mem0_scale_{n_scale}"
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": f"scale_{n_scale}", "path": str(mem_dir),
            "embedding_model_dims": 1024, "on_disk": True}},
        "llm": {"provider": "openai", "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0,
            "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
            "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
        "embedder": {"provider": "huggingface", "config": {
            "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
    }
    mem = Memory.from_config(cfg)
    out = mem.search(QUERIES[1], top_k=TOP_K, filters={"user_id": "scale"}, threshold=0.0)
    results = out.get("results", [])
    mems = [str(x.get("memory", "")) for x in results]
    return {
        "n_returned": len(mems),
        "memories_tokens": sum(_tok(m) for m in mems),  # 记忆内容注入 token（cl100k）
        "memories_chars": sum(len(m) for m in mems),
    }


def main() -> None:
    _load_dotenv()
    results = {}
    for n_scale in SCALES:
        print(f"\n{'=' * 60}\n[scale={n_scale}] 检索侧", flush=True)
        ari = measure_ariadne(n_scale)
        mem0 = measure_mem0(n_scale)
        results[n_scale] = {"ariadne": ari, "mem0": mem0}
        print(f"  Ariadne:  候选={ari['n_candidates']}  story字符={ari['story_chars']}  "
              f"story注入token={ari['story_tokens']}", flush=True)
        print(f"  Mem0:     返回={mem0['n_returned']} 条  记忆注入token={mem0['memories_tokens']}  "
              f"(search=0 LLM)", flush=True)

    out = ROOT / "eval/outputs/scale_retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scales": SCALES, "results": results, "top_k": TOP_K,
        "note": ("检索注入量：Ariadne=候选数+story token；Mem0=search 返回条数(top_k封顶)+记忆内容token；"
                 "Ariadne search 的 LLM 成本见 scale_internal_cost.json；Mem0 search=0 LLM；"
                 "注入量用 cl100k_base（与 judge 上下文同标尺）"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}", flush=True)


if __name__ == "__main__":
    main()
