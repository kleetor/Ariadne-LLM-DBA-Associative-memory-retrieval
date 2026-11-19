# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
系统「内部 LLM 调用」token 成本测量。

核心：Ariadne 是 tool-calling（search 才花 LLM，add 纯图插入 = 0 LLM）；
Mem0/Graphiti 侵入式（每轮 add/ingest 都做 LLM 抽取）。

测量各操作的单次 LLM token（input+output），再按「多轮会话 + 1 次换话题」的
调用频率模型汇总出会话级内部成本。

  python eval/session_internal_cost.py
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

from langchain_openai import ChatOpenAI  # noqa: E402

GRAPH = "data/dba_eval/smoke_companion.yaml"
SESSION = {
    "n_turns": 6,
    "topic_switch_turn": 3,   # 第 4 轮换话题（0-based 索引）
}
QUERIES = [
    "最近项目排得太满了，天天加班到十一二点，感觉整个人都被掏空了……",
    "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？",
    "感觉压力好大，不知道怎么熬过这段日子……",
]


class Recorder:
    """包裹 langchain ChatOpenAI.invoke，累加每次内部调用的 API token_usage。"""

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0
        self.orig = None

    def reset(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def __enter__(self):
        self.orig = ChatOpenAI.invoke
        rec = self

        def wrapped(self_obj, *a, **k):
            r = rec.orig(self_obj, *a, **k)
            meta = getattr(r, "response_metadata", {}) or {}
            tu = meta.get("token_usage") or meta.get("usage") or {}
            inp = tu.get("input_tokens") or tu.get("prompt_tokens") or 0
            out = tu.get("output_tokens") or tu.get("completion_tokens") or 0
            rec.prompt += inp
            rec.completion += out
            rec.calls += 1
            return r

        ChatOpenAI.invoke = wrapped
        return self

    def __exit__(self, *a):
        ChatOpenAI.invoke = self.orig
        return None

    @property
    def total(self):
        return self.prompt + self.completion


def _load_dotenv(path=None):
    from dotenv import load_dotenv
    load_dotenv(path or ROOT / ".env")


def main() -> None:
    _load_dotenv()
    print("session 模型:", SESSION, flush=True)

    rec = Recorder()

    # ---------- Ariadne：search 操作（infer_purpose + story_rank） ----------
    from eval.run_p_baseline import build_retriever
    graph, retriever = build_retriever(GRAPH)
    rec.reset()
    with rec:
        out = retriever.retrieve_with_story(QUERIES[0], with_response=False)
    story = out.get("stories", [""])[0]
    ari_search = {"prompt_tokens": rec.prompt, "completion_tokens": rec.completion, "llm_calls": rec.calls,
                  "story_chars": len(story)}
    print(f"\n[Ariadne search-op] LLM调用={rec.calls}  input={rec.prompt}  output={rec.completion}  "
          f"total={rec.total}  story_chars={len(story)}", flush=True)

    # ---------- Ariadne：写入/抽取（maintain = triage + node抽取 + edge连接） ----------
    # 添加节点其实也有 token（MemoryDBA.maintain 走 LLM 抽取），不再计 0。
    from dba_pipeline.extraction.dba import MemoryDBA
    from dba_pipeline.extraction.graph_builder import GraphBuilder
    from dba_pipeline.embedding.store import VectorStore
    import threading
    vs = VectorStore(embeddings=None, backend="faiss", persist_dir=None)
    db_builder = GraphBuilder(graph=graph, vector_store=vs, lock=threading.Lock())
    dba = MemoryDBA(llm=retriever.inference.llm, graph=graph, vector_store=vs, graph_builder=db_builder)
    rec.reset()
    try:
        with rec:
            dba.maintain(QUERIES[0])
    except Exception as e:  # apply_ops 可能因 None embeddings 报错，但抽取 LLM 已被记录
        print(f"  (maintain apply_ops 异常，忽略以保留抽取 token 计数): {type(e).__name__}", flush=True)
    ari_write = {"prompt_tokens": rec.prompt, "completion_tokens": rec.completion, "llm_calls": rec.calls}
    print(f"[Ariadne write-op(maintain)]  LLM调用={rec.calls}  input={rec.prompt}  output={rec.completion}  "
          f"total={rec.total}", flush=True)

    # ---------- Mem0：真实 add(infer=True) + search，包裹 openai client ----------
    # mem0 走记忆系统的 LLM（extraction + infer 的增/删/改决策），故用其真实 add() 测量，
    # 并在 mem0 的 OpenAI client 上包计数器（与 graphiti 同口径），而非简化 prompt 估算。
    import frameworks.mem0_runner_zh_patch as zh
    zh.apply_patch()
    from mem0 import Memory
    import shutil
    with open(ROOT / GRAPH, encoding="utf-8") as f:
        id2content = {n["id"]: n["content"] for n in yaml.safe_load(f)["nodes"]}
    mem_dir = ROOT / "eval/.cache/mem0_qdrant_usage"
    if mem_dir.exists():
        shutil.rmtree(mem_dir, ignore_errors=True)
    from dba_pipeline.embedding.store import local_model_path
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "usage_cmp", "path": str(mem_dir),
            "embedding_model_dims": 1024, "on_disk": True}},
        "llm": {"provider": "openai", "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0.1,
            "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
            "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
        "embedder": {"provider": "huggingface", "config": {
            "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
    }
    mem = Memory.from_config(cfg)

    class _OpenAITok:
        def __init__(self):
            self.prompt = self.completion = self.calls = 0

        def wrap(self, create_fn):
            me = self

            def w_create(*a, **k):
                r = create_fn(*a, **k)
                u = getattr(r, "usage", None)
                if u is not None:
                    me.prompt += getattr(u, "prompt_tokens", 0) or 0
                    me.completion += getattr(u, "completion_tokens", 0) or 0
                    me.calls += 1
                return r

            return w_create

        @property
        def total(self):
            return self.prompt + self.completion

    tok = _OpenAITok()
    completions = mem.llm.client.chat.completions
    orig_create = completions.create
    completions.create = tok.wrap(orig_create)

    # 每次 add 一条用户消息（infer=True 会做抽取 + 增/删/改决策）
    mem.add({"role": "user", "content": next(iter(id2content.values()))}, user_id="cmp", infer=True)
    mem0_add = {"prompt_tokens": tok.prompt, "completion_tokens": tok.completion, "llm_calls": tok.calls}
    print(f"[Mem0 add-op(infer)]  LLM调用={tok.calls}  input={tok.prompt}  output={tok.completion}  "
          f"total={tok.total}", flush=True)

    # search：默认 rerank=False 且未配 reranker → 应为 0 LLM（向量+BM25）
    tok.prompt = tok.completion = tok.calls = 0
    mem.search("最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？", top_k=8,
               filters={"user_id": "cmp"}, threshold=0.0)
    mem0_search = {"prompt_tokens": tok.prompt, "completion_tokens": tok.completion, "llm_calls": tok.calls}
    print(f"[Mem0 search-op]      LLM调用={tok.calls}  input={tok.prompt}  output={tok.completion}  "
          f"total={tok.total}  (应为 0 = 非LLM)", flush=True)
    completions.create = orig_create

    # ---------- 按「多轮 + 1 次换话题」汇总 ----------
    n = SESSION["n_turns"]
    sw = SESSION["topic_switch_turn"]
    ari_searches = 2  # 冷启动(第0轮) + 换话题(第sw轮)；其余轮仅 add(写入抽取)
    model = {
        "Ariadne": {
            "llm_calls": ari_searches * ari_search["llm_calls"] + n * ari_write["llm_calls"],
            "prompt_tokens": ari_searches * ari_search["prompt_tokens"] + n * ari_write["prompt_tokens"],
            "completion_tokens": ari_searches * ari_search["completion_tokens"] + n * ari_write["completion_tokens"],
        },
        "Mem0": {"llm_calls": n * mem0_add["llm_calls"],
                 "prompt_tokens": n * mem0_add["prompt_tokens"],
                 "completion_tokens": n * mem0_add["completion_tokens"]},
    }
    for name, d in model.items():
        d["total_tokens"] = d["prompt_tokens"] + d["completion_tokens"]
    model["Ariadne"]["searches"] = ari_searches
    model["Ariadne"]["writes"] = n

    print("\n" + "=" * 60)
    print(f"会话：{n} 轮 + 1 次换话题")
    print(f"  Ariadne: search {ari_searches} 次 + 每轮 write(maintain 抽取) {n} 次")
    print(f"  Mem0/Graphiti: 每轮抽取 {n} 次（Graphiti 走独立 venv，另行测量）")
    print(f"{'系统':<10}{'LLM调用':<8}{'input':<10}{'output':<10}{'总token':<10}")
    for name, d in model.items():
        print(f"{name:<10}{d['llm_calls']:<8}{d['prompt_tokens']:<10}{d['completion_tokens']:<10}{d['total_tokens']:<10}")

    out = ROOT / "eval/outputs/session_internal_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "session": SESSION,
        "per_op": {"ariadne_search": ari_search, "ariadne_write": ari_write,
                   "mem0_add": mem0_add, "mem0_search": mem0_search},
        "session_model": model,
        "note": ("Ariadne search=目的推断+StoryRank；Ariadne write=maintain(triage+node+edge)抽取；"
                 "Mem0 add(infer)=抽取+增删改决策；Mem0/graphiti search=向量+BM25 非LLM；"
                 "嵌入模型 token 不计；Graphiti 另测"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}", flush=True)


if __name__ == "__main__":
    main()
