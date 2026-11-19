# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
系统「内部 LLM 调用」token 随记忆库/图谱规模增长趋势测量（主环境：Ariadne + Mem0）。

复用 session_internal_cost.py 的方法：
- Ariadne search = infer_purpose + StoryRank（retrieve_with_story）；
- Ariadne write = MemoryDBA.maintain（triage + 节点 + 边连接）；
- Mem0 add(infer=True) = 抽取 + 增/删/改决策（包裹 openai client）；
- Mem0 search = 向量+BM25 非 LLM（应为 0）。

规模轴：数据集内容条数 N ∈ {50, 100, 200}（同源递增，见 frameworks/scale_dataset.py）。
对每个 N，测「单次 operation token」，并按「6 轮 + 1 次换话题」会话模型聚合
（Ariadne search 2 次 + 每轮 write N... 注：write 每轮 6 次）。

过程要点：
- Ariadne 在同规模图谱上做 search 与 maintain；
- Mem0 先把图内容按 infer=False 预载入（形成规模 N 的库），再测一次 infer=True 的 add
  （此时库已有 N 条上下文，add 的 infer 模块会把已有记忆带进 prompt → 体现规模效应）。

  python eval/scale_internal_cost.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_openai import ChatOpenAI  # noqa: E402

SCALES = [50, 100, 200]
SESSION = {"n_turns": 6, "topic_switch_turn": 3}
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
        self.prompt = self.completion = self.calls = 0

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


def _graph_contents(n_scale: int) -> dict:
    """读 scale_N yaml 的节点 → {id: content}"""
    with open(ROOT / f"data/dba_eval/scale_{n_scale}.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {n["id"]: n["content"] for n in data["nodes"]}


def measure_ariadne(rec: Recorder, n_scale: int) -> dict:
    """在同规模图谱上测 search 与 write 单次 token。"""
    from eval.run_p_baseline import build_retriever
    from dba_pipeline.extraction.dba import MemoryDBA
    from dba_pipeline.extraction.graph_builder import GraphBuilder
    from dba_pipeline.embedding.store import VectorStore
    import threading

    graph_path = f"data/dba_eval/scale_{n_scale}.yaml"
    graph, retriever = build_retriever(graph_path)

    # 测新一轮对话写入（写入内容固定为一条短倾诉，**只变图规模**，隔离"对话长度"变量）
    raw_conv = ("[2026-03-01 20:00] 用户：最近项目排得太满了，天天加班到晚上11点，"
                "颈椎又酸又僵，感觉整个人都被掏空了……\n"
                "[2026-03-01 20:05] 助手：听起来你最近真的被工作压得喘不过气，身体也开始报警了。")

    # ---- search-op ----
    rec.reset()
    with rec:
        out = retriever.retrieve_with_story(QUERIES[0], with_response=False)
    stories = out.get("stories", [])
    story_len = len(stories[0]) if stories else 0
    search = {"prompt_tokens": rec.prompt, "completion_tokens": rec.completion,
              "llm_calls": rec.calls, "story_chars": story_len}

    # ---- write-op (maintain) ----
    vs = VectorStore(embeddings=None, backend="faiss", persist_dir=None)
    db_builder = GraphBuilder(graph=graph, vector_store=vs, lock=threading.Lock())
    dba = MemoryDBA(llm=retriever.inference.llm, graph=graph, vector_store=vs, graph_builder=db_builder)
    rec.reset()
    try:
        with rec:
            dba.maintain(raw_conv)
    except Exception as e:
        print(f"  (maintain apply_ops 异常，忽略以保留抽取 token 计数): {type(e).__name__}", flush=True)
    write = {"prompt_tokens": rec.prompt, "completion_tokens": rec.completion, "llm_calls": rec.calls}

    return {"search": search, "write": write}


def measure_mem0(n_scale: int) -> dict:
    """测 Mem0 在规模 N 库上下文下的 add(infer=True) 与 search token。"""
    import frameworks.mem0_runner_zh_patch as zh
    zh.apply_patch()
    from mem0 import Memory
    from dba_pipeline.embedding.store import local_model_path

    id2content = _graph_contents(n_scale)
    mem_dir = ROOT / f"eval/.cache/mem0_scale_{n_scale}"
    if mem_dir.exists():
        shutil.rmtree(mem_dir, ignore_errors=True)
    cfg = {
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": f"scale_{n_scale}", "path": str(mem_dir),
            "embedding_model_dims": 1024, "on_disk": True}},
        "llm": {"provider": "openai", "config": {
            "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0.1,
            "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
            "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
        "embedder": {"provider": "huggingface", "config": {
            "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
    }
    mem = Memory.from_config(cfg)

    # 预载：把图内容全部以 infer=False 入库（形成规模 N 库，不动 LLM 计费点）
    for c in id2content.values():
        mem.add({"role": "user", "content": c}, user_id="scale", infer=False)

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

    # 测一次真实 add(infer=True)：此时库已有 N 条记忆，infer 会把现有记忆带进上下文
    mem.add({"role": "user", "content": QUERIES[0]}, user_id="scale", infer=True)
    add = {"prompt_tokens": tok.prompt, "completion_tokens": tok.completion, "llm_calls": tok.calls}

    # search：应为 0 LLM
    tok.prompt = tok.completion = tok.calls = 0
    mem.search(QUERIES[1], top_k=8, filters={"user_id": "scale"}, threshold=0.0)
    search = {"prompt_tokens": tok.prompt, "completion_tokens": tok.completion, "llm_calls": tok.calls}
    completions.create = orig_create
    return {"add": add, "search": search}


def session_model(per_op: dict) -> dict:
    """按「6 轮 + 1 次换话题」聚合（Ariadne search 2 次 + 每轮 write；Mem0 每轮 add）。"""
    n = SESSION["n_turns"]
    ari_searches = 2
    model = {
        "Ariadne": {
            "llm_calls": ari_searches * per_op["ariadne_search"]["llm_calls"] + n * per_op["ariadne_write"]["llm_calls"],
            "prompt_tokens": ari_searches * per_op["ariadne_search"]["prompt_tokens"] + n * per_op["ariadne_write"]["prompt_tokens"],
            "completion_tokens": ari_searches * per_op["ariadne_search"]["completion_tokens"] + n * per_op["ariadne_write"]["completion_tokens"],
            "searches": ari_searches, "writes": n,
        },
        "Mem0": {
            "llm_calls": n * per_op["mem0_add"]["llm_calls"],
            "prompt_tokens": n * per_op["mem0_add"]["prompt_tokens"],
            "completion_tokens": n * per_op["mem0_add"]["completion_tokens"],
            "writes": n,
        },
    }
    for name, d in model.items():
        d["total_tokens"] = d["prompt_tokens"] + d["completion_tokens"]
    return model


def main() -> None:
    _load_dotenv()
    rec = Recorder()
    results = {}
    for n_scale in SCALES:
        print(f"\n{'=' * 66}\n[scale={n_scale}]", flush=True)
        ari = measure_ariadne(rec, n_scale)
        mem0 = measure_mem0(n_scale)
        per_op = {
            "ariadne_search": ari["search"], "ariadne_write": ari["write"],
            "mem0_add": mem0["add"], "mem0_search": mem0["search"],
        }
        s = session_model(per_op)
        results[n_scale] = {"per_op": per_op, "session_model": s}
        print(f"  Ariadne search: llm={ari['search']['llm_calls']} in={ari['search']['prompt_tokens']} "
              f"out={ari['search']['completion_tokens']} total={ari['search']['prompt_tokens']+ari['search']['completion_tokens']} "
              f"story_chars={ari['search']['story_chars']}", flush=True)
        print(f"  Ariadne write:  llm={ari['write']['llm_calls']} in={ari['write']['prompt_tokens']} "
              f"out={ari['write']['completion_tokens']} total={ari['write']['prompt_tokens']+ari['write']['completion_tokens']}", flush=True)
        print(f"  Mem0 add:       llm={mem0['add']['llm_calls']} in={mem0['add']['prompt_tokens']} "
              f"out={mem0['add']['completion_tokens']} total={mem0['add']['prompt_tokens']+mem0['add']['completion_tokens']}", flush=True)
        print(f"  Mem0 search:    llm={mem0['search']['llm_calls']} total={mem0['search']['prompt_tokens']+mem0['search']['completion_tokens']} (应为0)", flush=True)
        print("  session_model:", json.dumps(s, ensure_ascii=False), flush=True)

    out = ROOT / "eval/outputs/scale_internal_cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "session": SESSION, "scales": SCALES, "results": results,
        "note": ("Ariadne search=目的推断+StoryRank；Ariadne write=maintain(triage+node+edge)抽取；"
                 "Mem0 add(infer)=抽取+增删改决策（预载 N 条后测，体现规模上下文）；"
                 "Mem0 search=向量+BM25 非LLM；嵌入模型 token 不计；Graphiti 另测(.venv_graphiti)"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}", flush=True)


if __name__ == "__main__":
    main()
