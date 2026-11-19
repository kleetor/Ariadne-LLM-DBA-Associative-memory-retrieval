import os, sys, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["EMBEDDING_LOCAL"] = "1"
from eval.run_p_baseline import _load_dotenv, build_retriever
_load_dotenv(ROOT / ".env")
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["EMBEDDING_LOCAL"] = "1"

GRAPH = "data/dba_eval/smoke_companion.yaml"
QUERIES = [
    ("q1", "最近项目排得太满了，天天加班到十一二点，感觉整个人都被掏空了……", "work_pressure"),
    ("q2", "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？", "health"),
    ("q3", "感觉压力好大，不知道怎么熬过这段日子……", "emotion"),
]
TOP_K = 5

from dba_pipeline.embedding.store import local_model_path
from frameworks import naive_rag

graph, retriever = build_retriever(GRAPH)
id2content = naive_rag.build_memories(GRAPH)
rag_store = naive_rag.build_store(id2content)

import frameworks.mem0_runner_zh_patch as zh
zh.apply_patch()
from mem0 import Memory
mem_dir = ROOT / "eval/.cache/mem0_qdrant_zh"
if mem_dir.exists(): shutil.rmtree(mem_dir, ignore_errors=True)
cfg = {
    "vector_store": {"provider": "qdrant", "config": {
        "collection_name": "smoke_cmp", "path": str(ROOT / "eval/.cache/mem0_qdrant_zh"),
        "embedding_model_dims": 1024, "on_disk": True}},
    "llm": {"provider": "openai", "config": {
        "model": os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"), "temperature": 0,
        "api_key": os.environ.get("OPENAI_API_KEY") or "not-needed",
        "openai_base_url": os.environ.get("OPENAI_API_BASE")}},
    "embedder": {"provider": "huggingface", "config": {
        "model": local_model_path("BAAI/bge-large-zh-v1.5"), "embedding_dims": 1024}},
}
mem = Memory.from_config(cfg)
for c in id2content.values():
    mem.add({"role": "user", "content": c}, user_id="cmp", infer=False)

def par_raw(q):
    res = retriever.retrieve(q)
    return " ".join(c for _, c in res.get("peak_memories", [])[:TOP_K])

def rag_raw(q):
    hits = rag_store.search(q, k=TOP_K)
    return " ".join(id2content.get(mid, mid) for mid, _, _ in hits)

def mem_raw(q):
    out = mem.search(q, top_k=TOP_K, filters={"user_id": "cmp"}, threshold=0.0)
    return " ".join(str(x.get("memory", "")) for x in out.get("results", []))

out_data = {}
for qid, query, scen in QUERIES:
    print("\n" + "=" * 70)
    print(f"[{qid} · {scen}] {query}")
    for name, fn in [("PAR", par_raw), ("Naive RAG", rag_raw), ("Mem0(zh)", mem_raw)]:
        raw = fn(query)
        out_data.setdefault(name, {})[qid] = raw
        print(f"\n  ── {name} ──")
        print("  " + raw.replace("\n", " "))

oj = ROOT / "eval/outputs/smoke_retrieved_raw.json"
oj.parent.mkdir(parents=True, exist_ok=True)
oj.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n\n原文已写入:", oj)
