# -*- coding: utf-8 -*-
"""
T1: P 链路评测脚本（LiFEMem）

对 LiFEMem 图逐条跑 P 检索（目的驱动联想检索：跳转轴 + 目的回归 + 寻峰终止），
输出每条查询的候选集、输出量 N（供外部框架 top-N 对齐）与因果路径
（nodes + edges，供 StoryRank 故事化）。

用法:
    python -m eval.run_p_baseline
    python -m eval.run_p_baseline --queries data/LiFEMem_queries.yaml \
        --yaml data/LiFEMem.yaml --out eval/outputs/p_candidates.json

可复现性: LLM temperature=0；每次运行新建 PathTracker（无跨查询状态残留）。
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 避免 torch 与 FAISS 的 OpenMP 运行时冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 模型已本地缓存，禁止 HF 联网检查（否则每次初始化重试 SSL 耗时 ~20-30s）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml
from langchain_openai import ChatOpenAI

from dba_pipeline.loader import load_graph
from dba_pipeline.core.path_tracker import PathTracker
from dba_pipeline.llm.inference import InferenceEngine
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever
from dba_pipeline.embedding.store import VectorStore, OpenAIEmbeddings, LocalEmbeddings


def _load_dotenv(env_path: Path) -> None:
    """加载项目 .env（已存在的环境变量优先）"""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_VECTOR_CACHE = "eval/.cache/lifemem_vectors.pkl"   # 兼容旧引用；实际路径见 _seed_vector_cache


def _seed_vector_cache(vector_store: VectorStore, mem_ids, contents, chunk: int = 64,
                       cache_key: str = "lifemem") -> None:
    """预填充向量缓存（.pkl），避免每次评测启动重复 embedding 322 条（~90s→~3s）

    **分块写入**（0930）：`embed_documents` 会把整批塞进**一次** API 请求，而端点对单请求
    输入条数有上限；线上图最多几百节点、从没触发，但合成大图（3000 节点）会直接被拒。
    分块对既有小图行为无影响（只有一块）。

    `cache_key` 决定用哪个 pkl（0930）：合成图有 3000 节点，**必须**单独缓存，否则每次
    重跑都要重算 3000 条向量。默认仍是 `lifemem`，行为不变。
    """
    cache_file = ROOT / f"eval/.cache/{cache_key}_vectors.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        id_set = set(mem_ids)
        for mid, vec in cached.items():
            if mid in id_set:
                vector_store._content_vectors[mid] = vec
    vector_store._contents = {mid: c for mid, c in zip(mem_ids, contents)}
    for i in range(0, len(mem_ids), chunk):
        vector_store.add_memories(mem_ids[i:i + chunk], contents[i:i + chunk])
    # 命中缓存时 `add_memories` 会把所有 id 视为“已在库”，不写 FAISS → store 仍为 None，
    # 后续向量检索全部返回空（表现为 PAR 每跳 0 候选）。缓存命中后必须从权威映射重建索引。
    if vector_store.store is None and vector_store._content_vectors:
        vector_store._rebuild_index()
    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(vector_store._content_vectors, f)


def build_retriever(yaml_path: str, llm_model: str = None, temperature: float = 0.0,
                    purpose_weight_coef: float = None, peak_tolerance: float = None,
                    peak_tolerance_ratio: float = None, vector_cache_key: str = "lifemem"):
    """构建完整 P 链路（与 mcp_server 初始化一致）

    purpose_weight_coef/peak_tolerance/peak_tolerance_ratio 为参数实验透传（None 用默认值）。
    """
    graph = load_graph(yaml_path)

    embedding_local = os.environ.get("EMBEDDING_LOCAL", "").strip().lower() in ("1", "true", "yes", "on")
    emb_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    if embedding_local:
        embeddings = LocalEmbeddings(model_name=emb_model)
    else:
        embeddings = OpenAIEmbeddings(
            api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            api_base=os.environ.get("EMBEDDING_API_BASE") or os.environ.get("OPENAI_API_BASE") or "",
            model=emb_model,
        )

    vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    # 启动时把图节点灌入向量存储（FAISS 不支持持久化，用向量缓存避免重复 embedding）
    if vector_store.store is None:
        mem_ids = [nid for nid in graph.graph.nodes() if graph.graph.nodes[nid].get("content")]
        contents = [graph.graph.nodes[nid].get("content") for nid in mem_ids]
        if mem_ids:
            _seed_vector_cache(vector_store, mem_ids, contents, cache_key=vector_cache_key)

    llm = ChatOpenAI(
        model=llm_model or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=temperature,
        max_tokens=16384,
        # 单次请求超时：不设就吃 openai SDK 默认的 600s，网关挂死时整条离线跑批会卡住
        # （0930 实测挂死 >10 分钟）。与 alm/engine.py 同口径。
        timeout=120.0,
        extra_body={"thinking": {"type": "disabled"}},  # 关 thinking：content 直出完整 JSON（api.gpt.ge 实测必需）
    )
    inference = InferenceEngine(llm)
    path_tracker = PathTracker()
    retriever = PurposeDrivenRetriever(
        llm=llm,
        embeddings=embeddings,
        graph=graph,
        vector_store=vector_store,
        inference=inference,
        path_tracker=path_tracker,
        purpose_weight_coef=purpose_weight_coef if purpose_weight_coef is not None else 0.5,
    )
    if peak_tolerance is not None:
        retriever.peak_finder.peak_tolerance = peak_tolerance
    # 0930：相对带宽透传（>0 时 δ = ratio × (peak_mean − min 轮均分)，随场景自适应）
    if peak_tolerance_ratio is not None:
        retriever.peak_tolerance_ratio = peak_tolerance_ratio
    return graph, retriever


def build_path(retriever: PurposeDrivenRetriever, result: dict) -> dict:
    """复用 retrieve_with_story 的路径构建逻辑（nodes + edges），供 StoryRank"""
    hop_history = result.get("hop_history", [])
    result_ids = [mid for mid, _ in result.get("peak_memories", [])]
    core_groups = retriever.select_core_nodes(hop_history, result_ids)
    all_nodes = []
    seen = set()
    for group in core_groups:
        for nid in group:
            if nid not in seen:
                seen.add(nid)
                all_nodes.append(nid)
    if not all_nodes:
        return {"nodes": [], "edges": []}
    return retriever._build_path(all_nodes, hop_history)


def main() -> None:
    parser = argparse.ArgumentParser(description="T1: P 链路评测")
    parser.add_argument("--queries", default="data/LiFEMem_queries.yaml", help="查询集 YAML")
    parser.add_argument("--yaml", default="data/LiFEMem.yaml", help="图谱 YAML")
    parser.add_argument("--out", default="eval/outputs/p_candidates.json", help="输出 JSON 路径")
    parser.add_argument("--llm-model", default=None, help="LLM 模型名（默认读 .env OPENAI_MODEL）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--purpose-mode", choices=["infer", "inject"], default="infer",
                        help="目的来源：infer=独立推断（默认）；inject=人工标注注入（模拟主 LLM）")
    parser.add_argument("--purposes", default="data/LiFEMem_purposes.yaml", help="目的标注文件")
    args = parser.parse_args()

    random.seed(args.seed)

    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever(args.yaml, llm_model=args.llm_model)

    with open(ROOT / args.queries, "r", encoding="utf-8") as f:
        qs = yaml.safe_load(f)
    queries = qs["queries"]

    inject_purposes = {}
    if args.purpose_mode == "inject":
        with open(ROOT / args.purposes, "r", encoding="utf-8") as f:
            inject_purposes = yaml.safe_load(f)
        print(f"[T1] purpose-mode=inject（人工标注注入 {len(inject_purposes)} 条）")

    results = []
    for q in queries:
        qid = q["id"]
        query = q["text"]
        purpose = inject_purposes.get(qid) if args.purpose_mode == "inject" else None
        res = retriever.retrieve(query, purpose=purpose)
        peak = res.get("peak_memories", [])
        scores = res.get("peak_scores", {})
        candidates = [
            {"id": mid, "content": content, "score": round(float(scores.get(mid, 0.0)), 4)}
            for mid, content in peak
        ]
        path = build_path(retriever, res)
        results.append({
            "query_id": qid,
            "category": q.get("category", q.get("scenario", "")),
            "query": query,
            "purpose": res.get("purpose", {}),
            "candidates": candidates,
            "total_candidates": len(candidates),
            "path": path,
        })
        print(f"[T1] {qid} ({q.get('category')}) -> {len(candidates)} 候选")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"meta": {"dataset": args.yaml, "queries": args.queries}, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[T1] 输出: {out_path}")
    print(f"[T1] 完成 {len(results)} 条查询")


if __name__ == "__main__":
    main()
