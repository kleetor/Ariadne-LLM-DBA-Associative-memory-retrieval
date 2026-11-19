# -*- coding: utf-8 -*-
"""探针：两套 embedding 的余弦尺度差异 → PAR 绝对阈值是否需要重标定。

背景：PAR 有两条**绝对阈值**判据——
  1) 目的过滤 `purpose_filter_threshold=0.30`（purpose_score < 0.30 的候选直接丢）
  2) 峰值容忍带 `peak_tolerance=0.10`（轮均分 μ ≥ peak − 0.10）
它们都作用在「余弦」上。换 embedding 会把余弦分布整体平移/压缩，绝对阈值随之失真：
阈值偏松 → 过滤形同虚设；偏紧 → 误杀正确节点。

本脚本量化：
  - 节点-节点余弦分布（同图不同模型的尺度）
  - query→节点 top1/top5/top20 余弦
  - purpose→节点余弦分布与 τ=0.30 的通过率
  - 各轮 μ 的极差（δ=0.10 相对于轮间差异是宽还是窄）

用法:
    python -m eval.probe_embedding_scale
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from dba_pipeline.embedding.store import OpenAIEmbeddings  # noqa: E402
from dba_pipeline.loader import load_graph  # noqa: E402
from eval.run_p_baseline import _load_dotenv  # noqa: E402

GOLD = "data/dba_eval/comprehensive_gold.yaml"
QUERIES = "data/dba_eval/comprehensive_queries.yaml"
E3_CTX = "eval/outputs/e3/context_e3_2x2_noedge.json"
MODELS = ("BAAI/bge-m3", "BAAI/bge-large-zh-v1.5")
TAU = 0.30          # purpose_filter_threshold
DELTA = 0.10        # peak_tolerance


def _pct(xs: List[float]) -> str:
    a = np.asarray(xs, dtype="float64")
    return (f"n={len(a)} mean={a.mean():.3f} p10={np.percentile(a, 10):.3f} "
            f"p50={np.percentile(a, 50):.3f} p90={np.percentile(a, 90):.3f} max={a.max():.3f}")


def _embed_nodes(model: str, ids: List[str], contents: List[str]) -> Dict[str, np.ndarray]:
    """节点向量：bge-m3 复用 E3 的 pkl，其余模型一次性嵌入后落盘。"""
    cache = ROOT / f"eval/.cache/comprehensive_{model.split('/')[-1]}_vectors.pkl"
    if model == "BAAI/bge-m3":
        src = ROOT / "eval/.cache/comprehensive_vectors.pkl"
        if src.exists():
            with open(src, "rb") as f:
                return pickle.load(f)
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    emb = OpenAIEmbeddings(
        api_key=os.environ.get("EMBEDDING_API_KEY", ""),
        api_base=os.environ.get("EMBEDDING_API_BASE", ""),
        model=model,
    )
    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(ids), 32):
        vecs = emb.embed_documents(contents[i:i + 32])
        out.update({mid: np.asarray(v, dtype="float32") for mid, v in zip(ids[i:i + 32], vecs)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def _purposes_of(qid: str, e3: Dict[str, dict]) -> List[str]:
    raw = (e3.get(qid) or {}).get("stats", {}).get("purposes") or "[]"
    try:
        return [str(p) for p in (ast.literal_eval(raw) if isinstance(raw, str) else list(raw))]
    except (ValueError, SyntaxError):
        return []


def hop_probe(n_queries: int) -> None:
    """对比两套 embedding 下的“各跳 μ 极差 / 容忍带轮数 / 输出节点数”。

    目的固定为 E3 跑出来的 purposes（注入），排除 LLM 推断方差；只变 embedding 与阈值。
    """
    from eval.run_p_baseline import build_retriever

    with open(ROOT / QUERIES, encoding="utf-8") as f:
        queries = [q for q in yaml.safe_load(f)["queries"] if q.get("category") != "abstention"]
    with open(ROOT / E3_CTX, encoding="utf-8") as f:
        e3 = {r["query_id"]: r for r in json.load(f)["queries"]}
    queries = [q for q in queries if _purposes_of(q["id"], e3)][:n_queries]

    configs = [("τ=0.30, δ=0.10(abs)", 0.30, None),
               ("τ=0.50, δ=0.10(abs)", 0.50, None),
               ("τ=0.30, δ=0.5×极差", 0.30, 0.5),
               ("τ=0.50, δ=0.5×极差", 0.50, 0.5)]

    for model in MODELS:
        os.environ["EMBEDDING_MODEL"] = model
        os.environ["EMBEDDING_LOCAL"] = "0"
        cache_key = "comprehensive" if model.endswith("bge-m3") else f"comprehensive_{model.split('/')[-1]}"
        _, retriever = build_retriever(str(ROOT / GOLD), vector_cache_key=cache_key)
        print(f"===== {model}（{len(queries)} 条，目的注入）=====", flush=True)
        for name, tau, ratio in configs:
            retriever.purpose_filter_threshold = tau
            retriever.peak_tolerance_ratio = ratio or 0.0
            spreads, nodes, rounds, band_ratios = [], [], [], []
            for q in queries:
                out = retriever.retrieve_with_story(
                    str(q["text"]), story=False, with_response=False,
                    purpose=_purposes_of(q["id"], e3),
                )
                mus = [float(h.get("mean_score", 0)) for h in out["hop_history"]]
                spreads.append(max(mus) - min(mus) if mus else 0.0)
                nodes.append(len((out.get("path") or {}).get("nodes") or []))
                rounds.append(len(mus))
                band_ratios.append(float(out["peak_band"].get("band_ratio") or 0.0))
            print(f"  {name:<20} 轮数均值={np.mean(rounds):.2f} "
                  f"μ极差均值={np.mean(spreads):.3f} "
                  f"输出节点均值={np.mean(nodes):.1f} "
                  f"带内占比={np.mean(band_ratios):.2f}", flush=True)
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="embedding 余弦尺度与 PAR 阈值标定探针")
    parser.add_argument("--hop", type=int, default=0, help="额外跑前 N 条 query 的跳级阈值对比（0=不跑）")
    parser.add_argument("--only-hop", action="store_true", help="只跑跳级阈值对比，跳过尺度统计")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    if args.only_hop:
        hop_probe(args.hop or 8)
        return
    graph = load_graph(str(ROOT / GOLD))
    ids = [nid for nid in graph.graph.nodes() if graph.graph.nodes[nid].get("content")]
    contents = [str(graph.graph.nodes[nid].get("content") or "") for nid in ids]

    with open(ROOT / QUERIES, encoding="utf-8") as f:
        queries = yaml.safe_load(f)["queries"]
    with open(ROOT / E3_CTX, encoding="utf-8") as f:
        e3 = {r["query_id"]: r for r in json.load(f)["queries"]}

    print(f"节点 {len(ids)}  查询 {len(queries)}  τ={TAU}  δ={DELTA}\n")
    for model in MODELS:
        print(f"===== {model} =====")
        vecs = _embed_nodes(model, ids, contents)
        missing = [i for i in ids if i not in vecs]
        if missing:
            print(f"  警告：{len(missing)} 个节点缺向量")
        mat = np.stack([vecs[i] for i in ids])
        mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)

        # 1) 节点-节点余弦（图内内容同质度）
        sim = mat @ mat.T
        off = sim[~np.eye(len(ids), dtype=bool)]
        print("  节点×节点 :", _pct(off.tolist()))

        # 2) query → 节点
        emb = OpenAIEmbeddings(
            api_key=os.environ.get("EMBEDDING_API_KEY", ""),
            api_base=os.environ.get("EMBEDDING_API_BASE", ""),
            model=model,
        )
        top1, top5, top20 = [], [], []
        for q in queries:
            qv = np.asarray(emb.embed_query(str(q["text"])), dtype="float32")
            qv = qv / max(np.linalg.norm(qv), 1e-12)
            sc = np.sort(mat @ qv)[::-1]
            top1.append(float(sc[0]))
            top5.append(float(sc[:5].mean()))
            top20.append(float(sc[:20].mean()))
        print("  query→top1:", _pct(top1))
        print("  query→top5:", _pct(top5))
        print("  query→top20:", _pct(top20))

        # 3) purpose → 节点（τ=0.30 的通过率）
        purpose_scores = []
        for q in queries:
            stats = (e3.get(q["id"]) or {}).get("stats", {})
            raw = stats.get("purposes") or "[]"
            try:
                purposes = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
            except (ValueError, SyntaxError):
                purposes = []
            if not purposes:
                continue
            pv = np.asarray(emb.embed_query("，".join(map(str, purposes))), dtype="float32")
            pv = pv / max(np.linalg.norm(pv), 1e-12)
            purpose_scores.extend((mat @ pv).tolist())
        if purpose_scores:
            arr = np.asarray(purpose_scores)
            print("  purpose→节点:", _pct(purpose_scores))
            print(f"    通过 τ={TAU}: {float((arr >= TAU).mean()):.3f}"
                  f"   ≥0.4: {float((arr >= 0.4).mean()):.3f}"
                  f"   ≥0.5: {float((arr >= 0.5).mean()):.3f}")
        print()

    if args.hop:
        hop_probe(args.hop)


if __name__ == "__main__":
    main()
