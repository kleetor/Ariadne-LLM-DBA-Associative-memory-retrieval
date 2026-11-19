# -*- coding: utf-8 -*-
"""0828 审计补算：stress 图 3 条查询上的 R@all（GT vs DBA 自动图）。

为什么要单独写：旧脚本 `memory/experiments/scripts/run_p0_retrieval_e2e.py` 仍用
`src.*` 导入与 SiliconFlow 旧接口，当前仓库跑不起来。本脚本用当前 `dba_pipeline`
重跑同一段 stress 对话，落一份可引用的 JSON。

口径：
- GT：`memory/data/formal/stress.yaml`（13 节点 / 12 边 / 3 条查询）。
- DBA：同一段对话经 `MemoryDBA.maintain()` 建图。
- R@all（原分母）：把 DBA 召回节点按 embedding 最近邻映射回 GT id（阈值 0.8），
  再与 GT 原始 expected 求交；避免旧脚本“只保留映射上的 expected”导致分母缩小。
- R@all（旧口径）：GT→DBA 映射后，用重映射 expected 计算，供与旧记录对照。
- 同时报告映射覆盖率；若覆盖低，R@all 只能作方向性证据。

用法：python -m eval.e0828_stress_rall
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from dba_pipeline.embedding.store import VectorStore  # noqa: E402
from dba_pipeline.extraction.dba import MemoryDBA  # noqa: E402
from dba_pipeline.extraction.graph_builder import GraphBuilder  # noqa: E402
from dba_pipeline.graph.memory_graph import MemoryGraph  # noqa: E402
from dba_pipeline.loader import load_all  # noqa: E402
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

STRESS = "memory/data/formal/stress.yaml"
OUT = "eval/outputs/audit/0828_stress_rall.json"

CONVERSATION = """user(10:00): 最近压力太大了
assistant: 听起来压力很大，发生了什么呢？
user(10:02): 经常加班到晚上十点，睡眠都不够了
assistant: 确实很辛苦，有没有什么放松的方式？
user(10:04): 喜欢喝咖啡提神，最爱拿铁了
assistant: 拿铁确实不错，有常去的咖啡店吗？
user(10:06): 公司附近有一家安静的咖啡馆，那里的音乐很舒缓
assistant: 挺好的，还有其他放松方式吗？
user(10:08): 有时会去公园散步，走一走感觉放松很多
assistant: 同事呢？工作上有伙伴吗？
user(10:10): 同事小王也经常一起加班，有时候会考虑换个工作"""


def _cos_matrix(embeddings, a_texts, b_texts):
    a = np.asarray(embeddings.embed_documents(a_texts), dtype="float32")
    b = np.asarray(embeddings.embed_documents(b_texts), dtype="float32")
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T


def _eval(retriever, queries):
    per_q = {}
    total_expected = 0
    total_hit = 0
    for text, info in queries.items():
        expected = set(info.get("expected") or [])
        if not expected:
            continue
        result = retriever.retrieve(text, seed_k=5, max_hops=5)
        retrieved = {mid for mid, _ in result.get("peak_memories", [])}
        hit = len(expected & retrieved)
        total_expected += len(expected)
        total_hit += hit
        per_q[text] = {
            "expected": sorted(expected),
            "retrieved": sorted(retrieved),
            "n_expected": len(expected),
            "hit": hit,
            "recall": round(hit / len(expected), 4),
        }
    return {
        "r_all": round(total_hit / total_expected, 4) if total_expected else None,
        "hit": total_hit,
        "expected": total_expected,
        "per_query": per_q,
    }


def main() -> None:
    _load_dotenv(ROOT / ".env")
    gt_graph, gt_queries = load_all(str(ROOT / STRESS))

    # 用当前 build_retriever 复用 embedding / LLM / inference；GT 图 13 节点，缓存很小
    _, gt_retriever = build_retriever(
        str(ROOT / STRESS), vector_cache_key="e0828_stress_gt")
    embeddings = gt_retriever.embeddings

    print(f"[0828] GT nodes={gt_graph.node_count} edges={gt_graph.edge_count} "
          f"queries={len(gt_queries)} emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)
    gt_eval = _eval(gt_retriever, gt_queries)

    # DBA 自动建图
    dba_graph = MemoryGraph()
    dba_vs = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    builder = GraphBuilder(dba_graph, dba_vs)
    dba = MemoryDBA(llm=gt_retriever.llm, graph=dba_graph,
                    vector_store=dba_vs, graph_builder=builder)
    res = dba.maintain(CONVERSATION)
    print(f"[0828] DBA nodes={dba_graph.node_count} edges={dba_graph.edge_count} "
          f"created={builder.stats.get('nodes_created')} errors={len(res.get('result', {}).get('errors', []))}",
          flush=True)

    dba_retriever = PurposeDrivenRetriever(
        llm=gt_retriever.llm, embeddings=embeddings, graph=dba_graph,
        vector_store=dba_vs, inference=gt_retriever.inference)

    # 映射：GT→DBA（旧口径）与 DBA→GT（原分母口径）
    gt_ids = [n for n in gt_graph.graph.nodes()]
    dba_ids = [n for n in dba_graph.graph.nodes()]
    gt_texts = [gt_graph.get_content(i) or "" for i in gt_ids]
    dba_texts = [dba_graph.get_content(i) or "" for i in dba_ids]
    sim = _cos_matrix(embeddings, gt_texts, dba_texts)  # gt × dba

    gt_to_dba = {}
    for i, gid in enumerate(gt_ids):
        j = int(np.argmax(sim[i]))
        if float(sim[i][j]) > 0.8:
            gt_to_dba[gid] = dba_ids[j]
    dba_to_gt = {}
    for j, did in enumerate(dba_ids):
        i = int(np.argmax(sim[:, j]))
        if float(sim[i][j]) > 0.8:
            dba_to_gt[did] = gt_ids[i]

    coverage = len(gt_to_dba) / len(gt_ids) if gt_ids else None
    print(f"[0828] mapping GT→DBA {len(gt_to_dba)}/{len(gt_ids)} "
          f"({coverage:.1%}) DBA→GT {len(dba_to_gt)}/{len(dba_ids)}", flush=True)

    # 旧口径：remap expected 后算 DBA R@all
    remapped = {}
    for text, info in gt_queries.items():
        exp = [gt_to_dba[e] for e in (info.get("expected") or []) if e in gt_to_dba]
        if exp:
            remapped[text] = {"expected": exp}
    dba_eval_remapped = _eval(dba_retriever, remapped)

    # 原分母口径：召回 DBA 节点 → 最近 GT → 与原始 expected 求交
    orig_expected_total = 0
    orig_hit_total = 0
    per_q_orig = {}
    for text, info in gt_queries.items():
        expected = set(info.get("expected") or [])
        if not expected:
            continue
        result = dba_retriever.retrieve(text, seed_k=5, max_hops=5)
        retrieved_gt = {dba_to_gt[mid] for mid, _ in result.get("peak_memories", [])
                        if mid in dba_to_gt}
        hit = len(expected & retrieved_gt)
        orig_expected_total += len(expected)
        orig_hit_total += hit
        per_q_orig[text] = {
            "expected": sorted(expected), "retrieved_gt": sorted(retrieved_gt),
            "n_expected": len(expected), "hit": hit,
            "recall": round(hit / len(expected), 4),
        }

    payload = {
        "meta": {
            "experiment": "0828-audit stress R@all",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stress": STRESS,
            "n_queries": len(gt_queries),
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "llm": os.environ.get("OPENAI_MODEL"),
            "note": ("旧脚本 run_p0_retrieval_e2e 使用 src.* 旧接口，无法直接运行；"
                     "本脚本用当前 dba_pipeline 重跑同一 stress 对话。"
                     "原分母口径把 DBA 召回映射回 GT 后与原始 expected 求交，避免分母缩小；"
                     "旧口径同时保留，供与历史 100% 对照。"),
        },
        "gt": {"nodes": gt_graph.node_count, "edges": gt_graph.edge_count, **gt_eval},
        "dba": {"nodes": dba_graph.node_count, "edges": dba_graph.edge_count,
                "created": builder.stats.get("nodes_created"),
                "errors": len(res.get("result", {}).get("errors", [])),
                "r_all_original_denominator": round(orig_hit_total / orig_expected_total, 4)
                if orig_expected_total else None,
                "r_all_original_hit": orig_hit_total,
                "r_all_original_expected": orig_expected_total,
                "per_query_original": per_q_orig,
                "r_all_remapped": dba_eval_remapped["r_all"],
                "per_query_remapped": dba_eval_remapped["per_query"],
                "mapping_gt_to_dba": len(gt_to_dba),
                "mapping_dba_to_gt": len(dba_to_gt),
                "mapping_coverage": round(coverage, 4) if coverage is not None else None},
    }
    out = ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[0828] GT R@all={gt_eval['r_all']} ({gt_eval['hit']}/{gt_eval['expected']})", flush=True)
    print(f"[0828] DBA R@all(原分母)={payload['dba']['r_all_original_denominator']} "
          f"({orig_hit_total}/{orig_expected_total})", flush=True)
    print(f"[0828] DBA R@all(旧口径)={dba_eval_remapped['r_all']}", flush=True)
    print(f"[0828] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
