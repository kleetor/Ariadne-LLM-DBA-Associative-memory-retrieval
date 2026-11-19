# -*- coding: utf-8 -*-
"""
DBA 抽取环节评测（LiFEMem 子集闭环，仅抽取图质量对比）

流程：对话语料（data/dba_eval/lifemem_subset_dialogs.yaml，100 节点还原为 15 批多轮对话）
  → 空图逐批 MemoryDBA.maintain() 自动建图
  → 与 gold 子图（LiFEMem.yaml 按 gold_nodes 提取）对比：
      节点匹配（embedding 余弦贪心匹配）→ 节点召回/精确、类型正确率
      边匹配（gold 边两端节点均匹配且有 DBA 边）→ 边召回/精确

用法:
    python -m eval.run_dba_extraction_eval
    python -m eval.run_dba_extraction_eval --out eval/outputs/dba_extraction_subset.json

可复现性: LLM temperature=0。
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import yaml
from langchain_openai import ChatOpenAI

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.embedding.store import VectorStore, OpenAIEmbeddings, LocalEmbeddings
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.extraction.dba import MemoryDBA
from dba_pipeline.loader import load_graph


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


def build_llm(llm_model: str = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=llm_model or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=16384,
        extra_body={"thinking": {"type": "disabled"}},  # 关 thinking：content 直出完整 JSON（实测必需）
    )


def build_embeddings():
    embedding_local = os.environ.get("EMBEDDING_LOCAL", "").strip().lower() in ("1", "true", "yes", "on")
    emb_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    if embedding_local:
        return LocalEmbeddings(model_name=emb_model)
    return OpenAIEmbeddings(
        api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
        api_base=os.environ.get("EMBEDDING_API_BASE") or os.environ.get("OPENAI_API_BASE") or "",
        model=emb_model,
    )


# ---- 匹配 ----

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_nodes(gold: list, dba: list, embed_fn, threshold: float = 0.75) -> list:
    """embedding 余弦贪心最大匹配。gold/dba: [{id, content, type}]。返回 [(g, d, sim)]"""
    pairs = []
    g_vecs = [embed_fn(g["content"]) for g in gold]
    d_vecs = [embed_fn(d["content"]) for d in dba]
    for gi, gv in enumerate(g_vecs):
        for di, dv in enumerate(d_vecs):
            sim = _cosine(gv, dv)
            if sim >= threshold:
                pairs.append((sim, gi, di))
    pairs.sort(reverse=True)
    used_g, used_d = set(), set()
    matched = []
    for sim, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matched.append((gold[gi], dba[di], sim))
    return matched


# ---- 评估 ----

def evaluate(gold_graph: MemoryGraph, gold_node_ids: set, dba_graph: MemoryGraph, embed_fn, threshold: float) -> dict:
    # gold 活跃节点
    gold_nodes = []
    for nid in gold_node_ids:
        n = gold_graph.get_node(nid)
        if n and not n.get("deprecated") and not n.get("forgotten"):
            gold_nodes.append({"id": nid, "content": n["content"],
                               "type": n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])})

    # DBA 活跃节点
    dba_nodes = []
    for nid in dba_graph.graph.nodes():
        n = dba_graph.get_node(nid)
        if n and not n.get("deprecated") and not n.get("forgotten"):
            dba_nodes.append({"id": nid, "content": n["content"],
                              "type": n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])})

    # 节点匹配
    matched = match_nodes(gold_nodes, dba_nodes, embed_fn, threshold)
    matched_g_ids = {g["id"] for g, _, _ in matched}
    matched_d_ids = {d["id"] for _, d, _ in matched}
    node_recall = len(matched_g_ids) / len(gold_nodes) if gold_nodes else 0.0
    node_precision = len(matched_d_ids) / len(dba_nodes) if dba_nodes else 0.0
    type_correct = sum(1 for g, d, _ in matched if g["type"] == d["type"]) / len(matched) if matched else 0.0

    # gold 内部边（两端都在子集内，活跃）
    gold_edges = []
    for u, v, data in gold_graph.graph.edges(data=True):
        if u in gold_node_ids and v in gold_node_ids:
            nu = gold_graph.get_node(u)
            nv = gold_graph.get_node(v)
            if nu and nv and not nu.get("deprecated") and not nv.get("deprecated"):
                gold_edges.append((u, v, data.get("rel_type", "")))
    d_edge_ids = set()
    for a, b in dba_graph.graph.edges():
        d_edge_ids.add((a, b))
        d_edge_ids.add((b, a))

    g2d = {g["id"]: d["id"] for g, d, _ in matched}
    d2g = {d["id"]: g["id"] for g, d, _ in matched}

    # 边召回：gold 边两端均匹配，且 DBA 中有连接
    edge_hit = 0
    for u, v, rt in gold_edges:
        if u in g2d and v in g2d and (g2d[u], g2d[v]) in d_edge_ids:
            edge_hit += 1
    edge_recall = edge_hit / len(gold_edges) if gold_edges else 0.0

    # 边精确：DBA 边两端均匹配，且 gold 中有连接（否则为 DBA 自主新边）
    dba_edges = [(a, b, data.get("rel_type", "")) for a, b, data in dba_graph.graph.edges(data=True)]
    edge_prec_hit = 0
    new_edges = []
    for a, b, rt in dba_edges:
        if a in d2g and b in d2g:
            if (d2g[a], d2g[b]) in {(x, y) for x, y, _ in gold_edges} | {(y, x) for x, y, _ in gold_edges}:
                edge_prec_hit += 1
            else:
                new_edges.append((a, b, rt))
    edge_precision = edge_prec_hit / len(dba_edges) if dba_edges else 0.0

    # 边类型一致性（仅对双边均匹配且连接的边，比较 rel_type）
    edge_type_match = 0
    edge_type_total = 0
    gold_rt = {(x, y): t for x, y, t in gold_edges}
    gold_rt.update({(y, x): t for x, y, t in gold_edges})
    for a, b, rt in dba_edges:
        if a in d2g and b in d2g and (d2g[a], d2g[b]) in gold_rt:
            edge_type_total += 1
            gt = gold_rt[(d2g[a], d2g[b])]
            if str(gt).lower() == str(rt).lower():
                edge_type_match += 1
    edge_type_accuracy = edge_type_match / edge_type_total if edge_type_total else 0.0

    return {
        "threshold": threshold,
        "gold_nodes": len(gold_nodes),
        "dba_active_nodes": len(dba_nodes),
        "matched_nodes": len(matched),
        "node_recall": round(node_recall, 4),
        "node_precision": round(node_precision, 4),
        "type_accuracy": round(type_correct, 4),
        "gold_edges": len(gold_edges),
        "dba_edges": len(dba_edges),
        "edge_recall": round(edge_recall, 4),
        "edge_precision": round(edge_precision, 4),
        "edge_type_accuracy": round(edge_type_accuracy, 4),
        "dba_new_edges": len(new_edges),
        "dba_deprecated_nodes": sum(1 for nid in dba_graph.graph.nodes()
                                    if dba_graph.get_node(nid).get("deprecated")),
        "dba_forgotten_nodes": sum(1 for nid in dba_graph.graph.nodes()
                                   if dba_graph.get_node(nid).get("forgotten")),
        "dba_node_detail": [
            {
                "id": d["id"],
                "content": d["content"],
                "type": d["type"],
                "deprecated": bool(dba_graph.get_node(d["id"]).get("deprecated")),
                "forgotten": bool(dba_graph.get_node(d["id"]).get("forgotten")),
                "matched_gold_id": next((g["id"] for g, dd, _ in matched if dd["id"] == d["id"]), None),
            }
            for d in dba_nodes
        ],
        "unmatched_gold": [
            {"id": g["id"], "content": g["content"], "type": g["type"]}
            for g in gold_nodes if g["id"] not in matched_g_ids
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DBA 抽取环节评测（子集闭环）")
    parser.add_argument("--yaml", default="data/LiFEMem.yaml", help="gold 图谱 YAML")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_subset_dialogs.yaml", help="对话语料 YAML")
    parser.add_argument("--out", default="eval/outputs/dba_extraction_subset.json", help="输出 JSON 路径")
    parser.add_argument("--llm-model", default=None, help="LLM 模型名（默认读 .env OPENAI_MODEL）")
    parser.add_argument("--threshold", type=float, default=0.75, help="节点语义匹配余弦阈值")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="每批合并对话轮数（批量抽取：一次 DBA 维护内连边/图遍历；15 轮 → 6/6/3 三批）")
    parser.add_argument("--orphan-threshold", type=int, default=100,
                        help="孤立节点遗忘阈值（评测语料为一次性批量，默认调大以隔离遗忘机制干扰，专注抽取质量）")
    parser.add_argument("--save-graph", default="eval/outputs/dba_graph_subset.yaml",
                        help="DBA 建图结果导出路径（供检索联动评测复用，置空则跳过）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)
    _load_dotenv(ROOT / ".env")

    # 1. gold 子图
    gold_graph = load_graph(ROOT / args.yaml)
    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    gold_node_ids = set(data["subset"]["gold_nodes"])
    dialogs = data["dialogs"]

    # 2. 空图 + DBA
    embeddings = build_embeddings()
    vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    graph = MemoryGraph()
    builder = GraphBuilder(graph, vector_store, orphan_threshold=args.orphan_threshold)
    llm = build_llm(args.llm_model)
    dba = MemoryDBA(llm=llm, graph=graph, vector_store=vector_store, graph_builder=builder)

    # 3. 批量维护（每 batch_size 轮对话合并一批，一次边连接/结构分析）
    batch_size = args.batch_size
    groups = [dialogs[i:i + batch_size] for i in range(0, len(dialogs), batch_size)]
    batch_log = []
    for gi, group in enumerate(groups):
        turns_all = [t for dialog in group for t in dialog["turns"]]
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in turns_all)
        res = dba.maintain(conversation)
        ops = res.get("ops", {})
        result = res.get("result", {})
        node_ops_summary = [f"{op.get('action')}:{op.get('content', op.get('target_id', ''))[:40]}"
                            for op in ops.get("node_ops", [])]
        batch_log.append({
            "batch": gi + 1,
            "dialogs": [d["id"] for d in group],
            "turn_count": len(turns_all),
            "skipped": res.get("skipped", False),
            "node_ops": node_ops_summary,
            "edge_ops_count": len(ops.get("edge_ops", [])),
            "created": result.get("created_ids", []),
            "nodes_total": graph.node_count,
            "edges_total": graph.edge_count,
        })
        print(f"[DBA] 批{gi + 1}（{', '.join(d['id'] for d in group)}，{len(turns_all)}轮）-> "
              f"node_ops={len(node_ops_summary)} edge_ops={len(ops.get('edge_ops', []))} "
              f"created={len(result.get('created_ids', []))} "
              f"(累计 {graph.node_count} 节点 / {graph.edge_count} 边)")

    # 4. 评估
    embed_fn = lambda text: np.array(embeddings.embed_query(text))
    eval_result = evaluate(gold_graph, gold_node_ids, graph, embed_fn, args.threshold)

    out = {
        "meta": {
            "dataset": args.yaml,
            "dialogs": args.dialogs,
            "subset": data["subset"]["name"],
            "gold_nodes_total": len(gold_node_ids),
            "dialog_count": len(dialogs),
            "batch_size": batch_size,
            "batch_count": len(groups),
        },
        "eval": eval_result,
        "batch_log": batch_log,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4.5 导出 DBA 建图（供检索联动评测）
    graph_path = None
    if args.save_graph:
        graph_path = ROOT / args.save_graph
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(yaml.safe_dump(graph.to_dict(), allow_unicode=True, sort_keys=False),
                              encoding="utf-8")
        print(f"[DBA] 建图已导出: {graph_path}")

    e = eval_result
    print("\n===== DBA 抽取评测（子集）=====")
    print(f"节点: gold={e['gold_nodes']} DBA活跃={e['dba_active_nodes']} 匹配={e['matched_nodes']}")
    print(f"  召回率={e['node_recall']:.2%}  精确率={e['node_precision']:.2%}  类型正确率={e['type_accuracy']:.2%}")
    print(f"边: gold={e['gold_edges']} DBA={e['dba_edges']} (新边 {e['dba_new_edges']})")
    print(f"  召回率={e['edge_recall']:.2%}  精确率={e['edge_precision']:.2%}  边类型一致率={e['edge_type_accuracy']:.2%}")
    print(f"DBA 废弃节点={e['dba_deprecated_nodes']}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
