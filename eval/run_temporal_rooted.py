# -*- coding: utf-8 -*-
"""
TEMPORAL 专项 · 真实时序场景分型诊断（跨期 + 细时间）

诊断目标：用「查询类型(TA/TB/TC/TD/TE) × 图来源(gold/基线抽取/temporal抽取)」的
隔离矩阵跑全链路，定位真实时序场景下系统弱势落在哪一层、根本原因是什么。

  TA = 事件→时间（when did X）        TB = 时间→事件（what happened at X）
  TC = 期内时间线重建                TD = 跨期历程
  TE = 事实+时间背景（causal+when）

判据：
  · gold 图上某类失败 → 检索层根因（缺 th→事件反向/一次性规则/锚点角色判别）
  · gold 成功但抽取图失败 → 抽取层根因（产不出细时间节点/TEMPORAL 边）
  · TD 只在多 th 齐聚时成功 → 佐证跨期多 th 是结构收益来源，需目的门控放行

用法:
    python -m eval.run_temporal_rooted
    python -m eval.run_temporal_rooted --out eval/outputs/temporal_rooted.json
    python -m eval.run_temporal_rooted --smoke
"""

import argparse
import copy
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
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever
from dba_pipeline.llm.inference import InferenceEngine
from dba_pipeline.loader import load_graph
import dba_pipeline.extraction.dba as dba_mod
from dba_pipeline.core import jump_axis

# 复用 run_temporal_experiment 的通用工具
from eval.run_temporal_experiment import (
    _load_dotenv, build_llm, build_embeddings, active_nodes, match_items,
    evaluate_extraction, build_store_from_graph, _cosine, gold_edges_within,
    TEMPORAL_NODE_EXTRA, TEMPORAL_EDGE_EXTRA, TEMPORAL_EXAMPLE_EXTRA,
)

# ---- prompt 变体快照（run_temporal_experiment.main 未跑，需在此快照）----
_BASE_NODE = copy.deepcopy(dba_mod.NODE_EXTRACTION_PROMPT)
_BASE_EDGE = copy.deepcopy(dba_mod.EDGE_LINKING_PROMPT)
_BASE_EXAMPLE = copy.deepcopy(dba_mod.EDGE_LINKING_FEWSHOT_EXAMPLE)

DEFAULT_RULES = copy.deepcopy(jump_axis.JUMP_AXIS_RULES)
BOOST_RULES = copy.deepcopy(DEFAULT_RULES)
for _nt, _rel_map in BOOST_RULES.items():
    if jump_axis.RelationType.TEMPORAL in _rel_map:
        BOOST_RULES[_nt][jump_axis.RelationType.TEMPORAL] = (0.9, 0.5)


def set_prompt_mode(mode: str) -> None:
    if mode == "base":
        dba_mod.NODE_EXTRACTION_PROMPT = _BASE_NODE
        dba_mod.EDGE_LINKING_PROMPT = _BASE_EDGE
        dba_mod.EDGE_LINKING_FEWSHOT_EXAMPLE = _BASE_EXAMPLE
    elif mode == "temporal":
        dba_mod.NODE_EXTRACTION_PROMPT = _BASE_NODE + TEMPORAL_NODE_EXTRA
        dba_mod.EDGE_LINKING_PROMPT = _BASE_EDGE + TEMPORAL_EDGE_EXTRA
        dba_mod.EDGE_LINKING_FEWSHOT_EXAMPLE = _BASE_EXAMPLE + TEMPORAL_EXAMPLE_EXTRA
    else:
        raise ValueError(f"未知 prompt 模式: {mode}")


def set_jump_mode(mode: str) -> None:
    if mode == "default":
        jump_axis.JUMP_AXIS_RULES = DEFAULT_RULES
    elif mode == "temporal_boost":
        jump_axis.JUMP_AXIS_RULES = BOOST_RULES
    else:
        raise ValueError(f"未知跳转模式: {mode}")


# ---------------------------------------------------------------- 抽取（同 run_temporal_experiment 的 run_extraction/run_step）
def _run_step(dba, conversation):
    node_context = dba._build_node_context(conversation)
    node_ops = dba._parse_response(dba.node_chain.invoke(node_context).content).get("node_ops", [])
    result1 = dba.builder.apply_ops(node_ops, [])
    new_ids = result1.get("created_ids", [])
    edge_context = dba._build_edge_context(conversation, new_ids)
    edge_ops = dba._parse_response(dba.edge_chain.invoke(edge_context).content).get("edge_ops", [])
    result2 = dba.builder.apply_ops([], edge_ops)
    return {
        "node_ops": node_ops, "edge_ops": edge_ops,
        "created_ids": result1.get("created_ids", []),
        "errors": result1.get("errors", []) + result2.get("errors", []),
    }


def _triage_need_rate(dba, conversation, n=3):
    need = 0
    for _ in range(n):
        if dba._should_maintain(conversation):
            need += 1
    return round(need / n, 3)


def run_extraction(dialogs, llm, embeddings, batch_size, mode):
    set_prompt_mode(mode)
    vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    graph = MemoryGraph()
    builder = GraphBuilder(graph, vector_store, orphan_threshold=100)
    dba = dba_mod.MemoryDBA(llm=llm, graph=graph, vector_store=vector_store, graph_builder=builder)
    groups = [dialogs[i:i + batch_size] for i in range(0, len(dialogs), batch_size)]
    batch_log = []
    for gi, group in enumerate(groups):
        turns_all = [t for dialog in group for t in dialog["turns"]]
        conversation = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in turns_all)
        res = _run_step(dba, conversation)
        triage_need = _triage_need_rate(dba, conversation)
        batch_log.append({
            "batch": gi + 1, "turn_count": len(turns_all),
            "node_ops": len(res["node_ops"]), "edge_ops": len(res["edge_ops"]),
            "created": res["created_ids"], "errors": res["errors"],
            "triage_need_rate": triage_need,
            "nodes_total": graph.node_count, "edges_total": graph.edge_count,
        })
        print(f"[DBA-{mode}] 批{gi + 1}（{len(turns_all)}轮）node_ops={len(res['node_ops'])} "
              f"edge_ops={len(res['edge_ops'])} created={len(res['created_ids'])} "
              f"triage_need={triage_need} (累计 {graph.node_count} 节点 / {graph.edge_count} 边)")
    return graph, vector_store, batch_log


# ---------------------------------------------------------------- 时序重建（动态 gold_trank + 分型）
def event_detection(graph, store, llm, embeddings, queries, embed_fn, gold_events, gold_trank,
                    d2g=None, jump_mode="default"):
    set_jump_mode(jump_mode)
    inference = InferenceEngine(llm)
    retriever = PurposeDrivenRetriever(
        llm=llm, embeddings=embeddings, graph=graph, vector_store=store, inference=inference,
    )
    temporal_anchor = {}
    for u, v, data in graph.graph.edges(data=True):
        rt = data.get("rel_type", "")
        rt_val = rt.value if hasattr(rt, "value") else str(rt)
        if rt_val.lower() == "temporal":
            temporal_anchor[u] = v

    def to_gold(gid):
        return d2g.get(gid, gid) if d2g else gid

    results = []
    for q in queries:
        expected = q["gold_expected"]
        expected_set = set(expected)
        r = retriever.retrieve(q["query"], seed_k=5, purpose=q["purposes"])
        peak_ids = [mid for mid, _ in r["peak_memories"]]
        R = []
        seen = set()
        for mid in peak_ids:
            gw = to_gold(mid)
            if gw in expected_set and gw not in seen:
                seen.add(gw)
                anchor = temporal_anchor.get(mid) or temporal_anchor.get(gw)
                rank = None
                if anchor:
                    gw_time = to_gold(anchor)
                    if gw_time in gold_trank:
                        rank = gold_trank[gw_time]
                R.append((gw, rank))
        r_all = len(R) / len(expected) if expected else 0.0
        anchored = [(g, rk) for g, rk in R if rk is not None]
        anchorable = len(anchored) / len(R) if R else 0.0
        pair_acc = None
        if len(anchored) >= 2:
            gold_pos = {g: i for i, g in enumerate(expected)}
            correct = 0
            total = 0
            n = len(anchored)
            for i in range(n):
                for j in range(i + 1, n):
                    gi, rki = anchored[i]
                    gj, rkj = anchored[j]
                    if rki == rkj:
                        continue
                    total += 1
                    if (rki < rkj) == (gold_pos[gi] < gold_pos[gj]):
                        correct += 1
            pair_acc = round(correct / total, 4) if total else None
        results.append({
            "query_id": q["id"], "type": q.get("type", ""),
            "query": q["query"],
            "retrieved_ids": peak_ids, "retrieved_gold": [g for g, _ in R],
            "expected": expected, "R_at_all": round(r_all, 4),
            "anchorable": round(anchorable, 4), "timeline_pairwise_acc": pair_acc,
            "retrieved_count": len(peak_ids),
        })
    return results


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="TEMPORAL 真实时序场景分型诊断")
    parser.add_argument("--yaml", default="data/dba_eval/lifemem_crossperiod.yaml")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_crossperiod_dialogs.yaml")
    parser.add_argument("--queries", default="data/dba_eval/lifemem_crossperiod_queries.yaml")
    parser.add_argument("--out", default="eval/outputs/temporal_rooted.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="只跑抽取第 1 批")
    args = parser.parse_args()

    random.seed(42)
    _load_dotenv(ROOT / ".env")

    gold_graph = load_graph(ROOT / args.yaml)
    with open(ROOT / args.dialogs, "r", encoding="utf-8") as f:
        dialog_data = yaml.safe_load(f)
    gold_node_ids = set(dialog_data["subset"]["gold_nodes"])
    dialogs = dialog_data["dialogs"]
    if args.smoke:
        dialogs = dialogs[:1]
    with open(ROOT / args.queries, "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)

    embeddings = build_embeddings()
    embed_fn = lambda text: np.array(embeddings.embed_query(text))
    llm = build_llm(args.llm_model)

    # gold 事件集合 & 时间秩（动态：所有 id 形如 T\\d+ 的时间节点）
    gold_active = active_nodes(gold_graph)
    gold_events = {n["id"] for n in gold_active
                   if n["type"] in ("action", "status", "emotion", "reason")}
    gold_trank = {}
    for n in gold_active:
        nid = n["id"]
        if nid.startswith("T") and nid[1:].isdigit():
            gold_trank[nid] = int(nid[1:])
    gold_store = build_store_from_graph(gold_graph, embeddings)

    print("===== 阶段 1：DBA 抽取（base vs temporal）=====")
    graphs = {"gold": gold_graph}
    stores = {"gold": gold_store}
    extract = {}
    for mode in ("base", "temporal"):
        g, vs, blog = run_extraction(dialogs, llm, embeddings, args.batch_size, mode)
        graphs[f"dba_{mode}"] = g
        stores[f"dba_{mode}"] = vs
        extract[mode] = {"batch_log": blog, "nodes": g.node_count, "edges": g.edge_count,
                         "eval": evaluate_extraction(gold_graph, gold_node_ids, g, embed_fn, args.threshold)}
        gpath = ROOT / f"eval/outputs/temporal_rooted_graph_{mode}.yaml"
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gpath.write_text(yaml.safe_dump(g.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[抽取-{mode}] 节点={g.node_count} 边={g.edge_count} "
              f"边召回={extract[mode]['eval']['edge_recall']:.2%} "
              f"temporal边={extract[mode]['eval']['edge_by_rel'].get('temporal', {})} "
              f"sequence边={extract[mode]['eval']['edge_by_rel'].get('sequence', {})}")

    print("\n===== 阶段 2：事件时间线重建（gold / dba_base / dba_temporal × 默认/增强跳转）=====")
    g2d_maps = {}
    for mode in ("base", "temporal"):
        g = graphs[f"dba_{mode}"]
        gold_nodes = [n for n in active_nodes(gold_graph) if n["id"] in gold_node_ids]
        _, _, d2g = match_items(gold_nodes, active_nodes(g), embed_fn, args.threshold)
        g2d_maps[mode] = d2g

    detect = {}
    for gname, graph in graphs.items():
        detect[gname] = {}
        for jmode in ("default", "temporal_boost"):
            d2g = g2d_maps[gname.replace("dba_", "")] if gname.startswith("dba_") else None
            detect[gname][jmode] = event_detection(
                graph, stores[gname], llm, embeddings, queries, embed_fn,
                gold_events, gold_trank, d2g=d2g, jump_mode=jmode,
            )

    out = {
        "meta": {"dataset": args.yaml, "dialogs": args.dialogs, "queries": args.queries,
                 "batch_size": args.batch_size, "smoke": args.smoke,
                 "gold_nodes": len(gold_node_ids), "time_nodes": sorted(gold_trank, key=lambda x: gold_trank[x]),
                 "dba_prompts": ["base", "temporal"], "jump_modes": ["default", "temporal_boost"]},
        "extraction": extract, "event_detection": detect,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {out_path}")

    # 打印汇总（按类型分型）
    print("\n--- 抽取保真度汇总 ---")
    for mode in ("base", "temporal"):
        e = extract[mode]["eval"]
        print(f"[{mode}] 节点召回={e['node_recall']:.2%} 边召回={e['edge_recall']:.2%} "
              f"temporal={e['edge_by_rel'].get('temporal')} sequence={e['edge_by_rel'].get('sequence')}")
    print("\n--- 事件检测汇总（R@all / 可锚定率 / 时间线pairwise），按 图×跳转×类型 ---")
    by_type = {}
    for gname in detect:
        for jmode in detect[gname]:
            for r in detect[gname][jmode]:
                t = r["type"]
                key = (gname, jmode, t)
                by_type.setdefault(key, []).append(r)
    for (gname, jmode, t), rows in sorted(by_type.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        ra = [r["R_at_all"] for r in rows]
        an = [r["anchorable"] for r in rows if r["anchorable"] is not None]
        pa = [r["timeline_pairwise_acc"] for r in rows if r["timeline_pairwise_acc"] is not None]
        print(f"[{gname} | {jmode} | {t}] n={len(rows)} R@all均值={sum(ra)/len(ra):.2%} "
              f"可锚定={sum(an)/len(an):.2%} pairwise={'%.2f' % (sum(pa)/len(pa)) if pa else 'N/A'}")


if __name__ == "__main__":
    main()
