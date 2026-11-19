# -*- coding: utf-8 -*-
"""
TEMPORAL 时序关系专项 · 端到端闭环实验

回答两个问题：
  Q1（抽取保真度）：当前 DBA 两步拆分能否抽出 TEMPORAL / SEQUENCE 边？
                   加「时间节点引导 + TEMPORAL few-shot」后提升多少？
  Q2（事件检测效用）：同一批事件检测查询下，含 TEMPORAL 边的图（gold / 增强抽取）
                    vs 不含 TEMPORAL 边的图（基线抽取）在「事件时间线重建」上的差距，
                    从而验证「TEMPORAL 在事件检测当中有大用」。

评测流程：
  1. 新数据集 lifemem_temporal.yaml（gold 图：10 事件 + 7 时间锚点 + temporal/sequence/causal）
     → lifemem_temporal_dialogs.yaml（复现为多轮对话，含显式时间词）
     → lifemem_temporal_queries.yaml（事件检测查询 + 手写目的 + gold 期望时序）
  2. 抽取：空图 + MemoryDBA.maintain，mode=base / temporal 两版 prompt
  3. 抽取保真度：与 gold 对比，节点/边召回，边按 rel_type 拆分（重点 temporal/sequence）
  4. 事件时间线重建：在上述图上对每条查询跑 P 检索（注入目的），
     从 graph 读 TEMPORAL 边得到事件→时间秩，重建顺序 vs gold 期望顺序：
       R@all、anchorable 比例、时间线相邻对一致率（pairwise accuracy）

用法:
    python -m eval.run_temporal_experiment
    python -m eval.run_temporal_experiment --out eval/outputs/temporal_experiment.json
    python -m eval.run_temporal_experiment --smoke          # 小样试跑（1 批抽取）
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
from dba_pipeline.embedding.store import VectorStore, OpenAIEmbeddings, LocalEmbeddings
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever
from dba_pipeline.llm.inference import InferenceEngine
from dba_pipeline.loader import load_graph
import dba_pipeline.extraction.dba as dba_mod
from dba_pipeline.core import jump_axis


# ---------------------------------------------------------------- 环境

def _load_dotenv(env_path: Path) -> None:
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
        extra_body={"thinking": {"type": "disabled"}},
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


# ---------------------------------------------------------------- prompt 变体

BASE_NODE = None
BASE_EDGE = None
BASE_EXAMPLE = None

TEMPORAL_NODE_EXTRA = """
7. 时间节点单独抽取：对话中出现的明确时间/时段（如「上周一晚上」「周二」「周三凌晨」「这周」）
   要抽成独立的 THING 节点，content 就是该时间词本身；不要把时间写进事件 content。
   一个事件若发生在某个时间，需同时抽出「事件节点」和「时间节点」两个节点。
"""

TEMPORAL_EDGE_EXTRA = """
7. 务必连 TEMPORAL 边（事件→时间，单向）：本轮新抽出的事件节点，若其发生时间也是一个已抽取的
   时间节点，用 TEMPORAL 单向连边（from=事件, to=时间）。例如「用户接到需求」--TEMPORAL-->「上周一晚上」。
8. 事件之间的先后关系用 SEQUENCE（先→后），不要与「时间定位」（TEMPORAL）混为一谈：
   SEQUENCE 是事件 A 在事件 B 之前；TEMPORAL 是事件发生在某个时间锚点上。
"""

TEMPORAL_EXAMPLE_EXTRA = """

示例 3 —— TEMPORAL 事件→时间 + SEQUENCE 事件先后：

对话：
user(周一 20:30): 上周一晚上临时又甩过来一个紧急需求。
user(周二 13:20): 这周二白天我一直在赶代码。
user(周三 01:20): 周三凌晨上线，结果出了个 P0 故障。

本轮新抽取的节点（事件节点与时间节点分开）：
[e0 ACTION] 用户接到紧急需求
[e1 ACTION] 用户加班赶代码
[e2 STATUS] 项目上线出P0故障
[t1 THING] 上周一晚上
[t2 THING] 上周二白天
[t3 THING] 上周三凌晨

正确输出（事件→时间用 TEMPORAL；事件之间先后用 SEQUENCE；因果用 CAUSAL）：
{{"edge_ops":[
  {{"action":"create","from":"e0","to":"t1","rel_type":"TEMPORAL"}},
  {{"action":"create","from":"e1","to":"t2","rel_type":"TEMPORAL"}},
  {{"action":"create","from":"e2","to":"t3","rel_type":"TEMPORAL"}},
  {{"action":"create","from":"e0","to":"e1","rel_type":"SEQUENCE"}},
  {{"action":"create","from":"e1","to":"e2","rel_type":"SEQUENCE"}},
  {{"action":"create","from":"e0","to":"e2","rel_type":"CAUSAL"}}
]}}"""


def set_prompt_mode(mode: str) -> None:
    """在构造 MemoryDBA 前切换节点抽取 / 边连接 prompt 变体"""
    if mode == "base":
        dba_mod.NODE_EXTRACTION_PROMPT = BASE_NODE
        dba_mod.EDGE_LINKING_PROMPT = BASE_EDGE
        dba_mod.EDGE_LINKING_FEWSHOT_EXAMPLE = BASE_EXAMPLE
    elif mode == "temporal":
        dba_mod.NODE_EXTRACTION_PROMPT = BASE_NODE + TEMPORAL_NODE_EXTRA
        dba_mod.EDGE_LINKING_PROMPT = BASE_EDGE + TEMPORAL_EDGE_EXTRA
        dba_mod.EDGE_LINKING_FEWSHOT_EXAMPLE = BASE_EXAMPLE + TEMPORAL_EXAMPLE_EXTRA
    else:
        raise ValueError(f"未知 prompt 模式: {mode}")


# ---------------------------------------------------------------- 抽取

def _run_step(dba, conversation):
    """绕过 triage 直接执行「节点抽取 → 边连接」两步（实验隔离：语料均为主观事实，triage 与 A/B 无关）"""
    node_context = dba._build_node_context(conversation)
    node_ops = dba._parse_response(dba.node_chain.invoke(node_context).content).get("node_ops", [])
    result1 = dba.builder.apply_ops(node_ops, [])
    new_ids = result1.get("created_ids", [])
    edge_context = dba._build_edge_context(conversation, new_ids)
    edge_ops = dba._parse_response(dba.edge_chain.invoke(edge_context).content).get("edge_ops", [])
    result2 = dba.builder.apply_ops([], edge_ops)
    return {
        "node_ops": node_ops,
        "edge_ops": edge_ops,
        "created_ids": result1.get("created_ids", []),
        "errors": result1.get("errors", []) + result2.get("errors", []),
    }


def _triage_need_rate(dba, conversation, n=3):
    """统计 triage 判定 NEEDED 的比例（反映该语料上的 triage 可靠性，仅供报告）"""
    need = 0
    for _ in range(n):
        if dba._should_maintain(conversation):
            need += 1
    return round(need / n, 3)


def run_extraction(dialogs, llm, embeddings, batch_size, mode):
    """空图 + MemoryDBA（批量模式，跳过 triage），返回 (graph, vector_store, batch_log)"""
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
            "batch": gi + 1,
            "turn_count": len(turns_all),
            "node_ops": len(res["node_ops"]),
            "edge_ops": len(res["edge_ops"]),
            "created": res["created_ids"],
            "errors": res["errors"],
            "triage_need_rate": triage_need,
            "nodes_total": graph.node_count,
            "edges_total": graph.edge_count,
        })
        print(f"[DBA-{mode}] 批{gi + 1}（{len(turns_all)}轮）node_ops={len(res['node_ops'])} "
              f"edge_ops={len(res['edge_ops'])} created={len(res['created_ids'])} "
              f"triage_need={triage_need} (累计 {graph.node_count} 节点 / {graph.edge_count} 边)")
    return graph, vector_store, batch_log


# ---------------------------------------------------------------- 匹配

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def match_items(gold_items, dba_items, embed_fn, threshold=0.75):
    """embedding 余弦贪心最大匹配。items: [{id, content, type}]。返回 (matched, g2d, d2g)"""
    pairs = []
    g_vecs = [embed_fn(g["content"]) for g in gold_items]
    d_vecs = [embed_fn(d["content"]) for d in dba_items]
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
        matched.append((gold_items[gi], dba_items[di], sim))
    g2d = {g["id"]: d["id"] for g, d, _ in matched}
    d2g = {d["id"]: g["id"] for g, d, _ in matched}
    return matched, g2d, d2g


def active_nodes(graph):
    out = []
    for nid in graph.graph.nodes():
        n = graph.get_node(nid)
        if n and not n.get("deprecated") and not n.get("forgotten"):
            out.append({"id": nid, "content": n["content"],
                        "type": n["node_type"].value if hasattr(n["node_type"], "value") else str(n["node_type"])})
    return out


def gold_edges_within(gold_graph, gold_node_ids):
    edges = []
    for u, v, data in gold_graph.graph.edges(data=True):
        if u in gold_node_ids and v in gold_node_ids:
            nu, nv = gold_graph.get_node(u), gold_graph.get_node(v)
            if nu and nv and not nu.get("deprecated") and not nv.get("deprecated"):
                rt = data.get("rel_type", "")
                edges.append((u, v, rt.value if hasattr(rt, "value") else str(rt)))
    return edges


# ---------------------------------------------------------------- 抽取保真度

def evaluate_extraction(gold_graph, gold_node_ids, dba_graph, embed_fn, threshold=0.75):
    gold_nodes = [n for n in active_nodes(gold_graph) if n["id"] in gold_node_ids]
    dba_nodes = active_nodes(dba_graph)
    matched, g2d, d2g = match_items(gold_nodes, dba_nodes, embed_fn, threshold)

    node_recall = len(g2d) / len(gold_nodes) if gold_nodes else 0.0
    node_precision = len(d2g) / len(dba_nodes) if dba_nodes else 0.0
    type_correct = sum(1 for g, d, _ in matched if g["type"] == d["type"]) / max(1, len(matched))

    gold_edges = gold_edges_within(gold_graph, gold_node_ids)
    dba_edge_set = set()
    for a, b in dba_graph.graph.edges():
        dba_edge_set.add((a, b))
        dba_edge_set.add((b, a))

    # 全局金边类型 dict（含反向）
    gold_rt = {(x, y): t for x, y, t in gold_edges}
    gold_rt.update({(y, x): t for x, y, t in gold_edges})

    edge_recall_total = 0
    edge_by_rel = {}
    for u, v, rt in gold_edges:
        if u in g2d and v in g2d and (g2d[u], g2d[v]) in dba_edge_set:
            edge_recall_total += 1
        edge_by_rel.setdefault(rt, {"gold": 0, "hit": 0})
        edge_by_rel[rt]["gold"] += 1
        # 命中统计（按 gold rel 类别计数命中）
        if u in g2d and v in g2d and (g2d[u], g2d[v]) in dba_edge_set:
            edge_by_rel[rt]["hit"] += 1

    # 边精确 + 边类型一致率
    dba_edges = [(a, b, d["rel_type"].value if hasattr(d["rel_type"], "value") else str(d["rel_type"]))
                 for a, b, d in dba_graph.graph.edges(data=True)]
    edge_prec_hit = 0
    edge_type_match = 0
    edge_type_total = 0
    for a, b, rt in dba_edges:
        if a in d2g and b in d2g and (d2g[a], d2g[b]) in gold_rt:
            edge_prec_hit += 1
            edge_type_total += 1
            if gold_rt[(d2g[a], d2g[b])].lower() == rt.lower():
                edge_type_match += 1

    return {
        "gold_nodes": len(gold_nodes),
        "dba_nodes": len(dba_nodes),
        "matched_nodes": len(g2d),
        "node_recall": round(node_recall, 4),
        "node_precision": round(node_precision, 4),
        "type_accuracy": round(type_correct, 4),
        "gold_edges": len(gold_edges),
        "dba_edges": len(dba_edges),
        "edge_recall": round(edge_recall_total / len(gold_edges), 4) if gold_edges else 0.0,
        "edge_precision": round(edge_prec_hit / len(dba_edges), 4) if dba_edges else 0.0,
        "edge_type_accuracy": round(edge_type_match / edge_type_total, 4) if edge_type_total else 0.0,
        "edge_by_rel": {
            k: {"gold": v["gold"], "hit": v["hit"],
                "recall": round(v["hit"] / v["gold"], 4) if v["gold"] else 0.0}
            for k, v in edge_by_rel.items()
        },
    }


# ---------------------------------------------------------------- 检索 / 事件检测

def build_store_from_graph(graph, embeddings):
    """用图谱活跃节点重建向量库（供 P 检索）"""
    vs = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    nodes = active_nodes(graph)
    ids = [n["id"] for n in nodes]
    contents = [n["content"] for n in nodes]
    if ids:
        vs.add_memories(ids, contents)
    return vs


DEFAULT_RULES = None
BOOST_RULES = None


def set_jump_mode(mode: str) -> None:
    if mode == "default":
        jump_axis.JUMP_AXIS_RULES = DEFAULT_RULES
    elif mode == "temporal_boost":
        jump_axis.JUMP_AXIS_RULES = BOOST_RULES
    else:
        raise ValueError(f"未知跳转模式: {mode}")


def event_detection(graph, store, llm, embeddings, queries, embed_fn, gold_events, gold_trank,
                    d2g=None, jump_mode="default"):
    """对每条查询跑 P 检索（注入目的），用 graph 的 TEMPORAL 边重建事件时间线。

    d2g: 该图节点 → gold 节点 id 的映射（gold 图传 None 表示恒等映射）。
    gold_events: gold 事件节点集合（用于判定检索结果是否命中 expected）。
    gold_trank: gold 时间节点 id → 时间秩（越小越早）。
    """
    set_jump_mode(jump_mode)
    inference = InferenceEngine(llm)
    retriever = PurposeDrivenRetriever(
        llm=llm, embeddings=embeddings, graph=graph, vector_store=store, inference=inference,
    )
    # 图内 TEMPORAL 边：graph 节点 -> 时间节点（读 direction：事件→时间）
    temporal_anchor = {}  # graph_node_id -> time_node_id
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

        # 检索结果映射到 gold 事件
        R = []  # (gold_event, rank_or_None)
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
        # 时间线一致性：对「能锚定到时间」的事件，按 graph TEMPORAL 边恢复的顺序
        # vs gold_expected 中的顺序，做 pairwise 一致率。
        # 仅统计「时间锚点不同」的事件对（同锚点无法用 TEMPORAL 定序，需 SEQUENCE，不纳入本指标）
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
                        continue  # 同时间锚点，TEMPORAL 无法定序
                    total += 1
                    if (rki < rkj) == (gold_pos[gi] < gold_pos[gj]):
                        correct += 1
            pair_acc = round(correct / total, 4) if total else None

        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "retrieved_ids": peak_ids,
            "retrieved_gold": [g for g, _ in R],
            "expected": expected,
            "R_at_all": round(r_all, 4),
            "anchorable": round(anchorable, 4),
            "timeline_pairwise_acc": pair_acc,
            "retrieved_count": len(peak_ids),
        })
    return results


# ---------------------------------------------------------------- main

def main() -> None:
    global BASE_NODE, BASE_EDGE, BASE_EXAMPLE, DEFAULT_RULES, BOOST_RULES

    parser = argparse.ArgumentParser(description="TEMPORAL 时序关系专项 · 端到端闭环实验")
    parser.add_argument("--yaml", default="data/dba_eval/lifemem_temporal.yaml")
    parser.add_argument("--dialogs", default="data/dba_eval/lifemem_temporal_dialogs.yaml")
    parser.add_argument("--queries", default="data/dba_eval/lifemem_temporal_queries.yaml")
    parser.add_argument("--out", default="eval/outputs/temporal_experiment.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="小样试跑：只用前 1 批对话抽取")
    args = parser.parse_args()

    random.seed(42)
    _load_dotenv(ROOT / ".env")

    # 基准 prompt 快照
    BASE_NODE = dba_mod.NODE_EXTRACTION_PROMPT
    BASE_EDGE = dba_mod.EDGE_LINKING_PROMPT
    BASE_EXAMPLE = dba_mod.EDGE_LINKING_FEWSHOT_EXAMPLE
    DEFAULT_RULES = copy.deepcopy(jump_axis.JUMP_AXIS_RULES)
    BOOST_RULES = copy.deepcopy(DEFAULT_RULES)
    for nt, rel_map in BOOST_RULES.items():
        if jump_axis.RelationType.TEMPORAL in rel_map:
            BOOST_RULES[nt][jump_axis.RelationType.TEMPORAL] = (0.9, 0.5)

    # gold
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

    # gold 事件集合 & 时间秩
    gold_active = active_nodes(gold_graph)
    gold_events = {n["id"] for n in gold_active
                   if n["type"] in ("action", "status", "emotion", "reason")}
    gold_trank = {}
    for nid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        gold_trank[nid] = int(nid[1:]) if nid[1:].isdigit() else len(gold_trank) + 1
    gold_store = build_store_from_graph(gold_graph, embeddings)
    gold_d2g = None  # 恒等映射

    print("===== 阶段 1：DBA 抽取（base vs temporal）=====")
    graphs = {"gold": gold_graph}
    stores = {"gold": gold_store}
    extract = {}
    for mode in ("base", "temporal"):
        g, vs, blog = run_extraction(dialogs, llm, embeddings, args.batch_size, mode)
        graphs[f"dba_{mode}"] = g
        stores[f"dba_{mode}"] = vs
        extract[mode] = {
            "batch_log": blog,
            "nodes": g.node_count,
            "edges": g.edge_count,
            "eval": evaluate_extraction(gold_graph, gold_node_ids, g, embed_fn, args.threshold),
        }
        # 导出图
        gpath = ROOT / f"eval/outputs/temporal_graph_{mode}.yaml"
        gpath.parent.mkdir(parents=True, exist_ok=True)
        gpath.write_text(yaml.safe_dump(g.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[抽取-{mode}] 节点={g.node_count} 边={g.edge_count} "
              f"边召回={extract[mode]['eval']['edge_recall']:.2%} "
              f"temporal边={extract[mode]['eval']['edge_by_rel'].get('temporal', {})} "
              f"sequence边={extract[mode]['eval']['edge_by_rel'].get('sequence', {})}")

    print("\n===== 阶段 2：事件时间线重建（gold / dba_base / dba_temporal × 默认/增强跳转）=====")
    # DBA 图的 d2g 映射：gold 事件节点 -> 共享（按全图匹配）
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
            if gname == "gold":
                d2g = None
            else:
                d2g = g2d_maps[gname.replace("dba_", "")]
            detect[gname][jmode] = event_detection(
                graph, stores[gname], llm, embeddings, queries, embed_fn,
                gold_events, gold_trank, d2g=d2g, jump_mode=jmode,
            )

    out = {
        "meta": {
            "dataset": args.yaml,
            "dialogs": args.dialogs,
            "queries": args.queries,
            "batch_size": args.batch_size,
            "smoke": args.smoke,
            "gold_nodes": len(gold_node_ids),
            "dba_prompts": ["base", "temporal"],
            "jump_modes": ["default", "temporal_boost"],
        },
        "extraction": extract,
        "event_detection": detect,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {out_path}")

    # 打印汇总
    print("\n--- 抽取保真度汇总 ---")
    for mode in ("base", "temporal"):
        e = extract[mode]["eval"]
        print(f"[{mode}] 节点召回={e['node_recall']:.2%} 边召回={e['edge_recall']:.2%} "
              f"temporal={e['edge_by_rel'].get('temporal')} sequence={e['edge_by_rel'].get('sequence')}")
    print("\n--- 事件检测汇总（R@all / anchorable / 时间线pairwise）---")
    for gname in detect:
        for jmode in detect[gname]:
            rows = detect[gname][jmode]
            ra = [r["R_at_all"] for r in rows]
            an = [r["anchorable"] for r in rows if r["anchorable"] is not None]
            pa = [r["timeline_pairwise_acc"] for r in rows if r["timeline_pairwise_acc"] is not None]
            print(f"[{gname} | {jmode}] R@all均值={sum(ra)/len(ra):.2%} "
                  f"anchorable均值={sum(an)/len(an):.2%} "
                  f"时间线pairwise均值={sum(pa)/len(pa) if pa else 'N/A'}")


if __name__ == "__main__":
    main()
