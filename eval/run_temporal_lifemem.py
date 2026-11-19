# -*- coding: utf-8 -*-
"""
LiFEMem 全量(322) · TEMPORAL 补充 + 时序（时间线重建）评测

背景：LiFEMem.yaml 已含 h(高中)/c(大专)/f(家庭)/t(旅行)/th(时间锚点) 叙事层，
但 TEMPORAL 边很稀疏（仅 8 条：h0/h20/h23→th0、c0/c11/c14→th1、w0/w13→th2），
绝大多数叙事事件未被锚定到时间纪元 → 无法重建跨期时序。

本脚本：
  1. 原始 LiFEMem（8 条 TEMPORAL）vs 补充版 LiFEMem（按前缀补 TEMPORAL 边：
     全部 h*→th0、c*→th1、w*→th2），导出补充版图。
  2. 对一组"叙事时序"查询做 P 检索（注入目的），重建事件时间线：
       时序排序 = 主序 TEMPORAL 纪元秩(th0<th1<th2) + 次序 SEQUENCE 拓扑序(期内先后)
     指标：R@all（命中 expected 比例）
           锚定率 anchorable（检索命中事件中有 TEMPORAL 纪元锚点的比例）
           时序一致性（重建顺序 vs gold_expected 顺序的 pairwise 一致率）
  3. 对比原始 vs 补充 的时间线重建效果，验证"补充 TEMPORAL 边后时序可重建"。

用法:
    python -m eval.run_temporal_lifemem --out eval/outputs/lifemem_temporal_cmp.json
"""

import argparse
import collections
import copy
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml

from dba_pipeline.loader import load_graph
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import RelationType
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever
from dba_pipeline.llm.inference import InferenceEngine

# 复用上一轮实验的环境/工具（避免重复实现）
from eval.run_temporal_experiment import (
    _load_dotenv, build_llm, build_embeddings, build_store_from_graph,
)

GOLD_YAML = "data/LiFEMem.yaml"
GOLD_TRANK = {"th0": 1, "th1": 2, "th2": 3}  # 高中 < 大专 < 工作

# 叙事时序查询：gold_expected 按人生叙事先后排序（用于对比重建顺序）
QUERIES = [
    {
        "id": "qA",
        "query": "用户高中那段初恋，从心动到分手是怎么一步步走过来的？",
        "purposes": ["了解经历", "还原时间线"],
        "gold_expected": ["h10", "h11", "h12", "h13", "h20"],  # 心动→递情书→答应→天台约会→分手
    },
    {
        "id": "qB",
        "query": "用户高考失利之后，是怎么进入大专的？",
        "purposes": ["了解经历", "还原时间线"],
        "gold_expected": ["h23", "c0"],  # 高考失利 → 上大专
    },
    {
        "id": "qC",
        "query": "用户是怎么从大专一步步转行做上后端开发的？",
        "purposes": ["了解经历", "还原时间线"],
        "gold_expected": ["c7", "c8", "c9", "c10", "c11", "w0"],  # 自学→爬虫→转行→刷题→offer→后端
    },
    {
        "id": "qD",
        "query": "用户高中那三年从成绩好到高考失利，期间经历了什么？",
        "purposes": ["了解经历", "还原时间线"],
        "gold_expected": ["h0", "h3", "h18", "h22", "h23"],  # 成绩好→下滑→高三压力→进考场→失利
    },
    {
        "id": "qE",
        "query": "用户从迷茫的大专走到现在的工作，这条人生转折线是怎样的？",
        "purposes": ["了解经历", "还原时间线"],
        "gold_expected": ["h23", "c0", "c1", "c3", "c4", "c5", "c6", "c7", "c9", "c11", "w0"],
    },
]


def build_augmented(graph: MemoryGraph) -> MemoryGraph:
    """按人生阶段前缀补 TEMPORAL 边：h*→th0(高中)、c*→th1(大专)、w*→th2(工作)。"""
    g = copy.deepcopy(graph)
    added = 0
    for nid in list(g.graph.nodes()):
        m = re.match(r"^([hcw])\d+$", nid)
        if not m:
            continue
        anchor = {"h": "th0", "c": "th1", "w": "th2"}[m.group(1)]
        if anchor in g.graph and not g.graph.has_edge(nid, anchor):
            g.add_edge(nid, anchor, RelationType.TEMPORAL)
            added += 1
    return g, added


def prog_rank_map(graph: MemoryGraph, rels=("sequence", "causal")) -> dict:
    """推进边拓扑序（Kahn）：边 u→v（rels 内）⇒ rank[u] < rank[v]；孤立/环置后。

    默认取 SEQUENCE+CAUSAL——数据集叙事链条的先后大量由 CAUSAL 承载（如递情书→答应），
    纯 SEQUENCE 会漏掉期内顺序；用两者共同定序更贴近真实时序。
    """
    relset = {r.lower() for r in rels}
    nodes = list(graph.graph.nodes())
    out = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}
    for u, v, d in graph.graph.edges(data=True):
        rt = d.get("rel_type", "")
        rt = rt.value if hasattr(rt, "value") else str(rt)
        if rt.lower() in relset:
            out[u].append(v)
            indeg[v] += 1
    q = collections.deque(n for n in nodes if indeg[n] == 0)
    rank, order = {}, 0
    while q:
        u = q.popleft()
        rank[u] = order
        order += 1
        for v in out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for n in nodes:  # 环/未覆盖
        if n not in rank:
            rank[n] = order
            order += 1
    return rank


def era_rank_map(graph: MemoryGraph) -> dict:
    """从 TEMPORAL 边读事件→纪元秩：边 u→v(temporal) 且 v 是时间锚点 ⇒ era[u]=GOLD_TRANK[v]"""
    era = {}
    for u, v, d in graph.graph.edges(data=True):
        rt = d.get("rel_type", "")
        rt = rt.value if hasattr(rt, "value") else str(rt)
        if rt.lower() == "temporal" and u not in era and v in GOLD_TRANK:
            era[u] = GOLD_TRANK[v]
    return era


def run_reconstruction(graph, store, llm, embeddings, queries, jump_default=True) -> list:
    inference = InferenceEngine(llm)
    retriever = PurposeDrivenRetriever(
        llm=llm, embeddings=embeddings, graph=graph, vector_store=store, inference=inference,
    )
    era = era_rank_map(graph)
    seqr = prog_rank_map(graph, rels=("sequence",))
    prog = prog_rank_map(graph, rels=("sequence", "causal"))

    def sort_key(rank):
        return (lambda eid: (era.get(eid, 10 ** 6), rank.get(eid, 10 ** 6)))

    key_seq = sort_key(seqr)
    key_prog = sort_key(prog)

    results = []
    for q in queries:
        expected = q["gold_expected"]
        exp_set = set(expected)
        r = retriever.retrieve(q["query"], seed_k=5, purpose=q["purposes"])
        R, seen = [], set()
        for mid, _ in r["peak_memories"]:
            if mid in exp_set and mid not in seen:
                seen.add(mid)
                R.append(mid)

        r_all = len(R) / len(expected) if expected else 0.0
        anchored = [eid for eid in R if eid in era]
        anchorable = len(anchored) / len(R) if R else 0.0

        # 拆分时序一致性：跨纪元用 TEMPORAL 纪元秩；期内用推进秩（seq-only 与 seq+causal 两版对比）
        def pairwise(anchored_list, sub_rank):
            gold_pos = {eid: i for i, eid in enumerate(expected)}
            rc_order = sorted(anchored_list, key=sub_rank)
            cc = ct = wc = wt = 0
            for i in range(len(anchored_list)):
                for j in range(i + 1, len(anchored_list)):
                    a, b = anchored_list[i], anchored_list[j]
                    gold_asc = gold_pos[a] < gold_pos[b]
                    rec_asc = rc_order.index(a) < rc_order.index(b)
                    if era[a] != era[b]:
                        ct += 1
                        if gold_asc == rec_asc:
                            cc += 1
                    else:
                        wt += 1
                        if gold_asc == rec_asc:
                            wc += 1
            return (round(cc / ct, 4) if ct else None, round(wc / wt, 4) if wt else None)

        cross_acc, within_acc = pairwise(anchored, key_seq)
        _, within_acc_prog = pairwise(anchored, key_prog)

        results.append({
            "query_id": q["id"],
            "query": q["query"],
            "retrieved_gold": R,
            "expected": expected,
            "R_at_all": round(r_all, 4),
            "anchorable": round(anchorable, 4),
            "cross_era_pairwise": cross_acc,
            "within_era_pairwise_seq": within_acc,
            "within_era_pairwise_causal": within_acc_prog,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LiFEMem 全量 TEMPORAL 补充 + 时序重建评测")
    parser.add_argument("--yaml", default=GOLD_YAML)
    parser.add_argument("--out", default="eval/outputs/lifemem_temporal_cmp.json")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--export-augmented", default="eval/outputs/lifemem_full_temporal_augmented.yaml")
    args = parser.parse_args()

    random.seed(42)
    _load_dotenv(ROOT / ".env")

    embeddings = build_embeddings()
    llm = build_llm(args.llm_model)

    gold = load_graph(ROOT / args.yaml)
    aug, added = build_augmented(gold)
    if args.export_augmented:
        p = ROOT / args.export_augmented
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(aug.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    store_gold = build_store_from_graph(gold, embeddings)
    store_aug = build_store_from_graph(aug, embeddings)

    print(f"[LiFEMem] 原始 TEMPORAL 边锚定事件数: {len(era_rank_map(gold))}；补充后新增 TEMPORAL 边: {added}，"
          f"锚定事件数: {len(era_rank_map(aug))}")

    res_gold = run_reconstruction(gold, store_gold, llm, embeddings, QUERIES)
    res_aug = run_reconstruction(aug, store_aug, llm, embeddings, QUERIES)

    out = {
        "meta": {
            "graph": args.yaml,
            "queries": len(QUERIES),
            "augmented_added_temporal": added,
            "temporal_covered_orig": len(era_rank_map(gold)),
            "temporal_covered_aug": len(era_rank_map(aug)),
        },
        "original": res_gold,
        "augmented": res_aug,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def agg(rows):
        ra = [r["R_at_all"] for r in rows]
        an = [r["anchorable"] for r in rows]
        ce = [r["cross_era_pairwise"] for r in rows if r["cross_era_pairwise"] is not None]
        we = [r["within_era_pairwise_seq"] for r in rows if r["within_era_pairwise_seq"] is not None]
        wc = [r["within_era_pairwise_causal"] for r in rows if r["within_era_pairwise_causal"] is not None]
        return (sum(ra) / len(ra) if ra else 0,
                sum(an) / len(an) if an else 0,
                sum(ce) / len(ce) if ce else None,
                sum(we) / len(we) if we else None,
                sum(wc) / len(wc) if wc else None)

    print("\n=== 时间线重建对比（各 5 条查询）===")
    for name, rows in (("原始(8条TEMPORAL)", res_gold), ("补充版(叙事全锚定)", res_aug)):
        ra, an, ce, we, wc = agg(rows)
        print(f"[{name}] R@all={ra:.2%}  可锚定率={an:.2%}  跨纪元pairwise={ce}  "
              f"期内(seq)={we}  期内(seq+causal)={wc}")

    print("\n--- 逐查询 ---")
    for name, rows in (("原始", res_gold), ("补充", res_aug)):
        print(f"  [{name}]")
        for r in rows:
            print(f"    {r['query_id']} R@all={r['R_at_all']:.2%} anchorable={r['anchorable']:.2%} "
                  f"跨纪元={r['cross_era_pairwise']} 期内-seq={r['within_era_pairwise_seq']} "
                  f"期内-causal={r['within_era_pairwise_causal']}")

    print(f"\n输出: {out_path}  |  补充版图: {ROOT / args.export_augmented}")


if __name__ == "__main__":
    main()
