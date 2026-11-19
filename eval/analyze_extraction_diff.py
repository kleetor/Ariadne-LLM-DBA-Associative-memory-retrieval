# -*- coding: utf-8 -*-
"""两份图谱的抽取/连边差分分析（离线，不调用任何 API）。

用途
----
配合 `eval/probe_llm_arm_ab.py --no-search --dump <dir>` 导出的两臂图谱，
回答「换 LLM 后，抽取内容与关系边到底哪里不一样」——这是制定调优策略的依据，
不是评分：只看结构、粒度与信息覆盖。

分析维度
--------
1. 规模：节点 / 边 / 平均度 / 孤立节点占比
2. 节点类型直方图（6 类）与关系类型直方图（8 类）
3. 各类节点的平均度：定位「哪一类节点丢了连接」
4. 内容粒度：按节点类型分组的内容长度分布（中位/均值/p90）
5. 信息覆盖：对给定词表（默认取查询集的 gold_facts）统计其在**节点文本**与
   **边两端文本**中的命中次数 —— 直接量化「该记的事实有没有记下来、有没有连起来」
6. 抽样对照：按类型并排若干节点与关系三元组，供人工判断抽取粒度

用法
----
    python -m eval.analyze_extraction_diff \
        --a eval/outputs/llm_arm/A/graph.yaml \
        --b eval/outputs/llm_arm/B/graph.yaml
    # 可选：--samples 5 --out eval/outputs/extraction_diff.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import yaml

# 复用 e2e 的归一化口径（去空白与常见中英标点）：LLM 会改写空格与标点——
# 例如 gpt-4o-mini 常写成「9 月 8 日」「先做 A 方案」，而 gold fact 是「9月8日」。
# 直接用 raw.count() 会把这类事实全判成"未覆盖"，得出错误结论。
from eval.test_alm_vs_mcp_e2e import _norm

_ROOT = Path(__file__).resolve().parent.parent

NODE_TYPES = ["status", "reason", "action", "thing", "person", "emotion"]
REL_TYPES = ["causal", "scenario", "sequence", "preference",
             "social", "attribute", "temporal", "taxonomic"]
QUERIES_PATH = _ROOT / "data" / "dba_eval" / "groupchat_mixed_queries.yaml"
SPEAKERS = ["小林", "老周", "阿哲", "小李", "陈哲", "周明"]


def load_graph(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    by_id = {n["id"]: n for n in nodes}
    degree: Dict[str, int] = {n["id"]: 0 for n in nodes}
    for edge in edges:
        for endpoint in (edge.get("from"), edge.get("to")):
            if endpoint in degree:
                degree[endpoint] += 1
    return {"path": str(path), "nodes": nodes, "edges": edges,
            "by_id": by_id, "degree": degree}


def _hist(items: List[str], keys: List[str]) -> Dict[str, int]:
    out = {k: 0 for k in keys}
    extra: Dict[str, int] = {}
    for value in items:
        if value in out:
            out[value] += 1
        else:
            extra[value] = extra.get(value, 0) + 1
    if extra:
        out["<其它>"] = sum(extra.values())
    return out


def _percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[idx]


def _len_stats(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {"n": 0, "median": 0, "mean": 0.0, "p90": 0, "max": 0}
    return {
        "n": len(lengths),
        "median": int(median(lengths)),
        "mean": round(mean(lengths), 1),
        "p90": _percentile(lengths, 0.9),
        "max": max(lengths),
    }


def profile(g: Dict[str, Any]) -> Dict[str, Any]:
    nodes, edges = g["nodes"], g["edges"]
    node_types = [str(n.get("type") or "") for n in nodes]
    rel_types = [str(e.get("type") or "") for e in edges]
    degree = g["degree"]
    isolated = [nid for nid, d in degree.items() if d == 0]

    # 每类节点的平均度：定位哪一类节点丢了连接
    by_type_degree: Dict[str, List[int]] = {t: [] for t in NODE_TYPES}
    for node in nodes:
        by_type_degree.setdefault(str(node.get("type") or ""), []).append(
            degree.get(node["id"], 0)
        )
    type_degree_avg = {
        t: round(mean(vals), 2) if vals else 0.0
        for t, vals in by_type_degree.items() if vals
    }

    # 每类关系涉及多少不同节点（覆盖广度，而不只是边的条数）
    rel_node_coverage: Dict[str, set] = {t: set() for t in REL_TYPES}
    for edge in edges:
        rel = str(edge.get("type") or "")
        bucket = rel_node_coverage.setdefault(rel, set())
        bucket.add(edge.get("from"))
        bucket.add(edge.get("to"))

    content_by_type: Dict[str, List[int]] = {}
    for node in nodes:
        content_by_type.setdefault(str(node.get("type") or ""), []).append(
            len(str(node.get("content") or ""))
        )

    joined_nodes = "\n".join(str(n.get("content") or "") for n in nodes)
    joined_edges = "\n".join(
        f"{g['by_id'].get(e.get('from'), {}).get('content', '')} "
        f"{g['by_id'].get(e.get('to'), {}).get('content', '')}"
        for e in edges
    )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "avg_degree": round(2 * len(edges) / len(nodes), 2) if nodes else 0.0,
        "isolated": len(isolated),
        "isolated_ratio": round(len(isolated) / len(nodes), 3) if nodes else 0.0,
        "node_types": _hist(node_types, NODE_TYPES),
        "rel_types": _hist(rel_types, REL_TYPES),
        "type_degree_avg": type_degree_avg,
        "rel_node_coverage": {t: len(v) for t, v in rel_node_coverage.items() if v},
        "content_len": {t: _len_stats(v) for t, v in content_by_type.items()},
        "speakers": {name: joined_nodes.count(name) for name in SPEAKERS},
        "collapse_user": joined_nodes.count("用户"),
        "assistant_label": joined_nodes.count("助手"),
        "joined_nodes": joined_nodes,
        "joined_edges": joined_edges,
        "norm_nodes": _norm(joined_nodes),
        "norm_edges": _norm(joined_edges),
    }


def load_fact_vocab() -> List[str]:
    with QUERIES_PATH.open("r", encoding="utf-8") as fh:
        queries = yaml.safe_load(fh) or []
    facts: List[str] = []
    for query in queries:
        for fact in query.get("gold_facts") or []:
            if fact not in facts:
                facts.append(fact)
    return facts


def _fmt_hist(a: Dict[str, int], b: Dict[str, int], keys: List[str]) -> None:
    print(f"  {'类型':<12}{'A':>7}{'B':>7}{'差':>9}")
    for key in keys + ["<其它>"]:
        va, vb = a.get(key, 0), b.get(key, 0)
        if va == 0 and vb == 0 and key == "<其它>":
            continue
        delta = f"{vb - va:+d}" if (va or vb) else "-"
        print(f"  {key:<12}{va:>7}{vb:>7}{delta:>9}")


def main() -> int:
    parser = argparse.ArgumentParser(description="两份图谱的抽取/连边差分")
    parser.add_argument("--a", required=True, help="A 臂图谱 YAML")
    parser.add_argument("--b", required=True, help="B 臂图谱 YAML")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--samples", type=int, default=4, help="每类抽样条数")
    parser.add_argument("--out", default=None, help="把结构化摘要写到该 JSON")
    args = parser.parse_args()

    ga, gb = load_graph(Path(args.a)), load_graph(Path(args.b))
    pa, pb = profile(ga), profile(gb)
    la, lb = args.label_a, args.label_b

    print(f"# 抽取/连边差分 | {la}={args.a} | {lb}={args.b}")

    print("\n" + "=" * 74)
    print("一、规模")
    print("=" * 74)
    print("  （边数取自落盘 YAML：双向关系只存一条，故小于探针汇总里的内存有向边数）")
    rows: List[Tuple[str, Any, Any]] = [
        ("节点数", pa["nodes"], pb["nodes"]),
        ("边数", pa["edges"], pb["edges"]),
        ("平均度(2E/V)", pa["avg_degree"], pb["avg_degree"]),
        ("孤立节点", pa["isolated"], pb["isolated"]),
        ("孤立占比", pa["isolated_ratio"], pb["isolated_ratio"]),
        ("「用户」出现", pa["collapse_user"], pb["collapse_user"]),
        ("「助手」出现", pa["assistant_label"], pb["assistant_label"]),
    ]
    print(f"  {'指标':<14}{la:>10}{lb:>10}")
    for label, va, vb in rows:
        print(f"  {label:<14}{str(va):>10}{str(vb):>10}")

    print("\n二、节点类型直方图")
    _fmt_hist(pa["node_types"], pb["node_types"], NODE_TYPES)

    print("\n三、关系类型直方图")
    _fmt_hist(pa["rel_types"], pb["rel_types"], REL_TYPES)

    print("\n四、各类节点的平均度（定位哪类节点丢了连接）")
    print(f"  {'节点类型':<12}{la:>10}{lb:>10}")
    for node_type in NODE_TYPES:
        va = pa["type_degree_avg"].get(node_type, 0.0)
        vb = pb["type_degree_avg"].get(node_type, 0.0)
        if va == 0 and vb == 0:
            continue
        print(f"  {node_type:<12}{va:>10}{vb:>10}")

    print("\n五、各关系类型涉及的节点数（覆盖广度）")
    print(f"  {'关系类型':<12}{la:>10}{lb:>10}")
    for rel in REL_TYPES:
        va = pa["rel_node_coverage"].get(rel, 0)
        vb = pb["rel_node_coverage"].get(rel, 0)
        if va == 0 and vb == 0:
            continue
        print(f"  {rel:<12}{va:>10}{vb:>10}")

    print("\n六、内容粒度（字符数：中位 / 均值 / p90 / 最长）")
    print(f"  {'节点类型':<12}{la + '(中位/均值/p90/最长)':>28}{lb + '(中位/均值/p90/最长)':>28}")
    for node_type in NODE_TYPES:
        sa, sb = pa["content_len"].get(node_type), pb["content_len"].get(node_type)
        if not sa and not sb:
            continue
        def fmt(s):
            return "-" if not s or not s["n"] else f"{s['median']}/{s['mean']}/{s['p90']}/{s['max']}"
        print(f"  {node_type:<12}{fmt(sa):>28}{fmt(sb):>28}")

    print("\n七、人称词频（节点文本内出现次数）")
    print(f"  {'称谓':<10}{la:>7}{lb:>7}")
    for name in SPEAKERS:
        print(f"  {name:<10}{pa['speakers'][name]:>7}{pb['speakers'][name]:>7}")

    facts = load_fact_vocab()
    print(f"\n八、信息覆盖（{len(facts)} 个 gold fact 在节点文本 / 边两端文本中的命中次数）")
    print("   （已按 e2e 口径去空白与标点后计数，故「9 月 8 日」与「9月8日」等价）")
    print(f"  {'fact':<14}{'A节点':>7}{'A边':>6}{'B节点':>7}{'B边':>6}")
    cover = []
    for fact in facts:
        key = _norm(fact)
        an, ae = pa["norm_nodes"].count(key), pa["norm_edges"].count(key)
        bn, be = pb["norm_nodes"].count(key), pb["norm_edges"].count(key)
        cover.append({"fact": fact, "a_node": an, "a_edge": ae, "b_node": bn, "b_edge": be})
        flag = ""
        if (an and not bn) or (ae and not be):
            flag = "  ← 仅 A 侧有"
        elif (bn and not an) or (be and not ae):
            flag = "  ← 仅 B 侧有"
        print(f"  {fact:<14}{an:>7}{ae:>6}{bn:>7}{be:>6}{flag}")
    only_a = [c["fact"] for c in cover if (c["a_node"] or c["a_edge"]) and not (c["b_node"] or c["b_edge"])]
    only_b = [c["fact"] for c in cover if (c["b_node"] or c["b_edge"]) and not (c["a_node"] or c["a_edge"])]
    both_missing = [c["fact"] for c in cover if not (c["a_node"] or c["a_edge"]) and not (c["b_node"] or c["b_edge"])]
    print(f"\n  仅 A 覆盖: {len(only_a)} → {only_a}")
    print(f"  仅 B 覆盖: {len(only_b)} → {only_b}")
    print(f"  两侧都没覆盖: {len(both_missing)} → {both_missing}")

    print("\n九、抽样对照（按节点类型）")
    for node_type in NODE_TYPES:
        na = [n for n in ga["nodes"] if str(n.get("type")) == node_type][: args.samples]
        nb = [n for n in gb["nodes"] if str(n.get("type")) == node_type][: args.samples]
        if not na and not nb:
            continue
        print(f"\n  ── {node_type} ──")
        for i in range(max(len(na), len(nb))):
            left = f"[{na[i]['id']}] {str(na[i].get('content'))[:110]}" if i < len(na) else "（无）"
            right = f"[{nb[i]['id']}] {str(nb[i].get('content'))[:110]}" if i < len(nb) else "（无）"
            print(f"    {la}: {left}")
            print(f"    {lb}: {right}")

    print("\n十、抽样对照（按关系类型）")
    for rel in REL_TYPES:
        ea = [e for e in ga["edges"] if str(e.get("type")) == rel][: args.samples]
        eb = [e for e in gb["edges"] if str(e.get("type")) == rel][: args.samples]
        if not ea and not eb:
            continue
        print(f"\n  ── {rel} ──")
        def triple(g, e):
            f = g["by_id"].get(e.get("from"), {})
            t = g["by_id"].get(e.get("to"), {})
            return (f"{str(f.get('content'))[:50]} → {str(t.get('content'))[:50]}")
        for i in range(max(len(ea), len(eb))):
            print(f"    {la}: {triple(ga, ea[i]) if i < len(ea) else '（无）'}")
            print(f"    {lb}: {triple(gb, eb[i]) if i < len(eb) else '（无）'}")

    if args.out:
        summary = {
            "a": {"path": args.a, "label": la, "nodes": pa["nodes"], "edges": pa["edges"],
                  "avg_degree": pa["avg_degree"], "isolated": pa["isolated"],
                  "node_types": pa["node_types"], "rel_types": pa["rel_types"],
                  "type_degree_avg": pa["type_degree_avg"],
                  "rel_node_coverage": pa["rel_node_coverage"],
                  "content_len": pa["content_len"], "speakers": pa["speakers"],
                  "collapse_user": pa["collapse_user"]},
            "b": {"path": args.b, "label": lb, "nodes": pb["nodes"], "edges": pb["edges"],
                  "avg_degree": pb["avg_degree"], "isolated": pb["isolated"],
                  "node_types": pb["node_types"], "rel_types": pb["rel_types"],
                  "type_degree_avg": pb["type_degree_avg"],
                  "rel_node_coverage": pb["rel_node_coverage"],
                  "content_len": pb["content_len"], "speakers": pb["speakers"],
                  "collapse_user": pb["collapse_user"]},
            "coverage": cover,
            "only_a": only_a, "only_b": only_b, "both_missing": both_missing,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化摘要已写入: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
