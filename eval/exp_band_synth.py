# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""峰值容忍带：**高同质大图**上的退化复现 + 相对带宽验证（单张最小验证）

问题
----
线上 600 次检索实测 `峰值/候选` 中位 **0.81**（候选≥300 档 0.79）——"寻峰"平均只筛掉 19%。
根因是 δ 为**绝对**带宽（`PeakFinder` 库默认 0.10，`retriever` 从未显式传入），而真实语料的
余弦分布很窄（scriptMem 18 次全在 0.5497~0.5884、全距仅 0.039，δ=0.10 已是全距的 2.5 倍）。
该退化与 embedding 无关（换 bge-m3 前后均 ≈0.80），属**判据尺度与数据尺度不匹配**。

为什么需要合成图
----------------
0925 那次实验的语料最大只有 322 节点（comprehensive 163 / lifemem 322）、同质度低，
**复现不了**线上的退化；而线上图**没有 gold**，无法量"压读出量会不会砍掉该保的节点"。
本脚本按**句式/词汇同质度**造一张 ~3000 节点、单一主题的图（对标线上 scriptMem 的
"Juror #N says …"），并**按构造注入 multi_hop gold**，于是两件事可同时量：

  ① 退化是否复现（轮均分全距、带比）
  ② 相对带宽 `ratio` 能否把带比压下来、**且不砍 gold**

gold 的构造是关键：四处证据分散在**第 0 / 4 / 9 幕**，其中一处是**不含主体名的第三方观察句**
（"The argument in the room has turned around."）——它几乎不可能靠种子直接召回，只能靠图扩展
捞回，因此**分数天然偏低**；四者之间只用**稀疏边**串成一条链（结构可达、内容不相似），
正是 0925 报告 §四.3 说的"被上限砍掉的那批"。这复现了真实多跳 gold 的性质，而不是造一批
"好找的" gold。

不调 LLM（目的由查询集注入），只调 embedding 建向量；只打印计数与 id，不打印记忆正文。

用法：
    python -m eval.exp_band_synth                 # 用缓存图（首次自动生成）
    python -m eval.exp_band_synth --regen         # 重新生成合成图
    python -m eval.exp_band_synth --ratios 0 0.3 0.5 0.7 --caps 0 40 20
"""

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402
from eval.exp_budget_cap_ab import _effective_seed_k  # noqa: E402

OUT_DIR = ROOT / "eval" / "outputs"
GRAPH_YAML = OUT_DIR / "synth_homog_3000.yaml"
PLAN_JSON = OUT_DIR / "synth_homog_3000.plan.json"

N_NODES = 3000
N_ACTS = 10
N_QUERIES = 12
MAX_HOPS = 5
EXPAND_K = 40

# 极小词表 + 8 个句式 → 任意两条文本共享绝大部分词汇，余弦高度接近，对标线上 scriptMem
_SUBJ = [f"Juror #{i}" for i in range(1, 13)]
_OBJ = ["knife", "window", "train", "apartment", "testimony",
        "clock", "switchblade", "witness", "old man", "stairs"]
_ADJ = ["clear", "wrong", "possible", "certain", "unclear", "meaningful", "doubtful", "solid"]
_FRAMES = [
    "{s} says the {o} is {a}.",
    "{s} asks whether the {o} was {a}.",
    "{s} insists the {o} is not {a}.",
    "{s} argues the {o} must be {a}.",
    "{s} looks at the {o} and calls it {a}.",
    "{s} repeats that the {o} is {a}.",
    "{s} wonders if the {o} is really {a}.",
    "{s} agrees the {o} is {a}.",
]
_OBS = "The argument in the room has turned around."   # 不含主体名 → 只能靠扩展捞回
# 情境从句：只为**消掉精确重复文本**（精确重复会造出完全并列的向量，比线上更极端）。
# 词表增得很小，所以整体相似度仍很高。
_CIRC = [
    "This came up before the first vote.",
    "This came up after the first vote.",
    "This happened near midnight.",
    "This happened in the hallway.",
    "This came up during the break.",
    "This happened at the window.",
]
_TYPES = ["action", "status", "reason", "emotion", "thing"]
_RELS = ["sequence", "scenario", "attribute", "temporal", "causal"]


def build_plan(seed: int = 20260930) -> Tuple[List[dict], List[dict], List[dict]]:
    """确定性地生成 (nodes, edges, queries) 三件套。"""
    rng = random.Random(seed)
    act_size = N_NODES // N_ACTS

    # ---- gold 的落点：每问三处证据，分散在第 0 / 4 / 9 幕 ----
    marks: List[dict] = []
    used_slots: set = set()
    for qi in range(N_QUERIES):
        subj = _SUBJ[qi % len(_SUBJ)]
        obj = _OBJ[qi % len(_OBJ)]
        adj = _ADJ[qi % len(_ADJ)]
        slots = []
        for act, tmpl in ((0, _FRAMES[0]), (4, _FRAMES[6]), (9, _FRAMES[2])):
            for _ in range(200):                      # 在幕内找一个未占用位置
                k = rng.randrange(act * act_size, (act + 1) * act_size)
                if k not in used_slots:
                    used_slots.add(k)
                    slots.append((k, tmpl))
                    break
        if len(slots) < 3:
            continue
        obs_k = min((9 + 1) * act_size - 1, max(1, slots[2][0] + 1))
        while obs_k in used_slots:
            obs_k -= 1
        used_slots.add(obs_k)
        marks.append({"qi": qi, "subj": subj, "obj": obj, "adj": adj,
                      "a": slots[0], "b": slots[1], "c": slots[2], "obs": obs_k,
                      "query": f"How did {subj} change his position about {obj} "
                               f"over the course of the deliberation?"})
    # 落点 → id
    slot2mark = {}
    for m in marks:
        for key in ("a", "b", "c"):
            slot2mark[m[key][0]] = (m, key, m[key][1])
    slot2obs = {m["obs"]: m for m in marks}

    nodes: List[dict] = []
    edges: List[dict] = []
    for k in range(N_NODES):
        nid = f"sn{k}"
        if k in slot2obs:
            content, ntype = _OBS, "reason"
        elif k in slot2mark:
            m, _key, tmpl = slot2mark[k]
            content, ntype = tmpl.format(s=m["subj"], o=m["obj"], a=m["adj"]), "action"
        else:
            content = (rng.choice(_FRAMES).format(
                s=rng.choice(_SUBJ), o=rng.choice(_OBJ), a=rng.choice(_ADJ))
                + " " + rng.choice(_CIRC))
            ntype = rng.choice(_TYPES)
        nodes.append({"id": nid, "content": content, "type": ntype,
                      "deprecated": False, "forgotten": False})

    # 边：幕内链（保连通、可逐跳走）+ 稀疏长程（造出跨幕的另一条通路）
    act_size_g = N_NODES // N_ACTS
    for k in range(N_NODES - 1):
        if (k + 1) % act_size_g != 0:          # 幕边界断开 → 跨幕必须靠下面的长程边
            edges.append({"from": f"sn{k}", "to": f"sn{k + 1}", "type": "sequence"})
    for _ in range(N_NODES // 2):
        a, b = rng.randrange(N_NODES), rng.randrange(N_NODES)
        if a != b:
            edges.append({"from": f"sn{a}", "to": f"sn{b}", "type": rng.choice(_RELS)})
    # **gold 的演变链**（本实验的关键）：三处证据用稀疏边串起来 —— 结构上可达（扩展捞得回），
    # 但内容与提问并不特别相似（分数天然偏低）。这正是线上多跳 gold 的性质，也是 0925 报告
    # §四.3"上限按分数截断必然先砍它们"的复现条件。
    for m in marks:
        edges.append({"from": f"sn{m['a'][0]}", "to": f"sn{m['b'][0]}", "type": "temporal"})
        edges.append({"from": f"sn{m['b'][0]}", "to": f"sn{m['c'][0]}", "type": "temporal"})
        edges.append({"from": f"sn{m['c'][0]}", "to": f"sn{m['obs']}", "type": "causal"})

    queries = [{
        "id": f"synth-q{m['qi']}",
        "text": m["query"],
        "purposes": ["track how a juror's position changed during the deliberation"],
        "gold": [f"sn{m['a'][0]}", f"sn{m['b'][0]}", f"sn{m['c'][0]}", f"sn{m['obs']}"],
        "category": "multi_hop",
    } for m in marks]
    return nodes, edges, queries


def _write_yaml(path: Path, nodes: List[dict], edges: List[dict]) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"nodes": nodes, "edges": edges}, fh,
                       allow_unicode=True, sort_keys=False)


def ensure_corpus(regen: bool) -> List[dict]:
    if regen or not GRAPH_YAML.exists() or not PLAN_JSON.exists():
        nodes, edges, queries = build_plan()
        _write_yaml(GRAPH_YAML, nodes, edges)
        PLAN_JSON.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已生成合成图: {GRAPH_YAML.name}（节点 {len(nodes)} / 边 {len(edges)}）"
              f" + 查询 {len(queries)} 条")
    return json.loads(PLAN_JSON.read_text(encoding="utf-8"))


def run_cell(retriever, queries, total: int, ratio: float, cap: int,
             seed_k: int, verbose: bool = False) -> Dict[str, Any]:
    retriever.peak_tolerance_ratio = ratio
    sizes, band_ratios, spreads, top40_hits = [], [], [], []
    g_hit = g_tot = 0
    first_means = None

    for q in queries:
        res = retriever.retrieve(q["text"], seed_k=seed_k, expand_k=EXPAND_K,
                                 max_hops=MAX_HOPS, purpose=q["purposes"] or None,
                                 max_peak_nodes=cap or None)
        pred = {mid for mid, _ in (res.get("peak_memories") or [])}
        sizes.append(len(pred))

        band = res.get("peak_band") or {}
        if band:
            band_ratios.append(float(band.get("band_ratio", 0.0)))
        means = [float(h.get("mean_score", 0)) for h in (res.get("hop_history") or [])]
        if means:
            spreads.append(max(means) - min(means))
            if first_means is None:
                first_means = means

        # cap 的伤害机制：gold 有多少落在"按分数排的前 40"里
        id_score = res.get("peak_scores") or {}
        top40 = {i for i, _ in sorted(id_score.items(), key=lambda x: -x[1])[:40]}
        gold = set(q["gold"])
        g_hit += len(pred & gold)
        g_tot += len(gold)
        top40_hits.append(len(top40 & gold) / len(gold) if gold else 0.0)

    if verbose and first_means:
        print(f"     轮均分: " + " ".join(f"{m:.4f}" for m in first_means)
              + f"   全距={max(first_means) - min(first_means):.4f}")
    return {
        "n_median": statistics.median(sizes) if sizes else 0,
        "n_max": max(sizes) if sizes else 0,
        "coverage": (statistics.median(sizes) / total) if sizes and total else 0.0,
        "band_ratio": statistics.median(band_ratios) if band_ratios else None,
        "hop_spread": statistics.median(spreads) if spreads else None,
        "gold_all": (g_hit / g_tot) if g_tot else None,
        "gold_top40": statistics.mean(top40_hits) if top40_hits else None,
    }


def _pct(v):
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def experiment(ratios: List[float], caps: List[int]) -> Dict[str, Any]:
    queries = ensure_corpus(regen=False)
    if not queries:
        raise SystemExit("合成图无可用查询（用 --regen 重新生成）")
    _load_dotenv(ROOT / ".env")
    # 合成图单独缓存向量（3000 条）：否则每次重跑都要重算一遍 embedding
    graph, retriever = build_retriever(str(GRAPH_YAML), vector_cache_key="synth_homog_3000")
    total = graph.node_count
    seed_k = _effective_seed_k(total)

    print(f"\n{'=' * 104}")
    print(f"合成图 {GRAPH_YAML.name} | 节点 {total} | 查询 {len(queries)} 条（均带 multi_hop gold）"
          f" | seed_k={seed_k} expand_k={EXPAND_K} max_hops={MAX_HOPS}")
    print(f"目的由查询集注入，不调 LLM；gold 四处证据分散在第 0/4/9 幕 + 一处不含主体名的"
          f"观察句（只能靠扩展捞回），彼此用稀疏边成链")
    print(f"{'=' * 104}")

    cells = []
    for ratio in ratios:
        print(f"\n── ratio={ratio}（{'绝对模式（现行）' if ratio <= 0 else '相对模式'}）"
              f"{'  ← 基线' if ratio == 0 else ''}")
        print(f"   {'cap':>6} {'N中位':>6} {'N_max':>6} {'占全图':>7} {'带比':>7} "
              f"{'轮均分全距':>10} {'gold全':>7} {'gold在top40':>11}")
        for cap in caps:
            m = run_cell(retriever, queries, total, ratio, cap, seed_k, verbose=(cap == caps[0]))
            m.update({"ratio": ratio, "cap": cap})
            cells.append(m)
            spread = "  n/a" if m["hop_spread"] is None else f"{m['hop_spread']:.4f}"
            print(f"   {cap or '不限':>6} {m['n_median']:>6.0f} {m['n_max']:>6} "
                  f"{_pct(m['coverage'])} {_pct(m['band_ratio'])} {spread:>10} "
                  f"{_pct(m['gold_all'])} {_pct(m['gold_top40'])}")
    return {"total_nodes": total, "seed_k": seed_k, "queries": len(queries), "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser(description="峰值容忍带：高同质大图退化复现 + 相对带宽验证")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--caps", type=int, nargs="+", default=[0, 40, 20])
    ap.add_argument("--regen", action="store_true", help="重新生成合成图")
    ap.add_argument("--out", default="eval/outputs/exp_band_synth.json")
    args = ap.parse_args()

    if args.regen:
        ensure_corpus(regen=True)
    result = experiment(args.ratios, args.caps)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "graph": GRAPH_YAML.name, "max_hops": MAX_HOPS, "expand_k": EXPAND_K,
        "ratios": args.ratios, "caps": args.caps, "result": result,
        "note": ("合成图（单一主题、极小词表、8 句式）复现线上高同质大图；gold 按构造注入，"
                 "含一处不含主体名的观察句（只能靠扩展捞回）。只记计数与 id，不记正文。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
