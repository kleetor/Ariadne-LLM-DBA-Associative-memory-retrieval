# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""检索预算 × 输出上限 正交 A/B

问题
----
同一份检索代码、同一张图，历史评测口径返回几十条，ALM 线上返回几百条。根因不是
平台异常，而是两边注入的检索预算不是一套：

  · 历史评测（eval 里绝大多数脚本、scale_retrieval.py）：走 `retriever.retrieve()`
    **自带默认** seed_k=5、expand_k=None→5；
  · ALM 线上（alm/space.py）：seed_k=`_effective_seed_k()`（200 节点→20、322→32）、
    expand_k=40。

实测同一张 200 节点图、同一查询、同一份代码，只改预算：23 条（11.5% 全图）→ 124 条
（62%）。而 0828 检索理论文的设计输出区间是 5–37 条，线上日志也印证了膨胀
（喂入点 max 379 / 采纳 max 378）。

本实验把两个可动旋钮做成**正交因子**，量出「压输出量」与「gold 覆盖」的取舍：

    因子 A  expand_k       ∈ {40(现行), 20, 10, 5}
    因子 B  max_peak_nodes ∈ {0(不限, 现行), 40, 20}

另加一行「历史评测口径」（seed_k=5 / expand_k=5 / 不限）作为锚点，使全表可回溯到
"当年那几十条"。

seed_k 固定为 ALM 现行取值（_effective_seed_k 等价实现），故输出差异只由 A/B 解释。
目的（purposes）由查询集直接注入，检索全程**不调用 LLM**，结果可复现。

只打印计数与节点 id，不打印记忆正文（同 0919 §五 合规口径）。

用法：
    python -m eval.exp_budget_cap_ab
    python -m eval.exp_budget_cap_ab --corpus lifemem
    python -m eval.exp_budget_cap_ab --corpus comprehensive --expand-ks 40 10 --caps 0 20
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

# ---- 因子档位 ----
DEFAULT_EXPAND_KS = [40, 20, 10, 5]
DEFAULT_CAPS = [0, 40, 20]
# 历史评测口径（retrieve() 自带默认），单独一行作锚点
LEGACY = {"label": "历史评测口径", "seed_k": 5, "expand_k": 5, "cap": 0}
MAX_HOPS = 5

DATA_DIR = ROOT / "data" / "dba_eval"

CORPORA = {
    "comprehensive": {
        "graph": DATA_DIR / "comprehensive_gold.yaml",
        "queries": DATA_DIR / "comprehensive_queries.yaml",
        "text_key": "text",
        "name_key": "id",
        "category_key": "category",
    },
    "scale200": {
        "graph": DATA_DIR / "scale_200.yaml",
        "queries": None,  # 无 gold，只量输出规模
        "inline": [
            {"id": "s1", "text": "最近项目排得太满了，天天加班到十一二点，感觉整个人都被掏空了……",
             "purposes": ["理解工作与情绪"], "gold_expected": []},
            {"id": "s2", "text": "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？",
             "purposes": ["理解健康近况"], "gold_expected": []},
            {"id": "s3", "text": "感觉压力好大，不知道怎么熬过这段日子……",
             "purposes": ["理解情绪"], "gold_expected": []},
        ],
        "text_key": "text",
        "name_key": "id",
        "category_key": None,
    },
    "lifemem": {
        "graph": ROOT / "data" / "LiFEMem.yaml",
        "queries": None,
        "inline_factory": "lifemem",
        "text_key": "query",
        "name_key": "id",
        "category_key": None,
    },
}


def _norm_queries(raw: Any) -> List[Dict[str, Any]]:
    """兼容 list / {queries: [...]} / {id: {...}} 三种查询集写法"""
    if isinstance(raw, dict):
        for key in ("queries", "items", "data"):
            if key in raw:
                return list(raw[key])
        return list(raw.values())
    return list(raw)


def _lifemem_queries() -> List[Dict[str, Any]]:
    """LiFEMem 322 节点的 5 条带 gold 查询。

    定义在 eval/run_temporal_lifemem.py 里（那是唯一来源），但该模块在 import 期
    会连带拉起 run_temporal_experiment 的构建工具。这里只做**受控导入**：任一环节
    失败就返回空集，让本实验退化为"只量规模、不量 gold"，而不是整脚本崩掉。
    """
    try:
        from eval.run_temporal_lifemem import QUERIES as LIFEMEM_QUERIES

        return [
            {
                "id": q["id"],
                "text": q["query"],
                "purposes": q.get("purposes") or [],
                "gold_expected": q.get("gold_expected") or [],
            }
            for q in LIFEMEM_QUERIES
        ]
    except Exception as exc:  # pragma: no cover - 依赖缺失时降级
        print(f"[warn] LiFEMem 查询集导入失败，退化为只量规模: {exc}")
        return []


def _load_corpus(spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = None
    if spec.get("queries") is not None:
        import yaml

        with open(spec["queries"], "r", encoding="utf-8") as fh:
            raw = _norm_queries(yaml.safe_load(fh))
    elif spec.get("inline_factory") == "lifemem":
        raw = _lifemem_queries()
    else:
        raw = spec.get("inline") or []

    queries = []
    for q in raw:
        text = q.get(spec["text_key"]) or q.get("query") or q.get("text")
        if not text:
            continue
        queries.append({
            "id": str(q.get(spec["name_key"]) or q.get("id") or "?"),
            "text": text,
            "purposes": list(q.get("purposes") or []),
            "gold": list(q.get("gold_expected") or []),
            "category": (q.get(spec["category_key"]) if spec.get("category_key") else None) or "-",
        })
    return queries, queries


def _effective_seed_k(total: int, seed_k: int = 40, seed_k_min: int = 5) -> int:
    """alm/space.py MemorySpace._effective_seed_k 的等价实现（ALM 现行规则）"""
    if total <= 0:
        return max(2, seed_k_min)
    if total <= seed_k:
        return max(2, total)
    return max(2, seed_k_min, min(seed_k, total // 10))


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = int(round(q * (len(ordered) - 1)))
    return ordered[max(0, min(pos, len(ordered) - 1))]


def _run_cell(retriever, queries, total: int, seed_k: int, expand_k: int,
              cap: int) -> Dict[str, Any]:
    sizes: List[int] = []
    gold_micro_hit = gold_micro_tot = 0
    per_cat: Dict[str, List[float]] = {}

    for q in queries:
        res = retriever.retrieve(
            q["text"],
            seed_k=seed_k,
            expand_k=expand_k,
            max_hops=MAX_HOPS,
            purpose=q["purposes"] or None,
            max_peak_nodes=cap or None,
        )
        pred = {mid for mid, _ in (res.get("peak_memories") or [])}
        sizes.append(len(pred))

        gold = set(q["gold"])
        if gold:
            hit = len(pred & gold)
            gold_micro_hit += hit
            gold_micro_tot += len(gold)
            per_cat.setdefault(q["category"], []).append(hit / len(gold))

    return {
        "n_median": statistics.median(sizes) if sizes else 0,
        "n_p90": _quantile([float(s) for s in sizes], 0.90),
        "n_max": max(sizes) if sizes else 0,
        "coverage": (statistics.median(sizes) / total) if sizes and total else 0.0,
        "gold_micro": (gold_micro_hit / gold_micro_tot) if gold_micro_tot else None,
        "gold_macro": (statistics.mean([v for vs in per_cat.values() for v in vs])
                       if per_cat else None),
        "per_cat": {k: statistics.mean(v) for k, v in sorted(per_cat.items())},
    }


def _fmt_pct(v: Optional[float]) -> str:
    return "  n/a" if v is None else f"{v * 100:5.1f}%"


def experiment(corpus: str, expand_ks: List[int], caps: List[int]) -> Dict[str, Any]:
    spec = CORPORA[corpus]
    queries, _ = _load_corpus(spec)
    if not queries:
        raise SystemExit(f"语料 {corpus} 无可用查询")

    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever(str(spec["graph"]))
    total = graph.node_count
    alm_seed_k = _effective_seed_k(total)

    has_gold = any(q["gold"] for q in queries)
    print(f"\n{'=' * 96}")
    print(f"语料 {corpus} | 图 {total} 节点 | 查询 {len(queries)} 条"
          f" | gold {'有' if has_gold else '无（只量规模）'}")
    print(f"seed_k 固定 = {alm_seed_k}（ALM 现行 _effective_seed_k） | max_hops={MAX_HOPS}"
          f" | 目的由查询集注入，不调 LLM")
    print(f"{'=' * 96}")
    header = (f"{'口径':<26} {'seed_k':>6} {'expand_k':>8} {'上限':>5} "
              f"{'N中位':>6} {'N_p90':>6} {'N_max':>6} {'占全图':>7} "
              f"{'gold微':>7} {'gold宏':>7}")
    print(header)
    print("-" * 96)

    cells: List[Dict[str, Any]] = []

    def _row(label: str, seed_k: int, expand_k: int, cap: int) -> Dict[str, Any]:
        m = _run_cell(retriever, queries, total, seed_k, expand_k, cap)
        cells.append({"label": label, "seed_k": seed_k, "expand_k": expand_k,
                      "cap": cap, **m})
        print(f"{label:<26} {seed_k:>6} {expand_k:>8} {cap or '不限':>5} "
              f"{m['n_median']:>6.0f} {m['n_p90']:>6.0f} {m['n_max']:>6} "
              f"{_fmt_pct(m['coverage'])} {_fmt_pct(m['gold_micro'])} "
              f"{_fmt_pct(m['gold_macro'])}")
        return m

    _row(LEGACY["label"], LEGACY["seed_k"], LEGACY["expand_k"], LEGACY["cap"])
    print("-" * 96)
    for expand_k in expand_ks:
        for cap in caps:
            _row(f"ALM口径 expand_k={expand_k}", alm_seed_k, expand_k, cap)

    # 按类别拆解（只对当前最优/偏差最大的两行做，避免刷屏）
    if has_gold:
        print(f"\n{'—' * 96}\n按能力切面的 gold 覆盖率（macro，只列有 gold 的类别）")
        cats = sorted({c for m in cells for c in m["per_cat"]})
        print(f"{'口径':<26} " + " ".join(f"{c[:13]:>14}" for c in cats))
        for m in cells:
            row = " ".join(
                (f"{m['per_cat'][c] * 100:13.1f}%" if c in m["per_cat"] else f"{'-':>14}")
                for c in cats
            )
            tag = f"{m['label']}(上限{m['cap'] or '不限'})"[:26]
            print(f"{tag:<26} {row}")

    return {"corpus": corpus, "total_nodes": total, "seed_k": alm_seed_k,
            "queries": len(queries), "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description="检索预算 × 输出上限 正交 A/B")
    parser.add_argument("--corpus", default="all",
                        choices=["all", *CORPORA.keys()])
    parser.add_argument("--expand-ks", type=int, nargs="+", default=DEFAULT_EXPAND_KS)
    parser.add_argument("--caps", type=int, nargs="+", default=DEFAULT_CAPS)
    parser.add_argument("--out", default="eval/outputs/exp_budget_cap_ab.json")
    args = parser.parse_args()

    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    results = [experiment(n, args.expand_ks, args.caps) for n in names]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "legacy": LEGACY,
        "max_hops": MAX_HOPS,
        "expand_ks": args.expand_ks,
        "caps": args.caps,
        "results": results,
        "note": ("因子 A=expand_k，因子 B=max_peak_nodes；seed_k 固定为 ALM 现行值。"
                 "目的由查询集注入，全程不调 LLM，结果可复现。"
                 "只记计数与 id，不记记忆正文（0919 §五）。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out}")


if __name__ == "__main__":
    main()
