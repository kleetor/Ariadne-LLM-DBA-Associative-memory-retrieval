# -*- coding: utf-8 -*-
"""E13：分位阈值扫描 + pAUC（L / M / H，bge-m3 与 bge-large-zh-v1.5）。

回答两个问题：
1. 只看高分段（FPR ≤ 0.10）时，可分性是否明显高于全局 AUC（pAUC）？
2. 把绝对阈值 τ 换成**分位阈值**（τ = Q_{1-p}(cos)，p 为保留比例）后，
   同一 p 在不同图、不同 embedding 上的混入率是否稳定（跨图波动 < 5pp）？

口径：
- 相关对 = 同主体 **且** 同场景；无关对 = 其余，与 E10 一致；
- pAUC：ROC 曲线下 FPR ∈ [0, 0.10] 的面积，除以随机基线 0.5 × 0.1² = 0.005；
  > 1 表示高分段优于随机；
- 分位阈值在**全图节点对余弦分布**上取（不是每跳候选分布），因此：
  `keep_within` / `keep_across` 是「保留的对占该类对的比例」，
  `mix` = 保留对里跨主体/跨场景对所占比例（越低越干净）。

用法：
    python -m eval.e13_quantile
    python -m eval.e13_quantile --cache-suffix zh15 --out eval/outputs/e13/quantile_zh15.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from eval.e10_probe import GRAPHS, _load_vectors  # noqa: E402
from eval.e14_auc_ci import _auc_from_scores, _pair_sets  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

P_GRID = (0.10, 0.15, 0.20, 0.30, 0.50)
PAUC_FPR = 0.10
RANDOM_PAUC = 0.5 * PAUC_FPR ** 2  # = 0.005
QUERIES = ROOT / "data/dba_eval/comprehensive_queries.yaml"


def _tpr_at(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    """在 ROC 上按线性插值取 FPR=target 的 TPR。"""
    if fpr.max() < target:
        return float(tpr[-1])
    return float(np.interp(target, fpr, tpr))


def _roc_metrics(within: np.ndarray, across: np.ndarray) -> Dict:
    y = np.concatenate([np.ones(within.size), np.zeros(across.size)])
    score = np.concatenate([within, across])
    fpr, tpr, _ = roc_curve(y, score)
    mask = fpr <= PAUC_FPR
    if mask.sum() >= 2:
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        raw = float(trapz(tpr[mask], fpr[mask]))
    else:
        raw = 0.0
    return {
        "auc": round(_auc_from_scores(within, across), 4),
        "pauc_raw": round(raw, 5),
        "pauc_norm": round(raw / RANDOM_PAUC, 2),
        "tpr_at_fpr05": round(_tpr_at(fpr, tpr, 0.05), 3),
        "tpr_at_fpr10": round(_tpr_at(fpr, tpr, 0.10), 3),
    }


def _quantile_curve(within: np.ndarray, across: np.ndarray, grid=P_GRID) -> List[Dict]:
    allc = np.concatenate([within, across])
    rows = []
    for p in grid:
        tau = float(np.quantile(allc, 1.0 - p))
        kw = float((within >= tau).mean())
        ka = float((across >= tau).mean())
        denom = kw + ka
        rows.append({
            "p": p,
            "tau": round(tau, 4),
            "keep_within": round(kw, 4),
            "keep_across": round(ka, 4),
            "mix": round(ka / denom, 4) if denom else None,
            "blocked_across": round(1.0 - ka, 4),
        })
    return rows


def _run(suffix: str) -> Dict:
    out = {}
    for name, (yaml_path, key) in GRAPHS.items():
        if suffix:
            key = f"{key}_{suffix}"
        ids, subs, M = _load_vectors(yaml_path, key)
        scenes = [n[1] if len(n) >= 2 else "?" for n in ids]
        within, across = _pair_sets(M @ M.T, subs, scenes)
        out[name] = {
            "n_nodes": len(ids),
            "n_within": int(within.size),
            "n_across": int(across.size),
            **_roc_metrics(within, across),
            "quantile": _quantile_curve(within, across),
        }
    return out


def _stability(graphs: Dict) -> Dict:
    """同一 p 下 mix / keep_across 的跨图极差（max−min）。"""
    out = {}
    for p in P_GRID:
        mixes = [g["quantile"][i]["mix"] for g in graphs.values() for i, r in enumerate(g["quantile"])
                 if r["p"] == p and g["quantile"][i]["mix"] is not None]
        keeps = [r["keep_across"] for g in graphs.values() for r in g["quantile"] if r["p"] == p]
        out[str(p)] = {
            "mix_range_pp": round((max(mixes) - min(mixes)) * 100, 2) if mixes else None,
            "keep_across_range_pp": round((max(keeps) - min(keeps)) * 100, 2) if keeps else None,
        }
    return out


def _load_queries() -> List[Dict]:
    with open(QUERIES, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [q for q in (data.get("queries") or [])
            if q.get("category") != "abstention" and q.get("gold_expected")]


def _query_vectors(suffix: str) -> Dict[str, np.ndarray]:
    """查询向量（按 embedding 模型缓存到 JSON，避免重复调用 API）。"""
    p = ROOT / f"eval/outputs/e13/query_vecs_{suffix or 'm3'}.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: np.asarray(v, dtype="float32") for k, v in raw.items()}
    key = f"e10_L_{suffix}" if suffix else "e10_L"
    _, retriever = build_retriever(str(ROOT / "data/dba_eval/e10_L.yaml"), vector_cache_key=key)
    out = {str(q["id"]): np.asarray(retriever.vector_store._embed(str(q["text"])), dtype="float32")
           for q in _load_queries()}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: v.tolist() for k, v in out.items()}, ensure_ascii=False),
                 encoding="utf-8")
    return out


def _query_level(suffix: str) -> Dict:
    """查询-节点余弦上的分位门控（离线近似，不是 PAR 逐跳目的过滤）。

    对每个查询，保留 cos(query, node) ≥ τ_p 的节点，记录平均输出条数与 gold 命中率。
    τ_p 仍取全图节点对余弦的分位，因此这是「同一分位参数跨图/跨模型」的可移植性检查。
    """
    qs = _load_queries()
    qv = _query_vectors(suffix)
    out = {}
    for name, (yaml_path, key) in GRAPHS.items():
        if suffix:
            key = f"{key}_{suffix}"
        ids, _, M = _load_vectors(yaml_path, key)
        allc = (M @ M.T)[np.triu_indices(len(ids), k=1)]
        pos = {nid: i for i, nid in enumerate(ids)}
        rows = {p: {"sizes": [], "recall": []} for p in P_GRID}
        for q in qs:
            v = qv.get(str(q["id"]))
            if v is None:
                continue
            v = v / (np.linalg.norm(v) + 1e-8)
            sim = M @ v
            gold = [g for g in (q.get("gold_expected") or []) if g in pos]
            if not gold:
                continue
            gold_idx = [pos[g] for g in gold]
            for p in P_GRID:
                tau = float(np.quantile(allc, 1.0 - p))
                keep = sim >= tau
                rows[p]["sizes"].append(int(keep.sum()))
                rows[p]["recall"].append(float(keep[gold_idx].mean()))
        out[name] = {
            "n_queries": len(rows[P_GRID[0]]["sizes"]),
            "quantile": [{
                "p": p,
                "mean_output": round(float(np.mean(rows[p]["sizes"])), 1),
                "gold_recall": round(float(np.mean(rows[p]["recall"])), 3),
            } for p in P_GRID],
        }
    return out


def _query_stability(ql: Dict) -> Dict:
    out = {}
    for i, p in enumerate(P_GRID):
        rec = [g["quantile"][i]["gold_recall"] for g in ql.values()]
        size = [g["quantile"][i]["mean_output"] for g in ql.values()]
        out[str(p)] = {
            "gold_recall_range_pp": round((max(rec) - min(rec)) * 100, 2),
            "mean_output_range": round(max(size) - min(size), 1),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E13：分位阈值 + pAUC")
    ap.add_argument("--out", default="eval/outputs/e13/quantile.json")
    ap.add_argument("--cache-suffix", default="", help="zh15 = bge-large-zh-v1.5")
    ap.add_argument("--emb-model", default=None, help="覆盖 EMBEDDING_MODEL（查询向量用）")
    ap.add_argument("--skip-query", action="store_true", help="跳过查询级 gold 命中（不调 embedding API）")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    if args.emb_model:
        os.environ["EMBEDDING_MODEL"] = args.emb_model
    elif args.cache_suffix == "zh15":
        os.environ["EMBEDDING_MODEL"] = "BAAI/bge-large-zh-v1.5"
    suffix = args.cache_suffix or "m3"
    graphs = _run(args.cache_suffix)
    stability = _stability(graphs)
    qlevel = {} if args.skip_query else _query_level(args.cache_suffix)
    qstab = _query_stability(qlevel) if qlevel else {}

    print(f"\n[{suffix}] 全局 AUC 与局部 pAUC")
    print(f"{'图':4s} {'AUC':>6s} {'pAUC(norm)':>11s} {'TPR@5%':>8s} {'TPR@10%':>8s}")
    for name, g in graphs.items():
        print(f"{name:4s} {g['auc']:6.3f} {g['pauc_norm']:11.2f} "
              f"{g['tpr_at_fpr05']:8.3f} {g['tpr_at_fpr10']:8.3f}")

    print(f"\n[{suffix}] 分位阈值（τ = Q_(1-p)）")
    print(f"{'图':4s} {'p':>5s} {'τ':>7s} {'keep_within':>12s} {'keep_across':>12s} {'mix':>7s}")
    for name, g in graphs.items():
        for r in g["quantile"]:
            print(f"{name:4s} {r['p']:5.2f} {r['tau']:7.3f} {r['keep_within']:12.3f} "
                  f"{r['keep_across']:12.3f} {str(r['mix']):>7s}")

    print(f"\n[{suffix}] 跨图稳定性（同一 p 的极差，pp；判据 mix < 5pp）")
    for p, s in stability.items():
        print(f"  p={p}: mix 极差 {s['mix_range_pp']}pp, keep_across 极差 "
              f"{s['keep_across_range_pp']}pp")

    if qlevel:
        print(f"\n[{suffix}] 查询级分位门控（离线近似：cos(query,node) ≥ τ_p）")
        print(f"{'图':4s} {'p':>5s} {'平均输出':>9s} {'gold recall':>12s}")
        for name, g in qlevel.items():
            for r in g["quantile"]:
                print(f"{name:4s} {r['p']:5.2f} {r['mean_output']:9.1f} "
                      f"{r['gold_recall']:12.3f}")
        print(f"\n[{suffix}] 查询级跨图稳定性")
        for p, s in qstab.items():
            print(f"  p={p}: recall 极差 {s['gold_recall_range_pp']}pp, "
                  f"输出条数极差 {s['mean_output_range']}")

    payload = {
        "meta": {
            "experiment": "E13",
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "cache_suffix": args.cache_suffix,
            "pauc_fpr": PAUC_FPR,
            "random_pauc": RANDOM_PAUC,
            "p_grid": list(P_GRID),
            "note": ("分位阈值取全图节点对余弦分布；mix = 保留对中无关对占比，"
                     "越低越干净。跨图极差 < 5pp 才算可移植。"
                     "查询级结果是 cos(query,node) 门控的离线近似，不是 PAR 逐跳目的过滤。"),
        },
        "graphs": graphs,
        "stability": stability,
        "query_level": qlevel,
        "query_stability": qstab,
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E13] wrote {p}")


if __name__ == "__main__":
    main()
