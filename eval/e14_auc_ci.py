# -*- coding: utf-8 -*-
"""E14：L / M / H 三图 AUC 的 bootstrap 95% CI。

为什么要做：正文里 0.717 / 0.744 / 0.767 只差 0.05，而 `e10/probe.json` 的配对数
（1.3 万–5.9 万）足够做 bootstrap。没有区间就不能说「随规模上升」。

口径与 E10 一致：
- 相关对 = 同主体 **且** 同场景；无关对 = 其余（含同主体跨场景、跨主体）。
- AUC = 随机取一相关对与一无关对、相关对余弦更高的概率。

两种重采样，分开报：
1. **pair bootstrap**（计划 §3.5 的写法）：分别对相关/无关对重采样，保留原类别样本量；
   缺点是把共享节点的对当成独立样本，区间会偏窄。
2. **node bootstrap**（更保守）：对节点重采样，重建节点对；区间更宽，作为稳健性检查。
   节点重采样次数默认 2000，够估 95% CI。

用法：
    python -m eval.e14_auc_ci
    python -m eval.e14_auc_ci --cache-suffix zh15 --out eval/outputs/e14/auc_ci_zh15.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from eval.e10_probe import GRAPHS, _load_vectors, _subject  # noqa: E402
from eval.run_p_baseline import _load_dotenv  # noqa: E402


def _pair_sets(sim: np.ndarray, subs: List[str], scenes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (within, across) 两组余弦值，口径同 e10_probe._metrics。"""
    sub, sc = np.array(subs), np.array(scenes)
    iu = np.triu_indices(len(subs), k=1)
    allc = sim[iu]
    same = (sub[iu[0]] == sub[iu[1]]) & (sc[iu[0]] == sc[iu[1]])
    return allc[same], allc[~same]


def _auc_from_scores(within: np.ndarray, across: np.ndarray) -> float:
    """AUC = P(pos > neg) + 0.5 P(tie)，用排序 + searchsorted 计算（与 roc_auc_score 等价）。"""
    if within.size == 0 or across.size == 0:
        return float("nan")
    neg = np.sort(across)
    left = np.searchsorted(neg, within, side="left")
    right = np.searchsorted(neg, within, side="right")
    return float((left.sum() + 0.5 * (right - left).sum()) / (neg.size * within.size))


def _boot_pair(within: np.ndarray, across: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    """对节点对重采样（保留两类样本量）。"""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        w = within[rng.integers(0, within.size, within.size)]
        a = across[rng.integers(0, across.size, across.size)]
        vals[i] = _auc_from_scores(w, a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _boot_node(sim: np.ndarray, subs: List[str], scenes: List[str],
               n_boot: int, seed: int) -> Tuple[float, float]:
    """对节点重采样（有放回），重建节点对。共享节点导致的配对相关被保留下来。"""
    rng = np.random.default_rng(seed)
    n = len(subs)
    sub = np.array(subs)
    sc = np.array(scenes)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        s = sim[np.ix_(idx, idx)]
        sb, scb = sub[idx], sc[idx]
        iu = np.triu_indices(n, k=1)
        allc = s[iu]
        same = (sb[iu[0]] == sb[iu[1]]) & (scb[iu[0]] == scb[iu[1]])
        w, a = allc[same], allc[~same]
        vals[i] = _auc_from_scores(w, a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="E14：AUC bootstrap CI")
    ap.add_argument("--out", default="eval/outputs/e14/auc_ci.json")
    ap.add_argument("--cache-suffix", default="",
                    help="向量缓存 key 后缀（zh15 = bge-large-zh-v1.5）")
    ap.add_argument("--n-boot-pair", type=int, default=10000)
    ap.add_argument("--n-boot-node", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    report: Dict[str, dict] = {}
    print(f"{'图':4s} {'对数':>8s} {'AUC':>7s} {'pair CI95':>20s} {'node CI95':>20s}")
    for name, (yaml_path, key) in GRAPHS.items():
        if args.cache_suffix:
            key = f"{key}_{args.cache_suffix}"
        ids, subs, M = _load_vectors(yaml_path, key)
        scenes = [n[1] if len(n) >= 2 else "?" for n in ids]
        sim = M @ M.T
        within, across = _pair_sets(sim, subs, scenes)
        auc = _auc_from_scores(within, across)
        plo, phi = _boot_pair(within, across, args.n_boot_pair, args.seed)
        nlo, nhi = _boot_node(sim, subs, scenes, args.n_boot_node, args.seed)
        report[name] = {
            "n_nodes": len(ids),
            "n_within": int(within.size),
            "n_across": int(across.size),
            "auc": round(auc, 4),
            "pair_ci95": [round(plo, 4), round(phi, 4)],
            "node_ci95": [round(nlo, 4), round(nhi, 4)],
        }
        print(f"{name:4s} {within.size + across.size:8d} {auc:7.3f} "
              f"[{plo:.3f}, {phi:.3f}]".rjust(21)
              + f" [{nlo:.3f}, {nhi:.3f}]".rjust(21))

    # 三档之间是否可区分：区间重叠即「无显著差异」
    names = list(report)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo = max(report[a]["node_ci95"][0], report[b]["node_ci95"][0])
            hi = min(report[a]["node_ci95"][1], report[b]["node_ci95"][1])
            overlap[f"{a}-{b}"] = bool(lo <= hi)
    print("\n节点 bootstrap 区间重叠（True = 无显著差异）：" + str(overlap))

    payload = {
        "meta": {
            "experiment": "E14",
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "cache_suffix": args.cache_suffix,
            "n_boot_pair": args.n_boot_pair,
            "n_boot_node": args.n_boot_node,
            "seed": args.seed,
            "note": "pair bootstrap 偏窄；node bootstrap 保留共享节点带来的相关，作主区间。",
        },
        "graphs": report,
        "node_ci_overlap": overlap,
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E14] wrote {p}")


if __name__ == "__main__":
    main()
