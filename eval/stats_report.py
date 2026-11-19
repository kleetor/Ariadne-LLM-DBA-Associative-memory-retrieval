# -*- coding: utf-8 -*-
"""主表统计：bootstrap 95% CI + paired Wilcoxon + 跨 judge 一致性（κ/ICC/Spearman）。

为什么单独一个脚本：
- 论文所有主表都要求"均值 + 区间"，而 `score_comprehensive.py` 只出点估计；
- 多 judge 交叉目前只报了「第一名一致 X/70」，缺 κ / ICC / Spearman 与 CI；
- 两类表（E1 系统级、E3 条件级）的还原逻辑相同（都靠 label_map），应共用一份实现，
  否则两处各写一遍必然口径漂移。

输入：若干 judge JSON（`eval/judge_comprehensive.py` 产物）+ 对应 ctx JSON。
若 judge 行内带 `label_map` 就直接用（E3）；否则原样用系统名（E1）。

用法：
    python -m eval.stats_report --judge eval/outputs/e1/judge_e1_qwen.json eval/outputs/e1/judge_e1_deepseek.json eval/outputs/e1/judge_e1_gpt.json --out eval/outputs/e1/stats_e1.json
    python -m eval.stats_report --judge eval/outputs/e3/judge_e3_bandfix_*.json --out eval/outputs/e3/stats_e3.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
N_BOOT = 10000
SEED = 20261117


# ---------- 载入与还原 ----------

def _ctx_label_maps(ctx_path: str | None) -> Dict[str, Dict[str, str]]:
    if not ctx_path:
        return {}
    p = ROOT / ctx_path
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for row in data.get("queries", []):
        lm = row.get("label_map")
        if lm:
            out[row["query_id"]] = lm
    return out


def load_judge(path: str):
    """返回 (judge_model, {qid: {system: total}}, {qid: {system: {dim: score}}}, {qid: category})"""
    with open(ROOT / path, encoding="utf-8") as f:
        data = json.load(f)
    model = (data.get("meta") or {}).get("judge_model", Path(path).stem)
    ctx_maps = _ctx_label_maps((data.get("meta") or {}).get("ctx"))
    totals: Dict[str, Dict[str, float]] = {}
    dims: Dict[str, Dict[str, Dict[str, float]]] = {}
    cats: Dict[str, str] = {}
    for r in data.get("results", []):
        qid = r["query_id"]
        lm = r.get("label_map") or ctx_maps.get(qid)
        sc = r.get("scores") or {}
        if lm:
            inv = {lab: cond for cond, lab in lm.items()}
            sc = {inv.get(k, k): v for k, v in sc.items()}
        if not sc:
            continue
        cats[qid] = str(r.get("category") or "")
        totals[qid] = {k: float((v or {}).get("total", 0)) for k, v in sc.items()}
        dims[qid] = {k: {d: float((v or {}).get(d, 0)) for d in DIMS} for k, v in sc.items()}
    return model, totals, dims, cats


# ---------- 统计工具 ----------

def boot_mean_ci(vals: Sequence[float], n: int = N_BOOT, seed: int = SEED) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对差：均值差 + bootstrap CI + Wilcoxon（双侧）+ 效应量。

    效应量用 **rank-biserial r**（与 Wilcoxon 同源，不假设正态）：
        r = (W+ − W−) / (W+ + W−)，取值 [−1, 1]，0 表示正负对称。
    Cohen's d_z 也一并给出（便于与常见论文数值对照），但主报 r。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    if n == 0:
        return {}
    lo, hi = boot_mean_ci(d)
    out = {
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(d.mean()),
        "diff_ci95": [lo, hi],
        "win": int((d > 0).sum()),
        "tie": int((d == 0).sum()),
        "loss": int((d < 0).sum()),
    }
    nz = d[d != 0]
    if nz.size >= 1:
        try:
            w = stats.wilcoxon(nz, zero_method="wilcox")
            out["wilcoxon_p"] = float(w.pvalue)
        except Exception:
            out["wilcoxon_p"] = float("nan")
        # rank-biserial：用符号秩和的差
        ranks = stats.rankdata(np.abs(nz))
        w_plus = ranks[nz > 0].sum()
        w_minus = ranks[nz < 0].sum()
        denom = w_plus + w_minus
        out["rank_biserial_r"] = float((w_plus - w_minus) / denom) if denom else 0.0
    if nz.size >= 2 and nz.std(ddof=1) > 0:
        out["cohen_dz"] = float(d.mean() / d.std(ddof=1))
    return out


def icc_2k(matrix: np.ndarray) -> Tuple[float, float]:
    """ICC(2,k)：双向随机效应、**绝对一致**、k 名评分者均值的信度。

    matrix: (n_subjects, k_raters)。
    为什么用 ICC(2,1)/ICC(2,k) 而不是去皮尔逊相关：judge 之间存在**系统性的分数水平差**
    （实测 gpt 整体比 qwen 低 0.2–0.3），相关会把这部分差异忽略掉，ICC(2,·) 不会。
    返回 (ICC2k, F)。
    """
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan")
    grand = m.mean()
    ms_r = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # 行（对象）间
    ms_c = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # 列（评分者）间
    resid = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (ms_c - ms_e) / n
    if denom <= 0:
        return float("nan"), float("nan")
    icc = (ms_r - ms_e) / denom
    f = ms_r / ms_e if ms_e > 0 else float("nan")
    return float(icc), float(f)


def krippendorff_alpha_ordinal(m: np.ndarray) -> float:
    """Krippendorff α（序数度量）——对 1–5 有序评分比 Fleiss κ 更合适。

    m: (n_units, k_raters)，允许 NaN（缺失）。
    """
    m = np.asarray(m, dtype=float)
    n_units, _ = m.shape
    vals = m[~np.isnan(m)]
    if vals.size < 2:
        return float("nan")
    uniq = np.unique(vals)
    # 序数度量：δ²(c,k) = (Σ_{g=c..k} n_g − (n_c+n_k)/2)²
    counts = {u: int((vals == u).sum()) for u in uniq}
    total = vals.size

    def delta2(c, k):
        lo, hi = (c, k) if c <= k else (k, c)
        s = 0.0
        for u in uniq:
            if lo <= u <= hi:
                s += counts[u]
        s -= (counts[c] + counts[k]) / 2.0
        return s ** 2

    # 观测到的差异
    do_num = 0.0
    do_den = 0.0
    for i in range(n_units):
        row = m[i][~np.isnan(m[i])]
        ni = row.size
        if ni < 2:
            continue
        for a, b in itertools.combinations(range(ni), 2):
            do_num += delta2(row[a], row[b])
            do_den += 1.0
    do = do_num / do_den if do_den else float("nan")
    # 期望差异
    de_num = 0.0
    for c, k in itertools.combinations_with_replacement(uniq, 2):
        nc, nk = counts[c], counts[k]
        pairs = nc * nk if c == k else nc * nk
        de_num += pairs * delta2(c, k)
    de = de_num / (total * (total - 1) / 2.0) if total > 1 else float("nan")
    if not de or np.isnan(de) or np.isnan(do):
        return float("nan")
    return float(1.0 - do / de)


def spearman_between(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    res = stats.spearmanr(a, b)
    return float(res.statistic)


def rank_of(d: Dict[str, float]) -> List[str]:
    """按分数降序的系统名（同分按名字定序，保证可比）"""
    return [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))]


# ---------- 报告 ----------

def report(paths: List[str], out_path: str | None, exclude_cats: Sequence[str] = ("abstention",)) -> None:
    loaded = [load_judge(p) for p in paths]
    models = [m for m, _, _, _ in loaded]
    totals = [t for _, t, _, _ in loaded]
    dims = [d for _, _, d, _ in loaded]
    cats = loaded[0][3]

    # 取各 judge 共有的题目，再取共有的系统（顺序取第一个 judge 的键序，稳定可读）
    excluded = {c.strip() for c in exclude_cats if c and c.strip()}
    common_q = sorted(set.intersection(*[set(t) for t in totals]))
    n_all = len(common_q)
    if excluded:
        common_q = [q for q in common_q if cats.get(q, "") not in excluded]
    systems = sorted(totals[0][common_q[0]].keys())
    print(f"judges={models}  题目={len(common_q)}（总 {n_all}，排除 {sorted(excluded) or '无'}）"
          f"  系统={systems}\n")

    # ---- 1) 每 judge：各系统均值 + bootstrap CI ----
    print("=" * 78)
    print("1) 各 judge 的系统均值（bootstrap 95% CI，题目重采样）")
    print("=" * 78)
    per_judge = {}
    for mi, m in enumerate(models):
        row = {}
        print(f"\n[{m}]")
        for s in systems:
            vals = [totals[mi][q][s] for q in common_q if s in totals[mi][q]]
            lo, hi = boot_mean_ci(vals)
            row[s] = {"mean": float(np.mean(vals)), "ci95": [lo, hi], "n": len(vals)}
            print(f"  {s:12s} {row[s]['mean']:.3f}  95%CI [{lo:.3f}, {hi:.3f}]  n={len(vals)}")
        per_judge[m] = row

    # ---- 2) 每 judge：两两配对对比 ----
    print("\n" + "=" * 78)
    print("2) 两两配对对比（差值的 bootstrap CI + Wilcoxon + 效应量）")
    print("=" * 78)
    pair_stats = {}
    for mi, m in enumerate(models):
        print(f"\n[{m}]")
        pair_stats[m] = {}
        for a, b in itertools.combinations(systems, 2):
            qs = [q for q in common_q if a in totals[mi][q] and b in totals[mi][q]]
            st = paired_stats([totals[mi][q][a] for q in qs], [totals[mi][q][b] for q in qs])
            pair_stats[m][f"{a} - {b}"] = st
            print(f"  {a} - {b}: {st['diff']:+.3f}  CI95 [{st['diff_ci95'][0]:+.3f}, "
                  f"{st['diff_ci95'][1]:+.3f}]  W/T/L={st['win']}/{st['tie']}/{st['loss']}  "
                  f"p={st.get('wilcoxon_p', float('nan')):.4g}  r_rb={st.get('rank_biserial_r', float('nan')):+.2f}")

    # ---- 3) 跨 judge 一致性 ----
    print("\n" + "=" * 78)
    print("3) 跨 judge 一致性")
    print("=" * 78)
    cons = {}
    # 3a) 逐条（题×系统）Spearman
    print("\n3a) 逐条（题×系统）Spearman ρ")
    for (i, mi), (j, mj) in itertools.combinations(list(enumerate(models)), 2):
        xs, ys = [], []
        for q in common_q:
            for s in systems:
                if s in totals[i][q] and s in totals[j][q]:
                    xs.append(totals[i][q][s])
                    ys.append(totals[j][q][s])
        rho = spearman_between(xs, ys)
        cons.setdefault("spearman_pairs", {})[f"{mi} vs {mj}"] = rho
        print(f"  {mi} vs {mj}: rho={rho:+.3f}  (n={len(xs)})")

    # 3b) 每题内排序一致率
    print("\n3b) 排序一致率（每题内系统完整排序 / 第一名）")
    for (i, mi), (j, mj) in itertools.combinations(list(enumerate(models)), 2):
        same_order = same_top = both = 0
        for q in common_q:
            if not all(s in totals[i][q] and s in totals[j][q] for s in systems):
                continue
            ri = rank_of({s: totals[i][q][s] for s in systems})
            rj = rank_of({s: totals[j][q][s] for s in systems})
            same_order += int(ri == rj)
            same_top += int(ri[0] == rj[0])
            both += 1
        key = f"{mi} vs {mj}"
        cons.setdefault("rank_agreement", {})[key] = {
            "full_order": same_order, "top1": same_top, "n": both}
        print(f"  {key}: 完整 {same_order}/{both}  第一名 {same_top}/{both}")

    # 3c) ICC(2,k) + Krippendorff α：对象 = 题×系统，评分者 = judge
    mat = []
    for q in common_q:
        for s in systems:
            if all(s in totals[i][q] for i in range(len(models))):
                mat.append([totals[i][q][s] for i in range(len(models))])
    if mat:
        M = np.asarray(mat, dtype=float)
        icc, f = icc_2k(M)
        alpha = krippendorff_alpha_ordinal(M)
        cons["icc2k"] = icc
        cons["icc_F"] = f
        cons["krippendorff_alpha_ordinal"] = alpha
        cons["n_units"] = int(M.shape[0])
        print(f"\n3c) ICC(2,k)={icc:.3f}  (F={f:.2f}, 单元={M.shape[0]})")
        print(f"    Krippendorff α(ordinal)={alpha:.3f}")

    # ---- 4) 维度分解（每 judge × 系统）----
    print("\n" + "=" * 78)
    print("4) 维度分解（每 judge × 系统）")
    print("=" * 78)
    dim_report = {}
    for mi, m in enumerate(models):
        print(f"\n[{m}]")
        for s in systems:
            ds = {d: [] for d in DIMS}
            for q in common_q:
                if s in dims[mi].get(q, {}):
                    for d in DIMS:
                        ds[d].append(dims[mi][q][s][d])
            means = {d: float(np.mean(v)) for d, v in ds.items() if v}
            dim_report.setdefault(m, {})[s] = means
            if means:
                print("  {:12s} ".format(s) + "  ".join(
                    f"{d}={means[d]:.2f}" for d in DIMS if d in means))

    payload = {
        "meta": {"judges": models, "systems": systems, "n_queries": len(common_q),
                 "n_queries_all": n_all, "excluded_categories": sorted(excluded),
                 "n_boot": N_BOOT, "seed": SEED,
                 "inputs": paths},
        "means": per_judge,
        "pairs": pair_stats,
        "consistency": cons,
        "dims": dim_report,
    }
    if out_path:
        p = ROOT / out_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="主表统计：CI + Wilcoxon + 跨 judge 一致性")
    ap.add_argument("--judge", nargs="+", required=True, help="若干 judge JSON 路径")
    ap.add_argument("--out", default=None, help="统计结果 JSON 输出路径")
    ap.add_argument("--exclude-categories", default="abstention",
                    help="逗号分隔，默认排除 abstention（与主表 70 条口径一致）；传空串则包含全部")
    args = ap.parse_args()
    report(args.judge, args.out,
           exclude_cats=[c for c in args.exclude_categories.split(",") if c.strip()])


if __name__ == "__main__":
    main()
