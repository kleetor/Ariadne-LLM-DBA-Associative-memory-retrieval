# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""真实图上的「容忍带 → 输出量」曲线（离线；节点零 embedding 成本）

为什么需要它
------------
线上 600 次检索实测 `峰值/候选` 中位 **0.81** ——"寻峰"几乎不筛。要回答"怎么把**过多的输出**
压下来"，必须在**真实图 + 真实向量 + 真实查询**上量，而不是在合成图上推。

数据来源：容器导出的 `alm_data/*.yaml` + `*.index/content_vectors.npz`
（**npz 里就是生产节点的 bge-m3 向量**，所以节点侧不需要任何 embedding 调用）。
查询文本取自 ALM 日志的 `[IO] search 入参`（仅在 `ALM_TRACE_IO=1` 的轮次有）。

每个查询只花 2 次 API 调用：query 向量（FAISS 检索用）+ 目的推断（LLM）。目的**只推断一次**
并在所有格子里复用，保证跨格子可比。

主指标（按"减少过多输出"的目标，不拿 gold 当目标）：
    N          —— 读出量（喂入 StoryRank 的图节点数）
    带比        —— 带内候选 / 候选总数（"寻峰筛掉了多少"）
    N/全图      —— 读出量占该空间节点总数的比例
护栏（压输出不能把图压成"一堆孤点"）：
    诱导边/点   —— 读出集合在图上诱导出的边数 ÷ 节点数
    分量数      —— 诱导子图的弱连通分量数（越接近 1 越连贯）

用法：
    python -u -m eval.exp_band_real --round 7b4608eb --ratios 0 0.3 0.5 0.7
    python -u -m eval.exp_band_real --round 67ef87ff --ratios 0 0.5
"""

import argparse
import glob
import json
import os
import pickle
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402
from eval.exp_budget_cap_ab import _effective_seed_k  # noqa: E402

DATA_DIR = ROOT / "eval" / ".cache" / "alm_data"
CACHE_DIR = ROOT / "eval" / ".cache"
MAX_HOPS = 5
EXPAND_K = 40
IN_RE = re.compile(r"\[IO\] search 入参 空间=(\S+) top_k=(\d+) query=(.*?)\n选项:\n", re.S)


def set_data_dir(p: str) -> None:
    """切换数据目录（0930：支持只导出的子集 alm_data_r3）"""
    global DATA_DIR
    DATA_DIR = Path(p)


def _find_log(prefix: str) -> str:
    for p in glob.glob(str(ROOT / "*-json.log")) + glob.glob(str(ROOT / "almlog" / "*-json.log")):
        if os.path.basename(p).startswith(prefix):
            return p
    raise SystemExit(f"找不到日志 {prefix}*")


def _load_log_queries(path: str) -> Dict[str, List[str]]:
    """空间掩码 → 该空间被问过的 query（保持顺序）

    注意：容器 json-file 驱动会把**多行日志拆成多条记录**（`[IO] search 入参` 与其后的
    `选项:` 往往不在同一条里），所以必须先按"时间戳开头"把记录拼回去，再匹配。
    """
    import json as _json
    ts_re = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} ")
    recs: List[str] = []
    cur = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                text = _json.loads(line).get("log", "")
            except Exception:
                continue
            if ts_re.match(text):
                if cur is not None:
                    recs.append(cur)
                cur = text
            elif cur is not None:
                cur += text
    if cur is not None:
        recs.append(cur)

    out: Dict[str, List[str]] = {}
    for text in recs:
        for m in IN_RE.finditer(text):
            out.setdefault(m.group(1), []).append(" ".join(m.group(3).split()))
    return out


def _space_files(mask: str) -> Tuple[str, str]:
    """掩码 → (yaml 路径, npz 路径)"""
    pre, suf = mask[:4], mask[-4:]
    for y in glob.glob(str(DATA_DIR / "*.yaml")):
        stem = os.path.basename(y)[:-5]
        if stem.startswith(pre) and stem.endswith(suf):
            npz = DATA_DIR / (stem + ".index") / "content_vectors.npz"
            if npz.exists():
                return y, str(npz)
    raise SystemExit(f"空间 {mask} 不在导出数据里")


def _ensure_vector_cache(npz: str, cache_key: str) -> None:
    """把 npz 的生产向量注入评测用的 pkl 缓存 → build_retriever 不再调 embedding"""
    cache = CACHE_DIR / f"{cache_key}_vectors.pkl"
    if cache.exists():
        return
    z = np.load(npz, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    vecs = z["vectors"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump({i: np.asarray(v, dtype=float) for i, v in zip(ids, vecs)}, fh)


def _fmt(v, nd=0, suffix=""):
    if v is None:
        return "  n/a"
    return f"{v:.{nd}f}{suffix}"


def run_round(log_prefix: str, ratios: List[float], limit: int, out_json: Path,
              tolerances: List[float] = None, spaces: List[str] = None,
              expand_ks: List[int] = None, purpose_thrs: List[float] = None) -> Dict[str, Any]:
    """三种扫描模式（按优先级）：
       1) expand_ks / purpose_thrs —— **真正的旋钮**：检索预算与目的发散控制
       2) tolerances —— 绝对带宽
       3) ratios     —— 相对带宽
    """
    log = _find_log(log_prefix)
    queries_by_space = _load_log_queries(log)
    if spaces:
        queries_by_space = {k: v for k, v in queries_by_space.items() if k in set(spaces)}
    print(f"日志 {os.path.basename(log)[:8]}：{len(queries_by_space)} 个空间有 query")
    _load_dotenv(ROOT / ".env")

    # 目的推断缓存：33 条查询 × 每次重跑都调 LLM 太慢，缓存到 json 复用
    pcache_file = CACHE_DIR / "purposes_cache.json"
    pcache: Dict[str, Any] = {}
    if pcache_file.exists():
        try:
            pcache = json.loads(pcache_file.read_text(encoding="utf-8"))
        except Exception:
            pcache = {}

    # 逐空间构建（每个空间一个 retriever；目的只推断一次）
    cases: List[Dict[str, Any]] = []
    for mask, qs in sorted(queries_by_space.items()):
        try:
            yaml_path, npz = _space_files(mask)
        except SystemExit:
            continue
        stem = os.path.basename(yaml_path)[:-5]
        cache_key = f"alm_{stem[:12]}"
        _ensure_vector_cache(npz, cache_key)
        graph, retriever = build_retriever(yaml_path, vector_cache_key=cache_key)
        total = graph.node_count
        for q in (qs[:limit] if limit else qs):
            if q in pcache:
                purposes = pcache[q] or None
            else:
                info = retriever.inference.infer_purpose(q)      # LLM：每查询 1 次（缓存后不再调）
                purposes = info.get("purposes") or None
                pcache[q] = purposes
            cases.append({"mask": mask, "q": q, "purpose": purposes,
                          "total": total, "graph": graph, "retriever": retriever})
    try:
        pcache_file.parent.mkdir(parents=True, exist_ok=True)
        pcache_file.write_text(json.dumps(pcache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    print(f"可用案例 {len(cases)} 条（查询 × 空间）")
    if not cases:
        raise SystemExit("没有可用案例——检查该轮是否有 query 留痕（需 ALM_TRACE_IO=1）"
                         "且空间是否在导出数据里")

    # 格子定义：检索预算/目的发散 > 绝对带宽 > 相对带宽
    #   purpose_thrs 里的 None = 不动该参数（沿用构建时的默认 0.30）
    if expand_ks or purpose_thrs:
        eks = expand_ks or [EXPAND_K]
        pths = purpose_thrs or [None]
        grid = []
        for e in eks:
            for p in pths:
                label = f"exp={e}" + (f"/pur={p}" if p is not None else "")
                grid.append({"mode": "预算/目的", "label": label, "expand_k": e,
                             "purpose_thr": p, "ratio": None, "tol": None})
    elif tolerances:
        grid = [{"mode": "绝对", "label": f"δ={t}", "ratio": None, "tol": t,
                 "expand_k": EXPAND_K, "purpose_thr": None} for t in tolerances]
    else:
        grid = [{"mode": "相对", "label": f"ratio={r}", "ratio": r, "tol": None,
                 "expand_k": EXPAND_K, "purpose_thr": None} for r in ratios]

    cells: List[Dict[str, Any]] = []
    per_case: List[Dict[str, Any]] = []
    for g in grid:
        ratio, tol = g["ratio"], g["tol"]
        n_list, br_list, sp_list, cov_list, epn_list, comp_list = [], [], [], [], [], []
        for c in cases:
            r = c["retriever"]
            if g["purpose_thr"] is not None:
                r.purpose_filter_threshold = g["purpose_thr"]     # 目的向量侧：发散方向的门槛
            if tol is not None:
                r.peak_finder.peak_tolerance = tol
                r.peak_tolerance_ratio = 0.0
            elif ratio is not None:
                r.peak_tolerance_ratio = ratio
            res = r.retrieve(c["q"], seed_k=_effective_seed_k(c["total"]),
                             expand_k=g["expand_k"], max_hops=MAX_HOPS,
                             purpose=c["purpose"], max_peak_nodes=None)
            fed = [mid for mid, _ in (res.get("peak_memories") or [])]
            fed = [mid for mid in fed if mid in c["graph"].graph]   # 护栏：只统计图内节点
            n_list.append(len(fed))
            cov_list.append(len(fed) / c["total"] if c["total"] else 0.0)
            band = res.get("peak_band") or {}
            br_list.append(float(band.get("band_ratio", 0.0)))
            means = [float(h.get("mean_score", 0)) for h in (res.get("hop_history") or [])]
            sp_list.append(max(means) - min(means) if means else 0.0)
            sub = c["graph"].graph.subgraph(fed)                  # 护栏：诱导子图
            epn_list.append(sub.number_of_edges() / len(fed) if fed else 0.0)
            comp_list.append(nx.number_weakly_connected_components(sub) if fed else 0)
            per_case.append({"label": g["label"], "mask": c["mask"], "q": c["q"][:60],
                             "total": c["total"], "n": len(fed), "cov": cov_list[-1],
                             "band_ratio": br_list[-1], "hop_spread": sp_list[-1],
                             "edge_per_node": epn_list[-1], "components": comp_list[-1],
                             "band_seen": bool(band)})
        m = {"label": g["label"], "mode": g["mode"], "expand_k": g["expand_k"],
             "purpose_thr": g["purpose_thr"],
             "n_median": statistics.median(n_list),
             "n_p90": sorted(n_list)[int(len(n_list) * 0.9)],
             "n_max": max(n_list), "n_min": min(n_list),
             "cov_median": statistics.median(cov_list),
             "band_ratio_median": statistics.median(br_list),
             "hop_spread_median": statistics.median(sp_list),
             "edge_per_node_median": statistics.median(epn_list),
             "components_median": statistics.median(comp_list)}
        cells.append(m)
        print(f"{m['label']:<18} N 中位={m['n_median']:>5.0f} p90={m['n_p90']:>5} max={m['n_max']:>5} "
              f"| N/全图={m['cov_median'] * 100:>5.1f}% | 带比={m['band_ratio_median'] * 100:>5.1f}% "
              f"| 轮均分全距={m['hop_spread_median']:.4f} | 诱导边/点={m['edge_per_node_median']:.2f} "
              f"| 分量={m['components_median']:.0f}")

    # 按图规模分层（小图有 139/210 个，会淹没整体统计；"过多输出"的问题只出在大图上）
    buckets = [("≤49", 1, 50), ("50-199", 50, 200), ("200-499", 200, 500), ("500+", 500, 10 ** 9)]
    print(f"\n{'—' * 112}\n按图规模分层（每格：案例数 / N中位 / N占该档图的中位 / 带比中位 / 走到容忍带的案例数）")
    print(f"{'规模档':<10}" + "".join(f"{c['label']:>26}" for c in cells))
    for name, lo, hi in buckets:
        line = f"{name:<10}"
        for c in cells:
            sel = [x for x in per_case if x["label"] == c["label"]
                   and lo <= x["total"] < hi]
            if not sel:
                line += f"{'-':>26}"
                continue
            nmed = statistics.median([x["n"] for x in sel])
            cmed = statistics.median([x["cov"] for x in sel])
            bmed = statistics.median([x["band_ratio"] for x in sel])
            seen = sum(1 for x in sel if x["band_seen"])
            line += f"{len(sel):>4} / {nmed:>5.0f} / {cmed * 100:>5.1f}% / {bmed * 100:>5.1f}% / {seen:>2}"
        print(line)
    cells_out = {"log": os.path.basename(log), "spaces": len(queries_by_space), "cases": len(cases),
                 "max_hops": MAX_HOPS, "expand_k": EXPAND_K, "cells": cells,
                 "per_case": per_case}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(cells_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写: {out_json}")
    return cells_out


def main() -> None:
    ap = argparse.ArgumentParser(description="真实图上的容忍带 → 输出量曲线")
    ap.add_argument("--round", default="7b4608eb", help="日志前缀（如 7b4608eb / 67ef87ff / 2744e17c）")
    ap.add_argument("--data-dir", default=None, help="图/向量目录（默认 eval/.cache/alm_data）")
    ap.add_argument("--spaces", default=None, help="只跑这些空间掩码，逗号分隔")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7],
                    help="相对带宽扫描（ratio × 轮均分全距）")
    ap.add_argument("--tols", type=float, nargs="+", default=None,
                    help="**绝对**带宽扫描（如 0.10 0.05 0.03）；给了它就忽略 --ratios")
    ap.add_argument("--expand-ks", type=int, nargs="+", default=None,
                    help="检索预算扫描（真正的瓶颈）；给了它优先于带宽")
    ap.add_argument("--purpose-thrs", type=float, nargs="+", default=None,
                    help="目的过滤阈值扫描（控制发散方向）")
    ap.add_argument("--limit", type=int, default=2, help="每个空间最多取几条 query（0=不限）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.data_dir:
        set_data_dir(args.data_dir)
    out = Path(args.out) if args.out else ROOT / f"eval/outputs/exp_band_real_{args.round}.json"
    run_round(args.round, args.ratios, args.limit, out, tolerances=args.tols,
              spaces=[s.strip() for s in args.spaces.split(",")] if args.spaces else None,
              expand_ks=args.expand_ks, purpose_thrs=args.purpose_thrs)


if __name__ == "__main__":
    main()
