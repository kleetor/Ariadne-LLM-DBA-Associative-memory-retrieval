# -*- coding: utf-8 -*-
"""E10-A / E10-B：分化度标定（AUC）与阈值分离能力曲线。

A：对 L / M / H 三套图算「同主体 vs 跨主体」余弦分层与 ROC AUC。
   判据（§8.18）：AUC ≥ 0.85 且最优 margin ≥ 0.55 才算"支持机制验证"；
   < 0.75 直接废弃。
B：在每套图上把 τ 从 0.30 扫到 0.85，看「可挡跨主体 / 误伤同主体 / margin / 最优 τ*」。
   要的是**宽平台**（τ* ±0.05 内 margin 掉 <10pp），不是尖峰。

主体判定按节点 id 首位：u/a/t → A（原主体），b → B（语文老师），c → C（电商运营）。
场景仍按第二位（用于确认跨主体对里也含跨场景结构）。

用法：python -m eval.e10_probe
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

GRAPHS = {
    "L": ("data/dba_eval/e10_L.yaml", "e10_L"),
    "M": ("data/dba_eval/e10_M.yaml", "e10_M"),
    "H": ("data/dba_eval/e10_H.yaml", "e10_H"),
}


def _subject(nid: str) -> str:
    c = nid[0]
    if c in ("u", "a", "t"):
        return "A"
    return c.upper()


def _load_vectors(yaml_path: str, cache_key: str):
    graph, retriever = build_retriever(str(ROOT / yaml_path), vector_cache_key=cache_key)
    vs = retriever.vector_store
    ids, subs = [], []
    for nid in graph.graph.nodes():
        if graph.graph.nodes[nid].get("content"):
            ids.append(nid)
            subs.append(_subject(nid))
    vecs = [v for v in vs.get_content_vectors(ids)]
    keep = [(n, s, v) for n, s, v in zip(ids, subs, vecs) if v is not None]
    M = np.vstack([np.asarray(v, dtype="float32") for _, _, v in keep])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return [n for n, _, _ in keep], [s for _, s, _ in keep], M


def _metrics(sim: np.ndarray, subs: List[str], scenes: List[str]) -> Dict:
    """统一口径（L/M/H 可比）：
      相关对 = 同主体 **且** 同场景
      无关对 = 跨主体，或同主体但跨场景
    L 图只有主体 A，于是退化为"同场景 vs 跨场景"，与 §8.15 的 0.717 一致；
    M/H 的无关对以跨主体为主，分化度应明显上升。"""
    sub, sc = np.array(subs), np.array(scenes)
    iu = np.triu_indices(len(subs), k=1)
    allc = sim[iu]
    same = (sub[iu[0]] == sub[iu[1]]) & (sc[iu[0]] == sc[iu[1]])
    cross_subject = sub[iu[0]] != sub[iu[1]]
    within, across = allc[same], allc[~same]
    if within.size == 0 or across.size == 0:
        return {}
    y = np.concatenate([np.ones(within.size), np.zeros(across.size)])
    auc = float(roc_auc_score(y, np.concatenate([within, across])))
    curve = []
    for t in np.arange(0.20, 0.91, 0.01):
        block = float((across < t).mean())
        hurt = float((within < t).mean())
        curve.append({"tau": round(float(t), 2), "block": block, "hurt": hurt,
                      "margin": block - hurt})
    best = max(curve, key=lambda x: x["margin"])
    near = [c["margin"] for c in curve if abs(c["tau"] - best["tau"]) <= 0.0501]
    return {
        "n_pairs": int(within.size + across.size),
        "n_cross_subject": int(cross_subject.sum()),
        "within_mean": round(float(within.mean()), 3),
        "across_mean": round(float(across.mean()), 3),
        "gap": round(float(within.mean() - across.mean()), 3),
        "auc": round(auc, 4),
        "best_margin": round(best["margin"], 3),
        "best_tau": best["tau"],
        "plateau_drop": round(best["margin"] - min(near), 3) if near else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="E10-A/B 分化度与阈值曲线")
    ap.add_argument("--out", default="eval/outputs/e10/probe.json")
    ap.add_argument("--emb-model", default=None, help="覆盖 EMBEDDING_MODEL（如 BAAI/bge-large-zh-v1.5）")
    ap.add_argument("--cache-suffix", default="",
                    help="向量缓存 key 后缀；换 embedding 时必须给，否则会读到旧模型的向量")
    args = ap.parse_args()
    _load_dotenv(ROOT / ".env")
    if args.emb_model:
        os.environ["EMBEDDING_MODEL"] = args.emb_model

    report = {}
    print(f"{'图':4s} {'节点':>5s} {'同主体':>8s} {'跨主体':>8s} {'gap':>7s} "
          f"{'AUC':>6s} {'最优margin':>10s} {'τ*':>5s} {'平台跌幅':>8s}")
    for name, (yaml_path, key) in GRAPHS.items():
        if args.cache_suffix:
            key = f"{key}_{args.cache_suffix}"
        ids, subs, M = _load_vectors(yaml_path, key)
        scenes = [n[1] if len(n) >= 2 else "?" for n in ids]
        sim = M @ M.T
        m = _metrics(sim, subs, scenes)
        report[name] = {"n_nodes": len(ids),
                        "subjects": {s: subs.count(s) for s in sorted(set(subs))}, **m}
        print(f"{name:4s} {len(ids):5d} {m['within_mean']:8.3f} {m['across_mean']:8.3f} "
              f"{m['gap']:+7.3f} {m['auc']:6.3f} {m['best_margin']:10.3f} "
              f"{m['best_tau']:5.2f} {m['plateau_drop']:8.3f}")

    print("\n判据（§8.18）：AUC ≥ 0.85 且 最优 margin ≥ 0.55 才支持机制验证；< 0.75 废弃。")
    for name, r in report.items():
        ok = r.get("auc", 0) >= 0.85 and r.get("best_margin", 0) >= 0.55
        print(f"  {name}: {'✅ 合格' if ok else '❌ 不合格'} "
              f"(AUC {r.get('auc')}, margin {r.get('best_margin')})")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": {"experiment": "E10-A/B",
                                      "embedding": os.environ.get("EMBEDDING_MODEL")},
                             "graphs": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E10] wrote {p}")


if __name__ == "__main__":
    main()
