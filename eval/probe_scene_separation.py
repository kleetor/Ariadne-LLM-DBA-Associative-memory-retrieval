# -*- coding: utf-8 -*-
"""探测：把"数据集分化度"量化成可比较的数——同场景 vs 跨场景的余弦落差。

为什么需要这个测量：
E4/E5 在 163 节点图上都没测出机制差异，而 `τ=0.30` 的过滤通过率是 100%（§8.1）。
此前只能笼统归因成"数据集同质度高"，但**同质到什么程度、τ 到底该多少才能筛到东西**，
没有任何读数。这里直接算节点向量的余弦分层：
- `within`：同一场景内两两余弦（场景内本来就该像）
- `across`：跨场景两两余弦（这才是"无关节点"的相似度）
- **分化宽度 = within − across**：它决定绝对阈值 τ 能落在哪里还能同时"挡住跨场景、留住同场景"

若 across 本身就在 0.5 以上，则任何 τ<0.5 都不可能筛掉跨场景节点 —— 这就是本图的实际情况。

场景分组按节点 id 的第二位（`uh/uc/uw/ud/us/ul/um` 的 h/c/w/d/s/l/m）。

只读向量缓存，不调任何 API。

用法：python -m eval.probe_scene_separation
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

SCENE = {
    "h": "高中",
    "c": "大专",
    "w": "工作",
    "d": "日常健康",
    "s": "社交",
    "l": "学习",
    "m": "杂项",
}


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def main() -> None:
    ap = argparse.ArgumentParser(description="场景分化度探测（可切 embedding 做对照）")
    ap.add_argument("--emb-model", default=None,
                    help="覆盖 EMBEDDING_MODEL（如 BAAI/bge-large-zh-v1.5）")
    ap.add_argument("--cache-key", default=None,
                    help="向量缓存 key；换 embedding 时必须换 key，否则会读到旧模型的向量")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    if args.emb_model:
        os.environ["EMBEDDING_MODEL"] = args.emb_model
    cache_key = args.cache_key or "comprehensive"
    graph, retriever = build_retriever(
        str(ROOT / "data/dba_eval/comprehensive_gold.yaml"),
        vector_cache_key=cache_key)
    vs = retriever.vector_store

    buckets: defaultdict[str, list[str]] = defaultdict(list)
    for nid in graph.graph.nodes():
        if len(nid) >= 2 and nid[1] in SCENE and graph.graph.nodes[nid].get("content"):
            buckets[SCENE[nid[1]]].append(nid)
    buckets = {k: sorted(v) for k, v in sorted(buckets.items())}

    ids, vecs, owner = [], [], []
    for scene, nids in buckets.items():
        got = vs.get_content_vectors(nids)
        for nid, v in zip(nids, got):
            if v is not None:
                ids.append(nid)
                vecs.append(np.asarray(v, dtype="float32"))
                owner.append(scene)
    M = np.vstack(vecs)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    sim = M @ M.T
    owner = np.array(owner)

    print(f"节点={len(ids)}  场景={ {k: len(v) for k, v in buckets.items()} }")
    print(f"embedding={os.environ.get('EMBEDDING_MODEL')}\n")

    # 全量上三角（不含对角线）
    iu = np.triu_indices(len(ids), k=1)
    allc = sim[iu]
    same = owner[iu[0]] == owner[iu[1]]
    within, across = allc[same], allc[~same]

    print("=== 全局 ===")
    print(f"  全部    p10={pct(allc,10):.3f} p50={pct(allc,50):.3f} p90={pct(allc,90):.3f} "
          f"min={allc.min():.3f} max={allc.max():.3f}")
    print(f"  同场景  p10={pct(within,10):.3f} p50={pct(within,50):.3f} p90={pct(within,90):.3f} "
          f"n={within.size}")
    print(f"  跨场景  p10={pct(across,10):.3f} p50={pct(across,50):.3f} p90={pct(across,90):.3f} "
          f"n={across.size}")
    print(f"  >> 分化宽度（同场景均值 − 跨场景均值）= "
          f"{within.mean():.3f} − {across.mean():.3f} = {within.mean()-across.mean():+.3f}")

    print("\n=== 绝对阈值能筛到多少（跨场景对里，低于阈值的比例 = 可被挡掉的无关节点）===")
    for t in (0.30, 0.40, 0.50, 0.60):
        print(f"  τ={t:.2f}  跨场景可挡 {float((across < t).mean()):6.1%}   "
              f"同场景误伤 {float((within < t).mean()):6.1%}")

    # 可分离性 AUC：把"同场景 vs 跨场景"当二分类，用余弦当分数。
    # AUC=0.5 表示两个分布完全重叠（任何阈值都不可能分离）；1.0 表示可完美分开。
    # 它比"最优 margin"更标准，也更能说明"这张图是否支持绝对阈值"。
    from sklearn.metrics import roc_auc_score  # 局部导入，避免脚本级依赖

    y = np.concatenate([np.ones(within.size), np.zeros(across.size)])
    s = np.concatenate([within, across])
    auc = float(roc_auc_score(y, s))
    # 最优 margin（跨场景可挡率 − 同场景误伤率）及其阈值
    margins = []
    for t in np.arange(0.20, 0.91, 0.01):
        margins.append((float((across < t).mean() - (within < t).mean()), float(t)))
    best_m, best_t = max(margins)
    print(f"\n=== 可分离性 ===")
    print(f"  ROC AUC（同场景 vs 跨场景）= {auc:.3f}   "
          f"（0.5=完全重叠、无法用任何阈值分离）")
    print(f"  最优 margin = {best_m:+.3f} @ τ={best_t:.2f}   "
          f"（=可挡跨场景 − 误伤同场景，上限 1.0）")

    print("\n=== 场景两两（均值）===")
    scenes = list(buckets)
    print("        " + "".join(f"{s:>10s}" for s in scenes))
    for a in scenes:
        row = []
        for b in scenes:
            m = sim[np.ix_(owner == a, owner == b)]
            if a == b:
                n = m.shape[0]
                val = (m.sum() - np.trace(m)) / (n * (n - 1)) if n > 1 else float("nan")
            else:
                val = m.mean()
            row.append(f"{val:10.3f}")
        print(f"{a:>7s} " + "".join(row))


if __name__ == "__main__":
    main()
