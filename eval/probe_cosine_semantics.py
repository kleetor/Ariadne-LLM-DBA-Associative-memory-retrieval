# -*- coding: utf-8 -*-
"""探测 1：余弦 0.3 在 bge-m3 上到底对应"多无关"。

为什么要单独验这个数：
`purpose_filter_threshold=0.30` 是照搬一个"整数好看"的默认值，从没标定过。
而 0.3 是不是合理，取决于**"完全无关的文本对"在这套 embedding 上能得多少分**——
如果无关对被压到 0.1，0.3 就是宽得离谱；如果无关对本来就有 0.35，0.3 就是空转。
两者结论相反，必须实测。

做法：以图内 7 个场景各取 1 条节点为锚，对它们算四类文本的余弦：
  A. 同场景节点（应高）
  B. 跨场景节点（本图的"无关"）+ 用户真实提出的跨场景对
  C. 完全无关的中文（财经 / 物理 / 菜谱）——**外部无关**
  D. 无关的英文 / 代码
若 C/D 也落在 0.3~0.45，说明 0.3 挡不住真正的无关内容，该阈值在本模型上语义失效。

只调 embedding，不调 LLM。

用法：python -m eval.probe_cosine_semantics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

# 锚点：每个场景取一条最有代表性的节点正文
ANCHORS = {
    "高中": "用户高三那年为了高考熬夜刷题",
    "大专": "用户大二下学期偶然接触了编程",
    "工作": "用户这周项目很赶，每天加班到十一点",
    "日常健康": "用户这周开始吃减脂餐",
    "社交": "用户这周六和朋友约了聚餐",
    "学习": "用户上周报了一个数据分析网课",
    "杂项": "用户曾经养过一只橘猫",
}

# 外部"完全无关"文本 —— 与用户记忆毫无语义关系
UNRELATED_ZH = [
    "今天沪深300指数下跌了2%，成交量萎缩。",
    "量子纠缠的贝尔不等式实验在低温超导条件下被验证。",
    "这道菜需要先把番茄炒出汁，再加高汤小火慢炖四十分钟。",
    "今年第三季度全国铁路旅客发送量同比增长百分之八。",
]
UNRELATED_EN = [
    "The Federal Reserve raised interest rates by 25 basis points.",
    "Photosynthesis converts light energy into chemical energy in chloroplasts.",
]
UNRELATED_CODE = [
    "def quicksort(arr): return sorted(arr)",
    "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 3",
]


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description="余弦 0.3 的语义标定（可切 embedding 做对照）")
    ap.add_argument("--emb-model", default=None,
                    help="覆盖 EMBEDDING_MODEL（如 BAAI/bge-large-zh-v1.5）")
    ap.add_argument("--cache-key", default=None, help="向量缓存 key（换 embedding 必须换）")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    if args.emb_model:
        os.environ["EMBEDDING_MODEL"] = args.emb_model
    graph, retriever = build_retriever(
        str(ROOT / "data/dba_eval/comprehensive_gold.yaml"),
        vector_cache_key=args.cache_key or "comprehensive")
    vs = retriever.vector_store

    scene_of = {}
    for nid in graph.graph.nodes():
        if len(nid) >= 2 and nid[1] in "hcwdslm":
            scene_of[nid] = nid[1]

    names = list(ANCHORS)
    anchor_vecs = [unit(vs._embed(t)) for t in ANCHORS.values()]

    # 取图内节点向量，按场景索引
    in_scene = {"h": [], "c": [], "w": [], "d": [], "s": [], "l": [], "m": []}
    for nid, sc in scene_of.items():
        c = graph.graph.nodes[nid].get("content")
        if c:
            in_scene[sc].append((nid, c))
    for k in in_scene:
        in_scene[k].sort()
    all_in_ids = [nid for k in in_scene for nid, _ in in_scene[k]]
    all_in_vecs = [unit(v) for v in vs.get_content_vectors(all_in_ids)]
    keep = [(nid, v) for nid, v in zip(all_in_ids, all_in_vecs) if v is not None]
    all_in_ids = [n for n, _ in keep]
    all_in_vecs = [v for _, v in keep]

    def sims(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        tv = [unit(vs._embed(t)) for t in texts]
        return np.array([[float(a @ b) for a in anchor_vecs] for b in tv])

    print(f"embedding={os.environ.get('EMBEDDING_MODEL')}  锚点={len(ANCHORS)}  "
          f"图内节点={len(all_in_ids)}\n")

    # A. 同场景（每个锚点用同场景节点）
    same = []
    for i, sc in enumerate("hcwdslm"):
        cand = [v for nid, v in zip(all_in_ids, all_in_vecs) if scene_of.get(nid) == sc]
        if cand:
            same.append(float(np.mean([anchor_vecs[i] @ v for v in cand])))
    # B. 跨场景
    cross = []
    for i, sc in enumerate("hcwdslm"):
        cand = [v for nid, v in zip(all_in_ids, all_in_vecs) if scene_of.get(nid) != sc]
        if cand:
            cross.append(float(np.mean([anchor_vecs[i] @ v for v in cand])))

    print("=== 图内（锚点 × 同/跨场景节点，逐锚点均值后再平均）===")
    print(f"  同场景  mean={np.mean(same):.3f}  range={min(same):.3f}~{max(same):.3f}")
    print(f"  跨场景  mean={np.mean(cross):.3f}  range={min(cross):.3f}~{max(cross):.3f}")

    for label, texts in (("无关·中文", UNRELATED_ZH),
                         ("无关·英文", UNRELATED_EN),
                         ("无关·代码", UNRELATED_CODE)):
        m = sims(texts)
        if m.size:
            per = m.mean(axis=1)
            print(f"  {label}  总体 mean={m.mean():.3f}  "
                  f"逐条={[round(float(x),3) for x in per]}  "
                  f"最大={m.max():.3f}")

    # 关键判定：外部无关文本的**最高**余弦 vs 阈值 0.30
    ext_max = max(
        float(sims(UNRELATED_ZH).max()) if UNRELATED_ZH else 0.0,
        float(sims(UNRELATED_EN).max()) if UNRELATED_EN else 0.0,
        float(sims(UNRELATED_CODE).max()) if UNRELATED_CODE else 0.0,
    )
    print(f"\n=== 判定 ===")
    print(f"  外部无关文本的最大余弦 = {ext_max:.3f}")
    print(f"  τ=0.30 能挡住它吗？ {'能' if ext_max < 0.30 else '不能（阈值低于无关基线，形同虚设）'}")
    print(f"  若要让 τ 真正只留'相关'，至少需高于 {ext_max:.2f}；"
          f"结合同场景均值 {np.mean(same):.2f}，可用区间约 "
          f"[{ext_max:.2f}, {np.mean(same):.2f}]")


if __name__ == "__main__":
    main()
