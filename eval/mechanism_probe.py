# -*- coding: utf-8 -*-
"""机制专项：目的过滤与寻峰各自防的是哪种"贪婪"，以及需要什么条件才生效。

机制解读（本轮修订）：
两者作用是**同向**的——都阻止跳转轴输出过多，但防的失控方向不同：
  · **峰值带**：防**纵向**失控——沿同一根因果链一路往下走（A→B→C→…），越走越远；
  · **目的过滤**：防**横向**失控——某节点有多条出边时，跳转轴向太多语义方向扩散。
因此目的过滤的阈值天然应当**宽松**（它只需挡住"明显不在 query 邻域"的分支）。

为什么此前测不出（三个必要条件都没满足）：
  1. embedding 用 bge-m3 时无关基线高达 0.40–0.55，任何合理 τ 都挡不住东西；
  2. 原图 163 节点平均出度低，"一个节点十条因果边"的结构并不存在；
  3. 种子固定为 5，横向贪婪被多个种子摊薄，看不出单节点的分叉行为。

本脚本用**受控合成图**直接检验，两个条件都构造出来：
  · 图 A（纵向深链）：s → n1 → … → n8，全部相关，测峰值带能否及时停下；
  · 图 B（横向多叉）：s 有 10 条因果出边，其中 3 条相关、7 条无关，测过滤能否只留相关。
两图都用 **seed_k = 1**，embedding 默认 `BAAI/bge-large-zh-v1.5`（无关基线 0.269，阈值才有意义）。

用法：
    python -m eval.mechanism_probe
    python -m eval.mechanism_probe --emb-model BAAI/bge-m3   # 对照：高基线模型
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

OUT_DIR = ROOT / "eval/outputs/mechanism"
QUERY = "最近项目太赶了，天天加班到很晚，感觉整个人都被掏空了"

# 图 A：单链因果（纵向）。都相关，但每深一层语义就更远一点。
LONGITUDINAL = {
    "nodes": [
        {"id": "s", "content": "用户这周每天加班到十一点，回家倒头就睡", "type": "emotion"},
        {"id": "n1", "content": "用户的这个项目本周必须上线", "type": "thing"},
        {"id": "n2", "content": "上线当晚出现了P0级严重故障", "type": "thing"},
        {"id": "n3", "content": "用户通宵定位并修复了这个故障", "type": "action"},
        {"id": "n4", "content": "第二天用户参加了项目复盘会", "type": "action"},
        {"id": "n5", "content": "复盘会上指出流程缺少灰度发布环节", "type": "reason"},
        {"id": "n6", "content": "用户开始整理一份发布检查清单", "type": "action"},
        {"id": "n7", "content": "用户把清单同步给了组内同事", "type": "action"},
        {"id": "n8", "content": "团队决定下个迭代开始执行新的发布流程", "type": "thing"},
    ],
    "edges": [{"from": f"n{i}" if i else "s", "to": f"n{i+1}", "type": "causal"}
              for i in range(8)],
}

# 图 B：中心节点十条因果出边（横向）。三条与"工作压力"相关，七条是别的主题。
LATERAL = {
    "nodes": [
        {"id": "s", "content": "用户这周每天加班到十一点，非常疲惫", "type": "emotion"},
        # 3 条相关
        {"id": "r1", "content": "用户的加班是因为需求临时增加", "type": "reason"},
        {"id": "r2", "content": "用户想换一个有双休的团队", "type": "status"},
        {"id": "r3", "content": "用户连续加班后开始失眠", "type": "status"},
        # 7 条无关（其他人生阶段/主题）
        {"id": "x1", "content": "用户高中时参加过数学竞赛拿了校三等奖", "type": "status"},
        {"id": "x2", "content": "用户养过一只橘猫", "type": "thing"},
        {"id": "x3", "content": "用户这周开始吃减脂餐", "type": "action"},
        {"id": "x4", "content": "用户大二下学期偶然接触了编程", "type": "action"},
        {"id": "x5", "content": "用户周六和朋友约了聚餐", "type": "action"},
        {"id": "x6", "content": "用户上周报了数据分析网课", "type": "action"},
        {"id": "x7", "content": "用户的家在城西", "type": "status"},
    ],
    "edges": [{"from": "s", "to": x, "type": "causal"}
              for x in ("r1", "r2", "r3", "x1", "x2", "x3", "x4", "x5", "x6", "x7")],
}


def _dump(name: str, g: Dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.yaml"
    p.write_text(yaml.safe_dump(g, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _run(retriever, text: str, purpose, max_hops: int, tau: float,
         peak_tol: float) -> Dict[str, Any]:
    retriever.purpose_filter_threshold = tau
    retriever.purpose_filter_sigma = 0.0
    retriever.peak_finder.peak_tolerance = peak_tol
    out = retriever.retrieve_with_story(
        text, seed_k=1, max_hops=max_hops, story=False, with_response=False,
        purpose=list(purpose))
    ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
    hops = [{"hop": h.get("hop"), "n": len(h.get("candidates") or []),
             "mu": round(float(h.get("mean_score") or 0), 4)}
            for h in (out.get("hop_history") or [])]
    return {"ids": ids, "n": len(ids), "hops": hops,
            "filter_stats": getattr(retriever, "_purpose_stats", []) or []}


def main() -> int:
    ap = argparse.ArgumentParser(description="机制专项：纵向峰值带 / 横向目的过滤")
    ap.add_argument("--emb-model", default="BAAI/bge-large-zh-v1.5")
    ap.add_argument("--cache-prefix", default="mech_zh")
    ap.add_argument("--out", default="eval/outputs/mechanism/report.json")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    os.environ["EMBEDDING_MODEL"] = args.emb_model

    pA = _dump("longitudinal", LONGITUDINAL)
    pB = _dump("lateral", LATERAL)

    report: Dict[str, Any] = {"meta": {"embedding": args.emb_model, "query": QUERY,
                                       "seed_k": 1}}
    print(f"[mech] emb={args.emb_model}  query={QUERY}", flush=True)

    # ---------- 图 A：纵向，比较"有峰值带 / 无峰值带" ----------
    _, rA = build_retriever(str(pA), vector_cache_key=f"{args.cache_prefix}_A")
    print("\n=== 图 A 纵向深链（s → n1 … → n8）===")
    report["longitudinal"] = {}
    for label, tol in (("peak_band_on", 0.10), ("peak_band_off", 10.0)):
        r = _run(rA, QUERY, ["倾诉工作压力", "表达疲惫"], max_hops=8, tau=0.30, peak_tol=tol)
        report["longitudinal"][label] = r
        print(f"  {label:14s} 输出={r['n']} 条  ids={r['ids']}")
        print(f"                  每跳 μ={[h['mu'] for h in r['hops']]}  "
              f"候选数={[h['n'] for h in r['hops']]}")

    # ---------- 图 B：横向，扫 τ ----------
    _, rB = build_retriever(str(pB), vector_cache_key=f"{args.cache_prefix}_B")
    print("\n=== 图 B 横向多叉（s 有 10 条因果出边：3 相关 + 7 无关）===")
    report["lateral"] = {}
    rel = {"r1", "r2", "r3"}
    for tau in (0.0, 0.25, 0.30, 0.35, 0.40, 0.50):
        r = _run(rB, QUERY, ["倾诉工作压力", "表达疲惫"], max_hops=2, tau=tau, peak_tol=0.10)
        got = set(r["ids"])
        keep_rel = len(got & rel)
        leak = len(got - rel - {"s"})
        report["lateral"][f"{tau:.2f}"] = {**r, "keep_relevant": keep_rel, "leak_irrelevant": leak}
        print(f"  τ={tau:.2f}  输出={r['n']:2d}  保留相关={keep_rel}/3  混入无关={leak}/7  "
              f"通过率={[round(s['out']/s['in'], 3) for s in r['filter_stats'] if s['hop']>=1]}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[mech] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
