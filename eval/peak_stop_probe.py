# -*- coding: utf-8 -*-
"""验证：**首峰即停 + 多种子互补** 是否是比"走完整条链"更好的取舍。

背景（设计意图，见 core/peak_find.PeakFinder 的 first_peak 说明）：
在实际图谱里，近义节点会被 LLM 的边连接拉成团，因果链之间既会交叉也会闭环。
若允许扩散沿单链走到底，代价是**大量重复节点**；而覆盖面的缺口可以由
**其他种子的第一个峰**补齐。因此"首峰即停"是取舍，不是缺陷。

本脚本用受控图直接比较四种组合：
  peak ∈ {patience2（现行，回升会重置）, first_peak（首个显著下降即停）}
  × seed_k ∈ {1, 5}

图 C 结构（模拟真实条件）：
  入口 s1 / s2（都与 query 相关）
  链 A：s1 → a1 … → a6      链 B：s2 → b1 … → b6
  交叉：a5 → b6、b5 → a6（因果交织）
  噪声：s1 → x1 … → x3（无关分支）

指标：输出节点数、gold recall（a/b 两组）、无关混入率、每跳 μ 与输出规模。

用法：python -m eval.peak_stop_probe
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

QUERY = "最近项目太赶了，天天加班到很晚，感觉整个人都被掏空了"

GRAPH = {
    "nodes": [
        {"id": "s1", "content": "用户这周每天加班到十一点，回家倒头就睡", "type": "emotion"},
        {"id": "s2", "content": "用户最近压力特别大，情绪一直很低落", "type": "emotion"},
        # 链 A
        {"id": "a1", "content": "用户的项目本周必须上线", "type": "thing"},
        {"id": "a2", "content": "上线当晚出现了P0级严重故障", "type": "thing"},
        {"id": "a3", "content": "用户通宵定位并修复了故障", "type": "action"},
        {"id": "a4", "content": "第二天用户参加了项目复盘会", "type": "action"},
        {"id": "a5", "content": "复盘指出流程缺少灰度发布", "type": "reason"},
        {"id": "a6", "content": "团队下个迭代开始执行新流程", "type": "thing"},
        # 链 B
        {"id": "b1", "content": "用户连续加班后开始失眠", "type": "status"},
        {"id": "b2", "content": "用户想换一个有双休的团队", "type": "status"},
        {"id": "b3", "content": "用户投了几份简历出去", "type": "action"},
        {"id": "b4", "content": "用户接到两家公司的面试邀约", "type": "thing"},
        {"id": "b5", "content": "用户面试时又提到长期加班的疲惫", "type": "emotion"},
        {"id": "b6", "content": "用户在权衡要不要接受新的offer", "type": "action"},
        # 噪声链
        {"id": "x1", "content": "用户高中时参加过数学竞赛", "type": "status"},
        {"id": "x2", "content": "用户养过一只橘猫", "type": "thing"},
        {"id": "x3", "content": "用户的家在城西", "type": "status"},
    ],
    "edges": (
        [{"from": f"a{i}", "to": f"a{i+1}", "type": "causal"} for i in range(1, 6)]
        + [{"from": "s1", "to": "a1", "type": "causal"}]
        + [{"from": "s2", "to": "b1", "type": "causal"}]
        + [{"from": f"b{i}", "to": f"b{i+1}", "type": "causal"} for i in range(1, 6)]
        + [{"from": "a5", "to": "b6", "type": "causal"},
           {"from": "b5", "to": "a6", "type": "causal"}]
        + [{"from": "s1", "to": "x1", "type": "causal"},
           {"from": "x1", "to": "x2", "type": "causal"},
           {"from": "x2", "to": "x3", "type": "causal"}]
    ),
}

GOLD = {f"a{i}" for i in range(1, 7)} | {f"b{i}" for i in range(1, 7)}
OFFTOPIC = {"x1", "x2", "x3"}


def _run(retriever, seed_k: int, first_peak: bool, max_hops: int = 8) -> Dict[str, Any]:
    retriever.peak_finder.first_peak = first_peak
    retriever.purpose_filter_threshold = 0.30
    retriever.purpose_filter_sigma = 0.0
    retriever.peak_finder.peak_tolerance = 0.10
    out = retriever.retrieve_with_story(
        QUERY, seed_k=seed_k, max_hops=max_hops, story=False, with_response=False,
        purpose=["倾诉工作压力", "表达疲惫"])
    ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
    hops = [{"hop": h.get("hop"), "n": len(h.get("candidates") or []),
             "mu": round(float(h.get("mean_score") or 0), 4)}
            for h in (out.get("hop_history") or [])]
    return {
        "ids": ids,
        "n": len(ids),
        "recall": round(len(set(ids) & GOLD) / len(GOLD), 3),
        "off_ratio": round(len(set(ids) & OFFTOPIC) / max(len(ids), 1), 3),
        "hops": hops,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="首峰即停 vs 走到底：多种子是否互补")
    ap.add_argument("--emb-model", default="BAAI/bge-large-zh-v1.5")
    ap.add_argument("--out", default="eval/outputs/mechanism/peak_stop.json")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    os.environ["EMBEDDING_MODEL"] = args.emb_model
    p = ROOT / "eval/outputs/mechanism/lateral_chain.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(GRAPH, allow_unicode=True, sort_keys=False), encoding="utf-8")

    _, r = build_retriever(str(p), vector_cache_key="mech_chain")
    print(f"[peak-stop] emb={args.emb_model} gold={len(GOLD)} 噪声={len(OFFTOPIC)}", flush=True)

    report: Dict[str, Any] = {"meta": {"embedding": args.emb_model}}
    for seed_k in (1, 5):
        for fp in (False, True):
            label = f"seed{seed_k}_{'first_peak' if fp else 'patience2'}"
            res = _run(r, seed_k, fp)
            report[label] = res
            print(f"  {label:20s} 输出={res['n']:2d}  recall={res['recall']:.3f}  "
                  f"无关占比={res['off_ratio']:.1%}  μ={[h['mu'] for h in res['hops']]}")
            print(f"  {'':20s} ids={res['ids']}")

    print("\n=== 判读 ===")
    a, b = report["seed1_patience2"], report["seed1_first_peak"]
    c, d = report["seed5_patience2"], report["seed5_first_peak"]
    print(f"  单种子：首峰即停 输出 {b['n']} vs 走到底 {a['n']}；"
          f"recall {b['recall']:.3f} vs {a['recall']:.3f}")
    print(f"  多种子：首峰即停 输出 {d['n']} vs 走到底 {c['n']}；"
          f"recall {d['recall']:.3f} vs {c['recall']:.3f}")
    print("  若『多种子+首峰』的 recall 接近『单种子+走到底』而输出更少，则取舍成立。")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
