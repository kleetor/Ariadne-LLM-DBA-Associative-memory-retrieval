#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""边类型消融 —— 确认「双向边数量」是不是检索覆盖的敏感变量。

背景（Plan/0921 + Plan/0822 两份报告的对撞点）
--------------------------------------------
三臂（off / low / high）在同一份语料上建出的图有一个明显的形态差异：

    臂     有向边   单向边   双向边
    off     142      53       46
    low     243      68       94
    high    380      61      161      ← 双向边 ×3.5，单向边几乎不动

而**检索覆盖三臂不动**（gold 28 / 28 / 30，极差落在噪声内）。0921 报告的猜测是：
承载覆盖的是**单向边**（因果 / 时序 / 递进 —— 也就是真正产生"跨场景跨越"的那类），
双向边（SCENARIO 等「同场景并存」）增长再多也不改变覆盖。

本探针不动抽取、不重建图，**只在检索前按 ``edge.type`` 过滤边**，直接测这个猜测。

变体（每臂一组，跑同一批查询）
----------------------------
    base       原图（同进程内对照，必须重跑：三臂原先各用自己的 thinking 档位）
    only_dir   只留单向边（causal / sequence / preference / temporal / taxonomic）
    only_bidi  只留双向边（scenario / social / attribute）
    ctrl_rand  随机保留与 only_dir 同量的边 —— **区分"类型"与"量"**

为什么必须有 ctrl_rand：删边同时会改变"读取量"，若只跑 only_dir 就分不清掉分是
"少了单向边"还是"少了这么多边"。判读规则：

    only_dir 掉的幅度 ≈ ctrl_rand 掉的幅度  → 敏感变量是**量**，与类型无关
    only_dir 明显掉得更多                    → 敏感变量是**类型**，趋势成立

零抽取成本
----------
直接复制已建好的 ``data/`` 再改 YAML。节点原样保留 → 索引指纹（只覆盖节点 id）不变
→ 索引可安全复用；图规模不变，故 ``读取比`` 与之前几轮可直接对照。

用法
----
    python -m eval.probe_edge_type_ablation --arms off                    # 冒烟
    python -m eval.probe_edge_type_ablation --arms off,low,high           # 全量
    python -m eval.probe_edge_type_ablation --arms high --variants only_bidi,ctrl_rand
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:      # 环境里无 dotenv 时按已注入的环境变量走
    pass

import yaml  # noqa: E402

from alm.engine import ALMEngine, space_filename  # noqa: E402
from eval.exp_thinking_tiers import (  # noqa: E402
    DEFAULT_BUDGET,
    OUT_ROOT,
    USER_ID,
    _base_config,
    _capture_read_set,
    _load_corpus,
)
from eval.probe_llm_arm_ab import _take_within, _vector_topk  # noqa: E402
from eval.test_alm_vs_mcp_e2e import match_facts  # noqa: E402

# 与 EDGE_LINKING_PROMPT 的「方向」说明一致：
#   CAUSAL/SEQUENCE/PREFERENCE/TEMPORAL/TAXONOMIC 单向；SCENARIO/SOCIAL/ATTRIBUTE 双向
DIRECTIONAL = ("causal", "sequence", "preference", "temporal", "taxonomic")
BIDIRECTIONAL = ("scenario", "social", "attribute")
VARIANTS = ("base", "only_dir", "only_bidi", "ctrl_rand")


# ---------------------------------------------------------------- 图变体

def _rewrite_edges(data_dir: Path, variant: str, seed: int) -> Dict[str, Any]:
    """按变体过滤空间本体里的 ``edges``，节点原样保留。"""
    path = data_dir / space_filename(USER_ID)
    if not path.exists():
        raise SystemExit(f"缺少空间本体：{path}")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    edges: List[dict] = list(doc.get("edges") or [])

    dist: Dict[str, int] = {}
    for edge in edges:
        key = str(edge.get("type"))
        dist[key] = dist.get(key, 0) + 1
    n_dir = sum(1 for e in edges if str(e.get("type")) in DIRECTIONAL)

    if variant == "base":
        kept = edges
    elif variant == "only_dir":
        kept = [e for e in edges if str(e.get("type")) in DIRECTIONAL]
    elif variant == "only_bidi":
        kept = [e for e in edges if str(e.get("type")) in BIDIRECTIONAL]
    elif variant == "ctrl_rand":
        # 等量对照：随机抽与 only_dir 保留量相同的边（下面判读全靠它）
        kept = list(random.Random(seed).sample(edges, min(n_dir, len(edges))))
    else:
        raise SystemExit(f"未知变体：{variant}")

    doc["edges"] = kept
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return {"dist": dist, "edges_total": len(edges), "edges_kept": len(kept),
            "n_directional": n_dir}


def _prepare(corpus: str, arm: str, variant: str, seed: int,
             src_root: Path, out_root: Path):
    src = src_root / corpus / arm / "data"
    if not src.exists():
        raise SystemExit(f"缺少已建图产物：{src}（先跑 exp_thinking_tiers --stage build）")
    dst = out_root / corpus / f"{arm}__{variant}" / "data"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst, _rewrite_edges(dst, variant, seed)


# ---------------------------------------------------------------- 检索

def run_variant(corpus: str, arm: str, variant: str, queries: List[dict],
                expand_k: int, max_hops: int, seed: int,
                src_root: Path, out_root: Path) -> Dict[str, Any]:
    data_dir, meta = _prepare(corpus, arm, variant, seed, src_root, out_root)

    cfg = _base_config(expand_k, max_hops)
    cfg.data_dir = data_dir
    # 不调用 _apply_thinking_mode：全部变体共用当前默认（思考关闭），
    # 使本探针的唯一变量只有"边类型集合"。
    engine = ALMEngine(cfg)
    try:
        space = engine.get_space(USER_ID, create=False)
        node_count = space.graph.node_count
        edge_count = space.graph.edge_count              # 内存有向边
        vec_k = space._effective_seed_k()
        box = _capture_read_set(space)

        rows: List[Dict[str, Any]] = []
        for query in queries:
            shot = time.time()
            data = space.search(query["query"], top_k=100)
            elapsed = time.time() - shot
            text = "\n".join(item["content"] for item in data)
            gold = query.get("gold_facts") or []
            graph_hit = match_facts(text, gold)
            _, vec_texts = _vector_topk(space, query["query"], vec_k)
            vec_hit = match_facts("\n".join(vec_texts), gold)
            rows.append({
                "id": query["id"], "query": query["query"],
                "count": len(data), "chars": len(text),
                "gold_total": len(gold), "gold_hit": graph_hit,
                "vec_hit": vec_hit,
                "vec_budget_hit": match_facts(
                    _take_within(vec_texts, max(len(text), 1)), gold),
                "only_graph": sorted(set(graph_hit) - set(vec_hit)),
                "read_set": box["read_set"], "adopted": box["adopted"],
                "read_ratio": (round(box["read_set"] / node_count, 3)
                               if box["read_set"] and node_count else None),
                "elapsed": elapsed,
            })
    finally:
        engine.close()

    gold_hit = sum(len(r["gold_hit"]) for r in rows)
    gold_total = sum(r["gold_total"] for r in rows)
    vec_hit_total = sum(len(r["vec_hit"]) for r in rows)
    only_graph = sum(len(r["only_graph"]) for r in rows)
    ratios = [r["read_ratio"] for r in rows if r["read_ratio"] is not None]
    adopted = [r["adopted"] for r in rows if r["adopted"] is not None]
    summary = {
        "arm": arm, "variant": variant, "corpus": corpus,
        "node_count": node_count, "edge_count": edge_count,
        "edges_kept": meta["edges_kept"], "edges_total": meta["edges_total"],
        "avg_degree": round(2 * edge_count / node_count, 2) if node_count else None,
        "dist": meta["dist"],
        "gold_hit": gold_hit, "gold_total": gold_total,
        "vec_hit": vec_hit_total, "only_graph": only_graph,
        "read_ratio_median": (sorted(ratios)[len(ratios) // 2] if ratios else None),
        "read_ratio_max": (max(ratios) if ratios else None),
        "adopted_median": (sorted(adopted)[len(adopted) // 2] if adopted else None),
        "rows": rows,
    }
    print(f"      gold={gold_hit}/{gold_total} | 纯向量(top-{vec_k})={vec_hit_total} "
          f"| ★仅图={only_graph} | 读取比 中位 {summary['read_ratio_median']} "
          f"（max {summary['read_ratio_max']}）| adopted 中位 {summary['adopted_median']}")
    return summary


# ---------------------------------------------------------------- 入口

def main() -> int:
    parser = argparse.ArgumentParser(description="边类型消融（零抽取成本）")
    parser.add_argument("--corpus", default="groupchat_mixed")
    parser.add_argument("--arms", default="off", help="逗号分隔，如 off,low,high")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--expand-k", type=int, default=DEFAULT_BUDGET["expand_k"])
    parser.add_argument("--max-hops", type=int, default=DEFAULT_BUDGET["max_hops"])
    parser.add_argument("--seed", type=int, default=42, help="ctrl_rand 抽样种子")
    parser.add_argument("--out", default=None,
                        help="变体产物的输出根（源图始终读 harness 的 OUT_ROOT，不受此参数影响）")
    parser.add_argument("--query-limit", type=int, default=None, help="冒烟用：只跑前 N 条查询")
    args = parser.parse_args()

    out_root = Path(args.out).resolve() if args.out else OUT_ROOT
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"未知变体 {v}（可选 {VARIANTS}）")

    _, queries = _load_corpus(args.corpus)
    if args.query_limit:
        queries = queries[:args.query_limit]

    print(f"===== 边类型消融：语料 {args.corpus} | 臂 {arms} | 变体 {variants} "
          f"| 查询 {len(queries)} 条 | expand_k={args.expand_k} max_hops={args.max_hops} =====")

    summaries: List[Dict[str, Any]] = []
    for arm in arms:
        for variant in variants:
            print(f"\n----- [ablation] 臂 {arm} × 变体 {variant} -----")
            try:
                summaries.append(run_variant(
                    args.corpus, arm, variant, queries,
                    args.expand_k, args.max_hops, args.seed, OUT_ROOT, out_root))
            except SystemExit:
                raise
            except Exception as exc:                       # 单臂失败不拖垮整轮
                print(f"      !! 失败：{type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- 汇总表 ----
    print("\n===== 汇总（gold = 图路径命中；仅图 = 相对同种子数纯向量基线的净增量）=====")
    print(f"{'臂':<6}{'变体':<11}{'节点':>5}{'有向边':>7}{'平均度':>7}"
          f"{'gold':>8}{'纯向量':>8}{'仅图':>6}{'读取比中位':>11}{'adopted':>9}")
    for s in summaries:
        if "gold_hit" not in s:
            continue
        print(f"{s['arm']:<6}{s['variant']:<11}{s['node_count']:>5}"
              f"{s['edge_count']:>7}{s['avg_degree']:>7}"
              f"{str(s['gold_hit']) + '/' + str(s['gold_total']):>8}"
              f"{s['vec_hit']:>8}{s['only_graph']:>6}"
              f"{str(s['read_ratio_median']):>11}{str(s['adopted_median']):>9}")

    stamp = time.strftime("%m%d-%H%M%S")
    out_path = out_root / "edge_ablation" / f"{args.corpus}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n产物：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
