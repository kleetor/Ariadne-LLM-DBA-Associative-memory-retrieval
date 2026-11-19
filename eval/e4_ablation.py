# -*- coding: utf-8 -*-
"""E4：PAR 机制消融（同图、同组装器、同 judge，只关一个机制）。

设计要点（为什么这样切）：
- **固定组装器为 par_story**：E3 已确定「PAR 选择 + StoryRank 叙事」是当前最优组合；
  消融要回答的是"在这个最优配置里，每个机制各贡献多少"，因此不让 raw/story 再变一次。
- **分两批判分，而不是五臂并排**：
  * 批次 `purpose`   = full / no_purpose / random_purpose
  * 批次 `structure` = full / no_jump_axis / no_peak_band
  理由：与 E3 的三臂判分协议（同题三块、匿名 A–C、逐题打乱）保持一致，单题上下文长度
  不变；`full` 作为两批共用的锚点，还能顺带估计"同一条块重复判分"的噪声量级。
- **所有条件等规模约束**：`no_peak_band` 关掉峰值筛选后候选会变多，若放任会靠"篇幅"
  白拿丰富度分。故它按 full 的节点数截断（`max_peak_nodes`），与 E3 里 vector 的
  `match_par` 同口径。

五个条件的实现方式（都走真实代码路径，不做"假关"）：

| 条件 | 做法 | 关掉的是什么 |
|---|---|---|
| `full` | 默认配置 | — |
| `no_purpose` | `purpose_blind=True`（不推断/不做目的种子/不过滤/组合分不含目的/寻峰改用跳转权重） | 目的驱动（四处同时） |
| `random_purpose` | 注入 3 条**随机节点正文**当目的 | 目的的"相关性"（流程保留） |
| `no_jump_axis` | 把 `get_jump_weight` 临时替换为恒 1.0（方向无关、类型无关） | 跳转轴的类型×方向权重 |
| `no_peak_band` | `peak_tolerance=10`（带宽大于任何 μ 差 → 全轮进带）+ `max_hops=3` | 寻峰/容忍带（改为固定 3 跳） |

用法：
    python -m eval.e4_ablation --batch purpose --limit 12
    python -m eval.e4_ablation --batch purpose
    python -m eval.e4_ablation --batch structure
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

import dba_pipeline.graph.memory_graph as mg  # noqa: E402
import dba_pipeline.retrieval.retriever as rt  # noqa: E402
from dba_pipeline.core.jump_axis import get_jump_weight as _real_jump_weight  # noqa: E402
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever  # noqa: E402
from eval.collect_ariadne_raw import ariadne_raw_block  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

BATCHES = {
    "purpose": ("full", "no_purpose", "random_purpose"),
    "structure": ("full", "no_jump_axis", "no_peak_band"),
}
ALL_CONDS = tuple(dict.fromkeys(c for v in BATCHES.values() for c in v))
_LABEL_POOL = ["系统A", "系统B", "系统C", "系统D"]

# no_peak_band：带宽取 10.0，远大于任何轮均分之差（实测轮间极差 ≤0.15），
# 于是"每轮都进带"= 不做峰值筛选。跳数上限**沿用与 full 相同的默认值**——
# 关掉的是"提前停止判据"，不是"跳得多远"；若把上限也改成 3，规模缩水会混进机制效应。
_NO_BAND_TOLERANCE = 10.0


def _label_map(qid: str, seed: int, conds: List[str]) -> Dict[str, str]:
    labels = _LABEL_POOL[: len(conds)]
    shuffled = list(conds)
    random.Random(f"{seed}:{qid}").shuffle(shuffled)
    return dict(zip(shuffled, labels))


def _uniform_jump_weight(*_a, **_k) -> float:
    """跳转轴消融：所有类型/方向一律等权 1.0。"""
    return 1.0


class _PatchedJumpAxis:
    """临时把跳转轴权重替换为恒 1.0（graph 侧与 retriever 侧都要换，二者都会调它）。"""

    def __enter__(self):
        self._mg, self._rt = mg.get_jump_weight, rt.get_jump_weight
        mg.get_jump_weight = _uniform_jump_weight
        rt.get_jump_weight = _uniform_jump_weight
        return self

    def __exit__(self, *_exc):
        mg.get_jump_weight = self._mg
        rt.get_jump_weight = self._rt
        return False


class _PatchedBand:
    """临时放宽峰值容忍带（等价于关掉寻峰筛选）。"""

    def __init__(self, retriever: PurposeDrivenRetriever, tolerance: float):
        self.r = retriever
        self.tol = tolerance

    def __enter__(self):
        self._old = self.r.peak_finder.peak_tolerance
        self._old_ratio = self.r.peak_tolerance_ratio
        self.r.peak_finder.peak_tolerance = self.tol
        self.r.peak_tolerance_ratio = 0.0
        return self

    def __exit__(self, *_exc):
        self.r.peak_finder.peak_tolerance = self._old
        self.r.peak_tolerance_ratio = self._old_ratio
        return False


def _random_purposes(graph, qid: str, k: int = 3) -> List[str]:
    """从图里随机抽 k 条节点正文当"目的"——有意义但**与当前 query 无关**。

    为什么不用随机字符串：随机串的嵌入会落在语义空间外，cos 分布整体塌陷，
    那测的是"嵌入退化"而不是"目的失配"。抽真实节点正文可保证目的向量本身正常，
    只是与 query 不相关，才是干净的 random-purpose 对照。
    """
    ids = sorted(graph.graph.nodes())
    rng = random.Random(f"rp:{qid}")
    picked = rng.sample(ids, min(k, len(ids)))
    return [str(graph.get_content(i) or "") for i in picked if graph.get_content(i)]


def _run_condition(
    cond: str,
    retriever: PurposeDrivenRetriever,
    graph,
    q: Dict[str, Any],
    args,
    full_nodes: Optional[int],
) -> Dict[str, Any]:
    """按条件跑一次检索，返回 {path, hop_history, purpose, note}。"""
    qid, qtext = str(q["id"]), str(q["text"])
    kw: Dict[str, Any] = {"story": False, "with_response": False}
    if args.max_hops is not None:
        kw["max_hops"] = args.max_hops
    # 默认注入查询集里标注的 purposes：目的推断是一次 LLM 调用，未固定 temperature 时
    # **同一 query 两次跑会给出不同目的**（冒烟实测 q02 的 full 两次分别 36/29 节点），
    # 消融里这层随机性会把机制效应淹没。注入后检索变成确定性的（向量全部命中缓存）。
    injected = q.get("purposes") if args.purpose_mode == "inject" else None
    if injected:
        kw["purpose"] = list(injected)

    note = ""
    ctx = None
    if cond == "full":
        pass
    elif cond == "no_purpose":
        kw.pop("purpose", None)
        kw["purpose_blind"] = True
    elif cond == "random_purpose":
        kw["purpose"] = _random_purposes(graph, qid, args.random_purpose_k)
        note = "random purposes from graph nodes"
    elif cond == "no_jump_axis":
        ctx = _PatchedJumpAxis()
    elif cond == "no_peak_band":
        ctx = _PatchedBand(retriever, _NO_BAND_TOLERANCE)
        # 等规模：关掉峰值筛选后候选变多，按 full 的条数截断（与 E3 的 match_par 同口径）
        if full_nodes:
            kw["max_peak_nodes"] = int(full_nodes)
        note = f"band disabled (tol={_NO_BAND_TOLERANCE}), capped to full n={full_nodes}"
    else:
        raise ValueError(f"未知条件 {cond}")

    if ctx is None:
        out = retriever.retrieve_with_story(qtext, **kw)
    else:
        with ctx:
            out = retriever.retrieve_with_story(qtext, **kw)
    return {
        "path": out.get("path") or {"nodes": [], "edges": []},
        "hop_history": out.get("hop_history") or [],
        "purpose": out.get("purpose") or {},
        "note": note,
    }


def _story(retriever: PurposeDrivenRetriever, query: str, path: Dict[str, Any]) -> Tuple[str, bool]:
    out = retriever.inference.story_rank(query, path)
    return str(out.get("story") or ""), bool(out.get("degraded"))


def _load_queries(path: Path, exclude: List[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    excluded = {c.strip() for c in exclude if c and c.strip()}
    return [q for q in data.get("queries") or []
            if str(q.get("category") or "") not in excluded]


def _build_row(retriever, graph, q: Dict[str, Any], args, conds: List[str]) -> Dict[str, Any]:
    qid, qtext = str(q["id"]), str(q["text"])

    # 先跑 full：既作为锚点，也给 no_peak_band 提供等规模上限
    full = _run_condition("full", retriever, graph, q, args, full_nodes=None)
    full_nodes = len(full["path"].get("nodes") or [])

    runs = {"full": full}
    for c in conds:
        if c in runs:
            continue
        runs[c] = _run_condition(c, retriever, graph, q, args, full_nodes=full_nodes)

    blocks: Dict[str, Dict[str, Any]] = {}
    stats: Dict[str, Any] = {"full_nodes": full_nodes}
    for c in conds:
        path = runs[c]["path"]
        story, degraded = _story(retriever, qtext, path)
        blocks[c] = {"story": story, "n_nodes": len(path.get("nodes") or [])}
        stats[c] = {
            "nodes": len(path.get("nodes") or []),
            "edges": len(path.get("edges") or []),
            "hops": len(runs[c]["hop_history"]),
            "story_chars": len(story),
            "degraded": degraded,
            "purposes": str((runs[c]["purpose"] or {}).get("purposes")),
            "note": runs[c]["note"],
        }

    lm = _label_map(qid, args.anon_seed, conds)
    labels = _LABEL_POOL[: len(conds)]
    return {
        "query_id": qid,
        "category": q.get("category", ""),
        "query": qtext,
        "systems": {label: blocks[cond] for label in labels
                    for cond in conds if lm[cond] == label},
        "label_map": lm,
        "stats": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 PAR 机制消融（两批三臂）")
    ap.add_argument("--batch", choices=sorted(BATCHES), required=True,
                    help="purpose=目的相关消融；structure=结构/终止消融")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--queries", default="data/dba_eval/comprehensive_queries.yaml")
    ap.add_argument("--out", default=None, help="默认 eval/outputs/e4/context_e4_<batch>.json")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--max-hops", type=int, default=None, help="覆盖默认最大跳数（no_peak_band 固定 3）")
    ap.add_argument("--random-purpose-k", type=int, default=3)
    ap.add_argument("--purpose-mode", choices=["inject", "infer"], default="inject",
                    help="inject=用查询集标注的 purposes（默认，检索确定性）；"
                         "infer=走内部目的推断（会引入 LLM 随机性，仅作端到端口径对照）")
    ap.add_argument("--anon-seed", type=int, default=20261117)
    ap.add_argument("--exclude-categories", default="abstention")
    ap.add_argument("--vector-cache-key", default="comprehensive")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    conds = list(BATCHES[args.batch])
    out_path = args.out or f"eval/outputs/e4/context_e4_{args.batch}.json"

    _load_dotenv(ROOT / ".env")
    queries = _load_queries(ROOT / args.queries, args.exclude_categories.split(","))
    if args.limit:
        queries = queries[: args.limit]
    graph, retriever = build_retriever(
        str(ROOT / args.yaml), llm_model=args.llm_model,
        vector_cache_key=args.vector_cache_key,
    )
    print(f"[E4] batch={args.batch} conds={conds} nodes={graph.node_count} "
          f"queries={len(queries)} emb={os.environ.get('EMBEDDING_MODEL')}", flush=True)

    rows, failed = [], []
    for q in queries:
        try:
            row = _build_row(retriever, graph, q, args, conds)
        except Exception as exc:
            failed.append({"query_id": str(q.get("id")),
                           "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {q.get('id')} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(row)
        st = row["stats"]
        print(f"  {row['query_id']} [{row['category']}] "
              + "  ".join(f"{c}={st[c]['nodes']}n/{st[c]['story_chars']}字"
                          for c in conds), flush=True)

    payload = {
        "meta": {
            "experiment": f"E4/{args.batch}",
            "conditions": conds,
            "yaml": args.yaml,
            "queries": args.queries,
            "seed_k": args.seed_k,
            "max_hops": args.max_hops,
            "no_band_tolerance": _NO_BAND_TOLERANCE,
            "random_purpose_k": args.random_purpose_k,
            "purpose_mode": args.purpose_mode,
            "anon_seed": args.anon_seed,
            "excluded_categories": args.exclude_categories,
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "purpose_filter_threshold": retriever.purpose_filter_threshold,
            "peak_tolerance": retriever.peak_finder.peak_tolerance,
            "note": ("组装器固定为 StoryRank 叙事（E3 最优组合）；只关一个机制。"
                     "判分匿名 系统A–C 且逐题打乱，label_map 可还原。"
                     "no_peak_band 按 full 的节点数等规模截断。"),
            "judge": "python -m eval.judge_comprehensive --ctx <this file>",
            "failed": failed,
        },
        "queries": rows,
    }
    p = ROOT / out_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E4] wrote {p}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
