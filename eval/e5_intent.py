# -*- coding: utf-8 -*-
"""E5：意图变化实验（同一句歧义话 × 三种目的，PAR vs 向量）。

**核心前提**（v2 修正）：变的是**目的**，不是文本。
数据集的每条 scenario 只有一个 `text`（字面不指定意图），配三组 purposes。
这样两臂才可比：
- 纯向量（仅文本）对目的一无所知 → 同句三次结果必然相同；
- PAR 有条件让同一句话在不同目的下被理解成不同的事。

三臂（主表只放前两个）：
| 臂 | 输入 | 作用 |
|---|---|---|
| `par` | 文本 + 目的（目的种子/过滤/加权） | 主表 |
| `vector_purpose` | 文本与 purposes **拼接后嵌入** | 主表（公平对照：向量也能"看到"目的） |
| `vector_text` | 仅文本 | 内部校验/下界（同句 Jaccard 恒=1，不进主表） |

三项指标：
1. **同情景意图间 Jaccard**（越低=越随意图分化）；
2. **gold facet recall**：每个意图下应被召回的节点组（数据集的 `gold`）被覆盖的比例——
   回答"分化是否发生在正确的方向上"，比 Jaccard 更有说服力；
3. **judge 四维**（`eval/judge_e5_intent.py`，含意图贴合度）。

组装器固定 StoryRank；匿名 A–B 且逐题打乱；目的注入（E4 已实测内部推断不确定）。

用法：
    python -m eval.e5_intent --limit 3
    python -m eval.e5_intent
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever  # noqa: E402
from eval.e3_par_vs_vector import _cosine, _path_from_ids, _story  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

ARMS = ("par", "vector_purpose", "vector_text")
MAIN_ARMS = ("par", "vector_purpose")   # 进判分与主表
_LABEL_POOL = ["系统A", "系统B"]


def _label_map(qid: str, seed: int) -> Dict[str, str]:
    shuffled = list(MAIN_ARMS)
    random.Random(f"{seed}:{qid}").shuffle(shuffled)
    return dict(zip(shuffled, _LABEL_POOL))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _vec_topk(retriever: PurposeDrivenRetriever, text: str, k: int):
    """按给定文本做纯向量 top-k（不做图扩展、不带边）。"""
    vs, graph = retriever.vector_store, retriever.graph
    qv = vs._embed(text)
    ids = [n for n in graph.graph.nodes() if graph.graph.nodes[n].get("content")]
    vecs = vs.get_content_vectors(ids)
    scored = [(n, _cosine(qv, v)) for n, v in zip(ids, vecs) if v is not None]
    scored.sort(key=lambda p: -p[1])
    return scored[:k]


def _flatten(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 scenario × intents 展平成条目；**同一 scenario 的 text 逐字相同**。"""
    out = []
    for sc in cfg.get("scenarios") or []:
        prefix = str(sc["scenario"]).split("_")[0]
        for it in sc.get("intents") or []:
            out.append({
                "id": f"{prefix}_{it['intent']}",
                "scenario": sc["scenario"],
                "text": sc["text"],
                "intent": it["intent"],
                "purposes": list(it.get("purposes") or []),
                "gold": list(it.get("gold") or []),
            })
    return out


def _build_row(retriever, q: Dict[str, Any], args) -> Dict[str, Any]:
    qid, text = q["id"], q["text"]
    purpose = q["purposes"] if args.purpose_mode == "inject" else None

    # ---- PAR：文本 + 注入目的 ----
    par = retriever.retrieve_with_story(
        text, seed_k=args.seed_k, story=False, with_response=False,
        purpose=list(purpose) if purpose else None)
    par_path = par.get("path") or {"nodes": [], "edges": []}
    par_ids = [n["id"] for n in (par_path.get("nodes") or [])]

    k = max(len(par_ids), 1)

    # ---- vector_purpose：文本与目的拼接后嵌入（公平对照）----
    joined = text + " " + " ".join(purpose or [])
    vp_path = _path_from_ids(retriever, _vec_topk(retriever, joined, k), with_edges=False)
    vp_ids = [n["id"] for n in (vp_path.get("nodes") or [])]

    # ---- vector_text：仅文本（对目的不可见；内部校验，不进判分）----
    vt_path = _path_from_ids(retriever, _vec_topk(retriever, text, k), with_edges=False)
    vt_ids = [n["id"] for n in (vt_path.get("nodes") or [])]

    stories = {}
    degraded = {}
    for arm, path in (("par", par_path), ("vector_purpose", vp_path)):
        s, d = _story(retriever, text, path)
        stories[arm] = s
        degraded[arm] = d

    gold = set(q["gold"])
    def recall(ids):
        return round(len(set(ids) & gold) / len(gold), 4) if gold else None

    blocks = {arm: {"story": stories[arm], "n_nodes": len(ids)}
              for arm, ids in (("par", par_ids), ("vector_purpose", vp_ids))}
    lm = _label_map(qid, args.anon_seed)
    return {
        "query_id": qid,
        "scenario": q["scenario"],
        "intent": q["intent"],
        "category": q["intent"],
        "query": text,
        "systems": {label: blocks[arm] for label in _LABEL_POOL for arm in MAIN_ARMS
                    if lm[arm] == label},
        "label_map": lm,
        "stats": {
            "par_ids": par_ids,
            "vp_ids": vp_ids,
            "vt_ids": vt_ids,
            "par_nodes": len(par_ids),
            "vp_nodes": len(vp_ids),
            "vt_nodes": len(vt_ids),
            "gold_recall": {"par": recall(par_ids), "vector_purpose": recall(vp_ids),
                            "vector_text": recall(vt_ids)},
            "par_story_chars": len(stories["par"]),
            "vp_story_chars": len(stories["vector_purpose"]),
            "degraded": degraded,
            "purposes": str(par.get("purpose", {}).get("purposes")),
        },
    }


def _group_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按 scenario 聚合：同情景三种意图两两 Jaccard + gold recall 均值。"""
    groups: Dict[str, Dict[str, Dict[str, set]]] = {}
    vt_lists: Dict[str, Dict[str, List[str]]] = {}
    recall: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        g = groups.setdefault(r["scenario"], {})
        for arm, key in (("par", "par_ids"), ("vector_purpose", "vp_ids"),
                         ("vector_text", "vt_ids")):
            g.setdefault(arm, {})[r["intent"]] = set(r["stats"][key])
        vt_lists.setdefault(r["scenario"], {})[r["intent"]] = list(r["stats"]["vt_ids"])
        for arm in ARMS:
            v = r["stats"]["gold_recall"].get(arm)
            if v is not None:
                recall.setdefault(arm, []).append(v)

    def _is_prefix(short: List[str], long_: List[str]) -> bool:
        n = min(len(short), len(long_))
        return short[:n] == long_[:n]

    out: Dict[str, Any] = {}
    for sc, g in groups.items():
        intents = sorted(g.get("par", {}))
        if len(intents) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(intents) for b in intents[i + 1:]]
        row = {"pairs": {}}
        for arm in ARMS:
            vals = [_jaccard(g[arm][a], g[arm][b]) for a, b in pairs]
            row[arm] = {
                "mean": round(float(np.mean(vals)), 4) if vals else None,
                "per_pair": {f"{a}|{b}": round(v, 4) for (a, b), v in zip(pairs, vals)},
            }
        # vector_text 的内部校验：文本相同 → 三次的排序**应完全相同**，差异只来自
        # 等规模截断（较短的那份应是较长者的前缀）。若这里为 False，说明图/向量有非确定性。
        row["vt_ranking_prefix_identical"] = all(
            _is_prefix(vt_lists[sc][a], vt_lists[sc][b])
            for a, b in pairs
        )
        out[sc] = row
    summary = {}
    for arm in ARMS:
        means = [v[arm]["mean"] for v in out.values() if v[arm]["mean"] is not None]
        summary[arm] = {
            "jaccard_mean": round(float(np.mean(means)), 4) if means else None,
            "recall_mean": round(float(np.mean(recall[arm])), 4) if recall.get(arm) else None,
        }
    summary["vector_text"]["ranking_prefix_identical_all"] = all(
        v.get("vt_ranking_prefix_identical") for v in out.values()) if out else None
    return {"per_scenario": out, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 意图变化（同句歧义 × 3 目的）")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--queries", default="data/dba_eval/intent_queries.yaml")
    ap.add_argument("--out", default="eval/outputs/e5/context_e5_intent.json")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部 30）")
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--purpose-mode", choices=["inject", "infer"], default="inject")
    ap.add_argument("--anon-seed", type=int, default=20261117)
    ap.add_argument("--vector-cache-key", default="comprehensive")
    ap.add_argument("--tau", type=float, default=None,
                    help="覆盖 purpose_filter_threshold（换 embedding 后必须重标）")
    ap.add_argument("--llm-model", default=None)
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / args.queries, encoding="utf-8") as f:
        queries = _flatten(yaml.safe_load(f) or {})
    if args.limit:
        queries = queries[: args.limit]
    graph, retriever = build_retriever(
        str(ROOT / args.yaml), llm_model=args.llm_model,
        vector_cache_key=args.vector_cache_key)
    if args.tau is not None:
        retriever.purpose_filter_threshold = float(args.tau)
    print(f"[E5] queries={len(queries)} nodes={graph.node_count} "
          f"emb={os.environ.get('EMBEDDING_MODEL')} tau={retriever.purpose_filter_threshold} "
          f"purpose_mode={args.purpose_mode}", flush=True)

    rows, failed = [], []
    for q in queries:
        try:
            row = _build_row(retriever, q, args)
        except Exception as exc:
            failed.append({"query_id": q["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {q['id']} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(row)
        st = row["stats"]
        print(f"  {row['query_id']} [{row['scenario']}/{row['intent']}] "
              f"par={st['par_nodes']}n(vp={st['vp_nodes']},vt={st['vt_nodes']}) "
              f"recall par={st['gold_recall']['par']} vp={st['gold_recall']['vector_purpose']}",
              flush=True)

    metrics = _group_metrics(rows)
    print("\n[E5] 同情景内意图间 Jaccard（越低=越随意图分化）")
    for sc, v in metrics["per_scenario"].items():
        print(f"  {sc:14s} par={v['par']['mean']:.3f}  vp={v['vector_purpose']['mean']:.3f}  "
              f"vt={v['vector_text']['mean']:.3f}")
    s = metrics["summary"]
    print(f"  {'平均':14s} par={s['par']['jaccard_mean']:.3f}  "
          f"vp={s['vector_purpose']['jaccard_mean']:.3f}  "
          f"vt={s['vector_text']['jaccard_mean']:.3f}")
    print("\n[E5] gold facet recall（越高=分化的方向越对）")
    print(f"  par={s['par']['recall_mean']}  vp={s['vector_purpose']['recall_mean']}  "
          f"vt={s['vector_text']['recall_mean']}")

    payload = {
        "meta": {
            "experiment": "E5",
            "design": "same ambiguous text x 3 purposes; main table = par vs vector_purpose",
            "yaml": args.yaml,
            "queries": args.queries,
            "judged_arms": list(MAIN_ARMS),
            "internal_control": ["vector_text"],
            "seed_k": args.seed_k,
            "purpose_mode": args.purpose_mode,
            "anon_seed": args.anon_seed,
            "embedding": os.environ.get("EMBEDDING_MODEL"),
            "purpose_filter_threshold": retriever.purpose_filter_threshold,
            "metrics": metrics,
            "note": ("同一 scenario 的 query 文本逐字相同，只换注入目的；"
                     "vector_purpose 把文本与 purposes 拼接后嵌入（公平对照）；"
                     "vector_text 仅文本、不进判分，仅作下界校验（同句 Jaccard 应=1）。"),
            "judge": "python -m eval.judge_e5_intent --ctx <this file>",
            "failed": failed,
        },
        "queries": rows,
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E5] wrote {p}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
