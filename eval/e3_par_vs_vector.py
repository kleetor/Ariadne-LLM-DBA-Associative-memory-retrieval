# -*- coding: utf-8 -*-
"""E3：PAR vs 纯向量的同图对照（默认三臂），可切回四臂 2×2 作敏感性。

默认条件（--conditions）：
- par_story : PAR 选择 + StoryRank 叙事
- par_raw   : PAR 选择 + 原始节点/边结构
- vector_raw: 纯向量 top-k + 原始节点（无叙事、无边）

`vector_story` 保留但默认不跑：StoryRank 的提示词面向「图路径（节点+关系）」，
纯向量返回的是排序队列，强行成文只能靠补关系，会把"记忆"写成"模型生成的叙事"，
既有幻觉风险、也不构成对检索机制的公平比较。

关键控制：
- 同一张图（data/dba_eval/comprehensive_gold.yaml）、同一 query、同一候选池；
- 同一 `inference.story_rank` 组装器、同一渲染函数（story_rank 输入格式）；
- 只改变「记忆选择机制」；向量条件按 PAR 的节点数做等规模对齐（--vec-k-mode match_par）。
- **向量条件不带边**（--vector-edges none）：纯向量只返回记忆文本，图内关系是 PAR 的产物；
  给它补边等于把图谱结构白送基线。
- 输出统计里带「编造审计」：故事中的因果连接词数 vs 输入里真实 CAUSAL 边数。
- 只做选择/组装，不做 judge；judge 走 eval.judge_comprehensive（同 rubric、同并排协议）。
- 输出块匿名成 系统A–D（A=par_story, B=vector_story, C=par_raw, D=vector_raw），
  映射见每行 label_map，避免 judge 看到机制名产生先验偏好。

用法：
    python -m eval.e3_par_vs_vector --limit 2
    python -m eval.e3_par_vs_vector --out eval/outputs/e3/context_e3_2x2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 与 run_p_baseline 一致：避免 torch/FAISS OpenMP 冲突、禁止 HF 联网检查
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml  # noqa: E402

from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever  # noqa: E402
from eval.collect_ariadne_raw import ariadne_raw_block  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

CONDITIONS = ("par_story", "par_raw", "vector_story", "vector_raw")
DEFAULT_CONDITIONS = ("par_story", "par_raw", "vector_raw")

# 判分时**匿名化**：judge 只看得到 系统A–C，不知道哪个是 PAR / 向量、哪个做了叙事化，
# 避免对"par/vector/story"这些名字的先验偏好。映射随输出落盘（label_map），汇总可还原。
# **条件与标签的对应关系按 query 用固定种子打乱**（--anon-order shuffle，默认）：
# 0908 e2e 文档 §4.1 要求"每条查询随机打乱、避免位置偏好"，固定映射会让同一条件永远
# 落在同一位置，位置偏好即变成机制差异（实测差距 0.35，与机制差异同量级）。
_LABEL_POOL = ["系统A", "系统B", "系统C", "系统D"]


def _label_map(qid: str, seed: int, order: str, conds: List[str]) -> Dict[str, str]:
    """条件 → 匿名称的映射。order=shuffle 时按 (seed, qid) 稳定打乱，可复现。"""
    labels = _LABEL_POOL[: len(conds)]
    if order == "fixed":
        return dict(zip(conds, labels))
    import random

    shuffled = list(conds)
    random.Random(f"{seed}:{qid}").shuffle(shuffled)
    return dict(zip(shuffled, labels))


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a.dot(b) / (na * nb))


def _vector_topk(retriever: PurposeDrivenRetriever, query: str, k: int) -> List[Tuple[str, float]]:
    """纯向量检索：只按「query × 节点内容」余弦取前 k，不做任何图扩展。

    与基线同义：这是 PAR 要超越的下限；直接读向量缓存（get_content_vectors 无 API 调用），
    只多一次 query 嵌入。
    """
    vs = retriever.vector_store
    graph = retriever.graph
    query_vec = vs._embed(query)
    ids = [nid for nid in graph.graph.nodes() if graph.graph.nodes[nid].get("content")]
    vectors = vs.get_content_vectors(ids)
    scored = [(nid, _cosine(query_vec, vec)) for nid, vec in zip(ids, vectors) if vec is not None]
    scored.sort(key=lambda pair: -pair[1])
    return scored[:k]


def _path_from_ids(retriever: PurposeDrivenRetriever, scored: List[Tuple[str, float]],
                   with_edges: bool = False) -> Dict[str, Any]:
    """把向量选中的节点集合构造为 `_build_path` 同构的 {nodes, edges}。

    默认 `with_edges=False`：纯向量检索**没有边**——它只返回一段记忆文本，图内关系是 PAR
    才有的产物。若在这里补上「选中节点之间的图内边」，等于把图谱结构白送给了基线，
    组装出来的故事会带上本不属于向量系统的因果/时序链，比较就不成立。
    `with_edges=True` 仅用于敏感性检查（"如果向量也能看图，会怎样"）。
    """
    graph = retriever.graph
    ids = [nid for nid, _ in scored]
    id_set = set(ids)
    score_map = dict(scored)
    nodes = []
    for nid in ids:
        data = graph.graph.nodes[nid]
        nt = data.get("node_type")
        nodes.append({
            "id": nid,
            "content": str(data.get("content") or ""),
            "node_type": nt.value if hasattr(nt, "value") else str(nt),
            "timestamp": data.get("timestamp"),
            "score": float(score_map.get(nid) or 0.0),
        })
    edges = []
    if with_edges:
        for u, v, data in graph.graph.edges(data=True):
            if u in id_set and v in id_set:
                rel = data.get("rel_type")
                edges.append({
                    "from": u,
                    "to": v,
                    "rel_type": rel.value if hasattr(rel, "value") else str(rel),
                    "is_reverse": False,
                })
    return {"nodes": nodes, "edges": edges}


def _causal_edges(path: Dict[str, Any]) -> int:
    return sum(1 for e in (path.get("edges") or []) if str(e.get("rel_type")) == "causal")


_CAUSAL_WORDS = ("因为", "导致", "所以", "于是", "因此", "结果", "这使得", "这让")


def _causal_words(story: str) -> int:
    return sum(story.count(w) for w in _CAUSAL_WORDS)


def _story(retriever: PurposeDrivenRetriever, query: str, path: Dict[str, Any]) -> Tuple[str, bool]:
    """返回 (story, degraded)；degraded=True 表示 StoryRank 解析失败、正文是原文拼接。"""
    out = retriever.inference.story_rank(query, path)
    return str(out.get("story") or ""), bool(out.get("degraded"))


def _load_queries(path: Path, exclude: List[str]) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    qs = data.get("queries") or []
    excluded = {c.strip() for c in exclude if c and c.strip()}
    return [q for q in qs if str(q.get("category") or "") not in excluded]


def _build_row(retriever: PurposeDrivenRetriever, q: Dict[str, Any], args) -> Dict[str, Any]:
    """单条 query 的四条件上下文块 + 统计（PAR 检索一次，story/raw 共用同一条 path）。"""
    qid = str(q["id"])
    qtext = str(q["text"])

    # ---- PAR：只跑一次检索，story/raw 共用同一条 path ----
    # purpose_mode=inject 时用查询集里人工标注的 purposes（模拟主 LLM 在会话中给出的目的），
    # 用于区分「选择机制」与「目的推断」两条误差来源。
    purpose = q.get("purposes") if args.purpose_mode == "inject" else None
    par = retriever.retrieve_with_story(
        qtext, seed_k=args.seed_k, max_hops=args.max_hops, expand_k=args.expand_k,
        story=False, with_response=False, purpose=purpose or None,
    )
    par_path = par.get("path") or {"nodes": [], "edges": []}
    par_nodes = len(par_path.get("nodes") or [])

    # ---- vector：同图、同 query；规模按 PAR 对齐 ----
    vec_k = par_nodes if args.vec_k_mode == "match_par" else args.seed_k
    vec_k = max(int(vec_k), 1)
    scored = _vector_topk(retriever, qtext, vec_k)
    vec_path = _path_from_ids(retriever, scored, with_edges=(args.vector_edges == "internal"))

    conds = args.conditions_list
    n_par_edges = len(par_path.get("edges") or [])
    n_vec_edges = len(vec_path.get("edges") or [])

    real_blocks = {}
    par_story = vec_story = ""
    par_degraded = vec_degraded = False
    if "par_story" in conds:
        par_story, par_degraded = _story(retriever, qtext, par_path)
        real_blocks["par_story"] = {"story": par_story, "n_nodes": par_nodes}
    if "par_raw" in conds:
        real_blocks["par_raw"] = {**ariadne_raw_block(par_path), "n_nodes": par_nodes}
    if "vector_story" in conds:
        # 仅作敏感性/附录：story 提示词是按「图路径」设计的，纯向量是排序队列，
        # 强行叙事化会让模型为成文而补关系，等于给向量基线加幻觉（默认不跑）。
        vec_story, vec_degraded = _story(retriever, qtext, vec_path)
        real_blocks["vector_story"] = {"story": vec_story, "n_nodes": len(vec_path["nodes"])}
    if "vector_raw" in conds:
        real_blocks["vector_raw"] = {**ariadne_raw_block(vec_path),
                                     "n_nodes": len(vec_path["nodes"])}

    lm = _label_map(qid, args.anon_seed, args.anon_order, conds)
    labels = _LABEL_POOL[: len(conds)]
    return {
        "query_id": qid,
        "category": q.get("category", ""),
        "query": qtext,
        "systems": {label: real_blocks[cond] for label in labels
                    for cond in conds if lm[cond] == label},
        "label_map": lm,
        "stats": {
            "purpose_mode": args.purpose_mode,
            "purposes": str((par.get("purpose") or {}).get("purposes")),
            "par_nodes": par_nodes,
            "par_edges": n_par_edges,
            "par_hops": len(par.get("hop_history") or []),
            "par_story_degraded": par_degraded,
            "vec_k": vec_k,
            "vec_nodes": len(vec_path["nodes"]),
            "vec_edges": n_vec_edges,
            "vec_story_degraded": vec_degraded,
            "par_story_chars": len(par_story),
            "vec_story_chars": len(vec_story),
            # 编造审计：故事里出现多少因果连接词 vs 其输入里真正标了 CAUSAL 的边
            "par_causal_edges": _causal_edges(par_path),
            "vec_causal_edges": _causal_edges(vec_path),
            "par_causal_words": _causal_words(par_story),
            "vec_causal_words": _causal_words(vec_story),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="E3 PAR vs vector 同图同组装器 2×2 上下文块")
    parser.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml", help="图谱 YAML")
    parser.add_argument("--queries", default="data/dba_eval/comprehensive_queries.yaml", help="查询 YAML")
    parser.add_argument("--out", default="eval/outputs/e3/context_e3_2x2.json")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    parser.add_argument("--seed-k", type=int, default=5, help="PAR 种子条数（与线上默认一致）")
    parser.add_argument("--max-hops", type=int, default=None, help="PAR 最大跳数（None=默认）")
    parser.add_argument("--expand-k", type=int, default=None, help="PAR 每跳扩展条数（None=默认）")
    parser.add_argument("--vec-k-mode", choices=["match_par", "seed_k"], default="match_par",
                        help="向量条件规模：match_par=与 PAR 节点数对齐（等规模对照）；seed_k=固定 k")
    parser.add_argument("--purpose-mode", choices=["infer", "inject"], default="infer",
                        help="PAR 目的来源：infer=检索器内部推断；inject=用查询集标注的 purposes")
    parser.add_argument("--vector-edges", choices=["none", "internal"], default="none",
                        help="向量条件是否携带图内边：none=纯向量（默认）；internal=敏感性检查")
    parser.add_argument("--anon-order", choices=["shuffle", "fixed"], default="shuffle",
                        help="匿名标签分配：shuffle=按 query 打乱（默认，避免位置偏好）；fixed=固定")
    parser.add_argument("--anon-seed", type=int, default=42, help="匿名打乱种子（可复现）")
    parser.add_argument("--exclude-categories", default="abstention",
                        help="逗号分隔的分类，默认排除 abstention（主判分 70 条非弃权）")
    parser.add_argument("--vector-cache-key", default="comprehensive", help="向量缓存 pkl 名")
    parser.add_argument("--llm-model", default=None, help="覆盖 .env OPENAI_MODEL（冒烟可用）")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                        help="逗号分隔的判分条件，默认 par_story,par_raw,vector_raw。"
                             "vector_story 只作敏感性/附录：story prompt 是按图路径设计的，"
                             "对纯向量排序队列强行成文会引入补关系（幻觉）风险")
    args = parser.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad or not conds:
        parser.error(f"--conditions 只能取 {CONDITIONS} 的非空组合，收到 {args.conditions!r}")
    args.conditions_list = conds

    _load_dotenv(ROOT / ".env")
    queries = _load_queries(ROOT / args.queries, args.exclude_categories.split(","))
    if args.limit:
        queries = queries[: args.limit]

    graph, retriever = build_retriever(
        str(ROOT / args.yaml), llm_model=args.llm_model, vector_cache_key=args.vector_cache_key,
    )
    print(f"[E3] nodes={graph.node_count} edges={graph.edge_count} queries={len(queries)}",
          flush=True)

    rows = []
    failed: List[Dict[str, str]] = []
    for q in queries:
        try:
            row = _build_row(retriever, q, args)
        except Exception as exc:  # 单条失败不中断整批（网络/网关抖动）
            failed.append({"query_id": str(q.get("id")), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {q.get('id')} 失败：{type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(row)
        st = row["stats"]
        print(f"  {row['query_id']} [{row['category']}] par={st['par_nodes']}n/{st['par_edges']}e "
              f"vec={st['vec_nodes']}n/{st['vec_edges']}e"
              + (f" par_story={st['par_story_chars']}字" if "par_story" in conds else "")
              + (f" vec_story={st['vec_story_chars']}字" if "vector_story" in conds else ""),
              flush=True)

    out = {
        "meta": {
            "experiment": "E3",
            "yaml": args.yaml,
            "queries": args.queries,
            "seed_k": args.seed_k,
            "max_hops": args.max_hops,
            "expand_k": args.expand_k,
            "vec_k_mode": args.vec_k_mode,
            "purpose_mode": args.purpose_mode,
            "vector_edges": args.vector_edges,
            "anon_order": args.anon_order,
            "anon_seed": args.anon_seed,
            "excluded_categories": args.exclude_categories,
            "conditions": args.conditions_list,
            "context_format": {c: ("{story}" if c.endswith("_story") else "{nodes,edges}")
                               for c in args.conditions_list},
            "note": ("条件共用同一图与同一渲染格式；只改记忆选择/是否叙事化。"
                     "判分侧匿名（系统A–C），逐题打乱，label_map 用于还原条件。"
                     "vector_story 默认不跑：story prompt 面向图路径，纯向量队列强行成文会补关系。"),
            "judge": "python -m eval.judge_comprehensive --ctx <this file>",
            "failed": failed,
        },
        "queries": rows,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E3] wrote {out_path}（{len(rows)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
