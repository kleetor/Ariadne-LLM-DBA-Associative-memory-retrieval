# -*- coding: utf-8 -*-
"""E9：失败类型编码（对 E1 的 Ariadne 产物做分类，供讨论章节用）。

为什么要做：主表只告诉我们"更好/更差"，回答不了"差在哪、错成什么样"。
失败类型是讨论部分唯一能给出**可操作改进方向**的证据。

六类（与清单 §E9 一致，互不排斥，一条产物可命中多类）：
  intent_miss     输出没覆盖当前诉求的核心侧面（如倾诉场景全是日程）
  purpose_error   检索主题整体跑偏（与 query 无关的领域）
  cross_domain    混入明显不同人生阶段/主题的节点，且未作为背景
  redundant       同一事实被重复叙述
  missing_fact    漏掉 query 里明确提到的时间/地点/人物/数值等约束
  over_story      把无因果关系的节点写成因果，或补了节点中没有的解释/推断

方法说明（写论文时必须交代）：这里是**LLM 辅助编码**，不是人工双标。
故输出只作"分布估计 + 案例抽取"，不作精确比例结论；抽样条目需人工复核。

用法：
    python -m eval.e9_failure_taxonomy --limit 8     # 冒烟
    python -m eval.e9_failure_taxonomy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import httpx  # noqa: E402
import yaml  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from eval.judge_comprehensive import _extract_json, _load_dotenv  # noqa: E402

FAIL_TYPES = ["intent_miss", "purpose_error", "cross_domain",
              "redundant", "missing_fact", "over_story"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是记忆系统的失败分析员。下面给出：用户的原话、这条查询的**期望事实**（gold）、
该次检索**实际提供给叙事的可用事实**（系统输入的记忆节点），以及系统产出的**记忆叙事**。

六类失败（**互不排斥，可多选**）。注意区分"编造"与"无关"——判据是**可用事实里有没有**：

- intent_miss：没有覆盖当前诉求的核心侧面（例：用户倾诉压力，叙事却通篇是日程安排）
- purpose_error：整体跑偏到与用户原话完全无关的领域/话题
- cross_domain：**可用事实里有**、但与当前诉求无关的内容占了显著篇幅（不是编造，是陪跑）
- redundant：同一事实被重复叙述
- missing_fact：漏掉了「用户原话」里**明确提到**的时间/地点/人物/数值等约束
- over_story：叙事中出现**可用事实之外**的内容——包括补出的动机、原因、心理、结果，
  或把可用事实之间**没有因果边**的两件事写成因果。
  ⚠️ 只要该事实出现在「可用事实」列表里，**无论多无关都不算 over_story**（应算 cross_domain）。

校准要求：
- 这是经过打磨的高质量候选叙事，**多数情况下只应命中 0–2 类**；命中 4 类以上通常说明判据松了。
- 每条命中的类别，`evidence` 必须**逐字引用叙事原文片段**；引不出来就不要判该类。
- 没有明显失败时，`failures` 返回空数组、`severity` 返回 `none`。

严格只输出 JSON：
{{"failures": ["..."], "severity": "none|minor|major", "evidence": "引用叙事原文的一句话"}}
severity：major = 会让下游回答出错或答非所问；minor = 影响有限；none = 无明显失败。"""),
    ("human", "用户原话：{query}\n\n期望事实（gold）：\n{gold}\n\n"
              "实际提供给叙事的可用事实：\n{avail}\n\n记忆叙事：\n{story}"),
])


def build_llm(model: str | None, api_key: str | None, api_base: str | None):
    return ChatOpenAI(
        model=model or os.environ.get("JUDGE_MODEL2") or "deepseek-v4-flash",
        api_key=api_key or os.environ.get("JUDGE_API_KEY") or "not-needed",
        base_url=api_base or os.environ.get("JUDGE_API_BASE"),
        http_client=httpx.Client(trust_env=False,
                                 timeout=httpx.Timeout(180.0, connect=15.0)),
        temperature=0,
        max_tokens=512,
        max_retries=1,
        extra_body={},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="E9 失败类型编码")
    ap.add_argument("--ctx", default="eval/outputs/e1/context_e1_current.json")
    ap.add_argument("--queries", default="data/dba_eval/comprehensive_queries.yaml")
    ap.add_argument("--gold", default="data/dba_eval/comprehensive_gold.yaml")
    ap.add_argument("--yaml", default="data/dba_eval/comprehensive_gold.yaml",
                    help="构建检索器用的图（与 E1 同一张）")
    ap.add_argument("--out", default="eval/outputs/e9/failure_taxonomy.json")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-retrieval", action="store_true",
                    help="跳过重新检索（此时无法判定 over_story，只作调试用）")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / args.queries, encoding="utf-8") as f:
        qs = {str(q["id"]): q for q in (yaml.safe_load(f) or {}).get("queries", [])}
    with open(ROOT / args.gold, encoding="utf-8") as f:
        nodes = {n["id"]: n.get("content", "")
                 for n in (yaml.safe_load(f) or {}).get("nodes", [])}
    with open(ROOT / args.ctx, encoding="utf-8") as f:
        rows = json.load(f)["queries"]
    if args.limit:
        rows = rows[: args.limit]

    # 重新检索一遍，拿到"实际提供给叙事的可用事实"。**必须做**：只给 gold 时，
    # 模型会把图里本来就有的事实误判成编造（冒烟里把 uw14「需求临时增加」判成了 over_story）。
    retriever = None
    if not args.no_retrieval:
        from eval.run_p_baseline import build_retriever
        _, retriever = build_retriever(str(ROOT / args.yaml),
                                       vector_cache_key="comprehensive")
        print("[E9] 已构建检索器，用于还原「可用事实」", flush=True)

    llm = build_llm(args.model, None, None)
    chain = PROMPT | llm
    print(f"[E9] 条数={len(rows)} model={llm.model_name}", flush=True)

    results: List[Dict[str, Any]] = []
    for r in rows:
        qid = r["query_id"]
        meta = qs.get(qid, {})
        story = str((r.get("systems") or {}).get("ariadne", {}).get("story") or "")
        gold_ids = list(meta.get("gold_expected") or [])
        gold_text = "\n".join(f"- {nodes.get(g, '')}" for g in gold_ids if nodes.get(g)) or "（无标注）"

        avail_ids: List[str] = []
        if retriever is not None:
            try:
                # **必须与 E1 的生成方式一致**：E1 调 `retrieve_with_story(text)` 未传目的，
                # 走内部推断。若这里注入 purposes，两次检索召回的节点不同，就会把
                # "故事写了另一版召回的内容" 误判成编造（over_story 假阳性）。
                out = retriever.retrieve_with_story(
                    str(meta.get("text") or r["query"]), seed_k=5, story=False,
                    with_response=False)
                avail_ids = [n["id"] for n in (out.get("path") or {}).get("nodes") or []]
            except Exception as exc:
                print(f"  !! {qid} 检索失败：{type(exc).__name__}", flush=True)
        avail_text = "\n".join(f"- {nodes.get(i, '')}" for i in avail_ids if nodes.get(i)) or "（未提供）"
        data = None
        for att in range(4):
            try:
                resp = chain.invoke({"query": r["query"], "gold": gold_text,
                                     "avail": avail_text, "story": story})
                data = _extract_json(resp.content or "")
                if isinstance(data, dict):
                    break
            except Exception:
                data = None
            time.sleep(2 * (att + 1))
        fails = [f for f in (data or {}).get("failures", []) if f in FAIL_TYPES]
        sev = str((data or {}).get("severity") or "unknown")
        results.append({
            "query_id": qid, "category": r.get("category", ""),
            "failures": fails, "severity": sev,
            "evidence": str((data or {}).get("evidence") or "")[:200],
            "story_chars": len(story),
        })
        print(f"  {qid} [{r.get('category','')}] {sev:7s} {','.join(fails) or '-'}", flush=True)

    cnt = Counter(f for x in results for f in x["failures"])
    sev_cnt = Counter(x["severity"] for x in results)
    n = len(results)
    print("\n=== 失败类型频次（n=%d，LLM 辅助编码，非人工双标）===" % n)
    for f in FAIL_TYPES:
        print(f"  {f:14s} {cnt.get(f,0):3d} 条  {cnt.get(f,0)/n:6.1%}")
    print(f"  严重度分布: {dict(sev_cnt)}")
    multi = Counter(len(x["failures"]) for x in results)
    print(f"  每条命中失败数分布: {dict(sorted(multi.items()))}")

    by_cat: Dict[str, Counter] = {}
    for x in results:
        c = by_cat.setdefault(x["category"], Counter())
        for f in x["failures"]:
            c[f] += 1
    print("\n=== 按类别 ===")
    for cat, c in sorted(by_cat.items()):
        print(f"  {cat:22s} " + "  ".join(f"{f}={v}" for f, v in c.most_common()))

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "meta": {"experiment": "E9", "n": n, "model": llm.model_name,
                 "fail_types": FAIL_TYPES,
                 "caveat": "LLM 辅助编码，非人工双标；只用于分布估计与案例抽取，需抽样复核"},
        "freq": dict(cnt), "severity": dict(sev_cnt), "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E9] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
