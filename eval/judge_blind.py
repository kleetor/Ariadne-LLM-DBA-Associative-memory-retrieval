# -*- coding: utf-8 -*-
"""prompt 偏好对照实验：区分「匿名化」与「取消三维明细」两个变量的独立影响。

三种判定方式（仅提示词不同，口径均为三系统对照 1–5 总分 + 排名）：
  - mode=blind : 匿名化（A/B/C，随机打乱）+ 取消三维（整体分）      → 原盲评
  - mode=named : 不匿名（保留真实系统名）+ 取消三维（整体分）        → 本版新增
对比基准（带名 + 三维明细）：judge_comprehensive.json / judge_deepseek.json

要点：named 与 blind 唯一差异 = 是否暴露系统名；其余（整体分 + 排名 + 短评）完全一致，
从而隔离「取消三维」的效应（= named vs 带名三维）、「匿名」的效应（= blind vs named）。

用法：
    python -m eval.judge_blind --mode blind --model qwen3.8-flash \
        --ctx eval/outputs/context_comprehensive_3sys.json --out eval/outputs/judge_blind_qwen.json
    python -m eval.judge_blind --mode named --model qwen3.8-flash \
        --ctx eval/outputs/context_comprehensive_3sys.json --out eval/outputs/judge_named_overall_qwen.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from eval.judge_comprehensive import (_extract_json, _load_dotenv, build_judge_llm, load_context)

DIMS = ["记忆丰富度", "易理解性", "用户建模度"]
REAL = ["Ariadne", "Mem0", "Graphiti"]
LETTERS = ["A", "B", "C"]
ABSTAIN = "abstention"

# 匿名 + 整体（保持 A/B/C，不给维度明细，含短评）
ANON_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与三个记忆系统各自检索并组装、"
     "将喂给模型的结构化记忆上下文。三个块已匿名化为 A、B、C（顺序不代表优劣）。\n"
     "请结合同一段倾诉，**凭直觉**对 A、B、C 三块做**总体对比评价**：哪块作为陪伴素材更好、"
     "更能帮助理解与共情用户。只需给出：\n"
     "1. 每块的总分（1–5，5=明显更好，1=明显更差）\n"
     "2. 排名（1=最好，2=次之，3=最差）\n"
     "3. 一句总体短评（不超过一句话，说明你把哪块放在第一、以及最看重的原因），放入 comment 字段\n"
     "只输出 JSON：{{\"A\":{{\"total\":0}},\"B\":{{\"total\":0}},\"C\":{{\"total\":0}},"
     "\"ranking\":[\"A\",\"B\",\"C\"],\"comment\":\"\"}}"),
    ("human", "用户的倾诉：{query}\n\n{blocks}"),
])

# 不匿名 + 整体（保留真实系统名，不给维度明细，含短评）
NAMED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与三个记忆系统（Ariadne / Mem0 / Graphiti）"
     "各自检索并组装、将喂给模型的结构化记忆上下文。\n"
     "请结合同一段倾诉，**凭直觉**对 Ariadne / Mem0 / Graphiti 三块做**总体对比评价**："
     "哪块作为陪伴素材更好、更能帮助理解与共情用户。只需给出：\n"
     "1. 每块的总分（1–5，5=明显更好，1=明显更差）\n"
     "2. 排名（1=最好，2=次之，3=最差）\n"
     "3. 一句总体短评（不超过一句话，说明你把哪块放在第一、以及最看重的原因），放入 comment 字段\n"
     "只输出 JSON：{{\"Ariadne\":{{\"total\":0}},\"Mem0\":{{\"total\":0}},\"Graphiti\":{{\"total\":0}},"
     "\"ranking\":[\"Ariadne\",\"Mem0\",\"Graphiti\"],\"comment\":\"\"}}"),
    ("human", "用户的倾诉：{query}\n\n{blocks}"),
])


def render_order(seed: int):
    """给 seed 生成一条 A/B/C->真实系统 的映射与显示顺序（可复现、条间不同）。"""
    rng = random.Random(seed)
    order = LETTERS[:]
    rng.shuffle(order)
    letter_to_real = {letter: real for letter, real in zip(order, REAL)}
    return order, letter_to_real


def judge_overall(judge_llm, query, blocks, mode, seed, max_attempt=5):
    if mode == "blind":
        order, letter_to_real = render_order(seed)
        prompt = ANON_PROMPT
        rendered = "\n\n".join(f"## {letter}\n{json.dumps(blocks[letter_to_real[letter]], ensure_ascii=False)}"
                               for letter in order)
        keys = LETTERS
        key_to_real = letter_to_real.get
    else:  # named
        prompt = NAMED_PROMPT
        rendered = "\n\n".join(f"## {n}\n{json.dumps(blocks[n], ensure_ascii=False)}" for n in REAL)
        keys = REAL
        key_to_real = lambda k: k

    chain = prompt | judge_llm
    last = None
    for att in range(max_attempt):
        try:
            resp = chain.invoke({"query": query, "blocks": rendered})
            d = _extract_json(resp.content) if (resp is not None and getattr(resp, "content", None)) else None
            if isinstance(d, dict) and any(float(d.get(k, {}).get("total", 0) or 0) > 0 for k in keys):
                last = d
                break
            last = d
        except Exception:
            last = None
        if att < max_attempt - 1:
            time.sleep(2 * (att + 1))
    d = last or {}
    totals = {key_to_real(k): float(d.get(k, {}).get("total", 0)) for k in keys}
    ranking = d.get("ranking")
    if isinstance(ranking, list) and len(ranking) == len(keys) and all(k in totals and k in d for k in ranking):
        rk = [key_to_real(k) for k in ranking]
    else:
        rk = sorted(REAL, key=lambda r: totals[r], reverse=True)
    comment = str(d.get("comment", "") or "")
    return {"totals": totals, "ranking": rk, "comment": comment}


def load_named_scores(path="eval/outputs/judge_comprehensive.json"):
    """带名 + 三维主判分每题的 total 排位，作对比基准。"""
    p = ROOT / path
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for r in d["results"]:
        sc = r.get("scores", {})
        tot = {s: sc[s]["total"] for s in REAL if sc.get(s)}
        rk = [x for x, _ in sorted(tot.items(), key=lambda kv: kv[1], reverse=True)] if len(tot) >= 2 else []
        out[r["query_id"]] = {"category": r.get("category"), "totals": tot, "ranking": rk}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blind", "named"], default="blind", help="blind=匿名+整体；named=不匿名+整体")
    ap.add_argument("--model", default="qwen3.8-flash")
    ap.add_argument("--ctx", default="eval/outputs/context_comprehensive_3sys.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="覆盖 max_tokens（gemini 等推理型模型需 4096，否则 JSON 被截断）")
    ap.add_argument("--named", default="eval/outputs/judge_comprehensive.json", help="带名多维判分（对比基准）")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    if args.max_tokens:
        judge_llm = ChatOpenAI(
            model=args.model,
            api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
            http_client=httpx.Client(trust_env=False),
            temperature=0, max_tokens=args.max_tokens)
    else:
        judge_llm = build_judge_llm(args.model)
    ctx = load_context(args.ctx)
    if args.limit:
        ctx["queries"] = ctx["queries"][:args.limit]

    suffix = "blind" if args.mode == "blind" else "named_overall"
    out = ROOT / (args.out or f"eval/outputs/judge_{suffix}_{args.model}.json")
    print(f"[{args.mode}] {len(ctx['queries'])} 条 | judge={judge_llm.model_name} | seed={args.seed}", flush=True)

    results = []
    for i, r in enumerate(ctx["queries"]):
        seed = args.seed + i
        bj = judge_overall(judge_llm, r["query"], r["blocks"], args.mode, seed)
        results.append({"query_id": r["query_id"], "category": r.get("category"),
                        "query": r["query"], "scores": bj["totals"], "ranking": bj["ranking"],
                        "comment": bj["comment"]})
        print(f"[{args.mode}] {r['query_id']} -> "
              "  ".join(f"{s}={bj['totals'][s]:.0f}" for s in REAL) + f"  排名={bj['ranking']}  "
              f"短评={bj['comment'][:36]}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {"judge_model": judge_llm.model_name, "ctx": args.ctx, "seed": args.seed,
                                        "mode": args.mode, "rubric": "overall_1pass"},
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.mode}] 结果已写: {out}（{len(results)} 条）")

    # ---- 统计 ----
    named = load_named_scores(args.named)
    non_abstain = [r for r in results if r["category"] != ABSTAIN]
    print(f"\n===== {args.mode} 统计（非弃权 {len(non_abstain)} 条） =====")
    for s in REAL:
        scores = [r["scores"][s] for r in non_abstain]
        first = sum(1 for r in non_abstain if r["ranking"][0] == s)
        print(f"  {s:<10} 总分均={sum(scores)/len(scores):.2f}   排名第一占比={first}/{len(non_abstain)} ({first/len(non_abstain):.1%})")
    print("\n  —— 对比带名(三维)主判分 ——")
    for s in REAL:
        named_first = sum(1 for qid, v in named.items() if v["category"] != ABSTAIN and v["ranking"] and v["ranking"][0] == s)
        n = sum(1 for v in named.values() if v["category"] != ABSTAIN)
        print(f"  {s:<10} 带名排名第一占比={named_first}/{n} ({named_first/n:.1%})")
    agree = tot = 0
    for r in non_abstain:
        nv = named.get(r["query_id"])
        if nv and nv["ranking"]:
            tot += 1
            if r["ranking"][0] == nv["ranking"][0]:
                agree += 1
    if tot:
        print(f"\n  {args.mode} vs 带名 top-1 一致率：{agree}/{tot} = {agree/tot:.2%}")

    summary = {"mode": args.mode, "judge_model": judge_llm.model_name, "non_abstain": len(non_abstain),
               "system_avg": {s: round(sum(r["scores"][s] for r in non_abstain)/len(non_abstain), 2) for s in REAL},
               "system_first_ratio": {s: round(sum(1 for r in non_abstain if r["ranking"][0] == s)/len(non_abstain), 3) for s in REAL},
               "top1_agree_with_named": round(agree/tot, 3) if tot else None}
    out2 = ROOT / "eval/outputs" / f"judge_{suffix}_summary_{args.model}.json"
    out2.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.mode}] 汇总已写: {out2}")


if __name__ == "__main__":
    main()
