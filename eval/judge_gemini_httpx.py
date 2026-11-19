# -*- coding: utf-8 -*-
"""gemini 直连判分器（httpx，绕过 langchain）。

gpt.ge 的 gemini 经 langchain 调用时响应不稳定（偶发垃圾/全0），
而 httpx 直连 + 严格重试（非200 || 解析全0 → 重试）能拿到合法分数。
用法：
    python -m eval.judge_gemini_httpx --ctx eval/outputs/context_comprehensive_3sys.json \
        --model gemini-3.7-flash --out eval/outputs/judge_repeat_story_gemini_1.json
"""
import argparse
import json
import os
import sys
import time

import httpx
from dotenv import load_dotenv

from eval.judge_comprehensive import JUDGE_JOINT_PROMPT, _extract_json, _load_dotenv, load_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIMS = ["记忆丰富度", "易理解性", "用户建模度"]


def gemini_judge(llm_cfg, query, blocks, max_attempt=8):
    # 与 judge_comprehensive.judge_joint 完全同构的渲染，保证 gemini 分与 qwen/deepseek/gpt 一致口径
    rendered = "\n\n".join(f"## {name}\n{json.dumps(blk, ensure_ascii=False)}" for name, blk in blocks.items())
    msgs = []
    for m in JUDGE_JOINT_PROMPT.format_messages(query=query, blocks=rendered):
        msgs.append({"role": "system" if m.type == "system" else "user", "content": m.content})
    body = {"model": llm_cfg["model"], "messages": msgs, "max_tokens": 4096, "temperature": 0}
    headers = {"Authorization": "Bearer " + llm_cfg["key"], "Content-Type": "application/json"}
    for att in range(max_attempt):
        try:
            r = httpx.post(llm_cfg["base"] + "/chat/completions", headers=headers, json=body,
                           timeout=90, trust_env=False)
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                data = _extract_json(content)
                if isinstance(data, dict) and any(
                        float(data.get(n, {}).get("total", 0) or 0) > 0 for n in blocks):
                    return data
        except Exception:
            pass
        if att + 1 < max_attempt:
            time.sleep(2 * (att + 1))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", default="eval/outputs/context_comprehensive_3sys.json")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-base", default=None)
    args = p.parse_args()

    _load_dotenv(ROOT / ".env")
    cfg = {
        "model": args.model,
        "key": args.api_key or os.environ.get("JUDGE_API_KEY"),
        "base": (args.api_base or os.environ.get("JUDGE_API_BASE") or "https://api.gpt.ge/v1").rstrip("/"),
    }
    ctx = load_context(args.ctx)
    if args.limit:
        ctx["queries"] = ctx["queries"][:args.limit]
    print(f"[gemini] {len(ctx['queries'])} 条 | model={cfg['model']} | ctx={args.ctx}", flush=True)

    results = []
    for r in ctx["queries"]:
        data = gemini_judge(cfg, r["query"], r["blocks"])
        sc = {}
        if data:
            for name in r["blocks"]:
                d = data.get(name, {}) or {}
                vals = [float(d.get(k, 0)) for k in DIMS]
                sc[name] = {**{k: float(d.get(k, 0)) for k in DIMS}, "total": round(sum(vals) / len(vals), 2)}
        else:
            # 全部重试仍失败：置 0（供后续标记为无效）
            for name in r["blocks"]:
                sc[name] = {**{k: 0.0 for k in DIMS}, "total": 0.0}
        results.append({"query_id": r["query_id"], "category": r["category"], "query": r["query"], "scores": sc})
        print(f"[gemini] {r['query_id']} -> " + "  ".join(f"{n}={sc[n]['total']:.2f}" for n in sc), flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {"judge_model": cfg["model"], "mode": "httpx_direct",
                                        "ctx": args.ctx}, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gemini] 结果已写: {out}（{len(results)} 条）")


if __name__ == "__main__":
    main()
