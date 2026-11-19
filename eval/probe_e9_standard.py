# -*- coding: utf-8 -*-
"""E9 标准对照：同一批叙事，用「严格服务于诉求」vs「用户建模」两套标准判无关内容。

若严格标准命中率远高于宽松标准，说明 92.1% 的 cross_domain 主要是标准选择的结果，
而不是叙事真的跑偏。

用法：python -m eval.probe_e9_standard [--n 30]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_openai import ChatOpenAI  # noqa: E402

from eval.run_p_baseline import _load_dotenv  # noqa: E402

PROMPT = """用户原话：{q}

记忆叙事：
{story}

请用两个标准分别判断这段叙事是否存在「无关内容」问题：
A（严格：只服务于当前诉求）——只要包含与用户原话诉求无关的内容，即算 1。
B（宽松：用户建模）——只要是同一用户的生活事实、不编造、不误导，即使与当前诉求无关，也算 0。

只输出 JSON：{{"A": 0或1, "B": 0或1, "reason": "一句话"}}
"""


def _parse(txt):
    txt = txt or ""
    # 优先按 JSON 解析：deepseek/gemini 常返回 ```json ... ``` 或带前后说明的 JSON
    try:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            data = json.loads(m.group(0))
            a, b = data.get("A"), data.get("B")
            if a is not None and b is not None:
                return int(a), int(b)
    except Exception:
        pass

    def val(k):
        m = re.search(rf'"{k}"\s*:\s*["\']?\s*(是|否|true|false|1|0)', txt, re.I)
        if not m:
            # 兜底：键不带引号，如 A: 1
            m = re.search(rf'\b{k}\b\s*[:：]\s*["\']?\s*(是|否|true|false|1|0)', txt, re.I)
        if not m:
            return None
        return 1 if m.group(1).lower() in ("是", "true", "1") else 0
    a, b = val("A"), val("B")
    if a is None or b is None:
        return None
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default="qwen3.8-flash")
    args = ap.parse_args()
    _load_dotenv(ROOT / ".env")

    llm = ChatOpenAI(
        model=args.model,
        api_key=os.environ.get("JUDGE_API_KEY") or "not-needed",
        base_url=os.environ.get("JUDGE_API_BASE"),
        temperature=0, max_tokens=256,
    )
    ctx = json.loads((ROOT / "eval/outputs/e1/context_e1_current.json").read_text(encoding="utf-8"))["queries"]
    rows = []
    for r in ctx[: args.n]:
        story = ((r.get("systems") or {}).get("ariadne") or {}).get("story") or ""
        got = _parse(llm.invoke(PROMPT.format(q=r["query"], story=story)).content)
        if got is None:
            print("  !! parse fail", r["query_id"], flush=True)
            continue
        a, b = got
        rows.append({"qid": r["query_id"], "A": a, "B": b})
        print(f"  {r['query_id']}: A={a} B={b}", flush=True)

    A = [x["A"] for x in rows]; B = [x["B"] for x in rows]
    summary = {"n": len(rows), "strict_rate": round(statistics.mean(A), 3),
               "user_modeling_rate": round(statistics.mean(B), 3),
               "both": sum(1 for x, y in zip(A, B) if x and y),
               "strict_only": sum(1 for x, y in zip(A, B) if x and not y),
               "loose_only": sum(1 for x, y in zip(A, B) if y and not x)}
    print("\n[summary]", summary)
    out = ROOT / "eval/outputs/e9_standard_probe.json"
    out.write_text(json.dumps({"model": args.model, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("[wrote]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
