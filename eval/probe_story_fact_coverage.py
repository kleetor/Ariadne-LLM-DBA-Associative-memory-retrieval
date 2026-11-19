# -*- coding: utf-8 -*-
"""验证一个想法：两图故事分数相当，是"组织不同但事实齐"，还是"rubric 对遗漏不敏感"。

做法：以 gold 故事为参照，抽取最多 8 条具体事实，判断 DBA 故事是否以相同含义覆盖
（措辞可不同）。同时用"gold 故事 vs 另一条 query 的 DBA 故事"作随机基线。

输出：eval/outputs/story_fact_coverage.json
用法：python -m eval.probe_story_fact_coverage
"""

from __future__ import annotations

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

PROMPT = """下面有两段关于同一用户的记忆叙述：A 是参考叙述，B 是另一段叙述。
当前用户消息（这决定了"哪些事实算相关"）：
{q}

请先从 A 中抽取最多 8 条**与上述消息相关**的具体事实（事件、状态、数字/时间、关系、偏好；
不要抽象概括、不要抽与该消息无关的背景），然后逐条判断 B 是否以**相同含义**提到了它——
措辞可以完全不同（例如"A 加班到 11 点"与"B 每天十一二点才下班"算覆盖）。

只输出 JSON：
{{"facts": ["事实1", "事实2"], "covered": [true, false]}}

A:
{a}

B:
{b}
"""


def _parse(txt: str):
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        facts, cov = o.get("facts") or [], o.get("covered") or []
        if len(facts) != len(cov) or not facts:
            return None
        return facts, [bool(x) for x in cov]
    except Exception:
        return None


def _judge(llm, q: str, a: str, b: str):
    resp = llm.invoke(PROMPT.format(q=q, a=a, b=b))
    return _parse(getattr(resp, "content", "") or "")


def main() -> int:
    _load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=1024,
        extra_body={"thinking": {"type": "disabled"}},
    )
    rows = json.loads((ROOT / "eval/outputs/dba_story_full.json").read_text(encoding="utf-8"))["results"]
    qids = [r["query_id"] for r in rows]

    out, covs = [], []
    for i, r in enumerate(rows):
        got = _judge(llm, r["query"], r["gold"]["story"], r["dba"]["story"])
        if not got:
            print(f"  !! {r['query_id']} 解析失败", flush=True)
            continue
        facts, cov = got
        c = sum(cov) / len(cov)
        covs.append(c)
        out.append({"query_id": r["query_id"], "n_facts": len(facts), "coverage": round(c, 3)})
        print(f"  {r['query_id']}: {sum(cov)}/{len(cov)} = {c:.2f}", flush=True)

    # 随机基线：gold 故事 vs 另一条 query 的 DBA 故事
    rnd = []
    for i in range(0, len(rows), 4):
        a = rows[i]["gold"]["story"]
        b = rows[(i + 7) % len(rows)]["dba"]["story"]
        got = _judge(llm, rows[i]["query"], a, b)
        if got:
            rnd.append(sum(got[1]) / len(got[1]))
        print(f"  [random {rows[i]['query_id']}] {rnd[-1] if rnd else 'fail'}", flush=True)

    summary = {
        "n": len(covs),
        "coverage_mean": round(statistics.mean(covs), 3) if covs else None,
        "coverage_median": round(statistics.median(covs), 3) if covs else None,
        "random_baseline_mean": round(statistics.mean(rnd), 3) if rnd else None,
        "n_random": len(rnd),
    }
    p = ROOT / "eval/outputs/story_fact_coverage_withq.json"
    p.write_text(json.dumps({"summary": summary, "rows": out}, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n[summary]", summary)
    print("[wrote]", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
