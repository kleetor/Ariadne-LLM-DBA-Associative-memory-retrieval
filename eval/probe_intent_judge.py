# -*- coding: utf-8 -*-
"""标准错配检查：用「意图条件相关性」重判 E4 产物（不看 gold、不看命中）。

对 30 条 query，取 full 与 no_purpose 两臂的故事，用同一套「是否服务于该意图」的
判据打分（1–5），比较两臂。若 full 不再落后，说明原来的 gold recall 判据误导了结论。

用法：python -m eval.probe_intent_judge
输出：eval/outputs/intent_judge_e4.json
"""

from __future__ import annotations

import ast
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

PROMPT = """用户此刻说：{q}
当前检索目的（由对话上下文确定）：{p}

下面是一段从记忆库中取出的上下文：
{story}

请只回答一个问题：**这段上下文是否服务于上述目的、能否帮助理解用户此刻的状态与需求。**
不要比较其他答案，不要考虑是否"覆盖了标准答案"。1–5 分：
1=偏离目的；3=部分相关但有明显无关内容；5=紧密服务于目的。

只输出 JSON：{{"score": n, "reason": "一句话"}}
"""


def _parse(txt: str):
    m = re.search(r"\{[\s\S]*\}", txt or "")
    if not m:
        return None
    try:
        return int(json.loads(m.group(0)).get("score"))
    except Exception:
        return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=["deepseek", "qwen", "gpt"], default="deepseek")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    _load_dotenv(ROOT / ".env")
    if args.line == "deepseek":
        model = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")
        base = os.environ.get("OPENAI_API_BASE")
        key = os.environ.get("OPENAI_API_KEY") or "not-needed"
    else:
        model = (os.environ.get("JUDGE_MODEL") if args.line == "qwen"
                 else os.environ.get("JUDGE_MODEL3"))
        base = os.environ.get("JUDGE_API_BASE")
        key = os.environ.get("JUDGE_API_KEY") or "not-needed"
    tag = args.tag if args.tag is not None else f"_{args.line}"
    kwargs = dict(model=model, api_key=key, base_url=base, temperature=0, max_tokens=256)
    if args.line == "deepseek":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    llm = ChatOpenAI(**kwargs)
    data = json.loads((ROOT / "eval/outputs/e4/context_e4_purpose.json").read_text(encoding="utf-8"))
    rows = []
    for q in data["queries"]:
        if q.get("category") == "abstention":
            continue
        lm = q.get("label_map") or {}
        inv = {v: k for k, v in lm.items()}
        if "full" not in inv.values() or "no_purpose" not in inv.values():
            continue
        try:
            purposes = ast.literal_eval(str(q["stats"]["full"].get("purposes", "[]")))
        except Exception:
            purposes = []
        ptext = "、".join(purposes) if purposes else "（未给出明确目的）"
        scores = {}
        for label, arm in inv.items():
            if arm not in ("full", "no_purpose"):
                continue
            story = (q["systems"].get(label) or {}).get("story") or ""
            s = _parse(llm.invoke(PROMPT.format(q=q["query"], p=ptext, story=story)).content)
            scores[arm] = s
        if len(scores) < 2 or None in scores.values():
            print("  !! skip", q["query_id"], scores, flush=True)
            continue
        rows.append({"qid": q["query_id"], "full": scores["full"], "no_purpose": scores["no_purpose"]})
        print(f"  {q['query_id']}: full={scores['full']} no_purpose={scores['no_purpose']}", flush=True)
        if len(rows) >= 30:
            break

    f = [r["full"] for r in rows]
    n = [r["no_purpose"] for r in rows]
    d = [a - b for a, b in zip(f, n)]
    win = sum(1 for x in d if x > 0)
    tie = sum(1 for x in d if x == 0)
    lose = sum(1 for x in d if x < 0)
    summary = {"n": len(rows), "full_mean": round(statistics.mean(f), 3),
               "no_purpose_mean": round(statistics.mean(n), 3),
               "win_tie_loss": [win, tie, lose]}
    try:
        from scipy.stats import wilcoxon
        if any(d):
            w = wilcoxon(f, n)
            summary["wilcoxon_p"] = float(w.pvalue)
    except Exception:
        pass
    print("\n[summary]", summary)
    out = ROOT / f"eval/outputs/intent_judge_e4{tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("[wrote]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
