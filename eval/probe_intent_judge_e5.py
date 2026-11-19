# -*- coding: utf-8 -*-
"""标准错配检查（E5）：用「意图条件相关性」重判 par vs vector_purpose。

30 条 query，同一意图下两臂故事，1–5 分，不看 gold。
用法：python -m eval.probe_intent_judge_e5
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
当前检索目的：{p}

记忆上下文：
{story}

只回答：这段上下文是否服务于上述目的、能否帮助理解用户此刻的状态与需求。1–5 分
（1=偏离；3=部分相关；5=紧密服务）。不要考虑是否覆盖了标准答案。

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
    _load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0, max_tokens=256,
        extra_body={"thinking": {"type": "disabled"}},
    )
    data = json.loads((ROOT / "eval/outputs/e5/context_e5_intent.json").read_text(encoding="utf-8"))
    rows = []
    for q in data["queries"]:
        lm = q.get("label_map") or {}
        inv = {v: k for k, v in lm.items()}
        if "par" not in inv.values() or "vector_purpose" not in inv.values():
            continue
        try:
            purposes = ast.literal_eval(str(q["stats"].get("purposes", "[]")))
        except Exception:
            purposes = []
        ptext = "、".join(purposes) if purposes else "（未给出明确目的）"
        sc = {}
        for label, arm in inv.items():
            if arm not in ("par", "vector_purpose"):
                continue
            story = (q["systems"].get(label) or {}).get("story") or ""
            sc[arm] = _parse(llm.invoke(PROMPT.format(q=q["query"], p=ptext, story=story)).content)
        if len(sc) < 2 or None in sc.values():
            print("  !! skip", q["query_id"], sc, flush=True)
            continue
        rows.append({"qid": q["query_id"], "par": sc["par"], "vector_purpose": sc["vector_purpose"]})
        print(f"  {q['query_id']}: par={sc['par']} vp={sc['vector_purpose']}", flush=True)

    if not rows:
        print("no rows"); return 1
    a = [r["par"] for r in rows]; b = [r["vector_purpose"] for r in rows]
    d = [x - y for x, y in zip(a, b)]
    summary = {"n": len(rows), "par_mean": round(statistics.mean(a), 3),
               "vector_purpose_mean": round(statistics.mean(b), 3),
               "win_tie_loss(par-vp)": [sum(x > 0 for x in d), sum(x == 0 for x in d),
                                        sum(x < 0 for x in d)]}
    try:
        from scipy.stats import wilcoxon
        if any(d):
            summary["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
    except Exception:
        pass
    print("\n[summary]", summary)
    out = ROOT / "eval/outputs/intent_judge_e5.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("[wrote]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
