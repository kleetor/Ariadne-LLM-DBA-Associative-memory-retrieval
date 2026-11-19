# -*- coding: utf-8 -*-
"""纯故事成对对比（不带查询，不打分）——LLM 直接判断哪个故事更好"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_OFFLINE"] = "1"

import yaml
from langchain_openai import ChatOpenAI


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


PAIR_PROMPT = """你是一位长期记忆系统评测员。下面是两个记忆系统从同一个用户记忆库中检索并整合生成的「记忆故事」——都是关于同一位用户经历的连贯叙述。

【故事 A】
{story_a}

【故事 B】
{story_b}

请在不考虑任何具体查询的情况下，仅作为「长期记忆」判断哪个故事质量更高。参考维度：
- 信息丰富度：覆盖用户多少维度的背景（人生经历、健康、情绪、人际、偏好、目标等）
- 真实性：内容是否具体可信、有细节支撑
- 组织连贯：叙述是否连贯、有逻辑
- 记忆价值：作为长期记忆是否更有助于理解用户

注意：不要把"故事更长"当作更好，以信息质量为准。若两者相当选择 tie。

只输出 JSON：{{"choice": "A" | "B" | "tie", "reason": "一句话理由"}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="纯故事成对对比（不带查询、纯选择 A/B/tie）")
    parser.add_argument("--queries", default="data/LiFEMem_queries_v3.yaml")
    parser.add_argument("--stories-a", default="eval/outputs/story_texts_v3.json")
    parser.add_argument("--name-a", default="P")
    parser.add_argument("--stories-b", default="eval/outputs/story_texts_graphiti.json")
    parser.add_argument("--name-b", default="Graphiti")
    parser.add_argument("--out", default="eval/outputs/story_pair_pg.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    stories_a = json.load(open(ROOT / args.stories_a, encoding="utf-8"))
    stories_b = json.load(open(ROOT / args.stories_b, encoding="utf-8"))
    qs = [q for q in yaml.safe_load(open(ROOT / args.queries, encoding="utf-8"))["queries"]]

    rng = random.Random(args.seed)
    results = []
    wins = {args.name_a: 0, args.name_b: 0, "tie": 0}
    for q in qs:
        qid = q["id"]
        if q.get("scenario") == "abstention":
            continue
        sa = stories_a[qid]["stories"][args.name_a]["story"]
        sb = stories_b[qid]["stories"][args.name_b]["story"]
        swap = rng.random() < 0.5
        story_a, story_b = (sb, sa) if swap else (sa, sb)
        prompt = PAIR_PROMPT.format(story_a=story_a, story_b=story_b)
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        choice, reason = None, text
        try:
            start = text.find("{")
            end = text.rfind("}")
            obj = json.loads(text[start:end + 1])
            choice = obj.get("choice")
            reason = obj.get("reason", text[:200])
        except Exception:
            for tok in ("A", "B", "tie"):
                if tok in text:
                    choice = tok
                    break
        winner = None
        if choice == "A":
            winner = args.name_b if swap else args.name_a
        elif choice == "B":
            winner = args.name_a if swap else args.name_b
        elif choice == "tie":
            winner = "tie"
        wins[winner] += 1
        results.append({
            "query_id": qid, "scenario": q["scenario"], "display_swap": swap,
            "choice": choice, "winner": winner, "reason": reason[:300],
            "story_len": {args.name_a: len(sa), args.name_b: len(sb)},
        })
        print(f"[PAIR] {qid} choice={choice} -> winner={winner}", flush=True)

    summary = {"total": len(results),
               f"{args.name_a}_wins": wins[args.name_a],
               f"{args.name_b}_wins": wins[args.name_b],
               "ties": wins["tie"],
               f"{args.name_a}_win_rate": round(wins[args.name_a] / len(results), 3),
               f"{args.name_b}_win_rate": round(wins[args.name_b] / len(results), 3)}
    out = {"summary": summary, "results": results}
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[PAIR] 汇总: {summary}")


if __name__ == "__main__":
    main()
