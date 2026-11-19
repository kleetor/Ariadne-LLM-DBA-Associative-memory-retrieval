# -*- coding: utf-8 -*-
"""方案B：成对偏好盲评——LLM 从 P/RAG 两个故事中选更有记忆价值的"""
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


PREFERENCE_PROMPT = """你是一位记忆系统评测员。下面是一条用户对聊天机器人的陈述，以及两个候选记忆故事（由不同记忆系统从同一个用户记忆库中检索并整合生成）。

【用户陈述】
{statement}

【记忆故事 A】
{story_a}

【记忆故事 B】
{story_b}

这两个故事将被提供给主聊天 LLM，用于生成对该用户的回应。请从「对生成有记忆、有温度、贴合用户实际处境的回应最有帮助」的角度，评估哪个故事更有价值。

评估参考维度：
- 是否准确反映用户的当前状态与核心诉求
- 是否提供足够的背景记忆（相关经历、偏好、情感模式、健康状态、缓解方式、人际关系等）
- 是否包含对回应有实际价值的联想信息（而非与用户无关的内容）
- 信息密度与聚焦度（是否在聚焦的同时保留有用信息）

注意：不要把"故事更长"当作更好；以信息价值为准。如果两者价值相当，选择 tie。

只输出 JSON：{{"choice": "A" | "B" | "tie", "reason": "一句话理由"}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="方案B：成对偏好盲评")
    parser.add_argument("--queries", default="data/LiFEMem_queries_v3.yaml")
    parser.add_argument("--stories", default="eval/outputs/story_texts_v3.json")
    parser.add_argument("--scores", default="eval/outputs/story_scores_v3.json")
    parser.add_argument("--out", default="eval/outputs/preference_v3.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    stories = json.load(open(ROOT / args.stories, encoding="utf-8"))
    scores = {r["query_id"]: r for r in json.load(open(ROOT / args.scores, encoding="utf-8"))["results"]}
    qs = [q for q in yaml.safe_load(open(ROOT / args.queries, encoding="utf-8"))["queries"]]

    rng = random.Random(args.seed)
    results = []
    wins = {"p": 0, "rag": 0, "tie": 0}
    for q in qs:
        qid = q["id"]
        if q.get("scenario") == "abstention":
            continue  # 弃权类不参与偏好对比（两边都无有效记忆）
        st = stories[qid]["stories"]
        story_p = st["P"]["story"]
        story_rag = st["Naive RAG"]["story"]
        # 盲评：随机决定 A/B 谁是 P
        swap = rng.random() < 0.5
        story_a, story_b = (story_rag, story_p) if swap else (story_p, story_rag)
        prompt = PREFERENCE_PROMPT.format(statement=q["text"], story_a=story_a, story_b=story_b)
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        # 提取 JSON
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
        # 映射到真实系统
        winner = None
        if choice == "A":
            winner = "rag" if swap else "p"
        elif choice == "B":
            winner = "p" if swap else "rag"
        elif choice == "tie":
            winner = "tie"
        wins[winner] += 1
        results.append({
            "query_id": qid, "scenario": q["scenario"], "statement": q["text"],
            "display_swap": swap, "choice": choice, "winner": winner,
            "reason": reason[:300],
            "story_len": {"P": len(story_p), "RAG": len(story_rag)},
        })
        print(f"[PREF] {qid} choice={choice} -> winner={winner} ({len(story_p)}/{len(story_rag)}字)", flush=True)

    summary = {"total": len(results), "p_wins": wins["p"], "rag_wins": wins["rag"],
               "ties": wins["tie"], "p_win_rate": round(wins["p"] / len(results), 3),
               "rag_win_rate": round(wins["rag"] / len(results), 3)}
    out = {"summary": summary, "results": results}
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[PREF] 汇总: {summary}")


if __name__ == "__main__":
    main()
