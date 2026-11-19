# -*- coding: utf-8 -*-
"""方案：单故事独立评价（不带查询）——仅评故事本身质量"""
import argparse
import json
import os
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


STORY_ONLY_PROMPT = """你是一位长期记忆系统评测员。下面是一段从用户记忆中检索并整合生成的「记忆故事」——一段关于某用户经历的连贯叙述。

【记忆故事】
{story}

请在不考虑任何具体查询的情况下，仅根据故事本身的质量评分（每项 1-5 分，整数）：
- richness 信息丰富度：是否覆盖用户多个维度的背景（人生经历、健康状态、情绪模式、人际关系、偏好、目标等）
- authenticity 真实性：内容是否具体、可信、有细节支撑，无空洞或可疑表述
- coherence 组织连贯：叙述是否连贯、有逻辑、层次清晰
- memory_value 记忆价值：作为长期记忆，是否有助于后续理解用户、生成个性化回应
- structure 时序结构：是否按时间线/时序组织——如「高中→大学→工作」的清晰推进、显式时间锚点或时代词（中学时/毕业后等）、跨期顺序正确、无时间倒置或混乱。若故事只是并列罗列而无时序推进则给低分。

只输出 JSON：{{"richness": n, "authenticity": n, "coherence": n, "memory_value": n, "structure": n, "comment": "一句话点评"}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="单故事独立评价（不带查询）")
    parser.add_argument("--stories", default="eval/outputs/story_texts_v3.json")
    parser.add_argument("--out", default="eval/outputs/story_alone_v3.json")
    parser.add_argument("--only-queries", default="")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    stories = json.load(open(ROOT / args.stories, encoding="utf-8"))
    only = [q.strip() for q in args.only_queries.split(",") if q.strip()]

    results = []
    DIMS = ["richness", "authenticity", "coherence", "memory_value", "structure"]
    agg = {"p": {d: [] for d in DIMS}, "rag": {d: [] for d in DIMS}}
    for qid, entry in stories.items():
        if only and qid not in only:
            continue
        for sys_name, key in (("p", "P"), ("rag", "Naive RAG")):
            story = entry["stories"].get(key, {}).get("story", "")
            if not story:
                continue
            prompt = STORY_ONLY_PROMPT.format(story=story)
            resp = llm.invoke(prompt)
            text = resp.content.strip()
            scores = {}
            try:
                start = text.find("{")
                end = text.rfind("}")
                obj = json.loads(text[start:end + 1])
                scores = {d: int(obj.get(d, 0)) for d in DIMS}
                comment = obj.get("comment", "")
            except Exception:
                comment = text[:200]
            for d in DIMS:
                agg[sys_name][d].append(scores.get(d, 0))
            results.append({"query_id": qid, "system": sys_name, "scores": scores,
                            "comment": comment[:200], "story_len": len(story)})
            print(f"[ALONE] {qid} {sys_name} {scores}", flush=True)

    summary = {"systems": {}}
    for sys_name in ("p", "rag"):
        summary["systems"][sys_name] = {
            d: round(sum(agg[sys_name][d]) / len(agg[sys_name][d]), 2) for d in DIMS
        }
    out = {"summary": summary, "results": results}
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ALONE] 汇总: {summary}")


if __name__ == "__main__":
    main()
