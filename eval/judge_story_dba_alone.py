# -*- coding: utf-8 -*-
"""
纯故事对比评测：gold 子图 vs DBA 图（激进 v4）

对 dba_story_cmp.json 中两图的 StoryRank 故事做「不带查询」的独立评价
（richness / authenticity / coherence / memory_value 四维，1-5 分），
对比两图故事的记忆价值。

用法:
    python -m eval.judge_story_dba_alone
    python -m eval.judge_story_dba_alone --stories eval/outputs/dba_story_cmp.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_openai import ChatOpenAI

from eval.run_p_baseline import _load_dotenv


STORY_ONLY_PROMPT = """你是一位长期记忆系统评测员。下面是一段从用户记忆中检索并整合生成的「记忆故事」——一段关于某用户经历的连贯叙述。

【记忆故事】
{story}

请在不考虑任何具体查询的情况下，仅根据故事本身的质量评分（每项 1-5 分，整数）：
- richness 信息丰富度：是否覆盖用户多个维度的背景（人生经历、健康状态、情绪模式、人际关系、偏好、目标等）
- authenticity 真实性：内容是否具体、可信、有细节支撑，无空洞或可疑表述
- coherence 组织连贯：叙述是否连贯、有逻辑、层次清晰
- memory_value 记忆价值：作为长期记忆，是否有助于后续理解用户、生成个性化回应

只输出 JSON：{{"richness": n, "authenticity": n, "coherence": n, "memory_value": n, "comment": "一句话点评"}}"""


def _score(llm, story: str) -> dict:
    resp = llm.invoke(STORY_ONLY_PROMPT.format(story=story))
    text = resp.content.strip()
    try:
        start = text.find("{")
        end = text.rfind("}")
        obj = json.loads(text[start:end + 1])
        return {d: int(obj.get(d, 0)) for d in ["richness", "authenticity", "coherence", "memory_value"]}
    except (json.JSONDecodeError, ValueError):
        return {"richness": 0, "authenticity": 0, "coherence": 0, "memory_value": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="纯故事对比：gold vs DBA 图")
    parser.add_argument("--stories", default="eval/outputs/dba_story_cmp.json")
    parser.add_argument("--out", default="eval/outputs/dba_story_alone.json")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=2048,
        extra_body={"thinking": {"type": "disabled"}},  # 关 thinking，避免 content 空/截断
    )

    data = json.loads((ROOT / args.stories).read_text(encoding="utf-8"))
    DIMS = ["richness", "authenticity", "coherence", "memory_value"]
    agg = {"gold": {d: [] for d in DIMS}, "dba": {d: [] for d in DIMS}}
    rows = []

    for r in data["results"]:
        qid = r["query_id"]
        sg, sd = r["gold"]["story"], r["dba"]["story"]
        sc_g = _score(llm, sg)
        sc_d = _score(llm, sd)
        for sys_name, sc in (("gold", sc_g), ("dba", sc_d)):
            for d in DIMS:
                agg[sys_name][d].append(sc[d])
        rows.append({"query_id": qid, "category": r["category"],
                     "gold": sc_g, "dba": sc_d})
        tg = sum(sc_g.values()) / len(DIMS)
        td = sum(sc_d.values()) / len(DIMS)
        print(f"[ALONE] {qid} gold={tg:.2f} dba={td:.2f}")

    def _avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    summary = {}
    for sys_name in ("gold", "dba"):
        summary[sys_name] = {d: _avg(agg[sys_name][d]) for d in DIMS}
        summary[sys_name]["total"] = round(
            sum(summary[sys_name][d] for d in DIMS) / len(DIMS), 3)
    summary["win"] = {"gold": 0, "dba": 0, "tie": 0}
    for r in rows:
        tg = sum(r["gold"].values()) / len(DIMS)
        td = sum(r["dba"].values()) / len(DIMS)
        if tg > td:
            summary["win"]["gold"] += 1
        elif td > tg:
            summary["win"]["dba"] += 1
        else:
            summary["win"]["tie"] += 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 纯故事对比（无查询）gold vs DBA 图 =====")
    for d in DIMS:
        print(f"  {d:14s}: gold={summary['gold'][d]}  dba={summary['dba'][d]}")
    print(f"  总分: gold={summary['gold']['total']}  dba={summary['dba']['total']}")
    print(f"  胜负: gold胜{summary['win']['gold']} / dba胜{summary['win']['dba']} / 平{summary['win']['tie']}")
    print(f"  输出: {out_path}")


if __name__ == "__main__":
    main()
