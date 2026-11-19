# -*- coding: utf-8 -*-
"""
T3: 盲评管道（故事质量五维评分）

对「查询 + 故事」进行盲评（评分时**不标注**故事来源系统），按五维 1-5 打分：
  相关性 relevance / 连贯因果 coherence / 完整性 completeness /
  真实性 faithfulness / 噪音过滤 noise_filter
弃权类查询单独判定：故事是否体现「无相关记忆」。

用法（验证）:
    python -m eval.judge_story
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from eval.run_p_baseline import _load_dotenv

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是长期记忆系统评测员。你会收到「用户查询」、「查询目的」与一段由检索到的记忆整理而成的「故事」。

重要：检索系统会**基于查询目的做联想扩展**（沿因果链发散）。故事中与「查询目的」相关的扩展内容属于**合理联想**，不算噪音。

请从五个维度各打 1-5 分（整数）：
1. relevance 相关性：故事与「查询目的」的一致程度（5=高度一致，1=完全无关）
2. coherence 连贯/因果：故事是否有清晰的逻辑与因果关系、读起来是否连贯（5=很有逻辑，1=杂乱无章）
3. completeness 完整性：与「查询目的」相关的关键信息是否被覆盖（5=覆盖充分，1=几乎缺失）
4. faithfulness 真实性：故事是否忠实于记忆内容、没有编造（5=完全忠实，1=明显编造）
5. noise_filter 噪音过滤：故事中「与查询目的无关」的干扰信息量（与目的相关的合理联想不算噪音）（5=干净，1=大量与目的无关内容）

只输出 JSON：
{{"relevance": 0, "coherence": 0, "completeness": 0, "faithfulness": 0, "noise_filter": 0, "total": 0}}
total = 五维平均分，保留一位小数。"""),
    ("human", """用户查询：
{query}

查询目的：
{purpose}

故事：
{story}"""),
])

# ---- CROSS_VALID：联想合理性独立判定 ----

CROSS_VALID_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是记忆系统联想合理性评测员。你会收到「用户查询」、「查询目的」与一段故事。

检索系统沿因果链做联想扩展，故事中可能包含三类内容：
- 查询字面直接相关的（不算扩展）
- 与「查询目的」一致的合理联想（扩展，但合理）
- 与「查询目的」无关的纯噪音（不合理）

请标注扩展内容（不含查询字面相关部分）的构成：
{{"reasonable": 0.0, "noise": 0.0}}
- reasonable：合理联想占比（与查询目的一致的因果扩展，0-1）
- noise：纯噪音占比（与查询目的无关，0-1）
- reasonable + noise 应等于故事中扩展内容的整体比例（不强制求和，按相对比例给即可）

只输出 JSON。"""),
    ("human", """用户查询：
{query}

查询目的：
{purpose}

故事：
{story}"""),
])

ABSTENTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是长期记忆系统评测员。以下「用户查询」所问的信息在记忆库中并不存在，正确的系统行为是**不检索出实质记忆**（返回空或明确表示不知道）。

请判定「故事」是否体现了正确行为：
- 正确（refused=true, score=1）：故事为空、或明确声明「没有相关记忆/无法回答」
- 错误（refused=false, score=0）：故事包含实质性的记忆内容（无论是否与查询相关）——说明系统过度检索、把无关记忆当答案呈现

注意：只认「声明不知道/空」为正确；一段讲述其他内容的实质性故事一律视为错误。

只输出 JSON：{{"refused": true, "score": 1}}"""),
    ("human", """用户查询：
{query}

故事：
{story}"""),
])


def judge_story(llm: ChatOpenAI, query: str, story: str, purpose: str = "") -> dict:
    """五维评分（盲评：不携带系统身份信息；噪音判定基于查询目的）"""
    import logging
    chain = JUDGE_PROMPT | llm
    resp = chain.invoke({"query": query, "purpose": purpose, "story": story})
    try:
        return json.loads(resp.content)
    except json.JSONDecodeError:
        logging.warning(f"评分 JSON 解析失败: {resp.content[:200]}")
        return {"relevance": 0, "coherence": 0, "completeness": 0,
                "faithfulness": 0, "noise_filter": 0, "total": 0.0}


def judge_cross_valid(llm: ChatOpenAI, query: str, story: str, purpose: str = "") -> dict:
    """CROSS_VALID：扩展内容合理性标注（reasonable / noise）"""
    import logging
    chain = CROSS_VALID_PROMPT | llm
    resp = chain.invoke({"query": query, "purpose": purpose, "story": story})
    try:
        data = json.loads(resp.content)
        r = float(data.get("reasonable", 0.0))
        n = float(data.get("noise", 0.0))
        data["cross_valid"] = r / (r + n) if (r + n) > 0 else 0.0
        return data
    except (json.JSONDecodeError, ValueError):
        logging.warning(f"CROSS_VALID JSON 解析失败: {resp.content[:200]}")
        return {"reasonable": 0.0, "noise": 0.0, "cross_valid": 0.0}


def judge_abstention(llm: ChatOpenAI, query: str, story: str) -> dict:
    """弃权类单独判定"""
    import logging
    chain = ABSTENTION_PROMPT | llm
    resp = chain.invoke({"query": query, "story": story})
    try:
        return json.loads(resp.content)
    except json.JSONDecodeError:
        logging.warning(f"弃权判定 JSON 解析失败: {resp.content[:200]}")
        return {"refused": False, "score": 0}


def _verify() -> None:
    """验证：对 q01（P 候选故事）盲评 + q31（弃权）判定"""
    _load_dotenv(ROOT / ".env")
    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    # 用 T2 生成的 P 故事（这里直接跑一次 q01 带边故事）
    from eval.story_rank_adapter import story_rank_for_path, story_rank_for_texts
    from eval.run_p_baseline import build_retriever

    graph, retriever = build_retriever("data/LiFEMem.yaml")
    with open(ROOT / "eval/outputs/p_candidates.json", encoding="utf-8") as f:
        data = json.load(f)
    res = {r["query_id"]: r for r in data["results"]}

    r01 = res["q01"]
    out = story_rank_for_path(retriever.inference, r01["query"], r01["path"])
    story = out.get("story", "")
    print(f"[q01] 故事({len(story)}字)")
    score = judge_story(llm, r01["query"], story)
    print("  五维评分:", score)

    r31 = res["q31"]
    out31 = story_rank_for_path(retriever.inference, r31["query"], r31["path"])
    story31 = out31.get("story", "")
    print(f"[q31 弃权] 故事({len(story31)}字)")
    absj = judge_abstention(llm, r31["query"], story31)
    print("  弃权判定:", absj)


if __name__ == "__main__":
    _verify()
