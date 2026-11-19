# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
情感陪伴场景 e2e 冒烟测试：检索 → 故事化 → 真实 LLM 回复 → 陪伴判分 → 3 次取平均 → 汇总

验证重设计后的完整链路（§4.5）跑得通：
  query → PAR 检索(候选) → StoryRank(故事) → 真实回复LLM(companion) → 陪伴judge → 3次均值 → 汇总

数据：data/dba_eval/smoke_companion.yaml（小陪伴图，~12 节点/11 边）
查询：3 条陈述式（工作压力/健康/情绪）
用法:
    python -m eval.smoke_run_e2e
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 重型依赖（mem0/fastembed/qdrant 等）在项目隔离依赖目录 pylibs，须优先加入路径
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import yaml  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from eval.run_p_baseline import _load_dotenv, build_retriever, build_path  # noqa: E402

GRAPH = "data/dba_eval/smoke_companion.yaml"
QUERIES = [
    {"id": "q1", "query": "最近项目排得太满了，天天加班到十一二点，感觉整个人都被掏空了……", "scenario": "work_pressure"},
    {"id": "q2", "query": "最近总是睡不好，颈椎也难受，我是不是该抽空去体检一下？", "scenario": "health"},
    {"id": "q3", "query": "感觉压力好大，不知道怎么熬过这段日子……", "scenario": "emotion"},
]
REPLY_ROUNDS = 3  # 3 次回复取平均

# ---- 真实回复 LLM（陪伴场景，temperature>0 以产生差异）----
REPLY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个温暖、共情的情感陪伴助手。根据对用户的记忆，温柔地回应 ta 此刻的情绪与处境，"
               "让对方感觉被理解和陪伴。语气自然、像朋友，不要机械列举信息，不要编造记忆里没有的事。"),
    ("human", "用户的倾诉：{query}\n\n【关于用户的记忆】\n{story}"),
])

# ---- 陪伴场景判分（temperature=0，5 维各 1-5）----
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是情感陪伴场景的评测员。请评估【回复】作为陪伴回应的质量，5 个维度各打 1-5 分：\n"
               "1. 共情贴合：是否理解并贴合用户当下的情绪/处境\n"
               "2. 回应自然度：语气是否自然、像朋友而非机械拼贴\n"
               "3. 陪伴价值：是否提供了情感支持、被理解感、有温度\n"
               "4. 真实性：是否基于真实记忆、没有编造（可参考提供的记忆）\n"
               "5. 聚焦无冗余：是否贴合当前陪伴诉求、不倾倒无关信息\n"
               "只输出 JSON：{{\"共情贴合\":0,\"回应自然度\":0,\"陪伴价值\":0,\"真实性\":0,\"聚焦\":0,\"total\":0}}"),
    ("human", "用户的倾诉：{query}\n\n【记忆参考（仅用于判断真实性）】\n{story}\n\n回复：{reply}"),
])


def build_reply_llm():
    """回复 LLM：默认用 OPENAI_*，可用 REPLY_* 覆盖（换模型以解耦）。"""
    return ChatOpenAI(
        model=os.environ.get("REPLY_MODEL") or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("REPLY_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("REPLY_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        temperature=0.9,
        max_tokens=1024,
        extra_body={"thinking": {"type": "disabled"}},
    )


def build_judge_llm():
    """判分 judge：默认用 OPENAI_*，可用 JUDGE_* 覆盖。
    注意：judge 与回复模型同源会放大自偏好（天花板/无区分度），建议用独立模型消除。"""
    return ChatOpenAI(
        model=os.environ.get("JUDGE_MODEL") or os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )


def gen_reply(reply_llm, query: str, story: str) -> str:
    chain = REPLY_PROMPT | reply_llm
    resp = chain.invoke({"query": query, "story": story})
    return resp.content


def judge_reply(judge_llm, query: str, story: str, reply: str) -> dict:
    chain = JUDGE_PROMPT | judge_llm
    resp = chain.invoke({"query": query, "story": story, "reply": reply})
    keys = ["共情贴合", "回应自然度", "陪伴价值", "真实性", "聚焦"]
    try:
        data = json.loads(resp.content)
        # total = 5 维均值（落回 1-5 分，避免模型把 total 当和）
        vals = [float(data.get(k, 0)) for k in keys]
        data["total"] = round(sum(vals) / len(vals), 2)
        return data
    except json.JSONDecodeError:
        return {**{k: 0 for k in keys}, "total": 0.0}


def main() -> None:
    _load_dotenv(ROOT / ".env")
    graph, retriever = build_retriever(GRAPH)
    reply_llm = build_reply_llm()
    judge_llm = build_judge_llm()

    all_rows = []
    for q in QUERIES:
        qid, query = q["id"], q["query"]
        rounds = []
        stories = set()
        for i in range(REPLY_ROUNDS):
            # 1) 检索（PAR）
            res = retriever.retrieve(query)
            # 2) 故事化（StoryRank：连通性粗筛 → 路径 → 故事）
            path = build_path(retriever, res)
            story = retriever.inference.story_rank(query, path)["story"]
            if not story:
                story = " ".join(c["content"] for c in res.get("peak_memories", []))
            stories.add(story)
            # 3) 真实回复 LLM（每次 temperature 采样不同）
            reply = gen_reply(reply_llm, query, story)
            # 4) 陪伴判分
            sc = judge_reply(judge_llm, query, story, reply)
            sc["round"] = i + 1
            sc["reply"] = reply
            rounds.append(sc)
        # 5) 三次取平均
        mean_total = sum(r["total"] for r in rounds) / len(rounds)
        row = {"id": qid, "scenario": q["scenario"], "query": query,
               "rounds": rounds, "mean_total": round(mean_total, 2),
               "distinct_stories": len(stories)}
        all_rows.append(row)
        print(f"\n=== {qid} [{q['scenario']}] {query}")
        for r in rounds:
            print(f"  r{r['round']} total={r['total']}  回复: {r['reply'][:60]!r}")

    print("\n" + "=" * 60)
    print("三条查询标准分（3 次取平均）:")
    for row in all_rows:
        print(f"  {row['id']} [{row['scenario']}] mean_total={row['mean_total']}  distinct_stories={row['distinct_stories']}")
    agg = sum(r["mean_total"] for r in all_rows) / len(all_rows)
    print(f"  总均分 = {agg:.2f}")

    out = ROOT / "eval" / "outputs" / "smoke_e2e_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"aggregate": agg, "queries": all_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写: {out}")


if __name__ == "__main__":
    main()
