# -*- coding: utf-8 -*-
"""ALM vs MCP 的 Answer → Eval 实评分对比。

背景：`eval/test_alm_vs_mcp_e2e.py` 只算「检索结果里出现哪些 gold_fact」，属检索级代理
指标；平台真正的判分是 **Answer（统一回答模型读检索内容）→ Eval（判 CORRECT/WRONG）**。
本脚本把这后两段补上，给出可比较的分数。

两段均照抄 AML 公开契约（见 agent-memory-leaderboard/data/locomo-refined/pipeline.py）：
  · Answer：OPEN_ENDED_ANSWER_TEMPLATE（逐字复制，仅把群聊记忆填入 speaker_1 槽位）
  · Eval  ：ACCURACY_PROMPT（逐字复制，Inclusion + Non-contradiction）

⚠️ 两处与平台口径的差异，读数时必须带上：
  1. Judge 用的是本地 JUDGE_MODEL，不是平台固定判分模型（**判分仍有噪声**：曾出现两侧
     回答都含 2/2 gold 事实却双双判 WRONG）；
  2. 本地数据集**没有金标答案**，只能用人工标注的 gold_facts 列表当 gold_answer。
     官方判分器对"金标是列表"的情形要求**全覆盖且多余项算错**，因此该口径偏严，
     绝对分数会明显低于平台真值 —— 两侧同口径比较仍然有效。

Answer 模型借用官方同名字段 `ANSWER_MODEL`（回落 `.env` 的 `GPT_MODEL`）近似平台口径。
平台侧真正的 Answer 模型由其评测契约固定、不对参赛方公开，故这里只能近似；
绝对分数不可当作平台预期值，两侧同口径比较仍然有效。

为此另给一个不受判分器宽严影响的指标：**Answer 文本对 gold_facts 的覆盖率**
（回答模型在给定检索内容的前提下，实际说出了多少关键事实）。

用法：python -u -m eval.score_alm_vs_mcp_answer [--queries q1,q2]
"""

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.run_p_baseline import _load_dotenv
from eval.test_alm_vs_mcp_e2e import load_dataset, mcp_conversation, match_facts

USER_ID = "eval:local:groupchat:conv-0"
SESSION_ID = "eval:local:groupchat:sample:0"

# ---- 官方 Answer 模板（逐字复制自 data/locomo-refined/pipeline.py）----
OPEN_ENDED_ANSWER_TEMPLATE = """You are asked to answer a question based on your memories of a conversation.
<instructions>
1. Use only the provided memories. Prefer the memory that answers the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memory even if the wording differs.
4. When multiple answers are possible, list all supported answers, not just the first.
5. For counts or time intervals, enumerate carefully before answering.
6. Preserve specific names, titles, places, and labels from the memories. Use "Rob" not "a colleague", "Sweden" not "home country".
7. Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. Keep week-based expressions relative.
8. If memories conflict, prefer the most recent supported memory.
9. For list questions, include all required items and no extras.
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>
<memories>
Memories for user {{speaker_1_name}}:
{{speaker_1_memories}}
Memories for user {{speaker_2_name}}:
{{speaker_2_memories}}
</memories>
Question: {{question}}
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""

# ---- 官方判分模板（逐字复制自 data/locomo-refined/pipeline.py）----
# 注意：TIME / Lists of DISTINCT facts / Preference 三节不可省略。此前本脚本曾漏抄这三节，
# 导致"官方口径"判分实为宽松版，q6（gold 两项均命中）被误判 WRONG 即与此有关。
ACCURACY_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.
Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.
TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.
- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.
Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.
Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label":
```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""

_LABEL_RE = re.compile(r'"label"\s*:\s*"?(CORRECT|WRONG)"?', re.IGNORECASE)


def build_models():
    """Answer 用本地近似平台口径的模型；Judge 用本地判分模型。

    模型名优先取官方命名的 `ANSWER_MODEL`，其次 `.env` 的 `GPT_MODEL`，
    最后才回退 `OPENAI_MODEL`；base / key 沿用 `JUDGE_*`（与 GPT_MODEL 同一网关）。
    """
    from langchain_openai import ChatOpenAI

    answer_model = (os.environ.get("ANSWER_MODEL") or os.environ.get("GPT_MODEL")
                    or os.environ["OPENAI_MODEL"])
    answer = ChatOpenAI(
        model=answer_model,
        api_key=(os.environ.get("ANSWER_API_KEY") or os.environ.get("JUDGE_API_KEY")
                 or os.environ.get("OPENAI_API_KEY") or "not-needed"),
        base_url=(os.environ.get("ANSWER_API_BASE") or os.environ.get("JUDGE_API_BASE")
                  or os.environ.get("OPENAI_API_BASE")),
        temperature=0,
    )
    judge_model = os.environ.get("JUDGE_MODEL") or os.environ["OPENAI_MODEL"]
    judge = ChatOpenAI(
        model=judge_model,
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )
    return answer, judge, judge_model, answer_model


def render_answer_prompt(memories: str, question: str) -> str:
    """官方模板有 speaker_1 / speaker_2 两个记忆槽位；本地是群聊单块记忆，整体填入 slot 1，
    slot 2 留空（与官方 render_answer_prompt 里 speaker_2_memories 的缺省值 "" 一致）。"""
    return (OPEN_ENDED_ANSWER_TEMPLATE
            .replace("{{speaker_1_name}}", "the conversation participants")
            .replace("{{speaker_1_memories}}", memories or "(no memories)")
            .replace("{{speaker_2_name}}", "")
            .replace("{{speaker_2_memories}}", "")
            .replace("{{question}}", question))


def invoke(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    return (getattr(response, "content", "") or "").strip()


def judge_answer(judge, question: str, gold: str, generated: str) -> str:
    text = invoke(judge, ACCURACY_PROMPT.format(
        question=question, gold_answer=gold, generated_answer=generated))
    match = _LABEL_RE.search(text)
    return match.group(1).upper() if match else "UNPARSED"


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM vs MCP 的 Answer→Eval 实评分对比")
    parser.add_argument("--queries", default=None, help="只跑这些查询 id（逗号分隔）")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL", file=sys.stderr)
        return 2

    dialogs, queries = load_dataset()
    queries = [q for q in queries if q.get("gold_facts")]
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]

    answer_llm, judge_llm, judge_model, answer_model = build_models()
    print(f"# 可判定查询 {len(queries)} 条")
    print(f"# Answer 模型: {answer_model}"
          + ("（本地近似平台口径所用）" if answer_model == "gpt-4o-mini"
             else "（非 gpt-4o-mini；平台实际 Answer 模型未知，偏差更大，仅作方向参考）"))
    print(f"# Judge  模型: {judge_model}（⚠ 非平台固定判分模型）")
    print("# ⚠ gold 用人工标注的 gold_facts 列表，官方判分器对列表口径偏严（须全覆盖、多余项算错）")

    # ---- 装配两侧 ----
    mcp_dir, mcp_yaml = new_workspace("score_mcp_")
    mcp_graph, _, _, mcp_handlers = build_server(str(mcp_yaml))

    from alm.config import ALMConfig
    from alm.contract import AddMessage, AddRequest, SearchRequest
    from alm.engine import ALMEngine

    alm_cfg = ALMConfig.from_env()
    alm_cfg.data_dir = Path(tempfile.mkdtemp(prefix="score_alm_"))
    engine = ALMEngine(alm_cfg)
    print(f"# MCP 工作目录: {mcp_dir}\n# ALM 数据目录: {alm_cfg.data_dir}")

    print("\n===== ① 写入（两侧同源）=====")
    base_ms = int(time.mktime(time.strptime("2026-09-01 09:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    for idx, dialog in enumerate(dialogs):
        call_tool(mcp_handlers, "dba_add_conversation", {"conversation": mcp_conversation(dialog)})
        engine.add(AddRequest(
            request_id=f"{USER_ID}:{dialog['id']}", user_id=USER_ID, session_id=SESSION_ID,
            messages=[AddMessage(role=t["role"], content=f"[{t['speaker']}] {t['text']}",
                                 timestamp=base_ms + idx * 86_400_000 + i * 300_000)
                      for i, t in enumerate(dialog["turns"])],
        ))
        print(f"  {dialog['id']} 写入完成（MCP + ALM）")

    print("\n===== ② 检索 → Answer → Judge =====")
    rows: List[Dict[str, Any]] = []
    for item in queries:
        query, facts = item["query"], item["gold_facts"]
        gold_text = "、".join(facts)

        mcp_res = call_tool(mcp_handlers, "dba_query_memory", {"query": query})
        # 0927 审查 C1 消费者同步：query_memory 不再返回采纳 id 列表（story_nodes），无法按 id
        # 取采纳节点正文；原"MCP 记忆 = 故事 + 采纳节点正文"退化为只用故事正文（res["stories"]），
        # 不做任何节点级编造。
        mcp_memories = "\n".join(mcp_res.get("stories") or [])
        alm_items = engine.search(SearchRequest(query=query, user_id=USER_ID, top_k=100))
        alm_memories = "\n".join(it.get("content") or "" for it in alm_items)

        row: Dict[str, Any] = {"id": item["id"], "category": item["category"],
                               "query": query, "facts": facts}
        for tag, memories in (("mcp", mcp_memories), ("alm", alm_memories)):
            generated = invoke(answer_llm, render_answer_prompt(memories, query))
            hit = match_facts(generated, facts)
            label = judge_answer(judge_llm, query, gold_text, generated)
            row[tag] = {"answer": generated, "fact_hits": len(hit), "facts_total": len(facts),
                        "label": label, "memory_chars": len(memories)}
        rows.append(row)

        print(f"\n  [{row['id']}] {row['category']}｜{query}")
        print(f"    gold: {facts}")
        for tag, name in (("mcp", "MCP"), ("alm", "ALM")):
            d = row[tag]
            print(f"    {name} 判分={d['label']:<9} 事实覆盖={d['fact_hits']}/{d['facts_total']}"
                  f"（检索内容 {d['memory_chars']} 字符）")
            print(f"        回答: {d['answer'][:110]}")

    print("\n===== ③ 汇总 =====")
    total = len(rows)
    fact_total = sum(len(r["facts"]) for r in rows)
    correct = {t: sum(1 for r in rows if r[t]["label"] == "CORRECT") for t in ("mcp", "alm")}
    unparsed = {t: sum(1 for r in rows if r[t]["label"] == "UNPARSED") for t in ("mcp", "alm")}
    fact_hits = {t: sum(r[t]["fact_hits"] for r in rows) for t in ("mcp", "alm")}
    print(f"  可判定查询 {total} 条 / gold_facts 共 {fact_total} 项")
    for tag, name in (("mcp", "MCP"), ("alm", "ALM")):
        print(f"  {name}  判分准确率 {correct[tag]}/{total} = {correct[tag] / total:.1%}"
              f"   |  回答事实覆盖 {fact_hits[tag]}/{fact_total} = {fact_hits[tag] / fact_total:.1%}"
              + (f"   |  判分解析失败 {unparsed[tag]}" if unparsed[tag] else ""))

    print("\n  逐条（MCP | ALM，判分 / 回答事实覆盖）:")
    for r in rows:
        marks = []
        for tag in ("mcp", "alm"):
            marks.append(f"{'O' if r[tag]['label'] == 'CORRECT' else 'X'}"
                         f"{r[tag]['fact_hits']}/{r[tag]['facts_total']}")
        print(f"    {r['id']:<5} {marks[0]:<10} | {marks[1]:<10} {r['category']}")

    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
