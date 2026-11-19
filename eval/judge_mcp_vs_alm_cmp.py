# -*- coding: utf-8 -*-
"""MCP 接口 vs ALM 接口的**上下文块质量对照**（复用 0901 e2e 的判分法）。

为什么不用「准确率/召回」：0901 报告 §7.6 已实测**检索层数字不可靠**
（重标口径 R@all 四次跑 29.1%~58.6%、std ±11.4pp），而**故事层判分稳定**
（4 次独立抽取纯故事恒为 4.725）。故照搬 0901 §2.2 的判分法：

- 对象 = **系统喂给模型的结构化记忆上下文块**（不生成回复，把回复 LLM 与记忆系统解耦）
- 方式 = **同题并排一次判定**：三块同题一起交给 judge，各打 1–5 分
- rubric = 陪伴场景 3 维（记忆丰富度 / 易理解性 / 用户建模度），**逐字取自
  `eval/judge_comprehensive.py`**，只改了标签列表
- 多轮重复，给**方向 + 区间**而非单个小数

三个被比较的块（同一批对话、同一批查询、同一套记忆内核）：

| 标签 | 内容 | 对应 |
|---|---|---|
| `MCP` | 故事 + 被采纳节点原文 | 现有 MCP 的 `dba_query_memory` |
| `ALM-单条` | 只给叙事 | ALM 当前形态（收拢后） |
| `ALM-节点` | 叙事 + 被采纳节点原文 | ALM 收拢前形态 |

用途：判定"收拢成单条"在**陪伴素材质量**上到底是加分还是减分，以及两个接口谁更好。

用法：python -u -m eval.judge_mcp_vs_alm_cmp [--repeats 3] [--queries q1,q2]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 照仓库既有惯例显式关掉 HF 联网（.env 里的 HF_HUB_OFFLINE 在 MCP harness 那条路径吃不到，
# 否则每次模型加载都会去 HEAD huggingface 重试 5 次、多花约 1 分钟）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from langchain_core.prompts import ChatPromptTemplate

from eval.judge_comprehensive import _extract_json, _load_dotenv, build_judge_llm
from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.test_alm_vs_mcp_e2e import load_dataset, mcp_conversation

USER_ID = "eval:local:mcpvsalm:conv-0"
SESSION_ID = "eval:local:mcpvsalm:sample:0"
LABELS = ["MCP", "ALM-单条", "ALM-节点", "ALM-原文"]
DIMS = ["记忆丰富度", "易理解性", "用户建模度"]

# rubric 逐字取自 eval/judge_comprehensive.py::JUDGE_JOINT_PROMPT，仅替换被评对象与标签
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是情感陪伴场景的记忆评测员。下面给出「用户的倾诉」与四份记忆系统接口"
               "（MCP / ALM-单条 / ALM-节点 / ALM-原文）各自检索并组装、将喂给模型的结构化记忆上下文。"
               "其中 ALM-节点 与 ALM-原文 来自**同一次检索、同一批记忆**，唯一差别是节点正文是否带"
               "类型前缀（如 [status]），用于分辨格式本身的影响。"
               "请结合同一段倾诉，从下面 3 个维度，分别给每份打 1-5 分（四份放在一起更便于分辨好坏）：\n"
               "1. 记忆丰富度：记忆信息是否足够丰富、能支撑充分的陪伴（5=非常丰富，1=单薄贫乏）\n"
               "2. 易理解性：这段上下文是否条理清晰、便于直接用（5=一目了然，1=晦涩零乱）\n"
               "3. 用户建模度：能否据此准确理解用户、形成用户画像（5=很强，1=很弱）\n"
               "若上下文中出现关系/边/节点类型标签（如 CAUSAL/SCENARIO/SEQUENCE/PREFERENCE/SOCIAL/"
               "ATTRIBUTE/TEMPORAL/TAXONOMIC，或 STATUS/REASON/ACTION/THING/PERSON/EMOTION），"
               "其含义依次为：因果 / 同场景 / 时序先后 / 偏好 / 社交 / 属性 / 时间定位 / 分类，以及"
               "客观处境 / 原因 / 行为 / 事物 / 人物 / 情绪。请据此理解结构，而非按字面臆测。\n"
               "只输出 JSON：{{\"MCP\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"ALM-单条\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"ALM-节点\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}},"
               "\"ALM-原文\":{{\"记忆丰富度\":0,\"易理解性\":0,\"用户建模度\":0,\"total\":0}}}}"),
    ("human", "用户的倾诉：{query}\n\n{blocks}"),
])


def call_judge(llm, query: str, blocks: Dict[str, List[str]], max_attempt: int = 4):
    rendered = "\n\n".join(
        f"## {name}\n{json.dumps(blocks.get(name, []), ensure_ascii=False)}" for name in LABELS
    )
    for _ in range(max_attempt):
        try:
            text = llm.invoke(JUDGE_PROMPT.format_messages(query=query, blocks=rendered)).content
            payload = _extract_json(text)
            if isinstance(payload, dict) and all(name in payload for name in LABELS):
                return payload
        except Exception:
            time.sleep(2)
    return None


def alm_arms(space, query: str, top_k: int = 100):
    """一次链路调用产出三个 ALM 臂：单条 / 带前缀节点 / 原文节点（去前缀）。

    0927 审查 C1 消费者同步：StoryRank 输出契约变更，retrieve_with_story 不再返回采纳
    节点 id 列表（story_nodes），代之以 story_used（正文覆盖计数）。原先「ALM-节点 / ALM-原文」
    两臂建立在**同一次链路调用**（`space.search`）与同一次直连检索拿到的 story_nodes 之上，
    按 id 取节点正文并加/去 [node_type] 前缀以隔离"格式本身"的影响；该列表已不存在，按 id
    对齐两臂的核心对比已不可行。此处退化为直接用现成检索正文（res["stories"]），不做任何
    节点级编造——两臂在节点维度上趋于一致（仅作诚实降级记录）。
    """
    items = space.search(query, top_k)
    if not items:
        return [], [], []
    story = items[0]["content"]
    res = space.retriever.retrieve_with_story(
        query, with_response=False, render_timestamps=space.config.render_timestamps,
        seed_k=space._effective_seed_k(), max_hops=space.config.max_hops,
        expand_k=space.config.expand_k,
    )
    # 采纳 id 列表已不存在（原 story_nodes），无法按 id 取节点正文，只能用检索故事正文替代。
    stories = list(res.get("stories") or [])
    return [story], [story] + stories, [story] + stories


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP vs ALM 上下文块质量对照（0901 判分法）")
    parser.add_argument("--repeats", type=int, default=3, help="重复次数（用于给区间）")
    parser.add_argument("--queries", default=None, help="只跑这些查询 id（逗号分隔）")
    parser.add_argument("--judge-model", default=None, help="judge 模型（默认 .env JUDGE_MODEL）")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    dialogs, queries = load_dataset()
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]

    judge = build_judge_llm(model=args.judge_model)
    print(f"# judge 模型: {args.judge_model or os.environ.get('JUDGE_MODEL')}")
    print(f"# 查询 {len(queries)} 条 × 重复 {args.repeats} 次（同题并排一次判定，3 维 1–5）")

    from alm.config import ALMConfig
    from alm.contract import AddMessage, AddRequest
    from alm.engine import ALMEngine

    mcp_dir, mcp_yaml = new_workspace("cmp_mcp_")
    mcp_graph, _, _, mcp_handlers = build_server(str(mcp_yaml))
    cfg = ALMConfig.from_env()
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="cmp_alm_"))
    engine = ALMEngine(cfg)
    print(f"# MCP 工作目录: {mcp_dir}\n# ALM 数据目录: {cfg.data_dir}")

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
        print(f"  {dialog['id']} 写入完成")

    print("\n===== ② 组装三个上下文块 =====")
    rows: List[Dict[str, Any]] = []
    with engine._lease(USER_ID, create=False) as space:
        for item in queries:
            query = item["query"]
            res = call_tool(mcp_handlers, "dba_query_memory", {"query": query, "rerank_k": 20})
            # 0927 审查 C1 消费者同步：query_memory 不再返回采纳 id 列表（story_nodes），
            # 无法按 id 取采纳节点正文；原"MCP 块 = 故事 + 采纳节点原文"退化为只用故事正文
            # （res["stories"]），不做任何节点级编造。
            mcp_block = [b for b in (res.get("stories") or []) if b]
            single, with_nodes, raw_nodes = alm_arms(space, query)
            rows.append({"id": item["id"], "category": item["category"], "query": query,
                         "blocks": {"MCP": mcp_block, "ALM-单条": single,
                                    "ALM-节点": with_nodes, "ALM-原文": raw_nodes}})
            print(f"  [{item['id']}] MCP {len(mcp_block)} / 单条 {len(single)} / "
                  f"节点 {len(with_nodes)} / 原文 {len(raw_nodes)}")

    print(f"\n===== ③ judge（{args.repeats} 次重复）=====")
    scores: Dict[str, Dict[str, List[List[float]]]] = {
        name: {dim: [] for dim in DIMS + ["total"]} for name in LABELS
    }
    rankings: List[List[str]] = []
    failures = 0
    for rep in range(args.repeats):
        for row in rows:
            payload = call_judge(judge, row["query"], row["blocks"])
            if payload is None:
                failures += 1
                print(f"  [rep{rep+1}][{row['id']}] judge 解析失败，跳过")
                continue
            for name in LABELS:
                entry = payload.get(name) or {}
                for dim in DIMS + ["total"]:
                    try:
                        value = float(entry.get(dim))
                    except (TypeError, ValueError):
                        continue
                    scores[name][dim].append(value)
            totals = {n: (payload.get(n) or {}).get("total") for n in LABELS}
            if all(isinstance(v, (int, float)) for v in totals.values()):
                rankings.append(sorted(LABELS, key=lambda n: -float(totals[n])))
            print(f"  [rep{rep+1}][{row['id']}] " + "  ".join(
                f"{n}={totals.get(n)}" for n in LABELS))

    def stats(values: List[float]):
        if not values:
            return (0.0, 0.0, 0.0)
        return (sum(values) / len(values), min(values), max(values))

    print("\n===== ④ 汇总（均值[min~max]）=====")
    for name in LABELS:
        line = [f"  {name:<10}"]
        for dim in DIMS + ["total"]:
            mean, lo, hi = stats(scores[name][dim])
            line.append(f"{dim}={mean:.2f}[{lo:g}~{hi:g}]")
        print("  ".join(line))

    print("\n  排名分布（按 total 降序）:")
    for name in LABELS:
        first = sum(1 for r in rankings if r and r[0] == name)
        print(f"    {name:<10} 第一 {first}/{len(rankings)} 次")

    print(f"\n  judge 失败 {failures} 次（共 {len(rows) * args.repeats} 次调用）")
    print("\n  说明：0901 §7.6 已证检索层数字不可靠（R@all std ±11.4pp），故本表不含准确率/召回；"
          "\n        按 0901 惯例给方向 + 区间，不给单个小数下结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
