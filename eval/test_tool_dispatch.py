# -*- coding: utf-8 -*-
"""
工具分流观察：让 LLM 在 dba_query_memory 与 dba_temporal_lookup 之间做路由决策。

观察 LLM 对问题的判断（是否会判断对"时间→事件"该用 temporal_lookup，其余用 query_memory），
以及它的工具调用（选了哪个工具、给了什么参数）。

用法:
    python -m eval.test_tool_dispatch
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from eval.run_p_baseline import _load_dotenv
from dba_pipeline.mcp_server import TOOL_SCHEMAS


def main() -> None:
    _load_dotenv(ROOT / ".env")

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
    )

    # 从 MCP schema 取两个路由工具
    wanted = {"dba_query_memory", "dba_temporal_lookup"}
    tool_dicts = []
    for ts in TOOL_SCHEMAS:
        if ts["name"] in wanted:
            tool_dicts.append({
                "type": "function",
                "function": {
                    "name": ts["name"],
                    "description": ts["description"],
                    "parameters": ts["inputSchema"],
                },
            })

    chain = llm.bind_tools(tool_dicts)

    queries = [
        ("TB-时间→事件", "昨天我做了什么？"),
        ("TB-时间→事件", "上周用户经历了什么累人的事？"),
        ("TB-时间→事件", "高中时期都发生了什么事？"),
        ("TA-事件→时间", "用户是什么时候开始学吉他的？"),
        ("TD-跨期", "从高中到工作，用户是怎么一路走过来的？"),
        ("因果", "为什么用户最近心情不好？"),
        ("事实", "用户喜欢喝什么咖啡？"),
    ]

    print("=" * 72)
    print("工具分流观察（期望：TB→dba_temporal_lookup；TA/TD/因果/事实→dba_query_memory）")
    print("=" * 72)
    for tag, q in queries:
        resp = chain.invoke([SystemMessage(
            content="你是记忆检索路由助手。根据用户问题，从给定工具中选择【最合适的一个】调用。"
                    "只选一个工具。"
        ), HumanMessage(content=q)])
        tcs = getattr(resp, "tool_calls", None) or []
        if not tcs:
            pick, args = "（未调用工具）", {} 
        else:
            tc = tcs[0]
            pick = tc.get("name", "")
            args = tc.get("args", {})
        print(f"[{tag}] {q}")
        print(f"    LLM 判断 → 工具: {pick}")
        if args:
            print(f"      参数: {args}")
        print()


if __name__ == "__main__":
    main()
