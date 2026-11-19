# -*- coding: utf-8 -*-
"""
任务 A 效果验证：「决策与取舍」抽取增强（MCP 全链路 · 真实 LLM）

背景：抽取只留"事实"节点，原文按有损压缩处理，**"当初为什么做这个选择"最先丢**——
REASON 的定义原本只管"导致状态/行为的原因"，不管"选择依据"；而且"只抽事实 + 每条
独立可读"两条原则会把"权衡过程"当噪音滤掉。本测试验证新增的补充规则
（`NODE_CHOICE_EXTRA`，仅 MCP 侧开启）是否真的把这几类信息抽了出来。

对比设计（抽样口径见计划 §8.3）：
    OFF（基线，ALM 侧行为） vs ON（MCP 侧），各跑 N 次，比较**中位数**——
    抽取依赖 LLM 且提示词很长，单次输出不可归因。

关注四类信息（本语料里各有一条，可程序化粗判）：
    ① 被放弃的选项 ② 放弃它的原因 ③ 最终选择 ④ 选择依据

用法:
    python -m eval.test_choice_extraction
"""

import os
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.run_p_baseline import _load_dotenv

REPEAT = 3

# 语料：含明确的两个被放弃选项 + 各自原因 + 最终折中选择 + 依据
CONVERSATION = "\n".join([
    "[用户] 装修这事我纠结了很久，一开始想全包给装修公司。",
    "[助手] 全包不是更省心吗？",
    "[用户] 省心是省心，但报价比半包贵了快一倍，预算不够，所以最后还是放弃了。",
    "[用户] 半包也考虑过，就是得自己盯工地，我实在抽不出那个时间。",
    "[助手] 那最后怎么定的？",
    "[用户] 折中了一下，主材自己买、施工包给师傅，钱省了一半，时间也没多花多少。",
])

# 程序化粗判用的关键词（只为得到可跨轮次比较的数字；结论仍需人工核对下面的打印）
KEYS_ABANDONED = ("全包", "半包")          # 被放弃的选项
KEYS_RATIONALE = ("预算", "成本", "贵", "钱")  # 依据之一：成本
KEYS_TIME = ("时间", "工期", "盯")          # 依据之二：时间


def _metrics(graph):
    nodes = [(nid, nd) for nid, nd in graph.graph.nodes(data=True)]
    contents = [(nid, nd.get("content") or "") for nid, nd in nodes]
    reason = [nid for nid, nd in nodes if "REASON" in str(nd.get("node_type")).upper()]
    isolated = [
        nid for nid in graph.graph.nodes()
        if graph.graph.in_degree(nid) + graph.graph.out_degree(nid) == 0
    ]
    return {
        "nodes": len(nodes),
        "reason": len(reason),
        "abandoned": sum(1 for _, c in contents if any(k in c for k in KEYS_ABANDONED)),
        "rationale": sum(1 for _, c in contents if any(k in c for k in KEYS_RATIONALE)),
        "time": sum(1 for _, c in contents if any(k in c for k in KEYS_TIME)),
        "isolated": len(isolated),
    }


def _run_once(enable_choice: bool, tag: str):
    _, yaml_path = new_workspace(f"choice_{tag}_")
    graph, _, _, handlers = build_server(str(yaml_path), enable_choice=enable_choice)
    res = call_tool(handlers, "dba_add_conversation", {"conversation": CONVERSATION})
    metrics = _metrics(graph)
    detail = [
        (nid, str(nd.get("node_type")).upper().split(".")[-1], nd.get("content"))
        for nid, nd in graph.graph.nodes(data=True)
    ]
    return metrics, detail, res


def main() -> None:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        sys.exit(2)

    print(f"# 语料（{len(CONVERSATION.splitlines())} 轮）:")
    for line in CONVERSATION.splitlines():
        print(f"    {line}")

    results = {}
    for label, flag in (("OFF", False), ("ON", True)):
        runs = []
        print(f"\n===== {label}（enable_choice_extraction={flag}）=====")
        for i in range(REPEAT):
            metrics, detail, res = _run_once(flag, label.lower())
            runs.append(metrics)
            print(f"  第{i + 1}次 {metrics}  maintained={res.get('maintained')}")
        median = {k: statistics.median([r[k] for r in runs]) for k in runs[0]}
        results[label] = median
        print(f"  中位数: {median}")
        print(f"  末次节点明细:")
        for nid, nt, content in detail:
            print(f"    {nid}  [{nt}]  {content!r}")

    print("\n" + "=" * 60)
    print("指标中位数对比（OFF → ON）:")
    improved = []
    for k in results["OFF"]:
        a, b = results["OFF"][k], results["ON"][k]
        mark = "↑" if b > a else ("↓" if b < a else "=")
        print(f"  {k:10}: {a} → {b}  {mark}")
        if k in ("reason", "abandoned", "rationale", "time") and b > a:
            improved.append(k)
    print()
    if improved:
        print(f"任务 A 生效 [OK]：以下指标的抽样中位数上升 {improved}。")
        print("（仍需人工核对上面的节点明细，确认抽出的内容是"
              "「明确说过的备选与依据」，而不是编造的动机。）")
    else:
        print("任务 A 未观察到增益 [NG]：补充规则没有带来更多 REASON / 备选 / 依据节点。")
        print("建议：核对上方明细确认是否规则被长提示词淹没，或语料本身没踩到规则边界。")
        sys.exit(1)


if __name__ == "__main__":
    main()
