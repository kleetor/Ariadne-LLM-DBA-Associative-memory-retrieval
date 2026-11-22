# -*- coding: utf-8 -*-
"""ALM 与 MCP 双链路 e2e 对比（群聊场景：时序 + 复杂人称 + 日常 + 工作）

目的：判断两条链路在同一份数据上的表现是否相近，并量化差异。

两侧装配（同一进程、同一套 LLM 与 embedding，仅链路不同）：
  · MCP：eval.mcp_harness.build_server → dba_add_conversation / dba_query_memory
  · ALM：alm.engine.ALMEngine   → POST /add、/search 的同等内部调用
         （engine.add / engine.search，与 HTTP 端点同一套逻辑，省掉网络层）

可比指标：每条查询应出现的 gold_facts（去空白与标点后判子串）命中数。
公平性检查：ALM 返回的文本量远大于 MCP（叙述 + 多条原子节点 vs 一两段故事），
  命中更多可能只是"信息量更大"。因此额外报告 ALM 在 top-1/3/5/10 处的累计命中率，
  用于判断优势是来自**排序靠前**还是**条数堆出来**的。
附带画像：图规模、归属词频、TEMPORAL 边、时间锚点，以及 ALM 检索文本里
「助手」「用户」的出现次数（用于观察 role 映射带来的称法问题）。

数据：data/dba_eval/groupchat_mixed_dialogs.yaml / _queries.yaml
用法：python -m eval.test_alm_vs_mcp_e2e
"""

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.run_p_baseline import _load_dotenv

DATA_DIR = ROOT / "data" / "dba_eval"
USER_ID = "eval:local:groupchat:conv-0"
SESSION_ID = "eval:local:groupchat:sample:0"

# 说话人（含全名，用于观察"同一人两种称法"）
SPEAKERS = ("阿哲", "小林", "老周", "小李", "陈哲", "周明")

# 时间锚点识别（用于统计锚点节点数）
_TIME_NODE_RE = re.compile(r"\d+月\d+日|\d+年|上周|这周|本周|上个月|这个月|去年|今天")

# 归一化：去空白与常见中英标点，便于子串匹配（LLM 会改写标点与空格）
_PUNCT_RE = re.compile(r"[\s，。、：:；;！!？?…—\-（）()「」『』\"'’‘“”·,.\[\]]")


def _norm(text: str) -> str:
    return _PUNCT_RE.sub("", text or "").lower()


def load_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(DATA_DIR / "groupchat_mixed_dialogs.yaml", encoding="utf-8") as f:
        dialogs = yaml.safe_load(f)["dialogs"]
    with open(DATA_DIR / "groupchat_mixed_queries.yaml", encoding="utf-8") as f:
        queries = yaml.safe_load(f)
    return dialogs, queries


def mcp_conversation(dialog: Dict[str, Any]) -> str:
    """MCP 口径：多说话人文本，保留 [昵称] 标识（否则无法区分事实归属）"""
    return "\n".join(f"[{t['speaker']}] {t['text']}" for t in dialog["turns"])


def graph_stats(graph, label: str) -> Dict[str, Any]:
    """图画像：规模 / 归属词频 / TEMPORAL 边 / 时间锚点 / 塌缩与错称指标"""
    contents = [(d.get("content") or "") for _, d in graph.graph.nodes(data=True)]
    joined = "\n".join(contents)
    temporal = 0
    for _, _, ed in graph.graph.edges(data=True):
        rt = ed.get("rel_type")
        rt = rt.value if hasattr(rt, "value") else str(rt)
        if rt == "temporal":
            temporal += 1
    return {
        "label": label,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "temporal_edges": temporal,
        "time_anchor_nodes": sum(1 for c in contents if _TIME_NODE_RE.search(c)),
        "speaker_hits": {name: sum(1 for c in contents if name in c) for name in SPEAKERS},
        "collapse_user": sum(1 for c in contents if "用户" in c),
        "assistant_label": sum(1 for c in contents if "助手" in c),
    }


def print_stats(stats: Dict[str, Any]) -> None:
    print(f"  [{stats['label']}] {stats['nodes']} 节点 / {stats['edges']} 边"
          f" | TEMPORAL 边 {stats['temporal_edges']} | 时间锚点节点 {stats['time_anchor_nodes']}")
    print(f"    归属词频: {stats['speaker_hits']}")
    print(f"    「用户」出现 {stats['collapse_user']} 次 | 「助手」出现 {stats['assistant_label']} 次")


def match_facts(text: str, facts: List[str]) -> List[str]:
    norm = _norm(text)
    return [f for f in facts if _norm(f) in norm]


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 与 MCP 双链路 e2e 对比")
    parser.add_argument("--queries", default=None,
                        help="只跑这些查询 id（逗号分隔，如 q4,q10,q14）；默认全部")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        return 2

    dialogs, queries = load_dataset()
    if args.queries:
        wanted = {q.strip() for q in args.queries.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]
        missing = wanted - {q["id"] for q in queries}
        if missing:
            print(f"错误: 查询 id 不存在: {sorted(missing)}", file=sys.stderr)
            return 2
    print(f"# 数据集: groupchat_mixed（{len(dialogs)} 段对话 / {len(queries)} 条查询）")

    # ---- 装配两侧链路 ----
    mcp_dir, mcp_yaml = new_workspace("almvmcp_mcp_")
    mcp_graph, _, _, mcp_handlers = build_server(str(mcp_yaml))

    from alm.config import ALMConfig
    from alm.contract import AddMessage, AddRequest, SearchRequest
    from alm.engine import ALMEngine

    alm_cfg = ALMConfig.from_env()
    alm_cfg.data_dir = Path(tempfile.mkdtemp(prefix="almvmcp_alm_"))
    engine = ALMEngine(alm_cfg)
    print(f"# MCP 工作目录: {mcp_dir}")
    print(f"# ALM 数据目录: {alm_cfg.data_dir}")
    print(f"# ALM 弃权配置: 硬下限={alm_cfg.abstain_cosine} "
          f"模糊带上限={alm_cfg.abstain_verify_hi} "
          f"判定top_n={alm_cfg.abstain_judge_top_n}")

    # ---- ① 写入（两侧同源数据）----
    print("\n===== ① 写入 =====")
    # 合成时间戳：每段对话往前推一天，段内每轮 +5 分钟（ALM 契约需要，MCP 侧无此入参）
    base_ms = int(time.mktime(time.strptime("2026-09-01 09:00:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    for idx, dialog in enumerate(dialogs):
        conversation = mcp_conversation(dialog)
        started = time.time()
        add_res = call_tool(mcp_handlers, "dba_add_conversation", {"conversation": conversation})
        mcp_secs = time.time() - started

        messages = [
            AddMessage(
                role=turn["role"],
                content=f"[{turn['speaker']}] {turn['text']}",
                timestamp=base_ms + idx * 86_400_000 + i * 300_000,
            )
            for i, turn in enumerate(dialog["turns"])
        ]
        started = time.time()
        alm_stats = engine.add(AddRequest(
            request_id=f"{USER_ID}:{dialog['id']}",
            user_id=USER_ID,
            session_id=SESSION_ID,
            messages=messages,
        ))
        alm_secs = time.time() - started

        print(f"  {dialog['id']}（{len(dialog['turns'])} 轮 / 场景 {dialog['scene']}）"
              f"  MCP: 建 {add_res.get('nodes_created')} 节点（{mcp_secs:.1f}s）"
              f"  ALM: 建 {alm_stats.get('nodes')} 节点（{alm_secs:.1f}s）")

    # ---- ② 图谱画像 ----
    print("\n===== ② 图谱画像 =====")
    alm_space = engine.get_space(USER_ID, create=False)
    print_stats(graph_stats(mcp_graph, "MCP"))
    print_stats(graph_stats(alm_space.graph, "ALM"))

    # ---- ③ 逐查询对比 ----
    print("\n===== ③ 检索对比（gold_facts 命中）=====")
    rows = []
    for item in queries:
        query, facts = item["query"], item.get("gold_facts") or []

        mcp_res = call_tool(mcp_handlers, "dba_query_memory", {"query": query})
        mcp_stories = "".join(mcp_res.get("stories") or [])
        # 0927 审查 C1 消费者同步：query_memory 不再返回采纳 id 列表（story_nodes），无法按 id
        # 取采纳节点正文；原"MCP 文本 = 故事 + 采纳节点正文"退化为只用故事正文，不做节点级编造。
        mcp_text = mcp_stories

        alm_items = engine.search(SearchRequest(query=query, user_id=USER_ID, top_k=100))
        best_cos = engine.retrieval_stats.last_cosine()
        alm_text = " ".join(it.get("content") or "" for it in alm_items)
        # 逐条归一化后留存，供 top-k 累计命中率使用（不再额外消耗 LLM）
        alm_norm_items = [_norm(it.get("content") or "") for it in alm_items]

        mcp_hits = match_facts(mcp_text, facts)
        alm_hits = match_facts(alm_text, facts)
        rows.append({
            "id": item["id"], "category": item["category"], "query": query,
            "facts": facts, "mcp_hits": mcp_hits, "alm_hits": alm_hits,
            "alm_norm_items": alm_norm_items,
            "alm_best_cos": best_cos if best_cos is not None else 0.0,
            "mcp_returned": len(mcp_res.get("stories") or []),
            "alm_returned": len(alm_items),
            "mcp_chars": len(mcp_text), "alm_chars": len(alm_text),
            "alm_assistant": alm_text.count("助手"), "alm_user": alm_text.count("用户"),
        })

        flag = "" if len(mcp_hits) == len(alm_hits) else "  ← 差异"
        print(f"\n  [{item['id']}] {item['category']}｜{query}{flag}")
        print(f"    gold: {facts}")
        print(f"    MCP 命中 {len(mcp_hits)}/{len(facts)} {mcp_hits}"
              f"（{rows[-1]['mcp_returned']} 条故事 / {rows[-1]['mcp_chars']} 字符）")
        print(f"    ALM 命中 {len(alm_hits)}/{len(facts)} {alm_hits}"
              f"（{rows[-1]['alm_returned']} 条 / {rows[-1]['alm_chars']} 字符，"
              f"cos={rows[-1]['alm_best_cos']:.4f} "
              f"{'放行' if alm_items else '弃权'}）")

    # ---- ④ 汇总 ----
    print("\n===== ④ 汇总 =====")
    scored = [r for r in rows if r["facts"]]
    total = sum(len(r["facts"]) for r in scored)
    mcp_total = sum(len(r["mcp_hits"]) for r in scored)
    alm_total = sum(len(r["alm_hits"]) for r in scored)
    print(f"  可判定查询 {len(scored)} 条 / gold_facts 共 {total} 项")
    print(f"  MCP 命中 {mcp_total}/{total} = {mcp_total / total:.1%}")
    print(f"  ALM 命中 {alm_total}/{total} = {alm_total / total:.1%}")

    same = [r for r in scored if len(r["mcp_hits"]) == len(r["alm_hits"])]
    print(f"  两侧命中数一致的查询: {len(same)}/{len(scored)}")
    mcp_only = [(r["id"], f) for r in scored for f in r["facts"]
                if f in r["mcp_hits"] and f not in r["alm_hits"]]
    alm_only = [(r["id"], f) for r in scored for f in r["facts"]
                if f in r["alm_hits"] and f not in r["mcp_hits"]]
    print(f"  仅 MCP 命中: {mcp_only or '无'}")
    print(f"  仅 ALM 命中: {alm_only or '无'}")

    both_missed = [(r["id"], f) for r in scored for f in r["facts"]
                   if f not in r["mcp_hits"] and f not in r["alm_hits"]]
    print(f"  两侧均未命中（需人工核对，可能是抽取丢失或 fact 过严）: {both_missed or '无'}")

    # ALM 的 top-k 累计命中率：判断优势来自"排序靠前"还是"条数堆出来"
    print("\n  ALM top-k 累计命中率（对齐信息量的公平性检查）:")
    for k in (1, 3, 5, 10):
        hits = 0
        for r in scored:
            text = "".join(r["alm_norm_items"][:k])
            hits += sum(1 for f in r["facts"] if _norm(f) in text)
        print(f"    top-{k:<3} {hits}/{total} = {hits / total:.1%}")
    print(f"    top-all {alm_total}/{total} = {alm_total / total:.1%}"
          f"（ALM 平均返回 {sum(r['alm_returned'] for r in scored) / len(scored):.0f} 条）")
    print(f"    MCP     {mcp_total}/{total} = {mcp_total / total:.1%}（对照，无法按 k 切分）")

    print("\n  逐条命中数（MCP | ALM）:")
    for r in rows:
        print(f"    {r['id']:<4}{len(r['mcp_hits'])} | {len(r['alm_hits'])}"
              f"   {r['category']}")

    # ---- ⑤ 无解答查询的行为差异 ----
    print("\n===== ⑤ 无解答查询（库中无对应记忆）=====")
    for item in queries:
        if not item.get("expect_no_answer"):
            continue
        row = next(r for r in rows if r["id"] == item["id"])
        print(f"  query={item['query']!r}")
        print(f"    MCP 返回 {row['mcp_returned']} 条故事 / {row['mcp_chars']} 字符"
              f"（无弃权机制，必返回内容）")
        print(f"    ALM 返回 {row['alm_returned']} 条 data / {row['alm_chars']} 字符"
              f"（弃权阈值命中则返回空数组）")

    # ---- ⑥ ALM 检索余弦（对应 ALM_ABSTAIN_COS 的标定视角）----
    print("\n===== ⑥ ALM 检索余弦分布 =====")
    print(engine.retrieval_stats.report())

    # ---- ⑦ 称法问题探针 ----
    print("\n===== ⑦ ALM 检索文本中的主体称法 =====")
    for r in rows:
        if r["alm_assistant"] or r["alm_user"]:
            print(f"  {r['id']}: 「助手」×{r['alm_assistant']}  「用户」×{r['alm_user']}"
                  f"  ← {r['category']}")

    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
