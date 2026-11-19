# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
记忆保真 e2e（仿 eval/smoke_run_e2e 的陪伴链路，但图谱由**本轮优化的抽取管线**现场构建）

与 smoke_run_e2e 的区别：
- smoke 用**预置图谱** data/dba_eval/smoke_companion.yaml，只测"检索→回复"的后半段；
- 本测试从**原始对话**起跑，把本轮六项优化全串成一条链：

    原始对话 → dba_add_conversation（A 决策理由 / B evidence / 说话人归属 / 时间戳 / C1 原文落盘）
             → 落盘→重载贯通（timestamp / evidence / source_ref 不丢）
             → 巡检 dba_review_graph(C2) / dba_review_sources(C3)
             → dba_query_memory（P 链路 + StoryRank 故事化）
             → 真实回复 LLM（陪伴）→ judge 判分 → N 次取平均 → 汇总

分两类断言：
- **硬失败**：确定性不变量（归属 / 贯通 / 巡检只读 / 脱敏 / 时序拒答 / 决策理由闭环）——
  任一不满足即 exit(1)；
- **软观测**：依赖 LLM 的指标（判分均值、溯源召回、批次波动）——只报告，判断"是否劣化"
  看多次采样的中位数，不看单轮。

数据集：**自建**（见 DATASET），每段对话都对应本轮的一个优化点（segment["target"] 标注）。
整体可复用——直接 import DATASET 即可。

用法:
    python -m eval.test_memory_fidelity_e2e                 # 3 次回复取平均
    set E2E_REPLY_ROUNDS=1 && python -m eval.test_memory_fidelity_e2e
"""

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 重型依赖（fastembed/qdrant 等）在项目隔离依赖目录 pylibs，须优先加入路径
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.mcp_harness import build_server, call_tool, new_workspace  # noqa: E402
from eval.run_p_baseline import _load_dotenv  # noqa: E402
# 复用 e2e 的回复/判分链路，避免重复维护两套 prompt
from eval.smoke_run_e2e import (  # noqa: E402
    build_judge_llm, build_reply_llm, gen_reply, judge_reply,
)

REPLY_ROUNDS = int(os.environ.get("E2E_REPLY_ROUNDS", "3"))

# ────────────────────────────── 数据集（自建 · 可复用） ──────────────────────────────

DATASET = {
    "name": "memory_fidelity_e2e",
    "note": "每段对话对应本轮的一个优化点；target 字段标注它考的是什么",
    "segments": [
        {
            "id": "s1",
            "target": "说话人归属（昵称不塌缩为「用户」）+ 具体时间锚点 + 标识脱敏",
            "text": "\n".join([
                "[阿哲] 上周三我们开会定的 A 方案，我这周三之前能出稿。",
                "[小林] 我这两天在改 bug，前天弄到凌晨两点。",
                "[老周] 客户上周五说要换主色，阿哲你那边来得及吗？",
                "[老周] 你加我微信吧，手机号 13812345678，邮箱 laozhou@example.com。",
            ]),
        },
        {
            "id": "s2",
            "target": "决策与取舍（任务 A 靶子：两个被放弃方案 + 取舍依据）+ 原文支撑（任务 B）",
            "text": "\n".join([
                "[用户] 这次选服务器方案我纠结了很久，本来想直接上云，省事。",
                "[助手] 那为什么最后没上云？",
                "[用户] 云的成本太高，预算砍了一半，所以放弃了。",
                "[用户] 也考虑过自建机房，但要自己运维，我实在抽不出人。",
                "[助手] 那最后怎么定的？",
                "[用户] 折中，核心业务上云、边缘自建，钱省了，人也没多花。",
            ]),
        },
        {
            "id": "s3",
            "target": "推测口吻 + 无法定位的相对时间（任务 C3 靶子：「过两天」）",
            "text": "\n".join([
                "[用户] 上周三评审完，过两天我可能要去一趟外地出差。",
                "[助手] 定了吗？",
                "[用户] 还没定，估计得看项目进度，说不定要拖到月底。",
            ]),
        },
    ],
    "queries": [
        {
            "id": "q1", "scenario": "why",
            "text": "当初为什么没直接上云？",
            # 任务 A 的价值闭环：故事里必须能查到「被放弃的选项」与「取舍依据」
            "expect_rationale": ("预算", "成本", "运维"),
            "expect_abandoned": ("上云", "云", "自建"),
        },
        {
            "id": "q2", "scenario": "companion",
            "text": "最近项目压力太大了，天天加班到十一二点，感觉自己快扛不住了……",
        },
        {
            "id": "q3", "scenario": "followup",
            "text": "客户要换主色那件事，后来怎么样了？",
        },
    ],
}

SPEAKERS = ("阿哲", "小林", "老周")
SPECULATION_KEYS = ("可能", "估计", "说不定", "应该")
UNLOCATABLE_TIME = "过两天"

# 人工注入的近重复对（C2 体检的确定性靶子）
DUP_BASE = "阿哲按新配色改好了稿子"


# ────────────────────────────── 工具 ──────────────────────────────

def _contents(graph):
    return [(nid, (nd.get("content") or "")) for nid, nd in graph.graph.nodes(data=True)]


def _node_types(graph):
    from collections import Counter
    c = Counter()
    for _, nd in graph.graph.nodes(data=True):
        nt = nd.get("node_type")
        c[(nt.value if hasattr(nt, "value") else str(nt))] += 1
    return dict(c)


# ────────────────────────────── 阶段一：抽取 ──────────────────────────────

def extract_graph() -> dict:
    """原始对话 → 抽取（A/B/C1）→ 返回图谱与观测/硬失败"""
    _, yaml_path = new_workspace("fidelity_e2e_")
    graph, _, server, handlers = build_server(str(yaml_path))
    obs, fails, batches, seg_nodes = {}, [], {}, {}

    for seg in DATASET["segments"]:
        before = set(graph.graph.nodes())
        r = call_tool(handlers, "dba_add_conversation", {"conversation": seg["text"]})
        new_ids = list(set(graph.graph.nodes()) - before)
        seg_nodes[seg["id"]] = new_ids
        obs[f"{seg['id']}_nodes"] = len(new_ids)
        if r.get("evidence_stats") is not None:
            obs[f"{seg['id']}_evidence"] = r["evidence_stats"]
        if r.get("batch_id"):
            batches[seg["id"]] = r["batch_id"]
        if not new_ids:
            fails.append(f"{seg['id']} 未产出任何节点")

    # ① 归属：多说话人段里，绝对代称「用户」必须为 0，昵称必须都在
    s1_joined = " ".join(c for nid, c in _contents(graph) if nid in set(seg_nodes["s1"]))
    obs["s1_placeholder_user"] = s1_joined.count("用户")
    obs["s1_speakers_hit"] = sum(1 for s in SPEAKERS if s in s1_joined)
    if obs["s1_placeholder_user"]:
        fails.append(f"S1 出现绝对代称「用户」{obs['s1_placeholder_user']} 次（昵称归属失效）")
    if obs["s1_speakers_hit"] < len(SPEAKERS):
        fails.append(f"S1 昵称归属不全（{obs['s1_speakers_hit']}/{len(SPEAKERS)}）")

    # ② 决策理由（任务 A）：REASON 节点必须存在，且被放弃选项/依据进入内容
    joined = " ".join(c for _, c in _contents(graph))
    types = _node_types(graph)
    obs["reason_nodes"] = types.get("reason", 0)
    obs["abandoned_hit"] = sum(1 for k in ("上云", "自建") if k in joined)
    obs["rationale_hit"] = sum(1 for k in ("预算", "成本", "运维") if k in joined)
    if obs["reason_nodes"] == 0:
        fails.append("未产出任何 REASON 节点（任务 A 失效）")
    if obs["abandoned_hit"] == 0:
        fails.append("被放弃的选项未保留（任务 A 失效）")

    # ③ evidence（任务 B）：字段链路可用
    obs["evidence_stats"] = [obs.get(f"{s['id']}_evidence") for s in DATASET["segments"]]
    if all(v is None for v in obs["evidence_stats"]):
        fails.append("evidence_stats 全程缺失（任务 B 未生效）")

    # ④ source_ref 覆盖（C1）
    obs["source_ref_cov"] = sum(1 for _, nd in graph.graph.nodes(data=True) if nd.get("source_ref"))
    if obs["source_ref_cov"] == 0:
        fails.append("没有任何节点带 source_ref（C1 原文溯源未贯通）")

    return {
        "graph": graph, "server": server, "handlers": handlers, "yaml_path": yaml_path,
        "batches": batches, "seg_nodes": seg_nodes, "obs": obs, "fails": fails,
    }


# ────────────────────────────── 阶段二：贯通 + 巡检 ──────────────────────────────

def check_persistence_and_review(ctx: dict) -> None:
    """落盘→重载贯通（C1）+ 只读巡检（C2/C3），直接改 ctx['obs'] / ctx['fails']"""
    graph, handlers, yaml_path = ctx["graph"], ctx["handlers"], ctx["yaml_path"]
    obs, fails = ctx["obs"], ctx["fails"]

    # ── 落盘 → 重载贯通 ──
    ckpt_dir = Path(yaml_path).parent / "ckpt"
    call_tool(handlers, "dba_checkpoint", {"save_dir": str(ckpt_dir)})
    from dba_pipeline.loader import load_graph
    g2 = load_graph(str(ckpt_dir / "memory_graph.yaml"))
    fields = ("timestamp", "evidence", "source_ref")
    lost = {f: 0 for f in fields}
    for nid, nd in graph.graph.nodes(data=True):
        backed = g2.get_node(nid) or {}
        for f in fields:
            if nd.get(f) is not None and backed.get(f) != nd.get(f):
                lost[f] += 1
    obs["reload_lost"] = lost
    if any(lost.values()):
        fails.append(f"重载丢字段: {lost}")

    # ── C1 脱敏：落盘原文里不得出现手机号/邮箱 ──
    from dba_pipeline.source_store import load_batch, render_batch_text
    s1_batch = ctx["batches"].get("s1")
    text = render_batch_text(load_batch(str(Path(yaml_path).parent / "sources"), s1_batch)) \
        if s1_batch else ""
    obs["source_redacted"] = ("13812345678" not in text) and ("laozhou@example.com" not in text)
    if not text:
        fails.append("S1 原文未落盘（C1 失效）")
    elif not obs["source_redacted"]:
        fails.append("落盘原文未脱敏（手机号/邮箱仍在）")

    # ── 人工注入近重复对（C2 确定性靶子）──
    call_tool(handlers, "dba_intervene", {
        "action": "create_node", "params": {"node_type": "status", "content": DUP_BASE}})
    b = call_tool(handlers, "dba_intervene", {
        "action": "create_node", "params": {"node_type": "status", "content": DUP_BASE + "并发给了老周"}})
    obs["dup_warning"] = bool(b.get("duplicate_warning"))

    # ── C2 图谱体检（只读）──
    nodes_before, edges_before = graph.node_count, graph.edge_count
    rg = call_tool(handlers, "dba_review_graph", {"threshold": 0.9})
    if graph.node_count != nodes_before or graph.edge_count != edges_before:
        fails.append("图谱体检修改了图谱（应只读）")
    pairs = rg.get("duplicate_candidates") or []
    obs["rg_candidates"] = len(pairs)
    obs["rg_found_pair"] = any(
        {p["a"]["content"], p["b"]["content"]} == {DUP_BASE, DUP_BASE + "并发给了老周"}
        for p in pairs
    )
    if not obs["rg_found_pair"]:
        fails.append("图谱体检未报出人工注入的重复对")

    # ── C3 溯源巡检（只读）——核对推测最多的 s3 批 ──
    s3_batch = ctx["batches"].get("s3")
    edges_before = graph.edge_count
    rs = call_tool(handlers, "dba_review_sources",
                   {"batch_id": s3_batch} if s3_batch else {"max_batches": 1})
    if rs.get("error"):
        fails.append(f"溯源巡检不可用: {rs['error']}")
    else:
        if graph.edge_count != edges_before:
            fails.append("溯源巡检修改了图谱（应只读）")
        batches = rs.get("batches") or []
        obs["rs_summary"] = rs.get("summary")
        # 硬不变量：工具确实做了「原文 ↔ 该批节点」的联结（不是空转）
        obs["rs_audited_n"] = sum(len(b.get("audited_nodes") or []) for b in batches)
        if obs["rs_audited_n"] == 0:
            fails.append("溯源巡检未联结到任何节点（原文↔节点 映射失效）")
        # 硬不变量：逐批核对不得静默失败。audit_source 解析失败会返回 {"error": "JSON 解析失败"}，
        # 而 review_sources 的汇总用 len(... or []) 计 0 —— 会把"解析失败"伪装成"没发现问题"。
        obs["rs_audit_errors"] = [b.get("error") for b in batches if b.get("error")]
        if obs["rs_audit_errors"]:
            fails.append(f"溯源核对静默失败（LLM 输出无法解析）: {obs['rs_audit_errors']}")
        # 软观测：unsupported / granularity 的召回**依赖 LLM 判断**，单轮不可归因
        # （同一数据集实测 1 条 ↔ 0 条波动——0 的正常含义是"LLM 核对后认为没问题"，
        # 不是"没操作"；上面两条硬不变量保证它确实跑了且没解析失败）。
        # 只看多次采样的中位数，故不作硬失败。参见计划 §六 C3 说明与 §8.3。
        obs["rs_unsupported_n"] = sum(len(b.get("unsupported") or []) for b in batches)
        obs["rs_granularity_n"] = sum(len(b.get("granularity") or []) for b in batches)

    # ── 时序工具闭环（exact / 拒答）──
    tl_exact = call_tool(handlers, "dba_temporal_lookup", {"query": "上周五发生了什么？"})
    tl_absent = call_tool(handlers, "dba_temporal_lookup", {"query": "2025年1月1日发生了什么？"})
    obs["tl_exact_type"] = tl_exact.get("match_type")
    obs["tl_absent_rejected"] = tl_absent.get("rejected")
    if tl_absent.get("rejected") is not True:
        fails.append("不存在的显式日期未被拒答")
    has_anchor = any("上周五" in c for _, c in _contents(graph))
    obs["tl_exact_anchor_exists"] = has_anchor
    if has_anchor and tl_exact.get("rejected"):
        fails.append("库中存在「上周五」锚点却被误拒")


# ────────────────────────────── 阶段三：检索 → 回复 → 判分 ──────────────────────────────

def run_companion_chain(ctx: dict) -> list:
    """每条 query：检索(MCP) → 故事 → 真实回复 → judge，N 轮取平均"""
    handlers = ctx["handlers"]
    reply_llm = build_reply_llm()
    judge_llm = build_judge_llm()
    rows = []

    for q in DATASET["queries"]:
        rounds, stories = [], set()
        for i in range(REPLY_ROUNDS):
            qres = call_tool(handlers, "dba_query_memory", {"query": q["text"], "rerank_k": 3})
            story = "\n".join(qres.get("stories") or [])
            stories.add(story)
            reply = gen_reply(reply_llm, q["text"], story)
            sc = judge_reply(judge_llm, q["text"], story, reply)
            sc["round"] = i + 1
            sc["reply"] = reply
            rounds.append(sc)

        joined_story = " ".join(stories)
        row = {
            "id": q["id"], "scenario": q["scenario"], "query": q["text"],
            "story": joined_story,
            "rounds": rounds,
            "mean_total": round(sum(r["total"] for r in rounds) / len(rounds), 2),
            "distinct_stories": len(stories),
        }
        # 任务 A 的价值闭环：故事里必须查得到「被放弃的选项」与「取舍依据」
        if "expect_rationale" in q:
            row["rationale_hit"] = any(k in joined_story for k in q["expect_rationale"])
            row["abandoned_hit"] = any(k in joined_story for k in q["expect_abandoned"])
        rows.append(row)

        print(f"\n=== {q['id']} [{q['scenario']}] {q['text']}")
        for r in rounds:
            print(f"  r{r['round']} total={r['total']}  回复: {r['reply'][:60]!r}")
    return rows


# ────────────────────────────── 主流程 ──────────────────────────────

def main() -> None:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        sys.exit(2)

    print("# 记忆保真 e2e：原始对话 → 抽取 → 贯通 → 巡检 → 检索 → 回复 → judge")
    print(f"# 数据集 {DATASET['name']}：{len(DATASET['segments'])} 段对话 / "
          f"{len(DATASET['queries'])} 条查询 / 回复 {REPLY_ROUNDS} 轮取平均")
    for seg in DATASET["segments"]:
        print(f"#   {seg['id']}: {seg['target']}")

    print("\n## 阶段一：抽取（A 决策理由 / B evidence / 归属 / 时间戳 / C1 原文落盘）")
    ctx = extract_graph()
    for k, v in ctx["obs"].items():
        print(f"  {k}: {_short(v)}")
    for f in ctx["fails"]:
        print(f"  ✗ {f}")
    if not ctx["fails"]:
        print("  ✓ 抽取阶段无硬失败")

    print("\n## 阶段二：贯通 + 只读巡检（C1 重载 / C2 体检 / C3 溯源 / 时序拒答）")
    fails_before = len(ctx["fails"])
    check_persistence_and_review(ctx)
    for k in ("reload_lost", "source_redacted", "dup_warning", "rg_candidates",
              "rg_found_pair", "rs_summary", "rs_audited_n", "rs_audit_errors",
              "rs_unsupported_n", "rs_granularity_n", "tl_exact_type",
              "tl_absent_rejected", "tl_exact_anchor_exists"):
        if k in ctx["obs"]:
            print(f"  {k}: {_short(ctx['obs'][k])}")
    new_fails = ctx["fails"][fails_before:]
    if new_fails:
        for f in new_fails:
            print(f"  ✗ {f}")
    else:
        print("  ✓ 贯通与巡检无硬失败")

    print("\n## 阶段三：检索 → 故事 → 真实回复 → 陪伴判分")
    rows = run_companion_chain(ctx)

    # 决策理由闭环（q1）——这是任务 A 的端到端价值主张，缺了就说明"为什么"没传下来
    q1 = next((r for r in rows if r["id"] == "q1"), None)
    if q1 is not None and not (q1.get("rationale_hit") and q1.get("abandoned_hit")):
        ctx["fails"].append(
            f"q1 故事未同时带回「取舍依据」与「被放弃选项」"
            f"(rationale={q1.get('rationale_hit')}, abandoned={q1.get('abandoned_hit')})"
        )

    print("\n" + "=" * 62)
    print("查询均值（回复 N 轮取平均）:")
    for r in rows:
        extra = ""
        if "rationale_hit" in r:
            extra = f"  rationale={r['rationale_hit']} abandoned={r['abandoned_hit']}"
        print(f"  {r['id']} [{r['scenario']}] mean_total={r['mean_total']} "
              f"distinct_stories={r['distinct_stories']}{extra}")
    agg = sum(r["mean_total"] for r in rows) / len(rows)
    print(f"  总均分 = {agg:.2f}")

    out = ROOT / "eval" / "outputs" / "memory_fidelity_e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "aggregate": round(agg, 2),
        "reply_rounds": REPLY_ROUNDS,
        "observations": ctx["obs"],
        "queries": rows,
        "hard_fails": ctx["fails"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写: {out}")

    print("\n" + "=" * 62)
    if ctx["fails"]:
        print(f"存在硬失败 {len(ctx['fails'])} 项：")
        for f in ctx["fails"]:
            print(f"  - {f}")
        sys.exit(1)
    print("记忆保真 e2e 通过 [OK]：抽取→贯通→巡检→检索→回复 全链路的确定性不变量均满足。")
    print("（判分均值与溯源召回依赖 LLM，属软观测；判断是否劣化应看多次采样的中位数。）")


def _short(v, n: int = 90):
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


if __name__ == "__main__":
    main()
