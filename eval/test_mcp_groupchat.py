# -*- coding: utf-8 -*-
"""
群聊场景专项测试（MCP 全链路 · 真实 LLM 闭环）

用一小段**多说话人**群聊文本走完整 MCP 链路：
    dba_add_conversation（抽取）→ dba_query_memory / dba_temporal_lookup（检索返回）

重点观察两件事：
  ① 归属：多人对话下，抽取出的节点内容是否按昵称归属，还是会塌缩成"用户"；
  ② 时序：文本里混入**模糊时序词**（前几天 / 最近 / 那天）与少量具体周次
     （上周三 / 这周二 / 下周一），看抽取产出什么锚点、检索又怎么处理。

    · 抽取侧守卫 has_temporal_signal 要求 >=2 个不同日期词才启用时序增强链，
      模糊词不在 TIME_PERIOD_RE 内，所以本轮是"具体词触发、模糊词掺在里面"的混合场景；
    · 检索侧 temporal_lookup 分三级校验：显式词（上周三）硬校验字面命中，模糊词
      （前几天/后来）按其映射出的可接受锚点集合软校验，无时间词才退回语义最近邻；
      返回值里的 match_type 会标明本次属于 exact / fuzzy / fallback / rejected 哪一级。

用法:
    python -m eval.test_mcp_groupchat
"""

import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.run_p_baseline import _load_dotenv

# 多说话人群聊：昵称 + 方括号标注。模糊时序词与具体周次混排。
GROUP_CHAT = (
    "[小林] 前几天那个需求评审到底定下来没有？\n"
    "[老周] 上周三开会不是定了吗，先做 A 方案，B 方案放后面\n"
    "[阿哲] 我这边稿子还没改完，客户最近又加了两版需求\n"
    "[小林] 我这两天一直在加班，前天改 bug 改到凌晨两点\n"
    "[阿哲] 上周五客户突然说要换主色，我那天通宵重做了一版\n"
    "[老周] 这周二我把新的排期发群里了，你们看下有没有问题\n"
    "[小林] 收到了，我下周一前给你答复\n"
    "[阿哲] 行，那我先按新配色出一版，明天发你\n"
)

SPEAKERS = ("阿哲", "小林", "老周", "用户")

# 归属 / 时序 混合检索（走主链路：PAR + StoryRank）
MEMORY_QUERIES = (
    "阿哲最近在忙什么？",
    "谁在抱怨加班？",
    "上周三开会定了什么？",
)

# 时序工具单跳（"时间→事件"）：(查询, 期望 rejected)。None 表示仅观察不判定。
TEMPORAL_CASES = (
    ("前几天群里聊了什么？", None),          # 模糊词 → 软校验，观察落到 fuzzy 还是 rejected
    ("后来群里怎么说的？", True),            # 模糊词且库中无"后来/第二天"锚点 → 应拒答
    ("上周三开会定了什么？", False),         # 显式词字面命中 → 不得拒答
    ("2026年3月15日 群里聊了什么？", True),  # 显式词但库中不存在 → 应拒答
)


def _legacy_contract_fields(res: dict):
    """0927 审查 C1 契约守卫：StoryRank 输出契约已变更，返回里不再有 story_nodes /
    discarded_nodes（采纳/丢弃 id 列表），代之以 story_used（后端按正文覆盖统计的计数）与
    story_degraded。

    原先"采纳列表与故事正文是否自洽"的启发式检查建立在 discarded_nodes id 列表之上
    （比对被丢弃节点内容是否仍出现在正文）；而 story_used 本就由后端按正文实际覆盖算出，
    天然自洽，该检查已无意义。故此处改为契约变更守卫：若返回里仍出现旧字段，说明上游未同步，
    报警提示（见 main 中的告警与末尾汇总）。
    """
    return [k for k in ("story_nodes", "discarded_nodes", "adopted_ids") if k in res]


def _print_stories(res: dict, limit: int = 400) -> None:
    stories = res.get("stories") or []
    if not stories:
        print("  （无故事输出）")
        return
    for i, s in enumerate(stories, 1):
        text = s if isinstance(s, str) else str(s)
        print(f"  [故事{i}] {text[:limit]}{'...' if len(text) > limit else ''}")
    # 0927 审查 C1：不再有采纳/丢弃 id 列表，改为 story_used（正文覆盖计数）与 story_degraded。
    print(f"  story_used={res.get('story_used')} story_degraded={res.get('story_degraded')}")


def main() -> None:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        sys.exit(2)

    workdir, yaml_path = new_workspace("groupchat_")
    print(f"# 工作目录（用完可删）: {workdir}")
    print("# 原始群聊文本:")
    for line in GROUP_CHAT.strip().splitlines():
        print(f"    {line}")

    graph, _, _, handlers = build_server(str(yaml_path))
    failures = []

    # ---- ① 抽取 ----
    before = set(graph.graph.nodes())
    t0 = int(time.time() * 1000)
    add_res = call_tool(handlers, "dba_add_conversation", {"conversation": GROUP_CHAT})
    t1 = int(time.time() * 1000)
    created = [n for n in graph.graph.nodes() if n not in before]

    print("\n===== ① 抽取结果 =====")
    print(f"  handler 返回: {add_res}")
    for nid in created:
        nd = graph.graph.nodes[nid]
        nt = nd.get("node_type")
        nt = nt.value if hasattr(nt, "value") else str(nt)
        print(f"  {nid}  [{nt}]  ts={nd.get('timestamp')}  {nd.get('content')!r}")
        if nd.get("evidence"):
            print(f"        evidence: {nd['evidence']!r}")

    # evidence（任务 B）：产出率 + 全部通过"是否为原文子串"的程序化校验
    from dba_pipeline.extraction.dba import normalize_evidence_text

    haystack = normalize_evidence_text(GROUP_CHAT)
    with_ev = [n for n in created if graph.graph.nodes[n].get("evidence")]
    bad_ev = [
        (n, graph.graph.nodes[n]["evidence"]) for n in with_ev
        if normalize_evidence_text(graph.graph.nodes[n]["evidence"]) not in haystack
    ]
    print(f"  evidence 覆盖: {len(with_ev)}/{len(created)} 节点；"
          f"非法（非原文子串）: {[n for n, _ in bad_ev] or '无'}")
    if bad_ev:
        failures.append(f"① evidence 子串校验失效: {[n for n, _ in bad_ev]}")
    elif created and not with_ev:
        print("  注：本轮未产出任何 evidence（规则没被遵循或都被程序化拦掉，需人工核对）")

    # C1：溯源引用（source_ref）是否与批次对上
    bid = add_res.get("batch_id")
    with_ref = [n for n in created if graph.graph.nodes[n].get("source_ref")]
    print(f"  批次 {bid}：带 source_ref 的节点 {len(with_ref)}/{len(created)}"
          f"，且均指向该批: {all(graph.graph.nodes[n]['source_ref'] == bid for n in with_ref)}")
    if bid and len(with_ref) != len(created):
        failures.append("① 部分节点缺 source_ref")

    ts_bad = [
        n for n in created
        if not isinstance(graph.graph.nodes[n].get("timestamp"), int)
        or not (t0 <= graph.graph.nodes[n]["timestamp"] <= t1)
    ]
    if not created:
        failures.append("① 未产出任何节点（triage 可能判定跳过）")
    elif ts_bad:
        failures.append(f"① 节点时间戳缺失/越界: {ts_bad}")

    # 归属观察
    counts = {
        name: sum(1 for _, d in graph.graph.nodes(data=True) if name in (d.get("content") or ""))
        for name in SPEAKERS
    }
    print(f"  归属词频（节点内容中出现的次数）: {counts}")
    if counts.get("用户"):
        print("  注：仍有节点写成「用户」→ 多人归属未完全按昵称落地。")

    # 时序锚点与 TEMPORAL 边
    temporal_edges = [
        (u, v) for u, v, ed in graph.graph.edges(data=True)
        if (ed.get("rel_type").value if hasattr(ed.get("rel_type"), "value") else str(ed.get("rel_type"))) == "temporal"
    ]
    print(f"  TEMPORAL 边: {temporal_edges or '无'}")
    print(f"  图规模: {graph.node_count} 节点 / {graph.edge_count} 边")

    # ---- ② 主链路检索（归属 + 时序混合）----
    print("\n===== ② dba_query_memory（PAR + StoryRank）=====")
    consistency_warn = []
    for q in MEMORY_QUERIES:
        res = call_tool(handlers, "dba_query_memory", {"query": q, "rerank_k": 5})
        print(f"\n  query={q!r}")
        print(f"  purpose={res.get('purpose')} total_matched={res.get('total_matched')} returned={res.get('returned')}")
        _print_stories(res)
        story_text = "".join(res.get("stories") or [])
        print(f"  记录时间渲染: {'出现「记录于」' if '记录于' in story_text else '未出现'}")
        # 0927 审查 C1：原"采纳列表与正文自洽"检查已失去意义（见 _legacy_contract_fields），
        # 改为契约变更守卫。
        legacy = _legacy_contract_fields(res)
        if legacy:
            consistency_warn.append((q, legacy))
            print(f"  [契约变更告警] 返回中仍含旧字段 {legacy}，上游 StoryRank 输出未同步 0927 C1")

    # ---- ③ 时序工具单跳 ----
    print("\n===== ③ dba_temporal_lookup（时间→事件）=====")
    for q, expect_reject in TEMPORAL_CASES:
        res = call_tool(handlers, "dba_temporal_lookup", {"query": q})
        rej = bool(res.get("rejected"))
        print(f"\n  query={q!r}")
        print(f"  match_type={res.get('match_type')} rejected={res.get('rejected')} count={res.get('count')}")
        if res.get("reason"):
            print(f"  reason: {res['reason']}")
        if res.get("note"):
            print(f"  note: {res['note']}")
        for grp in res.get("matches") or []:
            ta = grp.get("time_anchor", {})
            print(f"  锚点 {ta.get('id')}: {ta.get('content')!r}")
            for f in grp.get("facts", []):
                print(f"    - {f['id']}: {f['content']!r}  time_consistent={f.get('time_consistent')}")
        if expect_reject is not None and rej != expect_reject:
            failures.append(f"③ 期望 rejected={expect_reject}，实际 {rej}：{q!r}")

    # ---- ④ 调度器批量路径（生产 MCP 的真实入口）----
    # 每一轮单独 on_conversation（各自带到达时间）→ 合并成一批维护。
    # 验证：① 每轮时间是否随文本进入抽取输入；② 时间前缀会不会被写进节点内容。
    print("\n===== ④ 调度器批量路径（每轮独立到达时间）=====")
    _, sched_yaml = new_workspace("groupchat_sched_")
    sched_graph, _, sched_server, _ = build_server(str(sched_yaml), with_scheduler=True)
    # 用公开回调等待异步维护真正结束（只看节点数会早退：节点先建、边后建）
    done = threading.Event()
    sched_server.scheduler.on_maintenance_done = lambda result: done.set()
    rounds = [ln for ln in GROUP_CHAT.strip().splitlines() if ln.strip()]
    for line in rounds:
        sched_server.add_conversation(line)
        time.sleep(0.15)  # 让各轮到达时间可区分
    print(f"  已入队 {len(rounds)} 轮，缓冲区={sched_server.scheduler.buffer_size}")
    sched_server.scheduler.flush_now()
    if not done.wait(timeout=300):
        failures.append("④ 调度器维护超时未完成")
    print(f"  维护后: {sched_graph.node_count} 节点 / {sched_graph.edge_count} 边")
    for nid, nd in sched_graph.graph.nodes(data=True):
        nt = nd.get("node_type")
        nt = nt.value if hasattr(nt, "value") else str(nt)
        print(f"  {nid}  [{nt}]  ts={nd.get('timestamp')}  {nd.get('content')!r}")

    # 前缀泄漏检查：节点内容里不应出现「YYYY-MM-DD」这种批次时间戳
    leak = re.compile(r"\d{4}-\d{2}-\d{2}")
    leaked = [
        nid for nid, nd in sched_graph.graph.nodes(data=True)
        if leak.search(nd.get("content") or "")
    ]
    batch_ts = [nd.get("timestamp") for _, nd in sched_graph.graph.nodes(data=True)]
    print(f"  时间前缀泄漏进节点内容: {leaked or '无'}")
    print(f"  节点时间戳取值集合: {sorted(set(batch_ts))}")
    if leaked:
        failures.append(f"④ 时间前缀泄漏进节点内容: {leaked}")
    if not sched_graph.node_count:
        failures.append("④ 调度器路径未产出任何节点")

    # ---- ⑤ 图谱体检（C2）：只读候选发现 ----
    # 复现 txt 里的重复节点场景（n25/n27）：手工造一对近重复，看体检能否报出来。
    print("\n===== ⑤ dba_review_graph（只读体检）=====")
    before_nodes = graph.node_count
    a = call_tool(handlers, "dba_intervene", {
        "action": "create_node",
        "params": {"node_type": "status", "content": "小林最近在加班改需求"},
    })
    b = call_tool(handlers, "dba_intervene", {
        "action": "create_node",
        "params": {"node_type": "status", "content": "小林最近一直在加班改需求"},
    })
    print(f"  造对: {a.get('node', {}).get('id')} / {b.get('node', {}).get('id')}"
          f" | 第二次 duplicate_warning={bool(b.get('duplicate_warning'))}")

    rep = call_tool(handlers, "dba_review_graph", {"threshold": 0.9})
    cand = rep.get("duplicate_candidates") or []
    print(f"  scanned={rep.get('scanned')} 候选对={len(cand)} 孤立={len(rep.get('isolated_nodes') or [])}")
    for p in cand[:5]:
        print(f"    {p['a']['id']}({p['a']['node_type']}) ≈ {p['b']['id']}({p['b']['node_type']})"
              f"  sim={p['similarity']}")
        print(f"      A: {p['a']['content']!r}")
        print(f"      B: {p['b']['content']!r}")
    hit = any({p["a"]["id"], p["b"]["id"]} == {a.get("node", {}).get("id"), b.get("node", {}).get("id")}
              for p in cand)
    print(f"  是否报出这对近重复: {hit}")
    print(f"  体检未改图: {graph.node_count == before_nodes + 2}")
    if not hit:
        failures.append("⑤ 体检未报出手工造的近重复节点对")
    if graph.node_count != before_nodes + 2:
        failures.append("⑤ 体检修改了图谱（应为只读）")

    # ---- ⑥ 溯源巡检（C3）：用原文核对节点，只报不改 ----
    print("\n===== ⑥ dba_review_sources（溯源巡检）=====")
    before_edges = graph.edge_count
    rep = call_tool(handlers, "dba_review_sources", {"max_batches": 1})
    if rep.get("error"):
        print(f"  {rep}")
        failures.append(f"⑥ 溯源巡检不可用: {rep.get('error')}")
    else:
        print(f"  summary={rep.get('summary')}")
        for b in rep.get("batches") or []:
            print(f"  批次 {b.get('batch_id')}（{b.get('rounds')} 轮，核对 {len(b.get('audited_nodes') or [])} 节点）")
            for item in b.get("unsupported") or []:
                print(f"    [无据] {item.get('id')}: {item.get('why')}")
            for item in b.get("missing") or []:
                print(f"    [漏抽] {item.get('quote')!r}: {item.get('why')}")
            for item in b.get("granularity") or []:
                print(f"    [粒度] {item.get('id')}: {item.get('why')}")
        if graph.edge_count != before_edges:
            failures.append("⑥ 溯源巡检修改了图谱（应为只读）")

    print("\n" + "=" * 60)
    if consistency_warn:
        print(f"[契约] {len(consistency_warn)} 条查询返回里仍含旧字段（story_nodes/discarded_nodes/"
              f"adopted_ids），上游未同步 0927 C1 契约变更")
    else:
        print("[契约] 返回字段符合 0927 C1 契约（无 story_nodes/discarded_nodes/adopted_ids）")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"群聊专项测试未通过：{len(failures)} 项")
        sys.exit(1)
    print("群聊专项测试通过 [OK]：抽取产出正常、节点时间戳完整；其余为观察项，见上方输出。")


if __name__ == "__main__":
    main()
