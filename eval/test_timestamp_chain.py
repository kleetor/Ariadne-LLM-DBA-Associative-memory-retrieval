# -*- coding: utf-8 -*-
"""
时间戳链路专项测试（MCP 全链路 · 真实 LLM 闭环）

被测链路：
    dba_add_conversation → 节点时间戳注入 → dba_checkpoint 落盘
    → load_graph 重载 → 时间戳保留 → dba_temporal_lookup 时序行为

背景：节点 timestamp 此前是"写得进、落得下、读不回"的死字段——保存走
`MemoryGraph.to_dict()`，载入却走 `dba_pipeline.loader.load_graph()` 的另一套代码，
两边的字段集不一致。本次修复后由 `PERSISTED_NODE_FIELDS` 统一登记、双向透传。
本测试同时验证贯通性（① ② ③）与时序工具的拒答契约（④ ⑤ ⑥）。

用法:
    python -m eval.test_timestamp_chain
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.run_p_baseline import _load_dotenv
from dba_pipeline.loader import load_graph

# 两个不同显式日期 → 命中 has_temporal_signal（>=2 个不同日期词），
# 触发 DBA 的时序叙事增强链，产出时间锚点节点 + TEMPORAL 边。
CONVERSATION = (
    "2026年8月30日 阿哲说他在赶平面设计的稿子，客户反复改需求，最近压力挺大。\n"
    "2026年9月7日 阿哲说稿子终于交了，当天晚上去吃了顿好的庆祝。"
)

# 库中绝不存在的日期，用于验证拒答（不得回退成"语义最近邻"）
ABSENT_DATE_QUERY = "2025年1月1日 阿哲做了什么？"
# 库中确实出现过的日期，用于观察命中情况（抽取质量相关，仅作信息输出）
PRESENT_DATE_QUERY = "2026年9月7日 阿哲做了什么？"
VAGUE_QUERY = "最近有什么烦心事？"


def main() -> None:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        sys.exit(2)

    workdir, yaml_path = new_workspace("ts_chain_")
    failures = []

    # ---- ⓪ 时期词覆盖检查（离线、确定性）----
    # 锚点白名单与查询校验曾是两份独立正则，出现过「今天 / 上个月 / 大二上学期」这类
    # "锚点能定位、查询却不校验"的词。这里用一份**外部维护**的已知词表做不变量检查：
    # 凡能被锚点识别的已知时期词，要么进入校验（explicit/fuzzy/era），要么在
    # 文档化的"仅锚点"例外清单里。词表由测试维护，不取自被测元组，避免自证。
    from dba_pipeline.retrieval.retriever import (
        TOOL_TIME_ANCHOR_RE, _ANCHOR_ONLY_TOKENS,
        _era_periods, _explicit_periods, _fuzzy_periods,
    )

    known_anchor_words = (
        "昨天", "今天", "明天", "前天", "上周", "这周", "本周", "下周", "上周三", "周三",
        "上个月", "这个月", "下个月", "去年", "今年", "明年", "前年",
        "寒假", "暑假", "上学期", "下学期", "上半年", "下半年", "年底", "年初",
        "高中", "大学", "大专", "高一", "大二", "大四",
        "前几天", "前两天", "这两天", "这几天", "最近", "近来", "这段时间",
        "那天", "当天", "当时",
        "后来", "之后", "第二天", "次日",
        "凌晨", "早上", "上午", "中午", "下午", "晚上", "那段时间", "近年来",
        "毕业", "入职",
    )
    # 仅在"仅锚点"清单里逐字出现（非正则片段）的词，视为已文档化的例外
    anchor_only_literals = {t for t in _ANCHOR_ONLY_TOKENS if not t.startswith("\\") and "^" not in t}

    print("\n===== ⓪ 时期词覆盖检查 =====")
    coverage = {"explicit": [], "fuzzy": [], "era": [], "anchor_only": []}
    drift = []
    for w in known_anchor_words:
        q = f"{w}发生了什么？"
        if _explicit_periods(q):
            cls = "explicit"
        elif _fuzzy_periods(q):
            cls = "fuzzy"
        elif _era_periods(q):
            cls = "era"
        else:
            cls = "anchor_only"
        coverage[cls].append(w)
        if not TOOL_TIME_ANCHOR_RE.search(w):
            drift.append(f"{w}(锚点正则不识别)")
        elif cls == "anchor_only" and w not in anchor_only_literals:
            drift.append(f"{w}(未校验且不在仅锚点清单)")
    for cls, words in coverage.items():
        print(f"  {cls:12}: {' '.join(words) or '（无）'}")
    if drift:
        failures.append(f"⓪ 时期词漂移: {drift}")
        print(f"  ✗ 漂移: {drift}")
    else:
        print("  ✓ 所有已知时期词均已归类（校验或文档化例外）")

    # 不变量：常见非时间语境词不得进入硬校验（否则正常查询会被误判为拒答）
    for q in ("阿哲工作顺利吗？", "他毕业了吗？", "晚上我总是睡不着"):
        if _explicit_periods(q):
            failures.append(f"⓪ 非时间语境被硬校验: {q!r} → {sorted(_explicit_periods(q))}")
    # 不变量：年份不参与硬校验（否则「2026年8月30日」会被「2026年」放宽到同年任意锚点）
    if "2026年" in _explicit_periods("2026年3月15日 群里聊了什么？"):
        failures.append("⓪ 年份进入了硬校验，会放宽跨期匹配")
    print(f"  硬校验通过 {sorted(_explicit_periods('今天群里聊了什么？'))}"
          f" / {sorted(_explicit_periods('上个月大家在忙什么？'))}"
          f" / 当天→{[w for w, _ in _fuzzy_periods('当天发生了什么？')]}"
          f" / 大二→{_era_periods('大二上学期发生了什么？')}")

    # ---- ① 时间戳注入 ----
    print(f"\n# 工作目录（用完可删）: {workdir}")
    print("# 构建 MCP 全链路（真实 LLM + embedding）...")
    graph, _, _, handlers = build_server(str(yaml_path))
    before = set(graph.graph.nodes())
    t0 = int(time.time() * 1000)
    add_res = call_tool(handlers, "dba_add_conversation", {"conversation": CONVERSATION})
    t1 = int(time.time() * 1000)
    created = [n for n in graph.graph.nodes() if n not in before]
    ts_map = {n: graph.graph.nodes[n].get("timestamp") for n in created}
    print("\n===== ① 时间戳注入 =====")
    print(f"  handler 返回: {add_res}")
    for nid in created:
        print(f"  {nid}  ts={ts_map[nid]}  {graph.graph.nodes[nid].get('content')!r}")
    ok_inject = bool(created) and all(
        isinstance(ts, int) and not isinstance(ts, bool) and t0 <= ts <= t1
        for ts in ts_map.values()
    )
    if not created:
        failures.append("① 未产出任何节点（triage 可能判定跳过）")
    elif not ok_inject:
        bad = {n: ts for n, ts in ts_map.items() if not isinstance(ts, int)}
        failures.append(f"① 时间戳缺失或越界: {bad}")

    # ---- ② 落盘透传 ----
    ckpt_dir = workdir / "ckpt"
    ck = call_tool(handlers, "dba_checkpoint", {"save_dir": str(ckpt_dir)})
    ckpt_yaml = ckpt_dir / "memory_graph.yaml"
    snap = yaml.safe_load(ckpt_yaml.read_text(encoding="utf-8"))
    missing = [n["id"] for n in snap.get("nodes", []) if "timestamp" not in n]
    print("\n===== ② checkpoint 落盘 =====")
    print(f"  {ck}")
    print(f"  {ckpt_yaml}")
    print(f"  缺 timestamp 的节点: {missing or '无'}")
    if missing:
        failures.append(f"② 落盘丢字段: {missing}")

    # ---- ③ 重载贯通（本次修复点）----
    g2 = load_graph(str(ckpt_yaml))
    mismatch = [
        (nid, ts, g2.get_node(nid).get("timestamp") if g2.get_node(nid) else None)
        for nid, ts in ts_map.items()
        if (g2.get_node(nid) or {}).get("timestamp") != ts
    ]
    print("\n===== ③ load_graph 重载 =====")
    print(f"  重载后节点数: {g2.node_count}")
    if mismatch:
        for nid, mem_ts, disk_ts in mismatch:
            print(f"  {nid}: 内存={mem_ts} 重载={disk_ts}")
        failures.append(f"③ 重载后时间戳不一致: {[m[0] for m in mismatch]}")
    else:
        print(f"  时间戳一致: {ts_map}")

    # ---- ④ 拒答：不存在的显式日期 ----
    rej = call_tool(handlers, "dba_temporal_lookup", {"query": ABSENT_DATE_QUERY})
    ok_reject = bool(rej.get("rejected")) and not rej.get("matches")
    print("\n===== ④ 拒答（不存在的显式日期）=====")
    print(f"  query={ABSENT_DATE_QUERY!r}")
    print(f"  rejected={rej.get('rejected')} count={rej.get('count')} matches={rej.get('matches')}")
    if rej.get("reason"):
        print(f"  reason: {rej['reason']}")
    if not ok_reject:
        failures.append("④ 不存在的显式日期未拒答（发生跨期回退）")

    # ---- ⑤ 命中（信息性：依赖抽取是否产出时间锚点）----
    hit = call_tool(handlers, "dba_temporal_lookup", {"query": PRESENT_DATE_QUERY})
    print("\n===== ⑤ 命中（信息性）=====")
    print(f"  query={PRESENT_DATE_QUERY!r}  rejected={hit.get('rejected')} count={hit.get('count')}")
    for grp in hit.get("matches") or []:
        ta = grp.get("time_anchor", {})
        print(f"  锚点 {ta.get('id')}: {ta.get('content')!r}")
        for f in grp.get("facts", []):
            print(f"    - {f['id']}: {f['content']!r}")
    if not hit.get("matches"):
        print("  （未定位到时间锚点：抽取未产出该日期的锚点节点/反向 TEMPORAL 边，非本测试断言项）")

    # ---- ⑥ 鲁棒：模糊表述不崩溃 ----
    vague = call_tool(handlers, "dba_temporal_lookup", {"query": VAGUE_QUERY})
    ok_vague = isinstance(vague, dict) and "matches" in vague
    print("\n===== ⑥ 鲁棒（模糊表述）=====")
    print(f"  query={VAGUE_QUERY!r}  count={vague.get('count')} rejected={vague.get('rejected')}")
    if not vague.get("rejected") and vague.get("matches"):
        print("  注：模糊表述不含显式日期词，无法做字面校验 → 仍走语义最近邻，"
              "调用方拿不到『命中/回退』信号（已知取舍，非本测试断言项）。")
    if not ok_vague:
        failures.append("⑥ 模糊表述返回结构异常")

    # ---- 附：存量节点回填能力（静态探测，只看 intervene 实现）----
    import inspect

    from dba_pipeline.mcp_server import DBAServer

    intervene_src = (
        inspect.getsource(DBAServer.intervene)
        + inspect.getsource(DBAServer._intervene_impl)
    )
    can_backfill = "timestamp" in intervene_src
    print("\n===== 附：存量节点时间戳回填 =====")
    print(f"  dba_intervene 是否接受 timestamp: {'是' if can_backfill else '否（存量节点暂无法补时间戳）'}")

    print("\n" + "=" * 60)
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"专项测试未通过：{len(failures)} 项")
        sys.exit(1)
    print("专项测试通过 [OK]：时间戳注入→落盘→重载贯通；时序工具对不存在的显式日期正确拒答。")


if __name__ == "__main__":
    main()
