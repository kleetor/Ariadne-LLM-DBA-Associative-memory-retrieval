# -*- coding: utf-8 -*-
"""
完整管线稳定性测试（模拟数据集 · 真实 LLM）

目的：本轮修了一堆东西（归属 / 时序 / 决策取舍 / evidence / 原文溯源 / 图谱体检 / 溯源巡检），
单点验证都过了，但**没有一次是从头到尾连起来跑的**。这里针对每个修复点都放一个"靶子"，
跑 N 轮看整条管线的稳定性——抽取依赖 LLM 且提示词很长，单轮结果不可归因，
所以关注的是**跨轮的波动范围与中位数**，以及有没有硬失败。

模拟数据集（4 段，逐段追加到同一张图）：

    S1 群聊·归属 + 时序    3 人带昵称；含具体日期词（触发时序增强链）
    S2 决策取舍            任务 A 的靶子：两个被放弃方案 + 自述依据
    S3 推测 + 无法定位时间  任务 C3 的靶子：全是推测口吻；含「过两天」这类无法定位的表述
    S4 语义重复            任务 C2/D 的靶子：与 S1 内容近重复

每轮检查：
    ① 抽取指标（节点数 / 归属 / 锚点数 / REASON 数 / evidence 合规率 / source_ref 覆盖）
    ② 落盘→重载贯通（timestamp / evidence / source_ref 全保留）
    ③ 检索（query_memory 非空；temporal_lookup 的 exact 命中与 must-reject）
    ④ 图谱体检 dba_review_graph 能跑通且只读
    ⑤ 溯源巡检 dba_review_sources 能报出推测 / 粒度问题

用法:
    python -m eval.test_pipeline_stability        # 默认 6 轮
    set STAB_ROUNDS=1 && python -m eval.test_pipeline_stability
"""

import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.mcp_harness import build_server, call_tool, new_workspace
from eval.run_p_baseline import _load_dotenv

ROUNDS = int(os.environ.get("STAB_ROUNDS", "6"))

S1_GROUP_CHAT = "\n".join([
    "[阿哲] 上周三我们开会定的 A 方案，我这周三之前能出稿。",
    "[小林] 我这两天在改 bug，前天弄到凌晨两点。",
    "[老周] 客户上周五说要换主色，阿哲你那边来得及吗？",
    "[阿哲] 来得及，我明天先发一版给你。",
])

S2_DECISION = "\n".join([
    "[用户] 这次选服务器方案我纠结了很久，本来想直接上云，省事。",
    "[助手] 那为什么最后没上云？",
    "[用户] 云的成本太高，预算砍了一半，所以放弃了。",
    "[用户] 也考虑过自建机房，但要自己运维，我实在抽不出人。",
    "[助手] 那最后怎么定的？",
    "[用户] 折中，核心业务上云、边缘自建，钱省了，人也没多花。",
])

S3_SPECULATION = "\n".join([
    "[用户] 上周三评审完，过两天我可能要去一趟外地出差。",
    "[助手] 定了吗？",
    "[用户] 还没定，估计得看项目进度，说不定要拖到月底。",
    "[用户] 反正应该是甲方那边催得紧，我也没办法。",
])

S4_DUPLICATE = "[阿哲] 补充一下，客户上周五说要换主色那件事，我已经按新配色改好了。"

# 归属检查：这些必须出现、而绝对代称「用户」必须为 0
SPEAKERS = ("阿哲", "小林", "老周")
# 任务 A 靶子：被放弃的方案 / 依据关键词
ABANDONED_KEYS = ("上云", "云", "自建")
RATIONALE_KEYS = ("预算", "成本", "运维", "人")
# 任务 C3 靶子：推测口吻的关键词（若这些被写成确定事实，巡检应报 unsupported）
SPECULATION_KEYS = ("可能", "估计", "说不定", "应该")
# 无法定位的时间表述（若被独立成节点，巡检应报 granularity）
UNLOCATABLE_TIME = "过两天"


def _node_types(graph):
    from collections import Counter
    c = Counter()
    for _, nd in graph.graph.nodes(data=True):
        nt = nd.get("node_type")
        c[(nt.value if hasattr(nt, "value") else str(nt))] += 1
    return dict(c)


def _contents(graph):
    return [(nid, (nd.get("content") or "")) for nid, nd in graph.graph.nodes(data=True)]


def run_round(idx: int) -> dict:
    """跑一轮完整管线，返回该轮的指标与检查结果"""
    _, yaml_path = new_workspace(f"stab_r{idx}_")
    graph, _, server, handlers = build_server(str(yaml_path))
    res = {"round": idx, "fail": [], "obs": {}}

    # ── S1：归属 + 时序 ──
    before = set(graph.graph.nodes())
    r1 = call_tool(handlers, "dba_add_conversation", {"conversation": S1_GROUP_CHAT})
    s1_nodes = [n for n in graph.graph.nodes() if n not in before]
    contents_all = dict(_contents(graph))
    joined = " ".join(contents_all.values())
    res["obs"]["S1_nodes"] = len(s1_nodes)
    res["obs"]["placeholder_user"] = joined.count("用户")
    res["obs"]["speakers_hit"] = sum(1 for s in SPEAKERS if s in joined)
    res["obs"]["evidence"] = r1.get("evidence_stats")
    res["obs"]["batch_id"] = r1.get("batch_id")
    if not s1_nodes:
        res["fail"].append("S1 未产出节点")
    if res["obs"]["placeholder_user"]:
        res["fail"].append(f"S1 出现绝对代称「用户」{res['obs']['placeholder_user']} 次")
    if res["obs"]["speakers_hit"] < 3:
        res["fail"].append(f"S1 昵称归属不全（只命中 {res['obs']['speakers_hit']}/3）")

    # 时间锚点（thing 类型且内容像时间词）
    anchors = [
        nid for nid, nd in graph.graph.nodes(data=True)
        if str(nd.get("node_type")).lower().endswith("thing")
        and len((nd.get("content") or "")) <= 8
    ]
    res["obs"]["anchors"] = len(anchors)

    # ── S2：决策取舍 ──
    r2 = call_tool(handlers, "dba_add_conversation", {"conversation": S2_DECISION})
    types = _node_types(graph)
    contents_all = dict(_contents(graph))
    joined = " ".join(contents_all.values())
    res["obs"]["S2_nodes_created"] = r2.get("nodes_created")
    res["obs"]["reason"] = types.get("reason", 0)
    res["obs"]["abandoned_hit"] = sum(1 for k in ABANDONED_KEYS if k in joined)
    res["obs"]["rationale_hit"] = sum(1 for k in RATIONALE_KEYS if k in joined)
    if res["obs"]["reason"] == 0:
        res["fail"].append("S2 未产出任何 REASON 节点（任务 A 失效）")
    if res["obs"]["abandoned_hit"] == 0:
        res["fail"].append("S2 未保留被放弃的选项")

    # ── S3：推测 + 无法定位时间 ──
    r3 = call_tool(handlers, "dba_add_conversation", {"conversation": S3_SPECULATION})
    s3_batch = r3.get("batch_id")
    res["obs"]["S3_nodes_created"] = r3.get("nodes_created")
    res["obs"]["S3_batch"] = s3_batch
    spec_nodes = [nid for nid, c in _contents(graph) if any(k in c for k in SPECULATION_KEYS)]
    res["obs"]["speculation_nodes"] = len(spec_nodes)
    rel_time_nodes = [nid for nid, c in _contents(graph) if UNLOCATABLE_TIME in c]
    res["obs"]["unlocatable_time_nodes"] = len(rel_time_nodes)

    # ── S4：语义重复（人工造对，确定性） ──
    base = "阿哲按新配色改好了稿子"
    a = call_tool(handlers, "dba_intervene", {
        "action": "create_node", "params": {"node_type": "status", "content": base}})
    b = call_tool(handlers, "dba_intervene", {
        "action": "create_node", "params": {"node_type": "status", "content": base + "并发给了老周"}})
    res["obs"]["dup_warning"] = bool(b.get("duplicate_warning"))

    # ── ② 落盘 → 重载贯通 ──
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
    res["obs"]["reload_lost"] = lost
    if any(lost.values()):
        res["fail"].append(f"重载丢字段: {lost}")
    res["obs"]["source_ref_cov"] = sum(
        1 for _, nd in graph.graph.nodes(data=True) if nd.get("source_ref"))

    # ── ③ 检索 ──
    q = call_tool(handlers, "dba_query_memory", {"query": "客户要换主色那件事怎么样了？"})
    res["obs"]["query_returned"] = q.get("returned")
    if not q.get("stories"):
        res["fail"].append("主链路检索返回空")

    tl = {
        "exact": call_tool(handlers, "dba_temporal_lookup", {"query": "上周五发生了什么？"}),
        "absent": call_tool(handlers, "dba_temporal_lookup", {"query": "2025年1月1日发生了什么？"}),
        "fuzzy": call_tool(handlers, "dba_temporal_lookup", {"query": "前几天发生了什么？"}),
    }
    res["obs"]["tl_types"] = {k: v.get("match_type") for k, v in tl.items()}
    if tl["absent"].get("rejected") is not True:
        res["fail"].append("不存在的显式日期未被拒答")
    # 「上周五」若被抽成锚点，就必须能精确命中；没抽到则只记录（抽取波动，非本项失败）
    has_anchor = any("上周五" in c for _, c in _contents(graph))
    res["obs"]["tl_exact_anchor_exists"] = has_anchor
    if has_anchor and tl["exact"].get("rejected"):
        res["fail"].append("库中存在「上周五」锚点却被误拒")

    # ── ④ 图谱体检（C2）──
    nodes_before, edges_before = graph.node_count, graph.edge_count
    rg = call_tool(handlers, "dba_review_graph", {"threshold": 0.9})
    res["obs"]["rg_candidates"] = len(rg.get("duplicate_candidates") or [])
    if graph.node_count != nodes_before or graph.edge_count != edges_before:
        res["fail"].append("图谱体检修改了图谱")
    hit_dup = any(
        {p["a"]["content"], p["b"]["content"]} == {base, base + "并发给了老周"}
        for p in (rg.get("duplicate_candidates") or [])
    )
    res["obs"]["rg_found_pair"] = hit_dup
    if not hit_dup:
        res["obs"].setdefault("notes", []).append("体检未报出人工造的对")

    # ── ⑤ 溯源巡检（C3）── 针对 S3（推测最多的一批）
    edges_before = graph.edge_count
    rs = call_tool(handlers, "dba_review_sources",
                   {"batch_id": s3_batch} if s3_batch else {"max_batches": 1})
    if rs.get("error"):
        res["fail"].append(f"溯源巡检不可用: {rs['error']}")
    else:
        batches = rs.get("batches") or []
        res["obs"]["rs_summary"] = rs.get("summary")
        res["obs"]["rs_unsupported"] = [
            (b.get("unsupported") or []) for b in batches
        ]
        res["obs"]["rs_granularity"] = [
            (b.get("granularity") or []) for b in batches
        ]
        if graph.edge_count != edges_before:
            res["fail"].append("溯源巡检修改了图谱")

    return res


def run_deterministic_suite() -> tuple:
    """确定性批量用例（不调 LLM，秒级）——把"能大批量跑"的部分做成大样本。

    覆盖本轮所有**非 LLM** 的修复点：时期词分类 / 模糊映射边界 / 硬校验的周X精确化 /
    evidence 子串校验 / 语义去重的阈值与排除规则 / 标识脱敏。

    Returns:
        (总用例数, 失败列表)
    """
    import numpy as np

    from dba_pipeline.core.jump_axis import NodeType
    from dba_pipeline.extraction.dba import sanitize_evidence
    from dba_pipeline.graph.review import find_duplicate_candidates
    from dba_pipeline.retrieval.retriever import (
        TOOL_TIME_ANCHOR_RE, _ANCHOR_ONLY_TOKENS, _anchor_acceptable_periods,
        _era_acceptable, _era_periods, _explicit_periods, _fuzzy_periods,
        _period_literals, _time_consistency,
    )
    from dba_pipeline.source_store import redact_identifiers

    total, fails = 0, []

    def check(cond, msg):
        nonlocal total
        total += 1
        if not cond:
            fails.append(msg)

    # ── 1. 时期词分类：每个已知词都必须被"锚点识别 + 归类到某一档" ──
    known = (
        "昨天", "今天", "明天", "前天", "上周", "这周", "本周", "下周", "上周三", "周三",
        "上个月", "这个月", "下个月", "去年", "今年", "明年", "前年",
        "寒假", "暑假", "上学期", "下学期", "上半年", "下半年", "年底", "年初",
        "高中", "大学", "大专", "高一", "大二", "大四",
        "前几天", "前两天", "这两天", "这几天", "最近", "近来", "这段时间",
        "那天", "当天", "当时", "后来", "之后", "第二天", "次日",
        "凌晨", "早上", "上午", "中午", "下午", "晚上", "那段时间", "近年来", "毕业", "入职",
    )
    anchor_only_literals = {t for t in _ANCHOR_ONLY_TOKENS if not t.startswith("\\") and "^" not in t}
    for w in known:
        q = f"{w}发生了什么？"
        cls = ("explicit" if _explicit_periods(q)
               else "fuzzy" if _fuzzy_periods(q)
               else "era" if _era_periods(q) else "anchor_only")
        check(bool(TOOL_TIME_ANCHOR_RE.search(w)), f"锚点正则不识别期限词: {w}")
        check(cls != "anchor_only" or w in anchor_only_literals,
              f"期限词既未校验也不在仅锚点清单: {w}")

    # ── 2. 模糊映射 / 锚点-事实一致性的时间跨度边界（正/负例各若干）──
    # 注：None = 无法判定（锚点没有映射，如已被移出校验的「最近」）
    span_cases = [
        ("前几天", "前天", True), ("前几天", "上周三", False), ("前几天", "2026年9月7日", False),
        ("前两天", "前天", True), ("前两天", "上周五", False),
        ("这两天", "昨天", True), ("这两天", "上周", False),
        ("这几天", "前天", True), ("这几天", "上个月", False),
        ("最近", "上周", None), ("最近", "去年", None),      # 已移出校验，只作锚点
        ("那天", "上周三", True), ("那天", "去年", False),    # 集合不交但字面包含 → 同义
        ("后来", "第二天", True), ("后来", "上周三", False),
        ("当时", "当天", True), ("当时", "下个月", False),
    ]
    for anchor, fact, expect in span_cases:
        got = _time_consistency(
            _anchor_acceptable_periods(anchor), fact, _period_literals(fact))
        check(got == expect, f"锚点-事实一致性错: 锚点{anchor} 事实{fact} 期望{expect} 实得{got}")

    # ── 3. 显式词硬校验：周X 精确化（「上周三」不得命中「上周五」）──
    hard_cases = [
        ("上周三", "上周三", True), ("上周三", "上周五", False), ("上周三", "上周", False),
        ("周三", "周三", True), ("周三", "周五", False),
        ("2026年3月15日", "2026年3月15日", True), ("2026年3月15日", "2026年9月7日", False),
        ("今天", "今天", True), ("今天", "昨天", False),
        ("上个月", "上个月", True), ("上个月", "这个月", False),
        ("大二", "大二上学期", True), ("大二", "大学时期", True), ("大二", "高一", False),
    ]
    for q, anchor, expect in hard_cases:
        qp = _explicit_periods(q)
        if qp:
            got = bool(qp & _explicit_periods(anchor))
        else:
            era = _era_acceptable(_era_periods(q))
            got = bool(era) and any(a in anchor for a in era)
        check(got == expect, f"硬校验错: 查询{q} 锚点{anchor} 期望{expect} 实得{got}")
    # 阶段词分族：跨学段不得互相匹配
    check(set(_era_acceptable(["大二"])) & {"高中", "高一", "高二", "高三"} == set(),
          "大学族与高中族未隔离")
    check("高三" in _era_acceptable(["高中"]), "高中族内未互相覆盖")

    # ── 4. evidence 子串校验（正/负例）──
    convo = "小林说：上周五客户突然说要换主色，我那天通宵重做了一版。预算也砍了一半。"
    ev_cases = [
        ("上周五客户突然说要换主色", True),          # 原文原话
        ("预算也砍了一半", True),                    # 原文原话
        ("上周五，客户突然说要换主色", True),         # 仅标点差异 → 应通过（去标点后是子串）
        ("客户要求更换主色调", False),                # 改写 → 拦
        ("小林通宵改稿", False),                     # 概括 → 拦
        ("通宵", False),                            # 过短 → 拦
        ("", False),                                # 空 → 拦
        ("不存在的句子内容在这里", False),             # 编造 → 拦
    ]
    for ev, expect_keep in ev_cases:
        cleaned, kept, _ = sanitize_evidence([{"action": "create", "evidence": ev}], convo)
        got = "evidence" in cleaned[0]
        check(got == expect_keep, f"evidence 校验错: {ev!r} 期望保留={expect_keep} 实得={got}")

    # ── 5. 语义去重的阈值语义与排除规则（合成向量，可批量）──
    rng = np.random.default_rng(20260911)
    dim, per_cluster, clusters = 32, 4, 12
    nodes, vecs, expect_pairs = [], {}, set()
    for c in range(clusters):
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        ids = []
        for k in range(per_cluster):
            v = base + rng.normal(scale=0.01, size=dim)
            v /= np.linalg.norm(v)
            nid = f"c{c}n{k}"
            ids.append(nid)
            nodes.append({"id": nid, "content": f"这是一条事实陈述{c}{k}", "node_type": NodeType.STATUS})
            vecs[nid] = v
        for i in range(per_cluster):
            for j in range(i + 1, per_cluster):
                expect_pairs.add(frozenset((ids[i], ids[j])))
    # 负控 1：thing 锚点，向量完全相同也必须被排除
    for k in (1, 2):
        nodes.append({"id": f"anchor{k}", "content": "2026年9月7日", "node_type": NodeType.THING})
        vecs[f"anchor{k}"] = np.ones(dim) / np.sqrt(dim)
    # 负控 2：过短内容，向量完全相同也必须被排除
    for k in (1, 2):
        nodes.append({"id": f"short{k}", "content": "短", "node_type": NodeType.STATUS})
        vecs[f"short{k}"] = np.ones(dim) / np.sqrt(dim)

    pairs = find_duplicate_candidates(nodes, vecs, threshold=0.9, max_pairs=10000)
    got_pairs = {frozenset((p["a"], p["b"])) for p in pairs}
    check(expect_pairs <= got_pairs,
          f"漏报同簇近邻 {len(expect_pairs - got_pairs)} 对")
    check(not any({"anchor1", "anchor2"} == set(p) for p in got_pairs), "thing 锚点未被排除")
    check(not any({"short1", "short2"} == set(p) for p in got_pairs), "过短内容未被排除")
    # 阈值语义：调高阈值应使候选数不增
    strict = find_duplicate_candidates(nodes, vecs, threshold=0.9999, max_pairs=10000)
    check(len(strict) <= len(pairs), "提高阈值后候选数反而增加")
    # max_pairs 截断
    capped = find_duplicate_candidates(nodes, vecs, threshold=0.9, max_pairs=5)
    check(len(capped) == 5, f"max_pairs 截断失效: {len(capped)}")

    # ── 6. 标识脱敏 ──
    redact_cases = [
        ("联系 13812345678", "13812345678" not in redact_identifiers("联系 13812345678")),
        ("邮箱 a.b@ex.com", "a.b@ex.com" not in redact_identifiers("邮箱 a.b@ex.com")),
        ("QQ@123456", "@123456" not in redact_identifiers("QQ@123456")),
        ("ID 3512593260", "3512593260" not in redact_identifiers("ID 3512593260")),
        ("日期 2026年9月7日", "2026年9月7日" in redact_identifiers("日期 2026年9月7日")),  # 不该误伤日期
    ]
    for src, ok in redact_cases:
        check(ok, f"脱敏异常: {src!r} → {redact_identifiers(src)!r}")

    return total, fails


def main() -> None:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        sys.exit(2)

    # ── ⓪ 确定性批量用例（大样本、秒级、不烧 token）──
    print("# ⓪ 确定性用例批量采样（时期词 / 模糊边界 / 硬校验 / evidence / 去重 / 脱敏）")
    det_total, det_fails = run_deterministic_suite()
    print(f"  用例总数 {det_total}，失败 {len(det_fails)}")
    for f in det_fails[:20]:
        print(f"    ✗ {f}")
    if det_fails:
        print("\n确定性用例未通过，先修再跑 LLM 轮次。")
        sys.exit(1)
    print("  ✓ 全部通过\n")

    if ROUNDS <= 0:
        print("（STAB_ROUNDS=0：只跑确定性套件，跳过 LLM 轮次）")
        return

    print(f"# 完整管线稳定性测试：{ROUNDS} 轮")
    print(f"# 模拟数据集：S1 群聊归属+时序({len(S1_GROUP_CHAT.splitlines())}行) / "
          f"S2 决策取舍({len(S2_DECISION.splitlines())}行) / "
          f"S3 推测+相对时间({len(S3_SPECULATION.splitlines())}行) / S4 语义重复")

    results = []
    for i in range(1, ROUNDS + 1):
        r = run_round(i)
        results.append(r)
        print(f"\n===== 第 {i} 轮 =====")
        for k, v in r["obs"].items():
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            print(f"  {k}: {v}")
        if r["fail"]:
            for f in r["fail"]:
                print(f"  ✗ {f}")
        else:
            print("  ✓ 本轮无硬失败")

    # ── 汇总：跨轮波动 ──
    numeric = [
        "S1_nodes", "placeholder_user", "speakers_hit", "anchors", "reason",
        "abandoned_hit", "rationale_hit", "speculation_nodes", "unlocatable_time_nodes",
        "source_ref_cov", "query_returned", "rg_candidates",
    ]
    print("\n" + "=" * 62)
    print("指标跨轮汇总（min / 中位数 / max）:")
    for k in numeric:
        vals = [r["obs"].get(k) for r in results if isinstance(r["obs"].get(k), (int, float))]
        if not vals:
            continue
        print(f"  {k:24}: {min(vals)} / {statistics.median(vals)} / {max(vals)}")

    print("\n检查项通过率（按轮）:")
    checks = {
        "hard_fail_free": sum(1 for r in results if not r["fail"]),
        "rg_found_pair": sum(1 for r in results if r["obs"].get("rg_found_pair")),
        "dup_warning": sum(1 for r in results if r["obs"].get("dup_warning")),
    }
    for k, v in checks.items():
        print(f"  {k:24}: {v}/{len(results)}")

    all_fail = [f for r in results for f in r["fail"]]
    print("\n" + "=" * 62)
    if all_fail:
        print(f"存在硬失败 {len(all_fail)} 项：")
        for f in all_fail:
            print(f"  - {f}")
        sys.exit(1)
    print(f"稳定性测试通过 [OK]：{ROUNDS} 轮全部无硬失败，"
          "字段贯通 / 检索 / 时序拒答 / 只读巡检均稳定。")
    print("（各指标的跨轮波动见上表；波动本身是 LLM 抽取的固有属性，"
          "判断「是否劣化」应以中位数对比，不看单轮。）")


if __name__ == "__main__":
    main()
