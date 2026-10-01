# SPDX-License-Identifier: AGPL-3.0-only

"""StoryRank 单元测试：轨迹记录、连通性筛选、故事化"""
import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType, RelationType
from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever
from dba_pipeline.llm.inference import (
    InferenceEngine, _STORY_MAX_CHARS, _truncate_at_sentence,
)


def _graph():
    g = MemoryGraph()
    g.add_memory("n1", "用户压力很大", NodeType.STATUS)
    g.add_memory("n2", "经常加班到10点", NodeType.REASON)
    g.add_memory("n3", "喝咖啡提神", NodeType.ACTION)
    g.add_edge("n1", "n2", RelationType.CAUSAL)
    g.add_edge("n2", "n3", RelationType.CAUSAL)
    return g


def _retriever(graph):
    r = PurposeDrivenRetriever.__new__(PurposeDrivenRetriever)
    r.graph = graph
    return r


def test_expand_with_trace_records_edge_source():
    graph = _graph()
    trace = graph.expand_with_trace(["n1"])
    assert "n2" in trace
    assert trace["n2"]["from"] == "n1"
    assert trace["n2"]["rel_type"] == RelationType.CAUSAL
    assert trace["n2"]["is_reverse"] is False
    assert trace["n2"]["weight"] > 0


def test_select_core_nodes_splits_components():
    graph = _graph()
    graph.add_memory("n6", "独立的旅行计划", NodeType.THING)
    r = _retriever(graph)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
            {"id": "n6", "content": "独立的旅行计划", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
        ]},
    ]

    groups = r.select_core_nodes(hop_history, ["n2", "n6"])
    # n2 回溯到 n1（一组），n6 是独立种子（另一组）
    assert len(groups) == 2
    flat = {nid for grp in groups for nid in grp}
    assert flat == {"n1", "n2", "n6"}


def test_select_core_nodes_drops_isolated_branch():
    graph = _graph()
    graph.add_memory("n7", "失眠", NodeType.STATUS)
    graph.add_edge("n1", "n7", RelationType.SEQUENCE)
    r = _retriever(graph)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
            {"id": "n7", "content": "失眠", "combined_score": 0.3,
             "from": "n1", "rel_type": RelationType.SEQUENCE, "is_reverse": False},
        ]},
    ]

    # 只有 n2 是结果，n7 不在结果回溯路径上，应被剔除
    groups = r.select_core_nodes(hop_history, ["n2"])
    assert len(groups) == 1
    assert set(groups[0]) == {"n1", "n2"}


def test_build_path_constructs_nodes_and_edges():
    graph = _graph()
    r = _retriever(graph)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
        ]},
    ]

    path = r._build_path(["n1", "n2"], hop_history)
    assert len(path["nodes"]) == 2
    assert path["nodes"][0]["node_type"] == "status"
    assert len(path["edges"]) == 1
    assert path["edges"][0] == {"from": "n1", "to": "n2",
                                "rel_type": "causal", "is_reverse": False}


def test_build_path_orders_by_level_not_by_root_block():
    """path 顺序 = 全局层次序（先所有根，再所有第二层…），不是"根组整块排"。

    回归自线上实测：老顺序下，子树长的根组会把后面更重要的根整体推后——种子第 1 名的
    证据节点落到 path 第 187 位（累计 14,702 字），而正文篇幅预算只有 2,000 字，
    于是证据一个字都写不进正文。
    """
    g = MemoryGraph()
    for nid in ("n1", "n2", "n3", "n4", "n9", "n10", "n11"):
        g.add_memory(nid, f"内容{nid}", NodeType.THING)
    r = _retriever(g)

    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "根A", "purpose_score": 0.9},
            {"id": "n9", "content": "根B", "purpose_score": 0.8},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "A1", "from": "n1"},
            {"id": "n3", "content": "A2", "from": "n2"},
            {"id": "n10", "content": "B1", "from": "n9"},
            {"id": "n11", "content": "B2", "from": "n9"},
        ]},
        {"hop": 2, "candidates": [
            {"id": "n4", "content": "A3", "from": "n3"},
        ]},
    ]

    # 传入的仍是"根组整块排"的老顺序：根A 的整棵子树，然后根B
    path = r._build_path(["n1", "n2", "n3", "n4", "n9", "n10", "n11"], hop_history)
    assert [n["id"] for n in path["nodes"]] == ["n1", "n9", "n2", "n10", "n11", "n3", "n4"]


def test_story_rank_parses_story_and_counts_used_backend():
    """`used` 由后端按"正文实际覆盖的节点数"计算（0927 审查 C4），不再采用模型自报值。

    正文逐字包含三条节点内容 → used == 3。
    """
    llm = FakeListChatModel(responses=[
        '{"story": "用户压力很大，因为经常加班到10点，常喝咖啡提神", "used": 999}'
    ])
    engine = InferenceEngine(llm)
    path = {
        "nodes": [
            {"id": "n1", "content": "用户压力很大", "node_type": "status", "score": 1.0},
            {"id": "n2", "content": "经常加班到10点", "node_type": "reason", "score": 0.9},
            {"id": "n3", "content": "喝咖啡提神", "node_type": "action", "score": 0.8},
        ],
        "edges": [
            {"from": "n1", "to": "n2", "rel_type": "causal", "is_reverse": False},
            {"from": "n2", "to": "n3", "rel_type": "causal", "is_reverse": False},
        ],
    }
    out = engine.story_rank("最近怎么样", path)
    assert out["story"] == "用户压力很大，因为经常加班到10点，常喝咖啡提神"
    assert out["used"] == 3          # 后端计数：模型自报的 999 被忽略
    assert out["degraded"] is False


def test_story_rank_used_is_backend_computed():
    """`used` 与正文严格一致：只计"内容真的出现在正文里"的节点，与节点总数无关。"""
    nodes = [
        {"id": "n1", "content": "用户压力很大", "node_type": "status", "score": 1.0},
        {"id": "n2", "content": "打算下个月去冰岛看极光", "node_type": "thing", "score": 0.9},
    ]
    # 正文只覆盖 n1（且模型仍自报一个数，应被忽略）
    engine = InferenceEngine(FakeListChatModel(responses=['{"story": "用户压力很大", "used": 999}']))
    out = engine.story_rank("最近怎么样", {"nodes": nodes, "edges": []})
    assert out["used"] == 1
    assert out["degraded"] is False

    # 正文与所有节点都不相关 → 0
    engine = InferenceEngine(FakeListChatModel(responses=['{"story": "今天天气不错"}']))
    assert engine.story_rank("最近怎么样", {"nodes": nodes, "edges": []})["used"] == 0

    # 模型省略 used 字段也照样由后端算出
    engine = InferenceEngine(FakeListChatModel(responses=['{"story": "用户压力很大"}']))
    assert engine.story_rank("最近怎么样", {"nodes": nodes, "edges": []})["used"] == 1


def test_story_rank_fallback_picks_top_n_by_score():
    """输出不可解析时按 combined_score 降序取前 N，而不是把全部候选平铺。

    背景：`" ".join(所有节点)` 在最难的题上会产出 1.2 万~2.3 万字符的原文流水账
    （0926 线上 7 次实测，采纳 217~443），而节点边界/边/方向会在拼接时全部抹平。
    """
    nodes = [
        {"id": f"n{i}", "content": f"事实{i}", "node_type": "status", "score": float(i)}
        for i in range(1, 46)  # 45 条，超过兜底上限 40
    ]
    llm = FakeListChatModel(responses=['{"story": "被截断的'])
    engine = InferenceEngine(llm)
    out = engine.story_rank("最近怎么样", {"nodes": nodes, "edges": []})
    assert out["degraded"] is True
    assert out["used"] == 40
    # 拼接以空格分隔 → 按词切分后逐条断言，避免 "事实1" 命中 "事实10" 这类子串误判
    picked = out["story"].split()
    assert len(picked) == 40                           # 上限 40：45 条不能全进去
    assert picked[0] == "事实45"                        # 按分数降序，最高分在最前
    assert "事实1" not in picked and "事实5" not in picked   # 最低分的被丢掉


def test_story_rank_retries_once_when_over_budget():
    """超预算 → 附加强制压缩指令重试一次（实测规则 6 的篇幅约束会被越过 4/46 次）"""
    over = json.dumps({"story": "很长的内容" * 600}, ensure_ascii=False)   # 3000 字符
    ok = json.dumps({"story": "压缩后的内容"}, ensure_ascii=False)
    llm = FakeListChatModel(responses=[over, ok])
    engine = InferenceEngine(llm)
    path = {"nodes": [{"id": "n1", "content": "x", "node_type": "status", "score": 1.0}],
            "edges": []}
    out = engine.story_rank("最近怎么样", path)
    # 拿到的是**第二条**响应 → 说明第一次超预算后确实重试了，且采用了重试结果
    assert out["story"] == "压缩后的内容"
    assert len(out["story"]) <= _STORY_MAX_CHARS
    assert out["degraded"] is False


def test_story_rank_shrinks_feed_before_truncating(caplog):
    """压缩重试后仍超预算 → **缩喂重试**（丢尾部节点），而不是先截正文。

    0930 需求：面对 Full 不能只靠兜底。超预算时先丢"优先级最低的尾部节点"重试，
    只有缩到下限仍超才硬截断——这样正文是**完整**的，不会把收束/结论那段写出来又切掉。
    """
    over = json.dumps({"story": "很长的内容" * 600}, ensure_ascii=False)   # 3000 字符
    ok = json.dumps({"story": "完整写完了"}, ensure_ascii=False)
    # 第 1 次超预算 → 第 2 次（压缩指令）仍超 → 第 3 次（缩喂后）落进预算
    llm = FakeListChatModel(responses=[over, over, ok])
    nodes = [{"id": f"n{i}", "content": "x", "node_type": "status", "score": 1.0}
             for i in range(20)]                                    # > _STORY_FIT_MIN_NODES
    with caplog.at_level("WARNING"):
        out = InferenceEngine(llm).story_rank("最近怎么样", {"nodes": nodes, "edges": []})

    assert out["story"] == "完整写完了"          # 采用缩喂那一次的结果
    assert len(out["story"]) <= _STORY_MAX_CHARS
    assert out["degraded"] is False
    msgs = [r.getMessage() for r in caplog.records]
    assert any("缩喂重试第 1 轮" in m for m in msgs), msgs   # 确实缩喂了
    assert not any("已硬截断" in m for m in msgs), msgs      # 且没有再截正文


def test_story_rank_truncates_when_retry_still_over_budget():
    """重试后仍超预算 → 硬截断到句边界，保证返回值不超过预算"""
    over = json.dumps({"story": "第一句。第二句。" * 400}, ensure_ascii=False)
    llm = FakeListChatModel(responses=[over, over])
    engine = InferenceEngine(llm)
    path = {"nodes": [{"id": "n1", "content": "x", "node_type": "status", "score": 1.0}],
            "edges": []}
    out = engine.story_rank("最近怎么样", path)
    assert len(out["story"]) <= _STORY_MAX_CHARS
    assert out["story"].endswith("。")             # 落在句末，不出现半句


def test_truncate_at_sentence_boundaries():
    assert _truncate_at_sentence("abc", 10) == "abc"          # 未超限原样返回
    # 优先切在句末标点
    assert _truncate_at_sentence("aa. bb. cc. dd", 12) == "aa. bb. cc."
    # 中文句号同理
    assert _truncate_at_sentence("甲甲。乙乙。丙丙。丁丁。戊戊。", 11) == "甲甲。乙乙。丙丙。"
    # 无标点时退到空白，不切断单词
    text = "word " * 20
    got = _truncate_at_sentence(text, 13)
    assert len(got) <= 13 and not got.endswith("wor")
    assert got == got.rstrip() and " " not in got[-1:]


def test_story_rank_empty_nodes_returns_used_zero():
    """无候选时的早退分支必须与正常分支**同形状**（曾漏改，仍回 adopted_ids）"""
    engine = InferenceEngine(FakeListChatModel(responses=["不会用到"]))
    assert engine.story_rank("最近怎么样", {"nodes": [], "edges": []}) == {
        "story": "", "used": 0, "degraded": False}


def test_story_rank_fallback_respects_char_budget():
    """降级拼接同样受篇幅预算约束，否则又会退化成"长而无结构"的流水账。"""
    long_node = "很长的节点内容" * 400                 # 单条就 3200 字符 > 预算 2000
    nodes = [{"id": "n1", "content": long_node, "node_type": "status", "score": 1.0}]
    engine = InferenceEngine(FakeListChatModel(responses=['{"story": "被截断的']))
    out = engine.story_rank("最近怎么样", {"nodes": nodes, "edges": []})
    assert out["degraded"] is True
    assert len(out["story"]) == _STORY_MAX_CHARS


def test_select_core_nodes_no_infinite_loop_on_revisit():
    """双向边折返（回访）时 parent 不应成环，回溯不死循环"""
    graph = _graph()
    r = _retriever(graph)

    # 模拟双向边折返：n1 → n2 → n1
    hop_history = [
        {"hop": 0, "candidates": [
            {"id": "n1", "content": "用户压力很大", "purpose_score": 0.5},
        ]},
        {"hop": 1, "candidates": [
            {"id": "n2", "content": "经常加班到10点", "combined_score": 0.6,
             "from": "n1", "rel_type": RelationType.CAUSAL, "is_reverse": False},
        ]},
        {"hop": 2, "candidates": [
            {"id": "n1", "content": "用户压力很大", "combined_score": 0.4,
             "from": "n2", "rel_type": RelationType.SCENARIO, "is_reverse": True},
        ]},
    ]

    groups = r.select_core_nodes(hop_history, ["n1", "n2"])
    assert len(groups) == 1
    assert set(groups[0]) == {"n1", "n2"}


def test_rule6_allows_juxtaposition_but_forbids_naming_the_change():
    """规则 6 的边界必须**成对**存在。

    0927 之前规则 6 自相矛盾：前半句要求"围绕『用户当前消息』问的**变化**写"，后半句又禁止
    写"立场"——问"立场如何演变"的题（平台 B2 · Causal chain/path/intermediate-step recovery，
    实测恒 0）因此被夹死，模型只能把全部节点平铺。

    修法是把禁令精确化为「**可并置、不可命名**」：允许同一主体不同时刻的事实按先后并列，
    但禁止给这个变化下结论。两个断言必须同时成立——任一边被单独改掉，这条题类都会再次失效。
    """
    from dba_pipeline.llm.inference import (
        STORY_RANK_PROMPT, STORY_RANK_PROMPT_TS,
        STORY_RANK_PROMPT_CONDENSE, STORY_RANK_PROMPT_STATE,
    )
    variants = [("base", STORY_RANK_PROMPT), ("ts", STORY_RANK_PROMPT_TS),
                ("condense", STORY_RANK_PROMPT_CONDENSE), ("state", STORY_RANK_PROMPT_STATE)]
    for name, prompt in variants:
        tpl = str(prompt.messages[0].prompt.template)
        assert "并置" in tpl, f"{name} 变体丢了「可并置先后事实」"
        assert "不得替它下结论" in tpl, f"{name} 变体丢了「不得替它下结论」"
        assert "不得臆造先后" in tpl, f"{name} 变体丢了「顺序依据不得臆造」"
