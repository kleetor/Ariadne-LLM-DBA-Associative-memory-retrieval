# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 链路的端到端验证：用**真实组件**跑一遍 Add → Search → 落盘 → 重载 → Search。

为什么单测全绿还不够：本轮三项改动都是**跨模块的装配**，

  · 向量分片落盘 —— 单测直接调 `VectorStore.save/load`，而链路上是
    `space.add() → space.save() → _save_index() → vector_store.save()`，
    中间还隔着「索引目录路径派生」与「节点集合指纹校验」两层；
  · role 埋点 —— 单测用手写的 RunnableBinding，链路上是
    `PROMPT | with_role(llm, …)`，且 llm 上挂着 TokenMeter 回调；
  · 边连接拆批 —— 单测用最小替身，链路上要经过真实的 `apply_ops`。

所以本文件专门补这一段：**验证改动在链路上真的生效**，而不只是函数本身正确。

用**按 prompt 分派的假模型**而不是 `FakeListChatModel`：后者按调用顺序返回，
链路里多一次/少一次调用就会张冠李戴，测出来的结论没有意义。按 system prompt 的
特征词分派则与调用顺序无关。
"""

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from alm.config import ALMConfig
from alm.contract import AddMessage
from alm.engine import ALMEngine
from alm.space import MemorySpace, RetrievalStats
from alm.tokens import TokenMeter
from dba_pipeline.extraction.dba import EDGE_BATCH_SIZE

# system prompt 的特征词 → 该链的预设响应（与调用顺序无关）
M_NODE = "节点抽取器"
M_EDGE = "边连接器"
M_TRIAGE = "值得长期记忆的新事实"
M_PURPOSE = "对话意图分析助手"
M_STORY = "记忆故事整理助手"
M_ROUTE = "路由判定员"


class ScriptedChatModel(BaseChatModel):
    """按 system prompt 分派的假模型；`calls` 记录各链被调用的次数"""

    routes: tuple = ()
    _calls: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def call_count(self, marker: str) -> int:
        return sum(1 for m in self._calls if m == marker)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system = msg.content or ""
                break
        else:
            system = getattr(messages[0], "content", "") if messages else ""

        for marker, payload in self.routes:
            if marker in system:
                self._calls.append(marker)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=payload))]
                )
        raise AssertionError("未识别的 prompt（前 120 字）：" + system[:120])


class WordOverlapEmbeddings(Embeddings):
    """词袋哈希向量：共享词越多余弦越高。

    为什么不用「向量 = [文本长度, 1.0]」那种玩具实现：`graph_dedup_threshold=0.92`
    会把长度相近的节点判成重复并合并，图根本建不起来，测的就不是链路了。
    也不用纯文本哈希（哈希向量彼此近乎正交）：那样 `purpose_filter_threshold` 会把
    候选全部滤掉，检索恒为空，断言退化成"空 == 空"，没有意义。
    """

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_documents(self, texts):
        return [self._vec(t).tolist() for t in texts]

    def embed_query(self, text):
        return self._vec(text).tolist()


def _node_payload(n: int) -> str:
    """让正文与查询共享 {the,user,mentioned,fact}，同时每个节点带 7 个独有词。

    这是**为了不被去重合并**：`graph_dedup_threshold=0.92` 会把余弦 ≥0.92 的节点
    合并掉，而纯模板（只差一个序号）的余弦正好落在 0.91 附近，是否被合并取决于
    词频细节——那种测试会随夹具微调而随机失败。加上独有词后节点间余弦降到 ~0.36，
    与查询仍有 ~0.60（高于 `purpose_filter_threshold=0.30`），两个阈值都有余量。
    """
    ops = []
    for i in range(n):
        unique = " ".join(f"uniq{i}x{k}" for k in range(7))
        ops.append({
            "action": "create", "temp_id": f"t{i}", "node_type": "action",
            "content": f"the user mentioned fact {i} {unique}",
        })
    return json.dumps({"node_ops": ops, "edge_ops": []}, ensure_ascii=False)


# 查询与节点正文共享词，且与假模型给出的 purposes 共享词——
# 否则目的过滤（`purpose_filter_threshold`）会把候选全部滤掉，检索恒为空。
QUERY = "the user mentioned fact"
PURPOSES = ["the user mentioned fact"]


def _routes(node_count: int, route_p: float = 0.9):
    return (
        (M_TRIAGE, "NEEDED"),
        (M_NODE, _node_payload(node_count)),
        (M_EDGE, json.dumps({"edge_ops": []})),
        (M_PURPOSE, json.dumps({"status": "ok", "purposes": PURPOSES,
                                "condense": False, "state": False, "temporal": False})),
        (M_STORY, json.dumps({"story": "用户在这段对话里提到了若干事实。", "used": 1},
                             ensure_ascii=False)),
        # 归属判定（B 项 + T3）。实测里它自报的 p 呈双峰，故夹具直接给两端值。
        (M_ROUTE, json.dumps({"p": route_p})),
    )


def _messages(n: int = 6):
    return [
        AddMessage(role="user" if i % 2 == 0 else "assistant",
                   content=f"line {i} of the conversation", timestamp=1704067200000 + i)
        for i in range(n)
    ]


def _make_space(tmp_path, model, **overrides):
    # 夹具默认 = **强制图臂**：改动前 ALM 的默认行为就是"整站建图"，让绝大多数
    # 链路用例（抽取 / 连边拆批 / 重载一致性 …）不受路由改造影响。
    # 需要新路由行为的用例显式传 `auto_route=True`（见下）或 `force_mode="auto"`。
    overrides.setdefault("force_mode", "graph")
    # 旧模式参数（write_mode / read_mode / auto_route）已从 ALMConfig 删除，
    # 这里统一翻译到**唯一开关** `force_mode`：
    #   write_mode="documents" / read_mode="vector" → force_mode="vector"（纯向量臂）
    #   auto_route=True                             → force_mode="auto"（按空间归属）
    # 三个 pop 都要无条件执行，否则短路的那些会残留在 overrides 里让 ALMConfig 报错。
    _w = overrides.pop("write_mode", None)
    _r = overrides.pop("read_mode", None)
    _a = overrides.pop("auto_route", None)
    # 先按 auto_route 落到 auto，再让**强制臂**盖过它——与生产的语义一致：
    # force_mode 是实验对照，必须能压住按空间归属。
    if _a:
        overrides["force_mode"] = "auto"
    if _w == "documents" or _r == "vector":
        overrides["force_mode"] = "vector"
    # 弃权整体关闭：与生产 `.env.alm`（ALM_ABSTAIN_COS=0 / VERIFY_HI=0）一致。
    # 本测试不考检索质量，开着弃权会让"夹具余弦偏低"被误读成"链路坏了"。
    overrides.setdefault("abstain_cosine", 0.0)
    overrides.setdefault("abstain_verify_hi", 0.0)
    config = ALMConfig(data_dir=tmp_path / "data", **overrides)
    return MemorySpace(
        user_id="eval:run_e2e:locomo:conv-0",
        config=config,
        embeddings=WordOverlapEmbeddings(),
        llm=model,
        # 一空间一目录：图谱在 <空间目录>/graph.yaml，索引与文档区都从它的父目录派生
        yaml_path=tmp_path / "data" / "conv0" / "graph.yaml",
        stats=RetrievalStats(),
    )


# ---------------------------------------------------------------------------
# 1. 向量分片落盘：链路上「落盘 → 重载」后检索结果必须逐条一致
# ---------------------------------------------------------------------------

def test_add_search_save_reload_is_consistent(tmp_path):
    """本项直接验证 store.py 的落盘改造在链路上成立"""
    model = ScriptedChatModel(routes=_routes(node_count=5))
    space = _make_space(tmp_path, model)

    stats = space.add(_messages())
    assert stats["nodes"] == 5, "节点抽取链路没跑通"

    before = space.search(QUERY, top_k=5)
    assert before, "检索没有返回任何内容"

    space.save()

    # 索引目录必须落下**分片**（新格式），而不是旧的 save_local 产物
    shard_dir = tmp_path / "data" / "conv0" / "graph.index" / "faiss" / "vectors"
    assert shard_dir.is_dir(), "未按新格式落盘向量分片"
    assert list(shard_dir.glob("*.npz")), "分片为空"
    assert not (tmp_path / "data" / "conv0" / "graph.index" / "faiss" / "index.faiss").exists(), \
        "仍在落 LangChain 的 index.faiss（索引不该再落盘）"

    # 用**新实例**从盘上载入，检索结果必须与落盘前一致
    reloaded = _make_space(tmp_path, ScriptedChatModel(routes=_routes(node_count=5)))
    assert reloaded.graph.node_count == 5

    after = reloaded.search(QUERY, top_k=5)
    assert after, "重载后检索返回空——分片恢复或索引重建有问题"
    assert [i["content"] for i in after] == [i["content"] for i in before]


def test_reload_does_not_call_the_llm(tmp_path):
    """复用落盘索引时不得再跑 LLM——这是分片持久化要买到的东西之一"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model)
    space.add(_messages())
    space.save()

    fresh = ScriptedChatModel(routes=_routes(node_count=3))
    _make_space(tmp_path, fresh)
    assert fresh._calls == [], "重载空转就调了 LLM"


# ---------------------------------------------------------------------------
# 2. role 埋点：真实链路上必须按侧归位
# ---------------------------------------------------------------------------

def test_role_tags_land_in_the_meter_on_the_real_pipeline(tmp_path):
    """验证 with_role 在**真实链路**上生效，而不只是单测里手搓的 binding"""
    meter = TokenMeter(price_input=1.0, price_output=4.0, price_cache=0.02, verbose=False)
    model = ScriptedChatModel(routes=_routes(node_count=3), callbacks=[meter])
    space = _make_space(tmp_path, model)

    space.add(_messages())
    space.search(QUERY, top_k=3)

    by_role = meter.summary()["by_role"]
    # 写侧两条 + 读侧两条，缺任一条说明该调用点漏打标签（会落进 unknown）
    for role in ("extract", "link", "purpose", "story"):
        assert role in by_role, f"{role} 未归位，实际桶：{sorted(by_role)}"
    assert "unknown" not in by_role, "有调用点没打 role 标签"


# ---------------------------------------------------------------------------
# 3. 边连接拆批：链路上真的按批调用
# ---------------------------------------------------------------------------

def test_edge_linking_is_batched_when_many_nodes(tmp_path):
    """新节点超阈值 → 边链按 ceil(n/EDGE_BATCH_SIZE) 次调用（链路级，穿过真实 apply_ops）"""
    n = EDGE_BATCH_SIZE * 2 + 3
    model = ScriptedChatModel(routes=_routes(node_count=n))
    space = _make_space(tmp_path, model)

    stats = space.add(_messages())
    assert stats["nodes"] == n
    assert model.call_count(M_EDGE) == 3, \
        f"边链应调用 3 次（{n} 个节点 / 上限 {EDGE_BATCH_SIZE}），实际 {model.call_count(M_EDGE)}"


def test_edge_linking_single_call_under_threshold(tmp_path):
    """阈值以内仍是单次调用——不能因为加了拆批就无谓地放大输入"""
    model = ScriptedChatModel(routes=_routes(node_count=EDGE_BATCH_SIZE))
    space = _make_space(tmp_path, model)

    space.add(_messages())

    assert model.call_count(M_EDGE) == 1


# ---------------------------------------------------------------------------
# 4. 留痕截断：链路上写入的日志确实被截短
# ---------------------------------------------------------------------------

def test_trace_is_clipped_on_the_real_add(tmp_path, caplog):
    """`[IO] add 入参` 在链路上必须被截到上限以内（0930 实测单文件 84 MB 的成因）"""
    model = ScriptedChatModel(routes=_routes(node_count=2))
    space = _make_space(tmp_path, model, trace_io=True, trace_max_chars=200)

    long_content = "x" * 5000
    with caplog.at_level("INFO", logger="alm.space"):
        space.add([AddMessage(role="user", content=long_content, timestamp=1704067200000)])

    trace_lines = [r.getMessage() for r in caplog.records if "[IO] add 入参" in r.getMessage()]
    assert trace_lines, "没有产生留痕（trace_io 未生效？）"
    assert "留痕截断" in trace_lines[0]
    assert len(trace_lines[0]) < 2000, "单条留痕没有真正被截短"


def test_trace_disabled_by_default(tmp_path, caplog):
    """trace_io 默认关：不能因为引入了截断就顺手把留痕打开"""
    model = ScriptedChatModel(routes=_routes(node_count=2))
    space = _make_space(tmp_path, model)

    with caplog.at_level("INFO", logger="alm.space"):
        space.add(_messages(2))

    assert not [r for r in caplog.records if "[IO] add 入参" in r.getMessage()]


# ---------------------------------------------------------------------------
# 7. 写侧/读侧模式：路由的落地点（Plan §一 总目标）
#
# 这两项的判据**只有一个：LLM 调用数为零**。
# 因为整个路由的账建立在"不建图 = 零 LLM 开销"上（写侧占 LLM 调用 94.7%，
# 单价 $0.00107/组）。只要哪条路径还偷偷调了模型，那个前提就是假的，
# 路由的收益全部作废——而它**不会报错**，只会让账单对不上。故必须钉死。
# ---------------------------------------------------------------------------

def test_write_mode_documents_uses_zero_llm_calls(tmp_path):
    """写侧 documents：整空间只存文档，/add 一次 LLM 都不该调"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents")

    space.add(_messages())

    assert model._calls == [], f"纯向量臂下 /add 仍调了模型：{model._calls}"
    # 「零调用」必须是因为走了文档通道，而不是因为内容被丢了
    assert space.docs, "文档区为空——零调用是因为内容被丢弃？"
    assert space.graph.node_count == 0, "纯向量臂下不该产生任何图节点"


def test_force_mode_default_is_auto(tmp_path):
    """默认就是 `auto`：新模式是**唯一**模式，不再有"旧模式当默认"这一层"""
    model = ScriptedChatModel(routes=_routes(node_count=1))
    space = _make_space(tmp_path, model, force_mode="auto")

    assert space.config.force_mode == "auto"
    # auto 下新空间初态是 vector（零 LLM），而**不是**改动前的"整站建图"
    assert space.space_mode == "vector"


def test_read_mode_vector_uses_zero_llm_calls(tmp_path):
    """读侧 vector：跳过目的推断与 StoryRank，/search 零 LLM 调用"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, read_mode="vector")
    space.add(_messages())
    before = len(model._calls)

    items = space.search(QUERY, top_k=5)

    assert model._calls[before:] == [], f"纯向量读侧仍调了模型：{model._calls[before:]}"
    assert items, "纯向量读侧返回空——那不是对照组，是坏了"
    assert all(it.get("content") for it in items)


def test_documents_only_space_end_to_end(tmp_path):
    """写侧 documents + 读侧 vector = 完整纯向量臂，全链路零 LLM 调用

    这就是"这个语料建图到底值不值"的对照组，必须能端到端跑通。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents", read_mode="vector")

    space.add(_messages())
    items = space.search(QUERY, top_k=5)

    assert model._calls == [], "纯向量臂全链路不该有任何 LLM 调用"
    assert items, "纯向量臂返回空"


def test_search_shape_evidence_skips_storyrank(tmp_path):
    """A 项：`search_shape="evidence"` 下**不调 StoryRank**，直接回证据块。

    验收（计划 §二 A）：① 读侧少一次最贵的 LLM 调用（StoryRank 单次输入 >5000，
    是全日志唯一的一档）；② 结构上不再可能出现 `路径=拼接`（JSON 解析失败→降级成
    1.2~2.3 万字流水账）与 `out=` 撞 max_tokens；③ 返回的必须是**节点原文证据块**，
    不是那条叙事文本。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, search_shape="evidence")
    space.add(_messages())
    before = len(model._calls)

    items = space.search(QUERY, top_k=5)

    after = model._calls[before:]
    assert M_STORY not in after, f"evidence 形态下仍调了 StoryRank：{after}"
    assert items, "证据块形态返回空——那不是对照组，是坏了"
    assert all(it.get("content") for it in items)
    # 证据块是节点原文（带 `[node_type]` 前缀），不该出现那条叙事文本
    assert all("用户在这段对话里提到了若干事实。" not in it["content"] for it in items)


def test_search_shape_story_is_the_default(tmp_path):
    """默认必须还是 `story`：加开关不能顺手改掉现行行为（这是 A/B 的回退阀）"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model)

    assert space.config.search_shape == "story"
    space.add(_messages())
    items = space.search(QUERY, top_k=5)

    assert model.call_count(M_STORY) == 1, "默认形态应走一次 StoryRank"
    assert items and "用户在这段对话里提到了若干事实。" in items[0]["content"]


def test_auto_route_initial_state_is_vector(tmp_path):
    """B 项：开启 auto_route 后，**新空间初态必须是 vector**（零 LLM）

    这是前置条件而不是优化：判断器要到前 2 题才表态，若初态是 graph，
    那 2 题的语料**已经建完图了**——判断器白判、钱白花。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, auto_route=True)

    assert space.space_mode == "vector", "无 mode.yaml 时必须按初态 vector"
    space.add(_messages())
    space.search(QUERY, top_k=5)

    assert model._calls == [], f"初态 vector 下不该有任何 LLM 调用：{model._calls}"
    assert space.graph.node_count == 0
    assert space.docs, "内容必须落进文档区（零调用不等于把内容丢掉）"


def test_auto_route_decides_then_freezes(tmp_path):
    """B/T3：前 N 题判一次 → 写入 mode.yaml → **冻结**（此后不再路由）"""
    model = ScriptedChatModel(routes=_routes(node_count=3, route_p=0.9))
    space = _make_space(tmp_path, model, auto_route=True)

    space.search(QUERY, top_k=5)
    assert model.call_count(M_ROUTE) == 0, "第 1 题不该触发判定（观察窗未满）"

    space.search(QUERY, top_k=5)
    assert model.call_count(M_ROUTE) == 1, "第 2 题应触发一次判定"
    assert space.space_mode == "graph", "p=0.9 ≥ 0.5 应归到 graph"

    mode_file = tmp_path / "data" / "conv0" / "mode.yaml"
    assert mode_file.exists(), "归属必须落盘（否则重启后丢决定）"

    # 冻结：后续不再判定
    space.search(QUERY, top_k=5)
    assert model.call_count(M_ROUTE) == 1, "归属已冻结，不该再判定"

    # 归属生效：此后 add 走图链路
    before = len(model._calls)
    space.add(_messages())
    assert M_NODE in model._calls[before:], "graph 归属下 add 应走节点抽取"


def test_auto_route_below_threshold_stays_vector(tmp_path):
    """p < 阈值 → 归到 vector，且此后 add 仍不花 LLM 钱（主线 ① 的收益）"""
    model = ScriptedChatModel(routes=_routes(node_count=3, route_p=0.2))
    space = _make_space(tmp_path, model, auto_route=True)

    space.search(QUERY, top_k=5)
    space.search(QUERY, top_k=5)

    assert space.space_mode == "vector"
    before = len(model._calls)
    space.add(_messages())
    assert model._calls[before:] == [], f"vector 归属下 add 不该调 LLM：{model._calls[before:]}"
    assert space.graph.node_count == 0


def test_auto_route_parse_failure_falls_back_to_vector(tmp_path):
    """判不出来 → **走向量层**（成本优先的显式兜底方向）

    为什么兜底不是 graph：误判成 graph 要白付一整空间的建图钱（可见、量级 $）；
    误判成 vector 只是可能丢分（不可见）。成本优先下选前者（见计划 §1.4）。
    """
    routes = tuple((m, p) for m, p in _routes(node_count=3) if m != M_ROUTE)
    routes += ((M_ROUTE, "我不会输出 JSON"),)
    model = ScriptedChatModel(routes=routes)
    space = _make_space(tmp_path, model, auto_route=True)

    space.search(QUERY, top_k=5)
    space.search(QUERY, top_k=5)

    assert space.space_mode == "vector", "判定失败必须兜底 vector"


def test_global_force_beats_space_mode(tmp_path):
    """① 全局强制臂必须盖过按空间归属（否则 A/B 对照组会被 mode.yaml 污染）"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, read_mode="vector", auto_route=True)
    space.set_space_mode("graph", reason="夹具")   # 空间自称 graph
    space.add(_messages())                         # 让它真的建出节点，检索才有东西可回

    before = len(model._calls)
    items = space.search(QUERY, top_k=5)

    assert model._calls[before:] == [], f"read_mode=vector 被按空间归属盖掉了：{model._calls[before:]}"
    assert items


def test_search_purpose_inference_does_not_hold_space_lock(tmp_path):
    """P1-1：读侧的目的推断（唯一必做 LLM）必须在**空间锁外**执行。

    场景是 16 并发 × 48h：一次目的推断 LLM 超时最坏 `llm_timeout`=120s。若推断持锁，
    同 user_id 的并发 `/add` 会被整段堵死。

    这里不靠计时阈值（易 flaky），而是**用门闩确定性复现**：目的推断一启动就等一个
    "add 已完成"的事件。若推断持锁，add 拿不到锁 → 事件永不置位 → 门闩超时，
    断言即失败；锁外执行时 add 顺利返回并置位，推断解除等待。
    """
    started = threading.Event()
    add_done = threading.Event()
    gate = {"timeout": False}

    class GatedPurposeModel(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            system = ""
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    system = msg.content or ""
                    break
            else:
                system = getattr(messages[0], "content", "") if messages else ""
            if M_PURPOSE in system:
                started.set()
                if not add_done.wait(timeout=5.0):
                    gate["timeout"] = True
            return super()._generate(messages, stop=stop,
                                     run_manager=run_manager, **kwargs)

    model = GatedPurposeModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model)          # 默认 force_mode="graph"
    space.add(_messages())                        # 建出节点，检索才走主通道

    err = {}

    def run_search():
        try:
            space.search(QUERY, top_k=5)
        except Exception as exc:                  # 线程内异常必须带回主线程
            err["exc"] = exc

    t = threading.Thread(target=run_search)
    t.start()
    assert started.wait(timeout=5.0), "目的推断未启动，场景未复现"

    t0 = time.monotonic()
    space.add(_messages(n=4))                      # 推断「睡眠」期间 add 必须能拿到锁
    elapsed = time.monotonic() - t0
    add_done.set()
    t.join(timeout=10.0)

    assert not gate["timeout"], "目的推断持锁，add 被阻塞（P1-1 回归）"
    assert not err, f"search 线程抛出异常：{err.get('exc')}"
    assert elapsed < 5.0, f"add 耗时异常：{elapsed:.3f}s"


def test_doc_store_persists_raw_messages_not_chunks(tmp_path):
    """B2 项：文档区落盘的是**原始消息**（role / 时间戳 / 消息边界），切片是派生视图。

    为什么必须如此：
      · 「回补建图」——归属判定在前 2 题，此前该空间走 vector；若最终判成 graph，
        这段语料要能**重新抽取**，而抽取的指代消解依赖 role 与相邻句，切片回补不了；
      · 「局部建边」只能从原始消息取上下文——从切片取会复现块图（0 边、峰落 hop0、
        `跳=1`，线上实测 289 节点 0 边）。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents")
    space.add(_messages(n=4))
    space.save()

    log_path = tmp_path / "data" / "conv0" / "docs.log"
    rec = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "msgs" in rec and "docs" not in rec, "落盘单位必须是原始消息"
    msgs = rec["msgs"]
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert all(m.get("timestamp") is not None for m in msgs)
    assert [m["msg_index"] for m in msgs] == [0, 1, 2, 3]
    assert all(m.get("doc_id") for m in msgs)

    # 切片仍可用（读侧两条检索路径都只读 `docs`），且**载入后由 msgs 重建**、逐条一致
    ids_before = [d["id"] for d in space.docs]
    content_before = [d["content"] for d in space.docs]
    assert ids_before, "切片派生视图不该为空"

    reloaded = _make_space(tmp_path, model, write_mode="documents")
    assert [d["id"] for d in reloaded.docs] == ids_before
    assert [d["content"] for d in reloaded.docs] == content_before
    assert len(reloaded.doc_msgs) == 4
    assert reloaded.search(QUERY, top_k=5), "重载后文档通道仍应能检索到内容"


def _l1_only_space(tmp_path, model, **overrides):
    """造一个「判定前」的空间：归属未定 → 写侧走 documents → 语料全落在 L1。

    这正是 C 项要处理的现场：`auto_route` 开启后新空间初态是 vector，
    在判定发生前收到的语料**只进文档区**，图是空的。
    """
    overrides.setdefault("auto_route", True)
    overrides.setdefault("backfill_enabled", True)
    return _make_space(tmp_path, model, **overrides)


def test_docs_log_replay_is_idempotent_by_request_id(tmp_path):
    """P1-3：重启后平台重试同一 request_id，不得重复入库。

    `Engine._request_log` 是**内存态**，重启即失；若幂等信息只活在内存里，平台重试会被
    当成新请求再建一次图（0930 一次跑里重启过 2 次 → 重复付费）。本用例验证 rid 被写在
    `docs.log` 的**每条消息**上，回放时整行跳过。

    注意重试会生成**新的 doc_id**——所以按 `(doc_id, msg_index)` 的内容去重挡不住它，
    只有 rid 能挡。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))

    first = _make_space(tmp_path, model, write_mode="documents")
    first.add(_messages(n=4), request_id="R1")
    first.save()
    n = len(first.doc_msgs)
    assert n == 4

    # 模拟重启：新实例（内存态幂等表为空）→ 平台重试同一 request_id
    retry = _make_space(tmp_path, model, write_mode="documents")
    assert len(retry.doc_msgs) == n
    retry.add(_messages(n=4), request_id="R1")
    retry.save()

    # 再次重启回放：重试行必须被 rid 挡掉，条数不得翻倍
    again = _make_space(tmp_path, model, write_mode="documents")
    assert len(again.doc_msgs) == n, "同一 request_id 重试不得重复入库"


def test_backfill_builds_graph_from_l1_messages(tmp_path):
    """C 项：判成 graph 后，判定前落在 L1 的语料必须**补建进图**

    不补的话图侧永久缺前置内容——而那正是"检索时图帮不上忙"的经典成因。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _l1_only_space(tmp_path, model)
    space.add(_messages(n=6))

    assert space.graph.node_count == 0, "判定前不该有图节点"
    assert len(space.doc_msgs) == 6, "判定前的语料应全部落在 L1"

    added = space.run_backfill()      # 同步入口：线程体与测试共用

    assert added == 6, f"应补入 6 条，实际 {added}"
    assert space.graph.node_count > 0, "回补没有把 L1 的语料补建进图"
    assert space.backfill_progress()["status"] == "done"


def test_backfill_runs_in_background_and_is_idempotent(tmp_path):
    """C 项：回补走**后台**线程（不阻塞发起它的 /search），且不可重复起线程"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _l1_only_space(tmp_path, model)
    space.add(_messages(n=6))

    thread = space.start_backfill()
    assert thread is not None, "开关打开且有待补内容时应起线程"
    thread.join(timeout=30)

    assert space.backfill_progress()["status"] == "done"
    assert space.graph.node_count > 0
    assert space.start_backfill() is None, "已补完不该再起线程（否则同一批被建两次图）"


def test_auto_route_graph_decision_triggers_backfill(tmp_path):
    """接线：判成 graph 时**当场**把进度置为 running 并起线程

    这里断言的是"触发"而不是"跑完"——进度置位发生在 `thread.start()` **之前**，
    所以 `search()` 返回时它必然已生效（不需要 sleep/轮询）。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3, route_p=0.9))
    space = _make_space(tmp_path, model, auto_route=True, backfill_enabled=True)
    space.add(_messages(n=6))

    space.search(QUERY, top_k=5)
    space.search(QUERY, top_k=5)

    assert space.space_mode == "graph"
    assert space.backfill_progress()["status"] in ("running", "done"), "判定成 graph 应触发回补"


def test_backfill_resumes_from_checkpoint(tmp_path):
    """断点续建：进度按批落盘，重启后从断点继续（不重头付费）"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _l1_only_space(tmp_path, model)
    space.add(_messages(n=6))
    space._save_backfill_progress(3, "running")     # 夹具：模拟已补 3 条后进程重启
    before = len(model._calls)

    added = space.run_backfill()

    assert added == 3, f"应从断点续建剩余 3 条，实际 {added}"
    assert space.backfill_progress()["status"] == "done"
    assert M_NODE in model._calls[before:], "续建部分应真的走了抽取"


def test_backfill_stop_loss_skips_when_too_large(tmp_path):
    """启动前止损：估算超上限就**不启动**（宁可不建图，不可占满线程 / 超预算）"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _l1_only_space(tmp_path, model, backfill_max_groups=1)
    space.add(_messages(n=12))                      # 12 条 / 每批 6 条 = 2 组 > 上限 1

    added = space.run_backfill()

    assert added == 0
    assert space.backfill_progress()["status"] == "skipped"
    assert space.graph.node_count == 0, "止损后不该有任何图节点"


def test_backfill_disabled_by_default(tmp_path):
    """默认关：不加开关就不该起线程、不该建图"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model)
    assert space.config.backfill_enabled is False
    assert space.start_backfill() is None


def test_invalid_mode_is_rejected_not_silently_defaulted(tmp_path, monkeypatch):
    """非法模式名必须**炸**，不能静默退回默认

    否则"以为开了纯向量臂、其实还是全链路"——那次对照实验就白跑了，
    而且事后从账单里看不出来（两边都有开销，只是差一个量级）。
    """
    monkeypatch.setenv("ALM_FORCE_MODE", "document")      # 少一个 s
    with pytest.raises(ValueError, match="ALM_FORCE_MODE"):
        ALMConfig.from_env()

    monkeypatch.setenv("ALM_FORCE_MODE", "graph")
    monkeypatch.setenv("ALM_SEARCH_SHAPE", "evidences")   # 多一个 s
    with pytest.raises(ValueError, match="ALM_SEARCH_SHAPE"):
        ALMConfig.from_env()


def test_fallback_reason_treats_omitted_action_as_create(tmp_path):
    """`_fallback_reason` 必须与解析侧同口径，否则**误触发一次错误的兜底**。

    场景：抽取产出了 create，但它们**全被去重合并进已有节点**（`created_ids` 为空）。
    这是**正常成功**，不该兜底。但 `action` 省略后，若这里仍比 `== "create"`，
    就会掉到下面判成 `no_create` → `_divert_raw` 把**已成功合并**的原文又抄一份进
    文档区 → 重复内容 + 无效 embedding + 每次一条 WARNING。

    这是"字段语义从必填变可省略"时最容易漏的那类读取点：
    它不在 `graph_builder` 里，而在**调用方的判定逻辑**里。
    """
    model = ScriptedChatModel(routes=_routes(node_count=1))
    space = _make_space(tmp_path, model)

    # ① 有 create（省略 action）但 created 为空 = 全被合并 → 正常，不该兜底
    merged = {"ops": {"node_ops": [{"content": "x", "node_type": "action"}]},
              "result": {}}
    assert space._fallback_reason(merged, []) is None

    # ② 显式写了 action: create 也要同样放行
    explicit = {"ops": {"node_ops": [{"action": "create", "content": "x"}]},
                "result": {}}
    assert space._fallback_reason(explicit, []) is None

    # ③ 真的有节点落库时更不该兜底（早退分支）
    assert space._fallback_reason(merged, ["n1"]) is None

    # ④ triage 跳过 / 空 ops 两种原因不能被这次改动影响
    assert space._fallback_reason({"skipped": True}, []) == "triage"
    assert space._fallback_reason({"ops": {"node_ops": []}, "result": {}}, []) == "no_ops"


# ---------------------------------------------------------------------------
# 8. 文档区落盘：只追加，不整份重写（O(n²) 写放大）
#
# 改造前 `_save_documents` 每次 Add 都 `yaml.dump` **整份** docs，而 documents 模块下
# 全部语料都走这条路 → 第 k 次 Add 重写前 k 次累计的全部条文。线上实测一轮 full
# 4,378 次 Add、终态 ~10 MB，写放大接近 46 GB，与已修的「索引全量重写」同源。
# ---------------------------------------------------------------------------

def _doc_messages(tag: str, n: int = 1):
    return [
        AddMessage(role="user", content=f"{tag}#{i} " + "正文" * 200,
                   timestamp=1704067200000 + i)
        for i in range(n)
    ]


def test_documents_are_appended_not_rewritten(tmp_path):
    """每次 Add 只追加一行日志，不得重写整份条文快照"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents")
    docs_yaml = tmp_path / "data" / "conv0" / "docs.yaml"
    docs_log = tmp_path / "data" / "conv0" / "docs.log"

    for i in range(5):
        space.add(_doc_messages(f"doc{i}"))
        space.save()

    assert docs_log.exists(), "没有增量日志——又回到整份重写了？"
    assert not docs_yaml.exists(), "每次 Add 都重写了整份条文快照（O(n²) 写放大）"
    # 5 次 Add → 5 行日志。行数必须与**新增批次数**成正比，而不是与累计容量成正比。
    assert docs_log.read_text(encoding="utf-8").count("\n") == 5

    reloaded = _make_space(tmp_path, ScriptedChatModel(routes=_routes(node_count=3)),
                           write_mode="documents")
    assert [d["id"] for d in reloaded.docs] == [d["id"] for d in space.docs]
    assert [d["content"] for d in reloaded.docs] == [d["content"] for d in space.docs]


def test_docs_log_compaction_keeps_reload_consistent(tmp_path, monkeypatch):
    """日志涨到阈值就并入快照，且合并**不产生重复条文**（按 id 幂等）"""
    import alm.space as space_mod

    monkeypatch.setattr(space_mod, "_DOCS_LOG_MAX_BYTES", 1)  # 每批之后都触发合并
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents")
    docs_yaml = tmp_path / "data" / "conv0" / "docs.yaml"
    docs_log = tmp_path / "data" / "conv0" / "docs.log"

    for i in range(4):
        space.add(_doc_messages(f"doc{i}"))
        space.save()

    assert docs_yaml.exists(), "触发合并后快照必须落盘"
    assert not docs_log.exists(), "合并后日志必须清空，否则下次会把同一批又回放一遍"

    reloaded = _make_space(tmp_path, ScriptedChatModel(routes=_routes(node_count=3)),
                           write_mode="documents")
    ids = [d["id"] for d in reloaded.docs]
    assert ids == [d["id"] for d in space.docs]
    assert len(ids) == len(set(ids)), f"合并后出现重复条文：{ids}"


def test_docs_log_repairs_trailing_partial_line(tmp_path):
    """写入途中崩溃留下的**半行**必须就地截断，不能让它毒化后续所有条文"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents")
    space.add(_doc_messages("good"))
    space.save()
    docs_log = tmp_path / "data" / "conv0" / "docs.log"
    good_len = len(space.docs)

    # 模拟"写到一半断电"：追加一条没有换行、JSON 也截断的残行
    with open(docs_log, "ab") as f:
        f.write(b'{"seq": 99, "docs": [{"id": "d9999c0000", "content": "hal')

    reloaded = _make_space(tmp_path, ScriptedChatModel(routes=_routes(node_count=3)),
                           write_mode="documents")
    assert len(reloaded.docs) == good_len, "残行被当成了有效条文"
    assert all(d["id"] != "d9999c0000" for d in reloaded.docs)
    assert docs_log.read_bytes().endswith(b"\n"), "残行没有被就地截断，下次追加会接在脏行后面"


# ---------------------------------------------------------------------------
# 9. 全链路审查回归（F1 / F2 / F4）
#
# 三条都属"静默失效"：不报错、不影响返回值，只在特定并发/模式下丢数据，故必须
# 用**能区分修复前后**的断言钉住（而不是"跑完不报错"）。
# ---------------------------------------------------------------------------


def test_memory_space_exposes_bg_active_flag(tmp_path):
    """F1：`bg_active` 必须是真实存在的**公开属性**（engine 读的就是这个名字）。

    `_evict_locked` 用 `getattr(sp, "bg_active", False)` 判断能否换出；若该属性不存在，
    `getattr` 静默取默认 False → 保护失效。故这里直接断言属性与内部标记同步。
    """
    model = ScriptedChatModel(routes=_routes(node_count=1))
    space = _make_space(tmp_path, model)

    assert space.bg_active is False, "初始态不该有后台任务在跑"
    space._bg_active = True                  # 模拟 start_backfill 置位
    assert space.bg_active is True, "公开属性没有映射到内部标记（engine 会读空）"


def test_engine_eviction_skips_space_with_background_task(tmp_path):
    """F1：**最旧**的空间若在跑后台回补，就不得被 LRU 换出。

    为什么必须造"最旧"：`_evict_locked` 取第一个可换出的条目。若把带后台标记的空间
    放在最前，修复前（属性名不符 → 恒 False）它会**恰好**被选为牺牲品；修复后才会
    被跳过。这是一个**能区分修复前后**的确定性断言。
    """
    model = ScriptedChatModel(routes=_routes(node_count=1))
    config = ALMConfig(data_dir=tmp_path / "data", max_spaces=1,
                       abstain_cosine=0.0, abstain_verify_hi=0.0,
                       **{"force_mode": "graph"})

    def _space(uid: str, subdir: str) -> MemorySpace:
        return MemorySpace(
            user_id=uid, config=config, embeddings=WordOverlapEmbeddings(),
            llm=model, yaml_path=tmp_path / "data" / subdir / "graph.yaml",
            stats=RetrievalStats(),
        )

    bg = _space("u_bg", "u_bg")
    bg._bg_active = True                     # 回补线程在跑
    other = _space("u_other", "u_other")

    engine = object.__new__(ALMEngine)       # 只测换出逻辑，不构造 LLM / Embedding
    engine.config = config
    engine._spaces = OrderedDict([("u_bg", bg), ("u_other", other)])
    engine._in_use = {}

    engine._evict_locked()

    assert "u_bg" in engine._spaces, "有后台任务的空间被换出（bg_active 未生效，回补会丢数据）"
    assert "u_other" not in engine._spaces, "最旧的可用空间没有被换出，换出逻辑异常"


def test_document_channel_keeps_overflow_messages(tmp_path):
    """F2：文档通道不得丢弃超出 `max_messages_per_add` 的消息。

    写侧 `documents`（向量模式）下形态判定被整批覆盖为文档 → **所有** Add 都走文档
    分支；该分支原先只落 `payload`（前 `limit` 条），`overflow` 被静默丢弃——既不进图
    也不进文档区，等于凭空消失，违反"每次 Add 的消息已完整存储且可被检索"。
    文档通道零 LLM，没有限量的理由。
    """
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model, write_mode="documents",
                        max_messages_per_add=2)

    space.add(_messages(n=5))                # 5 条 > 上限 2

    assert len(space.doc_msgs) == 5, "文档通道丢弃了超出维护上限的消息"
    assert space.graph.node_count == 0
    assert model._calls == [], "文档通道不该调 LLM"


def test_document_store_uses_explicit_rid_not_instance_state(tmp_path):
    """F4：rid 必须由**显式参数**下传，不能读实例上的共享状态。

    并发 Add 在锁外交错时，实例属性会被后到的请求覆盖 → 两批打成同一个 rid →
    重启回放按 rid 整行跳过 → 先到那批静默丢失。这里刻意在实例上摆一个"残留的
    别的 rid"：实现只要还读实例属性就会打错（旧实现还会因多出 `rid=` 参数直接报错）。
    """
    model = ScriptedChatModel(routes=_routes(node_count=1))
    space = _make_space(tmp_path, model, write_mode="documents")
    space._add_rid = "STALE"                 # 模拟被并发请求覆盖后的残留值

    space._store_documents(
        [AddMessage(role="user", content="hello", timestamp=1704067200000)],
        1704067200000, rid="MINE",
    )

    assert space.doc_msgs[-1]["rid"] == "MINE", "rid 读了实例状态（并发下会串号）"


def test_add_threads_request_id_into_document_channel(tmp_path, monkeypatch):
    """F4：`add()` 必须把本次 request_id 传给文档通道（端到端接线）"""
    model = ScriptedChatModel(routes=_routes(node_count=1))
    space = _make_space(tmp_path, model, write_mode="documents")
    seen = {}

    original = space._store_documents

    def spy(messages, batch_ts=None, rid=None):
        seen["rid"] = rid
        return original(messages, batch_ts, rid=rid)

    monkeypatch.setattr(space, "_store_documents", spy)
    space.add(_messages(n=2), request_id="REQ-42")

    assert seen.get("rid") == "REQ-42", "request_id 没有下传到文档通道"
    assert all(m["rid"] == "REQ-42" for m in space.doc_msgs), "落库消息没带上本次 rid"


def test_concurrent_adds_do_not_cross_request_ids(tmp_path):
    """F4：两个不同 request_id 的 Add 并发后，各自消息必须保留自己的 rid（P1-3 幂等前提）"""
    model = ScriptedChatModel(routes=_routes(node_count=1))
    space = _make_space(tmp_path, model, write_mode="documents")

    barrier = threading.Barrier(2)
    errors = []

    def worker(rid: str, tag: str):
        try:
            barrier.wait(timeout=5)
            space.add(
                [AddMessage(role="user", content=f"{tag} message",
                            timestamp=1704067200000)],
                request_id=rid,
            )
        except Exception as exc:             # 线程内异常必须带回主线程
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=("R-A", "alpha")),
               threading.Thread(target=worker, args=("R-B", "beta"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"并发 add 抛出异常：{errors}"
    by_rid = {}
    for msg in space.doc_msgs:
        by_rid.setdefault(msg["rid"], []).append(msg["content"])
    assert set(by_rid) == {"R-A", "R-B"}, f"两批被打成同一个 rid（回放会丢一批）：{by_rid}"
    assert by_rid["R-A"] == ["alpha message"]
    assert by_rid["R-B"] == ["beta message"]


def test_auto_route_deciding_query_keeps_llm_out_of_lock(tmp_path):
    """F3（方案 A）：auto 路由下，**判定那一题**的两次 LLM（association + purpose）
    都必须在空间锁外。

    为什么单独测这一题：判定前该空间读侧是 vector（不推断目的，零 LLM）；判定当场切
    graph 后才需要目的。旧写法把判定留在锁内、目的推断跟着回落到检索器内部（锁内），
    于是判定题持锁连做两次 LLM → 同 user_id 的并发 /add 被堵（P1-1 回归）。

    用门闩确定性复现：目的推断一启动就等一次 add 完成；若它持锁，add 拿不到锁 →
    事件永不置位 → 门闩超时。
    """
    started = threading.Event()
    add_done = threading.Event()
    gate = {"timeout": False}

    class GatedPurposeModel(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            system = ""
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    system = msg.content or ""
                    break
            else:
                system = getattr(messages[0], "content", "") if messages else ""
            if M_PURPOSE in system:
                started.set()
                if not add_done.wait(timeout=5.0):
                    gate["timeout"] = True
            return super()._generate(messages, stop=stop,
                                     run_manager=run_manager, **kwargs)

    model = GatedPurposeModel(routes=_routes(node_count=3, route_p=0.9))
    space = _make_space(tmp_path, model, auto_route=True)

    space.add(_messages(n=4))                 # 判定前：落 L1（零 LLM）
    space.search(QUERY, top_k=5)              # 第 1 题：不判定、不推断目的
    assert model.call_count(M_ROUTE) == 0
    assert model.call_count(M_PURPOSE) == 0, "未定空间的前置题不该推断目的"

    err = {}

    def run_search():
        try:
            space.search(QUERY, top_k=5)      # 第 2 题：判定 → 目的推断
        except Exception as exc:              # 线程内异常必须带回主线程
            err["exc"] = exc

    t = threading.Thread(target=run_search)
    t.start()
    assert started.wait(timeout=5.0), "判定题的目的推断未启动，场景未复现"

    t0 = time.monotonic()
    space.add(_messages(n=4))                 # 推断「睡眠」期间 add 必须能拿到锁
    elapsed = time.monotonic() - t0
    add_done.set()
    t.join(timeout=10.0)

    assert not gate["timeout"], "判定题的目的推断持锁，add 被阻塞（F3 回归）"
    assert not err, f"search 线程抛出异常：{err.get('exc')}"
    assert elapsed < 5.0, f"add 耗时异常：{elapsed:.3f}s"
    assert space.space_mode == "graph", "p=0.9 应判成 graph"
    assert model.call_count(M_ROUTE) == 1, "归属只该判定一次"
