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

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from alm.config import ALMConfig
from alm.contract import AddMessage
from alm.space import MemorySpace, RetrievalStats
from alm.tokens import TokenMeter
from dba_pipeline.extraction.dba import EDGE_BATCH_SIZE

# system prompt 的特征词 → 该链的预设响应（与调用顺序无关）
M_NODE = "节点抽取器"
M_EDGE = "边连接器"
M_TRIAGE = "值得长期记忆的新事实"
M_PURPOSE = "对话意图分析助手"
M_STORY = "记忆故事整理助手"


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


def _routes(node_count: int):
    return (
        (M_TRIAGE, "NEEDED"),
        (M_NODE, _node_payload(node_count)),
        (M_EDGE, json.dumps({"edge_ops": []})),
        (M_PURPOSE, json.dumps({"status": "ok", "purposes": PURPOSES,
                                "condense": False, "state": False, "temporal": False})),
        (M_STORY, json.dumps({"story": "用户在这段对话里提到了若干事实。", "used": 1},
                             ensure_ascii=False)),
    )


def _messages(n: int = 6):
    return [
        AddMessage(role="user" if i % 2 == 0 else "assistant",
                   content=f"line {i} of the conversation", timestamp=1704067200000 + i)
        for i in range(n)
    ]


def _make_space(tmp_path, model, **overrides):
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


def test_write_mode_graph_is_the_default(tmp_path):
    """默认必须还是 graph：加开关不能顺手改掉现行行为"""
    model = ScriptedChatModel(routes=_routes(node_count=3))
    space = _make_space(tmp_path, model)

    assert space.config.write_mode == "graph"
    space.add(_messages())
    assert model.call_count(M_NODE) == 1
    assert space.graph.node_count > 0


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


def test_invalid_mode_is_rejected_not_silently_defaulted(tmp_path, monkeypatch):
    """非法模式名必须**炸**，不能静默退回默认

    否则"以为开了纯向量臂、其实还是全链路"——那次对照实验就白跑了，
    而且事后从账单里看不出来（两边都有开销，只是差一个量级）。
    """
    monkeypatch.setenv("ALM_WRITE_MODE", "document")      # 少一个 s
    with pytest.raises(ValueError, match="ALM_WRITE_MODE"):
        ALMConfig.from_env()

    monkeypatch.setenv("ALM_WRITE_MODE", "graph")
    monkeypatch.setenv("ALM_READ_MODE", "vectors")        # 多一个 s
    with pytest.raises(ValueError, match="ALM_READ_MODE"):
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
