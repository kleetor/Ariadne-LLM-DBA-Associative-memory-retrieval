# SPDX-License-Identifier: AGPL-3.0-only

"""按 user_id 隔离的记忆空间。

ALM 以 user_id 为唯一检索作用域，且 `MemoryGraph.to_dict()` 的 YAML 序列化
**不保留节点 metadata**，因此这里采用「每个 user_id 一份独立图谱 + 独立向量索引」
的隔离方式：不做任何核心改动即可保证「禁止跨 user_id 返回记忆」。

隔离代价是空间数量随 user_id 增长，由上层 ALMEngine 用 LRU 控制内存驻留数。
"""

import hashlib
import logging
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.core.path_tracker import PathTracker
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph

from alm.config import ALMConfig
from alm.contract import AddMessage
from alm.judge import RelevanceJudge

try:
    from dba_pipeline.embedding.store import VectorStore
    from dba_pipeline.extraction.dba import MemoryDBA
    from dba_pipeline.extraction.graph_builder import GraphBuilder
    from dba_pipeline.llm.inference import InferenceEngine
    from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever

    HAS_DEPS = True
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确报错
    HAS_DEPS = False
    _IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

# 兜底节点单条内容的上限（与 embedding 端的截断预算保持一致）
_FALLBACK_CONTENT_LIMIT = 2000


def _cosine(a, b) -> float:
    """余弦相似度；任一输入缺失或为零向量时返回 0（仅服务于弃权判定）"""
    if a is None or b is None:
        return 0.0
    vec_a = np.asarray(a, dtype=float)
    vec_b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class RetrievalStats:
    """检索可观测性：累计每次 Search 的「候选最高余弦」与弃权次数。

    存在的理由是**标定弃权阈值**：ALM 真实数据不可离线观测，但平台每次 smoke 都会打
    我们的 `/search`，把这些请求的 best_cos 记下来（只记数值，**不记 query 正文**，
    符合平台的合规要求），跑完一轮就能看到真实查询实际落在什么区间：

    - 若 p10 已经高于阈值 → 阈值偏低，漏弃权会伤 H（无记忆时没返回空）；
    - 若 p90 低于阈值 → 阈值偏高，过度弃权会成片伤 A（有记忆也返回空）；
    - 若阈值正好落在分布中间 → 必须重标定，此时任何一个方向的偏移都在扣分。

    Add / Search 在线程池中并发执行，故计数加锁。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.searches = 0
        self.abstained = 0
        self._cosines: List[float] = []

    def record_search(self, best_cos: float) -> None:
        """每次检索都调用（含最终未弃权的），保证余弦分布完整"""
        with self._lock:
            self.searches += 1
            self._cosines.append(round(float(best_cos), 4))

    def record_abstain(self) -> None:
        """确认弃权时调用。

        与 record_search 分开，是因为两段式判定的弃权点落在检索**之后**
        （模糊带要等 top-N 排好才能交给 LLM 判定），无法在记余弦时一并确定。
        """
        with self._lock:
            self.abstained += 1

    def last_cosine(self) -> Optional[float]:
        """最近一次检索的候选最高余弦；尚未记录过则返回 None。

        供标定脚本按查询取值用（串行发起检索即可一一对应）。
        """
        with self._lock:
            return self._cosines[-1] if self._cosines else None

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        pos = int(round(q * (len(ordered) - 1)))
        return ordered[max(0, min(pos, len(ordered) - 1))]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            searches = self.searches
            abstained = self.abstained
            cosines = list(self._cosines)
        return {
            "searches": searches,
            "abstained": abstained,
            "abstain_rate": round(abstained / searches, 4) if searches else 0.0,
            "min": round(min(cosines), 4) if cosines else 0.0,
            "p10": round(self._quantile(cosines, 0.10), 4),
            "p25": round(self._quantile(cosines, 0.25), 4),
            "p50": round(self._quantile(cosines, 0.50), 4),
            "p75": round(self._quantile(cosines, 0.75), 4),
            "p90": round(self._quantile(cosines, 0.90), 4),
            "max": round(max(cosines), 4) if cosines else 0.0,
        }

    def report(self) -> str:
        """人类可读的分布汇总（进程退出时打印到 stderr）"""
        data = self.summary()
        lines = [
            "=" * 60,
            "[ALM] 检索余弦分布（标定 ALM_ABSTAIN_COS 用）",
            "=" * 60,
            f"  检索次数 : {data['searches']}",
            f"  弃权次数 : {data['abstained']}（{data['abstain_rate'] * 100:.1f}%）",
            f"  best_cos : min={data['min']}  p10={data['p10']}  p25={data['p25']}  "
            f"p50={data['p50']}",
            f"             p75={data['p75']}  p90={data['p90']}  max={data['max']}",
            "=" * 60,
        ]
        return "\n".join(lines)


class MemorySpace:
    """单个 user_id 的完整记忆栈：图谱 + 向量 + DBA 维护 + PAR 检索"""

    def __init__(
        self,
        user_id: str,
        config: ALMConfig,
        embeddings,
        llm,
        yaml_path: Path,
        stats: Optional[RetrievalStats] = None,
    ):
        if not HAS_DEPS:
            raise RuntimeError(
                f"ALM 记忆链路依赖未安装（{_IMPORT_ERROR}）；请先执行 pip install -e ."
            )

        self.user_id = user_id
        self.config = config
        self.yaml_path = Path(yaml_path)
        self.stats = stats
        # 串行化单个用户空间内的图谱写入与 YAML 落盘
        self._lock = threading.RLock()
        # 自上次落盘以来图谱是否被改动过。add() 置位、save() 成功后清位。
        # 作用不只是省一次写盘：留存储存期清理（alm/cleanup.py）按 mtime 判定
        # 空间最后一次被写入的时间，而 engine.close() 会对**全部驻留空间**调 save()。
        # 若无条件重写，停服这一步就会把每个空间的 mtime 刷成停服时刻，
        # 使「先停服再清理」的既定流程永远判定不到过期空间（静默空转）。
        self._dirty = False

        self.graph = (
            load_graph(str(self.yaml_path)) if self.yaml_path.exists() else MemoryGraph()
        )
        self.vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
        if not self._load_index():
            self._reindex()

        self.builder = GraphBuilder(
            graph=self.graph,
            vector_store=self.vector_store,
            orphan_threshold=self.config.orphan_threshold,
            lock=self._lock,
        )
        self.dba = MemoryDBA(
            llm=llm, graph=self.graph, vector_store=self.vector_store, graph_builder=self.builder
        )
        self.retriever = PurposeDrivenRetriever(
            llm=llm,
            embeddings=embeddings,
            graph=self.graph,
            vector_store=self.vector_store,
            inference=InferenceEngine(llm),
            path_tracker=PathTracker(),
        )
        # 弃权模糊带的二次判定（仅当 config.abstain_verify_hi > 0 时被调用）
        self.judge = RelevanceJudge(
            llm,
            top_n=config.abstain_judge_top_n,
            max_chars=config.abstain_judge_max_chars,
        )

    # ---- 初始化 ----

    def _reindex(self):
        """从图谱内容重建向量索引。

        进程重启后不依赖 FAISS 落盘，避免索引与图谱不同步；
        代价是冷启动需重新嵌入一次，空间在内存中以 LRU 缓存摊薄。
        """
        node_ids = [
            nid for nid in self.graph.graph.nodes()
            if self.graph.graph.nodes[nid].get("content")
        ]
        if not node_ids:
            return
        contents = [self.graph.graph.nodes[nid].get("content", "") for nid in node_ids]
        self.vector_store.add_memories(node_ids, contents)
        logger.info("空间 %s 重建向量索引: %d 条", self._masked_id(), len(node_ids))

    # ---- 向量索引持久化 ----
    # 无索引落盘时，每次空间载入（LRU 换出后再访问、或进程重启）都要把整图重新嵌入
    # 一遍——API embedding 下这是成百上千次外部调用。改为随 YAML 一起落盘索引，
    # 并用「节点集合指纹」校验一致性，不一致才回退重建。

    def _index_paths(self):
        """索引与其指纹文件的路径。

        必须让每个空间独占一个子目录：VectorStore.save 会把向量缓存写到索引路径的
        同级目录，若直接放在 data_dir 下，多个空间会互相覆盖。
        """
        base = self.yaml_path.parent / (self.yaml_path.stem + ".index")
        return base / "faiss", base / "stamp.yaml"

    def _index_digest(self) -> Dict[str, Any]:
        node_ids = sorted(str(n) for n in self.graph.graph.nodes())
        return {
            "count": len(node_ids),
            "digest": hashlib.sha1("|".join(node_ids).encode("utf-8")).hexdigest(),
        }

    def _load_index(self) -> bool:
        """复用落盘索引；与当前图谱不一致时返回 False 以触发重建"""
        index_path, stamp_path = self._index_paths()
        try:
            if not index_path.exists() or not stamp_path.exists():
                return False
            with open(stamp_path, "r", encoding="utf-8") as f:
                stamp = yaml.safe_load(f) or {}
            if stamp != self._index_digest():
                logger.info("空间 %s 索引与图谱不一致，改为重建", self._masked_id())
                return False
            self.vector_store.load(str(index_path), embeddings=self.vector_store.embeddings)
            logger.info("空间 %s 复用落盘索引: %d 条", self._masked_id(), stamp.get("count"))
            return True
        except Exception as exc:
            logger.warning("空间 %s 索引加载失败，改为重建: %s", self._masked_id(), exc)
            return False

    def _save_index(self):
        """落盘 FAISS 索引与向量缓存（失败不影响主流程，仅降级为下次重建）"""
        index_path, stamp_path = self._index_paths()
        try:
            self.vector_store.save(str(index_path))
            stamp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stamp_path, "w", encoding="utf-8") as f:
                yaml.dump(self._index_digest(), f, allow_unicode=True)
        except Exception as exc:
            logger.warning("空间 %s 索引保存失败: %s", self._masked_id(), exc)

    # ---- 写入 ----

    def add(self, messages: List[AddMessage]) -> Dict[str, int]:
        """同步写入一批消息：DBA 维护完成后才返回（对齐 ALM 契约）"""
        limit = self.config.max_messages_per_add
        payload = messages[:limit]
        overflow = messages[limit:]
        conversation = self._format_conversation(payload)
        batch_ts = self._batch_timestamp(payload)

        with self._lock:
            # 先置位再改图：maintain 中途抛错时图谱可能已被部分修改，
            # 此时保持 dirty 让退出路径仍会落盘，宁多写一次也不丢内容。
            self._dirty = True
            result = self.dba.maintain(conversation, timestamp=batch_ts)
            created = (result.get("result") or {}).get("created_ids") or []
            if self._should_fallback(result, created):
                created = self._store_raw_fallback(payload, batch_ts)
            if overflow:
                # 条数上限只用于控制 LLM 成本，**不等于可以丢弃内容**：契约要求每次
                # Add 的消息已完整存储。超出部分改走无需 LLM 的兜底落库。
                logger.warning(
                    "空间 %s 单次 Add %d 条超过维护上限 %d，超出 %d 条改走兜底落库",
                    self._masked_id(), len(messages), limit, len(overflow),
                )
                created = created + self._store_raw_fallback(overflow, batch_ts)

        logger.info(
            "空间 %s 写入完成: messages=%d nodes=%d",
            self._masked_id(), len(messages), len(created),
        )
        return {"messages": len(messages), "nodes": len(created)}

    @staticmethod
    def _batch_timestamp(messages: List[AddMessage]) -> Optional[int]:
        """取本批消息的代表时间（最后一个携带的时间戳，Unix 毫秒）。

        ALM 的 Add 以批次为单位，节点抽取不区分单条消息，因此时间戳按批次注入；
        取最后一个非空时间戳，等价于「本批最新事件时间」，与时间序上升的会话一致。
        """
        for msg in reversed(messages):
            if msg.timestamp is not None:
                return msg.timestamp
        return None

    def _should_fallback(self, result: Dict[str, Any], created: List[str]) -> bool:
        """是否需要兜底落库。

        ALM 要求每次 Add 的消息「已完整存储且可被检索」，而 Ariadne 的
        triage 会在判定无维护价值时整段跳过、LLM 也可能抽不出节点——
        这两种情况都必须兜底，否则该会话记忆永久缺失。

        但「LLM 发起了 create 却一个都没落地」不属于这两种情况：那说明这些内容
        命中了语义去重，即图中已有等价记忆。此时再兜底是有害的——兜底刻意把去重
        阈值提到 2.0（相当于关闭去重），会把同一批内容以原文形式重复写入。平台重试
        同一次 Add 时正好走这条链：去重命中 → created 为空 → 兜底关去重 → 必然重复。
        """
        if result.get("skipped"):
            return True
        if created:
            return False
        node_ops = (result.get("ops") or {}).get("node_ops") or []
        if not node_ops:
            return True
        if any(op.get("action") == "create" for op in node_ops):
            return False
        return not any(op.get("action") in ("update", "deprecate", "fix_type") for op in node_ops)

    def _store_raw_fallback(
        self, messages: List[AddMessage], batch_ts: Optional[int] = None
    ) -> List[str]:
        """兜底落库：把消息原文逐条存为可检索节点

        原文兜底必须绕过语义去重，否则与既有记忆语义相近的消息会被判为重复而整条
        丢弃——而 add() 仍返回 200，形成静默丢数据（实测：同批「随便聊聊第N段」
        只有第 1 条落库）。调用方 add() 已持有 space 锁，builder 共享同一把锁，
        因此这里的阈值临时调整不会被并发观察到。
        """
        limit = self.config.fallback_max_nodes
        if len(messages) > limit:
            logger.warning(
                "空间 %s 兜底落库条数超上限: %d → %d 条，超出部分未存储",
                self._masked_id(), len(messages), limit,
            )

        ops = []
        clipped = 0
        for msg in messages[:limit]:
            text = msg.content.strip()
            if not text:
                continue
            raw = f"{msg.role}: {text}"
            if len(raw) > _FALLBACK_CONTENT_LIMIT:
                clipped += 1
            ops.append({
                "action": "create",
                "content": raw[:_FALLBACK_CONTENT_LIMIT],
                "node_type": NodeType.THING.value,
            })
        if clipped:
            logger.warning(
                "空间 %s 兜底落库 %d 条内容超 %d 字符被截断",
                self._masked_id(), clipped, _FALLBACK_CONTENT_LIMIT,
            )
        if not ops:
            return []

        saved_threshold = self.builder.dedup_threshold
        self.builder.dedup_threshold = self.config.fallback_dedup_threshold
        try:
            result = self.builder.apply_ops(ops, [], batch_timestamp=batch_ts)
        finally:
            self.builder.dedup_threshold = saved_threshold

        created = result.get("created_ids") or []
        logger.info("空间 %s 兜底落库: %d 条原文节点", self._masked_id(), len(created))
        return created

    @staticmethod
    def _format_conversation(messages: List[AddMessage]) -> str:
        """把结构化消息拼成 DBA 维护所需的对话文本"""
        lines = []
        for msg in messages:
            prefix = f"[{msg.role}]"
            if msg.timestamp is not None:
                prefix = f"[{msg.role}@{msg.timestamp}]"
            lines.append(f"{prefix} {msg.content}")
        return "\n".join(lines)

    # ---- 检索 ----

    def _effective_seed_k(self) -> int:
        """按图规模自适应确定 seed 数量。

        单 user 空间的图往往只有几十到几百个节点，固定 seed_k 会造成两种失真：
        - 过大（如 40）在百节点级图上会把整张图召回为种子，PAR 的图扩展无事可做，
          T2/T3 梯度全部为空；
        - 过小在孤立节点占比高时直接限制召回——孤立节点没有任何边，无法通过
          图扩展被找回，只能作为种子被召回。

        规则：图规模不超过 seed_k 时全量取种子；否则按 1/10 取值并夹在
        [seed_k_min, seed_k]。下界不低于 2，因为底层检索器的 purpose_seeds
        取 k = seed_k - seed_k//2，seed_k=1 时 k=0 会触发 FAISS 断言崩溃。
        """
        total = self.graph.node_count
        if total <= 0:
            return max(2, self.config.seed_k_min)
        if total <= self.config.seed_k:
            return max(2, total)
        return max(2, self.config.seed_k_min, min(self.config.seed_k, total // 10))

    @staticmethod
    def _story_id(node_ids: List[str]) -> str:
        """由所涉记忆集合派生稳定 id。

        叙述内容随查询变化，不能像普通节点那样由内容派生 id；改为对采纳的节点集合
        取哈希，这样同一批记忆在同一作用域下的 id 保持稳定。
        """
        if not node_ids:
            return ""
        joined = "|".join(sorted(node_ids))
        return "story:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def _story_id_from_text(text: str) -> str:
        """采纳节点集合为空时的兜底 id：改由叙述正文派生，避免跨查询撞同一个常量 id"""
        return "story:" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]

    def _render_content(self, node_id: str, content: str) -> str:
        """为输出内容附上节点类型，保留实体/属性的类型信息。

        content 是唯一能传给平台回答模型的信道，节点类型（person / emotion /
        preference 等）不带出去就彻底丢失，而它正是 A（显式事实：实体与属性）与
        B（关系组合）判断「这条是事实、偏好还是情绪」的依据。
        """
        try:
            node_type = self.graph.get_node_type(node_id)
        except Exception:
            return content
        label = getattr(node_type, "value", None) or str(node_type or "")
        return f"[{label}] {content}" if label else content

    def _node_created_at(self, node_id: str) -> Optional[str]:
        """节点的记录时间，渲染成 `YYYY-MM-DD`；无时间戳则返回 None（不输出该字段）。

        契约里 `created_at` 是可选字段，但平台侧确实会用：CLBench 的
        `format_selected_memories()` 把每条记忆渲染成 `- [<created_at>] 正文`，缺失即整条
        记忆丢掉时间线索；官方 Answer 模板第 7 条还要求把 yesterday / last month 这类相对
        时间换算成日期，前提同样是记忆里带时间。

        注意语义：节点 `timestamp` 是**记录时间**（写入图谱的时刻），不是事件发生时间
        （见 dba_pipeline/llm/inference.py 顶部说明），因此这里只提供日期，不做任何换算。
        """
        try:
            ts = self.graph.graph.nodes[node_id].get("timestamp")
        except Exception:
            return None
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or ts <= 0:
            return None
        try:
            return datetime.fromtimestamp(ts / 1000.0).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None

    def _best_cosine(self, query_vec, candidates: List[Dict[str, Any]]) -> float:
        """候选与查询的最大余弦相似度（弃权判定用）"""
        if query_vec is None or not candidates:
            return 0.0
        vectors = self.vector_store.get_content_vectors([c["id"] for c in candidates])
        return max((_cosine(query_vec, v) for v in vectors), default=0.0)

    def _abstain(self, best_cos: float, reason: str) -> List[Dict[str, Any]]:
        """按契约返回空数组；同时记账与打日志"""
        if self.stats is not None:
            self.stats.record_abstain()
        logger.info(
            "空间 %s 弃权: best_cos=%.4f 硬下限=%.4f 原因=%s",
            self._masked_id(), best_cos, self.config.abstain_cosine, reason,
        )
        return []

    def search(
        self, query: str, top_k: int, options: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """检索并按 ALM 契约组装 data 数组。

        返回内容 = MCP `query_memory` 对外给出的记忆内容（即叙事），检索直接使用共享检索器，
        不做候选扩召、重排或形态切换。`options` 只是契约字段（选择题才下发，见 contract.py），
        检索只消费题干，不对选项做任何加工。

        全程持有空间锁：add() 改图时持的是同一把锁，检索若不持锁就可能读到「字典
        边迭代边被改」的中间态；代价是同 user_id 的并发检索串行（跨 user_id 互不
        影响，而评测里同一 user_id 的并发度远低于跨 user_id 的并发度）。
        """
        with self._lock:
            return self._search_locked(query, top_k)

    def _route_purposes(self, query: str) -> Optional[List[str]]:
        """时序路由：一次目的判定 →（仅当判定为时序）追加锚点/事实文本。

        返回最终 purpose 列表（供 retrieve_with_story 注入）。任何失败/退化都
        **静默退回**：抛出异常时返回 None（由检索器内部自行推断）、工具无产出时返回
        原始 purposes。

        为什么必须由 ALM 显式调用 infer_purpose：`Retriever.retrieve()` 的 `purpose`
        语义是「提供时**跳过**内部独立推断」（见 retriever.py 的 `if purpose is not None`
        分支），故显式传 purpose 就等于放弃内部推断，必须自己先把 purposes 推断出来，
        否则会完全丢失「目的驱动种子」。`seed_ids` 是内部计算的，外部无法注入种子，
        所以用时序产出驱动主通道的唯一途径就是把产出并入 purpose。

        四条约束（Plan §3.2）：
          ① 判定只看 query、一次定死：本方法只接收 query，infer_purpose 只调用一次，
             且在任何检索结果产生之前完成；
          ② 时序产出只**追加**：merged = purposes + 追加文本，不改写既有目的；
          ③ 不碰控制流：本方法不决定是否调用主通道（由调用方无条件执行）；
          ④ 全过程打日志：判定结果 / match_type / 锚点 id / 追加条数，只记数值与 ID，
             **不记 query 正文**（合规要求）。
        """
        # ① 目的判定（时序标签与 purposes 合并在同一次 LLM 调用里，不新增调用）
        try:
            info = self.retriever.inference.infer_purpose(query)
            # infer_purpose 直接返回 json.loads 的结果：合法 JSON 也可能是数组或标量，
            # 那种情况下 .get 会抛 AttributeError。放进同一个 try，按退化处理。
            purposes = list(info.get("purposes") or []) if isinstance(info, dict) else []
            is_temporal = bool(info.get("temporal")) if isinstance(info, dict) else False
        except Exception as exc:
            logger.warning(
                "空间 %s 目的推断失败，改由检索器内部推断: %s", self._masked_id(), exc
            )
            return None

        if not purposes:
            # 退化（LLM 返回了合法 JSON 但无 purposes）：退回内部推断，避免空目的向量
            logger.warning("空间 %s 目的推断无结果，改由检索器内部推断", self._masked_id())
            return None

        logger.info(
            "空间 %s 时序路由: 判定=%s 目的数=%d",
            self._masked_id(), "时序" if is_temporal else "非时序", len(purposes),
        )
        if not is_temporal:
            return purposes

        # ② 时序工具：时间锚点 → 共时事实（仅向量检索 + 图单跳，无 LLM 调用）
        try:
            res = self.retriever.temporal_lookup(
                query,
                k_seed=self.config.temporal_k_seed,
                max_facts=self.config.temporal_max_facts,
                max_anchors=self.config.temporal_max_anchors,
            )
        except Exception as exc:
            logger.warning(
                "空间 %s 时序查询失败，静默退回仅目的通道: %s", self._masked_id(), exc
            )
            return purposes

        extra: List[str] = []
        seen_text = set()
        anchor_ids: List[str] = []
        fact_count = 0
        for group in res.get("matches") or []:
            anchor = group.get("time_anchor") or {}
            if anchor.get("id"):
                anchor_ids.append(anchor["id"])
            # 锚点自身也并入目的：查询问的往往是「何时」，锚点文本直接给出时间线索
            texts = [anchor.get("content") or ""]
            for fact in group.get("facts") or []:
                # time_consistent 为 False 表示事实自带的时期词与该锚点完全不相交
                # （疑似 TEMPORAL 边错配）。宁漏不误：这类事实不并入目的，避免引偏。
                if fact.get("time_consistent") is False:
                    continue
                texts.append(fact.get("content") or "")
            for text in texts:
                text = text.strip()
                if text and text not in seen_text:
                    seen_text.add(text)
                    extra.append(text)
            fact_count += len(group.get("facts") or [])

        logger.info(
            "空间 %s 时序查询: match_type=%s 锚点=%s 事实数=%d 可并入=%d",
            self._masked_id(), res.get("match_type"), anchor_ids, fact_count, len(extra),
        )
        if not extra:
            logger.info("空间 %s 时序路由无可用锚点，退回仅目的通道", self._masked_id())
            return purposes

        # ③ 只追加不覆盖
        merged = purposes + extra
        logger.info(
            "空间 %s 时序路由追加: 目的 %d → %d（追加 %d 条锚点/事实文本）",
            self._masked_id(), len(purposes), len(merged), len(extra),
        )
        return merged

    def _search_locked(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.retriever.path_tracker is not None:
            self.retriever.path_tracker.start_session()

        # 时序路由（Plan §3.2）：开启时由 ALM 先用 query 做一次目的判定，命中时序则把
        # temporal_lookup 的锚点/事实文本**追加**进 purpose，再驱动下面的主通道。
        # 关闭时返回 None → 检索器内部自行推断，链路与改造前逐字一致。
        injected_purpose = self._route_purposes(query) if self.config.temporal_route else None

        # with_response=False：ALM 只消费 stories / story_nodes，GenerateResponse 的产物
        # 从不被读取，却要多打一次 LLM（约 2.7s；Search 并发可达 256）。同时主办方把
        # 「Search 阶段生成最终答案」划为红线，即便结果被丢弃也不应触发该链路。
        # 主通道无条件执行（约束 ③）：无论时序路由是否命中、是否失败，这里都必须走到。
        result = self.retriever.retrieve_with_story(
            query,
            with_response=False,
            render_timestamps=self.config.render_timestamps,
            seed_k=self._effective_seed_k(),
            max_hops=self.config.max_hops,
            expand_k=self.config.expand_k,
            purpose=injected_purpose,
        )
        # 约束 ④：主通道是否执行必须可观测（失效模式是静默的）
        logger.info(
            "空间 %s 主通道已执行: 峰值=%d 叙事=%d",
            self._masked_id(),
            len(result.get("peak_memories") or []),
            len(result.get("stories") or []),
        )

        # 查询向量只服务于弃权判定（返回内容本身不再依赖它），但**不允许静默降级**：
        # 拿不到查询向量就返回空数组，与「库里确实没有相关记忆」在平台侧完全无法区分，
        # 等于把 embedding 故障变成一条不重试的错误答案。这里直接抛出，由 /search
        # 转成 5xx 交给平台重试（embedding 层内部已先做过有界退避重试）。
        try:
            query_vec = self.vector_store._embed(query)
        except Exception as exc:
            logger.error(
                "空间 %s 查询向量计算失败（embedding 不可用）: %s", self._masked_id(), exc
            )
            raise RuntimeError(
                "查询向量计算失败：embedding 服务不可用，请检查 EMBEDDING_MODEL / "
                "EMBEDDING_API_BASE / EMBEDDING_API_KEY"
            ) from exc

        items: List[Dict[str, Any]] = []

        # 只返回叙事一条。MCP 的 `query_memory` 对外给出的记忆内容就是 `stories`
        # （`story_nodes` 只是一个 id 列表，正文本就不是它对外的记忆产物），所以这是对
        # MCP 最严格的忠实映射。实测（Plan §7.10.6）：叙事首条已覆盖全部命中的 gold fact，
        # 其后的原子节点边际贡献为 0（top-1 累计命中 == top-all），而平台也没有任何检索侧
        # 指标会因条数变化——故节点条目是纯冗余，收拢掉。
        stories = result.get("stories") or []
        if stories:
            story = stories[0]
            # adopted_ids 可能全部落在候选之外（LLM 幻觉或重编号），此时集合为空。
            # 对空集合取哈希会让所有这类查询共用同一个常量 id，故退回由正文派生。
            story_id = self._story_id(result.get("story_nodes") or []) \
                or self._story_id_from_text(story)
            # 叙述刻意不带 created_at：它跨多条记忆、时间跨度可能很宽，给单一日期会让
            # Answer 模型误以为整段叙述都发生在同一天。
            items.append({"id": story_id, "content": story, "score": 1.0})
        else:
            # 叙事缺失（LLM 调用失败等）时**不能静默返回空数组**——那与「弃权」无法区分，
            # 会把「检索到了但没整理成文」误报成「库里没有相关记忆」。此时退回采纳节点原文。
            logger.warning("空间 %s 叙事缺失，退回采纳节点原文", self._masked_id())
            for node_id in result.get("story_nodes") or []:
                node = self.graph.get_node(node_id)
                if not node or node.get("deprecated") or node.get("forgotten"):
                    continue
                content = node.get("content")
                if not content:
                    continue
                entry = {
                    "id": node_id,
                    "content": self._render_content(node_id, content),
                    # 占位分：最终分数由 _normalize_scores 按返回顺序折算（首条恒 1.0）
                    "score": 1.0,
                }
                created_at = self._node_created_at(node_id)
                if created_at:
                    entry["created_at"] = created_at
                items.append(entry)

        # 契约要求返回条数不超过 top_k
        items = items[:top_k]

        # 弃权（唯一保留的 ALM 自造能力）：best_cos 取候选池 T1/T2 的最大余弦。
        # 每次检索都记录（只记数值、不记 query 正文）：平台 smoke 打进来的真实查询
        # 无法离线复现，只能靠这组数字反推分布来标定阈值。
        candidates = self._collect_par_candidates(result)
        best_cos = self._best_cosine(query_vec, candidates)
        if self.stats is not None:
            self.stats.record_search(best_cos)
        logger.info("空间 %s 检索: best_cos=%.4f", self._masked_id(), best_cos)

        # 弃权第一段（硬下限）：低于它一律弃权，连判定调用都不必发。
        if self.config.abstain_cosine > 0.0 and best_cos < self.config.abstain_cosine:
            return self._abstain(best_cos, "低于硬下限")

        # 弃权第二段（模糊带判定）：余弦落在 [abstain_cosine, abstain_verify_hi) 时，
        # 单看数值分不清「库里确实没有」与「有、但提问措辞与节点原文距离远」——实测
        # 两类样本的余弦区间重叠（见 alm/judge.py 顶部注释）。故把已排好序的 top-N
        # 交给 LLM 判一次；判否才弃权，判定失败按放行处理。
        if self.config.abstain_verify_hi > 0.0 and best_cos < self.config.abstain_verify_hi:
            verdict = self.judge.is_relevant(query, items)
            if verdict is False:
                return self._abstain(best_cos, "模糊带判定为无相关信息")
            logger.info(
                "空间 %s 模糊带放行: best_cos=%.4f verdict=%s",
                self._masked_id(), best_cos, verdict,
            )

        return self._normalize_scores(items)

    def _collect_par_candidates(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """T1 种子 + T2 图扩展的候选并集，仅供弃权判定取 best_cos 用。

        直接取 hop_history（它比 peak 容忍带更全）。返回内容不再参与排序，故这里的
        排序只影响可读性。
        """
        par_scores = result.get("peak_scores") or {}
        candidates: Dict[str, Dict[str, Any]] = {}

        for entry in result.get("hop_history") or []:
            hop = int(entry.get("hop", 0))
            for cand in entry.get("candidates") or []:
                node_id = cand.get("id")
                if not node_id or node_id in candidates:
                    continue
                content = cand.get("content") or self.graph.get_content(node_id)
                if not content:
                    continue
                candidates[node_id] = {
                    "id": node_id,
                    "content": content,
                    "par_score": float(par_scores.get(node_id, 0.0)),
                    "tier": 1 if hop == 0 else 2,
                }

        ordered = list(candidates.values())
        ordered.sort(key=lambda item: (item["tier"], -item["par_score"]))
        return ordered

    @staticmethod
    def _normalize_scores(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把重排得分归一化到 (0, 1]，保证「数值越大越相关」

        注意：此处按**单次查询**的最大值归一化，因此每次查询的首条恒为 1.0，
        分数**不跨查询可比**。若平台要求跨查询同尺度，需改用绝对分而非归一化。
        """
        if not items:
            return items
        top = max(item["score"] for item in items)
        if top <= 0:
            # 极端情况下没有可用分数，则按返回顺序给出单调递减的占位分
            total = len(items)
            for rank, item in enumerate(items):
                item["score"] = round(1.0 - rank / total, 6)
            return items
        for item in items:
            item["score"] = round(item["score"] / top, 6)
        return items

    # ---- 持久化 ----

    def save(self, force: bool = False):
        """原子写回 YAML，避免进程中断损坏主文件

        仅在图谱确有改动时落盘（`force=True` 可强制）。这不只是省 I/O：cleanup.py
        按 mtime 判断空间最后一次被写入的时间，无条件重写会让「停服」本身刷新 mtime，
        使 30 天留存清理永远判定不到过期空间。详见 __init__ 中 _dirty 的说明。
        """
        with self._lock:
            if not self._dirty and not force:
                return
            data = self.graph.to_dict()
            self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(self.yaml_path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                os.replace(tmp_path, self.yaml_path)
                self._save_index()
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            # 写盘成功后才清位：中途失败时保持 dirty，下次仍会重试落盘
            self._dirty = False

    # ---- 内部工具 ----

    def _masked_id(self) -> str:
        """日志中不输出完整 user_id（评测数据与标识不宜落入日志）"""
        if len(self.user_id) <= 8:
            return "***"
        return f"{self.user_id[:4]}***{self.user_id[-4:]}"
