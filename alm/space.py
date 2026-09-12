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
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.core.path_tracker import PathTracker
from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph

from alm.config import ALMConfig
from alm.contract import AddMessage
from alm.rerank import cosine, rerank

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


class MemorySpace:
    """单个 user_id 的完整记忆栈：图谱 + 向量 + DBA 维护 + PAR 检索"""

    def __init__(
        self,
        user_id: str,
        config: ALMConfig,
        embeddings,
        llm,
        yaml_path: Path,
    ):
        if not HAS_DEPS:
            raise RuntimeError(
                f"ALM 记忆链路依赖未安装（{_IMPORT_ERROR}）；请先执行 pip install -e ."
            )

        self.user_id = user_id
        self.config = config
        self.yaml_path = Path(yaml_path)
        # 串行化单个用户空间内的图谱写入与 YAML 落盘
        self._lock = threading.RLock()

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
        payload = messages[: self.config.max_messages_per_add]
        conversation = self._format_conversation(payload)
        batch_ts = self._batch_timestamp(payload)

        with self._lock:
            result = self.dba.maintain(conversation, timestamp=batch_ts)
            created = (result.get("result") or {}).get("created_ids") or []
            if self._should_fallback(result, created):
                created = self._store_raw_fallback(payload, batch_ts)

        logger.info(
            "空间 %s 写入完成: messages=%d nodes=%d",
            self._masked_id(), len(payload), len(created),
        )
        return {"messages": len(payload), "nodes": len(created)}

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
        """
        if result.get("skipped"):
            return True
        if created:
            return False
        # DBA 虽未新建节点，但可能改写了既有节点，此时内容已被覆盖
        node_ops = (result.get("ops") or {}).get("node_ops") or []
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
        ops = []
        for msg in messages[: self.config.fallback_max_nodes]:
            text = msg.content.strip()
            if not text:
                continue
            ops.append({
                "action": "create",
                "content": f"{msg.role}: {text}"[:_FALLBACK_CONTENT_LIMIT],
                "node_type": NodeType.THING.value,
            })
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
        joined = "|".join(sorted(node_ids))
        return "story:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]

    def _collect_temporal_candidates(self, query: str, exclude_ids) -> List[Dict[str, Any]]:
        """时序补召（T4）：语义命中时间锚点后，沿 TEMPORAL 边反向取共时事件。

        默认关闭（见 ALMConfig.temporal_recall）。实测该通道与主检索高度冗余：
        产出节点 97% 已被主链路召回；10 条 temporal 查询上新增的 8 个节点无一命中 gold；
        而新增节点的 gold 重叠为 0，意味着提权只会挤占正确节点。故保留实现以备在真实
        数据上重新评估，默认不启用。时间锚点自身一并带上，便于回答模型定位「何时」。
        """
        if not self.config.temporal_recall:
            return []
        try:
            res = self.retriever.temporal_lookup(
                query,
                k_seed=self.config.temporal_k_seed,
                max_facts=self.config.temporal_max_facts,
                max_anchors=self.config.temporal_max_anchors,
            )
        except Exception as exc:
            logger.warning("时序补召失败: %s", exc)
            return []

        out: List[Dict[str, Any]] = []
        for group in res.get("matches") or []:
            anchor = group.get("time_anchor") or {}
            items = ([anchor] if anchor.get("id") else []) + list(group.get("facts") or [])
            for item in items:
                node_id = item.get("id")
                if not node_id or node_id in exclude_ids:
                    continue
                exclude_ids.add(node_id)
                out.append({
                    "id": node_id,
                    "content": item.get("content") or "",
                    "par_score": 0.0,
                    "tier": 4,
                })
        return out

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

    def _best_cosine(self, query_vec, candidates: List[Dict[str, Any]]) -> float:
        """候选与查询的最大余弦相似度（弃权判定用）"""
        if query_vec is None or not candidates:
            return 0.0
        vectors = self.vector_store.get_content_vectors([c["id"] for c in candidates])
        return max((cosine(query_vec, v) for v in vectors), default=0.0)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """检索并按梯度重排为 ALM 的 data 数组（只返回节点原文，不经过 StoryRank）

        候选按证据来源分梯度：T1 种子 → T2 图扩展（1 跳及以上）→ T3 语义救援
        （被目的回归过滤掉、但语义上可能相关的邻居）。梯度以**软先验乘子**参与打分，
        因此高相关的低梯度节点仍可越过弱相关的高梯度节点。
        """
        if self.retriever.path_tracker is not None:
            self.retriever.path_tracker.start_session()

        # story / hybrid 形态下，同一次调用内顺带产出记忆整理叙述，避免重复检索
        retrieve_kwargs = dict(
            seed_k=self._effective_seed_k(),
            max_hops=self.config.max_hops,
            expand_k=self.config.expand_k,
        )
        if self.config.search_shape == "raw":
            result = self.retriever.retrieve(query, **retrieve_kwargs)
        else:
            result = self.retriever.retrieve_with_story(query, **retrieve_kwargs)

        mode = self.config.rerank_mode
        # 查询向量同时服务于重排与弃权判定，故无论重排模式如何都先算出来
        query_vec = None
        try:
            query_vec = self.vector_store._embed(query)
        except Exception as exc:
            logger.warning("查询向量计算失败，退回 PAR 排序: %s", exc)
            mode = "off"

        candidates = self._collect_par_candidates(result)
        if not candidates:
            return []

        # 弃权：候选与查询的最大余弦低于阈值时视为「无相关记忆」，按契约返回空数组
        best_cos = self._best_cosine(query_vec, candidates)
        if self.config.abstain_cosine > 0.0 and best_cos < self.config.abstain_cosine:
            logger.info(
                "空间 %s 弃权: 最高余弦 %.4f < 阈值 %.4f",
                self._masked_id(), best_cos, self.config.abstain_cosine,
            )
            return []

        if mode != "off":
            candidates += self._collect_rescue_candidates(
                self._seed_ids(result), {item["id"] for item in candidates}, query_vec
            )

        # 时序补召与目的过滤无关，故不受重排模式影响
        candidates += self._collect_temporal_candidates(
            query, {item["id"] for item in candidates}
        )

        pool = self.config.rerank_pool
        if pool > 0:
            candidates = candidates[:pool]

        content_vecs = None
        if mode != "off":
            content_vecs = self.vector_store.get_content_vectors(
                [item["id"] for item in candidates]
            )

        items = rerank(
            candidates,
            mode=mode,
            query_vec=query_vec,
            content_vecs=content_vecs,
            weight_par=self.config.rerank_weight_par,
            weight_embedding=self.config.rerank_weight_embedding,
            tier_weights=self.config.rerank_tier_weights,
            llm=self.retriever.llm,
            query=query,
            pool=pool,
        )
        shape = self.config.search_shape
        stories = result.get("stories") or []
        if shape == "raw" or not stories:
            atom_limit = top_k
        elif shape == "story":
            atom_limit = 0
        else:  # hybrid：叙述占首条，其余名额留给原子节点
            atom_limit = max(0, top_k - 1)

        # 只透出 ALM 契约声明的字段（id / content / score）
        projected = [
            {
                "id": item["id"],
                "content": self._render_content(item["id"], item["content"]),
                "score": item["score"],
            }
            for item in items[:atom_limit]
        ]
        if shape != "raw" and stories:
            # 叙述置顶：由 _normalize_scores 归一化后其 score 恒为 1.0
            top_score = max((item["score"] for item in projected), default=1.0)
            projected = [{
                "id": self._story_id(result.get("story_nodes") or []),
                "content": stories[0],
                "score": top_score,
            }] + projected

        return self._normalize_scores(projected)

    @staticmethod
    def _seed_ids(result: Dict[str, Any]) -> List[str]:
        """hop 0 的候选即种子节点"""
        for entry in result.get("hop_history") or []:
            if int(entry.get("hop", -1)) == 0:
                return [c["id"] for c in (entry.get("candidates") or []) if c.get("id")]
        return []

    def _collect_par_candidates(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """T1 种子 + T2 图扩展。

        直接取 hop_history（它比 peak 容忍带更全），每条候选自带 hop 与关系边信息，
        因此梯度无需额外推断。截断时先保 T1、再按组合分排 T2。
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

    def _collect_rescue_candidates(
        self, seed_ids: List[str], known: set, query_vec
    ) -> List[Dict[str, Any]]:
        """T3 语义救援：种子的 1 跳邻居里，被目的回归过滤掉的那些。

        这批节点的画像很明确——purpose 分低于阈值（所以被丢弃），但语义相似度可能很高。
        这里只按语义相似度取前 N 个，避免把噪声全捞回来。
        """
        limit = self.config.rerank_rescue
        if limit <= 0 or not seed_ids or query_vec is None:
            return []

        try:
            expanded = self.graph.expand_with_trace(seed_ids)
        except Exception as exc:
            logger.warning("救援候选扩展失败: %s", exc)
            return []

        pool: List[Dict[str, Any]] = []
        for node_id in expanded:
            if node_id in known:
                continue
            node = self.graph.get_node(node_id)
            if not node or node.get("deprecated") or node.get("forgotten"):
                continue
            content = node.get("content")
            if not content:
                continue
            pool.append({"id": node_id, "content": content, "par_score": 0.0, "tier": 3})

        if not pool:
            return []

        vectors = self.vector_store.get_content_vectors([item["id"] for item in pool])
        scored = sorted(
            zip(pool, vectors), key=lambda pair: cosine(query_vec, pair[1]), reverse=True
        )
        return [item for item, _ in scored[:limit]]

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

    def save(self):
        """原子写回 YAML，避免进程中断损坏主文件"""
        with self._lock:
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

    # ---- 内部工具 ----

    def _masked_id(self) -> str:
        """日志中不输出完整 user_id（评测数据与标识不宜落入日志）"""
        if len(self.user_id) <= 8:
            return "***"
        return f"{self.user_id[:4]}***{self.user_id[-4:]}"
