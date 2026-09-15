# SPDX-License-Identifier: AGPL-3.0-only

"""
DBA MCP Server

将 DBA 记忆图谱系统封装为标准 MCP (Model Context Protocol) Server，
任何遵循 MCP 的 LLM Agent 均可通过 stdio 或 SSE 调用。

Usage:
    # 本地 stdio 模式（Agent 直接调用）
    python -m dba_pipeline.mcp_server --yaml your_memory_graph.yaml

    # SSE 网络模式（远程 Agent 调用）
    python -m dba_pipeline.mcp_server --yaml your_memory_graph.yaml --sse --port 8766

依赖:
    pip install mcp
"""

import os

# 避免 torch 与 FAISS 的 OpenMP 运行时冲突导致进程 Aborted
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import sys
import tempfile
import threading
import time
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# Allow running as module
# Package installed via pip

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.graph.review import find_duplicate_candidates
from dba_pipeline.loader import load_graph, load_into
from dba_pipeline.core.jump_axis import NodeType, RelationType, get_jump_weight
from dba_pipeline.core.path_tracker import PathTracker
from dba_pipeline.source_store import (
    default_source_dir, list_batches, load_batch, render_batch_text, source_store_enabled,
)
from dba_pipeline.extraction.review_scheduler import IdleReviewScheduler, ReviewConfig
from dba_pipeline import envfile, filelock, oplog
from dba_pipeline import params as params_mod
from dba_pipeline import webauth
from dba_pipeline.viz import logbus

# ---- 可选 DBA 管线导入 ----
try:
    from dba_pipeline.extraction.dba import MemoryDBA
    from dba_pipeline.extraction.graph_builder import GraphBuilder
    from dba_pipeline.embedding.store import VectorStore, OpenAIEmbeddings, LocalEmbeddings
    from langchain_openai import ChatOpenAI
    HAS_DBA = True
except ImportError:
    HAS_DBA = False

# ---- MCP SDK imports (需要 pip install mcp) ----
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool, ListToolsResult, CallToolResult, TextContent,
        PaginatedRequestParams, CallToolRequestParams,
    )
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


NODE_TYPES = {t.value.upper(): t for t in NodeType}
REL_TYPES = {t.value: t for t in RelationType}

# 单次对话输入的最大字符数（防止异常输入打爆 LLM token / 缓冲内存）
MAX_CONVERSATION_LENGTH = 20000


class DBAServer:
    """DBA 核心逻辑，与 MCP 协议层解耦"""

    def __init__(self, graph: MemoryGraph, yaml_path: str = None,
                 dba=None, scheduler=None, retriever=None, vector_store=None,
                 lock=None, params_path: str = None):
        self.graph = graph
        self.yaml_path = yaml_path
        self.dba = dba          # 可选: MemoryDBA 实例
        self.scheduler = scheduler  # 可选: MaintenanceScheduler 实例
        self.retriever = retriever  # 可选: PurposeDrivenRetriever 实例（完整 P 链路）
        self.vector_store = vector_store  # 可选: VectorStore（人工干预时同步向量）
        # 可重入锁，与 GraphBuilder 共享，串行化 graph 修改与 YAML 写回
        self._lock = lock if lock is not None else threading.RLock()
        # 跨进程写锁：与 WebUI 面板共用同一把 <yaml>.lock，串行化「读-改-写」；
        # 维护可能耗时较长（LLM 调用），故 MCP 侧等待上限放宽到 5 分钟。
        self._file_lock = filelock.FileLock(filelock.lock_path_for(yaml_path)) if yaml_path else None
        self._file_lock_timeout = 300.0
        # 记录上次读到的 YAML mtime，用于检测其它进程（面板）的写入
        self._last_yaml_mtime = self._yaml_mtime()
        # 向量库已对齐到的 YAML mtime（0 表示尚未对齐，首次访问会补做一次）
        self._vector_mtime = 0
        # 巡检工具级互斥：避免两次巡检并发产出重复建议（非阻塞获取）。
        # 用 RLock：空闲巡检会在**整个巡检期间**持锁，然后调用 review_graph()，
        # 而 review_graph 自己也会获取这把锁——同一线程重入必须放行，否则体检结果
        # 会被静默替换成"已有一次巡检正在进行"的错误字典。跨线程仍互斥。
        self._review_running = threading.RLock()
        # 活动追踪：空闲巡检据此判断"现在是不是没人用"，以及巡检中途要不要让路
        self._last_activity = time.time()
        self._activity_lock = threading.Lock()
        self._next_node_id = self._compute_next_id()
        # 运行时检索参数（权重矩阵 / 种子 / 目的回归）：与 WebUI 面板共享同一份 YAML，
        # 面板改完落盘，这里按 mtime 检测并就地生效（无需重启）。
        self.params_path = params_path or (
            params_mod.params_path_for(yaml_path) if yaml_path else None
        )
        self._params_mtime = 0
        self.params = params_mod.default_params()
        self.reload_params(force=True)

    # ---- 跨进程一致性 ----

    def _yaml_mtime(self) -> int:
        """YAML 文件 mtime（纳秒）；不存在时返回 0"""
        if not self.yaml_path:
            return 0
        try:
            return os.stat(self.yaml_path).st_mtime_ns
        except OSError:
            return 0

    def _reload_if_changed(self) -> bool:
        """若 YAML 被其它进程（如 WebUI 面板）改动，则**就地**重载。

        必须就地重载：GraphBuilder / Retriever 等组件持有同一个 MemoryGraph
        实例，若是新建实例再赋值，它们仍指向旧对象，重载等于没做。

        返回是否发生了重载；重载失败会抛异常——写路径据此中止，避免拿旧数据
        覆盖磁盘上的最新内容。
        """
        if not self.yaml_path:
            return False
        mtime = self._yaml_mtime()
        if mtime == self._last_yaml_mtime:
            return False
        with self._lock:
            load_into(self.graph, self.yaml_path)
            self._next_node_id = self._compute_next_id()
            self._last_yaml_mtime = mtime
        logging.getLogger("mcp").info("检测到外部改动，已重载 YAML: %s", self.yaml_path)
        return True

    def _graph_contents(self) -> dict:
        """图谱中「有内容」节点的 {id: content} 快照（向量库期望集）。"""
        with self._lock:
            return {
                nid: (d.get("content") or "")
                for nid, d in self.graph.graph.nodes(data=True)
                if (d.get("content") or "").strip()
            }

    def reconcile_vectors(self, force: bool = False) -> dict:
        """把向量库对齐到当前磁盘图谱（补增 / 更新变更 / 删除多余）。

        WebUI 面板只写 YAML、不碰向量库，因此这里以**图谱为权威**做对账，
        保证面板的增删改能进入/退出向量检索。仅当 YAML 自上次对齐后发生变化
        才真正执行（一次 os.stat 的代价）。
        """
        if self.vector_store is None:
            self._reload_if_changed()
            return {"skipped": True, "reason": "no vector store"}
        mtime = self._yaml_mtime()
        if not force and mtime and mtime == self._vector_mtime:
            return {"changed": False}
        self._reload_if_changed()
        desired = self._graph_contents()
        if not desired:
            # 图谱为空（如 --yaml 指错）时不清空既有向量，避免误删索引
            logging.getLogger("mcp").warning("图谱为空，跳过向量同步（不清空既有向量库）")
            self._vector_mtime = self._yaml_mtime()
            return {"skipped": True, "reason": "empty graph"}
        stats = self.vector_store.reconcile(desired)
        self._vector_mtime = self._yaml_mtime()
        if stats.get("added") or stats.get("updated") or stats.get("removed"):
            logging.getLogger("mcp").info(
                "向量库已对齐图谱: +%s ~%s -%s",
                stats.get("added"), stats.get("updated"), stats.get("removed"),
            )
        return stats

    def _refresh_before_read(self) -> None:
        """读取前对齐（重载图谱 + 同步向量 + 加载参数）；失败只告警，不阻断读取。"""
        try:
            self.reconcile_vectors()
        except Exception as e:
            logging.getLogger("mcp").warning("读取前同步失败（沿用现有索引）: %s", e)
        self._load_params_if_changed()  # 面板调参后，下一次检索即用新参数

    def _refresh_graph_only(self) -> None:
        """仅重载图谱（不触发嵌入），用于让统计/巡检与面板编辑保持一致。"""
        try:
            self._reload_if_changed()
        except Exception as e:
            logging.getLogger("mcp").warning("重载图谱失败（沿用现有内存图）: %s", e)

    # ---- 运行时参数（权重矩阵 / 种子 / 目的回归）----

    def _params_file_mtime(self) -> int:
        """参数文件 mtime（纳秒）；不存在时返回 0"""
        if not self.params_path:
            return 0
        try:
            return os.stat(self.params_path).st_mtime_ns
        except OSError:
            return 0

    def reload_params(self, force: bool = False) -> bool:
        """读取参数文件并就地生效（权重矩阵 + retriever 系数）。返回是否重新加载。"""
        if not self.params_path:
            return False
        mtime = self._params_file_mtime()
        if not force and mtime == self._params_mtime:
            return False
        loaded = params_mod.load(self.params_path)
        stats = params_mod.apply(loaded, retriever=self.retriever)
        self.params = loaded
        self._params_mtime = mtime
        if force or stats.get("weights") or stats.get("retriever"):
            logging.getLogger("mcp").info(
                "已加载检索参数: 权重改动=%s 打分系数改动=%s（%s）",
                stats.get("weights"), stats.get("retriever"), self.params_path,
            )
        return True

    def _load_params_if_changed(self) -> None:
        """按 mtime 检测参数变更；失败只告警（退回默认值），不影响主流程。"""
        try:
            self.reload_params()
        except Exception as e:
            logging.getLogger("mcp").warning("加载检索参数失败（沿用现有参数）: %s", e)

    def set_params(self, patch: dict) -> dict:
        """合并并落盘参数（供 dba_intervene(action=set_params) 与面板共用同一份文件）。

        走 ``params_mod.update``：在跨进程文件锁内完成「读-改-写」，因此与面板
        同时调参也各自基于锁内最新版本合并，不会互相覆盖。
        """
        if not self.params_path:
            return {"error": "未配置参数文件路径"}
        saved = params_mod.update(self.params_path, patch or {})
        stats = params_mod.apply(saved, retriever=self.retriever)
        self.params = saved
        self._params_mtime = self._params_file_mtime()
        logging.getLogger("mcp").info("检索参数已更新: %s", stats)
        return {"path": self.params_path, "applied": stats, "params": saved}

    @contextmanager
    def write_guard(self):
        """跨进程写临界区：文件锁 → 重载外部改动并同步向量 → 执行写操作。

        所有会落盘的操作都必须包在这里，否则会被其它进程的写入覆盖（丢失更新）。
        """
        if self._file_lock is None:
            self.reconcile_vectors()
            self._load_params_if_changed()
            yield
            return
        with self._file_lock.hold(timeout=self._file_lock_timeout):
            self.reconcile_vectors()
            self._load_params_if_changed()
            yield

    # ---- 序列化 ----

    def _save(self):
        """原子写回 YAML：先写临时文件再替换，避免崩溃损坏主文件"""
        if not self.yaml_path:
            return
        with self._lock:
            data = self.graph.to_dict()
            dir_name = os.path.dirname(os.path.abspath(self.yaml_path)) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                os.replace(tmp_path, self.yaml_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            self._last_yaml_mtime = self._yaml_mtime()  # 自己的写入不算「外部改动」
            # 注意：这里**不**更新 _vector_mtime。写路径虽然大多自行同步了向量
            # （intervene / GraphBuilder），但删除等路径未必；留待下次对账时以图谱
            # 为权威做一次 diff（无差异时开销极小），可自愈任何遗漏。

    def _semantic_duplicate_hint(self, content: str) -> Optional[dict]:
        """语义疑似重复提示（仅提示，不阻断创建）。

        人工路径原先直接 add_node，完全不走去重，实测产生过重复 person 节点
        （txt 的 n25/n27）。这里复用 DBA 维护路径同一套 `_find_duplicate`
        （FAISS top-3 + 余弦阈值），但**不据此拒绝**：日期锚点这类短内容在向量
        空间里彼此高度相似，按余弦拒绝会挡住合法的新锚点。
        """
        builder = getattr(self.dba, "builder", None)
        if builder is None:
            return None
        try:
            dup_id = builder._find_duplicate(content)
        except Exception as e:
            logging.warning(f"人工建节点语义去重检查失败，按不重复处理: {e}")
            return None
        if not dup_id:
            return None
        return {
            "duplicate_of": dup_id,
            "existing_content": (self.graph.get_node(dup_id) or {}).get("content", ""),
            "message": (f"该内容与已有节点 {dup_id} 语义高度相似，请确认是否应当改用 "
                        "update_node 更新既有节点，而不是新建。"),
        }

    def _compute_next_id(self) -> int:
        max_id = 0
        for nid in self.graph.graph.nodes():
            try:
                num = int(nid[1:])
                if num > max_id:
                    max_id = num
            except (ValueError, IndexError):
                pass
        return max_id + 1

    # ---- Tool Handlers ----

    def add_conversation(self, conversation: str) -> dict:
        """追加对话文本，若 DBA 管线已接入则维护图谱"""
        if not conversation or len(conversation) > MAX_CONVERSATION_LENGTH:
            return {"error": f"conversation 长度必须在 1~{MAX_CONVERSATION_LENGTH} 字符之间"}
        # P0: 优先走调度器批量累积（多轮合并为一次 LLM 维护），降低 token
        if self.scheduler:
            self.scheduler.on_conversation(conversation)
            return {
                "queued": True,
                "buffered": self.scheduler.buffer_size,
                "message": "conversation queued for batched maintenance",
            }
        if self.dba:
            try:
                # 跨进程写临界区：先重载面板等外部改动，再抽取落盘，避免相互覆盖
                with self.write_guard():
                    # 无外部时间戳来源，按「实际落库时刻」注入（Unix 毫秒）
                    result = self.dba.maintain(conversation, timestamp=int(time.time() * 1000))
                    self._save()
                out = {
                    "maintained": True,
                    "nodes_created": len(result.get("result", {}).get("created_ids", [])),
                    "stats": {
                        k: v for k, v in self.dba.builder.stats.items()
                        if isinstance(v, (int, float))
                    },
                }
                # evidence 产出情况（开启时才有）：用于区分"LLM 主动留空"与"被子串校验拦掉"
                if result.get("evidence_stats") is not None:
                    out["evidence_stats"] = result["evidence_stats"]
                # 批次标识（开启原文存储时才有）：用于事后调 dba_review_sources 核对这一批
                if result.get("batch_id"):
                    out["batch_id"] = result["batch_id"]
                return out
            except Exception as e:
                return {
                    "maintained": False,
                    "error": str(e),
                    "conversation_length": len(conversation),
                }
        return {
            "maintained": False,
            "error": "DBA pipeline not wired",
        }

    def query_memory(self, query: str, rerank_k: int = 20) -> dict:
        """检索记忆：走完整 P 链路 + StoryRank 故事化，返回记忆故事片段"""
        if self.retriever is None:
            return {"error": "检索链路未初始化（需要 embedding）", "stories": []}

        # 检索前对齐：面板等外部进程对图谱的增删改需先进入向量库
        self._refresh_before_read()
        try:
            # 每次检索视为一次联想会话，重置同会话饱和计数（跨会话终身增强保留）
            if self.retriever.path_tracker is not None:
                self.retriever.path_tracker.start_session()
            # StoryRank：检索 → 连通性粗筛 → 故事化（不生成回复，交由 Agent 处理）
            # render_timestamps=True：把节点「记录于 <日期>」交给叙事整理，让上层 LLM
            # 自己判断时间关系，而不依赖独立的时序锚点网络。
            # 运行时参数（面板可调）：种子数量 / 最大跳数 / 每轮扩展数
            rp = self.params.get("retrieval", {})
            result = self.retriever.retrieve_with_story(
                query, with_response=False, render_timestamps=True,
                seed_k=int(rp.get("seed_k", 5)),
                max_hops=int(rp.get("max_hops", 5)),
                expand_k=(int(rp.get("expand_k", 0)) or None),
            )
            stories = result.get("stories", [])
            # rank-k：最多返回 rerank_k 个故事片段（0 表示不截断）
            if rerank_k > 0:
                stories = stories[:rerank_k]
            return {
                "stories": stories,
                "story_nodes": result.get("story_nodes", []),
                "discarded_nodes": result.get("discarded_nodes", []),
                "total_matched": result.get("total_candidates", 0),
                "returned": len(stories),
                "query": query,
                "purpose": result.get("purpose"),
                "method": "story_rank",
            }
        except Exception as e:
            logging.error(f"StoryRank 检索失败: {e}", exc_info=True)
            return {"error": str(e), "stories": [], "method": "story_rank_failed"}

    def run_chain_trace(self, query: str, on_stage=None, rerank_k: int = 20) -> dict:
        """面板「调用测试」用：走同一条真实链路，但把每一段轨迹都带回来。

        与 query_memory 的差别**只在返回体**：那边是给 MCP 工具调用方的精简结果，
        这里返回完整轨迹（意图 / 每跳候选与分数 / 峰值 / storyrank 采纳），并支持
        on_stage 回调把阶段事件实时抛给面板。检索与打分逻辑完全共用，不另写一份链路。
        """
        if self.retriever is None:
            raise RuntimeError("检索链路未初始化（需要 embedding 配置）")

        def _safe(value):
            """转成可 JSON 序列化的形式：枚举取 value、numpy 标量取 item、元组转列表"""
            if isinstance(value, dict):
                return {str(k): _safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_safe(v) for v in value]
            if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
                return value
            if isinstance(value, float):
                return round(value, 6)
            if hasattr(value, "value"):        # 枚举（如 RelationType）
                return _safe(value.value)
            if hasattr(value, "item"):         # numpy 标量
                return _safe(value.item())
            return str(value)

        # 检索前对齐：面板等外部进程对图谱的增删改需先进入向量库
        self._refresh_before_read()
        # 每次检索视为一次联想会话，重置同会话饱和计数（跨会话终身增强保留）
        if self.retriever.path_tracker is not None:
            self.retriever.path_tracker.start_session()
        # 运行时参数与 query_memory 保持同一来源（面板可调）
        rp = self.params.get("retrieval", {})
        seed_k = int(rp.get("seed_k", 5))
        max_hops = int(rp.get("max_hops", 5))
        expand_k = int(rp.get("expand_k", 0)) or None

        result = self.retriever.retrieve_with_story(
            query, with_response=False, render_timestamps=True,
            seed_k=seed_k, max_hops=max_hops, expand_k=expand_k,
            on_stage=on_stage,
        )
        stories = result.get("stories", [])
        if rerank_k > 0:
            stories = stories[:rerank_k]
        # 「经过的节点」= PAR 每跳候选的并集（面板据此在图谱里做激活呈现）
        visited = []
        for hop in result.get("hop_history", []):
            for cand in hop.get("candidates", []):
                cid = cand.get("id")
                if cid and cid not in visited:
                    visited.append(cid)
        return _safe({
            "query": query,
            "purpose": result.get("purpose"),
            "hop_history": result.get("hop_history", []),
            "peak_memories": result.get("peak_memories", []),
            "peak_scores": result.get("peak_scores", {}),
            "visited_ids": visited,
            "stories": stories,
            "story_nodes": result.get("story_nodes", []),
            "discarded_nodes": result.get("discarded_nodes", []),
            "total_candidates": result.get("total_candidates", 0),
            "method": "story_rank",
            "params": {"seed_k": seed_k, "max_hops": max_hops, "expand_k": expand_k},
        })

    def temporal_lookup(self, query: str, k_seed: int = None, max_facts: int = None,
                        max_anchors: int = None) -> dict:
        """独立单跳时序工具（仅"时间→事件"反向）。

        走检索链路自身的 temporal_lookup：语义匹配到时间锚点 → 沿 TEMPORAL 反向取共时事件。
        不进入主检索/故事化流程，不挤占主检索候选集。
        """
        if self.retriever is None:
            return {"error": "检索链路未初始化", "time_anchor": {"id": None, "content": ""}, "facts": []}
        self._refresh_before_read()  # 同步外部进程对图谱的改动
        rp = self.params.get("retrieval", {})
        k_seed = int(rp.get("temporal_k_seed", 8)) if k_seed is None else k_seed
        max_facts = int(rp.get("temporal_max_facts", 8)) if max_facts is None else max_facts
        max_anchors = int(rp.get("temporal_max_anchors", 3)) if max_anchors is None else max_anchors
        try:
            res = self.retriever.temporal_lookup(query, k_seed=k_seed, max_facts=max_facts,
                                                 max_anchors=max_anchors)
            if isinstance(res, dict) and not res.get("matches"):
                if res.get("rejected"):
                    # 拒答 ≠ 检索失败：库里确实没有该时段的锚点。此处不能引导改走主检索，
                    # 否则上层会反复换查询，而真实原因是"该时间没有记忆"。
                    res["note"] = ("查询指定的时间在库中无对应锚点，已按拒答返回空（不是检索失败）。"
                                   "若该时段确实有记忆，说明抽取未产出时间锚点，可让用户补充具体日期。")
                else:
                    res["note"] = ("未定位到可用时间锚点；这可能是‘事件→时间/因果联想’类问题，"
                                   "考虑改用 dba_query_memory。")
            return res
        except Exception as e:
            logging.error(f"temporal_lookup 失败: {e}", exc_info=True)
            return {"error": str(e), "time_anchor": {"id": None, "content": ""}, "facts": []}

    # ---- 活动追踪（供空闲巡检）----

    def touch(self, source: str = ""):
        """记一次活动。任何工具的调用都算——空闲巡检据此判断是否需要让路。

        注意巡检工具自己也会 touch（它们在 dispatch 层统一记录），这让巡检天然
        "自我节流"：刚跑完一次巡检，空闲计时重新开始，不会连轴转。
        """
        with self._activity_lock:
            self._last_activity = time.time()

    def last_activity(self) -> float:
        with self._activity_lock:
            return self._last_activity

    def idle_seconds(self) -> float:
        return time.time() - self.last_activity()

    def is_idle(self, idle_seconds: float) -> bool:
        return self.idle_seconds() >= idle_seconds

    def review_sources(self, batch_id: str = None, max_batches: int = 3) -> dict:
        """溯源巡检（C3）：把"该批原文 + 该批产出的节点"交给 LLM 核对，**只报不改**。

        依赖 C1（原文按批次落盘，`ARIADNE_SOURCE_STORE=1`）。仅读取 `sources/` 目录，
        不写图谱；LLM 调用在锁外进行（持锁只做一次节点快照）。

        与 review_graph 共用 `_review_running` 非阻塞互斥（该锁为 RLock：空闲巡检整轮
        持锁后再调用本方法，同线程重入必须放行）。这样两次手动调用、以及手动与自动
        巡检之间都不会并发跑 LLM。
        """
        if not self._review_running.acquire(blocking=False):
            return {"error": "已有一次巡检正在进行，请稍后再试", "running": True}
        try:
            self._refresh_graph_only()  # 体检前重载外部改动

            return self._review_sources_impl(batch_id=batch_id, max_batches=max_batches)
        finally:
            self._review_running.release()

    def _review_sources_impl(self, batch_id: str = None, max_batches: int = 3) -> dict:
        source_dir = getattr(self.dba, "source_dir", None)
        if not source_dir:
            return {
                "error": "未开启原文存储，无法做溯源核对",
                "hint": "设置环境变量 ARIADNE_SOURCE_STORE=1 后重启；"
                        "注意这会落盘原始对话，请在确认隐私与体积可接受后再开。",
            }
        inference = getattr(self.retriever, "inference", None)
        if inference is None:
            return {"error": "检索链路未初始化，无法调用 LLM 做核对"}

        if batch_id:
            targets = [{"batch_id": batch_id}]
        else:
            targets = list_batches(source_dir, limit=max_batches)
        if not targets:
            return {"error": "没有可用的批次原文", "source_dir": str(source_dir)}

        reports = []
        for t in targets:
            bid = t.get("batch_id")
            batch = load_batch(source_dir, bid)
            if not batch:
                continue
            nodes_text, node_ids = self._snapshot_batch_nodes(bid)
            if not nodes_text:
                reports.append({
                    "batch_id": bid,
                    "note": "该批产出的节点已不在图中（可能被合并、废弃或遗忘），跳过核对",
                })
                continue
            verdict = inference.audit_source(render_batch_text(batch), nodes_text)
            report = {
                "batch_id": bid,
                "rounds": len(batch.get("rounds") or []),
                "audited_nodes": node_ids,
            }
            report.update(verdict)
            reports.append(report)

        total_unsupported = sum(len(r.get("unsupported") or []) for r in reports)
        total_missing = sum(len(r.get("missing") or []) for r in reports)
        return {
            "batches": reports,
            "summary": {
                "batches_checked": len(reports),
                "unsupported": total_unsupported,
                "missing": total_missing,
            },
            "note": ("本报告只做核对，**未修改图谱**。确认后请用 dba_intervene 落盘："
                     "无据节点用 deprecate（保留痕迹）或 delete_node（彻底删除），"
                     "漏抽的事实用 create_node 补上（可带 evidence 与 timestamp）。"
                     "注意：原文落盘时已做标识脱敏，个别片段可能带 ***。"),
        }

    def _snapshot_batch_nodes(self, batch_id: str):
        """持锁取一次快照：source_ref == batch_id 的节点（只读，锁外做 LLM 调用）"""
        with self._lock:
            lines, ids = [], []
            for nid, nd in self.graph.graph.nodes(data=True):
                if nd.get("source_ref") != batch_id:
                    continue
                if nd.get("deprecated") or nd.get("forgotten"):
                    continue
                nt = nd.get("node_type")
                nt_val = nt.value if hasattr(nt, "value") else str(nt)
                ev = f'  evidence: {nd["evidence"]}' if nd.get("evidence") else ""
                lines.append(f"[{nid}]（{nt_val}）{nd.get('content') or ''}{ev}")
                ids.append(nid)
        return "\n".join(lines), ids

    def review_graph(self, threshold: float = 0.85, max_pairs: int = 50,
                     include_isolated: bool = True) -> dict:
        """图谱体检（**只读**）：找语义高度相似的节点对与孤立节点，不修改图谱。

        设计取舍：
        - **只发现候选，不下判断**——是否真的同一条事实、要不要合并，交给上层结合
          上下文判断，确认后用 `dba_intervene`（update_node / delete_node）落盘。
          因此本工具不调 LLM、不写图，也就不存在与检索的写冲突。
        - 相似度计算是 O(n²)，**不能持锁跑**：只在开始时持锁取一次快照
          （节点字段 + 向量缓存浅拷贝），随后放锁做纯计算。
        """
        if not self._review_running.acquire(blocking=False):
            return {"error": "已有一次巡检正在进行，请稍后再试", "running": True}
        try:
            self._refresh_graph_only()  # 体检前重载外部改动
            with self._lock:
                active = []
                for nid, nd in self.graph.graph.nodes(data=True):
                    if nd.get("deprecated") or nd.get("forgotten"):
                        continue
                    active.append({
                        "id": nid,
                        "content": nd.get("content") or "",
                        "node_type": nd.get("node_type"),
                    })
                # 浅拷贝：向量数组本身不被就地修改，替换只会换引用 → 天然构成快照
                vectors = dict(self.vector_store._content_vectors) if self.vector_store else {}

            pairs = find_duplicate_candidates(
                active, vectors, threshold=threshold, max_pairs=max_pairs,
            )
            index = {n["id"]: n for n in active}
            for p in pairs:
                for side in ("a", "b"):
                    nd = index.get(p[side]) or {}
                    p[side] = {
                        "id": p[side],
                        "content": nd.get("content", ""),
                        "node_type": (str(nd.get("node_type") or "").split(".")[-1].lower()),
                    }

            isolated = []
            if include_isolated:
                with self._lock:
                    isolated = [
                        {"id": nid, "content": (nd.get("content") or "")}
                        for nid, nd in self.graph.graph.nodes(data=True)
                        if not nd.get("deprecated") and not nd.get("forgotten")
                        and self.graph.graph.in_degree(nid) + self.graph.graph.out_degree(nid) == 0
                    ]

            return {
                "scanned": len(active),
                "duplicate_candidates": pairs,
                "isolated_nodes": isolated,
                "threshold": threshold,
                "note": ("本报告只做候选发现，**未修改图谱**。确认后请用 dba_intervene "
                         "（update_node 合并内容 / delete_node 删除冗余）落盘；"
                         "注意节点类型为 thing 的短内容（如日期锚点）已默认排除在比对之外。"),
            }
        finally:
            self._review_running.release()

    def inspect_graph(self, node_id: str = None) -> dict:
        """查看图谱：指定节点展开 1-hop 邻居"""
        self._refresh_graph_only()  # 让查看结果与面板编辑保持一致
        if not node_id:
            return {"error": "缺少 node_id"}
        if node_id not in self.graph.graph.nodes:
            return {"error": f"节点不存在: {node_id}"}

        nid = node_id
        nodes = []
        edges = []
        seen_nodes = {nid}

        node = self.graph.graph.nodes[nid]
        nt = node.get("node_type")
        entry = {
            "id": nid,
            "content": node.get("content", ""),
            "node_type": nt.value if hasattr(nt, "value") else str(nt),
            "deprecated": node.get("deprecated", False),
            "forgotten": node.get("forgotten", False),
        }
        # 原文支撑片段（opt-in 抽取时才有）：用于核查这条节点凭什么存在
        if node.get("evidence"):
            entry["evidence"] = node["evidence"]
        nodes.append(entry)

        # 出边
        for _, tgt, edata in self.graph.graph.out_edges(nid, data=True):
            rt = edata["rel_type"]
            tgt_node = self.graph.graph.nodes[tgt]
            tgt_nt = tgt_node.get("node_type")
            if tgt not in seen_nodes:
                seen_nodes.add(tgt)
                nodes.append({
                    "id": tgt,
                    "content": tgt_node.get("content", ""),
                    "node_type": tgt_nt.value if hasattr(tgt_nt, "value") else str(tgt_nt),
                })
            edges.append({
                "from": nid,
                "to": tgt,
                "type": rt.value if hasattr(rt, "value") else str(rt),
            })

        # 入边
        for src, _, edata in self.graph.graph.in_edges(nid, data=True):
            rt = edata["rel_type"]
            src_node = self.graph.graph.nodes[src]
            src_nt = src_node.get("node_type")
            if src not in seen_nodes:
                seen_nodes.add(src)
                nodes.append({
                    "id": src,
                    "content": src_node.get("content", ""),
                    "node_type": src_nt.value if hasattr(src_nt, "value") else str(src_nt),
                })
            edges.append({
                "from": src,
                "to": nid,
                "type": rt.value if hasattr(rt, "value") else str(rt),
            })

        return {"nodes": nodes, "edges": edges}

    def intervene(self, action: str, params: dict) -> dict:
        """人工干预 CRUD 操作（跨进程文件锁 + 进程内锁保护）"""
        with self.write_guard():
            with self._lock:
                return self._intervene_impl(action, params)

    def _intervene_impl(self, action: str, params: dict) -> dict:
        """intervene 实际实现（在锁内调用）"""
        result = {"action": action, "success": True}
        try:
            if action == "create_node":
                nt = NODE_TYPES.get(params.get("node_type", "").upper())
                if not nt:
                    return {"error": f"无效的节点类型: {params.get('node_type')}"}
                content = params.get("content", "")
                # 空内容会建出脏节点（且会用空串做 embedding 污染向量库），在入口挡掉
                if not isinstance(content, str) or not content.strip():
                    return {"error": "content 不能为空", "action": action}
                # 精确重复：直接拒绝（零误判）
                for nid, nd in self.graph.graph.nodes(data=True):
                    if (nd.get("content") or "").strip() == content.strip():
                        return {
                            "action": action,
                            "success": False,
                            "duplicate_of": nid,
                            "existing_content": nd.get("content", ""),
                            "message": (f"内容与已有节点 {nid} 完全相同，已拒绝创建；"
                                        "如需修改请改用 update_node。"),
                        }

                # 语义疑似重复：只提示、不阻断。短内容（如「2026年9月7日」这类日期锚点）
                # 在向量空间里彼此高度相似，据余弦拒绝会挡住合法的日期锚点建设。
                hint = self._semantic_duplicate_hint(content)

                # 可选时间戳：与抽取路径一致，允许调用方给人工节点注入记录时间
                ts = params.get("timestamp")
                if ts is not None and (isinstance(ts, bool) or not isinstance(ts, int)):
                    return {"error": "timestamp 必须为 Unix 毫秒整数，或省略", "action": action}

                # 动态计算 ID，避免与 DBA 维护新建节点冲突导致静默覆盖
                self._next_node_id = self._compute_next_id()
                nid = f"n{self._next_node_id}"
                self._next_node_id += 1
                self.graph.graph.add_node(
                    nid,
                    content=content,
                    node_type=nt,
                    metadata={},
                    deprecated=False,
                    forgotten=False,
                )
                if ts is not None:
                    self.graph.graph.nodes[nid]["timestamp"] = ts
                # 同步向量库，使人工创建的节点可被 P 链路检索
                if self.vector_store is not None:
                    try:
                        self.vector_store.add_memories([nid], [content])
                    except Exception:
                        self.graph.graph.remove_node(nid)
                        raise
                result["node"] = {"id": nid, "content": content,
                                  "node_type": params.get("node_type", "").upper()}
                if ts is not None:
                    result["node"]["timestamp"] = ts
                if hint:
                    result["duplicate_warning"] = hint
                self._save()

            elif action == "update_node":
                nid = params["node_id"]
                if nid not in self.graph.graph.nodes:
                    return {"error": f"节点不存在: {nid}"}
                node = self.graph.graph.nodes[nid]
                if "content" in params:
                    old_content = node.get("content", "")
                    node["content"] = params["content"]
                    # 同步向量（失败时回滚内容并向调用方报错，与 GraphBuilder 一致）
                    if self.vector_store is not None:
                        try:
                            self.vector_store.update_memories([nid], [params["content"]])
                        except Exception as e:
                            node["content"] = old_content
                            return {"error": f"向量更新失败，内容已回滚: {e}", "action": action}
                if "node_type" in params:
                    nt = NODE_TYPES.get(params["node_type"].upper())
                    if not nt:
                        return {"error": f"无效的节点类型: {params['node_type']}"}
                    node["node_type"] = nt
                if "deprecated" in params:
                    node["deprecated"] = bool(params["deprecated"])
                if "timestamp" in params:
                    # 允许给存量节点补时间戳（节点的创建时间无法由抽取侧回填，只能手工注入）
                    ts = params["timestamp"]
                    if ts is not None and (isinstance(ts, bool) or not isinstance(ts, int)):
                        return {"error": "timestamp 必须为 Unix 毫秒整数，或 null 表示清空",
                                "action": action}
                    node["timestamp"] = ts
                result["node"] = {"id": nid, "content": node.get("content", ""),
                                  "timestamp": node.get("timestamp")}
                self._save()

            elif action == "delete_node":
                nid = params["node_id"]
                if nid not in self.graph.graph.nodes:
                    return {"error": f"节点不存在: {nid}"}
                # 先清向量、再删图节点：原先"先删图、后清向量"在向量清理失败时会留下
                # 「内存图已删 / YAML 未删」的部分状态（且 return 跳过了 _save）。
                # 现在失败即原地返回，图与向量保持一致。
                if self.vector_store is not None:
                    try:
                        self.vector_store.remove_memories([nid])
                    except Exception as e:
                        return {"error": f"向量清理失败，节点未删除: {e}", "action": action}
                self.graph.graph.remove_node(nid)
                result["deleted"] = nid
                self._save()

            elif action == "create_edge":
                src, tgt = params["source"], params["target"]
                rt = REL_TYPES.get(params.get("rel_type", "").lower())
                if not rt:
                    return {"error": f"无效的边类型: {params.get('rel_type')}"}
                if src == tgt:
                    return {"error": "不能创建自环边"}
                if src not in self.graph.graph.nodes:
                    return {"error": f"源节点不存在: {src}"}
                if tgt not in self.graph.graph.nodes:
                    return {"error": f"目标节点不存在: {tgt}"}
                # 跳转轴权重校验：权重为 0 的方向不允许建边
                src_type = self.graph.get_node_type(src)
                if src_type and get_jump_weight(src_type, rt, is_reverse=False) <= 0:
                    return {"error": f"边 {src}--[{rt.value}]-->{tgt} 在该节点类型上权重为 0，不允许创建"}
                if self.graph.graph.has_edge(src, tgt):
                    return {"error": f"边已存在: {src} -> {tgt}"}
                self.graph.graph.add_edge(src, tgt, rel_type=rt)
                # 双向类型自动补反向边
                if rt in (RelationType.SCENARIO, RelationType.SOCIAL, RelationType.ATTRIBUTE):
                    if not self.graph.graph.has_edge(tgt, src):
                        self.graph.graph.add_edge(tgt, src, rel_type=rt)
                result["edge"] = {"source": src, "target": tgt, "rel_type": params["rel_type"]}
                self._save()

            elif action == "delete_edge":
                src, tgt = params["source"], params["target"]
                if not self.graph.graph.has_edge(src, tgt):
                    return {"error": f"边不存在: {src} -> {tgt}"}
                rt = self.graph.graph.edges[src, tgt].get("rel_type")
                self.graph.graph.remove_edge(src, tgt)
                # 双向类型同步删除反向边
                if rt in (RelationType.SCENARIO, RelationType.SOCIAL, RelationType.ATTRIBUTE):
                    if self.graph.graph.has_edge(tgt, src):
                        self.graph.graph.remove_edge(tgt, src)
                result["deleted"] = f"{src} -> {tgt}"
                self._save()

            elif action == "set_params":
                # 调整运行时检索参数（权重矩阵 / 种子 / 目的回归）。
                # 允许两种写法：params 直接是参数补丁，或 {"patch": {...}}。
                raw = params.get("patch") if isinstance(params.get("patch"), dict) else params
                out = self.set_params(raw)
                if out.get("error"):
                    return {"error": out["error"], "action": action}
                result["applied"] = out.get("applied")
                result["params"] = out.get("params")
                result["path"] = out.get("path")
                result["note"] = ("参数已写入共享文件并即时生效；"
                                  "WebUI 面板的「参数」页看到的是同一份配置。")

            else:
                return {"error": f"未知的干预类型: {action}"}

        except Exception as e:
            return {"error": str(e), "action": action}

        return result

    def checkpoint(self, save_dir: str = None) -> dict:
        """保存检查点（若 DBA 管线已接入则保存完整状态）"""
        if self.dba:
            target = save_dir or "snapshots/latest"
            try:
                self.dba.save_checkpoint(target)
                # 额外保存调度器状态
                scheduler_state = None
                if self.scheduler:
                    sched_path = Path(target) / "scheduler_state.json"
                    with open(sched_path, "w", encoding="utf-8") as f:
                        json.dump(self.scheduler.save_state(), f, ensure_ascii=False, indent=2)
                    scheduler_state = "saved"
                return {
                    "save_dir": target,
                    "nodes": self.graph.node_count,
                    "edges": self.graph.edge_count,
                    "vectors": "saved",
                    "builder_state": "saved",
                    "scheduler_state": scheduler_state,
                }
            except Exception as e:
                return {"error": str(e), "save_dir": target}

        # 回退：仅保存 YAML
        target = save_dir or (str(Path(self.yaml_path).parent) if self.yaml_path else "snapshots/latest")
        Path(target).mkdir(parents=True, exist_ok=True)
        out_path = Path(target) / "memory_graph.yaml"
        data = self.graph.to_dict()
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return {
            "save_dir": str(out_path),
            "nodes": self.graph.node_count,
            "edges": self.graph.edge_count,
        }

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        self._refresh_graph_only()  # 统计口径与面板编辑保持一致
        nodes = list(self.graph.graph.nodes(data=True))
        deprecated = sum(1 for _, d in nodes if d.get("deprecated"))
        forgotten = sum(1 for _, d in nodes if d.get("forgotten"))

        node_types = {}
        for _, d in nodes:
            nt = d.get("node_type")
            key = nt.value if hasattr(nt, "value") else str(nt)
            node_types[key] = node_types.get(key, 0) + 1

        return {
            "total_nodes": len(nodes),
            "total_edges": self.graph.edge_count,
            "deprecated": deprecated,
            "forgotten": forgotten,
            "active": len(nodes) - deprecated - forgotten,
            "node_types": node_types,
        }


# ---- MCP Tool Schema ----

TOOL_SCHEMAS = [
    {
        "name": "dba_add_conversation",
        "description": (
            "当用户在对话中透露新的个人信息、状态、偏好、经历、人际关系、"
            "计划等值得长期记忆的事实时，自动调用此工具记录，无需用户明确要求。"
            "纯寒暄、客套、追问细节等无新信息的内容不需要调用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation": {
                    "type": "string",
                    "description": (
                        "要记录的对话文本。多人对话请保留说话人标识"
                        "（如「[阿哲] 我最近压力挺大」），否则无法区分事实归属；"
                        "单人对话无需标识。"
                    ),
                }
            },
            "required": ["conversation"],
        },
    },
    {
        "name": "dba_query_memory",
        "description": (
            "在回答用户问题、给出建议或延续话题之前，先调用此工具检索与查询相关的"
            "历史记忆，返回按因果链路整理好的故事片段，以便结合用户过去的上下文做出更贴合的回答。\n"
            "故事里若出现「记录于 YYYY-MM-DD」，那是这条记忆**被记下**的时间，"
            "**不是事件发生时间**，不要据此断言事情发生在哪一天。\n"
            "适用范围（除了下面 dba_temporal_lookup 明确的【在某个时间发生了什么】之外，都用这个）：\n"
            "- 问原因/为什么（因果）、现状、整体看法、与谁/什么有关、偏爱、特性\n"
            "- 问某个具体事件发生在什么时候（事件→时间）、跨多个时期的完整经历（从高中到工作…）\n"
            "- 需要跨节点联想/推理/先后顺序的记忆检索"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询文本，用于联想检索相关的记忆",
                },
                "rerank_k": {
                    "type": "integer",
                    "description": "最多返回的故事片段数（默认 20，0 表示不截断）",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "dba_temporal_lookup",
        "description": (
            "【只在用户问“在某个具体时间点发生了哪些事/那个时间做了什么”时调用】——即“时间→事件”的共时查询。\n"
            "典型信号：昨天/今天/上周/上个月/高中学期/大二上/高三下学期 等时间词 + 做了什么/发生了什么事/经历了什么。\n"
            "它会定位到时间锚点，返回同一时间发生的几个事实节点（单跳，扁平列表），用于回答“那个时间我都干了啥”。\n"
            "注意：查询可能命中多个时间锚点（如“三年”可指大专三年或工作三年），工具会**返回全部候选锚点的事实组**，"
            "请你结合问题判断最相关的一个/多个。\n"
            "返回值里的 match_type 标明本次命中的强度，请据此决定要不要采信：\n"
            "- exact：查询里的显式日期（如“上周三”“9月7日”）字面命中了锚点，可放心使用；\n"
            "- fuzzy：查询用的是模糊时间词（前几天/最近/那天/后来），锚点来自该类词允许的时间范围，"
            "可用但精度较低，回答时不要把它说成精确日期；\n"
            "- fallback：查询不含任何时间词，返回的只是**语义最近**的锚点，**不代表时间吻合**——"
            "回答时不要断言具体日期，必要时向用户确认；\n"
            "- rejected：库里没有该时间段的记忆，已主动拒答（不是检索失败），此时 matches 为空。\n"
            "rejected 为 true 时会附 reason 说明判定依据。**此时应直接告诉用户“没有那个时段的记录”，"
            "不要拿其他时段的事实顶替，也不要反复换查询**；\n"
            "matches 为空但未拒答时另附 note，提示是否该改用 dba_query_memory。\n"
            "每条事实另带 time_consistent：false 表示该事实自带的时期词与锚点时间不相交，"
            "很可能是连边错配（时间上对不上），请勿据此断言；null 表示事实里没有时期词、无法判定。\n"
            "【不要用于】以下场景——这些请改用 dba_query_memory：\n"
            "- 问“什么时候做了某件事”（事件→时间，如‘什么时候开始学吉他’）\n"
            "- 问“为什么/现状/整体经历/跨多个时期的完整历程”（需因果联想/多期跨越）\n"
            "- 没有明确单一时间锚点的联想类问题"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询文本，用于定位时间锚点并取该时间发生的共时事实",
                },
                "k_seed": {
                    "type": "integer",
                    "description": "用于语义定位时间锚点的候选数（默认 8）",
                },
                "max_facts": {
                    "type": "integer",
                    "description": "每个锚点最多返回的共时事实数（默认 8，0 表示不截断）",
                },
                "max_anchors": {
                    "type": "integer",
                    "description": "最多返回的时间锚点候选数（默认 3，按相关性从高到低）",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "dba_inspect_graph",
        "description": "查看图谱：指定节点展开 1-hop 邻居",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "节点 ID（如 n12），展开其 1-hop 邻居",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "dba_intervene",
        "description": "人工干预：创建/更新/删除节点或边，或调整运行时检索参数（权重矩阵/种子/目的回归）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create_node", "update_node", "delete_node", "create_edge", "delete_edge",
                             "set_params"],
                    "description": "干预操作类型",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "操作参数。"
                        "create_node: {node_type, content, timestamp?}；"
                        "update_node: {node_id, content?, node_type?, deprecated?, timestamp?}"
                        "（timestamp 为 Unix 毫秒整数，null 表示清空；给存量节点补记录时间用它）；"
                        "delete_node: {node_id}；"
                        "create_edge: {source, target, rel_type}；"
                        "delete_edge: {source, target}；"
                        "set_params: {retrieval?: {seed_k, max_hops, expand_k, temporal_*},"
                        " scoring?: {distance_decay, jump_weight_coef, purpose_weight_coef,"
                        " purpose_filter_threshold, purpose_filter_decay},"
                        " weights?: {节点类型: {关系类型: [正向, 反向]}}}"
                        "（只传要改的部分即可，未传字段保持现值；权重与系数取值区间 0~1）。"
                        "返回值提示：create_node 命中**完全相同**的内容时 success=false 并给出 duplicate_of；"
                        "仅是语义高度相似时创建仍成功，但附 duplicate_warning"
                        "（此时请判断是否应当改用 update_node，以免造出重复节点）。"
                    ),
                },
            },
            "required": ["action", "params"],
        },
    },
    {
        "name": "dba_checkpoint",
        "description": "保存当前图谱到检查点文件",
        "inputSchema": {
            "type": "object",
            "properties": {
                "save_dir": {
                    "type": "string",
                    "description": "保存目录路径（默认为 yaml 文件所在目录）",
                },
            },
        },
    },
    {
        "name": "dba_get_stats",
        "description": "获取当前图谱的统计信息（节点数、边数、废弃节点数等）",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "dba_review_graph",
        "description": (
            "【图谱体检 · 只读】找出语义高度相似的节点对（疑似重复）与孤立节点，"
            "返回报告但**不修改图谱**。适合在记忆攒了一段时间后人工触发一次。\n"
            "- 不调用 LLM、不写图，因此不会与正在进行的检索/维护冲突。\n"
            "- 语义去重只对「事实陈述」有意义：类型为 thing 的短内容（日期锚点这类）"
            "默认排除——否则「2026年9月7日」与「2026年8月31日」会大量误报。\n"
            "- 孤立节点会因连续多轮无人访问而被遗忘，报告里提前暴露，可按需连边或删除。\n"
            "- 确认某对确实是同一条事实后，用 dba_intervene 落盘："
            "update_node 把内容合并到保留的那条，delete_node 删掉冗余的那条。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "余弦相似度阈值，默认 0.85；调低会报出更多弱候选",
                },
                "max_pairs": {
                    "type": "integer",
                    "description": "最多返回多少对候选（默认 50，0 表示不截断）",
                },
                "include_isolated": {
                    "type": "boolean",
                    "description": "是否同时返回孤立节点列表（默认 true）",
                },
            },
        },
    },
    {
        "name": "dba_review_sources",
        "description": (
            "【溯源巡检 · 只读】用**原始对话**核对从它抽出来的节点，报三类问题且不修改图谱：\n"
            "- unsupported：节点的内容在原文里找不到支撑（把推测写成了事实这类）；\n"
            "- missing：原文里有、但没抽出来的关键事实，重点是「为什么做某个选择」"
            "（被放弃的选项、取舍依据）；\n"
            "- granularity：粒度不对（相对时间表述被独立成节点、多件事被合并成一条）。\n"
            "需要服务端开启原文存储（环境变量 ARIADNE_SOURCE_STORE=1）才能用；"
            "未开启时会返回提示。不传 batch_id 则核对最近几批。\n"
            "报告只是建议：确认后用 dba_intervene 落盘（无据节点 deprecate/delete_node，"
            "漏抽的用 create_node 补）。原文落盘时已做标识脱敏，个别片段可能带 ***。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "batch_id": {
                    "type": "string",
                    "description": "指定要核对哪一批（形如 b20260911-143000-ab12）；不传则取最近几批",
                },
                "max_batches": {
                    "type": "integer",
                    "description": "不指定 batch_id 时，最多核对最近多少批（默认 3）",
                },
            },
        },
    },
]


def _build_tool_list() -> list:
    """将 TOOL_SCHEMAS 转为 MCP Tool 对象列表"""
    tools = []
    for ts in TOOL_SCHEMAS:
        tools.append(Tool(
            name=ts["name"],
            description=ts["description"],
            input_schema=ts["inputSchema"],
        ))
    return tools


def create_mcp_server(dba: DBAServer) -> "Server":
    """构建 MCP Server 实例 (MCP v2.0 API)"""
    server = Server("ariadne")
    # 操作日志路径（与 MCP 图谱 YAML 同目录，便于追溯）
    log_path = oplog.default_oplog_path(dba.yaml_path)

    async def handle_list_tools(ctx, params: PaginatedRequestParams):
        return ListToolsResult(tools=_build_tool_list())

    async def handle_call_tool(ctx, params: CallToolRequestParams):
        name = params.name
        arguments = params.arguments or {}
        # 统一记一次活动：空闲巡检据此判断"现在有人用"，从而不启动或中途让路。
        # 放在 dispatch 层而不是各 handler 里，是为了不漏——新增工具自动被覆盖。
        dba.touch(source=name)
        # 记录本次 LLM 操作（source=llm）
        session_id = getattr(ctx, "session_id", "") if ctx is not None else ""
        actor = {
            "type": "llm",
            "model": os.environ.get("OPENAI_MODEL", ""),
        }
        if session_id:
            actor["session_id"] = session_id

        try:
            if name == "dba_add_conversation":
                result = dba.add_conversation(**arguments)
            elif name == "dba_query_memory":
                result = dba.query_memory(**arguments)
            elif name == "dba_temporal_lookup":
                result = dba.temporal_lookup(**arguments)
            elif name == "dba_inspect_graph":
                result = dba.inspect_graph(**arguments)
            elif name == "dba_intervene":
                result = dba.intervene(**arguments)
            elif name == "dba_checkpoint":
                result = dba.checkpoint(**arguments)
            elif name == "dba_get_stats":
                result = dba.get_stats()
            elif name == "dba_review_graph":
                result = dba.review_graph(**arguments)
            elif name == "dba_review_sources":
                result = dba.review_sources(**arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            oplog.log_operation("llm", op=name, request=arguments,
                                result={"status": "error", "error": str(e)},
                                actor=actor, tool=name, path=log_path)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))]
            )

        oplog.log_operation("llm", op=name, request=arguments, result=result,
                            actor=actor, tool=name, path=log_path)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        )

    server.add_request_handler("tools/list", PaginatedRequestParams, handle_list_tools)
    server.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

    return server


async def run_stdio(server: "Server"):
    """启动 stdio 模式的 MCP Server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse(server: "Server", host: str, port: int, store: "webauth.AuthStore"):
    """启动 SSE 模式的 MCP Server

    与面板共用同一份凭据文件（``<图谱目录>/auth.json``），客户端用 HTTP Basic；
    也可用 ``ARIADNE_MCP_TOKEN`` 走 Bearer。凭据文件里仍是初始密码时拒绝
    Basic，避免默认账号被长期使用（先在 WebUI 面板改密再接入）。
    """
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import Response
    from starlette.routing import Mount, Route

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )
        # 返回空响应，避免 Starlette 因 endpoint 返回 None 报 "NoneType is not callable"
        return Response()

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )
    # MCP 没有登录页：未通过一律 401
    starlette_app.add_middleware(
        webauth.AuthMiddleware,
        store=store,
        bearer_token=webauth.load_bearer_token(),
        exempt_paths=(),
        login_path=None,
    )

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    http_server = uvicorn.Server(config)
    await http_server.serve()


def _env_flag(name: str) -> bool:
    """将环境变量解析为布尔值（用于 store_true 的默认值）"""
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


def _local_embeddings_available() -> bool:
    """检测 sentence-transformers 是否已安装（EMBEDDING_LOCAL=true 时需要）"""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _print_status(graph, dba, retriever, vector_ready, args) -> None:
    """输出统一的状态日志"""
    retrieval = "P 链路（跳转轴+目的回归+寻峰）" if retriever else "未启用"
    llm = args.llm_model or "未配置"
    if args.embedding_model:
        emb = f"{args.embedding_model}（{'本地' if args.embedding_local else 'API'}）"
    else:
        emb = "未配置"
    transport = "SSE" if args.sse else "stdio"

    print("=" * 60, file=sys.stderr)
    print("[DBA MCP] 运行状态", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  检索模式   : {retrieval}", file=sys.stderr)
    print(f"  LLM        : {llm}", file=sys.stderr)
    print(f"  Embedding  : {emb}", file=sys.stderr)
    print(f"  向量索引   : {'已就绪' if vector_ready else '未启用'}", file=sys.stderr)
    print(f"  传输模式   : {transport}", file=sys.stderr)
    print(f"  图谱规模   : {graph.node_count} 节点 / {graph.edge_count} 边", file=sys.stderr)
    print(f"  工具        : dba_add_conversation / dba_query_memory / dba_temporal_lookup / "
          f"dba_inspect_graph / dba_intervene / dba_review_graph / dba_review_sources / "
          f"dba_checkpoint / dba_get_stats", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def main():
    if not HAS_MCP:
        print("错误: MCP SDK 未安装。请先运行: pip install mcp", file=sys.stderr)
        sys.exit(1)

    envfile.load_dotenv()

    parser = argparse.ArgumentParser(description="DBA MCP Server")
    parser.add_argument("--yaml", required=True, help="YAML checkpoint 文件路径")
    parser.add_argument("--params", default=None,
                        help="运行时检索参数文件（权重矩阵/种子/目的回归）；"
                             "默认与图谱同目录 retrieval_params.yaml")
    parser.add_argument("--sse", action="store_true", help="使用 SSE 网络模式（默认 stdio）")
    parser.add_argument("--host", default="127.0.0.1", help="SSE 绑定地址")
    parser.add_argument("--port", type=int, default=8765, help="SSE 端口")

    # ---- 可选 DBA 管线参数（优先级：命令行 > 环境变量）----
    parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL"),
                        help="LLM 模型名（默认读环境变量 OPENAI_MODEL）")
    parser.add_argument("--llm-api-key", default=os.environ.get("OPENAI_API_KEY"),
                        help="LLM API Key（默认读环境变量 OPENAI_API_KEY）")
    parser.add_argument("--llm-base-url", default=os.environ.get("OPENAI_API_BASE"),
                        help="LLM API Base URL（默认读环境变量 OPENAI_API_BASE）")
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL"),
                        help="Embedding 模型名（默认读环境变量 EMBEDDING_MODEL）")
    parser.add_argument("--embedding-api-key", default=os.environ.get("EMBEDDING_API_KEY"),
                        help="Embedding API Key（默认读环境变量 EMBEDDING_API_KEY）")
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBEDDING_API_BASE"),
                        help="Embedding API Base URL（默认读环境变量 EMBEDDING_API_BASE）")
    parser.add_argument("--embedding-local", action="store_true",
                        default=_env_flag("EMBEDDING_LOCAL"),
                        help="使用本地 sentence-transformers 模型（默认读环境变量 EMBEDDING_LOCAL）")
    parser.add_argument("--vector-index", default=None, help="FAISS 索引文件路径（可选，用于恢复向量索引）")
    parser.add_argument("--restore-dir", default=None, help="从 checkpoint 目录完整恢复（图谱+向量+构建器+调度器状态）")
    parser.add_argument("--log-level", default=os.environ.get("ARIADNE_LOG_LEVEL", "INFO"),
                        help="服务日志级别（写入共享日志文件，供 WebUI 面板聚合展示；默认 INFO）")
    args = parser.parse_args()

    # 跨进程服务日志：写入共享 JSONL 文件（默认 <图谱目录>/ariadne.log），
    # WebUI 面板 tail 该文件即可看到 MCP 的运行日志；控制台仍只保留 WARNING 以上。
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _log_file = logbus.default_log_file_path(args.yaml)
    logbus.install_service_logging("mcp", _log_file, level=getattr(logging, args.log_level.upper(), logging.INFO))
    logging.getLogger("mcp").info("MCP 服务启动: yaml=%s log=%s", args.yaml, _log_file)

    print(f"[DBA MCP] 加载图数据: {args.yaml}", file=sys.stderr)
    graph = load_graph(args.yaml)

    # 共享锁：串行化 graph 修改（DBA 维护 + 人工干预）
    graph_lock = threading.RLock()

    # 构建 DBA 管线
    dba_instance = None
    scheduler_instance = None
    retriever_instance = None
    vector_store = None
    vector_ready = False

    # 只保留完整 DBA 链路：缺少 LLM / DBA 依赖 / 本地 embedding 依赖时直接退出
    if not args.llm_model:
        print("错误: 需要配置 LLM（--llm-model 或环境变量 OPENAI_MODEL）", file=sys.stderr)
        sys.exit(1)
    if not HAS_DBA:
        print("错误: DBA 管线依赖未安装（langchain 等），无法启动", file=sys.stderr)
        sys.exit(1)
    if args.embedding_model and args.embedding_local and not _local_embeddings_available():
        print("错误: EMBEDDING_LOCAL=true 但未安装 sentence-transformers。", file=sys.stderr)
        print("请安装本地依赖后再启动: pip install -e \".[local]\"", file=sys.stderr)
        sys.exit(1)

    if args.llm_model and HAS_DBA:
        print("[DBA MCP] 构建 DBA 管线...", file=sys.stderr)
        try:
            from dba_pipeline.extraction.maintenance_scheduler import MaintenanceScheduler, ScheduleConfig

            # Embeddings: API 或本地
            if args.embedding_model:
                if args.embedding_local:
                    embeddings = LocalEmbeddings(model_name=args.embedding_model)
                else:
                    embeddings = OpenAIEmbeddings(
                        api_key=args.embedding_api_key or args.llm_api_key or "",
                        api_base=args.embedding_base_url or args.llm_base_url or "",
                        model=args.embedding_model,
                    )
            else:
                embeddings = None

            # VectorStore
            vector_store = VectorStore(
                embeddings=embeddings,
                backend="faiss",
                persist_dir=None,
            )
            if args.vector_index:
                if os.path.exists(args.vector_index):
                    vector_store.load(args.vector_index, embeddings=embeddings)

            # GraphBuilder
            builder = GraphBuilder(graph=graph, vector_store=vector_store, lock=graph_lock)

            # LLM：远程 API 必须显式配置 key；本地模型（自定义 base_url）才用占位符
            if not args.llm_api_key and not args.llm_base_url:
                print("错误: 需要配置 LLM API Key（--llm-api-key 或环境变量 OPENAI_API_KEY）"
                      "或指定 --llm-base-url（本地模型）", file=sys.stderr)
                sys.exit(1)
            llm = ChatOpenAI(
                model=args.llm_model,
                api_key=args.llm_api_key or "not-needed",
                base_url=args.llm_base_url,
                temperature=0,
            )

            # DBA
            dba_instance = MemoryDBA(
                llm=llm,
                graph=graph,
                vector_store=vector_store,
                graph_builder=builder,
                # 「决策与取舍」补充规则（把"被放弃的选项 / 选择依据"也抽成 REASON 节点）。
                # 仅 MCP 侧开启；ALM 侧保持关闭，避免动到待提交的评测行为。
                enable_choice_extraction=True,
                # 每条节点附带 evidence（原文支撑片段），供核查与后续溯源巡检使用；
                # 不进检索/叙事，仅在 inspect_graph 可见，且受子串校验兜底。
                enable_evidence=True,
                # 原文按批次落盘（溯源巡检的前提）。**默认关闭**：存的是原始对话，
                # 需显式设 ARIADNE_SOURCE_STORE=1 才开；落盘时做标识脱敏 + 30 天 TTL。
                enable_source_store=source_store_enabled(),
                source_dir=default_source_dir(args.yaml) if source_store_enabled() else None,
            )

            # Scheduler（异步批量维护）
            config = ScheduleConfig()
            scheduler_instance = MaintenanceScheduler(
                dba=dba_instance,
                config=config,
            )

            # 完整检索链路（跳转轴 + 目的回归 + 寻峰终止），失败不影响 DBA 维护
            if embeddings is not None:
                try:
                    from dba_pipeline.llm.inference import InferenceEngine
                    from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever

                    # 启动时把当前图节点灌入向量存储（仅在索引为空时）
                    if vector_store.store is None:
                        mem_ids = [
                            nid for nid in graph.graph.nodes()
                            if graph.graph.nodes[nid].get("content")
                        ]
                        contents = [graph.graph.nodes[nid].get("content") for nid in mem_ids]
                        if mem_ids:
                            vector_store.add_memories(mem_ids, contents)

                    inference = InferenceEngine(llm)
                    path_tracker = PathTracker()
                    retriever_instance = PurposeDrivenRetriever(
                        llm=llm,
                        embeddings=embeddings,
                        graph=graph,
                        vector_store=vector_store,
                        inference=inference,
                        path_tracker=path_tracker,
                    )
                    vector_ready = vector_store.store is not None
                except Exception as e:
                    print(f"[DBA MCP] 检索链路初始化失败（检索不可用）: {e}", file=sys.stderr)

            print(f"[DBA MCP] DBA 管线就绪: LLM={args.llm_model}, "
                  f"检索={'P链路' if retriever_instance else '未启用'}", file=sys.stderr)
        except Exception as e:
            print(f"[DBA MCP] DBA 管线初始化失败: {e}", file=sys.stderr)
            print("错误: DBA 管线是核心功能，初始化失败将退出（避免带病启动）", file=sys.stderr)
            sys.exit(1)

    params_path = args.params or params_mod.params_path_for(args.yaml)
    dba_server = DBAServer(graph, yaml_path=args.yaml,
                           dba=dba_instance, scheduler=scheduler_instance,
                           retriever=retriever_instance, vector_store=vector_store,
                           lock=graph_lock, params_path=params_path)
    print(f"[DBA MCP] 运行时参数: {params_path}"
          f"{'' if os.path.exists(params_path) else '（尚未创建，用默认值）'}", file=sys.stderr)

    # 完整恢复 checkpoint（如果指定）
    if args.restore_dir and dba_instance:
        try:
            dba_instance.restore_checkpoint(args.restore_dir)
            if scheduler_instance:
                sched_path = os.path.join(args.restore_dir, "scheduler_state.json")
                if os.path.exists(sched_path):
                    with open(sched_path, encoding="utf-8") as f:
                        scheduler_instance.load_state(json.load(f))
            dba_server._next_node_id = dba_server._compute_next_id()
            # 同步向量库：若 checkpoint 未含向量索引，以恢复后的图谱为准重建
            if vector_store is not None and not os.path.exists(
                os.path.join(args.restore_dir, "faiss_index")
            ):
                mem_ids = [
                    nid for nid in graph.graph.nodes()
                    if graph.graph.nodes[nid].get("content")
                ]
                if mem_ids:
                    vector_store.clear_vectors()
                    vector_store.add_memories(
                        mem_ids,
                        [graph.graph.nodes[nid].get("content") for nid in mem_ids],
                    )
            print(f"[DBA MCP] 已从 checkpoint 恢复: {args.restore_dir}", file=sys.stderr)
        except Exception as e:
            print(f"[DBA MCP] checkpoint 恢复失败: {e}", file=sys.stderr)
            sys.exit(1)

    # 启动即对齐一次：把向量库与磁盘图谱同步（面板可能在 MCP 上次运行期间改过图谱）
    if vector_store is not None:
        try:
            _sync = dba_server.reconcile_vectors(force=True)
            if _sync.get("added") or _sync.get("updated") or _sync.get("removed"):
                print(f"[DBA MCP] 向量库启动对齐: +{_sync.get('added')} "
                      f"~{_sync.get('updated')} -{_sync.get('removed')}", file=sys.stderr)
        except Exception as e:
            print(f"[DBA MCP] 向量库启动对齐失败（沿用现有索引）: {e}", file=sys.stderr)

    # P0: 启动调度器，维护完成后自动保存 YAML
    if scheduler_instance:
        def _on_maintenance_done(result):
            dba_server._save()
            # 维护本身也是"系统活动"：让空闲巡检重新计时，避免巡检和维护撞在同一段空闲里
            dba_server.touch(source="maintenance")

        scheduler_instance.on_maintenance_done = _on_maintenance_done
        # 维护（LLM 改图 + 落盘）整段放入跨进程写临界区，避免与 WebUI 面板相互覆盖
        scheduler_instance.maintain_context = dba_server.write_guard
        scheduler_instance.start()

    # 空闲巡检：只在没人用的时段跑，一有人来就让路。默认关闭（见 review_scheduler 模块说明）。
    review_scheduler = None
    review_cfg = ReviewConfig.from_env(
        maintenance_idle_timeout=scheduler_instance.config.idle_timeout
        if scheduler_instance else None
    )
    if review_cfg and dba_instance is not None:
        def _on_review(report):
            if report.get("skipped") or report.get("aborted"):
                return
            summary = (report.get("graph") or {}).get("duplicate_candidates")
            print(f"[DBA MCP] 空闲巡检完成: 候选重复对 {len(summary or [])} 组, "
                  f"核对批次 {len(report.get('sources') or [])} 批; 报告见 {review_scheduler.review_dir}",
                  file=sys.stderr)

        review_scheduler = IdleReviewScheduler(
            dba_server, review_cfg, yaml_path=args.yaml, on_report=_on_review)
        review_scheduler.start()
        print(f"[DBA MCP] 空闲巡检已启动: 空闲阈值 {review_cfg.idle_seconds:.0f}s, "
              f"最小间隔 {review_cfg.min_interval:.0f}s, 每轮核对 {review_cfg.max_batches} 批原文",
              file=sys.stderr)
    else:
        print("[DBA MCP] 空闲巡检未启动（设 ARIADNE_REVIEW_IDLE=<秒> 开启；"
              "它会消耗 LLM 额度，故默认关闭）", file=sys.stderr)

    server = create_mcp_server(dba_server)

    _print_status(graph=graph, dba=dba_instance, retriever=retriever_instance,
                  vector_ready=vector_ready, args=args)

    try:
        if args.sse:
            import asyncio
            store = webauth.AuthStore.load_or_create(args.yaml)
            token = webauth.load_bearer_token()
            print(f"[DBA MCP] SSE 模式: http://{args.host}:{args.port}/sse  "
                  f"(鉴权: 开启，用户 {store.user}"
                  f"{'，另有 Bearer Token' if token else ''})", file=sys.stderr)
            if store.must_change:
                print("提示: 凭据仍是初始密码，请先在 WebUI 面板改密，再让客户端接入 MCP",
                      file=sys.stderr)
            warning = webauth.backdoor_warning(webauth.load_backdoor())
            if warning:
                print(warning, file=sys.stderr)
            asyncio.run(run_sse(server, args.host, args.port, store))
        else:
            import asyncio
            print("[DBA MCP] stdio 模式", file=sys.stderr)
            asyncio.run(run_stdio(server))
    finally:
        # 优雅退出：先停空闲巡检（它可能正在调 LLM），再 flush 调度器缓冲中的对话
        if review_scheduler:
            review_scheduler.stop()
        if scheduler_instance:
            scheduler_instance.stop()


if __name__ == "__main__":
    main()
