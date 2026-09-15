# SPDX-License-Identifier: AGPL-3.0-only

"""面板「调用测试」：在面板进程内跑 MCP 那条真实链路，并把中间阶段抓出来。

面板与 MCP 用的是同一份链路代码（``DBAServer`` + ``PurposeDrivenRetriever``），
这里**不另写一套检索**，只做三件事：

1. 按 ``dba_pipeline/mcp_server.main`` 的方式装配（同 embedding / LLM / 参数 / 对账），
   且**首次调用时才装配**——没用到就不付 embedding 与模型加载的成本，也不拖慢面板启动；
2. 把 retriever 的 ``on_stage`` 阶段回调转发给面板 SSE 总线，前端据此显示实时进度；
3. 用 ``DBAServer.run_chain_trace`` 拿到完整轨迹（每跳候选、分数、峰值、故事采纳）。

只读：不写图谱 YAML、不写向量索引（写盘仍只发生在面板 CRUD 与 MCP 的 add 类工具）。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("viz")

# 阶段回调：stage 名称 + 附加信息（会被转成 SSE 事件）
StageCb = Optional[Callable[[str, Dict[str, Any]], None]]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class ChainRunner:
    """懒加载的检索链路执行器（每个面板进程一份）"""

    def __init__(self, yaml_path: str):
        self.yaml_path = str(Path(yaml_path).resolve())
        self._server: Any = None
        self._run_lock = threading.Lock()      # 同一时刻只跑一次：链路与 PathTracker 非线程安全
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _init_failure(self) -> RuntimeError:
        return RuntimeError(
            f"检索链路装配失败: {self._init_error}"
            "（已记录，不再重试；修好配置后重启面板即可重试）"
        )

    # ── 装配 ────────────────────────────────────────────────────────────────

    def _build(self) -> Any:
        from langchain_openai import ChatOpenAI

        from dba_pipeline.core.path_tracker import PathTracker
        from dba_pipeline.embedding.store import (
            LocalEmbeddings, OpenAIEmbeddings, VectorStore,
        )
        from dba_pipeline.extraction.dba import MemoryDBA
        from dba_pipeline.extraction.graph_builder import GraphBuilder
        from dba_pipeline.llm.inference import InferenceEngine
        from dba_pipeline.loader import load_graph
        from dba_pipeline.mcp_server import DBAServer
        from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever

        emb_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        if _env_flag("EMBEDDING_LOCAL"):
            embeddings = LocalEmbeddings(model_name=emb_model)
        else:
            embeddings = OpenAIEmbeddings(
                api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
                api_base=os.environ.get("EMBEDDING_API_BASE") or os.environ.get("OPENAI_API_BASE") or "",
                model=emb_model,
            )
        llm = ChatOpenAI(
            model=os.environ.get("OPENAI_MODEL", ""),
            api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=os.environ.get("OPENAI_API_BASE"),
            temperature=0,
        )

        graph = load_graph(self.yaml_path)
        lock = threading.RLock()
        vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
        builder = GraphBuilder(graph=graph, vector_store=vector_store, lock=lock)
        dba = MemoryDBA(
            llm=llm, graph=graph, vector_store=vector_store, graph_builder=builder,
            # 与 mcp_server 一致；调用测试只走检索，不触发抽取，这几项仅保持装配同构
            enable_choice_extraction=True,
            enable_evidence=True,
            enable_source_store=False,   # 面板测试不落原文
            source_dir=None,
        )
        retriever = PurposeDrivenRetriever(
            llm=llm, embeddings=embeddings, graph=graph, vector_store=vector_store,
            inference=InferenceEngine(llm), path_tracker=PathTracker(),
        )
        return DBAServer(
            graph, yaml_path=self.yaml_path, dba=dba, retriever=retriever,
            vector_store=vector_store, lock=lock, scheduler=None,
        )

    def _get(self, on_stage: StageCb = None) -> Any:
        if self._server is not None:
            return self._server
        # 装配失败通常是确定性的（缺依赖 / 缺密钥 / 模型不可用），重跑只会再付一次
        # 加载 embedding 的代价，因此缓存错误直接复用。
        if self._init_error is not None:
            raise self._init_failure()
        with self._init_lock:
            if self._server is not None:
                return self._server
            if self._init_error is not None:
                raise self._init_failure()
            # 首次装配要加载 embedding 模型 + 建 LLM 客户端，先给前端一个提示
            if on_stage:
                on_stage("init_start", {"yaml": self.yaml_path})
            try:
                self._server = self._build()
            except Exception as e:
                self._init_error = str(e)
                logger.error("检索链路装配失败: %s", e, exc_info=True)
                raise
            if on_stage:
                on_stage("init_done", {})
            return self._server

    # ── 执行 ────────────────────────────────────────────────────────────────

    def run(self, query: str, on_stage: StageCb = None) -> Dict[str, Any]:
        """跑一次真实链路并返回完整轨迹。

        on_stage(stage, info) 会在每个阶段被调用（装配 / 向量对齐 / 意图识别 / 每跳 /
        寻峰 / storyrank），调用方负责把它转发给前端。
        """
        text = (query or "").strip()
        if not text:
            raise ValueError("query 不能为空")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("已有一次调用测试在跑，等它结束再试")

        def emit(stage: str, info: Dict[str, Any]) -> None:
            if on_stage is None:
                return
            try:
                on_stage(stage, info)
            except Exception:
                logger.debug("阶段事件转发失败（忽略）", exc_info=True)

        try:
            server = self._get(emit)
            # 读前会把向量库对齐到磁盘图谱：刚改过图谱时这一步包含 embedding，会明显偏慢
            emit("align_start", {"note": "对齐向量库（若图谱刚改过，这里会补 embedding）"})
            result = server.run_chain_trace(text, on_stage=on_stage)
            # done 事件带完整轨迹：SSE 通道先到，前端不必干等 POST 响应
            # （长连接/代理下 POST 响应可能迟迟不到，那时面板会一直卡在"运行中"）
            emit("done", {
                "visited": len(result.get("visited_ids", [])),
                "adopted": len(result.get("story_nodes", [])),
                "visited_ids": result.get("visited_ids", []),
                "story_nodes": result.get("story_nodes", []),
                "result": result,
            })
            return result
        finally:
            self._run_lock.release()
