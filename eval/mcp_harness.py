# -*- coding: utf-8 -*-
"""
MCP 全链路测试的公共装配层。

把「图谱 + 向量 + DBA + PAR 检索 + MCP Server」的装配与工具调用抽出来，
供 eval/test_*.py 的 MCP 全链路测试复用。装配方式与 `dba_pipeline/mcp_server.main`
一致，差异只在参数来源：这里读环境变量，且**不挂异步调度器**——调度器把
add 变成排队，测试内无法确定性断言。

本模块只做装配与协议层调用，不产生 LLM 调用（LLM 只在实际 add / query 时发生）。
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def new_workspace(prefix: str = "mcp_eval_") -> Tuple[Path, Path]:
    """新建隔离工作目录 + 空图谱 YAML，避免污染 data/"""
    import yaml

    workdir = Path(tempfile.mkdtemp(prefix=prefix))
    yaml_path = workdir / "memory_graph.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"nodes": [], "edges": []}, f, allow_unicode=True)
    return workdir, yaml_path


def build_pipeline(yaml_path: str, enable_choice: bool = True, enable_evidence: bool = True,
                   enable_source_store: bool = True):
    """构建 MCP 全链路：返回 (graph, vector_store, dba, retriever, lock)

    enable_choice: 是否启用「决策与取舍」抽取增强（默认开，与 mcp_server 主程序一致）
    enable_evidence: 是否让抽取附带 evidence 原文支撑片段（默认开，同上）
    enable_source_store: 是否把本批原文落盘到 <yaml 同目录>/sources/（默认开，便于测试
        C1/C3；生产由 ARIADNE_SOURCE_STORE 环境变量控制，默认关闭）
    """
    from langchain_openai import ChatOpenAI

    from dba_pipeline.core.path_tracker import PathTracker
    from dba_pipeline.source_store import default_source_dir
    from dba_pipeline.embedding.store import (
        LocalEmbeddings, OpenAIEmbeddings, VectorStore,
    )
    from dba_pipeline.extraction.dba import MemoryDBA
    from dba_pipeline.extraction.graph_builder import GraphBuilder
    from dba_pipeline.llm.inference import InferenceEngine
    from dba_pipeline.retrieval.retriever import PurposeDrivenRetriever

    embedding_local = os.environ.get("EMBEDDING_LOCAL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    emb_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    if embedding_local:
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

    graph = load_graph(yaml_path)
    lock = threading.RLock()
    vector_store = VectorStore(embeddings=embeddings, backend="faiss", persist_dir=None)
    builder = GraphBuilder(graph=graph, vector_store=vector_store, lock=lock)
    dba = MemoryDBA(
        llm=llm, graph=graph, vector_store=vector_store, graph_builder=builder,
        enable_choice_extraction=enable_choice,
        enable_evidence=enable_evidence,
        enable_source_store=enable_source_store,
        source_dir=default_source_dir(yaml_path) if enable_source_store else None,
    )
    retriever = PurposeDrivenRetriever(
        llm=llm, embeddings=embeddings, graph=graph, vector_store=vector_store,
        inference=InferenceEngine(llm), path_tracker=PathTracker(),
    )
    return graph, vector_store, dba, retriever, lock


def build_server(yaml_path: str, with_scheduler: bool = False, enable_choice: bool = True,
                 enable_evidence: bool = True, enable_source_store: bool = True):
    """构建 DBAServer + MCP Server：返回 (graph, vector_store, server, handlers)

    with_scheduler=True 时挂上异步批量调度器（生产 MCP 的真实路径）。
    调度器是异步的，调用方需自行 flush 并等待维护完成。
    enable_choice: 是否启用「决策与取舍」抽取增强。
    enable_evidence: 是否让抽取附带 evidence 原文支撑片段。
    enable_source_store: 是否把本批原文落盘（C1/C3 验证需要）。
    """
    from dba_pipeline.mcp_server import DBAServer, create_mcp_server

    graph, vector_store, dba, retriever, lock = build_pipeline(
        yaml_path, enable_choice=enable_choice, enable_evidence=enable_evidence,
        enable_source_store=enable_source_store,
    )
    scheduler = None
    if with_scheduler:
        from dba_pipeline.extraction.maintenance_scheduler import (
            MaintenanceScheduler, ScheduleConfig,
        )
        scheduler = MaintenanceScheduler(dba=dba, config=ScheduleConfig())
    server = DBAServer(
        graph, yaml_path=str(yaml_path), dba=dba, retriever=retriever,
        vector_store=vector_store, lock=lock, scheduler=scheduler,
    )
    handlers = getattr(create_mcp_server(server), "_request_handlers", {})
    return graph, vector_store, server, handlers


def _resolve(callable_obj):
    for attr in ("handler", "fn", "func", "_handler"):
        cand = getattr(callable_obj, attr, None)
        if callable(cand):
            return cand
    return callable_obj if callable(callable_obj) else None


def call_tool(handlers: Dict, name: str, arguments: Dict) -> Dict:
    """经 MCP 协议层（tools/call handler）调用工具，返回解析后的 dict"""
    from mcp.types import CallToolRequestParams

    h = _resolve(handlers.get("tools/call"))
    if h is None:
        raise RuntimeError("tools/call handler 无法解析")
    params = CallToolRequestParams(name=name, arguments=arguments)
    res = asyncio.run(h(None, params))
    return json.loads(res.content[0].text)


def dump_nodes(graph, only: List[str] = None) -> None:
    """打印节点表（id / 类型 / 时间戳 / 内容）"""
    for nid, nd in graph.graph.nodes(data=True):
        if only is not None and nid not in only:
            continue
        nt = nd.get("node_type")
        nt = nt.value if hasattr(nt, "value") else str(nt)
        print(f"  {nid}  [{nt}]  ts={nd.get('timestamp')}  {nd.get('content')!r}")


def dump_edges(graph, rel_filter: str = None) -> None:
    """打印边表；rel_filter 非空时只打印该关系类型（如 temporal）"""
    for u, v, ed in graph.graph.edges(data=True):
        rt = ed.get("rel_type")
        rt = rt.value if hasattr(rt, "value") else str(rt)
        if rel_filter and rt != rel_filter:
            continue
        print(f"  {u} --{rt}--> {v}")


def load_graph(yaml_path: str):
    from dba_pipeline.loader import load_graph as _load

    return _load(yaml_path)
