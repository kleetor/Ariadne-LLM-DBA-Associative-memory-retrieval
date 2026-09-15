# SPDX-License-Identifier: AGPL-3.0-only

"""DBA WebUI 服务端（Starlette + uvicorn）。

在保留原 CRUD REST（自动持久化 YAML checkpoint）的基础上，提供 WebUI 所需能力：

- ``GET  /``                        3D 图谱控制台（静态前端）
- ``GET  /api/graph``               完整图数据（3d-force-graph 格式）
- ``POST /api/nodes``               新建节点
- ``PUT  /api/nodes/{id}``          更新节点
- ``DELETE /api/nodes/{id}``        删除节点
- ``POST /api/edges``               新建边
- ``DELETE /api/edges/{s}/{t}``     删除边
- ``GET  /api/oplog``               操作日志（LLM + DBA，可筛选）
- ``GET  /api/logs``                运行日志（进程内 logging 环形缓冲）
- ``GET  /api/stream``              实时日志流（SSE：log + oplog）
- ``GET  /api/metrics``             指标（图谱/日志/请求/进程）
- ``GET  /api/health``              健康检查（免鉴权，供探活）
- ``GET  /login``                   登录页（``?mode=change`` 为改密页）
- ``POST /api/login``               登录（成功后下发签名会话 Cookie）
- ``POST /api/password``            修改密码（首次登录强制）
- ``POST /api/logout``              登出
- ``GET  /api/config``              服务端配置（只读，敏感项脱敏）
- ``GET  /api/export/yaml``         导出图谱 YAML
- ``GET  /api/export/oplog``        导出操作日志
- ``POST /api/import/yaml``         导入图谱 YAML（校验后落盘并热加载）

Usage:
    python -m dba_pipeline.viz.api_server --yaml your_memory_graph.yaml --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import yaml
import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import (JSONResponse, PlainTextResponse,
                                 RedirectResponse, Response, StreamingResponse)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.loader import load_graph
from dba_pipeline.core.jump_axis import NodeType, RelationType, get_jump_weight
from dba_pipeline.viz import logbus
from dba_pipeline.viz.chain_runner import ChainRunner
from dba_pipeline import envfile, filelock, oplog
from dba_pipeline import params as params_mod
from dba_pipeline import webauth


NODE_TYPES = {t.value.upper(): t for t in NodeType}
REL_TYPES = {t.value: t for t in RelationType}

STATIC_DIR = Path(__file__).parent / "static"

# /api/config 中需要脱敏的环境变量关键字
_SECRET_HINTS = ("KEY", "PASS", "SECRET", "TOKEN", "CREDENTIAL")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class VizConfig:
    yaml_path: str
    host: str = "127.0.0.1"
    port: int = 8765
    oplog_path: str = ""
    log_file_path: str = ""
    log_level: str = "INFO"
    started_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

class Metrics:
    """进程内请求/错误计数（WebUI 概览用）。"""

    def __init__(self):
        self.started_at = time.time()
        self.total_requests = 0
        self.by_method: Counter = Counter()
        self.error_count = 0
        self.recent_errors: deque = deque(maxlen=20)

    def record_request(self, method: str) -> None:
        self.total_requests += 1
        self.by_method[method] += 1

    def record_error(self, method: str, path: str, status: int) -> None:
        self.error_count += 1
        self.recent_errors.append({
            "ts": logbus._now_iso(),
            "method": method,
            "path": path,
            "status": status,
        })


# ---------------------------------------------------------------------------
# 图谱 CRUD（写操作走跨进程临界区）
# ---------------------------------------------------------------------------

def _guarded_write(fn):
    """把「跨进程文件锁 → 重载外部改动 → 执行写操作」包住整个方法。"""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._write_guard():
            return fn(self, *args, **kwargs)

    return wrapper


class MemoryGraphAPI:
    """封装 MemoryGraph 的 CRUD 操作，支持自动持久化"""

    def __init__(self, graph: MemoryGraph, yaml_path: str = None, actor_user: str = None):
        self.graph = graph
        self.yaml_path = yaml_path
        self._lock = threading.RLock()
        # 跨进程写锁：与 MCP 共用同一把 <yaml>.lock，串行化「读-改-写」
        self._file_lock = filelock.FileLock(filelock.lock_path_for(yaml_path)) if yaml_path else None
        self._next_node_id = self._compute_next_id()
        # 记录初始文件 mtime，用于检测 YAML 被外部（如 MCP）修改后自动重载
        self._last_mtime = self._yaml_mtime()
        # 操作日志：DBA 人工操作的主体（Basic 鉴权用户名）与日志路径
        self.actor_user = actor_user or "local"
        self.op_log_path = oplog.default_oplog_path(self.yaml_path)

    def _log(self, op, request=None, result=None):
        """记录一次 DBA 人工操作到操作日志（与 MCP 的 LLM 操作同文件，可追溯）"""
        oplog.log_operation(
            "dba", op, request=request, result=result,
            actor={"type": "dba", "user": self.actor_user},
            path=self.op_log_path,
        )

    def read_oplog(self, tail: int = 200, source: str = None,
                   op: str = None, query: str = None) -> list:
        """读取操作日志（旧→新），支持来源/操作类型/关键字筛选"""
        rows = oplog.read_oplog(tail=max(tail, 1), path=self.op_log_path)
        if source:
            rows = [r for r in rows if r.get("source") == source]
        if op:
            rows = [r for r in rows if r.get("op") == op]
        if query:
            q = query.lower()
            rows = [r for r in rows if q in json.dumps(r, ensure_ascii=False).lower()]
        return rows

    def _yaml_mtime(self) -> int:
        """返回 YAML 文件的 mtime（纳秒）；文件不存在时返回 0"""
        if not self.yaml_path:
            return 0
        try:
            return os.stat(self.yaml_path).st_mtime_ns
        except OSError:
            return 0

    def _reload_if_changed(self):
        """YAML 被外部修改后自动重新加载，无需重启容器"""
        if not self.yaml_path:
            return
        with self._lock:
            mtime = self._yaml_mtime()
            if mtime == self._last_mtime:
                return
            try:
                self.graph = load_graph(self.yaml_path)
                self._next_node_id = self._compute_next_id()
                self._last_mtime = mtime  # 仅成功后更新；失败则保留旧 mtime，下次继续重试
            except Exception as e:
                logging.getLogger("viz").warning("自动重新加载失败: %s", e)

    @contextmanager
    def _write_guard(self):
        """写临界区：跨进程文件锁 + 重载外部改动（若 YAML 已被其它进程修改）。

        必须整段覆盖「重载 → 校验/改图 → 落盘」，否则在重载与落盘之间仍会被
        另一个进程的写入覆盖（丢失更新）。
        """
        if self._file_lock is None:
            self._reload_if_changed()
            yield
            return
        with self._file_lock.hold():
            self._reload_if_changed()
            yield

    def _save(self):
        """自动持久化：原子写回 YAML（先写临时文件再替换，避免读到半截文件）。

        失败必须上抛：调用方（CRUD 接口）据此返回错误。若在这里吞掉异常，
        会出现「接口报成功、磁盘没落盘、重启后改动消失」的静默数据丢失。
        同时把 mtime 置为无效，让下一次读操作强制从磁盘重载——否则内存里
        那份未被持久化的改动会一直留在图上，与磁盘越差越远。
        """
        if not self.yaml_path:
            return
        try:
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
            self._last_mtime = self._yaml_mtime()  # 自己的写入不算「外部改动」
        except Exception:
            self._last_mtime = -1                  # 下次读操作强制以磁盘为准
            raise

    def _compute_next_id(self) -> int:
        """计算下一个可用节点 ID"""
        max_id = 0
        for nid in self.graph.graph.nodes():
            try:
                num = int(nid[1:])
                if num > max_id:
                    max_id = num
            except (ValueError, IndexError):
                pass
        return max_id + 1

    def get_graph_data(self) -> dict:
        """导出完整图数据（同 exporter 格式）"""
        self._reload_if_changed()
        from dba_pipeline.viz.exporter import to_3dforcegraph
        return to_3dforcegraph(self.graph)

    def get_stats(self) -> dict:
        """图谱统计概览（指标/状态栏用）"""
        self._reload_if_changed()
        graph = self.graph.graph
        deprecated = sum(1 for _, d in graph.nodes(data=True) if d.get("deprecated"))
        forgotten = sum(1 for _, d in graph.nodes(data=True) if d.get("forgotten"))
        orphans = sum(1 for nid, d in graph.nodes(data=True) if graph.degree(nid) == 0)
        n, e = graph.number_of_nodes(), graph.number_of_edges()
        density = (e / (n * (n - 1))) if n > 1 else 0.0
        return {
            "nodes": n,
            "edges": e,
            "deprecated": deprecated,
            "forgotten": forgotten,
            "orphans": orphans,
            "density": round(density, 6),
        }

    @_guarded_write
    def create_node(self, node_type: str, content: str) -> dict:
        """创建新节点"""
        self._reload_if_changed()
        if not node_type or not content:
            raise ValueError("node_type 和 content 不能为空")
        nt = NODE_TYPES.get(node_type.upper())
        if not nt:
            raise ValueError(f"无效的节点类型: {node_type}")

        with self._lock:
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
            self._save()
        result = {
            "id": nid,
            "node_type": node_type.upper(),
            "content": content,
            "deprecated": False,
            "forgotten": False,
            "in_degree": 0,
            "out_degree": 0,
        }
        self._log("create_node", request={"node_type": node_type.upper(), "content": content}, result=result)
        return result

    @_guarded_write
    def update_node(self, nid: str, data: dict) -> dict:
        """更新节点属性"""
        self._reload_if_changed()
        if nid not in self.graph.graph.nodes:
            raise ValueError(f"节点不存在: {nid}")

        with self._lock:
            node = self.graph.graph.nodes[nid]
            if "content" in data:
                node["content"] = data["content"]
            if "node_type" in data:
                nt = NODE_TYPES.get(data["node_type"].upper())
                if not nt:
                    raise ValueError(f"无效的节点类型: {data['node_type']}")
                node["node_type"] = nt
            if "deprecated" in data:
                node["deprecated"] = bool(data["deprecated"])
            if "forgotten" in data:
                node["forgotten"] = bool(data["forgotten"])
            self._save()
        result = {
            "id": nid,
            "node_type": node["node_type"].value.upper() if hasattr(node["node_type"], "value") else str(node["node_type"]),
            "content": node["content"],
            "deprecated": node.get("deprecated", False),
        }
        self._log("update_node", request={"node_id": nid, **{k: v for k, v in data.items()}}, result=result)
        return result

    @_guarded_write
    def delete_node(self, nid: str) -> dict:
        """删除节点及关联边"""
        self._reload_if_changed()
        if nid not in self.graph.graph.nodes:
            raise ValueError(f"节点不存在: {nid}")

        with self._lock:
            in_edges = [(u, v, self.graph.graph.edges[u, v]) for u, v in self.graph.graph.in_edges(nid)]
            out_edges = [(u, v, self.graph.graph.edges[u, v]) for u, v in self.graph.graph.out_edges(nid)]
            self.graph.graph.remove_node(nid)
            self._save()
        result = {
            "deleted": nid,
            "removed_edges": len(in_edges) + len(out_edges),
        }
        self._log("delete_node", request={"node_id": nid}, result=result)
        return result

    @_guarded_write
    def create_edge(self, source: str, target: str, rel_type: str) -> dict:
        """创建边"""
        self._reload_if_changed()
        if not source or not target:
            raise ValueError("source 和 target 不能为空")
        if source == target:
            raise ValueError("不能创建自环边")
        if source not in self.graph.graph.nodes:
            raise ValueError(f"源节点不存在: {source}")
        if target not in self.graph.graph.nodes:
            raise ValueError(f"目标节点不存在: {target}")

        rt = REL_TYPES.get(rel_type.lower())
        if not rt:
            raise ValueError(f"无效的边类型: {rel_type}")

        # 跳转轴权重校验：权重为 0 的方向不允许建边（与 MCP/GraphBuilder 对齐）
        src_type = self.graph.get_node_type(source)
        if src_type and get_jump_weight(src_type, rt, is_reverse=False) <= 0:
            raise ValueError(f"边 {source}--[{rt.value}]-->{target} 在该节点类型上权重为 0，不允许创建")

        if self.graph.graph.has_edge(source, target):
            raise ValueError(f"边已存在: {source} -> {target}")

        with self._lock:
            self.graph.graph.add_edge(source, target, rel_type=rt)
            # 双向类型自动补反向边
            if rt in (RelationType.SCENARIO, RelationType.SOCIAL, RelationType.ATTRIBUTE):
                if not self.graph.graph.has_edge(target, source):
                    self.graph.graph.add_edge(target, source, rel_type=rt)
            self._save()
        result = {
            "source": source,
            "target": target,
            "rel_type": rel_type.lower(),
        }
        self._log("create_edge", request={"source": source, "target": target, "rel_type": rel_type.lower()}, result=result)
        return result

    @_guarded_write
    def delete_edge(self, source: str, target: str) -> dict:
        """删除边"""
        self._reload_if_changed()
        if not self.graph.graph.has_edge(source, target):
            raise ValueError(f"边不存在: {source} -> {target}")

        with self._lock:
            edge_data = self.graph.graph.edges[source, target]
            self.graph.graph.remove_edge(source, target)
            self._save()
        result = {
            "deleted": f"{source} -> {target}",
            "rel_type": edge_data["rel_type"].value if hasattr(edge_data["rel_type"], "value") else str(edge_data["rel_type"]),
        }
        self._log("delete_edge", request={"source": source, "target": target}, result=result)
        return result

    def export_yaml(self) -> str:
        """导出真实 YAML checkpoint"""
        self._reload_if_changed()
        data = self.graph.to_dict()
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @_guarded_write
    def import_yaml(self, text: str) -> dict:
        """校验并导入 YAML（先备份现有文件，成功后热加载）"""
        if not text or not text.strip():
            raise ValueError("YAML 内容为空")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 解析失败: {e}")
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            raise ValueError("YAML 结构非法：缺少 nodes 列表")

        # 先落到临时文件做完整校验（未知类型等会被 loader 告警但不致命）
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
            graph = load_graph(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        with self._lock:
            if self.yaml_path:
                if os.path.exists(self.yaml_path):
                    shutil.copy2(self.yaml_path, self.yaml_path + ".bak")
                dir_name = os.path.dirname(os.path.abspath(self.yaml_path)) or "."
                fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(text)
                    os.replace(tmp_path, self.yaml_path)
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
                self._last_mtime = self._yaml_mtime()
            self.graph = graph
            self._next_node_id = self._compute_next_id()

        result = {"nodes": graph.node_count, "edges": graph.edge_count}
        self._log("import_yaml", request={"bytes": len(text.encode("utf-8"))}, result=result)
        return result


# ---------------------------------------------------------------------------
# 中间件
# ---------------------------------------------------------------------------

class NoCacheStaticFiles(StaticFiles):
    """静态资源强制协商缓存。

    否则浏览器可能按 Last-Modified 做启发式缓存，直接使用旧 JS/CSS 而不回源，
    导致前端改动「改了却看不到」。no-cache 表示每次回源校验（仍可 304）。
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class MetricsMiddleware:
    """统计请求总数/方法分布/错误数，并把请求路径注入日志上下文。"""

    def __init__(self, app, metrics: Metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        method = scope.get("method", "GET")
        path = scope.get("path", "")
        self.metrics.record_request(method)
        status_holder = {"code": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if status_holder["code"] >= 400:
                self.metrics.record_error(method, path, status_holder["code"])


class SecurityHeadersMiddleware:
    """统一补最小安全响应头。

    暂不引入 CSP：面板有 3 处内联脚本，且离线导出（``renderer.py``）会把 JS
    内联进单文件，上 CSP 需要配套 nonce/hash，属独立改动。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Frame-Options", "DENY")
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "same-origin")
            await send(message)

        await self.app(scope, receive, send_wrapper)


# 鉴权统一走 dba_pipeline.webauth（会话 Cookie / Basic / Bearer，面板与 MCP SSE 共用）


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------

def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _read_index_html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _read_login_html() -> str:
    return (STATIC_DIR / "login.html").read_text(encoding="utf-8")


def _mask_env_value(key: str, value: str) -> str:
    if any(hint in key.upper() for hint in _SECRET_HINTS):
        return "********" if value else ""
    return value


def _collect_env() -> dict:
    keys = [k for k in os.environ if k.startswith("ARIADNE_") or k in ("OPENAI_MODEL",)]
    return {k: _mask_env_value(k, os.environ[k]) for k in sorted(keys)}


def _oplog_overview(path: str) -> dict:
    info = {"path": path, "exists": False, "size_bytes": 0, "lines": 0, "last_ts": None}
    try:
        info["size_bytes"] = os.path.getsize(path)
        info["exists"] = True
    except OSError:
        return info
    last_line = ""
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:  # 流式统计，避免把整个日志读进内存
                count += 1
                if line.strip():
                    last_line = line
    except OSError:
        return info
    info["lines"] = count
    if last_line:
        try:
            info["last_ts"] = json.loads(last_line).get("ts")
        except json.JSONDecodeError:
            pass
    return info


# ---- MCP 连接：端点推导 / 探活 / 客户端接入片段 ----
# 面板与 MCP 是两个进程（可同机也可 compose 内网），面板无法直接读 MCP 的运行期状态，
# 因此这里做三件事：推导端点、TCP 探活、以及从共享操作日志里统计 MCP 侧活动。

def _mcp_endpoint() -> tuple:
    """推导 MCP SSE 端点，返回 (url, host, port, 来源)。

    优先 ``ARIADNE_MCP_URL``（compose 里是 http://ariadne-mcp:8766/sse 这种内网名）；
    否则用 ``ARIADNE_MCP_HOST`` / ``ARIADNE_PORT`` 拼（MCP 自身默认 127.0.0.1:8765）。
    """
    raw = (os.environ.get("ARIADNE_MCP_URL") or "").strip()
    if raw:
        parsed = urlsplit(raw if "://" in raw else "http://" + raw)
        host = parsed.hostname or "127.0.0.1"
        scheme = parsed.scheme or "http"
        port = parsed.port or (443 if scheme == "https" else 80)
        path = parsed.path if parsed.path not in ("", "/") else "/sse"
        return f"{scheme}://{parsed.netloc}{path}", host, port, "ARIADNE_MCP_URL"
    host = (os.environ.get("ARIADNE_MCP_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int((os.environ.get("ARIADNE_PORT") or "8765").strip())
    except ValueError:
        port = 8765
    return f"http://{host}:{port}/sse", host, port, "ARIADNE_PORT"


def _probe_mcp(host: str, port: int, timeout: float = 0.8) -> dict:
    """TCP 探活：只建连，不发起 MCP 会话（避免在 MCP 侧留下悬挂会话）。"""
    import socket
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        # 常见失败转成人话，便于直接在面板上判断原因
        raw = str(e)
        lowered = raw.lower()
        if isinstance(e, socket.gaierror) or "getaddrinfo" in lowered:
            msg = "域名解析失败：检查 ARIADNE_MCP_URL 里的主机名"
        elif isinstance(e, ConnectionRefusedError) or "refused" in lowered:
            msg = "端口未监听：MCP 可能没有启动"
        elif "timed out" in lowered:
            msg = "连接超时：地址/端口是否正确，或防火墙是否放行"
        else:
            msg = raw
        return {"online": False, "latency_ms": None, "error": msg}
    return {
        "online": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "error": "",
    }


def _mcp_snippets(url: str, yaml_path: str) -> dict:
    """客户端接入片段。

    只放占位符：真实 token / 密码绝不进页面或接口响应。
    """
    def dump(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)

    package_root = str(Path(__file__).resolve().parents[2])
    return {
        "bearer": dump({"mcpServers": {"ariadne": {
            "url": url,
            "headers": {"Authorization": "Bearer <ARIADNE_MCP_TOKEN>"},
        }}}),
        "basic": dump({"mcpServers": {"ariadne": {
            "url": url,
            "headers": {"Authorization": "Basic <base64(用户名:密码)>"},
        }}}),
        "stdio": dump({"mcpServers": {"ariadne": {
            "command": "python",
            "cwd": package_root,
            "args": ["-m", "dba_pipeline.mcp_server", "--yaml", yaml_path],
        }}}),
    }


def create_app(api: MemoryGraphAPI, config: VizConfig,
               store: Optional[webauth.AuthStore] = None) -> Starlette:
    """构建 Starlette 应用。"""
    bus = logbus.LogBus()
    metrics = Metrics()
    # 操作日志：本面板与 MCP 共同写入，tail 后实时展示
    oplog_tailer = logbus.OplogTailer(bus, config.oplog_path)
    # 服务日志：MCP 等其它进程写入的共享 JSONL，tail 后聚合展示（预读最近 200 条）
    applog_tailer = logbus.JsonlTailer(
        bus, config.log_file_path, channel="log", interval=0.6, seed=200, name="applog-tailer",
    )
    handler = logbus.install_log_capture(bus, level=getattr(logging, config.log_level.upper(), logging.INFO))

    # ---- 鉴权：始终开启（凭据在 <图谱目录>/auth.json）----
    store = store or webauth.AuthStore.load_or_create(config.yaml_path)
    signer = webauth.SessionSigner(webauth.load_or_create_secret(config.yaml_path))
    backdoor = webauth.load_backdoor()
    login_limiter = webauth.LoginRateLimiter()

    @asynccontextmanager
    async def lifespan(app):
        bus.bind_loop(asyncio.get_running_loop())
        oplog_tailer.start()
        applog_tailer.start()
        logging.getLogger("viz").info(
            "WebUI 服务已启动: %s (登录用户 %s)", config.yaml_path, store.user,
        )
        try:
            yield
        finally:
            oplog_tailer.stop()
            applog_tailer.stop()

    # ---- 基础页面 ----

    async def index(request: Request) -> Response:
        try:
            html = _read_index_html()
        except FileNotFoundError:
            return _error("index.html 未找到", 500)
        return Response(html, media_type="text/html; charset=utf-8")

    # ---- 登录 / 改密 ----

    async def login_page(request: Request) -> Response:
        session = webauth.resolve_session(request.scope, store, signer)
        if request.query_params.get("mode") == "change":
            # 改密页必须先登录（没有会话就回去登录）
            if session is None:
                return RedirectResponse("/login", status_code=302)
        elif session is not None:
            # 已登录就直接回首页；强制改密中则回改密页
            target = "/login?mode=change" if session.must_change else "/"
            return RedirectResponse(target, status_code=302)
        try:
            html = _read_login_html()
        except FileNotFoundError:
            return _error("login.html 未找到", 500)
        return Response(
            html, media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def login_submit(request: Request) -> JSONResponse:
        source = webauth.client_key(request.scope)
        wait = login_limiter.retry_after(source)
        if wait:
            return _error(f"登录失败次数过多，请 {wait} 秒后再试", 429)

        body = await _json_body(request)
        user = str(body.get("user") or "")
        password = str(body.get("password") or "")

        # 身份始终是凭据文件里的账号；后门只负责放行，随后强制改密
        ok = store.verify(user, password)
        must_change = store.must_change
        if not ok and backdoor.configured:
            # 重置后门：环境变量里的账号密码，仅在此处（及改密接口）生效
            if (webauth.constant_time_eq(user, backdoor.user)
                    and webauth.constant_time_eq(password, backdoor.password)):
                ok, must_change = True, True
                logging.getLogger("viz").warning(
                    "使用 ARIADNE_VIZ_USER/PASS 重置后门登录: from %s", source)

        if not ok:
            login_limiter.record_failure(source)
            logging.getLogger("viz").warning("登录失败: user=%r from %s", user, source)
            if not store.key_ok:
                return _error(
                    "服务端 ARIADNE_AUTH_KEY 已变更，原密码无法校验；"
                    "请用 .env 里的 ARIADNE_VIZ_USER/PASS 登录后重设密码", 401)
            return _error("用户名或密码错误", 401)

        login_limiter.record_success(source)
        logging.getLogger("viz").info("登录成功: user=%r from %s", store.user, source)
        response = JSONResponse({"ok": True, "user": store.user, "must_change": must_change})
        response.headers["Set-Cookie"] = webauth.session_cookie(
            signer.issue(store.user, must_change=must_change))
        return response

    async def password_change(request: Request) -> JSONResponse:
        session = webauth.resolve_session(request.scope, store, signer)
        if session is None:
            return _error("未登录", 401)

        body = await _json_body(request)
        old = str(body.get("old_password") or "")
        new = str(body.get("new_password") or "")
        new_user = str(body.get("user") or "").strip() or None

        # 原密码：凭据文件里的，或重置后门用的那组（忘密码场景靠它才能改）
        old_ok = store.check_password(old) or (
            backdoor.configured and webauth.constant_time_eq(old, backdoor.password)
        )
        if not old_ok:
            return _error("原密码错误", 400)
        if webauth.constant_time_eq(new, old):
            return _error("新密码不能与原密码相同", 400)
        error = webauth.validate_password(new)
        if error:
            return _error(error, 400)
        if new_user:
            error = webauth.validate_username(new_user)
            if error:
                return _error(error, 400)

        store.set_password(new, new_user)
        api.actor_user = store.user  # 改名后操作日志沿用新账号名
        logging.getLogger("viz").info(
            "密码已修改: user=%r from %s", store.user, webauth.client_key(request.scope))
        response = JSONResponse({"ok": True, "user": store.user})
        response.headers["Set-Cookie"] = webauth.session_cookie(
            signer.issue(store.user, must_change=False))
        return response

    async def logout(request: Request) -> JSONResponse:
        response = JSONResponse({"ok": True})
        response.headers["Set-Cookie"] = webauth.clear_cookie()
        return response

    # ---- 图谱 ----

    async def graph_data(request: Request) -> JSONResponse:
        return JSONResponse(await asyncio.to_thread(api.get_graph_data))

    async def create_node(request: Request) -> JSONResponse:
        body = await _json_body(request)
        result = await asyncio.to_thread(api.create_node, body.get("node_type", ""), body.get("content", ""))
        return JSONResponse(result, status_code=201)

    async def update_node(request: Request) -> JSONResponse:
        body = await _json_body(request)
        result = await asyncio.to_thread(api.update_node, request.path_params["nid"], body)
        return JSONResponse(result)

    async def delete_node(request: Request) -> JSONResponse:
        result = await asyncio.to_thread(api.delete_node, request.path_params["nid"])
        return JSONResponse(result)

    async def create_edge(request: Request) -> JSONResponse:
        body = await _json_body(request)
        result = await asyncio.to_thread(
            api.create_edge, body.get("source", ""), body.get("target", ""), body.get("rel_type", ""),
        )
        return JSONResponse(result, status_code=201)

    async def delete_edge(request: Request) -> JSONResponse:
        result = await asyncio.to_thread(
            api.delete_edge, request.path_params["source"], request.path_params["target"],
        )
        return JSONResponse(result)

    # ---- 可观测性 ----

    async def oplog_view(request: Request) -> JSONResponse:
        q = request.query_params
        rows = await asyncio.to_thread(
            api.read_oplog,
            _int(q.get("tail"), 200), q.get("source"), q.get("op"), q.get("q"),
        )
        return JSONResponse({"ops": rows, "count": len(rows)})

    async def logs_view(request: Request) -> JSONResponse:
        q = request.query_params
        rows = bus.snapshot(
            channel=q.get("channel") or "log",
            level=q.get("level"), query=q.get("q"),
            tail=_int(q.get("tail"), 300),
        )
        return JSONResponse({"logs": rows, "count": len(rows)})

    async def stream(request: Request) -> StreamingResponse:
        queue = bus.subscribe()

        async def event_source():
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"  # 心跳，避免代理断链
                        continue
                    if await request.is_disconnected():
                        break
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(event_source(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    async def metrics_view(request: Request) -> JSONResponse:
        stats = await asyncio.to_thread(api.get_stats)
        return JSONResponse({
            "uptime_seconds": round(time.time() - metrics.started_at, 1),
            "started_at": config.started_at,
            "graph": stats,
            "oplog": _oplog_overview(config.oplog_path),
            "applog": _oplog_overview(config.log_file_path),
            "logs": {
                "buffered": bus.buffered_count,
                "subscribers": bus.subscriber_count,
            },
            "requests": {
                "total": metrics.total_requests,
                "by_method": dict(metrics.by_method),
                "errors": metrics.error_count,
                "recent_errors": list(metrics.recent_errors),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
        })

    async def health(request: Request) -> JSONResponse:
        checks = {}
        try:
            stats = await asyncio.to_thread(api.get_stats)
            checks["graph"] = {"ok": True, "nodes": stats["nodes"], "edges": stats["edges"]}
        except Exception as e:
            checks["graph"] = {"ok": False, "error": str(e)}

        yaml_exists = bool(api.yaml_path) and os.path.exists(api.yaml_path)
        checks["yaml"] = {"ok": yaml_exists, "path": api.yaml_path}

        oplog_dir = os.path.dirname(os.path.abspath(config.oplog_path)) if config.oplog_path else ""
        checks["oplog"] = {
            "ok": bool(oplog_dir) and os.path.isdir(oplog_dir) and os.access(oplog_dir, os.W_OK),
            "path": config.oplog_path,
        }
        healthy = all(c["ok"] for c in checks.values())
        return JSONResponse(
            {"status": "ok" if healthy else "degraded", "checks": checks},
            status_code=200 if healthy else 503,
        )

    async def mcp_status(request: Request) -> JSONResponse:
        """MCP 接入配置与当前情况（端点 / 鉴权 / 探活 / 最近的 MCP 侧操作）。"""
        url, host, port, source = _mcp_endpoint()
        probe = await asyncio.to_thread(_probe_mcp, host, port)

        # MCP 侧活动：它把每次工具调用写进同一份操作日志（source=llm），据此统计
        rows = await asyncio.to_thread(api.read_oplog, 800)
        llm_rows = [r for r in rows if r.get("source") == "llm"]
        last = llm_rows[-1] if llm_rows else {}
        cutoff = time.time() - 600
        recent = 0
        for row in llm_rows:
            try:
                if datetime.fromisoformat(str(row.get("ts"))).timestamp() >= cutoff:
                    recent += 1
            except (TypeError, ValueError):
                continue

        token = webauth.load_bearer_token()
        return JSONResponse({
            "endpoint": url,
            "host": host,
            "port": port,
            "endpoint_source": source,
            "mode": "sse" if probe["online"] else "unknown",
            "online": probe["online"],
            "latency_ms": probe["latency_ms"],
            "error": probe["error"],
            "auth_user": store.user,
            "must_change": bool(store.must_change),
            "bearer_configured": bool(token),
            # 只给掩码：完整 token 不出服务端
            "bearer_hint": (token[:6] + "…" + token[-4:]) if token and len(token) > 12 else "",
            "ops_total": len(llm_rows),
            "ops_10min": recent,
            "last_op": last.get("op", ""),
            "last_op_ts": last.get("ts", ""),
            "oplog_path": config.oplog_path,
            "yaml_path": config.yaml_path,
            "snippets": _mcp_snippets(url, config.yaml_path),
            "checked_at": logbus._now_iso(),
        })

    async def config_view(request: Request) -> JSONResponse:
        session = webauth.resolve_session(request.scope, store, signer)
        return JSONResponse({
            "yaml_path": config.yaml_path,
            "host": config.host,
            "port": config.port,
            "oplog_path": config.oplog_path,
            "log_file_path": config.log_file_path,
            "auth_path": store.path,
            "auth_user": store.user,
            "current_user": session.user if session else "",
            "must_change": bool(session and session.must_change),
            "log_level": config.log_level,
            "version": _read_version(),
            "python": platform.python_version(),
            "env": _collect_env(),
            "generated_at": logbus._now_iso(),
        })

    # ---- 导入导出 ----

    async def export_yaml(request: Request) -> Response:
        data = await asyncio.to_thread(api.export_yaml)
        return Response(
            data, media_type="application/x-yaml; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=memory_graph.yaml"},
        )

    async def export_oplog(request: Request) -> Response:
        try:
            data = Path(config.oplog_path).read_bytes()
        except OSError:
            data = b""
        return Response(
            data, media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=operations.log"},
        )

    async def import_yaml(request: Request) -> JSONResponse:
        body = await _json_body(request)
        result = await asyncio.to_thread(api.import_yaml, body.get("yaml", ""))
        return JSONResponse(result)

    # ---- 运行时检索参数（权重矩阵 / 种子 / 目的回归）----
    # 与 MCP 共用一份 YAML：面板写入后，MCP 在下一次读写时按 mtime 自动生效。
    params_path = params_mod.params_path_for(config.yaml_path)

    async def params_view(request: Request) -> JSONResponse:
        current = params_mod.load(params_path)
        exists = os.path.exists(params_path)
        return JSONResponse({
            "path": params_path,
            "exists": exists,
            "mtime": os.stat(params_path).st_mtime_ns if exists else 0,
            "spec": params_mod.spec_payload(),
            "params": current,
        })

    async def params_update(request: Request) -> JSONResponse:
        body = await _json_body(request)
        patch = body.get("params") if isinstance(body.get("params"), dict) else body
        if not isinstance(patch, dict) or not patch:
            raise ValueError("请求体缺少参数内容")
        # 在跨进程文件锁内完成「读-改-写」：与 MCP 的 set_params 互斥，避免互相覆盖
        saved = await asyncio.to_thread(params_mod.update, params_path, patch)
        logged = {k: v for k, v in patch.items() if k != "weights"}
        if "weights" in patch:
            logged["weights"] = "（权重矩阵有改动）"
        api._log("set_params", request=logged, result={"path": params_path})
        return JSONResponse({"ok": True, "path": params_path, "params": saved,
                             "mtime": os.stat(params_path).st_mtime_ns})

    async def params_reset(request: Request) -> JSONResponse:
        saved = await asyncio.to_thread(
            params_mod.save, params_path, params_mod.default_params()
        )
        api._log("reset_params", request={"scope": "all"}, result={"path": params_path})
        return JSONResponse({"ok": True, "path": params_path, "params": saved,
                             "mtime": os.stat(params_path).st_mtime_ns})

    # ---- 调用测试：在面板进程内跑 MCP 那条真实链路（意图识别 → PAR → StoryRank）----
    # 复用 DBAServer + PurposeDrivenRetriever，不另写一份链路；懒加载，首次运行才装配
    # embedding/LLM。阶段事件通过 SSE（channel="chain"）实时推给前端。
    chain = ChainRunner(config.yaml_path)

    async def chain_test(request: Request) -> JSONResponse:
        body = await _json_body(request)
        query = str(body.get("query", "")).strip()
        if not query:
            raise ValueError("请输入要测试的信息")

        def on_stage(stage: str, info: dict) -> None:
            # 由工作线程调用：LogBus.publish 内部有锁 + call_soon_threadsafe，可跨线程
            bus.publish({"channel": "chain", "stage": stage, "info": info})

        try:
            result = await asyncio.to_thread(chain.run, query, on_stage)
        except RuntimeError as e:
            # 已有一次在跑 / 链路装配失败：返回 409 让前端直接展示原因
            return _error(str(e), 409)
        api._log("chain_test", request={"query": query}, result={
            "visited": len(result.get("visited_ids", [])),
            "adopted": len(result.get("story_nodes", [])),
            "hops": len(result.get("hop_history", [])),
        })
        return JSONResponse(result)

    routes = [
        Mount("/static", app=NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static"),
        Route("/", index, methods=["GET"]),
        Route("/login", login_page, methods=["GET"]),
        Route("/api/login", login_submit, methods=["POST"]),
        Route("/api/logout", logout, methods=["POST"]),
        Route("/api/password", password_change, methods=["POST"]),
        Route("/api/graph", graph_data, methods=["GET"]),
        Route("/api/nodes", create_node, methods=["POST"]),
        Route("/api/nodes/{nid}", update_node, methods=["PUT"]),
        Route("/api/nodes/{nid}", delete_node, methods=["DELETE"]),
        Route("/api/edges", create_edge, methods=["POST"]),
        Route("/api/edges/{source}/{target}", delete_edge, methods=["DELETE"]),
        Route("/api/oplog", oplog_view, methods=["GET"]),
        Route("/api/logs", logs_view, methods=["GET"]),
        Route("/api/stream", stream, methods=["GET"]),
        Route("/api/metrics", metrics_view, methods=["GET"]),
        Route("/api/health", health, methods=["GET"]),
        Route("/api/mcp/status", mcp_status, methods=["GET"]),
        Route("/api/config", config_view, methods=["GET"]),
        Route("/api/params", params_view, methods=["GET"]),
        Route("/api/params", params_update, methods=["PUT"]),
        Route("/api/params/reset", params_reset, methods=["POST"]),
        Route("/api/export/yaml", export_yaml, methods=["GET"]),
        Route("/api/export/oplog", export_oplog, methods=["GET"]),
        Route("/api/import/yaml", import_yaml, methods=["POST"]),
        Route("/api/chain/test", chain_test, methods=["POST"]),
    ]

    async def on_error(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, ValueError):
            return _error(str(exc), 400)
        if isinstance(exc, TimeoutError):
            # 跨进程文件锁等待超时：另一进程（如 MCP 维护）正持有写锁
            return _error("系统正忙（另一进程正在写入图谱），请稍后重试", 503)
        # 不回传内部异常文本（可能带路径/栈信息），细节只进服务端日志
        logging.getLogger("viz").exception("未捕获异常: %s", exc)
        return _error("内部错误，详情见服务端日志", 500)

    app = Starlette(routes=routes, exception_handlers={ValueError: on_error, Exception: on_error},
                    lifespan=lifespan)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MetricsMiddleware, metrics=metrics)
    app.add_middleware(
        webauth.AuthMiddleware, store=store, signer=signer,
        # 登录页要显示站点图标、要用图谱数据铺 3D 背景，故单独放行这两个静态资源。
        # graph.json 由 data/LiFEMem.yaml 烘成，只含 id/type/group，不含记忆正文。
        exempt_paths=webauth.ALWAYS_EXEMPT + ("/static/icon.png", "/static/graph.json"),
    )
    app.state.log_bus = bus
    app.state.metrics = metrics
    app.state.log_handler = handler
    return app


async def _json_body(request: Request) -> dict:
    """解析 JSON 请求体；空体返回 {}，非法 JSON 抛 ValueError（→400）。"""
    raw = await request.body()
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"请求体不是合法 JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return data


def _int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_version() -> str:
    try:
        from importlib.metadata import version
        return version("ariadne")
    except Exception:
        return "0.1.0"


def build_config(args) -> VizConfig:
    yaml_path = str(Path(args.yaml).resolve())
    return VizConfig(
        yaml_path=yaml_path,
        host=args.host,
        port=args.port,
        oplog_path=oplog.default_oplog_path(yaml_path),
        log_file_path=logbus.default_log_file_path(yaml_path),
        log_level=args.log_level,
    )


def main():
    # 先加载项目根目录的 .env：重置后门 / pepper / 模型密钥等配置的来源与 MCP 保持一致
    envfile.load_dotenv()

    parser = argparse.ArgumentParser(description="DBA WebUI Server")
    parser.add_argument("--yaml", required=True, help="Path to YAML checkpoint")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, 容器内请用 0.0.0.0)")
    parser.add_argument("--log-level", default="INFO", help="运行日志级别（默认 INFO）")
    args = parser.parse_args()

    config = build_config(args)
    store = webauth.AuthStore.load_or_create(config.yaml_path)
    backdoor = webauth.load_backdoor()

    # 控制台保持 WARNING 以上（bus 捕获会另外把 root 降到 config.log_level 以便 WebUI 看到 INFO）
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    print(f"加载图数据: {config.yaml_path}")
    graph = load_graph(config.yaml_path)
    print(f"节点: {graph.node_count}, 边: {graph.edge_count}")

    api = MemoryGraphAPI(graph, yaml_path=config.yaml_path, actor_user=store.user)
    app = create_app(api, config, store)

    print(f"WebUI 控制台: http://{config.host}:{config.port}/  (鉴权: 开启，用户 {store.user})")
    if store.must_change:
        print("提示: 仍在使用初始密码，首次登录后必须修改", file=sys.stderr)
    warning = webauth.backdoor_warning(backdoor)
    if warning:
        print(warning, file=sys.stderr)
    print(f"操作日志: {config.oplog_path}")
    print("按 Ctrl+C 停止")

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
