"""AstrBot 插件：把 Ariadne 长期记忆接入 AstrBot。

Ariadne 把对话沉淀为一张可联想的记忆图谱；本插件负责在 AstrBot 内安装、配置与托管
它的 MCP / 面板服务，不重写任何记忆逻辑。

设计边界（与项目 README 一致）：
- 记忆能力仍然由独立的 `ariadne-mcp-server` 提供，本插件不重写任何 MCP 工具；
- 插件只做三件事：安装依赖（requirements.txt）、把配置映射成环境变量、托管进程并提供状态视图。

进程：
- mcp   → `python -m dba_pipeline.mcp_server --sse ...`（默认随插件启动）
- panel → `python -m dba_pipeline.viz.api_server ...`（可选，默认关闭）

WebUI：
- `_conf_schema.json` 自动生成配置表单；
- `views/status/` 提供状态卡、启停/重启、日志与 MCP 片段复制。
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import socket
import sys
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request, stream_response
from starlette.responses import Response

try:  # AstrBot >= 4.x
    from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
except Exception:  # pragma: no cover - 兼容旧版本
    get_astrbot_plugin_data_path = None

PLUGIN_NAME = "astrbot_plugin_ariadne"

# AstrBot 自带 mcp<2，而 ariadne-mcp-server 需要 mcp>=2.2；同环境安装会把 AstrBot 的
# MCP SDK 顶掉。因此服务一律通过 uvx 在独立虚拟环境里运行，不往 AstrBot 的 Python
# 环境装任何东西。可用 ARIADNE_UVX_FROM 固定版本（例如 ariadne-mcp-server==0.1.6）。
UVX_FROM = os.environ.get("ARIADNE_UVX_FROM", "ariadne-mcp-server")
BINARIES = {"mcp": "ariadne-mcp", "panel": "ariadne-api"}

# 服务名 → 配置分组
SERVICE_SPECS: dict[str, dict[str, str]] = {
    "mcp": {"cfg": "mcp"},
    "panel": {"cfg": "panel"},
}

# 允许通过视图 / API 修改的字段（白名单，避免任意写配置）
ALLOWED_PATHS: set[tuple[str, str]] = {
    ("llm", "api_key"),
    ("llm", "api_base"),
    ("llm", "model"),
    ("llm", "thinking"),
    ("embedding", "mode"),
    ("embedding", "api_key"),
    ("embedding", "api_base"),
    ("embedding", "model"),
    ("storage", "graphs_dir"),
    ("storage", "yaml"),
    ("mcp", "auto_start"),
    ("mcp", "host"),
    ("mcp", "port"),
    ("mcp", "token"),
    ("panel", "enabled"),
    ("panel", "host"),
    ("panel", "port"),
    ("panel", "user"),
    ("panel", "password"),
    ("advanced", "log_level"),
    ("advanced", "source_store"),
    ("advanced", "review_idle"),
}

SECRET_PATHS: set[tuple[str, str]] = {
    ("llm", "api_key"),
    ("embedding", "api_key"),
    ("mcp", "token"),
    ("panel", "password"),
}

INT_PATHS = {
    ("mcp", "port"),
    ("panel", "port"),
    ("advanced", "review_idle"),
}

BOOL_PATHS = {
    ("llm", "thinking"),
    ("mcp", "auto_start"),
    ("panel", "enabled"),
    ("advanced", "source_store"),
}


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    probe = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.create_connection((probe, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class AriadnePlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.logs: deque[str] = deque(maxlen=800)
        self._pumps: set[asyncio.Task[None]] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._data_dir().mkdir(parents=True, exist_ok=True)

        context.register_web_api(
            f"/{PLUGIN_NAME}/status", self.api_status, ["GET"], "Ariadne 运行状态"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/action", self.api_action, ["POST"], "启动 / 停止 / 重启"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/logs", self.api_logs, ["GET"], "最近日志"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/config", self.api_config_get, ["GET"], "读取当前配置"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/config/save", self.api_config_save, ["POST"], "保存配置"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/snippet", self.api_snippet, ["GET"], "MCP 接入片段"
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/proxy",
            self.api_viz_proxy,
            ["POST"],
            "Ariadne 面板 API 代理",
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/stream",
            self.api_viz_stream,
            ["GET"],
            "Ariadne 面板事件流代理",
        )
        context.register_web_api(
            f"/{PLUGIN_NAME}/download",
            self.api_viz_download,
            ["GET"],
            "Ariadne 面板导出下载代理",
        )

    # ---------- 生命周期 ----------

    async def initialize(self) -> None:
        """AstrBot 加载插件后调用；按配置自动拉起服务。"""
        try:
            self._ensure_mcp_token()
            if self._get("mcp", "auto_start", True):
                logger.info("[ariadne] %s", await self._start("mcp"))
            if self._get("panel", "enabled", False):
                logger.info("[ariadne] %s", await self._start("panel"))
        except Exception as exc:  # 进程问题不应拖垮 AstrBot
            logger.error("[ariadne] 自动启动失败: %s", exc)

    async def terminate(self) -> None:
        """插件卸载 / 重载时停止所有子进程。"""
        for name in list(self.procs):
            await self._stop(name)

    # ---------- 配置读取 ----------

    def _data_dir(self) -> Path:
        if get_astrbot_plugin_data_path is not None:
            try:
                return Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
            except Exception:  # pragma: no cover
                pass
        return Path(__file__).resolve().parent / "data"

    def _get(self, section: str, key: str, default: Any = None) -> Any:
        node = self.config.get(section) if hasattr(self.config, "get") else None
        if not isinstance(node, dict):
            return default
        value = node.get(key, default)
        if value is None or value == "":
            return default
        return value

    def _ensure_mcp_token(self) -> str:
        """确保有一个内部 Bearer Token，用于面板 API 代理与服务鉴权。

        没有配置时自动生成并保存，避免面板在“无鉴权”状态下被代理访问。
        """
        token = str(self._get("mcp", "token", "") or "")
        if token:
            return token
        token = secrets.token_urlsafe(24)
        if not isinstance(self.config.get("mcp"), dict):
            self.config["mcp"] = {}
        self.config["mcp"]["token"] = token
        try:
            self.config.save_config()
        except Exception as exc:  # 保存失败仍可本次使用
            logger.debug("[ariadne] 保存内部 token 失败：%s", exc)
        return token

    def _graphs_dir(self) -> Path:
        raw = str(self._get("storage", "graphs_dir", "data")).strip()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self._data_dir() / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _graph_path(self) -> Path | None:
        raw = str(self._get("storage", "yaml", "")).strip()
        if raw:
            path = Path(raw).expanduser()
            return path if path.is_absolute() else self._graphs_dir() / path
        preferred = self._graphs_dir() / "memory_graph.yaml"
        if preferred.exists():
            return preferred
        candidates = sorted(self._graphs_dir().glob("*.yaml"))
        return candidates[0] if candidates else None

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        mode = str(self._get("embedding", "mode", "api")).lower()

        mapping = {
            "OPENAI_API_KEY": self._get("llm", "api_key"),
            "OPENAI_API_BASE": self._get("llm", "api_base"),
            "OPENAI_MODEL": self._get("llm", "model"),
            "OPENAI_THINKING": "1" if self._get("llm", "thinking", False) else "0",
            "EMBEDDING_LOCAL": "true" if mode == "local" else "false",
            "EMBEDDING_API_KEY": self._get("embedding", "api_key"),
            "EMBEDDING_API_BASE": self._get("embedding", "api_base"),
            "EMBEDDING_MODEL": self._get("embedding", "model"),
            "ARIADNE_GRAPHS_DIR": str(self._graphs_dir()),
            "ARIADNE_MCP_TOKEN": self._ensure_mcp_token(),
            "ARIADNE_VIZ_USER": self._get("panel", "user", ""),
            "ARIADNE_VIZ_PASS": self._get("panel", "password", ""),
            "ARIADNE_LOG_LEVEL": self._get("advanced", "log_level", "INFO"),
            "ARIADNE_SOURCE_STORE": "1" if self._get("advanced", "source_store", True) else "0",
            "ARIADNE_REVIEW_IDLE": str(self._get("advanced", "review_idle", 300)),
        }
        yaml_path = self._graph_path()
        if yaml_path is not None:
            mapping["ARIADNE_YAML"] = str(yaml_path)

        for key, value in mapping.items():
            if value not in (None, ""):
                env[key] = str(value)
        return env

    # ---------- 进程管理 ----------

    def _running(self, name: str) -> bool:
        proc = self.procs.get(name)
        return proc is not None and proc.returncode is None

    def _uv_ready(self) -> bool:
        return shutil.which("uv") is not None and shutil.which("uvx") is not None

    async def _start(self, name: str) -> str:
        spec = SERVICE_SPECS.get(name)
        if spec is None:
            return f"未知服务：{name}"
        if self._running(name):
            return f"{name} 已在运行（pid {self.procs[name].pid}）"
        if not self._uv_ready():
            return "未找到 uv/uvx：插件用 uvx 隔离运行服务，请先安装 uv（https://docs.astral.sh/uv/）"

        cfg = spec["cfg"]
        host = str(self._get(cfg, "host", "127.0.0.1"))
        port = int(self._get(cfg, "port", 8766 if name == "mcp" else 8765))
        args = ["uvx", "--from", UVX_FROM, BINARIES[name]]

        if name == "mcp":
            args += ["--sse", "--host", host, "--port", str(port)]
        else:
            args += ["--host", host, "--port", str(port)]
        args += ["--graph-dir", str(self._graphs_dir()), "--log-level",
                 str(self._get("advanced", "log_level", "INFO"))]
        yaml_path = self._graph_path()
        if yaml_path is not None:
            args += ["--yaml", str(yaml_path)]

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self._data_dir()),
                env=self._build_env(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            return f"{name} 启动失败：{exc}"

        self.procs[name] = proc
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                task = asyncio.create_task(self._pump(name, stream))
                self._pumps.add(task)
                task.add_done_callback(self._pumps.discard)

        await asyncio.sleep(0.8)
        if proc.returncode is not None:
            tail = "\n".join(list(self.logs)[-6:])
            return f"{name} 启动后立即退出（code {proc.returncode}）\n{tail}"
        if name == "mcp":
            # AstrBot 的核心 MCP 客户端在插件之前初始化，此刻大概率还没连上；
            # 端口就绪后再让它重连一次，避免“插件已启动但工具不可用”。
            task = asyncio.create_task(self._wake_astrbot_mcp(port))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return f"{name} 已启动（pid {proc.pid}，端口 {port}）"

    async def _stop(self, name: str) -> str:
        proc = self.procs.get(name)
        if proc is None or proc.returncode is not None:
            self.procs.pop(name, None)
            return f"{name} 未在运行"
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=8)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        self.procs.pop(name, None)
        return f"{name} 已停止"

    async def _restart(self, name: str) -> str:
        stopped = await self._stop(name)
        started = await self._start(name)
        return f"{stopped}；{started}"

    async def _wake_astrbot_mcp(self, port: int) -> None:
        """等端口就绪后，让 AstrBot 自己的 MCP 客户端重连一次（尽力而为）。

        AstrBot 在插件加载前就初始化 MCP 客户端，而插件此时才拉起服务，必然先失败。
        这里读取 AstrBot 的 mcp_server.json，只重连指向本插件端口的 SSE 条目；
        失败不影响插件本身（用户仍可在 MCP 页面手动重连）。
        """
        for _ in range(20):
            if _port_open("127.0.0.1", port):
                break
            await asyncio.sleep(1)
        else:
            return

        try:
            manager = getattr(self.context, "provider_manager", None)
            llm_tools = getattr(manager, "llm_tools", None)
            if llm_tools is None:
                return
            servers = (llm_tools.load_mcp_config() or {}).get("mcpServers", {})
            for name, cfg in servers.items():
                url = str(cfg.get("url", ""))
                if f":{port}" not in url or not cfg.get("active", True):
                    continue
                try:
                    await llm_tools.disable_mcp_server(name, timeout=3)
                except Exception:
                    pass
                await llm_tools.enable_mcp_server(name, cfg)
                logger.info("[ariadne] AstrBot MCP 客户端已重连：%s", name)
                return
        except Exception as exc:  # AstrBot 内部 API 变动时静默跳过
            logger.debug("[ariadne] 跳过 AstrBot MCP 重连：%s", exc)

    async def _pump(self, name: str, stream: asyncio.StreamReader) -> None:
        log_file = self._data_dir() / f"{name}.log"
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            if not text:
                continue
            self.logs.append(f"[{name}] {text}")
            try:
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(f"{text}\n")
            except OSError:
                pass

    # ---------- 状态 ----------

    def _config_view(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for section, key in sorted(ALLOWED_PATHS):
            value = self._get(section, key)
            bucket = out.setdefault(section, {})
            if (section, key) in SECRET_PATHS:
                bucket[key] = {"set": bool(value)}
            else:
                bucket[key] = value
        return out

    def _status(self) -> dict[str, Any]:
        services: dict[str, Any] = {}
        for name, spec in SERVICE_SPECS.items():
            cfg = spec["cfg"]
            host = str(self._get(cfg, "host", "127.0.0.1"))
            port = int(self._get(cfg, "port", 8766 if name == "mcp" else 8765))
            proc = self.procs.get(name)
            running = self._running(name)
            services[name] = {
                "running": running,
                "pid": proc.pid if running and proc else None,
                "exit_code": None if proc is None else proc.returncode,
                "host": host,
                "port": port,
                "healthy": _port_open(host, port) if running else False,
                "enabled": bool(self._get(cfg, "enabled", name == "mcp")),
            }
        return {
            "plugin": PLUGIN_NAME,
            "installed": self._uv_ready(),
            "runtime": f"uvx --from {UVX_FROM}",
            "python": sys.executable,
            "data_dir": str(self._data_dir()),
            "graphs_dir": str(self._graphs_dir()),
            "services": services,
            "config": self._config_view(),
        }

    def _graph_stats_sync(self) -> dict[str, Any]:
        path = self._graph_path()
        if path is None or not path.exists():
            return {"path": None, "nodes": None, "edges": None}
        try:
            import yaml  # 由 ariadne-mcp-server 的依赖提供

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {
                "path": str(path),
                "nodes": len(data.get("nodes") or []),
                "edges": len(data.get("edges") or []),
            }
        except Exception as exc:
            return {"path": str(path), "error": str(exc)}

    # ---------- Web API ----------

    async def api_status(self):
        payload = self._status()
        payload["graph"] = await asyncio.to_thread(self._graph_stats_sync)
        return json_response(payload)

    async def api_action(self):
        payload = await request.json(default={})
        action = str(payload.get("action", "")).lower()
        target = str(payload.get("target", "mcp")).lower()
        if target not in SERVICE_SPECS:
            return error_response("target 只能是 mcp 或 panel", status_code=400)
        if action == "start":
            message = await self._start(target)
        elif action == "stop":
            message = await self._stop(target)
        elif action == "restart":
            message = await self._restart(target)
        else:
            return error_response("action 只能是 start / stop / restart", status_code=400)
        return json_response({"message": message, "status": self._status()})

    async def api_logs(self):
        limit = request.query.get("limit", 120, type=int)
        limit = max(1, min(limit, len(self.logs) or 1))
        return json_response({"lines": list(self.logs)[-limit:]})

    async def api_config_get(self):
        return json_response({"config": self._config_view()})

    async def api_config_save(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求体必须是 JSON 对象", status_code=400)

        changed: list[str] = []
        for section, key in ALLOWED_PATHS:
            group = payload.get(section)
            if not isinstance(group, dict) or key not in group:
                continue
            raw = group[key]
            if (section, key) in SECRET_PATHS and raw == "":
                continue  # 留空表示保持原值
            if (section, key) in INT_PATHS:
                try:
                    value: Any = int(raw)
                except (TypeError, ValueError):
                    return error_response(f"{section}.{key} 必须是整数", status_code=400)
            elif (section, key) in BOOL_PATHS:
                value = bool(raw)
            else:
                value = str(raw)
            if not isinstance(self.config.get(section), dict):
                self.config[section] = {}
            self.config[section][key] = value
            changed.append(f"{section}.{key}")

        if changed:
            try:
                self.config.save_config()
            except Exception as exc:
                return error_response(f"保存失败：{exc}", status_code=500)
        return json_response({"changed": changed, "status": self._status()})

    async def api_snippet(self):
        graphs = str(self._graphs_dir())
        host = str(self._get("mcp", "host", "127.0.0.1"))
        port = int(self._get("mcp", "port", 8766))
        args = ["--from", UVX_FROM, "ariadne-mcp", "--graph-dir", graphs]
        yaml_path = self._graph_path()
        if yaml_path is not None:
            args += ["--yaml", str(yaml_path)]
        env = {
            "OPENAI_API_KEY": "<YOUR_KEY>",
            "OPENAI_API_BASE": self._get("llm", "api_base", "https://api.deepseek.com/v1"),
            "OPENAI_MODEL": self._get("llm", "model", "deepseek-v4-flash"),
            "EMBEDDING_LOCAL": "true"
            if str(self._get("embedding", "mode", "api")).lower() == "local"
            else "false",
            "EMBEDDING_API_KEY": "<YOUR_KEY>",
            "EMBEDDING_API_BASE": self._get("embedding", "api_base", "https://api.siliconflow.cn/v1"),
            "EMBEDDING_MODEL": self._get("embedding", "model", "BAAI/bge-m3"),
            "ARIADNE_GRAPHS_DIR": graphs,
        }
        return json_response(
            {
                "stdio": {"command": "uvx", "args": args, "env": env},
                "sse": {
                    "url": f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/sse",
                    "transport": "sse",
                },
            }
        )

    # ---------- 面板 API 代理 ----------
    #
    # 插件视图的 iframe 没有 allow-same-origin（无 Cookie / localStorage），面板前端
    # 通过 AstrBot 官方 bridge 把请求交给父页面，父页面带 Dashboard 登录态打到下面
    # 三个端点；插件再用内部 Bearer 访问容器内的面板服务。面板只监听 127.0.0.1，
    # 不需要端口映射、域名或宿主机 nginx。

    def _panel_url(self, path: Any, query: Any = "") -> str | None:
        """拼出面板上游 URL；只允许 /api/ 路径，避免代理变成任意内网请求。"""
        target = str(path or "").strip()
        if not target.startswith("/"):
            target = "/" + target
        if not target.startswith("/api/") or ".." in target or "://" in target:
            return None
        port = int(self._get("panel", "port", 8765))
        url = f"http://127.0.0.1:{port}{target}"
        raw_query = str(query or "").lstrip("?")
        if raw_query:
            url += "?" + raw_query
        return url

    async def api_viz_proxy(self):
        """转发面板 JSON API：bridge.apiPost('proxy', {method,path,query,body})。"""
        if not self._running("panel"):
            return error_response("面板未运行：请先在状态视图启动 panel", status_code=503)
        payload = await request.json(default={})
        payload = payload if isinstance(payload, dict) else {}
        method = str(payload.get("method") or "GET").upper()
        if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            return error_response("不支持的请求方法", status_code=400)
        url = self._panel_url(payload.get("path"), payload.get("query"))
        if url is None:
            return error_response("非法的面板路径", status_code=400)

        headers = {"Authorization": f"Bearer {self._ensure_mcp_token()}"}
        content = None
        body = payload.get("body")
        if body is not None:
            content = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, read=3600.0), trust_env=False
            ) as client:
                upstream = await client.request(method, url, headers=headers, content=content)
        except Exception as exc:
            return error_response(f"面板不可达：{exc}", status_code=502)

        content_type = upstream.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = upstream.json()
            except Exception:
                data = {"error": upstream.text[:500]}
            is_health = url.split("?", 1)[0].endswith("/api/health")
            if upstream.status_code >= 400 and not is_health:
                message = None
                if isinstance(data, dict):
                    message = data.get("error") or data.get("detail")
                if not message:
                    message = upstream.text[:300] or f"面板返回 {upstream.status_code}"
                return error_response(str(message), status_code=upstream.status_code)
            return json_response(data)

        if upstream.status_code >= 400:
            return error_response(
                upstream.text[:300] or f"面板返回 {upstream.status_code}",
                status_code=upstream.status_code,
            )
        return json_response({"text": upstream.text})

    async def api_viz_stream(self):
        """转发面板 SSE：bridge.subscribeSSE('stream', handlers, {path})。"""
        if not self._running("panel"):
            return error_response("面板未运行：请先在状态视图启动 panel", status_code=503)
        url = self._panel_url(request.query.get("path", "/api/stream"))
        if url is None:
            return error_response("非法的面板路径", status_code=400)

        headers = {
            "Authorization": f"Bearer {self._ensure_mcp_token()}",
            "Accept": "text/event-stream",
        }
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=None), trust_env=False
        )
        try:
            upstream_request = client.build_request("GET", url, headers=headers)
            upstream = await client.send(upstream_request, stream=True)
        except Exception as exc:
            await client.aclose()
            return error_response(f"面板不可达：{exc}", status_code=502)

        if upstream.status_code >= 400:
            data = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            return error_response(
                data.decode("utf-8", "replace")[:300] or f"面板返回 {upstream.status_code}",
                status_code=upstream.status_code,
            )

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return stream_response(relay())

    async def api_viz_download(self):
        """转发面板导出文件：bridge.download('download', {path, query})。"""
        if not self._running("panel"):
            return error_response("面板未运行：请先在状态视图启动 panel", status_code=503)
        url = self._panel_url(
            request.query.get("path", "/api/export/yaml"),
            request.query.get("query", ""),
        )
        if url is None:
            return error_response("非法的面板路径", status_code=400)

        headers = {"Authorization": f"Bearer {self._ensure_mcp_token()}"}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, read=3600.0), trust_env=False
            ) as client:
                upstream = await client.get(url, headers=headers)
        except Exception as exc:
            return error_response(f"面板不可达：{exc}", status_code=502)

        if upstream.status_code >= 400:
            return error_response(
                upstream.text[:300] or f"面板返回 {upstream.status_code}",
                status_code=upstream.status_code,
            )

        extra_headers = {}
        disposition = upstream.headers.get("content-disposition")
        if disposition:
            extra_headers["Content-Disposition"] = disposition
        media_type = (
            upstream.headers.get("content-type", "").split(";")[0]
            or "application/octet-stream"
        )
        return Response(
            content=upstream.content,
            media_type=media_type,
            headers=extra_headers or None,
        )

    # ---------- 聊天指令 ----------

    @filter.command("ariadne")
    async def ariadne_command(self, event: AstrMessageEvent):
        parts = (getattr(event, "message_str", "") or "").split()
        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub == "status":
            status = self._status()
            graph = await asyncio.to_thread(self._graph_stats_sync)
            lines = [
                f"Ariadne · 依赖{'已安装' if status['installed'] else '未安装'}",
                f"图谱：{graph.get('nodes', '-')} 节点 / {graph.get('edges', '-')} 关系",
            ]
            for name, info in status["services"].items():
                state = "运行中" if info["running"] else "已停止"
                lines.append(
                    f"{name}：{state} · {info['host']}:{info['port']} · 端口"
                    f"{'可达' if info['healthy'] else '不可达'}"
                )
            text = "\n".join(lines)
        elif sub in {"start", "stop", "restart"}:
            target = parts[2].lower() if len(parts) > 2 else "mcp"
            if target not in SERVICE_SPECS:
                text = "用法：/ariadne start|stop|restart [mcp|panel]"
            elif sub == "start":
                text = await self._start(target)
            elif sub == "stop":
                text = await self._stop(target)
            else:
                text = await self._restart(target)
        elif sub == "logs":
            text = "\n".join(list(self.logs)[-15:]) or "暂无日志"
        else:
            text = "用法：/ariadne [status|start|stop|restart|logs] [mcp|panel]"

        yield event.plain_result(text)
