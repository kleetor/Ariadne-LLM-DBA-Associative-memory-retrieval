# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 同步 Add / Search HTTP 服务。

契约端点：
    POST /add     同步写入记忆
    POST /search  同步检索记忆
    GET  /health  健康检查（免鉴权）

Usage:
    python -m alm --data-dir data/alm --port 8770
    ariadne-alm --port 8770
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from alm.config import ALMConfig
from alm.contract import (
    ContractError,
    build_add_response,
    build_search_response,
    parse_add_request,
    parse_search_request,
)

if TYPE_CHECKING:  # 仅用于类型标注；运行时延迟导入，见 main()
    from alm.engine import ALMEngine

logger = logging.getLogger("alm")


def _error(status: int, reason: str) -> JSONResponse:
    """ALM 约定的错误体格式：{"detail": {"reason": "..."}}"""
    return JSONResponse({"detail": {"reason": reason}}, status_code=status)


def _extract_key(request: Request) -> str:
    """从 X-Api-Key / Authorization: Bearer|Token 中取出凭据"""
    key = request.headers.get("x-api-key", "").strip()
    if key:
        return key
    auth = request.headers.get("authorization", "")
    scheme, _, value = auth.partition(" ")
    if scheme.lower() in ("bearer", "token"):
        return value.strip()
    return ""


def _authorized(request: Request, config: ALMConfig) -> bool:
    """未配置 Key 时放行（仅限本地自测 / 公开 smoke）"""
    if not config.api_keys:
        return True
    provided = _extract_key(request)
    if not provided:
        return False
    return any(hmac.compare_digest(provided, key) for key in config.api_keys)


def _mask(user_id: str) -> str:
    if len(user_id) <= 8:
        return "***"
    return f"{user_id[:4]}***{user_id[-4:]}"


def create_app(engine: ALMEngine, config: ALMConfig) -> Starlette:
    """构建 Starlette 应用"""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def add(request: Request) -> JSONResponse:
        if not _authorized(request, config):
            return _error(401, "认证失败：缺少或无效的 API Key")

        try:
            body: Any = await request.json()
        except Exception:
            return _error(400, "请求体不是合法 JSON")

        try:
            parsed = parse_add_request(body)
        except ContractError as exc:
            return _error(exc.status, exc.message)

        try:
            # 阻塞式 LLM 维护放到线程池，避免阻塞事件循环（正式评测 Add 并发可达 16–64）
            await run_in_threadpool(engine.add, parsed)
        except Exception as exc:
            logger.exception("Add 处理失败 user_id=%s", _mask(parsed.user_id))
            return _error(500, f"内部错误: {type(exc).__name__}: {str(exc)[:200]}")

        return JSONResponse(
            build_add_response(parsed.request_id, parsed.user_id, parsed.session_id)
        )

    async def search(request: Request) -> JSONResponse:
        if not _authorized(request, config):
            return _error(401, "认证失败：缺少或无效的 API Key")

        try:
            body: Any = await request.json()
        except Exception:
            return _error(400, "请求体不是合法 JSON")

        try:
            parsed = parse_search_request(body, max_top_k=config.max_top_k)
        except ContractError as exc:
            return _error(exc.status, exc.message)

        try:
            items = await run_in_threadpool(engine.search, parsed)
        except Exception as exc:
            logger.exception("Search 处理失败 user_id=%s", _mask(parsed.user_id))
            return _error(500, f"内部错误: {type(exc).__name__}: {str(exc)[:200]}")

        return JSONResponse(build_search_response(items))

    async def on_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未捕获异常: %s", exc)
        return _error(500, "内部错误")

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/add", add, methods=["POST"]),
            Route("/search", search, methods=["POST"]),
        ],
        exception_handlers={Exception: on_error},
    )


def _configure_logging(level: str) -> None:
    """核心链路的 info 日志会打印记忆正文，默认只保留 warning 以上（数据合规）"""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("alm").setLevel(getattr(logging, level.upper(), logging.INFO))


def _find_dotenv() -> Optional[Path]:
    """从当前文件向上查找项目根目录的 .env"""
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_dotenv() -> None:
    """把 .env 写入环境变量。

    必须早于任何重依赖导入（langchain / huggingface_hub 等）：这些库在 import
    时就把 HF_HUB_OFFLINE 等读入模块常量，之后再 setdefault 不会生效——那会导致
    sentence-transformers 尝试联网下载 adapter_config.json 而长时间阻塞。
    因此这里只做纯文件解析，不 import 任何项目内重模块。
    """
    env_path = _find_dotenv()
    if env_path is None:
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # 已存在的环境变量优先（不覆盖外部显式设置）
            os.environ.setdefault(key, value)


def _print_status(config: ALMConfig, engine: ALMEngine) -> None:
    auth = f"开启（{len(config.api_keys)} 个 Key）" if config.api_keys else "未开启（仅限本地/公开 smoke）"
    embedding = f"{config.embedding_model}（{'本地' if config.embedding_local else 'API'}）"
    lines = [
        "=" * 60,
        "[ALM] 运行状态",
        "=" * 60,
        f"  LLM        : {config.llm_model}",
        f"  Embedding  : {embedding}",
        f"  数据目录   : {config.data_dir}",
        f"  鉴权       : {auth}",
        f"  检索参数   : seed_k={config.seed_k} expand_k={config.expand_k} "
        f"max_hops={config.max_hops} max_top_k={config.max_top_k}",
        f"  重排       : {config.rerank_mode}（pool={config.rerank_pool} "
        f"tiers={config.rerank_tier_weights} rescue={config.rerank_rescue}）",
        f"  监听       : http://{config.host}:{config.port}",
        f"  端点       : POST /add  POST /search  GET /health",
        "=" * 60,
    ]
    print("\n".join(lines), file=sys.stderr)


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(description="ALM 同步 Add / Search 服务")
    parser.add_argument("--host", default=None, help="绑定地址（默认取 ALM_HOST，0.0.0.0）")
    parser.add_argument("--port", type=int, default=None, help="端口（默认取 ALM_PORT，8770）")
    parser.add_argument("--data-dir", default=None, help="数据目录（默认取 ALM_DATA_DIR，data/alm）")
    parser.add_argument("--log-level", default="INFO", help="alm 日志级别（核心链路固定 WARNING 以上）")
    args = parser.parse_args()

    config = ALMConfig.from_env()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.data_dir:
        config.data_dir = Path(args.data_dir)

    _configure_logging(args.log_level)

    # 延迟导入：确保 .env（HF_HUB_OFFLINE 等）在 langchain / huggingface_hub
    # 被导入之前就已写入环境变量，否则本地 embedding 会走网络并长时间阻塞。
    from alm.engine import ALMEngine

    try:
        engine = ALMEngine(config)
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_status(config, engine)
    app = create_app(engine, config)

    try:
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
