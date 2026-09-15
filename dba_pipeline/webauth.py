# SPDX-License-Identifier: AGPL-3.0-only

"""对外入口的统一鉴权。

面板（``viz/api_server``）与 MCP SSE（``mcp_server.run_sse``）共用本模块，
保证「所有对外暴露的接口」都走同一套判定：

- **始终鉴权**：账号密码存在 ``<图谱目录>/auth.json``，只存
  PBKDF2-HMAC-SHA256 加盐散列（不可逆），首次启动自动创建默认账号
  ``ariadne / ariadne`` 并要求首次登录后修改；
- **浏览器**：登录页换取签名会话 Cookie（HttpOnly，默认 7 天）；
- **脚本 / 第三方客户端**：HTTP Basic（MCP 另可用 ``ARIADNE_MCP_TOKEN`` 走 Bearer）；
- **重置后门**：``ARIADNE_VIZ_USER`` / ``ARIADNE_VIZ_PASS`` 仅在忘记密码时用于
  登录（登录后强制改密），不参与日常校验；
- **可选 pepper**：``ARIADNE_AUTH_KEY`` 存在时参与散列，相当于在散列之外再加
  一个只存在环境里的密钥——配置文件和散列值被看到也无法离线爆破。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence
from urllib.parse import quote

from starlette.datastructures import Headers
from starlette.responses import JSONResponse, RedirectResponse

SESSION_COOKIE = "ariadne_session"
SESSION_TTL = 7 * 24 * 3600      # 会话有效期（秒）
SECRET_FILENAME = ".ariadne_secret"
AUTH_FILENAME = "auth.json"      # 账号密码存储（与图谱同目录）

DEFAULT_USER = "ariadne"
DEFAULT_PASSWORD = "ariadne"
MIN_PASSWORD_LENGTH = 8
PBKDF2_ITERATIONS = 200_000

LOGIN_WINDOW = 300               # 登录失败统计窗口（秒）
LOGIN_MAX_FAILURES = 5           # 窗口内允许的失败次数，超出则暂时拒绝

# 这些路径无需鉴权：登录流程自身 + 健康探针
ALWAYS_EXEMPT = ("/api/health", "/api/login", "/api/logout", "/login")

# 强制改密期间仍可访问的路径；其余接口一律 403，
# 避免初始密码（或重置后门）被当成长期凭据用下去。
STALE_ALLOWED = ("/", "/login", "/api/password", "/api/logout", "/api/config", "/api/health")


# ---------------------------------------------------------------------------
# 散列
# ---------------------------------------------------------------------------

def load_backdoor() -> "Backdoor":
    """重置后门（ARIADNE_VIZ_USER / ARIADNE_VIZ_PASS），只配一半视为未配置。"""
    return Backdoor(
        user=os.environ.get("ARIADNE_VIZ_USER") or "",
        password=os.environ.get("ARIADNE_VIZ_PASS") or "",
    )


def backdoor_warning(backdoor: "Backdoor") -> Optional[str]:
    """后门相关的启动提醒（含「只配了一半」这种误配）；无需提醒时返回 None。"""
    user = os.environ.get("ARIADNE_VIZ_USER") or ""
    password = os.environ.get("ARIADNE_VIZ_PASS") or ""
    if bool(user) != bool(password):
        return "警告: ARIADNE_VIZ_USER 与 ARIADNE_VIZ_PASS 需同时设置，重置后门不会生效"
    if not backdoor.configured:
        return None
    return (
        "提示: 检测到 ARIADNE_VIZ_USER / ARIADNE_VIZ_PASS 重置后门已配置。"
        "它可绕过凭据文件直接登录（登录后强制改密）；密码改好后建议从 .env 中移除。"
    )


def load_bearer_token() -> Optional[str]:
    """MCP SSE 可选的非浏览器凭据（ARIADNE_MCP_TOKEN）。"""
    return os.environ.get("ARIADNE_MCP_TOKEN") or None


def _pepper() -> bytes:
    """可选 pepper：只存在于环境里，不落盘。"""
    return (os.environ.get("ARIADNE_AUTH_KEY") or "").encode("utf-8")


def _pepper_id(pepper: bytes) -> str:
    """pepper 指纹，用于识别「key 被换过导致原密码无法校验」。"""
    return hashlib.sha256(pepper).hexdigest()[:12] if pepper else ""


def _derive(password: str, salt: bytes, iterations: int, pepper: bytes) -> str:
    secret = hmac.new(pepper, password.encode("utf-8"), hashlib.sha256).digest() \
        if pepper else password.encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", secret, salt, iterations).hex()


def constant_time_eq(left: str, right: str) -> bool:
    """常量时间比较两个字符串。

    注意：``hmac.compare_digest`` 不接受含非 ASCII 字符的 str（会抛 TypeError），
    故统一先编码成 bytes——中文用户名/密码同样能用。
    """
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def validate_username(user: str) -> Optional[str]:
    """返回错误文案；合法返回 None。"""
    if not user or len(user) > 64 or "|" in user or any(c.isspace() for c in user):
        return '用户名不能为空、不含空白字符与 “|”，且不超过 64 个字符'
    return None


def validate_password(password: str) -> Optional[str]:
    """返回错误文案；合法返回 None。"""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"新密码至少 {MIN_PASSWORD_LENGTH} 位"
    return None


# ---------------------------------------------------------------------------
# 凭据文件
# ---------------------------------------------------------------------------

def _data_dir(yaml_path: Optional[str]) -> str:
    return os.path.dirname(os.path.abspath(yaml_path)) if yaml_path else "."


def auth_path_for(yaml_path: Optional[str] = None) -> str:
    return os.path.join(_data_dir(yaml_path), AUTH_FILENAME)


def _mtime_of(path: str) -> int:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return 0


def _read_record(path: str) -> dict:
    """读取凭据文件；不存在抛 FileNotFoundError，内容非法抛 ValueError。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("不是 JSON 对象")
    try:
        record = {
            "user": str(raw["user"]),
            "salt": str(raw["salt"]),
            "hash": str(raw["hash"]),
            "iterations": int(raw["iterations"]),
            "key_id": str(raw.get("key_id", "")),
            "must_change": bool(raw.get("must_change", False)),
        }
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"字段非法: {e}") from e
    bytes.fromhex(record["salt"])          # salt 必须是合法 hex
    if record["iterations"] < 1000:
        raise ValueError("iterations 过小")
    return record


class AuthStore:
    """账号密码存储（``<图谱目录>/auth.json``）。

    面板与 MCP 可能同时读这份文件，故按 mtime 门控重载；写入走
    「临时文件 + 原子替换」，读方永远看不到半截内容。
    """

    def __init__(self, path: str, user: str, salt: str, password_hash: str,
                 iterations: int, key_id: str, must_change: bool):
        self.path = path
        self.user = user
        self.salt = salt
        self.password_hash = password_hash
        self.iterations = iterations
        self.key_id = key_id
        self.must_change = must_change
        self._mtime = 0

    # ---- 读取 ----

    @classmethod
    def load_or_create(cls, yaml_path: Optional[str] = None, notice: bool = True) -> "AuthStore":
        """加载凭据；文件不存在（或损坏）时用默认账号初始化并提醒。"""
        path = auth_path_for(yaml_path)
        try:
            record = _read_record(path)
        except FileNotFoundError:
            store = cls._default(path)
            store.save()
            if notice:
                _print_default_notice(store)
            return store
        except (OSError, ValueError) as e:
            store = cls._default(path)
            broken = path + ".broken"
            try:
                os.replace(path, broken)
            except OSError:
                broken = "（备份失败）"
            store.save()
            if notice:
                print(f"警告: 凭据文件 {path} 无法读取（{e}），"
                      f"已备份为 {broken} 并恢复默认账号", file=sys.stderr)
                _print_default_notice(store)
            return store
        store = cls(path=path, user=record["user"], salt=record["salt"],
                    password_hash=record["hash"], iterations=record["iterations"],
                    key_id=record["key_id"], must_change=record["must_change"])
        store._mtime = _mtime_of(path)
        return store

    @classmethod
    def _default(cls, path: str) -> "AuthStore":
        salt = secrets.token_bytes(16)
        pepper = _pepper()
        return cls(
            path=path, user=DEFAULT_USER, salt=salt.hex(),
            password_hash=_derive(DEFAULT_PASSWORD, salt, PBKDF2_ITERATIONS, pepper),
            iterations=PBKDF2_ITERATIONS, key_id=_pepper_id(pepper), must_change=True,
        )

    def refresh(self) -> None:
        """凭据文件被其它进程改写后重读（文件消失时保留内存里的值）。"""
        mtime = _mtime_of(self.path)
        if not mtime or mtime == self._mtime:
            return
        try:
            record = _read_record(self.path)
        except (OSError, ValueError):
            return
        self.user = record["user"]
        self.salt = record["salt"]
        self.password_hash = record["hash"]
        self.iterations = record["iterations"]
        self.key_id = record["key_id"]
        self.must_change = record["must_change"]
        self._mtime = mtime

    # ---- 校验 / 写入 ----

    @property
    def key_ok(self) -> bool:
        """当前 ARIADNE_AUTH_KEY 与散列时所用的一致？不一致则原密码无法校验。"""
        return self.key_id == _pepper_id(_pepper())

    def check_password(self, password: str) -> bool:
        self.refresh()
        if not self.key_ok:
            return False
        try:
            salt = bytes.fromhex(self.salt)
        except ValueError:
            return False
        derived = _derive(password, salt, self.iterations, _pepper())
        return constant_time_eq(derived, self.password_hash)

    def verify(self, user: str, password: str) -> bool:
        """校验账号密码；用户名不匹配也照算散列，避免按用户名短路泄漏时序。"""
        self.refresh()
        password_ok = self.check_password(password)
        return bool(password_ok and constant_time_eq(user, self.user))

    def set_password(self, password: str, user: Optional[str] = None) -> None:
        """改密（可同时改名），并清除「必须改密」标记。"""
        self.refresh()
        salt = secrets.token_bytes(16)
        pepper = _pepper()
        if user:
            self.user = user
        self.salt = salt.hex()
        self.iterations = PBKDF2_ITERATIONS
        self.password_hash = _derive(password, salt, self.iterations, pepper)
        self.key_id = _pepper_id(pepper)
        self.must_change = False
        self.save()

    def save(self) -> None:
        data = {
            "version": 1,
            "user": self.user,
            "salt": self.salt,
            "hash": self.password_hash,
            "iterations": self.iterations,
            "key_id": self.key_id,
            "must_change": self.must_change,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self._mtime = _mtime_of(self.path)


def _print_default_notice(store: AuthStore) -> None:
    print(
        f"已创建凭据文件: {store.path}\n"
        f"  默认账号: {DEFAULT_USER} / {DEFAULT_PASSWORD}（首次登录后必须修改密码）",
        file=sys.stderr,
    )


@dataclass
class Backdoor:
    """回落凭据（环境变量），仅用于忘记密码时登录并强制改密。"""

    user: str = ""
    password: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.user and self.password)


@dataclass
class Session:
    """一次请求的登录态。"""

    user: str
    must_change: bool = False


# ---------------------------------------------------------------------------
# 会话签名
# ---------------------------------------------------------------------------

def secret_path_for(yaml_path: Optional[str]) -> str:
    return os.path.join(_data_dir(yaml_path), SECRET_FILENAME)


def load_or_create_secret(yaml_path: Optional[str] = None) -> bytes:
    """会话签名密钥：优先 ARIADNE_SECRET_KEY，否则落在图谱同目录并复用。

    持久化是为了让服务重启后登录态不失效；文件权限收紧到 0o600。
    """
    env = os.environ.get("ARIADNE_SECRET_KEY")
    if env:
        return env.encode("utf-8")

    path = secret_path_for(yaml_path)
    try:
        with open(path, "rb") as f:
            existing = f.read().strip()
        if existing:
            return existing
    except OSError:
        pass

    fresh = secrets.token_bytes(32)
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(fresh)
    except OSError:
        pass  # 落盘失败则退化为本次进程内有效
    return fresh


class SessionSigner:
    """无状态会话：Cookie 内容为 ``base64(user|过期时间|改密标记|HMAC 签名)``。"""

    def __init__(self, secret: bytes, ttl: int = SESSION_TTL):
        self.secret = secret
        self.ttl = ttl

    def _sign(self, payload: str) -> str:
        return hmac.new(self.secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def issue(self, user: str, must_change: bool = False) -> str:
        payload = f"{user}|{int(time.time()) + self.ttl}|{1 if must_change else 0}"
        raw = f"{payload}|{self._sign(payload)}"
        return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")

    def verify(self, token: str) -> Optional[Session]:
        """校验并返回登录态；签名不符或已过期返回 None。"""
        if not token:
            return None
        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
            user, expires, flag, signature = raw.rsplit("|", 3)
            deadline = int(expires)
        except Exception:
            return None
        if not hmac.compare_digest(signature, self._sign(f"{user}|{deadline}|{flag}")):
            return None
        if deadline < time.time():
            return None
        return Session(user=user, must_change=flag == "1")


# ---------------------------------------------------------------------------
# 登录限速
# ---------------------------------------------------------------------------

class LoginRateLimiter:
    """按来源 IP 统计登录失败次数，超出阈值后暂时拒绝（防爆破）。"""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window: int = LOGIN_WINDOW):
        self.max_failures = max_failures
        self.window = window
        self._failures = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> list:
        kept = [t for t in self._failures.get(key, []) if now - t < self.window]
        if kept:
            self._failures[key] = kept
        else:
            self._failures.pop(key, None)
        return kept

    def retry_after(self, key: str) -> int:
        """被限速时返回需等待的秒数，否则 0。"""
        now = time.time()
        with self._lock:
            recent = self._prune(key, now)
            if len(recent) < self.max_failures:
                return 0
            return max(1, int(self.window - (now - recent[0])))

    def record_failure(self, key: str) -> None:
        now = time.time()
        with self._lock:
            recent = self._prune(key, now)
            recent.append(now)
            self._failures[key] = recent

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


# ---------------------------------------------------------------------------
# 统一鉴权中间件
# ---------------------------------------------------------------------------

def _basic_ok(header: str, store: AuthStore) -> bool:
    if not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:]).decode("utf-8")
    except Exception:
        return False
    store.refresh()
    # 初始密码尚未修改前不接受 Basic：脚本客户端会绕过强制改密流程
    if store.must_change:
        return False
    user, _, password = decoded.partition(":")
    return store.verify(user, password)


def _bearer_ok(header: str, token: str) -> bool:
    scheme, _, value = header.partition(" ")
    if scheme.lower() not in ("bearer", "token"):
        return False
    return constant_time_eq(value.strip(), token)


class AuthMiddleware:
    """统一鉴权：会话 Cookie / HTTP Basic / Bearer Token 任一通过即可。

    HTML 导航未通过：302 跳登录页；API 与静态资源未通过：401 JSON；
    强制改密期间访问其它接口：403（带 ``must_change`` 标记供前端跳转）。
    """

    def __init__(self, app, store: AuthStore, signer: Optional[SessionSigner] = None,
                 bearer_token: Optional[str] = None,
                 exempt_paths: Sequence[str] = ALWAYS_EXEMPT,
                 login_path: Optional[str] = "/login"):
        self.app = app
        self.store = store
        self.signer = signer
        self.bearer_token = bearer_token
        self.exempt_paths = tuple(exempt_paths)
        self.login_path = login_path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path in self.exempt_paths:
            return await self.app(scope, receive, send)

        session = resolve_session(scope, self.store, self.signer, self.bearer_token)
        if session is None:
            # 浏览器直接导航（非接口、非静态资源）→ 引导到登录页；
            # login_path 为 None 表示没有登录页（如 MCP SSE），一律返回 401
            if self.login_path and self._wants_html(scope, path):
                target = self.login_path
                if path and path != "/":
                    target += "?next=" + _quote(path)
                await RedirectResponse(target, status_code=302)(scope, receive, send)
                return
            await JSONResponse(
                {"error": "未登录或凭据无效"}, status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="ariadne"'},
            )(scope, receive, send)
            return

        if session.must_change and not _stale_allowed(path):
            await JSONResponse(
                {"error": "请先修改初始密码", "must_change": True}, status_code=403,
            )(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    def _wants_html(scope, path: str) -> bool:
        if scope.get("method") != "GET":
            return False
        if path.startswith("/api/") or path.startswith("/static/"):
            return False
        accept = Headers(scope=scope).get("accept", "")
        return "text/html" in accept or accept == ""


def _stale_allowed(path: str) -> bool:
    return path in STALE_ALLOWED or path.startswith("/static/")


def resolve_session(scope, store: AuthStore,
                    signer: Optional[SessionSigner] = None,
                    bearer_token: Optional[str] = None) -> Optional[Session]:
    """从请求中解析登录态；未通过鉴权返回 None。"""
    headers = Headers(scope=scope)

    if signer is not None:
        cookie = headers.get("cookie", "")
        if cookie:
            session = signer.verify(_cookie_value(cookie, SESSION_COOKIE))
            if session is not None:
                return session

    authorization = headers.get("authorization", "")
    if _basic_ok(authorization, store):
        return Session(user=store.user, must_change=store.must_change)
    if bearer_token and _bearer_ok(authorization, bearer_token):
        return Session(user="token")
    return None


def _cookie_value(cookie_header: str, name: str) -> str:
    for part in cookie_header.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value
    return ""


def _quote(value: str) -> str:
    return quote(value, safe="")


def session_cookie(token: str, ttl: int = SESSION_TTL) -> str:
    """构造 Set-Cookie 值。

    注：本地多为 http，故不加 Secure；HttpOnly 防脚本读取，SameSite=Lax 防 CSRF。
    """
    return (
        f"{SESSION_COOKIE}={token}; Path=/; Max-Age={ttl}; "
        "HttpOnly; SameSite=Lax"
    )


def clear_cookie() -> str:
    return f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"


def client_key(scope) -> str:
    """限速用的来源标识（优先取代理透传的真实 IP）。"""
    client = scope.get("client")
    if client:
        return str(client[0])
    return "-"
