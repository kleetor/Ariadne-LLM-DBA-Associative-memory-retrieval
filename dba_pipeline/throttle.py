# SPDX-License-Identifier: Apache-2.0
"""出站配额节流：**字符速率**令牌桶（等待式）。

为什么按字符而不是请求数：线上 429 的响应体原文是
``Details: TPM limit reached.``——限的是 **token/分钟**，不是请求数。字符是
token 的线性代理（1 token ≈ 1~4 字符），按字符计费最贴近真实配额。

为什么"等待"而不是"快速失败"：调用方以 HTTP 200 判定"本次完成"，且**单请求允许
最长 30 分钟**。所以在本服务内**排队等待**（合法占用请求时间）比打爆配额后吃 429
（→ 5xx → 调用方重试 → 更多流量）对系统更安全——即"以时间换稳定"。

自适应（AIMD）：供应商配额不可知 → 遇 429 速率**减半**，此后每次成功**缓慢回升**
（+10%，不超过初始配置），自动收敛到"略低于限速线"。

默认**关闭**：只有调用方显式 ``configure(...)`` 才启用，故 MCP / 单测不受影响。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_ENV_EMBED = "ALM_EMBED_CHARS_PER_MIN"
_ENV_LLM = "ALM_LLM_CHARS_PER_MIN"
_ENV_WAIT = "ALM_THROTTLE_MAX_WAIT"


class CharBucket:
    """字符令牌桶：``acquire(n)`` 取不到就等，最多等 ``max_wait`` 秒（超时放行）。

    超时**放行**（fail-open）是有意的：把这次略超配额，比返回 5xx 让调用方重试更安全；
    而 AIMD 已经把速率压下来了，所以放行的规模有限。
    """

    def __init__(self, name: str, chars_per_min: float, max_wait: float = 120.0):
        self.name = name
        self.max_wait = max(0.0, float(max_wait))
        self._lock = threading.Lock()
        self._ceiling = max(1.0, float(chars_per_min))   # 配置上限（回升的天花板）
        self._cap = self._ceiling                        # 当前速率（字符/分）
        self._tokens = self._cap / 6.0                   # 突发额度 = 10 秒的量
        self._ts = time.monotonic()
        # 可观测
        self.waits = 0
        self.total_wait = 0.0
        self.rate_cuts = 0
        self.overflow = 0

    # ---- 自适应 ----
    def note_429(self) -> None:
        """遇 429：速率减半（下限 2000 字符/分）。"""
        with self._lock:
            self._cap = max(2000.0, self._cap * 0.5)
            self.rate_cuts += 1
            cap = self._cap
        logger.warning("节流[%s]: 遇 429，速率降至 %.0f 字符/分", self.name, cap)

    def note_success(self) -> None:
        """成功一次：若曾被降速，则**缓慢**回升（+2%，不超过配置上限）。

        步长取 2% 而不是 10%：`note_success` 是**按调用**触发的，一次 Add/Search 里
        embedding 与 LLM 各有几十次调用，10% 会让减半后的速率在个位数次成功内就回满，
        于是在限速线附近反复冲顶 → 429 抖动（这正是"收敛到略低于限速线"要避免的）。
        2% 下从减半回到上限约需 35 次成功，给上游留出稳定的观察窗。
        """
        with self._lock:
            if self._cap < self._ceiling:
                self._cap = min(self._ceiling, self._cap * 1.02)

    @property
    def rate(self) -> float:
        with self._lock:
            return self._cap

    def stats(self) -> dict:
        with self._lock:
            return {"rate": round(self._cap), "waits": self.waits,
                    "total_wait": round(self.total_wait, 2),
                    "rate_cuts": self.rate_cuts, "overflow": self.overflow,
                    "tokens": round(self._tokens, 1)}

    # ---- 取令牌 ----
    def acquire(self, n: float) -> float:
        """取 ``n`` 字符额度；不足则等待。返回本次实际等待秒数。

        ⚠️ 单次请求可能**大于突发额度**（`_cap/6`，即 10 秒的量）：例如一次 /search 的
        rerank 输入 ≈ 50 条 × 1800 字 ≈ 9 万字，而 30 万字/分的桶突发额度只有 5 万。
        若仍按 `tokens >= n` 判定，就**永远等不到**（桶的物理上限就是 5 万）→ 空转到
        `max_wait`（120s）才放行，等于每次调用白等两分钟。故判据用
        `need = min(n, 突发额度)`：桶一攒满就放行，并把**整笔 n 记成欠账**（`_tokens`
        允许为负，下限 -突发额度），由后续调用等待补齐——平均速率仍被压住，而单次
        超大请求不会被卡死。
        """
        n = max(0.0, float(n))
        if n <= 0 or self._cap <= 0:
            return 0.0
        waited = 0.0
        deadline = time.monotonic() + self.max_wait
        while True:
            with self._lock:
                now = time.monotonic()
                burst = self._cap / 6.0
                # 突发额度随**当前速率**伸缩：被 429 降速后，"一口气能突发多少"必须一起
                # 收紧，否则降速只压住了平均速率、仍允许一次性打出一大口（配额照样爆）。
                self._tokens = min(
                    burst,
                    self._tokens + (now - self._ts) * self._cap / 60.0,
                )
                self._ts = now
                need = min(n, burst)
                if self._tokens >= need:
                    # 欠账下限 -burst：给"补齐欠账"的等待封顶（最多约 10s 攒回一个突发量）。
                    # 不限下限时，一次全量重建（数百万字符）会让之后**每次**调用都等满 max_wait。
                    self._tokens = max(-burst, self._tokens - n)
                    if waited > 0:
                        self.waits += 1
                        self.total_wait += waited
                    return waited
                remain = deadline - now
                wait = min((need - self._tokens) / (self._cap / 60.0), max(0.0, remain))
            if wait <= 0:
                with self._lock:
                    self.overflow += 1
                logger.warning("节流[%s]: 已等 %.0fs 仍不足，放行本次（%.0f 字符）",
                               self.name, self.max_wait, n)
                return waited
            time.sleep(wait)
            waited += wait


# ---- 进程级单例（默认关闭）----
_embed_bucket: Optional[CharBucket] = None
_llm_bucket: Optional[CharBucket] = None
_rerank_bucket: Optional[CharBucket] = None


def configure(embed_chars_per_min: float = 0, llm_chars_per_min: float = 0,
              rerank_chars_per_min: float = 0, max_wait: float = 120.0) -> None:
    """启用节流（由 ALM 启动时调用）。三者都 ≤0 表示关闭。

    rerank 单独一只桶、**不与 embedding 共用**：两者虽同一供应商，但 embedding 在
    **写侧**（/add 要把整批文档全部嵌入，是出站量的大头），rerank 在**读侧**。共用一只
    桶会让读侧的 rerank 挤占写侧预算 → 拖慢调用方写入进度。分开计量既不互相饿死，
    也各自保有收敛空间。
    """
    global _embed_bucket, _llm_bucket, _rerank_bucket
    _embed_bucket = (CharBucket("embedding", embed_chars_per_min, max_wait)
                     if embed_chars_per_min and embed_chars_per_min > 0 else None)
    _llm_bucket = (CharBucket("llm", llm_chars_per_min, max_wait)
                   if llm_chars_per_min and llm_chars_per_min > 0 else None)
    _rerank_bucket = (CharBucket("rerank", rerank_chars_per_min, max_wait)
                      if rerank_chars_per_min and rerank_chars_per_min > 0 else None)
    if _embed_bucket or _llm_bucket or _rerank_bucket:
        logger.info(
            "出站节流已启用: embedding=%s 字符/分, llm=%s 字符/分, rerank=%s 字符/分, "
            "单次最长等待=%.0fs",
            round(_embed_bucket.rate) if _embed_bucket else "关",
            round(_llm_bucket.rate) if _llm_bucket else "关",
            round(_rerank_bucket.rate) if _rerank_bucket else "关",
            max_wait,
        )


def embed_bucket() -> Optional[CharBucket]:
    return _embed_bucket


def llm_bucket() -> Optional[CharBucket]:
    return _llm_bucket


def rerank_bucket() -> Optional[CharBucket]:
    return _rerank_bucket


def snapshot() -> dict:
    return {
        "embedding": _embed_bucket.stats() if _embed_bucket else None,
        "llm": _llm_bucket.stats() if _llm_bucket else None,
        "rerank": _rerank_bucket.stats() if _rerank_bucket else None,
    }


try:  # LangChain 回调（计量与限速共用 SDK，故这里是可选依赖）
    from langchain_core.callbacks import BaseCallbackHandler

    class ThrottleCallback(BaseCallbackHandler):
        """在 ``on_llm_start`` 里取令牌 → 同步阻塞，等效给 LLM 调用限速。

        LLM 侧不做 AIMD（拿不到响应状态码）：只用配置的固定速率 + 等待。
        """

        def __init__(self, bucket: CharBucket, reserve_output_chars: int = 0):
            self.bucket = bucket
            self.reserve = max(0, int(reserve_output_chars))

        def on_llm_start(self, serialized, prompts, **kwargs):  # noqa: D401
            try:
                n = sum(len(p) for p in (prompts or [])) + self.reserve
                self.bucket.acquire(n)
            except Exception as exc:  # 节流本身绝不能把请求弄挂
                logger.warning("节流[llm] 取令牌异常，放行: %s", exc)

except Exception:  # pragma: no cover
    ThrottleCallback = None  # type: ignore
