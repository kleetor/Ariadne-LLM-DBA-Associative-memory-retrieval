# SPDX-License-Identifier: AGPL-3.0-only

"""Cross-encoder 重排（SiliconFlow `/v1/rerank`，模型 `BAAI/bge-reranker-v2-m3`）。

与 embedding 走**同一条端点 / 同一把密钥**（同为免费额度），但职责完全不同：
embedding 决定"召回什么"，rerank 只决定"召回出来的这几条**怎么排**"。因此两者对
失败的处置**刻意相反**：

  · embedding 失败 → 查询向量拿不到，只能抛错（否则与"库里确实没有相关记忆"在调用方侧
    无法区分，会把故障伪装成一条不重试的错误答案）；
  · rerank 失败 → **静默降级**（返回 None，调用方保留原序）。它只是排序优化，挂了不该
    让 /search 失败。

上下文上限 8k：单条文档超限会被端点**静默截断**（实测 15k/30k/100k 字符均返回 200），
这里显式截断到 `max_chars`，既避免送无意义的尾部、也顺带压 token。

配额：用 `dba_pipeline.throttle` 的**独立** rerank 字符桶（`ALM_RERANK_CHARS_PER_MIN`，
默认关）。不与 embedding 共用——embedding 在写侧（/add 要嵌入整批文档），rerank 在读侧，
共用会让读侧挤占写侧预算、拖慢调用方写入进度。
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

_MAX_CHARS = 8000          # bge-reranker-v2-m3 的上下文上限
_TIMEOUT = 30.0
_MAX_ATTEMPTS = 2
_BACKOFF = 1.5
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _bucket():
    """rerank 专用的字符令牌桶（未启用时返回 None）。"""
    try:
        from dba_pipeline import throttle
        return throttle.rerank_bucket()
    except Exception:
        return None


class Reranker:
    """`/v1/rerank` 的薄客户端：一次调用返回 `[(候选下标, 相关度), ...]`（按分数降序）。"""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        max_chars: int = _MAX_CHARS,
        timeout: float = _TIMEOUT,
        max_attempts: int = _MAX_ATTEMPTS,
    ):
        self.model = model
        self.url = api_base.rstrip("/") + "/rerank"
        self.api_key = api_key
        self.max_chars = max(1, int(max_chars))
        self.timeout = max(1.0, float(timeout))
        self.max_attempts = max(1, int(max_attempts))

    @classmethod
    def from_env(cls) -> Optional["Reranker"]:
        """按 `ALM_RERANK_*` 构造；端点/密钥缺省回落 EMBEDDING_*。缺项则返回 None。

        **启动期由调用方按 `ALM_RERANK` 决定是否启用**——本方法只负责"能不能构造出来"。
        """
        def env(name, default=None):
            value = os.environ.get(name)
            return value if value not in (None, "") else default

        model = env("ALM_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        base = env("ALM_RERANK_API_BASE", env("EMBEDDING_API_BASE"))
        key = env("ALM_RERANK_API_KEY", env("EMBEDDING_API_KEY"))
        if not (model and base and key):
            logger.warning(
                "rerank 启用但缺少配置（model=%s base=%s key=%s），已关闭",
                bool(model), bool(base), bool(key),
            )
            return None
        try:
            return cls(
                model=model,
                api_base=base,
                api_key=key,
                max_chars=int(env("ALM_RERANK_MAX_CHARS", str(_MAX_CHARS)) or _MAX_CHARS),
                timeout=float(env("ALM_RERANK_TIMEOUT", str(_TIMEOUT)) or _TIMEOUT),
                max_attempts=int(env("ALM_RERANK_MAX_ATTEMPTS", str(_MAX_ATTEMPTS)) or _MAX_ATTEMPTS),
            )
        except (TypeError, ValueError) as exc:
            logger.warning("rerank 配置解析失败，已关闭: %s", exc)
            return None

    # ---- 调用 ----
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
    ) -> Optional[List[Tuple[int, float]]]:
        """对 `documents` 按 `query` 重排。

        返回**全部候选**的 `(下标, 相关度)`（按相关度降序；`top_n` 缺省 = 全量），
        由调用方自行截断——这样"重排"与"截断多少条"是两个独立决策，不互相绑死。
        任何失败都返回 None（调用方保留原序）。
        """
        docs = [(d or "")[: self.max_chars] for d in documents]
        if not docs:
            return []
        if len(docs) == 1:
            return [(0, 0.0)]      # 单条无需打分（也省一次网络调用）

        n = len(docs) if not top_n or top_n <= 0 else min(int(top_n), len(docs))
        q = (query or "")[: self.max_chars]
        payload = {"model": self.model, "query": q, "documents": docs, "top_n": n}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        bucket = _bucket()
        last_err: Optional[str] = None

        for attempt in range(1, self.max_attempts + 1):
            if bucket is not None:
                bucket.acquire(len(q) + sum(len(d) for d in docs))
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=self.timeout
                )
            except Exception as exc:                       # 网络层异常同样重试
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts:
                    time.sleep(_BACKOFF * attempt)
                continue

            if resp.status_code == 200:
                if bucket is not None:
                    bucket.note_success()
                parsed = self._parse(resp)
                if parsed is not None:
                    return parsed
                last_err = "响应结构不可解析"
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code == 429 and bucket is not None:
                    bucket.note_429()
                if resp.status_code in _RETRY_STATUS and attempt < self.max_attempts:
                    time.sleep(_BACKOFF * attempt)
                    continue
                break

        logger.warning("rerank 调用失败（保留原序）: %s", last_err)
        return None

    @staticmethod
    def _parse(resp) -> Optional[List[Tuple[int, float]]]:
        try:
            data = resp.json()
            results = data.get("results")
            if not isinstance(results, list):
                return None
            pairs: List[Tuple[int, float]] = []
            for item in results:
                idx = int(item["index"])
                score = float(item["relevance_score"])
                pairs.append((idx, score))
        except Exception:
            return None
        pairs.sort(key=lambda x: -x[1])
        return pairs
