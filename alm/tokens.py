# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 侧 LLM token 计量。

目的：把一次压测 / 一次评测消耗的 token 记下来，便于按调用量与单价估算成本。
计量器挂在 ``ALMEngine`` 的共享 LLM 实例上（Add / Search 共用同一个实例），
因此覆盖全部 LLM 调用点，无需在各调用处埋点。

- 每次调用打一行 INFO（logger ``alm.tokens``），含本次与累计用量；
- 进程退出时由 ``ALMEngine.close()`` 打一份汇总，含按单价折算的估算成本。

单价从环境变量读取（USD / 每百万 token），默认为 gpt-4o-mini 官方价——
正式评测规定 Add / Search 必须用 gpt-4o-mini，用该价目表可直接折算正式成本；
本地用其它模型（如 deepseek-v4-flash）时按供应商价目表覆盖：

    ALM_PRICE_INPUT_PER_M   默认 0.15
    ALM_PRICE_OUTPUT_PER_M  默认 0.60
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("alm.tokens")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _pick_int(usage: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def extract_usage(response: Any) -> Optional[Dict[str, int]]:
    """从 LLMResult 中取出 {input, output, total}；供应商未返回用量时给 None。

    LangChain 不同版本把用量放在两处：``llm_output["token_usage"]``（旧）
    或 ``generations[].message.usage_metadata``（新），两者都尝试。
    """
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage")
    if not isinstance(usage, dict) or not usage:
        usage = None
        for batch in getattr(response, "generations", None) or []:
            for generation in batch:
                metadata = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if isinstance(metadata, dict) and metadata:
                    usage = metadata
                    break
            if usage:
                break
    if not usage:
        return None

    prompt = _pick_int(usage, "prompt_tokens", "input_tokens")
    completion = _pick_int(usage, "completion_tokens", "output_tokens")
    if prompt is None and completion is None:
        return None
    prompt = prompt or 0
    completion = completion or 0
    total = _pick_int(usage, "total_tokens") or (prompt + completion)
    return {"input": prompt, "output": completion, "total": total}


class TokenMeter(BaseCallbackHandler):
    """累计 LLM token 用量。

    Add 在线程池中并发执行，故计数加锁；回调本身不做任何重活，不影响主链路。
    """

    def __init__(self, price_input: float, price_output: float, verbose: bool = True):
        super().__init__()
        self.price_input = price_input
        self.price_output = price_output
        self.verbose = verbose
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.unmetered_calls = 0

    @classmethod
    def from_env(cls) -> "TokenMeter":
        return cls(
            price_input=_env_float("ALM_PRICE_INPUT_PER_M", 0.15),
            price_output=_env_float("ALM_PRICE_OUTPUT_PER_M", 0.60),
            verbose=_env_float("ALM_TOKEN_VERBOSE", 1.0) > 0,
        )

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D102 - 回调协议
        usage = extract_usage(response)
        with self._lock:
            self.calls += 1
            if usage is None:
                self.unmetered_calls += 1
                return
            self.input_tokens += usage["input"]
            self.output_tokens += usage["output"]
            calls, total_in, total_out = self.calls, self.input_tokens, self.output_tokens
        if self.verbose:
            logger.info(
                "LLM #%d: in=%d out=%d | 累计 in=%d out=%d",
                calls, usage["input"], usage["output"], total_in, total_out,
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            calls = self.calls
            total_in = self.input_tokens
            total_out = self.output_tokens
            unmetered = self.unmetered_calls
        cost = total_in / 1e6 * self.price_input + total_out / 1e6 * self.price_output
        return {
            "calls": calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "unmetered_calls": unmetered,
            "price_input_per_m": self.price_input,
            "price_output_per_m": self.price_output,
            "estimated_cost_usd": round(cost, 4),
        }

    def report(self) -> str:
        """人类可读的汇总（进程退出时打印到 stderr）"""
        data = self.summary()
        calls = str(data["calls"])
        if data["unmetered_calls"]:
            calls += f"（其中 {data['unmetered_calls']} 次未返回用量）"
        lines = [
            "=" * 60,
            "[ALM] LLM token 用量",
            "=" * 60,
            f"  调用次数 : {calls}",
            f"  input    : {data['input_tokens']:,} tokens",
            f"  output   : {data['output_tokens']:,} tokens",
            f"  合计     : {data['total_tokens']:,} tokens",
            f"  估算成本 : ≈ ${data['estimated_cost_usd']}"
            f"（每百万 input ${data['price_input_per_m']} / output ${data['price_output_per_m']}）",
            "=" * 60,
        ]
        return "\n".join(lines)
