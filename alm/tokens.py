# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 侧 LLM token 计量。

目的：把一次压测 / 一次评测消耗的 token 记下来，便于按调用量与单价估算成本。
计量器挂在 ``ALMEngine`` 的共享 LLM 实例上（Add / Search 共用同一个实例），
因此覆盖全部 LLM 调用点，无需在各调用处埋点。

- 每次调用打一行 INFO（logger ``alm.tokens``），含本次与累计用量；
- 进程退出时由 ``ALMEngine.close()`` 打一份汇总，含按单价折算的估算成本。

**按侧归因（role）**：全部调用点共用同一个 ``ChatOpenAI`` 实例，故靠 LangChain 的
运行标签区分"这一次是写侧的抽取、还是读侧的叙事"。调用点用 ``with_role()`` 打标签，
计量器在 ``on_llm_end`` 的 ``kwargs["tags"]`` 里读（已实测该版本会传，见
``tests/test_tokens_role.py``）。**新增调用点只要包一层 ``with_role`` 就被覆盖**，
不会像"在每个调用点手工累加"那样漏计。

**缓存命中**：deepseek 用 ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``，
OpenAI 用 ``prompt_tokens_details.cached_tokens``。命中价与未命中价相差极大
（deepseek 折后 1/50、gpt-4o-mini 1/2），**不分开记就无法核对缓存策略的收益**。

单价从环境变量读取（默认币种 USD / 每百万 token），默认取 gpt-4o-mini 官方价。
**该默认值只是基准价**：工业榜单不限制 Add / Search 使用的模型，本系统实际使用
deepseek-v4-flash，故须按供应商价目表覆盖，否则折算出的成本会失真：

    ALM_PRICE_INPUT_PER_M   默认 0.15   未命中输入
    ALM_PRICE_CACHE_PER_M   默认 = 上面的值（即不区分命中）  命中输入
    ALM_PRICE_OUTPUT_PER_M  默认 0.60
    ALM_PRICE_CURRENCY      默认 USD（仅用于打印，不做换算）
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler

# 角色标签的定义在 `dba_pipeline.llm.roles`（那里是中立层：alm 依赖 dba_pipeline，
# 反向导入会成环）。这里重新导出，调用方可以只 import 一个模块。
from dba_pipeline.llm.roles import ROLE_TAG_PREFIX, role_of, with_role  # noqa: F401

logger = logging.getLogger("alm.tokens")


def _pct(part: int, whole: int) -> str:
    return "-" if not whole else "%.0f%%" % (100.0 * part / whole)


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


def extract_finish_reason(response: Any) -> Optional[str]:
    """从 LLMResult 里取 ``finish_reason``（取不到给 None）。

    为什么要在计量器里取：0927 线上实测有一次写入侧抽取 `in=4533 out=32768`——**撞满
    输出上限**。而 `finish_reason` 当时只在 ``story_rank`` 里记录，所以这条完全是隐形的，
    只能靠"out 恰好等于上限"反推。挂在计量器上则写入/检索两侧一并覆盖。
    LangChain 不同版本放在 ``generation_info``、``message.response_metadata`` 或
    ``llm_output`` 三处，逐个尝试。
    """
    for batch in getattr(response, "generations", None) or []:
        for generation in batch:
            info = getattr(generation, "generation_info", None) or {}
            if info.get("finish_reason"):
                return str(info["finish_reason"])
            metadata = getattr(getattr(generation, "message", None), "response_metadata", None) or {}
            if metadata.get("finish_reason"):
                return str(metadata["finish_reason"])
    llm_output = getattr(response, "llm_output", None) or {}
    finish = llm_output.get("finish_reason")
    return str(finish) if finish else None


def extract_usage(response: Any) -> Optional[Dict[str, int]]:
    """从 LLMResult 中取出用量；供应商未返回用量时给 None。

    返回 ``{input, output, total, cache_hit, cache_miss}``。

    LangChain 不同版本把用量放在两处：``llm_output["token_usage"]``（旧）
    或 ``generations[].message.usage_metadata``（新），两者都尝试。

    缓存命中字段各供应商叫法不同，**都尝试**（理由见模块 docstring）：
      · deepseek：``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
      · OpenAI  ：``prompt_tokens_details.cached_tokens``（新）
                  ``input_tokens_details.cached_tokens``（Responses API 形态）
    取不到时 hit=0、miss=全部输入 —— 等价于"按未命中计价"，是**偏保守**的降级
    （宁可高估成本，也不要低估）。
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

    cache_hit = _pick_int(usage, "prompt_cache_hit_tokens")
    if cache_hit is None:
        for key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(key)
            if isinstance(details, dict):
                cache_hit = _pick_int(details, "cached_tokens")
                if cache_hit is not None:
                    break
    if cache_hit is None:
        cache_hit = 0
    # 命中不可能超过输入；供应商偶尔给不自洽的数，夹一下避免负的 miss
    cache_hit = max(0, min(cache_hit, prompt))
    cache_miss = _pick_int(usage, "prompt_cache_miss_tokens")
    if cache_miss is None:
        cache_miss = prompt - cache_hit
    cache_miss = max(0, cache_miss)

    return {
        "input": prompt,
        "output": completion,
        "total": total,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
    }


class TokenMeter(BaseCallbackHandler):
    """累计 LLM token 用量。

    Add 在线程池中并发执行，故计数加锁；回调本身不做任何重活，不影响主链路。
    """

    def __init__(
        self,
        price_input: float,
        price_output: float,
        price_cache: Optional[float] = None,
        currency: str = "USD",
        verbose: bool = True,
    ):
        super().__init__()
        self.price_input = price_input
        self.price_output = price_output
        # 命中价缺省 = 未命中价（即"不区分命中"，与改造前口径逐字一致）
        self.price_cache = price_input if price_cache is None else price_cache
        self.currency = currency
        self.verbose = verbose
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0
        self.unmetered_calls = 0
        # 撞输出上限被截断的调用数（finish_reason=length）。这是"模型跑飞"的唯一
        # 直接信号：截断后的输出往往是畸形 JSON，而 out 恰好等于上限只是间接指认。
        self.truncated_calls = 0
        # 按角色分桶：{role: {calls,input,output,cache_hit,cache_miss,truncated,unmetered}}
        self.by_role: Dict[str, Dict[str, int]] = {}

    @classmethod
    def from_env(cls) -> "TokenMeter":
        cache_raw = os.environ.get("ALM_PRICE_CACHE_PER_M")
        try:
            price_cache = float(cache_raw) if cache_raw else None
        except ValueError:
            price_cache = None
        return cls(
            price_input=_env_float("ALM_PRICE_INPUT_PER_M", 0.15),
            price_output=_env_float("ALM_PRICE_OUTPUT_PER_M", 0.60),
            price_cache=price_cache,
            currency=os.environ.get("ALM_PRICE_CURRENCY") or "USD",
            verbose=_env_float("ALM_TOKEN_VERBOSE", 1.0) > 0,
        )

    def _bucket(self, role: str) -> Dict[str, int]:
        return self.by_role.setdefault(role, {
            "calls": 0, "input": 0, "output": 0,
            "cache_hit": 0, "cache_miss": 0, "truncated": 0, "unmetered": 0,
        })

    def on_llm_end(self, response, run_id=None, parent_run_id=None, **kwargs) -> None:  # noqa: D102
        # 角色来自 `with_role()` 打的 tag；LangChain 把运行标签原样放进回调 kwargs
        # （已实测，见 tests/test_tokens_role.py），拿不到时归 "unknown" 而不是丢弃。
        role = role_of(kwargs.get("tags"))
        usage = extract_usage(response)
        finish = extract_finish_reason(response)
        truncated = finish == "length"
        with self._lock:
            self.calls += 1
            bucket = self._bucket(role)
            bucket["calls"] += 1
            if truncated:
                self.truncated_calls += 1
                bucket["truncated"] += 1
            if usage is None:
                self.unmetered_calls += 1
                bucket["unmetered"] += 1
                return
            self.input_tokens += usage["input"]
            self.output_tokens += usage["output"]
            self.cache_hit_tokens += usage["cache_hit"]
            self.cache_miss_tokens += usage["cache_miss"]
            for key in ("input", "output", "cache_hit", "cache_miss"):
                bucket[key] += usage[key]
            calls = self.calls
            total_in, total_out = self.input_tokens, self.output_tokens
            truncated_total = self.truncated_calls
            hit, miss = self.cache_hit_tokens, self.cache_miss_tokens
        if self.verbose:
            logger.info(
                "LLM #%d [%s]: in=%d(hit=%d miss=%d) out=%d finish=%s | "
                "累计 in=%d out=%d 命中=%s 截断=%d",
                calls, role, usage["input"], usage["cache_hit"], usage["cache_miss"],
                usage["output"], finish or "?", total_in, total_out,
                _pct(hit, hit + miss), truncated_total,
            )

    def cost_of(self, bucket: Dict[str, int]) -> float:
        """按单价折算一个桶的成本。

        输入侧**分开算命中与未命中**：deepseek 命中价是未命中的 1/50，混在一起算会
        把缓存策略的收益完全抹掉（这正是改造前"算不出钱花在哪"的原因）。
        """
        return (
            bucket.get("cache_miss", 0) / 1e6 * self.price_input
            + bucket.get("cache_hit", 0) / 1e6 * self.price_cache
            + bucket.get("output", 0) / 1e6 * self.price_output
        )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            calls = self.calls
            total_in = self.input_tokens
            total_out = self.output_tokens
            unmetered = self.unmetered_calls
            truncated = self.truncated_calls
            hit, miss = self.cache_hit_tokens, self.cache_miss_tokens
            by_role = {role: dict(b) for role, b in self.by_role.items()}
        overall = {"input": total_in, "output": total_out,
                   "cache_hit": hit, "cache_miss": miss}
        return {
            "calls": calls,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "cache_hit_rate": (hit / (hit + miss)) if (hit + miss) else None,
            "unmetered_calls": unmetered,
            "truncated_calls": truncated,
            "price_input_per_m": self.price_input,
            "price_cache_per_m": self.price_cache,
            "price_output_per_m": self.price_output,
            "currency": self.currency,
            # 保留 6 位：单次调用常在 1e-5 量级，四舍五入到 4 位会把小额直接抹成 0，
            # 而这正是"按 role 分侧后每侧还剩多少"要看的数。
            "estimated_cost": round(self.cost_of(overall), 6),
            "by_role": {
                role: {**b, "cost": round(self.cost_of(b), 6)}
                for role, b in by_role.items()
            },
        }

    def report(self) -> str:
        """人类可读的汇总（进程退出时打印到 stderr）"""
        data = self.summary()
        calls = str(data["calls"])
        if data["unmetered_calls"]:
            calls += f"（其中 {data['unmetered_calls']} 次未返回用量）"
        truncated = f"{data['truncated_calls']} 次"
        if data["truncated_calls"]:
            # 截断＝模型跑飞（常见于抽取侧），这类输出多为畸形 JSON，须显著提示
            truncated += "  ← finish_reason=length，输出撞上限，须检查是否在跑飞"
        hit_rate = data["cache_hit_rate"]
        lines = [
            "=" * 60,
            "[ALM] LLM token 用量",
            "=" * 60,
            f"  调用次数 : {calls}",
            f"  被截断   : {truncated}",
            f"  input    : {data['input_tokens']:,} tokens"
            f"（命中 {data['cache_hit_tokens']:,} / 未命中 {data['cache_miss_tokens']:,}"
            f"，命中率 {'-' if hit_rate is None else '%.0f%%' % (100 * hit_rate)}）",
            f"  output   : {data['output_tokens']:,} tokens",
            f"  合计     : {data['total_tokens']:,} tokens",
            f"  估算成本 : ≈ {data['currency']} {data['estimated_cost']}"
            f"（每百万 未命中 {data['price_input_per_m']} / 命中 {data['price_cache_per_m']}"
            f" / 输出 {data['price_output_per_m']}）",
        ]
        if data["by_role"]:
            # 按侧归因：这是回答"钱花在写链还是读链"的唯一依据
            lines.append("-" * 60)
            lines.append("  按角色（role）:")
            lines.append("    %-10s %7s %12s %12s %8s %14s" %
                         ("role", "调用", "input", "命中率", "output", "成本"))
            for role in sorted(data["by_role"],
                               key=lambda r: -data["by_role"][r]["cost"]):
                b = data["by_role"][role]
                lines.append("    %-10s %7d %12d %12s %8d %14.6f" % (
                    role, b["calls"], b["input"],
                    _pct(b["cache_hit"], b["cache_hit"] + b["cache_miss"]),
                    b["output"], b["cost"],
                ))
        lines.append("=" * 60)
        return "\n".join(lines)
