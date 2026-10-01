# SPDX-License-Identifier: AGPL-3.0-only

"""按侧归因（role）与缓存命中的契约测试。

两件事必须有测试钉住：

1. **标签能不能从调用点传到计量器**。整条 `with_role(...)` 的设计建立在一个实现
   细节上——LangChain 把运行标签放进 `on_llm_end` 的 `kwargs["tags"]`。该细节一旦
   随版本变化，归因会**静默退化成全部计入 unknown**，而 `unknown` 桶看起来"也有数据"，
   很难被发现。故这里做一次端到端断言。
2. **命中/未命中要分开计价**。deepseek 命中价是未命中的 1/50，混算等于把缓存策略的
   收益抹平——那正是"算不出钱花在哪"的根因。
"""

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from alm.tokens import TokenMeter, extract_usage, role_of, with_role


# --------------------------------------------------------------------------
# 1. 标签传递（端到端）
# --------------------------------------------------------------------------

def test_tags_reach_meter_on_llm_end():
    """`with_role` 的标签必须出现在计量器的 role 分桶里，而不是落进 unknown"""
    meter = TokenMeter(price_input=0.15, price_output=0.60, verbose=False)
    llm = FakeListChatModel(responses=["ok"])

    ctx = {"callbacks": [meter]}
    with_role(llm, "extract").invoke("a", config=ctx)
    with_role(llm, "extract").invoke("b", config=ctx)
    with_role(llm, "story").invoke("c", config=ctx)
    llm.invoke("d", config=ctx)  # 没打标签

    assert meter.by_role["extract"]["calls"] == 2
    assert meter.by_role["story"]["calls"] == 1
    assert meter.by_role["unknown"]["calls"] == 1


def test_role_of_picks_only_role_tag():
    assert role_of(["foo", "role:story", "bar"]) == "story"
    assert role_of(["foo"]) == "unknown"
    assert role_of(None) == "unknown"
    assert role_of([]) == "unknown"


# --------------------------------------------------------------------------
# 2. 缓存命中字段的解析（三种供应商形态 + 缺失降级）
# --------------------------------------------------------------------------

def _result(usage):
    return LLMResult(
        generations=[[Generation(text="x")]],
        llm_output={"token_usage": usage},
    )


def test_deepseek_style_cache_fields():
    u = extract_usage(_result({
        "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
        "prompt_cache_hit_tokens": 640, "prompt_cache_miss_tokens": 360,
    }))
    assert (u["cache_hit"], u["cache_miss"], u["input"]) == (640, 360, 1000)


def test_openai_style_cached_tokens_details():
    u = extract_usage(_result({
        "prompt_tokens": 1000, "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 900},
    }))
    assert u["cache_hit"] == 900
    assert u["cache_miss"] == 100


def test_responses_api_style_details():
    u = extract_usage(_result({
        "input_tokens": 500, "output_tokens": 50,
        "input_tokens_details": {"cached_tokens": 125},
    }))
    assert u["cache_hit"] == 125
    assert u["cache_miss"] == 375


def test_missing_cache_fields_degrade_conservatively():
    """拿不到命中字段时按"全部未命中"计——宁可高估成本，不要低估"""
    u = extract_usage(_result({"prompt_tokens": 800, "completion_tokens": 20}))
    assert u["cache_hit"] == 0
    assert u["cache_miss"] == 800


def test_inconsistent_cache_numbers_are_clamped():
    """供应商偶尔给不自洽的数（命中 > 输入），必须夹住，否则 miss 会变负"""
    u = extract_usage(_result({
        "prompt_tokens": 100, "completion_tokens": 1,
        "prompt_cache_hit_tokens": 500,
    }))
    assert u["cache_hit"] == 100
    assert u["cache_miss"] == 0


# --------------------------------------------------------------------------
# 3. 计价与汇总
# --------------------------------------------------------------------------

def test_cost_separates_cache_hit_and_miss():
    meter = TokenMeter(price_input=1.0, price_output=4.0, price_cache=0.02, verbose=False)
    meter.on_llm_end(_result({
        "prompt_tokens": 1_000_000, "completion_tokens": 0,
        "prompt_cache_hit_tokens": 900_000, "prompt_cache_miss_tokens": 100_000,
    }))
    # 100k 未命中 × 1/M + 900k 命中 × 0.02/M = 0.1 + 0.018
    assert meter.summary()["estimated_cost"] == pytest.approx(0.118, abs=1e-4)


def test_price_cache_defaults_to_input_price():
    """不给命中价 = 不区分命中（与改造前口径逐字一致）"""
    meter = TokenMeter(price_input=0.15, price_output=0.60, verbose=False)
    meter.on_llm_end(_result({
        "prompt_tokens": 1_000_000, "completion_tokens": 0,
        "prompt_cache_hit_tokens": 1_000_000,
    }))
    assert meter.summary()["price_cache_per_m"] == 0.15
    assert meter.summary()["estimated_cost"] == pytest.approx(0.15, abs=1e-4)


def test_summary_reports_hit_rate_and_by_role():
    meter = TokenMeter(price_input=1.0, price_output=1.0, verbose=False)
    meter.on_llm_end(_result({
        "prompt_tokens": 100, "completion_tokens": 10,
        "prompt_cache_hit_tokens": 25, "prompt_cache_miss_tokens": 75,
    }))
    s = meter.summary()
    assert s["cache_hit_rate"] == pytest.approx(0.25)
    assert "unknown" in s["by_role"]
    assert s["by_role"]["unknown"]["cost"] == pytest.approx(100 / 1e6 + 10 / 1e6)


def test_report_prints_role_table_and_currency():
    meter = TokenMeter(price_input=0.15, price_output=0.60, currency="CNY", verbose=False)
    meter.on_llm_end(_result({"prompt_tokens": 10, "completion_tokens": 2}))
    text = meter.report()
    assert "按角色" in text
    assert "unknown" in text
    assert "CNY" in text


def test_meter_is_a_callback_handler():
    """计量器要挂进 ChatOpenAI(callbacks=...)，必须仍是回调处理器"""
    assert isinstance(TokenMeter(0.15, 0.60, verbose=False), BaseCallbackHandler)
