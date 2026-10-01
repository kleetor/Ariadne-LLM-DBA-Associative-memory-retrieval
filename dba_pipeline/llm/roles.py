# SPDX-License-Identifier: AGPL-3.0-only

"""LLM 调用的**角色标签**：给每个调用点一个稳定名字，供计量器按侧归因。

为什么需要：全部调用点共用同一个 ``ChatOpenAI`` 实例（见 ``alm/engine.py``），
所以"这次调用属于写链还是读链"没有任何结构性区分——账单只能给一个总数。
2026-09-30 的 Full 首跑正是卡在这一点上：算得出"44 次调用/题"，算不出
"写侧占多少、读侧占多少"，因而无法判断该先优化哪一侧。

为什么标签走 LangChain 的 ``tags`` 而不是自建累加：

- ``tags`` 是**逐运行**（per-run）传递的，不依赖全局状态，多角色并发不会串；
- 计量器在 ``on_llm_end`` 的 ``kwargs["tags"]`` 里直接拿到（已实测，
  见 ``tests/test_tokens_role.py``）；
- **新增调用点只要包一层 :func:`with_role` 就被覆盖**，不像在每个调用点手工
  累加那样容易漏计。

本模块刻意放在 ``dba_pipeline``（而不是 ``alm``）：``alm`` 依赖 ``dba_pipeline``，
反向导入会形成循环。这里只依赖 ``langchain_core``，两侧都能安全使用。

**角色词表**（新增调用点请复用，不要临时造名字——计量汇总按名字分组，同义异名会让
统计被拆成两行）：

| role | 调用点 | 侧 |
|---|---|---|
| ``extract`` | 节点抽取（含 temporal / choice 变体） | 写 |
| ``link`` | 边连接（含 temporal 变体） | 写 |
| ``triage`` | 维护判断前置 | 写 |
| ``purpose`` | 目的推断 | 读 |
| ``status`` | 状态推断（ALM 链路当前无调用者） | 读 |
| ``story`` | StoryRank 叙事整理 | 读 |
| ``abstain`` | 弃权模糊带判定 | 读 |
| ``audit`` | 溯源审计（MCP 巡检） | 旁路 |
| ``answer`` | 生成回复（ALM 传 ``with_response=False``，不执行） | 旁路 |
"""

from typing import Optional, Sequence

# 前缀用 ``role:`` 而不是裸名字，是为了与其它 tag（如 langsmith 自带的）区分开。
ROLE_TAG_PREFIX = "role:"


def role_of(tags: Optional[Sequence[str]]) -> str:
    """从 LangChain 的运行标签里取角色名；没有则 ``"unknown"``"""
    for tag in tags or ():
        if isinstance(tag, str) and tag.startswith(ROLE_TAG_PREFIX):
            return tag[len(ROLE_TAG_PREFIX):]
    return "unknown"


def with_role(runnable, role: str):
    """给一个 Runnable 打角色标签。

    用法：``chain = PROMPT | with_role(llm, "extract")``。
    返回的是 ``RunnableBinding``，可以继续参与 ``|`` 组合。
    """
    return runnable.with_config(tags=[f"{ROLE_TAG_PREFIX}{role}"])
