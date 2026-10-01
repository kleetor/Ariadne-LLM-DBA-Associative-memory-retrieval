# -*- coding: utf-8 -*-
"""LLM 供应商能力判定。

目前只有一项：请求体 ``thinking`` 参数是否受支持。

为什么要判定：deepseek 系端点**默认开启思考**（effort=high），关闭必须走请求体
``extra_body={"thinking": {"type": "disabled"}}``（见 Plan/alm/0921 报告实测）。但 OpenAI 兼容
端点收到未知参数会直接 400 —— 实测 api.gpt.ge 的 gpt-4o-mini 报
``Unrecognized request argument supplied: thinking``。

若不加判定地无条件注入该参数，一旦把链路指向非 deepseek 模型，**整条链路的每个 LLM 调用
都会 400**（Add / Search 全挂）。故统一由本模块给出"该不该带这个参数"。
"""

from typing import Any, Dict, Optional


def supports_thinking_param(model: Optional[str], base_url: Optional[str]) -> bool:
    """端点是否支持请求体 ``thinking`` 参数（以模型名或 base_url 是否含 deepseek 判定）。"""
    text = f"{model or ''} {base_url or ''}".lower()
    return "deepseek" in text


def thinking_kwargs(model: Optional[str], base_url: Optional[str],
                    enabled: bool) -> Dict[str, Any]:
    """返回应并入 ``ChatOpenAI(...)`` 的关键字参数。

    ``enabled=True``  —— 不干预，交回端点默认（deepseek 即思考开启 / effort=high）。
    ``enabled=False`` —— 注入关闭指令；**但仅在端点支持该参数时**，否则返回空 dict，
                          避免对 gpt-4o-mini 之类的模型触发 400。
    """
    if enabled or not supports_thinking_param(model, base_url):
        return {}
    return {"extra_body": {"thinking": {"type": "disabled"}}}
