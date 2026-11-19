# -*- coding: utf-8 -*-
"""Probe: 探测 api.gpt.ge deepseek-v4-flash-0731 关闭 thinking 的请求参数"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import AsyncOpenAI


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


PARAMS = {
    "no_extra": {},
    "thinking_disabled": {"extra_body": {"thinking": {"type": "disabled"}}},
    "reasoning_disabled": {"extra_body": {"reasoning": {"enabled": False}}},
    "enable_thinking_false": {"extra_body": {"enable_thinking": False}},
    "reasoning_none": {"extra_body": {"reasoning": None}},
}


async def main() -> None:
    load_dotenv(ROOT / ".env")
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_API_BASE", ""),
    )
    model = os.environ.get("OPENAI_MODEL", "deepseek-v4-flash")
    for name, extra in PARAMS.items():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "用 JSON 输出 1+1 等于几：{\"answer\": 2}"}],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
                **extra,
            )
            msg = resp.choices[0].message
            content = getattr(msg, "content", None) or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            print(f"[{name}] finish={resp.choices[0].finish_reason} "
                  f"content_len={len(content)} reasoning_len={len(reasoning)} "
                  f"content={content[:60]!r}")
        except Exception as e:
            print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    asyncio.run(main())
