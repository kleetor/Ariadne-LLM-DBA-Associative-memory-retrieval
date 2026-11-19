# -*- coding: utf-8 -*-
"""临时：用 Mem0 的实际抽取 prompt 调用 api.gpt.ge，定位 content=None 原因"""
import os
import sys
from pathlib import Path

ROOT = Path(r"c:\Users\makot\Desktop\Ariadne")
sys.path.insert(0, str(ROOT))
os.environ["MEM0_TELEMETRY"] = "false"
os.environ["MEM0_DIR"] = str(ROOT / "eval" / ".cache" / "mem0_home")
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT, generate_additive_extraction_prompt
from openai import OpenAI

c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
model = os.environ["OPENAI_MODEL"]

user_prompt = generate_additive_extraction_prompt(
    existing_memories=[],
    new_messages="user: The user often works overtime until 11 p.m.\n",
    last_k_messages=[],
    custom_instructions=None,
)
print("user_prompt len:", len(user_prompt))
try:
    r = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ADDITIVE_EXTRACTION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    msg = r.choices[0].message
    print("content:", str(msg.content)[:150])
    print("reasoning:", str(getattr(msg, "reasoning_content", None))[:80])
    print("finish:", r.choices[0].finish_reason)
    print("usage:", r.usage)
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:300])
