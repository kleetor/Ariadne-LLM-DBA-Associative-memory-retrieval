# -*- coding: utf-8 -*-
"""临时：探测 DeepSeek(api.gpt.ge) 对 Mem0 抽取 prompt 的响应形态"""
import os
import sys
from pathlib import Path

ROOT = Path(r"c:\Users\makot\Desktop\Ariadne")
sys.path.insert(0, str(ROOT))
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

from openai import OpenAI

c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
model = os.environ["OPENAI_MODEL"]

for rf in [None, {"type": "json_object"}]:
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": 'Say {"memory": []} only.'},
            ],
            temperature=0,
            response_format=rf,
        )
        msg = r.choices[0].message
        print(f"rf={rf}: content={str(msg.content)[:120]!r} reasoning={str(getattr(msg,'reasoning_content',None))[:60]!r} finish={r.choices[0].finish_reason}")
    except Exception as e:
        print(f"rf={rf}: FAIL {type(e).__name__} {str(e)[:200]}")
