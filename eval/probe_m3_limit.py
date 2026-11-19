# -*- coding: utf-8 -*-
"""实测 SiliconFlow / BAAI-bge-m3 的真实单条字符上限。

背景：`store._MAX_CHARS = 500` 是 bge-large-zh-v1.5（512 token）时代的标定。
改用 bge-m3（最大序列 8192 token）后，500 字符过于保守——但**不能凭模型规格推断**，
供应商网关可能有自己的限制，必须实测。

方法：直接打 API（绕过 store 的截断逻辑），二分/递进测英文与中文的 400 边界。
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KEY = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE = (os.environ.get("EMBEDDING_API_BASE") or os.environ.get("OPENAI_API_BASE") or "").rstrip("/")
URL = f"{BASE}/embeddings"

EN = ("The user is a backend developer working at an internet company in Hangzhou. "
      "They mainly use Go and PostgreSQL, and recently started studying distributed "
      "storage systems. Their neck often hurts because of long working hours. ")
ZH = ("用户是杭州一家互联网公司的后端开发工程师，主要使用 Go 与 PostgreSQL，"
      "最近开始学习分布式存储系统。由于长期加班，颈椎经常酸痛，"
      "需要定期做拉伸和理疗。他喜欢在周末徒步，也养了两只猫。")

print(f"端点 {URL}\n模型 BAAI/bge-m3\n")
print(f"{'目标字符':>9}{'英文实测':>12}{'英文实际':>10}{'中文实测':>12}{'中文实际':>10}")
print("-" * 56)


def probe(text: str, want: int, lang: str):
    """把文本截断/拼接到目标长度后打一次 API，返回 (状态, 实际长度)"""
    if lang == "en":
        s = (text * (want // len(text) + 2))[:want]
    else:
        s = (text * (want // len(text) + 2))[:want]
    try:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": "BAAI/bge-m3", "input": s, "encoding_format": "float"},
            timeout=90,
        )
    except Exception as exc:
        return f"EXC:{type(exc).__name__}", len(s)
    if r.status_code == 200:
        try:
            dim = len(r.json()["data"][0]["embedding"])
            return f"200(dim={dim})", len(s)
        except Exception:
            return "200(bad)", len(s)
    body = r.text[:120].replace("\n", " ")
    return f"{r.status_code} {body[:60]}", len(s)


for want in (500, 1000, 2000, 4000, 8000, 16000):
    a, la = probe(EN, want, "en")
    b, lb = probe(ZH, want, "zh")
    print(f"{want:>9}{a:>12}{la:>10}{b:>12}{lb:>10}")

print()
print("参考：bge-m3 规格 8192 token。英文约 3.5 字符/token，中文约 1.5 字符/token。")
