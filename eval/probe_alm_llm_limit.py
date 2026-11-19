# -*- coding: utf-8 -*-
"""实测 ALM 侧 LLM 端点（deepseek-v4-flash）的真实输出 token 上限。

背景：0926 平台复验里，StoryRank 的 7 次 `JSON 解析失败` 与 7 次
`alm.tokens: out=8192` 一一对应——输出被 8192 截断，JSON 在 `adopted_ids`
数组中途断掉。但 `engine._build_llm()` **没有设 `max_tokens`**，只有两个可能：

  · 8192 是**端点默认值**   → 显式抬高即可（补丁）
  · 8192 是**模型硬上限**   → 抬高会 400，必须改输出契约（根治）

**不能凭模型规格推断**——模型名 `deepseek-v4-flash` 不在官方公开列表里，
且网关可能有自己的限制。故直接打 API 扫一遍，并记录 `finish_reason`：

  finish_reason=length → 撞上限被截断（这正是生产里那 7 次的形态）
  finish_reason=stop   → 正常自然结束
  HTTP 400             → 该 max_tokens 不被接受（硬上限）

用法：python eval/probe_alm_llm_limit.py
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# ALM 的 LLM 凭据在 .env.alm（与根 .env 分离）；两个都读，.env.alm 优先
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.alm", override=True)

MODEL = os.environ.get("OPENAI_MODEL") or "deepseek-v4-flash"
BASE = (os.environ.get("OPENAI_API_BASE") or "").rstrip("/")
KEY = os.environ.get("OPENAI_API_KEY") or ""
URL = f"{BASE}/chat/completions"

# 强制长输出的提示：让它必须一直写，从而真正撞到上限
PROMPT = ("请从 1 开始连续编号，逐行输出。每行格式严格为："
          "「第N行：这是一段用于测试输出长度上限的填充文字，请勿省略。」"
          "不要总结、不要停、不要解释，尽可能多地输出，直到无法继续为止。")

print(f"端点 {URL}\n模型 {MODEL}\nkey  {'已加载' if KEY else '缺失'}\n")
if not BASE or not KEY:
    sys.exit("缺少 OPENAI_API_BASE / OPENAI_API_KEY（检查 .env.alm）")

# 含"不传"一档，用来判断 8192 是默认值还是硬上限
SWEEP = [None, 4096, 8192, 16384, 32768, 65536]

print(f"{'max_tokens':>11}{'HTTP':>6}{'finish':>9}{'out_tok':>9}{'正文字符':>10}  备注")
print("-" * 74)

results = []
for mt in SWEEP:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
    }
    if mt is not None:
        body["max_tokens"] = mt
    try:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=300,
        )
    except Exception as exc:
        print(f"{str(mt):>11}{'-':>6}{'-':>9}{'-':>9}{'-':>10}  EXC:{type(exc).__name__}")
        results.append((mt, None, None, None, f"EXC:{type(exc).__name__}"))
        continue

    if r.status_code != 200:
        note = r.text[:100].replace("\n", " ")
        print(f"{str(mt):>11}{r.status_code:>6}{'-':>9}{'-':>9}{'-':>10}  {note}")
        results.append((mt, r.status_code, None, None, note))
        continue

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    finish = choice.get("finish_reason", "?")
    content = (choice.get("message") or {}).get("content") or ""
    out_tok = (data.get("usage") or {}).get("completion_tokens")
    note = ""
    if finish == "length":
        note = "撞上限被截断"
    elif finish == "stop":
        note = "自然结束"
    print(f"{str(mt):>11}{200:>6}{finish:>9}{str(out_tok):>9}{len(content):>10}  {note}")
    results.append((mt, 200, finish, out_tok, note))

# ---- 第二轮：复现生产请求形态（thinking disabled）----
# 生产里 out 恒为 8192，而上面"不传 max_tokens"却到了 15016 → 差异只能来自
# `extra_body={"thinking": {"type": "disabled"}}`（ALM_LLM_THINKING=0）。
# 故把这一维单独拉出来对照，并记录 reasoning_tokens 以确认 token 构成。
print("\n" + "=" * 74)
print("第二轮：复现生产请求形态（thinking disabled，engine._build_llm 的实际形状）")
print("=" * 74)
print(f"{'thinking':>9}{'max_tokens':>11}{'HTTP':>6}{'finish':>9}{'out_tok':>9}"
      f"{'reason':>8}{'正文字符':>10}")
print("-" * 74)

ARMS2 = [
    ("off", None), ("off", 8192), ("off", 32768), ("off", 65536),
    ("on", 32768), ("on", None),
]
for th, mt in ARMS2:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
    }
    if th == "off":
        body["thinking"] = {"type": "disabled"}
    if mt is not None:
        body["max_tokens"] = mt
    try:
        r = requests.post(
            URL,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json=body, timeout=300,
        )
    except Exception as exc:
        print(f"{th:>9}{str(mt):>11}{'-':>6}{'-':>9}{'-':>9}{'-':>8}{'-':>10}  EXC:{type(exc).__name__}")
        continue
    if r.status_code != 200:
        print(f"{th:>9}{str(mt):>11}{r.status_code:>6}{'-':>9}{'-':>9}{'-':>8}{'-':>10}"
              f"  {r.text[:70]}")
        continue
    data = r.json()
    ch = (data.get("choices") or [{}])[0]
    u = data.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    reasoning = det.get("reasoning_tokens")
    print(f"{th:>9}{str(mt):>11}{200:>6}{ch.get('finish_reason','?'):>9}"
          f"{str(u.get('completion_tokens')):>9}{str(reasoning):>8}"
          f"{len((ch.get('message') or {}).get('content') or ''):>10}")

print()
ok = [x for x in results if x[1] == 200]
raised = [x for x in ok if x[0] and x[0] > 8192 and x[2] == "length"]
trunc_at = sorted({x[3] for x in ok if x[3]})
print("判读：")
if any(x[1] == 400 for x in results if x[0] and x[0] > 8192):
    print("  · 存在 >8192 档位返回 400 → **8192 是模型/网关硬上限**，抬高不可行，"
          "只能改输出契约（拆 adopted_ids + story 长度预算）。")
elif raised:
    print(f"  · >8192 档位能正常返回且 finish=length → **上限可抬**；"
          f"观测到的 out_tok 上界 {max(trunc_at)}。可显式设 max_tokens 作为止血。")
else:
    print(f"  · 各档位均未触顶（out_tok={trunc_at}）→ 模型会自然结束，"
          f"8192 更可能是**默认值**；显式抬高即可。")
print(f"  · 实测 completion_tokens 取值集合: {trunc_at}")
