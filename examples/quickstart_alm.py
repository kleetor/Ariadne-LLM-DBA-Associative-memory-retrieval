# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 同步 HTTP 服务快速开始：Health -> Add -> Search。

前置条件：
    - ALM 服务已启动（默认 http://127.0.0.1:8770）
    - 若服务开启了鉴权，需要提供 API Key

用法：
    python examples/quickstart_alm.py
    python examples/quickstart_alm.py --base-url https://alm.example.com --api-key sk-xxx
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 快速开始")
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--api-key", default=None, help="X-Api-Key；服务未开鉴权时留空")
    parser.add_argument("--user-id", default="quickstart:user")
    parser.add_argument("--session-id", default="quickstart:session")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-Api-Key"] = args.api_key

    # 1) 健康检查
    resp = requests.get(f"{base}/health", timeout=10)
    resp.raise_for_status()
    print(f"[OK] /health -> {resp.status_code}")

    # 2) 写入一段记忆
    now_ms = int(time.time() * 1000)
    add_body = {
        "request_id": "quickstart:req-1",
        "user_id": args.user_id,
        "session_id": args.session_id,
        "messages": [
            {"role": "user", "timestamp": now_ms, "content": "我叫小林，在杭州做后端开发。"},
            {"role": "assistant", "timestamp": now_ms + 1000, "content": "了解，主要用什么技术栈？"},
            {"role": "user", "timestamp": now_ms + 2000, "content": "主要是 Go 和 PostgreSQL。"},
        ],
    }
    resp = requests.post(f"{base}/add", json=add_body, headers=headers, timeout=args.timeout)
    resp.raise_for_status()
    print("[OK] /add ->", json.dumps(resp.json(), ensure_ascii=False))

    # 3) 检索记忆
    search_body = {"query": "小林在哪里工作？做什么？", "user_id": args.user_id, "top_k": 5}
    resp = requests.post(f"{base}/search", json=search_body, headers=headers, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    print(f"[OK] /search -> {len(data)} 条")
    for item in data:
        print(f"  - [{item.get('id')}] {item.get('content', '')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
