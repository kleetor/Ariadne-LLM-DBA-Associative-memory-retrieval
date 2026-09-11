# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 契约本地自测客户端。

对运行中的 ALM 服务发一轮 Add → Search，校验请求 / 响应是否符合官方规范。

Usage:
    python -m alm.smoke --base-url http://127.0.0.1:8770
    python -m alm.smoke --base-url https://your-host --api-key sk-xxx
"""

import argparse
import json
import sys
import time

import requests


def _fail(message: str) -> int:
    print(f"[FAIL] {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 契约自测客户端")
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--api-key", default=None, help="X-Api-Key（服务未开鉴权时留空）")
    parser.add_argument("--user-id", default="eval:local:smoke:conv-0")
    parser.add_argument("--session-id", default="eval:local:smoke:sample:0")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时秒数")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-Api-Key"] = args.api_key

    # 1) health（免鉴权，任意 2xx）
    try:
        resp = requests.get(f"{base}/health", timeout=10)
    except Exception as exc:
        return _fail(f"health 请求失败: {exc}")
    if not (200 <= resp.status_code < 300):
        return _fail(f"health 返回 {resp.status_code}")
    print(f"[OK] /health -> {resp.status_code}")

    # 2) Add
    messages = [
        {"role": "user", "timestamp": 1704067200000,
         "content": "我叫小林，在杭州的一家互联网公司做后端开发。"},
        {"role": "assistant", "timestamp": 1704067260000,
         "content": "了解，后端开发主要用什么技术栈？"},
        {"role": "user", "timestamp": 1704067320000,
         "content": "主要是 Go 和 PostgreSQL，最近在啃分布式存储。"},
    ]
    add_body = {
        "request_id": "eval:local:smoke:conv-0:chunk-0",
        "messages": messages,
        "user_id": args.user_id,
        "session_id": args.session_id,
    }
    started = time.time()
    try:
        resp = requests.post(f"{base}/add", json=add_body, headers=headers, timeout=args.timeout)
    except Exception as exc:
        return _fail(f"add 请求失败: {exc}")
    if resp.status_code != 200:
        return _fail(f"add 返回 {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if data.get("success") is not True:
        return _fail(f"add.success 必须为布尔 true，实际: {data.get('success')!r}")
    for field in ("request_id", "user_id", "session_id"):
        if data.get(field) != add_body[field]:
            return _fail(f"add.{field} 未原样回显: {data.get(field)!r}")
    print(f"[OK] /add -> 200（{time.time() - started:.1f}s）")

    # 3) Search
    search_body = {"query": "小林在哪里工作？做什么？", "user_id": args.user_id, "top_k": 100}
    started = time.time()
    try:
        resp = requests.post(f"{base}/search", json=search_body, headers=headers, timeout=args.timeout)
    except Exception as exc:
        return _fail(f"search 请求失败: {exc}")
    if resp.status_code != 200:
        return _fail(f"search 返回 {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    if not isinstance(data.get("data"), list):
        return _fail(f"search.data 必须为顶层数组，实际: {type(data.get('data')).__name__}")
    items = data["data"]
    if len(items) > search_body["top_k"]:
        return _fail(f"search 返回 {len(items)} 条，超过 top_k={search_body['top_k']}")
    for idx, item in enumerate(items):
        if not item.get("id") or not item.get("content"):
            return _fail(f"data[{idx}] 缺少必填 id / content")

    print(f"[OK] /search -> 200（{time.time() - started:.1f}s，{len(items)} 条）")
    for item in items[:3]:
        print(f"    - [{item['id']}] score={item.get('score')} {item['content'][:60]}")

    if not items:
        print("[WARN] 检索结果为空，请确认 Add 的模型 / Embedding 配置是否生效", file=sys.stderr)
        return 1

    print(json.dumps({"add": "ok", "search": "ok", "returned": len(items)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
