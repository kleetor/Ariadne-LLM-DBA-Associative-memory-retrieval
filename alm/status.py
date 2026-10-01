# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 运行状况快速检查（只读，不写入任何记忆）。

与 ``alm/smoke.py`` 的分工：

- ``smoke`` ：完整契约自测，会真实 Add 一批消息（**写入数据**）
- ``status``：只读巡检，不写库，可高频跑（cron 定时探活 / 上线前人工确认）

四项检查：

1. ``GET /health`` 免鉴权可达（期望 2xx）
2. 无 Key 与错 Key 调 ``/search`` 必须 401（证明鉴权真的生效，且反代没吞掉请求头）
3. 带 Key 对**空 user_id** 调 ``/search`` 必须 200 且 ``data`` 为数组
4. ``/docs`` 与 ``/openapi.json`` 必须 404（公网不应暴露 API schema）

第 3 项是端到端硬信号，也是本脚本的核心价值：服务在走弃权判定**之前**，会先跑一次
目的推断（LLM，见 ``dba_pipeline/retrieval/retriever.py`` 的 ``infer_purpose``）与查询
向量计算（embedding，见 ``alm/space.py``），任一环失败都会抛成 500
（``alm/server.py`` 的 search handler 统一兜成 5xx）。因此「200 且 data 为数组」
即可证明 LLM 与 embedding 两条出网链路都通——**而不需要写入任何记忆**。

注意：空空间的响应通常只有 1–2 秒，比对有数据的 user 检索（4 秒上下）快，
因为后者多了一次叙事生成的 LLM 调用（那是链路上最贵的一次）。**快不代表空转**，
上面两处调用在空空间上同样执行。查询本身走弃权路径（候选为空 → 返回空数组），
所以输出里 data 通常为 0 条。若把这项误判为"没实际调用"，会让脚本失去告警意义。

Usage::

    python -m alm.status --base-url https://alm.ariadne.top --api-key <key>
    docker exec ariadne-alm python -m alm.status \\
        --base-url http://127.0.0.1:8770 --api-key <key>

退出码：0 全部通过；1 有任一项失败（用于 cron 告警）。
"""

import argparse
import sys
import time

import requests


def _fail(message: str) -> int:
    print(f"[FAIL] {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 运行状况快速检查（只读）")
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--api-key", default=None,
                        help="X-Api-Key；留空则跳过链路检查（无法验证 LLM / embedding 出网）")
    parser.add_argument("--timeout", type=int, default=60,
                        help="单项请求超时秒数；链路检查含 2 次上游 LLM/embedding 调用，别设太小")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    # 查询文本刻意保持中性：它只用于触发一次目的推断，不写入任何内容。
    probe_body = {"query": "健康检查", "user_id": "alm:status:probe", "top_k": 5}

    # ---- 1/4 可达性与鉴权豁免 ----
    started = time.time()
    try:
        resp = requests.get(f"{base}/health", timeout=args.timeout)
    except Exception as exc:
        return _fail(f"服务不可达：{base}/health 请求异常 {type(exc).__name__}: {exc}")
    elapsed = time.time() - started
    if not 200 <= resp.status_code < 300:
        return _fail(f"服务不可达：/health 返回 {resp.status_code}（期望 2xx）")
    print(f"[OK]   1/4 可达性   /health 免鉴权 -> {resp.status_code}（{elapsed:.2f}s）")

    # ---- 2/4 鉴权拒绝：无 Key 与错 Key 都必须是 401 ----
    # 这一项同时验证了「鉴权已开启」与「反代透传了请求头」：
    # 若 Key 在网络层被丢弃，错 Key 也会返回 200，而本项会失败。
    for label, headers in (("无 Key", {}), ("错 Key", {"X-Api-Key": "wrong-key-alm-status"})):
        try:
            resp = requests.post(f"{base}/search", json=probe_body, headers=headers,
                                 timeout=args.timeout)
        except Exception as exc:
            return _fail(f"鉴权检查（{label}）请求异常：{type(exc).__name__}: {exc}")
        if resp.status_code != 401:
            return _fail(
                f"鉴权检查（{label}）期望 401，实际 {resp.status_code}。"
                "若为 200，说明服务未配置 ALM_API_KEYS（公网部署下等于记忆库完全开放）"
            )
    print("[OK]   2/4 鉴权     无 Key / 错 Key -> 401")

    # ---- 3/4 端到端链路（LLM + embedding）----
    if not args.api_key:
        print("[SKIP] 3/4 链路     未提供 --api-key，跳过（无法验证 LLM / embedding 出网）")
    else:
        started = time.time()
        try:
            resp = requests.post(f"{base}/search", json=probe_body,
                                 headers={"X-Api-Key": args.api_key}, timeout=args.timeout)
        except Exception as exc:
            return _fail(f"链路检查请求异常：{type(exc).__name__}: {exc}")
        elapsed = time.time() - started
        if resp.status_code != 200:
            return _fail(
                f"链路检查失败：/search 返回 {resp.status_code}。"
                "5xx 通常意味着 LLM 或 embedding 出网异常（密钥 / 端点 / 网络），"
                "或验签通过但 Key 值不对"
            )
        try:
            payload = resp.json()
        except ValueError:
            return _fail(f"链路检查失败：响应不是合法 JSON：{resp.text[:200]}")
        data = payload.get("data")
        if not isinstance(data, list):
            return _fail(
                f"链路检查失败：响应缺少顶层 data 数组，实际 {type(data).__name__}"
                "（契约要求 Search 响应为顶层 data 数组）"
            )
        print(f"[OK]   3/4 链路     /search（空 user_id）-> 200，data {len(data)} 条"
              f"（{elapsed:.2f}s）")

    # ---- 4/4 端点边界：公网不应暴露 API schema ----
    for path in ("/docs", "/openapi.json"):
        try:
            resp = requests.get(f"{base}{path}", timeout=args.timeout)
        except Exception as exc:
            return _fail(f"端点边界检查 {path} 请求异常：{type(exc).__name__}: {exc}")
        if resp.status_code != 404:
            return _fail(f"端点边界检查：{path} 返回 {resp.status_code}，期望 404")
    print("[OK]   4/4 边界     /docs 与 /openapi.json -> 404")

    print(f"\n结论: PASS（{base}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
