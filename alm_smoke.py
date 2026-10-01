#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ALM 服务器快速体检：端口健康 → 参数核对 → 小型烟雾（add + search）。

为什么单独放一个脚本：排查时最常问的三件事——"服务活着吗"、"跑的是不是那份配置"、
"链路现在还通不通"——此前要手敲 curl + docker exec 好几条。这里一次跑完并给退出码，
可直接串进部署流程（部署完 → 体检 → 通过才申报）。

用法（**在本地开发机跑，打服务器的公网地址**）：
    python alm_smoke.py --url https://alm.ariadne.top --key <KEY>
    python alm_smoke.py --url https://alm.ariadne.top/add            # 给端点也行，自动剥成 base
    python alm_smoke.py --url https://alm.ariadne.top --skip-write    # 只查健康，不写数据
    python alm_smoke.py --url https://alm.ariadne.top --json          # 机器可读（CI 用）

凭据与默认地址：优先 `--key/--url`；其次读仓库根 `.env.alm` 的 `ALM_API_KEYS` 首个与
`ALM_PUBLIC_URL`；都没有就打 `http://127.0.0.1:$ALM_PORT`。

⚠️ **参数断言必须在服务器上做**：`precision_route / max_tokens / timeout` 这些只有读到
服务在用的那份 env 才算数，公网接口不返回它们（`/health` 只有 `{"status":"ok"}`）。所以
本地跑时这些项是 `[SKIP]`，要在服务器上补一次：
    docker cp alm_smoke.py ariadne-alm:/app/
    docker exec ariadne-alm python /app/alm_smoke.py --in-container --skip-write

退出码：0 = 全通过；1 = 有 FAIL（明细见输出）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 体检参数（改这里即可调断言口径）
EXPECT = {
    "precision_route": True,        # 精确证据路由：题面带作答指令时收拢读出
    "precision_max_hops": 1,        # 收拢到根 + 1 hop
    "precision_max_peak_nodes": 0,  # 0 = 按篇幅预算自适应（>0 才是定长）
    "llm_max_tokens": 32768,        # 8k 会让 StoryRank 在最难的题上撞顶（0926）
    "llm_timeout": 120.0,           # 不设会吃 SDK 默认 600s × 重试 = 最坏 40 分钟
}

# 烟雾用的中文素材（写进去要能检索得到，且第二次检索故意做成"选择题形态"）
ADD_MESSAGES = [
    {"role": "user", "content": "我平时早上都喝手冲咖啡，最近胃有点不舒服。"},
    {"role": "user", "content": "医生建议我这段时间别喝咖啡，改成喝小米粥。"},
]
SEARCH_PLAIN = "我早上一般喝什么？"
SEARCH_CHOICE = (
    "结合医生建议，我今天早上应该喝什么？"
    " A. 手冲咖啡 B. 小米粥 C. 都可以 D. 不确定"
    " 请只回答正确选项，例如 (X)。"
)


class Probe:
    """一次体检的结果收集（每条一个 PASS/FAIL/SKIP）。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok, detail: str = "", ms: float = 0.0) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append({"name": name, "status": status,
                          "detail": detail, "ms": round(ms, 1)})

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if r["status"] == "FAIL"]

    def render(self, as_json: bool) -> None:
        if as_json:
            print(json.dumps(self.rows, ensure_ascii=False, indent=2))
            return
        width = max(len(r["name"]) for r in self.rows)
        print("=" * 78)
        for r in self.rows:
            print(f"  [{r['status']}] {r['name']:<{width}}  {r['ms']:>7.1f}ms  {r['detail']}")
        print("=" * 78)
        bad = len(self.failed)
        print(f"  合计 {len(self.rows)} 项：通过 "
              f"{sum(1 for r in self.rows if r['status'] == 'PASS')} / 跳过 "
              f"{sum(1 for r in self.rows if r['status'] == 'SKIP')} / 失败 {bad}")


def _env_file(path: Path) -> dict:
    """极简 .env 解析（不引依赖；只取 KEY=VALUE 首值）。"""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def _normalize_base(url: str) -> str:
    """接受 base（`https://host`）或**任一端点**（`.../add`、`.../search`、`.../health`）→ base。

    为什么容忍端点：部署单/浏览器里给的通常就是 `https://alm.ariadne.top/add` 这种完整
    地址，让脚本自己剥掉，比要求人记住"只填 host"更不容易出错。
    """
    text = (url or "").strip().rstrip("/")
    for suffix in ("/add", "/search", "/health"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _request(url: str, body: dict | None, key: str, timeout: float) -> tuple[int, dict, float]:
    """返回 (status, json, 秒)。HTTP 错误码也当结果返回，不抛异常（体检要把 401 记成结果）。"""
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw or "{}"), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            payload = {"raw": raw[:200]}
        return exc.code, payload, time.perf_counter() - started
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # 连不上 / 超时也要**当成结果报出来**：体检脚本在"服务没起来"时崩掉是最没用的行为
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - started


def check_config(probe: Probe, assert_ok: bool) -> None:
    """参数核对。**只在能 import 到 alm.config 时做**（容器内或本机带依赖均可）。

    `assert_ok` 决定这些项是 PASS/FAIL 还是 SKIP：本地进程读的是**本机** `.env`，
    与服务器在用的那份无关，拿它断言只会报假警；所以只有 `--in-container`（读到服务
    真正在用的 env）才断言。
    """
    try:
        from alm.config import ALMConfig  # noqa: PLC0415
    except Exception as exc:  # 本机没装依赖是常态，不当失败
        probe.add("参数核对（导入 alm.config）", None, f"跳过：{type(exc).__name__}")
        return

    cfg = ALMConfig.from_env()
    shown = {
        "precision_route": cfg.precision_route,
        "precision_max_hops": cfg.precision_max_hops,
        "precision_max_peak_nodes": cfg.precision_max_peak_nodes,
        "llm_max_tokens": cfg.llm_max_tokens,
        "llm_timeout": cfg.llm_timeout,
        "trace_io": cfg.trace_io,
        "doc_route": getattr(cfg, "doc_route", None),
        "abstain_cos": getattr(cfg, "abstain_cos", None),
        "max_hops": cfg.max_hops,
        "seed_k": cfg.seed_k,
        "max_top_k": cfg.max_top_k,
    }
    print("  参数：" + " ".join(f"{k}={v}" for k, v in shown.items()))
    for field, want in EXPECT.items():
        got = getattr(cfg, field)
        probe.add(f"参数 {field}={want}", (got == want) if assert_ok else None,
                  f"实际 {got}" + ("" if assert_ok else "（本机值，仅供参考）"))

    if not assert_ok:
        probe.add("参数断言（需 --in-container）", None,
                  "本地读的是本机 .env；容器内加 --in-container 才会断言")


def run(args: argparse.Namespace) -> Probe:
    probe = Probe()
    env = _env_file(ROOT / ".env.alm")
    port = env.get("ALM_PORT", "8770")
    # 目标优先级：--url > .env.alm 的 ALM_PUBLIC_URL > 本机回环
    base = _normalize_base(args.url or env.get("ALM_PUBLIC_URL") or f"http://127.0.0.1:{port}")
    key = args.key or (env.get("ALM_API_KEYS", "").split(",")[0].strip())
    space = args.space or f"u_smoke_{uuid.uuid4().hex[:12]}"
    print(f"目标 {base}  空间 {space}  凭据 {'有' if key else '无'}")
    print(f"  端点 {base}/health  {base}/add  {base}/search")

    # 1) 端口健康（免鉴权，见契约）
    status, payload, ms = _request(f"{base}/health", None, "", args.timeout)
    probe.add("GET /health 可达", status == 200,
              f"HTTP {status} {payload.get('error', '')}".strip(), ms)
    probe.add("GET /health 返回 ok", payload.get("status") == "ok", str(payload)[:60])

    # 2) 参数
    check_config(probe, assert_ok=args.in_container)

    # 3) 鉴权：配了 Key 时，不带凭据必须 401
    if key:
        status, _, ms = _request(f"{base}/search", {"query": "ping", "user_id": space,
                                                    "top_k": 1}, "", args.timeout)
        probe.add("无凭据被拒（401）", status == 401, f"HTTP {status}", ms)

    if args.skip_write:
        return probe

    # 4) 写入：契约要求原样回显三个 ID
    req_id, sess_id = f"smoke-{uuid.uuid4().hex[:8]}", f"sess-{uuid.uuid4().hex[:8]}"
    status, payload, ms = _request(f"{base}/add", {
        "request_id": req_id, "messages": ADD_MESSAGES,
        "user_id": space, "session_id": sess_id,
    }, key, args.timeout)
    echo_ok = (payload.get("request_id") == req_id
               and payload.get("user_id") == space
               and payload.get("session_id") == sess_id)
    probe.add("POST /add 成功", status == 200, f"HTTP {status} {str(payload)[:80]}", ms)
    probe.add("POST /add 回显三 ID", echo_ok, "")

    # 5) 检索：普通问句 + **选择题形态**（后者用来验证精确证据路由的判型/收拢）
    for label, query in (("普通问句", SEARCH_PLAIN), ("选择题形态", SEARCH_CHOICE)):
        status, payload, ms = _request(f"{base}/search", {
            "query": query, "user_id": space, "top_k": args.top_k,
        }, key, args.timeout)
        items = payload.get("data") if isinstance(payload.get("data"), list) else []
        probe.add(f"POST /search {label} 有返回", status == 200 and bool(items),
                  f"HTTP {status} 条目 {len(items)}", ms)

    return probe


def main() -> int:
    ap = argparse.ArgumentParser(description="ALM 服务器快速体检 + 小型烟雾")
    ap.add_argument("--url", help="服务地址（默认 http://127.0.0.1:$ALM_PORT）")
    ap.add_argument("--key", help="API Key（默认取 .env.alm 的 ALM_API_KEYS 首个）")
    ap.add_argument("--space", help="指定 user_id（默认建临时空间 u_smoke_xxx）")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=180.0, help="单请求超时秒（search 可能较慢）")
    ap.add_argument("--skip-write", action="store_true", help="只查健康与参数，不写数据")
    ap.add_argument("--in-container", action="store_true",
                    help="在容器内运行（此时参数就是服务在用的那份）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args()

    probe = run(args)
    probe.render(args.json)
    if probe.failed:
        print("  ✗ 有失败项。选择题那条若失败，多半是精确路由/篇幅预算的问题，"
              "配合 `docker logs | grep -E '判型|精检|缩喂|已硬截断'` 看。")
    return 1 if probe.failed else 0


if __name__ == "__main__":
    sys.exit(main())
