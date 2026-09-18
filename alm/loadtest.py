# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 压力测试客户端。

按主办方给出的正式评测口径施压：**Add 并发 16–64、Search 并发 16–256、
单请求超时 1200s**（见 Plan §7.2）。除吞吐外，重点验证 §7.3 列出的三项风险：

    同 user 上 Add 与 Search 并发（图读写一致性）
    跨 user 高速切换、空间被 LRU 换出后重新载入（注册表锁与索引重载）
    Search 并发上限高于原预估（原仅按 Add 16–64 估）

阶段（`--scenario` 按序执行，可单独跑，数据由 --users 与下标确定性生成）：

    write   每个 user 写入一条身份事实（Add 并发 = --add-concurrency）
    read    每个 user 用同一问题检索（Search 并发 = --search-concurrency）
    mixed   同 user 上并发发起 Add 与 Search —— 冲击图读写一致性
    churn   重新检索最早写入的 user（超过 ALM_MAX_SPACES 时它们已被换出）

不变量（任一失败即整体 FAIL，退出码非 0）：

    1. Add 返回 success=true，且 request_id / user_id / session_id 原样回显
    2. 同 user「Add → Search」必须命中本次写入的事实（可检索性）
    3. Search 不得返回其它 user 的事实（跨 user 隔离）
    4. data 为顶层数组、条数 ≤ top_k、id / content 非空
    5. 全程无 5xx / 无连接错误

同时统计单次响应 `sum(len(content))`，用于核对平台 300000 字符总预算（§7.2）。

成本提示：每次 Add ≈ 1 次维护（多条 LLM 调用），每次 Search ≈ 2 次 LLM 调用；
`--scenario all --users 512` 是百量级 LLM 调用，请先小规模试跑。

Usage:
    # 完整流程，按正式评测的 Add 并发上限
    python -m alm.loadtest --scenario all --add-concurrency 64

    # 只压 Search 到平台上限
    python -m alm.loadtest --scenario read --search-concurrency 256 --users 512

    # 同 user 混合并发
    python -m alm.loadtest --scenario mixed --users 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# 用于跨 user 隔离判定的姓名格式：李000 ~ 李999
_NAME_RE = re.compile(r"李\d{3}")

_CITIES = ("杭州", "成都", "西安", "厦门", "长沙", "青岛", "昆明", "郑州")
_JOBS = ("后端开发", "平面设计", "中学教师", "临床医生", "审计师", "数据分析")

# 单次响应内容总长的观察上限（平台 300000 字符预算）
_CONTENT_BUDGET = 300000


def user_id_of(idx: int) -> str:
    return f"eval:loadtest:conv-{idx:04d}"


def session_id_of(idx: int) -> str:
    return f"eval:loadtest:sample:{idx:04d}"


def name_of(idx: int) -> str:
    """每个 user 唯一的姓名，同时充当「可检索性」与「跨 user 隔离」的判定锚点"""
    return f"李{idx:03d}"


def city_of(idx: int) -> str:
    return _CITIES[idx % len(_CITIES)]


def query_of(idx: int) -> str:
    """检索探针。

    问题里显式带上该 user 的姓名与城市：一是保证候选与查询的余弦能越过弃权阈值
    （``ALM_ABSTAIN_COS`` 默认 0.52，纯短问句如「我叫什么名字？」实测余弦只有
    0.23–0.41，会被判为「无相关记忆」直接返回空数组，实测 8/8 全弃权）；二是姓名
    是该 user 的唯一锚点，顺带把跨 user 隔离检查一起做了。
    """
    return f"我叫{name_of(idx)}，我在{city_of(idx)}做什么工作？"


def content_of(idx: int) -> str:
    city = city_of(idx)
    job = _JOBS[idx % len(_JOBS)]
    return f"我叫{name_of(idx)}，我在{city}做{job}，已经在这座城市住了五年。"


def content2_of(idx: int) -> str:
    return f"{name_of(idx)}最近开始学吉他了。"


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(pos, len(ordered) - 1))]


@dataclass
class Stats:
    """按操作类型累计延迟与失败"""

    label: str
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def record(self, seconds: float, error: Optional[str] = None) -> None:
        self.latencies.append(seconds)
        if error:
            self.errors.append(error)

    def summary(self) -> Dict[str, Any]:
        return {
            "op": self.label,
            "count": len(self.latencies),
            "failed": len(self.errors),
            "p50": round(_pct(self.latencies, 50), 2),
            "p95": round(_pct(self.latencies, 95), 2),
            "p99": round(_pct(self.latencies, 99), 2),
            "max": round(max(self.latencies) if self.latencies else 0.0, 2),
            "mean": round(sum(self.latencies) / len(self.latencies), 2) if self.latencies else 0.0,
            "first_error": self.errors[0] if self.errors else None,
        }


class Client:
    """线程安全的 ALM 客户端（每线程复用一条连接）"""

    def __init__(self, base_url: str, api_key: Optional[str], timeout: float):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-Api-Key"] = api_key
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def health(self) -> Tuple[bool, str]:
        try:
            resp = self._session().get(f"{self.base}/health", timeout=10)
        except Exception as exc:
            return False, f"health 连接失败: {exc}"
        if not (200 <= resp.status_code < 300):
            return False, f"health 返回 {resp.status_code}"
        return True, ""

    def add(self, idx: int, content: str, tag: str) -> Tuple[bool, float, str]:
        body = {
            "request_id": f"{user_id_of(idx)}:{tag}:{int(time.time() * 1000)}",
            "messages": [{"role": "user", "content": content}],
            "user_id": user_id_of(idx),
            "session_id": session_id_of(idx),
        }
        started = time.time()
        try:
            resp = self._session().post(
                f"{self.base}/add", json=body, headers=self.headers, timeout=self.timeout
            )
        except Exception as exc:
            return False, time.time() - started, f"add 请求失败: {exc}"
        elapsed = time.time() - started
        if resp.status_code != 200:
            return False, elapsed, f"add 返回 {resp.status_code}: {resp.text[:200]}"
        try:
            payload = resp.json()
        except ValueError:
            return False, elapsed, f"add 响应非 JSON: {resp.text[:200]}"
        if payload.get("success") is not True:
            return False, elapsed, f"add.success 为 {payload.get('success')!r}"
        for field_name in ("request_id", "user_id", "session_id"):
            if payload.get(field_name) != body[field_name]:
                return False, elapsed, f"add.{field_name} 未原样回显"
        return True, elapsed, ""

    def search(self, idx: int, query: str, top_k: int) -> Tuple[bool, float, str, List[dict]]:
        body = {"query": query, "user_id": user_id_of(idx), "top_k": top_k}
        started = time.time()
        try:
            resp = self._session().post(
                f"{self.base}/search", json=body, headers=self.headers, timeout=self.timeout
            )
        except Exception as exc:
            return False, time.time() - started, f"search 请求失败: {exc}", []
        elapsed = time.time() - started
        if resp.status_code != 200:
            return False, elapsed, f"search 返回 {resp.status_code}: {resp.text[:200]}", []
        try:
            payload = resp.json()
        except ValueError:
            return False, elapsed, f"search 响应非 JSON: {resp.text[:200]}", []
        items = payload.get("data")
        if not isinstance(items, list):
            return False, elapsed, "search.data 不是顶层数组", []
        if len(items) > top_k:
            return False, elapsed, f"返回 {len(items)} 条 > top_k={top_k}", items
        for position, item in enumerate(items):
            if not item.get("id") or not item.get("content"):
                return False, elapsed, f"data[{position}] 缺少 id / content", items
        return True, elapsed, "", items


@dataclass
class Check:
    """不变量检查结果的累积器"""

    failures: List[str] = field(default_factory=list)
    max_content_chars: int = 0
    over_budget: int = 0
    empty_results: int = 0

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def note_response(self, items: List[dict]) -> None:
        total = sum(len(item.get("content") or "") for item in items)
        if total > self.max_content_chars:
            self.max_content_chars = total
        if total > _CONTENT_BUDGET:
            self.over_budget += 1
        if not items:
            self.empty_results += 1


def _run_concurrent(tasks: List[Callable[[], Any]], concurrency: int) -> List[Any]:
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(lambda fn: fn(), tasks))


def _check_retrieval(check: Check, idx: int, items: List[dict]) -> None:
    """不变量 2 / 3：自己写的必须找得到，别人的绝不能出现"""
    texts = "\n".join(item.get("content") or "" for item in items)
    own = name_of(idx)
    if own not in texts:
        check.fail(f"user[{idx}] 检索未命中自己写入的 {own}（可检索性失败）")
    leaked = {found for found in _NAME_RE.findall(texts) if found != own}
    if leaked:
        check.fail(f"user[{idx}] 检索到其它 user 的姓名 {sorted(leaked)[:5]}（跨 user 泄漏）")


def phase_write(client: Client, args: argparse.Namespace, stats: Dict[str, Stats]) -> None:
    stat = stats["add"]
    indices = list(range(args.users))

    def task(idx: int):
        ok, elapsed, error = client.add(idx, content_of(idx), "w")
        stat.record(elapsed, None if ok else error)
        return ok

    print(f"[write] Add × {len(indices)}，并发 {args.add_concurrency} ...")
    started = time.time()
    results = _run_concurrent([lambda i=i: task(i) for i in indices], args.add_concurrency)
    print(f"[write] 完成 {sum(1 for r in results if r)}/{len(indices)}，"
          f"耗时 {time.time() - started:.1f}s")


def phase_read(client: Client, args: argparse.Namespace, stats: Dict[str, Stats],
               check: Check, indices: List[int], query: Optional[str] = None,
               concurrency: Optional[int] = None, label: str = "read") -> None:
    stat = stats["search"] if label == "read" else stats.get(f"search[{label}]", stats["search"])
    limit = concurrency or args.search_concurrency

    def task(idx: int):
        ok, elapsed, error, items = client.search(idx, query or query_of(idx), args.top_k)
        stat.record(elapsed, None if ok else error)
        if not ok:
            check.fail(f"user[{idx}] search 失败: {error}")
            return False
        check.note_response(items)
        _check_retrieval(check, idx, items)
        return True

    print(f"[{label}] Search × {len(indices)}，并发 {limit} ...")
    started = time.time()
    results = _run_concurrent([lambda i=i: task(i) for i in indices], limit)
    print(f"[{label}] 完成 {sum(1 for r in results if r)}/{len(indices)}，"
          f"耗时 {time.time() - started:.1f}s")


def phase_mixed(client: Client, args: argparse.Namespace, stats: Dict[str, Stats],
                check: Check) -> None:
    """同一 user 上**并发**发起 Add 与 Search —— 图正在被改写时的检索一致性。

    关键约束：两类请求必须落在同一个 user_id 上。若按 user 奇偶把 Add / Search 拆开，
    测到的只是「跨 user 并发」，碰不到 §7.3-2 的竞态（`add` 持锁改图、`search` 不持锁
    读图）。因此这里每个 user 都同时投递一个 Add 与一个 Search。
    """
    jobs: List[Tuple[str, int]] = []
    for idx in range(args.users):
        jobs.append(("add", idx))
        jobs.append(("search", idx))

    add_stat, search_stat = stats["add"], stats["search"]

    def task(job: Tuple[str, int]):
        kind, idx = job
        if kind == "add":
            ok, elapsed, error = client.add(idx, content2_of(idx), "m")
            add_stat.record(elapsed, None if ok else error)
            if not ok:
                check.fail(f"mixed user[{idx}] add 失败: {error}")
            return ok
        ok, elapsed, error, items = client.search(idx, query_of(idx), args.top_k)
        search_stat.record(elapsed, None if ok else error)
        if not ok:
            check.fail(f"mixed user[{idx}] search 失败: {error}")
            return False
        check.note_response(items)
        _check_retrieval(check, idx, items)
        return True

    print(f"[mixed] 同 user Add + Search × {len(jobs)}（{args.users} 个 user），"
          f"并发 {args.mixed_concurrency} ...")
    started = time.time()
    results = _run_concurrent([lambda job=job: task(job) for job in jobs], args.mixed_concurrency)
    print(f"[mixed] 完成 {sum(1 for r in results if r)}/{len(jobs)}，"
          f"耗时 {time.time() - started:.1f}s")


def print_summary(stats: Dict[str, Stats], check: Check, elapsed: float) -> bool:
    print("\n" + "=" * 72)
    print("ALM 压测结果")
    print("=" * 72)
    header = f"{'op':<16}{'count':>7}{'failed':>8}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"
    print(header)
    total_failed = 0
    for key in sorted(stats):
        summary = stats[key].summary()
        if summary["count"] == 0:
            continue
        total_failed += summary["failed"]
        print(f"{summary['op']:<16}{summary['count']:>7}{summary['failed']:>8}"
              f"{summary['p50']:>9}{summary['p95']:>9}{summary['p99']:>9}{summary['max']:>9}")
    print("-" * 72)
    print(f"总耗时            : {elapsed:.1f}s")
    print(f"单次响应最大内容量: {check.max_content_chars} 字符"
          f"（平台预算 {_CONTENT_BUDGET}，超限次数 {check.over_budget}）")
    print(f"空结果次数        : {check.empty_results}")
    print(f"请求失败          : {total_failed}")

    if total_failed:
        for key in sorted(stats):
            if stats[key].errors:
                print(f"  [{key}] 首个错误: {stats[key].errors[0]}")
    if check.failures:
        print(f"\n不变量失败 {len(check.failures)} 项（最多显示 10 条）:")
        for message in check.failures[:10]:
            print(f"  - {message}")

    passed = total_failed == 0 and not check.failures
    print("\n结论: " + ("PASS" if passed else "FAIL"))
    print("=" * 72)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 压力测试客户端")
    parser.add_argument("--base-url", default="http://127.0.0.1:8770")
    parser.add_argument("--api-key", default=None, help="X-Api-Key（服务未开鉴权时留空）")
    parser.add_argument("--timeout", type=float, default=1200,
                        help="单次请求超时秒数（对齐平台默认 1200s）")
    parser.add_argument("--scenario", default="all",
                        choices=["all", "write", "read", "mixed", "churn"],
                        help="all = write → read → mixed → churn")
    parser.add_argument("--users", type=int, default=64, help="参与的 user_id 数量")
    parser.add_argument("--add-concurrency", type=int, default=16,
                        help="write 阶段并发（正式评测 Add 可选 16–64）")
    parser.add_argument("--search-concurrency", type=int, default=16,
                        help="read 阶段并发（正式评测 Search 可选 16–256）")
    parser.add_argument("--mixed-concurrency", type=int, default=32,
                        help="mixed 阶段并发")
    parser.add_argument("--churn-users", type=int, default=32,
                        help="churn 阶段重新检索最早的多少个 user")
    parser.add_argument("--top-k", type=int, default=100, help="Search 的 top_k（正式评测为 100）")
    parser.add_argument("--out", default=None, help="把结果摘要写入 JSON 文件")
    args = parser.parse_args()

    client = Client(args.base_url, args.api_key, args.timeout)
    ok, message = client.health()
    if not ok:
        print(f"[FAIL] {message}", file=sys.stderr)
        return 1
    print(f"[OK] /health 可达: {args.base_url}")

    stats: Dict[str, Stats] = {
        "add": Stats("add"),
        "search": Stats("search"),
        "search[churn]": Stats("search[churn]"),
    }
    check = Check()
    started = time.time()

    if args.scenario in ("all", "write"):
        phase_write(client, args, stats)
    if args.scenario in ("all", "read"):
        phase_read(client, args, stats, check, list(range(args.users)))
    if args.scenario in ("all", "mixed"):
        phase_mixed(client, args, stats, check)
    if args.scenario in ("all", "churn"):
        indices = list(range(min(args.churn_users, args.users)))
        phase_read(client, args, stats, check, indices, concurrency=args.search_concurrency,
                   label="churn")

    passed = print_summary(stats, check, time.time() - started)

    if args.out:
        report = {
            "base_url": args.base_url,
            "scenario": args.scenario,
            "users": args.users,
            "passed": passed,
            "stats": {key: stats[key].summary() for key in stats},
            "invariant_failures": check.failures,
            "max_content_chars": check.max_content_chars,
            "empty_results": check.empty_results,
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"报告已写入: {args.out}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
