# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 评测数据留存清理（AML 要求 30 天内删除）。

AML 要求评测数据仅用于本次评测、不得训练 / 分析 / 共享，并在任务结束后 **30 天内删除**。
ALM 侧每个 user_id 在 `--data-dir` 下落两份产物：

    <escaped user_id>.yaml      # 记忆图谱（节点 / 边 / 正文）
    <escaped user_id>.index/    # FAISS 索引 + 向量缓存 + 指纹

因此"删除评测数据" = 按留存期清理这两类文件。判定时间用**文件 mtime**：每次 Add 都会
原子重写 YAML 并随后落盘索引，而 Search 不写盘，所以 mtime 就是该空间最后一次被写入的
时间（即最后一个触及它的评测任务的时间）。

安全性：默认**只报告不删除**（预演），确认无误后加 `--apply` 才真正删除；`data_dir`
不存在或为空时直接返回，不做任何事。

用法：
    python -m alm.cleanup                      # 预演：列出超过 30 天未写入的空间
    python -m alm.cleanup --apply              # 执行删除
    python -m alm.cleanup --older-than-days 7 --apply
    python -m alm.cleanup --data-dir data/alm

⚠️ 两条约束互相牵扯，顺序不能错：
  ① 规则同时要求「接口在提交后至少 30 天稳定可访问」，**不能评测一结束就停服**；
  ② 清理必须先停服务：引擎退出时会把内存中驻留的空间重新落盘
     （`alm.engine.ALMEngine.close()`），在服务运行期删除文件会被这一步写回来。

因此清理应在**可访问期结束之后**执行：

    docker compose -f docker-compose.alm.yml stop ariadne-alm
    docker compose -f docker-compose.alm.yml run --rm --no-deps ariadne-alm \
        python -m alm.cleanup --older-than-days 30 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

YAML_SUFFIX = ".yaml"
INDEX_SUFFIX = ".index"


def _newest_mtime(path: Path) -> float:
    """目录取「自身 + 递归内容」的最新 mtime；文件取自身。"""
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    if path.is_file():
        return newest
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return newest


def _dir_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def collect_spaces(data_dir: Path) -> Tuple[Dict[str, List[Path]], List[Path]]:
    """把 data_dir 下的产物按 user 空间归组。

    返回 (spaces, ignored)：
      spaces  —— {stem: [yaml?, index_dir?]}，stem 即转义后的 user_id
      ignored —— 既不是 *.yaml 也不是 *.index/ 的顶层条目（不碰，只报告，避免误删）
    """
    spaces: Dict[str, List[Path]] = {}
    ignored: List[Path] = []
    for entry in sorted(data_dir.iterdir()):
        name = entry.name
        if entry.is_file() and name.endswith(YAML_SUFFIX):
            spaces.setdefault(name[: -len(YAML_SUFFIX)], []).append(entry)
        elif entry.is_dir() and name.endswith(INDEX_SUFFIX):
            spaces.setdefault(name[: -len(INDEX_SUFFIX)], []).append(entry)
        else:
            ignored.append(entry)
    return spaces, ignored


def _human_age(seconds: float) -> str:
    days = seconds / 86400.0
    return f"{days:.0f} 天前" if days >= 1 else f"{seconds / 3600.0:.1f} 小时前"


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def purge(data_dir: Path, older_than_days: float, apply: bool) -> int:
    if not data_dir.is_dir():
        print(f"data_dir 不存在，无需清理: {data_dir}")
        return 0

    spaces, ignored = collect_spaces(data_dir)
    if not spaces:
        print(f"没有 user 空间产物可清理: {data_dir}")
        return 0

    now = time.time()
    cutoff = now - older_than_days * 86400.0
    mode = "执行删除" if apply else "预演"
    print(f"ALM 留存清理 | data_dir={data_dir} | 留存期={older_than_days:g} 天 "
          f"| 截止={time.strftime('%Y-%m-%d %H:%M', time.localtime(cutoff))} | 模式={mode}")

    expired: List[Tuple[str, List[Path], float, int]] = []
    kept = 0
    for stem, paths in sorted(spaces.items()):
        last_touch = max(_newest_mtime(p) for p in paths)
        if last_touch >= cutoff:
            kept += 1
            continue
        expired.append((stem, paths, last_touch, sum(_dir_size(p) for p in paths)))

    if expired:
        total_bytes = sum(item[3] for item in expired)
        print(f"\n超过留存期的空间 {len(expired)} 个，合计 {_human_size(total_bytes)}：")
        for stem, paths, last_touch, size in expired:
            kinds = "+".join(
                "yaml" if p.is_file() else "index" for p in sorted(paths, key=lambda x: x.is_file())
            )
            print(f"  - {stem:<48} 最后写入 {time.strftime('%Y-%m-%d', time.localtime(last_touch))}"
                  f"（{_human_age(now - last_touch)}）  {kinds}  {_human_size(size)}")
        if apply:
            removed = 0
            for _stem, paths, _last_touch, _size in expired:
                for path in paths:
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                        removed += 1
                    except OSError as exc:
                        print(f"    [WARN] 删除失败 {path}: {exc}", file=sys.stderr)
            print(f"\n已删除 {removed}/{sum(len(p) for _, p, _, _ in expired)} 个产物")
        else:
            print("\n[预演] 未删除任何文件；确认无误后加 --apply 执行")
    else:
        print("\n没有超过留存期的空间")

    print(f"保留 {kept} 个空间（最后一次写入在留存期内）")
    if ignored:
        print(f"忽略 {len(ignored)} 个非空间条目（未触碰）: "
              f"{', '.join(p.name for p in ignored[:5])}"
              f"{' …' if len(ignored) > 5 else ''}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 评测数据留存清理（默认预演）")
    parser.add_argument("--data-dir", default=None,
                        help="记忆数据目录（默认取 ALM_DATA_DIR，再回落到 data/alm）")
    parser.add_argument("--older-than-days", type=float, default=30.0,
                        help="留存期天数，超过则清理（AML 口径为 30）")
    parser.add_argument("--apply", action="store_true",
                        help="真正执行删除；缺省只预演并打印将要删除的内容")
    args = parser.parse_args()

    if args.older_than_days < 0:
        print("错误: --older-than-days 不能为负", file=sys.stderr)
        return 2

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # 与 alm.config 的口径一致：ALM_DATA_DIR 优先，否则 data/alm
        import os

        data_dir = Path(os.environ.get("ALM_DATA_DIR") or "data/alm")

    return purge(data_dir, args.older_than_days, args.apply)


if __name__ == "__main__":
    sys.exit(main())
