# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 运行数据留存清理（默认留存 30 天）。

运行数据仅用于本次服务、不得用于训练 / 分析 / 共享，并应在留存期结束后删除。
ALM 侧每个 user_id 在 `--data-dir` 下独占**一个目录**，目录内是该空间的全部产物：

    <escaped user_id>/
        graph.yaml         # 记忆图谱（节点 / 边 / 正文）
        graph.index/       # 图谱的 FAISS 索引 + 向量缓存 + 指纹
        docs.yaml          # 文档区条文快照（知识库式长文档的原文块）
        docs.log           # 文档区增量日志（快照之后新追加的原文块）
        docs.index/        # 文档区的 FAISS 索引 + 向量缓存

因此"删除运行数据" = 按留存期删除这些空间目录。判定时间用目录内**最新的 mtime**：
每次 Add 都会原子重写图谱 YAML 并随后落盘索引，而 Search 不写盘，所以 mtime 就是该
空间最后一次被写入的时间（即最后一个触及它的运行任务的时间）。

"一空间一目录"把**归组这件事从规则里拿掉了**：删空间就是删目录，不存在"认出了一个
产物、漏掉另一个"的可能。旧布局把五份产物**平铺**在 data_dir 下，靠 `.docs` 后缀
剥离来归组——那正是留存条款最容易失效的地方（多一个产物就多一次漏删的机会）。

对**旧布局的历史遗留**（`<user>.yaml` / `<user>.index/` / …）仍会识别并清理，
否则它们会以"非空间条目"的身份被永久跳过；其余顶层条目一律只报告、不删除，
避免误删挂载进来的其它东西。

安全性：默认**只报告不删除**（预演），确认无误后加 `--apply` 才真正删除；`data_dir`
不存在或为空时直接返回，不做任何事。

用法：
    python -m alm.cleanup                      # 预演：列出超过 30 天未写入的空间
    python -m alm.cleanup --apply              # 执行删除
    python -m alm.cleanup --older-than-days 7 --apply
    python -m alm.cleanup --data-dir data/alm

⚠️ 两条约束互相牵扯，顺序不能错：
  ① 规则同时要求「接口在提交后至少 30 天稳定可访问」，**不能运行一结束就停服**；
  ② 清理必须先停服务：引擎退出时会把内存中驻留的空间重新落盘
     （`alm.engine.ALMEngine.close()`），在服务运行期删除文件会被这一步写回来。

因此清理应在**可访问期结束之后**执行：

    docker compose --env-file alm/.env.alm -f deploy/docker-compose.alm.yml stop ariadne-alm
    docker compose --env-file alm/.env.alm -f deploy/docker-compose.alm.yml run --rm --no-deps ariadne-alm \
        python -m alm.cleanup --older-than-days 30 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 旧布局（一空间五份产物**平铺**在 data_dir 下）留下的后缀。新布局下不会再产生，
# 但服务器上有历史数据；它们装的同样是记忆正文，必须按同一留存期清理，
# 否则"删了新空间、留着旧文件"仍然是违反 30 天留存条款的。
#
# 与空间目录名**不会歧义**：`engine.space_dirname` 会把 '.' 转义成 `%2E`，
# 所以新布局的目录名里不可能出现点号，凡是以这些后缀结尾的必然是旧布局产物。
_LEGACY_SUFFIXES = (".docs.yaml", ".docs.log", ".docs.index", ".index", ".yaml")


def _legacy_stem(name: str) -> Optional[str]:
    """旧布局产物 → 它所属的 user_id 目录名；不是旧布局产物则返回 None（顺序：长后缀优先）"""
    for suffix in _LEGACY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


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
    """把 data_dir 下的空间目录按 user 空间归组。

    返回 (spaces, ignored)：
      spaces  —— {转义后的 user_id: [该空间的产物路径]}，新布局下长度恒为 1
      ignored —— 既不是空间目录、也不像旧布局产物的顶层条目（不碰，只报告）

    新布局：**一个顶层目录 = 一个空间**（`engine.space_dirname` 决定目录名），
    "这个文件属于哪个空间"不再需要推断，删空间就是删目录。

    旧布局：五份产物平铺 + 按后缀命名（`<user>.yaml` / `<user>.index/` / …）。
    服务器上可能有这类历史遗留，它们装的同样是记忆正文，故按**同一留存期**一并清理
    （与同名的空间目录并组）。不这么做的话，旧文件会以"非空间条目"的身份被永久跳过
    ——那正是 30 天留存条款要避免的。
    """
    spaces: Dict[str, List[Path]] = {}
    ignored: List[Path] = []
    for entry in sorted(data_dir.iterdir()):
        # 先判旧布局后缀（文件与目录都可能），再判"这是个空间目录"。
        # 顺序不能反：`<user>.index/` 是**目录**，先按目录认领就会把它当成一个
        # 名叫 `xxx.index` 的独立空间，删得掉但与它的图谱脱钩（不致命，但难以对账）。
        stem = _legacy_stem(entry.name)
        if stem is not None:
            spaces.setdefault(stem, []).append(entry)
        elif entry.is_dir():
            spaces.setdefault(entry.name, []).append(entry)
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
    for name, paths in sorted(spaces.items()):
        last_touch = max(_newest_mtime(p) for p in paths)
        if last_touch >= cutoff:
            kept += 1
            continue
        expired.append((name, paths, last_touch, sum(_dir_size(p) for p in paths)))

    if expired:
        total_bytes = sum(item[3] for item in expired)
        print(f"\n超过留存期的空间 {len(expired)} 个，合计 {_human_size(total_bytes)}：")
        for name, paths, last_touch, size in expired:
            print(f"  - {name:<48} 最后写入 {time.strftime('%Y-%m-%d', time.localtime(last_touch))}"
                  f"（{_human_age(now - last_touch)}）  {len(paths)} 项产物  {_human_size(size)}")
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
    parser = argparse.ArgumentParser(description="ALM 运行数据留存清理（默认预演）")
    parser.add_argument("--data-dir", default=None,
                        help="记忆数据目录（默认取 ALM_DATA_DIR，再回落到 data/alm）")
    parser.add_argument("--older-than-days", type=float, default=30.0,
                        help="留存期天数，超过则清理（默认 30 天）")
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
