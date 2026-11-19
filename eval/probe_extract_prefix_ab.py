# -*- coding: utf-8 -*-
"""阶段一：Add 文本化的 A/B —— ALM 自加的 `[role@timestamp]` 前缀是否影响建图抽取。

背景（Plan/0920——ALM系统分析与改造计划.md §2.1）：
  ALM 与 MCP 共用同一个建图内核（GraphBuilder + MemoryDBA.maintain），
  唯一变量是 ALM 会把消息渲染成 `[role@timestamp] 内容` 再喂入，MCP 不代劳。

两臂定义（唯一差别就是外层前缀）：

  A  现状：`[user@1704067200000] [小林] 上周体检结果出来了`
  B  纯文本：`[小林] 上周体检结果出来了`
     —— content 里的 `[昵称]` 标识是语料自带的，两臂完全一致，不参与变量。

注意：抽取是 LLM 非确定性，**单轮只能看方向**；要坐实差异需重复 2–3 轮取中位数
（依据：Plan §7.10.5 实测检索层波动远大于故事层，R@all std ±11.4pp）。

数据：data/dba_eval/groupchat_mixed_dialogs.yaml
用法：python -m eval.probe_extract_prefix_ab [--dialogs N]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # 与其它 eval 脚本一致：先从 .env 装载模型与路径配置
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import yaml

from alm.config import ALMConfig
from alm.contract import AddMessage, AddRequest
from alm.engine import ALMEngine
from alm.space import MemorySpace as Space
from eval.test_alm_vs_mcp_e2e import graph_stats, print_stats

DIALOGS_PATH = _ROOT / "data" / "dba_eval" / "groupchat_mixed_dialogs.yaml"
USER_ID = "eval:probe:prefix:conv-0"
SESSION_ID = "eval:probe:prefix:s0"
BASE_MS = 1_704_067_200_000  # 2024-01-01


def load_dialogs(limit: int | None) -> list[dict]:
    with DIALOGS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    dialogs = data["dialogs"] if isinstance(data, dict) else data
    return dialogs[:limit] if limit else dialogs


def build_messages(dialog: dict, dialog_idx: int) -> list[AddMessage]:
    """与 e2e 同口径：对话序号按天递增，轮次间隔 5 分钟（时间锚点/TEMPORAL 边依赖它）"""
    return [
        AddMessage(
            role=turn["role"],
            content=f"[{turn['speaker']}] {turn['text']}",
            timestamp=BASE_MS + dialog_idx * 86_400_000 + i * 300_000,
        )
        for i, turn in enumerate(dialog["turns"])
    ]


def ingest(dialogs: list[dict], arm: str) -> tuple[dict, float]:
    """灌一个臂，返回（图画像, 耗时秒）。arm ∈ {A, B}"""
    cfg = ALMConfig.from_env()
    cfg.data_dir = Path(tempfile.mkdtemp(prefix=f"almprefix_{arm}_"))
    engine = ALMEngine(cfg)

    original = Space._format_conversation
    if arm == "B":
        # 唯一变量：去掉外层 [role@timestamp]，只拼 content
        Space._format_conversation = staticmethod(
            lambda messages: "\n".join(m.content for m in messages)
        )

    started = time.time()
    try:
        for dialog_idx, dialog in enumerate(dialogs):
            engine.add(
                AddRequest(
                    request_id=f"{USER_ID}:{dialog['id']}",
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    messages=build_messages(dialog, dialog_idx),
                )
            )
        elapsed = time.time() - started
        space = engine.get_space(USER_ID, create=False)
        stats = graph_stats(space.graph, arm)
    finally:
        Space._format_conversation = original
        try:
            engine.close()
        except Exception:
            pass
        _rmtree(cfg.data_dir)

    return stats, elapsed


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _pct(a: int, b: int) -> str:
    if not b:
        return "n/a"
    return f"{(a - b) / b * 100:+.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogs", type=int, default=None,
                        help="只取前 N 段对话（默认全部）")
    args = parser.parse_args()

    dialogs = load_dialogs(args.dialogs)
    print(f"# A/B 探针：Add 文本化的 [role@timestamp] 前缀")
    print(f"# 语料 {DIALOGS_PATH.name}：{len(dialogs)} 段对话")
    print("# A = 现状（带 [role@timestamp] 前缀）| B = 纯文本（只拼 content）")
    print("# 单轮只作方向判断：抽取是 LLM 非确定性，跨轮不可逐条对比\n")

    results: dict[str, dict] = {}
    for arm in ("A", "B"):
        print(f"===== 臂 {arm} =====")
        stats, elapsed = ingest(dialogs, arm)
        results[arm] = stats
        print_stats(stats)
        print(f"    写入耗时 {elapsed:.1f}s\n")

    a, b = results["A"], results["B"]
    print("===== 差异（B 相对 A）=====")
    rows = [
        ("节点数", "nodes"), ("边数", "edges"),
        ("TEMPORAL 边", "temporal_edges"), ("时间锚点节点", "time_anchor_nodes"),
        ("「用户」出现", "collapse_user"), ("「助手」出现", "assistant_label"),
    ]
    print(f"  {'指标':<14}{'A':>6}{'B':>6}{'差':>10}")
    for label, key in rows:
        va, vb = int(a.get(key) or 0), int(b.get(key) or 0)
        print(f"  {label:<14}{va:>6}{vb:>6}{_pct(vb, va):>10}")

    print("\n===== 判读 =====")
    print("  · 节点/边差在噪声内 → 前缀不影响建图，§2.1 的『待确认』可关闭；")
    print("  · 时间锚点或 TEMPORAL 边显著变少 → 前缀参与了时间信息的注入，不可删；")
    print("  · 「用户」/「助手」称法差异 → 前缀影响归属判定，需人工核对样本。")
    print("  单轮结论不可下，需 2–3 轮取中位数后再定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
