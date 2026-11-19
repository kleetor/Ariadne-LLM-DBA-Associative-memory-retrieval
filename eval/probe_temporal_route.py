# -*- coding: utf-8 -*-
"""临时探针：验证 /search 的「时序工具 → 主通道」串联路由（Plan §3.2 阶段三）。

真数据灌一段对话进临时 ALMEngine，打两发：
  · 时序问题 → 期望日志显示命中时序 + 追加锚点 + 主通道执行
  · 非时序问题 → 期望日志显示未命中时序，且返回与关闭开关时一致

用完即删（eval/ 已被 .gitignore 忽略）。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import yaml

from alm.config import ALMConfig
from alm.contract import AddMessage, AddRequest
from alm.engine import ALMEngine

DIALOGS_PATH = _ROOT / "data" / "dba_eval" / "groupchat_mixed_dialogs.yaml"
USER_ID = "eval:probe:temporal:conv-0"
SESSION_ID = "eval:probe:temporal:s0"
BASE_MS = 1_704_067_200_000  # 2024-01-01

TEMPORAL_QUERY = "上周三的需求评审会定了什么方案"
NONTEMPORAL_QUERIES = ["用户叫什么名字", "小林喜欢什么运动"]


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _load_dialogs(limit: int) -> list[dict]:
    with DIALOGS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    dialogs = data["dialogs"] if isinstance(data, dict) else data
    return dialogs[:limit]


def _build_messages(dialog: dict, idx: int) -> list[AddMessage]:
    return [
        AddMessage(
            role=turn["role"],
            content=f"[{turn['speaker']}] {turn['text']}",
            timestamp=BASE_MS + idx * 86_400_000 + i * 300_000,
        )
        for i, turn in enumerate(dialog["turns"])
    ]


def _print_route_log(cap: _Capture, title: str):
    print(f"\n  --- {title} 相关日志 ---")
    keys = ("时序路由", "时序查询", "主通道已执行", "弃权", "检索:")
    for line in cap.lines:
        if any(k in line for k in keys):
            print("   ", line)


def _shot(space, cap: _Capture, query: str, label: str) -> list[dict]:
    cap.lines.clear()
    started = time.time()
    data = space.search(query, top_k=100)
    print(f"\n[{label}] query={query!r}  用时 {time.time() - started:.1f}s")
    info = space.retriever.inference.infer_purpose(query)
    print(f"  目的推断: temporal={info.get('temporal')} purposes={info.get('purposes')}")
    _print_route_log(cap, label)
    print(f"  返回 {len(data)} 条")
    for item in data[:3]:
        print(f"    - score={item['score']} content={item['content'][:160]}")
    return data


def _deterministic_equivalence(space, query: str) -> bool:
    """确定性等价检验：把目的推断固定后，比较「注入 purpose」与「内部推断」两条路径。

    `retrieve_with_story` 的 LLM（StoryRank）与弃权判定都是非确定性的，会掩盖路由本身的
    影响。这里绕开它们，直接对 `retrieve()`（纯确定性）比较峰值节点集合与每跳候选得分，
    证明「注入 purpose」与「内部推断出同一 purpose」等价。
    """
    retr = space.retriever
    fixed = ["获取信息", "确认细节"]
    orig = retr.inference.infer_purpose
    retr.inference.infer_purpose = lambda q: {
        "status": "fixed", "query_type": "information",
        "temporal": False, "purposes": list(fixed),
    }
    kwargs = dict(
        seed_k=space._effective_seed_k(),
        max_hops=space.config.max_hops,
        expand_k=space.config.expand_k,
    )
    try:
        injected = retr.retrieve(query, purpose=list(fixed), **kwargs)
        internal = retr.retrieve(query, **kwargs)
    finally:
        retr.inference.infer_purpose = orig

    ids_inj = [mid for mid, _ in injected.get("peak_memories") or []]
    ids_int = [mid for mid, _ in internal.get("peak_memories") or []]
    hops_inj = [(h["hop"], [c["id"] for c in h["candidates"]]) for h in injected["hop_history"]]
    hops_int = [(h["hop"], [c["id"] for c in h["candidates"]]) for h in internal["hop_history"]]
    same = ids_inj == ids_int and hops_inj == hops_int
    print(f"\n# 确定性等价 {query!r}：注入purpose 峰值数={len(ids_inj)} 内部推断峰值数={len(ids_int)}"
          f" 一致={same}")
    return same


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    cap = _Capture()
    space_log = logging.getLogger("alm.space")
    space_log.setLevel(logging.INFO)
    space_log.addHandler(cap)

    dialogs = _load_dialogs(2)
    cfg = ALMConfig.from_env()
    cfg.data_dir = Path(tempfile.mkdtemp(prefix="alm_troute_"))
    engine = ALMEngine(cfg)
    print(f"# 时序路由探针 | 语料 {DIALOGS_PATH.name} | 灌入 {len(dialogs)} 段对话")
    print(f"# temporal_route(默认)={cfg.temporal_route}")

    try:
        started = time.time()
        for idx, dialog in enumerate(dialogs):
            engine.add(
                AddRequest(
                    request_id=f"{USER_ID}:{dialog['id']}",
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    messages=_build_messages(dialog, idx),
                )
            )
        print(f"# 写入耗时 {time.time() - started:.1f}s，图节点数={engine.get_space(USER_ID).graph.node_count}")

        space = engine.get_space(USER_ID, create=False)
        anchors = [
            n for n in space.graph.graph.nodes()
            if space.retriever._tool_is_time_node(n)
            and space.retriever._has_reverse_temporal(n)
        ]
        print(f"\n# 真时间锚点 {len(anchors)} 个（含反向 TEMPORAL 邻居）:")
        for n in anchors:
            print(f"    {n}  {space.graph.get_content(n)!r}")

        # —— 第 0 步：确定性等价检验（隔离 LLM 噪声）——
        _deterministic_equivalence(space, "用户叫什么名字")
        _deterministic_equivalence(space, TEMPORAL_QUERY)

        # —— 第 1 发：时序问题（开关开启，默认）——
        t_on = _shot(space, cap, TEMPORAL_QUERY, "时序/开关ON")

        # —— 第 2 发：非时序问题（开关开启 vs 关闭，应完全一致）——
        for q in NONTEMPORAL_QUERIES:
            n_on = _shot(space, cap, q, "非时序/开关ON")
            cfg.temporal_route = False
            n_off = _shot(space, cap, q, "非时序/开关OFF")
            cfg.temporal_route = True
            same = json.dumps(n_on, ensure_ascii=False, sort_keys=True) == \
                json.dumps(n_off, ensure_ascii=False, sort_keys=True)
            print(f"\n# 非时序 {q!r}：开关ON 与 OFF 返回完全一致 = {same}")
        print(f"# 时序 {TEMPORAL_QUERY!r}：开关ON 返回条数={len(t_on)}")
    finally:
        try:
            engine.close()
        except Exception:
            pass
        shutil.rmtree(cfg.data_dir, ignore_errors=True)
        space_log.removeHandler(cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
