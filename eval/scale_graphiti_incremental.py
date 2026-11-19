# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
Graphiti 在「已建 N 规模图」上再 ingest 1 条新内容的增量单次 token。

与 Ariadne `MemoryDBA.maintain` / Mem0 `add(infer=True)` 对等：
三者都是「在规模 N 的库/图上，写入 1 条」的单次抽取成本。

之前 scale_graphiti_usage.py 只用「全量建图 total / N」当单条均摊，那不等于
「在 N 规模图上加 1 条的增量成本」（均摊会掩盖大图下消歧成本上升）。

在 .venv_graphiti 内运行：
  python eval/scale_graphiti_incremental.py
"""

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import frameworks.graphiti_runner as gr  # noqa: E402

SCALES = [50, 100, 200]
OUT = ROOT / "eval/outputs/scale_graphiti_incremental.json"
# 一条与任何已有内容都有实体重叠的新倾诉，强制触发与图中实体的消歧/合并
NEW_EPISODE_TEXT = (
    "用户说最近加班太多，颈椎又酸又僵，晚上睡不好，体检还查出轻度脂肪肝，"
    "他担心身体扛不住，常常在日记里写自己的焦虑。"
)


class Recorder:
    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def wrap(self, create_fn):
        rec = self

        async def w_create(*a, **k):
            r = await create_fn(*a, **k)
            u = getattr(r, "usage", None)
            if u is not None:
                rec.prompt += getattr(u, "prompt_tokens", 0) or 0
                rec.completion += getattr(u, "completion_tokens", 0) or 0
                rec.calls += 1
            return r

        return w_create

    @property
    def total(self):
        return self.prompt + self.completion


async def measure_scale(scale: int, llm, embedder) -> dict:
    from datetime import datetime, timezone
    from graphiti_core.nodes import EpisodeType

    graphiti = gr.build_graphiti("neo4j", llm, embedder)
    group_id = f"scale_{scale}"

    # 在已建 N 规模图的 group 里再加 1 条 episode（逐条 add，真实路径）
    rec = Recorder()
    completions = llm.client.chat.completions
    orig_create = completions.create
    completions.create = rec.wrap(orig_create)
    try:
        await graphiti.add_episode(
            name="inc_1",
            episode_body=NEW_EPISODE_TEXT,
            source_description="评测增量单条写入",
            reference_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
            source=EpisodeType.message,
            group_id=group_id,
        )
    finally:
        completions.create = orig_create

    info = {
        "scale": scale,
        "incremental_llm_calls": rec.calls,
        "incremental_prompt_tokens": rec.prompt,
        "incremental_completion_tokens": rec.completion,
        "incremental_total": rec.total,
    }
    print(f"[scale={scale}] 增量单条ingest: LLM={rec.calls} in={rec.prompt} out={rec.completion} "
          f"total={rec.total}", flush=True)
    return info


async def main() -> None:
    gr._load_dotenv(ROOT / ".env")
    gr.force_combined_extraction()
    llm, embedder = gr.build_clients()

    results = []
    for s in SCALES:
        try:
            results.append(await measure_scale(s, llm, embedder))
        except Exception as e:
            print(f"[scale={s}] 失败: {type(e).__name__}: {e}", flush=True)
            results.append({"scale": s, "error": f"{type(e).__name__}: {e}"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scales": SCALES, "results": results,
                              "note": ("在已建 N 规模图 group 上逐条 ingest 1 条新内容的增量单次 token；"
                                       "与 Ariadne maintain/Mem0 add 对等；NEW_EPISODE_TEXT 含实体重叠以触发消歧")},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
