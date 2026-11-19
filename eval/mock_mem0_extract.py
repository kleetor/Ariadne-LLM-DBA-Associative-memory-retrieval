# -*- coding: utf-8 -*-
"""
模拟 Mem0 前置记忆管理（add infer=True 的 LLM 抽取阶段）——离线批量抽取

背景：Mem0 的 infer=True 在 api.gpt.ge（DeepSeek 兼容）上偶发 content=None
（reasoning 模型把结果放进 reasoning_content），导致 LLM 抽取失败。用户指示
绕过该故障：按 Mem0 ADDITIVE_EXTRACTION_PROMPT 的产出格式直接批量抽取一份，
保存成文件（mem0 抽取产物），再让 Mem0 用这份产物跑检索。

产出格式（对齐 Mem0 Phase 2 的解析逻辑 main.py#L978）：
    {"memory": [{"id": "0", "text": "...", "attributed_to": "user"}, ...]}

实现：把 98 条英文节点分批（每批 25 条）喂给 LLM，要求按 Mem0 质量标准
（上下文丰富、自包含、15-80 词、时间锚定、保留专名）抽取每条记忆，
输出 {"memory": [...]}；每条提取结果映射回源 node_id 存为边文件。

用法:
    python -m eval.mock_mem0_extract
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["MEM0_DIR"] = str(ROOT / "eval" / ".cache" / "mem0_home")

import yaml
from openai import OpenAI

SUB_YAML = "data/LiFEMem_en_subset.yaml"
OUT_MEM = "eval/outputs/mem0_en_extracted.json"      # {"node_id": [抽取文本...]}（溯源映射）
OUT_RAW = "eval/outputs/mem0_en_extraction_raw.json"  # 每批原始响应（复现用）

EXTRACTION_SYSTEM = """You are a Memory Extractor — a precise, evidence-bound processor responsible for extracting rich, contextual memories from a list of factual memory notes. For EACH note, produce ONE self-contained memory statement following these standards:

- Contextually rich, NOT atomic: capture fact + surrounding context in a single statement.
- Clean factual statement: keep emotional states, motivations, subjective descriptions; remove filler.
- Self-contained: replace pronouns with "the user" (third person).
- Concise but complete: 15-80 words, 1-2 sentences.
- Temporally grounded: preserve dates/durations; convert relative to absolute only when context gives it.
- Numerically precise: keep exact quantities.
- Preserve specific details: keep proper nouns (brands, places, person names), titles, exact numbers.

You enrich the note with plausible context ONLY from the note itself — do NOT invent facts that are not stated or clearly implied. If the note is already good, lightly enrich it.

Output STRICT JSON only: {"memory": [{"id": "0", "text": "..."}, {"id": "1", "text": "..."}]} — one entry per input note, in order, no extra keys."""


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def main() -> None:
    _load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_API_BASE"])
    model = os.environ["OPENAI_MODEL"]

    with open(ROOT / SUB_YAML, encoding="utf-8") as f:
        sub = yaml.safe_load(f)
    entries = [(n["id"], n["content"]) for n in sub["nodes"]]
    print(f"[extract] 待抽取节点: {len(entries)}")

    mapping = {}
    raws = []
    B = 25
    for start in range(0, len(entries), B):
        batch = entries[start:start + B]
        numbered = "\n".join(f"[{i}] {c}" for i, (_, c) in enumerate(batch))
        user = f"Extract one memory per note:\n{numbered}"
        ok = False
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = r.choices[0].message.content
                if not content:  # DeepSeek reasoning 偶发把结果放 reasoning_content
                    content = getattr(r.choices[0].message, "reasoning_content", None) or ""
                import re
                if content.strip().startswith("```"):
                    content = re.sub(r"^```(?:json)?|```$", "", content.strip()).strip()
                data = json.loads(content)
                mems = data.get("memory", [])
                if len(mems) < len(batch):
                    raise ValueError(f"期望 {len(batch)} 条，实际 {len(mems)} 条")
                for i, m in enumerate(mems):
                    nid = batch[i][0]
                    mapping.setdefault(nid, []).append(m.get("text", ""))
                raws.append({"start": start, "request": numbered, "response": content})
                print(f"[extract] 批 {start//B+1}: {len(mems)} 条 OK", flush=True)
                ok = True
                break
            except Exception as e:
                print(f"[extract] 批 {start//B+1} 第{attempt+1}次失败: {type(e).__name__} {str(e)[:100]}", flush=True)
        if not ok:
            raise RuntimeError(f"批 {start//B+1} 抽取失败")

    out = ROOT / OUT_MEM
    out.write_text(json.dumps({"meta": {"source": SUB_YAML, "model": model,
                                        "format": "mem0 ADDITIVE extraction mock",
                                        "count": len(mapping)},
                               "memories": mapping}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    (ROOT / OUT_RAW).write_text(json.dumps(raws, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[extract] 输出: {out}（{len(mapping)} 节点已映射）")
    # 抽查
    for nid in ["w1", "he0", "e27"]:
        if nid in mapping:
            print(f"  {nid}: {mapping[nid][0][:120]}")


if __name__ == "__main__":
    main()
