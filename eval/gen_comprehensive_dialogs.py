# -*- coding: utf-8 -*-
"""
综合能力测评 · 双主体对话语料生成器（大叙事 + 双主体 + 时序）

从 data/dba_eval/comprehensive_gold.yaml 读取节点（含主体 user/assistant/time），
按生命周期场景分组、场景内按主体轮转切批（保证每批都混合三类主体），逐批调用 LLM
还原为口语化多轮对话，写出 data/dba_eval/comprehensive_dialogs.yaml。

与 gen_full_dialogs.py（仅用户自述）的关键差异：
  · 信息点带主体 subject，assistant 也承载 subject=assistant 的行为/产出信息点；
  · 每批强制自然出现 ≥2 个不同时间词，驱动 TEMPORAL 链；
  · 学生时代（高中/大专）批次侧重大叙事回顾，工作批次侧重当下，共同支撑跨期时间线。

用法:
    python -m eval.gen_comprehensive_dialogs --dry-run   # 只打印批次计划与主体分布
    python -m eval.gen_comprehensive_dialogs             # 生成
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

from eval.build_comprehensive_gold import SCENE_ORDER, SCENE_PREFIX


def _extract_json(raw: str):
    """从 LLM 输出中鲁棒提取 JSON 对象（剥离代码围栏/前后杂文本）。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.S)
        raw = re.sub(r"\s*```$", "", raw, flags=re.S)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"未找到 JSON 对象: {raw[:120]}")
    return json.loads(raw[start:end + 1])


def _scene_of(nid: str) -> str:
    return SCENE_PREFIX.get(nid.rstrip("0123456789")[:2], "其他")


def _subject_of(nid: str) -> str:
    pref = nid.rstrip("0123456789")[:2]
    if pref.startswith("u"):
        return "user"
    if pref.startswith("a"):
        return "assistant"
    return "_TIME"


GEN_PROMPT = """你是记忆整理师。以下是"本次对话"中会自然谈到的信息点（每行格式：id | 主体 | content | 类型）：

{items}

请生成一段自然的用户与助手对话（3-6 轮，user/assistant 交替，时间戳 HH:MM 递增），要求：
1. 口语化自然表达，绝不能照念 content，允许换说法、加口语词（如"哎""其实""反正"）；
2. 必须覆盖上面【所有】信息点，每个点至少被自然带出一次；
3. 主体规则：主体=user 的信息点必须由「用户」说出；主体=assistant 的信息点必须由「助手」说出并表现为助手完成的动作（如"我帮你整理了""我给你写了…"），助手不要只当倾听者；主体=_TIME 的时间词（如"周一晚上""这周""高三""去年暑假"）需自然出现在对话文本里（由带出该时间的事件方说出即可）；
4. 同一批对话应自然出现【至少 2 个不同】时间词（如"这周""高三""去年暑假""开学前"），把事件按时间先后串联起来。若信息点里有"暑假/开学/去年/今年/上周/这周/周X/几月几日"这类可识别时间词，请尽量在句子里自然说出来（学生时代回忆也可用"去年暑假""高三那年开学"等），而不要只用"高中时/那时"这种模糊说法；
5. 允许少量指代省略、寒暄、与主题无关的插话，但不要喧宾夺主。

严格输出 JSON（不要任何额外文字）：
{{"id": "d01", "scene": "{scene}", "covers": [{covers}], "turns": [{{"role": "user", "time": "09:00", "text": "..."}}]}}
其中 covers 是本批全部信息点 id 的列表（原样列出）。"""


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key:
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_nodes(yaml_path: Path) -> list:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    nodes = []
    for n in data["nodes"]:
        nodes.append({
            "id": n["id"],
            "content": n["content"],
            "type": n.get("type", ""),
            "subject": _subject_of(n["id"]),
        })
    return nodes


def build_batches(nodes: list, batch_size: int = 7) -> list:
    """按生命周期场景分组；场景内按主体轮转切批，保证每批混合 user/assistant/time 三主体。"""
    grouped = defaultdict(list)
    for n in nodes:
        grouped[_scene_of(n["id"])].append(n)

    batches = []
    for scene_name, desc in SCENE_ORDER:
        bucket = {"user": [], "assistant": [], "_TIME": []}
        for n in grouped.get(scene_name, []):
            bucket[n["subject"]].append(n)

        piles = [bucket["user"], bucket["assistant"], bucket["_TIME"]]
        cur = []
        while any(piles):
            progressed = False
            for pile in piles:
                if pile:
                    cur.append(pile.pop(0))
                    progressed = True
                    if len(cur) >= batch_size:
                        break
            if len(cur) >= batch_size or not progressed:
                if cur:
                    batches.append({"scene": scene_name, "desc": desc, "items": cur[:]})
                    cur = cur[batch_size:]
                if len(cur) >= batch_size:
                    continue
        if cur:
            batches.append({"scene": scene_name, "desc": desc, "items": cur})

    batches.sort(key=lambda b: [s for s, _ in SCENE_ORDER].index(b["scene"]))
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="综合性对话语料生成（双主体）")
    parser.add_argument("--gold", default="data/dba_eval/comprehensive_gold.yaml")
    parser.add_argument("--out", default="data/dba_eval/comprehensive_dialogs.yaml")
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="只打印批次计划")
    args = parser.parse_args()

    nodes = load_nodes(ROOT / args.gold)
    print(f"gold 节点: {len(nodes)}")
    batches = build_batches(nodes, args.batch_size)
    print(f"批次总数: {len(batches)}（每批 {args.batch_size} 节点）")

    if args.dry_run:
        for bi, b in enumerate(batches, 1):
            subs = defaultdict(int)
            for n in b["items"]:
                subs[n["subject"]] += 1
            ids = [n["id"] for n in b["items"]]
            print(f"  d{bi:02d} [{b['scene']}] {dict(subs)}: {','.join(ids)}")
        return

    _load_dotenv(ROOT / ".env")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=16384,
        extra_body={"thinking": {"type": "disabled"}},
    )
    from dba_pipeline.extraction.dba import has_temporal_signal

    dialogs = []
    for bi, b in enumerate(batches, 1):
        items = "\n".join(f"{n['id']} | {n['subject']} | {n['content']} | {n['type']}" for n in b["items"])
        covers = ", ".join(f'"{n["id"]}"' for n in b["items"])
        prompt = GEN_PROMPT.format(items=items, scene=b["scene"], covers=covers)
        dlg = None
        for attempt in range(3):  # 每批最多重试 3 次（LLM 偶发非严格 JSON）
            raw = llm.invoke(prompt).content.strip()
            try:
                dlg = _extract_json(raw)
                break
            except (json.JSONDecodeError, ValueError) as e:
                print(f"    [{bi}] JSON 解析失败（第{attempt + 1}次重试）: {e}")
                dlg = None
        if dlg is None:
            raise RuntimeError(f"批次 d{bi:02d} 连续 3 次解析失败，请检查 LLM 输出")
        dlg["id"] = f"d{bi:02d}"
        dlg["scene"] = b["scene"]
        dlg["covers"] = [n["id"] for n in b["items"]]
        conv = "\n".join(f"{t['role']}({t['time']}): {t['text']}" for t in dlg["turns"])
        dlg["temporal"] = has_temporal_signal(conv)
        dialogs.append(dlg)
        print(f"[{bi}/{len(batches)}] {dlg['id']} [{b['scene']}] {len(dlg['turns'])}轮 "
              f"temporal={dlg['temporal']} -> {dlg['covers'][0]}..{dlg['covers'][-1]}")

    out_data = {
        "subset": {
            "name": "comprehensive",
            "scenes": [s for s, _ in SCENE_ORDER],
            "gold_nodes": [n["id"] for n in nodes],
        },
        "dialogs": dialogs,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\n已写出: {out_path}（{len(dialogs)} 批）")

    covered = set()
    for d in dialogs:
        covered.update(d["covers"])
    missing = set(n["id"] for n in nodes) - covered
    temporal_batches = sum(1 for d in dialogs if d.get("temporal"))
    print(f"覆盖节点: {len(covered)}/{len(nodes)}，缺失: {sorted(missing)}")
    print(f"触发时序的批次: {temporal_batches}/{len(dialogs)}")


if __name__ == "__main__":
    main()
