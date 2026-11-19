# -*- coding: utf-8 -*-
"""
全量 LiFEMem（322 节点）对话语料生成器

从 data/LiFEMem.yaml 读取全部 322 节点，按叙事顺序分组切批（每批 ~7 节点），
逐批调用 LLM 还原为口语化多轮对话（覆盖本批全部节点），汇总写出
data/dba_eval/lifemem_full_dialogs.yaml（格式对齐 subset 语料）。

用法:
    python -m eval.gen_full_dialogs            # 全量生成（约 45 次 LLM 调用）
    python -m eval.gen_full_dialogs --dry-run  # 只打印批次计划，不调用 LLM
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

# 叙事顺序（场景分组 → 批次顺序），保证对话流连贯
SCENE_ORDER = [
    ("学生时代·高中", ["h"], "高中生活的回忆"),
    ("学生时代·大专转码", ["c"], "大专与自学转码的回忆"),
    ("学生时代·学习", ["ln"], "学习历程（成绩/晚自习/转码）"),
    ("工作", ["w"], "工作经历"),
    ("健康", ["he"], "健康与体检"),
    ("社交", ["so"], "社交与朋友"),
    ("情绪", ["e"], "情绪状态"),
    ("家庭", ["f"], "家庭成员"),
    ("生活", ["t", "sh", "th"], "旅行/购物/人生阶段总结"),
    ("噪声", ["n"], "日常见闻闲聊（与用户无关）"),
]

GEN_PROMPT = """你是记忆整理师。以下是"用户"在本次对话中会自然谈到的信息点（来自其个人记忆库，每行格式：id | content | 类型）：

{items}

请生成一段自然的用户与助手闲聊对话（3-6 轮，user/assistant 交替，时间戳 HH:MM），要求：
1. 口语化自然表达，绝不能照念 content，允许换说法、加口语词（如"哎""其实""反正"）；
2. 必须覆盖上面【所有】信息点，每个点至少被自然带出一次；
3. 允许少量指代省略、寒暄、与主题无关的插话，但不要喧宾夺主；
4. 用户以第一人称自述，所有记忆内容由用户说出；assistant 是倾听回应角色；
5. 若信息点属于【日常见闻/噪声】（与他人的见闻、天气、短信等与用户自身无关），以用户"随口提及"的方式带出。

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
            "deprecated": bool(n.get("deprecated", False)),
            "forgotten": bool(n.get("forgotten", False)),
        })
    return nodes


def _scene_of(nid: str) -> str:
    """返回节点所属场景名；多前缀歧义时取最长匹配（如 he 优先于 h）"""
    best = None
    for scene_name, prefixes, _ in SCENE_ORDER:
        for p in prefixes:
            if nid.startswith(p) and (best is None or len(p) > len(best[1])):
                best = (scene_name, p)
    return best[0] if best else "其他"


def build_batches(nodes: list, batch_size: int = 7) -> list:
    """按 SCENE_ORDER 分组（最长前缀归属），同场景内顺序切批。返回 [{scene, items}]"""
    grouped = {}
    for scene_name, _, _ in SCENE_ORDER:
        grouped[scene_name] = []
    for n in nodes:
        grouped[_scene_of(n["id"])].append(n)

    seen = {n["id"] for lst in grouped.values() for n in lst}
    assert len(seen) == len(nodes), f"分组覆盖不全或重复: {len(nodes)} vs {len(seen)}"

    batches = []
    for scene_name, _, desc in SCENE_ORDER:
        scene_nodes = sorted(grouped[scene_name], key=lambda n: n["id"])
        for i in range(0, len(scene_nodes), batch_size):
            batches.append({"scene": scene_name, "desc": desc, "items": scene_nodes[i:i + batch_size]})
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="全量 322 节点对话语料生成")
    parser.add_argument("--yaml", default="data/LiFEMem.yaml")
    parser.add_argument("--out", default="data/dba_eval/lifemem_full_dialogs.yaml")
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="只打印批次计划")
    args = parser.parse_args()

    nodes = load_nodes(ROOT / args.yaml)
    print(f"全量节点: {len(nodes)}")
    batches = build_batches(nodes, args.batch_size)
    print(f"批次总数: {len(batches)}（每批 {args.batch_size} 节点）")

    if args.dry_run:
        for bi, b in enumerate(batches, 1):
            ids = [n["id"] for n in b["items"]]
            print(f"  d{bi:02d} [{b['scene']}] ({len(ids)}): {','.join(ids)}")
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

    dialogs = []
    for bi, b in enumerate(batches, 1):
        items = "\n".join(f"{n['id']} | {n['content']} | {n['type']}" for n in b["items"])
        covers = ", ".join(f'"{n["id"]}"' for n in b["items"])
        prompt = GEN_PROMPT.format(items=items, scene=b["scene"], covers=covers)
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
        dlg = json.loads(raw)
        dlg["id"] = f"d{bi:02d}"
        dlg["scene"] = b["scene"]
        dlg["covers"] = [n["id"] for n in b["items"]]
        dialogs.append(dlg)
        print(f"[{bi}/{len(batches)}] {dlg['id']} [{b['scene']}] {len(dlg['turns'])}轮 -> {dlg['covers'][0]}..{dlg['covers'][-1]}")

    out_data = {
        "subset": {
            "name": "lifemem_full_322",
            "scenes": [s for s, _, _ in SCENE_ORDER],
            "gold_nodes": [n["id"] for n in nodes],
        },
        "dialogs": dialogs,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\n已写出: {out_path}（{len(dialogs)} 批）")

    # 覆盖校验
    covered = set()
    for d in dialogs:
        covered.update(d["covers"])
    missing = set(n["id"] for n in nodes) - covered
    print(f"覆盖节点: {len(covered)}/322，缺失: {sorted(missing)}")


if __name__ == "__main__":
    main()
