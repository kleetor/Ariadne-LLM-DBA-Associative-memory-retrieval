# -*- coding: utf-8 -*-
"""E10 数据集构造：在现有单主体图上叠加 1~2 个不同语域的主体，形成低/中/高三档分化。

为什么这样造：
E10-A 的判据是 ROC AUC（同场景 vs 跨场景）。现有 163 节点只有 0.717，因为七个场景
同属"一个用户的同一语域"。要拉开分化度，必须引入**真正无关**的内容——
不是"另一段中文"（实测无关中文仍有 0.405），而是**另一个主体 / 另一种语域**。

三档：
  L（低）  : 现有 163 节点                      —— 已知 AUC 0.717，作 baseline
  M（中）  : L + 主体 B（45 岁中学语文老师，成都，家庭/教学/健康/旅行）
  H（高）  : M + 主体 C（28 岁跨境电商运营，深圳，外贸/物流/健身/游戏，商务语域）

边用**规则**生成（E10 测的是检索机制，不是抽取质量）：
同场景内相邻节点连 SEQUENCE、每 8 条一组内连 SCENARIO、组首连 ATTRIBUTE 到组内其余节点。
这样 PAR 的跳转轴（node_type × relation）在新增节点上同样有定义。

用法：
    python -m eval.e10_build_graphs              # 复用已有生成结果（若有）
    python -m eval.e10_build_graphs --regenerate  # 强制重新调 LLM 生成 B/C
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import httpx  # noqa: E402
import yaml  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from eval.judge_comprehensive import _extract_json, _load_dotenv  # noqa: E402

BASE_GOLD = "data/dba_eval/comprehensive_gold.yaml"
CACHE = ROOT / "eval/outputs/e10/generated_nodes.json"
OUT_DIR = ROOT / "data/dba_eval"

# 主体画像：刻意与 A（后端开发 / 高考 / 大专 / 编程）在职业、年龄、语域上拉开
PERSONAS = {
    "B": {
        "desc": "45 岁的中学语文老师，女性，住成都，已婚有一个上初中的女儿，"
                "关注教学、家庭、健康、旅行与读书",
        "scenes": [
            ("bh", "高中教学与班级管理"),
            ("bf", "家庭与女儿"),
            ("bd", "健康与作息"),
            ("bt", "旅行与读书"),
            ("bs", "社交与同事"),
            ("bm", "日常琐事"),
        ],
    },
    "C": {
        "desc": "28 岁的跨境电商运营，男性，住深圳，单身，"
                "关注外贸订单、物流、健身、游戏与城市生活，语域偏商务",
        "scenes": [
            ("cw", "跨境电商运营与订单"),
            ("cl", "物流与供应链"),
            ("cg", "健身与体能"),
            ("cv", "游戏与娱乐"),
            ("cs", "社交与同城活动"),
            ("cm", "日常与居住"),
        ],
    },
}

TYPES = ["status", "action", "thing", "person", "emotion", "reason"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是记忆数据构造器。请为一个人物生成记忆节点。

人物设定：{persona}

本次需要两个场景：
  场景 1（前缀 {p1}）：{s1}
  场景 2（前缀 {p2}）：{s2}

生成要求：
1. 每个场景 **15 条**，合计 30 条；
2. 每条一句话，用「用户……」的第三人称写法（与该数据集既有风格一致）；
3. 类型从 {types} 中选，六个类型都要出现，但不必均分；
4. **必须与该人物设定强相关**；严禁出现编程、代码、高考、大专、后端开发、数据分析网课等词
   （那些属于另一个人物的记忆）；
5. 内容要具体（有职业细节、时间词、人物关系），不要空泛；
6. 场景 1 的 15 条 `id` 用 {p1}1..{p1}15，场景 2 用 {p2}1..{p2}15。

只输出 JSON：
{{"nodes":[{{"id":"{p1}1","content":"用户……","type":"action"}}]}}"""),
    ("human", "请生成。"),
])


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("E10_MODEL") or "deepseek-v4-flash",
        api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "x",
        base_url=os.environ.get("JUDGE_API_BASE"),
        http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(180.0, connect=15.0)),
        temperature=0,
        max_tokens=4096,
        max_retries=2,
    )


def _generate(persona_key: str, force: bool) -> List[Dict[str, Any]]:
    cache: Dict[str, List[Dict[str, Any]]] = {}
    if CACHE.exists():
        cache = json.load(open(CACHE, encoding="utf-8"))
    if not force and persona_key in cache:
        print(f"  [{persona_key}] 复用缓存 {len(cache[persona_key])} 条")
        return cache[persona_key]

    p = PERSONAS[persona_key]
    chain = PROMPT | _llm()
    scenes = p["scenes"]
    nodes: List[Dict[str, Any]] = []
    for i in range(0, len(scenes), 2):
        (p1, s1), (p2, s2) = scenes[i], scenes[i + 1]
        data = None
        for att in range(4):
            try:
                resp = chain.invoke({"persona": p["desc"], "p1": p1, "s1": s1,
                                     "p2": p2, "s2": s2, "types": "/".join(TYPES)})
                data = _extract_json(resp.content or "")
                if isinstance(data, dict) and data.get("nodes"):
                    break
            except Exception as exc:
                print(f"    !! 生成失败 {type(exc).__name__}", flush=True)
            time.sleep(2 * (att + 1))
        got = (data or {}).get("nodes") or []
        for n in got:
            c, t = str(n.get("content") or "").strip(), str(n.get("type") or "").strip()
            if c and t in TYPES:
                nodes.append({"id": str(n.get("id") or "").strip(), "content": c, "type": t})
        print(f"  [{persona_key}] {s1}/{s2} -> {len(got)} 条", flush=True)

    # id 去重 + 补缺（LLM 偶发重复/漏号）
    seen, fixed = set(), []
    for i, n in enumerate(nodes):
        nid = n["id"] or f"{persona_key.lower()}{i+1}"
        while nid in seen:
            nid = f"{nid}x"
        n["id"] = nid
        seen.add(nid)
        fixed.append(n)
    cache[persona_key] = fixed
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return fixed


def _scene_of(nid: str) -> str:
    return nid[1] if len(nid) >= 2 else "?"


def _build_edges(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """规则建边：同场景相邻 SEQUENCE；每 8 条组内 SCENARIO；组首到组内 ATTRIBUTE。

    字段名必须是 `type`（与 `comprehensive_gold.yaml` 一致）；写成 `rel_type` 会被
    MemoryGraph 当成未知边类型整条丢弃（首跑 180 条新节点的边全被跳过）。
    """
    edges: List[Dict[str, Any]] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        groups.setdefault(_scene_of(n["id"]), []).append(n)
    for scene, items in groups.items():
        for i in range(len(items) - 1):
            edges.append({"from": items[i]["id"], "to": items[i + 1]["id"],
                          "type": "sequence"})
        for g0 in range(0, len(items), 8):
            chunk = items[g0:g0 + 8]
            for a in range(len(chunk)):
                for b in range(a + 1, len(chunk)):
                    edges.append({"from": chunk[a]["id"], "to": chunk[b]["id"],
                                  "type": "scenario"})
            if len(chunk) > 1:
                for b in range(1, len(chunk)):
                    edges.append({"from": chunk[0]["id"], "to": chunk[b]["id"],
                                  "type": "attribute"})
    # 去重
    uniq, seen = [], set()
    for e in edges:
        k = (e["from"], e["to"], e["type"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="E10 数据集构造（低/中/高分化）")
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    _load_dotenv(ROOT / ".env")
    with open(ROOT / BASE_GOLD, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    base_nodes = base.get("nodes") or []
    base_edges = base.get("edges") or []
    print(f"[E10] 基础图：{len(base_nodes)} 节点 / {len(base_edges)} 边")

    extra: Dict[str, List[Dict[str, Any]]] = {}
    for key in PERSONAS:
        print(f"[E10] 生成主体 {key} …", flush=True)
        extra[key] = _generate(key, args.regenerate)
        print(f"[E10] 主体 {key} 合计 {len(extra[key])} 节点")

    def dump(name: str, nodes, edges) -> None:
        p = OUT_DIR / f"e10_{name}.yaml"
        p.write_text(yaml.safe_dump({"nodes": nodes, "edges": edges},
                                    allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        print(f"  -> {p.name}: {len(nodes)} 节点 / {len(edges)} 边")

    # L：仅原图
    dump("L", base_nodes, base_edges)
    # M：+ 主体 B
    b_nodes = extra["B"]
    dump("M", base_nodes + b_nodes, base_edges + _build_edges(b_nodes))
    # H：+ B + C
    c_nodes = extra["C"]
    dump("H", base_nodes + b_nodes + c_nodes,
         base_edges + _build_edges(b_nodes) + _build_edges(c_nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
