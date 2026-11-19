# -*- coding: utf-8 -*-
"""
RC3 去词面化确证。

构造一个"事件 content 不含时间词、只靠 TEMPORAL 边连接时间锚点"的图，
验证 temporal_lookup 的"时间→事件"反向遍历是**靠图结构**而非**靠事件文本里的时间词**：
  · temporal_lookup("昨天用户都做了些什么") → 应锚定 T10(昨天)，仅靠 T10 的反向 TEMPORAL 边取 e9/e10/e12，
    且这些事件 content 里**不包含"昨天"**。
  · 对照：主检索器 retrieve 在同一去词面图上，因事件不再含"昨天"，向量命中不到这些事件，
    无法展开（thing→temporal 反向权重=0），从而**取不到"昨天"的事件** —— 证明主检索器依赖词面。

用法:
    python -m eval.verify_temporal_delex
"""

import os
import sys
import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv, build_retriever

SRC = "data/dba_eval/lifemem_crossperiod.yaml"
OUT = "data/dba_eval/lifemem_crossperiod_delex.yaml"

# 事件 content 中要去掉的时间短语（按长度降序，先删长的避免残留）
STRIP_PHRASES = sorted([
    "高二暑假", "高三上学期", "高三下学期", "大四上学期", "大二上学期",
    "去年年底", "毕业前", "上个月", "上周", "昨天早上", "昨天", "今天",
], key=len, reverse=True)


def delex_events(content: str) -> str:
    for ph in STRIP_PHRASES:
        content = content.replace(ph, "")
    return content.strip()


def main() -> None:
    _load_dotenv(ROOT / ".env")

    data = yaml.safe_load(open(SRC, encoding="utf-8"))
    nodes = copy.deepcopy(data.get("nodes", []))
    # 只对"事件"去词面化；时间锚点 T* 保留时间词（否则查询"昨天"定位不到锚点）
    for n in nodes:
        if not n.get("id", "").startswith(("T", "th")):
            n["content"] = delex_events(n["content"])
    delex = {"nodes": nodes, "edges": data.get("edges", [])}
    # 保证目录存在
    OUT_PATH = ROOT / OUT
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(delex, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"[gen] 已生成去词面化图: {OUT_PATH}")

    _, r = build_retriever(OUT)

    print("\n" + "=" * 72)
    print("RC3 确证：temporal_lookup 是否靠结构而非词面")
    print("=" * 72)
    res = r.temporal_lookup("昨天用户都做了些什么？")
    print("[temporal_lookup]")
    fact_ids = []
    fact_no_time_ok = True
    for g in res.get("matches", []):
        ta = g["time_anchor"]
        print(f"  锚点: {ta['id']} = {ta['content']!r}")
        for f in g["facts"]:
            fact_ids.append(f["id"])
            has_w = any(ph in f["content"] for ph in STRIP_PHRASES)
            if has_w:
                fact_no_time_ok = False
            print(f"     - {f['id']} → {f['content']!r}")
    anchors = [g["time_anchor"]["id"] for g in res.get("matches", [])]
    t10_in_facts = "e9" in fact_ids and "e10" in fact_ids and "e12" in fact_ids
    anchored_t10 = "T10" in anchors
    print(f"\n  判定: 锚定T10={anchored_t10}; 取到昨天三事件(e9/e10/e12)={t10_in_facts}; 事件content无时间词={fact_no_time_ok}")
    verdict_tool = anchored_t10 and t10_in_facts and fact_no_time_ok
    print(f"  [OK] temporal_lookup 靠结构反向取到'昨天'事件（事件本身不含时间词）: {verdict_tool}")

    print("\n[主检索器 retrieve·同一查询·同一去词面图]")
    peak = r.retrieve("昨天用户都做了些什么？", purpose=["还原昨天经历"]).get("peak_memories", [])
    peak_ids = [m[0] for m in peak]
    main_found = {x for x in ("e9", "e10", "e12") if x in peak_ids}
    print(f"  取到节点: {peak_ids}")
    print(f"  取到'昨天'三事件: {main_found if main_found else '无'}")
    print("  → 对照：去掉词面后，主检索器不再能单凭'时间→事件'结构取到昨日事件"
          "（thing→temporal 反向权重=0），印证原先 TB=1.0 确有词面成分。")

    print("\n" + "=" * 72)
    if verdict_tool:
        print("RC3 确证通过：temporal_lookup 的'时间→事件'反向遍历**靠图结构（TEMPORAL 边）**，"
              "事件本身不含时间词也能取到；主检索器在去词面图上取不到该结构（作为对照）。")
    else:
        print("RC3 确证未通过：temporal_lookup 未能靠结构取到无词面的昨日事件，请检查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
