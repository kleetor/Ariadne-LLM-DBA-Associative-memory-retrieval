# SPDX-License-Identifier: AGPL-3.0-only

# -*- coding: utf-8 -*-
"""
C3 溯源巡检的**准确性基准**（precision / recall）

为什么单独建这个：e2e 里 C3 的召回是"1 / 0 / 1"式波动，但那没有 ground truth——
既不知道 0 是"判断干净"还是"漏报"，也不知道报出来的是真问题还是误报。
要量"判得准不准"，必须手工标注靶子。

做法：手工构造 (原文, 节点列表) 用例，每个节点都带 ground-truth 标签：
    clean        —— 不该被报（含"保留限定口吻的推测""允许的代指消解/补全省略"
                    "具体时间锚点"「上周五」"已纳入映射的近期表述"「前几天」这些**陷阱负例**）
    unsupported  —— 该被报为 unsupported（推测写成事实 / 凭空补信息 / 过度概括）
    granularity  —— 该被报为 granularity（「过两天」独立成节点 / 多事合并）
另有 expect_missing_any：原文里"有但没抽"的关键事实（重点考"为什么"）。

每轮对每个用例调一次 `InferenceEngine.audit_source`（真实 LLM），按节点 id 精确匹配：
    TP = 标注为该类且被报为该类
    FP = 标注为 clean 却被报（误报，含标错类别）
    FN = 标注为该类却没被报（含被报到别的类别下）
missing 按"期望关键词是否出现在某条 quote 里"计命中；缺失类**只测召回**——
"原文还有什么该抽没抽"是开放集，无法算精确率。

用法:
    python -m eval.test_c3_audit_accuracy            # 每用例 5 轮
    set C3_ROUNDS=3 && python -m eval.test_c3_audit_accuracy
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/.cache/pylibs"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.run_p_baseline import _load_dotenv  # noqa: E402

ROUNDS = int(os.environ.get("C3_ROUNDS", "5"))
SWEEP_ROUNDS = int(os.environ.get("C3_SWEEP_ROUNDS", "3"))

# ────────────────────────────── 标注用例 ──────────────────────────────
# 每个节点：id / type / content / label（clean | unsupported | granularity）

CASES = [
    {
        "id": "c1-推测写成事实",
        "batch_text": "\n".join([
            "[用户] 上周三评审完，过两天我可能要去一趟外地出差。",
            "[用户] 还没定，估计得看项目进度。",
        ]),
        "nodes": [
            {"id": "n1", "type": "status", "content": "用户要去外地出差", "label": "unsupported"},
            {"id": "n2", "type": "thing", "content": "上周三", "label": "clean"},
            {"id": "n3", "type": "status", "content": "用户的出差还没定下来", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c2-凭空补信息",
        "batch_text": "[阿哲] 客户上周五说要换主色，我这两天在改稿。",
        "nodes": [
            {"id": "n1", "type": "status", "content": "阿哲在改稿", "label": "clean"},
            {"id": "n2", "type": "status", "content": "客户要求把主色换成蓝色", "label": "unsupported"},
            {"id": "n3", "type": "thing", "content": "上周五", "label": "clean"},
            {"id": "n4", "type": "status", "content": "阿哲养了两只猫", "label": "unsupported"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c3-过度概括",
        "batch_text": "[小林] 我这三天都在加班，前天弄到凌晨两点。",
        "nodes": [
            {"id": "n1", "type": "status", "content": "小林连续三天通宵到凌晨两点", "label": "unsupported"},
            {"id": "n2", "type": "thing", "content": "前天", "label": "clean"},
            {"id": "n3", "type": "status", "content": "小林最近在加班", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c4-保留限定口吻(陷阱负例)",
        "batch_text": "\n".join([
            "[用户] 下周我可能要请两天假。",
            "[助手] 定了吗？",
            "[用户] 还没准，看项目进度。",
        ]),
        "nodes": [
            {"id": "n1", "type": "status", "content": "用户下周可能请两天假", "label": "clean"},
            {"id": "n2", "type": "status", "content": "用户的请假还没确定", "label": "clean"},
            {"id": "n3", "type": "thing", "content": "下周", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c5-代指消解与补全省略(陷阱负例)",
        "batch_text": "[阿哲] 我上周三把稿子交了，老周看完说没问题。",
        "nodes": [
            {"id": "n1", "type": "status", "content": "阿哲上周三交了稿子", "label": "clean"},
            {"id": "n2", "type": "status", "content": "老周看了阿哲的稿子并确认没问题", "label": "clean"},
            {"id": "n3", "type": "thing", "content": "上周三", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c6-无法定位的模糊表述",
        "batch_text": "[用户] 评审完，过两天我可能要去外地出差。",
        "nodes": [
            {"id": "n1", "type": "thing", "content": "过两天", "label": "granularity"},
            {"id": "n2", "type": "status", "content": "用户可能要去外地出差", "label": "clean"},
            {"id": "n3", "type": "status", "content": "用户参加了一次评审", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c7-具体时间锚点(陷阱负例)",
        "batch_text": "\n".join([
            "[阿哲] 上周五客户说要换主色。",
            "[小林] 下周一我给他答复。",
        ]),
        "nodes": [
            {"id": "n1", "type": "thing", "content": "上周五", "label": "clean"},
            {"id": "n2", "type": "thing", "content": "下周一", "label": "clean"},
            {"id": "n3", "type": "status", "content": "客户说要换主色", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c8-已纳入映射的近期表述(陷阱负例)",
        "batch_text": "[小林] 我前几天在改 bug，这两天快弄完了。",
        "nodes": [
            {"id": "n1", "type": "thing", "content": "前几天", "label": "clean"},
            {"id": "n2", "type": "thing", "content": "这两天", "label": "clean"},
            {"id": "n3", "type": "status", "content": "小林在改 bug 且快完成了", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c9-多件事合并",
        "batch_text": "\n".join([
            "[老周] 客户上周五说要换主色。",
            "[阿哲] 我昨天把方案改了。",
            "[小林] 我今天把测试跑完了。",
        ]),
        "nodes": [
            {"id": "n1", "type": "status",
             "content": "老周说客户要换主色、阿哲改了方案、小林跑完了测试", "label": "granularity"},
            {"id": "n2", "type": "thing", "content": "上周五", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c10-漏抽为什么",
        "batch_text": "\n".join([
            "[用户] 我本来想直接上云，省事。",
            "[助手] 那为什么没上云？",
            "[用户] 云的成本太高，预算砍了一半，所以放弃了。",
        ]),
        "nodes": [
            {"id": "n1", "type": "status", "content": "用户最终没有采用上云方案", "label": "clean"},
        ],
        "expect_missing_any": [("预算", "成本")],
    },
    {
        "id": "c11-未发生写成已发生",
        "batch_text": "[用户] 我本来想报个健身班，后来没报。",
        "nodes": [
            {"id": "n1", "type": "status", "content": "用户报了健身班", "label": "unsupported"},
            {"id": "n2", "type": "status", "content": "用户考虑过报健身班但最终没报", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c12-凭空编造冲突",
        "batch_text": "[小林] 我对这次分配有点不爽，但没说。",
        "nodes": [
            {"id": "n1", "type": "status", "content": "小林因为分配的事和别人吵了一架", "label": "unsupported"},
            {"id": "n2", "type": "status", "content": "小林对这次分配不太满意", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c13-过几天vs明天",
        "batch_text": "[阿哲] 过几天我把稿子发你。明天我先跟客户对一下需求。",
        "nodes": [
            {"id": "n1", "type": "thing", "content": "过几天", "label": "granularity"},
            {"id": "n2", "type": "thing", "content": "明天", "label": "clean"},
            {"id": "n3", "type": "status", "content": "阿哲过几天把稿子发给对方", "label": "clean"},
            {"id": "n4", "type": "status", "content": "阿哲明天和客户对接需求", "label": "clean"},
        ],
        "expect_missing_any": [],
    },
    {
        "id": "c14-漏抽第二个原因",
        "batch_text": "[老周] 客户要换主色，因为他们的品牌刚升级。",
        "nodes": [
            {"id": "n1", "type": "status", "content": "客户要换主色", "label": "clean"},
        ],
        "expect_missing_any": [("品牌", "升级")],
    },
]


# ── 时间词扫描：ground truth 直接取自检索侧词表，避免"人工标准"与系统漂移 ──
# 判定规则与系统设计同源：只要该词能被 检索侧 识别为一种时间锚点
# （显式时期 / 模糊映射 / 人生阶段 / 仅锚点），独立成 thing 节点就是**既定设计**，不该被报；
# 只有系统完全不认识的"无法定位"表述（「过两天」「回头」…）才该报 granularity。
SWEEP_GROUPS = {
    "A-显式日周月年": [
        "今天", "明天", "昨天", "前天", "上周", "这周", "本周", "下周", "上周三", "周三",
        "上个月", "这个月", "下个月", "去年", "今年", "明年", "前年",
        "寒假", "暑假", "上学期", "下学期", "上半年", "下半年", "年底", "年初",
    ],
    "B-模糊_阶段_仅锚点": [
        "前几天", "前两天", "这两天", "这几天", "那天", "当天", "当时", "后来", "之后",
        "第二天", "次日", "高中", "大学", "大专", "高一", "大二", "大四",
        "凌晨", "早上", "上午", "中午", "下午", "晚上", "毕业", "入职",
        "最近", "近来", "这段时间", "那段时间", "近年来",
    ],
    "C-无法定位(该报)": [
        "过两天", "过几天", "过阵子", "回头", "改天", "以后", "晚点", "有空", "再说", "将来",
    ],
}


def expected_time_label(word: str) -> str:
    """该词独立成 thing 节点时，按**系统既定设计**它应否被报为 granularity"""
    from dba_pipeline.retrieval.retriever import (
        TOOL_TIME_ANCHOR_RE, _era_periods, _explicit_periods, _fuzzy_periods,
    )
    mapped = bool(_explicit_periods(word) or _fuzzy_periods(word) or _era_periods(word))
    recognized = bool(TOOL_TIME_ANCHOR_RE.search(word))
    return "clean" if (mapped or recognized) else "granularity"


def build_sweep_cases():
    cases = []
    for gid, words in SWEEP_GROUPS.items():
        nodes, lines = [], []
        for i, w in enumerate(words, 1):
            nodes.append({"id": f"w{i}", "type": "thing", "content": w,
                          "label": expected_time_label(w)})
            lines.append(f"[用户] {w}我把这件事处理一下。")
        cases.append({"id": f"sw-{gid}", "batch_text": "\n".join(lines),
                      "nodes": nodes, "expect_missing_any": []})
    return cases


def run_sweep(engine, rounds: int):
    """扫描每个时间词被误报的比例，返回 {word: {expected, flagged, rounds}}"""
    stats = {}
    for case in build_sweep_cases():
        nodes_text = build_nodes_text(case["nodes"])
        for n in case["nodes"]:
            stats.setdefault(n["content"], {"expected": n["label"], "flagged": 0, "rounds": 0})
        for _ in range(rounds):
            v = engine.audit_source(case["batch_text"], nodes_text)
            rep = {i["id"] for i in (v.get("granularity") or []) + (v.get("unsupported") or [])
                   if isinstance(i, dict) and i.get("id")}
            for n in case["nodes"]:
                st = stats[n["content"]]
                st["rounds"] += 1
                if n["id"] in rep:
                    st["flagged"] += 1
    return stats


def build_nodes_text(nodes) -> str:
    """与 `DBAServer._snapshot_batch_nodes` 同格式（本基准不带 evidence 行）"""
    return "\n".join(f"[{n['id']}]（{n['type']}）{n['content']}" for n in nodes)


def build_engine():
    from langchain_openai import ChatOpenAI
    from dba_pipeline.llm.inference import InferenceEngine

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", ""),
        api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        base_url=os.environ.get("OPENAI_API_BASE"),
        temperature=0,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return InferenceEngine(llm)


def score_case(case: dict, verdict: dict) -> dict:
    """按节点 id 精确匹配。

    类别错放也算错：标为 unsupported 却被报成 granularity → unsupported 记 FN，
    granularity 记 FP（说明它把真问题的性质也判歪了）。
    """
    rep_un = {i["id"] for i in (verdict.get("unsupported") or []) if isinstance(i, dict) and i.get("id")}
    rep_gr = {i["id"] for i in (verdict.get("granularity") or []) if isinstance(i, dict) and i.get("id")}
    rep_any = rep_un | rep_gr

    tp = {"unsupported": set(), "granularity": set()}
    fp = {"unsupported": set(), "granularity": set()}
    fn = {"unsupported": set(), "granularity": set()}

    for n in case["nodes"]:
        nid, label = n["id"], n["label"]
        if label == "clean":
            if nid in rep_un:
                fp["unsupported"].add(nid)
            if nid in rep_gr:
                fp["granularity"].add(nid)
        elif label == "unsupported":
            if nid in rep_un:
                tp["unsupported"].add(nid)
            else:
                fn["unsupported"].add(nid)
                if nid in rep_gr:
                    fp["granularity"].add(nid)      # 报错了类别
        elif label == "granularity":
            if nid in rep_gr:
                tp["granularity"].add(nid)
            else:
                fn["granularity"].add(nid)
                if nid in rep_un:
                    fp["unsupported"].add(nid)      # 报错了类别

    # missing 只测召回：期望关键词出现在任一 quote 里即命中
    quotes = " ".join(str(i.get("quote") or "") for i in (verdict.get("missing") or [])
                      if isinstance(i, dict))
    miss_total = len(case["expect_missing_any"])
    miss_hit = sum(1 for kws in case["expect_missing_any"] if any(k in quotes for k in kws))

    # 干净节点误报：只有"标注为 clean 的节点被报"才算（rep_any 含真问题节点，不能直接当分子）
    clean_ids = {n["id"] for n in case["nodes"] if n["label"] == "clean"}

    return {"tp": tp, "fp": fp, "fn": fn,
            "clean_reported": len(clean_ids & rep_any),
            "missing_hit": miss_hit, "missing_total": miss_total,
            "error": verdict.get("error")}


def main() -> None:
    _load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_MODEL"):
        print("错误: 未配置 OPENAI_MODEL（.env 或环境变量）", file=sys.stderr)
        sys.exit(2)

    print("# C3 溯源巡检准确性基准（precision / recall）")
    n_nodes = sum(len(c["nodes"]) for c in CASES)
    n_clean = sum(1 for c in CASES for n in c["nodes"] if n["label"] == "clean")
    n_issue = n_nodes - n_clean
    print(f"# {len(CASES)} 个用例 / {n_nodes} 个节点（{n_clean} 干净 + {n_issue} 预置问题）"
          f" / 每用例 {ROUNDS} 轮\n")

    engine = build_engine()
    agg = {
        "unsupported": Counter(), "granularity": Counter(),
        "clean_nodes": 0, "clean_reported": 0,
        "missing_hit": 0, "missing_total": 0,
        "errors": 0, "runs": 0,
    }
    per_case = []
    examples = {"fp": [], "fn": []}

    for case in CASES:
        nodes_text = build_nodes_text(case["nodes"])
        clean_ids = {n["id"]: n["content"] for n in case["nodes"] if n["label"] == "clean"}
        id2content = {n["id"]: n["content"] for n in case["nodes"]}
        verdicts = []
        for _ in range(ROUNDS):
            v = engine.audit_source(case["batch_text"], nodes_text)
            s = score_case(case, v)
            verdicts.append(s)
            agg["runs"] += 1
            if s["error"]:
                agg["errors"] += 1
            agg["clean_nodes"] += len(clean_ids)
            agg["clean_reported"] += s["clean_reported"]
            agg["missing_hit"] += s["missing_hit"]
            agg["missing_total"] += s["missing_total"]
            for cat in ("unsupported", "granularity"):
                agg[cat]["tp"] += len(s["tp"][cat])
                agg[cat]["fp"] += len(s["fp"][cat])
                agg[cat]["fn"] += len(s["fn"][cat])

            for cat in ("unsupported", "granularity"):
                for nid in sorted(s["fp"][cat]):
                    if len(examples["fp"]) < 6:
                        examples["fp"].append(
                            f"{case['id']} / [{cat}] {nid}「{id2content.get(nid, '')}」")
                for nid in sorted(s["fn"][cat]):
                    if len(examples["fn"]) < 6:
                        c = next((n["content"] for n in case["nodes"] if n["id"] == nid), "")
                        examples["fn"].append(f"{case['id']} / [{cat}] {nid}「{c}」")

        # 跨轮稳定性：每轮的报告（按类别）是否完全一致
        def _sig(s):
            return (tuple(sorted(s["fp"]["unsupported"])), tuple(sorted(s["fp"]["granularity"])),
                    tuple(sorted(s["fn"]["unsupported"])), tuple(sorted(s["fn"]["granularity"])))
        stable = len({_sig(s) for s in verdicts}) == 1
        tp_u = sum(len(s["tp"]["unsupported"]) for s in verdicts)
        tp_g = sum(len(s["tp"]["granularity"]) for s in verdicts)
        per_case.append({
            "id": case["id"], "stable_across_rounds": stable,
            "unsupported_tp": tp_u, "granularity_tp": tp_g,
            "fp_total": sum(len(s["fp"]["unsupported"]) + len(s["fp"]["granularity"])
                            for s in verdicts),
            "missing_hit": sum(s["missing_hit"] for s in verdicts),
            "missing_total": sum(s["missing_total"] for s in verdicts),
        })
        print(f"  {case['id']:26} 稳定={'是' if stable else '否'}  "
              f"un 命中={tp_u:2}  gr 命中={tp_g:2}  误报={per_case[-1]['fp_total']:2}  "
              f"missing={per_case[-1]['missing_hit']}/{per_case[-1]['missing_total']}")

    # ── 汇总 ──
    def prf(c):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) and p == p and r == r else float("nan")
        return tp, fp, fn, p, r, f1

    print("\n" + "=" * 72)
    print("按类别的准确率（跨用例 × 跨轮累计）:")
    summary = {}
    for cat in ("unsupported", "granularity"):
        tp, fp, fn, p, r, f1 = prf(agg[cat])
        summary[cat] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
        print(f"  {cat:12}: TP={tp:3} FP={fp:3} FN={fn:3}  "
              f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    fp_rate = agg["clean_reported"] / agg["clean_nodes"] if agg["clean_nodes"] else float("nan")
    miss_r = agg["missing_hit"] / agg["missing_total"] if agg["missing_total"] else float("nan")
    print(f"  干净节点误报率 : {agg['clean_reported']}/{agg['clean_nodes']} = {fp_rate:.3f}")
    print(f"  missing 召回    : {agg['missing_hit']}/{agg['missing_total']} = {miss_r:.3f}")
    print(f"  解析失败次数    : {agg['errors']}/{agg['runs']}")
    stable_n = sum(1 for c in per_case if c["stable_across_rounds"])
    print(f"  跨轮结果稳定    : {stable_n}/{len(CASES)} 个用例（其余会在不同轮给出不同报告）")

    if examples["fp"]:
        print("\n误报样例（干净节点被报）:")
        for e in examples["fp"]:
            print(f"  - {e}")
    if examples["fn"]:
        print("\n漏报样例（预置问题未被报）:")
        for e in examples["fn"]:
            print(f"  - {e}")

    # ── 时间词扫描：还有哪些"没写进排除清单却被误报"的词 ──
    print("\n" + "=" * 72)
    print(f"时间词扫描（每个词独立成 thing 节点，跑 {SWEEP_ROUNDS} 轮；"
          "ground truth 取自检索侧词表）")
    sweep = run_sweep(engine, SWEEP_ROUNDS)
    n_words = len(sweep)
    over = {w: s for w, s in sweep.items() if s["expected"] == "clean" and s["flagged"]}
    under = {w: s for w, s in sweep.items() if s["expected"] == "granularity" and not s["flagged"]}
    print(f"  扫描 {n_words} 个词；**误报词 {len(over)} 个**，**漏报词 {len(under)} 个**")
    if over:
        print("  误报（系统认得、不该报，却被报）——按误报轮数降序:")
        for w, s in sorted(over.items(), key=lambda kv: -kv[1]["flagged"]):
            print(f"    {w:8} 误报 {s['flagged']}/{s['rounds']} 轮")
    if under:
        print("  漏报（系统不认识、该报，却没报）:")
        for w, s in sorted(under.items()):
            print(f"    {w:8} 0/{s['rounds']} 轮")
    if not over and not under:
        print("  ✓ 无偏差：该报的都报了、不该报的都没报")

    out = ROOT / "eval" / "outputs" / "c3_audit_accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rounds_per_case": ROUNDS,
        "summary": summary,
        "clean_fp_rate": round(fp_rate, 3),
        "clean_nodes": agg["clean_nodes"], "clean_reported": agg["clean_reported"],
        "missing_recall": round(miss_r, 3),
        "missing_hit": agg["missing_hit"], "missing_total": agg["missing_total"],
        "parse_errors": agg["errors"], "runs": agg["runs"],
        "stable_cases": f"{stable_n}/{len(CASES)}",
        "per_case": per_case,
        "examples": examples,
        "sweep": {"rounds": SWEEP_ROUNDS, "words": sweep,
                  "false_alarm_words": sorted(over), "missed_words": sorted(under)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写: {out}")

    print("\n" + "=" * 72)
    print("读法：C3 的 prompt 纪律是「发现可疑就报，漏报代价高于误报」——")
    print("所以 **高召回、允许一定误报** 是**设计预期**，不是缺陷；")
    print("但误报率若高到淹没真问题（干净节点误报率 ≫ 0.2），复核者会被噪声劝退，就该收紧判据。")
    print("具体问题是否真被上报，单轮不算数——请看上面的跨轮稳定性与 F1。")


if __name__ == "__main__":
    main()
