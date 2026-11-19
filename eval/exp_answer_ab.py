# -*- coding: utf-8 -*-
"""剥离内联选项前后：检索 → StoryRank → 本地 answer 模型，看选中哪一项。

A 组 = 平台原始 query（题干 + 全部选项 + 作答指令，内联在一坨里）——即 `_strip_option_block`
        上线**前**的行为；
B 组 = 只题干——即上线**后**的行为。

两组的检索参数完全一致（seed_k 按图规模、max_hops=5、expand_k=40、容忍带/目的阈值用默认），
只有进检索的文本不同；目的推断各自按自己的输入推一次（这正是改动要修正的第二处）。

answer 模型 = 本地 .env 配的 OPENAI_MODEL（与链路内部 LLM 同源，仅作"能不能选中"的判据）。

用法：
    python -u eval/exp_answer_ab.py                 # enemy + angry（angry 的官方答案是已知的，做对照）
    python -u eval/exp_answer_ab.py --only enemy
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

import dba_pipeline.llm.inference as _inf  # noqa: E402
from alm.space import _strip_option_block  # noqa: E402
from eval.run_p_baseline import _load_dotenv, build_retriever  # noqa: E402

_load_dotenv(ROOT / ".env")

# 判定 F 的三段必要证据是否**真的写进了正文**（不是进没进图）：
#   怀疑→赞赏的移动、最后一句咒誓（foremost man）、随后的社交靠拢（testimonial/举杯）
F_EVIDENCE = ("foremost man", "testimonial", "Aslaksen", "Your health", "good health",
              "pagan", "incredible", "country patient")
# angry 的官方答案（B）依赖"公寓前门被链锁"，看这段事实在序列里的位置
ANG_EVIDENCE = ("chain", "front door", "stairway", "the hall", "locked")

# 证据阶梯：手工把证据**直接放进记忆**，看模型还选不选得对。
# 目的：把"证据不足"与"模型读法"两件事分开——如果连显式编排过的证据都选不对，
# 那就不是链路缺料，而是 answer 模型这一层的问题。
EV8 = ["n181", "n250", "n276", "n277", "n295", "n325", "n385", "n388"]
EV6 = ["n250", "n276", "n295", "n385", "n388", "n396"]
# "命中组"＝上一轮**正文里真的出现过**的那几项（关键词 foremost man / testimonial /
# Aslaksen / pagan / country patient 对应的节点）。用来判定：是"这几项本身就把模型
# 引向 A"，还是"被别的枝（无关节点）稀释/带偏"。
EV_HIT = ["n250", "n276", "n385", "n388", "n391"]
# 编排用的**纯时间连接词**：只标注先后，不给这个变化命名（与规则 6 的"可并置、不可命名"一致）
_CONNECTIVES = ["起初，", "后来，", "随后，", "接着，", "再后来，", "此后，", "最后，", "最终，"]

# 证据集合注册表：键 →（标签, 节点 id 列表, 是否加时间连接词）
SETS = {
    "empty": ("空记忆（仅先验）", [], False),
    "hit5_raw": ("5项·命中组（原文）", EV_HIT, False),
    "hit5_arr": ("5项·命中组（编排）", EV_HIT, True),
    "six_raw": ("6项·原文罗列", EV6, False),
    "six_arr": ("6项·编排（只加时间连接词）", EV6, True),
    "eight_raw": ("8项·原文罗列", EV8, False),
    "eight_arr": ("8项·编排（只加时间连接词）", EV8, True),
}

R3 = ROOT / "eval/.cache/alm_data_r3"
CACHE = ROOT / "eval/.cache"

ANSWER_SYS = (
    "You are answering a multiple-choice question strictly from the provided memory. "
    "The memory is the only admissible evidence. Reply with ONLY the option letter in "
    "parentheses, e.g., (F). No explanation."
)

RT = re.compile(r"\(([A-H])\)")


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _find_query(marker: str) -> str:
    """从 purposes_cache 里取平台原样下发的 query（日志 `[IO] search 入参 query=` 抓下来的那条）"""
    data = json.loads((CACHE / "purposes_cache.json").read_text(encoding="utf-8"))
    hits = [k for k in data if marker in k and "\n" not in k]
    if not hits:
        raise SystemExit(f"purposes_cache 里找不到 {marker!r} 的平台 query")
    return max(hits, key=len)


CASES = {
    "enemy": {
        "marker": "Billing",
        "yaml": R3 / "u_886f4595eb0ad67ff1d6d33535f90310ba1c58dfb77a64b31f00da9d66572c39.yaml",
        "cache": "alm_r3_u_886f4595eb",
        "gold": "F（推断；平台实测官方 ≠ C）",
        "evidence": F_EVIDENCE,
    },
    "angry": {
        "marker": "chain-locked",
        "yaml": R3 / "u_1cea534932bcf690dfaecb2439a59e00dee296ac195c59768ac1d42d8bca7fa8.yaml",
        "cache": "alm_u_1cea534932",
        "gold": "B（官方，平台实测 1.0）",
        "evidence": ANG_EVIDENCE,
    },
}


def _answer(llm, memory: str, question_blob: str) -> str:
    msg = [
        SystemMessage(content=ANSWER_SYS),
        HumanMessage(content=f"MEMORY:\n{memory}\n\nQUESTION:\n{question_blob}"),
    ]
    out = llm.invoke(msg)
    text = getattr(out, "content", str(out))
    m = RT.search(text or "")
    if m:
        return f"({m.group(1)})"
    m = re.search(r"\b([A-H])\b", text or "")
    return f"({m.group(1)})" if m else "（未给出字母）"


def _build_answer_llm(model: str):
    """answer 模型（判题用）：沿用仓库既有口径 ANSWER_* → JUDGE_* → OPENAI_*。

    链路内部的 deepseek-v4-flash 只是**记忆整理**模型，不一定是合格的判题模型；
    `--answer-model` 用来换成 .env 里已配好的更强网关（JUDGE_API_BASE）。
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=(os.environ.get("ANSWER_API_KEY") or os.environ.get("JUDGE_API_KEY")
                 or os.environ.get("OPENAI_API_KEY") or "not-needed"),
        base_url=(os.environ.get("ANSWER_API_BASE") or os.environ.get("JUDGE_API_BASE")
                  or os.environ.get("OPENAI_API_BASE")),
        temperature=0,
        # 单次请求超时（与 alm/engine.py 同口径）：不设就吃 SDK 默认 600s，网关挂死时
        # 整个跑批会卡住（0930 实测 >10 分钟，且无任何进度输出）。
        timeout=120.0,
    )


def _trace_path(r, res, label: str, dump: list, evidence) -> None:
    """把**实际喂给 StoryRank 的 path 序列**打出来，定位证据在序列中的位置。

    正文是模型按这个顺序写的（规则 6：最相关的写在最前），所以「证据在第几位、
    前面累计多少字符」直接决定它能不能落进 `_STORY_MAX_CHARS` 的预算里。
    顺序 = hop 根→叶（见 `_build_path` 注释），不是分数序。
    """
    from eval.run_p_baseline import build_path

    nodes = (build_path(r, res) or {}).get("nodes") or []
    print(f"   path 节点 {len(nodes)} 条（顺序=hop 根→叶）")
    cum, lines, has_ev = 0, [], False
    for i, nd in enumerate(nodes, 1):
        c = nd.get("content") or ""
        cum += len(c) + 1
        hit = next((k for k in evidence if k.lower() in c.lower()), "")
        if hit:
            has_ev = True
        flag = f"  <<< 证据:{hit}" if hit else ""
        lines.append(f"#{i:>4} 累计{cum:>6}字 score={nd.get('score', 0.0):.3f} "
                     f"{nd.get('id'):>5} {c[:200]}{flag}")
        if hit or i <= 5:
            print(f"     #{i:>4} 累计{cum:>6}字 score={nd.get('score', 0.0):.3f} "
                  f"{nd.get('id'):>5} {c[:70]}{flag}")
    dump.append(f"\n##### [{label}] path 序列（共 {len(nodes)} 条）\n" + "\n".join(lines) + "\n")
    print(f"   证据是否出现在 path 里：{'是（见 <<< 标记）' if has_ev else '否'}")


def run_evidence_ladder(name: str, answer_models=(), sets=None, repeat: int = 1) -> None:
    """手工把证据编排进记忆，逐级看模型的选择（不检索、不叙事）。

    `repeat > 1` 用于看**稳定性**：同一份记忆问 N 次，答案是否在抖。
    """
    cfg = CASES[name]
    full = _find_query(cfg["marker"])
    graph, r = build_retriever(str(cfg["yaml"]), vector_cache_key=cfg["cache"])

    llms = [("deepseek-v4-flash(链路内部)", r.llm)]
    llms += [(m, _build_answer_llm(m)) for m in answer_models]

    keys = list(sets) if sets else list(SETS)
    print(f"\n### {name}   官方 = {cfg['gold']}   每条件重复 {repeat} 次")
    print("说明：记忆只含人工挑出的证据本身，不带任何检索/叙事产物。")

    dump, table = [], []
    for key in keys:
        label, ids, arranged = SETS[key]
        sents = [c for c in ((graph.get_content(nid) or "").strip() for nid in ids) if c]
        if not sents:
            mem = "（没有任何记忆，只凭题目本身与你的知识回答）"
        elif arranged:
            mem = " ".join(a + s for a, s in zip(_CONNECTIVES, sents))
        else:
            mem = "\n".join(sents)
        print(f"\n===== {label} ===== 记忆 {len(mem)} 字、{len(sents)} 条")
        if sents:
            print("   " + " | ".join(s[:46] for s in sents))
        dump.append(f"##### {label}  记忆 {len(mem)} 字\n{mem}\n")
        picks = {mlabel: [] for mlabel, _ in llms}
        for _ in range(repeat):
            for mlabel, llm in llms:
                picks[mlabel].append(_answer(llm, mem, full))
        for mlabel, _ in llms:
            seq = picks[mlabel]
            print(f"   {mlabel:<28} {' '.join(seq)}   [{'稳定' if len(set(seq)) == 1 else '抖'}]")
        dump.append("".join(f"{m}: {' '.join(v)}\n" for m, v in picks.items()) + "\n")
        table.append((label, picks))

    out_dir = ROOT / "eval/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"evidence_ladder_{name}.txt").write_text("\n".join(dump), encoding="utf-8")
    print(f"\n{'—' * 100}\n{name} 汇总（官方 = {cfg['gold']}）："
          f"A=否认有变化 / E=谴责但只是短暂烦躁 / F=怀疑→赞赏→靠拢")
    print("  " + f"{'条件':<26}" + "".join(f"{m:<28}" for m, _ in llms))
    for label, picks in table:
        print("  " + f"{label:<26}" + "".join(f"{' '.join(picks[m]):<28}" for m, _ in llms))
    print(f"  记忆原文已写: eval/outputs/evidence_ladder_{name}.txt")


def run_case(name: str, extra_budgets=(), controls_only: bool = False,
             answer_model: str = None, trace: bool = False, max_hops: int = 5,
             max_peak_nodes: int = None, no_controls: bool = False,
             repeat: int = 1) -> None:
    cfg = CASES[name]
    full = _find_query(cfg["marker"])
    stem = _strip_option_block(full)
    print(f"\n{'=' * 100}\n### {name}   官方答案 = {cfg['gold']}")
    print(f"平台 query {len(full)} 字 → 检索用题干 {len(stem)} 字")
    print(f"题干: {_clean(stem)}")
    assert stem != full, "剥离没生效，检查 _strip_option_block"

    vecs = pickle.load(open(CACHE / f"{cfg['cache']}_vectors.pkl", "rb"))
    graph, r = build_retriever(str(cfg["yaml"]), vector_cache_key=cfg["cache"])
    llm = _build_answer_llm(answer_model) if answer_model else r.llm
    print(f"answer 模型 = {answer_model or '链路内部 ' + str(getattr(r.llm, 'model_name', '?'))}")
    total = graph.node_count
    seed_k = max(5, min(40, total // 10))
    ids = [nid for nid in graph.graph.nodes() if nid in vecs]
    M = np.array([vecs[nid] for nid in ids], dtype=float)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    contents = {nid: (graph.graph.nodes[nid].get("content") or "") for nid in ids}
    print(f"图节点 {total}，有向量的 {len(ids)}，seed_k={seed_k}")

    rows, dump = [], []
    conds = [("A 带选项", full, None), ("B 只题干", stem, None)]
    conds += [(f"C 只题干 预算{b}", stem, b) for b in extra_budgets]
    for label, q, budget in conds if not controls_only else []:
        # 模块级常量：story_rank 内部按名字读它。每次都显式设回，避免跨条件/跨案例残留。
        _inf._STORY_MAX_CHARS = budget or 2000

        qv = np.asarray(r._embed(q), dtype=float)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        s = M @ qv
        order = np.argsort(-s)
        top = ", ".join(f"{ids[k]}({s[k]:.3f})" for k in order[:5])

        purposes = (r.inference.infer_purpose(q) or {}).get("purposes") or []
        res = r.retrieve_with_story(
            q, with_response=False, seed_k=seed_k, max_hops=max_hops, expand_k=40,
            purpose=purposes or None, max_peak_nodes=max_peak_nodes,
        )
        story = (res.get("stories") or [""])[0]
        # 判题重复 N 次（同一份叙事）→ 看的是**判定稳定性**，不是检索稳定性。
        pick = " ".join(_answer(llm, story, full) for _ in range(repeat))
        hit = [k for k in cfg["evidence"] if k.lower() in story.lower()]
        rows.append((label, len(story), res.get("path_nodes", 0), res.get("path_edges", 0),
                     purposes, top, pick, hit))
        dump.append(f"##### {label}  选中={pick}  正文={len(story)}字  "
                    f"F 证据命中={hit}\n{story}\n")
        print(f"\n-- {label} --")
        print(f"   目的推断: {purposes}")
        print(f"   种子 top5: {top}")
        print(f"   喂入点={res.get('path_nodes', 0)} 喂入边={res.get('path_edges', 0)} "
              f"叙事={len(story)} 字  F证据命中={hit}")
        print(f"   叙事开头: {_clean(story)[:150]}")
        print(f"   叙事结尾: …{_clean(story)[-150:]}")
        if trace:
            _trace_path(r, res, f"{name}/{label}", dump, cfg["evidence"])

    # 两端对照（不检索、每题 1 次调用）：
    #   D 零记忆 —— 看本地 answer 模型的先验，区分"选了 E 是因为读记忆"还是"模型本来就偏向 E"
    #   E 全图原文 —— 记忆上限：把 400 个节点原文全给它。若它仍选 E，说明瓶颈不在叙事
    all_text = "\n".join(contents[nid] for nid in ids)
    for label, mem in (() if no_controls else (
            ("D 零记忆(仅先验)", "（没有任何记忆，只凭题目本身与你的知识回答）"),
            ("E 上限对照=全图原文", all_text))):
        pick = " ".join(_answer(llm, mem, full) for _ in range(repeat))
        rows.append((label, len(mem), 0, 0, [], "", pick, []))
        dump.append(f"##### {label}  选中={pick}  正文={len(mem)}字\n{mem[:3000]}\n")
        print(f"\n-- {label} --  正文={len(mem)}字  选中={pick}")

    out_dir = ROOT / "eval/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"answer_ab_{name}.txt").write_text("\n".join(dump), encoding="utf-8")

    print(f"\n{'—' * 100}\n{name} 结论：官方 = {cfg['gold']}")
    for label, slen, pn, pe, _purposes, _top, pick, hit in rows:
        tag = ""
        if repeat > 1:
            seq = pick.split()
            tag = " [稳定]" if len(set(seq)) == 1 else " [抖]"
        print(f"  {label:<16} 叙事={slen:>5}字 喂入点={pn:>3} 选中={pick}{tag} "
              f"F证据={len(hit)}/{len(cfg['evidence'])}")
    print(f"  全文已写: eval/outputs/answer_ab_{name}.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=["enemy", "angry"])
    ap.add_argument("--budgets", type=int, nargs="*", default=[],
                    help="额外条件：只题干 + 该篇幅预算（默认 2000，即生产值）")
    ap.add_argument("--controls-only", action="store_true",
                    help="只跑两端对照（零记忆 / 全图原文），不跑检索条件")
    ap.add_argument("--answer-model", default=None,
                    help="answer 模型名（默认用链路内部模型；给名字则走 ANSWER_*/JUDGE_* 网关）")
    ap.add_argument("--trace", action="store_true",
                    help="打印喂给 StoryRank 的 path 序列，定位证据在第几位/累计多少字符")
    ap.add_argument("--answer-models", default="",
                    help="逗号分隔的 answer 模型名单（走 ANSWER_*/JUDGE_* 网关）")
    ap.add_argument("--evidence-ladder", action="store_true",
                    help="手工把 8 项/6 项证据编排进记忆，逐级看模型选择（不检索）")
    ap.add_argument("--sets", default="",
                    help=f"证据集合（逗号分隔）；可选 {','.join(SETS)}；默认全部")
    ap.add_argument("--repeat", type=int, default=1, help="每个条件重复次数（看稳定性）")
    ap.add_argument("--max-hops", type=int, default=5,
                    help="检索展开跳数（1 = 模拟精确路由把读出收拢到根与 1 hop 内）")
    ap.add_argument("--max-peak-nodes", type=int, default=None,
                    help="读出条数上限（按 combined_score 取前 N；模拟精确路由的收拢）")
    ap.add_argument("--no-controls", action="store_true", help="跳过零记忆/全图原文两端对照")
    args = ap.parse_args()
    models = tuple(m.strip() for m in args.answer_models.split(",") if m.strip())
    sets = tuple(s.strip() for s in args.sets.split(",") if s.strip())
    for name in ([args.only] if args.only else ["enemy", "angry"]):
        if args.evidence_ladder:
            run_evidence_ladder(name, models, sets or None, args.repeat)
            continue
        run_case(name, tuple(args.budgets), controls_only=args.controls_only,
                 answer_model=args.answer_model, trace=args.trace, max_hops=args.max_hops,
                 max_peak_nodes=args.max_peak_nodes, no_controls=args.no_controls,
                 repeat=args.repeat)


if __name__ == "__main__":
    main()
