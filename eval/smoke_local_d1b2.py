# -*- coding: utf-8 -*-
"""本地针对性烟雾：直接驱动 MemorySpace，重点验这几件事

① 语言：英文输入 → 节点与叙事是否保持英文（不再被译成中文）
② 节点膨胀：每个节点是否仍是单一事实、内容长度是否受控
③ 分批：条数 + 字符双约束是否生效（各组规模）
④ 值变更（D1）：换工作/换技术栈这类值变化是否触发 update / deprecate
⑤ 文档通道：长文档是否不进图、只进文档区
⑥ 边型（B2）：causal 与 scenario 的占比、跨批连边是否发生

真实调用 LLM + embedding（用 .env 里的生产配置），不是 mock。
"""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"c:\Users\makot\Desktop\Ariadne")
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
# 强制走 API embedding（与线上一致）。根 .env 的 EMBEDDING_LOCAL=1 需要本地模型，
# 而 bge-m3 有 2.2GB 且未缓存——本地跑本烟雾没有理由拉它。
os.environ["EMBEDDING_LOCAL"] = "false"

from alm.config import ALMConfig
from alm.contract import AddMessage
from alm.engine import ALMEngine
from alm.space import MemorySpace

CJK = re.compile(r"[\u3400-\u9fff]")
LAT = re.compile(r"[A-Za-z]")
DIRECTED = {"causal", "sequence", "preference", "temporal", "taxonomic"}


def cjk_ratio(s: str) -> float:
    c, l = len(CJK.findall(s)), len(LAT.findall(s))
    return (c / (c + l)) if (c + l) else 0.0


def M(role, content, ts=None):
    return AddMessage(role=role, content=content, timestamp=ts)


cfg = ALMConfig.from_env()
print("=" * 96)
print("配置")
print("=" * 96)
print(f"  llm={cfg.llm_model}  embedding={cfg.embedding_model}")
print(f"  add_batch_size={cfg.add_batch_size}  add_max_chars={cfg.add_max_chars}  "
      f"doc_route={cfg.doc_route}  doc_chunk_chars={cfg.doc_chunk_chars}")
import dba_pipeline.extraction.dba as dba_mod
print(f"  _TRIAGE_TRUST_MAX_CHARS={dba_mod._TRIAGE_TRUST_MAX_CHARS}  "
      f"_MAX_CHARS(embed)={__import__('dba_pipeline.embedding.store', fromlist=['x']).OpenAIEmbeddings._MAX_CHARS}")

tmp = Path(tempfile.mkdtemp(prefix="alm_smoke_"))
llm = ALMEngine._build_llm(cfg)
emb = ALMEngine._build_embeddings(cfg)
sp = MemorySpace(user_id="smoke:d1b2", config=cfg, embeddings=emb, llm=llm,
                 yaml_path=tmp / "smoke.yaml")

# 记录每次（每组）maintain 的真实 LLM 产出，用于统计值变更
calls = []
_orig = sp.dba.maintain


def _rec(conversation, timestamp=None, source_rounds=None):
    out = _orig(conversation, timestamp=timestamp, source_rounds=source_rounds)
    calls.append({"chars": len(conversation), "out": out})
    return out


sp.dba.maintain = _rec

B1 = [  # 建立基线
    ("user", "Hi, I'm a backend developer at an internet company here in Hangzhou."),
    ("assistant", "Got it. What does your stack look like?"),
    ("user", "Mostly Go and PostgreSQL these days."),
    ("assistant", "How long have you been on that team?"),
    ("user", "About three years. I've been doing a lot of overtime lately."),
    ("assistant", "That sounds rough. How is your body holding up?"),
    ("user", "My neck is killing me from sitting at the desk all day."),
    ("assistant", "You should look into a better ergonomic setup."),
]
B2 = [  # 值变更（D1）
    ("user", "Big news — I switched teams last week."),
    ("assistant", "Oh? What changed?"),
    ("user", "I moved to the data platform team, so I'm not writing Go anymore."),
    ("assistant", "What are you using now?"),
    ("user", "Python and Spark, mostly."),
    ("assistant", "How do you like it?"),
    ("user", "It's much better for my neck actually, less frantic."),
    ("assistant", "Good to hear."),
]
B3 = [  # 因果链（B2）
    ("user", "Remember the neck pain I told you about?"),
    ("assistant", "Yes, from all the overtime."),
    ("user", "I finally bought an ergonomic chair last weekend."),
    ("assistant", "That should help a lot."),
    ("user", "It is helping — the pain is much less now."),
    ("assistant", "Glad to hear it."),
    ("user", "I also started doing stretches every morning."),
    ("assistant", "That's a good habit to build."),
]
DOC = (
    "SCENE.—DR. STOCKMANN'S sitting-room. It is evening. The room is plainly but neatly "
    "appointed and furnished. Mrs. Stockmann: You see, if you come an hour late, Mr. Billing, "
    "you have to put up with cold meat. Billing: It is uncommonly good, thank you—remarkably good. "
    "Mrs. Stockmann: My husband makes such a point of having his meals punctually, you know. "
) * 10

TS = 1704153600000
for i, (label, batch) in enumerate([("批1 基线", B1), ("批2 换团队/换栈", B2), ("批3 因果链", B3)]):
    before = len(calls)
    msgs = [M(r, c, ts=TS + (i * 10 + k) * 60000) for k, (r, c) in enumerate(batch)]
    ret = sp.add(msgs)
    print()
    print("=" * 96)
    print(f"{label}：{len(batch)} 条 / {sum(len(c) for _, c in batch)} 字符  → add() 返回 {ret}")
    print("=" * 96)
    CRE, UPD, DEP = 0, 0, 0
    for j, rec in enumerate(calls[before:], 1):
        ops = (rec["out"].get("ops") or {}).get("node_ops") or []
        acts = {}
        for op in ops:
            acts[op.get("action")] = acts.get(op.get("action"), 0) + 1
        CRE += acts.get("create", 0); UPD += acts.get("update", 0); DEP += acts.get("deprecate", 0)
        print(f"  组{j}: 对话 {rec['chars']} 字符 | ops={acts or '{}'}"
              f"{' | triage SKIP' if rec['out'].get('skipped') else ''}")
    print(f"  → 合计 create={CRE} update={UPD} deprecate={DEP}")
    created = ret["nodes"]
    new_nodes = list(sp.graph.graph.nodes(data=True))[-created:] if created else []
    for nid, nd in new_nodes:
        body = str(nd.get("content") or "")
        ntype = nd.get("node_type")
        ntype = getattr(ntype, "value", None) or ntype or "-"
        print(f"    [{nid}]({ntype}) len={len(body):<4} cjk={cjk_ratio(body):.0%}  {body[:110]}")

# ---- 文档通道 ----
print()
print("=" * 96)
print(f"批4 中等长度文本 {len(DOC)} 字符（< 极端闸 {cfg.doc_chars_per_msg}）")
print("  → 走抽取 → 预计 no_ops → 由 _divert_raw 转入文档通道（新设计的主路径）")
print("=" * 96)
before_calls = len(calls)
r4 = sp.add([M("user", DOC, ts=TS + 999 * 60000)])
print(f"  add() 返回 {r4}；dba.maintain 调用次数 = {len(calls) - before_calls}")

DOC_BIG = DOC * 5
print()
print("=" * 96)
print(f"批5 超长文本 {len(DOC_BIG)} 字符（≥ 极端闸 {cfg.doc_chars_per_msg}）")
print("  → 形态闸直接判文档，不必再打一次抽取（保护 LLM 不被超大输入打爆）")
print("=" * 96)
before_calls = len(calls)
r5 = sp.add([M("user", DOC_BIG, ts=TS + 1000 * 60000)])
print(f"  add() 返回 {r5}；dba.maintain 调用次数 = {len(calls) - before_calls}（应为 0）")

# 监测「图内兜底块」的产出函数是否仍被调用——这是比"节点类型=thing"更精确的判据：
# THING 既是合法的抽取类型（物品/地点），也是旧兜底块用的类型，靠类型区分不开。
raw_fallback_calls = []
_orig_raw_fb = sp._store_raw_fallback


def _rec_raw_fb(messages, batch_ts=None):
    raw_fallback_calls.append(len(messages))
    return _orig_raw_fb(messages, batch_ts)


sp._store_raw_fallback = _rec_raw_fb

print()
print("=" * 96)
print("非图内容区（本轮改动核心）：没能变成图节点的原文应当落在这里，而不是灌进图")
print("=" * 96)
print(f"  文档区块数 = {len(sp.docs)}")
print(f"  图谱节点   = {sp.graph.graph.number_of_nodes()}  边 = {sp.graph.graph.number_of_edges()}")
print(f"  图内兜底块函数被调用次数 = {len(raw_fallback_calls)}  {raw_fallback_calls}")
print(f"  ★ 断言：doc_route 开启时不应用「图内兜底块」-> "
      f"{'PASS' if not raw_fallback_calls else 'FAIL'}")
for d in sp.docs[:5]:
    print(f"    [{d['id']}] chunk {d['chunk_index']}/{d['chunk_total']} len={len(d['content'])}"
          f"  {d['content'][:64]}")

# ---- 图结构 ----
print()
print("=" * 96)
print("图结构（B2 判据）")
print("=" * 96)
from collections import Counter
edges = sp.graph.to_dict().get("edges") or []
et = Counter(str(e.get("type")) for e in edges)
tmpc = Counter()
for e in edges:
    f = f"n{e.get('from')}" if not str(e.get("from")).startswith("n") else str(e.get("from"))
    t = f"n{e.get('to')}" if not str(e.get("to")).startswith("n") else str(e.get("to"))
    tmpc[(f, t)] = 1
dep = sum(1 for _, nd in sp.graph.graph.nodes(data=True) if nd.get("deprecated"))
n = sp.graph.graph.number_of_nodes()
print(f"  节点 {n}  边 {len(edges)}  边/节点 {len(edges)/max(1,n):.2f}  废弃节点 {dep}")
print(f"  边型 {dict(et.most_common())}")
directed = sum(v for k, v in et.items() if k in DIRECTED)
print(f"  有向边(causal/sequence/…) 占比 {100*directed/max(1,len(edges)):.0f}%")

# ---- 检索 ----
for q in ["What does the user use at work now?",       # D1：值应已更新为 Python/Spark
          "Why did the user buy an ergonomic chair?"]:  # B2：需因果链
    print()
    print("=" * 96)
    print(f"检索：{q}")
    print("=" * 96)
    items = sp.search(q, 100)
    for it in items:
        print(f"  [{it['id']}] score={it.get('score')} cjk={cjk_ratio(it['content']):.0%}")
        print(f"    {it['content'][:400]}")

import shutil
shutil.rmtree(tmp, ignore_errors=True)
