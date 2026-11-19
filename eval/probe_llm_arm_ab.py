# -*- coding: utf-8 -*-
"""LLM 双臂探针：把同一段 ALM 链路分别跑在两个模型上，比较建图与检索产物。

动机
----
抽取（GraphBuilder + MemoryDBA.maintain）与叙事（StoryRank）**全部是在
deepseek-v4-flash 上标定与调优的**。换用其它模型后（例如开源方法榜强制要求的
gpt-4o-mini，或任何新供应商）同一条链路是否还成立、产物是否退化，必须先实测再
决定部署配置。本探针即该对比工具，也可用于同一模型下比较不同配置。

设计
----
**唯一变量 = LLM**。两臂共用：同一份语料、同样的时间戳、同样的 embedding 端点与
模型、同样的检索参数（seed_k / max_hops / expand_k / 弃权阈值）、同样的代码路径。
配置差异只有 `llm_model / llm_api_key / llm_base_url` 三项。

指标（均不需要 judge，可离线复算）
  · 建图：节点数 / 边数 / TEMPORAL 边 / 时间锚点 / 归属词频（复用 e2e 的 graph_stats）
  · 每段 Add 的 messages→nodes 落地量（可看出 triage 跳过与兜底落库）
  · 检索：返回条数 / 总字符 / gold fact 命中（复用 e2e 的 match_facts）
  · **图相对纯向量检索的净增量**（核心判据）：对同一查询做一次不做图扩展的纯相似度
    top-100 检索，统计「图命中、而纯向量拿不到」的事实数。图的边际价值就在这里；
    同时给"等字符预算"口径，避免向量凭条数多而占便宜。
  · 退化信号：WARNING/ERROR 计数，以及「兜底落库 / 叙事缺失 / 弃权 / 解析失败」出现次数
  · 成本与延迟：token 用量（TokenMeter，挂在共享 LLM 实例上）、写入与检索耗时

注意：抽取与叙事都是 LLM 非确定性输出，**单轮只作方向判断**，不能逐条对比
（依据 Plan §7.10.5：检索层跨轮 std ±11.4pp）。

用法
----
    # 比较两个模型（A 沿用 .env 的 OPENAI_*，B 取 GPT_MODEL/JUDGE_*）
    python -m eval.probe_llm_arm_ab

    # 指定端点跑 A/B
    python -m eval.probe_llm_arm_ab --arm-a-model <M> --arm-a-base <URL> --arm-a-key <K> \
        --arm-b-model <M2> --arm-b-base <URL2> --arm-b-key <K2>

    # 只比建图产物、不做检索（快一倍），并导出图谱供离线差分
    python -m eval.probe_llm_arm_ab --no-search --dump eval/outputs/llm_arm
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# 关掉逐次调用的 token INFO，避免淹没对比输出（汇总仍在各自 summary 里）
os.environ.setdefault("ALM_TOKEN_VERBOSE", "0")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # 与其它 eval 脚本一致：先从 .env 装载模型与路径配置
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:
    pass

import yaml

from alm.config import ALMConfig
from alm.contract import AddMessage, AddRequest
from alm.engine import ALMEngine, space_filename
from alm.space import _cosine
from eval.test_alm_vs_mcp_e2e import graph_stats, match_facts, print_stats

DIALOGS_PATH = _ROOT / "data" / "dba_eval" / "groupchat_mixed_dialogs.yaml"
QUERIES_PATH = _ROOT / "data" / "dba_eval" / "groupchat_mixed_queries.yaml"
USER_ID = "eval:probe:llmarm:conv-0"
SESSION_ID = "eval:probe:llmarm:s0"
BASE_MS = 1_704_067_200_000  # 2024-01-01

# 退化信号关键词 → 说明该路径在某一臂上被触发
_DEGRADE_KEYS = {
    "兜底落库": "triage 跳过或抽不出节点，改走原文兜底",
    "叙事缺失": "叙事整理失败，退回节点原文",
    "弃权": "返回空数组（硬下限或模糊带判定）",
    "解析失败": "LLM 返回的 JSON 无法解析",
    "降级": "某一步降级处理",
}


class _Capture(logging.Handler):
    """收集日志用于计数与抽样，不打印正文"""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def count_level(self, level: int) -> int:
        return sum(1 for r in self.records if r.levelno >= level)

    def degrade_hits(self) -> Dict[str, int]:
        hits = {k: 0 for k in _DEGRADE_KEYS}
        for record in self.records:
            message = record.getMessage()
            for key in _DEGRADE_KEYS:
                if key in message:
                    hits[key] += 1
        return hits

    def samples(self, level: int, limit: int = 4) -> List[str]:
        out: List[str] = []
        for record in self.records:
            if record.levelno >= level:
                text = f"[{record.levelname}] {record.name}: {record.getMessage()}"
                out.append(text[:220])
                if len(out) >= limit:
                    break
        return out


def _load_dialogs(limit: Optional[int]) -> List[dict]:
    with DIALOGS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    dialogs = data["dialogs"] if isinstance(data, dict) else data
    return dialogs[:limit] if limit else dialogs


def _load_queries(which: str) -> List[dict]:
    with QUERIES_PATH.open("r", encoding="utf-8") as fh:
        queries = yaml.safe_load(fh)
    if which == "all":
        return queries
    wanted = {q.strip() for q in which.split(",") if q.strip()}
    return [q for q in queries if q["id"] in wanted]


def _build_messages(dialog: dict, idx: int) -> List[AddMessage]:
    """与 e2e 同口径：对话序号按天递增，轮次间隔 5 分钟（时间锚点依赖它）"""
    return [
        AddMessage(
            role=turn["role"],
            content=f"[{turn['speaker']}] {turn['text']}",
            timestamp=BASE_MS + idx * 86_400_000 + i * 300_000,
        )
        for i, turn in enumerate(dialog["turns"])
    ]


def _vector_topk(space, query: str, top_k: int):
    """纯向量检索基线：只按「查询 × 节点内容」的余弦排序取前 k，不做任何图扩展。

    这是本系统要超越的下限——PAR 的价值就等于「图检索结果」减去「同预算下的纯向量结果」。
    直接读向量缓存（get_content_vectors 无 API 调用），只多一次查询嵌入。
    """
    query_vec = space.vector_store._embed(query)
    ids = [nid for nid in space.graph.graph.nodes()
           if space.graph.graph.nodes[nid].get("content")]
    vectors = space.vector_store.get_content_vectors(ids)
    scored = [(nid, _cosine(query_vec, vec)) for nid, vec in zip(ids, vectors) if vec is not None]
    scored.sort(key=lambda pair: -pair[1])
    top = scored[:top_k]
    texts = [str(space.graph.get_content(nid) or "") for nid, _ in top]
    return top, texts


def _take_within(texts: List[str], budget: int) -> str:
    """按整条累积到字符预算为止，用于与图返回内容做**等字符预算**的公平对比"""
    out: List[str] = []
    used = 0
    for text in texts:
        if used + len(text) > budget and out:
            break
        out.append(text)
        used += len(text)
    return "\n".join(out)


def _arm_config(
    model: Optional[str],
    base: Optional[str],
    key: Optional[str],
) -> ALMConfig:
    """按参数覆写 LLM 三项；其余（embedding、检索参数）全部继承 .env"""
    cfg = ALMConfig.from_env()
    if model:
        cfg.llm_model = model
    if base:
        cfg.llm_base_url = base
    if key:
        cfg.llm_api_key = key
    # 探针固定走 API embedding：与云端部署一致，且避免本地加载向量模型拖慢双臂
    cfg.embedding_local = False
    return cfg


# 保存未被改写的原始构造器：换臂时必须从它出发重建，否则会叠加多层 patch
_ORIGINAL_BUILD_LLM = ALMEngine._build_llm

# 官方文档（api-docs.deepseek.com/guides/thinking_mode）：
#   · 开关  extra_body={"thinking": {"type": "enabled"/"disabled"}}
#   · 强度  reasoning_effort = "low" / "high" / "max"（medium/xhigh→high，minimal→low）
#   · 默认  思考开启，effort=high
# 两者都走 extra_body：OpenAI SDK 会把 extra_body 合并进请求体，故无需依赖 langchain
# 是否把 reasoning_effort 声明成字段，跨版本更稳。
THINKING_MODES = ("off", "low", "high", "max")


def _apply_thinking_mode(mode: Optional[str]) -> None:
    """让 ``ALMEngine._build_llm`` 产出的 ChatOpenAI 带上指定的 thinking 档位。

    ``mode`` 取 ``None``（不干预，即端点默认：思考开启 / effort=high）或
    ``THINKING_MODES`` 之一。

    为什么在探针里打补丁而不是加生产开关：该项仍在**验证阶段**，结论出来前不该改动
    线上行为（与之前 strict_extraction 的处理一致）。若最终证明某一档更优，再落成配置项。

    注意还原时必须用 ``staticmethod()`` 包回去：直接取 ``ALMEngine._build_llm`` 得到的是
    **普通函数**，直接赋回类会退化成实例方法，``self._build_llm(config, meter)`` 就会把
    ``self`` 当第一个参数传进去而报 "takes 1 to 2 positional arguments but 3 were given"。
    """
    if mode is None:
        ALMEngine._build_llm = staticmethod(_ORIGINAL_BUILD_LLM)
        return
    if mode not in THINKING_MODES:
        raise ValueError(f"未知 thinking 档位 {mode!r}，可选 {THINKING_MODES}")

    def patched(config, meter=None):
        llm = _ORIGINAL_BUILD_LLM(config, meter)
        body = dict(llm.extra_body or {})
        if mode == "off":
            body["thinking"] = {"type": "disabled"}
        else:
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = mode
        llm.extra_body = body
        return llm

    ALMEngine._build_llm = staticmethod(patched)


def run_arm(
    label: str,
    cfg: ALMConfig,
    dialogs: List[dict],
    queries: List[dict],
    dump_dir: Optional[Path] = None,
    do_search: bool = True,
) -> Dict[str, Any]:
    cfg.data_dir = Path(tempfile.mkdtemp(prefix=f"almllm_{label}_"))
    cap = _Capture()
    root_logger = logging.getLogger()
    root_logger.addHandler(cap)
    root_logger.setLevel(logging.INFO)
    logging.getLogger("alm").setLevel(logging.INFO)
    # 与服务端一致：dba_pipeline 只看 WARNING 以上（解析失败/降级都在这档）
    logging.getLogger("dba_pipeline").setLevel(logging.WARNING)

    engine = ALMEngine(cfg)
    print(f"\n===== 臂 {label}：{cfg.llm_model} @ {cfg.llm_base_url} =====")
    print(f"      embedding={cfg.embedding_model}（API）  data_dir={cfg.data_dir}")

    try:
        per_add: List[Dict[str, Any]] = []
        started = time.time()
        for idx, dialog in enumerate(dialogs):
            result = engine.add(
                AddRequest(
                    request_id=f"{USER_ID}:{dialog['id']}",
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    messages=_build_messages(dialog, idx),
                )
            )
            per_add.append({"id": dialog["id"], **result})
        add_elapsed = time.time() - started
        print(f"      写入耗时 {add_elapsed:.1f}s")
        print(f"      每段落地: " + ", ".join(
            f"{row['id']}={row['messages']}msg/{row['nodes']}node" for row in per_add
        ))

        space = engine.get_space(USER_ID, create=False)
        stats = graph_stats(space.graph, label)
        print_stats(stats)

        rows: List[Dict[str, Any]] = []
        if do_search:
            for query in queries:
                shot = time.time()
                data = space.search(query["query"], top_k=100)
                elapsed = time.time() - shot
                text = "\n".join(item["content"] for item in data)
                gold = query.get("gold_facts") or []
                graph_hit = match_facts(text, gold)
                # 纯向量基线：同一批节点上的纯相似度检索（无图扩展），
                # 同时给"全文 top-100"与"等字符预算"两种口径，后者才与图返回公平可比
                _, vec_texts = _vector_topk(space, query["query"], 100)
                vec_hit_full = match_facts("\n".join(vec_texts), gold)
                vec_hit_budget = match_facts(_take_within(vec_texts, max(len(text), 1)), gold)
                rows.append({
                    "id": query["id"],
                    "query": query["query"],
                    "count": len(data),
                    "chars": len(text),
                    "gold_hit": graph_hit,
                    "gold_total": len(gold),
                    "elapsed": elapsed,
                    "head": (data[0]["content"][:220] if data else ""),
                    "vec_hit_full": vec_hit_full,
                    "vec_hit_budget": vec_hit_budget,
                    # 图检索命中、而纯向量 top-100 拿不到的事实 —— 这才是 PAR 的净增量
                    "only_graph": sorted(set(graph_hit) - set(vec_hit_full)),
                })

        token_summary = engine.token_meter.summary()
        degrade = cap.degrade_hits()
        warnings = cap.count_level(logging.WARNING)
        errors = cap.count_level(logging.ERROR)

        gold_hit_total = sum(len(r["gold_hit"]) for r in rows)
        gold_total = sum(r["gold_total"] for r in rows)
        vec_hit_total = sum(len(r["vec_hit_full"]) for r in rows)
        vec_budget_total = sum(len(r["vec_hit_budget"]) for r in rows)
        only_graph_total = sum(len(r["only_graph"]) for r in rows)
        if do_search:
            print(f"      检索 {len(rows)} 条 | gold fact 图={gold_hit_total}/{gold_total}"
                  f" | 纯向量(全文)={vec_hit_total} (等预算)={vec_budget_total}"
                  f" | **仅图命中 {only_graph_total}**")
            print(f"      返回空数组 {sum(1 for r in rows if r['count'] == 0)} 条")
        else:
            print("      检索已跳过（--no-search）：只比较建图产物")
        print(f"      token: calls={token_summary['calls']} "
              f"in={token_summary['input_tokens']:,} out={token_summary['output_tokens']:,}"
              f"（未计量 {token_summary['unmetered_calls']} 次）")
        print(f"      日志: WARNING={warnings} ERROR={errors} | 退化信号={degrade}")
        for line in cap.samples(logging.WARNING):
            print(f"        {line}")

        return {
            "label": label,
            "model": cfg.llm_model,
            "base": cfg.llm_base_url,
            "add_elapsed": add_elapsed,
            "per_add": per_add,
            "stats": stats,
            "rows": rows,
            "tokens": token_summary,
            "warnings": warnings,
            "errors": errors,
            "degrade": degrade,
            "gold_hit_total": gold_hit_total,
            "gold_total": gold_total,
            "vec_hit_total": vec_hit_total,
            "vec_budget_total": vec_budget_total,
            "only_graph_total": only_graph_total,
        }
    finally:
        try:
            engine.close()
        except Exception:
            pass
        root_logger.removeHandler(cap)
        # 保留建图产物：临时目录会被删掉，先复制出图谱 YAML 供离线分析
        if dump_dir is not None:
            src = cfg.data_dir / space_filename(USER_ID)
            if src.exists():
                arm_dir = Path(dump_dir) / label
                arm_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, arm_dir / "graph.yaml")
                print(f"      图谱已导出: {arm_dir / 'graph.yaml'}")
        shutil.rmtree(cfg.data_dir, ignore_errors=True)


def _report(a: Dict[str, Any], b: Dict[str, Any], queries: List[dict]) -> None:
    print("\n" + "=" * 78)
    print("汇总对比（A = 现用模型，B = 候选模型；唯一变量就是 LLM）")
    print("=" * 78)
    print(f"  A: {a['model']}")
    print(f"  B: {b['model']}")

    print(f"\n  {'指标':<18}{'A':>10}{'B':>10}")
    rows = [
        ("写入耗时(s)", a["add_elapsed"], b["add_elapsed"]),
        ("节点数", a["stats"]["nodes"], b["stats"]["nodes"]),
        ("边数", a["stats"]["edges"], b["stats"]["edges"]),
        ("TEMPORAL 边", a["stats"]["temporal_edges"], b["stats"]["temporal_edges"]),
        ("时间锚点节点", a["stats"]["time_anchor_nodes"], b["stats"]["time_anchor_nodes"]),
        ("gold fact 命中", f"{a['gold_hit_total']}/{a['gold_total']}",
         f"{b['gold_hit_total']}/{b['gold_total']}"),
        ("纯向量(全文 top100)", a["vec_hit_total"], b["vec_hit_total"]),
        ("纯向量(等字符预算)", a["vec_budget_total"], b["vec_budget_total"]),
        ("★仅图命中（净增量）", a["only_graph_total"], b["only_graph_total"]),
        ("返回空数组", sum(1 for r in a["rows"] if r["count"] == 0),
         sum(1 for r in b["rows"] if r["count"] == 0)),
        ("LLM 调用次数", a["tokens"]["calls"], b["tokens"]["calls"]),
        ("input tokens", a["tokens"]["input_tokens"], b["tokens"]["input_tokens"]),
        ("output tokens", a["tokens"]["output_tokens"], b["tokens"]["output_tokens"]),
        ("WARNING", a["warnings"], b["warnings"]),
        ("ERROR", a["errors"], b["errors"]),
    ]
    for key, label in _DEGRADE_KEYS.items():
        rows.append((f"退化·{key}", a["degrade"][key], b["degrade"][key]))
    for label, va, vb in rows:
        if isinstance(va, float) or isinstance(vb, float):
            va, vb = f"{va:.1f}", f"{vb:.1f}"
        print(f"  {label:<18}{str(va):>10}{str(vb):>10}")

    if not a["rows"]:
        print("\n（--no-search：跳过逐查询对照，建图产物已导出供离线分析）")
        return

    print("\n" + "-" * 78)
    print("逐查询对照（条数 / 字符 / gold 命中 / 耗时s）")
    print("-" * 78)
    by_id_b = {r["id"]: r for r in b["rows"]}
    for row_a in a["rows"]:
        row_b = by_id_b.get(row_a["id"])
        if row_b is None:
            continue
        print(f"\n[{row_a['id']}] {row_a['query']}")
        print(f"  A 图 {row_a['count']:>2}条 {row_a['chars']:>5}字 "
              f"命中 {len(row_a['gold_hit'])}/{row_a['gold_total']} {row_a['elapsed']:.1f}s"
              f" | 向量 {len(row_a['vec_hit_full'])}"
              f"（等预算 {len(row_a['vec_hit_budget'])}）"
              f" | 仅图 {row_a['only_graph']}")
        print(f"  B 图 {row_b['count']:>2}条 {row_b['chars']:>5}字 "
              f"命中 {len(row_b['gold_hit'])}/{row_b['gold_total']} {row_b['elapsed']:.1f}s"
              f" | 向量 {len(row_b['vec_hit_full'])}"
              f"（等预算 {len(row_b['vec_hit_budget'])}）"
              f" | 仅图 {row_b['only_graph']}")
        print(f"    A 叙事: {row_a['head'][:150] or '（空）'}")
        print(f"    B 叙事: {row_b['head'][:150] or '（空）'}")

    print("\n" + "-" * 78)
    print("判读提示")
    print("-" * 78)
    print("  · 判据是「图相对纯向量的净增量」：★仅图命中 = 图检索命中、而纯向量 top-100 拿不到")
    print("    的事实数。该值越大，说明图扩展确实带回了相似度检索够不着的记忆；接近 0 说明")
    print("    这份图对检索没有边际价值（边再密也没用）。")
    print("  · 「等字符预算」那一列才是公平对照：图只返回一段叙事，向量却可铺满 top-100，")
    print("    只有把向量截到同样的字符量，两者才可比。")
    print("  · 退化信号（兜底落库 / 叙事缺失 / 解析失败）显著变多 → 该配置撑不住现有 prompt。")
    print("  · LLM 非确定性：单轮只作方向判断，要坐实需重复 2–3 轮取中位数。")


def main() -> int:
    parser = argparse.ArgumentParser(description="ALM 链路的 LLM 双臂探针")
    parser.add_argument("--dialogs", type=int, default=None, help="只取前 N 段对话（默认全部）")
    parser.add_argument("--queries", default="all", help="查询 id 逗号分隔，或 all")
    parser.add_argument("--arm-a-model", default=None, help="A 臂模型（默认沿用 .env 的 OPENAI_MODEL）")
    parser.add_argument("--arm-a-base", default=None, help="A 臂端点（默认沿用 .env）")
    parser.add_argument("--arm-a-key", default=None, help="A 臂密钥（默认沿用 .env）")
    parser.add_argument("--arm-b-model", default=os.environ.get("GPT_MODEL") or "gpt-4o-mini",
                        help="B 臂模型名（默认取 GPT_MODEL）")
    parser.add_argument("--arm-b-base", default=os.environ.get("JUDGE_API_BASE"),
                        help="B 臂端点（默认取 JUDGE_API_BASE）")
    parser.add_argument("--arm-b-key", default=os.environ.get("JUDGE_API_KEY"),
                        help="B 臂密钥（默认取 JUDGE_API_KEY）")
    parser.add_argument("--arm-a-thinking", choices=THINKING_MODES, default=None,
                        help="A 臂 thinking 档位；不给则不干预（遵循 ALMConfig.llm_thinking，"
                             "现默认关闭思考）")
    parser.add_argument("--arm-b-thinking", choices=THINKING_MODES, default=None,
                        help="B 臂 thinking 档位；不给则不干预（遵循 ALMConfig.llm_thinking，"
                             "现默认关闭思考）")
    parser.add_argument("--no-search", action="store_true",
                        help="跳过检索，只灌入并比较建图产物（快一倍）")
    parser.add_argument("--dump", default=None,
                        help="把各臂的图谱 YAML 导出到该目录（形如 <dir>/A/graph.yaml）")
    args = parser.parse_args()

    dialogs = _load_dialogs(args.dialogs)
    queries = _load_queries(args.queries)
    if not queries:
        print("错误：没有匹配的查询", file=sys.stderr)
        return 2

    cfg_a = _arm_config(args.arm_a_model, args.arm_a_base, args.arm_a_key)
    cfg_b = _arm_config(args.arm_b_model, args.arm_b_base, args.arm_b_key)
    if not cfg_b.llm_api_key:
        print("错误：B 臂缺少密钥（用 --arm-b-key 指定，或设置 JUDGE_API_KEY）", file=sys.stderr)
        return 2
    # 允许两臂同模型同端点——只比 thinking 档位时就是这种用法；三项全同才无从对比
    if (cfg_a.llm_model == cfg_b.llm_model and cfg_a.llm_base_url == cfg_b.llm_base_url
            and args.arm_a_thinking == args.arm_b_thinking):
        print("错误：两臂完全同配置（模型/端点/thinking 档位都相同），无法对比", file=sys.stderr)
        return 2

    def _thinking_tag(mode: Optional[str]) -> str:
        return "thinking=不干预（默认关）" if mode is None else f"thinking={mode}"

    print(f"# ALM 链路探针 | 语料 {DIALOGS_PATH.name}：{len(dialogs)} 段对话"
          f" | 查询 {len(queries)} 条 | gold fact "
          f"{sum(len(q.get('gold_facts') or []) for q in queries)} 项")
    print(f"# A 臂: {cfg_a.llm_model} @ {cfg_a.llm_base_url} | {_thinking_tag(args.arm_a_thinking)}")
    print(f"# B 臂: {cfg_b.llm_model} @ {cfg_b.llm_base_url} | {_thinking_tag(args.arm_b_thinking)}")
    print("# embedding / 语料 / 时间戳 / 检索参数 / 代码路径完全一致；"
          "评价口径 = 图检索相对**纯向量检索**的净增量\n")

    _apply_thinking_mode(args.arm_a_thinking)
    a = run_arm("A", cfg_a, dialogs, queries, dump_dir=args.dump, do_search=not args.no_search)
    _apply_thinking_mode(args.arm_b_thinking)
    b = run_arm("B", cfg_b, dialogs, queries, dump_dir=args.dump, do_search=not args.no_search)
    _report(a, b, queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
