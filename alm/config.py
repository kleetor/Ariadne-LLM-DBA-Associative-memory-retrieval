# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 专项接入配置。

全部通过环境变量注入，与 MCP / 可视化链路隔离；
命令行参数优先级高于环境变量（由 server.main 覆盖）。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class ALMConfig:
    """ALM 服务运行配置"""

    # ---- 服务 ----
    host: str = "0.0.0.0"
    port: int = 8770

    # ---- 存储 ----
    data_dir: Path = Path("data/alm")
    # 内存中同时驻留的 user_id 空间上限（超出按 LRU 换出，磁盘 YAML 保留）
    max_spaces: int = 256

    # ---- 鉴权 ----
    # 逗号分隔的可用 Key；为空表示不鉴权（仅适用于本地自测 / 公开 smoke）
    api_keys: List[str] = field(default_factory=list)

    # ---- LLM ----
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    # 链路内部是否开启 deepseek 的思考模式（该端点**默认开启**，effort=high）。
    # **默认关闭**，依据是两阶段对照实验（见 Plan/0921——thinking档位对照实验报告.md）：
    # 关闭后 Add/Search 的 output token 降到约 1/16、写入快 2.5 倍，而 gold 覆盖
    # 28/28/30 三档极差仅 2（落在噪声内），且开启档位反而出现读取饱和。
    # 抽取、目的推断、叙事整理共用同一个 LLM 实例，故这一处开关覆盖整条链路；
    # 叙事整理只是把已检索记忆拼装成叙事，无长上下文推理需求。
    llm_thinking: bool = False

    # ---- Embedding ----
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_local: bool = False

    # ---- 写入 ----
    # 单次 Add 最多处理的消息条数（防止异常输入打爆 LLM）
    max_messages_per_add: int = 64
    # triage 跳过时的兜底落库上限（保证本次会话内容仍可被检索）。
    # 必须 ≥ max_messages_per_add：维护链路只处理前 N 条，其余改走兜底，
    # 兜底上限若小于 N 会让「超量消息」这条补救路径再次丢内容。
    fallback_max_nodes: int = 256

    # ---- 检索 ----
    # seed_k 为 seed 数量上限；seed_k_min 为下限。
    # 实际取值按图规模自适应（见 MemorySpace._effective_seed_k）：单 user 空间的图
    # 往往只有几十到几百个节点，固定值会造成两种失真——过大则把整张图召回为种子、
    # PAR 图扩展无事可做；过小则直接限制召回，孤立节点无法通过扩展被找回。
    seed_k: int = 40
    seed_k_min: int = 5
    expand_k: int = 40
    max_hops: int = 5
    # 单次 Search 允许返回的最大条数（对齐 ALM 正式评测 top_k=100）
    max_top_k: int = 100

    # ---- 兜底落库与遗忘 ----
    # 兜底落库时临时提升的语义去重阈值。余弦相似度上界为 1，故 >1 即彻底关闭去重：
    # 原文兜底若被去重拦截，消息会整条丢失而 add() 仍返回 200，属静默丢数据。
    fallback_dedup_threshold: float = 2.0
    # 孤立节点自动遗忘阈值。ALM 契约要求每次 Add 的消息「已完整存储且可被检索」，
    # 而兜底落库的原文节点必然是孤立节点，默认的 3 次即遗忘会静默清掉它们，
    # 因此默认设为极大值以关闭该启发式（用户显式要求的遗忘仍走 deprecate 路径）。
    orphan_threshold: int = 10 ** 9

    # ---- 弃权 ----
    # 候选与查询的最大余弦低于该值时，视为「无相关记忆」，按 ALM 契约返回空数组。
    # 注意这是**两段式弃权的硬下限**，置 0 只关掉第一段；第二段（模糊带判定）由
    # abstain_verify_hi 单独控制，**也必须一并置 0 才能整体关闭弃权**（见 space.py）。
    # 0.52 由 comprehensive 数据集用 bge-large-zh 标定：该阈值下 6/6 弃权题正确弃权，
    # 70 条可作答题中仅 1 条被误弃权（总准确 0.9211 → 0.9868）。
    # 换数据集或换 embedding 模型后需重新标定。
    # 标定手段：每次 Search 都会把候选最高余弦记进 RetrievalStats（见 space.py），
    # 平台 smoke 打一轮后从服务端日志读分位数，即可判断该值是偏低还是偏高。
    #
    # 实测（eval/probe_abstain_threshold.py，刻意拉开措辞的低余弦探针集，20 条）：
    #   有答案查询余弦 min=0.3954 中位=0.5064；无答案查询 max=0.4701 —— 两类区间
    #   重叠，单一阈值必然错一边；0.52 恰好落在有答案分布的中位附近，误弃权 6~7/12。
    #   纯阈值最优为 0.40；配合下面的模糊带判定可做到 A 召回 95% / H 准确 100%。
    # 因此本值取 0.35 作为**硬下限**（低于它连判定都不发），模糊带交给判定处理。
    abstain_cosine: float = 0.35

    # ---- 弃权模糊带二次判定（见 alm/judge.py）----
    # abstain_cosine ≤ best_cos < abstain_verify_hi 时，把已排序的 top-N 片段交给
    # LLM 判一次「是否够回答问题」，判否才弃权；调用失败或解析失败一律放行。
    # 依据：单一余弦无法区分「库里没有」与「有但措辞距离远」（两类区间实测重叠）。
    #
    # 实测（同上探针集）：t_low=0.35 / t_hi=0.50 / top_n=3 时
    #   A 召回 95.0% / H 准确 100%，三种权重下综合均优于纯阈值；
    #   对 t_low∈[0.30,0.40]、t_hi≥0.50、top_n∈[3,10] 均不敏感。
    #   t_hi 低于 0.50 会漏判边界样本（实测 n08 cos=0.4701 未被拦下）。
    # 0 表示关闭二次判定（退回纯阈值行为）。
    abstain_verify_hi: float = 0.50
    # 判定时喂给 LLM 的片段条数与单条截断长度（实测 top_n=3 已饱和）
    abstain_judge_top_n: int = 3
    abstain_judge_max_chars: int = 200

    # ---- 记忆整理的时间渲染 ----
    # 开启后，叙事 prompt 切换为带规则 9 的变体（「记录于 YYYY-MM-DD」是记录时间、
    # 不是事件发生时间），节点记录时间因此进入叙事 prompt。
    #
    # 默认保持**关闭**。0920 曾一度改为 True（理由：官方 Answer 模板第 7 条要求把
    # yesterday / last month 换算成日期，而 MCP 一直硬编码 True），但 A/B 补测证明
    # **净收益为 0**：同一份记忆只切换该开关，13 条查询的回答级事实覆盖逐条完全相同
    # （26/34 vs 26/34），26 次叙事生成里仅 1 次把「记录于」写进输出——开了近乎空转。
    # 按本项自身的判据（改动须由 A/B 分数决定）故回退。
    #
    # 注意：日期信息另有独立通道，不受本开关影响——`created_at` 由 `_node_created_at()`
    # 无条件输出给平台（见 space.py）。详见 Plan §7.10.5 D2/F。
    render_timestamps: bool = False

    # ---- 时序路由（Plan §3.2：/search 内部还原「时序工具 → 主通道」串联）----
    # 开启后，_search_locked 先用 query 做一次目的判定（时序标签与目的推断合并在同一次
    # infer_purpose 调用里，不新增 LLM 调用）；判定为时序问题时调 temporal_lookup 取
    # 时间锚点 + 共时事实，把这些文本**追加**进 purpose 再驱动主通道。非时序问题时
    # purposes 原样透传（行为与改造前一致）。关闭时 ALM 不显式调用目的推断，直接走
    # retrieve_with_story 内部推断，链路与改造前逐字一致。
    #
    # 默认 False（保守）：链路已实现并验证（Plan §3.2），但"时序产出并入 purpose"对种子
    # 选择的实际权重尚未标定（Plan §六风险 2），且我们没有 A/B 证据。按本项目对冻结行为
    # 的既定处理（render_timestamps 即"改动须由 A/B 分数决定"，净 0 即回退），在拿到净收益
    # 证据前不默认改变线上行为。置 ALM_TEMPORAL_ROUTE=1 即启用完整串联链路。
    temporal_route: bool = False
    # temporal_lookup 的三个参数（沿用已删 T4 实现的默认值）
    temporal_k_seed: int = 8
    temporal_max_facts: int = 8
    temporal_max_anchors: int = 3

    @classmethod
    def from_env(cls) -> "ALMConfig":
        keys = [k.strip() for k in (_env("ALM_API_KEYS") or "").split(",") if k.strip()]
        return cls(
            host=_env("ALM_HOST", "0.0.0.0"),
            port=_env_int("ALM_PORT", 8770),
            data_dir=Path(_env("ALM_DATA_DIR", "data/alm")),
            max_spaces=_env_int("ALM_MAX_SPACES", 256),
            api_keys=keys,
            llm_model=_env("OPENAI_MODEL"),
            llm_api_key=_env("OPENAI_API_KEY"),
            llm_base_url=_env("OPENAI_API_BASE"),
            llm_thinking=_env_bool("ALM_LLM_THINKING", False),
            embedding_model=_env("EMBEDDING_MODEL"),
            embedding_api_key=_env("EMBEDDING_API_KEY"),
            embedding_base_url=_env("EMBEDDING_API_BASE"),
            embedding_local=_env_bool("EMBEDDING_LOCAL"),
            max_messages_per_add=_env_int("ALM_MAX_MESSAGES_PER_ADD", 64),
            fallback_max_nodes=_env_int("ALM_FALLBACK_MAX_NODES", 256),
            seed_k=_env_int("ALM_SEED_K", 40),
            seed_k_min=_env_int("ALM_SEED_K_MIN", 5),
            expand_k=_env_int("ALM_EXPAND_K", 40),
            max_hops=_env_int("ALM_MAX_HOPS", 5),
            max_top_k=_env_int("ALM_MAX_TOP_K", 100),
            fallback_dedup_threshold=_env_float("ALM_FALLBACK_DEDUP_THRESHOLD", 2.0),
            orphan_threshold=_env_int("ALM_ORPHAN_THRESHOLD", 10 ** 9),
            abstain_cosine=_env_float("ALM_ABSTAIN_COS", 0.35),
            abstain_verify_hi=_env_float("ALM_ABSTAIN_VERIFY_HI", 0.50),
            abstain_judge_top_n=_env_int("ALM_ABSTAIN_JUDGE_TOP_N", 3),
            abstain_judge_max_chars=_env_int("ALM_ABSTAIN_JUDGE_MAX_CHARS", 200),
            render_timestamps=_env_bool("ALM_RENDER_TIMESTAMPS", False),
            temporal_route=_env_bool("ALM_TEMPORAL_ROUTE", False),
            temporal_k_seed=_env_int("ALM_TEMPORAL_K_SEED", 8),
            temporal_max_facts=_env_int("ALM_TEMPORAL_MAX_FACTS", 8),
            temporal_max_anchors=_env_int("ALM_TEMPORAL_MAX_ANCHORS", 3),
        )
