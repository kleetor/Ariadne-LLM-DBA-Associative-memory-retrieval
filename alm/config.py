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


def _env_float_list(name: str, default: List[float]) -> List[float]:
    raw = _env(name)
    if raw is None:
        return default
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError:
        return default
    return values or default


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

    # ---- 检索形态 ----
    # raw    只返回检索层原始节点文本（结构不可读）
    # story  只返回整理后的连贯叙述（返回条数固定为 1）
    # hybrid 叙述在前 + 采纳节点原文在后（默认：易理解性与事实精度兼得）
    # 注：story / hybrid 每条查询多一次 LLM 调用（用于把节点与关系边整理成叙述），
    # 但这是让「图结构」被平台统一回答模型看见的唯一通道——content 是唯一信息载体。
    search_shape: str = "hybrid"

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

    # ---- 选择题 options ----
    # 文本赛道含 Streaming 选择题，平台只在选择题的 Search 顶层下发 options（不含金标），
    # 开放题不下发。两个开关都默认关闭，因此开放题路径与既有评测行为完全不变；
    # 是否启用由本地 MC 集 A/B 的分数决定（见 eval/probe_options_mc.py）。
    #
    # abstain_guard：有 options 时跳过两段式弃权。选择题按定义是可回答的，而它的题干
    #   往往比开放题短（真正的判别信息在选项里），余弦容易被压低——实测短问句余弦可低至
    #   0.23~0.41（Plan §7.6），一旦落到硬下限之下就是整题归零，代价远大于多返几条无关内容。
    # contrast：把「各选项与候选的最大余弦」并入层内融合。动机是选择题题干常常欠定，
    #   只用题干做种子与重排会丢失判别信号；逐选项单独编码再取 max 聚合，而不是把选项
    #   拼进题干——拼接会让长选项稀释题干，并让干扰项把「匹配错误选项」的记忆拉上来。
    options_abstain_guard: bool = False
    options_contrast: bool = False
    # 对比信号在层内融合中的占比（仅 rerank_mode != off 时生效）
    options_contrast_weight: float = 0.35

    # ---- 记忆整理的时间渲染 ----
    # 开启后，叙事 prompt 切换为带规则 9 的变体（「记录于 YYYY-MM-DD」是记录时间、
    # 不是事件发生时间），节点时间戳因此进入 content。默认关闭：ALM 侧原本逐字使用
    # 基础 prompt，属于刻意冻结的评测行为，改动须由 smoke A/B 的分数决定。
    render_timestamps: bool = False

    # ---- 时序补召（C 维度：时间与事件序列）----
    # 主检索器的图扩展方向对时序是单向的（反向权重为 0），启用后额外调用时序工具，
    # 语义命中时间锚点后沿 TEMPORAL 边反向取共时事件，作为 T4 梯度并入候选。
    #
    # 默认关闭。实测（comprehensive）该通道与主检索高度冗余：
    #   · 76 条查询中 43 条触发、产出 278 个节点，但 97% 已被主链路召回；
    #   · 10 条 temporal 查询上仅新增 8 个节点，命中 gold 的为 0；
    #   · gold 含时间性事实占比 73.2%（30/41），可排除「gold 未标注时序」的测量假象。
    #   · 非时序查询上 T4 未挤占 top10（0/60 位移），说明低权重确实压得住，但这同时
    #     意味着它不产生任何增益；而新增节点 gold 重叠为 0，提权必然有害。
    # 结论：不增益也不（在低权重下）减益，故默认关闭以免被后续调参误激活。
    # 若真实数据上主链路确实抓不住时间问题，置 true 重新评估。
    temporal_recall: bool = False
    temporal_k_seed: int = 8
    temporal_max_facts: int = 8
    temporal_max_anchors: int = 3

    # ---- 重排 ----
    # off       沿用 PAR 组合分排序（基线）
    # embedding 仅做 PAR 组合分与语义相似度的融合
    # tiered    在融合分上再乘梯度先验（T1 种子 / T2 图扩展 / T3 语义救援）—— 默认
    # llm       用 gpt-4o-mini 对候选打分（ALM 规定 Add/Search 的 LLM 必须是 gpt-4o-mini）
    rerank_mode: str = "tiered"
    # 参与重排的候选池上限
    rerank_pool: int = 80
    # 候选池中预留给低梯度（T3 救援 / T4 时序）的名额。T3/T4 在候选序列里排在 T1/T2
    # 之后且 par_score 恒为 0，若不预留就会在 PAR 候选接近 pool 时被整段切掉，使救援
    # 与时序通道在大图上静默失效。0 表示退回「只取前 pool 条」的旧行为。
    rerank_pool_reserve: int = 20
    # 梯度先验乘子，依次对应 T1 种子 / T2 图扩展（1 跳）/ T3 语义救援 / T4 时序补召。
    # T4 高于 T3：它来自语义命中的时间锚点，比"仅因被目的过滤掉才需救援"的节点更可信。
    rerank_tier_weights: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.4, 0.65])
    # T3 语义救援的候选上限（0 表示关闭救援）
    rerank_rescue: int = 40
    # 层内融合权重：PAR 组合分与语义相似度
    rerank_weight_par: float = 0.3
    rerank_weight_embedding: float = 0.7

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
            rerank_mode=(_env("ALM_RERANK_MODE", "tiered") or "tiered").strip().lower(),
            rerank_pool=_env_int("ALM_RERANK_POOL", 80),
            rerank_pool_reserve=_env_int("ALM_RERANK_POOL_RESERVE", 20),
            fallback_dedup_threshold=_env_float("ALM_FALLBACK_DEDUP_THRESHOLD", 2.0),
            orphan_threshold=_env_int("ALM_ORPHAN_THRESHOLD", 10 ** 9),
            abstain_cosine=_env_float("ALM_ABSTAIN_COS", 0.35),
            abstain_verify_hi=_env_float("ALM_ABSTAIN_VERIFY_HI", 0.50),
            abstain_judge_top_n=_env_int("ALM_ABSTAIN_JUDGE_TOP_N", 3),
            abstain_judge_max_chars=_env_int("ALM_ABSTAIN_JUDGE_MAX_CHARS", 200),
            options_abstain_guard=_env_bool("ALM_OPTIONS_ABSTAIN_GUARD", False),
            options_contrast=_env_bool("ALM_OPTIONS_CONTRAST", False),
            options_contrast_weight=_env_float("ALM_OPTIONS_CONTRAST_WEIGHT", 0.35),
            render_timestamps=_env_bool("ALM_RENDER_TIMESTAMPS", False),
            temporal_recall=_env_bool("ALM_TEMPORAL_RECALL", False),
            temporal_k_seed=_env_int("ALM_TEMPORAL_K_SEED", 8),
            temporal_max_facts=_env_int("ALM_TEMPORAL_MAX_FACTS", 8),
            temporal_max_anchors=_env_int("ALM_TEMPORAL_MAX_ANCHORS", 3),
            search_shape=(_env("ALM_SEARCH_SHAPE", "hybrid") or "hybrid").strip().lower(),
            rerank_tier_weights=_env_float_list("ALM_RERANK_TIER_W", [1.0, 0.7, 0.4, 0.65]),
            rerank_rescue=_env_int("ALM_RERANK_RESCUE", 40),
            rerank_weight_par=_env_float("ALM_RERANK_W_PAR", 0.3),
            rerank_weight_embedding=_env_float("ALM_RERANK_W_EMB", 0.7),
        )
