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
    # triage 跳过时的兜底落库上限（保证本次会话内容仍可被检索）
    fallback_max_nodes: int = 64

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
    # 0 表示关闭弃权判定（永远返回候选）。
    # 0.52 由 comprehensive 数据集用 bge-large-zh 标定：该阈值下 6/6 弃权题正确弃权，
    # 70 条可作答题中仅 1 条被误弃权（总准确 0.9211 → 0.9868）。
    # 换数据集或换 embedding 模型后需重新标定。
    abstain_cosine: float = 0.52

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
            fallback_max_nodes=_env_int("ALM_FALLBACK_MAX_NODES", 64),
            seed_k=_env_int("ALM_SEED_K", 40),
            seed_k_min=_env_int("ALM_SEED_K_MIN", 5),
            expand_k=_env_int("ALM_EXPAND_K", 40),
            max_hops=_env_int("ALM_MAX_HOPS", 5),
            max_top_k=_env_int("ALM_MAX_TOP_K", 100),
            rerank_mode=(_env("ALM_RERANK_MODE", "tiered") or "tiered").strip().lower(),
            rerank_pool=_env_int("ALM_RERANK_POOL", 80),
            fallback_dedup_threshold=_env_float("ALM_FALLBACK_DEDUP_THRESHOLD", 2.0),
            orphan_threshold=_env_int("ALM_ORPHAN_THRESHOLD", 10 ** 9),
            abstain_cosine=_env_float("ALM_ABSTAIN_COS", 0.52),
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
