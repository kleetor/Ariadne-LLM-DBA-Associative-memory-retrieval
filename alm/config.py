# SPDX-License-Identifier: AGPL-3.0-only

"""ALM 专项接入配置。

全部通过环境变量注入，与 MCP / 可视化链路隔离；
命令行参数优先级高于环境变量（由 server.main 覆盖）。
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    # 下限校验：非法/越界值一律静默回退默认，不让一个配错的环境变量把服务拖进
    # 死循环或异常状态（见 0927 审查 B4：doc_chunk_chars<=0 会让 _split_by_chars 死循环）。
    if minimum is not None and value < minimum:
        logger.warning(
            "%s=%s 低于下限 %s，已回退默认值 %s", name, value, minimum, default
        )
        return default
    return value


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
    # **默认关闭**，依据是两阶段对照实验（见 Plan/alm/0921——thinking档位对照实验报告.md）：
    # 关闭后 Add/Search 的 output token 降到约 1/16、写入快 2.5 倍，而 gold 覆盖
    # 28/28/30 三档极差仅 2（落在噪声内），且开启档位反而出现读取饱和。
    # 抽取、目的推断、叙事整理共用同一个 LLM 实例，故这一处开关覆盖整条链路；
    # 叙事整理只是把已检索记忆拼装成叙事，无长上下文推理需求。
    llm_thinking: bool = False
    # 链路内部 LLM 的**输出** token 上限。0 = 不传该参数，交端点默认。
    #
    # 为什么必须显式设（0926 实测，eval/probe_alm_llm_limit.py）：deepseek 端点在
    # `thinking={"type":"disabled"}`（即上面 `llm_thinking=False` 的默认形态）下，
    # **不传 max_tokens 时输出硬上限就是 8192**（finish_reason=length）；而显式传
    # 32768 可正常返回 19001 token、31892 字符。这直接决定 StoryRank 的成败：
    #   线上 7 次 `StoryRank JSON 解析失败` 与 7 次 `alm.tokens: out=8192` 一一对应
    #   —— 输出撞 8192 被截断，JSON 在输出中途断掉。
    # 注意这一项**只抬高上限、不改行为**（按实际用量计费），故可安全设大。
    # 但它只是**止血**：根治是收紧 StoryRank 的输出契约（见 dba_pipeline/llm/inference.py
    # 规则 6/8 的选材与 `_STORY_MAX_CHARS` 篇幅预算），否则即便不截断，返回的仍是"全部候选的复述"。
    llm_max_tokens: int = 0

    # 单次上游 LLM 请求的超时（秒）。**必须显式设**：langchain-openai 不传 `request_timeout`
    # 时用的是 openai SDK 的默认 600 秒，配上 `max_retries=3`，一个挂死的上游会把 `/search`
    # 拖住最长 **40 分钟**——期间该用户的空间锁一直被占（同 user_id 的检索被串行阻塞），
    # 平台侧只会看到超时/5xx。0930 本地实测过网关单次调用挂死 >10 分钟。
    # 120 秒的取法：正常 StoryRank 长输出（2,000~4,000 字符 ≈ 1,000~2,000 token）实测在
    # 几十秒量级，留 2 倍余量；要更激进可置 `ALM_LLM_TIMEOUT`（如 30）——代价是长输出
    # 的题会被判超时，请配合日志里的 `out=` / `finish_reason` 观察后再调。
    llm_timeout: float = 120.0

    # ---- Embedding ----
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_local: bool = False

    # ---- 写入 ----
    # 单次 Add 最多处理的消息条数（防止异常输入打爆 LLM）
    max_messages_per_add: int = 64
    # triage 跳过时的兜底落库上限，单位是**节点数**——长消息按 500 字符切块、
    # 每块一个节点（见 space.py 的 _FALLBACK_CHUNK_CHARS），不再是"消息条数"。
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

    # 峰值容忍带的输出条数上限（0 = 不限，保持改造前行为）。
    #
    # 动机：容忍带取的是"整轮并集"，而 expand_k=40 的宽前沿在大图上会把读出量顶到
    # 全图六成——实测同一张 200 节点图、同一查询，只改预算就从 23 条（seed_k=5/
    # expand_k=5，即历史评测口径）涨到 124 条（seed_k=20/expand_k=40，即本服务口径）。
    # 而 0828 检索理论文的设计输出区间是 5–37 条；线上日志也印证了膨胀
    # （喂入点 max 379 / 采纳 max 378）。喂给 Answer 模型的块若近乎全图，"因果链
    # 恢复 / 新值覆盖 / 全局压缩"三项能力就只能靠 Answer 模型自己分辨。
    #
    # 默认 0（关闭）：按本项目对冻结行为的既定处理，改动须由 A/B 分数决定。
    max_peak_nodes: int = 0

    # 目的回归过滤阈值（候选节点与「目的向量」的余弦下限，低于它不进候选集）。
    #
    # 这个值**与 embedding 模型强绑定**，不是通用常数：0.2 的来历见 0828 检索理论文
    # §3.3.2——bge-large-zh-v1.5 下「缓解 vs 喝咖啡」的余弦就落在 0.2–0.3，取 0.3 会误杀
    # 大量跳转轴扩展出的功能等价结果，故放宽到 0.2。换 embedding 模型（如 bge-m3）后
    # 余弦分布整体平移，**本值必须在新模型上重标**，否则会成片误过滤（表现为扩展候选被
    # 清空、`跳=1` 占比升高）。
    #
    # 此前该值硬编码在 `PurposeDrivenRetriever.__init__` 的默认参数里、ALM 无法覆盖；
    # 这里提上来是为了能在不重建镜像的前提下做标定与 A/B（env ALM_PURPOSE_FILTER_THRESHOLD）。
    # 另两个同源参数 `purpose_filter_decay` / `distance_decay` 未提升：前者实测为 0（固定阈值），
    # 后者只作用于跳转轴项、与 embedding 无关。
    #
    # ⚠️ 0925 换 bge-m3 后上移为 **0.30**。依据（eval/probe_threshold_retune.py，同样本换模型）：
    #   必须放过的那一类「功能等价」样本（动机→行为，如 "work stress" ↔ "drinks coffee"）
    #   最低余弦 0.303 → **0.404**。阈值只要高于它就会误杀这批——这正是 0828 把 0.3 放宽到
    #   0.2 的原因。故新值取 0.404 − 0.10 ≈ **0.30**，保留与旧值同构的安全余量。
    purpose_filter_threshold: float = 0.30

    # ⚠️ 标定语料的代表性（0930 复审补记）：`eval/probe_threshold_retune.py` 的样本集是
    #   **14 组中英对照样本**（见 版本变更记录 §六.2），规模小，且都是"动机→行为"这类
    #   **同质度低**的对照对。在节点多、节点间普遍相近的图上，目的分的分布会被整体压缩，
    #   0.30 这个**绝对**阈值就可能退化成"几乎全过"或"几乎全杀"——与峰值容忍带同一类
    #   尺度问题。
    #   故本轮**不动该值**（在拿到大图读数前改它，等于又做一次小图标定），改为先把
    #   **逐跳通过率**打出来（见 retriever 的 `purpose_pass` 与 space 的 `检索:` 行），
    #   有读数再定标。

    # ---- 峰值容忍带带宽（0930 复审接入）----
    #
    # 历史遗留：`peak_tolerance` 定义在 `PeakFinder`（库默认 0.10），但 `retriever`（第 266 行）
    # **从未显式传入**（0815——debugPlan 已记录），所以线上生效值一直是库默认；而 0.10 当年的
    # 标定语料是自建小数据集（0925 实验：comprehensive 163 节点 / lifemem 322 节点，同质场景少）。
    #
    # 问题（0927 实测，600 次线上检索）：`峰值/候选` 中位 **0.81**、候选≥300 档 **0.79**——即
    # "寻峰"平均只筛掉 19%。原因是 δ 为**绝对**带宽，而真实语料的余弦分布很窄
    # （scriptMem 18 次全在 0.5497~0.5884、全距 0.039，δ=0.10 已是全距的 2.5 倍）。
    # 该退化**与 embedding 无关**：换 bge-m3 前后均为 ≈0.80。
    #
    # 两个参数（默认都不改变现行行为）：
    #   peak_tolerance       —— 绝对带宽，**默认与现状逐字一致（0.10）**；
    #   peak_tolerance_ratio —— 相对带宽：>0 时 δ = ratio × (peak_mean − min(轮均分))，
    #       即"只保留观测范围的上半部"。它**尺度无关**，因而**自动随场景（图规模/同质程度）**
    #       调整，不需要人工分档；各轮均分全等时 δ=0（只留峰值轮，不会清空结果）。
    #       **默认 0 = 关**（走绝对模式）。建议起始值 0.5，待本地大图复算后再定。
    peak_tolerance: float = 0.10
    peak_tolerance_ratio: float = 0.0

    # 语义去重阈值（GraphBuilder.dedup_threshold）：新节点与已有节点的余弦达此值即判为
    # 「同一条记忆」，不再新建。同样是**余弦阈值**，与 embedding 模型强绑定。
    #
    # 它直接决定 D1「新值覆盖与当前状态」能否成立：同一属性的**新取值**若被判为重复，
    # 新值节点根本不会落库（表现为 update/deprecate 计数长期为 0）。实测（0925 阈值重标
    # 探针）「同一模板只改数值」的句子在 bge-m3 下余弦 p50 已到 0.902，而旧值为 0.85——
    # 会**成片误吞新值**。故随 embedding 更换一并上移到 0.92。
    # （注意该档两类样本本身有重叠：不应去重 max 0.948 > 应去重 min 0.915，单阈值分不干净，
    #   0.92 是折中；根治要靠写侧改进而非阈值。）
    graph_dedup_threshold: float = 0.92

    # ---- 写入分批（对齐 0828 §5.5 的 M=6）----
    # 单次 Add 的消息按此条数切分，**逐组**调用 DBA 维护。
    #
    # 为什么必须有：ALM 的 add() 原先把整批消息（上限 max_messages_per_add）一次性交给
    # dba.maintain()，而 MCP 侧走的是 MaintenanceScheduler 的 proactive_batch_size=6。
    # 平台 smoke 实测单次 Add 最多 20 条、中位 8 条，即 ALM 侧长期跑在 M=8~20 上。
    # 后果（线上图文件实测 + 日志统计）：
    #   · 值变更（update/deprecate）在 >6 条的 73 个批次里**一次都没出现**，
    #     而 8 次 update 全落在 3~6 条批次、4 次 deprecate 全落在 1~2 条批次；
    #   · 节点产出率从 5.47 节点/条（1-2 条）单调掉到 1.21（15+ 条）；
    #   · 抽取调用 in token 中位 2376、max 21843，out 触顶 8192 → 出现
    #     「LLM 输出疑似被截断，已挽救 node_ops=0 edge_ops=343」这类整批丢节点的情形。
    # 0828 §5.5 已扫过 M=3/6/9：M=6 为生产默认（M=9 在已有图上耗时翻倍、误 deprecate 率高）；
    # §五 另测「整段一次灌入 → 压缩更深、联想路径断裂」。故此处对齐 6。
    # 0 或负数表示不切分（退回旧行为，用于 A/B 对照）。
    add_batch_size: int = 6

    # 单组的字符预算：与 add_batch_size 构成**双约束**（见 space._split_add_batches）。
    #
    # 为什么条数不够：0828 §5.5 的 M 是"多少轮**本地短对话**"，而平台的输入形态不同——
    # 实测有"3~6 条一批、单条均长 505 字"这类长消息对话，6 条就是 3000 字。只按条数
    # 切时线上仍有 9% 的组超过 1500 字（最大 2934），这些组因超过 triage 可信上限而
    # 整段直送抽取；输入一长，抽取 LLM 就"少切、切粗"，节点内容从单一事实退化成概述
    # → 意图模糊 → 连边判不出关系类型（线上边型 scenario 65.6% / causal 仅 21.9%）。
    # 双约束后 0 组超限，代价是组数 292 → 307（+5%）。
    #
    # 取 1500 与 `dba._TRIAGE_TRUST_MAX_CHARS` 对齐：这样每个组的窗口结论都可信，
    # triage 得以正常发挥，而不是被长度闸整段放行。
    # 0 = 不按字符切（仅按条数），用于 A/B 对照。
    add_max_chars: int = 1500

    # ---- 文档通道：异构输入路由 ----
    # 平台输入混了两类形态完全不同的数据：**对话记忆**（本系统的设计目标）与
    # **知识库式长文档**（整部剧本、论文+系统提示词、NTSB 摘录、任务轨迹）。
    # 后者交给节点/边抽取是范畴错误：抽取产出为空 → 走兜底 → 原文被切成几百个
    # 孤立节点灌进图。线上图文件实测：32 个平台图节点全是兜底块、边数恰好为 0，
    # 最大一个 289 个节点 0 条边（原文约 14.5 万字符，是一整部易卜生剧本）。
    # 这些孤立块随后被当成记忆种子 → 峰值出现在 hop0 → `跳=1` → 图扩展零次。
    #
    # 故在 add() 入口按**形态**路由（纯计数，不调 LLM）：
    #   文档型 → 切块 + 向量化存入**独立文档区**（不进图，不参与种子/扩展/峰值）
    #   会话型 → 现有链路（含 add_batch_size 分批）
    #   噪音   → 直接丢弃
    doc_route: bool = True

    # triage 维护判断（"这段对话值不值得维护"）是否启用。
    #
    # **0927 起默认 True（与 MCP 一致）**。此前 ALM 侧默认关闭，理由是 2026-09-26 冒烟实测：
    # 写入分批（add_max_chars=1500）把一次 Add 切成小组后，多数组长度落回
    # dba._TRIAGE_TRUST_MAX_CHARS=900 以下，于是 triage 重新参与判定——**421 个组里
    # 173 个被判 SKIP（41%）**，兜底率 11.1% → 50.7%，图只拿到约 59% 的内容。
    #
    # **为什么又改回开**（战略转向"保系统健康"，见 版本变更记录 §七）：
    #   ① "SKIP 会丢内容"这条**已被 B5 修复**——MCP/ALM 两侧现在都有原文兜底
    #      （`_divert_raw` / `DBA.store_raw`），SKIP 不再等于丢数据；
    #   ② 于是"关掉 triage"的剩余收益只是"图更完整"，代价是**失去 triage 的省调用效果**
    #      （关掉后每个组都要真跑一次抽取）→ 它变成纯成本/质量取舍，不再是健康必需；
    #   ③ 形态/口径类改动一律"加开关、默认关"（此处即回到原始行为）。
    # 仍需注意的是**判定质量**本身（113 字符、窗口覆盖 100% 的组仍被判 SKIP，见
    # 版本变更记录 §六.4）——那是要改 prompt 判据的事，与开不开无关。
    triage_enabled: bool = True

    # 形态判据：单条平均长度 ≥ doc_chars_per_msg **且** 消息条数 ≤ doc_max_messages。
    #
    # ⚠️ 这个判据只作**极端保护**，不是主要分流手段。为什么：长度分不出「长对话」和「文档」。
    # 用线上 135 批回测过——把阈值定在 500 时，28 个「≤2 条」的批里有 18 批被判为文档，
    # 而其中 **14 批抽取成功、合计 155 个节点**会被白白丢掉；把阈值抬到 3000 仍误吞 7 批、
    # 5000 仍误吞 4 批。长度这条路走不通。
    #
    # 真正可靠的判据是**抽取结果**：已确认为文档的那些批次（导出图里 289/58/40/21 个节点、
    # 0 条边、节点全是兜底块）在日志里对应的那批 `create=0`；而可抽取的内容一律 `create>0`。
    # 故主要分流点放在**抽取之后**（见 space.add()：no_ops / triage / 超量 → 原文送文档
    # 通道，不再兜底成图节点），此处只保留一个"输入量级明显异常、先去保护 LLM"的闸门。
    #
    # 10000 的来历：已确认的文档型输入规模是 **1 万~14.5 万字符**（21/40/58/289 块 × 500），
    # 而日志里抽取成功过的最大单条是 **7987 字符**（`create=13`）。取 10000 落在两者之间，
    # 既能挡住超大输入，又不会误吞任何观测到过的可抽取内容。
    doc_chars_per_msg: int = 10000
    doc_max_messages: int = 2

    # 文档块字符数。**实测**（eval/probe_m3_limit.py，SiliconFlow）：
    #   英文 16000 字符仍 200；中文 8000 字符 200、16000 报 400 code=20015。
    # 取 1800：中英都安全、留有充足余量，且比图节点的 500 字块（那是
    # bge-large-zh 512-token 时代为对话短句定的）大 3 倍多，块数少一个数量级。
    # 文档块不经过图，故此值放大不会波及节点粒度与图结构。
    doc_chunk_chars: int = 1800

    # 文档通道召回条数与相邻块扩展半径。文档块之间有**天然顺序**（块序），
    # 这是文档唯一的"边"：命中块时把 ±doc_neighbor 个邻块一并取回——
    # 「谁说了这句」这类问题往往需要跨块上下文，纯 top-k 会丢。
    doc_lookup_k: int = 8
    doc_neighbor: int = 1

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
    #
    # ⚠️ 0925 换 embedding（bge-large-zh-v1.5 → bge-m3）后上移为 **0.48**。依据：
    #   eval/probe_threshold_retune.py 实测（同端点、同样本、只换模型），
    #   「无答案」样本的最低余弦 0.360 → **0.506**、p50 0.479 → 0.552。
    #   0.35 已**落在这类分布之下**，等于硬下限从不生效；且"无关"一侧整体上移的幅度
    #   （+0.073）大于"有答案"一侧（+0.018），两类间距被压缩，旧值必然失效。
    #   新值沿用旧关系「硬下限略低于无答案最低值」：0.506 − 0.03 ≈ **0.48**。
    abstain_cosine: float = 0.48

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
    #
    # ⚠️ 0925 换 embedding 后上移为 **0.60**。依据：bge-m3 下「无答案」样本的最高余弦
    #   0.520 → **0.583**、p90 0.520 → 0.583。若上界仍取 0.50，则**无答案查询全部落在
    #   上界之上**——既进不了模糊带（无 LLM 判定兜底），又高于硬下限（不被拦），
    #   结果是**该弃权的查询一律放行返回内容**，H 维度会直接崩。取 0.60 略高于该类上限。
    abstain_verify_hi: float = 0.60
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
    # 注意：日期信息另有独立通道，不受本开关影响——兜底（原文）条目的 `created_at` 由
    # `space._node_created_at()` 输出；**叙事条目的 `created_at` 由下面的
    # `story_created_at` 单控**（0927：叙事条目曾刻意不带它，与本句此前的措辞相矛盾）。
    # 详见 Plan §7.10.5 D2/F。
    render_timestamps: bool = False

    # 叙事条目是否也带 `created_at`（= 本次叙事主证据里**最新的记录时点**）。
    #
    # 0927 曾无条件开启（记为 L2a），动机是让 "yesterday / last month" 这类相对时间有锚点
    # 可换算。**平台实测为无效**：开启后 `created_at` 确实从 46/46 `None` 变成 24/46 带日期
    # （如 locomo1 的 2023-01-20），但该题的平台回写仍是 `predicted=Yesterday.` / score=0
    # ——锚点给了、下游没用（平台是否把它渲染给 Answer 模型，我们这一侧无法确认）。
    # 故按"保系统健康"（无效改动不占默认位）**默认关闭**；置 `ALM_STORY_CREATED_AT=1` 开启。
    story_created_at: bool = False

    # 叙事是否**按场景 / 时段分段**（StoryRank 规则 12）。**默认关闭**。
    #
    # 起因（0930）：线上 story 实测把**互不相邻的多个场景**塞在同一个自然段里，且时序前后颠倒
    # （例：同段内先"陪审团落座"、再"走廊里人来人往"、又回到"投票举手"），造成逻辑断裂。
    # 结构成因有三，本条只解其中一条：
    #   ① 供料顺序 = **全局层次序**（先所有根、再逐层铺开），**与故事时间无关**
    #      （`retriever._build_path` → `_order_by_level`：只按"根/层"排，不按分数也不按时间）；
    #   ② 默认 `show_ts=False` → 节点**没有任何时间标注**；
    #   ③ 规则 2 要求「只输出一段文字」——不同场景因此没有边界。
    # 规则 12 解除 ③（允许 2~5 段、同场景同段），并限定定序依据只能来自 SEQUENCE/TEMPORAL 边
    # 或「记录于」时间，**都没有时保持中性并列、不得臆造先后与因果**。
    # ①② 未解 → 若要进一步改善，需在供给端按时间/序列边排序（下一步）。
    story_scene_paragraphs: bool = False

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

    # 压缩路由：与 temporal_route 同形状（复用同一次 infer_purpose 调用拿布尔标签、
    # 产出只影响输出形态、不碰检索控制流、任何失败静默退回）。
    #
    # 动机：F1「摘要与压缩/长历史全局综合」连续多轮 0 分。而用真实检索器离线实测，
    # 该能力的 gold 事实 **97% 已进入 StoryRank、83% 被故事采纳**——检索侧不是瓶颈。
    # 错配在**输出形态**：主通路的设计目标是「联想召回 + 铺成连贯叙事」，检索会带回
    # 覆盖全图 41%~64% 的节点集（实测峰值 133~205 / 322），StoryRank 再把这些**铺开**
    # 成一段故事——那是**扩写**，而 F1 要的是「压缩成短文本并保住精确事实」。
    #
    # 因此判定为压缩类问题时，切到 STORY_RANK_PROMPT_CONDENSE：只保留精确事实
    # （人名/时间/数值/实体），禁止把关系铺开成句子、禁止补充节点外的内容。
    #
    # 默认 False（0927 起）。它属"输出形态补丁"：只为把 F1 拉起来，且依赖 `infer_purpose`
    # 的判定——而该推断有 ~8% 静默失败率（审查 C2），失败即悄悄退回主变体。
    # 按"保系统健康"的处置原则（形态类一律加开关、默认关）改回 False；
    # 置 `ALM_CONDENSE_ROUTE=1` 即启用。
    condense_route: bool = False

    # 状态路由：与 condense_route 同形状（复用同一次 infer_purpose 调用拿布尔标签、
    # 只影响 StoryRank 的 prompt 变体与时间渲染、不碰检索控制流、失败静默退回）。
    #
    # 动机：D1「新值覆盖与当前状态」连续 5 轮 0 分。官方能力定义里写作
    # "Can it distinguish order, updates, and the latest valid state?"——要的是**最新
    # 有效状态**。而 ALM 的输出默认**不带任何时间信息**（render_timestamps 关闭，
    # 正常路径的 story 也不带 created_at），Answer 侧无从裁决同一属性的多个取值。
    #
    # 开启后：判定为「问当前/最新取值」时，改为渲染节点的记录时间（复用既有 TS 变体）
    # 并叠加 STORY_RANK_PROMPT_STATE 的规则 11——把记录时间显式用作新旧判据。
    #
    # 注意：0920 对 render_timestamps 的 A/B 是**整体分**净 0，故其默认仍为 False；
    # 本条不是把它全局打开，而是**按意图门控**——只影响问当前取值的那一小类问题。
    # 默认 False（0927 起）：同 `condense_route`——形态补丁 + 依赖会静默失败的推断，
    # 按"保系统健康"改回关闭；置 `ALM_STATE_ROUTE=1` 即启用。
    state_route: bool = False

    # 精确证据路由：**提问要唯一/精确答案时，把读出收拢到根节点与 1 hop 以内**。
    #
    # 动机（0930 实测）：本系统的联想设计是"抓一大把相关节点"，对召回与叙事丰富度有利，
    # 但在要求精确的问答上，多出来的那部分就是噪声。同一张 400 节点图上，判题模型的表现
    # 只跟**证据密度**走：
    #   · 记忆 = 该主体的 5~6 条证据本身（343~403 字）→ gemini/qwen **三次全选 F**（正确）；
    #   · 记忆 = 叙事产物（约 1,800 字，证据被无关节点稀释到 20~25%）→ 同样两个模型读 A。
    # 而 hop 深度直接决定密度，**单靠 StoryRank 一层收不拢**——它只能按给定集合写。
    #
    # 与 condense/state 两个路由的区别：那两个只改 prompt 变体、不碰检索控制流；本路由是
    # 第一个**动检索深度**的路由，故必须一眼可见何时生效（`检索:` 行打 `精检=是/否`）。
    # 判据是**本地形态判别**（`alm.space._is_precision_query`：内联选项块 / 作答指令 /
    # 契约的 `options` 字段），零 LLM 调用、确定性、可单测；**目的向量仍只由题干编码**，
    # 两者互不干扰（选项参与判型、不参与嵌入）。
    # 默认 False（沿用 condense/state 的口径：新路由先关，置 `ALM_PRECISION_ROUTE=1` 启用）。
    precision_route: bool = False

    # 命中精确路由时的展开跳数（1 = 只留根节点与其 1 hop 邻居）。
    precision_max_hops: int = 1

    # 命中精确路由时的读出上限（按 combined_score 取前 N）。
    #
    # 为什么还需要条数上限：只收跳数不收条数时，读出仍有 148~152 条（seed_k=40 × 每跳
    # expand_k=40），叙事照样写到 5~10 倍预算再被硬截断，证据密度没上去（实测正文证据
    # 命中反而从 4~5/8 掉到 1~2/8）。标定依据：叙事产出与输入长度大致 1:1（148 条 → 10,543
    # 字 ≈ 71 字/条），而篇幅预算是 2,000 字、节点正文中位约 75 字 → **约 24 条**正好让整段
    # 叙事落在预算内、"写满"= "写完"，密度接近 100%。
    #
    # 默认 **0 = 自适应**：由检索层按篇幅预算与**本图节点的实际长度**反推条数
    # （`retriever._auto_peak_cap`，夹在 4~40）。定长 24 只在这张图上标定过，换图（节点长短、
    # 叙事繁简）都会偏；>0 才是固定条数（排查时用）。
    precision_max_peak_nodes: int = 0

    # 边连接步骤的旧节点上下文也走「逐消息并集」宽召回（见 MemoryDBA.edge_wide）。
    #
    # 动机：B2「因果链、路径与中间步骤恢复」连续 5 轮 0 分。离线实测图的因果/时序边是
    # **批内短程**的——以节点顺序作时间代理，causal/sequence 边的中位索引距离只有 8/9，
    # 而随机基线是 107；只有 42% 的节点有 causal/sequence 出边，能走满 3 跳的仅 21%。
    # 而边连接提示词的原则 3 早就要求「新事实必须尽量接入已有图谱结构，能连则连」——
    # 说明**不是提示词没要求，而是 LLM 看不到该连的旧节点**：边步骤的旧节点上下文只有
    # 15 条、按整段对话相似度竞争，长对话下被稀释（与节点步骤同一个问题）。
    edge_wide: bool = True

    # ---- 冒烟期输入输出留痕 ----
    # 开启后把 /add 的入参原文与落库节点原文、/search 的 query/options 与返回条目原文
    # 整段写进运行日志。
    #
    # 为什么可以这么做：ALM 冒烟用的是平台自带的公开测试集（user_id 为 `u_*` 合成 id），
    # **不含真实用户记忆**，因此留痕不涉及隐私口径——而离线复现不了平台查询，只看计数
    # 无法判断「叙事是否退化成了全图概览」「选择题选项是否被消费」这类问题，必须看到
    # 实际文本才能定案。
    #
    # **默认关闭，且 Full 评测前必须确认仍为关闭**：Freeze 之后平台会灌入真实数据，
    # 那一刻起留痕就会把真实记忆正文写进日志，与本项目对外的合规声明冲突。
    # 置 ALM_TRACE_IO=1 开启（仅用于自测与冒烟排障）。
    trace_io: bool = False

    # 留痕**单条**的字符上限（`trace_io=1` 时生效）。0 = 不截断。
    #
    # 为什么必须有界：0930 Full 实测 `trace_io=1` 时**单个日志文件 84 MB**、约 60 秒写满
    # 一次轮转（100m×10）；平台上确有把整份文档当 query 送进来的轨道，单条 `[IO]` 可达
    # 数 MB。截断保留**头部**（形态判定看的是开头）并标注原长，grep 仍可用。
    #
    # 注意它不改变"是否留痕"（那是 trace_io），只把单条记录的体积钉住——
    # 于是"开着 trace_io 也能跑完一轮"成为可能，而 Full 期仍应把 trace_io 置 0。
    trace_max_chars: int = 4000

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
            llm_max_tokens=_env_int("ALM_LLM_MAX_TOKENS", 0),
            llm_timeout=_env_float("ALM_LLM_TIMEOUT", 120.0),
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
            max_peak_nodes=_env_int("ALM_MAX_PEAK_NODES", 0),
            purpose_filter_threshold=_env_float("ALM_PURPOSE_FILTER_THRESHOLD", 0.30),
            peak_tolerance=_env_float("ALM_PEAK_TOLERANCE", 0.10),
            peak_tolerance_ratio=_env_float("ALM_PEAK_TOLERANCE_RATIO", 0.0),
            graph_dedup_threshold=_env_float("ALM_GRAPH_DEDUP_THRESHOLD", 0.92),
            add_batch_size=_env_int("ALM_ADD_BATCH_SIZE", 6),
            add_max_chars=_env_int("ALM_ADD_MAX_CHARS", 1500),
            triage_enabled=_env_bool("ALM_TRIAGE", True),
            doc_route=_env_bool("ALM_DOC_ROUTE", True),
            doc_chars_per_msg=_env_int("ALM_DOC_CHARS_PER_MSG", 10000),
            doc_max_messages=_env_int("ALM_DOC_MAX_MESSAGES", 2),
            doc_chunk_chars=_env_int("ALM_DOC_CHUNK_CHARS", 1800, minimum=1),
            doc_lookup_k=_env_int("ALM_DOC_LOOKUP_K", 8),
            doc_neighbor=_env_int("ALM_DOC_NEIGHBOR", 1),
            fallback_dedup_threshold=_env_float("ALM_FALLBACK_DEDUP_THRESHOLD", 2.0),
            orphan_threshold=_env_int("ALM_ORPHAN_THRESHOLD", 10 ** 9),
            abstain_cosine=_env_float("ALM_ABSTAIN_COS", 0.48),
            abstain_verify_hi=_env_float("ALM_ABSTAIN_VERIFY_HI", 0.60),
            abstain_judge_top_n=_env_int("ALM_ABSTAIN_JUDGE_TOP_N", 3),
            abstain_judge_max_chars=_env_int("ALM_ABSTAIN_JUDGE_MAX_CHARS", 200),
            render_timestamps=_env_bool("ALM_RENDER_TIMESTAMPS", False),
            story_created_at=_env_bool("ALM_STORY_CREATED_AT", False),
            story_scene_paragraphs=_env_bool("ALM_STORY_SCENE_PARAGRAPHS", False),
            temporal_route=_env_bool("ALM_TEMPORAL_ROUTE", False),
            temporal_k_seed=_env_int("ALM_TEMPORAL_K_SEED", 8),
            temporal_max_facts=_env_int("ALM_TEMPORAL_MAX_FACTS", 8),
            temporal_max_anchors=_env_int("ALM_TEMPORAL_MAX_ANCHORS", 3),
            condense_route=_env_bool("ALM_CONDENSE_ROUTE", False),
            state_route=_env_bool("ALM_STATE_ROUTE", False),
            precision_route=_env_bool("ALM_PRECISION_ROUTE", False),
            precision_max_hops=_env_int("ALM_PRECISION_MAX_HOPS", 1),
            precision_max_peak_nodes=_env_int("ALM_PRECISION_MAX_PEAK_NODES", 0, minimum=0),
            edge_wide=_env_bool("ALM_EDGE_WIDE", True),
            trace_io=_env_bool("ALM_TRACE_IO", False),
            trace_max_chars=_env_int("ALM_TRACE_MAX_CHARS", 4000),
        )
