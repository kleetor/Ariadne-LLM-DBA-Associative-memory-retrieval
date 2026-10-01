# SPDX-License-Identifier: AGPL-3.0-only

"""
LangChain LLM 封装：状态推断 + 目的推断 + 回复生成
"""

from datetime import datetime
import logging
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dba_pipeline.llm.roles import with_role

logger = logging.getLogger(__name__)


def format_ts_ms(ts) -> str:
    """Unix 毫秒 → 本地日期字符串；非整数/越界返回空串。

    注意语义：节点 timestamp 是**记录时间**（写入图谱的时刻），不是事件发生时间。
    渲染时必须显式标注「记录于」，否则会被误当成事件日期。
    """
    if not isinstance(ts, int) or isinstance(ts, bool):
        return ""
    try:
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return ""


# ---- 状态推断 Prompt ----

STATUS_INFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个心理状态分析助手。分析用户当前消息，推断其状态。

输出格式（严格 JSON）：
{{
    "status": "当前状态",
    "emotion": "情绪标签"
}}

状态示例：压力大、疲惫、开心、困惑、焦虑、放松
情绪示例：负面、正面、中性
只输出 JSON。"""),
    ("human", "{query}"),
])

# ---- 目的推断 Prompt ----
# 该 prompt 同时承担一个**时序二元标签**（temporal），供 ALM 的 /search 决定是否调用
# 独立时序工具 temporal_lookup：把它合并在这一次调用里，是为了「不新增 LLM 调用」，
# 且判定只依赖 query（见 Plan §3.2）。判据刻意保守——误判为 true 会引入无关时间锚点、
# 稀释主通道，故要求「拿不准一律 false」。

PURPOSE_INFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是对话意图分析助手。根据用户消息，先判断查询类型，再推断用户的目的。

[OUTPUT LANGUAGE — HARD CONSTRAINT]
Write the `purposes` and `status` strings in the SAME LANGUAGE as the user message.
English message → English phrases (e.g. "get information", "confirm details").
Chinese message → Chinese phrases. Never translate. This rule outranks the language of
these instructions and of the examples below.

查询类型：
- information：信息查询（询问事实/偏好/经历/细节，如「用户喜欢喝什么咖啡？」「他高考考得怎么样？」）
- emotional：情绪表达（倾诉、吐槽、寻求安慰、分享感受）

输出格式（严格 JSON）：
{{
    "status": "用户当前状态",
    "query_type": "information 或 emotional",
    "temporal": true 或 false,
    "condense": true 或 false,
    "state": true 或 false,
    "purposes": ["目的1", "目的2", "目的3"]
}}

规则：
- **purposes 必须使用与用户消息相同的语言**：消息是英文就用英文短语（如 "get information"、
  "confirm details"、"reconstruct the timeline"），是中文才用中文短语（如「获取信息」「确认细节」）；
  `status` 同理。目的短语会被编码成向量，用于「目的驱动种子」与「目的回归过滤」两处余弦匹配——
  语言与库中节点不一致会**系统性**拉低匹配度（表现为候选被过滤光、跳=1）。**禁止把英文查询的目的写成中文。**
- information 类型：目的必须是信息获取类，禁止社交 / 情绪类目的
- emotional 类型：目的可以是倾诉 / 寻求建议 / 理解原因 / 寻求安慰 等
- temporal 是**二元判断**：只有当查询明确要求还原「某个时间点/时间段发生了什么」时才置 true，
  如「上周体检结果怎么样」「昨天下午做了什么」「寒假去了哪」「高中时成绩如何」；
  其余（问偏好、问身份、问属性、问原因、闲聊、情绪表达等）一律 false。
  **拿不准一律 false**：误判为 true 会引入无关的时间锚点。
- temporal 只作独立布尔标签，不影响 purposes 的推断。
- condense 是**二元判断**：只有当用户明确要求「把内容概括 / 压缩 / 总结成简短文本」时才置 true，
  如「把上面这段总结成三句话」「概括一下我们都聊了什么」「帮我压缩成要点」；
  其余（问某件事怎么样、问细节、问原因、闲聊、情绪表达等）一律 false。
  **拿不准一律 false**：误判为 true 会让回答从叙事切成要点式压缩。
- state 是**二元判断**：只有当查询在问「某个属性的**当前 / 最新取值**」、且该取值**可能随时间变过**
  时才置 true，如「用户现在住哪」「他最近改用什么技术栈了」「目前的工作是什么」；
  问过去某个时间点的状态（「高中时成绩如何」）、问原因、问偏好、闲聊一律 false。
  **拿不准一律 false**。
只输出 JSON。"""),
    ("human", "{query}"),
])

# ---- 记忆故事整理 StoryRank Prompt ----

# 叙事（story）的字符上限。**必须定义在 _STORY_RANK_SYSTEM 之前**：该常量会注入到
# 规则 6 的正文里（单一来源，避免数字两处漂移）。
#
# 为什么需要显式预算：0926 实测，取消输出上限（max_tokens 8192→32768 真生效）后，
# 叙事不再被截断，于是模型写出了**19,962 字符的全剧概览**（采纳 30 / 喂入点 265，
# 665 字符每节点），而该题得 0 分；同类最长返回 p90 从 2,884 涨到 4,663 字符。
# 根因不是"上限不够"，而是**没有任何篇幅约束**——规则只说"以讲清链路为限"，
# 模型便把几十个节点逐个展开。取 2000：健康返回的中位数在 800~950 字符
# （0926 两轮实测 815 / 952），2000 留了一倍余量，同时把 2 万字符的尾巴彻底切掉。
_STORY_MAX_CHARS = 2000

# 篇幅预算的**动态兜底**（0930）：超预算时先"强制压缩指令重试"，仍超则**缩喂重试**——
# 按当前顺序（全局层次序：根在前）逐步收缩喂入节点，直到正文落进预算。
#
# 为什么不直接截正文：截正文丢的是**已写出来的尾部**（跨场景的收束/结论），而缩喂丢的
# 是**优先级最低的尾部节点**——后者在问答里可替代性高得多。0930 线上实测 46 次检索里
# 18 次超预算、8 次走到硬截断，被丢的正是"变化/收束"那一段。
_STORY_FIT_MAX_ROUNDS = 3      # 缩喂最多几轮（每轮一次 LLM 调用）
_STORY_FIT_SHRINK = 0.6        # 每轮保留的节点比例
_STORY_FIT_MIN_NODES = 6       # 缩到该条数就停（再少就不成叙事）

_STORY_RANK_SYSTEM = """你是一个记忆故事整理助手。你会收到用户当前消息，以及一组「记忆节点与关系」（可能包含多条因果链路）。

[OUTPUT LANGUAGE — HARD CONSTRAINT]
Write the whole `story` in the SAME LANGUAGE as the user's current message. English message →
English story. Chinese message → Chinese story. Never translate, never mix two languages
(keep proper nouns / tech stack names / organisation names as-is). Reason: this text is handed
directly to the downstream answering model as memory evidence; a language mismatch forces it to
translate before answering. This rule outranks the language of the instructions and examples below.

**输出语言（硬约束）**：story 必须使用与「用户当前消息」**相同的语言**——消息是英文就整段写英文，
是中文才写中文。禁止翻译、禁止中英混排（专有名词、技术栈、机构名的原文可保留）。
理由：这段文本会直接作为记忆证据交给下游回答模型，语言与提问不一致会迫使它先做一次翻译再作答。

你的任务：
1. 理解这些节点与关系：每个节点是一个记忆片段（带类型），边表示节点之间的关系（带关系类型）。
2. 把所有记忆整合成一段连贯、自然的故事，让读者能顺畅理解用户的状态与经历。只输出一段文字，不要分段列举。
   **只陈述节点里已经写明的事实：不要解释这些事实意味着什么，不要推断任何人的立场、
   动机、意图或心理，也不要用"因为…所以…"去替读者下结论——判断留给下游回答模型。
   你多写一句解释，就可能把它带偏到错误的选项上。**
3. 关系语义要自然融入句子，例如：
   - CAUSAL（因果）→ 因为 / 导致 / 所以（**只在边上确实标了 CAUSAL 时才可用**）
   - SEQUENCE（时序）→ 然后 / 之后 / 接着
   - PREFERENCE（偏好）→ 偏爱 / 喜欢
   - SCENARIO（场景）→ 在…场景下 / 常去
   - ATTRIBUTE（属性）→ 具有…特征
   - TEMPORAL（时间）→ 时间上
   - SOCIAL（社交）→ 与…有关
   - TAXONOMIC（分类）→ 属于 / 是一种
4. 因果要克制：只有边类型为 CAUSAL 的两点之间才写成因果；相邻、并列或仅有时序关系的节点，
   只能平铺成"同时/随后"，**不得自行推断出原因、动机或后果**。
5. 主体不能串：多方对话里，每条事实的动作主体必须与该节点内容一致。
   不得把甲的遭遇安到乙头上，也不得用"这也让某人…"把两个人的事件缝成一条因果。
6. **先定位，再叙述**。「用户当前消息」问的通常是一个**具体的时刻 / 动作 / 变化**，
   而不是某个话题。先找出节点里真正对应该时刻 / 动作 / 变化的那么几条（彼此相邻、
   互有边相连者优先），围绕它们写，并把直接的前因后果接上。
   仅与问题**同一话题但发生在别的时刻**的节点——人物介绍、开场布置、他处进行的活动
   ——只作背景，最多一句话带过，不要展开。
   **例外（问题问的是"变化 / 演变 / 一贯性"时）**：若「用户当前消息」问的是**同一主体的
   前后变化、立场演变或长期一贯性**，那么该主体在**不同时刻**的事实**不得当作背景丢掉**：
   按时间先后把它们**并置列出**（可用"起初…；后来…"这类只有先后、没有评价的写法），
   顺序依据**只能**来自节点上标明的 SEQUENCE / TEMPORAL 边或「记录于」的时间，
   不得臆造先后。
   自检：与「用户当前消息」相关的**事实**是否都已写进正文？没写到的补上，写进去的
   按事实陈述。**可以并置先后事实，但不得替它下结论**——不要给"变化"命名
   （"因此他改变了立场""这说明他反对""他之所以…是因为…"），含义、动机、意图一概不写；
   判断留给下游回答模型。
   **篇幅上限：story 不超过 __STORY_MAX_CHARS__ 字符**（含标点）。宁可短而中靶，
   不可长而全——把几十上百个节点逐一复述会平铺成流水账，下游回答模型反而找不到
   答案；而且一旦超出输出上限被截断，整段作废。
7. 忠实于节点内容：节点里没写的动机、原因、结果一律不补；宁可少写，不可写错。
8. `used` 是**可选整数**：填**正文实际写到的节点数**（不是候选总数，也不是节点总数）。
   估不准就省略，不算输出错误。**不要为了凑这个数字去扩写正文。**

输出格式（严格 JSON）：
{{
    "story": "一段连贯的故事文本",
    "used": 12
}}
只输出 JSON。"""
# 上一条规则里的篇幅上限由 _STORY_MAX_CHARS 单一来源注入，避免数字两处漂移。
_STORY_RANK_SYSTEM = _STORY_RANK_SYSTEM.replace("__STORY_MAX_CHARS__", str(_STORY_MAX_CHARS))

_STORY_RANK_HUMAN = """用户当前消息：
{query}

记忆因果链路：
{path}"""

# 仅当渲染记录时间时追加：时间戳是"写入图谱的时刻"，不是事件发生时间，必须显式区分。
_STORY_RANK_TS_RULE = """

9. 节点若标注「记录于 YYYY-MM-DD」，那是这条记忆的**记录时间**（写入图谱的时刻），
   **不是事件发生时间**。讲述事件经过时不要把它当作事情发生的日期；
   只有当用户问"什么时候记下的/什么时候知道的"时才可以引用它。"""

STORY_RANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _STORY_RANK_SYSTEM),
    ("human", _STORY_RANK_HUMAN),
])

# 带记录时间的变体（MCP 侧使用；ALM 侧在 render_timestamps=False 时走上面的 STORY_RANK_PROMPT）。
# 两个变体共用同一份 _STORY_RANK_SYSTEM，故「输出语言」等约束对两侧同时生效——这是刻意的，
# 避免 MCP 与 ALM 的叙事语言规则分叉。
STORY_RANK_PROMPT_TS = ChatPromptTemplate.from_messages([
    ("system", _STORY_RANK_SYSTEM + _STORY_RANK_TS_RULE),
    ("human", _STORY_RANK_HUMAN),
])

# 仅当判定为「压缩 / 概括 / 总结」类问题时追加（ALM 的 condense_route）。
#
# 为什么需要变体：主通路的设计目标是「联想召回 + 铺成连贯叙事」，而离线实测该能力的
# gold 事实 97% 已进入本步骤、83% 被采纳——**检索侧不是瓶颈，错配在输出形态**：检索会
# 带回覆盖全图 41%~64% 的节点集，规则 1/2 又要求把它们铺开成一段连贯故事，那是**扩写**。
# 本条把压缩类问题的形态反过来：短、并列、只留精确事实。
_STORY_RANK_CONDENSE_RULE = """

10. **若用户消息要求概括 / 压缩 / 总结成简短文本**，本条优先于上面第 1、2、3 条：
   - 只输出**压缩后的短文本**，篇幅应显著短于所给节点的总长度，不要逐条铺开、不要写成完整故事；
   - **必须保住精确事实**：人名、时间（日期/时段/相对时间）、数值、地点、机构、技术栈等
     一律不得省略、不得模糊化、不得改写；
   - **不得把关系铺开成句子**（不要"因为…所以…""之后他…"这类展开），可改为要点式并列；
   - 不得补充节点里没有的内容；宁可少写，不可写错。
   若用户消息不要求概括/压缩，忽略本条。"""

STORY_RANK_PROMPT_CONDENSE = ChatPromptTemplate.from_messages([
    ("system", _STORY_RANK_SYSTEM + _STORY_RANK_CONDENSE_RULE),
    ("human", _STORY_RANK_HUMAN),
])

# 仅当判定为「问当前 / 最新取值」类问题时叠加（ALM 的 state_route）。
#
# 为什么需要：D1「新值覆盖与当前状态」连续 5 轮 0 分。官方能力定义里这一条写作
# "Can it distinguish order, updates, and the latest valid state?"——要的就是**最新
# 有效状态**。而我们的叙事默认**不带任何时间信息**（render_timestamps 关闭，正常路径
# 的 story 也不带 created_at），Answer 侧无从判断同一属性的多个取值里哪个是当前值。
#
# 这条规则把「记录时间」显式用作**新旧判据**——注意它与规则 9 的分工：规则 9 防止把
# 记录时间误当事件发生时间；本条只在问"当前值"时才允许用它做新旧的裁决。
_STORY_RANK_STATE_RULE = """

11. **若用户消息在问某个属性的当前 / 最新取值**，本条与规则 9 的限定分工如下：
   - 节点标注的「记录于 YYYY-MM-DD」**可以用作新旧判据**：同一属性出现多个不同取值时，
     **以记录时间最晚的那条为当前值**，其余作为历史值（可不写或明确标为"此前"）；
   - 但**不得**把记录时间当作事件发生时间叙述（规则 9 仍适用于事件经过）；
   - 若同一属性的多条记忆记录时间相同、无法判断先后，则并列陈述，不要臆断。
   若用户消息不是在问当前/最新取值，忽略本条。"""

STORY_RANK_PROMPT_STATE = ChatPromptTemplate.from_messages([
    ("system", _STORY_RANK_SYSTEM + _STORY_RANK_TS_RULE + _STORY_RANK_STATE_RULE),
    ("human", _STORY_RANK_HUMAN),
])

# 仅当开启「按场景分段」时追加（0930，ALM 的 `story_scene_paragraphs`；**默认关闭**）。
#
# 为什么需要：ALM 默认路径下，模型拿到的节点是「**全局层次序**」——它**与故事时间无关**
# （见 retriever._build_path → _order_by_level：先所有根、再逐层铺开，同层同根组内保持
# hop 序；**只按"根/层"排，不按分数也不按时间**）；且 `show_ts=False` 时
# **一个时间标注都没有**；而规则 2 又要求「只输出一段文字」。三者叠加 → 不同场景/不同时刻的
# 事实被强行塞进一个自然段，出现"上一句在会场、下一句在走廊"这类断裂（实测线上 story：
# 同一段里承载了 5+ 个互不相邻的场景，且时序前后颠倒）。
#
# 本规则**只放松"必须一段"**并给出定序依据，不改"只陈述事实、不下结论"的既有约束。
_STORY_RANK_SCENE_RULE = """

12. **分段与顺序（本条优先于第 2 条的「只输出一段文字」）**：
   - 按**场景 / 时段**把故事分成 **2~5 个自然段**：**同一场景**（同一地点、同一时段里连续发生的
     事）写在同一个自然段里；**不得把不同场景的节点硬塞进同一个自然段**。
   - 段内与段间的先后，依据**只允许**来自：① 节点上标明的 SEQUENCE / TEMPORAL 边；
     ② 「记录于」的时间。**两者都没有时保持中性并列**（"此外 / 另一方面 / 与此同时"这类），
     **不得臆造先后、不得编造因果**——不要用"于是 / 因此 / 结果"把没有因果边的两件事串起来。
   - 分段**不等于列举**：仍然写成叙事（成句、有主语与动作），不是要点清单。
   - 仍然只陈述节点里写明的事实：不解释、不替读者下结论、不给"变化"命名（同规则 2 / 6 / 7）。
   - 篇幅上限不变。"""

# 规则 12 必须插在**规则列表末尾、输出格式之前**，不能拼在整个 system 之后——
# 否则它落在「只输出 JSON。」之后，权重太低（0930 首测：两版都还是 1 段，规则未生效）。
_SCENE_ANCHOR = "\n\n输出格式（严格 JSON）："


def _with_scene_rule(sys_text: str) -> str:
    """把规则 12 插到规则列表末尾（输出格式之前）"""
    return sys_text.replace(_SCENE_ANCHOR, _STORY_RANK_SCENE_RULE + _SCENE_ANCHOR)

# ---- 溯源核对 Prompt（任务 C3）----

SOURCE_AUDIT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是记忆核查员。下面给你一段**原始对话**，以及从这段对话抽取出的**若干记忆节点**。
逐条核对，只输出 JSON。

核对三件事：
1. unsupported：哪些节点的内容在原文里**找不到支撑**（推断、过度概括、或凭空补出的信息）。
   注意：节点允许"消解指代、补全省略"，但**不得引入原文没有的事实**。
   找不到支撑的节点通常是"把推测写成了事实"，这是最重要的一类问题。
2. missing：原文里有、但**没有被抽出来**的关键事实——重点看"为什么做某个选择"：
   被放弃的选项、取舍依据（成本/工期/偏好/外部约束）有没有留下。
3. granularity：粒度不对的节点。**只报两类**：
   - 把**无法定位的模糊表述**单独抽成了节点——**仅限这一类词**：「过两天」「过几天」「过些天」
     「过阵子」「回头」「改天」「以后」「晚点」「有空」「再说」「将来」等**既落不到具体日期、
     也无法与库中时段对应**的说法，脱离上下文就会漂；
   - 把本应拆开的多件事合并成了一条（**这条与时间词无关，该报照报，不要漏**）。
   **下面"不要报"的限定只针对第一类（时间词独立成节点）**：除第一类词之外，任何时间表述
   单独成节点都不算粒度问题，一律不要报。包括但不限于：
   - 具体日期，与裸周 / 裸月 / 裸年 / 学期（「上周五」「下周一」「明天」「上周」「这周」
     「下周」「上个月」「下个月」「去年」「今年」「寒假」「暑假」「上学期」「上半年」）；
   - 已被系统纳入时段映射、或仅作锚点的相对/指示表述（「前几天」「前两天」「这两天」
     「这几天」「那天」「当天」「当时」「后来」「之后」「第二天」「次日」「最近」「近来」
     「这段时间」「那段时间」「近年来」「毕业」「入职」「凌晨」「早上」「晚上」）。
   这些要么能落到具体时段，要么是系统时序检索的**既定设计**，独立成节点是正确的。

判定纪律：
- 只依据给定的原文，不要用常识补全、不要假设原文之外的信息；
- **逐条节点都要过一遍**，不要跳读；**发现可疑就报**——这份报告是给人工/上层 agent
  复核用的线索，漏报的代价高于误报。但每条都必须给出判断依据（why），
  好让复核者能快速否掉；
- 不确定"算不算问题"时，宁可按更轻的类别报出来（例如先报 granularity），也不要直接放过。

输出格式（严格 JSON）：
{{
  "unsupported": [{{"id": "n1", "why": "..."}}],
  "missing": [{{"quote": "原文片段", "why": "..."}}],
  "granularity": [{{"id": "n1", "why": "..."}}]
}}
没有问题的项返回空数组。只输出 JSON。"""),
    ("human", """── 原始对话 ──
{batch_text}

── 抽取出的节点 ──
{nodes_text}"""),
])

# ---- 回复生成 Prompt ----

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个有共情能力的对话助手。根据以下信息生成自然回复。

{context}

要求：
1. 回复自然流畅，如朋友间聊天
2. 如果有相关记忆，自然地融入回复
3. 不要机械地列举信息
4. 保持简洁温暖"""),
    ("human", "{query}"),
])


# StoryRank 输出不可解析时，降级拼接保留的节点条数。
#
# 上限取 40 而非"全部"：`" ".join(所有节点)` 在最难的题上会产出 1.2 万~2.3 万字符的
# 原文流水账（0926 线上 7 次实测，采纳 217~443），而**节点边界、边、方向会在拼接时
# 全部抹平**——这三样正是 B2「因果链、路径与中间步骤恢复」要考的东西，抹平即归零。
# 40 与 0828 检索理论文的设计输出区间 5–37 同量级；按 combined_score 取前 40 条，
# 至少保住"与查询最相关"的少数节点及其顺序。
_STORY_FALLBACK_N = 40

# 超预算重试时追加在节点列表之前的指令。
#
# 为什么需要重试：0927 实测规则 6 的"不超过 2000 字符"只在 42/46 次被接受，另有 4 次
# 写出 2033 / 2520 / 4087 / **10487** 字符——篇幅约束对模型只是"倾向"而非"约束"。
# 重试前置一句强指令，比调小数字更有效（数字调小同样会被越过）。
# 用拼在 `path` 前面而不是改 prompt 模板：模板是模块级常量、被 4 个变体共享，
# 为一个偶发分支去改它会把所有调用的缓存与 diff 都搅动。
_STORY_RETRY_NOTICE = (
    "⚠️ 上一次输出超出了篇幅上限，已被判为不合格。这一次**必须**把 story 压到 "
    f"{_STORY_MAX_CHARS} 字符以内：只保留与「用户当前消息」最直接相关的那几条记忆，"
    "其余全部省略。\n\n"
)


def _truncate_at_sentence(text: str, limit: int) -> str:
    """把文本截到 limit 以内，且尽量落在句子边界上（不切断词、不留下半句）。

    只在"重试后仍然超预算"时使用：此时必须保证返回值不超过预算，否则篇幅约束形同虚设。
    优先切在最后一个句子结束符；实在够不到就退到最后一个空白，再不行才硬切。
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    best = max(head.rfind(ch) for ch in "。！？.!?")
    if best >= limit // 2:
        return head[: best + 1]
    space = head.rfind(" ")
    return head[:space] if space >= limit // 2 else head


# 判定"某条节点是否真的写进了正文"的字符二元组覆盖率阈值（见 count_story_used）。
_USED_COVERAGE_MIN = 0.5


def _bigrams(text: str) -> set:
    """归一化（去空白、转小写）后的字符二元组集合；单字符文本退化为该字符"""
    s = "".join((text or "").split()).lower()
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def count_story_used(story: str, nodes: List[dict]) -> int:
    """统计正文**实际覆盖**的节点数——后端计算，替代模型自报的 `used`。

    背景（0927 审查 C4）：模型自报的 `used` 与正文不自洽（自报 40/45/60，而正文被
    篇幅预算截断后实际写进去的远少于该数；降级分支同样不自洽）。改为后端按
    「节点内容是否真的出现在正文里」计数，使 `used` 与正文一致。

    判据是**字符二元组覆盖率**（节点与正文的二元组交集 / 节点自身二元组数 ≥
    `_USED_COVERAGE_MIN`）。为什么不用子串包含：叙事是改写，几乎不会有逐字子串；
    二元组对改写更稳健，而"整条被省略"仍会得到接近 0 的分数。

    局限：这是**近似**度量。对高度概括/改写的句子可能**低估**；对与正文其他内容共享
    大段措辞的节点可能**高估**（例如"事实1"会命中正文里的"事实10"）。它只用于观测
    （"选择层是否真在做减法"），不参与任何控制流；降级路径不用它（那里可精确记账）。
    """
    story_bg = _bigrams(story)
    if not story_bg:
        return 0
    used = 0
    for n in nodes:
        nb = _bigrams(n.get("content", ""))
        if nb and len(nb & story_bg) / len(nb) >= _USED_COVERAGE_MIN:
            used += 1
    return used


class InferenceEngine:
    """基于 LangChain ChatOpenAI 的推理引擎"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # 目的推断**最终失败**次数（观测用，不参与控制流）。0927 审查 C2：该失败此前
        # 只能靠 WARNING 文本反推，无法做指标；失败时静默回退默认目的会让
        # condense/state/时序路由同时失活（每轮约 3.6 次）。
        self.purpose_failures = 0

    def infer_status(self, query: str) -> dict:
        """推断用户状态和情绪"""
        import json
        chain = STATUS_INFERENCE_PROMPT | with_role(self.llm, "status")
        response = chain.invoke({"query": query})
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # 只记长度，不记原文：原始响应由 query 派生，落进日志即等于落评测正文
            # （合规要求日志不含记忆内容）。用模块 logger 而非 root，才能被
            # server._configure_logging 对 dba_pipeline 的级别闸统一管控。
            logger.warning("状态推断 JSON 解析失败（响应长度 %d），按未知处理", len(response.content or ""))
            return {"status": "unknown", "emotion": "中性"}

    def infer_purpose(self, query: str) -> dict:
        """推断用户隐含目的

        0927 审查 C2：该 prompt 与模型的组合有 ~8% 的**系统性**退化（跨 16 轮共 57 次、
        每轮约 3.6 次），失败形态高度一致——响应只有 2~3 个字符（对应 `out=2`），
        不是随机抖动。失败时静默回退默认目的会让 condense/state/时序路由同时失活，
        故这里重试一次；并对"响应过短"直接判失败（此前它可能恰好被解析成残缺 JSON）。
        """
        import json
        chain = PURPOSE_INFERENCE_PROMPT | with_role(self.llm, "purpose")
        for attempt in (1, 2):
            content = chain.invoke({"query": query}).content or ""
            parsed = None
            # 过短响应（如 3 字符）一律判失败：正常输出是一个带 purposes/condense/state
            # 的完整 JSON，长度远大于此阈值；把它当成功会让残缺 JSON 混过解析。
            if len(content.strip()) > 5:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
            if isinstance(parsed, dict) and parsed:
                return parsed
            # 只记长度，不记原文：原始响应由 query 派生，落进日志即等于落评测正文
            # （合规要求日志不含记忆内容）。
            logger.warning(
                "目的推断 JSON 解析失败（第 %d 次，响应长度 %d）%s",
                attempt, len(content), "，重试一次" if attempt == 1 else "，按默认目的处理",
            )
        self.purpose_failures += 1
        return {"status": "unknown", "purposes": ["理解"], "condense": False, "state": False}

    def story_rank(self, query: str, path: dict, render_timestamps: bool = False,
                   condense: bool = False, state: bool = False,
                   scene: bool = False) -> dict:
        """把一条检索因果链路理解成故事片段

        Args:
            query: 用户当前消息
            path: {"nodes": [{"id", "content", "node_type", "timestamp", "score"}],
                   "edges": [{"from", "to", "rel_type", "is_reverse"}]}
                `score` 是 combined_score，**只用于 JSON 解析失败时按相关度降级拼接**
                （节点入参顺序是 hop 顺序而非分数序，故兜底必须自己重排）。
            render_timestamps: 是否把节点的记录时间渲染进 prompt，并切换带规则 9 的
                prompt 变体（规则 9 专门声明「记录于」是记录时间、非事件时间）。
                MCP 侧硬编码开启；ALM 侧默认关闭（0920 A/B 补测显示净收益为 0，
                见 Plan §7.10.5 F），可用 ALM_RENDER_TIMESTAMPS=1 打开。
            condense: 压缩类问题（见 STORY_RANK_PROMPT_CONDENSE）。优先于
                render_timestamps 的变体选择。

        Returns:
            {"story": str, "used": int, "degraded": bool}
                `used` 是**后端统计的**"正文实际覆盖的节点数"（见 `count_story_used`），
                与正文严格一致，是**选择层是否真在做减法**的读数，只用于观测，不参与
                任何控制流。0927 前这里是模型自报的整数——它与正文不自洽（自报 40/45/60
                而正文被篇幅预算截断后远少于该数），故改为后端按覆盖率计算（审查 C4）。
                更早前这里是 `adopted_ids`（id 列表）——那份列表对外毫无用处（ALM 只把
                `story` 作为 content 交给下游），却把**输出体积与节点数绑死**，直接触发
                了 8192 截断；且上限 30 反过来诱导模型"凑满 30 条"，实测全体最长返回的
                `采纳` 都恰好是 30。
                `degraded=True` 表示 JSON 解析失败、已按相关度降级拼接——调用方必须把它
                透传到检索日志，否则这类返回在聚合指标上与正常叙事无法区分。
        """
        import json

        nodes = path.get("nodes", [])
        if not nodes:
            return {"story": "", "used": 0, "degraded": False}

        # 格式化链路。**渲染抽成函数**是为了支持"缩喂重试"（见 _STORY_FIT_*）：
        # 超预算时可以只渲染前 N 个节点，并自动剔除两端已不在集合内的边（不写悬空边）。
        def _render(sel):
            out = ["节点："]
            # state（问当前取值）同样需要渲染记录时间——没有时间就无从裁决新旧
            show_ts_ = render_timestamps or state
            for n in sel:
                nt = n.get("node_type", "")
                tag = ""
                if show_ts_:
                    ts_text = format_ts_ms(n.get("timestamp"))
                    if ts_text:
                        tag = f"(记录于 {ts_text}) "
                out.append(f'  [{n["id"]}]（{nt}）{tag}{n.get("content", "")}')
            ids = {n["id"] for n in sel}
            out.append("关系：")
            for e in path.get("edges", []):
                if e.get("from") not in ids or e.get("to") not in ids:
                    continue
                rel = e.get("rel_type", "")
                if e.get("is_reverse"):
                    out.append(f'  [{e["to"]}] --{rel}--> [{e["from"]}]')
                else:
                    out.append(f'  [{e["from"]}] --{rel}--> [{e["to"]}]')
            return "\n".join(out)

        path_text = _render(nodes)

        if condense:
            sys_text = _STORY_RANK_SYSTEM + _STORY_RANK_CONDENSE_RULE
        elif state:
            sys_text = _STORY_RANK_SYSTEM + _STORY_RANK_TS_RULE + _STORY_RANK_STATE_RULE
        elif render_timestamps:
            sys_text = _STORY_RANK_SYSTEM + _STORY_RANK_TS_RULE
        else:
            sys_text = _STORY_RANK_SYSTEM
        # 0930「按场景分段」：把规则 12 插到**规则列表末尾**（默认关闭）
        if scene:
            sys_text = _with_scene_rule(sys_text)
        chain = ChatPromptTemplate.from_messages(
            [("system", sys_text), ("human", _STORY_RANK_HUMAN)]
        ) | with_role(self.llm, "story")

        def _invoke_once(notice: str, sub=None):
            """调用一次并解析，返回 (story, degraded, used_hint)

            `sub` 给定时只渲染这批节点（缩喂重试用）；不给则是全量 `path_text`。
            `used_hint` 仅在**降级路径**给出（那时正文是我们自己拼的，可精确算出"完整
            写进正文的节点数"）；正常路径返回 None，由 `count_story_used` 事后估算。
            """
            response = chain.invoke({
                "query": query,
                "path": notice + (_render(sub) if sub is not None else path_text),
            })

            # 输出触顶必须留痕：0926 线上 7 次 `StoryRank JSON 解析失败` 全部伴随
            # `alm.tokens: out=8192`——输出撞端点上限被截断，JSON 在数组中途断掉。
            # 此前这个信号只能靠 token 数反推，故显式记录 finish_reason。
            # （0927 起 JSON 里只剩 story/used，篇幅由规则 6 的 _STORY_MAX_CHARS 约束，
            #   正常情况下不该再触顶；这个告警保留作为"预算未生效"的兜底观测。）
            try:
                finish = (response.response_metadata or {}).get("finish_reason")
            except AttributeError:
                finish = None
            if finish == "length":
                logger.warning(
                    "StoryRank 输出被 max_tokens 截断（finish_reason=length，喂入 %d 个节点）"
                    "——JSON 大概率不完整，将按相关度降级拼接", len(nodes),
                )

            try:
                cleaned = response.content.strip()
                if cleaned.startswith("```"):
                    lines_ = cleaned.split("\n")
                    lines_ = [l for l in lines_ if not l.startswith("```")]
                    cleaned = "\n".join(lines_)
                data = json.loads(cleaned)
                # 只取 story。模型自报的 `used` **不再采用**——它与正文不自洽，改由
                # `count_story_used` 在正文定稿后按实际覆盖计算（见 0927 审查 C4）。
                return data.get("story", ""), False, None
            except (json.JSONDecodeError, AttributeError):
                # 降级**按相关度取前 N**，不再把全部候选平铺（理由见 _STORY_FALLBACK_N）。
                # 注意入参顺序是 hop 顺序（根→叶）而非分数序，故这里必须重排。
                ranked = sorted(nodes, key=lambda n: float(n.get("score") or 0.0), reverse=True)
                picked = ranked[:_STORY_FALLBACK_N]
                logger.warning(
                    "StoryRank JSON 解析失败，降级为按相关度拼接前 %d 条节点（候选 %d 条）",
                    len(picked), len(nodes),
                )
                # 拼接同样受篇幅预算约束：否则降级产物又会退化成"长而无结构"的流水账。
                story = " ".join(n.get("content", "") for n in picked)[:_STORY_MAX_CHARS]
                # 降级路径的正文由我们自己拼，故"写进正文的节点数"可**精确**统计：
                # 只算完整放进预算的那些节点（被预算截断的那条不计）。
                acc = exact = 0
                for n in picked:
                    need = len(n.get("content", "")) + (1 if exact else 0)
                    if acc + need > _STORY_MAX_CHARS:
                        break
                    acc += need
                    exact += 1
                return story, True, exact

        story, degraded, used_hint = _invoke_once("")

        # 篇幅预算是**硬约束**：0927 实测它只在 42/46 次被接受（另有 4 次写出
        # 2033/2520/4087/10487 字符）。超限就附加强制压缩指令重试一次——数字写小
        # 同样会被越过，所以这里靠"重试 + 强指令"而不是靠调参数。
        if not degraded and len(story) > _STORY_MAX_CHARS:
            logger.warning(
                "叙事超出篇幅预算: 正文 %d 字符 > 上限 %d（喂入 %d 节点）——附加强制压缩指令重试",
                len(story), _STORY_MAX_CHARS, len(nodes),
            )
            story, degraded, used_hint = _invoke_once(_STORY_RETRY_NOTICE)

        # 第二段：仍超限 → **缩喂重试**（动态调参，而不是先截正文）。
        # 线索永不因篇幅而丢：先丢"优先级最低的尾部节点"，只有缩到下限还超才截正文。
        fit_nodes = nodes
        round_ = 0
        while (not degraded and len(story) > _STORY_MAX_CHARS
               and round_ < _STORY_FIT_MAX_ROUNDS and len(fit_nodes) > _STORY_FIT_MIN_NODES):
            round_ += 1
            keep = max(_STORY_FIT_MIN_NODES, int(len(fit_nodes) * _STORY_FIT_SHRINK))
            if keep >= len(fit_nodes):
                break
            logger.warning(
                "叙事仍超预算（%d 字符），缩喂重试第 %d 轮: %d → %d 节点（不截正文）",
                len(story), round_, len(fit_nodes), keep,
            )
            fit_nodes = fit_nodes[:keep]
            story, degraded, used_hint = _invoke_once(_STORY_RETRY_NOTICE, fit_nodes)

        if not degraded and len(story) > _STORY_MAX_CHARS:
            # 缩喂到底仍超限 → 硬截断到句边界（最后手段）。这一步不能省：篇幅约束一旦
            # 可以被越过就不再是约束。
            before = len(story)
            story = _truncate_at_sentence(story, _STORY_MAX_CHARS)
            logger.warning(
                "叙事缩喂至 %d 节点后仍超出预算，已硬截断 %d → %d 字符",
                len(fit_nodes), before, len(story),
            )

        # `used` 在正文**定稿之后**确定，保证与正文一致（0927 审查 C4）：降级路径用
        # 精确记账（used_hint），正常路径按正文覆盖率估算。
        used = used_hint if used_hint is not None else count_story_used(story, nodes)
        return {"story": story, "used": used, "degraded": degraded}

    def audit_source(self, batch_text: str, nodes_text: str) -> dict:
        """溯源核对（任务 C3）：把"该批原文 + 该批产出的节点"交给 LLM，只报判断结果。

        Returns:
            {"unsupported": [{"id","why"}], "missing": [{"quote","why"}],
             "granularity": [{"id","why"}]}；解析失败时返回 {"error": ...}
        """
        import json
        import logging

        if not batch_text or not nodes_text:
            return {"unsupported": [], "missing": [], "granularity": []}

        chain = SOURCE_AUDIT_PROMPT | with_role(self.llm, "audit")
        response = chain.invoke({"batch_text": batch_text, "nodes_text": nodes_text})
        raw = (response.content or "").strip()
        # 容错：模型可能套 ```json 代码块
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            logging.warning(f"溯源核对 JSON 解析失败，原始响应: {raw[:200]}")
            return {"error": "JSON 解析失败", "raw": raw[:500]}

        out = {}
        for key in ("unsupported", "missing", "granularity"):
            items = data.get(key)
            out[key] = items if isinstance(items, list) else []
        return out

    def generate_response(
        self,
        query: str,
        context_items: List[str],
    ) -> str:
        """注入检索到的记忆上下文，生成回复"""
        if context_items:
            context_text = "【关于用户的已知信息】\n" + "\n".join(
                f"- {item}" for item in context_items
            )
        else:
            context_text = "【关于用户的已知信息】\n暂无相关记忆。"

        chain = RESPONSE_PROMPT | with_role(self.llm, "answer")
        response = chain.invoke({
            "query": query,
            "context": context_text,
        })
        return response.content
