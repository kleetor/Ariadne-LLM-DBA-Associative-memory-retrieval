# SPDX-License-Identifier: AGPL-3.0-only

"""
LLM DBA：记忆图谱的自主维护者

每次对话后异步触发，LLM 收到三层上下文（对话、记忆、结构），
自主决定对图谱执行 CRUD 操作（新增/更新/修正/废弃节点，新增/删除边）。
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dba_pipeline.graph.memory_graph import MemoryGraph
from dba_pipeline.core.jump_axis import NodeType
from dba_pipeline.embedding.store import VectorStore
from dba_pipeline.extraction.graph_builder import GraphBuilder
from dba_pipeline.llm.roles import with_role
from dba_pipeline.source_store import new_batch_id, save_batch

logger = logging.getLogger(__name__)


# ---- Step 1：节点抽取 Prompt（2026-08-22 起，节点抽取与边连接拆成两步）----

# 输出形状（单一来源，便于按开关替换；注意 `{{` 是交给 ChatPromptTemplate 的转义）
#
# ⚠️ **不要给 create 写 `action`**（P1-2 输出瘦身，见 版本变更记录 §4.26）：
# 实测 99.8% 的节点 op / 100% 的边 op 都是 create，把它做成默认值后
# 每次输出都少写十几个 token。解析侧已加默认（`graph_builder._apply_*_op`）。
_NODE_OUTPUT_EXAMPLE = (
    '严格输出 JSON：{{"node_ops":[{{"content":"...","node_type":"ACTION"}},'
    '{{"action":"update","target_id":"n3","content":"...","reason":"..."}},'
    '{{"action":"fix_type","target_id":"n7","node_type":"STATUS","reason":"..."}},'
    '{{"action":"deprecate","target_id":"n3","reason":"..."}}]}}'
    '\n`action` 省略即视为 `create`；**只有非 create 操作才写 `action`**。'
)

# 开启 evidence 时的输出形状：每条 create/update 额外带 evidence（原文支撑片段）
_NODE_OUTPUT_EXAMPLE_EVIDENCE = (
    '严格输出 JSON：{{"node_ops":[{{"content":"...","node_type":"ACTION","evidence":"..."}},'
    '{{"action":"update","target_id":"n3","content":"...","evidence":"...","reason":"..."}},'
    '{{"action":"fix_type","target_id":"n7","node_type":"STATUS","reason":"..."}},'
    '{{"action":"deprecate","target_id":"n3","reason":"..."}}]}}'
    '\n`action` 省略即视为 `create`；**只有非 create 操作才写 `action`**。'
)

# ---- 输出语言指令：注入到 user 消息末尾，而不是只写在 system prompt 里 ----
#
# 为什么必须这么做（2026-09-26 本地烟雾实测，英文输入 + 生产模型 deepseek-v4-flash）：
# **只把语言约束写在中文 system prompt 顶部不生效**——首批（图里还没有任何「已有节点」
# 可供上下文污染）依然输出了中文节点，如「用户是杭州一家互联网公司的后端开发工程师」。
# 判断原因是两点叠加：① LLM 对**紧邻数据**的指令遵循度高于 system 里的规则；
# ② 中文的指令正文与中文举例在持续"拉"它说中文，一段英文声明压不住。
# 故改由代码判定源文语言，把一行指令拼到 user 消息的**末尾**（近因位置），
# system prompt 里那段约束保留作为兜底。
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_LANGUAGE_DIRECTIVE = """
── 输出语言（最高优先级，覆盖以上所有规则与示例）──
每条 `content` / `reason` 必须与**该条事实所在的原文段**同一种语言：原文是英文就写英文、
原文是中文就写中文；同批含多种语言时**按各自原文段分别处理**，不要整批统一成一种语言
（这与「抽取原则 7」一致）。本批以 {lang} 为主，仅供参考；判断依据是源对话本身，
不是本提示词的语言。**禁止翻译**成任何其它语言。
"""


def source_language(text: str) -> str:
    """粗判源文语言，返回注入指令里用的语言名（"中文" / "English"）。

    用字符构成而非语言识别模型：这个判断只决定一行指令的措辞，判错的代价是退回改造前
    的"跟随指令语言"行为，而引入模型会让每次 Add 多一次调用，不划算。

    判据是**汉字在「汉字+拉丁字母」里的占比 ≥ 10%**，而不是简单比大小。理由是两类文本
    的构成本来就不对称：
      · 英文文本几乎不含汉字——占比会接近 0，判英文不会错；
      · 中文文本经常夹英文专名（"我用 Go 和 PostgreSQL" 汉字 3 / 拉丁 12 = 20%），
        按"比大小"会把它误判成英文，按占比则正确判中文。
    阈值 10% 的边界：整段英文里偶尔出现一个中文词（2 汉字 / 数百字母 ≈ 0.4%）仍判英文，
    不会被个别字带偏。
    """
    cjk = len(_CJK_RE.findall(text))
    if cjk == 0:
        return "English"
    return "中文" if cjk / (cjk + len(_LATIN_RE.findall(text))) >= 0.10 else "English"


NODE_EXTRACTION_PROMPT = """你是记忆图谱的「节点抽取器」：从对话中抽取独立事实——既包括关于「用户」的事实，也包括「助手」自身经历/行为的事实，只输出节点维护操作，不负责连边。

[OUTPUT LANGUAGE — HARD CONSTRAINT]
Write every `content` and `reason` string in the SAME LANGUAGE as the dialogue it came from.
English dialogue → English output. Chinese dialogue → Chinese output. Never translate,
never paraphrase into another language. This rule outranks the language of the instructions
and of the examples below (which are written in Chinese merely for brevity).

## 节点类型（6）
STATUS=客观处境 | REASON=导致状态/行为的原因 | ACTION=主动行为 | THING=物品/地点 | PERSON=社交关系人 | EMOTION=主观情绪
边界：客观处境→STATUS/主观感受→EMOTION；用户做的事→ACTION/导致原因→REASON；物品本身→THING/互动→ACTION

## 操作
create(新事实) | update(事实变化) | fix_type(修正类型) | deprecate(标记过时,不物理删除)

## 抽取原则（重要）
1. 只抽事实，来源有二：用户的事实、助手自身经历/行为（助手为响应用户请求实际做过的事，如「助手帮用户整理了一份项目清单」「助手用Python写了个排序示例」）；忽略无信息量的寒暄/客套/共情
2. 一个事实一个节点：一句话含多个事实时必须拆成多个节点，禁止合并
   （如「加班又总吃外卖」→ 加班节点 + 外卖节点；「小王一起加班还帮分担任务」→ 两人同加班 + 小王帮分担 两个节点）
3. content 为消解指代、补全省略后的完整陈述：单人对话中，对话方转成「用户」、助手自身转成「助手」；多人对话中，各人按其姓名/昵称归属（如「阿哲最近在赶设计稿」），不得把其他人一律并入「用户」
4. 保留时间维度：区分长期习惯（「用户每天晨跑半小时」）与单次事件（「用户今早跑了一次步」），不得混淆
5. 与已有节点语义重复时用 update 更新而非新建；与已有节点矛盾时 deprecate 旧节点 + create 新节点
6. 每条独立可读
7. **content 必须使用源对话的语言**：原文是英文就写英文、是中文就写中文；同批含多种语言时，
   以该条事实所在原文段的语言为准。**禁止翻译**（英文原文不得译成中文，反之亦然）——
   译文一旦落库，检索侧会与同语言的查询向量跨语言失配，而且后续批次的「已有相关节点」
   也全是译文，误差会逐批累积。

> 上面各条举例中的中文**只是示意写法，不代表输出语言**；输出语言一律以原则 7 为准。

## 输出
""" + _NODE_OUTPUT_EXAMPLE + """
无需抽取时返回空 node_ops。只输出 JSON。"""

NODE_EXTRACTION_USER_PROMPT = """── 对话上下文 ──

{conversation}

── 已有相关节点 ──

{current_nodes}

{language_directive}

── 请输出节点维护操作 ──"""


# ---- Step 2：边连接 Prompt（与节点抽取拆分，2026-08-22）----

EDGE_LINKING_PROMPT = """你是记忆图谱的「边连接器」：节点已经抽取完成，你只负责为节点建立语义关系边。

## 边类型（8）
CAUSAL=因→果 | SCENARIO=同场景 | SEQUENCE=先→后 | PREFERENCE=偏好 | SOCIAL=社交 | ATTRIBUTE=属性 | TEMPORAL=事件→时间 | TAXONOMIC=子→父
方向：CAUSAL/SEQUENCE/PREFERENCE/TEMPORAL/TAXONOMIC 单向；SCENARIO/SOCIAL/ATTRIBUTE 双向
边界：A不发生B还会发生?不会=CAUSAL,可能=SEQUENCE；B是环境=SCENARIO/自身属性=ATTRIBUTE；喜欢=PREFERENCE/需要=CAUSAL；上位类别=TAXONOMIC（个人记忆里通常是「X 是一种 Y」这类常识归类，很少出现，没有就不连）

## 操作
create(连边) | delete(删错误/过时的边)

## 连边原则（激进版：宁可多连，不可漏连）
1. **尽量全连**：同一场景内（工作/健康/社交/情绪/学生时代）所有存在关系的事实之间都要连边——场景内高相关节点是检索枢纽，应连接尽可能多的节点（如「后端开发」连接项目加班、同事、主管、工作年限、换工作意向等）
2. **跨场景大胆桥接**：工作→健康→情绪→社交 之间，只要找得到因果/场景/时序/偏好/人物关系之一就连——跨场景联想是检索的价值所在
3. 连接「本轮新节点」与「已有相关节点」：新事实必须尽量接入已有图谱结构，能连则连
4. 边类型要准确：喜欢/偏好→PREFERENCE、自身属性→ATTRIBUTE、同场景并存→SCENARIO、先后来→SEQUENCE、因果→CAUSAL、人物关系→SOCIAL、上位类别→TAXONOMIC
5. 保留时间维度：长期习惯与单次事件按语义连（CAUSAL/SEQUENCE）
6. 边审计：删除已过时/被新事实覆盖的边；本轮未提到但可能仍成立的保留
7. `reason` 等自然语言字段使用**源对话的语言**（边本体只有类型与节点 id，不受语言影响）

> 下面的 few-shot 示例是中文写的，**只用于示意连边策略，不代表输出语言**。

## 输出
严格输出 JSON：{{"edge_ops":[{{"from":"n1","to":"n2","rel_type":"CAUSAL"}},{{"action":"delete","from":"n3","to":"n5"}}]}}
`action` 省略即视为 `create`；**只有 delete 才写 `action`**。
无需连边时返回空 edge_ops。只输出 JSON。"""

EDGE_LINKING_FEWSHOT_EXAMPLE = """参考示例（激进连边模式）：

示例 1 —— 同一场景内全连 + 跨场景也连（工作场景）：

对话：
user(09:00): 今天又要加班，这个项目真的排得太满了，天天搞到十一二点。
user(09:05): 我在这家互联网公司做后端开发，项目一上线就得赶进度。脖子又酸了，加班太多颈椎扛不住。
user(09:10): 同事小王跟我一样惨，天天一起加班，不过他有时候会帮我分担点任务。下午得靠咖啡续命，最爱拿铁。

本轮新抽取的节点：
[n1 STATUS] 用户在一家互联网公司做后端开发
[n2 ACTION] 用户项目上线时经常加班到十一二点
[n3 STATUS] 用户加班导致颈椎经常酸痛
[n5 ACTION] 用户喜欢喝咖啡提神
[n6 THING] 拿铁是用户最常点的咖啡
[n8 PERSON] 同事小王经常和用户一起加班

正确输出（场景内全连 + 跨场景桥接）：
{{"edge_ops":[{{"from":"n1","to":"n2","rel_type":"CAUSAL"}},{{"from":"n1","to":"n8","rel_type":"SOCIAL"}},{{"from":"n1","to":"n3","rel_type":"CAUSAL"}},{{"from":"n2","to":"n3","rel_type":"CAUSAL"}},{{"from":"n2","to":"n5","rel_type":"CAUSAL"}},{{"from":"n2","to":"n8","rel_type":"SOCIAL"}},{{"from":"n5","to":"n6","rel_type":"PREFERENCE"}},{{"from":"n3","to":"n5","rel_type":"SCENARIO"}}]}}

示例 2 —— 跨场景大胆桥接：

对话：
user(20:00): 复查结果出来了，指标降了，还挺欣慰的。
user(20:07): 就是戒不掉烧烤，那家店的烤羊排真是我的最爱。

本轮新抽取的节点：
[n20 ACTION] 用户每天早上绕小区晨跑半小时
[n27 EMOTION] 用户看到复查指标下降很欣慰
[n40 THING] 烧烤店的烤羊排是用户的最爱

已有相关节点：
[n2 ACTION] 用户项目上线时经常加班到十一二点
[e0 STATUS] 用户今天心情很低落

正确输出（晨跑接入工作/情绪/饮食多个场景）：
{{"edge_ops":[{{"from":"n2","to":"e0","rel_type":"CAUSAL"}},{{"from":"n20","to":"n27","rel_type":"CAUSAL"}},{{"from":"n2","to":"n20","rel_type":"SEQUENCE"}},{{"from":"n20","to":"e0","rel_type":"CAUSAL"}},{{"from":"n20","to":"n40","rel_type":"SCENARIO"}}]}}"""

EDGE_LINKING_USER_PROMPT = """── 对话上下文 ──

{conversation}

── 本轮新抽取的节点 ──

{new_nodes}

── 相关已有节点 ──

{current_nodes}

已有边：
{current_edges}

── 结构上下文（相关节点一跳邻居）──

{neighbor_info}

图谱概况：总节点 {total_nodes} 个，总边 {total_edges} 条

{language_directive}

── 请输出边维护操作 ──"""


# ---- 时序叙事（TEMPORAL）增强：仅在“时序叙事守卫”命中时叠加，避免常规语料过度拆分 ----

NODE_TEMPORAL_EXTRA = """
8. 时间节点单独抽取：对话中出现的明确时间/时段（如「上周一晚上」「周二」「周三凌晨」「这周」）要抽成独立的 THING 节点，content 就是该时间词本身；不要把时间写进事件 content。一个事件若发生在某个时间，需同时抽出「事件节点」和「时间节点」两个节点。"""

NODE_CHOICE_EXTRA = """
## 决策与取舍（补充规则）
REASON 除「导致状态/行为的原因」外，**还包括「做出某个选择 / 取舍的依据」**。
出现选择、取舍、权衡、最终决定这类表述时，除结果本身外还必须抽出：
1. 被放弃的选项：对话里明确说过还考虑过什么、为什么没有选它；
2. 选择依据：是什么条件决定了最终的取舍（成本、工期、偏好、外部约束等）。
这两项各写成独立节点，node_type 用 REASON。内容允许带最小限度的上下文指代
（如「当时考虑到成本，放弃了 B 方案」），不必强求单条脱离上下文也完全可读。
边界：只能抽对话里**明确说过**的备选与依据，不得推断、补全或想象未说出口的动机。
"""

EDGE_TEMPORAL_EXTRA = """
8. 务必连 TEMPORAL 边（事件→时间，单向）：本轮新抽出的事件节点，若其发生时间也是一个已抽取的时间节点，用 TEMPORAL 单向连边（from=事件, to=时间）。事件之间的先后关系用 SEQUENCE（先→后），不要与「时间定位」（TEMPORAL）混为一谈：SEQUENCE 是事件 A 在事件 B 之前；TEMPORAL 是事件挂在某个时间锚点上。"""

# ---- evidence：给每条节点留一个可核查的原文依据（opt-in，见 enable_evidence）----

NODE_EVIDENCE_EXTRA = """
## evidence（补充字段）
每条 create / update 的 node_op 额外给出 "evidence"：**对话原文中支撑这条节点的最短连续片段**。
- 必须是原文原话：可以截取，但不得改写、概括或拼接不同位置的句子；
- 只取支撑该节点所必需的那一句或半句，不要整段照抄；
- 若这条节点是对原文的归纳或推断、原文里找不到直接支撑，evidence 留空字符串 ""。
"""

# 去标点/空白后做子串判定，避免 LLM 微调标点导致误判
_EVIDENCE_STRIP_RE = re.compile(r"[\s，。、：:；;！!？?…—\-（）()「」『』\"'’‘“”·]+")
EVIDENCE_MIN_CHARS = 4
EVIDENCE_MAX_CHARS = 300


def normalize_evidence_text(text: str) -> str:
    """去掉标点与空白，用于 evidence「是否为原文子串」的判定"""
    return _EVIDENCE_STRIP_RE.sub("", text or "")


def sanitize_evidence(node_ops, conversation):
    """校验 node_ops 里的 evidence：不是该批原文子串的一律丢弃。

    程序化兜底——LLM 可能把 evidence 写成概括或改写，那种"证据"无法核查，留着反而
    误导巡检。返回 (清理后的 node_ops, 保留数, 丢弃数)。
    """
    haystack = normalize_evidence_text(conversation)
    kept = dropped = 0
    cleaned = []
    for op in node_ops:
        if not isinstance(op, dict) or "evidence" not in op:
            cleaned.append(op)
            continue
        op = dict(op)
        ev_norm = normalize_evidence_text(str(op.get("evidence") or ""))
        if (len(ev_norm) < EVIDENCE_MIN_CHARS
                or len(ev_norm) > EVIDENCE_MAX_CHARS
                or ev_norm not in haystack):
            op.pop("evidence", None)
            dropped += 1
        else:
            op["evidence"] = str(op["evidence"]).strip()[:EVIDENCE_MAX_CHARS]
            kept += 1
        cleaned.append(op)
    return cleaned, kept, dropped

EDGE_TEMPORAL_EXAMPLE = """

示例 3 —— TEMPORAL 事件→时间 + SEQUENCE 事件先后：

对话：
user(周一 20:30): 上周一晚上临时又甩过来一个紧急需求。
user(周二 13:20): 这周二白天我一直在赶代码。
user(周三 01:20): 周三凌晨上线，结果出了个 P0 故障。

本轮新抽取的节点（事件节点与时间节点分开）：
[e0 ACTION] 用户接到紧急需求
[e1 ACTION] 用户加班赶代码
[e2 STATUS] 项目上线出P0故障
[t1 THING] 上周一晚上
[t2 THING] 上周二白天
[t3 THING] 上周三凌晨

正确输出（事件→时间用 TEMPORAL；事件之间先后用 SEQUENCE；因果用 CAUSAL）：
{{"edge_ops":[{{"from":"e0","to":"t1","rel_type":"TEMPORAL"}},{{"from":"e1","to":"t2","rel_type":"TEMPORAL"}},{{"from":"e2","to":"t3","rel_type":"TEMPORAL"}},{{"from":"e0","to":"e1","rel_type":"SEQUENCE"}},{{"from":"e1","to":"e2","rel_type":"SEQUENCE"}},{{"from":"e0","to":"e2","rel_type":"CAUSAL"}}]}}"""


TIME_PERIOD_RE = re.compile(
    r"昨天|明天|前天|上周|这周|本周|下周|周[一二三四五六日天]|"
    r"\d{1,2}月(\d{1,2}[日号])?|\d{1,2}号|\d{1,2}日|"
    r"去年|今年|明年|前年|寒假|暑假|开学|年底|年初|年初|"
    r"春天|夏天|秋天|冬天|上半年|下半年"
)


def has_temporal_signal(text: str) -> bool:
    """时序叙事守卫：仅当出现 >=2 个**不同的日期/时期词**（昨天/上周/周三/这周/去年等）时判定为时序叙事。

    刻意排除「凌晨/中午/晚上/每天/近来/有时」这类日常时间副词，避免常规生活语料过度拆分；
    也仅凭「高中/大学」等 era 词**不**触发——它们常出现在非时序的设定陈述里（如「大学时学的钢琴」），
    单靠 era 会让大量常规语料误触发（0822 子集即如此）。真正的时序叙事靠「多个具体日期锚点」识别。
    """
    if not text:
        return False
    periods = set(m.group(0) for m in TIME_PERIOD_RE.finditer(text))
    return len(periods) >= 2


# ---- 维护判断（triage）Prompt ----

DBA_TRIAGE_PROMPT = """判断对话是否包含值得长期记忆的新事实或助手自身经历/行为，只回答 NEEDED 或 SKIP。

SKIP：纯寒暄/打招呼、共情安慰、追问细节无新信息、仅延续话题。
NEEDED：用户的新事实、事实变化、与已有记忆矛盾、话题切换；或助手做了值得记住的具体产出/行为（如生成文件、写代码、整理清单、给出可执行建议），这类「助手做过的事」同样需要记录。

对话：
{conversation}

回答（NEEDED 或 SKIP）："""

# triage 的判定窗口，以及"窗口结论可信"的长度上限。
#
# 背景（2026-09-24 线上实测，见 MemoryDBA._should_maintain 的说明）：ALM 侧单次 Add
# 的对话长度中位 2522、最长 11196 字符，而判定只取尾部 _TRIAGE_WINDOW 字符，覆盖率
# 低到 5%~20%，"事实在中段、头尾寒暄"的对话会被整批判 SKIP，进而落成孤立原文节点、
# 零条边。故超过 _TRIAGE_TRUST_MAX_CHARS 的对话不再采信窗口结论，直接放行。
#
# ⚠️ 2026-09-25 重标：上限由 1500 下调为 **900**。依据是线上日志里 9 次 triage 判 SKIP
# 的对话长度分布 —— 45 / 80 / 310 / 541 / 930 / 1019 / 1047 / 1381 / 1412，
# 其中 5 次（930~1412）都落在 1500 以下，于是 triage 照跑、照 SKIP，长度闸形同不存在。
# 1500 是按离线探测集定的，而那批对话中位 2522 —— **探针集里没有 1000~1500 这一段**，
# 线上漏判却正好全挤在这里（取样区间没覆盖目标分布，不是阈值方向错）。
#
# 900 与 ALM 侧新增的分组预算 `alm.config.add_max_chars=1500` 配合后语义更清楚：
# 单组长度中位 583、p90 1328，取 900 意味着"窗口覆盖 ≥56% 才采信 triage 结论"，
# 覆盖率不足的组一律放行——triage 只在它确实看得清的时候说话。
#
# 这两个值是**设计参数**（由实测标定，非运行期旋钮）：声明版本与行为一一对应，
# 要调整就重建镜像，避免同一版本在不同部署上行为不一致。
_TRIAGE_WINDOW = 500
_TRIAGE_TRUST_MAX_CHARS = 900

# 边连接的单批新节点上限。超过就拆批调用（见 `_link_edges`）。
#
# 为什么需要：`edge_ops` 的条数与「本轮新节点 × 候选旧节点」同阶，单次输出会线性膨胀，
# 0927 实测有一次 `out=32768` 撞顶、JSON 在数组中途断掉，该批的边整批丢。
#
# 取值理由：正常的 `add_batch_size=6 / add_max_chars=1500` 下一批通常只产出个位数节点
# （0930 Full 实测单次 Add 平均 10.8 条消息、约 5.1 组，每组十几个节点），故 24 是
# "确实异常多"的门槛——多数调用**不会**触发拆批，行为与改造前逐字一致。
EDGE_BATCH_SIZE = 24

# 节点抽取上下文的「宽召回」参数：对每条消息各做一次向量检索并取并集。
#
# 为什么需要：值变更句往往只占对话的一两轮，用**整段对话**去检索时会被其余内容
# 稀释，旧值节点可能挤不进 top-k（离线实测在 224 条候选里排到第 53 / 80 名）。而
# 旧值节点不在上下文里，LLM 就没有 target_id 可废弃——只能 create，于是新旧两值
# 同时活跃。线上实测「覆盖机制」几乎不触发：134 批里只有 4 批发出 deprecate、
# 2 批发出 update（合计 update 5 次 / deprecate 7 次，对 create 1962 次）。
#
# 按构造，并集是原 top-k 的**超集**，只可能提高旧值召回、不会更差；代价是节点
# 步骤的输入多出至多 _WIDE_EXTRA_CAP 个节点（对话正文才是 prompt 的大头）。
_WIDE_K_PER_MSG = 5
_WIDE_EXTRA_CAP = 15

# ---- DBA 类 ----

class MemoryDBA:
    """记忆图谱的 LLM 数据库管理员"""

    def __init__(
        self,
        llm: ChatOpenAI,
        graph: MemoryGraph,
        vector_store: VectorStore,
        graph_builder: GraphBuilder,
        current_state_k: int = 15,
        edge_wide: bool = True,
        triage_enabled: bool = True,
        enable_choice_extraction: bool = False,
        enable_evidence: bool = False,
        enable_source_store: bool = False,
        source_dir=None,
        variant_tail: bool = False,
    ):
        """
        Args:
            llm: LangChain ChatOpenAI 实例
            graph: 目标 MemoryGraph
            vector_store: 向量存储
            graph_builder: GraphBuilder 实例
            current_state_k: 记忆上下文检索多少条相关节点
            edge_wide: 边连接步骤的旧节点上下文是否也走「逐消息并集」宽召回。
                **必须开**：提示词原则 3 早已要求「本轮新节点必须尽量接入已有图谱结构，
                能连则连」，但实测因果/时序边仍是**批内短程**的——以节点顺序作时间代理，
                causal/sequence 边的中位索引距离只有 8/9，而随机基线是 107；只有 42% 的
                节点有 causal/sequence 出边，能走满 3 跳的仅 21%（B2「因果链、路径与
                中间步骤恢复」连续 5 轮 0 分即源于此）。根因是边步骤的旧节点上下文只有
                15 条、且按**整段对话**相似度竞争，长对话下被稀释——与节点步骤同一个
                问题（见 _WIDE_K_PER_MSG）。
            triage_enabled: 是否启用 triage 维护判断（"这段对话值不值得维护"）。默认 True
                保持既有行为不变。**ALM 侧应关闭**，理由见 `_should_maintain` 的说明：
                分批之后多数组会落回长度闸以下，triage 会重新参与判定，实测把 41% 的组
                判成 SKIP → 内容被转出图（2026-09-26 冒烟：兜底率 11.1% → 50.7%）。
            enable_choice_extraction: 是否启用「决策与取舍」补充规则（把"被放弃的选项 /
                选择依据"也抽成 REASON 节点）。默认关闭以保持既有抽取行为不变；
                记忆保真任务 A 效果验证通过后再考虑对 ALM 侧开启。
            enable_evidence: 是否让抽取额外产出 evidence（原文支撑片段）并落盘。
                默认关闭；evidence 属原文片段，有隐私与体积属性，且不进检索/叙事。
            enable_source_store: 是否把本批**原文**落盘（供溯源巡检反查）。默认关闭——
                存储的是原始对话，是隐私与体积的主要来源，需显式开启。
            source_dir: 批次原文的落盘目录（由 `source_store.default_source_dir(yaml)` 得到）；
                enable_source_store=True 时必填，否则跳过落盘。
        """
        self.llm = llm
        self.graph = graph
        self.vector_store = vector_store
        self.builder = graph_builder
        self.current_state_k = current_state_k
        self.edge_wide = edge_wide
        self.triage_enabled = triage_enabled
        self.enable_choice_extraction = enable_choice_extraction
        self.enable_evidence = enable_evidence
        self.enable_source_store = enable_source_store
        self.source_dir = source_dir
        # 变体增量的落点。False（默认，现行行为）= 插到 `## 输出` **之前**；
        # True = **追加到 system 末尾**。
        #
        # 为什么要试末尾：`_node_chain` 的 4 个变体（base/temporal/choice/choice+temporal）
        # 在"插入点之前"共享前缀，插在中间会把 base 系统切成两半——实测跨变体调用的
        # 前缀缓存命中只有 **512 tok**，而 base 系统本身约 1,088 tok，于是节点抽取链的
        # 缓存命中率上限被压到 **37.8%**（见 版本变更记录 §4.25.2）。
        # 挪到末尾后，基础系统成为公共前缀，上限可望接近 80%。
        #
        # ⚠️ 但它同时改变了"规则 vs 输出格式"的相对位置，**会动质量**，
        # 故默认关；须先过 `alm.tools.alm_cache_probe`（上限）+ `alm.tools.alm_batch_quality_exp`（质量）两道。
        self.variant_tail = variant_tail

        # 组装 Prompt 链：节点抽取与边连接拆成两步（2026-08-22）
        def _node_chain(*extras):
            """把增强规则并入节点抽取链。

            `variant_tail=False`（默认）：插到 `## 输出` 之前 —— 现行行为。
            `variant_tail=True`：**追加到 system 末尾** —— 让基础系统成为变体间的公共前缀，
            换取缓存命中率（理由与代价见 `self.variant_tail` 的注释）。
            """
            parts = [e for e in extras if e]
            if enable_evidence:
                parts.append(NODE_EVIDENCE_EXTRA)
            system = NODE_EXTRACTION_PROMPT
            if parts:
                if self.variant_tail:
                    system = system + "\n" + "".join(parts)
                else:
                    system = system.replace("\n## 输出", "".join(parts) + "\n\n## 输出")
            if enable_evidence:
                # evidence 改变的是输出形状，替换示例而不是追加规则，避免两套 schema 打架
                system = system.replace(_NODE_OUTPUT_EXAMPLE, _NODE_OUTPUT_EXAMPLE_EVIDENCE)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", NODE_EXTRACTION_USER_PROMPT),
            ])
            return prompt | with_role(self.llm, "extract")

        self.node_chain = _node_chain()
        self.edge_prompt = ChatPromptTemplate.from_messages([
            ("system", EDGE_LINKING_PROMPT),
            ("human", EDGE_LINKING_FEWSHOT_EXAMPLE),
            ("human", EDGE_LINKING_USER_PROMPT),
        ])
        self.edge_chain = self.edge_prompt | with_role(self.llm, "link")

        # 时序叙事增强链：仅当 has_temporal_signal(conversation) 命中才启用，
        # 把时间节点抽取与 TEMPORAL 连边规则叠加到标准 prompt，避免常规语料过度拆分。
        self.node_chain_temporal = _node_chain(NODE_TEMPORAL_EXTRA)
        _edge_sys_temporal = (
            EDGE_LINKING_PROMPT + "\n" + EDGE_TEMPORAL_EXTRA
            if self.variant_tail
            else EDGE_LINKING_PROMPT.replace("\n## 输出", EDGE_TEMPORAL_EXTRA + "\n\n## 输出")
        )
        self.edge_chain_temporal = ChatPromptTemplate.from_messages([
            ("system", _edge_sys_temporal),
            ("human", EDGE_LINKING_FEWSHOT_EXAMPLE + EDGE_TEMPORAL_EXAMPLE),
            ("human", EDGE_LINKING_USER_PROMPT),
        ]) | with_role(self.llm, "link")

        # 决策与取舍增强链（opt-in，见 enable_choice_extraction）；关闭时不参与任何路径
        self.node_chain_choice = _node_chain(NODE_CHOICE_EXTRA)
        self.node_chain_choice_temporal = _node_chain(NODE_TEMPORAL_EXTRA, NODE_CHOICE_EXTRA)

        # 维护判断前置（triage）：先用极小 prompt 判断是否值得维护
        self.triage_chain = ChatPromptTemplate.from_messages([
            ("system", DBA_TRIAGE_PROMPT),
        ]) | with_role(self.llm, "triage")

    # ---- 主入口 ----

    def maintain(
        self,
        conversation: str,
        timestamp=None,
        source_rounds=None,
    ) -> Dict:
        """对话完成后执行一次数据库维护（两步：节点抽取 → 边连接）

        Args:
            conversation: 本轮对话的完整文本（含角色和时间戳）
            timestamp: 本批对话的发生时间（Unix 毫秒），写入本批新建节点。
                缺省 None 则不写时间戳。
            source_rounds: 本批各轮的 [{"ts", "text"}]，供原文落盘时保留轮次时间；
                缺省则退化为"单轮 = 整段 conversation"。

        Returns:
            {
                "ops": {"node_ops": [...], "edge_ops": [...]},  # LLM 原始输出
                "result": {"created_ids": [...], "skipped": [...], "errors": [...]},  # 执行结果
                "context": {...},  # 组装上下文（调试用）
            }
        """
        # 0. 维护判断前置：明显无需维护的对话直接跳过，省掉完整 prompt
        if not self._should_maintain(conversation):
            # 刻意用 WARNING 而非 INFO：ALM 容器把 dba_pipeline 钉在 WARNING 以上
            # （合规需要，防止 graph_builder 在 INFO 打印节点正文），INFO 在这里永远
            # 不可见。而「整批是否被 triage 跳过」是定位兜底率偏高的必要信号——
            # 本行只有对话长度，不含正文，可以安全放行。
            logger.warning("DBA 维护: triage 判定跳过（对话 %d 字符）", len(conversation))
            return {
                "ops": {"node_ops": [], "edge_ops": []},
                "result": {"created_ids": [], "skipped": [], "errors": []},
                "context": None,
                "skipped": True,
            }

        # Step 1:节点抽取(对话 + 相关旧节点 → node_ops → 执行)。时序叙事守卫命中则叠加 TEMPORAL 时间节点规则。
        temporal = has_temporal_signal(conversation)
        # 批次标识：一次 maintain 即一批；开启原文落盘时用它把节点与原文关联起来
        batch_id = new_batch_id() if self.enable_source_store else None
        if self.enable_choice_extraction:
            node_chain = self.node_chain_choice_temporal if temporal else self.node_chain_choice
        else:
            node_chain = self.node_chain_temporal if temporal else self.node_chain
        node_context = self._build_node_context(conversation)
        node_ops = self._parse_response(node_chain.invoke(node_context).content).get("node_ops", [])
        evidence_stats = None
        if self.enable_evidence:
            # 程序化校验：不是该批原文子串的 evidence 一律丢弃（LLM 可能写成概括/改写）
            node_ops, ev_kept, ev_dropped = sanitize_evidence(node_ops, conversation)
            evidence_stats = {"kept": ev_kept, "dropped": ev_dropped}
            if ev_dropped:
                logger.info(f"evidence 校验: 保留 {ev_kept} 条, 丢弃 {ev_dropped} 条非法片段")
        result1 = self.builder.apply_ops(node_ops, [], batch_timestamp=timestamp, batch_id=batch_id)
        # 原文落盘（opt-in）：只在本批确实产出了操作时存，避免把 triage 空转的对话堆到磁盘
        if batch_id and node_ops:
            self._persist_source(batch_id, source_rounds, conversation, timestamp)

        # Step 2:边连接(对话 + 本轮新节点 + 相关旧节点/一跳邻居/已有边 → edge_ops → 执行)。守卫同样驱动 TEMPORAL 连边规则与示例。
        new_ids = result1.get("created_ids", [])
        edge_chain = self.edge_chain_temporal if temporal else self.edge_chain
        edge_ops, edge_contexts = self._link_edges(edge_chain, conversation, new_ids)
        result2 = self.builder.apply_ops([], edge_ops)

        logger.info(
            f"DBA 维护完成: 节点创建 {len(result1.get('created_ids', []))} 更新 {len(node_ops)} "
            f"边创建/删除 {len(edge_ops)} (累计 {self.graph.node_count} 节点 / {self.graph.edge_count} 边)"
        )

        # 跨批连边观测锚点：本批新建的边里，有多少条**一端在本轮新节点、另一端是已有记忆**。
        # 这是 B2「因果链、路径与中间步骤恢复」的判决依据——提示词原则上要求跨批接入，
        # 但离线实测因果/时序边仍是批内短程（中位索引距离 8/9 vs 随机基线 107），
        # 说明跨批连边几乎没发生。只记计数，不记正文。
        new_set = set(new_ids)
        inner = cross = 0
        for op in edge_ops:
            # `action` 省略即 create（P1-2 输出瘦身后模型不再写 create）。
            # 这里若仍直接比 `!= "create"`，统计会**静默归零**——功能不受影响，
            # 但"跨批连边有没有发生"这个判决依据会断掉（见 版本变更记录 §4.26）。
            if (op.get("action") or "create") != "create":
                continue
            f, t = op.get("from"), op.get("to")
            if f in new_set and t in new_set:
                inner += 1
            elif (f in new_set) != (t in new_set):
                cross += 1
        # 刻意用 WARNING 而非 INFO：ALM 容器把 dba_pipeline 钉在 WARNING 以上
        # （合规需要，防止 graph_builder 在 INFO 打印节点正文），INFO 在这里永远不可见。
        # 而「跨批连边有没有发生」是定位 B2 的必要信号——本行只有计数，可以安全放行。
        logger.warning("边连边分布: 批内=%d 跨批=%d（本轮新节点 %d 个）", inner, cross, len(new_set))

        out = {
            "ops": {"node_ops": node_ops, "edge_ops": edge_ops},
            "result": {
                "created_ids": result1.get("created_ids", []),
                "skipped": result1.get("skipped", []) + result2.get("skipped", []),
                "errors": result1.get("errors", []) + result2.get("errors", []),
            },
            "context": {
                "node": node_context,
                # 拆批后"边上下文"不再唯一：这里给的是**第一批**，批数见 `edge_batches`。
                # 不给出批数的话，读日志的人会把这批的 new_nodes 误当成本轮全部。
                "edge": edge_contexts[0],
                "edge_batches": len(edge_contexts),
            },
        }
        if evidence_stats is not None:
            out["evidence_stats"] = evidence_stats
        if batch_id:
            out["batch_id"] = batch_id
        return out

    def _link_edges(self, edge_chain, conversation: str, new_ids: List[str]):
        """边连接：新节点多时**拆批**调用，避免单次输出撞上限。

        `edge_ops` 的条数与「本轮新节点 × 候选旧节点」同阶，单次输出会线性膨胀——
        0927 实测出现过一次 `out=32768` 撞顶，JSON 在数组中途断掉（`_salvage_truncated_ops`
        只能救回完整对象），**该批的边整批丢**。拆批让单次输出有界。

        代价是旧节点上下文（`current_nodes` / `current_edges` / `neighbor_info`）要按批
        **重复发送**，所以只在确实很多时才拆：`len(new_ids) <= EDGE_BATCH_SIZE` 时行为与
        改造前**逐字一致**（含 `new_ids` 为空时仍照常调用一次）。

        Returns:
            ``(edge_ops, contexts)`` —— `contexts` 是**实际用过**的上下文列表（拆批时多条）。
            必须一并返回：`maintain` 的调试字段要回填它，而拆批后已经没有"唯一的那份上下文"。
            这正是本方法签名从 `-> List[Dict]` 改成返回二元组的原因。
        """
        if len(new_ids) <= EDGE_BATCH_SIZE:
            edge_context = self._build_edge_context(conversation, new_ids)
            ops = self._parse_response(edge_chain.invoke(edge_context).content).get("edge_ops", [])
            return ops, [edge_context]

        batches = -(-len(new_ids) // EDGE_BATCH_SIZE)  # 向上取整
        # 用 WARNING 而非 INFO：ALM 容器把 dba_pipeline 钉在 WARNING 以上（防节点正文
        # 落进日志），拆批是"输出曾经撞上限"的信号，不能随之被压掉。本行只有计数。
        logger.warning(
            "边连接拆批: 本轮新节点 %d 个 > 上限 %d，拆成 %d 批（单次调用时输出会撞上限）",
            len(new_ids), EDGE_BATCH_SIZE, batches,
        )
        merged: List[Dict] = []
        contexts: List[Dict] = []
        for i in range(0, len(new_ids), EDGE_BATCH_SIZE):
            edge_context = self._build_edge_context(conversation, new_ids[i:i + EDGE_BATCH_SIZE])
            contexts.append(edge_context)
            merged.extend(
                self._parse_response(edge_chain.invoke(edge_context).content).get("edge_ops", [])
            )
        return merged, contexts

    def _persist_source(self, batch_id, source_rounds, conversation, timestamp):
        """把本批原文落盘（opt-in）。失败只记日志，绝不影响主流程。"""
        if not self.source_dir:
            logger.warning("enable_source_store=True 但未配置 source_dir，跳过原文落盘")
            return
        rounds = source_rounds or [{"ts": timestamp, "text": conversation}]
        try:
            save_batch(self.source_dir, batch_id, rounds)
        except Exception as e:
            logger.warning(f"原文落盘失败（批次 {batch_id}）: {e}")

    def store_raw(self, conversation: str, timestamp: Optional[int] = None) -> List[str]:
        """把未经抽取的原文兜底落库为可检索节点（MCP 侧专用）。

        背景（0927 审查 B5）：`_should_maintain` 判 SKIP 时，MCP 侧内容**被直接丢弃**
        （ALM 侧此时会经 `space._divert_raw` 转文档通道；MCP 没有文档通道）。这既造成
        记忆缺失，又完全静默——调用方只看到"空转跳过"，无从得知内容没被存储。

        本方法只为 MCP 链路补上这一步：原文**兜底落库**，并绕过去重（阈值提到 2.0），
        保证内容一定可检索。`_should_maintain` 只在对话长度 ≤ `_TRIAGE_TRUST_MAX_CHARS`
        时才采信 triage、才会返回 SKIP，故这里通常只是一段短文，单节点即可。

        Returns: 新建节点 id 列表。
        """
        text = (conversation or "").strip()
        if not text:
            return []
        ops = [{"action": "create", "content": text, "node_type": NodeType.THING.value}]
        saved_threshold = self.builder.dedup_threshold
        self.builder.dedup_threshold = 2.0  # 关闭语义去重，保证原文一定落地
        try:
            result = self.builder.apply_ops(ops, [], batch_timestamp=timestamp)
        finally:
            self.builder.dedup_threshold = saved_threshold
        created = result.get("created_ids") or []
        # 用 WARNING：ALM 容器把 dba_pipeline 钉在 WARNING 以上，INFO 不可见。
        # 本行只有计数，不含正文，可安全放行。
        logger.warning("原文兜底落库: %d 字符 → %d 个节点", len(text), len(created))
        return created

    # ---- 上下文组装 ----

    def _build_node_context(self, conversation: str) -> Dict:
        """Step 1 节点抽取上下文：对话 + 相关旧节点（用于去重/更新判断）"""
        if self.graph.node_count == 0:
            nodes_text = "（暂无已有记忆）"
        else:
            # wide=True：额外做逐消息并集检索，提高「旧值节点」进入上下文的概率
            # （值变更场景的根因，见 _WIDE_K_PER_MSG 的说明）。
            nodes_text, _ = self._build_memory_context(conversation, wide=True)
        return {
            "conversation": conversation,
            "current_nodes": nodes_text,
            "language_directive": _LANGUAGE_DIRECTIVE.format(
                lang=source_language(conversation)
            ),
        }

    def _build_edge_context(self, conversation: str, new_ids: List[str]) -> Dict:
        """Step 2 边连接上下文：对话 + 本轮新节点(全) + 相关旧节点 + 一跳邻居 + 已有边"""
        if self.graph.node_count == 0:
            current_nodes_text = current_edges_text = "（暂无已有记忆）"
        else:
            current_nodes_text, current_edges_text = self._build_memory_context(
                conversation, wide=self.edge_wide
            )
        neighbor_text = self._build_structure_context(conversation)

        new_lines = []
        for nid in new_ids:
            node = self.graph.get_node(nid)
            if node is None:
                continue
            type_val = node["node_type"].value if hasattr(node["node_type"], "value") else str(node["node_type"])
            new_lines.append(f"  [{nid}] {type_val}: {node['content'][:120]}")
        new_nodes_text = "\n".join(new_lines) if new_lines else "（本轮无新节点）"

        return {
            "conversation": conversation,
            "new_nodes": new_nodes_text,
            "language_directive": _LANGUAGE_DIRECTIVE.format(
                lang=source_language(conversation)
            ),
            "current_nodes": current_nodes_text,
            "current_edges": current_edges_text,
            "neighbor_info": neighbor_text,
            "total_nodes": self.graph.node_count,
            "total_edges": self.graph.edge_count,
        }

    def _widen_by_message(
        self,
        conversation: str,
        base: List[Tuple[str, float, dict]],
    ) -> List[Tuple[str, float, dict]]:
        """对对话的每条消息各检索一次，把新命中的节点并到 base 之后

        消息按 _format_conversation 的约定逐行排列（每行一个 `[role] 正文`），因此
        这里按行切分即可；正文自身含换行时该行会被切成片段，退化成一次额外的近似
        检索，不影响正确性（并集是超集，不会挤掉原 top-k）。

        向量化走一次批量调用（embed_batch），检索走本地 FAISS，不产生逐条 API 往返。
        任何失败都静默退回原 top-k。
        """
        lines = [ln.strip() for ln in conversation.split("\n") if ln.strip()]
        if not lines:
            return base
        try:
            vecs = self.vector_store.embed_batch(lines)
        except Exception as exc:
            logger.warning("宽召回向量化失败，退回单路检索: %s", type(exc).__name__)
            return base

        seen = {mid for mid, _, _ in base}
        extra: List[Tuple[str, float, dict]] = []
        for vec in vecs:
            for mid, score in self.vector_store.search_by_vector(vec, k=_WIDE_K_PER_MSG):
                if mid in seen:
                    continue
                seen.add(mid)
                extra.append((mid, float(score), {}))
                if len(extra) >= _WIDE_EXTRA_CAP:
                    return base + extra
        return base + extra

    def _build_memory_context(self, conversation: str, wide: bool = False) -> Tuple[str, str]:
        """构建记忆上下文：语义相关节点 + 已有边

        wide=True 时额外做「逐消息并集」宽召回（见 _WIDE_K_PER_MSG 的说明），只用于
        节点抽取步骤；边连接步骤不做宽召回——它的输入本就很大（线上实测 in=14644）。
        """
        if self.graph.node_count == 0:
            return "（暂无已有记忆）", "（暂无已有边）"

        results = list(self.vector_store.search(conversation, k=self.current_state_k))
        if wide:
            results = self._widen_by_message(conversation, results)

        node_lines = []
        shown_ids = set()
        for mid, score, _ in results:
            node = self.graph.get_node(mid)
            # 与 retriever._is_active 对齐：deprecated 与 forgotten 都不该再进上下文
            if node is None or node.get("deprecated") or node.get("forgotten"):
                continue
            type_val = node["node_type"].value if hasattr(node["node_type"], "value") else str(node["node_type"])
            content = node["content"][:100]
            node_lines.append(f"  [{mid}] {type_val}: {content}")
            shown_ids.add(mid)

        # 收集这些节点之间的边
        edge_lines = []
        shown_edges = set()
        for u in shown_ids:
            for v in shown_ids:
                if u == v:
                    continue
                if self.graph.graph.has_edge(u, v):
                    edge_key = (u, v)
                    if edge_key in shown_edges:
                        continue
                    shown_edges.add(edge_key)
                    edge_data = self.graph.graph.edges[u, v]
                    rel_type = edge_data.get("rel_type")
                    rel_str = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
                    edge_lines.append(f"  {u} --[{rel_str}]--> {v}")

        nodes_text = "\n".join(node_lines) if node_lines else "（未找到相关已有记忆）"
        edges_text = "\n".join(edge_lines) if edge_lines else "（相关节点间暂无直接边）"

        return nodes_text, edges_text

    def _build_structure_context(self, conversation: str) -> str:
        """构建结构上下文：相关节点的一跳邻居"""
        if self.graph.node_count == 0:
            return "（暂无图谱结构）"

        results = self.vector_store.search(conversation, k=5)
        neighbor_set = set()
        for mid, _, _ in results:
            node = self.graph.get_node(mid)
            if node is None or node.get("deprecated") or node.get("forgotten"):
                continue
            for neighbor_id, rel_type, is_reverse in self.graph.get_neighbors(mid):
                neighbor_node = self.graph.get_node(neighbor_id)
                if (neighbor_node and not neighbor_node.get("deprecated")
                        and not neighbor_node.get("forgotten")):
                    direction = "←" if is_reverse else "→"
                    rel_str = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
                    neighbor_set.add(f"  {mid} --{direction}[{rel_str}]-- {neighbor_id}")

        if neighbor_set:
            return "\n".join(sorted(neighbor_set)[:20])
        return "（相关节点暂无邻居）"

    # ---- LLM 调用 ----

    def _should_maintain(self, conversation: str) -> bool:
        """维护判断前置：极小 prompt 判断是否值得维护。

        **ALM 侧应整体关闭（`triage_enabled=False`）。** 2026-09-26 冒烟实测：写入分批
        （`add_max_chars=1500`）把一次 Add 切成若干小组后，多数组的长度落回
        `_TRIAGE_TRUST_MAX_CHARS=900` **以下**，于是 triage 重新参与判定——而改造前整批
        2000~3000 字 > 1500，它是被整体放行的。后果：**421 个组里 173 个被判 SKIP（41%）**，
        这些内容经 `space._divert_raw` 转出图，兜底率 11.1% → 50.7%，图只拿到约 59% 的内容。

        为什么结论是"关"而不是"再调阈值"：错判不在窗口覆盖率上——实测有个组对话只有
        113 字符、窗口覆盖 100%，仍把含事实的 "It's much better for my neck actually"
        判成 SKIP。而 ALM 的数据全是评测对话，"值得存"的先验极高，分组也已经把单次输入
        压到合理规模：triage 省下的那一次抽取，远不值"丢记忆"的代价。

        保留本方法与下面的长度闸，是为了 MCP 侧（默认 `triage_enabled=True`）行为不变；
        这段判断对**短寒暄**仍然有效，故不删。
        """
        if not self.triage_enabled:
            return True
        if len(conversation) > _TRIAGE_TRUST_MAX_CHARS:
            return True
        snippet = conversation[-_TRIAGE_WINDOW:]
        resp = self.triage_chain.invoke({"conversation": snippet})
        return "NEEDED" in resp.content.upper()

    # ---- 响应解析 ----

    def _parse_response(self, response: str) -> Dict:
        """解析 LLM 输出，提取 node_ops 和 edge_ops"""
        # 尝试直接解析 JSON
        cleaned = response.strip()

        # 去掉可能的 markdown 代码块包裹
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # 去掉第一行 ```json 和最后一行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            ops = json.loads(cleaned)
            return {
                "node_ops": ops.get("node_ops", []),
                "edge_ops": ops.get("edge_ops", []),
            }
        except json.JSONDecodeError:
            logger.warning(f"LLM 输出 JSON 解析失败，原始响应前 200 字符: {response[:200]}")

            # 尝试提取 JSON 片段
            return self._extract_json_fallback(response)

    def _extract_json_fallback(self, response: str) -> Dict:
        """从非标准格式的 LLM 输出中尝试提取 JSON"""
        # 尝试找到第一个 { 和最后一个 }
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        # 截断挽救：输出撞 max_tokens 上限时 JSON 会在数组中途断掉，上面两条路都会
        # 失败，于是**整批操作被丢弃**——线上实测边连接步骤撞 8192 上限 4 次 / 369 次
        # 调用，其中 2 次解析失败，那一批的边全部没建。但断点之前的完整对象其实都还
        # 在，按花括号配对捞回来，部分结果远好于全丢。
        salvaged = self._salvage_truncated_ops(response)
        if salvaged["node_ops"] or salvaged["edge_ops"]:
            logger.warning(
                "LLM 输出疑似被截断，已挽救 node_ops=%d edge_ops=%d",
                len(salvaged["node_ops"]), len(salvaged["edge_ops"]),
            )
            return salvaged

        logger.error(f"无法从 LLM 输出中提取有效的 JSON 操作指令")
        return {"node_ops": [], "edge_ops": []}

    @staticmethod
    def _salvage_truncated_ops(response: str) -> Dict:
        """按花括号配对，从被截断的 JSON 里捞出所有完整的操作对象

        只在常规解析全部失败后调用，因此正常路径零影响。扫描时跳过字符串内部
        （含转义处理），避免把正文里的 `{` / `}` 误当作结构。
        """
        out: Dict[str, List[Dict]] = {"node_ops": [], "edge_ops": []}
        for key in ("node_ops", "edge_ops"):
            key_pos = response.find(f'"{key}"')
            if key_pos == -1:
                continue
            arr = response.find("[", key_pos)
            if arr == -1:
                continue

            depth, obj_start, in_str, esc = 0, None, False, False
            for i in range(arr + 1, len(response)):
                ch = response[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and obj_start is not None:
                            try:
                                obj = json.loads(response[obj_start:i + 1])
                            except json.JSONDecodeError:
                                obj = None
                            if isinstance(obj, dict):
                                out[key].append(obj)
                            obj_start = None
                elif ch == "]" and depth == 0:
                    break
        return out

    # ---- 检查点 ----

    def save_checkpoint(self, save_dir: str):
        """保存完整检查点（图谱 + 向量 + 构建器状态）"""
        import yaml
        import json
        import os
        from datetime import datetime

        os.makedirs(save_dir, exist_ok=True)

        # 1. 图谱 → YAML
        graph_path = os.path.join(save_dir, "memory_graph.yaml")
        with open(graph_path, "w", encoding="utf-8") as f:
            yaml.dump(self.graph.to_dict(), f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)

        # 2. 向量存储
        vs_path = os.path.join(save_dir, "faiss_index")
        if self.vector_store and self.vector_store.has_vectors():
            self.vector_store.save(vs_path)

        # 3. 构建器状态 → JSON
        builder_path = os.path.join(save_dir, "builder_state.json")
        with open(builder_path, "w", encoding="utf-8") as f:
            json.dump(self.builder.save_state(), f, ensure_ascii=False)

        # 4. 检查点元数据
        checkpoint_path = os.path.join(save_dir, "checkpoint.json")
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1",
                "timestamp": datetime.now().isoformat(),
                "nodes": self.graph.node_count,
                "edges": self.graph.edge_count,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"检查点已保存: {save_dir} ({self.graph.node_count} 节点, {self.graph.edge_count} 边)")
        return save_dir

    def restore_checkpoint(self, save_dir: str) -> bool:
        """从检查点恢复全部状态"""
        import yaml
        import json
        import os

        # 1. 图谱
        graph_path = os.path.join(save_dir, "memory_graph.yaml")
        if os.path.exists(graph_path):
            restored = MemoryGraph.from_dict(
                yaml.safe_load(open(graph_path, encoding="utf-8"))
            )
            self.graph.graph = restored.graph  # 替换内部 nx 图
            logger.info(f"图谱已恢复: {self.graph.node_count} 节点, {self.graph.edge_count} 边")
        else:
            logger.warning("未找到图谱检查点")

        # 2. 向量存储
        vs_path = os.path.join(save_dir, "faiss_index")
        if os.path.exists(vs_path):
            self.vector_store.load(vs_path, embeddings=self.vector_store.embeddings)

        # 3. 构建器状态
        builder_path = os.path.join(save_dir, "builder_state.json")
        if os.path.exists(builder_path):
            with open(builder_path, encoding="utf-8") as f:
                self.builder.load_state(json.load(f))
            logger.info("构建器状态已恢复")

        return True
