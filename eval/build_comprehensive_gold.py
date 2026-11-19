# -*- coding: utf-8 -*-
"""
综合能力测评 · gold 图谱构建器（大叙事 + 双主体 + 时序，~200 节点）

参照三套既有数据集的能力融合：
  · LiFEMem（完整生命周期叙事：高中→大专→工作，7 类查询）
  · agent_user（双主体：用户事实 + 助手自身行为，content 前缀区分）
  · tememporal/crossperiod（事件级时间锚点 + TEMPORAL/SEQUENCE/CAUSAL 驱动时间线重建）

设计（规模目标 ~200 节点，跨期大叙事 + 双主体 + 事件级时序）：
  1. 生命周期主轴：高中(h)→大专(c)→工作(w)，以时间锚点 T 承载跨期纪元秩。
  2. 双主体：每个场景同时有【用户 u】与【助手 a】节点（content 前缀「用户…/助手…」，
     助手也承载行为/产出信息点），时间锚点为 t。
  3. 事件级时序：每场景 ≥2 个不同事件级时间词，TEMPORAL 边(事件→时间) 驱动 has_temporal_signal，
     SEQUENCE 边承载事件先后，CAUSAL 承载因果。
  4. 关系类型覆盖：causal/scenario/sequence/temporal + attribute/person 属性边，保证跳转轴可走。

本脚本负责：校验（id 唯一 / 边两端存在 / NodeType·RelationType 合法 / content 前缀与主体匹配）、
双向边自动补全（scenario/attribute，social 亦可）、输出 data/dba_eval/comprehensive_gold.yaml。

用法:
    python -m eval.build_comprehensive_gold
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from dba_pipeline.core.jump_axis import NodeType, RelationType

VALID_NODE_TYPE = {t.value for t in NodeType}
VALID_REL_TYPE = {r.value for r in RelationType}

# 主体前缀 → 场景（供生成器分组与统计）
SCENE_PREFIX = {
    "uh": "学生时代/高中", "ah": "学生时代/高中", "th": "学生时代/高中",
    "uc": "学生时代/大专", "ac": "学生时代/大专", "tc": "学生时代/大专",
    "uw": "工作开发", "aw": "工作开发", "tw": "工作开发",
    "ud": "日常起居/健康", "ad": "日常起居/健康", "td": "日常起居/健康",
    "us": "社交/约会", "as": "社交/约会", "ts": "社交/约会",
    "ul": "学习/探索", "al": "学习/探索", "tl": "学习/探索",
    "um": "混合闲聊", "am": "混合闲聊",
}

SCENE_ORDER = [
    ("学生时代/高中", "高中学习与早恋"),
    ("学生时代/大专", "大专迷茫与自学"),
    ("工作开发", "项目上线周期与排期"),
    ("日常起居/健康", "晨跑体检与作息"),
    ("社交/约会", "聚会与生日会"),
    ("学习/探索", "数据分析学习"),
    ("混合闲聊", "感冒奶茶与养猫"),
]


# (id, content, type)
NODES = [
    # ================= 场景 h：学生时代/高中（用户 + 助手 + 时间锚点） =================
    ("uh1", "用户高中时成绩一直名列前茅", "status"),
    ("uh2", "用户高三那年为了高考熬夜刷题", "action"),
    ("uh3", "用户高考前感到紧张和压力", "emotion"),
    ("uh4", "用户高中时喜欢在晚自习后去操场跑步", "action"),
    ("uh5", "用户高中时和同桌关系很铁", "status"),
    ("uh6", "用户高中时暗恋过隔壁班的一个女生", "emotion"),
    ("uh7", "用户高考那几天失眠了", "action"),
    ("uh8", "用户最后考上了一所普通大专", "status"),
    ("uh9", "用户高一时参加了数学竞赛拿了校三等奖", "status"),
    ("uh10", "用户高二开始沉迷打游戏", "action"),
    ("uh11", "用户高二成绩跌到班级中下游", "status"),
    ("uh12", "用户高中时吃学校门口小卖部的辣条", "action"),
    ("uh13", "用户高中时晚自习是觉得最安静的时光", "emotion"),
    ("uh14", "用户高中时每天骑车上下学", "action"),
    ("uh15", "用户高中时的同桌上课总爱传纸条", "action"),
    ("uh16", "用户高考完那个暑假第一次去旅行", "action"),
    ("uh17", "用户高三课桌上贴满了激励自己的便签", "thing"),

    ("ah1", "助手帮用户把高中三年的成绩变化梳理成了曲线", "action"),
    ("ah2", "助手帮用户回忆高考前那段压力的来源", "action"),
    ("ah3", "助手给用户梳理了高中到大学的时间线", "action"),
    ("ah4", "助手帮用户整理了几条缓解学习焦虑的建议", "action"),
    ("ah5", "助手陪用户回忆了高中打游戏荒废学业那段", "action"),
    ("ah6", "助手帮用户整理了高中到工作的成长脉络", "action"),

    ("th1", "高一", "thing"),
    ("th2", "高三", "thing"),
    ("th3", "高考前", "thing"),
    ("th4", "高考那几天", "thing"),
    ("th5", "高二", "thing"),
    ("th6", "高中时", "thing"),
    ("th7", "高考后的暑假", "thing"),
    ("th8", "晚自习", "thing"),

    # ================= 场景 c：学生时代/大专（用户 + 助手 + 时间锚点） =================
    ("uc1", "用户大专时一度浑浑噩噩打游戏", "action"),
    ("uc2", "用户大二被室友拉着参加社团", "action"),
    ("uc3", "用户在大二下学期偶然接触了编程", "action"),
    ("uc4", "用户大三开始系统自学编程", "action"),
    ("uc5", "用户毕业时拿到了后端开发的offer", "status"),
    ("uc6", "用户大专时对未来感到迷茫", "emotion"),
    ("uc7", "用户的大专室友关系很好", "status"),
    ("uc8", "用户毕业前参加了学校的招聘会", "action"),
    ("uc9", "用户大专时参加过编程社团的培训课", "action"),
    ("uc10", "用户从迷茫中走出来靠的是编程找到了兴趣", "status"),
    ("uc11", "用户大一时经常逃课", "action"),
    ("uc12", "用户大一期末挂过一科", "status"),
    ("uc13", "用户大二开始规律地上自习", "action"),
    ("uc14", "用户大专时在食堂经常吃麻辣香锅", "action"),
    ("uc15", "用户大三在宿舍熬夜写代码", "action"),
    ("uc16", "用户毕业时对室友有点舍不得", "emotion"),
    ("uc17", "用户在校招会上投了很多简历", "action"),
    ("uc18", "用户大专时加过编程学习群", "action"),

    ("ac1", "助手帮用户回顾了大专时期从迷茫到找到方向的转变", "action"),
    ("ac2", "助手帮用户整理了自学编程的学习路线", "action"),
    ("ac3", "助手帮用户梳理了职业起点的时间线", "action"),
    ("ac4", "助手帮用户复盘了大一挂科的原因", "action"),
    ("ac5", "助手帮用户规划了大三的求职准备", "action"),
    ("ac6", "助手帮用户整理了社团活动里的收获", "action"),

    ("tc1", "大一", "thing"),
    ("tc2", "大二下学期", "thing"),
    ("tc3", "大三", "thing"),
    ("tc4", "毕业前", "thing"),
    ("tc5", "大二", "thing"),
    ("tc6", "大专时", "thing"),
    ("tc7", "校招", "thing"),

    # ================= 场景 w：工作开发 =================
    ("uw1", "用户在一家互联网公司做后端开发", "status"),
    ("uw2", "用户这周项目很赶，每天加班到十一点", "action"),
    ("uw3", "用户周一晚上临时接了一个紧急需求", "action"),
    ("uw4", "用户周三凌晨把项目上线", "action"),
    ("uw5", "用户的项目上线时出了一个P0级严重故障", "status"),
    ("uw6", "用户周四深夜通宵定位并修复故障", "action"),
    ("uw7", "用户周五参加了项目复盘会", "action"),
    ("uw8", "用户对临时加需求感到烦躁", "emotion"),
    ("uw9", "用户这周加班后非常疲惫", "emotion"),
    ("uw10", "用户习惯把踩过的坑写成技术笔记", "action"),
    ("uw11", "用户想换一个有双休的团队", "status"),
    ("uw12", "用户的技术主管叫老马，对代码质量要求很严格", "person"),
    ("uw13", "用户和同事小王经常一起加班", "action"),
    ("uw14", "用户加班多是因为需求临时增加", "reason"),

    ("aw1", "助手帮用户把下周排期整理成清单并标注优先级", "action"),
    ("aw2", "助手在排期清单中标出周三的需求评审会", "action"),
    ("aw3", "助手用Python写了一个自动拆分任务的脚本", "action"),
    ("aw4", "助手帮用户梳理了P0故障的排查要点", "action"),
    ("aw5", "助手帮用户生成了复盘提纲", "action"),
    ("aw6", "助手给用户推荐了一个压测工具", "action"),
    ("aw7", "助手把用户的技术笔记整理成了文档", "action"),
    ("aw8", "助手帮用户对比了几个双休团队的在招岗位", "action"),
    ("aw9", "助手提醒用户连续加班要注意休息", "action"),

    ("tw1", "周一晚上", "thing"),
    ("tw2", "周二白天", "thing"),
    ("tw3", "周三凌晨", "thing"),
    ("tw4", "周四深夜", "thing"),
    ("tw5", "周五", "thing"),
    ("tw6", "这周", "thing"),
    ("tw7", "下周", "thing"),

    # ================= 场景 d：日常起居/健康 =================
    ("ud1", "用户平时每天早上绕小区晨跑半小时", "action"),
    ("ud2", "用户昨天没跑成晨跑", "action"),
    ("ud3", "用户上周一去体检了", "action"),
    ("ud4", "用户这次体检指标比去年好了一些", "status"),
    ("ud5", "用户这周开始吃减脂餐", "action"),
    ("ud6", "用户对改善作息挺有成就感", "emotion"),
    ("ud7", "用户晚上经常十二点以后才睡", "status"),
    ("ud8", "用户打算明天去办一张健身卡", "action"),
    ("ud9", "用户以前颈椎不太好", "status"),

    ("ad1", "助手帮用户把一周的晨跑安排在了日历上", "action"),
    ("ad2", "助手帮用户整理了一份减脂餐食谱", "action"),
    ("ad3", "助手提醒用户记录每天的睡眠时间", "action"),
    ("ad4", "助手帮用户对比了附近几家健身房的年卡", "action"),
    ("ad5", "助手把用户的体检报告重点摘了出来", "action"),
    ("ad6", "助手给用户设计了一个循序渐进的作息调整计划", "action"),

    ("td1", "昨天", "thing"),
    ("td2", "上周一", "thing"),
    ("td3", "这周", "thing"),
    ("td4", "明天", "thing"),
    ("td5", "今年", "thing"),

    # ================= 场景 s：社交/约会 =================
    ("us1", "用户这周六和朋友约了聚餐", "action"),
    ("us2", "用户周五下班后经常去朋友家打桌游", "action"),
    ("us3", "用户上周三参加了同事的生日会", "action"),
    ("us4", "用户帮朋友准备了生日礼物", "action"),
    ("us5", "用户和表妹关系很好", "status"),
    ("us6", "用户打算下周日陪表妹逛街", "action"),
    ("us7", "用户想到聚会有点期待", "emotion"),
    ("us8", "用户喜欢在群里组织周末活动", "action"),
    ("us9", "用户的老朋友陈浩性格开朗爱开玩笑", "person"),
    ("us10", "用户的表妹还在读大学", "person"),

    ("as1", "助手帮用户列了聚餐地点清单", "action"),
    ("as2", "助手帮用户订了周六的餐厅", "action"),
    ("as3", "助手帮用户写了生日祝福卡片文案", "action"),
    ("as4", "助手提醒用户聚会别太晚睡", "action"),

    ("ts1", "上周三", "thing"),
    ("ts2", "周五", "thing"),
    ("ts3", "周六", "thing"),
    ("ts4", "下周日", "thing"),
    ("ts5", "这周", "thing"),

    # ================= 场景 l：学习/探索 =================
    ("ul1", "用户最近想学数据分析", "status"),
    ("ul2", "用户有一定的Python编程基础", "status"),
    ("ul3", "用户上周报了一个数据分析网课", "action"),
    ("ul4", "用户这周在赶网课的作业", "action"),
    ("ul5", "用户周末会刷LeetCode", "action"),
    ("ul6", "用户想先把数据分析做熟再转向AI", "status"),
    ("ul7", "用户打算明天把第一个项目跑起来", "action"),
    ("ul8", "用户对学习进展有点焦虑", "emotion"),

    ("al1", "助手建议用户先从Python数据分析入门", "action"),
    ("al2", "助手帮用户整理了3个入门项目", "action"),
    ("al3", "助手给用户推荐了一门适配其编程基础的数据分析基础课", "action"),
    ("al4", "助手帮用户把网课作业拆成了每日计划", "action"),
    ("al5", "助手帮用户写了一段数据处理示例代码", "action"),
    ("al6", "助手帮用户整理了刷题节奏安排", "action"),
    ("al7", "助手给用户梳理了数据分析到AI的学习路线图", "action"),
    ("al8", "助手给用户提供了缓解焦虑的节奏建议", "action"),

    ("tl1", "上周", "thing"),
    ("tl2", "这周", "thing"),
    ("tl3", "周日", "thing"),
    ("tl4", "明天", "thing"),
    ("tl5", "明年", "thing"),

    # ================= 场景 m：混合闲聊 =================
    ("um1", "用户今天有点感冒", "status"),
    ("um2", "用户刚才下楼买了杯热奶茶", "action"),
    ("um3", "用户很喜欢猫", "status"),
    ("um4", "用户曾经养过一只橘猫", "action"),
    ("um5", "用户的家在城西", "status"),
    ("um6", "用户对今年的安排还挺满意", "emotion"),

    ("am1", "助手劝用户多喝热水注意休息", "action"),
    ("am2", "助手给用户推荐了一家好喝的热奶茶店", "action"),
    ("am3", "助手给用户讲了几个养猫的小贴士", "action"),
    ("am4", "助手帮用户把今年想做的事列了个清单", "action"),
    ("am5", "助手陪用户聊了会儿天", "action"),
]


# (from, to, rel_type)；SCENARIO/ATTRIBUTE/SOCIAL 自动补反向
EDGES = [
    # ---- 场景 h：temporal（事件→时间）----
    ("uh1", "th1", "temporal"), ("uh2", "th2", "temporal"),
    ("uh2", "th3", "temporal"), ("uh7", "th4", "temporal"),
    ("uh8", "th4", "temporal"), ("uh4", "th1", "temporal"),
    ("ah1", "th2", "temporal"), ("ah2", "th3", "temporal"), ("ah3", "th4", "temporal"),
    ("uh9", "th1", "temporal"), ("uh10", "th5", "temporal"), ("uh11", "th5", "temporal"),
    ("uh12", "th6", "temporal"), ("uh13", "th8", "temporal"), ("uh14", "th6", "temporal"),
    ("uh16", "th7", "temporal"), ("uh17", "th2", "temporal"),
    ("ah4", "th3", "temporal"), ("ah5", "th5", "temporal"), ("ah6", "th7", "temporal"),
    # ---- 场景 h：sequence（先后）----
    ("uh1", "uh2", "sequence"), ("uh2", "uh3", "sequence"), ("uh3", "uh7", "sequence"),
    ("uh7", "uh8", "sequence"), ("ah1", "ah2", "sequence"),
    ("uh1", "uh10", "sequence"), ("uh10", "uh11", "sequence"), ("uh11", "uh2", "sequence"),
    ("uh16", "uh8", "sequence"), ("uh16", "ah6", "sequence"), ("uh9", "uh1", "sequence"),
    # ---- 场景 h：causal ----
    ("uh2", "uh3", "causal"), ("uh3", "uh7", "causal"), ("uh6", "uh3", "causal"),
    ("uh5", "uh6", "causal"), ("uh7", "uh8", "causal"),
    ("uh10", "uh11", "causal"), ("uh11", "uh2", "causal"), ("uh11", "uh13", "causal"),
    ("uh13", "ah4", "causal"), ("uh6", "uh2", "causal"),
    # ---- 场景 h：scenario（双向）----
    ("uh5", "uh6", "scenario"), ("uh4", "uh1", "scenario"), ("ah1", "uh2", "scenario"),
    ("uh15", "uh5", "scenario"), ("uh12", "uh14", "scenario"), ("uh17", "uh2", "scenario"),
    ("uh9", "uh1", "scenario"), ("ah5", "uh10", "scenario"),

    # ---- 场景 c：temporal ----
    ("uc1", "tc1", "temporal"), ("uc2", "tc2", "temporal"), ("uc3", "tc2", "temporal"),
    ("uc4", "tc3", "temporal"), ("uc8", "tc4", "temporal"), ("uc5", "tc4", "temporal"),
    ("ac1", "tc3", "temporal"), ("ac3", "tc4", "temporal"),
    ("uc11", "tc1", "temporal"), ("uc12", "tc1", "temporal"), ("uc13", "tc5", "temporal"),
    ("uc14", "tc6", "temporal"), ("uc15", "tc3", "temporal"), ("uc17", "tc7", "temporal"),
    ("uc18", "tc6", "temporal"), ("ac4", "tc1", "temporal"), ("ac5", "tc4", "temporal"),
    ("ac6", "tc5", "temporal"),
    # ---- 场景 c：sequence ----
    ("uc1", "uc2", "sequence"), ("uc2", "uc3", "sequence"), ("uc3", "uc4", "sequence"),
    ("uc4", "uc5", "sequence"), ("uc4", "uc7", "sequence"),
    ("uc1", "uc11", "sequence"), ("uc11", "uc12", "sequence"), ("uc12", "uc13", "sequence"),
    ("uc13", "uc4", "sequence"), ("uc15", "uc5", "sequence"), ("uc17", "uc5", "sequence"),
    # ---- 场景 c：causal ----
    ("uc1", "uc6", "causal"), ("uc6", "uc3", "causal"), ("uc3", "uc10", "causal"),
    ("uc4", "uc10", "causal"), ("uc10", "uc5", "causal"), ("uc2", "uc7", "causal"),
    ("uc11", "uc12", "causal"), ("uc12", "uc13", "causal"),
    ("uc13", "uc18", "causal"), ("uc16", "uc7", "causal"),
    # ---- 场景 c：scenario ----
    ("uc2", "uc9", "scenario"), ("uc3", "ac2", "scenario"), ("uc4", "ac2", "scenario"),
    ("uc7", "uc2", "scenario"), ("uc10", "ac1", "scenario"),
    ("uc9", "ac6", "scenario"), ("uc14", "uc7", "scenario"), ("uc15", "uc4", "scenario"),
    ("uc17", "ac5", "scenario"), ("uc12", "ac4", "scenario"),

    # ---- 场景 w：temporal ----
    ("uw3", "tw1", "temporal"), ("uw4", "tw3", "temporal"), ("uw5", "tw3", "temporal"),
    ("uw6", "tw4", "temporal"), ("uw7", "tw5", "temporal"), ("uw2", "tw6", "temporal"),
    ("aw1", "tw7", "temporal"), ("uw11", "tw7", "temporal"),
    # ---- 场景 w：sequence ----
    ("uw3", "uw4", "sequence"), ("uw4", "uw5", "sequence"), ("uw5", "uw6", "sequence"),
    ("uw6", "uw7", "sequence"), ("aw1", "aw2", "sequence"),
    # ---- 场景 w：causal ----
    ("uw3", "uw8", "causal"), ("uw2", "uw9", "causal"), ("uw6", "uw9", "causal"),
    ("uw9", "aw9", "causal"), ("uw2", "uw5", "causal"),
    # ---- 场景 w：scenario ----
    ("uw3", "aw1", "scenario"), ("uw4", "aw3", "scenario"), ("uw5", "aw4", "scenario"),
    ("uw7", "aw5", "scenario"), ("uw11", "aw8", "scenario"), ("uw10", "aw7", "scenario"),
    ("uw12", "aw9", "scenario"), ("uw13", "uw14", "scenario"), ("uw2", "aw3", "scenario"),
    # ---- 场景 w：attribute / social ----
    ("uw12", "uw1", "attribute"), ("uw13", "uw12", "social"), ("uw14", "uw2", "causal"),

    # ---- 场景 d：temporal ----
    ("ud1", "td5", "temporal"), ("ud2", "td1", "temporal"), ("ud3", "td2", "temporal"),
    ("ud5", "td3", "temporal"), ("ud8", "td4", "temporal"),
    # ---- 场景 d：sequence ----
    ("ud3", "ud5", "sequence"), ("ad1", "ud1", "sequence"), ("ad6", "ud7", "sequence"),
    # ---- 场景 d：causal ----
    ("ud3", "ud4", "causal"), ("ud7", "ud2", "causal"), ("ud9", "ud7", "causal"),
    # ---- 场景 d：scenario ----
    ("ud1", "ad1", "scenario"), ("ud5", "ad2", "scenario"), ("ud8", "ad4", "scenario"),
    ("ud6", "ad6", "scenario"), ("ud1", "ad3", "scenario"),

    # ---- 场景 s：temporal ----
    ("us3", "ts1", "temporal"), ("us2", "ts2", "temporal"), ("us1", "ts3", "temporal"),
    ("us6", "ts4", "temporal"), ("us8", "ts5", "temporal"),
    # ---- 场景 s：sequence ----
    ("as1", "as2", "sequence"), ("us1", "us7", "sequence"),
    # ---- 场景 s：causal ----
    ("us5", "us6", "causal"), ("us7", "as2", "causal"), ("us4", "us3", "causal"),
    # ---- 场景 s：scenario ----
    ("us1", "as1", "scenario"), ("us1", "as2", "scenario"), ("us2", "us8", "scenario"),
    ("us3", "us4", "scenario"), ("us2", "as4", "scenario"),
    ("us9", "us8", "scenario"), ("us10", "us6", "scenario"),
    # ---- 场景 s：attribute / social ----
    ("us9", "us8", "social"), ("us10", "us5", "social"),

    # ---- 场景 l：temporal ----
    ("ul3", "tl1", "temporal"), ("ul4", "tl2", "temporal"), ("ul5", "tl3", "temporal"),
    ("ul7", "tl4", "temporal"), ("ul6", "tl5", "temporal"),
    # ---- 场景 l：sequence ----
    ("ul1", "ul3", "sequence"), ("ul3", "ul4", "sequence"), ("al2", "al4", "sequence"),
    ("al1", "al2", "sequence"),
    # ---- 场景 l：causal ----
    ("ul8", "al8", "causal"), ("ul2", "al3", "causal"), ("ul3", "ul8", "causal"),
    ("ul6", "al7", "causal"),
    # ---- 场景 l：scenario ----
    ("ul1", "al1", "scenario"), ("ul1", "al3", "scenario"), ("ul1", "al2", "scenario"),
    ("ul6", "al7", "scenario"), ("ul1", "al5", "scenario"), ("ul5", "al6", "scenario"),

    # ---- 场景 m：scenario / causal（无时序锚，测混合记忆为主）----
    ("um1", "am1", "causal"), ("um2", "am2", "scenario"),
    ("um3", "am3", "scenario"), ("um4", "am3", "scenario"),
    ("um6", "am4", "scenario"), ("um1", "um2", "sequence"), ("am3", "am4", "sequence"),

    # ---- 跨期时间轴（SEQUENCE：高中 < 大专 < 工作，承载跨期大叙事）----
    ("th2", "tc1", "sequence"), ("tc4", "tw1", "sequence"),
    ("th4", "tc1", "sequence"), ("tc4", "tw6", "sequence"),
    ("th1", "th2", "sequence"), ("th3", "th4", "sequence"),
    # ---- 跨期因果/causal（人生低谷→回升主线）----
    ("uh8", "uc1", "causal"), ("uc6", "uc10", "causal"), ("uc10", "uw1", "causal"),
    ("uh3", "uh8", "causal"),
]

BIDIRECTIONAL = {"scenario", "attribute", "social"}


def scene_of(nid: str) -> str:
    pref = nid.rstrip("0123456789")
    return SCENE_PREFIX.get(pref[:2], "其他")


def main() -> None:
    errors = []
    ids = [n[0] for n in NODES]
    if len(ids) != len(set(ids)):
        dup = [x for x in ids if ids.count(x) > 1]
        errors.append(f"存在重复节点 id: {sorted(set(dup))}")
    nodes_by_id = {n[0]: n for n in NODES}

    # 主体前缀校验：content 开头必须与主体一致
    for nid, content, ntype in NODES:
        if ntype not in VALID_NODE_TYPE:
            errors.append(f"{nid}: 非法类型 {ntype}")
        pref = nid.rstrip("0123456789")[:2]
        if pref.startswith("u") and not content.startswith("用户"):
            errors.append(f"{nid}: 用户节点 content 应以「用户」开头: {content}")
        elif pref.startswith("a") and not content.startswith("助手"):
            errors.append(f"{nid}: 助手节点 content 应以「助手」开头: {content}")
        elif pref.startswith("t") and ntype != "thing":
            errors.append(f"{nid}: 时间锚点应为 thing 类型，实际 {ntype}: {content}")

    out_nodes = []
    for nid, content, ntype in NODES:
        out_nodes.append({"id": nid, "content": content, "type": ntype,
                          "deprecated": False, "forgotten": False})

    # 边引用 + 类型校验，双向自动补
    out_edges = []
    seen = set()
    for src, tgt, rt in EDGES:
        if rt not in VALID_REL_TYPE:
            errors.append(f"边 {src}->{tgt}: 非法 rel_type {rt}")
        if src not in nodes_by_id:
            errors.append(f"边 {src}->{tgt}: 源节点不存在 {src}")
        if tgt not in nodes_by_id:
            errors.append(f"边 {src}->{tgt}: 目标节点不存在 {tgt}")
        if src == tgt:
            errors.append(f"边 {src}->{tgt}: 自环")
        key = (src, tgt, rt)
        if key in seen:
            errors.append(f"重复边 {src}->{tgt} [{rt}]")
        seen.add(key)
        out_edges.append({"from": src, "to": tgt, "type": rt})
        if rt in BIDIRECTIONAL:
            rev = {"from": tgt, "to": src, "type": rt}
            if (tgt, src, rt) not in seen:
                seen.add((tgt, src, rt))
                out_edges.append(rev)

    if errors:
        print("校验失败：")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    data = {"nodes": out_nodes, "edges": out_edges}
    out_path = ROOT / "data/dba_eval/comprehensive_gold.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    # 统计
    from collections import Counter, defaultdict
    body = Counter()
    scene_cnt = defaultdict(int)
    type_cnt = Counter()
    for nid, content, ntype in NODES:
        pref = nid.rstrip("0123456789")[:2]
        body["用户" if pref.startswith("u") else ("助手" if pref.startswith("a") else "时间锚点")] += 1
        scene_cnt[scene_of(nid)] += 1
        type_cnt[ntype] += 1
    edge_rt = Counter(e["type"] for e in out_edges)

    print(f"gold 已写出: {out_path}")
    print(f"  节点总数={len(NODES)}  边总数(含双向补全)={len(out_edges)}")
    print(f"  主体分布: {dict(body)}")
    print(f"  场景分布: {dict(scene_cnt)}")
    print(f"  节点类型: {dict(type_cnt)}")
    print(f"  边类型: {dict(edge_rt)}")


if __name__ == "__main__":
    main()
