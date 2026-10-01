# SPDX-License-Identifier: AGPL-3.0-only

"""
寻峰终止：基于目的关联度均值的斜率判断

论文第五章：联想过程中的目的关联度先上升后下降，
不是单调的，需要在斜率转负时停止并回到峰值轮。
"""

from typing import List, Sequence


def resolve_peak_tolerance(
    peak_mean: float,
    hop_means: Sequence[float],
    absolute: float = 0.10,
    ratio: float = 0.0,
) -> float:
    """解析本轮生效的**峰值容忍带带宽**（0930 复审新增）。

    背景（为什么需要它）：容忍带判据是"轮均分 μ ≥ peak_mean − δ"。δ 是一个**绝对**常数
    （库默认 0.10），而候选余弦的分布宽度随语料/图规模变化——当分布被压得很窄时
    （实测线上 18 次全在 0.5497~0.5884、全距仅 0.039，而 δ=0.10 是全距的 2.5 倍），
    这个带会吞掉几乎**全部**候选：600 次线上检索里 `峰值/候选` 中位 **0.81**（候选≥300 档 0.79），
    即"寻峰"平均只筛掉 19%。该退化在换 embedding 之前就已存在（前后均为 ≈0.80），
    属**判据尺度与数据尺度不匹配**，与向量模型无关。

    两种模式：
      * `ratio <= 0`（默认）：返回 `absolute` —— **现行行为，逐字不变**。
      * `ratio > 0`：返回 `ratio × (peak_mean − min(hop_means))`，即"只保留落在观测范围
        上半部的轮"。它是**尺度无关**的，因而**自动随场景变**：高同质（分布窄）时带宽自动
        收窄，分散时自动放宽，无需人工按图规模分档。
        各轮均值全等时 spread=0 → δ=0 → 只保留峰值轮（仍非空，不会把结果清空）。

    注：这里的 `hop_means` 是**各跳的轮均分**（`hop_history[*]["mean_score"]`），不是节点余弦
    ——带宽判据作用在轮均分上，故尺度也要取轮均分的尺度。
    """
    if ratio <= 0.0:
        return max(0.0, float(absolute))
    if not hop_means:
        return max(0.0, float(absolute))
    spread = float(peak_mean) - min(float(m) for m in hop_means)
    return max(0.0, float(ratio) * spread)


class PeakFinder:
    """寻峰终止器 + 峰值容忍带

    行为说明（patience=2, min_delta=0.015 为例）：
      1. 得分上升 → 更新 peak_index，继续
      2. 首次显著下降（delta < -min_delta）→ 标记 seen_decline，开始计时
      3. 后续 patience 轮内若得分回升 → 重置计时，继续探索
      4. 若 patience 轮后仍未回升 → 返回 peak_found，输出峰值容忍带
      5. 微小波动（|delta| < min_delta）视为持平，不触发上升/下降判定，
         但在 seen_decline 期间仍计入额外探索轮数
    """

    def __init__(self, patience: int = 2, min_delta: float = 0.015,
                 peak_tolerance: float = 0.10):
        """
        Args:
            patience: 首次下降后允许额外探索的轮数（默认 2 轮）
            min_delta: 最小变化阈值，|Δμ| < min_delta 视为持平
            peak_tolerance: 峰值容忍常数，均值在 [peak-δ, peak] 内的轮都参与输出
        """
        self.patience = patience
        self.min_delta = min_delta
        self.peak_tolerance = peak_tolerance
        self.history: List[float] = []
        self.peak_index: int = 0
        self.seen_decline: bool = False
        self.extra_rounds: int = 0  # 首次下降后已探索的额外轮数

    def add_round(self, mean_score: float) -> str:
        """记录本轮得分，返回 'continue' 或 'peak_found'"""
        self.history.append(mean_score)

        if len(self.history) == 1:
            return "continue"

        prev = self.history[-2]
        curr = self.history[-1]
        delta = curr - prev

        # 显著上升 → 重置下降计时
        if delta > self.min_delta:
            self.peak_index = len(self.history) - 1
            self.seen_decline = False
            self.extra_rounds = 0
            return "continue"

        # 首次显著下降 → 标记并开始计时
        if not self.seen_decline:
            if delta < -self.min_delta:
                self.seen_decline = True
                self.extra_rounds = 0
            return "continue"

        # 已标记下降 → 累计额外探索轮数（含持平轮）
        self.extra_rounds += 1
        if self.extra_rounds >= self.patience:
            return "peak_found"
        return "continue"

    def get_peak_values(self, rounds_data: list) -> list:
        """返回峰值轮对应的数据"""
        return rounds_data[self.peak_index]

    def reset(self):
        """重置状态"""
        self.history = []
        self.peak_index = 0
        self.seen_decline = False
        self.extra_rounds = 0
