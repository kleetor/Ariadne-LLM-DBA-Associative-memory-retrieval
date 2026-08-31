# SPDX-License-Identifier: AGPL-3.0-only

"""
路径固化追踪器（PathTracker）— 双因子动态权重

核心机制：
  路径综合权重 = base_weight
               × min(2.0, 1.0 + 0.05 × lifetime_activations)   // 跨会话累积增强
               × max(0.3, 1.0 - 0.15 × same_session_count)     // 同会话内饱和抑制

  session 切换 → same_session_count 重置为 0
              → satiation factor 恢复到 1.0
              → lifetime_bonus 保持不变

设计原则：
  - 不修改 jump_axis.py 的静态权重矩阵
  - 动态权重作为乘数叠加在静态权重之上
  - 路径由 (节点序列, 边类型序列) 定义，精确匹配才算同一路径
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PathTrackerConfig:
    """双因子参数配置"""
    lifetime_coef: float = 0.05       # 每次跨会话激活的终身增强系数
    lifetime_cap: float = 2.0         # 终身增强上限
    session_satiation_coef: float = 0.15  # 同会话每次重复的饱和系数
    session_satiation_floor: float = 0.3  # 饱和因子下限
    session_boundary_hops: int = 3    # 用 hop 数模拟 session 边界（简化）


class PathTracker:
    """路径激活计数器 + 双因子动态权重计算

    用法：
        tracker = PathTracker()

        # Session 1
        tracker.start_session()
        weight1 = tracker.get_dynamic_weight(path_key, base_weight)

        # 切换到 Session 2
        tracker.start_session()
        weight2 = tracker.get_dynamic_weight(path_key, base_weight)  # satiation 重置
    """

    def __init__(self, config: PathTrackerConfig = None):
        self.config = config or PathTrackerConfig()

        # 路径 → 生命周期总激活次数（跨所有 session）
        self.lifetime_counts: Dict[str, int] = defaultdict(int)

        # 路径 → 当前 session 内激活次数
        self.session_counts: Dict[str, int] = defaultdict(int)

        # session 计数器
        self.session_id: int = 0

        # 历史记录（用于曲线分析）
        self.history: List[Dict] = []

    # ── Session 管理 ──

    def start_session(self, session_id: int = None):
        """开始新 session，重置 session 计数器"""
        if session_id is not None:
            self.session_id = session_id
        else:
            self.session_id += 1

        # 重置同 session 内的激活计数
        self.session_counts.clear()

    def reset(self):
        """完全重置所有状态"""
        self.lifetime_counts.clear()
        self.session_counts.clear()
        self.session_id = 0
        self.history.clear()

    # ── 路径构造 ──

    @staticmethod
    def make_path_key(
        src_node: str,
        dst_node: str,
        edge_type: str,
        hop: int = 0,
    ) -> str:
        """构造路径键。

        用于精确匹配：同一 (src, dst, edge_type) 组合算同一路径。
        hop 参数可区分不同跳数的同一类路径。
        """
        return f"{src_node}→{dst_node}@{edge_type}"

    @staticmethod
    def make_extended_path_key(
        node_seq: Tuple[str, ...],
        edge_seq: Tuple[str, ...],
    ) -> str:
        """构造扩展路径键（完整序列匹配）"""
        nodes = "→".join(node_seq)
        edges = "→".join(edge_seq)
        return f"[{nodes}]|[{edges}]"

    # ── 激活计数 ──

    def record_activation(self, path_key: str):
        """记录一条路径被激活"""
        self.lifetime_counts[path_key] += 1
        self.session_counts[path_key] += 1

        # 记录历史
        self.history.append({
            "session": self.session_id,
            "path": path_key,
            "lifetime": self.lifetime_counts[path_key],
            "session_count": self.session_counts[path_key],
            "lifetime_bonus": self._calc_lifetime_bonus(path_key),
            "session_satiation": self._calc_session_satiation(path_key),
            "combined_weight": self._calc_lifetime_bonus(path_key) * self._calc_session_satiation(path_key),
        })

    def record_batch(self, path_keys: List[str]):
        """批量记录多条路径激活（一次检索中的全部扩展）"""
        for pk in path_keys:
            self.record_activation(pk)

    # ── 双因子计算 ──

    def _calc_lifetime_bonus(self, path_key: str) -> float:
        """终身累积增强因子"""
        n = self.lifetime_counts[path_key]
        bonus = 1.0 + self.config.lifetime_coef * n
        return min(self.config.lifetime_cap, bonus)

    def _calc_session_satiation(self, path_key: str) -> float:
        """同 session 内饱和抑制因子"""
        n = self.session_counts[path_key]
        satiation = 1.0 - self.config.session_satiation_coef * n
        return max(self.config.session_satiation_floor, satiation)

    def get_dynamic_weight(self, path_key: str, base_weight: float = 1.0) -> float:
        """获取叠加了双因子的动态权重"""
        lifetime = self._calc_lifetime_bonus(path_key)
        satiation = self._calc_session_satiation(path_key)
        return base_weight * lifetime * satiation

    # ── 批量权重计算（用于注入 jump_axis） ──

    def get_edge_weight_multiplier(
        self,
        src_node: str,
        dst_node: str,
        edge_type: str,
    ) -> float:
        """获取某条边的动态权重乘数

        用于叠加到 jump_axis 的静态权重之上：
            effective_weight = static_weight * multiplier
        """
        pk = self.make_path_key(src_node, dst_node, edge_type)

        # 检查是否有激活记录
        lifetime = self.lifetime_counts.get(pk, 0)
        session = self.session_counts.get(pk, 0)

        if lifetime == 0:
            return 1.0  # 从未激活过，不影响

        bonus = min(self.config.lifetime_cap, 1.0 + self.config.lifetime_coef * lifetime)
        satiation = max(self.config.session_satiation_floor,
                        1.0 - self.config.session_satiation_coef * session)

        return bonus * satiation

    # ── 查询接口 ──

    def get_path_stats(self, path_key: str) -> Dict:
        """获取某条路径的完整统计"""
        return {
            "lifetime_count": self.lifetime_counts.get(path_key, 0),
            "session_count": self.session_counts.get(path_key, 0),
            "lifetime_bonus": self._calc_lifetime_bonus(path_key),
            "session_satiation": self._calc_session_satiation(path_key),
            "combined_weight": self._calc_lifetime_bonus(path_key) * self._calc_session_satiation(path_key),
        }

    def get_session_summary(self) -> Dict:
        """获取当前 session 的摘要统计"""
        if not self.session_counts:
            return {"total_paths": 0, "max_satiation": 1.0, "paths_at_floor": 0}

        counts = list(self.session_counts.values())
        sats = [self._calc_session_satiation(pk) for pk in self.session_counts]

        return {
            "session_id": self.session_id,
            "total_paths": len(self.session_counts),
            "total_activations": sum(counts),
            "max_repetition": max(counts),
            "min_satiation": min(sats),
            "paths_at_floor": sum(1 for s in sats if s <= self.config.session_satiation_floor + 0.01),
        }

    def get_lifetime_summary(self) -> Dict:
        """获取生命周期摘要统计"""
        if not self.lifetime_counts:
            return {"total_paths": 0, "max_bonus": 1.0}

        counts = list(self.lifetime_counts.values())
        bonuses = [self._calc_lifetime_bonus(pk) for pk in self.lifetime_counts]

        return {
            "total_unique_paths": len(self.lifetime_counts),
            "total_activations": sum(counts),
            "max_activations": max(counts),
            "max_bonus": max(bonuses),
            "paths_at_cap": sum(1 for b in bonuses if b >= self.config.lifetime_cap - 0.01),
            "mean_bonus": sum(bonuses) / len(bonuses),
        }

    def export_history(self) -> List[Dict]:
        """导出完整激活历史（用于绘制四条曲线）"""
        return list(self.history)
