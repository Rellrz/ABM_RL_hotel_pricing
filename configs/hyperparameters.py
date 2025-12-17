#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数配置文件 - 双智能体酒店动态定价系统
Hotel Dynamic Pricing System with Dual-Agent RL and ABM

本配置文件包含系统所有超参数，分为以下模块：
1. 项目路径配置
2. ABM客户行为模型参数
3. 双智能体RL参数（酒店 + OTA）
4. 环境配置（库存、定价）
5. 训练配置
6. 数据处理配置
7. 日志和可视化配置
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# =============================================================================
# 项目路径配置
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class PathConfig:
    """路径配置"""
    # 数据路径
    raw_data_dir: str = os.path.join(PROJECT_ROOT, 'data', 'raw')
    processed_data_dir: str = os.path.join(PROJECT_ROOT, 'data', 'processed')
    hotel_bookings_csv: str = os.path.join(PROJECT_ROOT, 'data', 'raw', 'hotel_bookings.csv')
    
    # 输出路径
    models_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'models')
    results_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'results')
    figures_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
    
    # 模型保存路径
    hotel_agent_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'hotel_agent.pkl')
    ota_agent_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'ota_agent.pkl')
    preprocessor_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'preprocessor.pkl')
    
    def __post_init__(self):
        """创建必要的目录"""
        for path in [self.raw_data_dir, self.processed_data_dir, 
                     self.models_dir, self.results_dir, self.figures_dir]:
            os.makedirs(path, exist_ok=True)


# =============================================================================
# ABM 客户行为模型参数
# =============================================================================
@dataclass
class ABMConfig:
    """ABM客户行为模型配置"""
    
    # 客户类型比例
    type_a_ratio: float = 0.2  # Type A (传统/忠诚型) 占比
    type_b_ratio: float = 0.8  # Type B (数字/价格敏感型) 占比
    
    # 客户生成参数
    lead_time_mu: float = 3.5  # 提前期对数正态分布参数 μ
    lead_time_sigma: float = 1.2  # 提前期对数正态分布参数 σ
    lead_time_min: int = 0  # 最小提前期（天）
    lead_time_max: int = 180  # 最大提前期（天）
    
    # WTP (支付意愿) 参数
    wtp_type_a_multiplier: float = 1.1  # Type A 的 WTP 倍数（相对于月均价）
    wtp_type_b_multiplier: float = 0.95  # Type B 的 WTP 倍数
    wtp_min: float = 10.0  # 最低 WTP（美元）
    
    # 价格敏感度参数
    beta_base: float = 1.0  # 基础价格敏感度
    beta_min: float = 0.8  # 最小敏感度
    beta_max: float = 1.2  # 最大敏感度
    
    # 决策效用函数参数
    urgency_weight: float = 7.5  # γ: 紧迫权重
    noise_std: float = 12.0  # σ_noise: 决策噪声标准差
    booking_threshold: float = 0.0  # U_threshold: 预订阈值
    
    # 佣金驱动的流量加权参数
    commission_tiers: List[float] = field(default_factory=lambda: [0.10, 0.15, 0.20, 0.25])  # 4档佣金比例
    traffic_boost: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # Tier 0 (10%): 基准流量
        1: 1.1,   # Tier 1 (15%): +10%
        2: 1.2,   # Tier 2 (20%): +20%
        3: 1.3    # Tier 3 (25%): +30%
    })


# =============================================================================
# 强化学习参数 - 酒店智能体
# =============================================================================
@dataclass
class HotelAgentConfig:
    """酒店智能体配置"""
    
    # 状态空间维度 (D, I, C, W, S) = 15 × 4 × 3 × 2 × 3 = 1080
    n_days_ahead: int = 15  # 提前期维度：0-14天
    n_inv_levels: int = 4  # 库存压力档位：空闲/正常/紧张/告急
    n_comp_status: int = 3  # OTA竞争力：劣势/均势/优势
    n_weekend: int = 2  # 周末标识：工作日/周末
    n_season: int = 3  # 季节：淡季/平季/旺季
    
    # 状态离散化阈值
    inv_thresholds: List[float] = field(default_factory=lambda: [0.30, 0.60, 0.90])  # 库存使用率分界点
    comp_thresholds: List[float] = field(default_factory=lambda: [0.95, 1.05])  # 价格比分界点
    
    # 动作空间：5个折扣档位 × 4个佣金档位 = 20个动作
    discount_levels: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90, 0.95, 1.00])  # 直销折扣系数
    commission_tiers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # 佣金档位索引
    
    # Q-learning 参数
    learning_rate: float = 0.05  # α: 学习率（降低以提高稳定性，配合奖励归一化）
    discount_factor: float = 0.95  # γ: 折扣因子（用于时间折扣奖励分配）
    epsilon_start: float = 0.9  # 初始探索率
    epsilon_end: float = 0.1  # 最终探索率
    epsilon_decay_episodes: int = 300  # 探索率衰减轮数
    
    # UCB 探索参数
    ucb_c: float = 2.0  # UCB 探索系数
    
    @property
    def n_states(self) -> int:
        """计算总状态数"""
        return self.n_days_ahead * self.n_inv_levels * self.n_comp_status * self.n_weekend * self.n_season
    
    @property
    def n_actions(self) -> int:
        """计算总动作数"""
        return len(self.discount_levels) * len(self.commission_tiers)


# =============================================================================
# 强化学习参数 - OTA智能体
# =============================================================================
@dataclass
class OTAAgentConfig:
    """OTA智能体配置"""
    
    # 状态空间维度 (D, I, M, W, S) = 15 × 4 × 3 × 2 × 3 = 1080
    n_days_ahead: int = 15  # 提前期维度：0-14天
    n_inv_levels: int = 4  # 库存压力档位
    n_margin_room: int = 3  # 套利空间：微利/正常/暴利
    n_weekend: int = 2  # 周末标识
    n_season: int = 3  # 季节
    
    # 状态离散化阈值
    inv_thresholds: List[float] = field(default_factory=lambda: [0.30, 0.60, 0.90])
    margin_thresholds: List[float] = field(default_factory=lambda: [1.1, 1.3])  # 套利空间分界点
    
    # 动作空间：5个补贴档位
    subsidy_levels: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8])  # 补贴系数
    
    # Q-learning 参数
    learning_rate: float = 0.01  # α: 学习率（降低以提高稳定性，配合奖励归一化）
    discount_factor: float = 0.95  # γ: 折扣因子（用于时间折扣奖励分配）
    epsilon_start: float = 0.9
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 250
    
    # UCB 探索参数
    ucb_c: float = 2.0
    
    @property
    def n_states(self) -> int:
        """计算总状态数"""
        return self.n_days_ahead * self.n_inv_levels * self.n_margin_room * self.n_weekend * self.n_season
    
    @property
    def n_actions(self) -> int:
        """计算总动作数"""
        return len(self.subsidy_levels)


# =============================================================================
# 环境配置
# =============================================================================
@dataclass
class EnvironmentConfig:
    """环境配置"""
    
    # 库存配置
    total_rooms: int = 100  # 酒店总房间数
    inventory_horizon: int = 180  # 库存管理时间窗口（天）
    decision_horizon: int = 15  # 决策时间窗口（天，0-14）
    
    # 季节划分（月份）
    low_season_months: List[int] = field(default_factory=lambda: [1, 2, 11, 12])  # 淡季
    mid_season_months: List[int] = field(default_factory=lambda: [3, 4, 5, 6])  # 平季
    high_season_months: List[int] = field(default_factory=lambda: [7, 8, 9, 10])  # 旺季
    
    # 周末定义
    weekend_days: List[int] = field(default_factory=lambda: [4, 5, 6])  # 周五、周六、周日 (0=周一)
    
    def get_season(self, month: int) -> int:
        """获取季节索引 (0=淡季, 1=平季, 2=旺季)"""
        if month in self.low_season_months:
            return 0
        elif month in self.mid_season_months:
            return 1
        else:
            return 2
    
    def is_weekend(self, weekday: int) -> bool:
        """判断是否为周末"""
        return weekday in self.weekend_days


# =============================================================================
# 训练配置
# =============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 训练轮数
    n_episodes: int = 100  # 总训练轮数
    episode_length: int = 365  # 每轮天数（1年）
    
    # 评估配置
    eval_frequency: int = 10  # 每N轮评估一次
    eval_episodes: int = 5  # 评估轮数
    
    # 保存配置
    save_frequency: int = 20  # 每N轮保存一次模型
    
    # 随机种子
    random_seed: int = 42
    
    # 训练模式
    simultaneous_training: bool = True  # 是否同时训练双智能体
    hotel_first: bool = True  # 顺序博弈时酒店是否先决策


# =============================================================================
# 数据处理配置
# =============================================================================
@dataclass
class DataConfig:
    """数据处理配置"""
    
    # 价格分位数计算
    p_base_quantile: float = 0.70  # P_base 使用70%分位数
    p_long_quantile: float = 0.50  # P_long 使用中位数
    
    # 数据过滤
    hotel_type: str = 'City Hotel'  # 只使用城市酒店数据
    filter_canceled: bool = True  # 计算价格时是否过滤取消订单
    
    # ADR 异常值过滤
    adr_min: float = 0.0  # 最小ADR
    adr_max: float = 500.0  # 最大ADR
    
    # 到达率计算
    use_actual_demand: bool = True  # 是否使用实际需求（未取消）计算λ_base


# =============================================================================
# 日志和可视化配置
# =============================================================================
@dataclass
class LoggingConfig:
    """日志和可视化配置"""
    
    # 日志级别
    log_level: str = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    
    # 日志文件
    log_to_file: bool = True
    log_file: str = os.path.join(PROJECT_ROOT, 'outputs', 'training.log')
    
    # 进度条
    use_tqdm: bool = True
    
    # 可视化
    plot_training_curves: bool = True
    plot_q_tables: bool = True
    plot_policy: bool = True
    save_plots: bool = True


# =============================================================================
# 全局配置实例
# =============================================================================
# 创建配置实例供全局使用
PATH_CONFIG = PathConfig()
ABM_CONFIG = ABMConfig()
HOTEL_AGENT_CONFIG = HotelAgentConfig()
OTA_AGENT_CONFIG = OTAAgentConfig()
ENV_CONFIG = EnvironmentConfig()
TRAINING_CONFIG = TrainingConfig()
DATA_CONFIG = DataConfig()
LOGGING_CONFIG = LoggingConfig()


# =============================================================================
# 配置验证函数
# =============================================================================
def validate_config():
    """验证配置的一致性和合理性"""
    errors = []
    
    # 验证客户类型比例
    if abs(ABM_CONFIG.type_a_ratio + ABM_CONFIG.type_b_ratio - 1.0) > 1e-6:
        errors.append(f"客户类型比例之和必须为1.0，当前为 {ABM_CONFIG.type_a_ratio + ABM_CONFIG.type_b_ratio}")
    
    # 验证佣金档位数量一致性
    if len(ABM_CONFIG.commission_tiers) != len(HOTEL_AGENT_CONFIG.commission_tiers):
        errors.append("ABM和酒店智能体的佣金档位数量不一致")
    
    # 验证决策窗口
    if ENV_CONFIG.decision_horizon > ENV_CONFIG.inventory_horizon:
        errors.append(f"决策窗口({ENV_CONFIG.decision_horizon})不能大于库存窗口({ENV_CONFIG.inventory_horizon})")
    
    # 验证状态空间大小
    expected_hotel_states = HOTEL_AGENT_CONFIG.n_states
    expected_ota_states = OTA_AGENT_CONFIG.n_states
    print(f"✓ 酒店智能体状态空间: {expected_hotel_states}")
    print(f"✓ OTA智能体状态空间: {expected_ota_states}")
    print(f"✓ 酒店智能体动作空间: {HOTEL_AGENT_CONFIG.n_actions}")
    print(f"✓ OTA智能体动作空间: {OTA_AGENT_CONFIG.n_actions}")
    
    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(errors))
    
    print("✓ 配置验证通过")


# =============================================================================
# 配置导出函数
# =============================================================================
def get_config_dict() -> Dict:
    """导出所有配置为字典格式"""
    return {
        'path': PATH_CONFIG.__dict__,
        'abm': ABM_CONFIG.__dict__,
        'hotel_agent': HOTEL_AGENT_CONFIG.__dict__,
        'ota_agent': OTA_AGENT_CONFIG.__dict__,
        'environment': ENV_CONFIG.__dict__,
        'training': TRAINING_CONFIG.__dict__,
        'data': DATA_CONFIG.__dict__,
        'logging': LOGGING_CONFIG.__dict__,
    }


if __name__ == '__main__':
    """测试配置"""
    print("="*60)
    print("双智能体酒店动态定价系统 - 配置验证")
    print("="*60)
    validate_config()
    print("\n配置摘要:")
    print(f"- 训练轮数: {TRAINING_CONFIG.n_episodes}")
    print(f"- 每轮天数: {TRAINING_CONFIG.episode_length}")
    print(f"- 库存窗口: {ENV_CONFIG.inventory_horizon}天")
    print(f"- 决策窗口: {ENV_CONFIG.decision_horizon}天")
    print(f"- 总房间数: {ENV_CONFIG.total_rooms}")
    print("="*60)
