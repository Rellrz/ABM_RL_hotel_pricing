#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, field #自动根据类属性生成构造函数__init__
from typing import Any, List, Dict, Tuple, Optional
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'hotel_bookings.csv')

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
    tensorboard_dir: str = os.path.join(PROJECT_ROOT, 'outputs', 'tensorboard_logs')
    
    # 模型保存路径
    abm_q_table_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'abm_q_table_{timestamp}.pkl')
    hotel_agent_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'hotel_agent_{timestamp}.pkl')
    ota_agent_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'ota_agent_{timestamp}.pkl')
    preprocessor_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'preprocessor_{timestamp}.pkl')
    results_path: str = os.path.join(PROJECT_ROOT, 'outputs', 'results', 'results_{timestamp}.pkl')

    def __post_init__(self):
        """创建必要的目录"""
        for path in [self.raw_data_dir, self.processed_data_dir, 
                     self.models_dir, self.results_dir, self.figures_dir]:
            os.makedirs(path, exist_ok=True)

# =================
# ABM配置辅助函数
# =================

# ABM配置辅助函数
def calculate_monthly_arrival_rates(historical_data: pd.DataFrame) -> Dict[int, float]:
    """
    从历史数据计算每月的日均到达率 λ_m
    
    Args:
        historical_data: 历史预订数据
        
    Returns:
        月份 -> 日均到达率的字典
    """
    df = historical_data.copy()
    
    if 'arrival_date_month' not in df.columns:
        return {m: 100.0 for m in range(1, 13)}
    
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    df['month_num'] = df['arrival_date_month'].map(month_map)
    monthly_counts = df.groupby('month_num').size()
    
    monthly_rates = {}
    for month in range(1, 13):
        if month in monthly_counts.index:
            monthly_rates[month] = monthly_counts[month] / 30.0
        else:
            monthly_rates[month] = 100.0
    
    return monthly_rates


def fit_lead_time_distribution(historical_data: pd.DataFrame) -> Dict[str, float]:
    """拟合提前期分布（指数分布，作为兜底）"""
    if 'lead_time' not in historical_data.columns:
        return {'mean': 104.0}

    lead_times = historical_data['lead_time'].dropna()
    lead_times = lead_times[lead_times >= 0]

    if len(lead_times) == 0:
        return {'mean': 104.0}

    return {'mean': float(lead_times.mean())}


def build_empirical_lead_time_distribution(
    historical_data: pd.DataFrame,
    max_lead_time_days: int = 90,
) -> Dict[str, Any]:

    lead_times = historical_data['lead_time'].dropna().astype(int)
    lead_times = lead_times[(lead_times >= 0) & (lead_times <= max_lead_time_days)]


    counts = lead_times.value_counts().sort_index()
    support = list(range(max_lead_time_days + 1))
    total = float(counts.sum())
    probabilities = [float(counts.get(d, 0)) / total for d in support]

    prob_sum = float(sum(probabilities))
    if prob_sum > 0:
        probabilities = [p / prob_sum for p in probabilities]

    return {
        'type': 'empirical',
        'max_days': max_lead_time_days,
        'support': support,
        'probabilities': probabilities,
    }


def fit_wtp_distribution(historical_data: pd.DataFrame) -> Dict[str, float]:
    """
    拟合支付意愿分布（正态分布）
    
    基于未取消订单的ADR（平均日房价）
    
    Args:
        historical_data: 历史预订数据
        
    Returns:
        分布参数字典
    """
    if 'adr' not in historical_data.columns:
        return {'mean': 100.0, 'std': 30.0}
    
    df = historical_data.copy()
    if 'is_canceled' in df.columns:
        df = df[df['is_canceled'] == 0]
    
    adr_values = df['adr'].dropna()
    adr_values = adr_values[(adr_values > 0) & (adr_values < 500)]
    
    if len(adr_values) == 0:
        return {'mean': 100.0, 'std': 30.0}
    
    return {'mean': float(adr_values.mean()), 'std': float(adr_values.std())}


def create_abm_config(data_path:str = None) -> 'ABMConfig':
    """
    创建ABM配置（工厂函数）
    
    Args:
        data_path: 历史数据文件路径，如果提供则从数据计算参数
        
    Returns:
        ABMConfig实例
    """
    if data_path is not None:
        historical_data = pd.read_csv(data_path)
        historical_data = historical_data[historical_data['hotel'] == 'City Hotel'].copy()
        monthly_arrival_rates = calculate_monthly_arrival_rates(historical_data)
        lead_time_params = build_empirical_lead_time_distribution(historical_data, max_lead_time_days=90)
        lead_time_params['mean'] = fit_lead_time_distribution(historical_data)['mean']
        wtp_params = fit_wtp_distribution(historical_data)
    else:
        monthly_arrival_rates = {m: 100.0 for m in range(1, 13)}
        lead_time_params = {'type': 'exponential', 'mean': 104.0}
        wtp_params = {'mean': 100.0, 'std': 30.0}
    
    return ABMConfig(
        monthly_arrival_rates=monthly_arrival_rates,
        lead_time_params=lead_time_params,
        wtp_params=wtp_params
    )


@dataclass
class ABMConfig:
    """ABM客户行为模型配置"""
    
    monthly_arrival_rates: Dict[int, float] = field(default_factory=lambda: {m: 100.0 for m in range(1, 13)})
    lead_time_params: Dict[str, Any] = field(default_factory=lambda: {'type': 'exponential', 'mean': 104.0})
    wtp_params: Dict[str, float] = field(default_factory=lambda: {'mean': 100.0, 'std': 30.0})
    
    urgency_weight: float = 20
    noise_std: float = 12.0
    booking_threshold: float = -15
    customer_type_ratio: Tuple[float, float] = (0.3, 0.7) #(ota_channel, ota and direct channel)
    online_discount_ratio: float = 0.8
    
    regret_coefficient: float = 0.75
    commitment_weight: float = 8.0
    shock_std: float = 15.0
    
    beta_base: float = 1.0
    beta_range: Tuple[float, float] = (0.8, 1.2)

# =================

@dataclass
class RLConfig:
    """强化学习配置
    
    支持两种算法：
    1. Q-learning: 离散动作空间（36个定价组合）
    2. Actor-Critic: 连续动作空间（价格范围80-200元）
    
    状态空间: 库存档位 × 季节 × 日期类型 = 5 × 3 × 2 = 30种状态
    """
    # ========== 算法选择 ==========
    algorithm: str = 'cem'  # 'q_learning' 或 'actor_critic'
    
    # ========== 通用参数 ==========
    n_states: int = 18
    n_actions: int = 144  # Q-learning使用
    discount_factor: float = 0.99
    
    # ========== Q-learning参数 ==========
    learning_rate: float = 0.05  # Q-learning学习率
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300  # 延长探索期，避免过早收敛
    epsilon_min: float = 0.01
    
    # ========== Actor-Critic参数 ==========
    actor_lr: float = 0.002  # Actor学习率（策略更新）- 降低以提高稳定性（0.005）
    critic_lr: float = 0.02  # Critic学习率（价值更新）（0.05）
    action_min: float = 80.0  # 最低价格
    action_max: float = 170.0  # 最高价格
    initial_std: float = 20.0  # 初始探索标准差
    min_std: float = 5.0  # 最小探索标准差 - 极小值减少后期波动
    std_decay: float = 0.999  # 标准差衰减率 - 持续衰减到500轮（0.99）
    
    # ========== 奖励函数参数 ==========
    reward_hotel_ratio: float = 0  # 个体收益权重（α）
    reward_ota_ratio: float = 1
    # ratio = 1 为完全自私
    # ratio = 0 为完全利他
    
    # ========== CEM参数 ==========
    cem_algorithm: str = 'cem'  # 'cem' (表格版) 或 'cem_nn' (神经网络版)
    cem_n_samples: int = 100  # CEM每次采样的动作数量
    cem_elite_frac: float = 0.3  # CEM精英样本比例（top-k）
    
    # ========== CEM-NN参数 ==========
    cem_nn_state_dim: int = 18  # 状态维度
    cem_nn_learning_rate: float = 0.001  # 学习率
    cem_nn_batch_size: int = 32  # 批次大小
    cem_nn_memory_size: int = 1000  # 经验回放大小
    cem_nn_hidden_dims: list = field(default_factory=lambda: [64, 64])  # 隐藏层维度
    cem_nn_min_std: float = 0.02  # 最小标准差（降低以减少补贴波动，0.02/0.8=2.5%相对标准差）
    cem_nn_initial_std: float = 0.1  # 初始标准差（从较小的值开始）
    
    # ========== 博弈系统参数 ==========
    enable_game_mode: bool = False  # 是否启用酒店-OTA博弈模式
    commission_rate: float = 0.30  # OTA佣金率（15%）
    subsidy_ratio_min: float = 0.0  # OTA补贴比例最小值（0%，不补贴）
    subsidy_ratio_max: float = 0.8  # OTA补贴比例最大值（80%，最多补贴佣金的80%）
    online_price_min: float = 80.0  # 线上基础价格最小值（需覆盖佣金）
    online_price_max: float = 180.0  # 线上基础价格最大值
    offline_price_min: float = 80.0  # 线下价格最小值
    offline_price_max: float = 180.0  # 线下价格最大值
    game_training_mode: str = 'simultaneous'  # 训练模式：'fixed_ota', 'alternating', 'simultaneous'
    
    episodes: int = 500  # Actor-Critic需要更多轮次
    online_learning_days: int = 90
    update_frequency: int = 7
    
    enable_online_learning: bool = False
    
    online_price_levels: List[int] = field(default_factory=lambda: [80,85, 90, 95, 100, 105,110, 115, 120, 125, 130, 135])
    offline_price_levels: List[int] = field(default_factory=lambda: [90,95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
    
    agent_paths: Dict[str, str] = field(default_factory=lambda: {
        'pretrained': os.path.join(PROJECT_ROOT, '02_训练模型', 'q_agent_pretrained.pkl'),
        'final': os.path.join(PROJECT_ROOT, '02_训练模型', 'q_agent_final.pkl'),
        'online': os.path.join(PROJECT_ROOT, '02_训练模型', 'q_agent_online.pkl')
    })

@dataclass
class EnvConfig:
    """酒店环境参数，模拟真实的酒店运营环境"""
    
    initial_inventory: int = 200
    max_inventory: int = 200
    min_inventory: int = 0

    booking_window_days: int = 91
    
    online_price_levels: List[int] = field(default_factory=lambda: [80,85, 90, 95, 100, 105,110, 115, 120, 125, 130, 135])
    offline_price_levels: List[int] = field(default_factory=lambda: [90,95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
    
    n_actions: int = 144
    
    demand_weight: float = 0.7
    inventory_weight: float = 0.3
    revenue_weight: float = 1.0
    booking_weight: float = 0.5


@dataclass
class SimulationConfig:
    """系统模拟和评估参数"""
    
    default_days: int = 90
    default_start_date: str = '2017-01-01'
    evaluation_episodes: int = 10
    results_path: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, '04_结果输出', 'simulation_results'))

@dataclass
class RandomConfig:
    """控制系统中的随机性，支持固定模式和随机模式"""
    
    random_mode: str = 'random'
    fixed_seed: int = 42
    description: str = '随机因子控制配置 - 支持固定和随机两种模式'


@dataclass
class SystemConfig:
    """系统级配置参数，控制硬件使用和全局行为"""
    
    use_cuda: bool = False
    device: str = 'cpu'
    random_seed: int = 42
    max_workers: int = 28
    memory_limit_gb: int = 24
    enable_gpu_memory_growth: bool = True
    mixed_precision: bool = False
    compile_models: bool = False
    profile_performance: bool = False


@dataclass
class LogConfig:
    """系统日志和输出配置"""
    
    log_level: str = 'INFO'
    log_file: str = field(default_factory=lambda: os.path.join(PROJECT_ROOT, '06_临时文件', 'hotel_pricing.log'))
    console_output: bool = True
    save_intermediate_results: bool = True


PATH_CONFIG = PathConfig()
ABM_CONFIG = create_abm_config(DATA_PATH)
RL_CONFIG = RLConfig()
ENV_CONFIG = EnvConfig()
SIMULATION_CONFIG = SimulationConfig()
RANDOM_CONFIG = RandomConfig()
SYSTEM_CONFIG = SystemConfig()
LOG_CONFIG = LogConfig()


def validate_config() -> bool:
    """
    验证配置有效性
    
    检查配置文件中的各项参数是否有效，包括路径存在性、参数范围、逻辑一致性等。
    提供详细的错误信息帮助定位和修复配置问题。
    
    Returns:
        bool: 配置有效返回True，无效返回False
        
    验证项目：
    - 数据文件存在性检查
    - BNN参数范围验证（输入维度、隐藏层维度等）
    - RL参数逻辑验证（学习率、折扣因子等）
    - 路径格式和权限检查
    - 数值参数范围检查
    
    Note:
        - 打印详细的错误信息便于调试
        - 检查关键路径的存在性和可访问性
        - 验证数值参数的合理范围
        - 提供配置修复建议
    """
    import os
    import typing
    
    # 数据文件检查已移除（DATA_CONFIG不存在）
    
    # BNN配置已移除，跳过相关检查
    
    # 检查RL配置
    epsilon_start = RL_CONFIG.epsilon_start
    epsilon_end = RL_CONFIG.epsilon_end
    discount_factor = RL_CONFIG.discount_factor
    
    if epsilon_start < 0 or epsilon_start > 1:
        print("错误：epsilon_start必须在0和1之间")
        return False
    
    if epsilon_end < 0 or epsilon_end > 1:
        print("错误：epsilon_end必须在0和1之间")
        return False
    
    if discount_factor < 0 or discount_factor > 1:
        print("错误：折扣因子必须在0和1之间")
        return False
    
    # 检查环境配置
    initial_inventory = ENV_CONFIG.initial_inventory
    online_price_levels = ENV_CONFIG.online_price_levels
    offline_price_levels = ENV_CONFIG.offline_price_levels
    n_actions = ENV_CONFIG.n_actions
    
    if initial_inventory <= 0:
        print("错误：初始库存必须大于0")
        return False
    
    if len(online_price_levels) <= 0 or len(offline_price_levels) <= 0:
        print("错误：线上和线下价格档位都必须大于0个")
        return False
    
    expected_actions = len(online_price_levels) * len(offline_price_levels)
    if n_actions != expected_actions:
        print(f"错误：动作数量必须为{expected_actions}（线上{len(online_price_levels)}×线下{len(offline_price_levels)}组合）")
        return False
    
    if RL_CONFIG.n_actions != n_actions:
        print(f"错误：RL_CONFIG.n_actions({RL_CONFIG.n_actions}) 必须与 ENV_CONFIG.n_actions({n_actions}) 一致")
        return False
    
    return True

# 初始化配置
# 获取项目根目录，用于构建所有相对路径的基准
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not validate_config():
    print("配置验证失败，请检查配置文件")
