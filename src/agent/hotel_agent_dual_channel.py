"""
双渠道酒店Agent模块

酒店代理，负责分别决策线上基础价格和线下价格
使用两个独立的CEM算法进行决策

业务逻辑：
- 线上价格：给OTA的基础价格（需覆盖佣金成本）
- 线下价格：直销渠道价格
- 考虑OTA补贴行为预测
- 平衡线上线下渠道收益
"""

import numpy as np
from collections import defaultdict, deque
from typing import Union, List, Dict, Any, Tuple
from src.algorithms.cem import CrossEntropyMethod
from src.algorithms.cem_nn import NeuralCrossEntropyMethod
from configs.config import RL_CONFIG


class HotelAgentDualChannel:
    """
    双渠道酒店Agent：使用两个独立的CEM分别决策线上和线下价格
    
    状态空间：
    - 库存水平
    - 季节
    - 是否周末
    - 提前预订天数
    
    动作空间：
    - price_online_base ∈ [90, 180]（线上基础价格，需覆盖佣金）
    - price_offline ∈ [80, 170]（线下价格）
    
    收益函数：
    - revenue = revenue_online + revenue_offline
    - revenue_online = bookings_online * price_online_base * (1 - commission_rate)
    - revenue_offline = bookings_offline * price_offline
    """
    
    def __init__(self,
                 n_states: int = 18,
                 commission_rate: float = 0.15,
                 online_price_min: float = 90.0,
                 online_price_max: float = 180.0,
                 offline_price_min: float = 80.0,
                 offline_price_max: float = 170.0,
                 n_samples: int = 20,
                 elite_frac: float = 0.2,
                 initial_std: float = 20.0,
                 min_std: float = 2.0,
                 std_decay: float = 0.99):
        """
        初始化双渠道酒店Agent
        
        Args:
            n_states: 状态空间大小
            commission_rate: OTA佣金率
            online_price_min: 线上基础价格最小值
            online_price_max: 线上基础价格最大值
            offline_price_min: 线下价格最小值
            offline_price_max: 线下价格最大值
            n_samples: CEM采样数量
            elite_frac: 精英样本比例
            initial_std: 初始标准差
            min_std: 最小标准差
            std_decay: 标准差衰减率
        """
        self.n_states = n_states
        self.n_base_states = 18
        self.n_stages = max(1, int(n_states) // self.n_base_states)
        self.commission_rate = commission_rate
        self.online_price_min = online_price_min
        self.online_price_max = online_price_max
        self.offline_price_min = offline_price_min
        self.offline_price_max = offline_price_max
        self.algorithm_type = RL_CONFIG.cem_algorithm
        
        # 根据配置选择算法
        if self.algorithm_type == 'cem_nn':
            # 使用神经网络版CEM
            self.cem_online = NeuralCrossEntropyMethod(
                state_dim=self.n_states,
                action_dim=1,
                action_min=online_price_min,
                action_max=online_price_max,
                discount_factor=0.99,
                n_samples=n_samples,
                elite_frac=elite_frac,
                learning_rate=RL_CONFIG.cem_nn_learning_rate,
                hidden_dims=RL_CONFIG.cem_nn_hidden_dims,
                batch_size=RL_CONFIG.cem_nn_batch_size,
                memory_size=RL_CONFIG.cem_nn_memory_size,
                min_std=min_std
            )
            
            self.cem_offline = NeuralCrossEntropyMethod(
                state_dim=self.n_states,
                action_dim=1,
                action_min=offline_price_min,
                action_max=offline_price_max,
                discount_factor=0.99,
                n_samples=n_samples,
                elite_frac=elite_frac,
                learning_rate=RL_CONFIG.cem_nn_learning_rate,
                hidden_dims=RL_CONFIG.cem_nn_hidden_dims,
                batch_size=RL_CONFIG.cem_nn_batch_size,
                memory_size=RL_CONFIG.cem_nn_memory_size,
                min_std=min_std
            )
        else:
            # 使用传统表格版CEM
            self.cem_online = CrossEntropyMethod(
                n_states=n_states,
                action_min=online_price_min,
                action_max=online_price_max,
                discount_factor=0.99,
                n_samples=n_samples,
                elite_frac=elite_frac,
                initial_std=initial_std,
                min_std=min_std,
                std_decay=std_decay
            )
            
            self.cem_offline = CrossEntropyMethod(
                n_states=n_states,
                action_min=offline_price_min,
                action_max=offline_price_max,
                discount_factor=0.99,
                n_samples=n_samples,
                elite_frac=elite_frac,
                initial_std=initial_std,
                min_std=min_std,
                std_decay=std_decay
            )
        
        # OTA补贴历史（用于预测OTA行为）
        self.ota_subsidy_history = deque(maxlen=30)
        
        # 统计信息
        self.total_revenue = 0.0
        self.total_revenue_online = 0.0
        self.total_revenue_offline = 0.0
        self.episode_count = 0
        
    def discretize_state(self, state: Dict, season: int = None, weekday: bool = None) -> int:
        """
        离散化状态
        
        状态空间：库存档位(3) × 季节(3) × 日期类型(2) = 18种状态
        
        关键修正：
        - 直接使用环境返回的inventory_level（0-2），不重新计算
        - 正确使用weekday维度，避免状态空间塌缩
        
        Args:
            state: 状态字典（必须包含inventory_level, season, weekday）
            season: 季节（0-2），如果提供则覆盖state中的值
            weekday: 是否周末（0-1），如果提供则覆盖state中的值
            
        Returns:
            state_idx: 离散化的状态索引（0-17）
        """
        if isinstance(state, (int, np.integer)):
            return int(state)
        
        # 提取状态特征
        if isinstance(state, dict):
            # 直接使用环境已经离散化好的inventory_level（0-2）
            inventory_level = state.get('inventory_level', 2)
            season = state.get('season', 0) if season is None else season
            weekday = state.get('weekday', 0) if weekday is None else weekday
        else:
            inventory_level = 2
            season = season if season is not None else 0
            weekday = weekday if weekday is not None else 0
        
        # 库存档位（0-2）：环境已离散化
        # 0: ≤33间, 1: 34-66间, 2: ≥67间
        inventory_level = int(inventory_level)
        # 季节（0-2）：淡季、平季、旺季
        season = int(season)
        # 日期类型（0-1）：工作日、周末
        weekday = int(weekday)
        
        stage_id = 0
        if isinstance(state, dict):
            stage_id = int(state.get('stage_id', 0))

        stage_id = int(np.clip(stage_id, 0, self.n_stages - 1))

        base_state_idx = inventory_level * 6 + season * 2 + weekday
        state_idx = base_state_idx * self.n_stages + stage_id

        return int(state_idx)
    
    def select_action(self, state: Union[Dict, int], deterministic: bool = False) -> np.ndarray:
        """
        分别决策线上基础价格和线下价格
        
        策略考虑：
        1. 线上基础价格要能覆盖佣金成本
        2. 预测OTA补贴行为，确保最终线上价格有竞争力
        3. 平衡线上线下渠道
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            [price_online_base, price_offline]
        """
        # 离散化状态
        if isinstance(state, dict):
            state_idx = self.discretize_state(state)
        else:
            state_idx = state
        
        # 分别选择动作
        price_online_base = self.cem_online.select_action(state_idx, deterministic)
        price_offline = self.cem_offline.select_action(state_idx, deterministic)
        
        # 约束1：线上基础价格要能覆盖佣金成本
        # 如果线上基础价格*(1-佣金率) < 线下价格，说明线上不划算
        #min_online_base = price_offline / (1 - self.commission_rate)
        #if price_online_base < min_online_base:
        #    price_online_base = min_online_base
        
        # 约束2：考虑OTA补贴预测
        # 预测OTA可能的补贴
        #if len(self.ota_subsidy_history) > 0:
        #    expected_subsidy = np.mean(self.ota_subsidy_history)
        #else:
        #    expected_subsidy = 5.0  # 默认预期补贴
        
        # 计算预期最终线上价格
        #expected_online_final = price_online_base - expected_subsidy
        
        # 如果预期最终线上价格还是比线下贵太多，调整基础价格
        #if expected_online_final > price_offline * 1.3:
            # 线上即使补贴后仍比线下贵30%以上，降低基础价格
            #price_online_base = price_offline * 1.3 + expected_subsidy
        
        # 约束3：确保价格在有效范围内
        price_online_base = np.clip(price_online_base, self.online_price_min, self.online_price_max)
        price_offline = np.clip(price_offline, self.offline_price_min, self.offline_price_max)
        
        return np.array([price_online_base, price_offline])
    
    def update(self, 
              state: Union[Dict, int], 
              action: np.ndarray, 
              reward: float, 
              next_state: Union[Dict, int], 
              done: bool,
              ota_subsidy: float = 0.0) -> None:
        """
        分别更新两个CEM
        
        Args:
            state: 当前状态
            action: [price_online_base, price_offline]
            reward: 酒店总收益（已扣除佣金）
            next_state: 下一状态
            done: 是否结束
            ota_subsidy: OTA实际补贴金额（用于学习预测）
        """
        price_online_base, price_offline = action
        
        # 更新OTA补贴历史
        self.ota_subsidy_history.append(ota_subsidy)
        
        # 离散化状态
        if isinstance(state, dict):
            state_idx = self.discretize_state(state)
        else:
            state_idx = state
            
        if isinstance(next_state, dict):
            next_state_idx = self.discretize_state(next_state)
        else:
            next_state_idx = next_state
        
        # 两个CEM使用相同的总收益（简化方案）
        # 更高级的方案可以根据渠道收益分配权重
        self.cem_online.update(state_idx, price_online_base, reward, next_state_idx, done)
        self.cem_offline.update(state_idx, price_offline, reward, next_state_idx, done)
        
        # 更新统计
        self.total_revenue += reward
    
    def end_episode(self) -> None:
        """结束episode，更新分布参数"""
        self.cem_online.end_episode()
        self.cem_offline.end_episode()
        self.episode_count += 1
    
    def calculate_revenue(self, 
                         bookings_online: int, 
                         bookings_offline: int,
                         price_online_base: float, 
                         price_offline: float) -> float:
        """
        计算酒店收益（扣除佣金）
        
        Args:
            bookings_online: 线上预订量
            bookings_offline: 线下预订量
            price_online_base: 线上基础价格
            price_offline: 线下价格
            
        Returns:
            total_revenue: 酒店总收益
        """
        # 线上收益（扣除佣金）
        revenue_online = bookings_online * price_online_base * (1 - self.commission_rate)
        
        # 线下收益（无佣金）
        revenue_offline = bookings_offline * price_offline
        
        # 总收益
        total_revenue = revenue_online + revenue_offline
        
        # 更新统计
        self.total_revenue_online += revenue_online
        self.total_revenue_offline += revenue_offline
        
        return total_revenue
    
    def get_epsilon(self, episode: int = 0) -> float:
        """
        获取探索率（CEM使用标准差，这里返回平均标准差作为参考）
        
        Args:
            episode: 当前episode
            
        Returns:
            探索率（标准差的归一化值）
        """
        if self.algorithm_type == 'cem_nn':
            # CEM-NN: 使用episode进行衰减估计
            initial_explore = 0.5
            min_explore = 0.1
            decay_rate = 0.995
            exploration_rate = max(min_explore, initial_explore * (decay_rate ** episode))
            return exploration_rate
        else:
            # 传统CEM: 使用标准差表
            std_online = np.mean([std for std in self.cem_online.std_table.values()]) if self.cem_online.std_table else self.cem_online.initial_std
            std_offline = np.mean([std for std in self.cem_offline.std_table.values()]) if self.cem_offline.std_table else self.cem_offline.initial_std
            avg_std = (std_online + std_offline) / 2.0
            
            # 归一化到0-1范围
            normalized_std = avg_std / self.cem_online.initial_std
            
            return normalized_std
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取酒店统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_revenue': self.total_revenue,
            'total_revenue_online': self.total_revenue_online,
            'total_revenue_offline': self.total_revenue_offline,
            'avg_revenue_per_episode': self.total_revenue / max(1, self.episode_count),
            'online_revenue_ratio': self.total_revenue_online / max(1, self.total_revenue),
            'episode_count': self.episode_count,
            'avg_ota_subsidy': np.mean(self.ota_subsidy_history) if len(self.ota_subsidy_history) > 0 else 0.0
        }
    
    def get_policy(self) -> Dict[str, Dict[Any, float]]:
        """获取当前策略"""
        return {
            'online': self.cem_online.get_policy(),
            'offline': self.cem_offline.get_policy()
        }
    
    @property
    def q_table(self) -> Dict:
        """兼容性接口：返回空字典（CEM不使用Q表）"""
        return {}
    
    def save(self) -> None:
        """
        保存Agent参数到JSON文件（易读格式）
        
        Args:
            filepath: 保存路径（自动添加.json后缀）
        """
        self.cem_online.save_model('hotel_online')
        self.cem_offline.save_model('hotel_offline')
        
    
    @classmethod
    def load(cls, filepath: str) -> 'HotelAgentDualChannel':
        """
        从文件加载Agent参数（支持JSON和PKL格式）
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的HotelAgentDualChannel实例
        """
        import json
        
        # 自动检测文件格式
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                save_dict = json.load(f)
            # JSON中的键是字符串，需要转换回元组
            save_dict['cem_online_means'] = {eval(k): v for k, v in save_dict['cem_online_means'].items()}
            save_dict['cem_online_stds'] = {eval(k): v for k, v in save_dict['cem_online_stds'].items()}
            save_dict['cem_online_state_visit_count'] = {eval(k): v for k, v in save_dict['cem_online_state_visit_count'].items()}
            save_dict['cem_offline_means'] = {eval(k): v for k, v in save_dict['cem_offline_means'].items()}
            save_dict['cem_offline_stds'] = {eval(k): v for k, v in save_dict['cem_offline_stds'].items()}
            save_dict['cem_offline_state_visit_count'] = {eval(k): v for k, v in save_dict['cem_offline_state_visit_count'].items()}
        else:
            # PKL格式
            import pickle
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
        
        # 创建新实例
        agent = cls(
            n_states=save_dict['n_states'],
            commission_rate=save_dict['commission_rate'],
            online_price_min=save_dict['online_price_min'],
            online_price_max=save_dict['online_price_max'],
            offline_price_min=save_dict['offline_price_min'],
            offline_price_max=save_dict['offline_price_max'],
            n_samples=save_dict['n_samples'],
            elite_frac=save_dict['elite_frac'],
            initial_std=save_dict['initial_std'],
            min_std=save_dict['min_std'],
            std_decay=save_dict['std_decay']
        )
        
        # 恢复CEM参数
        agent.cem_online.means = defaultdict(lambda: (save_dict['online_price_min'] + save_dict['online_price_max']) / 2, 
                                             save_dict['cem_online_means'])
        agent.cem_online.stds = defaultdict(lambda: save_dict['initial_std'], 
                                           save_dict['cem_online_stds'])
        agent.cem_online.current_std = save_dict['cem_online_current_std']
        agent.cem_online.state_visit_count = defaultdict(int, save_dict['cem_online_state_visit_count'])
        
        agent.cem_offline.means = defaultdict(lambda: (save_dict['offline_price_min'] + save_dict['offline_price_max']) / 2,
                                              save_dict['cem_offline_means'])
        agent.cem_offline.stds = defaultdict(lambda: save_dict['initial_std'],
                                            save_dict['cem_offline_stds'])
        agent.cem_offline.current_std = save_dict['cem_offline_current_std']
        agent.cem_offline.state_visit_count = defaultdict(int, save_dict['cem_offline_state_visit_count'])
        
        # 恢复统计信息
        agent.total_revenue = save_dict['total_revenue']
        agent.total_revenue_online = save_dict['total_revenue_online']
        agent.total_revenue_offline = save_dict['total_revenue_offline']
        agent.episode_count = save_dict['episode_count']
        agent.ota_subsidy_history = deque(save_dict['ota_subsidy_history'], maxlen=100)
        
        print(f"✅ 酒店Agent参数已从 {filepath} 加载")
        return agent
    