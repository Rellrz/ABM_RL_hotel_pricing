"""
OTA Agent模块

OTA（在线旅行社）代理，负责决策补贴金额以最大化利润
使用CEM算法进行决策

业务逻辑：
- OTA从酒店获取线上基础价格
- OTA决策补贴金额
- 最终线上价格 = 基础价格 - 补贴
- OTA利润 = 佣金收入 - 补贴支出
"""

import numpy as np
from collections import defaultdict, deque
from typing import Union, List, Dict, Any
from src.algorithms.cem import CrossEntropyMethod
from src.algorithms.cem_nn import NeuralCrossEntropyMethod
from configs.config import RL_CONFIG


class OTAAgent:
    """
    OTA代理：使用CEM算法决策补贴金额
    
    状态空间：
    - 酒店线上基础价格
    - 酒店线下价格
    - 库存水平
    - 季节
    - 是否周末
    
    动作空间：
    - subsidy_ratio ∈ [0, 0.8]（补贴占佣金的比例）
    
    收益函数：
    - profit = commission_revenue * (1 - subsidy_ratio)
    - commission_revenue = bookings_online * price_online_base * commission_rate
    - subsidy_amount = commission_revenue * subsidy_ratio
    - 最终线上价格 = price_online_base - subsidy_amount / bookings_online
    """
    
    def __init__(self, 
                 commission_rate: float = 0.15,
                 subsidy_ratio_min: float = 0.0,
                 subsidy_ratio_max: float = 0.8,
                 n_states: int = 36,
                 n_samples: int = 20,
                 elite_frac: float = 0.2,
                 initial_std: float = 0.2,
                 min_std: float = 0.02,
                 std_decay: float = 0.99):
        """
        初始化OTA Agent
        
        Args:
            commission_rate: 佣金率（默认15%）
            subsidy_ratio_min: 补贴比例最小值（0%）
            subsidy_ratio_max: 补贴比例最大值（80%）
            n_states: 状态空间大小
            n_samples: CEM采样数量
            elite_frac: 精英样本比例
            initial_std: 初始标准差
            min_std: 最小标准差
            std_decay: 标准差衰减率
        """
        self.commission_rate = commission_rate
        self.subsidy_ratio_min = subsidy_ratio_min
        self.subsidy_ratio_max = subsidy_ratio_max
        self.n_states = int(n_states)
        self.n_base_states = 90
        self.n_stages = max(1, self.n_states // self.n_base_states)
        self.algorithm_type = RL_CONFIG.cem_algorithm
        
        # 根据配置选择算法
        if self.algorithm_type == 'cem_nn':
            # 使用神经网络版CEM（针对补贴比例的小范围，使用更小的标准差）
            # 补贴范围0-0.8，使用initial_std=0.1, min_std=0.02来控制波动
            self.cem = NeuralCrossEntropyMethod(
                state_dim=self.n_states,
                action_dim=1,
                action_min=subsidy_ratio_min,
                action_max=subsidy_ratio_max,
                discount_factor=0.99,
                n_samples=n_samples,
                elite_frac=elite_frac,
                learning_rate=RL_CONFIG.cem_nn_learning_rate,
                hidden_dims=RL_CONFIG.cem_nn_hidden_dims,
                batch_size=RL_CONFIG.cem_nn_batch_size,
                memory_size=RL_CONFIG.cem_nn_memory_size,
                min_std=RL_CONFIG.cem_nn_min_std,
                initial_std=RL_CONFIG.cem_nn_initial_std  # 使用较小的初始标准差
            )
        else:
            # 使用传统表格版CEM
            self.cem = CrossEntropyMethod(
                n_states=n_states,
                action_min=subsidy_ratio_min,
                action_max=subsidy_ratio_max,
                discount_factor=0.99,
                n_samples=n_samples,
                elite_frac=elite_frac,
                initial_std=initial_std,
                min_std=min_std,
                std_decay=std_decay
            )
        
        # 统计信息
        self.total_profit = 0.0
        self.total_commission = 0.0
        self.total_subsidy_cost = 0.0
        self.episode_count = 0
        
    def extract_state(self, 
                     hotel_price_online: float, 
                     hotel_price_offline: float, 
                     env_state: Dict) -> int:
        """
        提取OTA的状态特征并离散化
        
        状态特征：
        1. 价格差异（线下 - 线上基础）：反映渠道竞争力
        2. 库存水平：影响需求预期
        3. 季节：影响需求强度
        4. 是否周末：影响需求模式
        
        Args:
            hotel_price_online: 酒店的线上基础价格
            hotel_price_offline: 酒店的线下价格
            env_state: 环境状态字典
            
        Returns:
            state_idx: 离散化的状态索引
        """
        # 提取环境状态
        # 直接使用环境已经离散化好的inventory_level（0-2）
        inventory_level = env_state.get('inventory_level', 2)
        season = env_state.get('season', 0)
        weekday = int(env_state.get('weekday', 0))
        
        # 特征1：价格差异（关键特征）
        # 正常情况：线上基础价格 > 线下价格（因为要覆盖佣金）
        # OTA需要通过补贴来降低最终线上价格，使其有竞争力
        price_gap = hotel_price_online - hotel_price_offline
        if price_gap < 0:
            price_gap_level = 0  # 线上基础价格更低（异常情况，几乎不补贴）
        elif price_gap < 5:
            price_gap_level = 1  # 价差很小，少补贴
        elif price_gap < 15:
            price_gap_level = 2  # 价差小，中等补贴
        elif price_gap < 25:
            price_gap_level = 3  # 价差中等，较多补贴
        else:
            price_gap_level = 4  # 价差大，需要大补贴才能吸引客户
        

        
        base_state_idx = (price_gap_level * 3 * 3 * 2 + 
                         inventory_level * 3 * 2 + 
                         season * 2 + 
                         weekday)

        stage_id = int(env_state.get('stage_id', 0))
        stage_id = int(np.clip(stage_id, 0, self.n_stages - 1))

        state_idx = base_state_idx * self.n_stages + stage_id
        return int(state_idx)
    
    def select_action(self, 
                     hotel_price_online: float, 
                     hotel_price_offline: float, 
                     env_state: Dict,
                     deterministic: bool = False) -> float:
        """
        决策补贴比例（占佣金的百分比）
        
        策略考虑：
        1. 价格差异：线下比线上贵越多，补贴比例越低
        2. 库存水平：库存高时可提高补贴比例吸引客户
        3. 季节性：旺季降低补贴比例，淡季提高补贴比例
        
        Args:
            hotel_price_online: 酒店的线上基础价格
            hotel_price_offline: 酒店的线下价格
            env_state: 环境状态
            deterministic: 是否使用确定性策略
            
        Returns:
            subsidy_ratio: 补贴比例（0-0.8）
        """
        # 提取状态
        state = self.extract_state(hotel_price_online, hotel_price_offline, env_state)
        
        # 使用CEM选择补贴比例
        subsidy_ratio = self.cem.select_action(state, deterministic)
        
        # 约束：补贴比例在0%-80%之间
        subsidy_ratio = np.clip(subsidy_ratio, self.subsidy_ratio_min, self.subsidy_ratio_max)
        
        return float(subsidy_ratio)
    
    def update(self, 
              hotel_price_online: float, 
              hotel_price_offline: float, 
              env_state: Dict,
              subsidy_ratio: float, 
              profit: float, 
              next_env_state: Dict, 
              done: bool) -> None:
        """
        更新CEM算法
        
        Args:
            hotel_price_online: 酒店的线上基础价格
            hotel_price_offline: 酒店的线下价格
            env_state: 当前环境状态
            subsidy_ratio: 执行的补贴比例
            profit: 获得的利润
            next_env_state: 下一个环境状态
            done: 是否结束
        """
        # 提取状态
        state = self.extract_state(hotel_price_online, hotel_price_offline, env_state)
        next_state = self.extract_state(hotel_price_online, hotel_price_offline, next_env_state)
        
        # 更新CEM
        self.cem.update(state, subsidy_ratio, profit, next_state, done)
        
        # 更新统计
        self.total_profit += profit
    
    def end_episode(self) -> None:
        """结束episode，更新分布参数"""
        self.cem.end_episode()
        self.episode_count += 1
    
    def calculate_profit(self, 
                        bookings_online: int, 
                        price_online_base: float, 
                        subsidy_ratio: float) -> float:
        """
        计算OTA利润
        
        利润 = 佣金收入 * (1 - 补贴比例)
        
        Args:
            bookings_online: 线上预订量
            price_online_base: 线上基础价格
            subsidy_ratio: 补贴比例（0-0.8）
            
        Returns:
            profit: OTA利润
        """
        # 佣金收入
        commission_revenue = bookings_online * price_online_base * self.commission_rate
        
        # 补贴支出 = 佣金收入 * 补贴比例
        subsidy_cost = commission_revenue * subsidy_ratio
        
        # 利润 = 佣金收入 - 补贴支出
        profit = commission_revenue - subsidy_cost
        
        # 更新统计
        self.total_commission += commission_revenue
        self.total_subsidy_cost += subsidy_cost
        
        return profit
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取OTA统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_profit': self.total_profit,
            'total_commission': self.total_commission,
            'total_subsidy_cost': self.total_subsidy_cost,
            'avg_profit_per_episode': self.total_profit / max(1, self.episode_count),
            'subsidy_ratio': self.total_subsidy_cost / max(1, self.total_commission),
            'episode_count': self.episode_count
        }
    
    def get_policy(self) -> Dict[Any, float]:
        """获取当前策略（各状态的最优补贴）"""
        return self.cem.get_policy()
    
    def save(self) -> None:
        """
        保存Agent参数到JSON文件（易读格式）
        
        Args:
            filepath: 保存路径（自动添加.json后缀）
        """
        import json
        
        self.cem.save_model('ota')


    
    @classmethod
    def load(cls) -> 'OTAAgent':
        """
        从文件加载Agent参数（支持JSON和PKL格式）
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的OTAAgent实例
        """
        import json
        from collections import defaultdict
        
        # 自动检测文件格式
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                save_dict = json.load(f)
            # JSON中的键是字符串，需要转换回元组
            save_dict['cem_means'] = {eval(k): v for k, v in save_dict['cem_means'].items()}
            save_dict['cem_stds'] = {eval(k): v for k, v in save_dict['cem_stds'].items()}
            save_dict['cem_state_visit_count'] = {eval(k): v for k, v in save_dict['cem_state_visit_count'].items()}
        else:
            # PKL格式
            import pickle
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
        
        # 创建新实例
        agent = cls(
            commission_rate=save_dict['commission_rate'],
            subsidy_ratio_min=save_dict['subsidy_ratio_min'],
            subsidy_ratio_max=save_dict['subsidy_ratio_max'],
            n_states=save_dict['n_states'],
            n_samples=save_dict['n_samples'],
            elite_frac=save_dict['elite_frac'],
            initial_std=save_dict['initial_std'],
            min_std=save_dict['min_std'],
            std_decay=save_dict['std_decay']
        )
        
        # 恢复CEM参数
        agent.cem.means = defaultdict(
            lambda: (save_dict['subsidy_ratio_min'] + save_dict['subsidy_ratio_max']) / 2,
            save_dict['cem_means']
        )
        agent.cem.stds = defaultdict(
            lambda: save_dict['initial_std'],
            save_dict['cem_stds']
        )
        agent.cem.current_std = save_dict['cem_current_std']
        agent.cem.state_visit_count = defaultdict(int, save_dict['cem_state_visit_count'])
        
        # 恢复统计信息
        agent.total_profit = save_dict['total_profit']
        agent.total_commission = save_dict['total_commission']
        agent.total_subsidy_cost = save_dict['total_subsidy_cost']
        agent.episode_count = save_dict['episode_count']
        
        print(f"✅ OTA Agent参数已从 {filepath} 加载")
        return agent
    
