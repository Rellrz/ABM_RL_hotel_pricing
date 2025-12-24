# 标准库导入
import pickle
import random
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import numpy as np
import pandas as pd
from scipy import stats

# 本地模块导入
from configs.config import RL_CONFIG
from src.utils.training_monitor import get_training_monitor
from src.algorithms.q_learning import QLearning

class HotelAgent:
    """
    Q-learning智能体
    
    实现Q-learning算法的智能体，用于酒店动态定价决策。
    支持ε-贪心探索策略、UCB探索增强、状态访问统计等功能。
    
    主要特性：
    - ε-贪心探索：平衡探索和利用
    - UCB增强：优先选择访问次数较少的状态-动作对
    - 状态离散化：将连续状态映射到离散状态空间
    - 访问统计：跟踪状态和动作访问次数
    - Q值更新：使用TD学习更新Q值
    
    状态空间：
    - 总状态数：30（库存等级5 × 季节3 × 日期类型2）
    - 状态编码：inventory_level × 6 + season × 2 + weekday
    
    动作空间：
    - 总动作数：36（线上6档 × 线下6档）
    - 动作映射：action_idx = online_idx * 6 + offline_idx
    - 线上价格档位：[80, 90, 100, 110, 120, 130]元
    - 线下价格档位：[90, 105, 120, 135, 150, 165]元
    
    学习参数：
    - 学习率：控制Q值更新速度
    - 折扣因子：权衡即时奖励和未来奖励
    - ε衰减：逐步减少探索概率
    
    Attributes:
        n_states (int): 状态数量（默认30）
        n_actions (int): 动作数量（默认6）
        learning_rate (float): 学习率
        discount_factor (float): 折扣因子
        epsilon_start (float): 初始探索概率
        epsilon_end (float): 最终探索概率
        epsilon_decay_steps (int): ε衰减步数
        q_table (Dict): Q值表，键为状态，值为动作Q值数组
        state_visit_count (Dict): 状态访问计数
        state_action_visit_count (Dict): 状态-动作访问计数
        training_history (List): 训练历史记录
        
    Note:
        - 使用defaultdict自动初始化Q值和访问计数
        - 支持UCB探索策略，优先探索访问次数少的状态-动作对
        - ε值随训练episode线性衰减
        - 状态离散化支持库存、季节、工作日类型组合
    """
    
    def __init__(self, n_states: int = 30, n_actions: int = None, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon_start: float = 0.8, epsilon_end: float = 0.01, epsilon_decay_steps: int = 50):
        
        # 如果未指定动作数，从配置文件读取
        if n_actions is None:
            n_actions = RL_CONFIG['n_actions']  # 36个动作组合（线上6价格 × 线下6价格）
        
        self.n_states = n_states
        self.n_actions = n_actions  # 6×6=36个动作组合（线上6价格 × 线下6价格）
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        
        # 初始化Q-learning算法
        self.q_learning = QLearning(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )
        
        # 训练历史
        self.training_history = []
    
    # 向后兼容的属性访问
    @property
    def q_table(self):
        """Q表（委托给Q-learning算法）"""
        return self.q_learning.q_table
    
    @property
    def state_visit_count(self):
        """状态访问计数（委托给Q-learning算法）"""
        return self.q_learning.state_visit_count
    
    @property
    def state_action_visit_count(self):
        """状态-动作访问计数（委托给Q-learning算法）"""
        return self.q_learning.state_action_visit_count
    
    def get_epsilon(self, episode: int) -> float:
        """获取当前的epsilon值 - 使用更快的指数衰减策略"""
        if episode >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            # 使用更快的指数衰减策略，使探索率快速下降
            # epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-episode / decay_rate)
            decay_rate = self.epsilon_decay_steps / 2  # 进一步加快衰减速率
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode / decay_rate)
            return epsilon
    
    def discretize_state(self, state_info: Dict[str, Any], season: int, weekday: int) -> int:
        """离散化状态 - 基于当前库存、季节和日期类型"""
        inventory_level = state_info['inventory_level']
        
        # 计算状态索引
        # inventory_level: 0-4 (5个等级)
        # season: 0-2 (3个季节)
        # weekday: 0-1 (工作日/周末)
        state_index = inventory_level * 6 + season * 2 + weekday
        
        return min(state_index, self.n_states - 1)  # 防止越界
    
    def select_action(self, state: Union[List, np.ndarray, int], episode: int) -> int:
        """选择动作（epsilon-greedy + 增强UCB探索策略）"""
        epsilon = self.get_epsilon(episode) # 获取当前探索系数，用于epsilon-greedy策略
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        q_values = self.q_learning.get_q_values(state)
        
        # 36个动作组合：action_idx = online_idx * 6 + offline_idx
        if random.random() < epsilon:
            # 增强探索策略：结合UCB和随机探索
            visit_counts = np.array([self.q_learning.state_action_visit_count.get((state_key, a), 0) for a in range(self.n_actions)])
            
            # 如果存在完全未探索的动作（访问次数为0），优先选择这些动作
            unvisited_actions = np.where(visit_counts == 0)[0]
            if len(unvisited_actions) > 0:
                # 如果有未探索的动作，随机选择一个
                return random.choice(unvisited_actions)
            
            # 否则使用UCB策略选择访问次数最少的动作
            min_visits = np.min(visit_counts)
            least_visited_actions = np.where(visit_counts == min_visits)[0]
            
            if len(least_visited_actions) > 1:
                # 如果有多个最少访问的动作，选择Q值较高的那个
                q_values_least = q_values[least_visited_actions]
                best_idx = np.argmax(q_values_least)
                return least_visited_actions[best_idx]
            else:
                return least_visited_actions[0]
        else:
            # 利用：选择Q值最大的动作
            # 如果有多个最大值，优先选择访问次数较少的
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            
            if len(best_actions) > 1:
                # 在最佳动作中选择访问次数最少的
                visit_counts = np.array([self.q_learning.state_action_visit_count.get((state_key, a), 0) for a in best_actions])
                least_visited_idx = np.argmin(visit_counts)
                return best_actions[least_visited_idx]
            else:
                return best_actions[0]
    
    def update_q_table(self, state: Union[List, np.ndarray, int], action: int, reward: float, next_state: Union[List, np.ndarray, int], done: bool) -> float:
        """更新Q表（委托给Q-learning算法）"""
        # 委托给Q-learning算法进行更新
        new_q = self.q_learning.update(state, action, reward, next_state, done)
        return new_q
    
    def get_policy(self) -> Dict[Any, int]:
        """获取当前策略（委托给Q-learning算法）"""
        return self.q_learning.get_policy()
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """获取Q值统计信息（委托给Q-learning算法）"""
        return self.q_learning.get_statistics()
    
    def save_agent(self, filepath: str) -> None:
        """
        保存智能体状态和训练历史到文件
        
        功能描述：
        将Q-learning智能体的完整状态保存到pickle文件，包括Q表、访问计数、训练历史、超参数等所有关键信息。
        
        参数:
            filepath (str): 保存文件的路径，应为.pkl文件
            
        保存内容:
        - q_table: Q值表，包含所有状态-动作对的Q值
        - state_visit_count: 状态访问计数统计
        - state_action_visit_count: 状态-动作对访问计数
        - training_history: 完整的训练历史记录
        - hyperparameters: 所有超参数设置
        
        文件格式:
        使用pickle格式保存，包含完整的智能体状态字典
        
        Note:
        - 自动将defaultdict转换为普通dict以便保存
        - 保存后打印确认信息
        - 文件可用于后续加载和继续训练
        - 包含所有必要的超参数信息
        """
        # 转换Q表为普通字典以便保存
        q_table_dict = dict(self.q_table)
        state_visit_dict = dict(self.state_visit_count)
        state_action_visit_dict = dict(self.state_action_visit_count)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'state_visit_count': state_visit_dict,
                'state_action_visit_count': state_action_visit_dict,
                'training_history': self.training_history,
                'hyperparameters': {
                    'n_states': self.n_states,
                    'n_actions': self.n_actions,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay_steps': self.epsilon_decay_steps
                }
            }, f)
        print(f"智能体已保存到：{filepath}")
    
    def load_agent(self, filepath: str) -> None:
        """
        从文件加载智能体状态和训练历史
        
        功能描述：
        从pickle文件恢复Q-learning智能体的完整状态，包括Q表、访问计数、训练历史等信息。
        
        参数:
            filepath (str): 加载文件的路径，应为之前保存的.pkl文件
            
        恢复内容:
        - q_table: Q值表，恢复所有状态-动作对的Q值
        - state_visit_count: 状态访问计数统计
        - state_action_visit_count: 状态-动作对访问计数  
        - training_history: 完整的训练历史记录
        - 超参数: 自动恢复保存时的超参数设置
        
        加载逻辑:
        1. 从pickle文件读取保存的数据字典
        2. 恢复Q表为defaultdict格式
        3. 恢复访问计数统计
        4. 恢复训练历史记录
        
        Note:
        - 自动将普通dict转换回defaultdict格式
        - 加载后打印确认信息
        - 可继续之前的训练过程
        - 保持与保存时相同的超参数设置
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 恢复Q表
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for state, q_values in data['q_table'].items():
            self.q_table[state] = q_values
        
        # 恢复其他属性
        self.state_visit_count = defaultdict(int, data['state_visit_count'])
        self.state_action_visit_count = defaultdict(int, data.get('state_action_visit_count', {}))
        self.training_history = data['training_history']
        
        print(f"智能体已从{filepath}加载")