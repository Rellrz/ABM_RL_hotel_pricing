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
from src.algorithms.actor_critic import TabularActorCritic
from src.algorithms.linear_sarsa import LinearSARSA
from src.algorithms.cem import CrossEntropyMethod
from src.algorithms.cem_nn import NeuralCrossEntropyMethod

class HotelAgent:
    """
    酒店定价智能体（支持多算法）
    
    支持多种强化学习算法：
    1. Q-learning: 离散动作空间（144个定价组合）
    2. Actor-Critic: 连续动作空间（价格80-170元）
    3. Linear SARSA: 连续动作空间（线性函数逼近）
    4. CEM: 连续动作空间（交叉熵方法，表格版本）
    5. CEM-NN: 连续动作空间（交叉熵方法，神经网络版本）
    
    算法切换：
    - 通过配置文件中的 RL_CONFIG.algorithm 参数选择
    - 'q_learning': 使用Q-learning算法
    - 'actor_critic': 使用Actor-Critic算法
    - 'linear_sarsa': 使用线性SARSA算法
    - 'cem': 使用交叉熵方法（表格版本）
    - 'cem_nn': 使用交叉熵方法（神经网络版本）
    
    状态空间：
    - 总状态数：30（库存等级5 × 季节3 × 日期类型2）
    - 状态编码：inventory_level × 6 + season × 2 + weekday
    
    Attributes:
        algorithm_type (str): 当前使用的算法类型
        algorithm: 算法实例（QLearning 或 TabularActorCritic）
        n_states (int): 状态数量
        n_actions (int): 动作数量（Q-learning使用）
    """
    
    def __init__(self):
        """初始化智能体，根据配置选择算法"""
        
        # 读取配置
        self.algorithm_type = RL_CONFIG.algorithm
        self.n_states = RL_CONFIG.n_states
        self.n_actions = RL_CONFIG.n_actions
        self.discount_factor = RL_CONFIG.discount_factor
        
        # 根据配置初始化算法
        if self.algorithm_type == 'q_learning':
            print(f"✅ 使用 Q-learning 算法（离散动作空间）")
            self.learning_rate = RL_CONFIG.learning_rate
            self.epsilon_start = RL_CONFIG.epsilon_start
            self.epsilon_end = RL_CONFIG.epsilon_end
            self.epsilon_decay_steps = RL_CONFIG.epsilon_decay_episodes
            
            self.algorithm = QLearning(
                n_states=self.n_states,
                n_actions=self.n_actions,
                learning_rate=self.learning_rate,
                discount_factor=self.discount_factor
            )
            
        elif self.algorithm_type == 'actor_critic':
            print(f"✅ 使用 Actor-Critic 算法（连续动作空间 + ε-greedy探索）")
            self.epsilon_start = RL_CONFIG.epsilon_start
            self.epsilon_end = RL_CONFIG.epsilon_end
            self.epsilon_decay_steps = RL_CONFIG.epsilon_decay_episodes
            
            self.algorithm = TabularActorCritic(
                n_states=self.n_states,
                action_min=RL_CONFIG.action_min,
                action_max=RL_CONFIG.action_max,
                actor_lr=RL_CONFIG.actor_lr,
                critic_lr=RL_CONFIG.critic_lr,
                discount_factor=self.discount_factor,
                initial_std=RL_CONFIG.initial_std,
                min_std=RL_CONFIG.min_std,
                std_decay=RL_CONFIG.std_decay,
                epsilon_start=self.epsilon_start,
                epsilon_end=self.epsilon_end,
                epsilon_decay_episodes=self.epsilon_decay_steps
            )
            
        elif self.algorithm_type == 'linear_sarsa':
            print(f"✅ 使用 Linear SARSA 算法（连续动作空间 + 线性函数逼近）")
            self.epsilon_start = RL_CONFIG.epsilon_start
            self.epsilon_end = RL_CONFIG.epsilon_end
            self.epsilon_decay_steps = RL_CONFIG.epsilon_decay_episodes
            
            self.algorithm = LinearSARSA(
                n_states=self.n_states,
                action_min=RL_CONFIG.action_min,
                action_max=RL_CONFIG.action_max,
                learning_rate=RL_CONFIG.learning_rate,
                discount_factor=self.discount_factor,
                epsilon_start=self.epsilon_start,
                epsilon_end=self.epsilon_end,
                epsilon_decay_episodes=self.epsilon_decay_steps,
                n_features=10
            )
            
        elif self.algorithm_type == 'cem':
            print(f"✅ 使用 CEM 算法（连续动作空间 + 交叉熵方法，表格版本）")
            self.epsilon_start = 0.0  # CEM不使用epsilon
            self.epsilon_end = 0.0
            self.epsilon_decay_steps = 1
            
            self.algorithm = CrossEntropyMethod(
                n_states=self.n_states,
                action_min=RL_CONFIG.action_min,
                action_max=RL_CONFIG.action_max,
                discount_factor=self.discount_factor,
                n_samples=RL_CONFIG.cem_n_samples if hasattr(RL_CONFIG, 'cem_n_samples') else 20,
                elite_frac=RL_CONFIG.cem_elite_frac if hasattr(RL_CONFIG, 'cem_elite_frac') else 0.2,
                initial_std=RL_CONFIG.initial_std,
                min_std=RL_CONFIG.min_std,
                std_decay=RL_CONFIG.std_decay
            )
            
        elif self.algorithm_type == 'cem_nn':
            print(f"✅ 使用 CEM-NN 算法（连续动作空间 + 交叉熵方法，神经网络版本）")
            self.epsilon_start = 0.0  # CEM不使用epsilon
            self.epsilon_end = 0.0
            self.epsilon_decay_steps = 1
            
            self.algorithm = NeuralCrossEntropyMethod(
                state_dim=self.n_states,
                action_dim=1,
                action_min=RL_CONFIG.action_min,
                action_max=RL_CONFIG.action_max,
                discount_factor=self.discount_factor,
                n_samples=RL_CONFIG.cem_n_samples if hasattr(RL_CONFIG, 'cem_n_samples') else 20,
                elite_frac=RL_CONFIG.cem_elite_frac if hasattr(RL_CONFIG, 'cem_elite_frac') else 0.2,
                learning_rate=RL_CONFIG.learning_rate if hasattr(RL_CONFIG, 'learning_rate') else 0.001,
                hidden_dims=[64, 64],
                batch_size=32,
                memory_size=1000,
                min_std=RL_CONFIG.min_std,
                device='cpu'
            )
        else:
            raise ValueError(f"不支持的算法类型: {self.algorithm_type}")
        
        # 训练历史
        self.training_history = []
    
    # 向后兼容的属性访问
    @property
    def q_table(self):
        """Q表（委托给算法）"""
        if self.algorithm_type == 'q_learning':
            return self.algorithm.q_table
        elif self.algorithm_type == 'actor_critic':
            # Actor-Critic使用actor_table作为策略表
            return self.algorithm.actor_table
        elif self.algorithm_type == 'linear_sarsa':
            # Linear SARSA使用权重向量，返回一个空字典以兼容
            return {}
        elif self.algorithm_type == 'cem':
            # CEM使用action_dist，返回一个空字典以兼容
            return {}
        elif self.algorithm_type == 'cem_nn':
            # CEM-NN使用神经网络，返回一个空字典以兼容
            return {}
        else:
            return {}
    
    @property
    def state_visit_count(self):
        """状态访问计数（委托给算法）"""
        return self.algorithm.state_visit_count
    
    @property
    def state_action_visit_count(self):
        """状态-动作访问计数（委托给算法）"""
        if self.algorithm_type == 'q_learning':
            return self.algorithm.state_action_visit_count
        else:
            # Actor-Critic使用state_visit_count
            return self.algorithm.state_visit_count
    
    def get_epsilon(self, episode: int) -> float:
        """获取当前的epsilon值"""
        if self.algorithm_type in ['actor_critic', 'linear_sarsa']:
            # Actor-Critic和Linear SARSA使用ε-greedy，直接返回current_epsilon
            return self.algorithm.current_epsilon
        elif self.algorithm_type in ['cem', 'cem_nn']:
            # CEM和CEM-NN不使用epsilon
            return 0.0
        else:
            # Q-learning使用epsilon-greedy
            return self.algorithm.get_epsilon(episode)

    
    def discretize_state(self, state_info: Dict[str, Any], season: int, weekday: int) -> int:
        """离散化状态 - 基于当前库存、季节和日期类型"""
        inventory_level = state_info['inventory_level']
        
        # 计算状态索引
        # inventory_level: 0-4 (5个等级)
        # season: 0-2 (3个季节)
        # weekday: 0-1 (工作日/周末)
        state_index = inventory_level * 6 + season * 2 + weekday
        
        return min(state_index, self.n_states - 1)  # 防止越界
    
    def select_action(self, state: Union[List, np.ndarray, int], episode: int) -> Union[int, float]:
        """
        选择动作
        
        Returns:
            int: Q-learning返回离散动作索引 (0-143)
            float: 连续动作算法返回连续价格值 (80-170)
        """
        # 连续动作算法：直接委托给算法的select_action方法
        if self.algorithm_type in ['actor_critic', 'linear_sarsa', 'cem', 'cem_nn']:
            return self.algorithm.select_action(state, deterministic=False)
        elif self.algorithm_type == 'q_learning':
            return self.algorithm.select_action(state, episode)

    
    def update_q_table(self, state: Union[List, np.ndarray, int], action: Union[int, float], 
                      reward: float, next_state: Union[List, np.ndarray, int], done: bool) -> float:
        """更新算法参数"""
        # 委托给具体算法进行更新
        new_value = self.algorithm.update(state, action, reward, next_state, done)
        return new_value
    
    def end_episode(self):
        """结束episode，更新相关参数"""
        if self.algorithm_type in ['actor_critic', 'linear_sarsa', 'cem', 'cem_nn']:
            self.algorithm.end_episode()
    
    def get_policy(self) -> Dict[Any, Union[int, float]]:
        """获取当前策略"""
        return self.algorithm.get_policy()
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """获取算法统计信息"""
        # 不同算法使用不同的方法名
        if hasattr(self.algorithm, 'get_value_stats'):
            return self.algorithm.get_value_stats()
        elif hasattr(self.algorithm, 'get_statistics'):
            return self.algorithm.get_statistics()
        else:
            # 返回默认统计信息
            return {
                'mean_q_value': 0.0,
                'std_q_value': 0.0,
                'min_q_value': 0.0,
                'max_q_value': 0.0,
                'exploration_coverage': 0.0
            }
    
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