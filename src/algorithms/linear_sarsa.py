"""
线性函数逼近SARSA算法

使用线性函数逼近Q值，支持连续动作空间。
通过特征工程将状态-动作对映射到特征向量，然后用线性模型估计Q值。

核心思想：
Q(s, a) = w^T φ(s, a)

其中：
- w: 权重向量
- φ(s, a): 特征向量（状态和动作的组合）

更新规则（SARSA）：
w ← w + α * δ * φ(s, a)
δ = r + γQ(s', a') - Q(s, a)
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.algorithms.base_algorithm import BaseRLAlgorithm


class LinearSARSA(BaseRLAlgorithm):
    """
    线性函数逼近SARSA算法
    
    特点：
    1. 使用线性函数逼近Q值
    2. 支持连续动作空间
    3. SARSA更新（on-policy）
    4. 通过求导找到最优连续动作
    """
    
    def __init__(self,
                 n_states: int,
                 action_min: float = 80.0,
                 action_max: float = 170.0,
                 learning_rate: float = 0.01,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 100,
                 n_features: int = 10):
        """
        初始化线性SARSA算法
        
        Args:
            n_states: 状态空间大小
            action_min: 最小动作值（最低价格）
            action_max: 最大动作值（最高价格）
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay_episodes: 探索率衰减的episode数
            n_features: 特征维度
        """
        self.n_states = n_states
        self.action_min = action_min
        self.action_max = action_max
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.current_epsilon = epsilon_start
        
        # 特征维度
        self.n_features = n_features
        
        # 权重向量（每个状态一个权重向量）
        self.weights = defaultdict(lambda: np.zeros(n_features))
        
        # 统计信息
        self.state_visit_count = defaultdict(int)
        self.episode_count = 0
        
        # 上一步的状态-动作对（用于SARSA更新）
        self.last_state = None
        self.last_action = None
        self.last_features = None
    
    def _extract_features(self, state: Union[List, np.ndarray, int], 
                         action: float) -> np.ndarray:
        """
        提取特征向量 φ(s, a)
        
        特征设计：
        1. 状态特征（归一化）
        2. 动作特征（价格及其多项式）
        3. 交互特征（状态×动作）
        
        Args:
            state: 状态
            action: 动作（价格）
            
        Returns:
            特征向量
        """
        # 归一化动作到[0, 1]
        normalized_action = (action - self.action_min) / (self.action_max - self.action_min)
        
        # 状态索引（假设是整数）
        state_idx = state if isinstance(state, int) else tuple(state)
        state_normalized = float(state_idx) / self.n_states if isinstance(state_idx, int) else 0.5
        
        # 构建特征向量
        features = np.array([
            1.0,                              # 偏置项
            state_normalized,                 # 状态特征
            normalized_action,                # 动作（线性）
            normalized_action ** 2,           # 动作（二次）
            normalized_action ** 3,           # 动作（三次）
            state_normalized * normalized_action,  # 状态×动作
            state_normalized * normalized_action ** 2,  # 状态×动作^2
            np.sin(2 * np.pi * normalized_action),  # 周期特征
            np.cos(2 * np.pi * normalized_action),  # 周期特征
            np.exp(-((normalized_action - 0.5) ** 2) / 0.1)  # 高斯特征
        ])
        
        return features[:self.n_features]
    
    def _estimate_q(self, state: Union[List, np.ndarray, int], 
                   action: float) -> float:
        """
        估计Q值：Q(s, a) = w^T φ(s, a)
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            Q值估计
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        features = self._extract_features(state, action)
        q_value = np.dot(self.weights[state_key], features)
        return float(q_value)
    
    def _find_optimal_action(self, state: Union[List, np.ndarray, int]) -> float:
        """
        找到最优动作（通过网格搜索）
        
        对于二次特征，理论上可以求解析解，但为了通用性使用网格搜索
        
        Args:
            state: 状态
            
        Returns:
            最优动作
        """
        # 网格搜索
        n_samples = 50
        actions = np.linspace(self.action_min, self.action_max, n_samples)
        q_values = [self._estimate_q(state, a) for a in actions]
        best_idx = np.argmax(q_values)
        
        return float(actions[best_idx])
    
    def select_action(self, state: Union[List, np.ndarray, int], 
                     deterministic: bool = False) -> float:
        """
        选择动作（ε-greedy + 连续动作）
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            连续动作值（价格）
        """
        if deterministic:
            # 确定性策略：选择最优动作
            return self._find_optimal_action(state)
        
        # ε-greedy探索
        if np.random.random() < self.current_epsilon:
            # 探索：随机动作
            action = np.random.uniform(self.action_min, self.action_max)
        else:
            # 利用：最优动作
            action = self._find_optimal_action(state)
        
        return float(action)
    
    def update(self, state: Union[List, np.ndarray, int], action: float,
              reward: float, next_state: Union[List, np.ndarray, int], 
              done: bool) -> float:
        """
        SARSA更新
        
        w ← w + α * δ * φ(s, a)
        δ = r + γQ(s', a') - Q(s, a)
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            TD目标值
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 提取当前特征
        features = self._extract_features(state, action)
        
        # 估计当前Q值
        current_q = self._estimate_q(state, action)
        
        # 选择下一个动作（SARSA的关键：使用实际策略选择的动作）
        next_action = self.select_action(next_state, deterministic=False)
        
        # 估计下一个Q值
        if done:
            next_q = 0.0
        else:
            next_q = self._estimate_q(next_state, next_action)
        
        # 计算TD误差
        td_target = reward + self.discount_factor * next_q
        td_error = td_target - current_q
        
        # 更新权重：w ← w + α * δ * φ(s, a)
        self.weights[state_key] += self.learning_rate * td_error * features
        
        # 更新统计
        self.state_visit_count[state_key] += 1
        
        # 保存当前状态-动作对（用于下一次更新）
        self.last_state = state
        self.last_action = action
        self.last_features = features
        
        return float(td_target)
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.episode_count < self.epsilon_decay_episodes:
            # 线性衰减
            decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_episodes
            self.current_epsilon = self.epsilon_start - decay_rate * self.episode_count
        else:
            self.current_epsilon = self.epsilon_end
        
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon)
    
    def end_episode(self):
        """结束一个episode"""
        self.episode_count += 1
        self.decay_epsilon()
        
        # 重置上一步信息
        self.last_state = None
        self.last_action = None
        self.last_features = None
    
    def get_policy(self) -> Dict[Any, float]:
        """
        获取当前策略（确定性）
        
        Returns:
            状态到最优动作的映射
        """
        policy = {}
        for state_key in self.weights.keys():
            policy[state_key] = self._find_optimal_action(state_key)
        return policy
    
    def get_value_stats(self) -> Dict[str, float]:
        """
        获取价值函数统计信息
        
        Returns:
            统计信息字典
        """
        # 计算每个状态的最优Q值
        optimal_q_values = []
        for state_key in self.weights.keys():
            optimal_action = self._find_optimal_action(state_key)
            optimal_q = self._estimate_q(state_key, optimal_action)
            optimal_q_values.append(optimal_q)
        
        # 计算权重统计
        all_weights = []
        for w in self.weights.values():
            all_weights.extend(w)
        
        explored_states = len(self.weights)
        exploration_coverage = (explored_states / self.n_states * 100) if self.n_states > 0 else 0
        
        return {
            'num_states': explored_states,
            'episode_count': self.episode_count,
            'current_epsilon': float(self.current_epsilon),
            'exploration_coverage': float(exploration_coverage),
            'optimal_q_mean': float(np.mean(optimal_q_values)) if optimal_q_values else 0.0,
            'optimal_q_std': float(np.std(optimal_q_values)) if optimal_q_values else 0.0,
            'optimal_q_min': float(np.min(optimal_q_values)) if optimal_q_values else 0.0,
            'optimal_q_max': float(np.max(optimal_q_values)) if optimal_q_values else 0.0,
            'weight_mean': float(np.mean(all_weights)) if all_weights else 0.0,
            'weight_std': float(np.std(all_weights)) if all_weights else 0.0,
            'num_state_visits': sum(self.state_visit_count.values()),
            'mean_q_value': float(np.mean(optimal_q_values)) if optimal_q_values else 0.0,
            'std_q_value': float(np.std(optimal_q_values)) if optimal_q_values else 0.0,
            'min_q_value': float(np.min(optimal_q_values)) if optimal_q_values else 0.0,
            'max_q_value': float(np.max(optimal_q_values)) if optimal_q_values else 0.0,
            'zero_q_percentage': 0.0,
            'explored_state_actions': explored_states,
            'total_state_actions': self.n_states
        }
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> float:
        """
        获取状态的最优Q值（用于兼容接口）
        
        Args:
            state: 状态
            
        Returns:
            最优Q值
        """
        optimal_action = self._find_optimal_action(state)
        return self._estimate_q(state, optimal_action)
    
    def reset(self):
        """重置算法状态"""
        self.weights = defaultdict(lambda: np.zeros(self.n_features))
        self.state_visit_count = defaultdict(int)
        self.current_epsilon = self.epsilon_start
        self.episode_count = 0
        self.last_state = None
        self.last_action = None
        self.last_features = None
