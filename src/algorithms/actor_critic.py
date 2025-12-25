"""
Actor-Critic算法模块 (Tabular with Gaussian Policy)

实现基于高斯策略的表格型Actor-Critic算法，支持连续动作空间
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Any, Union, List, Tuple


class TabularActorCritic:
    """
    表格型Actor-Critic算法（高斯策略）
    
    使用高斯策略处理连续动作空间，适用于酒店动态定价场景。
    
    核心组件：
    - Actor（策略网络）：输出高斯分布的均值μ(s)，用于生成连续动作
    - Critic（价值网络）：评估状态价值函数V(s)
    
    算法流程：
    1. Actor根据当前状态和策略参数生成动作（从高斯分布采样）
    2. 执行动作，观察奖励和下一个状态
    3. Critic计算TD误差：δ = r + γV(s') - V(s)
    4. 更新Critic：V(s) ← V(s) + α_critic * δ
    5. 更新Actor：μ(s) ← μ(s) + α_actor * δ * ∇log π(a|s)
    
    高斯策略：
    - 均值μ(s)：由策略参数表格决定
    - 标准差σ：可固定或自适应衰减
    - 动作采样：a ~ N(μ(s), σ²)
    - 策略梯度：∇log π(a|s) = (a - μ(s)) / σ²
    
    适用场景：
    - 连续动作空间（如价格：80-200元）
    - 需要探索-利用平衡
    - 状态空间可离散化（表格型方法）
    """
    
    def __init__(self, 
                 n_states: int,
                 action_min: float = 80.0,
                 action_max: float = 200.0,
                 actor_lr: float = 0.01,
                 critic_lr: float = 0.05,
                 discount_factor: float = 0.9,
                 initial_std: float = 20.0,
                 min_std: float = 5.0,
                 std_decay: float = 0.995,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay_episodes: int = 100):
        """
        初始化Actor-Critic算法
        
        Args:
            n_states: 状态空间大小
            action_min: 动作最小值（最低价格）
            action_max: 动作最大值（最高价格）
            actor_lr: Actor学习率（策略更新步长）
            critic_lr: Critic学习率（价值更新步长）
            discount_factor: 折扣因子γ
            initial_std: 初始标准差（控制探索程度）
            min_std: 最小标准差（保持最小探索）
            std_decay: 标准差衰减率（每个episode）
            epsilon_start: 初始探索率（ε-greedy）
            epsilon_end: 最终探索率
            epsilon_decay_episodes: 探索率衰减的episode数
        """
        self.n_states = n_states
        self.action_min = action_min
        self.action_max = action_max
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        
        # 高斯策略参数
        self.initial_std = initial_std
        self.current_std = initial_std
        self.min_std = min_std
        self.std_decay = std_decay
        
        # ε-greedy探索参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.current_epsilon = epsilon_start
        
        # Actor表：状态 -> 动作均值μ(s)
        # 初始化为动作空间的中点
        initial_mean = (action_min + action_max) / 2.0
        self.actor_table = defaultdict(lambda: initial_mean)
        
        # Critic表：状态 -> 状态价值V(s)
        self.critic_table = defaultdict(float)
        
        # 访问统计
        self.state_visit_count = defaultdict(int)
        self.episode_count = 0
        
    def select_action(self, state: Union[List, np.ndarray, int], 
                     deterministic: bool = False) -> float:
        """
        根据当前策略选择动作（改进版：ε-greedy + 高斯探索）
        
        探索策略：
        1. 以ε概率进行均匀随机探索（和Q-learning一样）
        2. 以(1-ε)概率使用策略均值（确定性）
        
        这种混合策略结合了：
        - ε-greedy的均匀探索（避免陷入局部最优）
        - Actor-Critic的策略学习（利用学到的知识）
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略（测试时使用）
            
        Returns:
            连续动作值（价格）
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 获取策略均值μ(s)
        mean = self.actor_table[state_key]
        
        if deterministic:
            # 确定性策略：直接返回均值（用于评估）
            action = mean
        else:
            # ε-greedy探索
            if np.random.random() < self.current_epsilon:
                # 探索：均匀随机选择动作（和Q-learning一样）
                action = np.random.uniform(self.action_min, self.action_max)
            else:
                # 利用：使用策略均值（确定性）
                action = mean
        
        # 裁剪到有效范围
        action = np.clip(action, self.action_min, self.action_max)
        
        return float(action)
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> np.ndarray:
        """
        获取状态的策略均值（用于兼容接口）
        
        注意：Actor-Critic不使用Q值，这里返回策略均值作为参考
        
        Args:
            state: 状态
            
        Returns:
            包含策略均值的数组
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        mean = self.actor_table[state_key]
        # 返回一个包含均值的数组，用于兼容Q-learning接口
        return np.array([mean])
    
    def update(self, 
               state: Union[List, np.ndarray, int],
               action: float,
               reward: float,
               next_state: Union[List, np.ndarray, int],
               done: bool) -> float:
        """
        更新Actor和Critic
        
        使用TD(0)算法更新价值函数和策略参数
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            TD误差（用于监控学习进度）
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        
        # 1. 计算TD误差（Critic的目标）
        current_value = self.critic_table[state_key]
        
        if done:
            td_target = reward
        else:
            next_value = self.critic_table[next_state_key]
            td_target = reward + self.discount_factor * next_value
        
        td_error = td_target - current_value
        
        # 2. 更新Critic（价值函数）
        # V(s) ← V(s) + α_critic * δ
        self.critic_table[state_key] += self.critic_lr * td_error
        
        # 3. 更新Actor（策略参数）
        # 计算策略梯度：∇log π(a|s) = (a - μ(s)) / σ²
        mean = self.actor_table[state_key]
        policy_gradient = (action - mean) / (self.current_std ** 2)
        
        # Actor更新：μ(s) ← μ(s) + α_actor * δ * ∇log π(a|s)
        self.actor_table[state_key] += self.actor_lr * td_error * policy_gradient
        
        # 裁剪策略均值到有效范围
        self.actor_table[state_key] = np.clip(
            self.actor_table[state_key], 
            self.action_min, 
            self.action_max
        )
        
        # 返回TD误差（用于兼容Q-learning接口，返回类似"新Q值"的概念）
        return float(td_target)
    
    def decay_epsilon(self):
        """衰减探索率（ε-greedy）"""
        if self.episode_count < self.epsilon_decay_episodes:
            # 线性衰减
            decay_rate = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_episodes
            self.current_epsilon = self.epsilon_start - decay_rate * self.episode_count
        else:
            self.current_epsilon = self.epsilon_end
        
        # 确保不低于最小值
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon)
    
    def decay_std(self):
        """衰减标准差（减少探索）- 保留用于兼容性"""
        self.current_std = max(
            self.min_std,
            self.current_std * self.std_decay
        )
    
    def end_episode(self):
        """结束一个episode，更新相关参数"""
        self.episode_count += 1
        self.decay_epsilon()  # 使用ε衰减代替标准差衰减
    
    def get_policy(self) -> Dict[Any, float]:
        """
        获取当前策略（确定性）
        
        Returns:
            状态到最优动作（均值）的映射
        """
        policy = {}
        for state, mean in self.actor_table.items():
            policy[state] = float(mean)
        return policy
    
    def get_value_function(self) -> Dict[Any, float]:
        """
        获取价值函数
        
        Returns:
            状态到价值的映射
        """
        return dict(self.critic_table)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取算法统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        if not self.actor_table:
            return {}
        
        # 策略统计
        policy_means = list(self.actor_table.values())
        
        # 价值函数统计
        values = list(self.critic_table.values())
        
        # 计算探索覆盖率（访问过的状态数 / 总状态数）
        explored_states = len(self.actor_table)
        exploration_coverage = (explored_states / self.n_states * 100) if self.n_states > 0 else 0
        
        return {
            'num_states': explored_states,
            'episode_count': self.episode_count,
            'current_std': float(self.current_std),
            'current_epsilon': float(self.current_epsilon),
            'exploration_coverage': float(exploration_coverage),
            'policy_mean_avg': float(np.mean(policy_means)) if policy_means else 0.0,
            'policy_mean_std': float(np.std(policy_means)) if policy_means else 0.0,
            'policy_mean_min': float(np.min(policy_means)) if policy_means else 0.0,
            'policy_mean_max': float(np.max(policy_means)) if policy_means else 0.0,
            'value_avg': float(np.mean(values)) if values else 0.0,
            'value_std': float(np.std(values)) if values else 0.0,
            'value_min': float(np.min(values)) if values else 0.0,
            'value_max': float(np.max(values)) if values else 0.0,
            'num_state_visits': sum(self.state_visit_count.values()),
            'mean_q_value': float(np.mean(values)) if values else 0.0,  # 兼容性字段
            'std_q_value': float(np.std(values)) if values else 0.0,    # 兼容性字段
            'min_q_value': float(np.min(values)) if values else 0.0,    # 兼容性字段
            'max_q_value': float(np.max(values)) if values else 0.0,    # 兼容性字段
            'zero_q_percentage': 0.0,  # AC不适用，设为0
            'explored_state_actions': explored_states,
            'total_state_actions': self.n_states
        }
    
    def reset(self):
        """重置算法状态"""
        initial_mean = (self.action_min + self.action_max) / 2.0
        self.actor_table = defaultdict(lambda: initial_mean)
        self.critic_table = defaultdict(float)
        self.state_visit_count = defaultdict(int)
        self.current_std = self.initial_std
        self.current_epsilon = self.epsilon_start
        self.episode_count = 0
