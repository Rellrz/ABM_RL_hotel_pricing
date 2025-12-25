"""
PPO算法模块 (Proximal Policy Optimization - Tabular)

实现基于高斯策略的表格型PPO算法，支持连续动作空间
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, Any, Union, List, Tuple


class TabularPPO:
    """
    表格型PPO算法（高斯策略）
    
    PPO是目前最流行的策略梯度算法之一，通过裁剪目标函数来限制策略更新幅度，
    从而提高训练稳定性和样本效率。
    
    核心特性：
    1. 裁剪目标函数：防止策略更新过大
    2. 经验回放：重复使用经验数据，提高样本效率
    3. GAE优势估计：减少方差，提高稳定性
    4. 高斯策略：处理连续动作空间
    
    算法流程：
    1. 收集一批经验数据（状态、动作、奖励）
    2. 计算优势函数 A(s,a) 使用GAE
    3. 多次更新策略和价值函数：
       - Actor更新：最大化裁剪的PPO目标
       - Critic更新：最小化价值函数误差
    4. 清空经验缓冲区，继续收集
    
    PPO目标函数：
    L^CLIP(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
    其中：r(θ) = π_θ(a|s) / π_θ_old(a|s) 是重要性采样比率
    
    适用场景：
    - 需要高稳定性的训练
    - 样本收集成本高
    - 连续动作空间
    - 需要快速收敛
    """
    
    def __init__(self,
                 n_states: int,
                 action_min: float = 80.0,
                 action_max: float = 200.0,
                 actor_lr: float = 0.003,
                 critic_lr: float = 0.01,
                 discount_factor: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 initial_std: float = 20.0,
                 min_std: float = 1.0,
                 std_decay: float = 0.995,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 buffer_size: int = 2048):
        """
        初始化PPO算法
        
        Args:
            n_states: 状态空间大小
            action_min: 动作最小值（最低价格）
            action_max: 动作最大值（最高价格）
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            discount_factor: 折扣因子γ
            gae_lambda: GAE的λ参数（0-1之间）
            clip_epsilon: PPO裁剪参数ε（通常0.1-0.3）
            initial_std: 初始标准差
            min_std: 最小标准差
            std_decay: 标准差衰减率
            batch_size: 小批量大小
            n_epochs: 每次更新的epoch数
            buffer_size: 经验缓冲区大小
        """
        self.n_states = n_states
        self.action_min = action_min
        self.action_max = action_max
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.buffer_size = buffer_size
        
        # 高斯策略参数
        self.initial_std = initial_std
        self.current_std = initial_std
        self.min_std = min_std
        self.std_decay = std_decay
        
        # Actor表：状态 -> 动作均值μ(s)
        initial_mean = (action_min + action_max) / 2.0
        self.actor_table = defaultdict(lambda: initial_mean)
        
        # Critic表：状态 -> 状态价值V(s)
        self.critic_table = defaultdict(float)
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []  # 存储旧策略的对数概率
        }
        
        # 统计信息
        self.state_visit_count = defaultdict(int)
        self.episode_count = 0
        self.update_count = 0
        
    def _gaussian_log_prob(self, action: float, mean: float, std: float) -> float:
        """计算高斯分布的对数概率"""
        var = std ** 2
        log_prob = -0.5 * ((action - mean) ** 2 / var + np.log(2 * np.pi * var))
        return log_prob
    
    def select_action(self, state: Union[List, np.ndarray, int], 
                     deterministic: bool = False) -> Tuple[float, float]:
        """
        根据当前策略选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            (action, log_prob): 动作和对数概率
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 获取策略均值μ(s)
        mean = self.actor_table[state_key]
        
        # 自适应探索
        use_deterministic = deterministic or (self.current_std <= 1.0)
        
        if use_deterministic:
            action = mean
            log_prob = 0.0  # 确定性策略的对数概率
        else:
            # 从高斯分布采样
            action = np.random.normal(mean, self.current_std)
            log_prob = self._gaussian_log_prob(action, mean, self.current_std)
        
        # 裁剪到有效范围
        action = np.clip(action, self.action_min, self.action_max)
        
        return float(action), float(log_prob)
    
    def store_transition(self, state: Union[List, np.ndarray, int],
                        action: float, reward: float,
                        next_state: Union[List, np.ndarray, int],
                        done: bool, log_prob: float):
        """存储经验到缓冲区"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        self.buffer['states'].append(state_key)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state_key)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        
        self.state_visit_count[state_key] += 1
    
    def _compute_gae(self) -> np.ndarray:
        """
        计算广义优势估计（GAE）
        
        GAE结合了TD误差和蒙特卡洛回报，平衡偏差和方差：
        A^GAE(s,a) = Σ(γλ)^t * δ_t
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])
        states = self.buffer['states']
        next_states = self.buffer['next_states']
        
        # 计算所有状态的价值
        values = np.array([self.critic_table[s] for s in states])
        next_values = np.array([self.critic_table[s] for s in next_states])
        
        # 计算TD误差
        deltas = rewards + self.discount_factor * next_values * (1 - dones) - values
        
        # 计算GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def update(self, state: Union[List, np.ndarray, int], action: float,
              reward: float, next_state: Union[List, np.ndarray, int],
              done: bool, log_prob: float = 0.0) -> float:
        """
        存储经验，当缓冲区满时触发PPO更新
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            log_prob: 旧策略的对数概率
            
        Returns:
            平均价值损失（用于监控）
        """
        # 存储经验
        self.store_transition(state, action, reward, next_state, done, log_prob)
        
        # 当缓冲区满时，执行PPO更新
        if len(self.buffer['states']) >= self.buffer_size:
            return self._ppo_update()
        
        # 未触发更新时返回当前价值
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        return float(self.critic_table[state_key])
    
    def _ppo_update(self) -> float:
        """执行PPO更新"""
        # 计算优势函数
        advantages = self._compute_gae()
        
        # 标准化优势（提高稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算回报（用于更新价值函数）
        returns = advantages + np.array([self.critic_table[s] for s in self.buffer['states']])
        
        # 转换为数组
        states = self.buffer['states']
        actions = np.array(self.buffer['actions'])
        old_log_probs = np.array(self.buffer['log_probs'])
        
        # 多次epoch更新
        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0
        
        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(states))
            
            # 小批量更新
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]
                
                # 提取批量数据
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 更新Actor
                actor_loss = self._update_actor(
                    batch_states, batch_actions, 
                    batch_advantages, batch_old_log_probs
                )
                
                # 更新Critic
                critic_loss = self._update_critic(batch_states, batch_returns)
                
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                n_updates += 1
        
        # 清空缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }
        
        self.update_count += 1
        
        # 返回平均critic损失
        return total_critic_loss / n_updates if n_updates > 0 else 0.0
    
    def _update_actor(self, states: List, actions: np.ndarray,
                     advantages: np.ndarray, old_log_probs: np.ndarray) -> float:
        """更新Actor（策略网络）"""
        total_loss = 0
        
        for i, state in enumerate(states):
            # 计算当前策略的对数概率
            mean = self.actor_table[state]
            new_log_prob = self._gaussian_log_prob(actions[i], mean, self.current_std)
            
            # 计算重要性采样比率
            ratio = np.exp(new_log_prob - old_log_probs[i])
            
            # PPO裁剪目标
            surr1 = ratio * advantages[i]
            surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
            actor_loss = -min(surr1, surr2)  # 负号因为我们要最大化
            
            # 计算策略梯度
            policy_gradient = (actions[i] - mean) / (self.current_std ** 2)
            
            # 更新策略均值
            self.actor_table[state] += self.actor_lr * advantages[i] * policy_gradient
            
            # 裁剪到有效范围
            self.actor_table[state] = np.clip(
                self.actor_table[state],
                self.action_min,
                self.action_max
            )
            
            total_loss += actor_loss
        
        return total_loss / len(states) if len(states) > 0 else 0.0
    
    def _update_critic(self, states: List, returns: np.ndarray) -> float:
        """更新Critic（价值网络）"""
        total_loss = 0
        
        for i, state in enumerate(states):
            # 计算价值函数误差
            value = self.critic_table[state]
            value_loss = (returns[i] - value) ** 2
            
            # 更新价值函数
            self.critic_table[state] += self.critic_lr * (returns[i] - value)
            
            total_loss += value_loss
        
        return total_loss / len(states) if len(states) > 0 else 0.0
    
    def decay_std(self):
        """衰减标准差（减少探索）"""
        self.current_std = max(
            self.min_std,
            self.current_std * self.std_decay
        )
    
    def end_episode(self):
        """结束一个episode"""
        self.episode_count += 1
        self.decay_std()
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> np.ndarray:
        """获取状态的策略均值（兼容接口）"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        mean = self.actor_table[state_key]
        return np.array([mean])
    
    def get_policy(self) -> Dict[Any, float]:
        """获取当前策略（确定性）"""
        policy = {}
        for state, mean in self.actor_table.items():
            policy[state] = float(mean)
        return policy
    
    def get_value_function(self) -> Dict[Any, float]:
        """获取价值函数"""
        return dict(self.critic_table)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取算法统计信息"""
        if not self.actor_table:
            return {}
        
        policy_means = list(self.actor_table.values())
        values = list(self.critic_table.values())
        
        explored_states = len(self.actor_table)
        exploration_coverage = (explored_states / self.n_states * 100) if self.n_states > 0 else 0
        
        return {
            'num_states': explored_states,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'current_std': float(self.current_std),
            'buffer_size': len(self.buffer['states']),
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
            # 兼容性字段
            'mean_q_value': float(np.mean(values)) if values else 0.0,
            'std_q_value': float(np.std(values)) if values else 0.0,
            'min_q_value': float(np.min(values)) if values else 0.0,
            'max_q_value': float(np.max(values)) if values else 0.0,
            'zero_q_percentage': 0.0,
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
        self.episode_count = 0
        self.update_count = 0
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }
