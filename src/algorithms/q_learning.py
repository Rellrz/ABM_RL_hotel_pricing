"""
Q-learning算法模块

提取Q-learning核心算法逻辑，独立于智能体实现
"""

import numpy as np
import random
from collections import defaultdict
from configs.config import RL_CONFIG
from typing import Dict, Any, Union, List, Tuple


class QLearning:
    """
    Q-learning算法
    
    实现标准的表格型Q-learning算法
    
    核心功能：
    - Q表管理：存储和更新状态-动作值函数
    - TD学习：使用时序差分方法更新Q值
    - 策略提取：从Q表生成贪心策略
    
    Q值更新公式：
    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
    
    参数：
    - α (learning_rate): 学习率，控制更新步长
    - γ (discount_factor): 折扣因子，权衡即时奖励和未来奖励
    """
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.9):
        """
        初始化Q-learning算法
        
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = RL_CONFIG.epsilon_start
        self.epsilon_end = RL_CONFIG.epsilon_end
        self.epsilon_decay_steps = RL_CONFIG.epsilon_decay_episodes
        
        # Q表：使用defaultdict自动初始化为零
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # 访问统计
        self.state_visit_count = defaultdict(int)
        self.state_action_visit_count = defaultdict(int)
    
    def select_action(self, state, episode):
        # Q-learning: epsilon-greedy + UCB探索
        epsilon = self.get_epsilon(episode)
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        q_values = self.get_q_values(state)
        
        # 144个动作组合：action_idx = online_idx * 12 + offline_idx
        if random.random() < epsilon:
            # 增强探索策略：结合UCB和随机探索
            visit_counts = np.array([self.state_action_visit_count.get((state_key, a), 0) for a in range(self.n_actions)])
            
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
                visit_counts = np.array([self.state_action_visit_count.get((state_key, a), 0) for a in best_actions])
                least_visited_idx = np.argmin(visit_counts)
                return best_actions[least_visited_idx]
            else:
                return best_actions[0]
    
    def get_epsilon(self, episode):
        if episode >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            decay_rate = self.epsilon_decay_steps / 2
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode / decay_rate)
            return epsilon

    def get_q_value(self, state: Union[List, np.ndarray, int], action: int) -> float:
        """
        获取指定状态-动作对的Q值
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            Q值
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        return self.q_table[state_key][action]
    
    def get_q_values(self, state: Union[List, np.ndarray, int]) -> np.ndarray:
        """
        获取状态的所有动作Q值
        
        Args:
            state: 状态
            
        Returns:
            Q值数组
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        return self.q_table[state_key]
    
    def get_best_action(self, state: Union[List, np.ndarray, int]) -> int:
        """
        获取状态的最优动作（贪心策略）
        
        Args:
            state: 状态
            
        Returns:
            最优动作索引
        """
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))
    
    def update(self, state: Union[List, np.ndarray, int], 
               action: int, 
               reward: float, 
               next_state: Union[List, np.ndarray, int], 
               done: bool) -> float:
        """
        更新Q值（Q-learning核心算法）
        
        使用TD(0)更新规则：
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            更新后的Q值
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        self.state_action_visit_count[(state_key, action)] += 1
        
        # 当前Q值
        current_q = self.q_table[state_key][action]
        
        # 计算TD目标
        if done:
            # 终止状态，未来奖励为0
            td_target = reward
        else:
            # 下一个状态的最大Q值
            max_next_q = np.max(self.q_table[next_state_key])
            td_target = reward + self.discount_factor * max_next_q
        
        # TD误差
        td_error = td_target - current_q
        
        # 更新Q值
        new_q = current_q + self.learning_rate * td_error
        self.q_table[state_key][action] = new_q
        
        return new_q
    
    def get_policy(self) -> Dict[Any, int]:
        """
        提取贪心策略
        
        Returns:
            状态到最优动作的映射字典
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = int(np.argmax(q_values))
        return policy
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取Q表统计信息
        
        Returns:
            包含各种统计指标的字典
        """
        if not self.q_table:
            return {}
        
        all_q_values = []
        zero_q_count = 0
        total_q_entries = 0
        
        for q_values in self.q_table.values():
            all_q_values.extend(q_values)
            zero_q_count += np.sum(q_values == 0)
            total_q_entries += len(q_values)
        
        all_q_values = np.array(all_q_values)
        
        # 计算探索覆盖率
        explored_state_actions = sum(1 for count in self.state_action_visit_count.values() if count > 0)
        total_state_actions = len(self.q_table) * self.n_actions
        exploration_coverage = (explored_state_actions / total_state_actions * 100) if total_state_actions > 0 else 0
        
        return {
            'mean_q_value': float(np.mean(all_q_values)),
            'std_q_value': float(np.std(all_q_values)),
            'min_q_value': float(np.min(all_q_values)),
            'max_q_value': float(np.max(all_q_values)),
            'num_states': len(self.q_table),
            'num_state_visits': sum(self.state_visit_count.values()),
            'zero_q_percentage': (zero_q_count / total_q_entries * 100) if total_q_entries > 0 else 0,
            'exploration_coverage': exploration_coverage,
            'explored_state_actions': explored_state_actions,
            'total_state_actions': total_state_actions
        }
    
    def reset(self):
        """重置Q表和统计信息"""
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.state_visit_count = defaultdict(int)
        self.state_action_visit_count = defaultdict(int)
