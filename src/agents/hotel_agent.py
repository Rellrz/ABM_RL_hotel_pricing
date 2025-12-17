#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店智能体 - Q-learning
Hotel Agent with Q-learning

状态空间 (D, I, C, W, S) = 15 × 4 × 3 × 2 × 3 = 1080
- D: days_ahead (0-14)
- I: inventory_level (0-3)
- C: competition_status (0-2)
- W: is_weekend (0-1)
- S: season_type (0-2)

动作空间 = 5 × 4 = 20
- 5个直销折扣档位
- 4个佣金档位
"""

import numpy as np
import pickle
from typing import Tuple, Dict
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import HOTEL_AGENT_CONFIG, ENV_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotelAgent:
    """酒店Q-learning智能体"""
    
    def __init__(self, config=None):
        """
        初始化酒店智能体
        
        Args:
            config: 配置对象，默认使用HOTEL_AGENT_CONFIG
        """
        self.config = config or HOTEL_AGENT_CONFIG
        
        # Q表：shape = (n_states, n_actions)
        self.q_table = np.zeros((self.config.n_states, self.config.n_actions))
        
        # 访问计数：用于UCB探索
        self.state_action_counts = np.zeros((self.config.n_states, self.config.n_actions))
        self.state_counts = np.zeros(self.config.n_states)
        
        # 训练参数
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        
        logger.info(f"酒店智能体初始化: 状态空间={self.config.n_states}, 动作空间={self.config.n_actions}")
    
    def discretize_state(
        self,
        days_ahead: int,
        inventory_usage: float,
        price_ratio: float,
        is_weekend: bool,
        season: int
    ) -> int:
        """
        离散化状态
        
        Args:
            days_ahead: 提前期 (0-14)
            inventory_usage: 库存使用率 (0-1)
            price_ratio: 价格比 P_ota / P_dir
            is_weekend: 是否周末
            season: 季节 (0-2)
            
        Returns:
            状态索引
        """
        # 1. days_ahead: 直接使用 (0-14)
        d = min(max(days_ahead, 0), 14)
        
        # 2. inventory_level: 离散化为4档
        if inventory_usage < self.config.inv_thresholds[0]:
            i = 0  # 空闲
        elif inventory_usage < self.config.inv_thresholds[1]:
            i = 1  # 正常
        elif inventory_usage < self.config.inv_thresholds[2]:
            i = 2  # 紧张
        else:
            i = 3  # 告急
        
        # 3. competition_status: 离散化为3档
        if price_ratio < self.config.comp_thresholds[0]:
            c = 0  # 劣势（OTA便宜）
        elif price_ratio <= self.config.comp_thresholds[1]:
            c = 1  # 均势
        else:
            c = 2  # 优势（OTA贵）
        
        # 4. is_weekend: 0或1
        w = 1 if is_weekend else 0
        
        # 5. season: 0-2
        s = min(max(season, 0), 2)
        
        # 计算状态索引
        # state = d * (4*3*2*3) + i * (3*2*3) + c * (2*3) + w * 3 + s
        state = (
            d * (self.config.n_inv_levels * self.config.n_comp_status * 
                 self.config.n_weekend * self.config.n_season) +
            i * (self.config.n_comp_status * self.config.n_weekend * self.config.n_season) +
            c * (self.config.n_weekend * self.config.n_season) +
            w * self.config.n_season +
            s
        )
        
        return state
    
    def action_to_params(self, action: int) -> Tuple[float, int]:
        """
        将动作索引转换为参数
        
        Args:
            action: 动作索引 (0-19)
            
        Returns:
            (discount_level, commission_tier)
        """
        n_commission_tiers = len(self.config.commission_tiers)
        
        discount_idx = action // n_commission_tiers
        commission_tier = action % n_commission_tiers
        
        discount_level = self.config.discount_levels[discount_idx]
        
        return discount_level, commission_tier
    
    def params_to_action(self, discount_level: float, commission_tier: int) -> int:
        """
        将参数转换为动作索引
        
        Args:
            discount_level: 折扣系数
            commission_tier: 佣金档位
            
        Returns:
            动作索引
        """
        discount_idx = self.config.discount_levels.index(discount_level)
        n_commission_tiers = len(self.config.commission_tiers)
        
        action = discount_idx * n_commission_tiers + commission_tier
        
        return action
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        选择动作（ε-greedy + UCB）
        
        Args:
            state: 状态索引
            training: 是否训练模式
            
        Returns:
            动作索引
        """
        if training and np.random.random() < self.epsilon:
            # 探索：使用UCB
            return self._ucb_action(state)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])
    
    def _ucb_action(self, state: int) -> int:
        """
        UCB探索策略
        
        Args:
            state: 状态索引
            
        Returns:
            动作索引
        """
        # 如果有未访问的动作，优先选择
        unvisited = np.where(self.state_action_counts[state] == 0)[0]
        if len(unvisited) > 0:
            return np.random.choice(unvisited)
        
        # UCB公式：Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        q_values = self.q_table[state]
        ucb_bonus = self.config.ucb_c * np.sqrt(
            np.log(self.state_counts[state] + 1) / (self.state_action_counts[state] + 1e-8)
        )
        ucb_values = q_values + ucb_bonus
        
        return np.argmax(ucb_values)
    
    def update_q_table(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool = False
    ):
        """
        更新Q表
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # 更新访问计数
        self.state_action_counts[state, action] += 1
        self.state_counts[state] += 1
        
        # Q-learning更新
        if done:
            td_target = reward
        else:
            td_target = reward + self.config.discount_factor * np.max(self.q_table[next_state])
        
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.config.learning_rate * td_error
    
    def update_epsilon(self):
        """更新探索率（指数衰减）"""
        decay_rate = (self.config.epsilon_end / self.config.epsilon_start) ** (
            1.0 / self.config.epsilon_decay_episodes
        )
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * decay_rate
        )
        self.episode += 1
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        获取贪婪策略
        
        Returns:
            策略数组 [n_states]，每个状态对应的最优动作
        """
        return np.argmax(self.q_table, axis=1)
    
    def save(self, filepath: str):
        """
        保存智能体
        
        Args:
            filepath: 保存路径
        """
        logger.info(f"保存酒店智能体: {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'state_action_counts': self.state_action_counts,
                'state_counts': self.state_counts,
                'epsilon': self.epsilon,
                'episode': self.episode
            }, f)
    
    @classmethod
    def load(cls, filepath: str, config=None):
        """
        加载智能体
        
        Args:
            filepath: 加载路径
            config: 配置对象
            
        Returns:
            智能体实例
        """
        logger.info(f"加载酒店智能体: {filepath}")
        
        agent = cls(config)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            agent.q_table = data['q_table']
            agent.state_action_counts = data['state_action_counts']
            agent.state_counts = data['state_counts']
            agent.epsilon = data['epsilon']
            agent.episode = data['episode']
        
        return agent


if __name__ == '__main__':
    """测试酒店智能体"""
    print("="*60)
    print("酒店智能体测试")
    print("="*60)
    
    agent = HotelAgent()
    
    # 测试状态离散化
    state = agent.discretize_state(
        days_ahead=5,
        inventory_usage=0.5,
        price_ratio=1.0,
        is_weekend=True,
        season=1
    )
    print(f"\n状态离散化测试: state={state}")
    
    # 测试动作选择
    action = agent.choose_action(state, training=True)
    discount, commission = agent.action_to_params(action)
    print(f"动作选择: action={action}, discount={discount}, commission_tier={commission}")
    
    # 测试Q表更新
    next_state = agent.discretize_state(6, 0.6, 0.95, True, 1)
    agent.update_q_table(state, action, 100.0, next_state)
    print(f"Q表更新: Q[{state},{action}]={agent.q_table[state, action]:.2f}")
    
    print("\n✓ 酒店智能体测试通过")
