#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OTA智能体 - Q-learning
OTA Agent with Q-learning

状态空间 (D, I, M, W, S) = 15 × 4 × 3 × 2 × 3 = 1080
- D: days_ahead (0-14)
- I: inventory_level (0-3)
- M: margin_room (0-2) 套利空间
- W: is_weekend (0-1)
- S: season_type (0-2)

动作空间 = 5
- 5个补贴系数档位
"""

import numpy as np
import pickle
from typing import Tuple
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import OTA_AGENT_CONFIG, ENV_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OTAAgent:
    """OTA Q-learning智能体"""
    
    def __init__(self, config=None):
        """
        初始化OTA智能体
        
        Args:
            config: 配置对象，默认使用OTA_AGENT_CONFIG
        """
        self.config = config or OTA_AGENT_CONFIG
        
        # Q表
        self.q_table = np.zeros((self.config.n_states, self.config.n_actions))
        
        # 访问计数
        self.state_action_counts = np.zeros((self.config.n_states, self.config.n_actions))
        self.state_counts = np.zeros(self.config.n_states)
        
        # 训练参数
        self.epsilon = self.config.epsilon_start
        self.episode = 0
        
        logger.info(f"OTA智能体初始化: 状态空间={self.config.n_states}, 动作空间={self.config.n_actions}")
    
    def discretize_state(
        self,
        days_ahead: int,
        inventory_usage: float,
        margin_ratio: float,
        is_weekend: bool,
        season: int
    ) -> int:
        """
        离散化状态
        
        Args:
            days_ahead: 提前期 (0-14)
            inventory_usage: 库存使用率 (0-1)
            margin_ratio: 套利空间 P_direct / P_wholesale
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
        
        # 3. margin_room: 离散化为3档
        if margin_ratio < self.config.margin_thresholds[0]:
            m = 0  # 微利
        elif margin_ratio < self.config.margin_thresholds[1]:
            m = 1  # 正常
        else:
            m = 2  # 暴利
        
        # 4. is_weekend: 0或1
        w = 1 if is_weekend else 0
        
        # 5. season: 0-2
        s = min(max(season, 0), 2)
        
        # 计算状态索引
        state = (
            d * (self.config.n_inv_levels * self.config.n_margin_room * 
                 self.config.n_weekend * self.config.n_season) +
            i * (self.config.n_margin_room * self.config.n_weekend * self.config.n_season) +
            m * (self.config.n_weekend * self.config.n_season) +
            w * self.config.n_season +
            s
        )
        
        return state
    
    def action_to_subsidy(self, action: int) -> float:
        """
        将动作索引转换为补贴系数
        
        Args:
            action: 动作索引 (0-4)
            
        Returns:
            补贴系数
        """
        return self.config.subsidy_levels[action]
    
    def subsidy_to_action(self, subsidy: float) -> int:
        """
        将补贴系数转换为动作索引
        
        Args:
            subsidy: 补贴系数
            
        Returns:
            动作索引
        """
        return self.config.subsidy_levels.index(subsidy)
    
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
        
        # UCB公式
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
            策略数组 [n_states]
        """
        return np.argmax(self.q_table, axis=1)
    
    def save(self, filepath: str):
        """保存智能体"""
        logger.info(f"保存OTA智能体: {filepath}")
        
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
        """加载智能体"""
        logger.info(f"加载OTA智能体: {filepath}")
        
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
    """测试OTA智能体"""
    print("="*60)
    print("OTA智能体测试")
    print("="*60)
    
    agent = OTAAgent()
    
    # 测试状态离散化
    state = agent.discretize_state(
        days_ahead=3,
        inventory_usage=0.7,
        margin_ratio=1.25,
        is_weekend=False,
        season=2
    )
    print(f"\n状态离散化测试: state={state}")
    
    # 测试动作选择
    action = agent.choose_action(state, training=True)
    subsidy = agent.action_to_subsidy(action)
    print(f"动作选择: action={action}, subsidy={subsidy}")
    
    # 测试Q表更新
    next_state = agent.discretize_state(4, 0.75, 1.15, False, 2)
    agent.update_q_table(state, action, 50.0, next_state)
    print(f"Q表更新: Q[{state},{action}]={agent.q_table[state, action]:.2f}")
    
    print("\n✓ OTA智能体测试通过")
