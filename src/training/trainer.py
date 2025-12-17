#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双智能体训练器
Dual-Agent Trainer

训练流程：
1. 顺序博弈：酒店先决策 → OTA观察后决策
2. ABM模拟客户行为
3. 双智能体同时更新Q表
4. 记录训练指标
"""

import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
import json
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import TRAINING_CONFIG, PATH_CONFIG, ENV_CONFIG, HOTEL_AGENT_CONFIG
from src.environment.hotel_env import HotelEnvironment
from src.agents.hotel_agent import HotelAgent
from src.agents.ota_agent import OTAAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualAgentTrainer:
    """双智能体训练器"""
    
    def __init__(
        self,
        env: HotelEnvironment,
        hotel_agent: HotelAgent,
        ota_agent: OTAAgent
    ):
        """
        初始化训练器
        
        Args:
            env: 环境
            hotel_agent: 酒店智能体
            ota_agent: OTA智能体
        """
        self.env = env
        self.hotel_agent = hotel_agent
        self.ota_agent = ota_agent
        
        # 训练历史
        self.training_history = {
            'episode_rewards_hotel': [],
            'episode_rewards_ota': [],
            'episode_bookings_direct': [],
            'episode_bookings_ota': [],
            'episode_avg_occupancy': [],
            'epsilon_hotel': [],
            'epsilon_ota': []
        }
        
        logger.info("训练器初始化完成")
    
    def train(self, n_episodes: int = None, episode_length: int = None):
        """
        训练双智能体
        
        Args:
            n_episodes: 训练轮数
            episode_length: 每轮天数
        """
        n_episodes = n_episodes or TRAINING_CONFIG.n_episodes
        episode_length = episode_length or TRAINING_CONFIG.episode_length
        
        logger.info("="*60)
        logger.info(f"开始训练: {n_episodes}轮 × {episode_length}天")
        logger.info("="*60)
        
        for episode in range(n_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode+1}/{n_episodes}")
            logger.info(f"{'='*60}")
            
            # 运行一轮
            episode_metrics = self._run_episode(episode_length)
            
            # 记录历史
            self.training_history['episode_rewards_hotel'].append(episode_metrics['total_reward_hotel'])
            self.training_history['episode_rewards_ota'].append(episode_metrics['total_reward_ota'])
            self.training_history['episode_bookings_direct'].append(episode_metrics['total_bookings_direct'])
            self.training_history['episode_bookings_ota'].append(episode_metrics['total_bookings_ota'])
            self.training_history['episode_avg_occupancy'].append(episode_metrics['avg_occupancy'])
            self.training_history['epsilon_hotel'].append(self.hotel_agent.epsilon)
            self.training_history['epsilon_ota'].append(self.ota_agent.epsilon)
            
            # 更新探索率
            self.hotel_agent.update_epsilon()
            self.ota_agent.update_epsilon()
            
            # 打印进度
            logger.info(f"酒店收益: {episode_metrics['total_reward_hotel']:.2f}")
            logger.info(f"OTA收益: {episode_metrics['total_reward_ota']:.2f}")
            logger.info(f"直销订单: {episode_metrics['total_bookings_direct']}")
            logger.info(f"OTA订单: {episode_metrics['total_bookings_ota']}")
            logger.info(f"平均入住率: {episode_metrics['avg_occupancy']:.2%}")
            logger.info(f"探索率: Hotel={self.hotel_agent.epsilon:.3f}, OTA={self.ota_agent.epsilon:.3f}")
            
            # 定期保存
            if (episode + 1) % TRAINING_CONFIG.save_frequency == 0:
                self.save_agents(episode + 1)
            
            # 定期评估
            #if (episode + 1) % TRAINING_CONFIG.eval_frequency == 0:
            #    self.evaluate()
        
        logger.info("\n" + "="*60)
        logger.info("训练完成")
        logger.info("="*60)
        
        # 最终保存
        self.save_agents('final')
        self.save_training_history()
    
    def _run_episode(self, episode_length: int) -> Dict:
        """
        运行一轮训练
        
        Args:
            episode_length: 轮次长度（天数）
            
        Returns:
            轮次指标
        """
        # 重置环境
        state_info = self.env.reset()
        
        # 初始化统计
        total_reward_hotel = 0.0
        total_reward_ota = 0.0
        total_bookings_direct = 0
        total_bookings_ota = 0
        occupancy_rates = []
        
        # 运行episode_length天
        for day in tqdm(range(episode_length), desc="仿真进度"):
            # 1. 为未来15天决策
            hotel_actions = {}
            ota_actions = {}
            hotel_states = {}
            ota_states = {}
            
            for days_ahead in range(ENV_CONFIG.decision_horizon):
                # 酒店决策
                hotel_state_dict = state_info['states_hotel'][days_ahead]
                hotel_state = self.hotel_agent.discretize_state(
                    days_ahead=hotel_state_dict['days_ahead'],
                    inventory_usage=hotel_state_dict['inventory_usage'],
                    price_ratio=hotel_state_dict['price_ratio'],
                    is_weekend=hotel_state_dict['is_weekend'],
                    season=hotel_state_dict['season']
                )
                hotel_states[days_ahead] = hotel_state
                
                hotel_action = self.hotel_agent.choose_action(hotel_state, training=True)
                discount, commission_tier = self.hotel_agent.action_to_params(hotel_action)
                hotel_actions[days_ahead] = (discount, commission_tier)
                
                # OTA决策（观察到酒店的佣金档位）
                ota_state_dict = state_info['states_ota'][days_ahead]
                ota_state = self.ota_agent.discretize_state(
                    days_ahead=ota_state_dict['days_ahead'],
                    inventory_usage=ota_state_dict['inventory_usage'],
                    margin_ratio=ota_state_dict['margin_ratio'],
                    is_weekend=ota_state_dict['is_weekend'],
                    season=ota_state_dict['season']
                )
                ota_states[days_ahead] = ota_state
                
                ota_action = self.ota_agent.choose_action(ota_state, training=True)
                subsidy_coef = self.ota_agent.action_to_subsidy(ota_action)
                ota_actions[days_ahead] = subsidy_coef
            
            # 2. 执行动作
            next_state_info, rewards, done = self.env.step(hotel_actions, ota_actions)
            
            # 3. 更新Q表（更新所有15天的状态-动作对）
            # 奖励归一化：缩放到合理范围，避免Q值过大
            REWARD_SCALE = 10000.0
            hotel_reward_normalized = rewards['hotel'] / REWARD_SCALE
            ota_reward_normalized = rewards['ota'] / REWARD_SCALE
            
            # 使用时间折扣分配奖励：近期决策获得更多奖励，远期决策获得较少奖励
            # 这样可以正确反映时间因果关系
            gamma = HOTEL_AGENT_CONFIG.discount_factor
            total_weight = sum([gamma**d for d in range(ENV_CONFIG.decision_horizon)])
            
            for days_ahead in range(ENV_CONFIG.decision_horizon):
                # 计算该天的奖励权重（近期权重大，远期权重小）
                weight = (gamma**days_ahead) / total_weight
                reward_per_day_hotel = hotel_reward_normalized * weight
                reward_per_day_ota = ota_reward_normalized * weight
                
                # 更新酒店智能体Q表
                hotel_state = hotel_states[days_ahead]
                hotel_action = self.hotel_agent.params_to_action(*hotel_actions[days_ahead])
                
                # 获取下一个状态（如果存在）
                if days_ahead in next_state_info['states_hotel']:
                    next_hotel_state_dict = next_state_info['states_hotel'][days_ahead]
                    next_hotel_state = self.hotel_agent.discretize_state(
                        days_ahead=next_hotel_state_dict['days_ahead'],
                        inventory_usage=next_hotel_state_dict['inventory_usage'],
                        price_ratio=next_hotel_state_dict['price_ratio'],
                        is_weekend=next_hotel_state_dict['is_weekend'],
                        season=next_hotel_state_dict['season']
                    )
                else:
                    next_hotel_state = hotel_state  # 使用当前状态作为下一状态
                
                self.hotel_agent.update_q_table(
                    state=hotel_state,
                    action=hotel_action,
                    reward=reward_per_day_hotel,
                    next_state=next_hotel_state,
                    done=done
                )
                
                # 更新OTA智能体Q表
                ota_state = ota_states[days_ahead]
                ota_action = self.ota_agent.subsidy_to_action(ota_actions[days_ahead])
                
                if days_ahead in next_state_info['states_ota']:
                    next_ota_state_dict = next_state_info['states_ota'][days_ahead]
                    next_ota_state = self.ota_agent.discretize_state(
                        days_ahead=next_ota_state_dict['days_ahead'],
                        inventory_usage=next_ota_state_dict['inventory_usage'],
                        margin_ratio=next_ota_state_dict['margin_ratio'],
                        is_weekend=next_ota_state_dict['is_weekend'],
                        season=next_ota_state_dict['season']
                    )
                else:
                    next_ota_state = ota_state
                
                self.ota_agent.update_q_table(
                    state=ota_state,
                    action=ota_action,
                    reward=reward_per_day_ota,
                    next_state=next_ota_state,
                    done=done
                )
            
            # 4. 累计统计
            total_reward_hotel += rewards['hotel']
            total_reward_ota += rewards['ota']
            total_bookings_direct += rewards['bookings_direct']
            total_bookings_ota += rewards['bookings_ota']
            
            # 计算入住率
            inv_status = self.env.get_current_inventory_status()
            occupancy_rates.append(inv_status['avg_occupancy_15days'])
            
            # 5. 更新状态
            state_info = next_state_info
        
        return {
            'total_reward_hotel': total_reward_hotel,
            'total_reward_ota': total_reward_ota,
            'total_bookings_direct': total_bookings_direct,
            'total_bookings_ota': total_bookings_ota,
            'avg_occupancy': np.mean(occupancy_rates)
        }
    
    def evaluate(self, n_episodes: int = None):
        """
        评估当前策略
        
        Args:
            n_episodes: 评估轮数
        """
        n_episodes = n_episodes or TRAINING_CONFIG.eval_episodes
        
        logger.info(f"\n开始评估 ({n_episodes}轮)...")
        
        # 保存当前探索率
        hotel_epsilon_backup = self.hotel_agent.epsilon
        ota_epsilon_backup = self.ota_agent.epsilon
        
        # 设置为贪婪策略
        self.hotel_agent.epsilon = 0.0
        self.ota_agent.epsilon = 0.0
        
        eval_rewards_hotel = []
        eval_rewards_ota = []
        
        for _ in range(n_episodes):
            metrics = self._run_episode(TRAINING_CONFIG.episode_length)
            eval_rewards_hotel.append(metrics['total_reward_hotel'])
            eval_rewards_ota.append(metrics['total_reward_ota'])
        
        # 恢复探索率
        self.hotel_agent.epsilon = hotel_epsilon_backup
        self.ota_agent.epsilon = ota_epsilon_backup
        
        logger.info(f"评估结果:")
        logger.info(f"  酒店平均收益: {np.mean(eval_rewards_hotel):.2f} ± {np.std(eval_rewards_hotel):.2f}")
        logger.info(f"  OTA平均收益: {np.mean(eval_rewards_ota):.2f} ± {np.std(eval_rewards_ota):.2f}")
    
    def save_agents(self, suffix=''):
        """
        保存智能体
        
        Args:
            suffix: 文件名后缀
        """
        hotel_path = os.path.join(PATH_CONFIG.models_dir, f'hotel_agent_{suffix}.pkl')
        ota_path = os.path.join(PATH_CONFIG.models_dir, f'ota_agent_{suffix}.pkl')
        
        self.hotel_agent.save(hotel_path)
        self.ota_agent.save(ota_path)
        
        logger.info(f"智能体已保存: {suffix}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(PATH_CONFIG.results_dir, 'training_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"训练历史已保存: {history_path}")


if __name__ == '__main__':
    """测试训练器"""
    print("="*60)
    print("双智能体训练器测试")
    print("="*60)
    
    print("\n请使用 experiments/train.py 运行完整训练流程")
