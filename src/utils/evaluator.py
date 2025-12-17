#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模块 - 评估训练好的策略
Evaluation Module

功能：
1. 使用贪婪策略运行仿真
2. 收集详细的评估指标
3. 生成评估报告
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import ENV_CONFIG, PATH_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """策略评估器"""
    
    def __init__(self, env, hotel_agent, ota_agent):
        """
        初始化评估器
        
        Args:
            env: 环境
            hotel_agent: 酒店智能体
            ota_agent: OTA智能体
        """
        self.env = env
        self.hotel_agent = hotel_agent
        self.ota_agent = ota_agent
        
    def evaluate(self, n_episodes: int = 1, episode_length: int = 365) -> Dict:
        """
        评估策略
        
        Args:
            n_episodes: 评估轮数
            episode_length: 每轮天数
            
        Returns:
            评估结果字典
        """
        logger.info("="*60)
        logger.info(f"开始策略评估: {n_episodes}轮 × {episode_length}天")
        logger.info("="*60)
        
        # 保存原始探索率
        hotel_epsilon_backup = self.hotel_agent.epsilon
        ota_epsilon_backup = self.ota_agent.epsilon
        
        # 设置为贪婪策略
        self.hotel_agent.epsilon = 0.0
        self.ota_agent.epsilon = 0.0
        
        all_episodes_results = []
        
        for episode in range(n_episodes):
            logger.info(f"\n评估轮次 {episode+1}/{n_episodes}")
            episode_result = self._run_evaluation_episode(episode_length)
            all_episodes_results.append(episode_result)
        
        # 恢复探索率
        self.hotel_agent.epsilon = hotel_epsilon_backup
        self.ota_agent.epsilon = ota_epsilon_backup
        
        # 汇总结果
        summary = self._summarize_results(all_episodes_results)
        
        logger.info("\n" + "="*60)
        logger.info("评估完成")
        logger.info("="*60)
        self._print_summary(summary)
        
        return summary
    
    def _run_evaluation_episode(self, episode_length: int) -> Dict:
        """
        运行一轮评估
        
        Args:
            episode_length: 轮次长度
            
        Returns:
            轮次结果
        """
        # 重置环境
        state_info = self.env.reset()
        
        # 初始化记录
        daily_records = []
        total_revenue_hotel = 0.0
        total_revenue_ota = 0.0
        total_bookings_direct = 0
        total_bookings_ota = 0
        
        for day in tqdm(range(episode_length), desc="评估进度", leave=False):
            # 1. 为未来15天决策（贪婪策略）
            hotel_actions = {}
            ota_actions = {}
            
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
                
                hotel_action = self.hotel_agent.choose_action(hotel_state, training=False)
                discount, commission_tier = self.hotel_agent.action_to_params(hotel_action)
                hotel_actions[days_ahead] = (discount, commission_tier)
                
                # OTA决策
                ota_state_dict = state_info['states_ota'][days_ahead]
                ota_state = self.ota_agent.discretize_state(
                    days_ahead=ota_state_dict['days_ahead'],
                    inventory_usage=ota_state_dict['inventory_usage'],
                    margin_ratio=ota_state_dict['margin_ratio'],
                    is_weekend=ota_state_dict['is_weekend'],
                    season=ota_state_dict['season']
                )
                
                ota_action = self.ota_agent.choose_action(ota_state, training=False)
                subsidy_coef = self.ota_agent.action_to_subsidy(ota_action)
                ota_actions[days_ahead] = subsidy_coef
            
            # 2. 执行动作
            next_state_info, rewards, done = self.env.step(hotel_actions, ota_actions)
            
            # 3. 记录数据
            inv_status = self.env.get_current_inventory_status()
            
            daily_record = {
                'day': day,
                'revenue_hotel': rewards['hotel'],
                'revenue_ota': rewards['ota'],
                'bookings_direct': rewards['bookings_direct'],
                'bookings_ota': rewards['bookings_ota'],
                'avg_occupancy': inv_status['avg_occupancy_15days'],
                'inventory_day0': inv_status['available_rooms'][0] if len(inv_status['available_rooms']) > 0 else 0,
                'inventory_day1': inv_status['available_rooms'][1] if len(inv_status['available_rooms']) > 1 else 0,
                'inventory_day2': inv_status['available_rooms'][2] if len(inv_status['available_rooms']) > 2 else 0,
                'inventory_day3': inv_status['available_rooms'][3] if len(inv_status['available_rooms']) > 3 else 0,
                'inventory_day4': inv_status['available_rooms'][4] if len(inv_status['available_rooms']) > 4 else 0,
            }
            daily_records.append(daily_record)
            
            # 4. 累计统计
            total_revenue_hotel += rewards['hotel']
            total_revenue_ota += rewards['ota']
            total_bookings_direct += rewards['bookings_direct']
            total_bookings_ota += rewards['bookings_ota']
            
            # 5. 更新状态
            state_info = next_state_info
        
        return {
            'daily_records': daily_records,
            'total_revenue_hotel': total_revenue_hotel,
            'total_revenue_ota': total_revenue_ota,
            'total_bookings_direct': total_bookings_direct,
            'total_bookings_ota': total_bookings_ota,
            'avg_occupancy': np.mean([r['avg_occupancy'] for r in daily_records])
        }
    
    def _summarize_results(self, all_episodes_results: List[Dict]) -> Dict:
        """
        汇总多轮评估结果
        
        Args:
            all_episodes_results: 所有轮次的结果
            
        Returns:
            汇总结果
        """
        # 提取指标
        revenues_hotel = [r['total_revenue_hotel'] for r in all_episodes_results]
        revenues_ota = [r['total_revenue_ota'] for r in all_episodes_results]
        bookings_direct = [r['total_bookings_direct'] for r in all_episodes_results]
        bookings_ota = [r['total_bookings_ota'] for r in all_episodes_results]
        occupancies = [r['avg_occupancy'] for r in all_episodes_results]
        
        # 合并所有daily_records
        all_daily_records = []
        for episode_result in all_episodes_results:
            all_daily_records.extend(episode_result['daily_records'])
        
        results_df = pd.DataFrame(all_daily_records)
        
        return {
            'n_episodes': len(all_episodes_results),
            'revenue_hotel_mean': np.mean(revenues_hotel),
            'revenue_hotel_std': np.std(revenues_hotel),
            'revenue_ota_mean': np.mean(revenues_ota),
            'revenue_ota_std': np.std(revenues_ota),
            'bookings_direct_mean': np.mean(bookings_direct),
            'bookings_direct_std': np.std(bookings_direct),
            'bookings_ota_mean': np.mean(bookings_ota),
            'bookings_ota_std': np.std(bookings_ota),
            'occupancy_mean': np.mean(occupancies),
            'occupancy_std': np.std(occupancies),
            'results_df': results_df,
            'all_episodes_results': all_episodes_results
        }
    
    def _print_summary(self, summary: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        print(f"评估轮数: {summary['n_episodes']}")
        print(f"\n酒店收益:")
        print(f"  平均: ${summary['revenue_hotel_mean']:.2f}")
        print(f"  标准差: ${summary['revenue_hotel_std']:.2f}")
        print(f"\nOTA收益:")
        print(f"  平均: ${summary['revenue_ota_mean']:.2f}")
        print(f"  标准差: ${summary['revenue_ota_std']:.2f}")
        print(f"\n直销订单:")
        print(f"  平均: {summary['bookings_direct_mean']:.1f}")
        print(f"  标准差: {summary['bookings_direct_std']:.1f}")
        print(f"\nOTA订单:")
        print(f"  平均: {summary['bookings_ota_mean']:.1f}")
        print(f"  标准差: {summary['bookings_ota_std']:.1f}")
        print(f"\n平均入住率:")
        print(f"  平均: {summary['occupancy_mean']:.2%}")
        print(f"  标准差: {summary['occupancy_std']:.2%}")
        print("="*60)
    
    def save_results(self, summary: Dict, filename: str = None):
        """
        保存评估结果
        
        Args:
            summary: 评估摘要
            filename: 文件名
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'evaluation_results_{timestamp}.csv'
        
        filepath = os.path.join(PATH_CONFIG.results_dir, filename)
        summary['results_df'].to_csv(filepath, index=False)
        logger.info(f"评估结果已保存: {filepath}")
        
        return filepath


if __name__ == '__main__':
    """测试评估器"""
    print("="*60)
    print("策略评估器测试")
    print("="*60)
    print("\n请使用训练脚本中的评估功能")
