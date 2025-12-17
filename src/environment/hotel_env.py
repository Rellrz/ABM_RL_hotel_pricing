#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店环境 - 双智能体强化学习
Hotel Environment for Dual-Agent RL

主要功能：
1. 180天滚动库存管理
2. 15天决策窗口（Day 0-14）
3. 顺序博弈：酒店先决策 → OTA观察后决策 → ABM模拟
4. 状态计算和奖励计算
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import ENV_CONFIG, ABM_CONFIG
from src.agents.customer_agent import HotelABM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotelEnvironment:
    """酒店环境"""
    
    def __init__(self, preprocessor, start_date: str = '2017-01-01'):
        """
        初始化环境
        
        Args:
            preprocessor: 数据预处理器
            start_date: 仿真开始日期
        """
        self.preprocessor = preprocessor
        self.abm = HotelABM(preprocessor)
        
        # 时间管理
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.current_date = self.start_date
        self.current_day = 0  # 仿真日计数器
        
        # 库存管理：180天滚动窗口
        self.inventory = np.full(ENV_CONFIG.inventory_horizon, ENV_CONFIG.total_rooms, dtype=int)
        
        # 价格缓存（用于状态计算）
        self.last_prices_direct = {}  # {target_date: price}
        self.last_prices_ota = {}
        self.last_commission_tier = 0
        
        logger.info(f"环境初始化: 开始日期={start_date}, 总房间数={ENV_CONFIG.total_rooms}")
    
    def reset(self) -> Dict:
        """
        重置环境
        
        Returns:
            初始状态信息
        """
        self.current_date = self.start_date
        self.current_day = 0
        self.inventory = np.full(ENV_CONFIG.inventory_horizon, ENV_CONFIG.total_rooms, dtype=int)
        self.last_prices_direct = {}
        self.last_prices_ota = {}
        self.last_commission_tier = 0
        
        # 返回初始状态信息
        return self._get_state_info()
    
    def step(
        self,
        hotel_actions: Dict[int, Tuple[float, int]],  # {days_ahead: (discount, commission_tier)}
        ota_actions: Dict[int, float]  # {days_ahead: subsidy_coef}
    ) -> Tuple[Dict, Dict, bool]:
        """
        执行一步仿真
        
        Args:
            hotel_actions: 酒店动作字典，为未来15天决策
            ota_actions: OTA动作字典，为未来15天决策
            
        Returns:
            (state_info, rewards, done)
        """
        # 1. 获取当前日期信息
        month = self.current_date.month
        weekday = self.current_date.weekday()
        is_weekend = weekday in ENV_CONFIG.weekend_days
        
        # 2. 计算价格（为未来15天）
        prices_direct = {}
        prices_ota = {}
        
        for days_ahead in range(ENV_CONFIG.decision_horizon):
            target_date_offset = days_ahead
            target_date_abs = self.current_day + days_ahead
            
            # 获取基准价格P_base
            p_base = self.preprocessor.get_price('p_base', month, is_weekend)
            
            # 酒店决策
            if days_ahead in hotel_actions:
                discount, commission_tier = hotel_actions[days_ahead]
            else:
                # 默认动作
                discount, commission_tier = 1.0, 0
            
            # 直销价格
            p_direct = p_base * discount
            prices_direct[target_date_abs] = p_direct
            
            # OTA决策
            if days_ahead in ota_actions:
                subsidy_coef = ota_actions[days_ahead]
            else:
                subsidy_coef = 0.0
            
            # OTA价格
            commission_rate = ABM_CONFIG.commission_tiers[commission_tier] 
            commission = p_base * commission_rate
            subsidy = commission * subsidy_coef
            p_ota = p_base - subsidy
            prices_ota[target_date_abs] = p_ota
        
        # 3. ABM模拟当日客户行为
        commission_tier = hotel_actions.get(0, (1.0, 0))[1]
        subsidy_coef = ota_actions.get(0, 0.0)
        
        abm_result = self.abm.simulate_day(
            current_date=self.current_day,
            month=month,
            is_weekend=is_weekend,
            prices_direct=prices_direct,
            prices_ota=prices_ota,
            commission_tier=commission_tier,
            inventory=self.inventory,
            current_inventory_offset=0
        )
        
        # 4. 计算奖励
        p_base = self.preprocessor.get_price('p_base', month, is_weekend)
        rewards = {
            'hotel': abm_result['total_revenue'],  
            'ota': self._calculate_ota_reward(abm_result, commission_tier, subsidy_coef, p_base),
            'bookings_direct': abm_result['bookings_direct'],
            'bookings_ota': abm_result['bookings_ota']
        }
        
        # 5. 更新状态
        self._advance_day()
        
        # 6. 缓存价格（用于下一步状态计算）
        self.last_prices_direct = prices_direct
        self.last_prices_ota = prices_ota
        self.last_commission_tier = commission_tier
        
        # 7. 获取新状态
        state_info = self._get_state_info()
        
        # 8. 检查是否结束
        done = False  # 由外部训练循环控制
        
        return state_info, rewards, done
    
    def _calculate_ota_reward(self, abm_result: Dict, commission_tier: int, subsidy_coef: float, p_base: float) -> float:
        """
        计算OTA奖励
        
        R_ota = (Commission - Subsidy) × Sales_ota
        
        Args:
            abm_result: ABM模拟结果
            commission_tier: 佣金档位
            subsidy_coef: 补贴系数（OTA的实际动作）
            p_base: 基准价格
            
        Returns:
            OTA收益
        """
        commission_rate = ABM_CONFIG.commission_tiers[commission_tier]
        commission_per_booking = p_base * commission_rate
        subsidy_per_booking = commission_per_booking * subsidy_coef
        
        net_profit_per_booking = commission_per_booking - subsidy_per_booking
        ota_reward = abm_result['bookings_ota'] * net_profit_per_booking
        
        return max(ota_reward, 0.0)
    
    def _advance_day(self):
        """推进一天"""
        # 1. 滚动库存窗口
        self.inventory = np.roll(self.inventory, -1)
        self.inventory[-1] = ENV_CONFIG.total_rooms  # 最远一天补充新库存
        
        # 2. 更新日期
        self.current_day += 1
        self.current_date += timedelta(days=1)
    
    def _get_state_info(self) -> Dict:
        """
        获取状态信息（为未来15天计算状态）
        
        Returns:
            状态信息字典
        """
        month = self.current_date.month
        weekday = self.current_date.weekday()
        is_weekend = weekday in ENV_CONFIG.weekend_days
        season = ENV_CONFIG.get_season(month)
        
        state_info = {
            'month': month,
            'weekday': weekday,
            'is_weekend': is_weekend,
            'season': season,
            'current_day': self.current_day,
            'inventory': self.inventory.copy(),
            'states_hotel': {},  # {days_ahead: state_index}
            'states_ota': {}
        }
        
        # 为未来15天计算状态
        for days_ahead in range(ENV_CONFIG.decision_horizon):
            # 库存使用率
            inv_idx = days_ahead
            if inv_idx < len(self.inventory):
                inventory_used = ENV_CONFIG.total_rooms - self.inventory[inv_idx]
                inventory_usage = inventory_used / ENV_CONFIG.total_rooms
            else:
                inventory_usage = 0.0
            
            # 价格比（用于竞争状态）
            target_date_abs = self.current_day + days_ahead
            p_direct_last = self.last_prices_direct.get(target_date_abs, 100.0)
            p_ota_last = self.last_prices_ota.get(target_date_abs, 100.0)
            price_ratio = p_ota_last / (p_direct_last + 1e-8)
            
            # 套利空间（用于OTA状态）
            p_base = self.preprocessor.get_price('p_base', month, is_weekend)
            commission_rate = ABM_CONFIG.commission_tiers[self.last_commission_tier]
            p_wholesale = p_base * (1 - commission_rate)
            margin_ratio = p_direct_last / (p_wholesale + 1e-8)
            
            # 存储状态信息
            state_info['states_hotel'][days_ahead] = {
                'days_ahead': days_ahead,
                'inventory_usage': inventory_usage,
                'price_ratio': price_ratio,
                'is_weekend': is_weekend,
                'season': season
            }
            
            state_info['states_ota'][days_ahead] = {
                'days_ahead': days_ahead,
                'inventory_usage': inventory_usage,
                'margin_ratio': margin_ratio,
                'is_weekend': is_weekend,
                'season': season
            }
        
        return state_info
    
    def get_current_inventory_status(self) -> Dict:
        """
        获取当前库存状态
        
        Returns:
            库存统计信息
        """
        return {
            'total_rooms': ENV_CONFIG.total_rooms,
            'available_rooms': self.inventory.copy(),
            'occupancy_rate': 1.0 - (self.inventory / ENV_CONFIG.total_rooms),
            'avg_occupancy_15days': np.mean(1.0 - (self.inventory[:15] / ENV_CONFIG.total_rooms))
        }


if __name__ == '__main__':
    """测试环境"""
    print("="*60)
    print("酒店环境测试")
    print("="*60)
    
    print("\n请先运行数据预处理:")
    print("python src/data/preprocessing.py")
    print("\n然后可以测试环境模块")
