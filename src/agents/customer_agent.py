#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABM客户行为模型 - 双智能体酒店动态定价系统
Customer Agent-Based Model

主要功能：
1. 客户生成（Type A/B，基于佣金的流量加权）
2. 客户决策（效用函数驱动的预订决策）
3. 渠道选择（直销 vs OTA）
4. 库存扣减和收益结算
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

# 导入配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import ABM_CONFIG, ENV_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerType(Enum):
    """客户类型枚举"""
    TYPE_A = "Traditional"  # 传统/忠诚型
    TYPE_B = "Digital"      # 数字/价格敏感型


class Channel(Enum):
    """预订渠道枚举"""
    DIRECT = "Direct"  # 直销
    OTA = "OTA"        # 分销
    NONE = "None"      # 未预订


@dataclass
class CustomerProfile:
    """客户画像"""
    customer_id: int
    customer_type: CustomerType  # Type A or Type B
    lead_time: int  # 提前期（天）
    target_date: int  # 目标入住日期（相对于当前日期的偏移）
    wtp: float  # 支付意愿（Willingness To Pay）
    price_sensitivity: float  # 价格敏感度 β


@dataclass
class BookingRecord:
    """预订记录"""
    customer_id: int
    customer_type: CustomerType
    channel: Channel  # 预订渠道
    target_date: int  # 入住日期
    price_paid: float  # 支付价格
    booking_date: int  # 预订日期


class HotelABM:
    """酒店ABM模型"""
    
    def __init__(self, preprocessor):
        """
        初始化ABM模型
        
        Args:
            preprocessor: 数据预处理器，包含价格表、到达率等信息
        """
        self.preprocessor = preprocessor
        self.customer_counter = 0  # 客户ID计数器
        
        logger.info("ABM模型初始化完成")
    
    def generate_customers(
        self, 
        current_date: int,
        month: int,
        commission_tier: int
    ) -> List[CustomerProfile]:
        """
        生成当日到达的客户
        
        Args:
            current_date: 当前日期（仿真日）
            month: 当前月份 (1-12)
            commission_tier: 酒店选择的佣金档位 (0-3)
            
        Returns:
            客户列表
        """
        # 1. 获取基础到达率
        lambda_base = self.preprocessor.get_arrival_rate(month)
        
        # 2. 获取流量加权系数
        boost = ABM_CONFIG.traffic_boost[commission_tier]
        
        # 3. 生成Type A客户（不受佣金影响）
        n_type_a = np.random.poisson(lambda_base * ABM_CONFIG.type_a_ratio)
        
        # 4. 生成Type B客户（受佣金影响）
        n_type_b = np.random.poisson(lambda_base * ABM_CONFIG.type_b_ratio * boost)
        
        # 5. 创建客户列表
        customers = []
        
        # 生成Type A客户
        for _ in range(n_type_a):
            customer = self._create_customer(
                current_date=current_date,
                month=month,
                customer_type=CustomerType.TYPE_A
            )
            customers.append(customer)
        
        # 生成Type B客户
        for _ in range(n_type_b):
            customer = self._create_customer(
                current_date=current_date,
                month=month,
                customer_type=CustomerType.TYPE_B
            )
            customers.append(customer)
        
        return customers
    
    def _create_customer(
        self,
        current_date: int,
        month: int,
        customer_type: CustomerType
    ) -> CustomerProfile:
        """
        创建单个客户
        
        Args:
            current_date: 当前日期
            month: 当前月份
            customer_type: 客户类型
            
        Returns:
            客户画像
        """
        # 1. 采样提前期（LogNormal分布）
        params = self.preprocessor.lead_time_params
        lead_time = int(np.random.lognormal(params['mu'], params['sigma']))
        lead_time = np.clip(lead_time, ABM_CONFIG.lead_time_min, ABM_CONFIG.lead_time_max)
        
        # 2. 计算目标入住日期
        target_date = current_date + lead_time
        
        # 3. 采样WTP（Normal分布）
        wtp_params = self.preprocessor.wtp_params[month]
        
        if customer_type == CustomerType.TYPE_A:
            wtp_mu = wtp_params['mu'] * ABM_CONFIG.wtp_type_a_multiplier
        else:
            wtp_mu = wtp_params['mu'] * ABM_CONFIG.wtp_type_b_multiplier
        
        wtp = np.random.normal(wtp_mu, wtp_params['sigma'])
        wtp = max(wtp, ABM_CONFIG.wtp_min)  # 确保WTP不低于最小值
        
        # 4. 采样价格敏感度（Uniform分布）
        price_sensitivity = np.random.uniform(ABM_CONFIG.beta_min, ABM_CONFIG.beta_max)
        
        # 5. 创建客户画像
        self.customer_counter += 1
        
        return CustomerProfile(
            customer_id=self.customer_counter,
            customer_type=customer_type,
            lead_time=lead_time,
            target_date=target_date,
            wtp=wtp,
            price_sensitivity=price_sensitivity
        )
    
    def customer_decision(
        self,
        customer: CustomerProfile,
        price_direct: float,
        price_ota: float,
        is_near_term: bool
    ) -> Tuple[Channel, float]:
        """
        客户预订决策
        
        Args:
            customer: 客户画像
            price_direct: 直销价格
            price_ota: OTA价格
            is_near_term: 是否为近期（L∈[0,14]）
            
        Returns:
            (预订渠道, 支付价格)
        """
        # 远期客户逻辑（L∈[15,180]）
        if not is_near_term:
            # 远期时价格相同：price_direct = price_ota = P_long
            if customer.customer_type == CustomerType.TYPE_A:
                # Type A选择直销
                utility = self._calculate_utility(
                    customer, price_direct, customer.lead_time
                )
                if utility > ABM_CONFIG.booking_threshold:
                    return Channel.DIRECT, price_direct
                else:
                    return Channel.NONE, 0.0
            else:
                # Type B默认选择OTA
                utility = self._calculate_utility(
                    customer, price_ota, customer.lead_time
                )
                if utility > ABM_CONFIG.booking_threshold:
                    return Channel.OTA, price_ota
                else:
                    return Channel.NONE, 0.0
        
        # 近期客户逻辑（L∈[0,14]）
        else:
            if customer.customer_type == CustomerType.TYPE_A:
                # Type A只关注直销
                utility_direct = self._calculate_utility(
                    customer, price_direct, customer.lead_time
                )
                
                if utility_direct > ABM_CONFIG.booking_threshold:
                    return Channel.DIRECT, price_direct
                else:
                    return Channel.NONE, 0.0
            
            else:
                # Type B比较两个渠道
                utility_direct = self._calculate_utility(
                    customer, price_direct, customer.lead_time
                )
                utility_ota = self._calculate_utility(
                    customer, price_ota, customer.lead_time
                )
                
                # 选择效用最大的渠道
                max_utility = max(utility_direct, utility_ota)
                
                if max_utility > ABM_CONFIG.booking_threshold:
                    if utility_ota >= utility_direct:
                        return Channel.OTA, price_ota
                    else:
                        return Channel.DIRECT, price_direct
                else:
                    return Channel.NONE, 0.0
    
    def _calculate_utility(
        self,
        customer: CustomerProfile,
        price: float,
        lead_time: int
    ) -> float:
        """
        计算效用函数
        
        U = (WTP - P) * β + γ/(L+1) + ε
        
        Args:
            customer: 客户画像
            price: 价格
            lead_time: 提前期
            
        Returns:
            效用值
        """
        # 价格效用
        price_utility = (customer.wtp - price) * customer.price_sensitivity
        
        # 紧迫效用
        urgency_utility = ABM_CONFIG.urgency_weight / (lead_time + 1)
        
        # 随机噪声
        noise = np.random.normal(0, ABM_CONFIG.noise_std)
        
        # 总效用
        utility = price_utility + urgency_utility + noise
        
        return utility
    
    def simulate_day(
        self,
        current_date: int,
        month: int,
        is_weekend: bool,
        prices_direct: Dict[int, float],  # {target_date: price}
        prices_ota: Dict[int, float],     # {target_date: price}
        commission_tier: int,
        inventory: np.ndarray,  # 库存数组 [180]
        current_inventory_offset: int = 0  # 当前日期在库存数组中的偏移
    ) -> Dict:
        """
        模拟一天的客户行为
        
        Args:
            current_date: 当前日期（仿真日）
            month: 当前月份
            is_weekend: 是否周末
            prices_direct: 直销价格字典（近期15天）
            prices_ota: OTA价格字典（近期15天）
            commission_tier: 佣金档位
            inventory: 库存数组
            current_inventory_offset: 当前日期在库存数组中的偏移
            
        Returns:
            模拟结果字典
        """
        # 1. 生成客户
        customers = self.generate_customers(current_date, month, commission_tier)
        
        # 2. 初始化统计
        bookings_direct = 0
        bookings_ota = 0
        revenue_direct = 0.0
        revenue_ota = 0.0
        booking_records = []
        
        # 3. 遍历客户进行决策
        for customer in customers:
            target_date = customer.target_date
            lead_time = customer.lead_time
            
            # 检查目标日期是否在库存窗口内
            inventory_idx = target_date - current_date + current_inventory_offset
            if inventory_idx < 0 or inventory_idx >= len(inventory):
                continue  # 超出库存窗口
            
            # 检查库存是否充足
            if inventory[inventory_idx] <= 0:
                continue  # 无库存
            
            # 判断是否为近期
            is_near_term = (lead_time <= 14)
            
            # 获取价格
            if is_near_term:
                # 近期：使用Agent决策的价格
                price_direct = prices_direct.get(target_date, 0)
                price_ota = prices_ota.get(target_date, 0)
            else:
                # 远期：使用规则价格P_long
                price_direct = self.preprocessor.get_price('p_long', month, is_weekend)
                price_ota = price_direct  # 远期价格相同
            
            # 客户决策
            channel, price_paid = self.customer_decision(
                customer, price_direct, price_ota, is_near_term
            )
            
            # 如果预订成功
            if channel != Channel.NONE:
                # 扣减库存
                inventory[inventory_idx] -= 1
                
                # 记录预订
                booking_records.append(BookingRecord(
                    customer_id=customer.customer_id,
                    customer_type=customer.customer_type,
                    channel=channel,
                    target_date=target_date,
                    price_paid=price_paid,
                    booking_date=current_date
                ))
                
                # 统计收益
                if channel == Channel.DIRECT:
                    bookings_direct += 1
                    revenue_direct += price_paid
                else:  # OTA
                    bookings_ota += 1
                    # OTA收益：酒店获得 P_base * (1 - commission%)
                    p_base = self.preprocessor.get_price('p_base', month, is_weekend)
                    commission_rate = ABM_CONFIG.commission_tiers[commission_tier]
                    revenue_ota += p_base * (1 - commission_rate)
        
        # 4. 返回结果
        return {
            'n_customers': len(customers),
            'n_type_a': sum(1 for c in customers if c.customer_type == CustomerType.TYPE_A),
            'n_type_b': sum(1 for c in customers if c.customer_type == CustomerType.TYPE_B),
            'bookings_direct': bookings_direct,
            'bookings_ota': bookings_ota,
            'revenue_direct': revenue_direct,
            'revenue_ota': revenue_ota,
            'total_revenue': revenue_direct + revenue_ota,
            'booking_records': booking_records
        }
    
    def get_demand_prediction(
        self,
        current_date: int,
        month: int,
        is_weekend: bool,
        prices_direct: Dict[int, float],
        prices_ota: Dict[int, float],
        commission_tier: int,
        inventory: np.ndarray,
        current_inventory_offset: int = 0,
        n_simulations: int = 100
    ) -> Dict:
        """
        通过蒙特卡洛模拟预测需求
        
        Args:
            (参数同simulate_day)
            n_simulations: 模拟次数
            
        Returns:
            需求预测结果（均值和方差）
        """
        revenues = []
        bookings_direct_list = []
        bookings_ota_list = []
        
        for _ in range(n_simulations):
            # 复制库存（避免修改原始库存）
            inv_copy = inventory.copy()
            
            result = self.simulate_day(
                current_date, month, is_weekend,
                prices_direct, prices_ota, commission_tier,
                inv_copy, current_inventory_offset
            )
            
            revenues.append(result['total_revenue'])
            bookings_direct_list.append(result['bookings_direct'])
            bookings_ota_list.append(result['bookings_ota'])
        
        return {
            'revenue_mean': np.mean(revenues),
            'revenue_std': np.std(revenues),
            'bookings_direct_mean': np.mean(bookings_direct_list),
            'bookings_ota_mean': np.mean(bookings_ota_list),
            'total_bookings_mean': np.mean(bookings_direct_list) + np.mean(bookings_ota_list)
        }


if __name__ == '__main__':
    """测试ABM模型"""
    print("="*60)
    print("ABM客户行为模型测试")
    print("="*60)
    
    # 这里需要先运行数据预处理才能测试
    print("\n请先运行数据预处理模块:")
    print("python src/data/preprocessing.py")
    print("\n然后可以测试ABM模型")
