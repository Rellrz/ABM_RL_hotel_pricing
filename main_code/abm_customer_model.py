#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABM客户行为模型 - 基于Mesa框架
用于替代NGBoost需求预测模块

主要功能：
1. 客户生成模块：基于泊松分布生成每日潜在客户
2. 客户决策模块：基于效用函数的预订决策
3. 客户取消模块：动态持有效用评估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import warnings
from config import ABM_CONFIG
warnings.filterwarnings('ignore')


@dataclass
class CustomerProfile:
    """客户特征配置文件"""
    lead_time: int              # 提前预订期（天）
    target_date: int            # 目标入住日期（仿真日期）
    wtp: float                  # 最高支付意愿（Willingness To Pay）
    price_sensitivity: float    # 价格敏感度系数 β
    customer_type: str          # 客户类型：'online' 或 'offline'


@dataclass
class BookingRecord:
    """预订记录"""
    customer_id: int
    booking_date: int           # 预订日期
    target_date: int            # 入住日期
    paid_price: float           # 成交价格
    wtp: float                  # 支付意愿
    price_sensitivity: float    # 价格敏感度
    customer_type: str          # 客户类型
    is_canceled: bool = False   # 是否已取消


class CustomerAgent(Agent):
    """
    客户智能体
    
    每个客户代表一个潜在的酒店预订者，具有独特的特征和决策逻辑。
    """
    
    def __init__(self, unique_id: int, model: 'HotelABMModel', profile: CustomerProfile):
        """
        初始化客户智能体
        
        Args:
            unique_id: 唯一标识符
            model: ABM模型实例
            profile: 客户特征配置
        """
        super().__init__(unique_id, model)
        self.profile = profile
        self.has_booked = False
        self.booking_record: Optional[BookingRecord] = None
        
    def evaluate_booking_utility(self, price: float, current_day: int) -> float:
        """
        评估预订效用
        
        基于效用函数计算客户的预订意愿：
        U_score = (WTP - P) * β + γ/(L+1) + ε
        
        Args:
            price: 酒店当前报价
            current_day: 当前仿真日期
            
        Returns:
            效用得分
        """
        # 如果价格超过支付意愿，直接返回负无穷
        if price > self.profile.wtp:
            return -np.inf
        
        # 经济盈余效用：(WTP - P) * β
        economic_surplus = (self.profile.wtp - price) * self.profile.price_sensitivity
        
        # 紧迫性效用：γ/(L+1)
        # 提前期越短，紧迫性越高
        gamma = self.model.params['urgency_weight']
        urgency_utility = gamma / (self.profile.lead_time + 1)
        
        # 随机噪声：模拟非理性因素
        noise_std = self.model.params['noise_std']
        noise = np.random.normal(0, noise_std)
        
        # 总效用
        utility = economic_surplus + urgency_utility + noise
        
        return utility
    
    def make_booking_decision(self, price: float, current_day: int) -> bool:
        """
        做出预订决策
        
        Args:
            price: 酒店报价
            current_day: 当前日期
            
        Returns:
            是否预订
        """
        if self.has_booked:
            return False
        
        # 计算效用
        utility = self.evaluate_booking_utility(price, current_day)
        
        # 决策阈值
        threshold = self.model.params['booking_threshold']
        
        # 做出决策
        if utility > threshold:
            self.has_booked = True
            self.booking_record = BookingRecord(
                customer_id=self.unique_id,
                booking_date=current_day,
                target_date=self.profile.target_date,
                paid_price=price,
                wtp=self.profile.wtp,
                price_sensitivity=self.profile.price_sensitivity,
                customer_type=self.profile.customer_type
            )
            return True
        
        return False
    
    def evaluate_holding_utility(self, current_price: float, days_until_checkin: int) -> float:
        """
        评估持有订单的效用（用于取消决策）
        
        U_hold = (WTP - P_paid) - β * max(0, P_paid - P_curr) + γ/(d+1) + ξ
        
        Args:
            current_price: 当前酒店报价
            days_until_checkin: 距离入住的剩余天数
            
        Returns:
            持有效用得分
        """
        if not self.has_booked or self.booking_record is None:
            return 0.0
        
        # 原始满意度：(WTP - P_paid)
        satisfaction = self.profile.wtp - self.booking_record.paid_price
        
        # 价格后悔：β * max(0, P_paid - P_curr)
        regret_coef = self.model.params['regret_coefficient']
        price_regret = regret_coef * max(0, self.booking_record.paid_price - current_price)
        
        # 临近承诺效应：γ/(d+1)
        commitment_weight = self.model.params['commitment_weight']
        commitment_utility = commitment_weight / (days_until_checkin + 1)
        
        # 每日随机冲击
        shock_std = self.model.params['shock_std']
        daily_shock = np.random.normal(0, shock_std)
        
        # 总持有效用
        holding_utility = satisfaction - price_regret + commitment_utility + daily_shock
        
        return holding_utility
    
    def evaluate_cancellation(self, current_price: float, current_day: int) -> bool:
        """
        评估是否取消订单
        
        Args:
            current_price: 当前酒店报价
            current_day: 当前日期
            
        Returns:
            是否取消
        """
        if not self.has_booked or self.booking_record is None or self.booking_record.is_canceled:
            return False
        
        # 计算距离入住的天数
        days_until_checkin = self.booking_record.target_date - current_day
        
        # 如果已经是入住日或过期，不取消
        if days_until_checkin <= 0:
            return False
        
        # 计算持有效用
        holding_utility = self.evaluate_holding_utility(current_price, days_until_checkin)
        
        # 如果持有效用为负，取消订单
        if holding_utility < 0:
            self.booking_record.is_canceled = True
            return True
        
        return False
    
    def step(self):
        """
        智能体的每步行为（由Mesa框架调用）
        """
        pass


class HotelABMModel(Model):
    """
    酒店ABM模型
    
    模拟酒店预订环境，包括客户生成、决策和取消行为。
    """
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 params: Optional[Dict] = None,
                 random_seed: Optional[int] = None):
        """
        初始化ABM模型
        
        Args:
            historical_data: 历史预订数据（hotel_bookings.csv）
            params: 模型参数字典
            random_seed: 随机种子
        """
        super().__init__()
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            self.random.seed(random_seed)
        
        # 存储历史数据
        self.historical_data = historical_data
        
        # 设置默认参数
        self.params = {
            # 客户生成参数
            'monthly_arrival_rates': self._calculate_monthly_arrival_rates(),
            'lead_time_params': self._fit_lead_time_distribution(),
            'wtp_params': self._fit_wtp_distribution(),
            
            # 决策参数
            'urgency_weight': 7.5,          # γ: 紧迫权重
            'noise_std': 12.0,              # σ_noise: 决策噪声
            'booking_threshold': 0.0,       # U_threshold: 预订阈值
            
            # 取消参数
            'regret_coefficient': 0.75,     # β: 后悔系数
            'commitment_weight': 8.0,       # γ: 承诺权重
            'shock_std': 15.0,              # σ_shock: 每日冲击
            
            # 价格敏感度参数
            'beta_base': 1.0,               # 基础敏感度
            'beta_range': (0.8, 1.2),       # 敏感度范围
        }
        
        # 更新用户自定义参数
        if params:
            self.params.update(params)
        
        # 初始化调度器，随机策略
        self.schedule = RandomActivation(self)
        
        # 当前仿真日期
        self.current_day = 0
        
        # ✅ 每日可用库存字典（由HotelEnvironment同步）
        from collections import defaultdict
        self.daily_available_rooms = defaultdict(lambda: self.params.get('max_inventory', 226))
        
        # ✅ 价格窗口（由HotelEnvironment同步）
        self.price_window_online = [100.0] * 5  # 未来5天的线上价格
        self.price_window_offline = [120.0] * 5  # 未来5天的线下价格
        self.booking_window_days = 5  # 预订窗口：只能预订未来5天
        
        # 活跃预订记录（未取消且未入住）
        self.active_bookings: List[BookingRecord] = []
        
        # 历史记录
        self.booking_history: List[BookingRecord] = []
        self.daily_stats: List[Dict] = []
        
        # 数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "total_customers": lambda m: m.schedule.get_agent_count(),
                "total_bookings": lambda m: len([b for b in m.booking_history if not b.is_canceled]),
                "total_cancellations": lambda m: len([b for b in m.booking_history if b.is_canceled]),
                "active_bookings": lambda m: len(m.active_bookings),
            }
        )
    
    def _calculate_monthly_arrival_rates(self) -> Dict[int, float]:
        """
        从历史数据计算每月的日均到达率 λ_m
        
        Returns:
            月份 -> 日均到达率的字典
        """
        df = self.historical_data.copy()
        
        # 确保有arrival_date_month列
        if 'arrival_date_month' not in df.columns:
            return {m: 100.0 for m in range(1, 13)}  # 默认值
        
        # 月份映射
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        # 转换月份
        df['month_num'] = df['arrival_date_month'].map(month_map)
        
        # 计算每月的预订量
        monthly_counts = df.groupby('month_num').size()
        
        # 假设每月平均30天，计算日均到达率
        monthly_rates = {}
        for month in range(1, 13):
            if month in monthly_counts.index:
                # 日均到达率 = 月总预订量 / 30
                monthly_rates[month] = monthly_counts[month] / 30.0
            else:
                monthly_rates[month] = 100.0  # 默认值
        
        return monthly_rates
    
    def _fit_lead_time_distribution(self) -> Dict[str, float]:
        """
        拟合提前期分布（指数分布）
        
        Returns:
            分布参数字典
        """
        if 'lead_time' not in self.historical_data.columns:
            return {'mean': 104.0}  # 默认值
        
        lead_times = self.historical_data['lead_time'].dropna()
        lead_times = lead_times[lead_times >= 0]  # 过滤负值
        
        if len(lead_times) == 0:
            return {'mean': 104.0}
        
        mean_lead_time = float(lead_times.mean())
        
        return {'mean': mean_lead_time}
    
    def _fit_wtp_distribution(self) -> Dict[str, float]:
        """
        拟合支付意愿分布（正态分布）
        
        基于未取消订单的ADR（平均日房价）
        
        Returns:
            分布参数字典
        """
        if 'adr' not in self.historical_data.columns:
            return {'mean': 100.0, 'std': 30.0}  # 默认值
        
        # 只使用未取消的订单
        df = self.historical_data.copy()
        if 'is_canceled' in df.columns:
            df = df[df['is_canceled'] == 0]
        
        adr_values = df['adr'].dropna()
        adr_values = adr_values[(adr_values > 0) & (adr_values < 500)]  # 过滤异常值
        
        if len(adr_values) == 0:
            return {'mean': 100.0, 'std': 30.0}
        
        mean_adr = float(adr_values.mean())
        std_adr = float(adr_values.std())
        
        return {'mean': mean_adr, 'std': std_adr}
    
    def generate_daily_customers(self, current_day: int) -> List[CustomerAgent]:
        """
        生成当日的潜在客户
        
        Args:
            current_day: 当前仿真日期
            
        Returns:
            客户智能体列表
        """
        # 确定当前月份（简化：假设每月30天）
        month = (current_day // 30) % 12 + 1
        
        # 获取该月的到达率 λ_m
        lambda_m = self.params['monthly_arrival_rates'].get(month, 100.0)
        
        # 从泊松分布采样当日客户数量
        num_customers = np.random.poisson(lambda_m)
        
        # 生成客户
        customers = []
        for i in range(num_customers):
            # 生成唯一ID
            customer_id = self.next_id()
            
            # 生成客户特征
            profile = self._generate_customer_profile(current_day)
            
            # 创建客户智能体
            customer = CustomerAgent(customer_id, self, profile)
            customers.append(customer)
            
            # 添加到调度器
            self.schedule.add(customer)
        
        return customers
    
    def _generate_customer_profile(self, current_day: int) -> CustomerProfile:
        """
        生成单个客户的特征向量
        
        Args:
            current_day: 当前日期
            
        Returns:
            客户特征配置
        """
        # 1. 提前预订期 L_i ~ Exp(μ_lead)
        #lead_time_mean = self.params['lead_time_params']['mean']
        #lead_time = int(np.random.exponential(lead_time_mean))
        #lead_time = max(0, min(lead_time, 365))  # 限制在0-365天
        # ✅ 1. 提前预订期 L_i：限制在预订窗口内（0-4天，即今天到第5天）
        # 原来：可以预订未来365天
        # 现在：只能预订未来5天（包括今天）
        lead_time = np.random.randint(0, self.booking_window_days)  # 0, 1, 2, 3, 4
        
        # 2. 目标入住日期 T_stay = CurrentDate + L
        target_date = current_day + lead_time
        
        # 3. 最高支付意愿 WTP_i ~ Normal(μ_adr, σ_adr)
        wtp_mean = self.params['wtp_params']['mean']
        wtp_std = self.params['wtp_params']['std']
        wtp = np.random.normal(wtp_mean, wtp_std)
        wtp = max(10.0, wtp)  # 确保不低于最低价
        
        # 4. 价格敏感度 β_i
        # 方案1：基于提前期的关联性
        # β_i = β_base + α * log(1 + L_i) + ε
        # 方案2：简单均匀分布（当前使用）
        beta_min, beta_max = self.params['beta_range']
        price_sensitivity = np.random.uniform(beta_min, beta_max)
        
        # 5. 客户类型（线上/线下）
        # 根据历史数据比例随机分配
        customer_type = np.random.choice(['online', 'offline'], p=[0.6, 0.4])
        
        return CustomerProfile(
            lead_time=lead_time,
            target_date=target_date,
            wtp=wtp,
            price_sensitivity=price_sensitivity,
            customer_type=customer_type
        )
    
    def simulate_day(self, 
                     price_online: float, 
                     price_offline: float,
                     max_inventory: int = 226) -> Dict:
        """
        模拟一天的酒店运营
        
        Args:
            price_online: 线上渠道报价
            price_offline: 线下渠道报价
            max_inventory: 最大库存
            
        Returns:
            当日统计数据
        """
        # 生成当日客户
        daily_customers = self.generate_daily_customers(self.current_day)
        
        # 统计变量
        new_bookings_online = 0
        new_bookings_offline = 0
        cancellations = 0
        
        # 客户决策阶段
        for customer in daily_customers:
            target_date = customer.profile.target_date
            days_ahead = target_date - self.current_day  # 0, 1, 2, 3, 4
            
            # ✅ 根据客户的target_date从价格窗口中选择对应的价格
            if 0 <= days_ahead < len(self.price_window_online):
                if customer.profile.customer_type == 'online':
                    price = self.price_window_online[days_ahead]
                else:
                    price = self.price_window_offline[days_ahead]
            else:
                # 超出预订窗口，跳过该客户
                continue
            
            # 做出预订决策
            #if customer.make_booking_decision(price, self.current_day):
            #    target_date = customer.booking_record.target_date
                
                # ✅ 正确的库存检查：检查目标日期的库存
            #    if self.daily_available_rooms[target_date] > 0:
                    # ✅ 扣减该日期的库存
            #        self.daily_available_rooms[target_date] -= 1
                    
            #        self.active_bookings.append(customer.booking_record)
            #        self.booking_history.append(customer.booking_record)
                    
            #        if customer.profile.customer_type == 'online':
            #            new_bookings_online += 1
            #        else:
            #            new_bookings_offline += 1
                # else: 该日期已满房，拒绝预订
            # 直接在simulate_day中创建booking_record，跳过效用函数
            if not customer.has_booked and self.daily_available_rooms[target_date] > 0:
                customer.has_booked = True
                booking_record = BookingRecord(
                    customer_id=customer.unique_id,
                    booking_date=self.current_day,
                    target_date=target_date,
                    paid_price=price,
                    wtp=customer.profile.wtp,
                    price_sensitivity=customer.profile.price_sensitivity,
                    customer_type=customer.profile.customer_type)
                customer.booking_record = booking_record
                
                target_date = customer.booking_record.target_date
                
                # ✅ 正确的库存检查：检查目标日期的库存
                if self.daily_available_rooms[target_date] > 0:
                # ✅ 扣减该日期的库存
                    self.daily_available_rooms[target_date] -= 1
                    
                    self.active_bookings.append(customer.booking_record)
                    self.booking_history.append(customer.booking_record)
                    
                    if customer.profile.customer_type == 'online':
                        new_bookings_online += 1
                    else:
                        new_bookings_offline += 1
                # else: 该日期已满房，拒绝预订

        
        # 取消评估阶段：遍历所有活跃预订
        cancellation_refund = 0.0  # 记录取消订单的退款总额
        active_bookings_copy = self.active_bookings.copy()
        for booking in active_bookings_copy:
            # 找到对应的客户（如果还在系统中）
            # 简化：直接基于预订记录评估
            days_until_checkin = booking.target_date - self.current_day
            
            if days_until_checkin <= 0:
                # 已入住，移除活跃预订
                self.active_bookings.remove(booking)
                continue
            
            # ✅ 评估取消：使用价格窗口中对应日期的价格
            days_ahead = booking.target_date - self.current_day
            if 0 <= days_ahead < len(self.price_window_online):
                if booking.customer_type == 'online':
                    current_price = self.price_window_online[days_ahead]
                else:
                    current_price = self.price_window_offline[days_ahead]
            else:
                # 超出窗口，使用默认价格
                current_price = price_online if booking.customer_type == 'online' else price_offline
            
            # 计算持有效用
            satisfaction = booking.wtp - booking.paid_price
            regret_coef = self.params['regret_coefficient']
            price_regret = regret_coef * max(0, booking.paid_price - current_price)
            commitment_weight = self.params['commitment_weight']
            commitment_utility = commitment_weight / (days_until_checkin + 1)
            shock_std = self.params['shock_std']
            daily_shock = np.random.normal(0, shock_std)
            
            holding_utility = satisfaction - price_regret + commitment_utility + daily_shock
            
            # 取消决策
            if holding_utility < 0 and not booking.is_canceled:
                booking.is_canceled = True
                # ✅ 释放该日期的库存
                self.daily_available_rooms[booking.target_date] += 1
                # ✅ 可退款政策：记录退款金额
                cancellation_refund += booking.paid_price
                self.active_bookings.remove(booking)
                cancellations += 1
        
        # 记录当日统计
        gross_revenue = new_bookings_online * price_online + new_bookings_offline * price_offline
        net_revenue = gross_revenue - cancellation_refund  # ✅ 净收益 = 新预订收益 - 取消退款
        
        daily_stat = {
            'day': self.current_day,
            'price_online': price_online,
            'price_offline': price_offline,
            'new_customers': len(daily_customers),
            'new_bookings_online': new_bookings_online,
            'new_bookings_offline': new_bookings_offline,
            'total_new_bookings': new_bookings_online + new_bookings_offline,
            'cancellations': cancellations,
            'cancellation_refund': cancellation_refund,  # ✅ 记录退款金额
            'active_bookings': len(self.active_bookings),
            'revenue_online': new_bookings_online * price_online,
            'revenue_offline': new_bookings_offline * price_offline,
            'gross_revenue': gross_revenue,  # ✅ 毛收益（新预订）
            'total_revenue': net_revenue,  # ✅ 净收益（扣除退款后）
        }
        
        self.daily_stats.append(daily_stat)
        
        # 更新仿真日期
        self.current_day += 1
        
        # 收集数据
        self.datacollector.collect(self)
        
        return daily_stat
    
    def get_demand_prediction(self, 
                             price_online: float, 
                             price_offline: float,
                             num_simulations: int = 10) -> Tuple[float, float]:
        """
        获取需求预测（替代NGBoost的接口）
        
        通过多次蒙特卡洛模拟估计需求的均值和方差
        
        Args:
            price_online: 线上价格
            price_offline: 线下价格
            num_simulations: 模拟次数
            
        Returns:
            (预测需求均值, 预测需求方差)
        """
        # 保存当前状态
        original_day = self.current_day
        original_bookings = self.active_bookings.copy()
        original_seed = np.random.get_state()
        
        # 多次模拟
        demand_samples = []
        
        for _ in range(num_simulations):
            # 重置到当前状态
            self.current_day = original_day
            self.active_bookings = original_bookings.copy()
            
            # 模拟一天
            stat = self.simulate_day(price_online, price_offline)
            demand_samples.append(stat['total_new_bookings'])
            
            # 恢复状态（避免影响主仿真）
            self.current_day = original_day
            self.active_bookings = original_bookings.copy()
        
        # 恢复随机状态
        np.random.set_state(original_seed)
        
        # 计算均值和方差
        mean_demand = np.mean(demand_samples)
        var_demand = np.var(demand_samples)
        
        return mean_demand, var_demand
    
    def reset(self):
        """重置模型到初始状态"""
        self.current_day = 0
        self.active_bookings = []
        self.booking_history = []
        self.daily_stats = []
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "total_customers": lambda m: m.schedule.get_agent_count(),
                "total_bookings": lambda m: len([b for b in m.booking_history if not b.is_canceled]),
                "total_cancellations": lambda m: len([b for b in m.booking_history if b.is_canceled]),
                "active_bookings": lambda m: len(m.active_bookings),
            }
        )
    
    def get_statistics(self) -> pd.DataFrame:
        """
        获取统计数据
        
        Returns:
            每日统计数据DataFrame
        """
        return pd.DataFrame(self.daily_stats)
    
    def step(self):
        """
        模型的一步（由Mesa框架调用）
        """
        self.schedule.step()
        self.datacollector.collect(self)
