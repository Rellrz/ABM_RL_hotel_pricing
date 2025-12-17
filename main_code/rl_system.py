#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 标准库导入
import pickle
import random
import warnings
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import numpy as np
import pandas as pd
from scipy import stats

# 本地模块导入
from config import BQL_CONFIG, ABM_CONFIG
from training_monitor import get_training_monitor

class HotelEnvironment:
    """
    酒店环境模拟器
    
    模拟酒店房间的动态定价环境，支持库存管理、需求预测、收益计算等功能。
    环境考虑了季节性、工作日类型、库存水平等因素对需求的影响。
    
    主要特性：
    - 多阶段库存管理：跟踪未来多天的可售房间数量
    - 动态需求预测：集成BNN模型进行需求预测
    - 收益优化：考虑当日收益和未来预期收益
    - 风险惩罚：基于预测方差的风险控制
    - 季节性调整：根据淡旺季调整定价策略
    
    状态空间：
    - inventory_level: 库存水平（离散化：0=极少，4=充足）
    - inventory_raw: 原始库存数量
    - future_inventory: 未来库存数组
    - day: 当前天数
    - season: 季节（0=淡季，1=平季，2=旺季）
    - weekday: 工作日类型（0=工作日，1=周末）
    
    动作空间：
    - 36个定价组合：线上6档 × 线下6档
    - 线上价格档位：[80, 90, 100, 110, 120, 130]元
    - 线下价格档位：[90, 105, 120, 135, 150, 165]元
    
    奖励函数：
    - 总收益 = 当日收益 + 未来预期收益
    - 风险惩罚 = λ × 预测方差
    - 最终奖励 = 总收益 - 风险惩罚
    
    Attributes:
        initial_inventory (int): 初始库存数量
        max_stay_nights (int): 最大入住天数
        cost_per_room (int): 每间房间的成本
        beta_distribution (List[float]): β系数分布，表示不同入住天数的比例
        future_inventory (List[int]): 未来库存数组
        current_inventory (int): 当前库存数量
        day (int): 当前天数
        total_revenue (float): 总收益
        total_bookings (int): 总预订数量
        daily_history (List[Dict]): 每日历史记录
        
    Note:
        - 状态编码：库存等级(0-4) × 季节(0-2) × 日期类型(0-1) = 30种状态
        - 价格档位：6档定价策略，间隔30元，覆盖60-210元区间
        - 风险惩罚系数按季节调整：旺季0.1，平季0.25，淡季0.5
        - 库存更新使用β系数分布，反映不同入住天数的影响
        - 支持90天周期模拟，支持自定义起始日期
    """
    
    def __init__(self, initial_inventory: int = None, max_stay_nights: int = 5, 
                 cost_per_room: int = 20, beta_distribution: Optional[List[float]] = None,
                 use_abm: bool = False, historical_data: Optional[Any] = None,
                 booking_window_days: int = 5):
        
        # 从配置文件读取客房数量，如果没有显式传递参数
        if initial_inventory is None:
            from config import ENV_CONFIG
            self.initial_inventory = ENV_CONFIG['initial_inventory']
        else:
            self.initial_inventory = initial_inventory
        self.max_stay_nights = max_stay_nights # 最大入住天数
        self.cost_per_room = cost_per_room # 每间客房的成本
        self.beta_distribution = beta_distribution or [0.2] * max_stay_nights # Beta分布参数，用于模拟未来需求
        
        # ✅ 预订窗口：客户只能预订未来N天（包括今天）
        self.booking_window_days = booking_window_days  # 默认5天
        
        # ABM模式配置
        self.use_abm = use_abm
        self.abm_model = None
        if use_abm:
            from abm_customer_model import HotelABMModel
            self.abm_model = HotelABMModel(
                historical_data=historical_data,
                random_seed=42,
                params=ABM_CONFIG
            )
        
        # ✅ 初始化未来库存数组：使用booking_window_days作为窗口大小
        # 跟踪当前及未来booking_window_days天的可售客房量
        # 例如：booking_window_days=5，则维护[今天, 明天, 后天, 大后天, 第5天]的库存
        self.future_inventory = None
        
        # ✅ 当前价格窗口：存储未来5天的价格
        self.current_price_window_online = [100.0] * booking_window_days
        self.current_price_window_offline = [120.0] * booking_window_days
        
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """
        重置酒店环境到初始状态
        
        将酒店环境重置到初始状态，包括：
        1. 恢复初始库存数量
        2. 重置天数计数器
        3. 清空收益和预订统计
        4. 初始化历史记录
        5. 设置未来库存数组
        6. 清空4+1队列系统
        
        Returns:
            Dict[str, Any]: 初始状态字典，包含库存水平、季节、工作日类型等信息
            
        状态包含字段：
        - inventory_level: 库存水平（0=极少，1=较少，2=中等，3=较多，4=充足）
        - inventory_raw: 原始库存数量
        - future_inventory: 未来库存数组
        - day: 当前天数
        - season: 季节（0=淡季，1=平季，2=旺季）
        - weekday: 工作日类型（0=工作日，1=周末）
            
        Note:
            - 每次新的episode开始时调用此方法
            - 返回的状态用于强化学习智能体的初始观察
            - 历史记录用于后续分析和可视化
            - 状态编码：库存等级(0-4) × 季节(0-2) × 日期类型(0-1) = 30种状态
        """
        self.current_inventory = self.initial_inventory
        self.day = 0
        self.total_revenue = 0
        self.total_bookings = 0
        self.daily_history = []
        
        # ✅ 初始化未来库存数组：使用booking_window_days作为窗口大小
        # 第t天起始时刻观察到当前及未来booking_window_days天的可售客房量
        # 例如：booking_window_days=5，则维护[Day0, Day1, Day2, Day3, Day4]的库存
        self.future_inventory = [self.initial_inventory] * self.booking_window_days
        
        # ✅ 重置价格窗口
        self.current_price_window_online = [100.0] * self.booking_window_days
        self.current_price_window_offline = [120.0] * self.booking_window_days
        
        # 重置ABM模型
        if self.use_abm and self.abm_model is not None:
            self.abm_model.reset()
        
        return self._get_state()
    
    def _get_state(self) -> Dict[str, Any]:
        """
        获取当前酒店环境状态
        
        计算当前环境状态，包括库存水平、季节、工作日类型等信息。
        
        Returns:
            Dict[str, Any]: 当前状态字典，包含以下字段：
                - inventory_level: 库存水平（0=极少，1=较少，2=中等，3=较多，4=充足）
                - inventory_raw: 原始库存数量
                - future_inventory: 未来库存数组
                - day: 当前天数
                - season: 季节（0=淡季，1=平季，2=旺季）
                - weekday: 工作日类型（0=工作日，1=周末）
                
        状态计算逻辑：
        1. 库存水平：根据当前库存数量离散化为5个等级
        2. 季节判断：基于当前天数计算月份，按季节划分规则确定
        3. 工作日类型：基于当前天数计算星期，周六日为周末
        
        Note:
            - 库存水平离散化：0-20=极少，21-40=较少，41-60=中等，61-80=较多，81-100=充足
            - 季节划分：11-2月=淡季，6-8月=旺季，其他=平季
            - 工作日类型：假设第0天为周一，周六日(5,6)为周末
            - 状态编码：库存等级(0-4) × 季节(0-2) × 日期类型(0-1) = 30种状态
        """
        # 当前库存离散化（s_t^1）- 注释掉库存数量区分
        current_inventory = self.future_inventory[0] if self.future_inventory else self.current_inventory
        
        # 注释掉库存等级区分，统一使用固定值
        if current_inventory <= int(self.initial_inventory * (1/3)):
             inventory_level = 0
        elif current_inventory <= int(self.initial_inventory * (2/3)):
             inventory_level = 1
        else:
             inventory_level = 2
        
        # 根据月份确定季节（方案要求：11-2月→淡季0，3-5/9-10月→平季1，6-8月→旺季2）
        month = (self.day // 30) % 12 + 1  # 简化：假设每月30天
        if month in [11, 12, 1, 2]:  # 11-2月：淡季
            season = 0
        elif month in [6, 7, 8]:  # 6-8月：旺季
            season = 2
        else:  # 3-5月, 9-10月：平季
            season = 1
        
        # 确定日期类型（工作日/周末）- 简化：假设第0天为周一
        weekday_type = 1 if (self.day % 7) in [5, 6] else 0  # 周六(5)、周日(6)为周末
        
        return {
            'inventory_level': inventory_level,
            'inventory_raw': current_inventory,
            'future_inventory': self.future_inventory.copy() if self.future_inventory else [],
            'day': self.day,
            'season': season,
            'weekday': weekday_type
        }
    
    def _get_state_for_day_offset(self, day_offset: int) -> Dict[str, Any]:
        """
        获取未来某一天的状态（用于5天窗口的Q-learning决策）
        
        Args:
            day_offset: 距离当前天的偏移量（0=今天, 1=明天, ..., 4=第5天）
        
        Returns:
            Dict[str, Any]: 该天的状态字典
        """
        # 计算目标日期
        target_day = self.day + day_offset
        
        # 获取该天的库存（从future_inventory窗口中）
        if day_offset < len(self.future_inventory):
            target_inventory = self.future_inventory[day_offset]
        else:
            target_inventory = self.initial_inventory  # 超出窗口，使用初始库存
        
        # 库存离散化
        if target_inventory <= int(self.initial_inventory * (1/3)):
            inventory_level = 0
        elif target_inventory <= int(self.initial_inventory * (2/3)):
            inventory_level = 1
        else:
            inventory_level = 2
        
        # 计算该天的季节
        month = (target_day // 30) % 12 + 1
        if month in [11, 12, 1, 2]:
            season = 0
        elif month in [6, 7, 8]:
            season = 2
        else:
            season = 1
        
        # 计算该天的工作日类型
        weekday_type = 1 if (target_day % 7) in [5, 6] else 0
        
        return {
            'inventory_level': inventory_level,
            'inventory_raw': target_inventory,
            'day': target_day,
            'day_offset': day_offset,  # 额外信息：距离当前天的偏移
            'season': season,
            'weekday': weekday_type
        }
    
    def _get_daily_inventory_dict(self) -> Dict[int, int]:
        """
        将future_inventory转换为ABM需要的字典格式
        
        ✅ 5天滚动窗口：
        - Day 0: future_inventory[0] → 今天（self.day）
        - Day 1: future_inventory[1] → 明天（self.day + 1）
        - Day 2: future_inventory[2] → 后天（self.day + 2）
        - Day 3: future_inventory[3] → 大后天（self.day + 3）
        - Day 4: future_inventory[4] → 第5天（self.day + 4）
        
        Returns:
            Dict[int, int]: 日期到库存数量的映射
        """
        from collections import defaultdict
        daily_inv = defaultdict(lambda: 0)  # 超出窗口的日期库存为0（不可预订）
        if self.future_inventory:
            for i, inv in enumerate(self.future_inventory):
                daily_inv[self.day + i] = inv
        return daily_inv
    
    def _step_with_abm(self, price_windows_online: List[float], price_windows_offline: List[float]) -> Tuple[int, float]:
        """
        使用ABM进行需求模拟
        
        ✅ 5天滚动窗口模式：
        - 传递当前5天的库存状态
        - 传递当前5天的价格窗口（每天都通过Q-learning确定）
        - ABM根据客户的target_date选择对应的价格
        
        Args:
            price_windows_online: 未来5天的线上价格数组 [Day0, Day1, Day2, Day3, Day4]
            price_windows_offline: 未来5天的线下价格数组 [Day0, Day1, Day2, Day3, Day4]
            
        Returns:
            Tuple[int, float]: (实际预订量, 总收益)
        """
        if self.abm_model is None:
            raise ValueError("ABM模型未初始化，请在创建环境时设置use_abm=True")
        
        # ✅ 更新价格窗口：使用传入的5天价格
        self.current_price_window_online = price_windows_online.copy()
        self.current_price_window_offline = price_windows_offline.copy()
        
        # ✅ 同步库存状态到ABM
        self.abm_model.daily_available_rooms = self._get_daily_inventory_dict()
        
        # ✅ 同步价格窗口到ABM
        self.abm_model.price_window_online = self.current_price_window_online.copy()
        self.abm_model.price_window_offline = self.current_price_window_offline.copy()
        self.abm_model.current_day = self.day  # 同步当前日期
        
        # 执行ABM模拟（使用今天的价格作为主价格，但ABM会根据target_date选择窗口价格）
        abm_stat = self.abm_model.simulate_day(
            price_online=price_windows_online[0],  # 今天的线上价格
            price_offline=price_windows_offline[0],  # 今天的线下价格
            max_inventory=self.initial_inventory
        )
        
        actual_bookings = abm_stat['total_new_bookings']
        total_revenue = abm_stat['total_revenue']
        
        return actual_bookings, total_revenue
    
    def step(self, action: Union[int, List[int]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        执行一步酒店定价决策
        
        根据给定的定价动作，模拟一天的酒店运营过程，包括：
        1. 确定定价：将动作索引转换为具体价格（支持线上线下双价格）
        2. 需求预测：使用线上和线下BNN模型预测需求分布，并相加结果
        3. 预订处理：根据库存限制确定实际预订量（优先满足线下用户）
        4. 收益计算：计算当日收益和未来预期收益
        5. 风险惩罚：基于预测方差添加风险惩罚
        6. 库存更新：根据β系数更新未来库存
        7. 状态转移：获取新的环境状态
        
        Args:
            action (Union[int, List[int]]): 定价动作索引（int为单价格，List[int]为双价格[线上, 线下]）
            
        Returns:
            Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: 
                - 新状态（包含库存、季节、工作日等信息）
                - 奖励（收益减去风险惩罚）
                - 是否结束（90天周期结束）
                - 额外信息（预测需求、方差、实际预订、收益等）
                
        收益计算逻辑：
        1. 当日收益 = (价格-成本) × 实际预订量 × 1.0（当天入住系数）
        2. 未来预期收益 = (价格-成本) × 预测需求 × Σβ₁₋₄（未来入住系数和）
        3. 总收益 = 当日收益 + 未来预期收益
        4. 风险惩罚 = λ × 预测方差（按季节调整λ系数）
        5. 最终奖励 = 总收益
                
        Note:
            - 价格档位：线上[80,90,100,110,120,130]，线下[90,105,120,135,150,165]
            - 需求预测为线上和线下BNN预测结果相加
            - 收益计算考虑当日入住和未来预期入住
            - 风险惩罚系数按季节调整（旺季0.1，平季0.25，淡季0.5）
            - 库存更新使用β系数分布，反映不同入住天数的影响
            - 支持90天周期模拟，episode在90天时结束
        """
        # print(f" \n {'+'*15}一个episode开始{'+'*15}")
        # 定价动作（线上线下双价格映射）
        # 从配置文件读取定价档位
        from config import ENV_CONFIG
        online_prices = ENV_CONFIG['online_price_levels']  # 线上价格档位（6个动作）
        offline_prices = ENV_CONFIG['offline_price_levels']  # 线下价格档位（6个动作）
        
        # ✅ ABM模式：使用客户行为模型（需要5天的价格窗口）
        if self.use_abm:
            # action应该是一个包含5个动作的列表：[action_day0, action_day1, ..., action_day4]
            if isinstance(action, (list, np.ndarray)) and len(action) == 5:
                # 5个动作：分别对应5天
                actions_window = action
                price_windows_online = []
                price_windows_offline = []
                for act in actions_window:
                    act_idx = int(act.item()) if hasattr(act, 'item') else int(act)
                    online_idx = act_idx // 6
                    offline_idx = act_idx % 6
                    price_windows_online.append(online_prices[online_idx])
                    price_windows_offline.append(offline_prices[offline_idx])
            elif isinstance(action, (int, np.integer)) or (hasattr(action, 'item') and not isinstance(action, (list, np.ndarray))):
                # 单个动作：应用到今天，其他天保持当前价格窗口
                action_idx = int(action.item()) if hasattr(action, 'item') else int(action)
                online_idx = action_idx // 6
                offline_idx = action_idx % 6
                price_online = online_prices[online_idx]
                price_offline = offline_prices[offline_idx]
                price_windows_online = [price_online] + self.current_price_window_online[1:]
                price_windows_offline = [price_offline] + self.current_price_window_offline[1:]
            else:
                raise ValueError(f"ABM模式需要1个或5个动作，收到: {action}, 类型: {type(action)}")
            
            actual_bookings, total_revenue = self._step_with_abm(price_windows_online, price_windows_offline)
            
            # 更新库存
            self._update_inventory(actual_bookings)
            
            # 更新统计
            self.total_revenue += total_revenue
            self.total_bookings += actual_bookings
            self.day += 1
            
            # 记录历史（使用今天的价格）
            price = price_windows_online[0]  # 今天的线上价格
            price_online = price_windows_online[0]
            price_offline = price_windows_offline[0]
            
            self.daily_history.append({
                'day': self.day,
                'price': price,
                'price_online': price_online,
                'price_offline': price_offline,
                'actual_demand': actual_bookings,
                'actual_bookings': actual_bookings,
                'inventory_before': self.current_inventory + actual_bookings,
                'inventory_after': self.current_inventory,
                'revenue': total_revenue,
                'reward': total_revenue
            })
            
            # 获取新状态
            new_state = self._get_state()
            done = (self.day >= 365)
            
            info = {
                'actual_bookings': actual_bookings,
                'revenue': total_revenue,
                'inventory_after': self.current_inventory
            }
            
            return new_state, total_revenue, done, info
    
    def _update_inventory(self, bookings: int) -> None:
        """
        更新酒店库存状态（5天滚动窗口模式）
        
        ✅ 滚动窗口更新逻辑：
        1. 库存已经在ABM中实时更新（通过daily_available_rooms）
        2. 这里主要负责窗口滚动：
           - 移除第0天（今天已结束）
           - 添加新的第5天
        3. 同时滚动价格窗口
        
        Args:
            bookings (int): 第t天的实际预订量（用于统计，库存已在ABM中更新）
            
        Returns:
            None
            
        滚动逻辑示例：
        Day 1结束前: [Day1, Day2, Day3, Day4, Day5]
        Day 1结束后: [Day2, Day3, Day4, Day5, Day6]  ← 滚动
        """
        if self.future_inventory:
            # ✅ 从ABM同步回最新的库存状态
            daily_inv = self.abm_model.daily_available_rooms if (self.use_abm and self.abm_model) else None
            if daily_inv:
                # 同步当前窗口的库存（已经被ABM更新过）
                for i in range(len(self.future_inventory)):
                    day_key = self.day + i
                    if day_key in daily_inv:
                        self.future_inventory[i] = daily_inv[day_key]
            
            # ✅ 滚动窗口：移除第0天，添加新的第N天
            # 例如：[Day1, Day2, Day3, Day4, Day5] → [Day2, Day3, Day4, Day5, Day6]
            self.future_inventory = self.future_inventory[1:] + [self.initial_inventory]
            
            # ✅ 滚动价格窗口
            self.current_price_window_online = self.current_price_window_online[1:] + [self.current_price_window_online[-1]]
            self.current_price_window_offline = self.current_price_window_offline[1:] + [self.current_price_window_offline[-1]]
            
            # 更新当前库存为新的第0天库存
            self.current_inventory = self.future_inventory[0]
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取酒店环境运行统计信息
        
        计算并返回酒店环境的运行统计信息，包括总天数、总收益、
        平均入住率、平均价格、需求满足率等关键指标。
        
        Returns:
            Dict[str, float]: 统计信息字典，包含以下字段：
                - total_days: 总运行天数
                - total_revenue: 总收益
                - total_bookings: 总预订数量
                - average_occupancy_rate: 平均入住率
                - average_daily_revenue: 平均每日收益
                - average_price: 平均价格
                - total_demand: 总需求
                - demand_satisfaction_rate: 需求满足率
                
        统计计算逻辑：
        1. 总天数：从daily_history中获取最大天数
        2. 总收益：累计所有天的收益
        3. 平均入住率：总预订量 / (初始库存 × 总天数)
        4. 平均价格：所有天价格的平均值
        5. 需求满足率：实际预订量 / 总需求量
                
        Note:
            - 基于daily_history数据计算统计信息
            - 入住率计算考虑初始库存和总天数
            - 需求满足率反映库存限制对需求的影响
            - 所有统计指标都基于历史运行数据
        """
        if not self.daily_history:
            return {}
        
        df_history = pd.DataFrame(self.daily_history)
        
        return {
            'total_days': self.day,
            'total_revenue': self.total_revenue,
            'total_bookings': self.total_bookings,
            'average_occupancy_rate': df_history['actual_bookings'].sum() / (self.initial_inventory * self.day) if self.day > 0 else 0,
            'average_daily_revenue': self.total_revenue / self.day if self.day > 0 else 0,
            'average_price': df_history['price'].mean(),
            'total_demand': df_history['actual_demand'].sum(),
            'demand_satisfaction_rate': df_history['actual_bookings'].sum() / df_history['actual_demand'].sum() if df_history['actual_demand'].sum() > 0 else 0
        }

class QLearningAgent:
    """
    Q-learning智能体
    
    实现Q-learning算法的智能体，用于酒店动态定价决策。
    支持ε-贪心探索策略、UCB探索增强、状态访问统计等功能。
    
    主要特性：
    - ε-贪心探索：平衡探索和利用
    - UCB增强：优先选择访问次数较少的状态-动作对
    - 状态离散化：将连续状态映射到离散状态空间
    - 访问统计：跟踪状态和动作访问次数
    - Q值更新：使用TD学习更新Q值
    
    状态空间：
    - 总状态数：30（库存等级5 × 季节3 × 日期类型2）
    - 状态编码：inventory_level × 6 + season × 2 + weekday
    
    动作空间：
    - 总动作数：36（线上6档 × 线下6档）
    - 动作映射：action_idx = online_idx * 6 + offline_idx
    - 线上价格档位：[80, 90, 100, 110, 120, 130]元
    - 线下价格档位：[90, 105, 120, 135, 150, 165]元
    
    学习参数：
    - 学习率：控制Q值更新速度
    - 折扣因子：权衡即时奖励和未来奖励
    - ε衰减：逐步减少探索概率
    
    Attributes:
        n_states (int): 状态数量（默认30）
        n_actions (int): 动作数量（默认6）
        learning_rate (float): 学习率
        discount_factor (float): 折扣因子
        epsilon_start (float): 初始探索概率
        epsilon_end (float): 最终探索概率
        epsilon_decay_steps (int): ε衰减步数
        q_table (Dict): Q值表，键为状态，值为动作Q值数组
        state_visit_count (Dict): 状态访问计数
        state_action_visit_count (Dict): 状态-动作访问计数
        training_history (List): 训练历史记录
        
    Note:
        - 使用defaultdict自动初始化Q值和访问计数
        - 支持UCB探索策略，优先探索访问次数少的状态-动作对
        - ε值随训练episode线性衰减
        - 状态离散化支持库存、季节、工作日类型组合
    """
    
    def __init__(self, n_states: int = 30, n_actions: int = None, learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon_start: float = 0.8, epsilon_end: float = 0.01, epsilon_decay_steps: int = 50):
        
        # 如果未指定动作数，从配置文件读取
        if n_actions is None:
            from config import RL_CONFIG
            n_actions = RL_CONFIG['n_actions']  # 36个动作组合（线上6价格 × 线下6价格）
        
        self.n_states = n_states
        self.n_actions = n_actions  # 6×6=36个动作组合（线上6价格 × 线下6价格）
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        
        # Q表
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # 状态访问计数
        self.state_visit_count = defaultdict(int)
        
        # 状态-动作访问计数（用于UCB探索）
        self.state_action_visit_count = defaultdict(int)
        
        # 训练历史
        self.training_history = []
    
    def get_epsilon(self, episode: int) -> float:
        """获取当前的epsilon值 - 使用更快的指数衰减策略"""
        if episode >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            # 使用更快的指数衰减策略，使探索率快速下降
            # epsilon = epsilon_end + (epsilon_start - epsilon_end) * exp(-episode / decay_rate)
            decay_rate = self.epsilon_decay_steps / 2  # 进一步加快衰减速率
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode / decay_rate)
            return epsilon
    
    def discretize_state(self, state_info: Dict[str, Any], season: int, weekday: int) -> int:
        """离散化状态 - 基于当前库存、季节和日期类型"""
        inventory_level = state_info['inventory_level']
        
        # 计算状态索引
        # inventory_level: 0-4 (5个等级)
        # season: 0-2 (3个季节)
        # weekday: 0-1 (工作日/周末)
        state_index = inventory_level * 6 + season * 2 + weekday
        
        return min(state_index, self.n_states - 1)  # 防止越界
    
    def select_action(self, state: Union[List, np.ndarray, int], episode: int) -> int:
        """选择动作（epsilon-greedy + 增强UCB探索策略）"""
        epsilon = self.get_epsilon(episode) # 获取当前探索系数，用于epsilon-greedy策略
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        q_values = self.q_table[state_key]
        
        # 36个动作组合：action_idx = online_idx * 6 + offline_idx
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
    
    def update_q_table(self, state: Union[List, np.ndarray, int], action: int, reward: float, next_state: Union[List, np.ndarray, int], done: bool) -> float:
        """更新Q表"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        self.state_action_visit_count[(state_key, action)] += 1
        
        # 当前Q值
        current_q = self.q_table[state_key][action]
        
        # 下一个状态的最大Q值
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # 更新Q表
        self.q_table[state_key][action] = new_q
        
        return new_q
    
    def get_policy(self) -> Dict[Any, int]:
        """
        获取当前策略（状态到动作的映射）
        
        功能描述：
        基于当前Q表生成确定性策略，为每个状态选择具有最高Q值的动作。
        
        返回值:
            Dict[Any, int]: 策略字典，键为状态，值为最优动作索引
            
        策略生成逻辑:
        - 遍历Q表中的所有状态
        - 对每个状态的Q值数组使用argmax获取最优动作
        - 返回状态到最优动作的映射字典
        
        Note:
        - 返回确定性策略（贪婪策略）
        - 如果Q表为空，返回空字典
        - 动作为0-7的整数，对应8个定价档位
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = np.argmax(q_values)
        return policy
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """
        获取Q值统计信息和学习进度指标
        
        功能描述：
        计算Q表的详细统计信息，包括Q值分布、探索覆盖率、学习进度等关键指标。
        
        返回值:
            Dict[str, float]: 统计信息字典，包含以下字段：
                - mean_q_value: Q值的平均值
                - std_q_value: Q值的标准差  
                - min_q_value: Q值的最小值
                - max_q_value: Q值的最大值
                - num_states: 已访问的状态数量
                - num_state_visits: 总状态访问次数
                - zero_q_percentage: 零值Q值所占百分比
                - exploration_coverage: 探索覆盖率（百分比）
                - explored_state_actions: 已探索的状态-动作对数量
                - total_state_actions: 总状态-动作对数量
                
        计算逻辑:
        1. 收集所有Q值并计算基本统计量（均值、标准差、极值）
        2. 统计零值Q值的数量和比例
        3. 计算探索覆盖率：已探索的状态-动作对 / 总状态-动作对
        4. 汇总状态访问和状态-动作访问计数
        
        Note:
        - 如果Q表为空，返回空字典
        - 探索覆盖率反映学习的完整性
        - 零值Q值比例可指示未充分探索的区域
        - 状态访问计数帮助分析学习重点
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
        
        # 计算探索覆盖率
        explored_state_actions = sum(1 for count in self.state_action_visit_count.values() if count > 0)
        total_state_actions = len(self.q_table) * self.n_actions
        exploration_coverage = explored_state_actions / total_state_actions if total_state_actions > 0 else 0
        
        return {
            'mean_q_value': np.mean(all_q_values),
            'std_q_value': np.std(all_q_values),
            'min_q_value': np.min(all_q_values),
            'max_q_value': np.max(all_q_values),
            'num_states': len(self.q_table),
            'num_state_visits': sum(self.state_visit_count.values()),
            'zero_q_percentage': (zero_q_count / total_q_entries) * 100 if total_q_entries > 0 else 0,
            'exploration_coverage': exploration_coverage * 100,
            'explored_state_actions': explored_state_actions,
            'total_state_actions': total_state_actions
        }
    
    def save_agent(self, filepath: str) -> None:
        """
        保存智能体状态和训练历史到文件
        
        功能描述：
        将Q-learning智能体的完整状态保存到pickle文件，包括Q表、访问计数、训练历史、超参数等所有关键信息。
        
        参数:
            filepath (str): 保存文件的路径，应为.pkl文件
            
        保存内容:
        - q_table: Q值表，包含所有状态-动作对的Q值
        - state_visit_count: 状态访问计数统计
        - state_action_visit_count: 状态-动作对访问计数
        - training_history: 完整的训练历史记录
        - hyperparameters: 所有超参数设置
        
        文件格式:
        使用pickle格式保存，包含完整的智能体状态字典
        
        Note:
        - 自动将defaultdict转换为普通dict以便保存
        - 保存后打印确认信息
        - 文件可用于后续加载和继续训练
        - 包含所有必要的超参数信息
        """
        # 转换Q表为普通字典以便保存
        q_table_dict = dict(self.q_table)
        state_visit_dict = dict(self.state_visit_count)
        state_action_visit_dict = dict(self.state_action_visit_count)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': q_table_dict,
                'state_visit_count': state_visit_dict,
                'state_action_visit_count': state_action_visit_dict,
                'training_history': self.training_history,
                'hyperparameters': {
                    'n_states': self.n_states,
                    'n_actions': self.n_actions,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay_steps': self.epsilon_decay_steps
                }
            }, f)
        print(f"智能体已保存到：{filepath}")
    
    def load_agent(self, filepath: str) -> None:
        """
        从文件加载智能体状态和训练历史
        
        功能描述：
        从pickle文件恢复Q-learning智能体的完整状态，包括Q表、访问计数、训练历史等信息。
        
        参数:
            filepath (str): 加载文件的路径，应为之前保存的.pkl文件
            
        恢复内容:
        - q_table: Q值表，恢复所有状态-动作对的Q值
        - state_visit_count: 状态访问计数统计
        - state_action_visit_count: 状态-动作对访问计数  
        - training_history: 完整的训练历史记录
        - 超参数: 自动恢复保存时的超参数设置
        
        加载逻辑:
        1. 从pickle文件读取保存的数据字典
        2. 恢复Q表为defaultdict格式
        3. 恢复访问计数统计
        4. 恢复训练历史记录
        
        Note:
        - 自动将普通dict转换回defaultdict格式
        - 加载后打印确认信息
        - 可继续之前的训练过程
        - 保持与保存时相同的超参数设置
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 恢复Q表
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for state, q_values in data['q_table'].items():
            self.q_table[state] = q_values
        
        # 恢复其他属性
        self.state_visit_count = defaultdict(int, data['state_visit_count'])
        self.state_action_visit_count = defaultdict(int, data.get('state_action_visit_count', {}))
        self.training_history = data['training_history']
        
        print(f"智能体已从{filepath}加载")

class BayesianQLearning:
    """
    贝叶斯Q-Learning (BQL) 实现
    
    在BQL中，我们维护Q(s,a)的概率分布信念，假设Q值服从高斯分布：
    Q(s,a) ~ N(μ_{s,a}, σ²_{s,a})
    
    更新过程使用贝叶斯推断，结合先验信念和观测证据来更新后验分布。
    """
    
    def __init__(self, n_states: int = 30, n_actions: int = 8, discount_factor: float = 0.9,
                 observation_noise_var: float = 1.0, prior_mean: float = 0.0, prior_var: float = 10.0,
                 q_value_max: float = 1000.0, reward_scale: float = 0.1):
        """
        初始化贝叶斯Q-Learning
        
        Args:
            n_states: 状态数量
            n_actions: 动作数量  
            discount_factor: 折扣因子γ
            observation_noise_var: 观测噪声方差σ²_r
            prior_mean: 先验均值
            prior_var: 先验方差
            q_value_max: Q值上限，防止极端值
            reward_scale: 奖励缩放因子，用于归一化奖励范围
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.observation_noise_var = observation_noise_var
        self.q_value_max = q_value_max
        self.reward_scale = reward_scale
        
        # Q值分布参数：每个状态-动作对的均值和方差
        # 使用字典存储，支持动态状态空间
        self.q_means = defaultdict(lambda: np.full(n_actions, prior_mean))  # μ_{s,a}
        self.q_vars = defaultdict(lambda: np.full(n_actions, prior_var))    # σ²_{s,a}
        
        # 状态访问计数
        self.state_visit_count = defaultdict(int)
        self.state_action_visit_count = defaultdict(int)
        
        # 训练历史
        self.training_history = []
        
        # 异常值检测参数
        self.q_value_history = defaultdict(list)  # 记录Q值历史用于异常检测
        self.max_q_value_change = 5.0  # 最大允许的Q值变化倍数
        self.min_variance = 0.1  # 最小方差，防止过度自信
    
    def get_state_distribution(self, state: Union[List, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
        """获取状态的Q值分布（均值和方差）"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        return self.q_means[state_key].copy(), self.q_vars[state_key].copy()
    
    def select_action(self, state: Union[List, np.ndarray, int], episode: int, 
                     exploration_strategy: str = "ucb") -> int:
        """
        选择动作（基于贝叶斯探索策略）
        
        Args:
            state: 当前状态
            episode: 当前episode编号
            exploration_strategy: 探索策略 ("ucb", "thompson", "epsilon_greedy")
        
        Returns:
            选择的动作索引
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        means = self.q_means[state_key]
        vars = self.q_vars[state_key]
        
        if exploration_strategy == "ucb":
            # 基于不确定性的上置信界
            n_visits = np.array([self.state_action_visit_count.get((state_key, a), 0) for a in range(self.n_actions)])
            
            # 使用配置中的UCB参数
            ucb_c = BQL_CONFIG.get('ucb_c', 2.5)
            ucb_bonus_scale = BQL_CONFIG.get('ucb_bonus_scale', 2.0)
            
            # 改进的UCB计算，避免初始阶段探索不足
            total_visits = self.state_visit_count.get(state_key, 0)
            if total_visits == 0:
                # 初始阶段：综合考虑先验均值和不确定性，引入随机扰动
                exploration_bonus = np.sqrt(vars) / np.max(np.sqrt(vars))  # 标准化不确定性
                random_noise = np.random.normal(0, 0.1, self.n_actions)  # 小幅度随机噪声
                # 平衡先验信念和探索：均值 + 探索奖励 + 随机扰动
                ucb_values = means + 0.5 * exploration_bonus + random_noise
            else:
                # 正常UCB计算
                log_total = np.log(total_visits + 1)
                # 避免除零，给未访问的动作最大探索奖励
                ucb_bonus = ucb_c * np.sqrt(log_total / (n_visits + 1e-6))
                ucb_values = means + ucb_bonus_scale * ucb_bonus * np.sqrt(vars)
            
            return np.argmax(ucb_values)
            
        elif exploration_strategy == "thompson":
            # Thompson采样：从高斯分布中采样
            sampled_values = np.random.normal(means, np.sqrt(vars))
            return np.argmax(sampled_values)
            
        else:  # epsilon_greedy
            # ε-贪心策略，使用均值
            epsilon = max(0.1, 1.0 / (episode + 1))
            if np.random.random() < epsilon:
                return np.random.randint(self.n_actions)
            else:
                return np.argmax(means)
    
    def _normalize_reward(self, reward: float) -> float:
        """归一化奖励值，防止极端值"""
        # 使用tanh函数将奖励压缩到合理范围
        normalized = np.tanh(reward * self.reward_scale)
        # 然后缩放到与先验均值匹配的范围
        return normalized * 100.0  # 假设先验均值在50左右
    
    def _detect_anomalous_q_value(self, state_key: Union[tuple, int], action: int, 
                                  new_mean: float, new_var: float) -> bool:
        """检测异常Q值更新"""
        # 检查Q值是否超出合理范围
        if abs(new_mean) > self.q_value_max:
            return True
        
        # 检查方差是否过小（过度自信）
        if new_var < self.min_variance:
            return True
        
        # 检查Q值变化是否过于剧烈
        history = self.q_value_history.get((state_key, action), [])
        if len(history) >= 3:  # 需要至少3个历史值
            recent_mean = np.mean(history[-3:])
            if recent_mean != 0 and abs(new_mean - recent_mean) / abs(recent_mean) > self.max_q_value_change:
                return True
        
        return False
    
    def update_bayesian_q_table(self, state: Union[List, np.ndarray, int], action: int, 
                               reward: float, next_state: Union[List, np.ndarray, int], 
                               done: bool) -> Tuple[float, float]:
        """
        使用贝叶斯推断更新Q值分布 - 改进版
        
        根据贝叶斯定理，更新后验分布参数：
        σ²_new = (1/σ²_old + 1/σ²_r)^(-1)
        μ_new = σ²_new * (μ_old/σ²_old + y/σ²_r)
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            
        Returns:
            Tuple[float, float]: 更新后的(均值, 方差)
        """
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_state_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state
        
        # 更新访问计数
        self.state_visit_count[state_key] += 1
        self.state_action_visit_count[(state_key, action)] += 1
        
        # 获取当前分布参数
        current_mean = self.q_means[state_key][action]
        current_var = self.q_vars[state_key][action]
        
        # 归一化奖励，防止极端值
        normalized_reward = self._normalize_reward(reward)
        
        # 计算TD目标 y = r + γ * max_a' μ_{s',a'}
        if done:
            td_target = normalized_reward
        else:
            next_means = self.q_means[next_state_key]
            # 使用鲁棒的max估计，避免异常值影响
            max_next_mean = np.percentile(next_means, 90)  # 使用90分位数而非最大值
            td_target = normalized_reward + self.discount_factor * max_next_mean
        
        # 限制TD目标的范围
        td_target = np.clip(td_target, -self.q_value_max, self.q_value_max)
        
        # 改进的贝叶斯更新：考虑TD目标的不确定性
        if not done:
            # 计算下一状态Q值的最大值的不确定性
            next_vars = self.q_vars[next_state_key]
            # 同样使用90分位数对应的不确定性
            top_10_percent_count = max(1, int(np.ceil(len(next_means) * 0.1)))
            max_idx = np.argsort(next_means)[-top_10_percent_count:]
            max_next_var = np.mean([next_vars[i] for i in max_idx]) if len(max_idx) > 0 else np.mean(next_vars)
            # TD目标的总不确定性 = 奖励噪声 + 折扣后的下一状态不确定性
            td_target_var = self.observation_noise_var + (self.discount_factor ** 2) * max_next_var
        else:
            td_target_var = self.observation_noise_var
        
        # 确保方差在合理范围内
        td_target_var = max(td_target_var, self.min_variance)
        
        # 贝叶斯更新，使用TD目标的总不确定性
        new_var = 1.0 / (1.0 / current_var + 1.0 / td_target_var)
        new_mean = new_var * (current_mean / current_var + td_target / td_target_var)
        
        # 检测异常Q值更新
        if self._detect_anomalous_q_value(state_key, action, new_mean, new_var):
            # 如果检测到异常，使用保守的更新策略
            learning_rate = 0.1  # 使用较小的学习率
            new_mean = current_mean + learning_rate * (td_target - current_mean)
            new_var = max(current_var * 0.99, self.min_variance)  # 稍微减小方差
        
        # 确保方差不小于最小值
        new_var = max(new_var, self.min_variance)
        
        # 限制Q值范围
        new_mean = np.clip(new_mean, -self.q_value_max, self.q_value_max)
        
        # 记录Q值历史
        self.q_value_history[(state_key, action)].append(new_mean)
        # 只保留最近的历史
        if len(self.q_value_history[(state_key, action)]) > 10:
            self.q_value_history[(state_key, action)].pop(0)
        
        # 更新分布参数
        self.q_means[state_key][action] = new_mean
        self.q_vars[state_key][action] = new_var
        
        return new_mean, new_var
    
    def get_uncertainty(self, state: Union[List, np.ndarray, int], action: int = None) -> Union[float, np.ndarray]:
        """获取状态-动作对的不确定性（标准差）- 改进版"""
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 确保状态存在，如果不存在则返回先验不确定性
        if state_key not in self.q_vars:
            prior_var = BQL_CONFIG.get('prior_var', 15.0)
            if action is not None:
                return np.sqrt(prior_var)
            else:
                return np.full(self.n_actions, np.sqrt(prior_var))
        
        if action is not None:
            return np.sqrt(max(self.q_vars[state_key][action], self.min_variance))
        else:
            return np.sqrt(np.maximum(self.q_vars[state_key], self.min_variance))
    
    def get_epsilon(self, episode: int) -> float:
        """获取当前探索率 - 改进版，支持动态探索"""
        # 贝叶斯Q-learning使用UCB或Thompson采样，不直接使用epsilon
        # 但为了兼容性，返回一个基于不确定性的动态探索率
        
        # 计算平均不确定性
        if self.q_means:
            total_uncertainty = 0.0
            count = 0
            for state_key in self.q_means.keys():
                uncertainties = self.get_uncertainty(state_key)
                total_uncertainty += np.mean(uncertainties)
                count += 1
            
            avg_uncertainty = total_uncertainty / count if count > 0 else 1.0
            
            # 基于不确定性调整探索率：不确定性高时探索更多
            base_epsilon = max(0.1, 1.0 / (episode + 1))
            uncertainty_factor = min(2.0, 1.0 + avg_uncertainty / 10.0)  # 不确定性因子
            
            return min(0.5, base_epsilon * uncertainty_factor)
        else:
            return max(0.1, 1.0 / (episode + 1))
    
    def get_q_value_stats(self) -> Dict[str, float]:
        """获取Q值统计信息 - 改进版，包含异常检测"""
        if not self.q_means:
            return {
                'zero_q_percentage': 100.0, 
                'exploration_coverage': 0.0, 
                'mean_q_value': 0.0, 
                'num_state_visits': 0,
                'explored_state_actions': 0,
                'total_state_actions': 0,
                'mean_uncertainty': 0.0,
                'std_uncertainty': 0.0,
                'min_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'anomalous_q_percentage': 0.0,
                'high_uncertainty_percentage': 0.0,
                'unvisited_percentage': 100.0
            }
        
        # 计算零值Q值比例和异常Q值比例
        all_means = []
        all_uncertainties = []
        anomalous_count = 0
        high_uncertainty_count = 0
        
        for state_key in self.q_means.keys():
            state_means = self.q_means[state_key]
            state_vars = self.q_vars[state_key]
            state_uncertainties = np.sqrt(state_vars)
            
            all_means.extend(state_means)
            all_uncertainties.extend(state_uncertainties)
            
            # 检测异常Q值
            for action in range(self.n_actions):
                if self._detect_anomalous_q_value(state_key, action, state_means[action], state_vars[action]):
                    anomalous_count += 1
                
                # 检测高不确定性（标准差大于先验标准差）
                if state_uncertainties[action] > np.sqrt(BQL_CONFIG.get('prior_var', 15.0)):
                    high_uncertainty_count += 1
        
        if not all_means:
            return {
                'zero_q_percentage': 100.0, 
                'exploration_coverage': 0.0, 
                'mean_q_value': 0.0, 
                'num_state_visits': 0,
                'explored_state_actions': 0,
                'total_state_actions': 0,
                'mean_uncertainty': 0.0,
                'std_uncertainty': 0.0,
                'min_uncertainty': 0.0,
                'max_uncertainty': 0.0,
                'anomalous_q_percentage': 0.0,
                'high_uncertainty_percentage': 0.0,
                'unvisited_percentage': 100.0
            }
        
        zero_q_count = sum(bool(abs(mean) < 0.01) for mean in all_means)
        zero_q_percentage = (zero_q_count / len(all_means)) * 100
        
        anomalous_q_percentage = (anomalous_count / len(all_means)) * 100
        high_uncertainty_percentage = (high_uncertainty_count / len(all_means)) * 100
        
        # 计算平均Q值（排除异常值）
        normal_means = [mean for mean in all_means if abs(mean) <= self.q_value_max]
        mean_q_value = np.mean(normal_means) if normal_means else 0.0
        
        # 计算不确定性统计
        mean_uncertainty = np.mean(all_uncertainties)
        std_uncertainty = np.std(all_uncertainties)
        min_uncertainty = np.min(all_uncertainties)
        max_uncertainty = np.max(all_uncertainties)
        
        # 计算探索覆盖率（已访问的状态-动作对比例）
        total_state_action_pairs = len(self.q_means) * self.n_actions
        visited_state_action_pairs = len(self.state_action_visit_count)
        exploration_coverage = (visited_state_action_pairs / total_state_action_pairs) * 100 if total_state_action_pairs > 0 else 0.0
        
        # 计算未访问的状态-动作对比例
        unvisited_percentage = 100.0 - exploration_coverage
        
        # 计算总状态访问次数
        num_state_visits = sum(self.state_visit_count.values())
        
        return {
            'zero_q_percentage': zero_q_percentage,
            'exploration_coverage': exploration_coverage,
            'mean_q_value': mean_q_value,
            'num_state_visits': num_state_visits,
            'explored_state_actions': visited_state_action_pairs,
            'total_state_actions': total_state_action_pairs,
            # 贝叶斯Q-learning特有的不确定性统计
            'mean_uncertainty': mean_uncertainty,
            'std_uncertainty': std_uncertainty,
            'min_uncertainty': min_uncertainty,
            'max_uncertainty': max_uncertainty,
            # 新增异常检测统计
            'anomalous_q_percentage': anomalous_q_percentage,
            'high_uncertainty_percentage': high_uncertainty_percentage,
            'unvisited_percentage': unvisited_percentage
        }
    
    def discretize_state(self, state_info: Dict[str, Any], season: int, weekday: int) -> int:
        """离散化状态"""
        inventory_level = state_info['inventory_level']
        # 确保库存水平在合理范围内
        inventory_level = max(0, min(inventory_level, 4))  # 库存等级为0-4
        
        # 修正状态映射：库存(5) × 季节(3) × 星期(2) = 30种状态
        # 季节只有3个值：0=淡季，1=平季，2=旺季
        state_index = inventory_level * 6 + season * 2 + weekday
        
        # 确保状态索引在有效范围内
        return min(state_index, self.n_states - 1)
    
    def save_agent(self, filepath: str):
        """保存智能体状态"""
        agent_state = {
            'q_means': dict(self.q_means),
            'q_vars': dict(self.q_vars),
            'state_visit_count': dict(self.state_visit_count),
            'state_action_visit_count': dict(self.state_action_visit_count),
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'discount_factor': self.discount_factor,
            'observation_noise_var': self.observation_noise_var
        }
        with open(filepath, 'wb') as f:
            pickle.dump(agent_state, f)
    
    def load_agent(self, filepath: str):
        """加载智能体状态"""
        with open(filepath, 'rb') as f:
            agent_state = pickle.load(f)
        
        # 获取配置中的先验参数，确保加载时使用正确的默认值
        prior_mean = BQL_CONFIG.get('prior_mean', 50.0)
        prior_var = BQL_CONFIG.get('prior_var', 15.0)
        
        self.q_means = defaultdict(lambda: np.full(self.n_actions, prior_mean), agent_state['q_means'])
        self.q_vars = defaultdict(lambda: np.full(self.n_actions, prior_var), agent_state['q_vars'])
        self.state_visit_count = defaultdict(int, agent_state['state_visit_count'])
        self.state_action_visit_count = defaultdict(int, agent_state['state_action_visit_count'])
        self.n_states = agent_state['n_states']
        self.n_actions = agent_state['n_actions']
        self.discount_factor = agent_state['discount_factor']
        self.observation_noise_var = agent_state['observation_noise_var']
    
    def train_episode(self, env: HotelEnvironment, online_booked_predictor: Optional[Any] = None, 
                     online_actual_predictor: Optional[Any] = None, offline_booked_predictor: Optional[Any] = None, 
                     offline_actual_predictor: Optional[Any] = None, date_features: Optional[pd.DataFrame] = None, 
                     episode: int = 0, exploration_strategy: str = "ucb") -> Tuple[float, int]:
        """
        使用贝叶斯Q-Learning训练一个episode
        
        Args:
            env: 酒店环境实例
            online_booked_predictor: 线上用户预定需求NGBoost预测器
            online_actual_predictor: 线上用户实际需求NGBoost预测器
            offline_booked_predictor: 线下用户预定需求NGBoost预测器
            offline_actual_predictor: 线下用户实际需求NGBoost预测器
            date_features: 日期特征数据
            episode: 当前episode编号
            exploration_strategy: 探索策略
            
        Returns:
            Tuple[float, int]: (总奖励, 步数)
        """
        state_info = env.reset()
        total_reward = 0.0  # 明确指定为float类型
        steps: int = 0
        day_index = 0  # 添加日期索引，避免使用steps作为日期索引
        
        # 初始化每日记录
        daily_rewards: List[float] = []
        daily_uncertainties: List[float] = []  # 记录不确定性
        
        # 获取季节和星期信息
        if date_features is not None and len(date_features) > 0:
            season = int(date_features['season'].iloc[0])
            weekday = int(date_features['is_weekend'].iloc[0])
        else:
            season = 0
            weekday = 0
        
        state = self.discretize_state(state_info, season, weekday)
        
        done = False
        while not done:
            # 选择动作（使用贝叶斯探索策略）
            action = self.select_action(state, episode, exploration_strategy)
            
            # 获取价格信息
            prices = [60, 80, 100, 110, 120, 130, 140, 150]
            price = prices[action]
            
            # 执行动作，使用四个NGBoost预测器
            next_state_info, reward, done, info = env.step(action, online_booked_predictor, online_actual_predictor, offline_booked_predictor, offline_actual_predictor, date_features)
            
            # 获取当前状态的不确定性
            current_uncertainty = self.get_uncertainty(state, action)
            
            # 获取下一状态的season和weekday信息
            if date_features is not None and day_index + 1 < len(date_features):
                next_season = int(date_features['season'].iloc[day_index + 1])
                next_weekday = int(date_features['is_weekend'].iloc[day_index + 1])
                day_index += 1
            else:
                next_season = season
                next_weekday = weekday
            
            next_state = self.discretize_state(next_state_info, next_season, next_weekday)
            
            # 使用贝叶斯更新Q表
            new_mean, new_var = self.update_bayesian_q_table(state, action, reward, next_state, done)
            
            # 记录信息
            daily_rewards.append(float(reward))
            daily_uncertainties.append(float(current_uncertainty))
            
            # 打印训练信息（包含不确定性）
            if steps % 10 == 0:
                print(f"Episode {episode}, Step {steps}: 动作={action}({price}元), "
                      f"奖励={reward:.2f}, Q均值={new_mean:.2f}, Q方差={new_var:.2f}, "
                      f"不确定性={current_uncertainty:.2f}")
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if steps >= 200:  # 防止无限循环
                break
        
        # 记录训练历史
        episode_history = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'avg_reward': float(total_reward / steps) if steps > 0 else 0.0,
            'avg_uncertainty': float(np.mean(daily_uncertainties)) if daily_uncertainties else 0.0,
            'exploration_strategy': exploration_strategy
        }
        self.training_history.append(episode_history)
        
        return total_reward, steps
