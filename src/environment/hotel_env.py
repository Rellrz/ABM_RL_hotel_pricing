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
from configs.config import ABM_CONFIG, RANDOM_CONFIG
from src.utils.training_monitor import get_training_monitor
from src.environment.abm_customer_model import HotelABMModel
from configs.config import ENV_CONFIG

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
    
    def __init__(
        self,
        initial_inventory: int = None,
        cost_per_room: int = 20,
        historical_data: Optional[Any] = None,
        booking_window_days: Optional[int] = None,
    ):
        
        # 从配置文件读取客房数量，如果没有显式传递参数
        if initial_inventory is None:
            from configs.config import ENV_CONFIG
            self.initial_inventory = ENV_CONFIG.initial_inventory
        else:
            self.initial_inventory = initial_inventory
        self.cost_per_room = cost_per_room # 每间客房的成本
        
        # ✅ 预订窗口：客户只能预订未来N天（包括今天）
        self.booking_window_days = int(ENV_CONFIG.booking_window_days if booking_window_days is None else booking_window_days)
        
        # ABM模式配置 - 根据RANDOM_CONFIG决定是否使用随机种子
        abm_random_seed = RANDOM_CONFIG.fixed_seed if RANDOM_CONFIG.random_mode == 'fixed' else None
        self.abm_model = HotelABMModel(
            historical_data=historical_data,
            random_seed=abm_random_seed,
            booking_window_days=self.booking_window_days,
        )
        
        # ✅ 初始化未来库存数组：使用booking_window_days作为窗口大小
        # 跟踪当前及未来booking_window_days天的可售客房量
        # 例如：booking_window_days=5，则维护[今天, 明天, 后天, 大后天, 第5天]的库存
        self.future_inventory = None
        
        # ✅ 当前价格窗口：存储未来N天的价格
        self.current_price_window_online = [100.0] * self.booking_window_days
        self.current_price_window_offline = [120.0] * self.booking_window_days
        
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
        
        # 保存ABM统计信息（用于博弈系统）
        self.last_abm_stat = abm_stat
        
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

        online_prices = ENV_CONFIG.online_price_levels  # 线上价格档位（6个动作）
        offline_prices = ENV_CONFIG.offline_price_levels  # 线下价格档位（6个动作）
        
        # ✅ 支持两种动作类型：
        # 1. 离散动作（Q-learning）：动作索引 0-35
        # 2. 连续动作（Actor-Critic）：直接价格值 80-200
        # 3. 博弈模式：[price_online, price_offline]
        
        if isinstance(action, (list, np.ndarray)) and len(action) == 2:
            # 博弈模式：直接指定线上和线下价格
            price_online, price_offline = action
            price_windows_online = [float(price_online)] + self.current_price_window_online[1:]
            price_windows_offline = [float(price_offline)] + self.current_price_window_offline[1:]
            
        elif isinstance(action, (list, np.ndarray)) and len(action) in (5, self.booking_window_days):
            actions_window = list(action)
            price_windows_online = []
            price_windows_offline = []
            
            for act in actions_window:
                # 检查是否是[price_online, price_offline]对（博弈模式）
                if isinstance(act, (list, np.ndarray)) and len(act) == 2:
                    # 博弈模式：直接指定线上和线下价格
                    price_online, price_offline = act
                    price_windows_online.append(float(price_online))
                    price_windows_offline.append(float(price_offline))
                # 判断是离散还是连续动作
                elif isinstance(act, (int, np.integer)) or (isinstance(act, (float, np.floating)) and act < 50):
                    # 离散动作：动作索引 (0-35)
                    act_idx = int(act.item()) if hasattr(act, 'item') else int(act)
                    n_offline = len(offline_prices)
                    online_idx = act_idx // n_offline
                    offline_idx = act_idx % n_offline
                    price_windows_online.append(online_prices[online_idx])
                    price_windows_offline.append(offline_prices[offline_idx])
                else:
                    # 连续动作：直接价格值 (80-200)
                    price = float(act.item()) if hasattr(act, 'item') else float(act)
                    # 线上和线下使用相同价格（简化处理）
                    price_windows_online.append(price)
                    price_windows_offline.append(price * 1.1)  # 线下价格略高于线上
                    
        elif isinstance(action, (int, np.integer)) or (isinstance(action, (float, np.floating)) and hasattr(action, 'item')):
            # 单个动作
            if isinstance(action, (int, np.integer)) or (isinstance(action, (float, np.floating)) and action < 50):
                # 离散动作
                action_idx = int(action.item()) if hasattr(action, 'item') else int(action)
                n_offline = len(offline_prices)
                online_idx = action_idx // n_offline
                offline_idx = action_idx % n_offline

                price_online = online_prices[online_idx]
                price_offline = offline_prices[offline_idx]
            else:
                # 连续动作
                price = float(action.item()) if hasattr(action, 'item') else float(action)
                price_online = price
                price_offline = price * 1.1
                
            price_windows_online = [price_online] + self.current_price_window_online[1:]
            price_windows_offline = [price_offline] + self.current_price_window_offline[1:]
        else:
            raise ValueError(f"不支持的动作类型: {action}, 类型: {type(action)}")
            
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
            
        # 构建info字典，包含渠道级数据（用于博弈系统）
        info = {
            'actual_bookings': actual_bookings,
            'revenue': total_revenue,
            'inventory_after': self.current_inventory
        }
        
        # 如果有ABM统计信息，添加渠道级数据
        if hasattr(self, 'last_abm_stat') and self.last_abm_stat:
            info.update({
                'new_bookings_online': self.last_abm_stat.get('new_bookings_online', 0),
                'new_bookings_offline': self.last_abm_stat.get('new_bookings_offline', 0),
                'revenue_online': self.last_abm_stat.get('revenue_online', 0),
                'revenue_offline': self.last_abm_stat.get('revenue_offline', 0),
                'price_online': self.last_abm_stat.get('price_online', 0),
                'price_offline': self.last_abm_stat.get('price_offline', 0),
                'new_customers': self.last_abm_stat.get('new_customers', 0),
                'bookings_by_day_offset': self.last_abm_stat.get('bookings_by_day_offset', [])  # ✅ 按day_offset分组的预订信息
            })
            
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
            daily_inv = self.abm_model.daily_available_rooms
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
