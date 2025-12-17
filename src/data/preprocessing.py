#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块 - 双智能体酒店动态定价系统
Data Preprocessing Module

主要功能：
1. 加载和清洗酒店预订数据
2. 计算基准价格 P_base (70%分位数，按月份+周末分段)
3. 计算远期价格 P_long (中位数，按月份+周末分段)
4. 计算基础到达率 λ_base (按月份统计日均订单量)
5. 拟合提前期分布参数 (LogNormal)
6. 拟合WTP分布参数 (Normal，按月份)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import pickle
import os
import logging

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import PATH_CONFIG, DATA_CONFIG, ABM_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotelDataPreprocessor:
    """酒店数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.data = None
        self.price_tables = {}  # 存储价格表
        self.arrival_rates = {}  # 存储到达率
        self.lead_time_params = {}  # 提前期分布参数
        self.wtp_params = {}  # WTP分布参数
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        加载酒店预订数据
        
        Args:
            filepath: 数据文件路径，默认使用配置中的路径
            
        Returns:
            加载的DataFrame
        """
        if filepath is None:
            filepath = PATH_CONFIG.hotel_bookings_csv
        
        logger.info(f"加载数据: {filepath}")
        self.data = pd.read_csv(filepath)
        logger.info(f"数据形状: {self.data.shape}")
        
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """
        清洗数据
        
        Returns:
            清洗后的DataFrame
        """
        logger.info("开始数据清洗...")
        
        # 1. 过滤酒店类型
        if DATA_CONFIG.hotel_type:
            self.data = self.data[self.data['hotel'] == DATA_CONFIG.hotel_type].copy()
            logger.info(f"过滤酒店类型: {DATA_CONFIG.hotel_type}, 剩余 {len(self.data)} 条记录")
        
        # 2. 过滤ADR异常值
        self.data = self.data[
            (self.data['adr'] >= DATA_CONFIG.adr_min) & 
            (self.data['adr'] <= DATA_CONFIG.adr_max)
        ].copy()
        logger.info(f"过滤ADR异常值 [{DATA_CONFIG.adr_min}, {DATA_CONFIG.adr_max}], 剩余 {len(self.data)} 条记录")
        
        # 3. 删除缺失值
        original_len = len(self.data)
        self.data = self.data.dropna(subset=['adr', 'lead_time', 'arrival_date_month'])
        logger.info(f"删除缺失值, 删除 {original_len - len(self.data)} 条记录")
        
        # 4. 添加月份数字列
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        self.data['month'] = self.data['arrival_date_month'].map(month_map)
        
        # 5. 添加周末标识（基于arrival_date_day_of_month和arrival_date_week_number推断）
        # 简化处理：使用数据集中的stays_in_weekend_nights作为代理
        # 如果weekend nights > 0，认为是周末入住
        self.data['is_weekend'] = (self.data['stays_in_weekend_nights'] > 0).astype(int)
        
        logger.info("数据清洗完成")
        return self.data
    
    def calculate_price_tables(self) -> Dict:
        """
        计算价格表：P_base 和 P_long
        按月份和周末分段计算
        
        Returns:
            价格表字典
        """
        logger.info("计算价格表...")
        
        # 如果需要过滤取消订单
        if DATA_CONFIG.filter_canceled:
            price_data = self.data[self.data['is_canceled'] == 0].copy()
            logger.info(f"过滤取消订单，用于价格计算的记录数: {len(price_data)}")
        else:
            price_data = self.data.copy()
        
        # 初始化价格表
        self.price_tables = {
            'p_base': {},  # 基准价格 (70%分位数)
            'p_long': {}   # 远期价格 (中位数)
        }
        
        # 按月份和周末分组计算
        for month in range(1, 13):
            for is_weekend in [0, 1]:
                key = (month, is_weekend)
                
                # 筛选数据
                mask = (price_data['month'] == month) & (price_data['is_weekend'] == is_weekend)
                adr_values = price_data.loc[mask, 'adr']
                
                if len(adr_values) > 0:
                    # 计算P_base (70%分位数)
                    p_base = np.percentile(adr_values, DATA_CONFIG.p_base_quantile * 100)
                    
                    # 计算P_long (中位数)
                    p_long = np.median(adr_values)
                    
                    self.price_tables['p_base'][key] = p_base
                    self.price_tables['p_long'][key] = p_long
                    
                    logger.debug(f"月份{month}, 周末{is_weekend}: P_base={p_base:.2f}, P_long={p_long:.2f}")
                else:
                    # 如果没有数据，使用全局平均值
                    logger.warning(f"月份{month}, 周末{is_weekend} 无数据，使用全局平均值")
                    self.price_tables['p_base'][key] = price_data['adr'].quantile(DATA_CONFIG.p_base_quantile)
                    self.price_tables['p_long'][key] = price_data['adr'].median()
        
        # 验证 P_long < P_base
        violations = []
        for key in self.price_tables['p_base'].keys():
            if self.price_tables['p_long'][key] >= self.price_tables['p_base'][key]:
                violations.append(key)
        
        if violations:
            logger.warning(f"发现 {len(violations)} 个违反 P_long < P_base 的情况: {violations}")
        else:
            logger.info("✓ 所有价格满足 P_long < P_base")
        
        logger.info(f"价格表计算完成，共 {len(self.price_tables['p_base'])} 个价格段")
        return self.price_tables
    
    def calculate_arrival_rates(self) -> Dict:
        """
        计算基础到达率 λ_base
        按月份统计日均订单量
        
        Returns:
            到达率字典 {month: λ_base}
        """
        logger.info("计算到达率...")
        
        # 如果使用实际需求（未取消订单）
        if DATA_CONFIG.use_actual_demand:
            arrival_data = self.data[self.data['is_canceled'] == 0].copy()
            logger.info(f"使用实际需求（未取消订单），记录数: {len(arrival_data)}")
        else:
            arrival_data = self.data.copy()
        
        # 每月天数（简化处理，使用平均值）
        days_per_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        
        self.arrival_rates = {}
        
        for month in range(1, 13):
            # 统计该月订单数
            month_orders = len(arrival_data[arrival_data['month'] == month])
            
            # 计算日均订单量
            days = days_per_month[month]
            lambda_base = month_orders / days
            
            self.arrival_rates[month] = lambda_base
            logger.debug(f"月份{month}: 订单数={month_orders}, 日均={lambda_base:.2f}")
        
        logger.info(f"到达率计算完成，共 {len(self.arrival_rates)} 个月份")
        return self.arrival_rates
    
    def fit_lead_time_distribution(self) -> Dict:
        """
        拟合提前期分布参数 (LogNormal)
        
        Returns:
            分布参数字典 {'mu': μ, 'sigma': σ}
        """
        logger.info("拟合提前期分布...")
        
        # 过滤提前期范围
        lead_times = self.data[
            (self.data['lead_time'] >= ABM_CONFIG.lead_time_min) & 
            (self.data['lead_time'] <= ABM_CONFIG.lead_time_max)
        ]['lead_time'].values
        
        # 过滤0值（对数正态分布要求正值）
        lead_times = lead_times[lead_times > 0]
        
        # 拟合对数正态分布
        shape, loc, scale = stats.lognorm.fit(lead_times, floc=0)
        
        # 转换为 μ 和 σ
        mu = np.log(scale)
        sigma = shape
        
        self.lead_time_params = {
            'mu': mu,
            'sigma': sigma,
            'shape': shape,
            'loc': loc,
            'scale': scale
        }
        
        logger.info(f"提前期分布参数: μ={mu:.3f}, σ={sigma:.3f}")
        logger.info(f"提前期统计: 均值={lead_times.mean():.2f}, 中位数={np.median(lead_times):.2f}")
        
        return self.lead_time_params
    
    def fit_wtp_distribution(self) -> Dict:
        """
        拟合WTP分布参数 (Normal)
        按月份分别拟合
        
        Returns:
            分布参数字典 {month: {'mu': μ, 'sigma': σ}}
        """
        logger.info("拟合WTP分布...")
        
        # 使用未取消订单的ADR作为WTP的代理
        wtp_data = self.data[self.data['is_canceled'] == 0].copy()
        
        self.wtp_params = {}
        
        for month in range(1, 13):
            adr_values = wtp_data[wtp_data['month'] == month]['adr'].values
            
            if len(adr_values) > 0:
                mu = np.mean(adr_values)
                sigma = np.std(adr_values)
                
                self.wtp_params[month] = {
                    'mu': mu,
                    'sigma': sigma
                }
                
                logger.debug(f"月份{month}: WTP μ={mu:.2f}, σ={sigma:.2f}")
            else:
                # 使用全局平均值
                logger.warning(f"月份{month} 无数据，使用全局平均值")
                self.wtp_params[month] = {
                    'mu': wtp_data['adr'].mean(),
                    'sigma': wtp_data['adr'].std()
                }
        
        logger.info(f"WTP分布拟合完成，共 {len(self.wtp_params)} 个月份")
        return self.wtp_params
    
    def get_price(self, price_type: str, month: int, is_weekend: bool) -> float:
        """
        获取指定价格
        
        Args:
            price_type: 'p_base' 或 'p_long'
            month: 月份 (1-12)
            is_weekend: 是否周末
            
        Returns:
            价格值
        """
        key = (month, int(is_weekend))
        return self.price_tables[price_type].get(key, 0.0)
    
    def get_arrival_rate(self, month: int) -> float:
        """
        获取指定月份的到达率
        
        Args:
            month: 月份 (1-12)
            
        Returns:
            到达率 λ_base
        """
        return self.arrival_rates.get(month, 0.0)
    
    def save(self, filepath: str = None):
        """
        保存预处理器
        
        Args:
            filepath: 保存路径，默认使用配置中的路径
        """
        if filepath is None:
            filepath = PATH_CONFIG.preprocessor_path
        
        logger.info(f"保存预处理器: {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'price_tables': self.price_tables,
                'arrival_rates': self.arrival_rates,
                'lead_time_params': self.lead_time_params,
                'wtp_params': self.wtp_params
            }, f)
        
        logger.info("保存完成")
    
    @classmethod
    def load(cls, filepath: str = None):
        """
        加载预处理器
        
        Args:
            filepath: 加载路径，默认使用配置中的路径
            
        Returns:
            预处理器实例
        """
        if filepath is None:
            filepath = PATH_CONFIG.preprocessor_path
        
        logger.info(f"加载预处理器: {filepath}")
        
        preprocessor = cls() # 相当于: HotelDataPreprocessor()
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            preprocessor.price_tables = data['price_tables']
            preprocessor.arrival_rates = data['arrival_rates']
            preprocessor.lead_time_params = data['lead_time_params']
            preprocessor.wtp_params = data['wtp_params']
        
        logger.info("加载完成")
        return preprocessor
    
    def run_preprocessing(self, filepath: str = None) -> 'HotelDataPreprocessor':
        """
        运行完整的预处理流程
        
        Args:
            filepath: 数据文件路径
            
        Returns:
            self
        """
        logger.info("="*60)
        logger.info("开始数据预处理流程")
        logger.info("="*60)
        
        # 1. 加载数据
        self.load_data(filepath)
        
        # 2. 清洗数据
        self.clean_data()
        
        # 3. 计算价格表
        self.calculate_price_tables()
        
        # 4. 计算到达率
        self.calculate_arrival_rates()
        
        # 5. 拟合提前期分布
        self.fit_lead_time_distribution()
        
        # 6. 拟合WTP分布
        self.fit_wtp_distribution()
        
        # 7. 保存预处理器
        self.save()
        
        logger.info("="*60)
        logger.info("数据预处理完成")
        logger.info("="*60)
        
        return self
    
    def print_summary(self):
        """打印预处理结果摘要"""
        print("\n" + "="*60)
        print("数据预处理摘要")
        print("="*60)
        
        print("\n1. 价格表 (P_base / P_long):")
        print(f"   共 {len(self.price_tables['p_base'])} 个价格段")
        
        # 打印几个示例
        for month in [1, 7]:  # 淡季和旺季示例
            for is_weekend in [0, 1]:
                key = (month, is_weekend)
                p_base = self.price_tables['p_base'].get(key, 0)
                p_long = self.price_tables['p_long'].get(key, 0)
                weekend_str = "周末" if is_weekend else "工作日"
                print(f"   月份{month:2d} {weekend_str}: P_base={p_base:6.2f}, P_long={p_long:6.2f}")
        
        print("\n2. 到达率 (λ_base):")
        for month in [1, 7, 12]:
            rate = self.arrival_rates.get(month, 0)
            print(f"   月份{month:2d}: λ={rate:6.2f} 订单/天")
        
        print("\n3. 提前期分布:")
        print(f"   μ = {self.lead_time_params['mu']:.3f}")
        print(f"   σ = {self.lead_time_params['sigma']:.3f}")
        
        print("\n4. WTP分布 (示例):")
        for month in [1, 7]:
            params = self.wtp_params.get(month, {})
            print(f"   月份{month:2d}: μ={params.get('mu', 0):6.2f}, σ={params.get('sigma', 0):6.2f}")
        
        print("="*60 + "\n")


if __name__ == '__main__':
    """测试数据预处理"""
    preprocessor = HotelDataPreprocessor()
    preprocessor.run_preprocessing()
    preprocessor.print_summary()
