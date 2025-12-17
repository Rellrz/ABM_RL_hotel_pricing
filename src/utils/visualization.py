#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块 - 生成训练和评估图表
Visualization Module

功能：
1. 训练曲线可视化
2. Q表分析和可视化
3. 评估结果可视化
4. 策略分析可视化
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import os
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.hyperparameters import PATH_CONFIG, HOTEL_AGENT_CONFIG, OTA_AGENT_CONFIG, ABM_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def generate_all_plots(
    hotel_agent,
    ota_agent,
    training_history: Dict,
    eval_summary: Dict,
    preprocessor
):
    """
    生成所有可视化图表
    
    Args:
        hotel_agent: 酒店智能体
        ota_agent: OTA智能体
        training_history: 训练历史
        eval_summary: 评估摘要
        preprocessor: 数据预处理器
    """

    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("生成可视化图表...")
    
    # 1. 训练曲线
    plot_training_curves(training_history, timestamp)
    
    # 2. Q表分析
    analyze_q_tables(hotel_agent, ota_agent, timestamp)
    
    # 3. 评估结果
    plot_evaluation_results(eval_summary, timestamp)
    
    # 4. 策略热图
    plot_policy_heatmaps(hotel_agent, ota_agent, timestamp)
    
    # 5. 价格分布
    plot_price_distribution(preprocessor, timestamp)
    
    logger.info(f"所有图表已保存到: {PATH_CONFIG.figures_dir}/")


def plot_training_curves(training_history: Dict, timestamp: str):
    """
    绘制训练曲线
    
    Args:
        training_history: 训练历史
        timestamp: 时间戳
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('训练过程监控', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(training_history['episode_rewards_hotel']) + 1)
    
    # 1. 酒店收益
    ax = axes[0, 0]
    ax.plot(episodes, training_history['episode_rewards_hotel'], 
            label='酒店收益', color='#3498db', linewidth=2)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('总收益 ($)')
    ax.set_title('酒店收益变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. OTA收益
    ax = axes[0, 1]
    ax.plot(episodes, training_history['episode_rewards_ota'], 
            label='OTA收益', color='#e74c3c', linewidth=2)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('总收益 ($)')
    ax.set_title('OTA收益变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 直销订单
    ax = axes[1, 0]
    ax.plot(episodes, training_history['episode_bookings_direct'], 
            label='直销订单', color='#2ecc71', linewidth=2)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('订单数')
    ax.set_title('直销订单变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. OTA订单
    ax = axes[1, 1]
    ax.plot(episodes, training_history['episode_bookings_ota'], 
            label='OTA订单', color='#f39c12', linewidth=2)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('订单数')
    ax.set_title('OTA订单变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 入住率
    ax = axes[2, 0]
    ax.plot(episodes, training_history['episode_avg_occupancy'], 
            label='平均入住率', color='#9b59b6', linewidth=2)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('入住率')
    ax.set_title('平均入住率变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # 6. 探索率
    ax = axes[2, 1]
    ax.plot(episodes, training_history['epsilon_hotel'], 
            label='酒店探索率', color='#3498db', linewidth=2)
    ax.plot(episodes, training_history['epsilon_ota'], 
            label='OTA探索率', color='#e74c3c', linewidth=2)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('探索率 (ε)')
    ax.set_title('探索率衰减')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(PATH_CONFIG.figures_dir, f'training_curves_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 训练曲线已保存: {save_path}")


def analyze_q_tables(hotel_agent, ota_agent, timestamp: str):
    """
    分析和可视化Q表
    
    Args:
        hotel_agent: 酒店智能体
        ota_agent: OTA智能体
        timestamp: 时间戳
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q表分析', fontsize=16, fontweight='bold')
    
    # 1. 酒店Q表统计
    ax = axes[0, 0]
    hotel_q_values = hotel_agent.q_table.flatten()
    hotel_q_values = hotel_q_values[hotel_q_values != 0]  # 过滤未访问的状态
    
    ax.hist(hotel_q_values, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Q值')
    ax.set_ylabel('频数')
    ax.set_title(f'酒店Q值分布 (均值: {hotel_q_values.mean():.2f})')
    ax.grid(True, alpha=0.3)
    
    # 2. OTA Q表统计
    ax = axes[0, 1]
    ota_q_values = ota_agent.q_table.flatten()
    ota_q_values = ota_q_values[ota_q_values != 0]
    
    ax.hist(ota_q_values, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Q值')
    ax.set_ylabel('频数')
    ax.set_title(f'OTA Q值分布 (均值: {ota_q_values.mean():.2f})')
    ax.grid(True, alpha=0.3)
    
    # 3. 酒店状态访问热图
    ax = axes[1, 0]
    hotel_visit_counts = hotel_agent.state_counts
    visited_states = np.sum(hotel_visit_counts > 0)
    total_states = len(hotel_visit_counts)
    coverage = visited_states / total_states * 100
    
    # 绘制访问次数的对数分布
    visit_hist = np.histogram(np.log10(hotel_visit_counts[hotel_visit_counts > 0] + 1), bins=30)
    ax.bar(range(len(visit_hist[0])), visit_hist[0], color='#3498db', alpha=0.7)
    ax.set_xlabel('log10(访问次数 + 1)')
    ax.set_ylabel('状态数')
    ax.set_title(f'酒店状态访问分布 (覆盖率: {coverage:.1f}%)')
    ax.grid(True, alpha=0.3)
    
    # 4. OTA状态访问热图
    ax = axes[1, 1]
    ota_visit_counts = ota_agent.state_counts
    visited_states = np.sum(ota_visit_counts > 0)
    total_states = len(ota_visit_counts)
    coverage = visited_states / total_states * 100
    
    visit_hist = np.histogram(np.log10(ota_visit_counts[ota_visit_counts > 0] + 1), bins=30)
    ax.bar(range(len(visit_hist[0])), visit_hist[0], color='#e74c3c', alpha=0.7)
    ax.set_xlabel('log10(访问次数 + 1)')
    ax.set_ylabel('状态数')
    ax.set_title(f'OTA状态访问分布 (覆盖率: {coverage:.1f}%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(PATH_CONFIG.figures_dir, f'q_table_analysis_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Q表分析已保存: {save_path}")
    
    # 保存Q表到CSV
    save_q_tables_to_csv(hotel_agent, ota_agent, timestamp)


def save_q_tables_to_csv(hotel_agent, ota_agent, timestamp: str):
    """
    保存Q表到CSV文件
    
    Args:
        hotel_agent: 酒店智能体
        ota_agent: OTA智能体
        timestamp: 时间戳
    """
    # 酒店Q表
    hotel_q_data = []
    for state in range(hotel_agent.q_table.shape[0]):
        if hotel_agent.state_counts[state] > 0:  # 只保存访问过的状态
            q_values = hotel_agent.q_table[state]
            best_action = np.argmax(q_values)
            discount, commission = hotel_agent.action_to_params(best_action)
            
            row = {
                'state': state,
                'visit_count': hotel_agent.state_counts[state],
                'best_action': best_action,
                'best_discount': discount,
                'best_commission_tier': commission,
                'best_q_value': q_values[best_action],
                'avg_q_value': np.mean(q_values)
            }
            
            # 添加所有动作的Q值
            for i, q in enumerate(q_values):
                row[f'q_action_{i}'] = q
            
            hotel_q_data.append(row)
    
    hotel_df = pd.DataFrame(hotel_q_data)
    hotel_path = os.path.join(PATH_CONFIG.results_dir, f'hotel_q_table_{timestamp}.csv')
    hotel_df.to_csv(hotel_path, index=False)
    logger.info(f"✓ 酒店Q表已保存: {hotel_path}")
    
    # OTA Q表
    ota_q_data = []
    for state in range(ota_agent.q_table.shape[0]):
        if ota_agent.state_counts[state] > 0:
            q_values = ota_agent.q_table[state]
            best_action = np.argmax(q_values)
            subsidy = ota_agent.action_to_subsidy(best_action)
            
            row = {
                'state': state,
                'visit_count': ota_agent.state_counts[state],
                'best_action': best_action,
                'best_subsidy': subsidy,
                'best_q_value': q_values[best_action],
                'avg_q_value': np.mean(q_values)
            }
            
            for i, q in enumerate(q_values):
                row[f'q_action_{i}'] = q
            
            ota_q_data.append(row)
    
    ota_df = pd.DataFrame(ota_q_data)
    ota_path = os.path.join(PATH_CONFIG.results_dir, f'ota_q_table_{timestamp}.csv')
    ota_df.to_csv(ota_path, index=False)
    logger.info(f"✓ OTA Q表已保存: {ota_path}")


def plot_evaluation_results(eval_summary: Dict, timestamp: str):
    """
    绘制评估结果
    
    Args:
        eval_summary: 评估摘要
        timestamp: 时间戳
    """
    results_df = eval_summary['results_df']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('策略评估结果', fontsize=16, fontweight='bold')
    
    # 1. 每日收益（酒店）
    ax = axes[0, 0]
    ax.plot(results_df['day'], results_df['revenue_hotel'], 
            color='#3498db', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('天数')
    ax.set_ylabel('收益 ($)')
    ax.set_title('酒店每日收益')
    ax.grid(True, alpha=0.3)
    
    # 2. 每日收益（OTA）
    ax = axes[0, 1]
    ax.plot(results_df['day'], results_df['revenue_ota'], 
            color='#e74c3c', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('天数')
    ax.set_ylabel('收益 ($)')
    ax.set_title('OTA每日收益')
    ax.grid(True, alpha=0.3)
    
    # 3. 订单对比
    ax = axes[0, 2]
    ax.plot(results_df['day'], results_df['bookings_direct'], 
            label='直销', color='#2ecc71', linewidth=1.5, alpha=0.7)
    ax.plot(results_df['day'], results_df['bookings_ota'], 
            label='OTA', color='#f39c12', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('天数')
    ax.set_ylabel('订单数')
    ax.set_title('每日订单对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 库存变化（Day 0-4）
    ax = axes[1, 0]
    for i, color in enumerate(['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']):
        col = f'inventory_day{i}'
        if col in results_df.columns:
            ax.plot(results_df['day'], results_df[col], 
                   label=f'Day {i}', color=color, linewidth=1.5, alpha=0.7)
    ax.set_xlabel('天数')
    ax.set_ylabel('剩余库存')
    ax.set_title('未来5天库存变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 入住率
    ax = axes[1, 1]
    ax.plot(results_df['day'], results_df['avg_occupancy'], 
            color='#9b59b6', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('天数')
    ax.set_ylabel('入住率')
    ax.set_title('平均入住率')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # 6. 渠道收益占比
    ax = axes[1, 2]
    total_revenue_hotel = results_df['revenue_hotel'].sum()
    total_revenue_ota = results_df['revenue_ota'].sum()
    
    ax.pie([total_revenue_hotel, total_revenue_ota], 
           labels=['酒店直销', 'OTA分销'],
           colors=['#3498db', '#e74c3c'],
           autopct='%1.1f%%',
           startangle=90)
    ax.set_title('总收益渠道占比')
    
    plt.tight_layout()
    save_path = os.path.join(PATH_CONFIG.figures_dir, f'evaluation_results_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 评估结果已保存: {save_path}")


def plot_policy_heatmaps(hotel_agent, ota_agent, timestamp: str):
    """
    绘制策略热图
    
    Args:
        hotel_agent: 酒店智能体
        ota_agent: OTA智能体
        timestamp: 时间戳
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('最优策略热图', fontsize=16, fontweight='bold')
    
    # 1. 酒店折扣策略（按库存和提前期）
    ax = axes[0]
    
    # 创建热图数据
    discount_matrix = np.zeros((HOTEL_AGENT_CONFIG.n_inv_levels, HOTEL_AGENT_CONFIG.n_days_ahead))
    
    for inv_level in range(HOTEL_AGENT_CONFIG.n_inv_levels):
        for days_ahead in range(HOTEL_AGENT_CONFIG.n_days_ahead):
            # 构造一个典型状态
            state = hotel_agent.discretize_state(
                days_ahead=days_ahead,
                inventory_usage=(inv_level + 0.5) / HOTEL_AGENT_CONFIG.n_inv_levels,
                price_ratio=1.0,
                is_weekend=False,
                season=1
            )
            
            if hotel_agent.state_counts[state] > 0:
                best_action = np.argmax(hotel_agent.q_table[state])
                discount, _ = hotel_agent.action_to_params(best_action)
                discount_matrix[inv_level, days_ahead] = discount
    
    im = ax.imshow(discount_matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)
    ax.set_xlabel('提前期（天）')
    ax.set_ylabel('库存压力等级')
    ax.set_title('酒店最优折扣策略')
    ax.set_yticks(range(HOTEL_AGENT_CONFIG.n_inv_levels))
    ax.set_yticklabels(['空闲', '正常', '紧张', '告急'])
    plt.colorbar(im, ax=ax, label='折扣系数')
    
    # 2. OTA补贴策略（按库存和提前期）
    ax = axes[1]
    
    subsidy_matrix = np.zeros((OTA_AGENT_CONFIG.n_inv_levels, OTA_AGENT_CONFIG.n_days_ahead))
    
    for inv_level in range(OTA_AGENT_CONFIG.n_inv_levels):
        for days_ahead in range(OTA_AGENT_CONFIG.n_days_ahead):
            state = ota_agent.discretize_state(
                days_ahead=days_ahead,
                inventory_usage=(inv_level + 0.5) / OTA_AGENT_CONFIG.n_inv_levels,
                margin_ratio=1.2,
                is_weekend=False,
                season=1
            )
            
            if ota_agent.state_counts[state] > 0:
                best_action = np.argmax(ota_agent.q_table[state])
                subsidy = ota_agent.action_to_subsidy(best_action)
                subsidy_matrix[inv_level, days_ahead] = subsidy
    
    im = ax.imshow(subsidy_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=0.8)
    ax.set_xlabel('提前期（天）')
    ax.set_ylabel('库存压力等级')
    ax.set_title('OTA最优补贴策略')
    ax.set_yticks(range(OTA_AGENT_CONFIG.n_inv_levels))
    ax.set_yticklabels(['空闲', '正常', '紧张', '告急'])
    plt.colorbar(im, ax=ax, label='补贴系数')
    
    plt.tight_layout()
    save_path = os.path.join(PATH_CONFIG.figures_dir, f'policy_heatmaps_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 策略热图已保存: {save_path}")


def plot_price_distribution(preprocessor, timestamp: str):
    """
    绘制价格分布
    
    Args:
        preprocessor: 数据预处理器
        timestamp: 时间戳
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('价格体系分析', fontsize=16, fontweight='bold')
    
    # 提取价格数据
    p_base_values = list(preprocessor.price_tables['p_base'].values())
    p_long_values = list(preprocessor.price_tables['p_long'].values())
    
    # 1. 价格分布对比
    ax = axes[0]
    ax.hist(p_base_values, bins=20, alpha=0.6, label='P_base (70%分位)', color='#e74c3c', edgecolor='black')
    ax.hist(p_long_values, bins=20, alpha=0.6, label='P_long (中位数)', color='#3498db', edgecolor='black')
    ax.set_xlabel('价格 ($)')
    ax.set_ylabel('频数')
    ax.set_title('基准价格 vs 远期价格分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 月度价格变化
    ax = axes[1]
    
    months = range(1, 13)
    p_base_weekday = [preprocessor.get_price('p_base', m, False) for m in months]
    p_base_weekend = [preprocessor.get_price('p_base', m, True) for m in months]
    p_long_weekday = [preprocessor.get_price('p_long', m, False) for m in months]
    p_long_weekend = [preprocessor.get_price('p_long', m, True) for m in months]
    
    x = np.arange(len(months))
    width = 0.2
    
    ax.bar(x - 1.5*width, p_base_weekday, width, label='P_base 工作日', color='#e74c3c', alpha=0.8)
    ax.bar(x - 0.5*width, p_base_weekend, width, label='P_base 周末', color='#c0392b', alpha=0.8)
    ax.bar(x + 0.5*width, p_long_weekday, width, label='P_long 工作日', color='#3498db', alpha=0.8)
    ax.bar(x + 1.5*width, p_long_weekend, width, label='P_long 周末', color='#2980b9', alpha=0.8)
    
    ax.set_xlabel('月份')
    ax.set_ylabel('价格 ($)')
    ax.set_title('月度价格变化')
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(PATH_CONFIG.figures_dir, f'price_distribution_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ 价格分布已保存: {save_path}")


if __name__ == '__main__':
    """测试可视化模块"""
    print("="*60)
    print("可视化模块测试")
    print("="*60)
    print("\n请使用训练脚本中的可视化功能")
