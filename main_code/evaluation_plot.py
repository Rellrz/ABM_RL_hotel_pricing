#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估仿真结果可视化模块
用于绘制评估模式下的各种分析图表
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import seaborn as sns
import platform



def visualize_evaluation_results(results_df: pd.DataFrame, 
                                 abm_model: Optional[object] = None,
                                 save_dir: str = '../05_分析报告'):
    """
    绘制评估仿真结果的综合可视化图表
    
    Args:
        results_df: 评估结果DataFrame，包含每天的详细数据
        abm_model: ABM模型对象，用于获取预订和入住详细信息
        save_dir: 图表保存目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    # 1. 绘制每天库存剩余量
    plot_daily_inventory(results_df, save_dir, timestamp)
    
    # 2. 绘制每天预订数量（按预订日期统计）
    if abm_model is not None:
        plot_daily_bookings_by_booking_date(abm_model, save_dir, timestamp)
        
        # 3. 绘制每天实际库存使用情况（最重要的图表）
        plot_daily_inventory_usage(results_df, save_dir, timestamp)
        
        # 4. 绘制预订vs取消对比
        plot_booking_cancellation_comparison(abm_model, results_df, save_dir, timestamp)
        
        # 5. 绘制综合对比图
        plot_comprehensive_comparison(results_df, abm_model, save_dir, timestamp)
    
    print(f"\n评估可视化图表已保存到: {save_dir}/")


def plot_daily_inventory(results_df: pd.DataFrame, save_dir: str, timestamp: str):
    """
    绘制每天库存剩余量图
    
    显示Day0到Day4的库存变化趋势
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    days = results_df['day'].values
    
    # 绘制5天窗口的库存
    inventory_cols = ['inventory_day0', 'inventory_day1', 'inventory_day2', 
                     'inventory_day3', 'inventory_day4']
    labels = ['今天剩余', '明天剩余', '后天剩余', '大后天剩余', '第5天剩余']
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
    
    plotted = False
    for col, label, color in zip(inventory_cols, labels, colors):
        if col in results_df.columns:
            ax.plot(days, results_df[col].values, label=label, 
                   linewidth=2, alpha=0.8, color=color)
            plotted = True
    
    ax.set_xlabel('天数', fontsize=12)
    ax.set_ylabel('剩余库存（间）', fontsize=12)
    ax.set_title('评估期间每天库存剩余量变化趋势', fontsize=14, fontweight='bold')
    
    # 只有在有数据时才显示图例
    if plotted:
        ax.legend(loc='best', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/evaluation_inventory_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 库存剩余量图已保存: {save_path}")


def plot_daily_bookings_by_booking_date(abm_model: object, save_dir: str, timestamp: str):
    """
    绘制每天预订数量图（按预订日期统计）
    
    统计每天有多少消费者进行了预订（不是入住日期）
    """
    # 从ABM模型的booking_history中统计每天的预订数量
    booking_by_date = {}
    
    for booking in abm_model.booking_history:
        booking_date = booking.booking_date  # 预订发生的日期
        if booking_date not in booking_by_date:
            booking_by_date[booking_date] = 0
        booking_by_date[booking_date] += 1
    
    # 转换为有序列表（确保覆盖所有天数）
    max_day = max(booking_by_date.keys()) if booking_by_date else 365
    days = list(range(max_day + 1))
    bookings = [booking_by_date.get(day, 0) for day in days]
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.bar(days, bookings, color='#3498db', alpha=0.7, edgecolor='navy', linewidth=0.5)
    ax.plot(days, bookings, color='#2c3e50', linewidth=1.5, alpha=0.6, label='趋势线')
    
    # 添加移动平均线
    if len(bookings) >= 7:
        ma_7 = pd.Series(bookings).rolling(window=7, center=True).mean()
        ax.plot(days, ma_7, color='#e74c3c', linewidth=2, 
               label='7天移动平均', linestyle='--')
    
    ax.set_xlabel('天数（预订发生日期）', fontsize=12)
    ax.set_ylabel('预订数量（间）', fontsize=12)
    ax.set_title('每天预订数量统计（按预订日期）', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    total_bookings = sum(bookings)
    avg_bookings = np.mean(bookings) if bookings else 0
    max_bookings = max(bookings) if bookings else 0
    
    stats_text = f'总预订: {total_bookings}间\n日均: {avg_bookings:.1f}间\n峰值: {max_bookings}间'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = f'{save_dir}/evaluation_bookings_by_date_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 每天预订数量图已保存: {save_path}")


def plot_booking_cancellation_comparison(abm_model: object, results_df: pd.DataFrame, 
                                         save_dir: str, timestamp: str):
    """
    绘制预订vs取消对比图
    
    对比每天的新预订数量和取消数量
    """
    # 从results_df获取每天的预订和取消数据
    if 'bookings' not in results_df.columns or 'cancellations' not in results_df.columns:
        print("⚠ 警告: 未找到预订/取消数据，跳过对比图绘制")
        return
    
    days = results_df['day'].values
    bookings = results_df['bookings'].values
    cancellations = results_df['cancellations'].values
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # 上图：预订vs取消柱状图
    x = np.arange(len(days))
    width = 0.35
    
    ax1.bar(x - width/2, bookings, width, label='新预订', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, cancellations, width, label='取消', color='#e74c3c', alpha=0.8)
    
    # 添加移动平均线
    if len(bookings) >= 7:
        ma_bookings = pd.Series(bookings).rolling(window=7, center=True).mean()
        ma_cancellations = pd.Series(cancellations).rolling(window=7, center=True).mean()
        ax1.plot(x, ma_bookings, color='#2c3e50', linewidth=2, label='预订7日均线', linestyle='--')
        ax1.plot(x, ma_cancellations, color='#c0392b', linewidth=2, label='取消7日均线', linestyle='--')
    
    ax1.set_xlabel('天数', fontsize=11)
    ax1.set_ylabel('数量（间）', fontsize=11)
    ax1.set_title('每天新预订 vs 取消数量对比', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 下图：净预订量（预订-取消）
    net_bookings = bookings - cancellations
    colors = ['#2ecc71' if nb >= 0 else '#e74c3c' for nb in net_bookings]
    
    ax2.bar(days, net_bookings, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 添加累计净预订线
    cumulative_net = np.cumsum(net_bookings)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(days, cumulative_net, color='#f39c12', linewidth=2.5, 
                 label='累计净预订', marker='o', markersize=2)
    ax2_twin.set_ylabel('累计净预订（间）', fontsize=11, color='#f39c12')
    ax2_twin.tick_params(axis='y', labelcolor='#f39c12')
    ax2_twin.legend(loc='upper left')
    
    ax2.set_xlabel('天数', fontsize=11)
    ax2.set_ylabel('净预订量（间）', fontsize=11)
    ax2.set_title('每天净预订量（预订 - 取消）', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 统计信息
    total_bookings = np.sum(bookings)
    total_cancellations = np.sum(cancellations)
    total_net = np.sum(net_bookings)
    cancel_rate = (total_cancellations / total_bookings * 100) if total_bookings > 0 else 0
    
    stats_text = f'总预订: {total_bookings:.0f}间\n总取消: {total_cancellations:.0f}间\n净预订: {total_net:.0f}间\n取消率: {cancel_rate:.1f}%'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = f'{save_dir}/evaluation_booking_cancellation_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 预订vs取消对比图已保存: {save_path}")


def plot_daily_inventory_usage(results_df: pd.DataFrame, save_dir: str, timestamp: str):
    """
    绘制每天库存剩余和占用情况图
    
    注意：inventory_day0就是当天的剩余库存，不需要再计算
    """
    if 'inventory_day0' not in results_df.columns:
        print("⚠ 警告: 未找到库存数据，跳过库存使用图绘制")
        return
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    days = results_df['day'].values
    remaining_inventory = results_df['inventory_day0'].values
    
    # 从config获取总库存（或从数据推断）
    total_inventory = 100  # 总房间数
    
    # 绘制剩余库存折线图
    ax.plot(days, remaining_inventory, color='#2ecc71', linewidth=2.5, 
           label='剩余可用库存', marker='o', markersize=3, alpha=0.8)
    
    # 添加总库存参考线
    ax.axhline(y=total_inventory, color='#e74c3c', linestyle='--', linewidth=2, 
              label=f'总库存容量 ({total_inventory}间)', alpha=0.7)
    
    # 添加零库存警戒线
    ax.axhline(y=0, color='#c0392b', linestyle='-', linewidth=1.5, 
              label='零库存线', alpha=0.5)
    
    # 填充区域
    ax.fill_between(days, 0, remaining_inventory, color='#2ecc71', alpha=0.2)
    ax.fill_between(days, remaining_inventory, total_inventory, color='#e74c3c', alpha=0.1)
    
    ax.set_xlabel('天数', fontsize=12)
    ax.set_ylabel('库存数量（间）', fontsize=12)
    ax.set_title('每天剩余库存变化趋势', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, total_inventory * 1.1)
    
    # 添加统计信息
    avg_remaining = np.mean(remaining_inventory)
    max_remaining = np.max(remaining_inventory)
    min_remaining = np.min(remaining_inventory)
    avg_occupancy_rate = ((total_inventory - avg_remaining) / total_inventory * 100) if total_inventory > 0 else 0
    
    # 统计零库存天数
    zero_inventory_days = np.sum(remaining_inventory == 0)
    
    stats_text = f'平均剩余: {avg_remaining:.1f}间\n最大剩余: {max_remaining:.0f}间\n最小剩余: {min_remaining:.0f}间\n平均占用率: {avg_occupancy_rate:.1f}%\n零库存天数: {zero_inventory_days}天'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_path = f'{save_dir}/evaluation_inventory_usage_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 库存使用情况图已保存: {save_path}")


def plot_comprehensive_comparison(results_df: pd.DataFrame, abm_model: object, 
                                  save_dir: str, timestamp: str):
    """
    绘制综合对比图：库存、预订、入住、收益的四合一分析
    """
    # 准备数据
    days = results_df['day'].values
    inventory_day0 = results_df['inventory_day0'].values if 'inventory_day0' in results_df.columns else None
    revenue = results_df['revenue'].values if 'revenue' in results_df.columns else None
    
    # 统计预订数据
    booking_by_date = {}
    
    for booking in abm_model.booking_history:
        # 预订日期统计
        booking_date = booking.booking_date
        if booking_date not in booking_by_date:
            booking_by_date[booking_date] = 0
        booking_by_date[booking_date] += 1
    
    max_day = int(max(days)) if len(days) > 0 else 365
    all_days = list(range(max_day + 1))
    bookings = [booking_by_date.get(day, 0) for day in all_days]
    
    # 从results_df获取每天的预订和取消数据
    daily_bookings = results_df['bookings'].values if 'bookings' in results_df.columns else None
    daily_cancellations = results_df['cancellations'].values if 'cancellations' in results_df.columns else None
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('评估仿真综合分析仪表盘', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 库存剩余量
    ax1 = axes[0, 0]
    if inventory_day0 is not None:
        ax1.fill_between(days, 0, inventory_day0, color='#3498db', alpha=0.3)
        ax1.plot(days, inventory_day0, color='#2c3e50', linewidth=2, label='当天剩余库存')
        ax1.set_ylabel('剩余库存（间）', fontsize=11)
        ax1.set_title('① 每天库存剩余量', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
    
    # 2. 每天预订数量
    ax2 = axes[0, 1]
    ax2.bar(all_days, bookings, color='#9b59b6', alpha=0.7, edgecolor='purple', linewidth=0.5)
    if len(bookings) >= 7:
        ma_7 = pd.Series(bookings).rolling(window=7, center=True).mean()
        ax2.plot(all_days, ma_7, color='#e74c3c', linewidth=2.5, label='7天移动平均')
    ax2.set_ylabel('预订数量（间）', fontsize=11)
    ax2.set_title('② 每天预订数量（按预订日期）', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 每天库存占用率
    ax3 = axes[1, 0]
    if inventory_day0 is not None:
        total_inventory = 100  # 总库存
        occupancy_rate = (total_inventory - inventory_day0) / total_inventory * 100
        
        # 绘制占用率柱状图
        colors = ['#e74c3c' if rate > 80 else '#f39c12' if rate > 60 else '#2ecc71' for rate in occupancy_rate]
        ax3.bar(days, occupancy_rate, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 添加移动平均线
        if len(occupancy_rate) >= 7:
            ma_7 = pd.Series(occupancy_rate).rolling(window=7, center=True).mean()
            ax3.plot(days, ma_7, color='#34495e', linewidth=2.5, label='7天移动平均')
        
        # 添加参考线
        ax3.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5, label='高占用(80%)')
        ax3.axhline(y=60, color='#f39c12', linestyle='--', linewidth=1, alpha=0.5, label='中占用(60%)')
        
        ax3.set_xlabel('天数', fontsize=11)
        ax3.set_ylabel('占用率（%）', fontsize=11)
        ax3.set_title('③ 每天库存占用率变化', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 105)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 每天收益
    ax4 = axes[1, 1]
    if revenue is not None:
        colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in revenue]
        ax4.bar(days, revenue, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 添加累计收益线
        cumulative_revenue = np.cumsum(revenue)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(days, cumulative_revenue, color='#f39c12', linewidth=2.5, 
                     label='累计收益', marker='o', markersize=2)
        ax4_twin.set_ylabel('累计收益（$）', fontsize=11, color='#f39c12')
        ax4_twin.tick_params(axis='y', labelcolor='#f39c12')
        ax4_twin.legend(loc='upper left')
        
        ax4.set_xlabel('天数', fontsize=11)
        ax4.set_ylabel('每日收益（$）', fontsize=11)
        ax4.set_title('④ 每天收益变化', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = f'{save_dir}/evaluation_comprehensive_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 综合对比图已保存: {save_path}")


def plot_inventory_heatmap(results_df: pd.DataFrame, save_dir: str, timestamp: str):
    """
    绘制5天库存窗口热力图
    
    展示每天对未来5天的库存预测
    """
    inventory_cols = ['inventory_day0', 'inventory_day1', 'inventory_day2', 
                     'inventory_day3', 'inventory_day4']
    
    # 检查列是否存在
    available_cols = [col for col in inventory_cols if col in results_df.columns]
    if not available_cols:
        print("⚠ 警告: 未找到库存窗口数据，跳过热力图绘制")
        return
    
    # 构建热力图数据
    inventory_matrix = results_df[available_cols].values.T
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(inventory_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    
    # 设置坐标轴
    ax.set_yticks(range(len(available_cols)))
    ax.set_yticklabels(['Day 0', 'Day 1', 'Day 2', 'Day 3', 'Day 4'][:len(available_cols)])
    ax.set_xlabel('仿真天数', fontsize=12)
    ax.set_ylabel('未来天数偏移', fontsize=12)
    ax.set_title('5天库存窗口热力图', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('库存数量（间）', fontsize=11)
    
    plt.tight_layout()
    save_path = f'{save_dir}/evaluation_inventory_heatmap_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 库存热力图已保存: {save_path}")


if __name__ == "__main__":
    # 测试代码
    print("评估可视化模块已加载")
    print("使用方法: visualize_evaluation_results(results_df, abm_model)")
