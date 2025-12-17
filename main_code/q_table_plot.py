# 标准库导入
import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

# 第三方库导入
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def analyze_and_visualize_q_table(rl_system, args):
    """
    分析和可视化Q表
    
    Args:
        rl_system: RL系统实例
        args: 命令行参数
    """
    print(f"\n=== Q表信息 ===")
    if hasattr(rl_system, 'agent'):
        # 获取Q值统计
        q_stats = rl_system.agent.get_q_value_stats()
        if q_stats:
            print(f"Q值统计:")
            print(f"  平均Q值: {q_stats['mean_q_value']:.2f}")
            print(f"  Q值标准差: {q_stats['std_q_value']:.2f}")
            print(f"  最小Q值: {q_stats['min_q_value']:.2f}")
            print(f"  最大Q值: {q_stats['max_q_value']:.2f}")
            print(f"  总状态访问次数: {q_stats['num_state_visits']}")
            print(f"  零值Q值占比: {q_stats['zero_q_percentage']:.1f}%")
            print(f"  探索覆盖率: {q_stats['exploration_coverage']:.1f}%")
            print(f"  已探索状态-动作对: {q_stats['explored_state_actions']}/{q_stats['total_state_actions']}")
        
        # 显示Q表内容
        q_table = rl_system.agent.q_table
        print(f"\nQ表状态数量: {len(q_table)}")
        
        # 显示前10个状态的Q值
        print(f"\n前10个状态的Q值:")
        # 36个动作组合配置（统一动作空间）
        online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个）
        offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个）
        
        # 生成36个动作的价格映射
        prices = []
        for online_idx in range(6):
            for offline_idx in range(6):
                prices.append(f"线上{online_prices[online_idx]}线下{offline_prices[offline_idx]}")
        for i, (state, q_values) in enumerate(list(q_table.items())[:10]):
            best_action = np.argmax(q_values)
            # 确保best_action在有效范围内
            if best_action < len(prices):
                print(f"状态 {state}: {[f'{q:.1f}' for q in q_values]} -> 最佳动作: {best_action} (价格: {prices[best_action]}元)")
            else:
                print(f"状态 {state}: {[f'{q:.1f}' for q in q_values]} -> 最佳动作: {best_action} (价格: 索引超出范围)")
        
        if len(q_table) > 10:
            print(f"... 还有 {len(q_table) - 10} 个状态")
        
        # 保存Q表到CSV文件
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据算法类型获取Q值数据
        q_table_data = []
        # 36个动作组合配置（统一动作空间）
        online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个）
        offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个）
        
        # 生成36个动作的价格映射
        prices = []
        for online_idx in range(6):
            for offline_idx in range(6):
                prices.append(f"线上{online_prices[online_idx]}线下{offline_prices[offline_idx]}")
        
        # 标准Q-learning
        q_data = rl_system.agent.q_table
        for state, q_values in q_data.items():
            best_action = np.argmax(q_values)
            
            # 动态创建动作列
            row = {'state': state}
            
            # 添加所有动作的Q值
            for i in range(len(q_values)):
                row[f'action_{i}'] = q_values[i]
            
            # 添加最佳动作信息（36个动作组合映射）
            if best_action < len(prices):
                best_price = prices[best_action]
                row.update({
                    'best_action': best_action,
                    'best_price': best_price,
                    'best_value': q_values[best_action]
                })
            else:
                # 对于超出36个动作的情况，显示警告但使用实际值
                row.update({
                    'best_action': best_action,
                    'best_price': f'超出范围({best_action})',
                    'best_value': q_values[best_action]
                })
            
            q_table_data.append(row)
        
        if q_table_data:
            q_table_df = pd.DataFrame(q_table_data)
            
            # 保存到CSV
            q_table_csv_path = f'../05_分析报告/q_table_main_{timestamp}.csv'
            q_table_df.to_csv(q_table_csv_path, index=False)
            print(f"\nQ表已保存到CSV文件: {q_table_csv_path}")
            
            # 如果提供了run_uuid，则尝试将Q表数据存储到临时文件中
            if args.run_uuid:
                try:
                    # 将Q表数据转换为字符串格式
                    q_table_str = q_table_df.to_csv(index=False)
                    
                    # 创建临时文件存储Q表数据
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, f"q_table_{args.run_uuid}.csv")
                    
                    # 将Q表数据写入临时文件
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(q_table_str)
                    
                    print(f"Q表数据已存储到临时文件: {temp_file_path}")
                except Exception as e:
                    print(f"无法将Q表数据存储到临时文件: {e}")
            
            # 同时保存Q表统计信息
            if q_stats:
                stats_data = {
                    'total_states': len(q_table_data),
                    'mean_q_value': q_stats['mean_q_value'],
                    'total_visits': q_stats['num_state_visits'],
                    'zero_q_percentage': q_stats['zero_q_percentage'],
                    'exploration_coverage': q_stats['exploration_coverage'],
                    'explored_state_actions': q_stats['explored_state_actions'],
                    'total_state_actions': q_stats['total_state_actions']
                }
                
                # 标准Q-learning的统计信息
                if 'std_q_value' in q_stats:
                    stats_data['std_q_value'] = q_stats['std_q_value']
                if 'min_q_value' in q_stats:
                    stats_data['min_q_value'] = q_stats['min_q_value']
                if 'max_q_value' in q_stats:
                    stats_data['max_q_value'] = q_stats['max_q_value']
                stats_data['algorithm'] = 'Standard Q-Learning'
                
                stats_df = pd.DataFrame([stats_data])
                
                stats_csv_path = f'../05_分析报告/q_table_stats_{timestamp}.csv'
                stats_df.to_csv(stats_csv_path, index=False)
                print(f"Q表统计信息已保存到: {stats_csv_path}")
        
        # 绘制Q表热力图
        import seaborn as sns
        
        print("\n=== 开始绘制Q表热力图 ===")
        
        # 设置中文字体 - 添加更多备选字体确保兼容性
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取Q值数据
        q_data = rl_system.agent.q_table
        states = sorted(q_data.keys())
        # 获取实际的动作数量（从Q表中第一个状态获取）
        num_actions = len(q_data[states[0]]) if states else 36
        actions = list(range(num_actions))
        
        # 创建Q值矩阵
        q_matrix = np.zeros((len(states), num_actions))
        for i, state in enumerate(states):
            q_matrix[i, :] = q_data[state]
        
        # 创建状态标签（库存等级 + 季节 + 日期类型）
        state_labels = []
        for state in states:
            # 状态编码：库存等级(0-4) × 3(季节) × 2(日期类型) = 30种状态
            state_value = state
            inventory_level = state_value // 6  # 5种库存等级 (0-4)
            remaining = state_value % 6
            season = remaining // 2  # 3种季节 (0-2)
            day_type = remaining % 2  # 2种日期类型 (0-1)
            
            # 库存等级描述 - 按照实际数值范围命名
            inventory_descriptions = ['1级库存', '2级库存', '3级库存']
            # 季节描述
            season_descriptions = ['淡季', '平季', '旺季']
            # 日期类型描述
            day_type_descriptions = ['工作日', '周末']
            
            # 使用实际换行符而不是转义字符
            state_label = f"{inventory_descriptions[inventory_level]}\n{season_descriptions[season]}\n{day_type_descriptions[day_type]}"
            state_labels.append(state_label)
        
        # 动作标签（价格）- 6×6价格组合格式
        online_prices = [80, 90, 100, 110, 120, 130]      # 线上价格档位（6个动作）
        offline_prices = [90, 105, 120, 135, 150, 165]    # 线下价格档位（6个动作）
        
        # 生成36个动作的标签（线上价格×线下价格组合）
        action_labels = []
        for online_idx in range(6):
            for offline_idx in range(6):
                online_price = online_prices[online_idx]
                offline_price = offline_prices[offline_idx]
                action_labels.append(f'线上¥{online_price}\n线下¥{offline_price}')
        
        # 创建Q值热力图 - 增加宽度防止文字重叠
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 标准Q-learning热力图
        sns.heatmap(q_matrix, 
                    xticklabels=action_labels, 
                    yticklabels=state_labels,
                    cmap='RdYlBu_r', 
                    center=0,
                    annot=True, 
                    fmt='.1f',
                    cbar_kws={'label': 'Q值'},
                    ax=ax)
        
        # 设置标题和标签
        ax.set_title(f'Q值热力图 - 酒店动态定价策略', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
        ax.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
        
        # 改善坐标轴标签显示 - 防止重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), fontsize=9)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存热力图
        heatmap_path = f'../04_结果输出/q_table_heatmap_{timestamp}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Q值热力图已保存到: {heatmap_path}")
        
        # 显示热力图
        plt.show()
        
        # 创建最佳策略热力图
        print("\n=== 绘制最佳策略热力图 ===")
        
        # 创建最佳动作矩阵
        best_action_matrix = np.zeros((len(states), len(actions)))
        for i, state in enumerate(states):
            best_action = np.argmax(q_data[state])
            best_action_matrix[i, best_action] = 1
            
        # 显示策略分析信息
        print(f"\n=== 策略分析 ===")
        print(f"总状态数: {len(states)}")
        print(f"动作数: {num_actions} (6×6价格组合模式)")
        
        # 将矩阵转换为整数类型以避免格式化错误
        best_action_matrix = best_action_matrix.astype(int)
        
        # 创建最佳策略热力图 - 增加宽度防止文字重叠
        fig2, ax2 = plt.subplots(figsize=(16, 10))
        
        # 使用离散颜色映射
        cmap = plt.cm.get_cmap('RdYlBu', 2)
        sns.heatmap(best_action_matrix, 
                    xticklabels=action_labels, 
                    yticklabels=state_labels,
                    cmap=cmap, 
                    vmin=0, vmax=1,
                    annot=True, 
                    fmt='d',  
                    cbar_kws={'label': '是否为最佳动作', 'ticks': [0, 1]},
                    ax=ax2)
        
        # 设置标题和标签
        ax2.set_title(f'最佳策略热力图 - 酒店动态定价', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('定价动作（价格）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('状态（库存等级 + 季节 + 日期类型）', fontsize=12, fontweight='bold')
        
        # 改善坐标轴标签显示 - 防止重叠
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax2.get_yticklabels(), fontsize=9)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存最佳策略热力图
        best_policy_path = f'../04_结果输出/best_policy_heatmap_{timestamp}.png'
        plt.savefig(best_policy_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 最佳策略热力图已保存到: {best_policy_path}")
        
        # 显示最佳策略热力图
        plt.show()
        
        print("[OK] Q值热力图绘制完成")