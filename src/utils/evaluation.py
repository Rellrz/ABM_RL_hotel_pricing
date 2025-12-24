# 标准库导入
import argparse
import os
import pickle
import sys
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

# 第三方库导入
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from configs.config import PATH_CONFIG

# 本地模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent.hotel_agent import HotelAgent
from configs.config import ENV_CONFIG
        
from src.plot.evaluation_plot import visualize_evaluation_results

def evaluate_trained_policy(rl_system:HotelAgent):
    """
    使用训练好的Q表进行一轮评估仿真
    
    Args:
        rl_system: 训练好的RL系统实例
        historical_data: 历史数据DataFrame
        
    Returns:
        dict: 评估结果，包含总收益、总预订、取消率等指标
    """
    print("\n" + "=" * 60)
    print("开始使用训练好的策略进行评估仿真...")
    print("=" * 60)
    
    # 重置环境
    rl_system = rl_system
    agent = rl_system.agent
    env = rl_system.env
    state = env.reset()
    
    # 评估指标
    total_revenue = 0.0
    total_bookings = 0
    total_cancellations = 0
    total_gross_revenue = 0.0
    total_refund = 0.0
    daily_results = []
    
    # 设置为贪婪策略（epsilon=0，不探索）
    original_epsilon_end = agent.epsilon_end
    agent.epsilon_end = 0.0
    final_inventory_list = []

    # 运行365天的仿真
    for day in range(365):
        # 为未来5天分别选择最佳动作（贪婪策略）
        actions_window = []
        states_window = []
        
        for day_offset in range(5):
            # 为每一天构建独立的状态
            state_for_day = env._get_state_for_day_offset(day_offset)
            state_idx_for_day = agent.discretize_state(
                state_for_day, 
                state_for_day['season'], 
                state_for_day['weekday']
            )
            states_window.append((state_for_day, state_idx_for_day))
            
            # 标准Q-learning：选择Q值最大的动作
            if state_idx_for_day in agent.q_table:
                action_for_day = np.argmax(agent.q_table[state_idx_for_day])
            else:
                action_for_day = np.random.randint(0, agent.n_actions)
            
            actions_window.append(action_for_day)
        
        # 执行动作
        next_state, reward, done, info = env.step(actions_window)
        
        # 记录指标
        total_revenue += reward  # reward已经是净收益（扣除退款）
        total_bookings += info.get('actual_bookings', 0)
        
        # 从ABM模型获取详细统计
        if hasattr(env, 'abm_model') and env.abm_model is not None:
            abm_stats = env.abm_model.daily_stats[-1] if env.abm_model.daily_stats else {}
            cancellations = abm_stats.get('cancellations', 0)
            gross_revenue = abm_stats.get('gross_revenue', 0.0)
            refund = abm_stats.get('cancellation_refund', 0.0)
            
            total_cancellations += cancellations
            total_gross_revenue += gross_revenue
            total_refund += refund

            final_inventory_list.append(env.current_inventory)
            
            daily_inv = env.abm_model.daily_available_rooms
            inventory_window = []
            for i in range(5):
                day_key = env.day + i
                inv = daily_inv.get(day_key, 0)
                inventory_window.append(inv)
            
            daily_results.append({
                'day': day,
                'revenue': reward,
                'gross_revenue': gross_revenue,
                'refund': refund,
                'bookings': info.get('actual_bookings', 0),
                'cancellations': cancellations,
                'inventory_day0': inventory_window[0],  # 今天剩余库存
                'inventory_day1': inventory_window[1],  # 明天剩余库存
                'inventory_day2': inventory_window[2],  # 后天剩余库存
                'inventory_day3': inventory_window[3],  # 大后天剩余库存
                'inventory_day4': inventory_window[4],  # 第5天剩余库存
                'actions': actions_window
            })
        
        if done:
            break
    
    # 恢复原始epsilon_end
    agent.epsilon_end = original_epsilon_end
    
    # 计算评估指标
    avg_daily_revenue = total_revenue / 365
    avg_daily_bookings = total_bookings / 365
    cancellation_rate = (total_cancellations / total_bookings * 100) if total_bookings > 0 else 0
    refund_rate = (total_refund / total_gross_revenue * 100) if total_gross_revenue > 0 else 0
    
    # 打印评估结果
    print(f"\n每天最终库存情况: {final_inventory_list}")
    print("\n" + "=" * 60)
    print("评估仿真完成！")
    print("=" * 60)
    print(f"\n总体指标:")
    print(f"  总净收益: ${total_revenue:,.2f}")
    print(f"  总毛收益: ${total_gross_revenue:,.2f}")
    print(f"  总退款: ${total_refund:,.2f}")
    print(f"  总预订: {total_bookings:,} 间")
    print(f"  总取消: {total_cancellations:,} 间")
    print(f"\n平均指标:")
    print(f"  日均净收益: ${avg_daily_revenue:,.2f}")
    print(f"  日均预订: {avg_daily_bookings:.1f} 间")
    print(f"  取消率: {cancellation_rate:.2f}%")
    print(f"  退款率: {refund_rate:.2f}%")
    
    # 保存评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(daily_results)
    results_path = PATH_CONFIG.results_path.format(timestamp=timestamp)
    results_df.to_csv(results_path, index=False)
    print(f"\n评估详细结果已保存到: {results_path}")
    
    # 生成评估可视化图表
    print("\n" + "=" * 60)
    print("开始生成评估可视化图表...")
    print("=" * 60)
    try:
        visualize_evaluation_results(results_df, env.abm_model, save_dir='../07_需求图')
        print("=" * 60)
        print("评估可视化完成！")
        print("=" * 60)
    except Exception as e:
        print(f"⚠ 警告: 评估可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 返回评估结果
    evaluation_results = {
        'total_revenue': total_revenue,
        'total_gross_revenue': total_gross_revenue,
        'total_refund': total_refund,
        'total_bookings': total_bookings,
        'total_cancellations': total_cancellations,
        'avg_daily_revenue': avg_daily_revenue,
        'avg_daily_bookings': avg_daily_bookings,
        'cancellation_rate': cancellation_rate,
        'refund_rate': refund_rate,
        'daily_results': daily_results
    }
    
    return evaluation_results