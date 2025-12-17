#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店动态定价系统 - 主程序
基于ABM和Q-learning/贝叶斯Q-learning强化学习
"""

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

# 本地模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import BQL_CONFIG, RL_CONFIG, ENV_CONFIG, DATA_CONFIG
        
from data_preprocessing import HotelDataPreprocessor
from evaluation_plot import visualize_evaluation_results
from rl_system import HotelEnvironment, QLearningAgent

# 配置警告过滤器
warnings.filterwarnings('ignore')

# 导入随机因子配置（自动设置随机模式）
from random_factor_config import current_random_config
from config import RANDOM_CONFIG
print(f"当前随机因子配置: {current_random_config['current_status']}")

# 确保所有随机种子设置与random_factor_config一致
if current_random_config['random_mode'] == 'fixed':
    # 固定模式：使用配置中的种子
    global_random_seed = RANDOM_CONFIG['fixed_seed']
    print(f"使用固定随机种子: {global_random_seed}")
else:
    # 随机模式：使用None作为种子
    global_random_seed = None
    print("使用随机模式，不设置固定种子")

def train_rl_system_with_abm(historical_data: pd.DataFrame, episodes: int = 100) -> Tuple[QLearningAgent, List, List]:
    """
    使用ABM训练RL智能体（替代NGBoost）
    
    Args:
        historical_data: 历史数据
        episodes: 训练轮数
        use_bayesian_rl: 是否使用贝叶斯RL（暂不支持）
        
    Returns:
        Tuple[QLearningAgent, List, List]: (智能体, 奖励列表, 收益列表)
    """
    print(f"\n=== 使用ABM训练RL智能体 ({episodes}轮) ===")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ✅ 创建集成ABM的RL环境
    env = HotelEnvironment(
        initial_inventory=ENV_CONFIG['initial_inventory'],
        use_abm=True,  # 启用ABM模式
        historical_data=historical_data  # 传入历史数据给ABM
    )
    
    # 根据参数选择智能体类型
    agent = QLearningAgent(
            n_states=RL_CONFIG['n_states'],  # 3个库存等级 × 3个季节 × 2个日期类型 = 18个状态
            n_actions=RL_CONFIG['n_actions'],
            learning_rate=RL_CONFIG['learning_rate'],
            discount_factor=RL_CONFIG['discount_factor'],
            epsilon_start=RL_CONFIG['epsilon_start'],
            epsilon_end=RL_CONFIG['epsilon_end'],
            epsilon_decay_steps=RL_CONFIG['epsilon_decay_episodes']
        )
    
    # 训练记录
    episode_rewards = []
    episode_revenues = []
    episode_bookings = []
    
    # 创建训练监控器
    from training_monitor import get_training_monitor
    monitor = get_training_monitor()
    
    print("\n开始训练...")
    
    for episode in range(episodes):
        state = env.reset()  # reset()已经包含了ABM的重置
        
        total_reward = 0
        total_revenue = 0
        total_bookings = 0
        
        # 365天模拟
        for _ in range(365):
            # ✅ 为未来5天分别执行Q-learning决策
            actions_window = []
            states_window = []  # 保存每天的状态，用于后续Q表更新
            
            for day_offset in range(5):  # Day0, Day1, Day2, Day3, Day4
                # ✅ 为每一天构建独立的状态
                state_for_day = env._get_state_for_day_offset(day_offset)
                state_idx_for_day = agent.discretize_state(
                    state_for_day, 
                    state_for_day['season'], 
                    state_for_day['weekday']
                )
                states_window.append((state_for_day, state_idx_for_day))
                
                # 基于该天的状态进行决策
                action_for_day = agent.select_action(state_idx_for_day, episode)
                actions_window.append(action_for_day)
            
            # ✅ 使用5个动作执行环境step
            next_state, reward, done, info = env.step(actions_window)
            
            # ✅ 更新Q表：更新所有5天的Q值
            # 这样可以确保未来几天的高库存状态也能得到有效学习
            next_state_idx = agent.discretize_state(next_state, next_state['season'], next_state['weekday'])
            
            # 为每一天分配reward（简化方案：平均分配）
            reward_per_day = reward / 5.0
            
            for i in range(5):
                state_day_i, state_idx_day_i = states_window[i]
                action_day_i = actions_window[i]
                
                # 确定该天的next_state
                if i < 4:
                    # Day 0-3: next_state是窗口中的下一天
                    next_state_day_i, next_state_idx_day_i = states_window[i + 1]
                else:
                    # Day 4: next_state是step后的新状态（滚动后的Day 0）
                    next_state_idx_day_i = next_state_idx
                
                # 更新该天的Q值
                agent.update_q_table(state_idx_day_i, action_day_i, reward_per_day, next_state_idx_day_i, done)
            
            total_reward += reward
            total_revenue += reward  # reward就是revenue
            total_bookings += info.get('actual_bookings', 0)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_revenues.append(total_revenue)
        episode_bookings.append(total_bookings)
        
        # 记录到监控器
        current_epsilon = agent.get_epsilon(episode)
        q_stats = agent.get_q_value_stats() if hasattr(agent, 'get_q_value_stats') else None
        monitor.record_rl_episode(
            episode=episode + 1,
            avg_reward=total_reward / 365,  # 平均每天的奖励
            episode_length=365,
            exploration_rate=current_epsilon,
            q_stats=q_stats
        )
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_revenue = np.mean(episode_revenues[-10:])
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Avg Reward={avg_reward:.2f}, "
                  f"Avg Revenue=${avg_revenue:.2f}, "
                  f"Avg Bookings={np.mean(episode_bookings[-10:]):.1f}, "
                  f"ε={current_epsilon:.3f}")
    
    print("\n训练完成！")
    print(f"最终平均收益: ${np.mean(episode_revenues[-10:]):.2f}")
    print(f"最终平均预订: {np.mean(episode_bookings[-10:]):.1f}间/episode")
    
    # 生成训练曲线图
    print("\n=== 生成训练曲线图 ===")
    monitor.plot_training_curves()
    
    # 保存模型（根据智能体类型保存不同的数据）
    # 标准Q-learning：保存q_table
    q_table_dict = dict(agent.q_table)
    q_table_path = f'../02_训练模型/abm_q_table_{timestamp}.pkl'
    with open(q_table_path, 'wb') as f:
        pickle.dump(q_table_dict, f)
    print(f"\nQ表已保存: {q_table_path}")
        
    agent_params = {
            'n_states': agent.n_states,
            'n_actions': agent.n_actions,
            'learning_rate': agent.learning_rate,
            'discount_factor': agent.discount_factor,
            'epsilon_start': agent.epsilon_start,
            'epsilon_end': agent.epsilon_end,
            'epsilon_decay_steps': agent.epsilon_decay_steps,
            'q_table': dict(agent.q_table),
            'state_visit_count': dict(agent.state_visit_count) if hasattr(agent, 'state_visit_count') else {},
            'state_action_visit_count': dict(agent.state_action_visit_count) if hasattr(agent, 'state_action_visit_count') else {}
        }
    
    agent_path = f'../02_训练模型/abm_agent_{timestamp}.pkl'
    with open(agent_path, 'wb') as f:
        pickle.dump(agent_params, f)
    print(f"智能体参数已保存: {agent_path}")
    
    # 创建一个简单的包装对象，使其兼容后续的可视化代码
    class ABMRLSystemWrapper:
        def __init__(self, agent):
            self.agent = agent
            self.env = env
        
        def is_bayesian_ql_agent(self):
            return self._use_bayesian
        
        def is_standard_ql_agent(self):
            return not self._use_bayesian
    
    rl_system_wrapper = ABMRLSystemWrapper(agent)
    
    return rl_system_wrapper, episode_rewards, episode_revenues


def evaluate_trained_policy(rl_system:QLearningAgent, historical_data:pd.DataFrame):
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
    results_path = f'../04_结果输出/evaluation_results_{timestamp}.csv'
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


def main() -> None:
    """
    酒店动态定价系统主函数
    
    系统入口点，负责整个定价系统的运行流程控制，包括：
    - 环境检查和配置验证
    - 数据加载和预处理
    - NGBoost模型训练和评估
    - 强化学习系统训练
    - 定价策略模拟和结果分析
    
    Args:
        无（使用命令行参数）
        
    命令行参数：
        --data: 数据文件路径，默认../03_数据文件/hotel_bookings.csv
        --use-bayesian-rl: 使用贝叶斯Q-learning算法
        --use-abm: 使用ABM模型替代NGBoost进行需求预测
        --abm-episodes: ABM模式下的训练轮数，默认100
        --run-uuid: 运行UUID，用于Q表存储和识别
        
    运行流程：
    1. 环境检查：验证Python环境和依赖库
    2. 数据准备：加载和预处理酒店预订数据
    3. 模型训练：根据参数训练BNN和RL模型
    4. 策略模拟：运行定价策略模拟
    5. 结果分析：生成分析报告和可视化图表
    
    Note:
        - 支持模型缓存避免重复训练
        - 提供详细的训练进度和性能报告
        - 生成完整的分析报告和可视化结果
        - 支持灵活的参数配置和运行模式
    """
    parser = argparse.ArgumentParser(description='酒店动态定价系统')
    parser.add_argument('--data', type=str, default='../03_数据文件/hotel_bookings.csv',
                       help='酒店预订数据文件路径')
    parser.add_argument('--use-abm', action='store_true',
                       help='使用ABM模型替代NGBoost进行需求预测')
    parser.add_argument('--abm-episodes', type=int, default=20,
                       help='ABM模式下的训练轮数（默认20）')
    parser.add_argument('--run-uuid', type=str, default=None,
                       help='运行UUID，用于Q表存储和识别')
    # parser.add_argument('--simulate-days', type=int, default=90,
    #                    help='模拟天数')
    # parser.add_argument('--start-date', type=str, default='2017-01-01',
    #                    help='模拟开始日期 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    algorithm = "Q-learning"
    demand_model = "ABM" if args.use_abm else "NGBoost"
    print(f"酒店动态定价系统 ({demand_model} + {algorithm})")
    print("=" * 60)
    
    # 如果使用ABM，直接调用ABM训练流程
    if args.use_abm:
        print("\n=== 使用ABM模型进行需求预测 ===")
        print(f"训练轮数: {args.abm_episodes}")
        
        # 加载历史数据
        historical_data = pd.read_csv(args.data)
        historical_data = historical_data[historical_data['hotel'] == 'City Hotel'].copy()
        print(f"数据加载完成，共 {len(historical_data)} 条记录")
        
        # 使用ABM训练RL系统
        rl_system, rewards, revenues = train_rl_system_with_abm(
            historical_data, 
            episodes=args.abm_episodes
        )
        
        print("\n" + "=" * 60)
        print("ABM + RL训练完成！")
        print(f"最终平均收益: ${np.mean(revenues[-10:]):.2f}")
        print(f"最终平均奖励: {np.mean(rewards[-10:]):.2f}")
        print("=" * 60)
    
    # 运行模拟功能已移除
    
    # 模拟结果保存功能已移除
    # results_path = f'../04_结果输出/simulation_results_{start_date.strftime("%Y%m%d")}_{args.simulate_days}days.csv'
    # simulation_results.to_csv(results_path, index=False)
    # print(f"\n模拟结果已保存到：{results_path}")
    
    # 输出Q表信息（仅在非仅训练NGBoost模式下）
    
    # 使用训练好的策略进行评估仿真
    if args.use_abm:
        evaluation_results = evaluate_trained_policy(
            rl_system, 
            historical_data
        )

    # 分析和可视化Q表
    from q_table_plot import analyze_and_visualize_q_table
    analyze_and_visualize_q_table(rl_system, args)

    print("\n" + "=" * 60)
    print("系统运行完成！")
    print("=" * 60)



if __name__ == "__main__":
    # 添加超参数搜索控制逻辑  
    main()
