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

# 本地模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.config import RL_CONFIG, ENV_CONFIG, PATH_CONFIG, RANDOM_CONFIG

from src.environment.hotel_env import HotelEnvironment
from src.agent.hotel_agent import HotelAgent



def train_rl_system_with_abm(historical_data: pd.DataFrame, episodes: int = 100) -> Tuple[HotelAgent, List, List]:
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
        initial_inventory=ENV_CONFIG.initial_inventory,
        historical_data=historical_data  # 传入历史数据给ABM
    )
    
    # 根据参数选择智能体类型
    agent = HotelAgent()
    
    # 训练记录
    episode_rewards = []
    episode_revenues = []
    episode_bookings = []
    
    # 创建训练监控器
    from src.utils.training_monitor import get_training_monitor
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
        
        # 结束episode（用于Actor-Critic的标准差衰减）
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
        
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
    q_table_path = PATH_CONFIG.abm_q_table_path
    with open(q_table_path, 'wb') as f:
        pickle.dump(q_table_dict, f)
    print(f"\nQ表已保存: {q_table_path}")
        
    # 保存agent参数（兼容Q-learning和Actor-Critic）
    agent_params = {
            'algorithm_type': agent.algorithm_type,
            'n_states': agent.n_states,
            'n_actions': agent.n_actions,
            'discount_factor': agent.discount_factor,
            'epsilon_start': agent.epsilon_start,
            'epsilon_end': agent.epsilon_end,
            'epsilon_decay_steps': agent.epsilon_decay_steps,
            'q_table': dict(agent.q_table),
            'state_visit_count': dict(agent.state_visit_count) if hasattr(agent, 'state_visit_count') else {},
            'state_action_visit_count': dict(agent.state_action_visit_count) if hasattr(agent, 'state_action_visit_count') else {}
        }
    
    # 添加算法特定参数
    if agent.algorithm_type == 'q_learning':
        agent_params['learning_rate'] = agent.learning_rate
    elif agent.algorithm_type == 'actor_critic':
        agent_params['actor_lr'] = agent.algorithm.actor_lr
        agent_params['critic_lr'] = agent.algorithm.critic_lr
        agent_params['action_min'] = agent.algorithm.action_min
        agent_params['action_max'] = agent.algorithm.action_max
        agent_params['current_std'] = agent.algorithm.current_std
    
    agent_path = PATH_CONFIG.hotel_agent_path
    with open(agent_path, 'wb') as f:
        pickle.dump(agent_params, f)
    print(f"智能体参数已保存: {agent_path}")
    
    # 创建一个简单的包装对象，使其兼容后续的可视化代码
    class ABMRLSystemWrapper:
        def __init__(self, agent):
            self.agent = agent
            self.env = env

    rl_system_wrapper = ABMRLSystemWrapper(agent)
    return rl_system_wrapper, episode_rewards, episode_revenues