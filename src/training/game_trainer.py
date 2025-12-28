"""
酒店-OTA博弈训练器

实现酒店和OTA的双层博弈训练
支持多种训练模式：
1. fixed_ota: 固定OTA策略，训练酒店
2. alternating: 交替训练两个Agent
3. simultaneous: 同步训练两个Agent
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime

from configs.config import RL_CONFIG, ENV_CONFIG
from src.environment.hotel_env import HotelEnvironment
from src.agent.hotel_agent_dual_channel import HotelAgentDualChannel
from src.agent.ota_agent import OTAAgent


def train_game_system(historical_data: pd.DataFrame, 
                     episodes: int = 100,
                     training_mode: str = 'simultaneous') -> Tuple[HotelAgentDualChannel, OTAAgent, List, List, List]:
    """
    训练酒店-OTA博弈系统
    
    Args:
        historical_data: 历史数据
        episodes: 训练轮数
        training_mode: 训练模式
            - 'fixed_ota': 固定OTA策略，只训练酒店
            - 'alternating': 交替训练
            - 'simultaneous': 同步训练
    
    Returns:
        hotel_agent: 酒店Agent
        ota_agent: OTA Agent
        episode_rewards_hotel: 酒店收益列表
        episode_rewards_ota: OTA利润列表
        episode_info: 详细信息列表
    """
    print(f"\n=== 训练酒店-OTA博弈系统 ({episodes}轮) ===")
    print(f"训练模式: {training_mode}")
    print(f"佣金率: {RL_CONFIG.commission_rate * 100:.1f}%")
    print(f"补贴比例范围: {RL_CONFIG.subsidy_ratio_min * 100:.1f}% - {RL_CONFIG.subsidy_ratio_max * 100:.1f}%")
    
    # 创建环境
    env = HotelEnvironment(
        initial_inventory=ENV_CONFIG.initial_inventory,
        historical_data=historical_data
    )
    
    # 创建酒店Agent
    hotel_agent = HotelAgentDualChannel(
        n_states=RL_CONFIG.n_states,
        commission_rate=RL_CONFIG.commission_rate,
        online_price_min=RL_CONFIG.online_price_min,
        online_price_max=RL_CONFIG.online_price_max,
        offline_price_min=RL_CONFIG.offline_price_min,
        offline_price_max=RL_CONFIG.offline_price_max,
        n_samples=RL_CONFIG.cem_n_samples,
        elite_frac=RL_CONFIG.cem_elite_frac,
        initial_std=RL_CONFIG.initial_std,
        min_std=RL_CONFIG.min_std,
        std_decay=RL_CONFIG.std_decay
    )
    
    # 创建OTA Agent
    ota_agent = OTAAgent(
        commission_rate=RL_CONFIG.commission_rate,
        subsidy_ratio_min=RL_CONFIG.subsidy_ratio_min,
        subsidy_ratio_max=RL_CONFIG.subsidy_ratio_max,
        n_states=120,  # OTA状态空间更大
        n_samples=RL_CONFIG.cem_n_samples,
        elite_frac=RL_CONFIG.cem_elite_frac,
        initial_std=0.2,
        min_std=0.02,
        std_decay=RL_CONFIG.std_decay
    )
    
    # 训练记录
    episode_rewards_hotel = []
    episode_rewards_ota = []
    episode_info = []
    
    # 创建训练监控器
    from src.utils.training_monitor import get_training_monitor
    monitor = get_training_monitor()
    
    print("\n开始训练...")
    print(f"✅ 酒店Agent: 双CEM（线上基础价格 + 线下价格）")
    print(f"✅ OTA Agent: CEM（补贴决策）")
    
    for episode in range(episodes):
        state = env.reset()
        
        total_reward_hotel = 0.0
        total_reward_ota = 0.0
        total_bookings_online = 0
        total_bookings_offline = 0
        total_subsidy = 0.0
        
        # 365天模拟
        for day in range(365):
            # 1. 酒店决策线上基础价格和线下价格
            action_hotel = hotel_agent.select_action(state, deterministic=False)
            price_online_base, price_offline = action_hotel
            
            # 2. OTA决策补贴比例
            if training_mode == 'fixed_ota':
                # 固定OTA策略：基于价格差异的简单规则
                price_gap = price_offline - price_online_base
                if price_gap > 20:
                    subsidy_ratio = 0.2  # 补贴佣金的20%
                elif price_gap > 10:
                    subsidy_ratio = 0.5  # 补贴佣金的50%
                else:
                    subsidy_ratio = 0.7  # 补贴佣金的70%
            else:
                # 使用OTA Agent决策补贴比例
                subsidy_ratio = ota_agent.select_action(
                    price_online_base, price_offline, state, deterministic=False
                )
            
            # 3. 计算补贴金额和最终线上价格
            # 预估佣金收入（假设有预订）
            estimated_commission = price_online_base * RL_CONFIG.commission_rate
            # 补贴金额 = 佣金 * 补贴比例
            subsidy_amount = estimated_commission * subsidy_ratio
            # 最终线上价格
            price_online_final = price_online_base - subsidy_amount
            
            # 4. 环境执行（传入最终价格）
            # 注意：这里需要修改环境以支持5天窗口
            # 简化版：只传入当天价格
            next_state, reward, done, info = env.step([price_online_final, price_offline])
            
            # 5. 计算各方收益
            bookings_online = info.get('new_bookings_online', 0)
            bookings_offline = info.get('new_bookings_offline', 0)
            
            # 酒店收益（扣除佣金）
            revenue_hotel = hotel_agent.calculate_revenue(
                bookings_online, bookings_offline,
                price_online_base, price_offline
            )
            
            # OTA利润（基于补贴比例）
            profit_ota = ota_agent.calculate_profit(
                bookings_online, price_online_base, subsidy_ratio
            )
            
            # 计算实际补贴金额（用于统计）
            actual_subsidy_amount = (bookings_online * price_online_base * 
                                    RL_CONFIG.commission_rate * subsidy_ratio)
            
            # 6. 更新Agent
            # 更新酒店Agent（传入实际补贴金额用于学习）
            hotel_agent.update(state, action_hotel, revenue_hotel, next_state, done, actual_subsidy_amount)
            
            # 更新OTA Agent（除非是fixed_ota模式）
            if training_mode != 'fixed_ota':
                ota_agent.update(
                    price_online_base, price_offline, state,
                    subsidy_ratio, profit_ota, next_state, done
                )
            
            # 累积统计
            total_reward_hotel += revenue_hotel
            total_reward_ota += profit_ota
            total_bookings_online += bookings_online
            total_bookings_offline += bookings_offline
            total_subsidy += actual_subsidy_amount
            
            state = next_state
            if done:
                break
        
        # Episode结束
        hotel_agent.end_episode()
        if training_mode != 'fixed_ota':
            ota_agent.end_episode()
        
        # 记录
        episode_rewards_hotel.append(total_reward_hotel)
        episode_rewards_ota.append(total_reward_ota)
        episode_info.append({
            'episode': episode + 1,
            'hotel_revenue': total_reward_hotel,
            'ota_profit': total_reward_ota,
            'bookings_online': total_bookings_online,
            'bookings_offline': total_bookings_offline,
            'total_subsidy': total_subsidy,
            'avg_subsidy_amount': total_subsidy / max(1, total_bookings_online),
            'avg_subsidy_ratio': subsidy_ratio  # 最后一天的补贴比例
        })
        
        # 监控
        exploration_rate = hotel_agent.get_epsilon(episode)
        monitor.record_rl_episode(
            episode=episode + 1,
            avg_reward=total_reward_hotel / 365,
            episode_length=365,
            exploration_rate=exploration_rate,
            q_stats=None
        )
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_hotel = np.mean(episode_rewards_hotel[-10:])
            avg_ota = np.mean(episode_rewards_ota[-10:])
            avg_bookings_online = np.mean([info['bookings_online'] for info in episode_info[-10:]])
            avg_bookings_offline = np.mean([info['bookings_offline'] for info in episode_info[-10:]])
            avg_subsidy_amount = np.mean([info['avg_subsidy_amount'] for info in episode_info[-10:]])
            avg_subsidy_ratio = np.mean([info['avg_subsidy_ratio'] for info in episode_info[-10:]])
            
            print(f"Episode {episode + 1}/{episodes}: "
                  f"Hotel=${avg_hotel:.2f}, "
                  f"OTA=${avg_ota:.2f}, "
                  f"Online={avg_bookings_online:.1f}, "
                  f"Offline={avg_bookings_offline:.1f}, "
                  f"SubsidyRatio={avg_subsidy_ratio*100:.1f}%, "
                  f"SubsidyAmt={avg_subsidy_amount:.2f}元, "
                  f"Explore={exploration_rate:.3f}")
    
    print("\n训练完成！")
    
    # 打印最终统计
    print("\n=== 最终统计 ===")
    hotel_stats = hotel_agent.get_statistics()
    ota_stats = ota_agent.get_statistics()
    
    print(f"\n酒店Agent:")
    print(f"  总收益: ${hotel_stats['total_revenue']:.2f}")
    print(f"  线上收益: ${hotel_stats['total_revenue_online']:.2f} ({hotel_stats['online_revenue_ratio']*100:.1f}%)")
    print(f"  线下收益: ${hotel_stats['total_revenue_offline']:.2f} ({(1-hotel_stats['online_revenue_ratio'])*100:.1f}%)")
    print(f"  平均每轮收益: ${hotel_stats['avg_revenue_per_episode']:.2f}")
    
    print(f"\nOTA Agent:")
    print(f"  总利润: ${ota_stats['total_profit']:.2f}")
    print(f"  总佣金收入: ${ota_stats['total_commission']:.2f}")
    print(f"  总补贴支出: ${ota_stats['total_subsidy_cost']:.2f}")
    print(f"  补贴率: {ota_stats['subsidy_ratio']*100:.1f}%")
    print(f"  平均每轮利润: ${ota_stats['avg_profit_per_episode']:.2f}")
    
    return hotel_agent, ota_agent, episode_rewards_hotel, episode_rewards_ota, episode_info


def plot_game_results(episode_rewards_hotel: List[float],
                     episode_rewards_ota: List[float],
                     episode_info: List[Dict],
                     save_path: str = None):
    """
    绘制博弈训练结果
    
    Args:
        episode_rewards_hotel: 酒店收益列表
        episode_rewards_ota: OTA利润列表
        episode_info: 详细信息列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 收益曲线
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards_hotel, label='Hotel Revenue', alpha=0.7)
    ax1.plot(episode_rewards_ota, label='OTA Profit', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Revenue/Profit ($)')
    ax1.set_title('Hotel vs OTA Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预订量对比
    ax2 = axes[0, 1]
    bookings_online = [info['bookings_online'] for info in episode_info]
    bookings_offline = [info['bookings_offline'] for info in episode_info]
    ax2.plot(bookings_online, label='Online Bookings', alpha=0.7)
    ax2.plot(bookings_offline, label='Offline Bookings', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Bookings')
    ax2.set_title('Online vs Offline Bookings')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均补贴
    ax3 = axes[1, 0]
    avg_subsidy = [info['avg_subsidy'] for info in episode_info]
    ax3.plot(avg_subsidy, color='orange', alpha=0.7)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Subsidy ($)')
    ax3.set_title('OTA Subsidy Strategy')
    ax3.grid(True, alpha=0.3)
    
    # 4. 渠道收益占比
    ax4 = axes[1, 1]
    episodes = len(episode_info)
    window = min(50, episodes // 10)
    if window > 0:
        online_ratio = []
        for i in range(window, episodes):
            recent_info = episode_info[i-window:i]
            total_bookings = sum(info['bookings_online'] + info['bookings_offline'] for info in recent_info)
            online_bookings = sum(info['bookings_online'] for info in recent_info)
            ratio = online_bookings / max(1, total_bookings)
            online_ratio.append(ratio * 100)
        
        ax4.plot(range(window, episodes), online_ratio, color='green', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Online Booking Ratio (%)')
        ax4.set_title('Channel Distribution')
        ax4.axhline(y=50, color='r', linestyle='--', alpha=0.3, label='50% baseline')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n博弈结果图已保存到: {save_path}")
    
    plt.show()
