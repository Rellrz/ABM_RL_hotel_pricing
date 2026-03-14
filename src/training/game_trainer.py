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
from typing import Optional, Tuple, List, Dict
from datetime import datetime
import os
from tensorboardX import SummaryWriter

from configs.config import RL_CONFIG, ENV_CONFIG, PATH_CONFIG
from src.environment.hotel_env import HotelEnvironment
from src.agent.hotel_agent_dual_channel import HotelAgentDualChannel
from src.agent.ota_agent import OTAAgent


def _default_buckets(n: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    if n <= 5:
        return [(i, i) for i in range(n)]
    edges = [0, 1, 2, 4, 7, 14, 30, 60, n]
    buckets: List[Tuple[int, int]] = []
    for i in range(len(edges) - 1):
        s = edges[i]
        e_excl = min(edges[i + 1], n)
        if s < n and e_excl > s:
            buckets.append((s, e_excl - 1))
    if buckets[0][0] != 0:
        buckets = [(0, buckets[0][0] - 1)] + buckets
    if buckets[-1][1] != n - 1:
        buckets[-1] = (buckets[-1][0], n - 1)
    merged: List[Tuple[int, int]] = []
    for s, e in buckets:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    for i in range(1, len(merged)):
        if merged[i][0] != merged[i - 1][1] + 1:
            raise ValueError("Buckets must be contiguous")
    if merged[0][0] != 0 or merged[-1][1] != n - 1:
        raise ValueError(f"Buckets must cover 0..{n-1}")
    return merged


def _parse_buckets(spec: Optional[str], n: int) -> List[Tuple[int, int]]:
    if n <= 5:
        return [(i, i) for i in range(max(0, n))]

    if spec is None or str(spec).strip() == "":
        return _default_buckets(n)
    tokens = [t.strip() for t in str(spec).replace(',', '|').split('|') if t.strip()]
    buckets: List[Tuple[int, int]] = []
    for t in tokens:
        if '-' in t:
            a, b = t.split('-', 1)
            s, e = int(a), int(b)
        else:
            s = e = int(t)
        buckets.append((s, e))
    buckets.sort(key=lambda x: x[0])
    if not buckets:
        return _default_buckets(n)
    if buckets[0][0] != 0:
        raise ValueError("Buckets must start at 0")
    if buckets[-1][1] != n - 1:
        raise ValueError(f"Buckets must end at {n-1}")
    for i, (s, e) in enumerate(buckets):
        if s < 0 or e < s or e >= n:
            raise ValueError(f"Invalid bucket: {(s, e)}")
        if i > 0 and s != buckets[i - 1][1] + 1:
            raise ValueError("Buckets must be contiguous")
    return buckets


def train_game_system(historical_data: pd.DataFrame, 
                      episodes: int = 100,
                      training_mode: str = 'simultaneous',
                      update_frequency: int = 10,
                      booking_window_days: int = 5,
                      decision_buckets: str = '') -> Tuple[HotelAgentDualChannel, OTAAgent, List, List, List]:
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
    
    buckets = _parse_buckets(decision_buckets, booking_window_days)
    n_stages = len(buckets) if buckets else 1

    env = HotelEnvironment(
        initial_inventory=ENV_CONFIG.initial_inventory,
        historical_data=historical_data,
        booking_window_days=booking_window_days
    )
    
    # 创建酒店Agent（18×K）
    hotel_agent = HotelAgentDualChannel(
        n_states=18 * n_stages,
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
    
    # 创建OTA Agent（90×K）
    ota_agent = OTAAgent(
        commission_rate=RL_CONFIG.commission_rate,
        subsidy_ratio_min=RL_CONFIG.subsidy_ratio_min,
        subsidy_ratio_max=RL_CONFIG.subsidy_ratio_max,
        n_states=90 * n_stages,
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
    
    # 初始化TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_suffix = "cem_nn" if RL_CONFIG.cem_algorithm == 'cem_nn' else "cem"
    log_dir = os.path.join(PATH_CONFIG.tensorboard_dir, f'game_{algorithm_suffix}_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    print("\n开始训练...")
    algorithm_name = "CEM-NN (神经网络)" if RL_CONFIG.cem_algorithm == 'cem_nn' else "CEM (表格版)"
    print(f"✅ 酒店Agent: 双{algorithm_name}（线上基础价格 + 线下价格）")
    print(f"✅ OTA Agent: {algorithm_name}（补贴决策）")
    print(f"📊 TensorBoard日志: {log_dir}")
    print(f"💡 查看训练曲线: tensorboard --logdir={PATH_CONFIG.tensorboard_dir}")
    
    for episode in range(episodes):
        state = env.reset()
        
        total_reward_hotel = 0.0
        total_reward_ota = 0.0
        total_bookings_online = 0
        total_bookings_offline = 0
        total_subsidy = 0.0
        
        # 是否是最后一个episode（用于记录详细数据）
        is_last_episode = (episode == episodes - 1)
        
        train_hotel = training_mode in ('simultaneous', 'fixed_ota') or (training_mode == 'alternating' and episode % 2 == 0)
        train_ota = training_mode == 'simultaneous' or (training_mode == 'alternating' and episode % 2 == 1)

        for day in range(365):
            price_online_base_window = []
            price_offline_window = []
            subsidy_ratio_window = []
            bucket_decisions = []

            for stage_id, (start, end) in enumerate(buckets):
                state_bucket = dict(env._get_state_for_day_offset(start))
                state_bucket['stage_id'] = int(stage_id)
                price_online_base, price_offline = hotel_agent.select_action(state_bucket, deterministic=False)

                if training_mode == 'fixed_ota':
                    price_gap = price_offline - price_online_base
                    if price_gap > 20:
                        subsidy_ratio = 0.2
                    elif price_gap > 10:
                        subsidy_ratio = 0.5
                    else:
                        subsidy_ratio = 0.7
                else:
                    subsidy_ratio = ota_agent.select_action(price_online_base, price_offline, state_bucket, deterministic=False)

                span = end - start + 1
                price_online_base_window.extend([float(price_online_base)] * span)
                price_offline_window.extend([float(price_offline)] * span)
                subsidy_ratio_window.extend([float(subsidy_ratio)] * span)
                bucket_decisions.append((int(stage_id), int(start), state_bucket, float(price_online_base), float(price_offline), float(subsidy_ratio), span))

            price_online_base_window = price_online_base_window[:booking_window_days]
            price_offline_window = price_offline_window[:booking_window_days]
            subsidy_ratio_window = subsidy_ratio_window[:booking_window_days]

            price_online_final_window = []
            subsidy_amount_window = []
            for i in range(booking_window_days):
                subsidy_amount = price_online_base_window[i] * RL_CONFIG.commission_rate * subsidy_ratio_window[i]
                price_online_final_window.append(price_online_base_window[i] - subsidy_amount)
                subsidy_amount_window.append(subsidy_amount)

            actions_window = [[price_online_final_window[i], price_offline_window[i]] for i in range(booking_window_days)]
            next_state, reward, done, info = env.step(actions_window)

            bookings_by_day_offset = info.get('bookings_by_day_offset', [])

            offset_idx = 0
            for stage_id, start, state_bucket, price_online_base, price_offline, subsidy_ratio, span in bucket_decisions:
                if offset_idx >= len(bookings_by_day_offset):
                    break

                end_idx = min(offset_idx + span, len(bookings_by_day_offset))
                bookings_online = sum(bookings_by_day_offset[j]['bookings_online'] for j in range(offset_idx, end_idx))
                bookings_offline = sum(bookings_by_day_offset[j]['bookings_offline'] for j in range(offset_idx, end_idx))
                offset_idx = end_idx

                if bookings_online == 0 and bookings_offline == 0:
                    continue

                next_state_bucket = dict(env._get_state_for_day_offset(start))
                next_state_bucket['stage_id'] = int(stage_id)

                revenue_hotel = hotel_agent.calculate_revenue(bookings_online, bookings_offline, price_online_base, price_offline)
                profit_ota = ota_agent.calculate_profit(bookings_online, price_online_base, subsidy_ratio)
                actual_subsidy_amount = bookings_online * price_online_base * RL_CONFIG.commission_rate * subsidy_ratio

                total_system_profit = revenue_hotel + profit_ota
                reward_hotel = RL_CONFIG.reward_hotel_ratio * revenue_hotel + (1 - RL_CONFIG.reward_hotel_ratio) * total_system_profit
                reward_ota = RL_CONFIG.reward_ota_ratio * profit_ota + (1 - RL_CONFIG.reward_ota_ratio) * total_system_profit

                if train_hotel:
                    hotel_agent.update(state_bucket, np.array([price_online_base, price_offline]), reward_hotel, next_state_bucket, done, actual_subsidy_amount)
                if train_ota and training_mode != 'fixed_ota':
                    ota_agent.update(price_online_base, price_offline, state_bucket, subsidy_ratio, reward_ota, next_state_bucket, done)

            revenue_hotel_day = 0.0
            profit_ota_day = 0.0
            actual_subsidy_amount_day = 0.0

            max_len = min(len(bookings_by_day_offset), booking_window_days)
            for i in range(max_len):
                bo = bookings_by_day_offset[i]['bookings_online']
                bf = bookings_by_day_offset[i]['bookings_offline']
                if bo == 0 and bf == 0:
                    continue
                pob = price_online_base_window[i]
                pof = price_offline_window[i]
                sr = subsidy_ratio_window[i]
                revenue_hotel_day += hotel_agent.calculate_revenue(bo, bf, pob, pof)
                profit_ota_day += ota_agent.calculate_profit(bo, pob, sr)
                actual_subsidy_amount_day += bo * pob * RL_CONFIG.commission_rate * sr

            total_bookings_online_day = info.get('new_bookings_online', 0)
            total_bookings_offline_day = info.get('new_bookings_offline', 0)

            total_reward_hotel += revenue_hotel_day
            total_reward_ota += profit_ota_day
            total_bookings_online += total_bookings_online_day
            total_bookings_offline += total_bookings_offline_day
            total_subsidy += actual_subsidy_amount_day

            last_subsidy_ratio = subsidy_ratio_window[0] if subsidy_ratio_window else 0.0

            if is_last_episode:
                # 记录价格数据
                writer.add_scalar('LastEpisode/Price_Online_Base', price_online_base_window[0], day)
                writer.add_scalar('LastEpisode/Price_Online_Final', price_online_final_window[0], day)
                writer.add_scalar('LastEpisode/Price_Offline', price_offline_window[0], day)
                writer.add_scalar('LastEpisode/Subsidy_Ratio', last_subsidy_ratio * 100, day)
                writer.add_scalar('LastEpisode/Subsidy_Amount', subsidy_amount_window[0], day)
                # 记录预订数据
                writer.add_scalar('LastEpisode/Bookings_Online', total_bookings_online_day, day)
                writer.add_scalar('LastEpisode/Bookings_Offline', total_bookings_offline_day, day)
                # 记录收益数据
                writer.add_scalar('LastEpisode/Revenue_Hotel', revenue_hotel_day, day)
                writer.add_scalar('LastEpisode/Profit_OTA', profit_ota_day, day)

            if update_frequency > 0 and (day + 1) % update_frequency == 0:
                if train_hotel:
                    hotel_agent.end_episode()
                if train_ota and training_mode != 'fixed_ota':
                    ota_agent.end_episode()

            state = next_state
            if done:
                break
        
        if update_frequency <= 0 or 365 % update_frequency != 0:
            if train_hotel:
                hotel_agent.end_episode()
            if train_ota and training_mode != 'fixed_ota':
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
            'avg_subsidy_ratio': last_subsidy_ratio  # 最后一天的补贴比例
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
        
        # TensorBoard记录
        writer.add_scalar('Reward/Hotel_Revenue', total_reward_hotel, episode)
        writer.add_scalar('Reward/OTA_Profit', total_reward_ota, episode)
        writer.add_scalar('Reward/Total_Revenue', total_reward_hotel + total_reward_ota, episode)
        writer.add_scalar('Bookings/Online', total_bookings_online, episode)
        writer.add_scalar('Bookings/Offline', total_bookings_offline, episode)
        writer.add_scalar('Bookings/Total', total_bookings_online + total_bookings_offline, episode)
        writer.add_scalar('Bookings/Online_Ratio', total_bookings_online / max(1, total_bookings_online + total_bookings_offline), episode)
        writer.add_scalar('Subsidy/Total_Amount', total_subsidy, episode)
        writer.add_scalar('Subsidy/Avg_Amount_Per_Booking', total_subsidy / max(1, total_bookings_online), episode)
        writer.add_scalar('Subsidy/Avg_Ratio', last_subsidy_ratio, episode)
        writer.add_scalar('Training/Exploration_Rate', exploration_rate, episode)
        
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
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"\n📊 TensorBoard日志已保存: {log_dir}")
    print(f"💡 查看训练曲线: tensorboard --logdir={PATH_CONFIG.tensorboard_dir}")
    
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
    
    # 3. 平均补贴金额和比例
    ax3 = axes[1, 0]
    avg_subsidy_amount = [info['avg_subsidy_amount'] for info in episode_info]
    avg_subsidy_ratio = [info['avg_subsidy_ratio'] * 100 for info in episode_info]  # 转换为百分比
    
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(avg_subsidy_amount, color='orange', alpha=0.7, label='Subsidy Amount ($)')
    line2 = ax3_twin.plot(avg_subsidy_ratio, color='purple', alpha=0.7, label='Subsidy Ratio (%)')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Subsidy Amount ($)', color='orange')
    ax3_twin.set_ylabel('Subsidy Ratio (%)', color='purple')
    ax3.set_title('OTA Subsidy Strategy')
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3_twin.tick_params(axis='y', labelcolor='purple')
    ax3.grid(True, alpha=0.3)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
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
