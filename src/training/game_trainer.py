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

def _parse_buckets(spec: Optional[str], n: int) -> List[Tuple[int, int]]:
    if n <= 5:
        return [(i, i) for i in range(max(0, n))]

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
    run_id = f"{timestamp}_{os.getpid()}"
    algorithm_suffix = "cem_nn" if RL_CONFIG.cem_algorithm == 'cem_nn' else "cem"
    log_dir = os.path.join(PATH_CONFIG.tensorboard_dir, f'game_{algorithm_suffix}_cap{ENV_CONFIG.initial_inventory}_{run_id}')
    writer = SummaryWriter(log_dir=log_dir)
    
    print("\n开始训练...")
    algorithm_name = "CEM-NN (神经网络)" if RL_CONFIG.cem_algorithm == 'cem_nn' else "CEM (表格版)"
    print(f"✅ 酒店Agent: 双{algorithm_name}（线上基础价格 + 线下价格）")
    print(f"✅ OTA Agent: {algorithm_name}（补贴决策）")
    print(f"📊 TensorBoard日志: {log_dir}")
    print(f"💡 查看训练曲线: tensorboard --logdir={PATH_CONFIG.tensorboard_dir}")
    
    for episode in range(episodes):
        env.reset()

        total_reward_hotel = 0.0
        total_reward_ota = 0.0
        total_bookings_online = 0
        total_bookings_offline = 0
        total_subsidy = 0.0

        is_last_episode = (episode == episodes - 1)

        bucket_of_offset = [0] * booking_window_days
        for sid, (s, e) in enumerate(buckets):
            for off in range(int(s), min(int(e) + 1, booking_window_days)):
                bucket_of_offset[off] = int(sid)

        trigger_offsets = sorted({int(e) for _, e in buckets if 0 <= int(e) < booking_window_days})

        train_hotel = training_mode in ('simultaneous', 'fixed_ota') or (training_mode == 'alternating' and episode % 2 == 0)
        train_ota = training_mode == 'simultaneous' or (training_mode == 'alternating' and episode % 2 == 1)

        price_online_base_by_offset = [0.0] * booking_window_days
        price_offline_by_offset = [0.0] * booking_window_days
        subsidy_ratio_by_offset = [0.0] * booking_window_days
        decision_state_by_offset = [None] * booking_window_days
        acc_bookings_online_by_offset = [0] * booking_window_days
        acc_bookings_offline_by_offset = [0] * booking_window_days

        for sid, (s, e) in enumerate(buckets):
            ref_off = int(min(int(e), booking_window_days - 1))
            st = dict(env._get_state_for_day_offset(ref_off))
            st['stage_id'] = int(sid)

            pob, pof = hotel_agent.select_action(st, deterministic=False)
            if training_mode == 'fixed_ota':
                gap = pof - pob
                if gap > 20:
                    sr = 0.2
                elif gap > 10:
                    sr = 0.5
                else:
                    sr = 0.7
            else:
                sr = ota_agent.select_action(pob, pof, st, deterministic=False)

            for off in range(int(s), min(int(e) + 1, booking_window_days)):
                price_online_base_by_offset[off] = float(pob)
                price_offline_by_offset[off] = float(pof)
                subsidy_ratio_by_offset[off] = float(sr)
                decision_state_by_offset[off] = dict(st)

        for day in range(365):
            for off in trigger_offsets:
                bo_acc = int(acc_bookings_online_by_offset[off])
                bf_acc = int(acc_bookings_offline_by_offset[off])
                if (bo_acc > 0 or bf_acc > 0) and decision_state_by_offset[off] is not None:
                    pob_prev = float(price_online_base_by_offset[off])
                    pof_prev = float(price_offline_by_offset[off])
                    sr_prev = float(subsidy_ratio_by_offset[off])

                    revenue_hotel_acc = bo_acc * pob_prev * (1 - RL_CONFIG.commission_rate) + bf_acc * pof_prev
                    commission_revenue_acc = bo_acc * pob_prev * RL_CONFIG.commission_rate
                    subsidy_cost_acc = commission_revenue_acc * sr_prev
                    profit_ota_acc = commission_revenue_acc - subsidy_cost_acc

                    total_system_profit_acc = revenue_hotel_acc + profit_ota_acc
                    reward_hotel_acc = RL_CONFIG.reward_hotel_ratio * revenue_hotel_acc + (1 - RL_CONFIG.reward_hotel_ratio) * total_system_profit_acc
                    reward_ota_acc = RL_CONFIG.reward_ota_ratio * profit_ota_acc + (1 - RL_CONFIG.reward_ota_ratio) * total_system_profit_acc

                    state_for_update = dict(decision_state_by_offset[off])
                    next_state_for_update = dict(env._get_state_for_day_offset(off))
                    next_state_for_update['stage_id'] = int(bucket_of_offset[off])

                    if train_hotel:
                        hotel_agent.update(state_for_update, np.array([pob_prev, pof_prev]), reward_hotel_acc, next_state_for_update, done=False, ota_subsidy=subsidy_cost_acc)
                    if train_ota and training_mode != 'fixed_ota':
                        ota_agent.update(pob_prev, pof_prev, state_for_update, sr_prev, reward_ota_acc, next_state_for_update, done=False)

                acc_bookings_online_by_offset[off] = 0
                acc_bookings_offline_by_offset[off] = 0

                sid = int(bucket_of_offset[off])
                st = dict(env._get_state_for_day_offset(off))
                st['stage_id'] = sid

                pob, pof = hotel_agent.select_action(st, deterministic=False)
                if training_mode == 'fixed_ota':
                    gap = pof - pob
                    if gap > 20:
                        sr = 0.2
                    elif gap > 10:
                        sr = 0.5
                    else:
                        sr = 0.7
                else:
                    sr = ota_agent.select_action(pob, pof, st, deterministic=False)

                price_online_base_by_offset[off] = float(pob)
                price_offline_by_offset[off] = float(pof)
                subsidy_ratio_by_offset[off] = float(sr)
                decision_state_by_offset[off] = dict(st)

            price_online_final_window = [
                price_online_base_by_offset[i] - price_online_base_by_offset[i] * RL_CONFIG.commission_rate * subsidy_ratio_by_offset[i]
                for i in range(booking_window_days)
            ]
            subsidy_amount_window = [
                price_online_base_by_offset[i] * RL_CONFIG.commission_rate * subsidy_ratio_by_offset[i]
                for i in range(booking_window_days)
            ]

            actions_window = [[price_online_final_window[i], price_offline_by_offset[i]] for i in range(booking_window_days)]
            _, _, done, info = env.step(actions_window)

            bookings_by_day_offset = info.get('bookings_by_day_offset', [])

            revenue_hotel_day = 0.0
            profit_ota_day = 0.0
            actual_subsidy_amount_day = 0.0

            max_len = min(len(bookings_by_day_offset), booking_window_days)
            for off in range(max_len):
                bo = bookings_by_day_offset[off]['bookings_online']
                bf = bookings_by_day_offset[off]['bookings_offline']
                if bo == 0 and bf == 0:
                    continue

                pob = float(price_online_base_by_offset[off])
                pof = float(price_offline_by_offset[off])
                sr = float(subsidy_ratio_by_offset[off])

                revenue = hotel_agent.calculate_revenue(bo, bf, pob, pof)
                profit = ota_agent.calculate_profit(bo, pob, sr)
                subsidy_cost = bo * pob * RL_CONFIG.commission_rate * sr

                revenue_hotel_day += revenue
                profit_ota_day += profit
                actual_subsidy_amount_day += subsidy_cost
                acc_bookings_online_by_offset[off] += int(bo)
                acc_bookings_offline_by_offset[off] += int(bf)

            total_bookings_online_day = info.get('new_bookings_online', 0)
            total_bookings_offline_day = info.get('new_bookings_offline', 0)

            total_reward_hotel += revenue_hotel_day
            total_reward_ota += profit_ota_day
            total_bookings_online += total_bookings_online_day
            total_bookings_offline += total_bookings_offline_day
            total_subsidy += actual_subsidy_amount_day

            last_subsidy_ratio = subsidy_ratio_by_offset[0] if subsidy_ratio_by_offset else 0.0

            if is_last_episode:
                writer.add_scalar('LastEpisode/Price_Online_Base', price_online_base_by_offset[0], day)
                writer.add_scalar('LastEpisode/Price_Online_Final', price_online_final_window[0], day)
                writer.add_scalar('LastEpisode/Price_Offline', price_offline_by_offset[0], day)
                writer.add_scalar('LastEpisode/Subsidy_Ratio', float(last_subsidy_ratio) * 100, day)
                writer.add_scalar('LastEpisode/Subsidy_Amount', subsidy_amount_window[0], day)
                writer.add_scalar('LastEpisode/Bookings_Online', total_bookings_online_day, day)
                writer.add_scalar('LastEpisode/Bookings_Offline', total_bookings_offline_day, day)
                writer.add_scalar('LastEpisode/Revenue_Hotel', revenue_hotel_day, day)
                writer.add_scalar('LastEpisode/Profit_OTA', profit_ota_day, day)

            if update_frequency > 0 and (day + 1) % update_frequency == 0:
                if train_hotel:
                    hotel_agent.end_episode()
                if train_ota and training_mode != 'fixed_ota':
                    ota_agent.end_episode()

            if done:
                break

            price_online_base_by_offset = price_online_base_by_offset[1:] + [price_online_base_by_offset[-1]]
            price_offline_by_offset = price_offline_by_offset[1:] + [price_offline_by_offset[-1]]
            subsidy_ratio_by_offset = subsidy_ratio_by_offset[1:] + [subsidy_ratio_by_offset[-1]]
            decision_state_by_offset = decision_state_by_offset[1:] + [decision_state_by_offset[-1]]
            acc_bookings_online_by_offset = acc_bookings_online_by_offset[1:] + [acc_bookings_online_by_offset[-1]]
            acc_bookings_offline_by_offset = acc_bookings_offline_by_offset[1:] + [acc_bookings_offline_by_offset[-1]]
        
        for off in range(booking_window_days):
            bo_acc = int(acc_bookings_online_by_offset[off])
            bf_acc = int(acc_bookings_offline_by_offset[off])
            if (bo_acc <= 0 and bf_acc <= 0) or decision_state_by_offset[off] is None:
                continue

            pob_prev = float(price_online_base_by_offset[off])
            pof_prev = float(price_offline_by_offset[off])
            sr_prev = float(subsidy_ratio_by_offset[off])

            revenue_hotel_acc = bo_acc * pob_prev * (1 - RL_CONFIG.commission_rate) + bf_acc * pof_prev
            commission_revenue_acc = bo_acc * pob_prev * RL_CONFIG.commission_rate
            subsidy_cost_acc = commission_revenue_acc * sr_prev
            profit_ota_acc = commission_revenue_acc - subsidy_cost_acc

            total_system_profit_acc = revenue_hotel_acc + profit_ota_acc
            reward_hotel_acc = RL_CONFIG.reward_hotel_ratio * revenue_hotel_acc + (1 - RL_CONFIG.reward_hotel_ratio) * total_system_profit_acc
            reward_ota_acc = RL_CONFIG.reward_ota_ratio * profit_ota_acc + (1 - RL_CONFIG.reward_ota_ratio) * total_system_profit_acc

            state_for_update = dict(decision_state_by_offset[off])
            next_state_for_update = dict(env._get_state_for_day_offset(off))
            next_state_for_update['stage_id'] = int(bucket_of_offset[off])

            if train_hotel:
                hotel_agent.update(state_for_update, np.array([pob_prev, pof_prev]), reward_hotel_acc, next_state_for_update, done=True, ota_subsidy=subsidy_cost_acc)
            if train_ota and training_mode != 'fixed_ota':
                ota_agent.update(pob_prev, pof_prev, state_for_update, sr_prev, reward_ota_acc, next_state_for_update, done=True)

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
    
    #plt.show()
