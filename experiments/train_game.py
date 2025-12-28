"""
酒店-OTA博弈系统训练脚本

训练酒店和OTA的双层博弈系统
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import RL_CONFIG, PATH_CONFIG
import pandas as pd
from src.training.game_trainer import train_game_system, plot_game_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练酒店-OTA博弈系统')
    parser.add_argument('--data', type=str, default='datasets/hotel_bookings.csv',
                       help='酒店预订数据文件路径')
    parser.add_argument('--episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--mode', type=str, default='simultaneous', 
                       choices=['fixed_ota', 'alternating', 'simultaneous'],
                       help='训练模式')
    parser.add_argument('--commission', type=float, default=0.15, help='OTA佣金率')
    parser.add_argument('--subsidy-ratio-max', type=float, default=0.8, help='最大补贴比例（占佣金的百分比）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("酒店-OTA博弈系统训练")
    print("=" * 60)
    print(f"训练轮数: {args.episodes}")
    print(f"训练模式: {args.mode}")
    print(f"佣金率: {args.commission * 100:.1f}%")
    print(f"补贴比例范围: 0%-{args.subsidy_ratio_max * 100:.0f}%")
    
    # 更新配置
    RL_CONFIG.commission_rate = args.commission
    RL_CONFIG.subsidy_ratio_max = args.subsidy_ratio_max
    
    # 加载数据
    print("\n=== 加载历史数据 ===")
    historical_data = pd.read_csv(args.data)
    historical_data = historical_data[historical_data['hotel'] == 'City Hotel'].copy()
    print(f"数据加载完成，共 {len(historical_data)} 条记录") 
    
    # 训练博弈系统
    hotel_agent, ota_agent, rewards_hotel, rewards_ota, episode_info = train_game_system(
        historical_data=historical_data,
        episodes=args.episodes,
        training_mode=args.mode
    )
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存Agent
    output_dir = os.path.join(PROJECT_ROOT, 'outputs', 'game_models')
    os.makedirs(output_dir, exist_ok=True)
    
    hotel_path = os.path.join(output_dir, f'hotel_agent_{timestamp}.pkl')
    ota_path = os.path.join(output_dir, f'ota_agent_{timestamp}.pkl')
    
    # 注意：需要实现save方法
    print(f"\n模型保存路径:")
    print(f"  酒店Agent: {hotel_path}")
    print(f"  OTA Agent: {ota_path}")
    
    # 绘制结果
    figure_dir = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    figure_path = os.path.join(figure_dir, f'game_results_{timestamp}.png')
    
    plot_game_results(rewards_hotel, rewards_ota, episode_info, figure_path)
    
    # 保存训练数据
    df = pd.DataFrame(episode_info)
    data_path = os.path.join(output_dir, f'training_data_{timestamp}.csv')
    df.to_csv(data_path, index=False)
    print(f"\n训练数据已保存到: {data_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
