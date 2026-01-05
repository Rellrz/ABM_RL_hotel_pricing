# 标准库导入
import argparse
import os
import sys
import warnings
from typing import List

# 第三方库导入
import numpy as np
import pandas as pd

# 本地模块导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import train_rl_system_with_abm
from src.utils.random_factor_config import current_random_config
from configs.config import RANDOM_CONFIG
print(f"当前随机因子配置: {current_random_config.current_status}")

# 确保所有随机种子设置与random_factor_config一致
if current_random_config.random_mode == 'fixed':
    # 固定模式：使用配置中的种子
    global_random_seed = RANDOM_CONFIG.fixed_seed
    print(f"使用固定随机种子: {global_random_seed}")
else:
    # 随机模式：使用None作为种子
    global_random_seed = None
    print("使用随机模式，不设置固定种子")


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
    parser.add_argument('--data', type=str, default='datasets/hotel_bookings.csv',
                       help='酒店预订数据文件路径')
    parser.add_argument('--abm-episodes', type=int, default=200,
                       help='ABM模式下的训练轮数（默认200）')
    parser.add_argument('--run-uuid', type=str, default=None,
                       help='运行UUID，用于Q表存储和识别')
    
    args = parser.parse_args()
    
    print("=" * 60)
    algorithm = "Q-learning"
    demand_model = "ABM" 
    print(f"酒店动态定价系统 ({demand_model} + {algorithm})")
    print("=" * 60)
    
    
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

    from src.utils.evaluation import evaluate_trained_policy
    evaluation_results = evaluate_trained_policy(
        rl_system,
    )

    # 分析和可视化Q表
    from src.plot.q_table_plot import analyze_and_visualize_q_table
    analyze_and_visualize_q_table(rl_system, args)

    print("\n" + "=" * 60)
    print("系统运行完成！")
    print("=" * 60)



if __name__ == "__main__":
    # 添加超参数搜索控制逻辑  
    main()
