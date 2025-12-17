#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主训练脚本 - 双智能体酒店动态定价系统
Main Training Script

使用方法:
    python experiments/train.py --episodes 100 --length 365
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.hyperparameters import (
    PATH_CONFIG, TRAINING_CONFIG, validate_config
)
from src.data.preprocessing import HotelDataPreprocessor
from src.environment.hotel_env import HotelEnvironment
from src.agents.hotel_agent import HotelAgent
from src.agents.ota_agent import OTAAgent
from src.training.trainer import DualAgentTrainer
from src.utils.evaluator import PolicyEvaluator

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双智能体酒店动态定价系统训练')
    
    parser.add_argument('--episodes', type=int, default=None,
                        help=f'训练轮数 (默认: {TRAINING_CONFIG.n_episodes})')
    parser.add_argument('--length', type=int, default=None,
                        help=f'每轮天数 (默认: {TRAINING_CONFIG.episode_length})')
    parser.add_argument('--seed', type=int, default=None,
                        help=f'随机种子 (默认: {TRAINING_CONFIG.random_seed})')
    parser.add_argument('--preprocess', action='store_true',
                        help='是否重新运行数据预处理')
    parser.add_argument('--load', action='store_true',
                        help='是否加载已有模型继续训练')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("="*80)
    print(" "*20 + "双智能体酒店动态定价系统")
    print(" "*25 + "训练程序")
    print("="*80)
    
    # 验证配置
    print("\n[1/6] 验证配置...")
    validate_config()
    
    # 设置随机种子
    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
        logger.info(f"设置随机种子: {args.seed}")
    
    # 数据预处理
    print("\n[2/6] 数据预处理...")
    if args.preprocess or not os.path.exists(PATH_CONFIG.preprocessor_path):
        logger.info("运行数据预处理...")
        preprocessor = HotelDataPreprocessor()
        preprocessor.run_preprocessing()
        preprocessor.print_summary()
    else:
        logger.info("加载已有预处理器...")
        preprocessor = HotelDataPreprocessor.load()
        preprocessor.print_summary()
    
    # 创建环境
    print("\n[3/6] 创建环境...")
    env = HotelEnvironment(preprocessor)
    logger.info("环境创建完成")
    
    # 创建智能体
    print("\n[4/6] 创建智能体...")
    if args.load and os.path.exists(PATH_CONFIG.hotel_agent_path):
        logger.info("加载已有智能体...")
        hotel_agent = HotelAgent.load(PATH_CONFIG.hotel_agent_path)
        ota_agent = OTAAgent.load(PATH_CONFIG.ota_agent_path)
    else:
        logger.info("创建新智能体...")
        hotel_agent = HotelAgent()
        ota_agent = OTAAgent()
    
    # 创建训练器
    print("\n[5/6] 创建训练器...")
    trainer = DualAgentTrainer(env, hotel_agent, ota_agent)
    logger.info("训练器创建完成")
    
    # 开始训练
    print("\n[6/6] 开始训练...")
    n_episodes = args.episodes or TRAINING_CONFIG.n_episodes
    episode_length = args.length or TRAINING_CONFIG.episode_length
    
    logger.info(f"训练配置: {n_episodes}轮 × {episode_length}天")
    
    try:
        trainer.train(n_episodes=n_episodes, episode_length=episode_length)
        
        print("\n" + "="*80)
        print(" "*30 + "训练完成!")
        print("="*80)
        print(f"\n模型保存位置: {PATH_CONFIG.models_dir}")
        print(f"结果保存位置: {PATH_CONFIG.results_dir}")
        
        # 训练完成后进行评估
        print("\n" + "="*80)
        print(" "*25 + "开始策略评估")
        print("="*80)
        
        evaluator = PolicyEvaluator(env, hotel_agent, ota_agent)
        eval_summary = evaluator.evaluate(n_episodes=5, episode_length=365)
        evaluator.save_results(eval_summary)
        
        # 生成可视化
        print("\n" + "="*80)
        print(" "*25 + "生成可视化图表")
        print("="*80)
        
        from src.utils.visualization import generate_all_plots
        generate_all_plots(
            hotel_agent=hotel_agent,
            ota_agent=ota_agent,
            training_history=trainer.training_history,
            eval_summary=eval_summary,
            preprocessor=preprocessor
        )
        
    except KeyboardInterrupt:
        logger.warning("\n训练被用户中断")
        print("\n保存当前进度...")
        trainer.save_agents('interrupted')
        trainer.save_training_history()
        print("进度已保存")
    
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        print("\n保存当前进度...")
        trainer.save_agents('error')
        trainer.save_training_history()
        raise


if __name__ == '__main__':
    main()
