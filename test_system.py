#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统测试脚本
快速验证所有模块是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """测试配置模块"""
    print("\n" + "="*60)
    print("测试1: 配置模块")
    print("="*60)
    
    from configs.hyperparameters import validate_config
    validate_config()
    print("✓ 配置验证通过")

def test_preprocessing():
    """测试数据预处理"""
    print("\n" + "="*60)
    print("测试2: 数据预处理")
    print("="*60)
    
    from src.data.preprocessing import HotelDataPreprocessor
    
    # 检查是否已有预处理器
    from configs.hyperparameters import PATH_CONFIG
    if os.path.exists(PATH_CONFIG.preprocessor_path):
        print("加载已有预处理器...")
        preprocessor = HotelDataPreprocessor.load()
    else:
        print("运行数据预处理...")
        preprocessor = HotelDataPreprocessor()
        preprocessor.run_preprocessing()
    
    print(f"✓ 价格表: {len(preprocessor.price_tables['p_base'])} 个价格段")
    print(f"✓ 到达率: {len(preprocessor.arrival_rates)} 个月份")
    print("✓ 数据预处理通过")
    
    return preprocessor

def test_agents():
    """测试智能体"""
    print("\n" + "="*60)
    print("测试3: 智能体")
    print("="*60)
    
    from src.agents.hotel_agent import HotelAgent
    from src.agents.ota_agent import OTAAgent
    
    hotel_agent = HotelAgent()
    ota_agent = OTAAgent()
    
    # 测试状态离散化
    hotel_state = hotel_agent.discretize_state(5, 0.5, 1.0, True, 1)
    ota_state = ota_agent.discretize_state(3, 0.7, 1.25, False, 2)
    
    # 测试动作选择
    hotel_action = hotel_agent.choose_action(hotel_state, training=False)
    ota_action = ota_agent.choose_action(ota_state, training=False)
    
    print(f"✓ 酒店智能体: 状态={hotel_state}, 动作={hotel_action}")
    print(f"✓ OTA智能体: 状态={ota_state}, 动作={ota_action}")
    print("✓ 智能体测试通过")
    
    return hotel_agent, ota_agent

def test_environment(preprocessor):
    """测试环境"""
    print("\n" + "="*60)
    print("测试4: 环境")
    print("="*60)
    
    from src.environment.hotel_env import HotelEnvironment
    
    env = HotelEnvironment(preprocessor)
    state_info = env.reset()
    
    print(f"✓ 库存窗口: {len(env.inventory)} 天")
    print(f"✓ 酒店状态数: {len(state_info['states_hotel'])}")
    print(f"✓ OTA状态数: {len(state_info['states_ota'])}")
    print("✓ 环境测试通过")
    
    return env

def test_training(env, hotel_agent, ota_agent):
    """测试训练"""
    print("\n" + "="*60)
    print("测试5: 训练流程")
    print("="*60)
    
    from src.training.trainer import DualAgentTrainer
    
    trainer = DualAgentTrainer(env, hotel_agent, ota_agent)
    
    print("运行微型训练 (2轮 × 5天)...")
    trainer.train(n_episodes=2, episode_length=5)
    
    print("✓ 训练流程测试通过")
    
    return trainer

def test_evaluation(env, hotel_agent, ota_agent):
    """测试评估"""
    print("\n" + "="*60)
    print("测试6: 评估模块")
    print("="*60)
    
    from src.utils.evaluator import PolicyEvaluator
    
    evaluator = PolicyEvaluator(env, hotel_agent, ota_agent)
    eval_summary = evaluator.evaluate(n_episodes=1, episode_length=5)
    
    print(f"✓ 酒店平均收益: ${eval_summary['revenue_hotel_mean']:.2f}")
    print(f"✓ OTA平均收益: ${eval_summary['revenue_ota_mean']:.2f}")
    print("✓ 评估模块测试通过")
    
    return eval_summary

def test_visualization():
    """测试可视化"""
    print("\n" + "="*60)
    print("测试7: 可视化模块")
    print("="*60)
    
    from src.utils.visualization import (
        plot_training_curves,
        analyze_q_tables,
        plot_evaluation_results,
        plot_policy_heatmaps,
        plot_price_distribution
    )
    
    print("✓ 所有可视化函数导入成功")
    print("✓ 可视化模块测试通过")

def main():
    """主测试流程"""
    print("\n" + "="*80)
    print(" "*25 + "系统测试开始")
    print("="*80)
    
    try:
        # 1. 测试配置
        test_config()
        
        # 2. 测试数据预处理
        preprocessor = test_preprocessing()
        
        # 3. 测试智能体
        hotel_agent, ota_agent = test_agents()
        
        # 4. 测试环境
        env = test_environment(preprocessor)
        
        # 5. 测试训练
        trainer = test_training(env, hotel_agent, ota_agent)
        
        # 6. 测试评估
        eval_summary = test_evaluation(env, hotel_agent, ota_agent)
        
        # 7. 测试可视化
        test_visualization()
        
        # 总结
        print("\n" + "="*80)
        print(" "*25 + "所有测试通过! ✓")
        print("="*80)
        print("\n系统已就绪，可以运行完整训练:")
        print("  python experiments/train.py --episodes 10 --length 30")
        print("\n或运行完整训练:")
        print("  python experiments/train.py --episodes 100 --length 365")
        
    except Exception as e:
        print("\n" + "="*80)
        print(" "*25 + "测试失败! ✗")
        print("="*80)
        print(f"\n错误信息: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
