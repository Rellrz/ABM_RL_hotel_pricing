# 双智能体酒店动态定价系统

基于Agent-Based Modeling (ABM) 和双智能体强化学习的酒店动态定价系统。

## 🎯 项目概述

本系统模拟酒店和OTA（在线旅行社）之间的定价博弈，通过Q-learning算法学习最优定价策略。

### 核心特性

- **双智能体博弈**：酒店智能体 vs OTA智能体
- **ABM客户模型**：Type A (传统/忠诚型) 和 Type B (数字/价格敏感型)
- **佣金驱动流量**：4档佣金机制影响OTA流量
- **180天库存管理**：长期库存规划
- **15天决策窗口**：近期精细化定价
- **远期规则定价**：提前15-180天的固定价格策略

## 📁 项目结构

```
ABM_hotel_pricing/
├── configs/                    # 配置文件
│   └── hyperparameters.py     # 统一超参数管理
├── src/                       # 源代码
│   ├── data/                  # 数据处理
│   │   └── preprocessing.py   # 数据预处理（P_base/P_long/λ_base）
│   ├── agents/                # 智能体
│   │   ├── customer_agent.py  # ABM客户模型
│   │   ├── hotel_agent.py     # 酒店Q-learning智能体
│   │   └── ota_agent.py       # OTA Q-learning智能体
│   ├── environment/           # 环境
│   │   └── hotel_env.py       # 酒店环境（库存、定价、仿真）
│   ├── training/              # 训练
│   │   └── trainer.py         # 双智能体训练器
│   └── utils/                 # 工具
│       └── logger.py          # 日志工具
├── experiments/               # 实验脚本
│   └── train.py              # 主训练脚本
├── data/                     # 数据文件
│   ├── raw/                  # 原始数据
│   └── processed/            # 处理后数据
├── outputs/                  # 输出
│   ├── models/              # 训练模型
│   ├── results/             # 结果
│   └── figures/             # 图表
└── notebooks/               # Jupyter notebooks
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
conda create -n hotel_pricing python=3.8
conda activate hotel_pricing

# 安装依赖
pip install -r requirements_abm.txt
```

### 2. 数据准备

将 `hotel_bookings.csv` 放入 `data/raw/` 目录。

### 3. 运行训练

```bash
# 基础训练（100轮 × 365天）
python experiments/train.py

# 自定义参数
python experiments/train.py --episodes 1000 --length 365

# 重新预处理数据
python experiments/train.py --preprocess

# 加载已有模型继续训练
python experiments/train.py --load
```

## 📊 系统架构

### 状态空间

**酒店智能体** (1080个状态):
- 提前期 (0-14天): 15档
- 库存压力 (空闲/正常/紧张/告急): 4档
- OTA竞争力 (劣势/均势/优势): 3档
- 周末标识: 2档
- 季节 (淡季/平季/旺季): 3档

**OTA智能体** (1080个状态):
- 提前期 (0-14天): 15档
- 库存压力: 4档
- 套利空间 (微利/正常/暴利): 3档
- 周末标识: 2档
- 季节: 3档

### 动作空间

**酒店智能体** (20个动作):
- 直销折扣: [0.80, 0.85, 0.90, 0.95, 1.00]
- 佣金档位: [10%, 15%, 20%, 25%]

**OTA智能体** (5个动作):
- 补贴系数: [0.0, 0.2, 0.4, 0.6, 0.8]

### 价格体系

- **P_base**: 基准价格（70%分位数，按月份+周末分段）
- **P_long**: 远期价格（中位数，用于L∈[15,180]天）
- **P_dir**: 直销价格（P_base × 折扣系数）
- **P_ota**: OTA价格（P_base - 补贴）

### 客户行为

**Type A (20%)**: 传统/忠诚型
- 只关注直销渠道
- 价格敏感度中等
- WTP较高

**Type B (80%)**: 数字/价格敏感型
- 比较直销和OTA价格
- 价格敏感度高
- 默认偏好OTA

**决策逻辑**:
- 近期 (L∈[0,14]): 看Agent决策价格
- 远期 (L∈[15,180]): 看规则价格P_long
- 效用函数: `U = (WTP - P) × β + γ/(L+1) + ε`

## ⚙️ 配置管理

所有超参数在 `configs/hyperparameters.py` 中统一管理：

```python
from configs.hyperparameters import (
    ABM_CONFIG,          # ABM参数
    HOTEL_AGENT_CONFIG,  # 酒店智能体参数
    OTA_AGENT_CONFIG,    # OTA智能体参数
    ENV_CONFIG,          # 环境配置
    TRAINING_CONFIG,     # 训练配置
    DATA_CONFIG          # 数据配置
)
```

## 📈 训练流程

1. **数据预处理**: 计算P_base, P_long, λ_base等参数
2. **环境初始化**: 创建180天库存窗口
3. **双智能体训练**:
   - 酒店先决策（折扣+佣金）
   - OTA观察后决策（补贴）
   - ABM模拟客户行为
   - **多天Q表更新**（更新所有15天的状态-动作对）
4. **定期评估**: 使用贪婪策略评估性能
5. **训练后评估**: 运行5轮×365天的完整评估
6. **可视化生成**: 自动生成所有分析图表
7. **保存模型**: 保存Q表、训练历史和评估结果

## 📝 输出结果

### 模型文件
- `outputs/models/hotel_agent_final.pkl` - 酒店智能体Q表
- `outputs/models/ota_agent_final.pkl` - OTA智能体Q表
- `outputs/models/preprocessor.pkl` - 数据预处理器

### 训练结果
- `outputs/results/training_history.json` - 训练历史
- `outputs/results/hotel_q_table_*.csv` - 酒店Q表详细数据
- `outputs/results/ota_q_table_*.csv` - OTA Q表详细数据
- `outputs/results/evaluation_results_*.csv` - 评估结果详细数据

### 可视化图表
- `outputs/figures/training_curves_*.png` - 训练曲线（6个子图）
  - 酒店收益、OTA收益
  - 直销订单、OTA订单
  - 平均入住率、探索率衰减
  
- `outputs/figures/q_table_analysis_*.png` - Q表分析（4个子图）
  - 酒店Q值分布、OTA Q值分布
  - 酒店状态访问分布、OTA状态访问分布
  
- `outputs/figures/evaluation_results_*.png` - 评估结果（6个子图）
  - 酒店每日收益、OTA每日收益
  - 每日订单对比、未来5天库存变化
  - 平均入住率、渠道收益占比
  
- `outputs/figures/policy_heatmaps_*.png` - 策略热图（2个子图）
  - 酒店最优折扣策略（按库存和提前期）
  - OTA最优补贴策略（按库存和提前期）
  
- `outputs/figures/price_distribution_*.png` - 价格分析（2个子图）
  - 基准价格vs远期价格分布
  - 月度价格变化（工作日/周末）

## 🔧 自定义开发

### 修改超参数

编辑 `configs/hyperparameters.py`:

```python
@dataclass
class ABMConfig:
    urgency_weight: float = 7.5  # 修改紧迫权重
    noise_std: float = 12.0      # 修改决策噪声
    # ...
```

### 扩展智能体

继承基类并重写方法:

```python
from src.agents.hotel_agent import HotelAgent

class CustomHotelAgent(HotelAgent):
    def choose_action(self, state, training=True):
        # 自定义动作选择逻辑
        pass
```

## 📚 技术文档

详细技术文档请参考：`RL_ABM酒店定价技术文档.md`

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👥 作者

Hotel Pricing Research Team

---

**注意**: 首次运行需要安装所有依赖并预处理数据，可能需要几分钟时间。
