# 算法切换使用指南

## 概述

本系统支持两种强化学习算法：
1. **Q-learning**：离散动作空间（36个定价组合）
2. **Actor-Critic**：连续动作空间（价格范围80-200元）

## 快速开始

### 1. 切换到Q-learning算法

在 `configs/config.py` 中设置：

```python
@dataclass
class RLConfig:
    algorithm: str = 'q_learning'  # 使用Q-learning
    # ... 其他参数
```

### 2. 切换到Actor-Critic算法

在 `configs/config.py` 中设置：

```python
@dataclass
class RLConfig:
    algorithm: str = 'actor_critic'  # 使用Actor-Critic
    # ... 其他参数
```

## 算法对比

| 特性 | Q-learning | Actor-Critic |
|------|-----------|--------------|
| **动作空间** | 离散（36个组合） | 连续（80-200元） |
| **价格精度** | 固定档位 | 任意价格 |
| **探索方式** | ε-greedy + UCB | 高斯噪声 |
| **收敛速度** | 较快 | 较慢 |
| **样本效率** | 高 | 中等 |
| **最优性** | 保证收敛 | 可能局部最优 |
| **适用场景** | 简单环境 | 复杂环境 |

## 参数配置详解

### Q-learning参数

```python
# Q-learning专用参数
learning_rate: float = 0.05          # 学习率
epsilon_start: float = 0.9           # 初始探索率
epsilon_end: float = 0.01            # 最终探索率
epsilon_decay_episodes: int = 100    # 探索率衰减轮数
```

**调优建议**：
- `learning_rate`：0.01-0.1，值越大学习越快但可能不稳定
- `epsilon_start`：0.8-1.0，初期需要充分探索
- `epsilon_decay_episodes`：根据总训练轮数调整，建议为总轮数的1/3

### Actor-Critic参数

```python
# Actor-Critic专用参数
actor_lr: float = 0.01               # Actor学习率（策略更新）
critic_lr: float = 0.05              # Critic学习率（价值更新）
action_min: float = 80.0             # 最低价格
action_max: float = 200.0            # 最高价格
initial_std: float = 20.0            # 初始探索标准差
min_std: float = 5.0                 # 最小探索标准差
std_decay: float = 0.995             # 标准差衰减率
```

**调优建议**：
- `actor_lr`：0.001-0.01，策略更新要谨慎，建议比critic_lr小
- `critic_lr`：0.01-0.1，价值估计可以更新快一些
- `initial_std`：价格范围的10-20%，控制初期探索幅度
- `min_std`：价格范围的2-5%，保持最小探索
- `std_decay`：0.99-0.999，每个episode衰减0.1-1%

## 使用示例

### 示例1：使用Q-learning训练

```python
# 1. 修改 configs/config.py
@dataclass
class RLConfig:
    algorithm: str = 'q_learning'
    learning_rate: float = 0.05
    epsilon_start: float = 0.9
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 100
    episodes: int = 200

# 2. 运行训练
python experiments/train.py --abm-episodes 200
```

**预期输出**：
```
✅ 使用 Q-learning 算法（离散动作空间）
Episode 10/200: Avg Reward=45000.00, ε=0.850
Episode 20/200: Avg Reward=48000.00, ε=0.720
...
```

### 示例2：使用Actor-Critic训练

```python
# 1. 修改 configs/config.py
@dataclass
class RLConfig:
    algorithm: str = 'actor_critic'
    actor_lr: float = 0.01
    critic_lr: float = 0.05
    initial_std: float = 20.0
    min_std: float = 5.0
    std_decay: float = 0.995
    episodes: int = 200

# 2. 运行训练
python experiments/train.py --abm-episodes 200
```

**预期输出**：
```
✅ 使用 Actor-Critic 算法（连续动作空间）
Episode 10/200: Avg Reward=43000.00, ε=0.670
Episode 20/200: Avg Reward=47000.00, ε=0.450
...
```

## 性能调优指南

### Q-learning调优

**问题1：收敛太慢**
- ✅ 增加 `learning_rate` (0.05 → 0.1)
- ✅ 减少 `epsilon_decay_episodes` (100 → 50)
- ✅ 增加训练轮数

**问题2：性能不稳定**
- ✅ 降低 `learning_rate` (0.1 → 0.05)
- ✅ 增加 `epsilon_decay_episodes` (50 → 100)
- ✅ 使用更多训练轮数

**问题3：探索不充分**
- ✅ 增加 `epsilon_start` (0.8 → 0.95)
- ✅ 增加 `epsilon_decay_episodes` (100 → 150)
- ✅ 提高 `epsilon_end` (0.01 → 0.05)

### Actor-Critic调优

**问题1：策略不收敛**
- ✅ 降低 `actor_lr` (0.01 → 0.005)
- ✅ 增加 `critic_lr` (0.05 → 0.1)
- ✅ 减慢 `std_decay` (0.995 → 0.998)

**问题2：探索不足**
- ✅ 增加 `initial_std` (20 → 30)
- ✅ 增加 `min_std` (5 → 10)
- ✅ 减慢 `std_decay` (0.995 → 0.998)

**问题3：价格波动太大**
- ✅ 降低 `initial_std` (20 → 15)
- ✅ 降低 `min_std` (5 → 3)
- ✅ 加快 `std_decay` (0.995 → 0.99)

## 算法选择建议

### 选择Q-learning的场景

✅ **适合使用Q-learning**：
- 动作空间较小（<100个动作）
- 需要快速收敛
- 对价格精度要求不高
- 计算资源有限
- 需要理论保证（收敛到最优）

### 选择Actor-Critic的场景

✅ **适合使用Actor-Critic**：
- 需要精细的价格控制
- 动作空间很大或连续
- 有充足的训练时间
- 环境复杂度高
- 需要平滑的策略变化

## 监控指标

### Q-learning监控

```python
# 关键指标
- exploration_coverage: 探索覆盖率（目标>90%）
- zero_q_percentage: 零值Q值占比（目标<10%）
- mean_q_value: 平均Q值（应逐渐增加）
- epsilon: 探索率（应逐渐衰减）
```

### Actor-Critic监控

```python
# 关键指标
- current_std: 当前标准差（应逐渐衰减）
- policy_mean_avg: 策略均值（应趋于稳定）
- value_avg: 平均价值（应逐渐增加）
- policy_mean_std: 策略标准差（反映策略多样性）
```

## 常见问题

### Q1: 如何验证算法切换成功？

查看训练开始时的输出：
```
✅ 使用 Q-learning 算法（离散动作空间）
# 或
✅ 使用 Actor-Critic 算法（连续动作空间）
```

### Q2: 两种算法可以同时使用吗？

不可以。每次训练只能使用一种算法。如需对比，请分别训练并保存结果。

### Q3: 切换算法后需要重新训练吗？

是的。两种算法的模型结构不同，无法直接迁移。

### Q4: Actor-Critic的价格范围如何设置？

根据业务需求设置 `action_min` 和 `action_max`。建议：
- 最低价格：成本价 × 1.2
- 最高价格：市场最高价或成本价 × 3

### Q5: 如何保存和加载不同算法的模型？

```python
# 保存时会自动包含算法类型
agent.save_agent('model_q_learning.pkl')
agent.save_agent('model_actor_critic.pkl')

# 加载时需确保配置文件中的algorithm参数匹配
```

## 实验建议

### 对比实验设计

1. **相同环境下对比**：
   ```python
   # 实验1：Q-learning
   RL_CONFIG.algorithm = 'q_learning'
   RL_CONFIG.episodes = 200
   
   # 实验2：Actor-Critic
   RL_CONFIG.algorithm = 'actor_critic'
   RL_CONFIG.episodes = 200
   ```

2. **记录关键指标**：
   - 最终平均收益
   - 收敛速度（达到稳定的轮数）
   - 训练时间
   - 价格分布

3. **可视化对比**：
   - 奖励曲线
   - 探索率/标准差曲线
   - 价格分布直方图

## 技术细节

### Q-learning实现

- 算法文件：`src/algorithms/q_learning.py`
- 核心类：`QLearning`
- 更新规则：Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

### Actor-Critic实现

- 算法文件：`src/algorithms/actor_critic.py`
- 核心类：`TabularActorCritic`
- Actor更新：μ(s) ← μ(s) + α_actor·δ·∇log π(a|s)
- Critic更新：V(s) ← V(s) + α_critic·δ
- 策略：a ~ N(μ(s), σ²)

## 更新日志

- **2024-12-24**：初始版本，支持Q-learning和Actor-Critic切换
- 未来计划：支持更多算法（PPO、SAC等）
