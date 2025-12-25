# 连续动作算法使用指南

本文档介绍如何使用新增的两种连续动作算法：**Linear SARSA** 和 **CEM**。

## 📋 算法概览

| 算法 | 类型 | 稳定性 | 收敛速度 | 适用场景 |
|------|------|--------|----------|----------|
| **Q-learning** | 离散动作 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 稳定训练，离散价格 |
| **Actor-Critic** | 连续动作 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 快速收敛，但不稳定 |
| **Linear SARSA** | 连续动作 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 稳定训练，连续价格 |
| **CEM** | 连续动作 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 最稳定，随机环境 |

---

## 1️⃣ Linear SARSA（线性函数逼近SARSA）

### **算法原理**

使用线性函数逼近Q值，支持连续动作空间：

```
Q(s, a) = w^T φ(s, a)
```

其中：
- `w`: 权重向量
- `φ(s, a)`: 特征向量（状态和动作的组合）

**更新规则（SARSA）**：
```
w ← w + α * δ * φ(s, a)
δ = r + γQ(s', a') - Q(s, a)
```

### **特点**

✅ **优势**：
- 比Actor-Critic更稳定（使用SARSA而非策略梯度）
- 真正的连续动作输出
- 通过网格搜索找到最优动作
- 适合on-policy学习

❌ **劣势**：
- 需要特征工程
- 收敛速度较慢
- 网格搜索有计算开销

### **使用方法**

#### **1. 修改配置文件**

编辑 `configs/config.py`:

```python
@dataclass
class RLConfig:
    # 算法选择
    algorithm: str = 'linear_sarsa'  # 切换到Linear SARSA
    
    # 通用参数
    n_states: int = 18
    discount_factor: float = 0.99
    
    # Linear SARSA参数
    learning_rate: float = 0.01  # 学习率（建议0.01-0.05）
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300
    
    # 连续动作范围
    action_min: float = 80.0
    action_max: float = 170.0
    
    episodes: int = 500
```

#### **2. 运行训练**

```bash
python experiments/train.py --abm-episodes 500
```

#### **3. 预期结果**

```
✅ 使用 Linear SARSA 算法（连续动作空间 + 线性函数逼近）

Episode 10/500: Avg Reward=3200000, ε=0.870
Episode 100/500: Avg Reward=3600000, ε=0.500
Episode 300/500: Avg Reward=3850000, ε=0.050
Episode 500/500: Avg Reward=3900000, ε=0.050

最终平均收益: $3,900,000
```

### **参数调优建议**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.01-0.05 | 太高会不稳定，太低收敛慢 |
| `epsilon_end` | 0.05-0.1 | 保持一定探索，避免局部最优 |
| `epsilon_decay_episodes` | 300-500 | 延长探索期 |
| `n_features` | 10 | 特征维度（代码中固定） |

---

## 2️⃣ CEM（交叉熵方法）

### **算法原理**

基于采样的优化方法，通过迭代更新动作分布：

**算法流程**：
1. 维护每个状态的动作分布 `N(μ(s), σ(s))`
2. 采样N个动作并评估
3. 选择top-k个最好的动作（精英样本）
4. 更新分布参数：
   ```
   μ(s) ← mean(elite_actions)
   σ(s) ← std(elite_actions)
   ```

### **特点**

✅ **优势**：
- **最稳定**：不使用梯度，基于采样
- 真正的连续动作输出
- 适合随机环境（如ABM）
- 不会崩溃

❌ **劣势**：
- 收敛较慢（需要收集足够样本）
- 需要更多episode
- 计算量稍大

### **使用方法**

#### **1. 修改配置文件**

编辑 `configs/config.py`:

```python
@dataclass
class RLConfig:
    # 算法选择
    algorithm: str = 'cem'  # 切换到CEM
    
    # 通用参数
    n_states: int = 18
    discount_factor: float = 0.99
    
    # CEM参数
    cem_n_samples: int = 20  # 每次采样数量（建议20-50）
    cem_elite_frac: float = 0.2  # 精英样本比例（建议0.1-0.3）
    
    # 动作分布参数
    initial_std: float = 20.0  # 初始标准差
    min_std: float = 2.0  # 最小标准差
    std_decay: float = 0.99  # 标准差衰减率
    
    # 连续动作范围
    action_min: float = 80.0
    action_max: float = 170.0
    
    episodes: int = 800  # CEM需要更多轮次
```

#### **2. 运行训练**

```bash
python experiments/train.py --abm-episodes 800
```

#### **3. 预期结果**

```
✅ 使用 CEM 算法（连续动作空间 + 交叉熵方法）

Episode 10/800: Avg Reward=3100000
Episode 100/800: Avg Reward=3400000
Episode 300/800: Avg Reward=3700000
Episode 500/800: Avg Reward=3850000
Episode 800/800: Avg Reward=3900000

最终平均收益: $3,900,000
```

### **参数调优建议**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `cem_n_samples` | 20-50 | 采样数量，越大越稳定但越慢 |
| `cem_elite_frac` | 0.1-0.3 | 精英比例，0.2表示top-20% |
| `initial_std` | 20-30 | 初始探索范围 |
| `min_std` | 2-5 | 最小探索范围 |
| `std_decay` | 0.98-0.995 | 标准差衰减速度 |
| `episodes` | 800-1000 | CEM需要更多轮次 |

---

## 📊 算法对比实验

### **实验设置**
- 环境：ABM酒店定价
- 训练轮数：500 episodes
- 评估指标：平均收益、稳定性、训练时间

### **预期结果**

| 算法 | 平均收益 | 标准差 | 训练时间 | 稳定性 |
|------|----------|--------|----------|--------|
| Q-learning | $3.9M | $0.1M | 15分钟 | ⭐⭐⭐⭐⭐ |
| Actor-Critic | $3.5M | $0.5M | 12分钟 | ⭐⭐ |
| Linear SARSA | $3.8M | $0.15M | 18分钟 | ⭐⭐⭐⭐ |
| CEM | $3.85M | $0.12M | 25分钟 | ⭐⭐⭐⭐⭐ |

---

## 🎯 选择建议

### **选择Linear SARSA，如果你需要**：
- ✅ 连续动作输出
- ✅ 稳定的训练过程
- ✅ 可解释的特征工程
- ✅ 中等训练时间

### **选择CEM，如果你需要**：
- ✅ 最稳定的训练
- ✅ 连续动作输出
- ✅ 适应随机环境
- ⚠️ 可以接受较慢的收敛

### **选择Q-learning，如果你需要**：
- ✅ 最快的训练
- ✅ 最稳定的结果
- ⚠️ 可以接受离散动作

### **选择Actor-Critic，如果你需要**：
- ✅ 最快的收敛
- ✅ 连续动作输出
- ⚠️ 可以接受不稳定性

---

## 🔧 故障排除

### **Linear SARSA收敛慢**
- 增加 `learning_rate` 到 0.05
- 减少 `epsilon_decay_episodes` 到 200
- 增加训练轮数到 800

### **CEM不收敛**
- 增加 `cem_n_samples` 到 50
- 增加 `cem_elite_frac` 到 0.3
- 增加训练轮数到 1000
- 减小 `std_decay` 到 0.98

### **输出动作不合理**
- 检查 `action_min` 和 `action_max` 设置
- 确保价格范围合理（80-170元）
- 查看训练日志中的动作分布

---

## 📝 代码示例

### **手动使用Linear SARSA**

```python
from src.algorithms.linear_sarsa import LinearSARSA

# 初始化
sarsa = LinearSARSA(
    n_states=18,
    action_min=80.0,
    action_max=170.0,
    learning_rate=0.01,
    discount_factor=0.99,
    epsilon_start=0.9,
    epsilon_end=0.05,
    epsilon_decay_episodes=300
)

# 选择动作
state = 5
action = sarsa.select_action(state, deterministic=False)
print(f"选择的价格: {action:.2f}元")

# 更新
reward = 12000.0
next_state = 6
sarsa.update(state, action, reward, next_state, done=False)

# Episode结束
sarsa.end_episode()
```

### **手动使用CEM**

```python
from src.algorithms.cem import CrossEntropyMethod

# 初始化
cem = CrossEntropyMethod(
    n_states=18,
    action_min=80.0,
    action_max=170.0,
    n_samples=20,
    elite_frac=0.2,
    initial_std=20.0,
    min_std=2.0
)

# 选择动作
state = 5
action = cem.select_action(state, deterministic=False)
print(f"选择的价格: {action:.2f}元")

# 存储经验
reward = 12000.0
next_state = 6
cem.update(state, action, reward, next_state, done=False)

# Episode结束（触发分布更新）
cem.end_episode()
```

---

## 📚 参考文献

1. **SARSA**: Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems.
2. **Linear Function Approximation**: Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
3. **Cross-Entropy Method**: Rubinstein, R. Y., & Kroese, D. P. (2004). The Cross-Entropy Method.

---

## 🆘 获取帮助

如有问题，请查看：
- `docs/ALGORITHM_SWITCHING_GUIDE.md` - 算法切换指南
- `src/algorithms/linear_sarsa.py` - Linear SARSA源码
- `src/algorithms/cem.py` - CEM源码
