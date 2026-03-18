# 酒店-OTA博弈系统

本仓库实现了一个基于 **ABM（消费者行为仿真） + 多智能体博弈（酒店 vs OTA） + CEM（Cross-Entropy Method）** 的动态定价系统，用于研究不同酒店容量、渠道结构与补贴策略下的价格策略与收益表现。

设计文档（与代码同步更新）：[基于ABM-MARL的酒店-OTA动态定价博弈系统设计文档.md](docs/%E5%9F%BA%E4%BA%8EABM-MARL%E7%9A%84%E9%85%92%E5%BA%97-OTA%E5%8A%A8%E6%80%81%E5%AE%9A%E4%BB%B7%E5%8D%9A%E5%BC%88%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md)
[基于ABM-MARL的酒店-OTA动态定价博弈系统设计文档.pdf](docs/基于ABM-MARL的酒店-OTA动态定价博弈系统设计文档.pdf)


### 查看训练后的模型参数

训练完成后，模型参数会保存为 JSON 格式，可以直接用文本编辑器查看：

```bash
# 查看保存的模型文件
ls outputs/models/

# 直接打开 JSON 文件查看
cat outputs/models/hotel_online_agent_20260118_200142.json
cat outputs/models/hotel_offline_agent_20260118_200142.json
cat outputs/models/ota_agent_20260118_200142.json
```

JSON 文件中的状态序号对应上表中的状态编码，例如：
```json
{
  "cem_online_means": {
    "0": 102.5,   // 状态0：低库存+淡季+工作日 → 线上价格均值102.5元
    "1": 125.0,   // 状态1：低库存+淡季+周末 → 线上价格均值125.0元
    "12": 105.8,  // 状态12：高库存+淡季+工作日 → 线上价格均值105.8元
    ...
  }
}
```

说明：
- CEM 表格版模型文件中会保存均值/标准差/访问次数表（key 为状态索引的字符串）。
- OTA 的模型文件也使用相同的保存接口，字段名可能沿用历史命名（例如仍出现 `cem_online_means`），以文件实际键为准。

## TensorBoard 可视化

```bash
tensorboard --logdir=outputs/tensorboard_logs
```

查看训练过程中的：
- 酒店和OTA的收益曲线
- 线上/线下预订量对比
- 补贴策略变化
- 最后一个episode的逐日价格与补贴曲线（day0为“今天”）

## 模拟流程（多智能体：酒店 vs OTA，当前代码版本）

本节描述当前“训练 + 仿真”主流程要点：
- 提前期：消费者 `lead_time ∈ [0, 90]`，优先按数据集（City Hotel，截断到0-90）经验分布采样（否则用指数分布兜底）
- 预订窗口：`booking_window_days = 91`（覆盖 `day_offset=0..90`），环境维护未来91天库存与报价窗口
- 定价维护方式：对每个“入住日轨道”（窗口内每个 `day_offset`）维护独立挂牌价；只在 `decision_buckets` 的边界触发日更新（其余天保持不变）
- 学习更新方式：一次“调价决策”对应一个阶段跨度的累计效果，奖励在触发边界时做阶段结算并更新（避免碎片化更新导致噪声）

涉及模块：
- 入口脚本：experiments/train_game.py
- 训练器：src/training/game_trainer.py
- 环境：src/environment/hotel_env.py
- ABM：src/environment/abm_customer_model.py
- Agent：src/agent/hotel_agent_dual_channel.py、src/agent/ota_agent.py

### 1) 数据与初始化

1. train_game.py 读取 datasets/hotel_bookings.csv 并筛选 City Hotel。
2. 构造 ABM 的 lead_time 经验分布（若提供数据路径）：统计 City Hotel 的 lead_time（仅保留 0..90），归一化为概率向量。
3. 创建 HotelEnvironment(booking_window_days=91)：
   - future_inventory：长度 91，表示今天及未来 90 天的可售库存（按入住日独立扣减）
   - current_price_window_online/offline：长度 91，表示每个提前期/入住日轨道的报价
4. 创建 Agent：
   - HotelAgentDualChannel：输出 [price_online_base, price_offline]
   - OTAAgent：输出 subsidy_ratio（补贴比例）

### 2) Episode 时间结构

- 训练按 episode 循环；每个 episode 运行 365 个仿真日。
- 每个仿真日会向环境传入完整的 91 天价格窗口；但窗口内的大部分入住日价格是“沿用历史挂牌价”，只有触发点上的入住日会在当天更新。

### 3) 每个仿真日的“触发更新 + 执行”流程（核心）

对某一仿真日 t：

1. 解析分桶：把 0..90 的提前期划分成若干连续区间（示例：0 | 1 | 2-3 | 4-6 | 7-13 | 14-29 | 30-59 | 60-90），桶索引为 `stage_id`。
2. 触发更新（只在桶边界）：对每个 `off in {bucket_end}`：
   - 先对该入住日轨道上一阶段累计预订做结算并更新策略
   - 再读取该入住日状态 `env._get_state_for_day_offset(off)`（附带 `stage_id`）并生成新挂牌价（酒店）与新补贴率（OTA）
3. 构造 91 天价格窗口：
   - 每个 `day_offset` 使用其当前挂牌价（若当天未触发更新则沿用）
   - 计算最终线上价：`price_online_final = price_online_base - price_online_base * commission_rate * subsidy_ratio`
4. 环境执行 `env.step(actions_window)`：
   - ABM 当天生成客户，按客户的 `lead_time` 选择对应 `day_offset` 的价格进行预订决策
   - 成交后按入住日库存扣减，并返回 `bookings_by_day_offset`

### 4) ABM 在 env.step() 内做了什么（修改后）

ABM（abm_customer_model.py）在 simulate_day() 中：
1. 生成当日客户数：按月份到达率 λ_m 采样 Poisson(λ_m)。
2. 为每个客户生成 profile：
   - lead_time：从经验分布采样（0..90），target_date = current_day + lead_time
   - wtp：按历史 ADR 拟合的正态分布采样
   - customer_type：online/offline 按历史比例采样
3. 客户决策：令 days_ahead = target_date - current_day（0..90），从 price_window_online/offline 取对应 day_offset 的价格做选择。
4. 库存扣减：按 target_date（入住日）检查并扣减 daily_available_rooms[target_date]。
5. 统计：输出 bookings_by_day_offset（长度 91，每个 day_offset 的线上/线下预订量与收入）。

### 5) 训练更新（按入住日轨道、在桶边界阶段结算）

env.step() 返回 info['bookings_by_day_offset']（0..90）。训练器将其累加到每个入住日轨道的“阶段累计预订量”，并在桶边界触发点：
- 计算该轨道上一阶段累计带来的酒店收益与OTA利润
- 用混合奖励（reward_hotel_ratio/reward_ota_ratio）构造 reward_hotel/reward_ota
- 将 reward 归因到“该轨道上一阶段的挂牌价决策状态”并更新 CEM

### 6) 日度统计与日志（修改后）

- 当天收益/补贴不再只用 day_offset=0 的价格估算，而是按 bookings_by_day_offset 与对应的 (price, subsidy) 在 0..90 上加总得到。
- TensorBoard 日志仍在 outputs/tensorboard_logs/game_*，但现在 day_offset 维度更长，记录曲线可选择只展示关键桶或前若干天。

### 7) 如何运行（修改后）

```bash
python experiments/train_game.py \
  --episodes 500 \
  --mode simultaneous \
  --commission 0.15 \
  --subsidy-ratio-max 0.8 \
  --update-frequency 30 \
  --booking-window-days 91 \
  --decision-buckets "0|1|2-3|4-6|7-13|14-29|30-59|60-90"
```

### 8) 容量并行实验（capacity sweep）

使用 [train_game_ac.py](file:///Users/raily/Desktop/hotel_pricing/ABM_MARL_hotel_pricing/experiments/train_game_ac.py) 可对不同酒店容量并行训练，并输出映射 JSON 便于后处理：

```bash
python experiments/train_game_ac.py --episodes 250 --workers 4
```

训练结束会输出：
- `outputs/results/capacity_to_csv_*.json`：capacity → 训练CSV路径
- `outputs/results/capacity_to_figure_*.json`：capacity → 结果图路径
- `outputs/results/capacity_to_model_*.json`：capacity → 模型文件路径（线上/线下/OTA）
