# 酒店-OTA博弈系统

## 状态空间编码对照表

### 酒店Agent状态编码（18种状态）

状态编码公式：`state_idx = inventory_level × 6 + season × 2 + weekday`

| 状态序号 | 库存水平 | 季节 | 日期类型 | 说明 |
|---------|---------|------|---------|------|
| 0 | 低库存 (≤33间) | 淡季 | 工作日 | 低库存+淡季+工作日 |
| 1 | 低库存 (≤33间) | 淡季 | 周末 | 低库存+淡季+周末 |
| 2 | 低库存 (≤33间) | 平季 | 工作日 | 低库存+平季+工作日 |
| 3 | 低库存 (≤33间) | 平季 | 周末 | 低库存+平季+周末 |
| 4 | 低库存 (≤33间) | 旺季 | 工作日 | 低库存+旺季+工作日 |
| 5 | 低库存 (≤33间) | 旺季 | 周末 | 低库存+旺季+周末 |
| 6 | 中库存 (34-66间) | 淡季 | 工作日 | 中库存+淡季+工作日 |
| 7 | 中库存 (34-66间) | 淡季 | 周末 | 中库存+淡季+周末 |
| 8 | 中库存 (34-66间) | 平季 | 工作日 | 中库存+平季+工作日 |
| 9 | 中库存 (34-66间) | 平季 | 周末 | 中库存+平季+周末 |
| 10 | 中库存 (34-66间) | 旺季 | 工作日 | 中库存+旺季+工作日 |
| 11 | 中库存 (34-66间) | 旺季 | 周末 | 中库存+旺季+周末 |
| 12 | 高库存 (≥67间) | 淡季 | 工作日 | 高库存+淡季+工作日 |
| 13 | 高库存 (≥67间) | 淡季 | 周末 | 高库存+淡季+周末 |
| 14 | 高库存 (≥67间) | 平季 | 工作日 | 高库存+平季+工作日 |
| 15 | 高库存 (≥67间) | 平季 | 周末 | 高库存+平季+周末 |
| 16 | 高库存 (≥67间) | 旺季 | 工作日 | 高库存+旺季+工作日 |
| 17 | 高库存 (≥67间) | 旺季 | 周末 | 高库存+旺季+周末 |

### 状态维度说明

**库存水平（inventory_level）**：
- 0: 低库存 - 剩余房间 ≤33间 (≤33%)
- 1: 中库存 - 剩余房间 34-66间 (34%-66%)
- 2: 高库存 - 剩余房间 ≥67间 (≥67%)

**季节（season）**：
- 0: 淡季 - 1-3月, 7-8月
- 1: 平季 - 4-6月, 11-12月
- 2: 旺季 - 9-10月

**日期类型（weekday）**：
- 0: 工作日 - 周一至周四
- 1: 周末 - 周五至周日

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

## TensorBoard 可视化

```bash
tensorboard --logdir=outputs/tensorboard_logs
```

查看训练过程中的：
- 酒店和OTA的收益曲线
- 线上/线下预订量对比
- 补贴策略变化
- 最后一个episode的每日价格和补贴曲线

## 模拟流程（多智能体：酒店 vs OTA，修改后版本）

本节描述你确认并应用以下修改后的“训练 + 仿真”主流程：
- 提前期：消费者 lead_time ∈ [0,90]，并严格按数据集（City Hotel，截断到0-90）经验分布采样
- 预订窗口：booking_window_days = 91（覆盖 day_offset=0..90）
- 决策方式：不再对 91 天逐日独立决策，而是用“提前期分桶”降低动作维度（每个桶输出一组价格/补贴，然后展开成 91 天窗口）

涉及模块：
- 入口脚本：experiments/train_game.py
- 训练器：src/training/game_trainer.py
- 环境：src/environment/hotel_env.py
- ABM：src/environment/abm_customer_model.py
- Agent：src/agent/hotel_agent_dual_channel.py、src/agent/ota_agent.py

### 1) 数据与初始化

1. train_game.py 读取 datasets/hotel_bookings.csv 并筛选 City Hotel。
2. 构造 ABM 的 lead_time 经验分布：统计 City Hotel 的 lead_time（仅保留 0..90），归一化为概率向量，用于采样。
3. 创建 HotelEnvironment(booking_window_days=91)：
   - future_inventory：长度 91，表示今天及未来 90 天的可售库存
   - current_price_window_online/offline：长度 91，表示每个提前期对应的报价
4. 创建 Agent：
   - HotelAgentDualChannel：输出 [price_online_base, price_offline]
   - OTAAgent：输出 subsidy_ratio（补贴比例）

### 2) Episode 时间结构

- 训练按 episode 循环；每个 episode 运行 365 个仿真日。
- 每个仿真日 t 的动作不再是 5 天窗口，而是“91 天价格窗口”（通过分桶展开得到）。

### 3) 每个仿真日的分桶决策与执行（核心）

对某一仿真日 t：

1. 分桶（bucket）：把 0..90 的提前期划分成若干连续区间（默认示例：0 | 1 | 2-3 | 4-6 | 7-13 | 14-29 | 30-59 | 60-90）。
2. 对每个 bucket 只决策一次：
   - 取该桶起点 day_offset=start 的状态：env._get_state_for_day_offset(start)
   - 酒店输出该桶的 [price_online_base, price_offline]
   - OTA 输出该桶的 subsidy_ratio（fixed_ota 模式则按规则生成）
3. 把 bucket 决策展开成 91 天窗口：
   - price_online_base_window[0..90]、price_offline_window[0..90]、subsidy_ratio_window[0..90]
4. 计算最终线上价（逐 day_offset）：
   - price_online_final[d] = price_online_base_window[d] - price_online_base_window[d] * commission_rate * subsidy_ratio_window[d]
5. 环境执行一步：
   - actions_window = [[price_online_final[d], price_offline_window[d]] for d in 0..90]
   - env.step(actions_window) 同步 91 天价格与 91 天库存到 ABM，并模拟“当天新增客户的预订”。

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

### 5) 训练更新（按 bucket 聚合更新）

env.step() 返回 info['bookings_by_day_offset']（0..90）。训练器按 bucket 聚合预订量，再做一次更新：
- 对每个 bucket，把该桶覆盖的 day_offset 区间内 bookings_online/bookings_offline 求和。
- 计算该 bucket 的：
  - 酒店收益：hotel_agent.calculate_revenue(Σbookings_online, Σbookings_offline, price_online_base_bucket, price_offline_bucket)
  - OTA 利润：ota_agent.calculate_profit(Σbookings_online, price_online_base_bucket, subsidy_ratio_bucket)
  - 实际补贴金额：Σbookings_online * price_online_base_bucket * commission_rate * subsidy_ratio_bucket
- 用混合奖励（reward_hotel_ratio/reward_ota_ratio）生成 reward_hotel/reward_ota
- 调用 update：
  - hotel_agent.update(state_for_bucket, [price_online_base_bucket, price_offline_bucket], reward_hotel, next_state, done, actual_subsidy_amount)
  - ota_agent.update(price_online_base_bucket, price_offline_bucket, state_for_bucket, subsidy_ratio_bucket, reward_ota, next_state, done)

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