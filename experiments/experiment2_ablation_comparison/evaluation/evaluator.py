"""统一评估器：训练结束后批量评估。"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from config import Experiment2Config
from env_wrappers.base_simulator import BucketPricingSimulator
from env_wrappers.ppo_env import PPOBucketEnv


StagePolicyFn = Callable[[int, dict], Tuple[float, float]]


def evaluate_policy(
    config: Experiment2Config,
    historical_data,
    seed: int,
    stage_policy_fn: StagePolicyFn,
    n_episodes: int,
) -> List[Dict[str, float]]:
    rewards: List[Dict[str, float]] = []
    for ep in range(n_episodes):
        sim = BucketPricingSimulator(config=config, seed=seed * 1000 + ep, historical_data=historical_data)
        sim.reset()
        total_hotel = 0.0
        total_ota = 0.0
        done = False
        while not done:
            stage_actions = []
            for sid in range(sim.n_stages):
                st = sim.get_state_by_stage(sid)
                stage_actions.append(stage_policy_fn(sid, st))
            out = sim.step_day(stage_actions)
            total_hotel += out.reward_hotel
            total_ota += out.reward_ota
            done = out.done
        rewards.append(
            {
                "EvalHotelRevenue": float(total_hotel),
                "EvalOTAProfit": float(total_ota),
                "EvalSystemProfit": float(total_hotel + total_ota),
            }
        )
    return rewards


def evaluate_ppo_model(
    config: Experiment2Config,
    historical_data,
    seed: int,
    model,
    n_episodes: int,
) -> List[Dict[str, float]]:
    rewards: List[Dict[str, float]] = []
    env = PPOBucketEnv(config=config, seed=seed, historical_data=historical_data)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        total_hotel = 0.0
        total_ota = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            del reward
            reward_hotel_by_stage = info.get("reward_hotel_by_stage", [])
            total_hotel += float(sum(reward_hotel_by_stage))
            reward_ota_by_stage = info.get("reward_ota_by_stage", [])
            total_ota += float(sum(reward_ota_by_stage))
            done = bool(terminated or truncated)
        rewards.append(
            {
                "EvalHotelRevenue": float(total_hotel),
                "EvalOTAProfit": float(total_ota),
                "EvalSystemProfit": float(total_hotel + total_ota),
            }
        )
    return rewards
