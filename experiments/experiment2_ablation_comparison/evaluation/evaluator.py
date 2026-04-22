"""统一评估器：每N步做确定性评估。"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from config import Experiment2Config
from env_wrappers.base_simulator import BucketPricingSimulator


StagePolicyFn = Callable[[int, dict], Tuple[float, float]]


def evaluate_policy(
    config: Experiment2Config,
    historical_data,
    seed: int,
    stage_policy_fn: StagePolicyFn,
    n_episodes: int,
) -> float:
    rewards: List[float] = []
    for ep in range(n_episodes):
        sim = BucketPricingSimulator(config=config, seed=seed * 1000 + ep, historical_data=historical_data)
        sim.reset()
        total = 0.0
        done = False
        while not done:
            stage_actions = []
            for sid in range(sim.n_stages):
                st = sim.get_state_by_stage(sid)
                stage_actions.append(stage_policy_fn(sid, st))
            out = sim.step_day(stage_actions)
            total += out.reward_hotel
            done = out.done
        rewards.append(total)
    return float(np.mean(rewards))
