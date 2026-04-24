"""PPO用 Gymnasium 环境（16维分桶动作）。"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import Experiment2Config
from env_wrappers.base_simulator import BucketPricingSimulator


class PPOBucketEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Experiment2Config, seed: int, historical_data):
        super().__init__()
        self.config = config
        self.sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
        self.n_stages = self.sim.n_stages

        low = np.array(
            [config.online_price_min, config.offline_price_min] * self.n_stages,
            dtype=np.float32,
        )
        high = np.array(
            [config.online_price_max, config.offline_price_max] * self.n_stages,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        obs_dim = config.booking_window_days + 1 + 12 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _decode_action(self, action: np.ndarray):
        arr = np.asarray(action, dtype=np.float64).reshape(self.n_stages, 2)
        stage_actions = []
        for row in arr:
            stage_actions.append((float(row[0]), float(row[1])))
        return stage_actions

    def reset(self, *, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            # Gymnasium reset seed 仅用于兼容接口，真正随机性由sim内部seed控制
            _ = int(seed)
        self.sim.reset()
        obs = self.sim.get_obs_vector_for_ppo()
        return obs, {}

    def step(self, action):
        stage_actions = self._decode_action(action)
        day_result = self.sim.step_day(stage_actions)
        obs = self.sim.get_obs_vector_for_ppo()
        reward = float(day_result.reward_hotel) / float(max(1.0, self.config.ppo_reward_scale))
        terminated = bool(day_result.done)
        truncated = False
        return obs, reward, terminated, truncated, day_result.info
