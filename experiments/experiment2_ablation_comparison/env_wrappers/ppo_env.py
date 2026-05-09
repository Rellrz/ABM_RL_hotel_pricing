"""PPO用 Gymnasium 环境（标准化动作，无结构性价格先验）。"""

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

        # PPO 始终在标准化动作空间 [-1, 1] 上探索。
        # 这样初始均值 0 对应真实价格区间中点，而不是价格下界。
        action_dim = self.n_stages * 2
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        obs_dim = config.booking_window_days + 1 + 12 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _decode_action(self, action: np.ndarray):
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        expected_dim = self.n_stages * 2
        if arr.size != expected_dim:
            raise ValueError(f"Expected {expected_dim}-dim action, got shape={arr.shape}")

        stage_actions = []
        for sid in range(self.n_stages):
            pon = self._denormalize_price(
                arr[2 * sid],
                self.config.online_price_min,
                self.config.online_price_max,
            )
            poff = self._denormalize_price(
                arr[2 * sid + 1],
                self.config.offline_price_min,
                self.config.offline_price_max,
            )
            stage_actions.append((float(pon), float(poff)))
        return stage_actions

    @staticmethod
    def _denormalize_price(value: float, low: float, high: float) -> float:
        # 标准化动作 0 -> 价格中点；[-1,1] 均匀覆盖整个价格区间
        clipped = float(np.clip(value, -1.0, 1.0))
        return float(low + (clipped + 1.0) * 0.5 * (high - low))

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
        reward = float(day_result.reward_hotel)
        terminated = bool(day_result.done)
        truncated = False
        return obs, reward, terminated, truncated, day_result.info
