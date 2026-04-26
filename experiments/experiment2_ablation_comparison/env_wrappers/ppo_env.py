"""PPO用 Gymnasium 环境（4维结构化动作 -> 分桶动作）。"""

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

        # 4维动作:
        # [base_online, base_offline, slope_early, slope_late]
        # 其中 slope 会被展开成各分桶的价格调整项。
        price_span = float(
            min(
                config.online_price_max - config.online_price_min,
                config.offline_price_max - config.offline_price_min,
            )
        )
        self._slope_span = 0.4 * price_span
        low = np.array(
            [
                config.online_price_min,
                config.offline_price_min,
                -self._slope_span,
                -self._slope_span,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                config.online_price_max,
                config.offline_price_max,
                self._slope_span,
                self._slope_span,
            ],
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
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.size != 4:
            raise ValueError(f"Expected 4-dim action, got shape={arr.shape}")

        base_on = float(arr[0])
        base_off = float(arr[1])
        slope_early = float(arr[2])
        slope_late = float(arr[3])
        gap_off = base_off - base_on

        stage_actions = []
        max_off = max(1.0, float(self.config.booking_window_days - 1))
        for sid, (s, e) in enumerate(self.sim.buckets):
            del sid
            center_off = 0.5 * (float(s) + float(e))
            t = float(center_off / max_off)  # near=0, far=1
            adj = (1.0 - t) * slope_early + t * slope_late
            pon = np.clip(
                base_on + adj,
                self.config.online_price_min,
                self.config.online_price_max,
            )
            poff = np.clip(
                pon + gap_off,
                self.config.offline_price_min,
                self.config.offline_price_max,
            )
            stage_actions.append((float(pon), float(poff)))
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
        reward = float(day_result.reward_hotel)
        terminated = bool(day_result.done)
        truncated = False
        return obs, reward, terminated, truncated, day_result.info
