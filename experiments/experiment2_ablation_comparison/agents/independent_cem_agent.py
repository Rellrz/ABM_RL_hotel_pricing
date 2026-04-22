"""Independent CEM：去协方差消融版。"""

from __future__ import annotations

import numpy as np

from config import Experiment2Config
from src.algorithms.cem import CrossEntropyMethod


class IndependentCEMAgent:
    """每个状态维护线上/线下两个独立一维高斯分布。"""

    def __init__(self, config: Experiment2Config):
        self.config = config
        self.cem_online = CrossEntropyMethod(
            n_states=config.q_n_states,
            action_min=config.online_price_min,
            action_max=config.online_price_max,
            discount_factor=0.99,
            n_samples=config.cem_n_samples,
            elite_frac=config.cem_elite_frac,
            initial_std=config.cem_initial_std,
            min_std=config.cem_min_std,
            std_decay=config.cem_std_decay,
            memory_size=400,
        )
        self.cem_offline = CrossEntropyMethod(
            n_states=config.q_n_states,
            action_min=config.offline_price_min,
            action_max=config.offline_price_max,
            discount_factor=0.99,
            n_samples=config.cem_n_samples,
            elite_frac=config.cem_elite_frac,
            initial_std=config.cem_initial_std,
            min_std=config.cem_min_std,
            std_decay=config.cem_std_decay,
            memory_size=400,
        )

    def select_action(self, state_idx: int, deterministic: bool = False) -> np.ndarray:
        pon = self.cem_online.select_action(state_idx, deterministic=deterministic)
        poff = self.cem_offline.select_action(state_idx, deterministic=deterministic)
        return np.array([pon, poff], dtype=np.float64)

    def update(self, s: int, a_pair: np.ndarray, r: float, s_next: int, done: bool) -> None:
        self.cem_online.update(s, float(a_pair[0]), float(r), s_next, done)
        self.cem_offline.update(s, float(a_pair[1]), float(r), s_next, done)

    def end_episode(self) -> None:
        self.cem_online.end_episode()
        self.cem_offline.end_episode()
