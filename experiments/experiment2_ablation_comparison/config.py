"""实验二配置：对比与消融实验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import ENV_CONFIG, RL_CONFIG


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = THIS_DIR / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
LOGS_DIR = ARTIFACTS_DIR / "logs"


DEFAULT_BUCKET_SPEC = "0|1|2-3|4-6|7-13|14-29|30-59|60-90"


@dataclass
class Experiment2Config:
    # -----------------------------
    # 运行规模（可切换）
    # -----------------------------
    run_mode: str = "debug"  # debug / medium / full
    post_eval_episodes: int = 30
    days_per_episode: int = 365
    n_jobs: int = 1

    # -----------------------------
    # 环境与业务参数（与现有项目对齐）
    # -----------------------------
    initial_inventory: int = int(ENV_CONFIG.initial_inventory)
    booking_window_days: int = int(ENV_CONFIG.booking_window_days)
    commission_rate: float = float(RL_CONFIG.commission_rate)
    reward_hotel_ratio: float = float(RL_CONFIG.reward_hotel_ratio)
    online_price_min: float = float(RL_CONFIG.online_price_min)
    online_price_max: float = float(RL_CONFIG.online_price_max)
    offline_price_min: float = float(RL_CONFIG.offline_price_min)
    offline_price_max: float = float(RL_CONFIG.offline_price_max)
    decision_buckets: str = DEFAULT_BUCKET_SPEC
    update_frequency: int = int(RL_CONFIG.update_frequency)
    ota_r_max: float = float(RL_CONFIG.subsidy_ratio_max)
    ota_delta_max: float = float(RL_CONFIG.ota_delta_max)
    ota_decay_lambda: float = float(RL_CONFIG.ota_decay_lambda)
    ota_noise_std: float = float(RL_CONFIG.ota_noise_std)
    ota_seed: int = int(RL_CONFIG.ota_seed)

    # -----------------------------
    # 离散化动作（Q-learning）
    # -----------------------------
    q_grid_size: int = 10
    q_alpha: float = 0.1
    q_gamma: float = 0.99
    q_eps_start: float = 1.0
    q_eps_end: float = 0.05
    q_eps_decay_steps: int = 300_000

    # -----------------------------
    # PPO参数
    # -----------------------------
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_gamma: float = 0.99
    ppo_net_arch: tuple = (128, 128)

    # -----------------------------
    # CEM参数（复用项目配置）
    # -----------------------------
    cem_n_samples: int = int(RL_CONFIG.cem_n_samples)
    cem_elite_frac: float = float(RL_CONFIG.cem_elite_frac)
    cem_initial_std: float = float(RL_CONFIG.initial_std)
    cem_min_std: float = float(RL_CONFIG.min_std)
    cem_std_decay: float = float(RL_CONFIG.std_decay)

    # -----------------------------
    # 路径
    # -----------------------------
    training_csv_path: Path = RESULTS_DIR / "experiment2_training.csv"
    evaluation_csv_path: Path = RESULTS_DIR / "experiment2_post_eval.csv"
    summary_json_path: Path = RESULTS_DIR / "experiment2_summary.json"
    stats_csv_path: Path = RESULTS_DIR / "experiment2_stats.csv"
    learning_curve_pdf: Path = FIGURES_DIR / "episode_revenue_curves.pdf"
    eval_bar_pdf: Path = FIGURES_DIR / "post_eval_bar_with_errorbars.pdf"
    performance_table_csv: Path = RESULTS_DIR / "performance_table.csv"

    def ensure_dirs(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def mode_profile(self) -> Dict[str, int]:
        profiles = {
            "debug": {"n_seeds": 5, "train_episodes": 137},
            "medium": {"n_seeds": 10, "train_episodes": 548},
            "full": {"n_seeds": 30, "train_episodes": 1370},
        }
        if self.run_mode not in profiles:
            raise ValueError(f"Unknown run_mode={self.run_mode}")
        return profiles[self.run_mode]

    @property
    def n_seeds(self) -> int:
        return int(self.mode_profile["n_seeds"])

    @property
    def train_episodes(self) -> int:
        return int(self.mode_profile["train_episodes"])

    @property
    def train_steps(self) -> int:
        return int(self.train_episodes * self.days_per_episode)

    @property
    def seed_list(self) -> List[int]:
        return list(range(1, self.n_seeds + 1))

    @property
    def n_stages(self) -> int:
        # 默认8个分桶，和现有配置一致
        return 8

    @property
    def q_n_states(self) -> int:
        # 3*3*2*8 = 144
        return 144

    @property
    def q_action_grid(self) -> np.ndarray:
        points_on = np.linspace(self.online_price_min, self.online_price_max, self.q_grid_size)
        points_off = np.linspace(self.offline_price_min, self.offline_price_max, self.q_grid_size)
        actions = []
        for pon in points_on:
            for poff in points_off:
                actions.append([float(pon), float(poff)])
        return np.asarray(actions, dtype=np.float64)
