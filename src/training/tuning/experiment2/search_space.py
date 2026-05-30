"""PPO调参搜索空间。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class SpaceBounds:
    ent_coef_low: float
    ent_coef_high: float
    lr_low: float
    lr_high: float
    clip_low: float
    clip_high: float
    gae_low: float
    gae_high: float
    n_steps_choices: List[int]


GLOBAL_BOUNDS = SpaceBounds(
    ent_coef_low=1e-3,
    ent_coef_high=2e-2,
    lr_low=1e-4,
    lr_high=5e-4,
    clip_low=0.1,
    clip_high=0.3,
    gae_low=0.95,
    gae_high=0.99,
    n_steps_choices=[256, 365, 512],
)


def suggest_ppo_params(trial, bounds: SpaceBounds | None = None) -> Dict[str, float]:
    b = bounds or GLOBAL_BOUNDS
    return {
        "ppo_ent_coef": float(trial.suggest_float("ppo_ent_coef", b.ent_coef_low, b.ent_coef_high, log=True)),
        "ppo_learning_rate": float(trial.suggest_float("ppo_learning_rate", b.lr_low, b.lr_high, log=True)),
        "ppo_clip_range": float(trial.suggest_float("ppo_clip_range", b.clip_low, b.clip_high)),
        "ppo_gae_lambda": float(trial.suggest_float("ppo_gae_lambda", b.gae_low, b.gae_high)),
        "ppo_n_steps": int(trial.suggest_categorical("ppo_n_steps", b.n_steps_choices)),
    }


def build_refine_bounds(trials_df: pd.DataFrame, top_k: int = 6) -> SpaceBounds:
    if trials_df is None or len(trials_df) == 0:
        return GLOBAL_BOUNDS
    use_df = trials_df.copy()
    if "Stable" in use_df.columns:
        stable_df = use_df[use_df["Stable"] == True]  # noqa: E712
        if len(stable_df) > 0:
            use_df = stable_df
    top_df = use_df.sort_values("Score", ascending=False).head(max(1, int(top_k)))

    def _range(col: str, g_low: float, g_high: float) -> tuple[float, float]:
        cmin = float(top_df[col].min())
        cmax = float(top_df[col].max())
        span = max((cmax - cmin) * 0.3, (g_high - g_low) * 0.05)
        low = max(g_low, cmin - span)
        high = min(g_high, cmax + span)
        if high <= low:
            high = min(g_high, low + (g_high - g_low) * 0.1)
        return low, high

    ent_low, ent_high = _range("ppo_ent_coef", GLOBAL_BOUNDS.ent_coef_low, GLOBAL_BOUNDS.ent_coef_high)
    lr_low, lr_high = _range("ppo_learning_rate", GLOBAL_BOUNDS.lr_low, GLOBAL_BOUNDS.lr_high)
    clip_low, clip_high = _range("ppo_clip_range", GLOBAL_BOUNDS.clip_low, GLOBAL_BOUNDS.clip_high)
    gae_low, gae_high = _range("ppo_gae_lambda", GLOBAL_BOUNDS.gae_low, GLOBAL_BOUNDS.gae_high)

    n_steps = sorted(top_df["ppo_n_steps"].dropna().astype(int).unique().tolist())
    n_steps_choices = n_steps if n_steps else GLOBAL_BOUNDS.n_steps_choices

    return SpaceBounds(
        ent_coef_low=ent_low,
        ent_coef_high=ent_high,
        lr_low=lr_low,
        lr_high=lr_high,
        clip_low=clip_low,
        clip_high=clip_high,
        gae_low=gae_low,
        gae_high=gae_high,
        n_steps_choices=n_steps_choices,
    )

