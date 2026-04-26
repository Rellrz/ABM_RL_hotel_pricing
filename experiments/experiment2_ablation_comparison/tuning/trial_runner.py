"""执行单个PPO调参trial。"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Tuple

import pandas as pd

from config import Experiment2Config
from trainers.ppo_trainer import run_ppo_single_seed


def run_ppo_trial(
    base_config: Experiment2Config,
    historical_data,
    trial_id: int,
    params: Dict[str, float],
    seeds: Iterable[int],
    train_episodes: int,
    post_eval_episodes: int,
    stage: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = deepcopy(base_config)
    cfg.override_train_episodes = int(train_episodes)
    cfg.post_eval_episodes = int(post_eval_episodes)
    cfg.ppo_use_sde = False

    for k, v in params.items():
        setattr(cfg, k, v)

    all_train = []
    all_eval = []
    algo_name = f"PPO_TRIAL_{int(trial_id)}"
    for sd in seeds:
        tr, ev, _ = run_ppo_single_seed(
            config=cfg,
            historical_data=historical_data,
            seed=int(sd),
            show_progress=False,
            algorithm_name=algo_name,
        )
        for r in tr:
            r["TrialID"] = int(trial_id)
            r["Stage"] = str(stage)
        for r in ev:
            r["TrialID"] = int(trial_id)
            r["Stage"] = str(stage)
        all_train.extend(tr)
        all_eval.extend(ev)

    return pd.DataFrame(all_train), pd.DataFrame(all_eval)

