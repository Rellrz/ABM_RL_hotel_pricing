"""PPO runner（SB3）。"""

from __future__ import annotations

import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from config import Experiment2Config
from evaluation.evaluator import evaluate_ppo_model
from env_wrappers.ppo_env import PPOBucketEnv


def run_ppo(config: Experiment2Config, historical_data) -> tuple[List[Dict], List[Dict]]:
    if importlib.util.find_spec("stable_baselines3") is None:  # pragma: no cover
        raise RuntimeError(
            "未检测到 stable-baselines3，请先在abm环境安装：pip install stable-baselines3"
        )

    all_train_records: List[Dict] = []
    all_eval_records: List[Dict] = []
    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="PPO Seeds", unit="seed"):
            tqdm.write(f"[PPO] Seed {seed} start")
            train_records, eval_records, _seed = _run_single_seed(config, historical_data, seed, show_progress=True)
            all_train_records.extend(train_records)
            all_eval_records.extend(eval_records)
            tqdm.write(f"[PPO] Seed {_seed} done: ep={len(train_records)}")
        return all_train_records, all_eval_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed, config, historical_data, seed, True))

        with tqdm(total=len(futures), desc="PPO Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                train_records, eval_records, seed = fut.result()
                all_train_records.extend(train_records)
                all_eval_records.extend(eval_records)
                pbar.update(1)
                tqdm.write(f"[PPO] Seed {seed} done: ep={len(train_records)}")
    return all_train_records, all_eval_records


def _run_single_seed(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
) -> tuple[List[Dict], List[Dict], int]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeRevenueCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards: List[float] = []
            self._current_reward = 0.0

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float64).reshape(-1)
            dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
            for reward, done in zip(rewards, dones):
                self._current_reward += float(reward)
                if bool(done):
                    self.episode_rewards.append(float(self._current_reward))
                    self._current_reward = 0.0
            return True

    train_records: List[Dict] = []
    eval_records: List[Dict] = []
    env = PPOBucketEnv(config=config, seed=seed, historical_data=historical_data)
    policy_kwargs = dict(net_arch=list(config.ppo_net_arch))
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.ppo_learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        gamma=config.ppo_gamma,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    callback = EpisodeRevenueCallback()
    learned = 0
    pbar = tqdm(
        total=config.train_steps,
        desc=f"PPO Seed {seed}",
        unit="step",
        leave=False,
        disable=not show_progress,
    )
    while learned < config.train_steps:
        chunk = min(config.days_per_episode, config.train_steps - learned)
        model.learn(
            total_timesteps=chunk,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        learned += chunk
        pbar.update(chunk)
        pbar.set_postfix({"ep": len(callback.episode_rewards)})
    pbar.close()

    for idx, rew in enumerate(callback.episode_rewards, start=1):
        train_records.append(
            {
                "Algorithm": "PPO",
                "Seed": seed,
                "Episode": idx,
                "EpisodeRevenue": float(rew),
            }
        )

    eval_rewards = evaluate_ppo_model(
        config=config,
        historical_data=historical_data,
        seed=seed + 400_000,
        model=model,
        n_episodes=config.post_eval_episodes,
    )
    for idx, rew in enumerate(eval_rewards, start=1):
        eval_records.append(
            {
                "Algorithm": "PPO",
                "Seed": seed,
                "EvalEpisode": idx,
                "EvalRevenue": float(rew),
            }
        )
    return train_records, eval_records, seed
