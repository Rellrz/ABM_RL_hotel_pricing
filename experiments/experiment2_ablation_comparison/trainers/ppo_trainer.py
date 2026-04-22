"""PPO runner（SB3）。"""

from __future__ import annotations

import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from config import Experiment2Config
from env_wrappers.ppo_env import PPOBucketEnv


def run_ppo(config: Experiment2Config, historical_data) -> List[Dict]:
    if importlib.util.find_spec("stable_baselines3") is None:  # pragma: no cover
        raise RuntimeError(
            "未检测到 stable-baselines3，请先在abm环境安装：pip install stable-baselines3"
        )

    records: List[Dict] = []
    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="PPO Seeds", unit="seed"):
            tqdm.write(f"[PPO] Seed {seed} start")
            seed_records, _seed = _run_single_seed(config, historical_data, seed, show_progress=True)
            records.extend(seed_records)
            tqdm.write(f"[PPO] Seed {_seed} done: eval={len(seed_records)}")
        return records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            futures.append(ex.submit(_run_single_seed, config, historical_data, seed, True))

        with tqdm(total=len(futures), desc="PPO Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                seed_records, seed = fut.result()
                records.extend(seed_records)
                pbar.update(1)
                tqdm.write(f"[PPO] Seed {seed} done: eval={len(seed_records)}")
    return records


def _run_single_seed(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
) -> tuple[List[Dict], int]:
    from stable_baselines3 import PPO

    records: List[Dict] = []
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

    learned = 0
    pbar = tqdm(
        total=config.train_steps,
        desc=f"PPO Seed {seed}",
        unit="step",
        leave=False,
        disable=not show_progress,
    )
    while learned < config.train_steps:
        chunk = min(config.steps_per_eval, config.train_steps - learned)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        learned += chunk
        pbar.update(chunk)

        eval_scores = []
        for ep in range(config.eval_episodes):
            obs, _ = env.reset(seed=seed * 1000 + ep)
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            eval_scores.append(total)

        records.append(
            {
                "Algorithm": "PPO",
                "Seed": seed,
                "Timesteps": learned,
                "EvalReward": float(np.mean(eval_scores)),
            }
        )
        pbar.set_postfix({"eval": f"{float(np.mean(eval_scores)):.1f}"})
    pbar.close()
    return records, seed
