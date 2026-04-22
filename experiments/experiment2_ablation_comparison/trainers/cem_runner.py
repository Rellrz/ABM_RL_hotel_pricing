"""CEM 系列 runner：Multivariate CEM 与 Independent CEM。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from agents.independent_cem_agent import IndependentCEMAgent
from common import state_to_144
from config import Experiment2Config
from env_wrappers.base_simulator import BucketPricingSimulator
from evaluation.evaluator import evaluate_policy
from src.algorithms.multivariate_cem import MultivariateCrossEntropyMethod


def _run_single_seed_multivariate(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    cem = MultivariateCrossEntropyMethod(
        n_states=config.q_n_states,
        action_mins=(config.online_price_min, config.offline_price_min),
        action_maxs=(config.online_price_max, config.offline_price_max),
        discount_factor=0.99,
        n_samples=config.cem_n_samples,
        elite_frac=config.cem_elite_frac,
        initial_std=config.cem_initial_std,
        min_std=config.cem_min_std,
        std_decay=config.cem_std_decay,
        memory_size=400,
    )
    sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
    sim.reset()
    init_actions = []
    for sid in range(sim.n_stages):
        st0 = sim.get_state_by_stage(sid)
        s0 = state_to_144(st0, sid)
        a0 = cem.select_action(s0, deterministic=False)
        init_actions.append((float(a0[0]), float(a0[1])))
    sim.initialize_episode_decisions(init_actions)
    records: List[Dict] = []
    cov_records: List[Dict] = []

    # 典型状态：day0的stage0状态
    prototype_idx = state_to_144(sim.get_state_by_stage(0), stage_id=0)
    steps = 0
    done = False
    pbar = tqdm(
        total=config.train_steps,
        desc=f"MV-CEM Seed {seed}",
        unit="step",
        leave=False,
        disable=not show_progress,
    )
    while steps < config.train_steps:
        if done:
            sim.reset()
            # 对齐game_trainer：新episode先初始化一套全窗口分桶决策
            init_actions = []
            for sid in range(sim.n_stages):
                st0 = sim.get_state_by_stage(sid)
                s0 = state_to_144(st0, sid)
                a0 = cem.select_action(s0, deterministic=False)
                init_actions.append((float(a0[0]), float(a0[1])))
            sim.initialize_episode_decisions(init_actions)
            done = False

        states = []
        actions = []
        for sid in range(sim.n_stages):
            st = sim.get_state_by_stage(sid)
            s_idx = state_to_144(st, sid)
            act = cem.select_action(s_idx, deterministic=False)
            states.append(s_idx)
            actions.append((float(act[0]), float(act[1])))

        out = sim.step_day(actions)
        update_events = out.info.get("update_events", [])
        for ev in update_events:
            s = state_to_144(ev.state, int(ev.state.get("stage_id", 0)))
            s_next = state_to_144(ev.next_state, int(ev.next_state.get("stage_id", 0)))
            a = np.asarray(ev.action_pair, dtype=np.float64)
            cem.update(s, a, float(ev.reward), s_next, bool(ev.done))

        steps += 1
        pbar.update(1)
        done = out.done

        if config.update_frequency > 0 and (sim.day % config.update_frequency == 0):
            cem.end_episode()
        if done and (config.update_frequency <= 0 or (sim.day % config.update_frequency != 0)):
            cem.end_episode()

        if steps % config.steps_per_eval == 0:
            def stage_policy_fn(stage_id: int, st: dict):
                s_idx = state_to_144(st, stage_id)
                action = cem.select_action(s_idx, deterministic=True)
                return float(action[0]), float(action[1])

            eval_reward = evaluate_policy(
                config=config,
                historical_data=historical_data,
                seed=seed + 100_000,
                stage_policy_fn=stage_policy_fn,
                n_episodes=config.eval_episodes,
            )
            records.append(
                {
                    "Algorithm": "Multivariate CEM",
                    "Seed": seed,
                    "Timesteps": steps,
                    "EvalReward": eval_reward,
                }
            )
            cov = np.asarray(cem.cov_table[prototype_idx], dtype=np.float64)
            cov_records.append(
                {
                    "Algorithm": "Multivariate CEM",
                    "Seed": seed,
                    "Timesteps": steps,
                    "StateIndex": int(prototype_idx),
                    "cov_00": float(cov[0, 0]),
                    "cov_01": float(cov[0, 1]),
                    "cov_10": float(cov[1, 0]),
                    "cov_11": float(cov[1, 1]),
                }
            )
            pbar.set_postfix({"eval": f"{eval_reward:.1f}", "day": sim.day})
    pbar.close()

    return records, cov_records


def _run_single_seed_independent(
    config: Experiment2Config,
    historical_data,
    seed: int,
    show_progress: bool = True,
) -> List[Dict]:
    agent = IndependentCEMAgent(config)
    sim = BucketPricingSimulator(config=config, seed=seed, historical_data=historical_data)
    sim.reset()
    init_actions = []
    for sid in range(sim.n_stages):
        st0 = sim.get_state_by_stage(sid)
        s0 = state_to_144(st0, sid)
        a0 = agent.select_action(s0, deterministic=False)
        init_actions.append((float(a0[0]), float(a0[1])))
    sim.initialize_episode_decisions(init_actions)
    records: List[Dict] = []
    steps = 0
    done = False
    pbar = tqdm(
        total=config.train_steps,
        desc=f"IND-CEM Seed {seed}",
        unit="step",
        leave=False,
        disable=not show_progress,
    )
    while steps < config.train_steps:
        if done:
            sim.reset()
            init_actions = []
            for sid in range(sim.n_stages):
                st0 = sim.get_state_by_stage(sid)
                s0 = state_to_144(st0, sid)
                a0 = agent.select_action(s0, deterministic=False)
                init_actions.append((float(a0[0]), float(a0[1])))
            sim.initialize_episode_decisions(init_actions)
            done = False

        states = []
        actions = []
        for sid in range(sim.n_stages):
            st = sim.get_state_by_stage(sid)
            s_idx = state_to_144(st, sid)
            act = agent.select_action(s_idx, deterministic=False)
            states.append(s_idx)
            actions.append((float(act[0]), float(act[1])))

        out = sim.step_day(actions)
        update_events = out.info.get("update_events", [])
        for ev in update_events:
            s = state_to_144(ev.state, int(ev.state.get("stage_id", 0)))
            s_next = state_to_144(ev.next_state, int(ev.next_state.get("stage_id", 0)))
            a = np.asarray(ev.action_pair, dtype=np.float64)
            agent.update(s, a, float(ev.reward), s_next, bool(ev.done))

        steps += 1
        pbar.update(1)
        done = out.done

        if config.update_frequency > 0 and (sim.day % config.update_frequency == 0):
            agent.end_episode()
        if done and (config.update_frequency <= 0 or (sim.day % config.update_frequency != 0)):
            agent.end_episode()

        if steps % config.steps_per_eval == 0:
            def stage_policy_fn(stage_id: int, st: dict):
                s_idx = state_to_144(st, stage_id)
                action = agent.select_action(s_idx, deterministic=True)
                return float(action[0]), float(action[1])

            eval_reward = evaluate_policy(
                config=config,
                historical_data=historical_data,
                seed=seed + 200_000,
                stage_policy_fn=stage_policy_fn,
                n_episodes=config.eval_episodes,
            )
            records.append(
                {
                    "Algorithm": "Independent CEM",
                    "Seed": seed,
                    "Timesteps": steps,
                    "EvalReward": eval_reward,
                }
            )
            pbar.set_postfix({"eval": f"{eval_reward:.1f}", "day": sim.day})
    pbar.close()

    return records


def run_cem_family(
    config: Experiment2Config,
    historical_data,
) -> Tuple[List[Dict], List[Dict]]:
    all_records: List[Dict] = []
    cov_records: List[Dict] = []
    if config.n_jobs <= 1:
        for seed in tqdm(config.seed_list, desc="CEM Family Seeds", unit="seed"):
            tqdm.write(f"[CEM] Seed {seed} start")
            rec_mv, cov_mv = _run_single_seed_multivariate(config, historical_data, seed, show_progress=True)
            rec_ind = _run_single_seed_independent(config, historical_data, seed, show_progress=True)
            all_records.extend(rec_mv)
            all_records.extend(rec_ind)
            cov_records.extend(cov_mv)
            tqdm.write(f"[CEM] Seed {seed} done: mv_eval={len(rec_mv)} ind_eval={len(rec_ind)}")
        return all_records, cov_records

    futures = []
    with ProcessPoolExecutor(max_workers=config.n_jobs) as ex:
        for seed in config.seed_list:
            fut = ex.submit(
                _run_seed_cem_bundle,
                config,
                historical_data,
                seed,
            )
            futures.append(fut)

        with tqdm(total=len(futures), desc="CEM Family Seeds", unit="seed") as pbar:
            for fut in as_completed(futures):
                rec_mv, rec_ind, cov_mv, seed = fut.result()
                all_records.extend(rec_mv)
                all_records.extend(rec_ind)
                cov_records.extend(cov_mv)
                pbar.update(1)
                tqdm.write(f"[CEM] Seed {seed} done: mv_eval={len(rec_mv)} ind_eval={len(rec_ind)}")
    return all_records, cov_records


def _run_seed_cem_bundle(
    config: Experiment2Config,
    historical_data,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict], int]:
    rec_mv, cov_mv = _run_single_seed_multivariate(config, historical_data, seed, show_progress=True)
    rec_ind = _run_single_seed_independent(config, historical_data, seed, show_progress=True)
    return rec_mv, rec_ind, cov_mv, seed
