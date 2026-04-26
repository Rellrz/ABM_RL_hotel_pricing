"""实验二专用：PPO最小调参入口。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
import pandas as pd
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from config import Experiment2Config, TUNING_FIGURES_DIR
from tuning.objective import summarize_trial
from tuning.report import generate_tuning_figures
from tuning.search_space import GLOBAL_BOUNDS, build_refine_bounds, suggest_ppo_params
from tuning.trial_runner import run_ppo_trial


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="实验二PPO最小调参")
    p.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    p.add_argument("--coarse-trials", type=int, default=24)
    p.add_argument("--refine-trials", type=int, default=12)
    p.add_argument("--coarse-episodes", type=int, default=300)
    p.add_argument("--refine-episodes", type=int, default=600)
    p.add_argument("--final-episodes", type=int, default=1000)
    p.add_argument("--coarse-seeds", type=int, default=1)
    p.add_argument("--refine-seeds", type=int, default=3)
    p.add_argument("--final-seeds", type=int, default=5)
    p.add_argument("--post-eval-episodes", type=int, default=30)
    p.add_argument("--sampler-seed", type=int, default=20260425)
    return p


def load_historical_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "datasets" / "hotel_bookings.csv"
    df = pd.read_csv(path)
    return df[df["hotel"] == "City Hotel"].copy()


def _run_stage(
    stage: str,
    n_trials: int,
    base_config: Experiment2Config,
    historical_data,
    train_episodes: int,
    post_eval_episodes: int,
    n_seeds: int,
    sampler_seed: int,
    start_trial_id: int,
    prior_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bounds = GLOBAL_BOUNDS if stage == "coarse" else build_refine_bounds(prior_df if prior_df is not None else pd.DataFrame())
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=sampler_seed))
    trial_rows: List[Dict] = []
    train_parts: List[pd.DataFrame] = []
    eval_parts: List[pd.DataFrame] = []
    seeds = list(range(1, int(n_seeds) + 1))

    for i in range(int(n_trials)):
        trial = study.ask()
        params = suggest_ppo_params(trial, bounds=bounds)
        trial_id = int(start_trial_id + i)
        print(f"[TUNE][{stage}] trial={trial_id} params={params}")
        tr_df, ev_df = run_ppo_trial(
            base_config=base_config,
            historical_data=historical_data,
            trial_id=trial_id,
            params=params,
            seeds=seeds,
            train_episodes=train_episodes,
            post_eval_episodes=post_eval_episodes,
            stage=stage,
        )
        row, metrics = summarize_trial(
            trial_id=trial_id,
            stage=stage,
            params=params,
            training_df=tr_df,
            eval_df=ev_df,
        )
        row["OptunaTrialNumber"] = int(trial.number)
        trial_rows.append(row)
        train_parts.append(tr_df)
        eval_parts.append(ev_df)
        study.tell(trial, float(metrics["Score"]))

    return (
        pd.DataFrame(trial_rows),
        pd.concat(train_parts, axis=0, ignore_index=True) if train_parts else pd.DataFrame(),
        pd.concat(eval_parts, axis=0, ignore_index=True) if eval_parts else pd.DataFrame(),
    )


def _best_row(trials_df: pd.DataFrame) -> Dict:
    if trials_df is None or len(trials_df) == 0:
        raise RuntimeError("No trial rows found.")
    stable = trials_df[trials_df["Stable"] == True]  # noqa: E712
    use = stable if len(stable) > 0 else trials_df
    return use.sort_values("Score", ascending=False).iloc[0].to_dict()


def main() -> None:
    if importlib.util.find_spec("stable_baselines3") is None:  # pragma: no cover
        raise RuntimeError("未检测到 stable-baselines3，请先安装。")
    args = build_parser().parse_args()
    config = Experiment2Config(run_mode=args.mode)
    config.ensure_dirs()
    historical_data = load_historical_data()

    print("=" * 72)
    print("实验二：PPO最小调参")
    print("=" * 72)
    print(f"mode={args.mode} coarse={args.coarse_trials} refine={args.refine_trials}")

    coarse_trials_df, coarse_train_df, coarse_eval_df = _run_stage(
        stage="coarse",
        n_trials=args.coarse_trials,
        base_config=config,
        historical_data=historical_data,
        train_episodes=args.coarse_episodes,
        post_eval_episodes=args.post_eval_episodes,
        n_seeds=args.coarse_seeds,
        sampler_seed=args.sampler_seed,
        start_trial_id=1,
        prior_df=None,
    )
    refine_trials_df, refine_train_df, refine_eval_df = _run_stage(
        stage="refine",
        n_trials=args.refine_trials,
        base_config=config,
        historical_data=historical_data,
        train_episodes=args.refine_episodes,
        post_eval_episodes=args.post_eval_episodes,
        n_seeds=args.refine_seeds,
        sampler_seed=args.sampler_seed + 1,
        start_trial_id=1 + args.coarse_trials,
        prior_df=coarse_trials_df,
    )

    trials_df = pd.concat([coarse_trials_df, refine_trials_df], axis=0, ignore_index=True)
    train_df = pd.concat([coarse_train_df, refine_train_df], axis=0, ignore_index=True)
    eval_df = pd.concat([coarse_eval_df, refine_eval_df], axis=0, ignore_index=True)
    best = _best_row(trials_df)
    best_trial_id = int(best["TrialID"])
    best_params = {
        "ppo_ent_coef": float(best["ppo_ent_coef"]),
        "ppo_learning_rate": float(best["ppo_learning_rate"]),
        "ppo_clip_range": float(best["ppo_clip_range"]),
        "ppo_gae_lambda": float(best["ppo_gae_lambda"]),
        "ppo_slope_span_ratio": float(best["ppo_slope_span_ratio"]),
        "ppo_n_steps": int(best["ppo_n_steps"]),
    }

    # 最终验证：best vs baseline（同预算同seed数）
    final_seeds = list(range(1, int(args.final_seeds) + 1))
    baseline_params = {
        "ppo_ent_coef": float(config.ppo_ent_coef),
        "ppo_learning_rate": float(config.ppo_learning_rate),
        "ppo_clip_range": float(config.ppo_clip_range),
        "ppo_gae_lambda": float(config.ppo_gae_lambda),
        "ppo_slope_span_ratio": float(config.ppo_slope_span_ratio),
        "ppo_n_steps": int(config.ppo_n_steps),
    }
    final_baseline_train, final_baseline_eval = run_ppo_trial(
        base_config=config,
        historical_data=historical_data,
        trial_id=-1,
        params=baseline_params,
        seeds=final_seeds,
        train_episodes=args.final_episodes,
        post_eval_episodes=args.post_eval_episodes,
        stage="final_baseline",
    )
    final_best_train, final_best_eval = run_ppo_trial(
        base_config=config,
        historical_data=historical_data,
        trial_id=-2,
        params=best_params,
        seeds=final_seeds,
        train_episodes=args.final_episodes,
        post_eval_episodes=args.post_eval_episodes,
        stage="final_best",
    )
    final_train_df = pd.concat([final_baseline_train, final_best_train], axis=0, ignore_index=True)
    final_eval_df = pd.concat([final_baseline_eval, final_best_eval], axis=0, ignore_index=True)
    full_train_df = pd.concat([train_df, final_train_df], axis=0, ignore_index=True)
    full_eval_df = pd.concat([eval_df, final_eval_df], axis=0, ignore_index=True)

    trials_df.to_csv(config.tuning_trials_csv_path, index=False)
    full_train_df.to_csv(config.tuning_train_csv_path, index=False)
    full_eval_df.to_csv(config.tuning_eval_csv_path, index=False)

    generate_tuning_figures(
        trials_df=trials_df,
        training_df=full_train_df,
        eval_df=full_eval_df,
        out_dir=TUNING_FIGURES_DIR,
        best_trial_id=-2,
        baseline_trial_id=-1,
    )

    best_json = {
        "best_trial_id": best_trial_id,
        "best_stage": best["Stage"],
        "best_score": float(best["Score"]),
        "best_stable": bool(best["Stable"]),
        "best_params": best_params,
        "coarse_trials": int(args.coarse_trials),
        "refine_trials": int(args.refine_trials),
    }
    with open(config.tuning_best_json_path, "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    summary = {
        "mode": args.mode,
        "n_trials_total": int(len(trials_df)),
        "n_training_rows": int(len(full_train_df)),
        "n_eval_rows": int(len(full_eval_df)),
        "files": {
            "trials_csv": str(config.tuning_trials_csv_path),
            "train_csv": str(config.tuning_train_csv_path),
            "eval_csv": str(config.tuning_eval_csv_path),
            "best_json": str(config.tuning_best_json_path),
            "figures_dir": str(TUNING_FIGURES_DIR),
        },
    }
    with open(config.tuning_summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[TUNE] 完成，汇总: {config.tuning_summary_json_path}")


if __name__ == "__main__":
    main()

