"""实验二主入口：对比与消融实验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plotter import plot_learning_curves, plot_post_eval_bar
from analysis.stats import build_performance_table, significance_tests
from config import Experiment2Config
from trainers.cem_runner import run_cem_family
from trainers.ppo_trainer import run_ppo
from trainers.qlearning_runner import run_qlearning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实验二：对比与消融")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-qlearning", action="store_true")
    parser.add_argument("--skip-cem", action="store_true")
    return parser


def load_historical_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "datasets" / "hotel_bookings.csv"
    df = pd.read_csv(path)
    return df[df["hotel"] == "City Hotel"].copy()


def main() -> None:
    args = build_parser().parse_args()
    config = Experiment2Config(run_mode=args.mode)
    if args.n_jobs is not None:
        config.n_jobs = int(args.n_jobs)
    config.ensure_dirs()

    print("=" * 72)
    print("实验二：对比与消融实验")
    print("=" * 72)
    print(f"运行档位: {config.run_mode}")
    print(f"Seeds: {config.n_seeds}, Train episodes: {config.train_episodes}, Train steps: {config.train_steps}")
    print(f"Bucket spec: {config.decision_buckets}")
    print(f"Training CSV: {config.training_csv_path}")
    print(f"Post-eval CSV: {config.evaluation_csv_path}")

    historical_data = load_historical_data()

    training_records = []
    eval_records = []

    if not args.skip_cem:
        print("\n[1/3] 训练 CEM 系列 ...")
        rec_train, rec_eval = run_cem_family(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[1/3] 跳过 CEM 系列（--skip-cem）")

    if not args.skip_qlearning:
        print("\n[2/3] 训练 Q-learning ...")
        rec_train, rec_eval = run_qlearning(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[2/3] 跳过 Q-learning（--skip-qlearning）")

    if not args.skip_ppo:
        print("\n[3/3] 训练 PPO ...")
        rec_train, rec_eval = run_ppo(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[3/3] 跳过 PPO（--skip-ppo）")

    train_df = pd.DataFrame(training_records)
    eval_df = pd.DataFrame(eval_records)
    train_df.to_csv(config.training_csv_path, index=False)
    eval_df.to_csv(config.evaluation_csv_path, index=False)

    perf_df = build_performance_table(train_df, eval_df)
    perf_df.to_csv(config.performance_table_csv, index=False)

    stats_df = significance_tests(eval_df)
    stats_df.to_csv(config.stats_csv_path, index=False)

    plot_learning_curves(config, train_df)
    plot_post_eval_bar(config, eval_df)

    summary = {
        "mode": config.run_mode,
        "n_training_records": int(len(train_df)),
        "n_eval_records": int(len(eval_df)),
        "algorithms": sorted(train_df["Algorithm"].unique().tolist()) if len(train_df) > 0 else [],
        "training_csv": str(config.training_csv_path),
        "evaluation_csv": str(config.evaluation_csv_path),
        "learning_curve_pdf": str(config.learning_curve_pdf),
        "eval_bar_pdf": str(config.eval_bar_pdf),
        "performance_table_csv": str(config.performance_table_csv),
        "stats_csv": str(config.stats_csv_path),
    }
    with open(config.summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成。汇总已写入: {config.summary_json_path}")


if __name__ == "__main__":
    main()
