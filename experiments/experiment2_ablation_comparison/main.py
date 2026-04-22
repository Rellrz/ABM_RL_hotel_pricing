"""实验二主入口：对比与消融实验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plotter import plot_covariance_evolution, plot_learning_curves
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
    print(f"Seeds: {config.n_seeds}, Train steps: {config.train_steps}")
    print(f"Bucket spec: {config.decision_buckets}")
    print(f"Results CSV: {config.results_csv_path}")

    historical_data = load_historical_data()

    records = []
    cov_records = []

    print("\n[1/3] 训练 CEM 系列 ...")
    rec_cem, rec_cov = run_cem_family(config, historical_data)
    records.extend(rec_cem)
    cov_records.extend(rec_cov)

    print("\n[2/3] 训练 Q-learning ...")
    records.extend(run_qlearning(config, historical_data))

    if not args.skip_ppo:
        print("\n[3/3] 训练 PPO ...")
        records.extend(run_ppo(config, historical_data))
    else:
        print("\n[3/3] 跳过 PPO（--skip-ppo）")

    df = pd.DataFrame(records)
    df.to_csv(config.results_csv_path, index=False)

    cov_df = pd.DataFrame(cov_records)
    if len(cov_df) > 0:
        cov_df.to_csv(config.results_csv_path.with_name("covariance_trace.csv"), index=False)

    perf_df = build_performance_table(df)
    perf_df.to_csv(config.performance_table_csv, index=False)

    stats_df = significance_tests(df)
    stats_df.to_csv(config.stats_csv_path, index=False)

    plot_learning_curves(config, df)
    plot_covariance_evolution(config, cov_df)

    summary = {
        "mode": config.run_mode,
        "n_records": int(len(df)),
        "algorithms": sorted(df["Algorithm"].unique().tolist()) if len(df) > 0 else [],
        "results_csv": str(config.results_csv_path),
        "learning_curve_pdf": str(config.learning_curve_pdf),
        "covariance_pdf": str(config.covariance_pdf),
        "performance_table_csv": str(config.performance_table_csv),
        "stats_csv": str(config.stats_csv_path),
    }
    with open(config.summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成。汇总已写入: {config.summary_json_path}")


if __name__ == "__main__":
    main()
