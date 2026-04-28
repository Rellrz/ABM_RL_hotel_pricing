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
from trainers.emsrb_trainer import run_emsrb
from trainers.ppo_trainer import run_ppo
from trainers.qlearning_runner import run_qlearning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实验二：对比与消融")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "medium", "full"])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-qlearning", action="store_true")
    parser.add_argument("--skip-cem", action="store_true")
    parser.add_argument("--skip-emsrb", action="store_true")
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

    if not args.skip_emsrb:
        print("\n[1/4] 运行 EMSR-b（0训练，直接评估） ...")
        rec_train, rec_eval = run_emsrb(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[1/4] 跳过 EMSR-b（--skip-emsrb）")

    if not args.skip_cem:
        print("\n[2/4] 训练 CEM 系列 ...")
        rec_train, rec_eval = run_cem_family(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[2/4] 跳过 CEM 系列（--skip-cem）")

    if not args.skip_qlearning:
        print("\n[3/4] 训练 Q-learning ...")
        rec_train, rec_eval = run_qlearning(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[3/4] 跳过 Q-learning（--skip-qlearning）")

    if not args.skip_ppo:
        print("\n[4/4] 训练 PPO ...")
        rec_train, rec_eval = run_ppo(config, historical_data)
        training_records.extend(rec_train)
        eval_records.extend(rec_eval)
    else:
        print("\n[4/4] 跳过 PPO（--skip-ppo）")

    train_columns = [
        "Algorithm",
        "Seed",
        "Episode",
        "EpisodeHotelRevenue",
        "EpisodeOTAProfit",
        "EpisodeSystemProfit",
        "EpisodeRevenue",
    ]
    eval_columns = [
        "Algorithm",
        "Seed",
        "EvalEpisode",
        "EvalHotelRevenue",
        "EvalOTAProfit",
        "EvalSystemProfit",
        "EvalRevenue",
    ]
    train_df = (
        pd.DataFrame(training_records)
        if len(training_records) > 0
        else pd.DataFrame(columns=train_columns)
    )
    eval_df = (
        pd.DataFrame(eval_records)
        if len(eval_records) > 0
        else pd.DataFrame(columns=eval_columns)
    )
    train_df.to_csv(config.training_csv_path, index=False)
    eval_df.to_csv(config.evaluation_csv_path, index=False)

    perf_hotel_df = build_performance_table(
        train_df,
        eval_df,
        training_metric_col="EpisodeHotelRevenue",
        eval_metric_col="EvalHotelRevenue",
        metric_name="Hotel Revenue",
    )
    perf_hotel_df.to_csv(config.performance_table_hotel_csv, index=False)
    perf_ota_df = build_performance_table(
        train_df,
        eval_df,
        training_metric_col="EpisodeOTAProfit",
        eval_metric_col="EvalOTAProfit",
        metric_name="OTA Profit",
    )
    perf_ota_df.to_csv(config.performance_table_ota_csv, index=False)
    perf_system_df = build_performance_table(
        train_df,
        eval_df,
        training_metric_col="EpisodeSystemProfit",
        eval_metric_col="EvalSystemProfit",
        metric_name="System Profit",
    )
    perf_system_df.to_csv(config.performance_table_system_csv, index=False)

    stats_hotel_df = significance_tests(eval_df, eval_metric_col="EvalHotelRevenue")
    stats_hotel_df.to_csv(config.stats_hotel_csv_path, index=False)
    stats_ota_df = significance_tests(eval_df, eval_metric_col="EvalOTAProfit")
    stats_ota_df.to_csv(config.stats_ota_csv_path, index=False)
    stats_system_df = significance_tests(eval_df, eval_metric_col="EvalSystemProfit")
    stats_system_df.to_csv(config.stats_system_csv_path, index=False)

    plot_learning_curves(
        config,
        train_df,
        metric_col="EpisodeHotelRevenue",
        ylabel="Episode Hotel Revenue",
        output_path=config.learning_curve_hotel_pdf,
    )
    plot_learning_curves(
        config,
        train_df,
        metric_col="EpisodeOTAProfit",
        ylabel="Episode OTA Profit",
        output_path=config.learning_curve_ota_pdf,
    )
    plot_learning_curves(
        config,
        train_df,
        metric_col="EpisodeSystemProfit",
        ylabel="Episode System Profit",
        output_path=config.learning_curve_system_pdf,
    )

    plot_post_eval_bar(
        config,
        eval_df,
        metric_col="EvalHotelRevenue",
        ylabel="Post-Training Evaluation Hotel Revenue",
        output_path=config.eval_bar_hotel_pdf,
    )
    plot_post_eval_bar(
        config,
        eval_df,
        metric_col="EvalOTAProfit",
        ylabel="Post-Training Evaluation OTA Profit",
        output_path=config.eval_bar_ota_pdf,
    )
    plot_post_eval_bar(
        config,
        eval_df,
        metric_col="EvalSystemProfit",
        ylabel="Post-Training Evaluation System Profit",
        output_path=config.eval_bar_system_pdf,
    )

    summary = {
        "mode": config.run_mode,
        "n_training_records": int(len(train_df)),
        "n_eval_records": int(len(eval_df)),
        "algorithms": sorted(set(train_df["Algorithm"].tolist()) | set(eval_df["Algorithm"].tolist())),
        "training_csv": str(config.training_csv_path),
        "evaluation_csv": str(config.evaluation_csv_path),
        "learning_curve_hotel_pdf": str(config.learning_curve_hotel_pdf),
        "learning_curve_ota_pdf": str(config.learning_curve_ota_pdf),
        "learning_curve_system_pdf": str(config.learning_curve_system_pdf),
        "eval_bar_hotel_pdf": str(config.eval_bar_hotel_pdf),
        "eval_bar_ota_pdf": str(config.eval_bar_ota_pdf),
        "eval_bar_system_pdf": str(config.eval_bar_system_pdf),
        "performance_table_hotel_csv": str(config.performance_table_hotel_csv),
        "performance_table_ota_csv": str(config.performance_table_ota_csv),
        "performance_table_system_csv": str(config.performance_table_system_csv),
        "stats_hotel_csv": str(config.stats_hotel_csv_path),
        "stats_ota_csv": str(config.stats_ota_csv_path),
        "stats_system_csv": str(config.stats_system_csv_path),
    }
    with open(config.summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n完成。汇总已写入: {config.summary_json_path}")


if __name__ == "__main__":
    main()
