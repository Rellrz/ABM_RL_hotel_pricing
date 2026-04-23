"""绘图：训练期 episode 收益曲线 + 训练后评估柱状图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import Experiment2Config


def plot_learning_curves(config: Experiment2Config, training_df: pd.DataFrame) -> None:
    if training_df is None or len(training_df) == 0:
        return
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=training_df,
        x="Episode",
        y="EpisodeRevenue",
        hue="Algorithm",
        errorbar=("ci", 95),
    )
    plt.xlabel("Episode")
    plt.ylabel("Episode Revenue")
    plt.tight_layout()
    plt.savefig(config.learning_curve_pdf, dpi=300)
    plt.close()


def plot_post_eval_bar(config: Experiment2Config, eval_df: pd.DataFrame) -> None:
    if eval_df is None or len(eval_df) == 0:
        return
    per_seed = (
        eval_df.groupby(["Algorithm", "Seed"], as_index=False)["EvalRevenue"]
        .mean()
        .rename(columns={"EvalRevenue": "SeedMeanEvalRevenue"})
    )
    if len(per_seed) == 0:
        return
    agg = (
        per_seed.groupby("Algorithm")["SeedMeanEvalRevenue"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "MeanEvalRevenue", "std": "StdEvalRevenue", "count": "N"})
    )
    agg["StdEvalRevenue"] = agg["StdEvalRevenue"].fillna(0.0)
    agg["ErrorBar95CI"] = 1.96 * agg["StdEvalRevenue"] / np.sqrt(np.maximum(agg["N"], 1))

    plt.figure(figsize=(8, 5))
    x = np.arange(len(agg))
    plt.bar(x, agg["MeanEvalRevenue"].values, yerr=agg["ErrorBar95CI"].values, capsize=4, alpha=0.85)
    plt.xticks(x, agg["Algorithm"].values, rotation=15, ha="right")
    plt.ylabel("Post-Training Evaluation Revenue")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.savefig(config.eval_bar_pdf, dpi=300)
    plt.close()
