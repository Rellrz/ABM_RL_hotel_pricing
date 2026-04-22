"""绘图：学习曲线 + 协方差演化。"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns

from config import Experiment2Config


def plot_learning_curves(config: Experiment2Config, df: pd.DataFrame) -> None:
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="Timesteps",
        y="EvalReward",
        hue="Algorithm",
        errorbar=("ci", 95),
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.tight_layout()
    plt.savefig(config.learning_curve_pdf, dpi=300)
    plt.close()


def _ellipse_from_cov(mean_xy, cov, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-8))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle


def plot_covariance_evolution(config: Experiment2Config, cov_df: pd.DataFrame) -> None:
    if cov_df is None or len(cov_df) == 0:
        return
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")

    # 取seed=1的轨迹作为示意图，时间点等间隔抽样
    sub = cov_df[cov_df["Seed"] == cov_df["Seed"].min()].sort_values("Timesteps")
    if len(sub) == 0:
        return
    pick_idx = np.linspace(0, len(sub) - 1, num=min(6, len(sub)), dtype=int)
    picked = sub.iloc[pick_idx]

    plt.figure(figsize=(6, 6))
    colors = sns.color_palette("Blues", n_colors=len(picked))
    mean = np.array([(config.online_price_min + config.online_price_max) / 2.0,
                     (config.offline_price_min + config.offline_price_max) / 2.0], dtype=float)
    for (_, row), color in zip(picked.iterrows(), colors):
        cov = np.array([[row["cov_00"], row["cov_01"]], [row["cov_10"], row["cov_11"]]], dtype=float)
        w, h, ang = _ellipse_from_cov(mean, cov, n_std=2.0)
        ell = Ellipse(xy=mean, width=w, height=h, angle=ang, fill=False, lw=2, color=color, alpha=0.9)
        plt.gca().add_patch(ell)
        plt.text(mean[0] + 0.2, mean[1] + 0.2, f"{int(row['Timesteps']/1000)}k", color=color, fontsize=8)

    plt.xlim(config.online_price_min - 5, config.online_price_max + 5)
    plt.ylim(config.offline_price_min - 5, config.offline_price_max + 5)
    plt.xlabel("Online Base Price")
    plt.ylabel("Offline Price")
    plt.title("Evolution of Covariance Ellipses")
    plt.tight_layout()
    plt.savefig(config.covariance_pdf, dpi=300)
    plt.close()
