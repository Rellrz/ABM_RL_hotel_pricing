"""显著性检验与性能汇总。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def holm_correction(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, (m - rank) * p_values[idx])
    # 保证单调性
    for i in range(1, m):
        prev_idx = order[i - 1]
        curr_idx = order[i]
        adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx])
    return adjusted.tolist()


def build_performance_table(training_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or len(eval_df) == 0:
        return pd.DataFrame()

    per_seed_eval = (
        eval_df.groupby(["Algorithm", "Seed"], as_index=False)["EvalRevenue"]
        .mean()
        .rename(columns={"EvalRevenue": "SeedMeanEvalRevenue"})
    )
    rows: List[Dict] = []
    for algo in sorted(per_seed_eval["Algorithm"].unique()):
        sub_eval = per_seed_eval[per_seed_eval["Algorithm"] == algo].copy()
        mean_eval = float(sub_eval["SeedMeanEvalRevenue"].mean())

        # 90% own max 收敛轮次：按每个seed自身训练收益定义
        conv_steps = []
        sub_train = training_df[training_df["Algorithm"] == algo].copy()
        for _, seed_df in sub_train.groupby("Seed"):
            seed_df = seed_df.sort_values("Episode")
            target = 0.9 * float(seed_df["EpisodeRevenue"].max())
            hit = seed_df[seed_df["EpisodeRevenue"] >= target]
            if len(hit) == 0:
                conv_steps.append(np.nan)
            else:
                conv_steps.append(float(hit.iloc[0]["Episode"]))
        mean_conv = float(np.nanmean(conv_steps))

        if algo == "Multivariate CEM":
            params = "144 x (2+3) = 720"
        elif algo == "Independent CEM":
            params = "144 x (2+2) = 576"
        elif algo == "Q-learning":
            params = "144 x 100 = 14,400"
        else:
            params = "Neural Net (auto-count)"

        rows.append(
            {
                "Algorithm": algo,
                "Mean Post-Eval Revenue (Seed Avg)": mean_eval,
                "Convergence Episode (90% Own Max)": mean_conv,
                "Parameter Count (Complexity)": params,
            }
        )
    return pd.DataFrame(rows)


def significance_tests(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df is None or len(eval_df) == 0:
        return pd.DataFrame()
    try:
        from scipy.stats import mannwhitneyu, ttest_ind
    except Exception:
        return pd.DataFrame(
            [{"note": "scipy not installed, significance tests skipped"}]
        )

    # 用每个seed的后评估均值做算法间比较
    per_seed_mean = (
        eval_df.groupby(["Algorithm", "Seed"])["EvalRevenue"]
        .mean()
        .reset_index()
    )
    ours = per_seed_mean[per_seed_mean["Algorithm"] == "Multivariate CEM"]["EvalRevenue"].values
    rows = []
    pvals = []
    tmp_rows = []
    for algo in sorted(per_seed_mean["Algorithm"].unique()):
        if algo == "Multivariate CEM":
            continue
        other = per_seed_mean[per_seed_mean["Algorithm"] == algo]["EvalRevenue"].values
        t_stat, t_p = ttest_ind(ours, other, equal_var=False)
        u_stat, u_p = mannwhitneyu(ours, other, alternative="two-sided")
        effect = float(np.mean(ours) - np.mean(other))
        tmp_rows.append(
            {
                "Compare": f"Multivariate CEM vs {algo}",
                "Welch_t_stat": float(t_stat),
                "Welch_p": float(t_p),
                "MWU_stat": float(u_stat),
                "MWU_p": float(u_p),
                "MeanDiff": effect,
            }
        )
        pvals.append(float(u_p))
    if pvals:
        holm = holm_correction(pvals)
        for row, p_adj in zip(tmp_rows, holm):
            row["MWU_p_holm"] = float(p_adj)
            rows.append(row)
    return pd.DataFrame(rows)
