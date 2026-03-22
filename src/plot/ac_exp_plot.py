import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.unicode_minus"] = False


def find_latest_capacity_json(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("capacity_to_csv_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"未在 {results_dir} 找到 capacity_to_csv_*.json")
    return candidates[-1]


def load_capacity_map(json_path: Path) -> dict[int, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cap2csv: dict[int, str] = {}
    for k, v in raw.items():
        cap2csv[int(k)] = str(v)
    return dict(sorted(cap2csv.items()))


def load_dfs(cap2csv: dict[int, str]) -> dict[int, pd.DataFrame]:
    dfs: dict[int, pd.DataFrame] = {}
    missing: list[tuple[int, str]] = []
    for cap, csv_path in cap2csv.items():
        p = Path(csv_path)
        if not p.exists():
            missing.append((cap, csv_path))
            continue
        dfs[cap] = pd.read_csv(p).sort_values("episode").reset_index(drop=True)
    if not dfs:
        raise FileNotFoundError("所有 capacity 的 CSV 都不存在，无法绘图。")
    if missing:
        print(f"跳过 {len(missing)} 个缺失文件。")
    return dfs


def plot_4x4_metric(dfs: dict[int, pd.DataFrame], metric: str, ylabel: str, title: str, output_pdf: Path, ma_window: int) -> None:
    caps = sorted(dfs.keys())
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True, sharey=False)
    axes = axes.flatten()
    palette = sns.color_palette("husl", n_colors=max(16, len(caps)))
    for i, cap in enumerate(caps):
        ax = axes[i]
        d = dfs[cap]
        if metric not in d.columns:
            ax.axis("off")
            continue
        x = d["episode"].values
        y = d[metric].astype(float).values
        y_ma = pd.Series(y).rolling(ma_window, min_periods=1).mean().values
        color = palette[i]
        ax.plot(x, y, alpha=0.20, linewidth=1.0, color=color, label="raw")
        ax.plot(x, y_ma, linewidth=2.5, color=color, label=f"MA{ma_window}")
        y_min = np.nanmin(np.r_[y, y_ma])
        y_max = np.nanmax(np.r_[y, y_ma])
        if np.isfinite(y_min) and np.isfinite(y_max):
            if y_max > y_min:
                pad = 0.08 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
            else:
                base = abs(y_max) if y_max != 0 else 1.0
                ax.set_ylim(y_min - 0.1 * base, y_max + 0.1 * base)
        ax.set_title(f"cap={cap}", fontsize=10)
        ax.grid(True, alpha=0.3)
    for j in range(len(caps), nrows * ncols):
        axes[j].axis("off")
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.supxlabel("Episode")
    fig.supylabel(ylabel)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_last_episode_summary(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cap, d in sorted(dfs.items()):
        if "bookings_online" not in d.columns or "bookings_offline" not in d.columns:
            continue
        last = d.sort_values("episode").iloc[-1]
        online = float(last["bookings_online"])
        offline = float(last["bookings_offline"])
        total = online + offline
        rows.append(
            {
                "capacity": int(cap),
                "episode": int(last["episode"]),
                "bookings_online": online,
                "bookings_offline": offline,
                "bookings_total": total,
                "ota_share": online / total if total > 0 else np.nan,
                "direct_share": offline / total if total > 0 else np.nan,
                "occupancy_rate_365": total / (int(cap) * 365.0),
            }
        )
    if not rows:
        raise ValueError("无法构建最后一轮 summary，缺少 bookings_online/offline。")
    return pd.DataFrame(rows).sort_values("capacity").reset_index(drop=True)


def plot_market_share(summary: pd.DataFrame, output_pdf: Path) -> None:
    x = np.arange(len(summary))
    caps = summary["capacity"].astype(int).values
    offline_vals = summary["bookings_offline"].values
    online_vals = summary["bookings_online"].values
    total_vals = summary["bookings_total"].values
    ota_share = summary["ota_share"].values
    fig, ax = plt.subplots(figsize=(11, 5.5))
    color_direct = "#4C78A8"
    color_ota = "#F58518"
    ax.bar(x, offline_vals, width=0.72, color=color_direct, label="Hotel Direct (offline)")
    ax.bar(x, online_vals, width=0.72, bottom=offline_vals, color=color_ota, label="OTA (online)")
    for i in range(len(summary)):
        if total_vals[i] > 0:
            ax.text(i, total_vals[i] * 1.01, f"OTA {ota_share[i]*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, caps)
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Bookings (last episode)")
    ax.set_title("Market Share by Capacity (Last Episode, Stacked)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_occupancy(summary: pd.DataFrame, output_pdf: Path) -> None:
    x = summary["capacity"].values
    y = summary["occupancy_rate_365"].values
    y_ma = pd.Series(y).rolling(3, min_periods=1).mean().values
    fig, ax = plt.subplots(figsize=(9, 5))
    color = "#4C78A8"
    ax.plot(x, y, alpha=0.20, linewidth=1.0, color=color, label="raw")
    ax.plot(x, y_ma, linewidth=2.5, color=color, label="MA3")
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="gray", alpha=0.8, label="100% reference")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.01, f"{yi*100:.1f}%", ha="center", va="bottom", fontsize=8)
    y_min = np.nanmin(np.r_[y, y_ma])
    y_max = np.nanmax(np.r_[y, y_ma])
    if y_max > y_min:
        pad = 0.08 * (y_max - y_min)
        ax.set_ylim(max(0, y_min - pad), y_max + pad)
    else:
        base = abs(y_max) if y_max != 0 else 1.0
        ax.set_ylim(max(0, y_min - 0.1 * base), y_max + 0.1 * base)
    ax.set_xlabel("Capacity")
    ax.set_ylabel("Occupancy Rate")
    ax.set_title("Occupancy Rate vs Capacity (Last Episode, 365-denominator)")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-json", type=str, default="", help="capacity_to_csv_*.json 路径；为空时自动选最新")
    parser.add_argument("--output-dir", type=str, default="outputs/figures/experiment2", help="输出目录")
    parser.add_argument("--ma-window", type=int, default=10, help="Episode 曲线移动平均窗口")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.capacity_json:
        capacity_json = Path(args.capacity_json)
    else:
        candidates = []
        for d in [root / "outputs" / "results", root / "outputs" / "results" / "experiment2"]:
            if d.exists():
                candidates.extend(d.glob("capacity_to_csv_*.json"))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"未找到 capacity_to_csv_*.json，请用 --capacity-json 显式指定。")
        capacity_json = candidates[-1]
    if not capacity_json.is_absolute():
        capacity_json = (root / capacity_json).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    set_style()
    cap2csv = load_capacity_map(capacity_json)
    dfs = load_dfs(cap2csv)

    plot_4x4_metric(
        dfs=dfs,
        metric="hotel_revenue",
        ylabel="Hotel Revenue",
        title="Hotel Revenue by Capacity (4x4)",
        output_pdf=output_dir / "hotel_revenue_4x4.pdf",
        ma_window=args.ma_window,
    )
    plot_4x4_metric(
        dfs=dfs,
        metric="ota_profit",
        ylabel="OTA Profit",
        title="OTA Profit by Capacity (4x4)",
        output_pdf=output_dir / "ota_profit_4x4.pdf",
        ma_window=args.ma_window,
    )
    plot_4x4_metric(
        dfs=dfs,
        metric="bookings_online",
        ylabel="Bookings Online",
        title="Bookings Online by Capacity (4x4)",
        output_pdf=output_dir / "bookings_online_4x4.pdf",
        ma_window=args.ma_window,
    )
    plot_4x4_metric(
        dfs=dfs,
        metric="avg_subsidy_ratio",
        ylabel="Avg Subsidy Ratio",
        title="Avg Subsidy Ratio by Capacity (4x4)",
        output_pdf=output_dir / "avg_subsidy_ratio_4x4.pdf",
        ma_window=args.ma_window,
    )

    summary = build_last_episode_summary(dfs)
    plot_market_share(summary, output_dir / "market_share_by_capacity_last_episode.pdf")
    plot_occupancy(summary, output_dir / "occupancy_rate_vs_capacity_last_episode.pdf")
    summary.to_csv(output_dir / "last_episode_summary.csv", index=False)

    print(f"capacity json: {capacity_json}")
    print(f"output dir: {output_dir}")
    print("saved pdf files:")
    for p in sorted(output_dir.glob("*.pdf")):
        print(p)


if __name__ == "__main__":
    main()
