#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用单一参数敏感性实验。

示例：
conda run -n abm python experiments/run_single_param_sensitivity.py \
  --param RL_CONFIG.cem_alpha \
  --values 0.1,0.2,0.3 \
  --episodes 100 \
  --tail-window 30 \
  --n-jobs 3
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import ABM_CONFIG, ENV_CONFIG, PATH_CONFIG, RANDOM_CONFIG, RL_CONFIG
from src.training.game_trainer import train_game_system


CONFIG_REGISTRY = {
    "RL_CONFIG": RL_CONFIG,
    "ABM_CONFIG": ABM_CONFIG,
    "ENV_CONFIG": ENV_CONFIG,
    "RANDOM_CONFIG": RANDOM_CONFIG,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用单一参数敏感性实验")
    parser.add_argument("--data", type=str, default="datasets/hotel_bookings.csv", help="酒店预订数据文件路径")
    parser.add_argument("--param", type=str, required=True, help="目标参数，如 RL_CONFIG.cem_alpha")
    parser.add_argument("--values", type=str, required=True, help="逗号分隔的候选值，如 0.1,0.2,0.3 或 1,2,3,5")
    parser.add_argument("--episodes", type=int, default=100, help="训练轮数")
    parser.add_argument("--episode-days", type=int, default=730, help="每个 episode 的模拟天数")
    parser.add_argument("--mode", type=str, default="simultaneous", choices=["fixed_ota", "alternating", "simultaneous"], help="训练模式")
    parser.add_argument("--update-frequency", type=int, default=30, help="CEM 参数更新频率")
    parser.add_argument("--booking-window-days", type=int, default=91, help="预订窗口长度")
    parser.add_argument("--decision-buckets", type=str, default="0|1|2-3|4-6|7-13|14-29|30-59|60-90", help="提前期分桶")
    parser.add_argument("--tail-window", type=int, default=30, help="收敛段统计窗口长度")
    parser.add_argument("--env-seed", type=int, default=42, help="固定环境随机种子")
    parser.add_argument("--n-jobs", type=int, default=4, help="并行进程数")
    parser.add_argument("--save-models", action="store_true", help="是否保存每个参数值对应的模型")
    return parser


def _tail_mean(series: pd.Series, window: int) -> float:
    n = max(1, min(len(series), int(window)))
    return float(series.tail(n).mean())


def _tail_std(series: pd.Series, window: int) -> float:
    n = max(1, min(len(series), int(window)))
    return float(series.tail(n).std(ddof=0))


def _parse_target(param_expr: str) -> tuple[str, str]:
    if "." not in str(param_expr):
        raise ValueError(f"参数写法错误，应为 CONFIG_NAME.field，收到: {param_expr}")
    config_name, field_name = str(param_expr).split(".", 1)
    config_name = config_name.strip()
    field_name = field_name.strip()
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"不支持的配置对象: {config_name}，可选: {sorted(CONFIG_REGISTRY)}")
    if not hasattr(CONFIG_REGISTRY[config_name], field_name):
        raise ValueError(f"{config_name} 中不存在字段: {field_name}")
    return config_name, field_name


def _coerce_single_value(raw_value: str, template_value: Any) -> Any:
    raw = str(raw_value).strip()
    if isinstance(template_value, bool):
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"无法解析布尔值: {raw_value}")
    if isinstance(template_value, int) and not isinstance(template_value, bool):
        return int(raw)
    if isinstance(template_value, float):
        return float(raw)
    if template_value is None:
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


def _parse_values(values_expr: str, template_value: Any) -> list[Any]:
    values = [token.strip() for token in str(values_expr).split(",") if token.strip()]
    if not values:
        raise ValueError("values 不能为空")
    return [_coerce_single_value(token, template_value) for token in values]


def _run_single_value(
    config_name: str,
    field_name: str,
    candidate_value: Any,
    args_dict: dict,
) -> tuple[dict, pd.DataFrame]:
    historical_data = pd.read_csv(args_dict["data"])
    historical_data = historical_data[historical_data["hotel"] == "City Hotel"].copy()

    old_random_mode = RANDOM_CONFIG.random_mode
    old_fixed_seed = RANDOM_CONFIG.fixed_seed
    target_config = CONFIG_REGISTRY[config_name]
    old_value = getattr(target_config, field_name)

    try:
        RANDOM_CONFIG.random_mode = "fixed"
        RANDOM_CONFIG.fixed_seed = int(args_dict["env_seed"])
        setattr(target_config, field_name, candidate_value)
        np.random.seed(int(args_dict["env_seed"]))

        hotel_agent, _, _, _, episode_info = train_game_system(
            historical_data=historical_data,
            episodes=int(args_dict["episodes"]),
            training_mode=str(args_dict["mode"]),
            update_frequency=int(args_dict["update_frequency"]),
            booking_window_days=int(args_dict["booking_window_days"]),
            decision_buckets=str(args_dict["decision_buckets"]),
            episode_days=int(args_dict["episode_days"]),
        )

        model_path = ""
        if bool(args_dict.get("save_models", False)) and getattr(hotel_agent, "cem_joint", None) is not None:
            safe_value = str(candidate_value).replace(".", "p").replace("/", "_")
            model_path = str(hotel_agent.cem_joint.save_model(f"hotel_joint_{field_name}_{safe_value}"))

        df = pd.DataFrame(episode_info)
        df["param_name"] = f"{config_name}.{field_name}"
        df["param_value"] = str(candidate_value)

        summary_row = {
            "param_name": f"{config_name}.{field_name}",
            "param_value": str(candidate_value),
            "episodes": int(args_dict["episodes"]),
            "tail_window": int(args_dict["tail_window"]),
            "hotel_last": float(df["hotel_revenue"].iloc[-1]),
            "hotel_best": float(df["hotel_revenue"].max()),
            "hotel_tail_mean": _tail_mean(df["hotel_revenue"], int(args_dict["tail_window"])),
            "hotel_tail_std": _tail_std(df["hotel_revenue"], int(args_dict["tail_window"])),
            "ota_tail_mean": _tail_mean(df["ota_profit"], int(args_dict["tail_window"])),
            "system_tail_mean": _tail_mean(df["hotel_revenue"] + df["ota_profit"], int(args_dict["tail_window"])),
            "online_tail_mean": _tail_mean(df["bookings_online"], int(args_dict["tail_window"])),
            "offline_tail_mean": _tail_mean(df["bookings_offline"], int(args_dict["tail_window"])),
            "subsidy_tail_mean": _tail_mean(df["total_subsidy"], int(args_dict["tail_window"])),
            "model_path": model_path,
        }
        return summary_row, df
    finally:
        setattr(target_config, field_name, old_value)
        RANDOM_CONFIG.random_mode = old_random_mode
        RANDOM_CONFIG.fixed_seed = old_fixed_seed


def main() -> None:
    args = build_parser().parse_args()
    config_name, field_name = _parse_target(args.param)
    template_value = getattr(CONFIG_REGISTRY[config_name], field_name)
    candidate_values = _parse_values(args.values, template_value)

    summary_rows: list[dict] = []
    episode_frames: list[pd.DataFrame] = []
    max_workers = max(1, min(int(args.n_jobs), len(candidate_values)))
    args_dict = vars(args).copy()

    print("=" * 72)
    print("单一参数敏感性实验")
    print("=" * 72)
    print(f"目标参数: {config_name}.{field_name}")
    print(f"候选值: {candidate_values}")
    print(f"环境种子: {args.env_seed}")
    print(f"训练轮数: {args.episodes}")
    print(f"并行进程数: {max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_single_value, config_name, field_name, candidate_value, args_dict): candidate_value
            for candidate_value in candidate_values
        }
        for fut in as_completed(futures):
            candidate_value = futures[fut]
            print("\n" + "-" * 72)
            print(f"完成参数值: {candidate_value}")
            print("-" * 72)
            summary_row, df = fut.result()
            summary_rows.append(summary_row)
            episode_frames.append(df)

    summary_df = pd.DataFrame(summary_rows).sort_values("hotel_tail_mean", ascending=False).reset_index(drop=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_stub = f"{config_name}_{field_name}".replace(".", "_")
    summary_path = os.path.join(PATH_CONFIG.results_dir, f"single_param_sensitivity_summary_{param_stub}_{timestamp}.csv")
    detail_path = os.path.join(PATH_CONFIG.results_dir, f"single_param_sensitivity_episodes_{param_stub}_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    episode_df.to_csv(detail_path, index=False)

    print("\n" + "=" * 72)
    print("实验完成")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"汇总结果已保存到: {summary_path}")
    print(f"逐轮结果已保存到: {detail_path}")


if __name__ == "__main__":
    main()
