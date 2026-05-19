"""项目通用工具（状态离散化、分桶映射、奖励计算等）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def parse_buckets(spec: str, n: int) -> List[Tuple[int, int]]:
    tokens = [t.strip() for t in str(spec).replace(",", "|").split("|") if t.strip()]
    buckets: List[Tuple[int, int]] = []
    for token in tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            s, e = int(a), int(b)
        else:
            s = e = int(token)
        buckets.append((s, e))
    buckets.sort(key=lambda x: x[0])

    if not buckets:
        raise ValueError("decision_buckets cannot be empty")
    if buckets[0][0] != 0 or buckets[-1][1] != n - 1:
        raise ValueError(f"Buckets must cover [0, {n-1}]")

    prev_end = -1
    for s, e in buckets:
        if s != prev_end + 1 or e < s:
            raise ValueError("Buckets must be contiguous and valid")
        prev_end = e
    return buckets


def build_bucket_mapping(buckets: List[Tuple[int, int]], window_days: int) -> Tuple[List[int], List[int], List[int]]:
    bucket_of_offset = [0] * window_days
    for sid, (s, e) in enumerate(buckets):
        for off in range(s, e + 1):
            bucket_of_offset[off] = sid
    entry_offsets = sorted({e for _, e in buckets if 0 <= e < window_days})
    exit_offsets = sorted({s for s, _ in buckets if 0 <= s < window_days})
    return bucket_of_offset, entry_offsets, exit_offsets


N_INVENTORY_LEVELS = 5
N_SEASONS = 3
N_WEEKDAY_TYPES = 2
N_STAGE_BUCKETS = 8
BASE_STATE_COUNT = N_INVENTORY_LEVELS * N_SEASONS * N_WEEKDAY_TYPES
TOTAL_Q_STATES = BASE_STATE_COUNT * N_STAGE_BUCKETS


def season_from_day(day: int) -> int:
    month = (int(day) // 30) % 12 + 1
    if month in (11, 12, 1, 2):
        return 0
    if month in (6, 7, 8):
        return 2
    return 1


def weekday_type_from_day(day: int) -> int:
    return 1 if (int(day) % 7) in (5, 6) else 0


def discretize_inventory_from_raw(
    inventory_raw: float,
    initial_inventory: float,
    n_inventory_levels: int = N_INVENTORY_LEVELS,
) -> int:
    inv = float(inventory_raw)
    init_inv = float(max(1.0, initial_inventory))
    if n_inventory_levels <= 1:
        return 0
    ratio = float(np.clip(inv / init_inv, 0.0, 1.0))
    # 5档默认阈值: 0.2 / 0.4 / 0.6 / 0.8
    if n_inventory_levels == 5:
        if ratio <= 0.2:
            return 0
        if ratio <= 0.4:
            return 1
        if ratio <= 0.6:
            return 2
        if ratio <= 0.8:
            return 3
        return 4
    level = int(np.floor(ratio * n_inventory_levels))
    return int(np.clip(level, 0, n_inventory_levels - 1))


def enrich_bucket_state(
    state: Dict,
    n_inventory_levels: int = N_INVENTORY_LEVELS,
) -> Dict:
    """将环境原始状态补齐为离散策略所需状态字段。"""
    out = dict(state)
    day = int(out.get("day", 0))
    if "season" not in out:
        out["season"] = int(season_from_day(day))
    if "weekday" not in out:
        out["weekday"] = int(weekday_type_from_day(day))
    if "inventory_level" not in out:
        inv_raw = float(out.get("inventory_raw", 0.0))
        init_inv = float(out.get("initial_inventory", max(1.0, inv_raw)))
        out["inventory_level"] = int(
            discretize_inventory_from_raw(
                inventory_raw=inv_raw,
                initial_inventory=init_inv,
                n_inventory_levels=n_inventory_levels,
            )
        )
    return out


def discretize_bucket_state(
    state: Dict,
    stage_id: int,
    n_inventory_levels: int = N_INVENTORY_LEVELS,
    n_seasons: int = N_SEASONS,
    n_weekday_types: int = N_WEEKDAY_TYPES,
    n_stage_buckets: int = N_STAGE_BUCKETS,
) -> int:
    """统一的CEM/Q状态离散函数（库存×季节×周末×bucket）。"""
    norm = enrich_bucket_state(state, n_inventory_levels=n_inventory_levels)
    inv = int(np.clip(int(norm.get("inventory_level", n_inventory_levels - 1)), 0, n_inventory_levels - 1))
    season = int(np.clip(int(norm.get("season", 0)), 0, n_seasons - 1))
    weekday = int(np.clip(int(norm.get("weekday", 0)), 0, n_weekday_types - 1))
    stage_id = int(np.clip(stage_id, 0, n_stage_buckets - 1))
    base_state = inv * (n_seasons * n_weekday_types) + season * n_weekday_types + weekday
    return int(base_state * n_stage_buckets + stage_id)


def state_to_q_state(state: Dict, stage_id: int) -> int:
    return discretize_bucket_state(state, stage_id=stage_id)


def state_to_144(state: Dict, stage_id: int) -> int:
    """Backward-compatible alias. The state space now has 240 states."""
    return discretize_bucket_state(state, stage_id=stage_id)


def compute_bucket_rewards(
    bookings_online: int,
    bookings_offline: int,
    price_online_base: float,
    price_offline: float,
    commission_rate: float,
    subsidy_ratio: float,
    reward_hotel_ratio: float,
) -> Dict[str, float]:
    """统一CEM奖励口径：酒店收益、OTA利润、系统收益与训练奖励。"""
    bo = int(max(0, bookings_online))
    bf = int(max(0, bookings_offline))
    pon = float(price_online_base)
    poff = float(price_offline)
    c = float(commission_rate)
    sr = float(np.clip(subsidy_ratio, 0.0, 1.0))
    r_h = float(np.clip(reward_hotel_ratio, 0.0, 1.0))

    revenue_hotel = bo * pon * (1.0 - c) + bf * poff
    commission_revenue = bo * pon * c
    subsidy_cost = commission_revenue * sr
    profit_ota = commission_revenue - subsidy_cost
    system_profit = revenue_hotel + profit_ota
    reward_hotel = r_h * revenue_hotel + (1.0 - r_h) * system_profit

    return {
        "revenue_hotel": float(revenue_hotel),
        "profit_ota": float(profit_ota),
        "system_profit": float(system_profit),
        "reward_hotel": float(reward_hotel),
        "subsidy_cost": float(subsidy_cost),
        "commission_revenue": float(commission_revenue),
    }


def q_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return float(eps_end)
    ratio = 1.0 - float(step) / float(decay_steps)
    return float(eps_end + (eps_start - eps_end) * ratio)

