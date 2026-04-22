"""实验二公共工具。"""

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


def build_bucket_mapping(buckets: List[Tuple[int, int]], window_days: int) -> Tuple[List[int], List[int]]:
    bucket_of_offset = [0] * window_days
    for sid, (s, e) in enumerate(buckets):
        for off in range(s, e + 1):
            bucket_of_offset[off] = sid
    trigger_offsets = sorted({e for _, e in buckets if 0 <= e < window_days})
    return bucket_of_offset, trigger_offsets


def state_to_144(state: Dict, stage_id: int) -> int:
    inv = int(np.clip(int(state.get("inventory_level", 2)), 0, 2))
    season = int(np.clip(int(state.get("season", 0)), 0, 2))
    weekday = int(np.clip(int(state.get("weekday", 0)), 0, 1))
    stage_id = int(np.clip(stage_id, 0, 7))
    base18 = inv * 6 + season * 2 + weekday
    return int(base18 * 8 + stage_id)


def q_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return float(eps_end)
    ratio = 1.0 - float(step) / float(decay_steps)
    return float(eps_end + (eps_start - eps_end) * ratio)
