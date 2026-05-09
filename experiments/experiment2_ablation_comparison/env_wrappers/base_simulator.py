"""实验二共用仿真内核：严格对齐 game_trainer 的分桶训练机制。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from common import build_bucket_mapping, parse_buckets, state_to_144
from config import Experiment2Config
from src.agent.ota_agent import OTASubsidyHeuristic
from src.environment.hotel_env import HotelEnvironment


@dataclass
class DayResult:
    reward_hotel: float
    reward_ota: float
    done: bool
    info: Dict


@dataclass
class UpdateEvent:
    state: Dict
    action_pair: Tuple[float, float]
    reward: float
    next_state: Dict
    done: bool
    ota_subsidy: float


class BucketPricingSimulator:
    """按分桶价格驱动的一天仿真。

    约定输入动作语义：
    - 每个bucket给出 `(p_online_base, p_offline)`；
    - 线上最终价由 OTA 补贴规则调整后得到。
    """

    def __init__(self, config: Experiment2Config, seed: int, historical_data):
        self.config = config
        self.env = HotelEnvironment(
            initial_inventory=config.initial_inventory,
            historical_data=historical_data,
            booking_window_days=config.booking_window_days,
            episode_days=config.days_per_episode,
        )
        self.ota = OTASubsidyHeuristic(
            commission_rate=config.commission_rate,
            r_max=config.ota_r_max,
            delta_max=config.ota_delta_max,
            decay_lambda=config.ota_decay_lambda,
            noise_std=config.ota_noise_std,
            seed=config.ota_seed if config.ota_seed >= 0 else seed,
        )
        self.buckets = parse_buckets(config.decision_buckets, config.booking_window_days)
        self.bucket_of_offset, self.entry_offsets, self.exit_offsets = build_bucket_mapping(
            self.buckets, config.booking_window_days
        )
        self.day = 0
        self.initialized = False

        self.price_online_base_by_offset: List[float] = []
        self.price_offline_by_offset: List[float] = []
        self.subsidy_ratio_by_offset: List[float] = []
        self.decision_state_by_offset: List[Optional[Dict]] = []
        self.acc_bookings_online_by_offset: List[int] = []
        self.acc_bookings_offline_by_offset: List[int] = []

    @property
    def n_stages(self) -> int:
        return len(self.buckets)

    def reset(self) -> Dict:
        self.day = 0
        self.initialized = False
        self.price_online_base_by_offset = [0.0] * self.config.booking_window_days
        self.price_offline_by_offset = [0.0] * self.config.booking_window_days
        self.subsidy_ratio_by_offset = [0.0] * self.config.booking_window_days
        self.decision_state_by_offset = [None] * self.config.booking_window_days
        self.acc_bookings_online_by_offset = [0] * self.config.booking_window_days
        self.acc_bookings_offline_by_offset = [0] * self.config.booking_window_days
        return self.env.reset()

    def get_state_by_stage(self, stage_id: int) -> Dict:
        _s, e = self.buckets[stage_id]
        ref_off = min(e, self.config.booking_window_days - 1)
        st = dict(self.env._get_state_for_day_offset(ref_off))
        st["stage_id"] = int(stage_id)
        return st

    def get_q_state_by_stage(self, stage_id: int) -> int:
        return state_to_144(self.get_state_by_stage(stage_id), stage_id=stage_id)

    def get_obs_vector_for_ppo(self) -> np.ndarray:
        st = self.env._get_state()
        future_inventory = np.asarray(st.get("future_inventory", [self.config.initial_inventory] * self.config.booking_window_days), dtype=np.float64)
        remaining_inventory = float(st.get("inventory_raw", self.config.initial_inventory))
        month = int(((self.day // 30) % 12) + 1)
        month_onehot = np.zeros(12, dtype=np.float64)
        month_onehot[month - 1] = 1.0
        weekend = float(st.get("weekday", 0))
        day_norm = float(self.day % self.config.days_per_episode) / float(max(1, self.config.days_per_episode - 1))

        vec = np.concatenate(
            [
                future_inventory,
                np.array([remaining_inventory], dtype=np.float64),
                month_onehot,
                np.array([weekend, day_norm], dtype=np.float64),
            ]
        )
        return vec.astype(np.float32)

    def _price_clipped(self, action_pair: Tuple[float, float]) -> Tuple[float, float]:
        pon = float(np.clip(action_pair[0], self.config.online_price_min, self.config.online_price_max))
        poff = float(np.clip(action_pair[1], self.config.offline_price_min, self.config.offline_price_max))
        return pon, poff

    def initialize_episode_decisions(self, stage_actions: List[Tuple[float, float]]) -> None:
        """对齐 game_trainer: episode开始时先按每个bucket初始化全窗口决策。"""
        if len(stage_actions) != self.n_stages:
            raise ValueError(f"Expected {self.n_stages} stage actions, got {len(stage_actions)}")
        for sid, (_s, e) in enumerate(self.buckets):
            ref_off = int(min(e, self.config.booking_window_days - 1))
            st = dict(self.env._get_state_for_day_offset(ref_off))
            st["stage_id"] = int(sid)
            pon, poff = self._price_clipped(stage_actions[sid])
            sr = float(self.ota.get_subsidy(pon, poff, lead_time=ref_off))
            for off in range(int(_s), min(int(e) + 1, self.config.booking_window_days)):
                self.price_online_base_by_offset[off] = pon
                self.price_offline_by_offset[off] = poff
                self.subsidy_ratio_by_offset[off] = sr
                self.decision_state_by_offset[off] = dict(st)
        self.initialized = True

    def _build_update_event(self, off: int, done_flag: bool) -> Optional[UpdateEvent]:
        bo_acc = int(self.acc_bookings_online_by_offset[off])
        bf_acc = int(self.acc_bookings_offline_by_offset[off])
        if (bo_acc <= 0 and bf_acc <= 0) or self.decision_state_by_offset[off] is None:
            return None

        pon = float(self.price_online_base_by_offset[off])
        poff = float(self.price_offline_by_offset[off])
        sr = float(self.subsidy_ratio_by_offset[off])

        revenue_hotel = bo_acc * pon * (1.0 - self.config.commission_rate) + bf_acc * poff
        commission_rev = bo_acc * pon * self.config.commission_rate
        subsidy_cost = commission_rev * sr
        profit_ota = commission_rev - subsidy_cost
        total_system_profit = revenue_hotel + profit_ota
        reward_hotel = (
            self.config.reward_hotel_ratio * revenue_hotel
            + (1.0 - self.config.reward_hotel_ratio) * total_system_profit
        )

        state_for_update = dict(self.decision_state_by_offset[off])
        next_state_for_update = dict(self.env._get_state_for_day_offset(off))
        next_state_for_update["stage_id"] = int(self.bucket_of_offset[off])
        return UpdateEvent(
            state=state_for_update,
            action_pair=(pon, poff),
            reward=float(reward_hotel),
            next_state=next_state_for_update,
            done=bool(done_flag),
            ota_subsidy=float(subsidy_cost),
        )

    def _rotate_offsets(self) -> None:
        self.price_online_base_by_offset = self.price_online_base_by_offset[1:] + [self.price_online_base_by_offset[-1]]
        self.price_offline_by_offset = self.price_offline_by_offset[1:] + [self.price_offline_by_offset[-1]]
        self.subsidy_ratio_by_offset = self.subsidy_ratio_by_offset[1:] + [self.subsidy_ratio_by_offset[-1]]
        self.decision_state_by_offset = self.decision_state_by_offset[1:] + [self.decision_state_by_offset[-1]]
        self.acc_bookings_online_by_offset = self.acc_bookings_online_by_offset[1:] + [self.acc_bookings_online_by_offset[-1]]
        self.acc_bookings_offline_by_offset = self.acc_bookings_offline_by_offset[1:] + [self.acc_bookings_offline_by_offset[-1]]

    def step_day(self, stage_actions: List[Tuple[float, float]]) -> DayResult:
        if len(stage_actions) != self.n_stages:
            raise ValueError(f"Expected {self.n_stages} stage actions, got {len(stage_actions)}")

        if not self.initialized:
            self.initialize_episode_decisions(stage_actions)

        update_events: List[UpdateEvent] = []

        # 在桶的右端点为新进入该桶的cohort重新定价。
        for off in self.entry_offsets:
            sid = int(self.bucket_of_offset[off])
            st = dict(self.env._get_state_for_day_offset(off))
            st["stage_id"] = sid
            pon, poff = self._price_clipped(stage_actions[sid])
            sr = float(self.ota.get_subsidy(pon, poff, lead_time=off))
            self.acc_bookings_online_by_offset[off] = 0
            self.acc_bookings_offline_by_offset[off] = 0
            self.price_online_base_by_offset[off] = pon
            self.price_offline_by_offset[off] = poff
            self.subsidy_ratio_by_offset[off] = sr
            self.decision_state_by_offset[off] = dict(st)

        final_online = [
            self.price_online_base_by_offset[i]
            - self.price_online_base_by_offset[i] * self.config.commission_rate * self.subsidy_ratio_by_offset[i]
            for i in range(self.config.booking_window_days)
        ]
        actions_window = [[final_online[i], self.price_offline_by_offset[i]] for i in range(self.config.booking_window_days)]

        _, _, done, info = self.env.step(actions_window)
        self.day += 1

        bookings = info.get("bookings_by_day_offset", [])
        reward_hotel = 0.0
        reward_ota = 0.0
        by_stage_hotel = [0.0] * self.n_stages
        by_stage_ota = [0.0] * self.n_stages
        for off in range(min(len(bookings), self.config.booking_window_days)):
            bo = int(bookings[off]["bookings_online"])
            bf = int(bookings[off]["bookings_offline"])
            if bo == 0 and bf == 0:
                continue
            sid = int(self.bucket_of_offset[off])
            p_on_base = float(self.price_online_base_by_offset[off])
            p_off = float(self.price_offline_by_offset[off])
            sr = float(self.subsidy_ratio_by_offset[off])
            hotel_gain = bo * p_on_base * (1.0 - self.config.commission_rate) + bf * p_off
            ota_gain = self.ota.calculate_profit(bo, p_on_base, sr)
            reward_hotel += hotel_gain
            reward_ota += ota_gain
            by_stage_hotel[sid] += hotel_gain
            by_stage_ota[sid] += ota_gain
            self.acc_bookings_online_by_offset[off] += int(bo)
            self.acc_bookings_offline_by_offset[off] += int(bf)

        done = bool(done or self.day >= self.config.days_per_episode)

        # 在桶的左端点结算该cohort完整经历该桶后的累计收益。
        for off in self.exit_offsets:
            ev = self._build_update_event(off, done_flag=done)
            if ev is not None:
                update_events.append(ev)
            self.acc_bookings_online_by_offset[off] = 0
            self.acc_bookings_offline_by_offset[off] = 0
            self.decision_state_by_offset[off] = None

        if done:
            # 对齐 game_trainer：episode结束后flush全部offset累计收益（done=True）
            for off in range(self.config.booking_window_days):
                ev = self._build_update_event(off, done_flag=True)
                if ev is not None:
                    update_events.append(ev)
                self.acc_bookings_online_by_offset[off] = 0
                self.acc_bookings_offline_by_offset[off] = 0
        else:
            self._rotate_offsets()

        info = dict(info)
        info["reward_hotel_by_stage"] = by_stage_hotel
        info["reward_ota_by_stage"] = by_stage_ota
        info["update_events"] = update_events
        return DayResult(
            reward_hotel=float(reward_hotel),
            reward_ota=float(reward_ota),
            done=done,
            info=info,
        )
