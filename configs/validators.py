#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .schema import EnvConfig, RLConfig


def validate_config(rl_config: RLConfig, env_config: EnvConfig) -> bool:
    if rl_config.cem_algorithm not in ("cem", "cem_nn"):
        print("错误：cem_algorithm必须为'cem'或'cem_nn'")
        return False

    if not (0.0 <= rl_config.cem_elite_frac <= 1.0):
        print("错误：cem_elite_frac必须在0和1之间")
        return False

    if rl_config.cem_n_samples <= 0:
        print("错误：cem_n_samples必须大于0")
        return False

    if rl_config.initial_std <= 0 or rl_config.min_std <= 0:
        print("错误：initial_std和min_std必须大于0")
        return False

    if rl_config.commission_rate < 0 or rl_config.commission_rate > 1:
        print("错误：commission_rate必须在0和1之间")
        return False

    if rl_config.subsidy_ratio_min < 0 or rl_config.subsidy_ratio_max > 1:
        print("错误：补贴比例必须在0和1之间")
        return False

    if rl_config.subsidy_ratio_min > rl_config.subsidy_ratio_max:
        print("错误：subsidy_ratio_min不能大于subsidy_ratio_max")
        return False

    if rl_config.ota_delta_max < 0:
        print("错误：ota_delta_max不能小于0")
        return False

    if rl_config.ota_decay_lambda < 0:
        print("错误：ota_decay_lambda不能小于0")
        return False

    if rl_config.ota_noise_std < 0:
        print("错误：ota_noise_std不能小于0")
        return False

    if rl_config.online_price_min <= 0 or rl_config.offline_price_min <= 0:
        print("错误：最低价格必须大于0")
        return False

    if rl_config.online_price_min > rl_config.online_price_max:
        print("错误：online_price_min不能大于online_price_max")
        return False

    if rl_config.offline_price_min > rl_config.offline_price_max:
        print("错误：offline_price_min不能大于offline_price_max")
        return False

    if env_config.initial_inventory <= 0:
        print("错误：初始库存必须大于0")
        return False

    if env_config.booking_window_days <= 0:
        print("错误：booking_window_days必须大于0")
        return False

    return True
