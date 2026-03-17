"""
Configuration dataclasses for gas price prediction model.

Centralizes magic numbers and tunable parameters for:
- Model simulation settings
- Trading parameters (Kelly, edge thresholds)
- Forecast variance adaptation
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Training data configuration
# Daily data starts Dec 2025; weekly data goes back to 1990.
# For AR(1) training, use daily data only (most recent period).
DAILY_DATA_START = "2025-12-01"

# Gas prices don't have a COVID-style gap, but we define a minimum
# training start for the daily model to avoid the weekly→daily transition.
TRAIN_START_DATE = "2025-12-01"


# Global sigma multiplier — corrects AR(1) sigma under-estimation.
# AR(1) sigma is estimated from ~103 daily observations (Dec 2025-Mar 2026),
# which produces an 8-day std ~4x tighter than the empirical Monday-to-Monday
# std from 1,850 weekly observations (1990-2026). Value calibrated from
# Part 0 historical sanity check (ratio = 0.26, multiplier = 1/0.26 ≈ 3.8).
GLOBAL_SIGMA_MULTIPLIER: float = 3.8

# Day-of-week variance ratios — placeholder, calibrate from backtest.
# Gas prices may have different DOW variance patterns than TSA.
DOW_VARIANCE_RATIOS: dict[int, float] = {
    0: 1.0,  # Monday
    1: 1.0,  # Tuesday
    2: 1.0,  # Wednesday
    3: 1.0,  # Thursday
    4: 1.0,  # Friday
    5: 1.0,  # Saturday
    6: 1.0,  # Sunday
}


# Seasonal variance multipliers — placeholder, calibrate from backtest.
SEASONAL_VARIANCE_MULTIPLIERS: dict[int, float] = {
    1: 1.0,   # Jan
    2: 1.0,   # Feb
    3: 1.0,   # Mar
    4: 1.0,   # Apr
    5: 1.0,   # May
    6: 1.0,   # Jun
    7: 1.0,   # Jul
    8: 1.0,   # Aug
    9: 1.0,   # Sep
    10: 1.0,  # Oct
    11: 1.0,  # Nov
    12: 1.0,  # Dec
}


# Conditioning variance scaler — how much to shrink simulation spread
# based on number of remaining sim days (7 - observed_days).
CONDITIONING_VARIANCE_SCALER: dict[int, float] = {
    7: 1.00,   # Day 0: 7 sim days
    6: 1.00,   # Day 1: 6 sim days
    5: 1.00,   # Day 2: 5 sim days
    4: 1.00,   # Day 3: 4 sim days
    3: 1.00,   # Day 4: 3 sim days
    2: 1.00,   # Day 5: 2 sim days
    1: 1.00,   # Day 6: 1 sim day
}


# Mean bias correction per observed day — placeholder, calibrate from backtest.
MEAN_BIAS_BY_DAY: dict[int, float] = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
}


def get_mean_bias_for_day(observed_days: int) -> float:
    """Get mean bias correction Z-score for given observed days."""
    return MEAN_BIAS_BY_DAY.get(observed_days, 0.0)


@dataclass(frozen=True)
class ModelConfig:
    """Model simulation and calibration settings."""
    n_sims: int = 25_000
    scaler_s: float = 1.0  # Probability scaling (1.0 = no stretching)


@dataclass(frozen=True)
class TradingConfig:
    """Trading and position sizing parameters."""
    bankroll: float = 700.0


@dataclass(frozen=True)
class MakerConfig:
    """Market making order submission configuration."""
    min_ev_cents: float = 2.0
    use_demo_api: bool = False
    post_only: bool = True
    cluster_cap_by_day: tuple = (0.50, 0.50, 1.0, 1.0, 2.0, 2.0, 2.0)


@dataclass(frozen=True)
class MakerDefensiveConfig:
    """Conservative maker gates."""
    min_prob: float = 0.75
    min_prob_by_day: tuple = (0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75)


@dataclass(frozen=True)
class MakerExitConfig:
    """Configuration for automatic maker exit (sell) orders."""
    min_exit_ev_c: float = 0.5
    enabled: bool = True


@dataclass(frozen=True)
class LiveMakerConfig:
    """Configuration for the real-time maker trading loop."""
    fast_tick_s: float = 5.0
    fill_check_s: float = 5.0
    full_cycle_s: float = 30.0
    mid_velocity_threshold_c: int = 10
    mid_velocity_window_s: int = 30
    mid_velocity_min_tickers: int = 2
    spread_blowout_mult: float = 3.0
    spread_blowout_floor_c: int = 10
    spread_trailing_window_s: int = 300
    fill_rate_max: int = 3
    fill_rate_window_s: int = 30
    min_cash_balance_c: int = 20000
    max_consecutive_api_errors: int = 5


@dataclass(frozen=True)
class WeeklyTradingConfig:
    """Intra-week dynamic trading configuration."""
    early_week_max_deployment: float = 0.60
    full_deployment_day: int = 3
    round_trip_fee_pct: float = 0.14
    reinvest_buffer_cents: float = 0.5
    min_remaining_edge_pp: float = 2.0
    take_profit_buffer_c: float = 1.0
    edge_gone_epsilon_c: float = 3.0


# Default instances
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_MAKER_CONFIG = MakerConfig()
DEFAULT_MAKER_DEFENSIVE_CONFIG = MakerDefensiveConfig()
DEFAULT_MAKER_EXIT_CONFIG = MakerExitConfig()
DEFAULT_LIVE_MAKER_CONFIG = LiveMakerConfig()
DEFAULT_WEEKLY_TRADING_CONFIG = WeeklyTradingConfig()

# No temporary blocklists for gas (no holiday blocking needed initially)
TEMP_BLOCKED_THRESHOLDS: tuple[str, ...] = ()


def get_maker_min_prob_for_day(weekday: int, config: MakerDefensiveConfig | None = None) -> float:
    if config is None:
        config = DEFAULT_MAKER_DEFENSIVE_CONFIG
    if 0 <= weekday <= 6:
        return config.min_prob_by_day[weekday]
    return config.min_prob


def get_maker_min_ev_for_zone(p_model_pct: float, config: MakerDefensiveConfig | None = None) -> tuple[float, str]:
    """min_ev = 0.5 * (100 - p) where p is traded side probability."""
    min_ev = 0.5 * (100.0 - p_model_pct)
    return min_ev, f"p={p_model_pct:.0f}%"


def get_maker_cluster_cap_for_day(observed_days: int, config: MakerConfig | None = None) -> float:
    if config is None:
        config = DEFAULT_MAKER_CONFIG
    if 0 <= observed_days <= 6:
        return config.cluster_cap_by_day[observed_days]
    return 2.0
