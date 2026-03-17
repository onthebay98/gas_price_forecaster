"""Forecast pipeline orchestration for gas prices.

Pipeline: load data → fit AR(1) → build features → condition on observed days
         → simulate 8 days → extract settlement day distribution → threshold probs

Settlement target: Kalshi KXAAAGASW settles on the AAA national average price
on a specific <date> — which is the Monday AFTER the trading week. The event
opens Monday, closes Sunday night, and settles on next Monday's AAA price.
Example: KXAAAGASW-26MAR23 opens Mar 16, closes Mar 22, settles on Mar 23 AAA.

The simulation runs 8 days (Mon through next Mon). The settlement distribution
is sims[7, :] — the 8th simulated day (next Monday).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DEFAULT_MODEL_CONFIG, ModelConfig
from src.features import FEATURE_NAMES, add_seasonal_norm, make_features
from src.io import load_aaa_csv, load_daily_prices, load_interpolated_daily
from src.model import (
    AR1Result,
    compute_threshold_probs,
    fit_ar1,
    simulate_future_days,
)

logger = logging.getLogger(__name__)


def get_week_dates(week_start: date) -> list[date]:
    """
    Get all 7 dates for a Mon-Sun week.

    Args:
        week_start: Monday of the target week

    Returns:
        List of 7 dates [Mon, Tue, ..., Sun]
    """
    return [week_start + timedelta(days=i) for i in range(7)]


def get_settlement_dates(week_start: date) -> list[date]:
    """
    Get 8 dates: Mon through next Mon (settlement day).

    Kalshi KXAAAGASW settles on the Monday after the trading week.
    We simulate 8 days so sims[7] is the settlement price.

    Args:
        week_start: Monday of the trading week

    Returns:
        List of 8 dates [Mon, Tue, ..., Sun, next Mon]
    """
    return [week_start + timedelta(days=i) for i in range(8)]


def auto_week_start(today: Optional[date] = None) -> date:
    """
    Determine the current trading week's Monday.

    Args:
        today: Override date (default: today)

    Returns:
        Monday of the current week
    """
    if today is None:
        today = date.today()
    # Monday of this week
    return today - timedelta(days=today.weekday())


def auto_asof_day(
    week_start: date,
    prices: pd.DataFrame,
) -> int:
    """
    Determine how many days of the current week have price observations.

    Args:
        week_start: Monday of the target week
        prices: DataFrame with DatetimeIndex and 'price' column

    Returns:
        Number of observed days (0-7)
    """
    week_dates = get_week_dates(week_start)
    n_observed = 0
    for d in week_dates:
        ts = pd.Timestamp(d)
        if ts in prices.index and pd.notna(prices.loc[ts, "price"]):
            n_observed += 1
        else:
            break  # Stop at first missing day
    return n_observed


def weekly_avg_distribution(
    week_start: date,
    asof_day: int = 0,
    thresholds: Optional[list[float]] = None,
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
    data_path: str = "data/aaa_daily.csv",
) -> dict:
    """
    Full forecast pipeline: produce settlement price distribution and threshold probs.

    Kalshi KXAAAGASW settles on the AAA price on the Monday after the trading week.
    Simulates 8 days (Mon through next Mon). The primary output ('weekly_avgs')
    is the next-Monday settlement price distribution.

    Args:
        week_start: Monday of the target week (trading opens this day)
        asof_day: Number of observed days (0 = no data, 7 = full week)
        thresholds: Price thresholds to compute probabilities for
        config: Model configuration
        data_path: Path to AAA CSV

    Returns:
        Dict with keys:
        - 'weekly_avgs': array of simulated settlement prices (next Monday)
        - 'probs': dict of threshold -> P(settlement_price > threshold)
        - 'mean', 'std', 'median', 'p5', 'p95': summary stats
        - 'observed_prices': array of observed prices (if any)
        - 'ar1': fitted AR1Result
        - 'n_observed': number of observed days
        - 'sims': full (8, n_sims) simulation array
        - 'settlement_date': the Monday settlement date
    """
    # Load data
    # - df_all: raw CSV (weekly pre-Dec 2025, daily after)
    # - df_daily: daily-frequency portion only (Dec 2025+)
    # - df_interp: full history interpolated to daily (for seasonal_norm only)
    df_all = load_aaa_csv(data_path)
    df_daily = load_daily_prices(data_path)
    df_interp = load_interpolated_daily(data_path)

    # Build features for training period (daily observations only)
    # Fourier/DOW are deterministic; seasonal_norm uses full 35-year history
    train_features = make_features(df_daily.index)
    train_features = add_seasonal_norm(train_features, df_interp["price"])

    # Fit AR(1) on daily data ONLY — phi requires consecutive daily observations.
    # Weekly data cannot contribute (7-day lag ≠ 1-day lag).
    ar1 = fit_ar1(df_daily["price"], train_features)

    # Build features for forecast period (8 days: Mon through next Mon)
    # Settlement is on next Monday — we need to simulate through that day.
    settlement_dates = get_settlement_dates(week_start)
    settlement_date = settlement_dates[-1]  # Next Monday
    forecast_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in settlement_dates])
    forecast_features = make_features(forecast_idx)
    forecast_features = add_seasonal_norm(forecast_features, df_interp["price"])

    # Get last observed price and diff for simulation start
    last_obs_date = df_daily["price"].dropna().index[-1]
    last_price = float(df_daily["price"].dropna().iloc[-1])

    # Guard against manual --asof-day 0 when data exists within the current week.
    # Without conditioning, sims[0] would simulate forward from last_price, double-counting
    # a day that should be fixed — subtle miscalibration that's hard to diagnose live.
    if asof_day == 0 and last_obs_date.date() >= week_start:
        raise ValueError(
            f"asof_day=0 but last observation ({last_obs_date.date()}) "
            f"is within the current week ({week_start}). "
            f"Use auto_asof_day() or set asof_day >= 1."
        )

    prices_clean = df_daily["price"].dropna()
    if len(prices_clean) >= 2:
        last_diff = float(prices_clean.iloc[-1] - prices_clean.iloc[-2])
    else:
        last_diff = 0.0

    # Get observed prices for conditioning (from trading week days only, not settlement day)
    week_dates = get_week_dates(week_start)
    observed_prices = None
    if asof_day > 0:
        obs = []
        for d in settlement_dates[:asof_day]:
            ts = pd.Timestamp(d)
            if ts in df_daily.index and pd.notna(df_daily.loc[ts, "price"]):
                obs.append(float(df_daily.loc[ts, "price"]))
            elif ts in df_all.index and pd.notna(df_all.loc[ts, "price"]):
                obs.append(float(df_all.loc[ts, "price"]))
            else:
                # Stop conditioning if we don't have this day's price
                break
        if obs:
            observed_prices = np.array(obs)
            asof_day = len(obs)  # Adjust if we found fewer than expected
        else:
            asof_day = 0

    # Simulate
    sims = simulate_future_days(
        ar1=ar1,
        last_price=last_price,
        last_daily_diff=last_diff,
        future_features=forecast_features,
        n_sims=config.n_sims,
        observed_prices=observed_prices,
        n_observed=asof_day,
    )

    # Extract settlement day price distribution (day 8 = index 7 = next Monday)
    # Kalshi KXAAAGASW settles on next Monday's AAA national average price.
    settlement_prices = sims[7, :]

    # Compute threshold probabilities against settlement prices
    probs = {}
    if thresholds:
        probs = compute_threshold_probs(settlement_prices, thresholds)

    return {
        "weekly_avgs": settlement_prices,  # Settlement target distribution
        "probs": probs,
        "mean": float(np.mean(settlement_prices)),
        "std": float(np.std(settlement_prices)),
        "median": float(np.median(settlement_prices)),
        "p5": float(np.percentile(settlement_prices, 5)),
        "p95": float(np.percentile(settlement_prices, 95)),
        "observed_prices": observed_prices,
        "ar1": ar1,
        "n_observed": asof_day,
        "last_price": last_price,
        "last_obs_date": str(last_obs_date.date()),
        "week_start": str(week_start),
        "settlement_date": str(settlement_date),
        "sims": sims,
    }


def generate_predictions_table(
    result: dict,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Build a predictions table from forecast results.

    Args:
        result: Output from weekly_avg_distribution()
        thresholds: Price thresholds ($/gal)

    Returns:
        DataFrame with columns: threshold, P_above, P_below, bucket
    """
    probs = result["probs"]
    rows = []
    for t in sorted(thresholds):
        p_above = probs.get(t, 0.5)
        rows.append({
            "threshold": t,
            "P_above": round(p_above, 4),
            "P_below": round(1 - p_above, 4),
            "P_model": round(p_above, 4),
            "bucket": f">{t:.2f}",
        })
    return pd.DataFrame(rows)


def save_predictions(
    predictions_df: pd.DataFrame,
    result: dict,
    output_csv: str = "data/latest_predict.csv",
    output_meta: str = "data/latest_predict_meta.json",
):
    """Save predictions to CSV and metadata to JSON."""
    predictions_df.to_csv(output_csv, index=False)

    meta = {
        "week_start": result["week_start"],
        "settlement_target": "next_monday_price",
        "settlement_date": result.get("settlement_date", ""),
        "n_observed": result["n_observed"],
        "last_obs_date": result["last_obs_date"],
        "last_price": result["last_price"],
        "forecast_mean": result["mean"],
        "forecast_std": result["std"],
        "forecast_median": result["median"],
        "forecast_p5": result["p5"],
        "forecast_p95": result["p95"],
        "ar1_phi": result["ar1"].phi,
        "ar1_sigma": result["ar1"].sigma,
        "ar1_r_squared": result["ar1"].r_squared,
        "ar1_n_obs": result["ar1"].n_obs,
        "ar1_train_start": result["ar1"].train_start,
        "ar1_train_end": result["ar1"].train_end,
        "n_sims": len(result["weekly_avgs"]),
    }
    if result["observed_prices"] is not None:
        meta["observed_prices"] = result["observed_prices"].tolist()

    Path(output_meta).parent.mkdir(parents=True, exist_ok=True)
    with open(output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved predictions to {output_csv} and {output_meta}")
