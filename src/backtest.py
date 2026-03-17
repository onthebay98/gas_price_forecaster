"""Walk-forward backtesting for gas price AR(1) model.

Evaluates model performance by conditioning level (observed_days 0-7).
For each week, trains AR(1) on expanding window of daily data through
the prior Sunday, conditions on observed_days prices, simulates 25k paths,
and compares P(settlement > threshold) to realized outcome.

Settlement: sims[7,:] = next Monday's AAA price.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.features import add_seasonal_norm, make_features
from src.io import load_aaa_csv, load_daily_prices, load_interpolated_daily
from src.model import AR1Result, compute_threshold_probs, fit_ar1, simulate_future_days
from src.predict_utils import get_settlement_dates

logger = logging.getLogger(__name__)


def kalshi_thresholds() -> list[float]:
    """Actual Kalshi KXAAAGASW market strikes.

    10c increments from $2.90 to $4.50, plus 5c exceptions at $3.35 and $3.45.
    """
    thresholds = []
    # 10c increments from 2.90 to 4.50
    t = 2.90
    while t <= 4.50 + 1e-9:
        thresholds.append(round(t, 2))
        t += 0.10

    # Add 5c exceptions
    for extra in (3.35, 3.45):
        if extra not in thresholds:
            thresholds.append(extra)

    return sorted(thresholds)


def default_thresholds() -> list[float]:
    """Default backtest thresholds — matches Kalshi market strikes."""
    return kalshi_thresholds()


def check_threshold_boundaries(thresholds: list[float], current_price: float) -> None:
    """Warn if current price is near the edge of the threshold grid."""
    lo, hi = min(thresholds), max(thresholds)
    margin = 0.10
    if current_price < lo + margin:
        logger.warning(
            f"Current price ${current_price:.3f} is within ${margin} of grid floor "
            f"${lo:.2f}. Consider extending thresholds downward."
        )
    if current_price > hi - margin:
        logger.warning(
            f"Current price ${current_price:.3f} is within ${margin} of grid ceiling "
            f"${hi:.2f}. Consider extending thresholds upward."
        )


def _th_label(th: float) -> str:
    """Format threshold for column names: 3.50 -> '3p50'."""
    return f"{th:.2f}".replace(".", "p")


def run_conditioned_backtest(
    *,
    df_daily: pd.DataFrame,
    df_interp: pd.DataFrame,
    observed_days: int,
    start_date: str,
    end_date: str,
    thresholds: list[float] | None = None,
    n_sims: int = 25_000,
    min_train_obs: int = 14,
    # Calibration overrides for simulate_future_days
    dow_variance_ratios: dict[int, float] | None = None,
    seasonal_variance_multipliers: dict[int, float] | None = None,
    conditioning_variance_scaler: dict[int, float] | None = None,
    mean_bias_by_day: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Walk-forward conditioned backtest for gas price model.

    For each Monday (week_start) in [start_date, end_date]:
    1. Train AR(1) on all daily data up to (but not including) week_start
    2. Build features for the 8-day forecast window
    3. If observed_days > 0, condition on observed prices
    4. Simulate n_sims paths, extract settlement distribution (sims[7,:])
    5. Compare P(settlement > threshold) to realized outcome

    Args:
        df_daily: Daily-frequency price data (Dec 2025+), DatetimeIndex + 'price' col
        df_interp: Full interpolated daily history (for seasonal_norm)
        observed_days: Number of days to condition on (0-7)
        start_date: First Monday to evaluate (ISO format)
        end_date: Last Monday to evaluate (ISO format)
        thresholds: Price thresholds; defaults to Kalshi strikes
        n_sims: Number of Monte Carlo simulations
        min_train_obs: Minimum daily training observations for AR(1)
        dow_variance_ratios: Override DOW variance ratios
        seasonal_variance_multipliers: Override seasonal variance multipliers
        conditioning_variance_scaler: Override conditioning variance scaler
        mean_bias_by_day: Override mean bias corrections

    Returns:
        DataFrame with one row per evaluated week.
    """
    if thresholds is None:
        thresholds = default_thresholds()

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    # Ensure start is a Monday
    if start.weekday() != 0:
        start = start + timedelta(days=(7 - start.weekday()) % 7)

    rows = []
    current = start
    while current <= end:
        settlement_dates = get_settlement_dates(current)
        settlement_date = settlement_dates[-1]  # next Monday

        # --- Training cutoff: all daily data before this Monday ---
        train_mask = df_daily.index < pd.Timestamp(current)
        train_data = df_daily[train_mask]

        if len(train_data) < min_train_obs:
            logger.debug(
                f"Skipping {current}: only {len(train_data)} training obs "
                f"(need {min_train_obs})"
            )
            current += timedelta(days=7)
            continue

        # --- Check realized settlement price exists ---
        settlement_ts = pd.Timestamp(settlement_date)
        # Look in both daily and full datasets
        realized_price = None
        if settlement_ts in df_daily.index:
            val = df_daily.loc[settlement_ts, "price"]
            if pd.notna(val):
                realized_price = float(val)

        if realized_price is None:
            # Try the interpolated dataset (which has weekly Monday prices)
            if settlement_ts in df_interp.index:
                val = df_interp.loc[settlement_ts, "price"]
                if pd.notna(val):
                    realized_price = float(val)

        if realized_price is None:
            logger.debug(f"Skipping {current}: no realized settlement price for {settlement_date}")
            current += timedelta(days=7)
            continue

        # --- Get observed prices for conditioning ---
        obs_prices = None
        actual_observed = 0
        if observed_days > 0:
            obs = []
            for d in settlement_dates[:observed_days]:
                ts = pd.Timestamp(d)
                if ts in df_daily.index and pd.notna(df_daily.loc[ts, "price"]):
                    obs.append(float(df_daily.loc[ts, "price"]))
                else:
                    break
            if obs:
                obs_prices = np.array(obs)
                actual_observed = len(obs)
            if actual_observed < observed_days:
                logger.debug(
                    f"Week {current}: requested {observed_days} obs days, "
                    f"only {actual_observed} available"
                )

        # --- Fit AR(1) on training data ---
        try:
            train_features = make_features(train_data.index)
            train_features = add_seasonal_norm(train_features, df_interp["price"])
            ar1 = fit_ar1(train_data["price"], train_features)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"Skipping {current}: AR(1) fit failed: {e}")
            current += timedelta(days=7)
            continue

        # --- Build forecast features ---
        forecast_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in settlement_dates])
        forecast_features = make_features(forecast_idx)
        forecast_features = add_seasonal_norm(forecast_features, df_interp["price"])

        # --- Get last price and diff ---
        prices_clean = train_data["price"].dropna()
        last_price = float(prices_clean.iloc[-1])
        if len(prices_clean) >= 2:
            last_diff = float(prices_clean.iloc[-1] - prices_clean.iloc[-2])
        else:
            last_diff = 0.0

        # --- Simulate ---
        sims = simulate_future_days(
            ar1=ar1,
            last_price=last_price,
            last_daily_diff=last_diff,
            future_features=forecast_features,
            n_sims=n_sims,
            observed_prices=obs_prices,
            n_observed=actual_observed,
            dow_variance_ratios=dow_variance_ratios,
            seasonal_variance_multipliers=seasonal_variance_multipliers,
            conditioning_variance_scaler=conditioning_variance_scaler,
            mean_bias_by_day=mean_bias_by_day,
        )

        # Settlement = sims[7,:] (next Monday)
        settlement_sims = sims[7, :]

        # --- Build result row ---
        row = {
            "week_start": str(current),
            "settlement_date": str(settlement_date),
            "settlement_price": realized_price,
            "n_train_obs": ar1.n_obs,
            "observed_days": observed_days,
            "n_observed": actual_observed,
            "ar1_phi": ar1.phi,
            "ar1_sigma": ar1.sigma,
            "ar1_c": ar1.c,
            "ar1_r_squared": ar1.r_squared,
            "model_mean": float(np.mean(settlement_sims)),
            "model_std": float(np.std(settlement_sims)),
            "model_median": float(np.median(settlement_sims)),
            "last_train_price": last_price,
        }

        # Threshold probabilities and realized outcomes
        for th in thresholds:
            label = _th_label(th)
            p = float(np.mean(settlement_sims > th))
            realized = float(realized_price > th)
            row[f"P_gt_{label}"] = p
            row[f"realized_gt_{label}"] = realized

        rows.append(row)
        current += timedelta(days=7)

    if not rows:
        logger.warning("No weeks evaluated — check date range and data availability")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_backtest_metrics(
    bt: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> dict:
    """Compute summary metrics from backtest results.

    Returns dict with:
    - mae: Mean absolute error of settlement forecast
    - mape: Mean absolute percentage error
    - brier: Average Brier score across all thresholds
    - dir_acc: Directional accuracy (model_mean vs last_train_price matches realized)
    - mean_z: Mean of standardized errors (should be ~0 if unbiased)
    - std_z: Std of standardized errors (should be ~1 if well-calibrated)
    - n: Number of weeks evaluated
    """
    if len(bt) == 0:
        return {"n": 0}

    if thresholds is None:
        thresholds = default_thresholds()

    n = len(bt)

    # MAE and MAPE
    errors = bt["model_mean"] - bt["settlement_price"]
    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / bt["settlement_price"]) * 100)

    # Brier score: average across all thresholds
    brier_scores = []
    for th in thresholds:
        label = _th_label(th)
        p_col = f"P_gt_{label}"
        r_col = f"realized_gt_{label}"
        if p_col in bt.columns and r_col in bt.columns:
            bs = ((bt[p_col] - bt[r_col]) ** 2).mean()
            brier_scores.append(bs)
    brier = float(np.mean(brier_scores)) if brier_scores else float("nan")

    # Directional accuracy: did model predict correct direction of move?
    model_dir = np.sign(bt["model_mean"] - bt["last_train_price"])
    actual_dir = np.sign(bt["settlement_price"] - bt["last_train_price"])
    dir_acc = float((model_dir == actual_dir).mean() * 100)

    # Z-scores: (model_mean - realized) / model_std
    # Filter out degenerate rows where model_std is near-zero
    valid_std = bt["model_std"] > 1e-6
    if valid_std.sum() > 0:
        z_scores = (bt.loc[valid_std, "model_mean"] - bt.loc[valid_std, "settlement_price"]) / bt.loc[valid_std, "model_std"]
    else:
        z_scores = pd.Series(dtype=float)
    z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()
    mean_z = float(z_scores.mean()) if len(z_scores) > 0 else float("nan")
    std_z = float(z_scores.std()) if len(z_scores) > 1 else float("nan")

    return {
        "mae": mae,
        "mape": mape,
        "brier": brier,
        "dir_acc": dir_acc,
        "mean_z": mean_z,
        "std_z": std_z,
        "n": n,
    }
