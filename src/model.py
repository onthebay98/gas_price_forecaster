"""AR(1) model on differenced gas prices with Monte Carlo simulation.

Model: diff_t = c + phi * diff_{t-1} + X_t @ beta + epsilon_t
where diff_t = price_t - price_{t-1} (consecutive daily observations only)

IMPORTANT: phi is estimated exclusively from the daily observation period
(Dec 2025+, ~102 observations). The weekly historical data (1990-2025) cannot
contribute to phi estimation because the 7-day lag autocorrelation is a
fundamentally different quantity than the 1-day lag autocorrelation.
For a daily AR(1) with phi=0.56, the implied 7-day autocorrelation is
phi^7 ≈ 0.017 — essentially zero.

The full 35-year history IS used for:
- seasonal_norm feature computation (in features.py)
- Fourier feature definitions (deterministic, no estimation needed)

Key differences from TSA SARIMAX:
- Simpler: AR(1) on diffs, not state-space
- No COVID gap handling
- No outlier imputation (gas prices don't have storm days)
- Unit root in levels (non-stationary), stationary diffs
- Training limited by daily data availability (~102 obs as of Mar 2026)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    CONDITIONING_VARIANCE_SCALER,
    DOW_VARIANCE_RATIOS,
    GLOBAL_SIGMA_MULTIPLIER,
    MEAN_BIAS_BY_DAY,
    SEASONAL_VARIANCE_MULTIPLIERS,
)

logger = logging.getLogger(__name__)


@dataclass
class AR1Result:
    """Result of AR(1) model fitting."""

    c: float  # Intercept (per-day drift)
    phi: float  # AR(1) coefficient on lagged daily diff
    beta: np.ndarray  # Exogenous feature coefficients
    sigma: float  # Residual standard deviation (per-day)
    feature_names: list[str]
    n_obs: int
    r_squared: float
    residuals: np.ndarray
    train_start: str  # First training date
    train_end: str  # Last training date


def fit_ar1(
    prices: pd.Series,
    features: Optional[pd.DataFrame] = None,
) -> AR1Result:
    """
    Fit AR(1) model on daily price diffs with exogenous features via OLS.

    diff_t = c + phi * diff_{t-1} + X_t @ beta + epsilon_t

    REQUIRES consecutive daily observations. The caller must pass only the
    daily-frequency portion of the price history. Do NOT pass weekly data —
    the lag structure would be incoherent (7-day lag ≠ 1-day lag).

    Args:
        prices: Daily price series with DatetimeIndex (consecutive days only)
        features: Optional feature DataFrame aligned with prices.
                 If None, fit without exogenous features.

    Returns:
        AR1Result with fitted parameters in per-day units
    """
    prices = prices.dropna().sort_index()

    if len(prices) < 10:
        raise ValueError(f"Too few observations ({len(prices)}) for AR(1) fit")

    # Validate that data is daily frequency
    gaps = prices.index.to_series().diff().dt.days.dropna()
    non_daily = (gaps > 1).sum()
    if non_daily > 0:
        pct = non_daily / len(gaps) * 100
        if pct > 10:
            logger.warning(
                f"{non_daily}/{len(gaps)} ({pct:.0f}%) gaps > 1 day in training data. "
                f"AR(1) phi requires consecutive daily observations."
            )

    # Compute daily diffs
    diffs = prices.diff().dropna()

    # Create lagged diff (AR(1) term)
    lagged_diff = diffs.shift(1)

    # Drop first row (no lagged diff available)
    valid_mask = lagged_diff.notna()
    valid_idx = diffs[valid_mask].index

    # Align features if provided
    if features is not None:
        valid_idx = valid_idx.intersection(features.index)

    y = diffs.loc[valid_idx].values
    lag = lagged_diff.loc[valid_idx].values

    n = len(y)
    if n < 10:
        raise ValueError(f"Too few aligned observations ({n}) for AR(1) fit")

    # Build design matrix
    if features is not None and len(features.columns) > 0:
        X_exog = features.loc[valid_idx].values
        X = np.column_stack([np.ones(n), lag, X_exog])
        feature_names = list(features.columns)
    else:
        X = np.column_stack([np.ones(n), lag])
        feature_names = []

    n_params = X.shape[1]
    if n <= n_params:
        raise ValueError(
            f"Degenerate fit: {n} observations but {n_params} parameters. "
            f"Need n > n_params for valid OLS."
        )

    # OLS fit
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        logger.warning("OLS failed, using zero coefficients")
        coeffs = np.zeros(X.shape[1])

    c = coeffs[0]
    phi = coeffs[1]
    beta = coeffs[2:] if len(coeffs) > 2 else np.array([])

    # Compute residuals and sigma
    y_hat = X @ coeffs
    resid = y - y_hat
    sigma = float(np.std(resid, ddof=min(X.shape[1], n - 1)))

    # R-squared
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info(
        f"AR(1) fit: c={c:.6f}, phi={phi:.4f}, sigma={sigma:.6f}, "
        f"R²={r_sq:.4f}, n={n}, "
        f"train=[{valid_idx.min().date()}, {valid_idx.max().date()}]"
    )

    return AR1Result(
        c=c,
        phi=phi,
        beta=beta,
        sigma=sigma,
        feature_names=feature_names,
        n_obs=n,
        r_squared=r_sq,
        residuals=resid,
        train_start=str(valid_idx.min().date()),
        train_end=str(valid_idx.max().date()),
    )


def simulate_future_days(
    ar1: AR1Result,
    last_price: float,
    last_daily_diff: float,
    future_features: Optional[pd.DataFrame] = None,
    n_days: int = 7,
    future_dates: Optional[pd.DatetimeIndex] = None,
    n_sims: int = 25_000,
    observed_prices: Optional[np.ndarray] = None,
    n_observed: int = 0,
    # Calibration overrides — default to config globals if None
    dow_variance_ratios: Optional[dict[int, float]] = None,
    seasonal_variance_multipliers: Optional[dict[int, float]] = None,
    conditioning_variance_scaler: Optional[dict[int, float]] = None,
    mean_bias_by_day: Optional[dict[int, float]] = None,
) -> np.ndarray:
    """
    Monte Carlo simulation of future daily prices.

    Generates n_sims price paths forward from last_price using the AR(1) model.
    All steps are 1-day (simulation always runs at daily frequency).
    If observed_prices are provided, those days are fixed (conditioning).

    Args:
        ar1: Fitted AR(1) model (parameters in per-day units)
        last_price: Most recent observed price
        last_daily_diff: Most recent 1-day price change (for AR(1) lag)
        future_features: Feature matrix for days to simulate (n_days × n_features).
                        If None, no exogenous contribution.
        n_days: Number of days to simulate (default 7 for a week)
        future_dates: DatetimeIndex for simulated days (for DOW/month variance scaling).
                     If None, uses future_features.index or generates generic dates.
        n_sims: Number of Monte Carlo simulations
        observed_prices: Array of observed prices for conditioning (first n_observed days)
        n_observed: Number of observed days in the forecast week

    Returns:
        Array of shape (n_days, n_sims) with simulated daily prices
    """
    if future_features is not None:
        n_days = len(future_features)
        feature_vals = future_features.values
        dates = future_features.index
    else:
        feature_vals = None
        if future_dates is not None:
            dates = future_dates
        else:
            dates = pd.date_range(pd.Timestamp.now(), periods=n_days, freq="D")

    # Resolve calibration parameters (override or config globals)
    _dow_var = dow_variance_ratios if dow_variance_ratios is not None else DOW_VARIANCE_RATIOS
    _seasonal_var = seasonal_variance_multipliers if seasonal_variance_multipliers is not None else SEASONAL_VARIANCE_MULTIPLIERS
    _cond_var = conditioning_variance_scaler if conditioning_variance_scaler is not None else CONDITIONING_VARIANCE_SCALER
    _bias = mean_bias_by_day if mean_bias_by_day is not None else MEAN_BIAS_BY_DAY

    # Determine number of days to actually simulate
    n_sim_days = n_days - n_observed

    # Get variance scaling
    cond_scaler = _cond_var.get(n_sim_days, 1.0)

    # Initialize simulation arrays
    sims = np.zeros((n_days, n_sims))

    # Fill in observed days
    if observed_prices is not None and n_observed > 0:
        for d in range(min(n_observed, n_days)):
            sims[d, :] = observed_prices[d]

    # Determine starting state for simulation
    if n_observed > 0 and observed_prices is not None:
        prev_price = np.full(n_sims, observed_prices[min(n_observed - 1, n_days - 1)])
        if n_observed >= 2:
            prev_daily_diff = np.full(
                n_sims, observed_prices[n_observed - 1] - observed_prices[n_observed - 2]
            )
        else:
            prev_daily_diff = np.full(n_sims, observed_prices[0] - last_price)
    else:
        prev_price = np.full(n_sims, last_price)
        prev_daily_diff = np.full(n_sims, last_daily_diff)

    # Simulate remaining days (1-day steps)
    start_day = n_observed
    for d in range(start_day, n_days):
        # AR(1) mean: c + phi * prev_daily_diff + X_t @ beta
        mean_diff = ar1.c + ar1.phi * prev_daily_diff

        if feature_vals is not None and len(ar1.beta) > 0:
            mean_diff = mean_diff + feature_vals[d] @ ar1.beta

        # Variance scaling
        dow = dates[d].dayofweek
        month = dates[d].month
        dow_scale = _dow_var.get(dow, 1.0)
        seasonal_scale = _seasonal_var.get(month, 1.0)
        effective_sigma = ar1.sigma * GLOBAL_SIGMA_MULTIPLIER * dow_scale * seasonal_scale * cond_scaler

        # Mean bias correction
        bias_z = _bias.get(n_observed, 0.0)
        mean_diff = mean_diff + bias_z * effective_sigma

        # Draw innovations
        innovations = np.random.normal(0, effective_sigma, n_sims)

        # Compute new diff and price
        new_diff = mean_diff + innovations
        new_price = prev_price + new_diff

        sims[d, :] = new_price
        prev_daily_diff = new_diff
        prev_price = new_price

    return sims


def compute_weekly_avg_distribution(
    sims: np.ndarray,
) -> np.ndarray:
    """
    Compute weekly average across all simulated days.

    Args:
        sims: Array of shape (n_days, n_sims) with daily price simulations

    Returns:
        Array of shape (n_sims,) with weekly averages
    """
    return sims.mean(axis=0)


def compute_threshold_probs(
    weekly_avgs: np.ndarray,
    thresholds: list[float],
) -> dict[float, float]:
    """
    Compute P(weekly_avg > threshold) for each threshold.

    Args:
        weekly_avgs: Array of simulated weekly averages
        thresholds: List of threshold values (in $/gal)

    Returns:
        Dict mapping threshold -> P(avg > threshold)
    """
    n = len(weekly_avgs)
    probs = {}
    for t in thresholds:
        probs[t] = float(np.sum(weekly_avgs > t) / n)
    return probs
