"""Calendar features for gas price forecasting.

Simpler than TSA: no holiday position bins needed.
Features: Fourier(K=2) annual seasonality, DOW dummies (7 days), seasonal_norm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build feature matrix for gas price model.

    Features (12 total):
    - Fourier K=2 (4 features): sin_y1, cos_y1, sin_y2, cos_y2
    - DOW dummies (7 features): dow_0 through dow_6
    - seasonal_norm (1 feature): smoothed day-of-year price level

    Args:
        dates: DatetimeIndex for which to compute features

    Returns:
        DataFrame with features, indexed by date
    """
    df = pd.DataFrame(index=dates)
    doy = dates.dayofyear
    year_frac = doy / 365.25

    # Fourier K=2 (annual seasonality: summer driving season)
    for k in [1, 2]:
        df[f"sin_y{k}"] = np.sin(2 * np.pi * k * year_frac)
        df[f"cos_y{k}"] = np.cos(2 * np.pi * k * year_frac)

    # DOW dummies (all 7 days — let the model learn weekend vs weekday effects)
    dow = dates.dayofweek
    for d in range(7):
        df[f"dow_{d}"] = (dow == d).astype(float)

    return df


def add_seasonal_norm(
    features: pd.DataFrame,
    price_history: pd.Series,
) -> pd.DataFrame:
    """
    Add seasonal_norm feature: smoothed day-of-year median price level.

    Uses the full interpolated price history to compute a robust seasonal
    baseline. Smoothed with a Gaussian kernel (sigma=14 days).

    Args:
        features: Feature DataFrame to augment
        price_history: Full daily price history (interpolated), indexed by date

    Returns:
        Features DataFrame with 'seasonal_norm' column added
    """
    if price_history.empty:
        features["seasonal_norm"] = 0.0
        return features

    # Compute day-of-year median from log prices
    log_prices = np.log(price_history.dropna())
    doy_medians = log_prices.groupby(log_prices.index.dayofyear).median()

    # Fill all 366 days
    all_doys = pd.Series(index=range(1, 367), dtype=float)
    all_doys.update(doy_medians)
    all_doys = all_doys.interpolate(method="linear").ffill().bfill()

    # Gaussian smoothing (circular, sigma=14 days)
    vals = all_doys.values
    n = len(vals)
    sigma = 14
    kernel_range = np.arange(-3 * sigma, 3 * sigma + 1)
    kernel = np.exp(-0.5 * (kernel_range / sigma) ** 2)
    kernel /= kernel.sum()

    # Circular convolution
    extended = np.concatenate([vals[-3 * sigma:], vals, vals[:3 * sigma]])
    smoothed = np.convolve(extended, kernel, mode="same")[3 * sigma: 3 * sigma + n]

    # Normalize to zero mean
    smoothed = smoothed - smoothed.mean()

    # Map to feature dates
    doy_lookup = pd.Series(smoothed, index=range(1, n + 1))
    feature_doys = features.index.dayofyear
    features["seasonal_norm"] = feature_doys.map(lambda d: doy_lookup.get(d, 0.0)).values

    return features


FEATURE_NAMES = [
    "sin_y1", "cos_y1", "sin_y2", "cos_y2",
    "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
    "seasonal_norm",
]
