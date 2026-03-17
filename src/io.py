"""Data loading for AAA gas price data.

AAA data has two regimes:
- 1990-Nov 2025: Weekly observations (every Monday)
- Dec 2025-present: Daily observations (7 days/week)

For model training on daily diffs, we use only the daily portion.
The full history is available for longer-term statistics and seasonal_norm.
"""
from __future__ import annotations

import pandas as pd


def load_aaa_csv(path: str = "data/aaa_daily.csv") -> pd.DataFrame:
    """
    Load AAA gas price data from CSV.

    Returns a DataFrame with DatetimeIndex and 'price' column.
    Does NOT interpolate — returns data as-is (weekly pre-Dec 2025, daily after).
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if df["price"].isna().all():
        raise ValueError("No valid price data found.")
    return df


def load_daily_prices(path: str = "data/aaa_daily.csv") -> pd.DataFrame:
    """
    Load only the daily-frequency portion of AAA data (Dec 2025+).

    Returns a DataFrame with DatetimeIndex, 'price' column, and complete
    daily frequency (reindexed to fill any gaps).
    """
    df = load_aaa_csv(path)

    # Detect daily data: find where consecutive gaps <= 1 day
    gaps = df.index.to_series().diff().dt.days
    daily_mask = gaps <= 1
    if not daily_mask.any():
        raise ValueError("No daily-frequency data found in AAA CSV.")

    # Find start of daily data
    first_daily_idx = daily_mask.idxmax()
    # Include the observation just before the first daily gap
    prev_idx = df.index[df.index < first_daily_idx]
    if len(prev_idx) > 0:
        first_daily_idx = prev_idx[-1]

    df_daily = df.loc[first_daily_idx:]

    # Reindex to fill any missing dates
    full_range = pd.date_range(df_daily.index.min(), df_daily.index.max(), freq="D")
    df_daily = df_daily.reindex(full_range)
    df_daily.index.name = "date"

    return df_daily


def load_interpolated_daily(path: str = "data/aaa_daily.csv") -> pd.DataFrame:
    """
    Load full AAA history with linear interpolation to daily frequency.

    The weekly pre-Dec 2025 data is interpolated to daily. This is useful
    for computing seasonal_norm and long-term statistics, NOT for AR(1) training.
    """
    df = load_aaa_csv(path)
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    df["price"] = df["price"].interpolate(method="linear")
    return df
