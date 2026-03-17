"""Phase 0: Market Validation & Feasibility Checks.

1. Conditioning value: var(weekly_avg | first N days) / var(weekly_avg)
2. Phi stability: AR(1) phi on rolling windows
3. Data summary statistics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.io import load_aaa_csv, load_daily_prices, load_interpolated_daily


def conditioning_value_check():
    """
    Compute variance ratio: var(weekly_avg | N days known) / var(weekly_avg unconditional).

    Uses the daily data to form complete Mon-Sun weeks, then checks how much
    knowing the first N days reduces variance of the weekly average.
    """
    print("\n" + "=" * 60)
    print("CONDITIONING VALUE CHECK")
    print("=" * 60)

    df = load_interpolated_daily()

    # Form complete Mon-Sun weeks
    df["weekday"] = df.index.dayofweek
    df["week_start"] = df.index - pd.to_timedelta(df["weekday"], unit="D")

    weeks = df.groupby("week_start")
    complete_weeks = []
    for ws, group in weeks:
        if len(group) == 7:
            prices = group["price"].values
            complete_weeks.append({"week_start": ws, "prices": prices, "avg": prices.mean()})

    if not complete_weeks:
        print("No complete weeks found!")
        return

    avgs = np.array([w["avg"] for w in complete_weeks])
    var_uncond = np.var(avgs)

    print(f"\nTotal complete weeks: {len(complete_weeks)}")
    print(f"Unconditional weekly avg: mean=${np.mean(avgs):.3f}, std=${np.std(avgs):.3f}")
    print(f"\nVariance ratio by observed days:")
    print(f"{'N days':>8}  {'Var ratio':>10}  {'Info gained':>12}")
    print("-" * 40)

    # The right way to measure conditioning value:
    # If I know the first N days, how uncertain am I about the weekly average?
    # weekly_avg = (sum_known + sum_unknown) / 7
    # Var(weekly_avg | N known) = Var(sum_unknown) / 49
    # We estimate Var(sum_unknown) empirically: for each week, compute the
    # "remaining sum" = sum of days N..6, then take variance across weeks.
    # But we must detrend first (price levels vary across decades).
    # Use within-week deviation: remaining_sum - E[remaining_sum | known_sum].

    # Approach: compute prediction error of weekly avg given first N days.
    # For each week with N known: best estimate = known_sum/7 + (7-N)/7 * E[day_price | context]
    # Simpler: just look at Var(remaining_sum) / Var(full_week_sum)
    # This tells us what fraction of weekly-avg variance comes from the unknown days.

    for n in range(0, 8):
        if n == 0:
            ratio = 1.0
        elif n >= 7:
            ratio = 0.0
        else:
            # For each week, compute the remaining (7-N) days sum
            remaining_sums = np.array([w["prices"][n:].sum() for w in complete_weeks])
            full_sums = np.array([w["prices"].sum() for w in complete_weeks])
            # Var(avg | N known) = Var(remaining_sum / 7)
            # Var(avg uncond) = Var(full_sum / 7)
            var_remain = np.var(remaining_sums) / 49.0
            var_full = np.var(full_sums) / 49.0
            ratio = var_remain / var_full if var_full > 0 else 0.0

        info = 1 - ratio
        print(f"  {n:>5}    {ratio:>8.4f}    {info*100:>8.1f}%")

    # More meaningful check: variance of weekly avg conditioned on first N days
    # being from the same regime (recent weeks only)
    print("\n--- Recent daily data conditioning check (Dec 2025+) ---")
    try:
        df_daily = load_daily_prices()
        df_daily["weekday"] = df_daily.index.dayofweek
        df_daily["week_start"] = df_daily.index - pd.to_timedelta(df_daily["weekday"], unit="D")

        daily_weeks = df_daily.groupby("week_start")
        daily_complete = []
        for ws, group in daily_weeks:
            clean = group["price"].dropna()
            if len(clean) == 7:
                daily_complete.append({"prices": clean.values, "avg": clean.values.mean()})

        if daily_complete:
            avgs_d = np.array([w["avg"] for w in daily_complete])
            var_d = np.var(avgs_d)
            print(f"Complete daily weeks: {len(daily_complete)}")
            print(f"Weekly avg std: ${np.std(avgs_d):.4f}")

            full_sums = np.array([w["prices"].sum() for w in daily_complete])
            var_full = np.var(full_sums) / 49.0
            for n in [1, 2, 3, 4, 5, 6]:
                remain = np.array([w["prices"][n:].sum() for w in daily_complete])
                var_remain = np.var(remain) / 49.0
                ratio = var_remain / var_full if var_full > 0 else 0.0
                print(f"  N={n}: var_ratio={ratio:.4f} (info={1-ratio:.1%})")
        else:
            print("No complete daily weeks found yet.")
    except ValueError as e:
        print(f"  Skipped: {e}")


def phi_stability_check():
    """
    Estimate AR(1) phi on rolling 3-year windows across full history.
    Uses weekly data where available.
    """
    print("\n" + "=" * 60)
    print("PHI STABILITY CHECK")
    print("=" * 60)

    df = load_aaa_csv()
    prices = df["price"].dropna()

    # Compute diffs (will be weekly diffs for weekly data, daily for daily)
    diffs = prices.diff().dropna()
    lagged = diffs.shift(1).dropna()

    # Align
    common = diffs.index.intersection(lagged.index)
    y = diffs.loc[common].values
    x = lagged.loc[common].values

    # Full sample phi
    phi_full = np.corrcoef(x, y)[0, 1]
    print(f"\nFull sample phi (autocorrelation of diffs): {phi_full:.4f}")
    print(f"Full sample N: {len(y)}")

    # Rolling windows
    window_years = 5
    dates = common
    print(f"\nRolling {window_years}-year phi estimates:")
    print(f"{'Window':>20}  {'Phi':>8}  {'N':>6}")
    print("-" * 40)

    for start_year in range(1995, 2026, 3):
        start = pd.Timestamp(f"{start_year}-01-01")
        end = pd.Timestamp(f"{start_year + window_years}-01-01")
        mask = (dates >= start) & (dates < end)
        if mask.sum() < 20:
            continue
        phi_w = np.corrcoef(x[mask], y[mask])[0, 1]
        print(f"  {start_year}-{start_year + window_years}     {phi_w:>7.4f}   {mask.sum():>5}")

    # Also check on daily data only
    print("\n--- Daily data only (Dec 2025+) ---")
    try:
        df_daily = load_daily_prices()
        p = df_daily["price"].dropna()
        d = p.diff().dropna()
        l = d.shift(1).dropna()
        common_d = d.index.intersection(l.index)
        if len(common_d) > 10:
            phi_d = np.corrcoef(d.loc[common_d].values, l.loc[common_d].values)[0, 1]
            print(f"  Daily phi: {phi_d:.4f} (N={len(common_d)})")
    except ValueError:
        print("  No daily data available")


def data_summary():
    """Print data summary statistics."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    df = load_aaa_csv()
    prices = df["price"].dropna()

    print(f"\nTotal observations: {len(prices)}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Current price: ${prices.iloc[-1]:.3f}")
    print(f"Price range: ${prices.min():.3f} - ${prices.max():.3f}")
    print(f"Mean: ${prices.mean():.3f}")
    print(f"Std: ${prices.std():.3f}")

    # Check frequency transition
    gaps = prices.index.to_series().diff().dt.days.dropna()
    print(f"\nGap statistics:")
    print(f"  1-day gaps (daily): {(gaps == 1).sum()}")
    print(f"  7-day gaps (weekly): {(gaps == 7).sum()}")
    print(f"  Other gaps: {((gaps != 1) & (gaps != 7)).sum()}")


if __name__ == "__main__":
    data_summary()
    phi_stability_check()
    conditioning_value_check()
