#!/usr/bin/env python3
"""Gas price model calibration diagnostics.

Runs in order:
  Part 0: Historical sanity check (model vs 35yr Monday-to-Monday data)
  Part 1: AR(1) trend bias check
  Part 2: DOW variance ratios from residuals
  Part 3: Seasonal variance (report only — single season in sample)
  Part 4: Mean bias by observed_days (from backtest Z-scores)
  Part 5: Conditioning variance scaler (from backtest std_z)

Usage:
    PYTHONPATH=. python scripts/calibrate_variance.py
    PYTHONPATH=. python scripts/calibrate_variance.py --backtest-dir data/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest import _th_label, compute_backtest_metrics, default_thresholds
from src.features import add_seasonal_norm, make_features
from src.io import load_aaa_csv, load_daily_prices, load_interpolated_daily
from src.model import fit_ar1


def part0_historical_sanity_check(
    df_all: pd.DataFrame,
    ar1_phi: float,
    ar1_sigma: float,
) -> float:
    """Compare model-implied 8-day std against empirical Monday-to-Monday std.

    Returns the ratio (model_implied / empirical_recent).
    """
    print("\n" + "=" * 65)
    print("PART 0: Historical Sanity Check — Model vs 35yr Monday Data")
    print("=" * 65)

    # Model-implied 8-day std: sigma * sqrt(sum_{i=0}^{7} phi^{2i})
    phi_sum = sum(ar1_phi ** (2 * i) for i in range(8))
    model_8day_std = ar1_sigma * np.sqrt(phi_sum)

    print(f"  AR(1) phi = {ar1_phi:.4f}, sigma = {ar1_sigma:.6f}")
    print(f"  Sum(phi^2i, i=0..7) = {phi_sum:.4f}")
    print(f"  Model-implied 8-day std = ${model_8day_std:.4f}")

    # Empirical Monday-to-Monday diffs from weekly data
    # Filter to Mondays only (weekly data is all Mondays)
    mondays = df_all[df_all.index.dayofweek == 0]["price"].dropna()
    weekly_diffs = mondays.diff().dropna()

    print(f"\n  Historical Monday-to-Monday diffs: {len(weekly_diffs)} observations")

    # Compute by era
    eras = {
        "Full sample (1990-2026)": weekly_diffs,
        "Recent 10yr (2016-2026)": weekly_diffs[weekly_diffs.index >= "2016-01-01"],
        "Recent 5yr (2021-2026)": weekly_diffs[weekly_diffs.index >= "2021-01-01"],
        "Recent 2yr (2024-2026)": weekly_diffs[weekly_diffs.index >= "2024-01-01"],
    }

    print(f"\n  {'Era':<30} {'N':>5} {'Std':>10} {'Ratio':>8}")
    print("  " + "-" * 55)

    ratios = {}
    for label, diffs in eras.items():
        if len(diffs) < 5:
            continue
        emp_std = float(diffs.std())
        ratio = model_8day_std / emp_std if emp_std > 0 else float("nan")
        ratios[label] = ratio
        flag = ""
        if ratio < 0.5:
            flag = " !! MODEL WAY TOO TIGHT"
        elif ratio > 2.0:
            flag = " !! MODEL WAY TOO WIDE"
        print(f"  {label:<30} {len(diffs):>5} ${emp_std:.4f} {ratio:>8.3f}{flag}")

    # Use recent 2yr as the most relevant comparison
    recent_ratio = ratios.get("Recent 2yr (2024-2026)", float("nan"))
    if np.isfinite(recent_ratio) and recent_ratio < 0.5:
        print(
            "\n  RED FLAG: Model is >2x too tight vs recent history."
            "\n  Confidence intervals are dangerously narrow."
            "\n  Consider a global variance multiplier before proceeding."
        )
        recommended_mult = 1.0 / recent_ratio
        print(f"  Suggested global multiplier: ~{recommended_mult:.2f}x")

    return recent_ratio


def part1_trend_bias_check(ar1_c: float, ar1_sigma: float, n_obs: int, residuals: np.ndarray):
    """Check if AR(1) constant implies systematic directional drift."""
    print("\n" + "=" * 65)
    print("PART 1: AR(1) Trend Bias Check")
    print("=" * 65)

    # Mean residual (should be ~0 by OLS construction, but verify)
    mean_resid = float(np.mean(residuals))
    se_resid = float(np.std(residuals) / np.sqrt(len(residuals)))
    t_resid = mean_resid / se_resid if se_resid > 0 else 0

    print(f"  AR(1) constant c = {ar1_c:.6f}")
    print(f"  Implied daily drift: ${ar1_c:.6f}/day")
    print(f"  Implied annual drift: ${ar1_c * 365:.4f}/year")
    print(f"  Implied 8-day drift: ${ar1_c * 8:.6f}")

    # c relative to sigma
    c_over_sigma = ar1_c / ar1_sigma if ar1_sigma > 0 else 0
    print(f"  c/sigma = {c_over_sigma:.3f} (drift per day in sigma units)")

    print(f"\n  Mean residual: {mean_resid:.8f} (t-stat: {t_resid:.3f})")
    if abs(c_over_sigma) > 0.1:
        print(
            f"\n  WARNING: c/sigma = {c_over_sigma:.3f} — model embeds a trend."
            f"\n  During sideways/reverting markets, this drift will create systematic bias."
        )
    else:
        print("  Trend component is small relative to noise. OK.")


def part2_dow_variance_ratios(residuals: np.ndarray, dates: pd.DatetimeIndex):
    """Compute DOW variance ratios from AR(1) residuals."""
    print("\n" + "=" * 65)
    print("PART 2: DOW Variance Ratios")
    print("=" * 65)

    df = pd.DataFrame({"resid": residuals, "dow": dates.dayofweek})
    overall_std = float(np.std(residuals))

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"\n  Overall std: {overall_std:.6f}")
    print(f"\n  {'DOW':<5} {'N':>4} {'Std':>10} {'Ratio':>8}")
    print("  " + "-" * 30)

    ratios = {}
    for dow in range(7):
        mask = df["dow"] == dow
        n = mask.sum()
        if n < 3:
            print(f"  {day_names[dow]:<5} {n:>4}   (too few)")
            continue
        dow_std = float(df.loc[mask, "resid"].std())
        ratio = dow_std / overall_std if overall_std > 0 else 1.0
        ratios[dow] = ratio
        print(f"  {day_names[dow]:<5} {n:>4} {dow_std:>10.6f} {ratio:>8.3f}")

    print(f"\n  N per DOW ≈ {len(residuals) // 7}. Too noisy for individual calibration.")
    print("  Recommendation: leave DOW_VARIANCE_RATIOS at 1.0.")
    print("  Revisit when N > 50 per DOW (~Sep 2026).")


def part3_seasonal_note():
    """Note that seasonal calibration is not possible with current data."""
    print("\n" + "=" * 65)
    print("PART 3: Seasonal Variance Multipliers")
    print("=" * 65)
    print("  Single season in sample (Dec 2025 – Mar 2026).")
    print("  Cannot calibrate. Leave SEASONAL_VARIANCE_MULTIPLIERS at 1.0.")


def part4_mean_bias_by_day(backtest_dir: str):
    """Compute mean bias per observed_days from backtest Z-scores."""
    print("\n" + "=" * 65)
    print("PART 4: Mean Bias by Observed Days (from backtest)")
    print("=" * 65)

    thresholds = default_thresholds()

    print(f"\n  {'Obs Days':>8} {'N':>4} {'Mean Z':>8} {'SE':>8} {'Sig?':>6} {'Recommended':>12}")
    print("  " + "-" * 50)

    for od in range(8):
        path = Path(backtest_dir) / f"backtest_ar1_asof_day{od}.csv"
        if not path.exists():
            continue
        bt = pd.read_csv(path)
        if len(bt) == 0:
            continue

        z_scores = (bt["model_mean"] - bt["settlement_price"]) / bt["model_std"]
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()

        if len(z_scores) < 3:
            continue

        mean_z = float(z_scores.mean())
        se = float(z_scores.std() / np.sqrt(len(z_scores)))
        sig = "YES" if abs(mean_z) > 1.0 else "no"
        correction = -mean_z if abs(mean_z) > 1.0 else 0.0

        print(
            f"  {od:>8} {len(z_scores):>4} {mean_z:>+8.3f} {se:>8.3f} "
            f"{sig:>6} {correction:>+12.3f}"
        )

    print("\n  Corrections applied only where |Mean Z| > 1.0 (SE ≈ 0.29 with N=12).")


def part5_conditioning_variance(backtest_dir: str):
    """Compute conditioning variance scaler from backtest std_z."""
    print("\n" + "=" * 65)
    print("PART 5: Conditioning Variance Scaler (from backtest)")
    print("=" * 65)

    thresholds = default_thresholds()

    print(f"\n  {'Obs Days':>8} {'N':>4} {'Std Z':>8} {'Scaler':>8} {'Note':>20}")
    print("  " + "-" * 55)

    all_z = []
    for od in range(8):
        path = Path(backtest_dir) / f"backtest_ar1_asof_day{od}.csv"
        if not path.exists():
            continue
        bt = pd.read_csv(path)
        if len(bt) == 0:
            continue

        z_scores = (bt["model_mean"] - bt["settlement_price"]) / bt["model_std"]
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna()

        if len(z_scores) < 3:
            continue

        std_z = float(z_scores.std())
        all_z.extend(z_scores.tolist())
        note = ""
        if std_z > 2.0:
            note = "OVERCONFIDENT"
        elif std_z < 0.5:
            note = "underconfident"

        # Scaler = current_std_z means we'd multiply sigma by std_z to correct
        print(f"  {od:>8} {len(z_scores):>4} {std_z:>8.3f} {std_z:>8.3f} {note:>20}")

    if all_z:
        pooled_std_z = float(np.std(all_z))
        print(f"\n  Pooled Std Z (all obs_days): {pooled_std_z:.3f}")
        if pooled_std_z > 1.5:
            print(f"  Model is overconfident. Suggested global sigma multiplier: {pooled_std_z:.2f}x")
        elif pooled_std_z < 0.7:
            print(f"  Model is underconfident. Suggested global sigma multiplier: {pooled_std_z:.2f}x")
        else:
            print("  Variance calibration is reasonable.")


def main():
    parser = argparse.ArgumentParser(description="Gas model calibration diagnostics")
    parser.add_argument(
        "--backtest-dir", type=str, default="data",
        help="Directory containing backtest_ar1_asof_day*.csv files"
    )
    parser.add_argument(
        "--data", type=str, default="data/aaa_daily.csv",
        help="AAA daily CSV path"
    )
    args = parser.parse_args()

    print("Loading data...")
    df_all = load_aaa_csv(args.data)
    df_daily = load_daily_prices(args.data)
    df_interp = load_interpolated_daily(args.data)

    # Fit AR(1) on all available daily data
    print("Fitting AR(1) on full daily dataset...")
    train_features = make_features(df_daily.index)
    train_features = add_seasonal_norm(train_features, df_interp["price"])
    ar1 = fit_ar1(df_daily["price"], train_features)

    # Part 0: Historical sanity check (RUN FIRST)
    ratio = part0_historical_sanity_check(df_all, ar1.phi, ar1.sigma)

    # Part 1: Trend bias check
    # Get the dates corresponding to residuals (lagged by 2 from training data)
    train_prices = df_daily["price"].dropna().sort_index()
    diffs = train_prices.diff().dropna()
    lagged = diffs.shift(1).dropna()
    valid_idx = diffs.index.intersection(lagged.index)
    if train_features is not None:
        valid_idx = valid_idx.intersection(train_features.index)
    part1_trend_bias_check(ar1.c, ar1.sigma, ar1.n_obs, ar1.residuals)

    # Part 2: DOW variance ratios
    part2_dow_variance_ratios(ar1.residuals, valid_idx[-len(ar1.residuals):])

    # Part 3: Seasonal note
    part3_seasonal_note()

    # Parts 4-5: Need backtest output
    bt_dir = Path(args.backtest_dir)
    has_backtests = any(
        (bt_dir / f"backtest_ar1_asof_day{od}.csv").exists()
        for od in range(8)
    )

    if has_backtests:
        part4_mean_bias_by_day(args.backtest_dir)
        part5_conditioning_variance(args.backtest_dir)
    else:
        print("\n" + "=" * 65)
        print("PARTS 4-5: Skipped — no backtest output found")
        print(f"  Run: PYTHONPATH=. python scripts/run_backtest.py --all")
        print(f"  Then re-run this script.")
        print("=" * 65)


if __name__ == "__main__":
    main()
