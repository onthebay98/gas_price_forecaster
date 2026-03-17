#!/usr/bin/env python3
"""CLI to run gas price backtests.

Usage:
    PYTHONPATH=. python scripts/run_backtest.py --observed-days 0
    PYTHONPATH=. python scripts/run_backtest.py --observed-days 3 --n-sims 10000
    PYTHONPATH=. python scripts/run_backtest.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest import (
    check_threshold_boundaries,
    compute_backtest_metrics,
    default_thresholds,
    run_conditioned_backtest,
)
from src.io import load_daily_prices, load_interpolated_daily


def run_single(
    observed_days: int,
    args: argparse.Namespace,
    df_daily: pd.DataFrame,
    df_interp: pd.DataFrame,
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict]:
    """Run backtest for a single observed_days level."""
    # Pass zero biases during backtest to measure raw model performance
    zero_bias = {i: 0.0 for i in range(8)}

    bt = run_conditioned_backtest(
        df_daily=df_daily,
        df_interp=df_interp,
        observed_days=observed_days,
        start_date=args.start,
        end_date=args.end,
        thresholds=thresholds,
        n_sims=args.n_sims,
        min_train_obs=args.min_train,
        mean_bias_by_day=zero_bias,
    )

    metrics = compute_backtest_metrics(bt, thresholds)

    # Save CSV
    output = args.output or f"data/backtest_ar1_asof_day{observed_days}.csv"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(output, index=False)

    return bt, metrics


def print_metrics(observed_days: int, metrics: dict, n_weeks: int) -> None:
    """Print summary for a single observed_days level."""
    print(f"\nGas Price AR(1) Backtest — observed_days={observed_days}")
    print(f"  Weeks evaluated: {n_weeks}")
    if metrics["n"] == 0:
        print("  No data")
        return
    print(f"  MAE:     ${metrics['mae']:.4f}")
    print(f"  MAPE:    {metrics['mape']:.2f}%")
    print(f"  Brier:   {metrics['brier']:.4f}")
    print(f"  Dir Acc: {metrics['dir_acc']:.1f}%")
    print(f"  Mean Z:  {metrics['mean_z']:+.3f}")
    print(f"  Std Z:   {metrics['std_z']:.3f}")


def print_comparison_table(all_metrics: dict[int, dict], naive_metrics: dict) -> None:
    """Print comparison table across all asof_days with naive baseline."""
    print("\n" + "=" * 75)
    print("Gas AR(1) Backtest — Comparison Table")
    print("Naive baseline: AR(1) at observed_days=0 (zero conditioning)")
    print("=" * 75)
    print(
        f"{'Obs Days':>8} {'N':>4} {'MAE':>8} {'MAPE':>7} "
        f"{'Brier':>8} {'vs Naive':>9} {'Dir Acc':>8} "
        f"{'Mean Z':>8} {'Std Z':>7}"
    )
    print("-" * 75)

    naive_brier = naive_metrics.get("brier", float("nan"))

    for od in sorted(all_metrics.keys()):
        m = all_metrics[od]
        if m["n"] == 0:
            continue
        brier = m["brier"]
        if np.isfinite(naive_brier) and np.isfinite(brier) and naive_brier > 0:
            improvement = (brier - naive_brier) / naive_brier * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "—"

        print(
            f"{od:>8} {m['n']:>4} ${m['mae']:.4f} {m['mape']:>6.2f}% "
            f"{brier:>8.4f} {imp_str:>9} {m['dir_acc']:>7.1f}% "
            f"{m['mean_z']:>+7.3f} {m['std_z']:>7.3f}"
        )
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Run gas price backtests")
    parser.add_argument(
        "--observed-days", type=int, default=None,
        help="Number of observed days (0-7). Use --all to run all."
    )
    parser.add_argument("--all", action="store_true", help="Run for all observed_days 0-7")
    parser.add_argument("--n-sims", type=int, default=25_000, help="Monte Carlo simulations")
    parser.add_argument("--start", type=str, default="2025-12-15", help="First Monday")
    parser.add_argument("--end", type=str, default="2026-03-16", help="Last Monday")
    parser.add_argument("--min-train", type=int, default=14, help="Min training obs")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--data", type=str, default="data/aaa_daily.csv", help="Data CSV")
    args = parser.parse_args()

    if not args.all and args.observed_days is None:
        parser.error("Specify --observed-days N or --all")

    # Load data once
    print("Loading data...")
    df_daily = load_daily_prices(args.data)
    df_interp = load_interpolated_daily(args.data)
    thresholds = default_thresholds()

    # Check grid boundaries
    last_price = float(df_daily["price"].dropna().iloc[-1])
    check_threshold_boundaries(thresholds, last_price)
    print(f"Last price: ${last_price:.3f}")
    print(f"Thresholds: {len(thresholds)} strikes (${min(thresholds):.2f} to ${max(thresholds):.2f})")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Simulations: {args.n_sims:,}")

    if args.all:
        all_metrics = {}
        naive_metrics = None

        for od in range(8):
            print(f"\n--- Running observed_days={od} ---")
            # Override output path per asof_day
            args_copy = argparse.Namespace(**vars(args))
            args_copy.output = f"data/backtest_ar1_asof_day{od}.csv"
            bt, metrics = run_single(od, args_copy, df_daily, df_interp, thresholds)
            all_metrics[od] = metrics
            print_metrics(od, metrics, len(bt))

            if od == 0:
                naive_metrics = metrics

        print_comparison_table(all_metrics, naive_metrics or {})

    else:
        bt, metrics = run_single(args.observed_days, args, df_daily, df_interp, thresholds)
        print_metrics(args.observed_days, metrics, len(bt))


if __name__ == "__main__":
    main()
