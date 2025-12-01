import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

DATA_PATH = Path("data/aaa_daily.csv")


def load_aaa_series() -> pd.DataFrame:
    """
    Load the AAA daily national regular gas price series.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'price'], sorted ascending by date.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run src.pipelines.log_aaa first to collect data."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    if df.empty:
        raise RuntimeError(f"{DATA_PATH} is empty.")

    df = df.sort_values("date").reset_index(drop=True)
    if "price" not in df.columns:
        raise RuntimeError(f"'price' column missing in {DATA_PATH}")

    return df


def get_next_saturday(from_date: Optional[date] = None) -> date:
    """
    Return the next Saturday (including today if today is Saturday).
    """
    if from_date is None:
        from_date = date.today()

    weekday = from_date.weekday()  # Monday=0 ... Sunday=6
    days_until_sat = (5 - weekday) % 7  # Saturday=5
    return from_date + timedelta(days=days_until_sat)


def forecast_price_for_date(target_date: date) -> Tuple[float, float]:
    """
    Very simple baseline forecast for the AAA national regular gas price
    on a given target date.

    Method:
    - Use recent daily price changes (last up to 14 days).
    - Compute mean daily change and standard deviation.
    - Assume changes are i.i.d. and approximate future distribution as:
        price_T ~ Normal(last_price + mean_delta * N, std_delta * sqrt(N))

    Parameters
    ----------
    target_date : date
        Date to forecast for.

    Returns
    -------
    mean_forecast : float
    std_forecast : float
        Standard deviation of forecast (used later to derive probabilities).

    Notes
    -----
    This is intentionally simple; we will replace it with a more
    sophisticated model later.
    """
    df = load_aaa_series()
    last_date = df["date"].iloc[-1].date()
    last_price = float(df["price"].iloc[-1])

    if target_date < last_date:
        raise ValueError(
            f"target_date {target_date} is before last available data {last_date}"
        )

    days_ahead = (target_date - last_date).days

    if days_ahead == 0:
        # Forecast is just the last observed price, zero uncertainty baseline.
        return last_price, 0.0

    # Use up to the last 14 daily deltas
    df = df.sort_values("date")
    df["delta"] = df["price"].diff()
    deltas = df["delta"].dropna().tail(14)

    if deltas.empty:
        # Not enough history; fallback to naive assumption: no drift, small std
        mean_delta = 0.0
        std_delta = 0.01  # 1 cent per day as a placeholder
    else:
        mean_delta = float(deltas.mean())
        std_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.01

        # Guard against zero std (completely flat prices)
        if std_delta == 0:
            std_delta = 0.01

    mean_forecast = last_price + mean_delta * days_ahead
    std_forecast = std_delta * np.sqrt(days_ahead)

    return mean_forecast, std_forecast


def main(argv=None) -> int:
    """
    CLI entry point.

    Usage:
        python -m src.pipelines.forecast        # forecast for next Saturday
        python -m src.pipelines.forecast 2025-12-06  # forecast for specific date
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        target = get_next_saturday()
    elif len(argv) == 1:
        try:
            target = date.fromisoformat(argv[0])
        except ValueError:
            print(f"Invalid date format: {argv[0]!r}. Use YYYY-MM-DD.", file=sys.stderr)
            return 1
    else:
        print("Usage: python -m src.pipelines.forecast [YYYY-MM-DD]", file=sys.stderr)
        return 1

    try:
        mean_f, std_f = forecast_price_for_date(target)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"Target date: {target.isoformat()}")
    print(f"Forecast mean price: ${mean_f:.3f}")
    print(f"Forecast std dev  : {std_f:.4f} (USD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
