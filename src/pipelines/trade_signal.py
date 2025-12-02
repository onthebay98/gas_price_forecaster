import sys
from dataclasses import dataclass
from datetime import date
from typing import Literal, Tuple, Optional

import numpy as np
import math

from .forecast import forecast_price_for_date, get_next_saturday


# ---------- probability helpers ----------

def normal_cdf(x: float) -> float:
    """
    Standard normal CDF Φ(x) using the error function.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))



def prob_above_threshold(mean: float, std: float, k: float) -> float:
    """
    P(X >= k) for X ~ N(mean, std^2).
    """
    if std <= 0:
        # Degenerate: all mass at mean
        return 1.0 if mean >= k else 0.0

    z = (k - mean) / std
    # P(X >= k) = 1 - Φ((k - μ) / σ)
    return 1.0 - normal_cdf(z)


def prob_in_range(mean: float, std: float, low: float, high: float) -> float:
    """
    P(low <= X <= high) for X ~ N(mean, std^2).
    """
    if low > high:
        raise ValueError("low must be <= high")

    if std <= 0:
        return 1.0 if (low <= mean <= high) else 0.0

    z_low = (low - mean) / std
    z_high = (high - mean) / std
    return normal_cdf(z_high) - normal_cdf(z_low)


# ---------- trade signal logic ----------

@dataclass
class TradeSignal:
    contract_type: Literal["above", "range"]
    target_date: date
    mean_forecast: float
    std_forecast: float
    model_prob_yes: float
    kalshi_yes_price: float
    edge: float  # model_prob_yes - market_price
    action: Literal["buy_yes", "sell_yes", "no_trade"]


def compute_signal_above(
    target_date: date,
    threshold: float,
    kalshi_yes_price: float,
    edge_threshold: float = 0.10,
) -> TradeSignal:
    """
    Contract: 'Will gas be >= threshold on target_date?'

    kalshi_yes_price: price of YES in [0,1] (e.g. 0.38 for 38¢)
    edge_threshold: minimum absolute edge to trade (e.g. 0.10 = 10pp).
    """
    mean_f, std_f = forecast_price_for_date(target_date)
    model_prob = prob_above_threshold(mean_f, std_f, threshold)
    edge = model_prob - kalshi_yes_price

    if edge > edge_threshold:
        action: Literal["buy_yes", "sell_yes", "no_trade"] = "buy_yes"
    elif edge < -edge_threshold:
        action = "sell_yes"
    else:
        action = "no_trade"

    return TradeSignal(
        contract_type="above",
        target_date=target_date,
        mean_forecast=mean_f,
        std_forecast=std_f,
        model_prob_yes=model_prob,
        kalshi_yes_price=kalshi_yes_price,
        edge=edge,
        action=action,
    )


def compute_signal_range(
    target_date: date,
    low: float,
    high: float,
    kalshi_yes_price: float,
    edge_threshold: float = 0.10,
) -> TradeSignal:
    """
    Contract: 'Will gas be between low and high (inclusive) on target_date?'
    """
    mean_f, std_f = forecast_price_for_date(target_date)
    model_prob = prob_in_range(mean_f, std_f, low, high)
    edge = model_prob - kalshi_yes_price

    if edge > edge_threshold:
        action: Literal["buy_yes", "sell_yes", "no_trade"] = "buy_yes"
    elif edge < -edge_threshold:
        action = "sell_yes"
    else:
        action = "no_trade"

    return TradeSignal(
        contract_type="range",
        target_date=target_date,
        mean_forecast=mean_f,
        std_forecast=std_f,
        model_prob_yes=model_prob,
        kalshi_yes_price=kalshi_yes_price,
        edge=edge,
        action=action,
    )


# ---------- CLI entry ----------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Usage examples:

    1) Above-threshold contract:
        python -m src.pipelines.trade_signal above 3.50 0.38

       -> interprets 0.38 as YES price, target date = next Saturday

    2) Above-threshold with explicit date:
        python -m src.pipelines.trade_signal above 3.50 0.38 2025-12-06

    3) Range contract:
        python -m src.pipelines.trade_signal range 3.40 3.50 0.30

    4) Range with explicit date:
        python -m src.pipelines.trade_signal range 3.40 3.50 0.30 2025-12-06
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 3:
        print("Usage:", file=sys.stderr)
        print("  above <threshold> <kalshi_yes_price> [YYYY-MM-DD]", file=sys.stderr)
        print("  range <low> <high> <kalshi_yes_price> [YYYY-MM-DD]", file=sys.stderr)
        return 1

    contract_type = argv[0]
    try:
        if contract_type == "above":
            threshold = float(argv[1])
            kalshi_yes_price = float(argv[2])
            target = (
                date.fromisoformat(argv[3])
                if len(argv) >= 4
                else get_next_saturday()
            )
            signal = compute_signal_above(target, threshold, kalshi_yes_price)
        elif contract_type == "range":
            low = float(argv[1])
            high = float(argv[2])
            kalshi_yes_price = float(argv[3])
            target = (
                date.fromisoformat(argv[4])
                if len(argv) >= 5
                else get_next_saturday()
            )
            signal = compute_signal_range(target, low, high, kalshi_yes_price)
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"Target date       : {signal.target_date.isoformat()}")
    print(f"Forecast mean     : ${signal.mean_forecast:.3f}")
    print(f"Forecast std dev  : {signal.std_forecast:.4f} (USD)")
    print(f"Model P(YES)      : {signal.model_prob_yes:.3f}")
    print(f"Kalshi YES price  : {signal.kalshi_yes_price:.3f}")
    print(f"Edge (model - mkt): {signal.edge:+.3f}")
    print(f"Suggested action  : {signal.action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
