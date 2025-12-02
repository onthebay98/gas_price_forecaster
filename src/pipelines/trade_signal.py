import sys
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

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

    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    fee_per_contract: float

    ev_buy_yes: float
    ev_buy_no: float

    best_action: Literal["buy_yes", "buy_no", "no_trade"]


def _build_prices_from_yes(yes_bid: float, yes_ask: float) -> tuple[float, float]:
    """
    Derive NO prices from YES bid/ask using the parity:
      no_bid = 1 - yes_ask
      no_ask = 1 - yes_bid
    """
    if not (0.0 <= yes_bid <= 1.0 and 0.0 <= yes_ask <= 1.0):
        raise ValueError("YES bid/ask must be between 0 and 1")
    if yes_bid > yes_ask:
        raise ValueError("YES bid cannot be greater than YES ask")

    no_bid = 1.0 - yes_ask
    no_ask = 1.0 - yes_bid
    return no_bid, no_ask


def _compute_ev(
    p_yes: float,
    yes_bid: float,
    yes_ask: float,
    fee_per_contract: float,
) -> tuple[float, float, float, float, Literal["buy_yes", "buy_no", "no_trade"]]:
    """
    Compute EV of buying YES vs buying NO, given
    model probability p_yes and YES bid/ask.

    Returns:
      (no_bid, no_ask, ev_buy_yes, ev_buy_no, best_action)
    """
    no_bid, no_ask = _build_prices_from_yes(yes_bid, yes_ask)

    # EV of buying YES at yes_ask and holding to expiry (per $1 contract)
    ev_buy_yes = p_yes - yes_ask - fee_per_contract

    # EV of buying NO at no_ask and holding to expiry (per $1 contract)
    p_no = 1.0 - p_yes
    ev_buy_no = p_no - no_ask - fee_per_contract

    # Choose best action by EV
    if ev_buy_yes <= 0 and ev_buy_no <= 0:
        best_action: Literal["buy_yes", "buy_no", "no_trade"] = "no_trade"
    elif ev_buy_yes >= ev_buy_no:
        best_action = "buy_yes"
    else:
        best_action = "buy_no"

    return no_bid, no_ask, ev_buy_yes, ev_buy_no, best_action


def compute_signal_above(
    target_date: date,
    threshold: float,
    yes_bid: float,
    yes_ask: float,
    fee_per_contract: float = 0.0,
    ev_threshold: float = 0.02,
) -> TradeSignal:
    """
    Contract: 'Will gas be >= threshold on target_date?'

    yes_bid / yes_ask: current order book for YES, in [0,1].
    fee_per_contract: per-contract fee (e.g. 0.01 = 1 cent).
    ev_threshold: minimum EV (in dollars per $1 contract) to issue a trade.
    """
    mean_f, std_f = forecast_price_for_date(target_date)
    model_prob = prob_above_threshold(mean_f, std_f, threshold)

    no_bid, no_ask, ev_buy_yes, ev_buy_no, best_action_raw = _compute_ev(
        model_prob,
        yes_bid,
        yes_ask,
        fee_per_contract,
    )

    # Enforce EV threshold
    if best_action_raw == "buy_yes" and ev_buy_yes < ev_threshold:
        best_action: Literal["buy_yes", "buy_no", "no_trade"] = "no_trade"
    elif best_action_raw == "buy_no" and ev_buy_no < ev_threshold:
        best_action = "no_trade"
    else:
        best_action = best_action_raw

    return TradeSignal(
        contract_type="above",
        target_date=target_date,
        mean_forecast=mean_f,
        std_forecast=std_f,
        model_prob_yes=model_prob,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        fee_per_contract=fee_per_contract,
        ev_buy_yes=ev_buy_yes,
        ev_buy_no=ev_buy_no,
        best_action=best_action,
    )


def compute_signal_range(
    target_date: date,
    low: float,
    high: float,
    yes_bid: float,
    yes_ask: float,
    fee_per_contract: float = 0.0,
    ev_threshold: float = 0.02,
) -> TradeSignal:
    """
    Contract: 'Will gas be between low and high (inclusive) on target_date?'
    """
    mean_f, std_f = forecast_price_for_date(target_date)
    model_prob = prob_in_range(mean_f, std_f, low, high)

    no_bid, no_ask, ev_buy_yes, ev_buy_no, best_action_raw = _compute_ev(
        model_prob,
        yes_bid,
        yes_ask,
        fee_per_contract,
    )

    # Enforce EV threshold
    if best_action_raw == "buy_yes" and ev_buy_yes < ev_threshold:
        best_action: Literal["buy_yes", "buy_no", "no_trade"] = "no_trade"
    elif best_action_raw == "buy_no" and ev_buy_no < ev_threshold:
        best_action = "no_trade"
    else:
        best_action = best_action_raw

    return TradeSignal(
        contract_type="range",
        target_date=target_date,
        mean_forecast=mean_f,
        std_forecast=std_f,
        model_prob_yes=model_prob,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        fee_per_contract=fee_per_contract,
        ev_buy_yes=ev_buy_yes,
        ev_buy_no=ev_buy_no,
        best_action=best_action,
    )


# ---------- CLI entry ----------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Usage examples:

    1) Above-threshold contract (no fee):
        python -m src.pipelines.trade_signal above 3.50 0.55 0.58

    2) Above-threshold with explicit date and fee (1 cent/contract):
        python -m src.pipelines.trade_signal above 3.50 0.55 0.58 2025-12-06 0.01

    3) Range contract:
        python -m src.pipelines.trade_signal range 3.40 3.50 0.45 0.50

    4) Range with explicit date and fee:
        python -m src.pipelines.trade_signal range 3.40 3.50 0.45 0.50 2025-12-06 0.01
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 4:
        print("Usage:", file=sys.stderr)
        print(
            "  above <threshold> <yes_bid> <yes_ask> [YYYY-MM-DD] [fee_per_contract]",
            file=sys.stderr,
        )
        print(
            "  range <low> <high> <yes_bid> <yes_ask> [YYYY-MM-DD] [fee_per_contract]",
            file=sys.stderr,
        )
        return 1

    contract_type = argv[0]
    try:
        if contract_type == "above":
            threshold = float(argv[1])
            yes_bid = float(argv[2])
            yes_ask = float(argv[3])

            # Optional date and fee
            target: date
            fee_per_contract: float

            if len(argv) >= 5:
                # Could be date or fee; detect using simple heuristic
                if "-" in argv[4]:
                    target = date.fromisoformat(argv[4])
                    fee_per_contract = float(argv[5]) if len(argv) >= 6 else 0.0
                else:
                    target = get_next_saturday()
                    fee_per_contract = float(argv[4])
            else:
                target = get_next_saturday()
                fee_per_contract = 0.0

            signal = compute_signal_above(
                target,
                threshold,
                yes_bid,
                yes_ask,
                fee_per_contract=fee_per_contract,
            )

        elif contract_type == "range":
            low = float(argv[1])
            high = float(argv[2])
            yes_bid = float(argv[3])
            yes_ask = float(argv[4])

            if len(argv) >= 6:
                if "-" in argv[5]:
                    target = date.fromisoformat(argv[5])
                    fee_per_contract = float(argv[6]) if len(argv) >= 7 else 0.0
                else:
                    target = get_next_saturday()
                    fee_per_contract = float(argv[5])
            else:
                target = get_next_saturday()
                fee_per_contract = 0.0

            signal = compute_signal_range(
                target,
                low,
                high,
                yes_bid,
                yes_ask,
                fee_per_contract=fee_per_contract,
            )
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"Target date         : {signal.target_date.isoformat()}")
    print(f"Forecast mean       : ${signal.mean_forecast:.3f}")
    print(f"Forecast std dev    : {signal.std_forecast:.4f} (USD)")
    print(f"Model P(YES)        : {signal.model_prob_yes:.3f}")
    print()
    print(f"YES bid / ask       : {signal.yes_bid:.3f} / {signal.yes_ask:.3f}")
    print(f"NO  bid / ask       : {signal.no_bid:.3f} / {signal.no_ask:.3f}")
    print(f"Fee per contract    : {signal.fee_per_contract:.3f}")
    print()
    print(f"EV buy YES (at ask) : {signal.ev_buy_yes:+.3f}")
    print(f"EV buy NO  (at ask) : {signal.ev_buy_no:+.3f}")
    print(f"Suggested action    : {signal.best_action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
