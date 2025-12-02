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

ActionType = Literal["buy_yes", "sell_yes", "buy_no", "sell_no", "no_trade"]


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
    ev_sell_yes: float
    ev_buy_no: float
    ev_sell_no: float

    best_action: ActionType
    best_ev: float


def _compute_all_evs(
    p_yes: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    fee_per_contract: float,
) -> dict[ActionType, float]:
    """
    Compute EV for buying/selling YES/NO given model probability and full
    YES/NO bid/ask.
    """
    for name, v in {
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
    }.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1 (got {v})")

    p_no = 1.0 - p_yes

    # Buy YES at ask
    ev_buy_yes = p_yes - yes_ask - fee_per_contract

    # Sell YES at bid
    ev_sell_yes = yes_bid - p_yes - fee_per_contract

    # Buy NO at ask
    ev_buy_no = p_no - no_ask - fee_per_contract

    # Sell NO at bid
    ev_sell_no = no_bid - p_no - fee_per_contract

    return {
        "buy_yes": ev_buy_yes,
        "sell_yes": ev_sell_yes,
        "buy_no": ev_buy_no,
        "sell_no": ev_sell_no,
        "no_trade": 0.0,
    }


def _select_best_action(
    evs: dict[ActionType, float],
    ev_threshold: float,
) -> tuple[ActionType, float]:
    """
    Pick the action with the highest EV that exceeds ev_threshold.
    If none exceed, return ("no_trade", 0).
    """
    # Ignore no_trade in max calculation
    best_action: ActionType = "no_trade"
    best_ev = 0.0

    for action in ["buy_yes", "sell_yes", "buy_no", "sell_no"]:
        ev = evs[action]  # type: ignore[index]
        if ev > best_ev:
            best_ev = ev
            best_action = action  # type: ignore[assignment]

    if best_ev < ev_threshold:
        return "no_trade", 0.0

    return best_action, best_ev


def compute_signal_above(
    target_date: date,
    threshold: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    fee_per_contract: float = 0.0,
    ev_threshold: float = 0.02,
) -> TradeSignal:
    """
    Contract: 'Will gas be >= threshold on target_date?'
    """
    mean_f, std_f = forecast_price_for_date(target_date)
    model_prob = prob_above_threshold(mean_f, std_f, threshold)

    evs = _compute_all_evs(
        model_prob,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        fee_per_contract=fee_per_contract,
    )

    best_action, best_ev = _select_best_action(evs, ev_threshold)

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
        ev_buy_yes=evs["buy_yes"],
        ev_sell_yes=evs["sell_yes"],
        ev_buy_no=evs["buy_no"],
        ev_sell_no=evs["sell_no"],
        best_action=best_action,
        best_ev=best_ev,
    )


def compute_signal_range(
    target_date: date,
    low: float,
    high: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    fee_per_contract: float = 0.0,
    ev_threshold: float = 0.02,
) -> TradeSignal:
    """
    Contract: 'Will gas be between low and high (inclusive) on target_date?'
    """
    mean_f, std_f = forecast_price_for_date(target_date)
    model_prob = prob_in_range(mean_f, std_f, low, high)

    evs = _compute_all_evs(
        model_prob,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        fee_per_contract=fee_per_contract,
    )

    best_action, best_ev = _select_best_action(evs, ev_threshold)

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
        ev_buy_yes=evs["buy_yes"],
        ev_sell_yes=evs["sell_yes"],
        ev_buy_no=evs["buy_no"],
        ev_sell_no=evs["sell_no"],
        best_action=best_action,
        best_ev=best_ev,
    )


# ---------- CLI entry ----------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Usage examples:

    1) Above-threshold contract (no fee):
       python -m src.pipelines.trade_signal above 2.95 0.59 0.61 0.04 0.06

       -> threshold = 2.95
          YES bid/ask = 0.59 / 0.61
          NO  bid/ask = 0.04 / 0.06
          target date = next Saturday
          fee = 0

    2) Above-threshold with explicit date and fee:
       python -m src.pipelines.trade_signal above 2.95 0.59 0.61 0.04 0.06 2025-12-06 0.01

    3) Range contract:
       python -m src.pipelines.trade_signal range 2.90 3.00 0.45 0.48 0.52 0.55
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 6:
        print("Usage:", file=sys.stderr)
        print(
            "  above <threshold> <yes_bid> <yes_ask> <no_bid> <no_ask> [YYYY-MM-DD] [fee_per_contract]",
            file=sys.stderr,
        )
        print(
            "  range <low> <high> <yes_bid> <yes_ask> <no_bid> <no_ask> [YYYY-MM-DD] [fee_per_contract]",
            file=sys.stderr,
        )
        return 1

    contract_type = argv[0]
    try:
        if contract_type == "above":
            threshold = float(argv[1])
            yes_bid = float(argv[2])
            yes_ask = float(argv[3])
            no_bid = float(argv[4])
            no_ask = float(argv[5])

            target: date
            fee_per_contract: float

            if len(argv) >= 7:
                if "-" in argv[6]:
                    target = date.fromisoformat(argv[6])
                    fee_per_contract = float(argv[7]) if len(argv) >= 8 else 0.0
                else:
                    target = get_next_saturday()
                    fee_per_contract = float(argv[6])
            else:
                target = get_next_saturday()
                fee_per_contract = 0.0

            signal = compute_signal_above(
                target,
                threshold,
                yes_bid,
                yes_ask,
                no_bid,
                no_ask,
                fee_per_contract=fee_per_contract,
            )

        elif contract_type == "range":
            low = float(argv[1])
            high = float(argv[2])
            yes_bid = float(argv[3])
            yes_ask = float(argv[4])
            no_bid = float(argv[5])
            no_ask = float(argv[6])

            if len(argv) >= 8:
                if "-" in argv[7]:
                    target = date.fromisoformat(argv[7])
                    fee_per_contract = float(argv[8]) if len(argv) >= 9 else 0.0
                else:
                    target = get_next_saturday()
                    fee_per_contract = float(argv[7])
            else:
                target = get_next_saturday()
                fee_per_contract = 0.0

            signal = compute_signal_range(
                target,
                low,
                high,
                yes_bid,
                yes_ask,
                no_bid,
                no_ask,
                fee_per_contract=fee_per_contract,
            )
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"Target date          : {signal.target_date.isoformat()}")
    print(f"Forecast mean        : ${signal.mean_forecast:.3f}")
    print(f"Forecast std dev     : {signal.std_forecast:.4f} (USD)")
    print(f"Model P(YES)         : {signal.model_prob_yes:.3f}")
    print()
    print(f"YES bid / ask        : {signal.yes_bid:.3f} / {signal.yes_ask:.3f}")
    print(f"NO  bid / ask        : {signal.no_bid:.3f} / {signal.no_ask:.3f}")
    print(f"Fee per contract     : {signal.fee_per_contract:.3f}")
    print()
    print(f"EV buy YES  (ask)    : {signal.ev_buy_yes:+.3f}")
    print(f"EV sell YES (bid)    : {signal.ev_sell_yes:+.3f}")
    print(f"EV buy NO   (ask)    : {signal.ev_buy_no:+.3f}")
    print(f"EV sell NO  (bid)    : {signal.ev_sell_no:+.3f}")
    print()
    print(f"Best action          : {signal.best_action}")
    print(f"Best EV              : {signal.best_ev:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
