from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Literal, Optional

from .trade_signal import (
    compute_signal_above,
    compute_signal_range,
    ActionType,
)
from .forecast import get_next_saturday


LOG_PATH = Path("data/trade_signals.csv")


@dataclass
class SignalLogRow:
    # metadata
    timestamp: str          # ISO datetime
    market_id: str          # e.g. Kalshi market slug or ID
    contract_type: Literal["above", "range"]

    # contract params
    target_date: str        # ISO date
    threshold: Optional[float]  # for "above"
    range_low: Optional[float]  # for "range"
    range_high: Optional[float]

    # order book
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    fee_per_contract: float

    # model outputs
    mean_forecast: float
    std_forecast: float
    model_prob_yes: float

    ev_buy_yes: float
    ev_sell_yes: float
    ev_buy_no: float
    ev_sell_no: float

    best_action: ActionType
    best_ev: float


def _ensure_log_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[field.name for field in SignalLogRow.__dataclass_fields__.values()],  # type: ignore
        )
        writer.writeheader()


def log_above_contract(
    market_id: str,
    threshold: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    target_date: Optional[date] = None,
    fee_per_contract: float = 0.0,
    ev_threshold: float = 0.02,
    log_path: Path = LOG_PATH,
) -> SignalLogRow:
    """
    Compute signal for an 'above' contract and append to CSV log.
    """
    if target_date is None:
        target_date = get_next_saturday()

    signal = compute_signal_above(
        target_date=target_date,
        threshold=threshold,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        fee_per_contract=fee_per_contract,
        ev_threshold=ev_threshold,
    )

    row = SignalLogRow(
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        market_id=market_id,
        contract_type="above",
        target_date=signal.target_date.isoformat(),
        threshold=threshold,
        range_low=None,
        range_high=None,
        yes_bid=signal.yes_bid,
        yes_ask=signal.yes_ask,
        no_bid=signal.no_bid,
        no_ask=signal.no_ask,
        fee_per_contract=signal.fee_per_contract,
        mean_forecast=signal.mean_forecast,
        std_forecast=signal.std_forecast,
        model_prob_yes=signal.model_prob_yes,
        ev_buy_yes=signal.ev_buy_yes,
        ev_sell_yes=signal.ev_sell_yes,
        ev_buy_no=signal.ev_buy_no,
        ev_sell_no=signal.ev_sell_no,
        best_action=signal.best_action,
        best_ev=signal.best_ev,
    )

    _ensure_log_header(log_path)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[field.name for field in SignalLogRow.__dataclass_fields__.values()],  # type: ignore
        )
        writer.writerow(asdict(row))

    return row


def log_range_contract(
    market_id: str,
    low: float,
    high: float,
    yes_bid: float,
    yes_ask: float,
    no_bid: float,
    no_ask: float,
    target_date: Optional[date] = None,
    fee_per_contract: float = 0.0,
    ev_threshold: float = 0.02,
    log_path: Path = LOG_PATH,
) -> SignalLogRow:
    """
    Compute signal for a 'range' contract and append to CSV log.
    """
    if target_date is None:
        target_date = get_next_saturday()

    signal = compute_signal_range(
        target_date=target_date,
        low=low,
        high=high,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        fee_per_contract=fee_per_contract,
        ev_threshold=ev_threshold,
    )

    row = SignalLogRow(
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
        market_id=market_id,
        contract_type="range",
        target_date=signal.target_date.isoformat(),
        threshold=None,
        range_low=low,
        range_high=high,
        yes_bid=signal.yes_bid,
        yes_ask=signal.yes_ask,
        no_bid=signal.no_bid,
        no_ask=signal.no_ask,
        fee_per_contract=signal.fee_per_contract,
        mean_forecast=signal.mean_forecast,
        std_forecast=signal.std_forecast,
        model_prob_yes=signal.model_prob_yes,
        ev_buy_yes=signal.ev_buy_yes,
        ev_sell_yes=signal.ev_sell_yes,
        ev_buy_no=signal.ev_buy_no,
        ev_sell_no=signal.ev_sell_no,
        best_action=signal.best_action,
        best_ev=signal.best_ev,
    )

    _ensure_log_header(log_path)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[field.name for field in SignalLogRow.__dataclass_fields__.values()],  # type: ignore
        )
        writer.writerow(asdict(row))

    return row


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI usage:

    ABOVE:
      python -m src.pipelines.log_signals above <market_id> <threshold> \
          <yes_bid> <yes_ask> <no_bid> <no_ask> [YYYY-MM-DD] [fee]

    RANGE:
      python -m src.pipelines.log_signals range <market_id> <low> <high> \
          <yes_bid> <yes_ask> <no_bid> <no_ask> [YYYY-MM-DD] [fee]
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 7:
        print("Usage:", file=sys.stderr)
        print(
            "  above <market_id> <threshold> <yes_bid> <yes_ask> <no_bid> <no_ask> [YYYY-MM-DD] [fee]",
            file=sys.stderr,
        )
        print(
            "  range <market_id> <low> <high> <yes_bid> <yes_ask> <no_bid> <no_ask> [YYYY-MM-DD] [fee]",
            file=sys.stderr,
        )
        return 1

    contract_type = argv[0]
    market_id = argv[1]

    try:
        if contract_type == "above":
            threshold = float(argv[2])
            yes_bid = float(argv[3])
            yes_ask = float(argv[4])
            no_bid = float(argv[5])
            no_ask = float(argv[6])

            target: Optional[date] = None
            fee = 0.0

            if len(argv) >= 8:
                if "-" in argv[7]:
                    target = date.fromisoformat(argv[7])
                    fee = float(argv[8]) if len(argv) >= 9 else 0.0
                else:
                    target = None
                    fee = float(argv[7])

            row = log_above_contract(
                market_id=market_id,
                threshold=threshold,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                target_date=target,
                fee_per_contract=fee,
            )

        elif contract_type == "range":
            low = float(argv[2])
            high = float(argv[3])
            yes_bid = float(argv[4])
            yes_ask = float(argv[5])
            no_bid = float(argv[6])
            no_ask = float(argv[7])

            target = None
            fee = 0.0

            if len(argv) >= 9:
                if "-" in argv[8]:
                    target = date.fromisoformat(argv[8])
                    fee = float(argv[9]) if len(argv) >= 10 else 0.0
                else:
                    target = None
                    fee = float(argv[8])

            row = log_range_contract(
                market_id=market_id,
                low=low,
                high=high,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                target_date=target,
                fee_per_contract=fee,
            )
        else:
            raise ValueError(f"Unknown contract type: {contract_type}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # Print a quick summary for you
    print(f"Logged signal at {row.timestamp}")
    print(f"Market ID        : {row.market_id}")
    print(f"Contract type    : {row.contract_type}")
    print(f"Target date      : {row.target_date}")
    if row.contract_type == "above":
        print(f"Threshold        : {row.threshold}")
    else:
        print(f"Range            : [{row.range_low}, {row.range_high}]")
    print(f"Model P(YES)     : {row.model_prob_yes:.3f}")
    print(f"YES bid/ask      : {row.yes_bid:.3f} / {row.yes_ask:.3f}")
    print(f"NO  bid/ask      : {row.no_bid:.3f} / {row.no_ask:.3f}")
    print(f"EV buy YES       : {row.ev_buy_yes:+.3f}")
    print(f"EV sell YES      : {row.ev_sell_yes:+.3f}")
    print(f"EV buy NO        : {row.ev_buy_no:+.3f}")
    print(f"EV sell NO       : {row.ev_sell_no:+.3f}")
    print(f"Best action      : {row.best_action}")
    print(f"Best EV          : {row.best_ev:+.3f}")
    print(f"Appended to      : {LOG_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
