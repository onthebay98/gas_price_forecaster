# src/live/kalshi_watcher.py

import logging
import time
from datetime import datetime
from typing import Optional

from src.pipelines.trade_signal import (  # adjust import if needed
    compute_signal_above,
    compute_signal_range,
    TradeSignal,
)
from src.pipelines.log_signals import (  # adjust import if needed
    log_above_contract,
    log_range_contract,
)

from .config import settings
from .logging_setup import setup_logging
from .metrics import metrics
from .kalshi_client import KalshiClient, KalshiContract
from .alerts import ping

logger = logging.getLogger(__name__)


def format_alert(signal: TradeSignal, market_id: str) -> str:
    return (
        f"[{datetime.utcnow().isoformat(timespec='seconds')}] "
        f"market={market_id} type={signal.contract_type} "
        f"target={signal.target_date} "
        f"p_yes={signal.model_prob_yes:.3f} "
        f"yes_bid={signal.yes_bid:.2f} yes_ask={signal.yes_ask:.2f} "
        f"no_bid={signal.no_bid:.2f} no_ask={signal.no_ask:.2f} "
        f"ev_buy_yes={signal.ev_buy_yes:+.3f} "
        f"ev_sell_yes={signal.ev_sell_yes:+.3f} "
        f"ev_buy_no={signal.ev_buy_no:+.3f} "
        f"ev_sell_no={signal.ev_sell_no:+.3f} "
        f"best={signal.best_action} best_ev={signal.best_ev:+.3f}"
    )


def process_contract(
    kc: KalshiContract,
    fee_per_contract: float,
    ev_threshold: float,
) -> Optional[TradeSignal]:
    """
    Compute EV for a single contract, log evaluation, and trigger alerts if needed.
    """
    if kc.contract_type == "above":
        signal = compute_signal_above(
            target_date=kc.target_date,
            threshold=kc.threshold,  # type: ignore[arg-type]
            yes_bid=kc.yes_bid,
            yes_ask=kc.yes_ask,
            no_bid=kc.no_bid,
            no_ask=kc.no_ask,
            fee_per_contract=fee_per_contract,
            ev_threshold=ev_threshold,
        )
        log_above_contract(
            market_id=kc.market_id,
            threshold=kc.threshold,  # type: ignore[arg-type]
            yes_bid=kc.yes_bid,
            yes_ask=kc.yes_ask,
            no_bid=kc.no_bid,
            no_ask=kc.no_ask,
            target_date=kc.target_date,
            fee_per_contract=fee_per_contract,
            ev_threshold=ev_threshold,
        )
    else:
        signal = compute_signal_range(
            target_date=kc.target_date,
            low=kc.range_low,  # type: ignore[arg-type]
            high=kc.range_high,  # type: ignore[arg-type]
            yes_bid=kc.yes_bid,
            yes_ask=kc.yes_ask,
            no_bid=kc.no_bid,
            no_ask=kc.no_ask,
            fee_per_contract=fee_per_contract,
            ev_threshold=ev_threshold,
        )
        log_range_contract(
            market_id=kc.market_id,
            low=kc.range_low,  # type: ignore[arg-type]
            high=kc.range_high,  # type: ignore[arg-type]
            yes_bid=kc.yes_bid,
            yes_ask=kc.yes_ask,
            no_bid=kc.no_bid,
            no_ask=kc.no_ask,
            target_date=kc.target_date,
            fee_per_contract=fee_per_contract,
            ev_threshold=ev_threshold,
        )

    if signal.best_action != "no_trade" and signal.best_ev >= ev_threshold:
        alert_msg = format_alert(signal, kc.market_id)
        ping(alert_msg)
        return signal

    return None


def run_once(client: KalshiClient) -> None:
    """
    One iteration of:
      - fetch contracts
      - evaluate all
      - send alerts
    """
    try:
        contracts = client.fetch_gas_contracts()
    except NotImplementedError:
        logger.error(
            "fetch_gas_contracts is not implemented. Wire it to Kalshi's API first."
        )
        return
    except Exception as e:
        logger.error("Error fetching contracts: %s", e, exc_info=True)
        return

    fee = settings.fee_per_contract
    ev_threshold = settings.ev_threshold

    for kc in contracts:
        try:
            process_contract(kc, fee_per_contract=fee, ev_threshold=ev_threshold)
        except Exception as e:
            logger.error(
                "Error processing contract %s: %s", kc.market_id, e, exc_info=True
            )


def main() -> int:
    setup_logging()

    logger.info(
        "Kalshi watcher starting. env=%s poll_seconds=%s ev_threshold=%.3f",
        settings.environment,
        settings.poll_seconds,
        settings.ev_threshold,
    )

    if metrics.enabled:
        metrics.start_server()

    client = KalshiClient()

    try:
        while True:
            loop_start = time.time()
            run_once(client)
            loop_duration = time.time() - loop_start
            if metrics.enabled:
                metrics.watcher_loop_duration_seconds.observe(loop_duration)

            # Ensure we don't hammer the API
            sleep_for = max(0.0, settings.poll_seconds - loop_duration)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        logger.info("Kalshi watcher shutting down due to KeyboardInterrupt.")
    except Exception as e:
        logger.error("Fatal error in watcher: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
