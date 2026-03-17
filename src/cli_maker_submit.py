"""CLI for automated market making order submission to Kalshi for gas prices.

Usage:
    # Preview orders without submitting
    python -m src.cli_maker_submit --dry-run

    # Submit validated orders
    python -m src.cli_maker_submit

    # Update existing orders if outbid (stay competitive)
    python -m src.cli_maker_submit --update

    # Resize orders to match Kelly
    python -m src.cli_maker_submit --resize

    # Cancel orders where EV went negative
    python -m src.cli_maker_submit --prune

    # Place exit (sell) orders for held positions
    python -m src.cli_maker_submit --place-exits

    # Kill switch - cancel all open orders
    python -m src.cli_maker_submit --cancel-all

    # List open orders / positions / balance
    python -m src.cli_maker_submit --list-open
    python -m src.cli_maker_submit --positions
    python -m src.cli_maker_submit --balance

Safeguards:
    - Unified validate_orders() with probability, EV, and robustness gates
    - Post-only orders (ensures maker fee)
    - Opposite-position blocking
    - Duplicate order prevention
    - Cash balance check (min $500)
    - Kill switch (--cancel-all)
    - Dry run mode (--dry-run)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.kalshi_client import KalshiClient
from src.orderbook import fetch_orderbook
from src.trade import (
    FeeSchedule,
    calc_ev_if_filled_c,
    ev_robust_under_slippage,
    generate_exit_orders,
    kalshi_fee_total_dollars,
)
from src.config import (
    DEFAULT_MAKER_DEFENSIVE_CONFIG,
    DEFAULT_MAKER_EXIT_CONFIG,
    TEMP_BLOCKED_THRESHOLDS,
    get_maker_min_ev_for_zone,
    get_maker_min_prob_for_day,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SubmissionSummary:
    """Summary of order submission session."""

    timestamp_utc: str
    mode: str  # "production", "dry_run"
    orders_attempted: int = 0
    orders_succeeded: int = 0
    orders_failed: int = 0
    orders_skipped: int = 0
    total_contracts: int = 0
    total_exposure_cents: int = 0
    results: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def is_model_data_fresh(
    meta_path: str = "data/latest_predict_meta.json",
    max_staleness_days: int = 1,
) -> tuple[bool, str]:
    """Check if the model was built on recent data.

    If the AAA scraper silently fails, the model conditions on stale prices
    and edge calculations are wrong. This gate prevents submitting orders
    against a model built on outdated data.

    Args:
        meta_path: Path to forecast metadata JSON
        max_staleness_days: Maximum allowed age of last observation (default: 1 day)

    Returns:
        (is_fresh, reason)
    """
    from datetime import date

    p = Path(meta_path)
    if not p.exists():
        return False, f"No metadata file: {meta_path}"

    try:
        meta = json.loads(p.read_text())
    except Exception as e:
        return False, f"Could not read metadata: {e}"

    last_obs = meta.get("last_obs_date")
    if not last_obs:
        return False, "No last_obs_date in metadata"

    try:
        last_obs_date = date.fromisoformat(last_obs)
    except ValueError:
        return False, f"Invalid last_obs_date: {last_obs}"

    today = date.today()
    staleness = (today - last_obs_date).days

    if staleness > max_staleness_days:
        return False, (
            f"Model data is {staleness} days stale "
            f"(last observation: {last_obs_date}, today: {today}). "
            f"AAA scraper may have failed. Run fetch_aaa.py and re-predict."
        )

    return True, f"Data fresh (last obs: {last_obs_date}, {staleness}d ago)"


def load_maker_orders(
    path: str = "data/latest_maker_orders.json",
    meta_path: str = "data/latest_predict_meta.json",
) -> tuple[list[dict], dict]:
    """Load maker orders from JSON file and associated metadata.

    Returns:
        Tuple of (order_list, metadata_dict)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Maker orders file not found: {path}")

    data = json.loads(p.read_text())

    # Handle both list format (from cli_predict --save) and dict format
    if isinstance(data, list):
        orders = data
    else:
        orders = data.get("maker_orders", [])

    # Load metadata
    metadata = {}
    mp = Path(meta_path)
    if mp.exists():
        try:
            metadata = json.loads(mp.read_text())
        except Exception:
            pass

    return orders, metadata


def deduplicate_orders(
    open_orders: list[dict],
    client: Optional[KalshiClient] = None,
    dry_run: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Deduplicate orders by (ticker, side), keeping best-priced.

    YES side: keep highest-priced (most aggressive bid).
    NO side: keep highest NO price (most aggressive bid).
    """
    from collections import defaultdict

    cancelled_results = []

    orders_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for o in open_orders:
        if o.get("remaining_count", 0) > 0 and o.get("action", "").lower() != "sell":
            key = (o.get("ticker", ""), o.get("side", "").upper())
            orders_by_key[key].append(o)

    deduped_orders = []
    for (ticker, side), group in orders_by_key.items():
        if len(group) > 1:
            if side == "YES":
                group.sort(key=lambda x: x.get("yes_price", 0), reverse=True)
            else:
                group.sort(key=lambda x: 100 - x.get("yes_price", 0), reverse=True)

            best = group[0]
            deduped_orders.append(best)

            for dup in group[1:]:
                dup_id = dup.get("order_id")
                dup_price = dup.get("yes_price", 0) if side == "YES" else 100 - dup.get("yes_price", 0)
                logger.warning(f"  {ticker} {side}: Cancelling duplicate order @ {dup_price}c")

                if not dry_run and client is not None:
                    cancel_result = client.cancel_order(dup_id)
                    if cancel_result.success:
                        logger.info(f"    Cancelled duplicate {dup_id}")
                    else:
                        logger.error(f"    Failed to cancel duplicate: {cancel_result.error_message}")

                cancelled_results.append({
                    "ticker": ticker,
                    "side": side,
                    "action": "cancelled_duplicate",
                    "price": dup_price,
                    "contracts": dup.get("remaining_count", 0),
                    "order_id": dup_id,
                    "dry_run": dry_run,
                })
        else:
            deduped_orders.append(group[0])

    return deduped_orders, cancelled_results


def validate_orders(
    orders: list[dict],
    observed_days: int,
    max_contracts: int = 999999,
    config=None,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Validate maker orders with all filters and gates in one pass.

    Filters applied (in order):
    1. Temp blocklist (TEMP_BLOCKED_THRESHOLDS)
    2. Day-of-week probability threshold
    3. Recalculate EV with real fees
    4. Continuous EV threshold: min_ev = 0.5 * (100 - p) cents
    5. EV robustness - edge survives +1 tick slippage
    6. Contract clipping to max_contracts
    """
    from zoneinfo import ZoneInfo

    if config is None:
        config = DEFAULT_MAKER_DEFENSIVE_CONFIG

    accepted = []
    skipped = []

    now_et = datetime.now(ZoneInfo("America/New_York"))
    weekday = now_et.weekday()
    min_prob_for_day = get_maker_min_prob_for_day(weekday) * 100

    orders_sorted = sorted(
        orders, key=lambda o: o.get("ev_if_filled_c", 0), reverse=True
    )

    for order in orders_sorted:
        ticker = order.get("ticker", "?")
        side = order.get("side", "YES").upper()
        limit_price_c = float(order.get("limit_price_c", 50))
        p_model_pct = float(order.get("p_model", 50))

        contracts_from_json = order.get("contracts", 100)
        contracts = min(max_contracts, contracts_from_json)

        # --- Filter 1: Temp blocklist ---
        if TEMP_BLOCKED_THRESHOLDS:
            blocked = any(t in ticker for t in TEMP_BLOCKED_THRESHOLDS)
            if blocked:
                skipped.append({**order, "skip_reason": "Temporarily blocked"})
                continue

        # --- Filter 2: Day-of-week probability threshold ---
        if p_model_pct < min_prob_for_day:
            skipped.append({
                **order,
                "skip_reason": f"P(side) {p_model_pct:.1f}% < day min {min_prob_for_day:.0f}%"
            })
            continue

        # Convert p_model to P(YES) for EV calculation
        if side == "YES":
            p_model = p_model_pct / 100.0
        else:
            p_model = 1.0 - (p_model_pct / 100.0)

        # --- Filter 3: Recalculate EV with real fees ---
        ev_c = calc_ev_if_filled_c(
            p_model=p_model,
            side=side,
            price_paid_c=limit_price_c,
            is_maker=True,
            qty=contracts,
        )

        # --- Filter 4: Zone-aware EV threshold ---
        effective_min_ev, zone_label = get_maker_min_ev_for_zone(p_model_pct)
        if ev_c < effective_min_ev:
            reason = f"EV {ev_c:.2f}c < min {effective_min_ev:.1f}c ({zone_label})"
            if verbose:
                logger.info(f"  {ticker} {side}: SKIP - {reason}")
            skipped.append({**order, "skip_reason": reason, "v4_ev_c": ev_c})
            continue

        # --- Filter 5: EV robustness ---
        is_robust, worst_ev = ev_robust_under_slippage(
            p_model=p_model,
            side=side,
            price_c=limit_price_c,
            is_maker=True,
            slippage_ticks=1,
            min_ev_c=0.0,
        )
        if not is_robust:
            reason = f"Fragile edge: EV after +1 tick = {worst_ev:.2f}c"
            if verbose:
                logger.info(f"  {ticker} {side}: SKIP - {reason}")
            skipped.append({**order, "skip_reason": reason, "v4_worst_ev": worst_ev})
            continue

        # --- All filters passed ---
        current_bid_c = float(order.get("current_bid_c", 0))
        current_ask_c = float(order.get("current_ask_c", 100))
        spread_c = current_ask_c - current_bid_c

        order["contracts_to_submit"] = contracts
        order["v4_ev_c"] = round(ev_c, 2)
        order["v4_spread_c"] = round(spread_c, 1)

        if verbose:
            logger.info(
                f"  {ticker} {side}: PASS - EV={ev_c:.2f}c, spread={spread_c:.0f}c, "
                f"contracts={contracts}"
            )

        accepted.append(order)

    return accepted, skipped


def submit_orders(
    client: KalshiClient,
    orders: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """Submit orders to Kalshi API."""
    results = []

    try:
        balance_data = client.get_balance()
        available_cash_cents = balance_data.get("balance", 0)
        logger.info(f"Available cash: ${available_cash_cents / 100:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch balance: {e}. Proceeding without balance check.")
        available_cash_cents = float('inf')

    for order in orders:
        ticker = order.get("ticker")
        side = order.get("side", "YES").lower()
        price_c = int(order.get("limit_price_c", 50))
        contracts = order.get("contracts_to_submit", 100)
        ev_c = order.get("ev_if_filled_c", 0)

        order_cost_cents = price_c * contracts

        # Reduce contracts if insufficient cash
        if order_cost_cents > available_cash_cents:
            usable_cash = int(available_cash_cents * 0.99)
            contracts = usable_cash // price_c
            if contracts <= 0:
                logger.warning(
                    f"Skipping {ticker} {side.upper()}: insufficient cash "
                    f"(need {price_c}c, have ${available_cash_cents / 100:.2f})"
                )
                results.append({
                    **order,
                    "submitted": False,
                    "success": False,
                    "error_message": "insufficient cash",
                    "contracts_to_submit": 0,
                })
                continue
            logger.info(
                f"Reducing {ticker} {side.upper()} from {order.get('contracts_to_submit', 0)} to {contracts} "
                f"(cash: ${available_cash_cents / 100:.2f})"
            )
            order_cost_cents = price_c * contracts

        # Convert NO price to YES price for API
        if side == "yes":
            yes_price = price_c
        else:
            yes_price = 100 - price_c

        if dry_run:
            logger.info(
                f"[DRY RUN] Would submit: {ticker} BUY {contracts}x {side.upper()} @ {price_c}c (EV: {ev_c:.2f}c)"
            )
            results.append({
                **order,
                "submitted": False,
                "dry_run": True,
                "success": None,
                "contracts_to_submit": contracts,
            })
            available_cash_cents -= order_cost_cents
            continue

        logger.info(
            f"Submitting: {ticker} BUY {contracts}x {side.upper()} @ {price_c}c (EV: {ev_c:.2f}c)"
        )

        result = client.create_order(
            ticker=ticker,
            side=side,
            action="buy",
            count=contracts,
            yes_price=yes_price,
            post_only=True,
        )

        results.append({
            **order,
            "submitted": True,
            "success": result.success,
            "order_id": result.order_id,
            "client_order_id": result.client_order_id,
            "status": result.status,
            "error_message": result.error_message,
            "error_code": result.error_code,
            "contracts_to_submit": contracts,
        })

        if result.success:
            logger.info(f"  SUCCESS: order_id={result.order_id}, status={result.status}")
            available_cash_cents -= order_cost_cents
        else:
            logger.error(f"  FAILED: {result.error_message} (code: {result.error_code})")

    return results


def update_orders(
    client: KalshiClient,
    maker_orders: list[dict],
    min_ev_c: float,
    min_prob_pct: float,
    dry_run: bool = False,
) -> list[dict]:
    """Update existing orders if outbid but still have positive EV.

    For each open order:
    1. Check current orderbook
    2. If someone has a better bid, calculate if we can beat them with +EV
    3. If yes, cancel old order and place new one at (best_bid + 1)
    """
    results = []
    open_orders = client.get_open_orders()

    if not open_orders:
        logger.info("No open orders to update")
        return results

    try:
        balance_data = client.get_balance()
        available_cash_cents = balance_data.get("balance", 0)
        logger.info(f"Available cash for updates: ${available_cash_cents / 100:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch balance: {e}")
        available_cash_cents = float('inf')

    # Build probability lookup from maker_orders
    prob_lookup = {}
    for mo in maker_orders:
        ticker = mo.get("ticker")
        side = mo.get("side", "").upper()
        prob_lookup[(ticker, side)] = mo.get("p_model", 0)

    # Deduplicate
    deduped_orders, dup_results = deduplicate_orders(open_orders, client, dry_run)
    results.extend(dup_results)

    logger.info(f"Checking {len(deduped_orders)} open orders for updates...")

    for order in deduped_orders:
        ticker = order.get("ticker")
        order_id = order.get("order_id")
        side = order.get("side", "").upper()
        our_price = order.get("yes_price", 0)
        remaining = order.get("remaining_count", 0)

        if remaining == 0:
            continue

        if order.get("action", "").lower() == "sell":
            continue

        if side == "NO":
            our_side_price = 100 - our_price
        else:
            our_side_price = our_price

        orderbook = fetch_orderbook(ticker)
        if not orderbook:
            logger.warning(f"  {ticker}: Could not fetch orderbook, skipping")
            continue

        if side == "YES":
            bids = orderbook.get("yes", [])
        else:
            bids = orderbook.get("no", [])

        if not bids:
            continue

        other_bids = [b[0] for b in bids if b[0] != our_side_price]
        next_best_bid = max(other_bids) if other_bids else 0

        target_price = min(99, max(1, next_best_bid + 1))

        if our_side_price == target_price:
            continue

        if our_side_price < target_price:
            direction = "outbid"
            direction_desc = f"Outbid! {our_side_price}c -> {next_best_bid}c"
        else:
            direction = "overpaying"
            direction_desc = f"Overpaying ({our_side_price}c vs next best {next_best_bid}c)"

        new_price = target_price

        p_model = prob_lookup.get((ticker, side), 0)
        if p_model < min_prob_pct:
            continue

        fee = kalshi_fee_total_dollars(
            price_dollars=new_price / 100.0,
            contracts=remaining,
            is_maker=True,
        )
        fee_per_c = (fee / remaining) * 100 if remaining > 0 else 0
        ev_c = (p_model / 100.0) * 100 - new_price - fee_per_c

        if ev_c < min_ev_c:
            logger.info(
                f"  {ticker} {side}: {direction_desc}, "
                f"but new price {new_price}c has EV {ev_c:.2f}c < min {min_ev_c:.1f}c"
            )
            continue

        logger.info(
            f"  {ticker} {side}: {direction_desc}. "
            f"Repricing {our_side_price}c -> {new_price}c (EV: {ev_c:.2f}c)"
        )

        if dry_run:
            results.append({
                "ticker": ticker,
                "side": side,
                "action": "would_update",
                "direction": direction,
                "old_price": our_side_price,
                "new_price": new_price,
                "contracts": remaining,
                "ev_c": ev_c,
                "dry_run": True,
            })
            continue

        # Cancel old order
        cancel_result = client.cancel_order(order_id)
        if not cancel_result.success:
            logger.error(f"    Failed to cancel order {order_id}: {cancel_result.error_message}")
            results.append({
                "ticker": ticker,
                "side": side,
                "action": "cancel_failed",
                "error": cancel_result.error_message,
            })
            continue

        logger.info(f"    Cancelled order {order_id}")

        # Cash check for new order
        order_cost_cents = new_price * remaining
        contracts_to_place = remaining

        if order_cost_cents > available_cash_cents:
            usable_cash = int(available_cash_cents * 0.99)
            contracts_to_place = usable_cash // new_price

            if contracts_to_place <= 0:
                # Try to restore original
                original_cost = our_side_price * remaining
                if original_cost <= available_cash_cents * 0.99:
                    logger.warning(f"    Can't afford new price, restoring at old price")
                    restore_order = client.create_order(
                        ticker=ticker, side=side.lower(), action="buy",
                        count=remaining, yes_price=our_price, post_only=True,
                    )
                    if restore_order.success:
                        results.append({
                            "ticker": ticker, "side": side,
                            "action": "update_skipped_restored",
                            "old_price": our_side_price,
                            "reason": "insufficient cash for new price",
                        })
                    else:
                        logger.error(f"    CRITICAL: Failed to restore: {restore_order.error_message}")
                        results.append({
                            "ticker": ticker, "side": side,
                            "action": "update_failed_no_restore",
                            "error": f"Restore failed: {restore_order.error_message}",
                        })
                else:
                    logger.warning(f"    Can't afford new price or restore - order lost")
                    results.append({
                        "ticker": ticker, "side": side,
                        "action": "update_failed_insufficient_cash",
                    })
                continue

            logger.info(f"    Reducing contracts from {remaining} to {contracts_to_place}")
            order_cost_cents = new_price * contracts_to_place

        # Place new order
        if side == "NO":
            yes_price = 100 - new_price
        else:
            yes_price = new_price

        new_order = client.create_order(
            ticker=ticker, side=side.lower(), action="buy",
            count=contracts_to_place, yes_price=yes_price, post_only=True,
        )

        if new_order.success:
            logger.info(f"    Placed new order at {new_price}c: {new_order.order_id}")
            results.append({
                "ticker": ticker, "side": side,
                "action": "updated", "direction": direction,
                "old_price": our_side_price, "new_price": new_price,
                "contracts": contracts_to_place,
                "new_order_id": new_order.order_id, "ev_c": ev_c,
            })
            available_cash_cents -= order_cost_cents
        else:
            logger.error(f"    Failed to place new order: {new_order.error_message}")
            # Recovery: restore original
            logger.warning(f"    Attempting to restore original order at {our_side_price}c...")
            restore_order = client.create_order(
                ticker=ticker, side=side.lower(), action="buy",
                count=remaining, yes_price=our_price, post_only=True,
            )
            if restore_order.success:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "update_failed_restored",
                    "error": new_order.error_message,
                })
            else:
                logger.error(f"    CRITICAL: Failed to restore: {restore_order.error_message}")
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "replace_failed_no_restore",
                    "error": f"Update failed: {new_order.error_message}; Restore failed: {restore_order.error_message}",
                })

    return results


def resize_orders(
    client: KalshiClient,
    maker_orders: list[dict],
    min_ev_c: float,
    min_prob_pct: float,
    max_contracts: int = 500,
    dry_run: bool = False,
) -> list[dict]:
    """Resize existing orders to match Kelly-sized contracts.

    For each open order:
    1. Look up the Kelly-sized target from maker_orders
    2. If significantly different (>5%), cancel and re-place
    3. If not in maker_orders, cancel (no hysteresis for gas — simpler approach)
    """
    results = []
    open_orders = client.get_open_orders()

    if not open_orders:
        logger.info("No open orders to resize")
        return results

    try:
        balance_data = client.get_balance()
        available_cash_cents = balance_data.get("balance", 0)
        logger.info(f"Available cash for resize: ${available_cash_cents / 100:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch balance: {e}")
        available_cash_cents = float('inf')

    deduped_orders, dup_results = deduplicate_orders(open_orders, client, dry_run)
    results.extend(dup_results)

    # Build target lookup: (ticker, side) -> {contracts, p_model, limit_price_c, ev_if_filled_c}
    target_lookup = {}
    for mo in maker_orders:
        ticker = mo.get("ticker")
        side = mo.get("side", "").upper()
        target_lookup[(ticker, side)] = {
            "contracts": min(max_contracts, mo.get("contracts", 100)),
            "p_model": mo.get("p_model", 0),
            "limit_price_c": mo.get("limit_price_c", 0),
            "ev_if_filled_c": mo.get("ev_if_filled_c", 0),
        }

    logger.info(f"Checking {len(deduped_orders)} open orders for resize...")

    for order in deduped_orders:
        ticker = order.get("ticker")
        order_id = order.get("order_id")
        side = order.get("side", "").upper()
        our_price = order.get("yes_price", 0)
        current_contracts = order.get("remaining_count", 0)

        if current_contracts == 0:
            continue

        if order.get("action", "").lower() == "sell":
            continue

        if side == "NO":
            our_side_price = 100 - our_price
        else:
            our_side_price = our_price

        target = target_lookup.get((ticker, side))
        if not target:
            # No target — cancel
            logger.info(f"  {ticker} {side}: No target in maker_orders, cancelling")
            if dry_run:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "would_cancel_no_target",
                    "old_contracts": current_contracts,
                    "price": our_side_price, "dry_run": True,
                })
                continue

            cancel_result = client.cancel_order(order_id)
            if cancel_result.success:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "cancelled_no_target",
                    "old_contracts": current_contracts,
                    "order_id": order_id,
                })
            else:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "cancel_failed",
                    "error": cancel_result.error_message,
                })
            continue

        target_contracts = target["contracts"]
        p_model = target["p_model"]
        ev_c = target["ev_if_filled_c"]

        if target_contracts == 0:
            logger.info(f"  {ticker} {side}: Target is 0 contracts, cancelling")
            if dry_run:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "would_cancel_zero_target",
                    "old_contracts": current_contracts, "dry_run": True,
                })
                continue

            cancel_result = client.cancel_order(order_id)
            if cancel_result.success:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "cancelled_zero_target",
                    "old_contracts": current_contracts, "order_id": order_id,
                })
            else:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "cancel_failed", "error": cancel_result.error_message,
                })
            continue

        # Skip small changes (5% threshold)
        diff = abs(target_contracts - current_contracts)
        pct_diff = diff / current_contracts if current_contracts > 0 else float('inf')
        if pct_diff <= 0.05:
            continue

        if p_model < min_prob_pct or ev_c < min_ev_c:
            continue

        logger.info(
            f"  {ticker} {side}: Resize {current_contracts} -> {target_contracts} (EV: {ev_c:.2f}c)"
        )

        if dry_run:
            results.append({
                "ticker": ticker, "side": side,
                "action": "would_resize",
                "old_contracts": current_contracts,
                "new_contracts": target_contracts,
                "price": our_side_price, "ev_c": ev_c, "dry_run": True,
            })
            continue

        # Cancel old
        cancel_result = client.cancel_order(order_id)
        if not cancel_result.success:
            results.append({
                "ticker": ticker, "side": side,
                "action": "cancel_failed", "error": cancel_result.error_message,
            })
            continue

        # Cash check
        order_cost_cents = our_side_price * target_contracts
        actual_target = target_contracts

        if order_cost_cents > available_cash_cents:
            usable_cash = int(available_cash_cents * 0.99)
            actual_target = usable_cash // our_side_price
            if actual_target <= 0:
                # Try to restore original
                original_cost = our_side_price * current_contracts
                if original_cost <= available_cash_cents * 0.99:
                    restore = client.create_order(
                        ticker=ticker, side=side.lower(), action="buy",
                        count=current_contracts, yes_price=our_price, post_only=True,
                    )
                    if restore.success:
                        results.append({
                            "ticker": ticker, "side": side,
                            "action": "resize_skipped_restored",
                            "reason": "insufficient cash for target",
                        })
                    else:
                        results.append({
                            "ticker": ticker, "side": side,
                            "action": "resize_failed_no_restore",
                            "error": f"Restore failed: {restore.error_message}",
                        })
                else:
                    results.append({
                        "ticker": ticker, "side": side,
                        "action": "resize_failed_insufficient_cash",
                    })
                continue
            order_cost_cents = our_side_price * actual_target

        # Place new
        new_order = client.create_order(
            ticker=ticker, side=side.lower(), action="buy",
            count=actual_target, yes_price=our_price, post_only=True,
        )

        if new_order.success:
            results.append({
                "ticker": ticker, "side": side,
                "action": "resized",
                "old_contracts": current_contracts,
                "new_contracts": actual_target,
                "price": our_side_price,
                "new_order_id": new_order.order_id, "ev_c": ev_c,
            })
            available_cash_cents -= order_cost_cents
        else:
            # Recovery: restore original
            restore = client.create_order(
                ticker=ticker, side=side.lower(), action="buy",
                count=current_contracts, yes_price=our_price, post_only=True,
            )
            if restore.success:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "resize_failed_restored",
                    "error": new_order.error_message,
                })
            else:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "resize_failed_no_restore",
                    "error": f"Resize failed: {new_order.error_message}; Restore failed: {restore.error_message}",
                })

    return results


def prune_orders(
    client: KalshiClient,
    maker_orders: list[dict],
    min_ev_c: float = 0.0,
    dry_run: bool = False,
    predict_csv_path: str = "data/latest_predict.csv",
) -> list[dict]:
    """Cancel orders where EV has gone negative based on updated model."""
    import pandas as pd

    results = []
    open_orders = client.get_open_orders()

    if not open_orders:
        logger.info("No open orders to prune")
        return results

    # Build probability lookup
    prob_lookup = {}
    try:
        predict_df = pd.read_csv(predict_csv_path)
        for _, row in predict_df.iterrows():
            ticker = row.get("ticker")
            p_yes = float(row.get("P_model", 0)) * 100
            if ticker:
                prob_lookup[(ticker, "YES")] = p_yes
                prob_lookup[(ticker, "NO")] = 100 - p_yes
    except Exception as e:
        logger.warning(f"Could not load prediction CSV: {e}")

    for mo in maker_orders:
        ticker = mo.get("ticker")
        side = mo.get("side", "").upper()
        prob_lookup[(ticker, side)] = mo.get("p_model", 0)

    logger.info(f"Checking {len(open_orders)} open orders for pruning...")

    for order in open_orders:
        ticker = order.get("ticker")
        order_id = order.get("order_id")
        side = order.get("side", "").upper()
        yes_price = order.get("yes_price", 0)
        remaining = order.get("remaining_count", 0)

        if remaining == 0:
            continue

        if order.get("action", "").lower() == "sell":
            continue

        side_price = (100 - yes_price) if side == "NO" else yes_price

        p_model = prob_lookup.get((ticker, side))
        if p_model is None:
            logger.warning(f"  {ticker} {side}: No model data, skipping")
            results.append({
                "ticker": ticker, "side": side,
                "action": "skipped_no_model", "price": side_price,
            })
            continue

        fee = kalshi_fee_total_dollars(
            price_dollars=side_price / 100.0,
            contracts=remaining,
            is_maker=True,
        )
        fee_per_c = (fee / remaining) * 100 if remaining > 0 else 0
        ev_c = (p_model / 100.0) * 100 - side_price - fee_per_c

        if ev_c >= min_ev_c:
            continue

        logger.warning(
            f"  {ticker} {side} @ {side_price}c: EV {ev_c:.2f}c < {min_ev_c}c - CANCELLING"
        )

        if dry_run:
            results.append({
                "ticker": ticker, "side": side,
                "action": "would_cancel",
                "price": side_price, "ev_c": ev_c, "p_model": p_model,
                "dry_run": True,
            })
            continue

        cancel_result = client.cancel_order(order_id)
        if cancel_result.success:
            results.append({
                "ticker": ticker, "side": side,
                "action": "cancelled",
                "price": side_price, "ev_c": ev_c, "order_id": order_id,
            })
        else:
            results.append({
                "ticker": ticker, "side": side,
                "action": "cancel_failed", "error": cancel_result.error_message,
            })

    return results


def place_exit_orders(
    client: KalshiClient,
    exit_orders: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """Place maker sell limit orders for held positions at +EV exit prices.

    Deduplication: checks for existing sell orders per ticker/side:
    - Same price + quantity -> skip (no-op)
    - Different price or quantity -> cancel old, place new (reprice)
    - No existing sell -> place new
    """
    results = []

    for order in exit_orders:
        ticker = order["ticker"]
        side = order["side"]
        sell_price_c = order["sell_price_c"]
        contracts = order["contracts"]
        fair_value_c = order["fair_value_c"]
        exit_ev_c = order["exit_ev_c"]

        if side == "YES":
            yes_price = sell_price_c
        else:
            yes_price = 100 - sell_price_c

        # Check for existing sell orders
        try:
            open_orders = client.get_open_orders(ticker=ticker)
        except Exception as e:
            results.append({
                "ticker": ticker, "side": side,
                "action": "error", "error": str(e),
            })
            continue

        existing_sell = None
        for o in open_orders:
            if (o.get("action", "").lower() == "sell"
                    and o.get("side", "").upper() == side.upper()
                    and o.get("remaining_count", 0) > 0):
                existing_sell = o
                break

        if existing_sell:
            existing_yes_price = existing_sell.get("yes_price", 0)
            existing_remaining = existing_sell.get("remaining_count", 0)

            if existing_yes_price == yes_price and existing_remaining == contracts:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "no_change",
                    "sell_price_c": sell_price_c, "contracts": contracts,
                })
                continue

            # Reprice
            old_sell_price_c = existing_yes_price if side == "YES" else (100 - existing_yes_price)
            action_label = "repriced"

            if dry_run:
                logger.info(
                    f"  [DRY RUN] Would reprice {ticker} {side}: "
                    f"SELL {old_sell_price_c}c -> {sell_price_c}c"
                )
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "would_reprice",
                    "old_price_c": old_sell_price_c,
                    "sell_price_c": sell_price_c, "contracts": contracts,
                    "exit_ev_c": exit_ev_c,
                })
                continue

            cancel_result = client.cancel_order(existing_sell.get("order_id"))
            if not cancel_result.success:
                results.append({
                    "ticker": ticker, "side": side,
                    "action": "cancel_failed", "error": cancel_result.error_message,
                })
                continue
        else:
            action_label = "placed"

        if dry_run:
            logger.info(
                f"  [DRY RUN] Would place {ticker} SELL {contracts}x {side} @ {sell_price_c}c "
                f"(fair={fair_value_c:.1f}c, EV={exit_ev_c:+.1f}c)"
            )
            results.append({
                "ticker": ticker, "side": side,
                "action": "would_place",
                "sell_price_c": sell_price_c, "contracts": contracts,
                "exit_ev_c": exit_ev_c,
            })
            continue

        logger.info(
            f"  Placing {ticker} SELL {contracts}x {side} @ {sell_price_c}c "
            f"(fair={fair_value_c:.1f}c, EV={exit_ev_c:+.1f}c)"
        )

        result = client.create_order(
            ticker=ticker, side=side.lower(), action="sell",
            count=contracts, yes_price=yes_price, post_only=True,
        )

        if result.success:
            results.append({
                "ticker": ticker, "side": side,
                "action": action_label,
                "sell_price_c": sell_price_c, "contracts": contracts,
                "exit_ev_c": exit_ev_c, "order_id": result.order_id,
            })
        else:
            results.append({
                "ticker": ticker, "side": side,
                "action": f"{action_label}_failed",
                "error": result.error_message,
            })

    return results


def cancel_stale_exit_orders(
    client: KalshiClient,
    positions: list[dict],
    predictions_df,
    exit_orders: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """Cancel sell orders for positions where no valid exit price exists anymore."""
    results = []

    valid_exit_tickers = {(eo["ticker"], eo["side"]) for eo in exit_orders}

    p_lookup = {}
    for _, row in predictions_df.iterrows():
        ticker = str(row.get("ticker", ""))
        p_model = float(row.get("P_model", 0.5))
        p_lookup[ticker] = p_model

    for pos in positions:
        pos_qty = pos.get("position", 0)
        if pos_qty == 0:
            continue

        ticker = pos.get("ticker", "")
        if ticker not in p_lookup:
            continue

        held_side = "YES" if pos_qty > 0 else "NO"

        if (ticker, held_side) in valid_exit_tickers:
            continue

        try:
            open_orders = client.get_open_orders(ticker=ticker)
        except Exception:
            continue

        for o in open_orders:
            if (o.get("action", "").lower() == "sell"
                    and o.get("side", "").upper() == held_side.upper()
                    and o.get("remaining_count", 0) > 0):
                order_id = o.get("order_id")
                yes_price = o.get("yes_price", 0)
                sell_price_c = yes_price if held_side == "YES" else (100 - yes_price)
                p_side = p_lookup[ticker] if held_side == "YES" else (1.0 - p_lookup[ticker])
                fair_value_c = p_side * 100.0

                logger.info(
                    f"  {ticker} {held_side}: Stale sell @ {sell_price_c}c "
                    f"(fair={fair_value_c:.1f}c) - CANCELLING"
                )

                if dry_run:
                    results.append({
                        "ticker": ticker, "side": held_side,
                        "action": "would_cancel_stale",
                        "sell_price_c": sell_price_c,
                    })
                else:
                    cancel_result = client.cancel_order(order_id)
                    if cancel_result.success:
                        results.append({
                            "ticker": ticker, "side": held_side,
                            "action": "cancelled_stale",
                            "sell_price_c": sell_price_c,
                        })
                    else:
                        results.append({
                            "ticker": ticker, "side": held_side,
                            "action": "cancel_stale_failed",
                            "error": cancel_result.error_message,
                        })
                break

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit maker orders to Kalshi API for gas price markets",
    )

    parser.add_argument("--orders-file", default="data/latest_maker_orders.json")
    parser.add_argument("--output", default="data/maker_submission_log.json")
    parser.add_argument("--min-ev", type=float, default=2.0,
                        help="Minimum EV in cents (default: 2.0)")
    parser.add_argument("--max-contracts", type=int, default=999999)
    parser.add_argument("--max-exposure", type=float, default=100000.0)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update", action="store_true",
                        help="Update existing orders if outbid")
    parser.add_argument("--prune", action="store_true",
                        help="Cancel orders where EV went negative")
    parser.add_argument("--resize", action="store_true",
                        help="Resize orders to match Kelly")
    parser.add_argument("--place-exits", action="store_true",
                        help="Place maker sell orders for held positions")
    parser.add_argument("--cancel-all", action="store_true",
                        help="KILL SWITCH: Cancel all open orders")
    parser.add_argument("--cancel-ticker", type=str,
                        help="Cancel all open orders for specific ticker")
    parser.add_argument("--list-open", action="store_true")
    parser.add_argument("--positions", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    mode = "dry_run" if args.dry_run else "production"

    # Initialize client
    client = None
    needs_client = (
        not args.dry_run or args.update or args.prune or args.resize
        or args.place_exits or args.list_open or args.positions or args.balance
    )
    if needs_client:
        try:
            client = KalshiClient()
            logger.info("Connected to Kalshi API")
        except ValueError as e:
            logger.error(f"Authentication error: {e}")
            sys.exit(1)

    # --balance
    if args.balance:
        if not client:
            logger.error("Cannot check balance in dry-run mode")
            sys.exit(1)
        balance = client.get_balance()
        available = balance.get("balance", 0) / 100.0
        print(f"\nAccount Balance: ${available:.2f}")
        sys.exit(0)

    # --cancel-all / --cancel-ticker
    if args.cancel_all or args.cancel_ticker:
        if not client:
            logger.error("Cannot cancel orders in dry-run mode")
            sys.exit(1)

        ticker = args.cancel_ticker
        scope = f" for {ticker}" if ticker else ""
        logger.warning(f"CANCELLING all open orders{scope}")

        results = client.cancel_all_orders(ticker=ticker)
        success_count = sum(1 for r in results if r.success)

        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error_message}"
            logger.info(f"  Cancel {r.order_id}: {status}")

        print(f"\nCancelled: {success_count}/{len(results)} orders")
        sys.exit(0 if success_count == len(results) else 1)

    # --list-open
    if args.list_open:
        if not client:
            logger.error("Cannot list orders in dry-run mode")
            sys.exit(1)

        orders = client.get_open_orders()
        print(f"\nOpen orders: {len(orders)}")
        if orders:
            print(f"{'Ticker':<40} {'Side':<4} {'Qty':>5} {'Price':>6}")
            print("-" * 60)
            for o in orders:
                ticker = o.get("ticker", "?")
                side = o.get("side", "?").upper()
                remaining = o.get("remaining_count", 0)
                price = o.get("yes_price", 0)
                print(f"{ticker:<40} {side:<4} {remaining:>5} {price:>5}c")
        sys.exit(0)

    # --positions
    if args.positions:
        if not client:
            logger.error("Cannot list positions in dry-run mode")
            sys.exit(1)

        positions = client.get_positions()
        open_positions = [p for p in positions if p.get("position", 0) != 0]
        closed_positions = [p for p in positions if p.get("position", 0) == 0 and p.get("realized_pnl", 0) != 0]

        print(f"\n{'='*60}")
        print("OPEN POSITIONS")
        print(f"{'='*60}")
        if open_positions:
            print(f"{'Ticker':<40} {'Side':<4} {'Qty':>6} {'Exposure':>10}")
            print("-" * 65)
            total_exposure = 0
            for p in open_positions:
                ticker = p.get("ticker", "?")
                position = p.get("position", 0)
                exposure = p.get("market_exposure", 0)
                side = "YES" if position > 0 else "NO"
                qty = abs(position)
                total_exposure += abs(exposure)
                print(f"{ticker:<40} {side:<4} {qty:>6} ${exposure/100:>8.2f}")
            print("-" * 65)
            print(f"Total open exposure: ${total_exposure / 100:.2f}")
        else:
            print("No open positions")

        print(f"\n{'='*60}")
        print("REALIZED P&L (closed positions)")
        print(f"{'='*60}")
        if closed_positions:
            print(f"{'Ticker':<40} {'P&L':>10} {'Fees':>8}")
            print("-" * 65)
            total_pnl = 0
            total_fees = 0
            for p in closed_positions:
                ticker = p.get("ticker", "?")
                pnl = p.get("realized_pnl", 0)
                fees = p.get("fees_paid", 0)
                total_pnl += pnl
                total_fees += fees
                print(f"{ticker:<40} ${pnl/100:>+8.2f} ${fees/100:>7.2f}")
            print("-" * 65)
            print(f"Total realized P&L: ${total_pnl/100:>+.2f} (fees: ${total_fees/100:.2f})")
        else:
            print("No closed positions")
        sys.exit(0)

    # Load maker orders
    try:
        all_orders, _ = load_maker_orders(args.orders_file)
        logger.info(f"Loaded {len(all_orders)} maker orders")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Data freshness check — block if AAA scraper silently failed
    is_fresh, freshness_reason = is_model_data_fresh()
    if not is_fresh:
        print(f"\nBLOCKED: {freshness_reason}")
        sys.exit(0)
    logger.info(freshness_reason)

    # Check minimum cash balance ($500)
    # NOTE: Revisit after first live dry-run. If typical gas contracts are
    # much larger/smaller than TSA, this threshold needs recalibration.
    MIN_CASH_DOLLARS = 500
    if client and not args.dry_run:
        try:
            balance = client.get_balance()
            available_cash = balance.get("balance", 0) / 100.0
            if available_cash < MIN_CASH_DOLLARS:
                print(f"\nBLOCKED: Available cash ${available_cash:.2f} < ${MIN_CASH_DOLLARS}")
                sys.exit(0)
            logger.info(f"Cash balance: ${available_cash:.2f}")
        except Exception as e:
            logger.warning(f"Could not check balance: {e}. Proceeding anyway.")

    if not all_orders:
        logger.info("No maker orders to submit")
        sys.exit(0)

    # Day-of-week probability threshold
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    weekday = now_et.weekday()
    min_prob_for_day = get_maker_min_prob_for_day(weekday) * 100
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"{day_names[weekday]}: min_prob = {min_prob_for_day:.0f}%")

    # Infer observed_days from latest_predict_meta.json
    observed_days = 0
    try:
        meta = json.loads(Path("data/latest_predict_meta.json").read_text())
        observed_days = meta.get("n_observed", 0)
    except Exception:
        pass

    # --update
    if args.update:
        if not client:
            logger.error("Cannot update orders: API connection failed")
            sys.exit(1)

        update_results = update_orders(
            client=client, maker_orders=all_orders,
            min_ev_c=args.min_ev, min_prob_pct=min_prob_for_day,
            dry_run=args.dry_run,
        )

        updated = [r for r in update_results if r.get("action") == "updated"]
        would_update = [r for r in update_results if r.get("action") == "would_update"]
        failed = [r for r in update_results if "failed" in r.get("action", "")]

        _save_log(args.output, "update", update_results)

        print(f"\n{'='*60}")
        print("UPDATE SUMMARY")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"Would update: {len(would_update)}")
            for r in would_update:
                print(f"  {r['ticker']} {r['side']}: {r['old_price']}c -> {r['new_price']}c (EV: {r['ev_c']:.2f}c)")
        else:
            print(f"Updated: {len(updated)}, Failed: {len(failed)}")
            for r in updated:
                print(f"  {r['ticker']} {r['side']}: {r['old_price']}c -> {r['new_price']}c")

        sys.exit(0 if not failed else 1)

    # --prune
    if args.prune:
        if not client:
            logger.error("Cannot prune orders: API connection failed")
            sys.exit(1)

        prune_results = prune_orders(
            client=client, maker_orders=all_orders,
            min_ev_c=args.min_ev, dry_run=args.dry_run,
        )

        cancelled = [r for r in prune_results if r.get("action") == "cancelled"]
        would_cancel = [r for r in prune_results if r.get("action") == "would_cancel"]

        _save_log(args.output, "prune", prune_results)

        print(f"\n{'='*60}")
        print("PRUNE SUMMARY")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"Would cancel: {len(would_cancel)}")
            for r in would_cancel:
                print(f"  {r['ticker']} {r['side']} @ {r['price']}c: EV {r['ev_c']:.2f}c")
        else:
            print(f"Cancelled: {len(cancelled)}")
            for r in cancelled:
                print(f"  {r['ticker']} {r['side']} @ {r['price']}c")

        sys.exit(0)

    # --resize
    if args.resize:
        if not client:
            logger.error("Cannot resize orders: API connection failed")
            sys.exit(1)

        resize_results = resize_orders(
            client=client, maker_orders=all_orders,
            min_ev_c=args.min_ev, min_prob_pct=min_prob_for_day,
            max_contracts=args.max_contracts, dry_run=args.dry_run,
        )

        resized = [r for r in resize_results if r.get("action") == "resized"]
        would_resize = [r for r in resize_results if r.get("action") == "would_resize"]
        cancelled = [r for r in resize_results if "cancelled" in r.get("action", "")]
        failed = [r for r in resize_results if "failed" in r.get("action", "")]

        _save_log(args.output, "resize", resize_results)

        print(f"\n{'='*60}")
        print("RESIZE SUMMARY")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"Would resize: {len(would_resize)}")
            for r in would_resize:
                print(f"  {r['ticker']} {r['side']}: {r['old_contracts']} -> {r['new_contracts']}")
        else:
            print(f"Resized: {len(resized)}, Cancelled: {len(cancelled)}, Failed: {len(failed)}")

        sys.exit(0 if not failed else 1)

    # --place-exits
    if args.place_exits:
        import pandas as pd

        exit_config = DEFAULT_MAKER_EXIT_CONFIG
        if not exit_config.enabled:
            print("Maker exit orders disabled")
            sys.exit(0)

        if not client:
            try:
                client = KalshiClient()
            except ValueError as e:
                logger.error(f"Authentication error: {e}")
                sys.exit(1)

        predict_csv_path = Path("data/latest_predict.csv")
        if not predict_csv_path.exists():
            logger.error("No predictions file found")
            sys.exit(1)

        predictions_df = pd.read_csv(predict_csv_path)

        try:
            positions = client.get_positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            sys.exit(1)

        open_positions = [p for p in positions if p.get("position", 0) != 0]
        if not open_positions:
            print("No open positions")
            sys.exit(0)

        exit_orders = generate_exit_orders(
            positions=open_positions,
            predictions_df=predictions_df,
            fee_schedule=FeeSchedule(),
            min_exit_ev_c=exit_config.min_exit_ev_c,
        )

        stale_results = cancel_stale_exit_orders(
            client=client, positions=open_positions,
            predictions_df=predictions_df, exit_orders=exit_orders,
            dry_run=args.dry_run,
        )

        if not exit_orders and not stale_results:
            print("No valid exit prices (positions hold to settlement)")
            sys.exit(0)

        exit_results = []
        if exit_orders:
            exit_results = place_exit_orders(
                client=client, exit_orders=exit_orders, dry_run=args.dry_run,
            )

        placed = [r for r in exit_results if r.get("action") == "placed"]
        repriced = [r for r in exit_results if r.get("action") == "repriced"]
        no_change = [r for r in exit_results if r.get("action") == "no_change"]
        would_place = [r for r in exit_results if r.get("action") == "would_place"]
        failed = [r for r in exit_results if "failed" in r.get("action", "")]

        print(f"\n{'='*60}")
        print("EXIT ORDER SUMMARY")
        print(f"{'='*60}")
        if args.dry_run:
            print(f"Would place: {len(would_place)}, No change: {len(no_change)}")
            for r in would_place:
                print(f"  PLACE: {r['ticker']} SELL {r['contracts']}x {r['side']} @ {r['sell_price_c']}c")
        else:
            print(f"Placed: {len(placed)}, Repriced: {len(repriced)}, No change: {len(no_change)}")

        # Save log
        log_path = Path("data/maker_exit_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "place_exits",
            "dry_run": args.dry_run,
            "results": exit_results + stale_results,
        }, indent=2, default=str))

        sys.exit(0 if not failed else 1)

    # === Default mode: validate and submit new orders ===

    # Check existing exposure and track open ticker/side pairs
    existing_exposure_dollars = 0.0
    existing_ticker_sides: set[tuple[str, str]] = set()
    if client:
        try:
            open_orders = client.get_open_orders()
            deduped_orders, _ = deduplicate_orders(open_orders, client, args.dry_run)

            for o in deduped_orders:
                remaining = o.get("remaining_count", 0)
                yes_price = o.get("yes_price", 0)
                side = o.get("side", "").upper()
                ticker = o.get("ticker", "")

                if ticker and remaining > 0:
                    existing_ticker_sides.add((ticker, side))

                side_price = (100 - yes_price) if side == "NO" else yes_price
                existing_exposure_dollars += (side_price * remaining) / 100.0

            if existing_exposure_dollars > 0:
                logger.info(f"Existing open order exposure: ${existing_exposure_dollars:.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch open orders: {e}")

    remaining_budget = args.max_exposure - existing_exposure_dollars
    if remaining_budget <= 0:
        print(f"\nExposure limit reached (${existing_exposure_dollars:.2f} >= ${args.max_exposure:.2f})")
        sys.exit(0)

    # Validate orders
    if args.verbose:
        print(f"\nValidating orders (day {observed_days})")
        print("-" * 50)

    accepted, skipped = validate_orders(
        orders=all_orders,
        observed_days=observed_days,
        max_contracts=args.max_contracts,
        verbose=args.verbose,
    )

    # Remove duplicates with existing open orders
    if existing_ticker_sides:
        deduped_accepted = []
        for order in accepted:
            ticker = order.get("ticker", "")
            side = order.get("side", "").upper()
            if (ticker, side) in existing_ticker_sides:
                logger.info(f"Skipping {ticker} {side}: already have open order")
                skipped.append({**order, "skip_reason": "Already have open order"})
            else:
                deduped_accepted.append(order)
        accepted = deduped_accepted

    # Block opposite-position orders
    if client and not args.dry_run:
        try:
            positions = client.get_positions()
            held_positions = {}
            for p in positions:
                pos_qty = p.get("position", 0)
                if pos_qty == 0:
                    continue
                ticker = p.get("ticker", "")
                held_side = "YES" if pos_qty > 0 else "NO"
                held_positions[ticker] = (held_side, abs(pos_qty))

            if held_positions:
                safe_accepted = []
                for order in accepted:
                    ticker = order.get("ticker", "")
                    order_side = order.get("side", "").upper()
                    if ticker in held_positions:
                        held_side, held_qty = held_positions[ticker]
                        if held_side != order_side:
                            logger.warning(
                                f"BLOCKING {ticker} {order_side}: holds {held_qty} {held_side}"
                            )
                            skipped.append({
                                **order,
                                "skip_reason": f"Opposite to held position ({held_qty} {held_side})"
                            })
                            continue
                    safe_accepted.append(order)
                accepted = safe_accepted
        except Exception as e:
            logger.warning(f"Could not check positions: {e}")

    logger.info(f"Orders: {len(accepted)} accepted, {len(skipped)} skipped")

    if not accepted:
        print(f"\nNo orders passed filters")
        sys.exit(0)

    # Preview
    print(f"\n{'='*70}")
    print(f"ORDERS TO SUBMIT ({mode.upper()} MODE)")
    print(f"{'='*70}")
    print(f"{'Ticker':<40} {'Side':<4} {'Qty':>5} {'Price':>6} {'EV':>8}")
    print("-" * 70)

    total_exposure_c = 0
    for o in accepted:
        ticker = o.get("ticker", "?")
        side = o.get("side", "?")
        contracts = o.get("contracts_to_submit", 0)
        price_c = o.get("limit_price_c", 0)
        ev_c = o.get("ev_if_filled_c", 0)
        exposure_c = price_c * contracts
        total_exposure_c += exposure_c
        print(f"{ticker:<40} {side:<4} {contracts:>5} {price_c:>5}c {ev_c:>+7.2f}c")

    print("-" * 70)
    print(f"Total exposure: ${total_exposure_c / 100:.2f}")
    print()

    # Submit
    if args.dry_run:
        logger.info("DRY RUN - no orders will be submitted")

    results = submit_orders(client, accepted, dry_run=args.dry_run)

    succeeded = [r for r in results if r.get("success")]
    failed = [r for r in results if r.get("submitted") and not r.get("success")]

    summary = SubmissionSummary(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        orders_attempted=len(accepted),
        orders_succeeded=len(succeeded),
        orders_failed=len(failed),
        orders_skipped=len(skipped),
        total_contracts=sum(r.get("contracts_to_submit", 0) for r in results),
        total_exposure_cents=total_exposure_c,
        results=results,
        skipped=[{"ticker": s.get("ticker"), "reason": s.get("skip_reason")} for s in skipped],
    )

    # Save log
    _save_log(args.output, "submit", results, summary=summary)

    # Print summary
    print(f"\n{'='*60}")
    print("SUBMISSION SUMMARY")
    print(f"{'='*60}")
    print(f"Attempted:  {summary.orders_attempted}")
    print(f"Succeeded:  {summary.orders_succeeded}")
    print(f"Failed:     {summary.orders_failed}")
    print(f"Skipped:    {summary.orders_skipped}")
    print(f"Total exposure: ${total_exposure_c / 100:.2f}")

    sys.exit(0 if not failed else 1)


def _save_log(output_path: str, mode: str, results: list, summary=None):
    """Save operation results to JSON log."""
    log_path = Path(output_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "results": results,
    }
    if summary:
        from dataclasses import asdict
        data["summary"] = asdict(summary)
    log_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
