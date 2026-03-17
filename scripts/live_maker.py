#!/usr/bin/env python3
"""Real-time maker trading loop for gas price markets.

Replaces the 15-minute cron-based maker management with a local loop that
polls every second and protects against adverse selection.

Four time scales:
  - Fast tick (5s): Poll orderbooks for active tickers, check adverse selection
  - Fill check (5s): Poll open orders to detect fills (remaining_count delta)
  - Full cycle (30s): Full order management — prune, update, place, resize, exits
  - AAA check (60s): Check for new AAA price, fetch if missing, re-predict

State machine: ACTIVE → DEFENSIVE (adverse selection)

Usage:
    ./venv/bin/python scripts/live_maker.py              # Production
    ./venv/bin/python scripts/live_maker.py --dry-run     # No orders, just monitor
    ./venv/bin/python scripts/live_maker.py --cancel-on-exit  # Cancel all on Ctrl+C
    ./venv/bin/python scripts/live_maker.py --no-aaa-poll # Disable AAA polling
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.kalshi_client import KalshiClient
from src.orderbook import fetch_orderbook, fetch_orderbooks_parallel
from src.trade import (
    FeeSchedule,
    calc_ev_if_filled_c,
    ev_robust_under_slippage,
    generate_exit_orders,
    generate_maker_orders,
    passes_toxicity_guard,
)
from src.cli_maker_submit import (
    deduplicate_orders,
    load_maker_orders,
    submit_orders,
    validate_orders,
)
from src.config import (
    DEFAULT_LIVE_MAKER_CONFIG,
    DEFAULT_MAKER_DEFENSIVE_CONFIG,
    DEFAULT_MAKER_EXIT_CONFIG,
    DEFAULT_TRADING_CONFIG,
    LiveMakerConfig,
)

ET = ZoneInfo("America/New_York")

# PID file — shared with dashboard/services/live_maker_service.py
_PID_FILE = Path(__file__).resolve().parent.parent / "data" / "live_maker.pid"

# ---------------------------------------------------------------------------
# Logging — colored terminal output + optional plain-text log file
# ---------------------------------------------------------------------------

TAG_ANSI = {
    "FILL": "\033[32m",       # green
    "DIFF": "\033[92m",       # light green
    "DIFF DRY": "\033[92m",
    "PRUNE": "\033[33m",      # yellow
    "PRUNE DRY": "\033[33m",
    "CYCLE": "\033[36m",      # cyan
    "STATE": "\033[33m",      # yellow
    "DEFENSIVE": "\033[91m",  # bright red
    "TAKER": "\033[35m",      # magenta
    "AAA": "\033[96m",        # bright cyan
    "EXIT": "\033[34m",       # blue
    "INIT": "\033[36m",       # cyan
    "LOOP": "\033[36m",       # cyan
    "PREFLIGHT": "\033[36m",
    "SHUTDOWN": "\033[33m",
    "DISCORD": "\033[35m",
    "TOXICITY": "\033[33m",
    "SESSION SUMMARY": "\033[36m",
    "HALT": "\033[91m",
    "FATAL": "\033[91m",
}
RESET = "\033[0m"
BOLD = "\033[1m"


class ColorFormatter(logging.Formatter):
    """Formatter that colors log lines based on [TAG] prefix."""

    LEVEL_COLORS = {
        logging.ERROR: "\033[91m",
        logging.CRITICAL: "\033[91m",
        logging.WARNING: "\033[33m",
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            return f"{self.LEVEL_COLORS[logging.ERROR]}{msg}{RESET}"
        if record.levelno >= logging.WARNING:
            return f"{self.LEVEL_COLORS[logging.WARNING]}{msg}{RESET}"
        for tag, color in TAG_ANSI.items():
            if f"[{tag}]" in msg:
                return f"{color}{msg}{RESET}"
        return msg


def setup_logging(log_file: str | None = None):
    """Configure logging with colored terminal output and optional log file."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)


logger = logging.getLogger("live_maker")


# ---------------------------------------------------------------------------
# Ticker shortener: KXAAAGASW-26MAR23-3.670 → 3.670
# ---------------------------------------------------------------------------

def t(ticker: str) -> str:
    """Shorten ticker to just the threshold value."""
    parts = ticker.rsplit("-", 1)
    if len(parts) == 2:
        try:
            float(parts[1])
            return parts[1]
        except ValueError:
            pass
    return ticker


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

@dataclass
class LiveState:
    """Mutable state for the live maker loop."""

    mode: Literal["ACTIVE", "DEFENSIVE"] = "ACTIVE"
    mode_since: float = 0.0
    defensive_trigger: str = ""

    # Price tracking — ticker → deque of (timestamp, mid_price)
    mid_prices: dict[str, deque] = field(default_factory=dict)
    # Spread tracking — ticker → deque of (timestamp, spread)
    spread_history: dict[str, deque] = field(default_factory=dict)

    # Fill tracking
    order_remaining: dict[str, int] = field(default_factory=dict)
    order_meta: dict[str, dict] = field(default_factory=dict)
    recent_fills: deque = field(default_factory=lambda: deque(maxlen=200))
    recently_cancelled: set[str] = field(default_factory=set)

    # API health
    consecutive_api_errors: int = 0
    low_balance_warned: bool = False

    # Tickers
    active_tickers: set[str] = field(default_factory=set)
    all_tickers: list[str] = field(default_factory=list)

    # Session stats
    total_fills: int = 0
    total_fills_qty: int = 0
    state_transitions: int = 0
    api_calls: int = 0
    full_cycles: int = 0

    # Cached resting orders for diffing — (ticker, side) → {price, contracts, order_id}
    resting_buy_orders: dict[tuple[str, str], dict] = field(default_factory=dict)

    # Exit orders that failed — don't retry same (ticker, side, price) within session
    failed_exits: set[tuple[str, str, int]] = field(default_factory=set)

    # AAA polling state
    last_aaa_check: float = 0.0
    aaa_updated_today: bool = False
    last_aaa_date: str = ""  # ISO date of latest AAA observation
    aaa_regen_count: int = 0

    # Taker opportunity notification dedup — (ticker, side, ask_price_c)
    notified_taker_opps: set[tuple[str, str, int]] = field(default_factory=set)
    taker_alerts_sent: int = 0
    taker_alerts_enabled: bool = True


# ---------------------------------------------------------------------------
# Orderbook helpers
# ---------------------------------------------------------------------------

def compute_mid(orderbook: dict, side: str) -> float | None:
    """Compute mid-price for a given side from orderbook."""
    if side == "YES":
        bids = orderbook.get("yes", [])
        asks_raw = orderbook.get("no", [])
        asks = [(100 - p, q) for p, q in asks_raw] if asks_raw else []
    else:
        bids = orderbook.get("no", [])
        asks_raw = orderbook.get("yes", [])
        asks = [(100 - p, q) for p, q in asks_raw] if asks_raw else []

    best_bid = max((b[0] for b in bids), default=None)
    best_ask = min((a[0] for a in asks), default=None)

    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    return None


def compute_spread(orderbook: dict, side: str) -> float | None:
    """Compute spread for a given side from orderbook."""
    if side == "YES":
        bids = orderbook.get("yes", [])
        asks_raw = orderbook.get("no", [])
        asks = [(100 - p, q) for p, q in asks_raw] if asks_raw else []
    else:
        bids = orderbook.get("no", [])
        asks_raw = orderbook.get("yes", [])
        asks = [(100 - p, q) for p, q in asks_raw] if asks_raw else []

    best_bid = max((b[0] for b in bids), default=None)
    best_ask = min((a[0] for a in asks), default=None)

    if best_bid is not None and best_ask is not None:
        return best_ask - best_bid
    return None


# ---------------------------------------------------------------------------
# Adverse selection detection
# ---------------------------------------------------------------------------

def check_mid_velocity(state: LiveState, config: LiveMakerConfig) -> str | None:
    """Check if multiple tickers' mid-prices moved too fast simultaneously."""
    now = time.time()
    cutoff = now - config.mid_velocity_window_s
    triggered = []
    for ticker, history in state.mid_prices.items():
        if len(history) < 2:
            continue
        recent = [(ts, mid) for ts, mid in history if ts >= cutoff]
        if len(recent) < 2:
            continue
        oldest_mid = recent[0][1]
        newest_mid = recent[-1][1]
        move = abs(newest_mid - oldest_mid)
        if move > config.mid_velocity_threshold_c:
            triggered.append((ticker, move))
    if len(triggered) >= config.mid_velocity_min_tickers:
        details = ", ".join(f"{tk} {m:.1f}c" for tk, m in triggered[:3])
        return f"mid velocity: {len(triggered)} tickers moved >{config.mid_velocity_threshold_c}c in {config.mid_velocity_window_s}s ({details})"
    return None


def check_spread_blowout(state: LiveState, config: LiveMakerConfig) -> str | None:
    """Check if any ticker's spread blew out relative to trailing average."""
    now = time.time()
    cutoff = now - config.spread_trailing_window_s
    for ticker, history in state.spread_history.items():
        if len(history) < 5:
            continue
        current_spread = history[-1][1]
        trailing = [s for ts, s in history if ts >= cutoff and ts < history[-1][0]]
        if not trailing:
            continue
        avg_spread = sum(trailing) / len(trailing)
        if avg_spread > 0 and current_spread > avg_spread * config.spread_blowout_mult and current_spread >= config.spread_blowout_floor_c:
            return (
                f"spread blowout: {t(ticker)} spread={current_spread:.0f}c "
                f"vs avg={avg_spread:.1f}c ({current_spread/avg_spread:.1f}x)"
            )
    return None


def check_fill_rate(state: LiveState, config: LiveMakerConfig) -> str | None:
    """Check if fills are happening too fast (informed flow)."""
    now = time.time()
    cutoff = now - config.fill_rate_window_s
    recent = [f for f in state.recent_fills if f[0] >= cutoff]
    if len(recent) > config.fill_rate_max:
        return f"fill rate spike: {len(recent)} fills in {config.fill_rate_window_s}s"
    return None


def check_book_depletion(
    state: LiveState, orderbooks: dict[str, dict], resting: dict[tuple[str, str], dict],
    min_depleted: int = 3,
) -> str | None:
    """Check if we're the sole best bid with no depth behind us on multiple tickers."""
    depleted = []
    for (ticker, side), order_info in resting.items():
        ob = orderbooks.get(ticker)
        if not ob:
            continue
        our_price = order_info["price"]
        if our_price <= 1 or our_price >= 95:
            continue
        if side == "YES":
            bids = ob.get("yes", [])
        else:
            bids = ob.get("no", [])
        if not bids:
            continue
        best_bid = max(b[0] for b in bids)
        if our_price == best_bid:
            behind = [b for b in bids if b[0] < our_price]
            if not behind:
                depleted.append(f"{t(ticker)} {side} @ {our_price}c")
    if len(depleted) >= min_depleted:
        return f"book depletion on {len(depleted)} tickers: {', '.join(depleted[:3])}"
    return None


def check_adverse_selection(
    state: LiveState,
    config: LiveMakerConfig,
    active_orderbooks: dict[str, dict] | None = None,
) -> str | None:
    """Run all adverse selection checks. Returns trigger reason or None."""
    signal = check_mid_velocity(state, config)
    if signal:
        return signal

    signal = check_spread_blowout(state, config)
    if signal:
        return signal

    if active_orderbooks:
        signal = check_book_depletion(state, active_orderbooks, state.resting_buy_orders)
        if signal:
            return signal

    return None


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def send_discord_defensive_alert(reason: str, cancelled: int):
    """Send Discord notification when DEFENSIVE mode is activated."""
    import requests
    webhook_url = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook_url:
        logger.warning("[DISCORD] DISCORD_WEBHOOK not set, skipping notification")
        return
    msg = f"\U0001f6a8 **GAS DEFENSIVE MODE ACTIVATED**\nTrigger: {reason}\nCancelled {cancelled} buy orders. Loop exiting."
    try:
        r = requests.post(webhook_url, json={"content": msg}, timeout=10)
        if 200 <= r.status_code < 300:
            logger.info("[DISCORD] Defensive alert sent")
        else:
            logger.warning(f"[DISCORD] Webhook failed: {r.status_code}")
    except Exception as e:
        logger.warning(f"[DISCORD] Webhook error: {e}")


def send_discord_taker_alert(opportunities: list[dict]):
    """Send Discord notification for +EV taker opportunities."""
    import requests
    webhook_url = os.getenv("DISCORD_WEBHOOK", "").strip()
    if not webhook_url:
        return
    lines = ["\U0001f4c8 **Gas Taker Opportunities**"]
    for opp in opportunities:
        lines.append(
            f"> {opp['bucket']} | {opp['side']} @ {opp['ask_c']}c | "
            f"P={opp['p_side_pct']:.1f}% | EV=+{opp['ev_c']:.1f}c/ct | "
            f"~{opp['contracts']} contracts"
        )
    msg = "\n".join(lines)
    try:
        r = requests.post(webhook_url, json={"content": msg}, timeout=10)
        if 200 <= r.status_code < 300:
            logger.info(f"[DISCORD] Taker alert sent ({len(opportunities)} opps)")
        else:
            logger.warning(f"[DISCORD] Taker webhook failed: {r.status_code}")
    except Exception as e:
        logger.warning(f"[DISCORD] Taker webhook error: {e}")


def scan_taker_opportunities(
    predict_csv_path: str,
    orderbooks: dict[str, dict],
    state: LiveState,
    min_ev_c: float = 0.0,
):
    """Scan for +EV taker opportunities and send Discord notifications for new ones."""
    import pandas as pd

    if not state.taker_alerts_enabled:
        return

    df = pd.read_csv(predict_csv_path)
    new_opps = []

    for _, row in df.iterrows():
        ticker = row.get("ticker", "")
        p_model = float(row.get("P_model", 0))
        bucket = row.get("bucket", "")
        ob = orderbooks.get(ticker)
        if not ob:
            continue

        for side in ("YES", "NO"):
            p_side = p_model if side == "YES" else (1.0 - p_model)

            if side == "YES":
                no_bids = ob.get("no", [])
                asks = [(100 - p, q) for p, q in no_bids] if no_bids else []
            else:
                yes_bids = ob.get("yes", [])
                asks = [(100 - p, q) for p, q in yes_bids] if yes_bids else []

            if not asks:
                continue
            best_ask_c = min(a[0] for a in asks)
            fillable = sum(q for p, q in asks if p == best_ask_c)

            ev_c = calc_ev_if_filled_c(
                p_model=p_model, side=side,
                price_paid_c=best_ask_c, is_maker=False, qty=max(fillable, 1),
            )
            if ev_c < min_ev_c:
                continue

            contracts = min(fillable, 250)
            if contracts <= 0:
                continue

            key = (ticker, side, int(best_ask_c))
            if key in state.notified_taker_opps:
                continue

            new_opps.append({
                "bucket": bucket, "ticker": ticker, "side": side,
                "ask_c": int(best_ask_c), "p_side_pct": p_side * 100,
                "ev_c": ev_c, "contracts": contracts,
            })
            state.notified_taker_opps.add(key)

    if new_opps:
        send_discord_taker_alert(new_opps)
        state.taker_alerts_sent += len(new_opps)
        for opp in new_opps:
            logger.info(
                f"[TAKER] {opp['bucket']} {opp['side']} @ {opp['ask_c']}c | "
                f"P={opp['p_side_pct']:.1f}% | EV=+{opp['ev_c']:.1f}c"
            )


def enter_defensive(state: LiveState, client: KalshiClient | None, reason: str, dry_run: bool = False):
    """Enter DEFENSIVE mode: cancel all buy orders, keep exit (sell) orders.

    DEFENSIVE is a terminal state — the loop will exit after cancelling orders.
    """
    prev_mode = state.mode
    state.mode = "DEFENSIVE"
    state.mode_since = time.time()
    state.defensive_trigger = reason
    state.state_transitions += 1
    logger.warning(f"[STATE] {prev_mode} -> DEFENSIVE ({reason})")

    print("\a", flush=True)

    cancelled = 0
    if client and not dry_run:
        try:
            open_orders = client.get_open_orders()
            buy_orders = [o for o in open_orders if o.get("action", "").lower() != "sell"]
            for o in buy_orders:
                oid = o.get("order_id")
                if oid:
                    result = client.cancel_order(oid)
                    if result.success:
                        cancelled += 1
                        state.recently_cancelled.add(oid)
            logger.info(f"[DEFENSIVE] Cancelled {cancelled} buy orders (kept sell/exit orders)")
            state.resting_buy_orders.clear()
        except Exception as e:
            logger.error(f"[DEFENSIVE] Error cancelling orders: {e}")

    send_discord_defensive_alert(reason, cancelled)


def enter_active(state: LiveState):
    """Enter ACTIVE mode."""
    prev_mode = state.mode
    state.mode = "ACTIVE"
    state.mode_since = time.time()
    state.defensive_trigger = ""
    state.state_transitions += 1
    logger.info(f"[STATE] {prev_mode} -> ACTIVE")


# ---------------------------------------------------------------------------
# Fill detection
# ---------------------------------------------------------------------------

def detect_fills(state: LiveState, client: KalshiClient | None, dry_run: bool = False):
    """Detect fills by comparing remaining_count of open orders."""
    if client is None:
        return

    try:
        current_orders = client.get_open_orders()
        state.api_calls += 1
    except Exception as e:
        state.consecutive_api_errors += 1
        logger.error(f"[FILL CHECK] API error: {e} (consecutive: {state.consecutive_api_errors})")
        return

    state.consecutive_api_errors = 0
    now = time.time()

    for order in current_orders:
        oid = order.get("order_id", "")
        remaining = order.get("remaining_count", 0)
        prev_remaining = state.order_remaining.get(oid, remaining)
        filled_qty = prev_remaining - remaining

        if filled_qty > 0:
            ticker = order.get("ticker", "?")
            side = order.get("side", "?").upper()
            action = order.get("action", "buy")
            price = order.get("yes_price", 0)
            side_price = (100 - price) if side == "NO" else price

            if action.lower() != "sell":
                state.recent_fills.append((now, oid, filled_qty))
            state.total_fills += 1
            state.total_fills_qty += filled_qty
            logger.info(
                f"[FILL] {t(ticker)} {action.upper()} {side} {filled_qty} ct "
                f"@ {side_price}c (remaining: {remaining})"
            )

        state.order_remaining[oid] = remaining
        state.order_meta[oid] = {
            "ticker": order.get("ticker", "?"),
            "side": order.get("side", "?").upper(),
            "action": order.get("action", "buy"),
            "yes_price": order.get("yes_price", 0),
        }

    # Detect fills from orders that vanished (fully filled between polls)
    current_ids = {o.get("order_id", "") for o in current_orders}
    stale_ids = [oid for oid in state.order_remaining if oid not in current_ids]
    for oid in stale_ids:
        prev_remaining = state.order_remaining[oid]
        if prev_remaining > 0 and oid not in state.recently_cancelled:
            meta = state.order_meta.get(oid, {})
            ticker = meta.get("ticker", "?")
            side = meta.get("side", "?")
            action = meta.get("action", "buy")
            price = meta.get("yes_price", 0)
            side_price = (100 - price) if side == "NO" else price

            if action.lower() != "sell":
                state.recent_fills.append((now, oid, prev_remaining))
            state.total_fills += 1
            state.total_fills_qty += prev_remaining
            logger.info(
                f"[FILL] {t(ticker)} {action.upper()} {side} {prev_remaining} ct "
                f"@ {side_price}c (fully filled, vanished from API)"
            )
        del state.order_remaining[oid]
        state.order_meta.pop(oid, None)
        state.recently_cancelled.discard(oid)


# ---------------------------------------------------------------------------
# Full cycle — order management
# ---------------------------------------------------------------------------

def build_side_lookup(maker_orders: list[dict]) -> dict[str, str]:
    """Build ticker → side lookup from maker orders for mid-price computation."""
    lookup = {}
    for o in maker_orders:
        ticker = o.get("ticker", "")
        side = o.get("side", "YES")
        lookup[ticker] = side
    return lookup


def refresh_resting_orders(state: LiveState, client: KalshiClient | None):
    """Refresh the cached resting buy orders from API."""
    if client is None:
        return
    try:
        open_orders = client.get_open_orders()
        state.api_calls += 1
        state.resting_buy_orders.clear()
        state.active_tickers.clear()
        for o in open_orders:
            if o.get("action", "").lower() == "sell":
                continue
            if o.get("remaining_count", 0) == 0:
                continue
            ticker = o.get("ticker", "")
            side = o.get("side", "").upper()
            price = o.get("yes_price", 0)
            side_price = (100 - price) if side == "NO" else price
            state.resting_buy_orders[(ticker, side)] = {
                "price": side_price,
                "contracts": o.get("remaining_count", 0),
                "order_id": o.get("order_id", ""),
                "yes_price": price,
            }
            state.active_tickers.add(ticker)
    except Exception as e:
        logger.error(f"Error refreshing resting orders: {e}")


def diff_and_submit_orders(
    desired: list[dict],
    resting: dict[tuple[str, str], dict],
    client: KalshiClient | None,
    dry_run: bool,
    state: LiveState | None = None,
) -> list[dict]:
    """Diff desired orders against resting, only submit changes."""
    results = []

    desired_keys = set()
    for order in desired:
        ticker = order.get("ticker", "")
        side = order.get("side", "YES").upper()
        key = (ticker, side)
        desired_keys.add(key)

        price_c = int(order.get("limit_price_c", 50))
        contracts = order.get("contracts_to_submit", order.get("contracts", 0))

        existing = resting.get(key)
        if existing:
            price_match = existing["price"] == price_c
            contract_diff = abs(existing["contracts"] - contracts)
            contract_pct = contract_diff / max(existing["contracts"], 1)
            if price_match and contract_pct < 0.05:
                continue  # No-op — within MC noise tolerance
            if client and not dry_run:
                client.cancel_order(existing["order_id"])
                if state:
                    state.recently_cancelled.add(existing["order_id"])
                logger.info(
                    f"[DIFF] {t(ticker)} {side}: reprice {existing['price']}c/{existing['contracts']}ct "
                    f"-> {price_c}c/{contracts}ct"
                )
            elif dry_run:
                logger.info(
                    f"[DIFF DRY] {t(ticker)} {side}: would reprice {existing['price']}c -> {price_c}c"
                )
        else:
            logger.info(f"[DIFF] {t(ticker)} {side}: new order @ {price_c}c x{contracts}")

        if contracts > 0:
            results.extend(submit_orders(client, [order], dry_run=dry_run))

    # Cancel resting orders not in desired set (pruned)
    for key, existing in resting.items():
        if key not in desired_keys:
            ticker, side = key
            if client and not dry_run:
                client.cancel_order(existing["order_id"])
                if state:
                    state.recently_cancelled.add(existing["order_id"])
                logger.info(f"[PRUNE] {t(ticker)} {side}: cancelled (no longer desired)")
            elif dry_run:
                logger.info(f"[PRUNE DRY] {t(ticker)} {side}: would cancel")

    return results


def run_full_cycle(
    client: KalshiClient | None,
    state: LiveState,
    config: LiveMakerConfig,
    dry_run: bool,
    maker_orders_path: str,
    predict_csv_path: str,
):
    """Run a full order management cycle."""
    state.full_cycles += 1
    cycle_start = time.time()

    try:
        # 1. Load predictions
        maker_orders, metadata = load_maker_orders(maker_orders_path)
        observed_days = metadata.get("n_observed", 0)

        if not maker_orders:
            logger.info("[CYCLE] No maker orders in file")
            return

        # 2. Fetch all orderbooks
        all_tickers = list({o.get("ticker", "") for o in maker_orders})
        state.all_tickers = all_tickers
        orderbooks = fetch_orderbooks_parallel(all_tickers)
        state.api_calls += len(all_tickers)

        # 3. Update orders with live prices from orderbooks
        updated_orders = []
        for order in maker_orders:
            ticker = order.get("ticker", "")
            side = order.get("side", "YES").upper()
            ob = orderbooks.get(ticker)
            if not ob:
                continue

            if side == "YES":
                bids = ob.get("yes", [])
                asks_raw = ob.get("no", [])
                asks = [(100 - p, q) for p, q in asks_raw] if asks_raw else []
            else:
                bids = ob.get("no", [])
                asks_raw = ob.get("yes", [])
                asks = [(100 - p, q) for p, q in asks_raw] if asks_raw else []

            best_bid = max((b[0] for b in bids), default=0)
            best_ask = min((a[0] for a in asks), default=100)

            # If our resting order IS the best bid, don't outbid ourselves
            resting = state.resting_buy_orders.get((ticker, side))
            if resting and resting["price"] == best_bid:
                candidate_price = best_bid  # Stay at our price
            else:
                candidate_price = best_bid + 1  # Improve by 1 tick
            if candidate_price >= best_ask:
                continue  # No room to place order

            order_copy = dict(order)
            order_copy["limit_price_c"] = candidate_price
            order_copy["current_bid_c"] = best_bid
            order_copy["current_ask_c"] = best_ask
            updated_orders.append(order_copy)

        if not updated_orders:
            logger.debug("[CYCLE] No valid orders after live price update")
            return

        # 4. Validate orders (bid+1 candidates)
        accepted, skipped = validate_orders(
            updated_orders,
            observed_days=observed_days,
            verbose=False,
        )

        # 4b. Retry EV-rejected orders at best bid (join) instead of bid+1 (hop)
        fallback_orders = []
        for order in skipped:
            reason = order.get("skip_reason", "")
            if "EV" not in reason and "Fragile" not in reason:
                continue
            bid_c = int(order.get("current_bid_c", 0))
            ask_c = int(order.get("current_ask_c", 100))
            if bid_c < 1 or bid_c >= ask_c:
                continue
            fallback = {k: v for k, v in order.items()
                        if not k.startswith("skip_") and k not in ("v4_ev_c", "v4_worst_ev", "v4_spread_c")}
            fallback["limit_price_c"] = bid_c
            fallback_orders.append(fallback)
        if fallback_orders:
            more_accepted, _ = validate_orders(
                fallback_orders,
                observed_days=observed_days,
                verbose=False,
            )
            accepted.extend(more_accepted)

        # 5. Apply toxicity guard
        toxicity_passed = []
        for order in accepted:
            ticker = order.get("ticker", "")
            side = order.get("side", "YES").upper()
            price_c = float(order.get("limit_price_c", 50))
            ev_c = order.get("v4_ev_c", order.get("ev_if_filled_c", 0))
            best_bid_c = float(order.get("current_bid_c", 0))

            passes, reason = passes_toxicity_guard(
                our_limit_c=price_c,
                best_bid_c=best_bid_c,
                ev_if_filled_c=ev_c,
            )
            if passes:
                toxicity_passed.append(order)
            else:
                logger.debug(f"[TOXICITY] {t(ticker)} {side}: blocked — {reason}")

        # 6. Check balance
        if client and not dry_run:
            try:
                balance = client.get_balance()
                state.api_calls += 1
                cash_c = balance.get("balance", 0)
                if cash_c < config.min_cash_balance_c:
                    if not state.low_balance_warned:
                        logger.warning(
                            f"[CYCLE] Low balance: ${cash_c/100:.2f} < "
                            f"${config.min_cash_balance_c/100:.2f} minimum — "
                            "skipping submissions until funded"
                        )
                        state.low_balance_warned = True
                    return
                state.low_balance_warned = False
            except Exception as e:
                logger.error(f"[CYCLE] Balance check failed: {e}")

        # 7. Diff against resting orders and submit changes
        refresh_resting_orders(state, client)
        diff_and_submit_orders(toxicity_passed, state.resting_buy_orders, client, dry_run, state=state)

        # 8. Place/update exit orders for held positions
        if client or dry_run:
            try:
                _place_exit_orders_cycle(client, predict_csv_path, dry_run, state)
            except Exception as e:
                logger.error(f"[CYCLE] Exit orders error: {e}")

        # 9. Refresh resting orders after changes
        refresh_resting_orders(state, client)

        # 10. Scan for taker opportunities and notify
        try:
            scan_taker_opportunities(predict_csv_path, orderbooks, state)
        except Exception as e:
            logger.error(f"[CYCLE] Taker scan error: {e}")

        elapsed = time.time() - cycle_start
        logger.info(
            f"[CYCLE #{state.full_cycles}] "
            f"{len(toxicity_passed)} orders active, "
            f"{len(skipped)} skipped, "
            f"{len(state.active_tickers)} tickers | "
            f"{elapsed:.1f}s"
        )

    except FileNotFoundError as e:
        logger.error(f"[CYCLE] File not found: {e}")
    except ValueError as e:
        logger.error(f"[CYCLE] Value error: {e}")
    except Exception as e:
        state.consecutive_api_errors += 1
        logger.error(f"[CYCLE] Error: {e}")


def _place_exit_orders_cycle(
    client: KalshiClient | None,
    predict_csv_path: str,
    dry_run: bool,
    state: LiveState | None = None,
):
    """Place/update exit orders for held positions within a full cycle."""
    import pandas as pd

    if not DEFAULT_MAKER_EXIT_CONFIG.enabled:
        return

    if client is None:
        return

    try:
        positions = client.get_positions()
        held = [p for p in positions if p.get("position", 0) != 0]
        if not held:
            return

        predictions_df = pd.read_csv(predict_csv_path)
        exit_orders = generate_exit_orders(
            held,
            predictions_df,
            min_exit_ev_c=DEFAULT_MAKER_EXIT_CONFIG.min_exit_ev_c,
        )

        from src.cli_maker_submit import place_exit_orders, cancel_stale_exit_orders
        stale_results = cancel_stale_exit_orders(
            client=client,
            positions=held,
            predictions_df=predictions_df,
            exit_orders=exit_orders,
            dry_run=dry_run,
            min_exit_ev_c=DEFAULT_MAKER_EXIT_CONFIG.min_exit_ev_c,
        )
        if stale_results:
            cancelled = [r for r in stale_results if r.get("action") == "cancelled_stale"]
            if cancelled:
                logger.info(f"[EXIT] Cancelled {len(cancelled)} stale sell order(s)")

        if not exit_orders:
            return

        # Filter out previously failed exits
        if state and state.failed_exits:
            filtered = []
            for eo in exit_orders:
                key = (eo.get("ticker", ""), eo.get("side", ""), int(eo.get("sell_price_c", 0)))
                if key in state.failed_exits:
                    continue
                filtered.append(eo)
            exit_orders = filtered
            if not exit_orders:
                return

        results = place_exit_orders(client, exit_orders, dry_run=dry_run)

        if state and results:
            for r in results:
                if "failed" in r.get("action", ""):
                    key = (r.get("ticker", ""), r.get("side", ""), int(r.get("sell_price_c", 0)))
                    state.failed_exits.add(key)
                    logger.info(f"[EXIT] Suppressing future retries for {key}")

    except Exception as e:
        logger.error(f"[EXIT ORDERS] Error: {e}")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def run_preflight_checks() -> tuple[bool, list[str]]:
    """Run pre-flight checks. Returns (passed, list_of_reasons)."""
    from src.cli_maker_submit import is_model_data_fresh

    issues = []

    # Check data freshness
    fresh, reason = is_model_data_fresh()
    if not fresh:
        issues.append(f"Data stale: {reason}")

    # Check maker orders file exists
    if not Path("data/latest_maker_orders.json").exists():
        issues.append("No maker orders file (run cli_predict --save --maker first)")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# AAA data polling
# ---------------------------------------------------------------------------

def get_local_latest_aaa_date() -> str:
    """Get the latest date in aaa_daily.csv."""
    try:
        import pandas as pd
        df = pd.read_csv("data/aaa_daily.csv", dtype={"date": str})
        return df["date"].iloc[-1]
    except Exception:
        return ""


def check_aaa_update(state: LiveState, dry_run: bool = False) -> bool:
    """Check for new AAA price. If today's price is missing, fetch it and re-predict.

    Returns True if predictions were regenerated.
    """
    today_str = date.today().isoformat()

    # Already have today's data
    if state.last_aaa_date == today_str:
        logger.debug(f"[AAA] Already have today's data ({today_str})")
        return False

    # Check CSV directly
    current_latest = get_local_latest_aaa_date()
    if current_latest == today_str:
        state.last_aaa_date = today_str
        state.aaa_updated_today = True
        return False  # Data exists but we didn't trigger a regen

    # Try to fetch today's price
    logger.info(f"[AAA] Fetching today's price ({today_str})...")
    try:
        fetch_result = subprocess.run(
            [sys.executable, "-m", "scripts.log_aaa"],
            capture_output=True, text=True, timeout=60,
        )
        if fetch_result.returncode != 0:
            logger.warning(f"[AAA] Fetch failed: {fetch_result.stderr[-200:]}")
            return False
    except Exception as e:
        logger.warning(f"[AAA] Fetch error: {e}")
        return False

    # Verify it was appended
    new_latest = get_local_latest_aaa_date()
    if new_latest != today_str:
        logger.debug(f"[AAA] Price not yet available for {today_str}")
        return False

    logger.info(f"[AAA] NEW DATA: {today_str} fetched successfully")

    # Re-predict with new data
    logger.info("[AAA] Running predict pipeline...")
    pipeline_start = time.time()
    predict_cmd = [
        sys.executable, "-m", "src.cli_predict",
        "--auto-week", "--auto-asof", "--save", "--maker",
    ]
    if not dry_run:
        predict_cmd.append("--auto-bankroll")
    pred_result = subprocess.run(predict_cmd, capture_output=True, text=True)
    if pred_result.returncode != 0:
        logger.error(f"[AAA] cli_predict failed:\n{pred_result.stderr[-500:]}")
        return False

    elapsed = time.time() - pipeline_start
    state.aaa_updated_today = True
    state.last_aaa_date = today_str
    state.aaa_regen_count += 1
    state.notified_taker_opps.clear()

    logger.info(f"[AAA] Pipeline complete in {elapsed:.0f}s — predictions refreshed")

    # Audible notification (macOS)
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
    except Exception:
        pass

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time gas maker trading loop with adverse selection protection"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="No orders placed, just monitoring and logging"
    )
    parser.add_argument(
        "--cancel-on-exit", action="store_true",
        help="Cancel all resting orders on Ctrl+C"
    )
    parser.add_argument(
        "--fast-tick", type=float, default=None,
        help="Override fast tick interval (seconds)"
    )
    parser.add_argument(
        "--full-cycle", type=float, default=None,
        help="Override full cycle interval (seconds)"
    )
    parser.add_argument(
        "--maker-orders", type=str, default="data/latest_maker_orders.json",
        help="Path to maker orders JSON"
    )
    parser.add_argument(
        "--predict-csv", type=str, default="data/latest_predict.csv",
        help="Path to prediction CSV"
    )
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight data freshness checks"
    )
    parser.add_argument(
        "--no-aaa-poll", action="store_true",
        help="Disable AAA data polling"
    )
    parser.add_argument(
        "--no-taker-alerts", action="store_true",
        help="Disable taker opportunity Discord notifications"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Also write plain-text logs to this file (for dashboard)"
    )
    return parser.parse_args()


def _acquire_pid_lock() -> None:
    """Write PID file, aborting if another instance is already running."""
    if _PID_FILE.exists():
        try:
            old_pid = int(_PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            print(f"live_maker already running (PID {old_pid}). Exiting.", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(f"live_maker PID {old_pid} exists (different user). Exiting.", file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            _PID_FILE.unlink(missing_ok=True)
    _PID_FILE.write_text(str(os.getpid()))


def _release_pid_lock() -> None:
    """Remove PID file if it belongs to this process."""
    try:
        if _PID_FILE.exists() and int(_PID_FILE.read_text().strip()) == os.getpid():
            _PID_FILE.unlink(missing_ok=True)
    except (ValueError, OSError):
        pass


def main():
    args = parse_args()
    setup_logging(log_file=args.log_file)

    _acquire_pid_lock()
    atexit.register(_release_pid_lock)

    # Build config with overrides
    overrides = {}
    if args.fast_tick is not None:
        overrides["fast_tick_s"] = args.fast_tick
    if args.full_cycle is not None:
        overrides["full_cycle_s"] = args.full_cycle

    if overrides:
        from dataclasses import asdict
        base = asdict(DEFAULT_LIVE_MAKER_CONFIG)
        base.update(overrides)
        config = LiveMakerConfig(**base)
    else:
        config = DEFAULT_LIVE_MAKER_CONFIG

    # Pre-flight checks
    if not args.skip_preflight:
        passed, issues = run_preflight_checks()
        if not passed:
            # When AAA polling is enabled, staleness is expected — poll will fix it
            stale_issues = [i for i in issues if i.startswith("Data stale")]
            hard_issues = [i for i in issues if not i.startswith("Data stale")]
            if stale_issues and not args.no_aaa_poll:
                for issue in stale_issues:
                    logger.warning(f"[PREFLIGHT] {issue} (AAA polling will update)")
            if hard_issues:
                logger.error("[PREFLIGHT] Failed:")
                for issue in hard_issues:
                    logger.error(f"  - {issue}")
                logger.error("Use --skip-preflight to override")
                sys.exit(1)
        logger.info("[PREFLIGHT] All checks passed")

    # Regenerate predictions at startup so orders reflect latest code
    logger.info("[INIT] Regenerating predictions...")
    predict_cmd = [
        sys.executable, "-m", "src.cli_predict",
        "--auto-week", "--auto-asof", "--save", "--maker",
    ]
    if not args.dry_run:
        predict_cmd.append("--auto-bankroll")
    result = subprocess.run(predict_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"[INIT] Prediction failed:\n{result.stderr[-500:]}")
        sys.exit(1)
    logger.info("[INIT] Predictions regenerated")

    # Initialize client
    client: KalshiClient | None = None
    if not args.dry_run:
        try:
            client = KalshiClient()
            logger.info("[INIT] Kalshi client initialized (PRODUCTION)")
        except Exception as e:
            logger.error(f"[INIT] Failed to initialize client: {e}")
            sys.exit(1)
    else:
        logger.info("[INIT] Dry run mode — no API client")

    # Fetch initial state
    if client:
        try:
            balance = client.get_balance()
            cash_c = balance.get("balance", 0)
            positions = client.get_positions()
            held = [p for p in positions if p.get("position", 0) != 0]
            open_orders = client.get_open_orders()
            buy_orders = [o for o in open_orders if o.get("action", "").lower() != "sell"]
            sell_orders = [o for o in open_orders if o.get("action", "").lower() == "sell"]

            logger.info(f"[INIT] Balance: ${cash_c/100:.2f}")
            logger.info(f"[INIT] Positions: {len(held)} held")
            logger.info(f"[INIT] Open orders: {len(buy_orders)} buy, {len(sell_orders)} sell")

            if cash_c < config.min_cash_balance_c:
                logger.error(
                    f"[INIT] Insufficient balance: ${cash_c/100:.2f} "
                    f"< ${config.min_cash_balance_c/100:.2f}"
                )
                sys.exit(1)
        except Exception as e:
            logger.error(f"[INIT] Failed to fetch account state: {e}")
            sys.exit(1)

    # Initialize state
    state = LiveState(mode_since=time.time())
    state.last_aaa_date = get_local_latest_aaa_date()
    state.taker_alerts_enabled = not args.no_taker_alerts

    # Graceful shutdown handler
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("[SHUTDOWN] Force exit")
            sys.exit(1)
        shutdown_requested = True
        logger.info("[SHUTDOWN] Caught signal, shutting down...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Main loop
    try:
        last_fast = 0.0
        last_fill = 0.0
        last_full = 0.0

        # Build side lookup from maker orders
        try:
            maker_orders, _ = load_maker_orders(args.maker_orders)
            side_lookup = build_side_lookup(maker_orders)
            state.all_tickers = list({o.get("ticker", "") for o in maker_orders})
        except Exception as e:
            logger.error(f"Failed to load maker orders: {e}")
            sys.exit(1)

        # Initial resting orders refresh
        refresh_resting_orders(state, client)

        logger.info(
            f"[LOOP] Starting — mode={state.mode}, "
            f"tickers={len(state.all_tickers)}, "
            f"active={len(state.active_tickers)}, "
            f"fast_tick={config.fast_tick_s}s, "
            f"full_cycle={config.full_cycle_s}s, "
            f"dry_run={args.dry_run}"
        )

        while not shutdown_requested:
            now = time.time()

            # --- Check API error limit ---
            if state.consecutive_api_errors >= config.max_consecutive_api_errors:
                logger.critical(
                    f"[HALT] {state.consecutive_api_errors} consecutive API errors. "
                    f"Stopping loop."
                )
                break

            now_et = datetime.now(ET)

            # --- Market close check: Sunday midnight ET (Mon 00:00 ET) ---
            if now_et.weekday() == 0 and now_et.hour == 0:
                logger.info("[MARKET CLOSE] Sunday midnight ET — market closed. Cancelling orders and shutting down.")
                if client:
                    try:
                        results = client.cancel_all_orders()
                        success = sum(1 for r in results if r.success)
                        logger.info(f"[MARKET CLOSE] Cancelled {success}/{len(results)} orders")
                    except Exception as e:
                        logger.error(f"[MARKET CLOSE] Error cancelling orders: {e}")
                break

            # --- AAA check (every 60s, not yet updated today) ---
            if (not args.no_aaa_poll
                    and not state.aaa_updated_today
                    and (now - state.last_aaa_check) >= 60.0):
                state.last_aaa_check = now
                if check_aaa_update(state, args.dry_run):
                    last_full = 0.0  # force immediate full cycle
                    # Reload side_lookup since maker_orders.json changed
                    maker_orders, _ = load_maker_orders(args.maker_orders)
                    side_lookup = build_side_lookup(maker_orders)
                    state.all_tickers = list({o.get("ticker", "") for o in maker_orders})

            # Midnight reset for overnight runs
            if now_et.hour == 0 and state.aaa_updated_today:
                state.aaa_updated_today = False
                state.last_aaa_date = get_local_latest_aaa_date()

            # --- Fast tick: orderbook monitoring ---
            if now - last_fast >= config.fast_tick_s:
                last_fast = now

                active_orderbooks = {}
                for ticker in list(state.active_tickers):
                    try:
                        ob = fetch_orderbook(ticker)
                        state.api_calls += 1
                        if ob:
                            active_orderbooks[ticker] = ob
                            side = side_lookup.get(ticker, "YES")
                            mid = compute_mid(ob, side)
                            spread = compute_spread(ob, side)
                            if mid is not None:
                                if ticker not in state.mid_prices:
                                    state.mid_prices[ticker] = deque(maxlen=300)
                                state.mid_prices[ticker].append((now, mid))
                            if spread is not None:
                                if ticker not in state.spread_history:
                                    state.spread_history[ticker] = deque(maxlen=300)
                                state.spread_history[ticker].append((now, spread))
                    except Exception:
                        state.consecutive_api_errors += 1

                if state.mode == "ACTIVE" and active_orderbooks:
                    signal_reason = check_adverse_selection(state, config, active_orderbooks)
                    if signal_reason:
                        enter_defensive(state, client, signal_reason, args.dry_run)

            if state.mode == "DEFENSIVE":
                break  # DEFENSIVE is terminal — exit loop

            # --- Fill check ---
            if now - last_fill >= config.fill_check_s:
                last_fill = now
                detect_fills(state, client, args.dry_run)

                if state.mode == "ACTIVE":
                    fill_signal = check_fill_rate(state, config)
                    if fill_signal:
                        enter_defensive(state, client, fill_signal, args.dry_run)
                        break  # DEFENSIVE is terminal

            # --- Full cycle (ACTIVE only) ---
            if state.mode == "ACTIVE" and now - last_full >= config.full_cycle_s:
                last_full = now

                # Reload side lookup in case orders changed
                try:
                    maker_orders, _ = load_maker_orders(args.maker_orders)
                    side_lookup = build_side_lookup(maker_orders)
                    state.all_tickers = list({o.get("ticker", "") for o in maker_orders})
                except Exception as e:
                    logger.error(f"Failed to reload maker orders: {e}")

                run_full_cycle(
                    client, state, config, args.dry_run,
                    args.maker_orders, args.predict_csv,
                )

            # 100ms base tick to avoid busy-wait
            time.sleep(0.1)

    except Exception as e:
        logger.error(f"[FATAL] Unexpected error: {e}", exc_info=True)

    # --- Shutdown ---
    if args.cancel_on_exit and client:
        logger.warning("[SHUTDOWN] Cancelling all resting orders...")
        try:
            results = client.cancel_all_orders()
            success = sum(1 for r in results if r.success)
            logger.info(f"[SHUTDOWN] Cancelled {success}/{len(results)} orders")
        except Exception as e:
            logger.error(f"[SHUTDOWN] Error cancelling orders: {e}")

    # Session summary
    uptime = time.time() - state.mode_since
    logger.info("=" * 60)
    logger.info("[SESSION SUMMARY]")
    logger.info(f"  Full cycles:       {state.full_cycles}")
    logger.info(f"  Total fills:       {state.total_fills} ({state.total_fills_qty} contracts)")
    logger.info(f"  State transitions: {state.state_transitions}")
    logger.info(f"  API calls:         {state.api_calls}")
    logger.info(f"  AAA regens:        {state.aaa_regen_count}")
    logger.info(f"  Taker alerts:      {state.taker_alerts_sent}")
    logger.info(f"  Final mode:        {state.mode}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
