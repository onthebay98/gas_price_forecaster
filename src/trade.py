from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd


Side = Literal["YES", "NO"]


# =============================================================================
# Phase 3-8: New EV, PnL, and Maker Controls (v4 plan)
# =============================================================================


def calc_ev_if_filled_c(
    p_model: float,
    side: str,
    price_paid_c: float,
    is_maker: bool,
    qty: int = 1,
    fee_schedule: "FeeSchedule | None" = None,
) -> float:
    """
    Compute EV if order fills (in cents, per contract).

    Uses the REAL fee engine (kalshi_fee_total_dollars).
    Returns: EV in cents per contract.

    CRITICAL INVARIANT: kalshi_fee_total_dollars() returns TOTAL fee for
    the entire order (not per-contract). We divide by qty to get per-contract.

    Args:
        p_model: Model probability for YES (P(YES wins))
        side: "YES" or "NO" - the side being traded
        price_paid_c: Price paid for this side in cents
        is_maker: True for maker orders (lower fees), False for taker
        qty: Number of contracts (for fee calculation)
        fee_schedule: Optional custom fee schedule

    Returns:
        EV in cents per contract
    """
    if fee_schedule is None:
        fee_schedule = FeeSchedule()

    # Convert p_model to p_win based on side
    # YES wins when event happens (p_model)
    # NO wins when event doesn't happen (1 - p_model)
    p_win = p_model if side == "YES" else (1.0 - p_model)

    price_paid_d = price_paid_c / 100.0

    # Get TOTAL fee for whole order (in dollars)
    total_fee_d = kalshi_fee_total_dollars(
        price_dollars=price_paid_d,
        contracts=qty,
        is_maker=is_maker,
        fee_schedule=fee_schedule,
    )

    # Convert to per-contract cents
    fee_per_contract_c = (total_fee_d / qty) * 100.0 if qty > 0 else 0.0

    # Payout if win = 100c, if lose = 0c
    # EV = p_win * 100 + (1 - p_win) * 0 - price_paid - fee
    expected_payout_c = p_win * 100.0
    ev_per_contract_c = expected_payout_c - price_paid_c - fee_per_contract_c
    return ev_per_contract_c


def calc_ev_per_cost(
    p_model: float,
    side: str,
    price_paid_c: float,
    is_maker: bool,
    qty: int = 1,
    fee_schedule: "FeeSchedule | None" = None,
) -> float:
    """
    Capital efficiency: EV cents / cost cents.

    Returns ratio (dimensionless). E.g., 0.05 = 5% return on capital.
    Use this as a secondary gate to avoid tying up bankroll for tiny EV.

    Args:
        p_model: Model probability for YES (P(YES wins))
        side: "YES" or "NO" - the side being traded
        price_paid_c: Price paid for this side in cents
        is_maker: True for maker orders
        qty: Number of contracts
        fee_schedule: Optional custom fee schedule

    Returns:
        EV/cost ratio (dimensionless)
    """
    ev_c = calc_ev_if_filled_c(p_model, side, price_paid_c, is_maker, qty, fee_schedule)
    if price_paid_c <= 0:
        return 0.0
    return ev_c / price_paid_c


@dataclass
class Position:
    """
    Represents a position for PnL and worst-case loss computation.

    threshold: The market threshold (e.g., 2_150_000 for "> 2.15M")
    side: "YES" or "NO"
    avg_price_paid_c: Average price paid per contract in cents
    qty: Number of contracts
    fees_paid_c: Total fees paid for this position in cents
    comparator: ">" or ">=" (from market rules, not bucket labels)
    """
    threshold: float
    side: str
    avg_price_paid_c: float
    qty: int
    fees_paid_c: float
    comparator: str = ">"


def compute_pnl(pos: Position, realized_avg: float) -> float:
    """
    Compute PnL for a single position given realized outcome.

    Payout:
    - win: +100c per contract
    - lose: +0c per contract

    Net PnL: qty * (payout_c - price_paid_c) - fees_paid_c

    Args:
        pos: Position to evaluate
        realized_avg: The realized weekly average (same units as threshold)

    Returns:
        PnL in cents (can be negative)
    """
    if pos.comparator == ">":
        threshold_exceeded = realized_avg > pos.threshold
    else:  # ">="
        threshold_exceeded = realized_avg >= pos.threshold

    # YES wins if threshold exceeded, NO wins if not exceeded
    wins = threshold_exceeded == (pos.side == "YES")

    payout_c = 100.0 if wins else 0.0
    pnl = pos.qty * (payout_c - pos.avg_price_paid_c) - pos.fees_paid_c
    return pnl


def compute_worst_case_loss(positions: list[Position]) -> float:
    """
    Compute worst-case loss over outcome grid.

    Grid: all thresholds ± epsilon + bounds derived from positions.
    Payoff only changes at threshold crossings.

    IMPORTANT: Does NOT use hardcoded bounds. Derives from actual positions.

    Args:
        positions: List of positions to evaluate

    Returns:
        Worst-case loss in cents (positive number, 0 if no loss possible)
    """
    if not positions:
        return 0.0

    # Grid: all thresholds from positions
    thresholds = sorted(set(p.threshold for p in positions))

    # Derive bounds from actual thresholds (NOT hardcoded)
    MARGIN = 1.0  # $1.00 margin (in $/gal, same unit as threshold)
    min_bound = min(thresholds) - MARGIN
    max_bound = max(thresholds) + MARGIN

    outcome_grid = list(thresholds) + [min_bound, max_bound]

    # Also include just below/above each threshold for boundary cases
    eps = 0.001  # $0.001/gal
    for t in thresholds:
        outcome_grid.extend([t - eps, t + eps])

    worst_loss = 0.0
    for realized_avg in outcome_grid:
        total_pnl = sum(compute_pnl(p, realized_avg) for p in positions)
        worst_loss = max(worst_loss, -total_pnl)

    return worst_loss


def _extract_threshold_from_ticker(ticker: str) -> float:
    """
    Extract the threshold value from a gas price ticker.

    Ticker format: KXAAAGASW-26MAR23-3.670
    The threshold is the last dash-separated segment (e.g., 3.670).

    Returns threshold in $/gal (e.g., 3.670 for $3.670/gal).
    """
    try:
        parts = ticker.split("-")
        if len(parts) >= 3:
            return float(parts[-1])
    except (ValueError, IndexError):
        pass
    # Default fallback
    return 3.50


def positions_from_api(api_positions: list[dict]) -> list[Position]:
    """
    Convert Kalshi API positions to Position objects for worst-case calc.

    Args:
        api_positions: Raw positions from client.get_positions()

    Returns:
        List of Position objects for compute_worst_case_loss()
    """
    positions = []
    for p in api_positions:
        if p.get("position", 0) == 0:
            continue

        ticker = p.get("ticker", "")
        position = p.get("position", 0)

        # Extract threshold from ticker
        threshold = _extract_threshold_from_ticker(ticker)

        if position > 0:
            side = "YES"
            qty = position
        else:
            side = "NO"
            qty = abs(position)

        # Use market_exposure / qty as proxy for avg_price_paid
        # market_exposure is in cents
        exposure_c = abs(p.get("market_exposure", 0))
        avg_price_c = (exposure_c / qty) if qty > 0 else 50

        positions.append(Position(
            threshold=threshold,
            side=side,
            avg_price_paid_c=avg_price_c,
            qty=qty,
            fees_paid_c=p.get("fees_paid", 0),
            comparator=">",
        ))

    return positions


def ev_robust_under_slippage(
    p_model: float,
    side: str,
    price_c: float,
    is_maker: bool,
    slippage_ticks: int = 2,
    min_ev_c: float = 0.5,
) -> tuple[bool, float]:
    """
    Check if EV stays >= min_ev after adverse price move.

    For takers: assume fill at worse price (slippage_ticks higher cost)
    For makers: assume price moves against us after fill

    Args:
        p_model: Model probability for YES (P(YES wins))
        side: "YES" or "NO"
        price_c: Current price in cents
        is_maker: True for maker orders
        slippage_ticks: Number of ticks of adverse move (default 2)
        min_ev_c: Minimum EV required after slippage (default 0.5c)

    Returns:
        (is_robust, worst_ev_c) where is_robust = True if EV stays above min
    """
    worst_price_c = price_c + slippage_ticks  # Worse fill / adverse move
    if worst_price_c > 99:
        worst_price_c = 99

    worst_ev_c = calc_ev_if_filled_c(p_model, side, worst_price_c, is_maker)
    return worst_ev_c >= min_ev_c, worst_ev_c


def passes_toxicity_guard(
    our_limit_c: float,
    best_bid_c: float,
    ev_if_filled_c: float,
    is_buy: bool = True,
    min_ev_c: float = 0.5,
    aggressive_improve_min_ev_c: float = 10.0,
) -> tuple[bool, str]:
    """
    Toxicity guard: don't aggressively improve the market.

    For maker orders, being "too aggressive" means you're likely
    to get filled exactly when the market knows something you don't.

    Rule: Don't improve best price by >1 tick unless EV is very high.

    NOTE: The action (BUY/SELL) determines improvement direction, NOT the side (YES/NO).
    - BUY orders post bids. Improving = bidding higher than current best bid.
    - SELL orders post asks. Improving = asking lower than current best ask.

    This codebase's maker orders are always BUY orders (acquiring contracts).

    Args:
        our_limit_c: Our proposed limit price in cents
        best_bid_c: Current best bid for this side's book (for BUY orders)
                    For SELL orders, pass best_ask instead.
        ev_if_filled_c: EV if our order fills (from calc_ev_if_filled_c)
        is_buy: True for BUY orders (default), False for SELL orders
        min_ev_c: Base EV threshold for 1-tick improvement
        aggressive_improve_min_ev_c: High EV bar for >1 tick improvement

    Returns:
        (passes, reason)
    """
    # How much are we improving the market?
    # BUY: improving = bidding higher (our_limit > best_bid)
    # SELL: improving = asking lower (our_limit < best_ask)
    if is_buy:
        improvement_c = our_limit_c - best_bid_c
    else:
        improvement_c = best_bid_c - our_limit_c  # best_bid_c is actually best_ask for sells

    # If not improving (joining or behind), always OK
    if improvement_c <= 0:
        return True, "joining or behind best"

    # If improving by 1 tick (1c), OK if meets base EV threshold
    if improvement_c <= 1.0:
        if ev_if_filled_c >= min_ev_c:
            return True, f"1-tick improve with EV={ev_if_filled_c:.1f}c >= {min_ev_c}c"
        return False, f"1-tick improve but EV={ev_if_filled_c:.1f}c < {min_ev_c}c"

    # If improving by >1 tick, require MUCH higher EV bar
    if ev_if_filled_c >= aggressive_improve_min_ev_c:
        return True, f">{improvement_c:.0f}-tick improve with EV={ev_if_filled_c:.1f}c >= {aggressive_improve_min_ev_c}c"
    return False, f">{improvement_c:.0f}-tick improve but EV={ev_if_filled_c:.1f}c < {aggressive_improve_min_ev_c}c"


@dataclass
class InfoReleaseWindow:
    """Defines a time window when major information is released."""
    weekday: int  # 0=Mon, 1=Tue, ...
    start_hour: int  # ET (24-hour)
    end_hour: int  # ET (24-hour)
    description: str


# AAA gas prices update daily — no specific high-impact release window.
# Placeholder: can add EIA weekly report (Wednesday 10:30 AM ET) if relevant.
GAS_RELEASE_WINDOWS: list[InfoReleaseWindow] = []


def should_pause_maker(
    current_time: datetime,
    release_windows: list[InfoReleaseWindow] | None = None,
) -> tuple[bool, str]:
    """
    Pause maker activity around major information releases.

    Don't rest maker orders through known info release times.
    You're most likely to get filled exactly when you're wrong.

    Args:
        current_time: Current datetime (assumed ET)
        release_windows: List of info release windows (default: GAS_RELEASE_WINDOWS)

    Returns:
        (should_pause, reason)
    """
    if release_windows is None:
        release_windows = GAS_RELEASE_WINDOWS

    current_weekday = current_time.weekday()
    current_hour = current_time.hour

    for window in release_windows:
        if current_weekday == window.weekday:
            if window.start_hour <= current_hour < window.end_hour:
                return True, window.description

    return False, ""


@lru_cache(maxsize=256)
def get_market_rules(ticker: str) -> dict:
    """
    Fetch market rules from Kalshi API (cached).

    CRITICAL: Bucket labels like "> 2.15M" are NOT authoritative.
    The actual resolution rules come from the market's rules_primary field.

    Args:
        ticker: Market ticker

    Returns:
        dict with:
        - comparator: ">" or ">="
        - rules_text: str (for debugging)
    """
    try:
        from src.kalshi_client import KalshiClient
        client = KalshiClient()
        market = client.get_market(ticker)

        if market is None:
            return {"comparator": ">", "rules_text": ""}

        rules_text = market.get("rules_primary", "") or ""

        # Parse comparator from rules text
        # Look for "greater than" vs "greater than or equal to" or "at least"
        rules_lower = rules_text.lower()
        if "greater than or equal" in rules_lower:
            comparator = ">="
        elif "at least" in rules_lower:
            comparator = ">="
        else:
            comparator = ">"  # Default to strict >

        return {
            "comparator": comparator,
            "rules_text": rules_text,
        }
    except Exception:
        # Fallback if API fails
        return {"comparator": ">", "rules_text": ""}


def get_threshold_comparator(ticker: str, default: str = ">") -> str:
    """
    Return ">" or ">=" from actual market rules.

    Falls back to default if rules fetch fails.

    Args:
        ticker: Market ticker
        default: Fallback comparator if API unavailable

    Returns:
        ">" or ">="
    """
    try:
        rules = get_market_rules(ticker)
        return rules.get("comparator", default)
    except Exception:
        return default


@dataclass
class TradeThresholds:
    """Thresholds for a specific trade opportunity.

    These are computed dynamically based on observed_days, hours_to_close, etc.
    """
    min_prob: float         # Secondary: enforce extremes (>=0.80)
    min_ev_c: float         # PRIMARY: net EV if filled (real fees)
    size_mult: float        # Kelly multiplier


def get_maker_thresholds(
    observed_days: int,
    config: "MakerDefensiveConfig | None" = None,
) -> TradeThresholds:
    """
    Compute maker thresholds for a specific opportunity.

    EV_maker_if_filled is the primary gate.

    Args:
        observed_days: Days of price data observed this week (0=Mon, 1=Tue, etc.)
        config: MakerDefensiveConfig (uses default if None)

    Returns:
        TradeThresholds for this opportunity
    """
    if config is None:
        from src.config import DEFAULT_MAKER_DEFENSIVE_CONFIG
        config = DEFAULT_MAKER_DEFENSIVE_CONFIG

    # min_ev_c=0 here; zone-based EV gating in validate_orders handles thresholds
    # size_mult=1.0; half-Kelly is applied in generate_maker_orders
    return TradeThresholds(
        min_prob=config.min_prob,
        min_ev_c=0.0,
        size_mult=1.0,
    )


@dataclass(frozen=True)
class FeeSchedule:
    # Kalshi Oct 2025 fee schedule:
    # taker: fees = round up(0.07 * C * P * (1-P))
    # maker: fees = round up(0.0175 * C * P * (1-P))
    # round up means to next cent
    taker_mult: float = 0.07
    maker_mult: float = 0.0175


def _round_up_cent(x: float) -> float:
    # rounds up to next cent
    return math.ceil(x * 100.0 - 1e-12) / 100.0


def kalshi_fee_total_dollars(
    *,
    price_dollars: float,
    contracts: int,
    is_maker: bool,
    fee_schedule: FeeSchedule = FeeSchedule(),
) -> float:
    """
    Fee formula from Kalshi fee schedule PDF:
      fees = round up(mult * C * P * (1-P))
    where P is price in dollars, C contracts.
    """  # Source: Kalshi fee schedule PDF. :contentReference[oaicite:2]{index=2}
    if contracts <= 0:
        return 0.0
    if price_dollars <= 0.0 or price_dollars >= 1.0:
        # At 0 or 1 this is degenerate; still handle safely.
        price_dollars = min(max(price_dollars, 0.0), 1.0)

    mult = fee_schedule.maker_mult if is_maker else fee_schedule.taker_mult
    raw = mult * float(contracts) * float(price_dollars) * float(1.0 - price_dollars)
    return _round_up_cent(raw)


def _ev_yes_per_contract(p_true: float, yes_price: float, fee_per_contract: float) -> float:
    # EV (profit) per contract in dollars
    # payout = 1 if true, 0 if false
    # EV = p_true*1 - price - fee
    return float(p_true) - float(yes_price) - float(fee_per_contract)


def _ev_no_per_contract(p_true: float, no_price: float, fee_per_contract: float) -> float:
    # NO pays 1 if event is false, 0 if true
    # EV = (1-p_true)*1 - no_price - fee
    return (1.0 - float(p_true)) - float(no_price) - float(fee_per_contract)


def add_trade_metrics(
    tbl: pd.DataFrame,
    *,
    is_maker: bool = False,
    fee_schedule: FeeSchedule = FeeSchedule(),
    contracts_for_fee: int = 100,
) -> pd.DataFrame:
    """
    Adds fee + EV columns based on prices and model probability (P_model).
    Uses VWAP columns if available (yes_vwap, no_vwap), otherwise falls back to ask prices.
    Uses contracts_for_fee to compute a realistic per-contract fee (rounding matters).
    """
    out = tbl.copy()

    # Use VWAP if available, otherwise fall back to best ask
    # VWAP reflects actual fill cost including liquidity/slippage
    if "yes_vwap" in out.columns:
        out["yes_ask_d"] = pd.to_numeric(out["yes_vwap"], errors="coerce") / 100.0
    else:
        out["yes_ask_d"] = pd.to_numeric(out["yes_ask"], errors="coerce") / 100.0

    if "no_vwap" in out.columns:
        out["no_ask_d"] = pd.to_numeric(out["no_vwap"], errors="coerce") / 100.0
    else:
        out["no_ask_d"] = pd.to_numeric(out["no_ask"], errors="coerce") / 100.0

    # fees depend on *price* and *contracts* and are rounded up to the cent on the total,
    # so compute total fee for a standard clip, then divide.
    def fee_per_contract(price_d: float) -> float:
        if not np.isfinite(price_d):
            return float("nan")
        fee_total = kalshi_fee_total_dollars(
            price_dollars=float(price_d),
            contracts=int(contracts_for_fee),
            is_maker=is_maker,
            fee_schedule=fee_schedule,
        )
        return float(fee_total) / float(contracts_for_fee)

    out["fee_yes_per"] = out["yes_ask_d"].apply(fee_per_contract)
    out["fee_no_per"] = out["no_ask_d"].apply(fee_per_contract)

    out["ev_yes_per"] = [
        _ev_yes_per_contract(p, y, f) if np.isfinite(p) and np.isfinite(y) and np.isfinite(f) else np.nan
        for p, y, f in zip(out["P_model"], out["yes_ask_d"], out["fee_yes_per"])
    ]
    out["ev_no_per"] = [
        _ev_no_per_contract(p, n, f) if np.isfinite(p) and np.isfinite(n) and np.isfinite(f) else np.nan
        for p, n, f in zip(out["P_model"], out["no_ask_d"], out["fee_no_per"])
    ]

    # Breakeven p_true for YES: p >= price + fee
    out["p_be_yes"] = out["yes_ask_d"] + out["fee_yes_per"]
    # Breakeven p_true for NO: (1-p) >= price + fee  -> p <= 1 - (price + fee)
    out["p_be_no"] = 1.0 - (out["no_ask_d"] + out["fee_no_per"])

    # Best side by EV
    def pick_side(ev_y, ev_n) -> Optional[str]:
        if not np.isfinite(ev_y) and not np.isfinite(ev_n):
            return None
        if not np.isfinite(ev_n):
            return "YES"
        if not np.isfinite(ev_y):
            return "NO"
        return "YES" if ev_y >= ev_n else "NO"

    out["best_side"] = [pick_side(y, n) for y, n in zip(out["ev_yes_per"], out["ev_no_per"])]
    out["best_ev_per"] = out[["ev_yes_per", "ev_no_per"]].max(axis=1, skipna=True)

    # Pretty cents versions
    out["ev_yes_c"] = out["ev_yes_per"] * 100.0
    out["ev_no_c"] = out["ev_no_per"] * 100.0
    out["best_ev_c"] = out["best_ev_per"] * 100.0

    return out


def calculate_maker_order(
    *,
    p_model: float,
    side: Side,
    current_bid: int,  # in cents
    current_ask: int,  # in cents
    fee_schedule: FeeSchedule = FeeSchedule(),
    contracts: int = 100,
    min_edge_pp: float = 3.0,
) -> dict:
    """
    Calculate optimal limit order price for market making.

    For market making, we place limit orders (bids) instead of taking asks.
    The limit price should:
    1. Be above current best bid (to be competitive)
    2. Be below our breakeven price (to maintain edge)
    3. Maintain minimum required edge

    Args:
        p_model: Model probability for YES
        side: "YES" or "NO"
        current_bid: Current best bid in cents
        current_ask: Current best ask in cents
        fee_schedule: Fee schedule for maker fees
        contracts: Number of contracts (for fee calculation)
        min_edge_pp: Minimum edge in percentage points

    Returns:
        dict with:
        - limit_price_c: Recommended limit price in cents
        - ev_if_filled_c: Expected value per contract if filled (cents)
        - edge_pp: Edge in percentage points
        - valid: Whether a valid maker order exists
        - reason: Explanation if not valid
    """
    if side == "YES":
        p_side = p_model
    else:
        p_side = 1.0 - p_model

    # Try bid+1 (hop) first for queue priority, then bid (join) as fallback
    candidates = []
    hop_c = current_bid + 1
    if 1 <= hop_c <= 99 and hop_c < current_ask:
        candidates.append(hop_c)
    if 1 <= current_bid <= 99 and current_bid < current_ask:
        candidates.append(current_bid)

    best_reject_reason = "No positive EV price available"

    for candidate_c in candidates:
        price_d = candidate_c / 100.0
        fee_total = kalshi_fee_total_dollars(
            price_dollars=price_d,
            contracts=contracts,
            is_maker=True,
            fee_schedule=fee_schedule,
        )
        fee_per = fee_total / contracts

        ev_per = p_side - price_d - fee_per
        ev_c = ev_per * 100.0
        edge_pp = (p_side - price_d) * 100.0

        if ev_c >= 0 and edge_pp >= min_edge_pp:
            return {
                "limit_price_c": candidate_c,
                "ev_if_filled_c": round(ev_c, 2),
                "edge_pp": round(edge_pp, 1),
                "valid": True,
                "reason": None,
            }

        # Track most informative rejection reason (edge < minimum is more specific)
        if ev_c >= 0 and edge_pp < min_edge_pp:
            best_reject_reason = f"Edge {edge_pp:.1f}pp < minimum {min_edge_pp}pp"

    return {
        "limit_price_c": None,
        "ev_if_filled_c": None,
        "edge_pp": None,
        "valid": False,
        "reason": best_reject_reason,
    }


def generate_maker_orders(
    tbl: pd.DataFrame,
    *,
    fee_schedule: FeeSchedule = FeeSchedule(),
    contracts_clip: int = 100,
    bankroll: float = 700.0,
    min_edge_very_extreme: float = 0.0,
    min_edge_extreme: float = 0.0,
    min_edge_near: float = 0.0,
    min_edge_middle: float = 0.0,
) -> pd.DataFrame:
    """
    Generate market making order recommendations from prediction table.

    Kelly sizing: Half Kelly (0.5) per bucket, bankroll split equally across N valid
    buckets with 3× oversize multiplier. Asymmetric exposure caps: YES side capped
    at 50% of bankroll (vulnerable to downward shocks), NO side at 100% (shocks help
    NO positions). 40-60% probability region skipped.

    Args:
        tbl: DataFrame with columns: bucket, ticker, P_model, yes_bid, yes_ask, no_bid, no_ask
        fee_schedule: Fee schedule for maker fees
        contracts_clip: Number of contracts for fee calculation (fallback)
        bankroll: Total bankroll for Kelly sizing (cash + positions)
        min_edge_*: Minimum edge by probability region

    Returns:
        DataFrame with maker order recommendations including Kelly-sized contracts
    """
    import numpy as np

    # Three-pass sizing: find valid orders, size with bankroll / N, then cap per-side exposure
    OVERSIZE_MULT = 3.0  # total max exposure = 3× bankroll (most limit orders won't fill)
    kelly_fraction = 1.0

    # Pass 1: collect valid orders without sizing
    pending_orders = []

    for _, row in tbl.iterrows():
        p_model = float(row["P_model"])

        # Determine minimum edge based on probability region
        if p_model <= 0.01 or p_model >= 0.99:
            min_edge = min_edge_very_extreme
        elif p_model < 0.10 or p_model > 0.90:
            min_edge = min_edge_extreme
        elif p_model < 0.20 or p_model > 0.80:
            min_edge = min_edge_near
        else:
            min_edge = min_edge_middle

        # Skip buckets with missing orderbook data
        if any(pd.isna(row[c]) or row[c] is None for c in ["yes_bid", "yes_ask", "no_bid", "no_ask"]):
            continue

        yes_order = calculate_maker_order(
            p_model=p_model,
            side="YES",
            current_bid=int(row["yes_bid"]),
            current_ask=int(row["yes_ask"]),
            fee_schedule=fee_schedule,
            contracts=contracts_clip,
            min_edge_pp=min_edge,
        )

        no_order = calculate_maker_order(
            p_model=p_model,
            side="NO",
            current_bid=int(row["no_bid"]),
            current_ask=int(row["no_ask"]),
            fee_schedule=fee_schedule,
            contracts=contracts_clip,
            min_edge_pp=min_edge,
        )

        # Skip 40-60% probability region (too uncertain)
        if 0.40 <= p_model <= 0.60:
            continue

        if yes_order["valid"] and no_order["valid"]:
            if yes_order["ev_if_filled_c"] >= no_order["ev_if_filled_c"]:
                best = yes_order
                best_side = "YES"
            else:
                best = no_order
                best_side = "NO"
        elif yes_order["valid"]:
            best = yes_order
            best_side = "YES"
        elif no_order["valid"]:
            best = no_order
            best_side = "NO"
        else:
            continue

        pending_orders.append((row, best, best_side, p_model))

    # Pass 2: size each order with equal bankroll slice
    n_orders = len(pending_orders)
    if n_orders == 0:
        return pd.DataFrame()

    per_bucket_bankroll = bankroll * OVERSIZE_MULT / n_orders
    orders = []

    for row, best, best_side, p_model in pending_orders:
        limit_price_c = best["limit_price_c"]
        price_d = limit_price_c / 100.0
        fee_total = kalshi_fee_total_dollars(
            price_dollars=price_d,
            contracts=contracts_clip,
            is_maker=True,
            fee_schedule=fee_schedule,
        )
        fee_per_d = fee_total / contracts_clip

        p_win = p_model if best_side == "YES" else 1 - p_model
        cost = price_d + fee_per_d

        if cost > 0 and cost < 1:
            kelly_full = max(0, (p_win - cost) / (1 - cost))
            kelly_scaled = kelly_full * kelly_fraction
            cost_per_contract = cost
            contracts = int(np.floor(per_bucket_bankroll * kelly_scaled / cost_per_contract)) if cost_per_contract > 0 else 0
            contracts = max(1, contracts)
        else:
            kelly_full = 0.0
            kelly_scaled = 0.0
            contracts = contracts_clip

        orders.append({
            "bucket": row["bucket"],
            "ticker": row["ticker"],
            "side": best_side,
            "limit_price_c": best["limit_price_c"],
            "current_bid_c": int(row["yes_bid"]) if best_side == "YES" else int(row["no_bid"]),
            "current_ask_c": int(row["yes_ask"]) if best_side == "YES" else int(row["no_ask"]),
            "p_model": round(p_model * 100, 1) if best_side == "YES" else round((1 - p_model) * 100, 1),
            "edge_pp": best["edge_pp"],
            "ev_if_filled_c": best["ev_if_filled_c"],
            "contracts": contracts,
            "kelly_full": round(kelly_full, 4),
            "kelly_scaled": round(kelly_scaled, 4),
        })

    if not orders:
        return pd.DataFrame()

    df = pd.DataFrame(orders)
    df = df.sort_values("ev_if_filled_c", ascending=False)
    return df


# =============================================================================
# Maker Exit (Sell) Order Functions
# =============================================================================


def compute_maker_exit_price(
    p_model: float,
    side: Side,
    qty: int,
    fee_schedule: FeeSchedule = FeeSchedule(),
    min_exit_ev_c: float = 0.5,
) -> dict:
    """
    Compute minimum maker sell price where selling is +EV vs holding to settlement.

    Entry cost is sunk — only the exit (sell) fee matters.

    For a held YES position with model probability p_model:
    - Fair value = p_model * 100 cents (expected settlement payout)
    - Sell is +EV when: sell_price - maker_fee(sell_price) > fair_value
    - Find smallest S in [1,99] where net >= fair_value + min_exit_ev_c

    Args:
        p_model: Model probability for YES (P(YES wins))
        side: "YES" or "NO" - the side we hold
        qty: Number of contracts held
        fee_schedule: Fee schedule for maker fees
        min_exit_ev_c: Minimum EV advantage of selling vs holding (cents)

    Returns:
        dict with:
        - sell_price_c: Minimum sell price in cents (or None if no valid price)
        - fair_value_c: Fair value based on model probability
        - net_proceeds_c: Net proceeds per contract at sell price
        - exit_ev_c: EV advantage of selling vs holding (per contract)
        - valid: Whether a valid sell price exists
    """
    p_side = p_model if side == "YES" else (1.0 - p_model)
    fair_value_c = p_side * 100.0

    # Search from ceil(fair_value) to 99
    start = max(1, math.ceil(fair_value_c))

    for s in range(start, 100):
        # Fee on selling at price s (in the side's terms)
        # When we sell, price_dollars is the sell price
        sell_price_d = s / 100.0
        fee_total_d = kalshi_fee_total_dollars(
            price_dollars=sell_price_d,
            contracts=qty,
            is_maker=True,
            fee_schedule=fee_schedule,
        )
        fee_per_contract_c = (fee_total_d / qty) * 100.0 if qty > 0 else 0.0

        net_c = s - fee_per_contract_c
        exit_ev_c = net_c - fair_value_c

        if exit_ev_c >= min_exit_ev_c:
            return {
                "sell_price_c": s,
                "fair_value_c": round(fair_value_c, 2),
                "net_proceeds_c": round(net_c, 3),
                "exit_ev_c": round(exit_ev_c, 3),
                "valid": True,
            }

    return {
        "sell_price_c": None,
        "fair_value_c": round(fair_value_c, 2),
        "net_proceeds_c": None,
        "exit_ev_c": None,
        "valid": False,
    }


def generate_exit_orders(
    positions: list[dict],
    predictions_df: pd.DataFrame,
    fee_schedule: FeeSchedule = FeeSchedule(),
    min_exit_ev_c: float = 0.5,
) -> list[dict]:
    """
    Generate maker exit (sell) orders for held positions.

    For each held position, computes the minimum +EV sell price and
    emits an exit order if one exists.

    Args:
        positions: Raw positions from client.get_positions()
        predictions_df: DataFrame with columns: ticker, P_model
        fee_schedule: Fee schedule for maker fees
        min_exit_ev_c: Minimum EV advantage of selling vs holding (cents)

    Returns:
        List of exit order dicts with:
        - ticker, side, action="sell", sell_price_c, contracts,
          fair_value_c, exit_ev_c
    """
    exit_orders = []

    # Build lookup: ticker -> P_model (probability for YES)
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

        p_model = p_lookup[ticker]
        held_side: Side = "YES" if pos_qty > 0 else "NO"
        contracts = abs(pos_qty)

        result = compute_maker_exit_price(
            p_model=p_model,
            side=held_side,
            qty=contracts,
            fee_schedule=fee_schedule,
            min_exit_ev_c=min_exit_ev_c,
        )

        if result["valid"]:
            exit_orders.append({
                "ticker": ticker,
                "side": held_side,
                "action": "sell",
                "sell_price_c": result["sell_price_c"],
                "contracts": contracts,
                "fair_value_c": result["fair_value_c"],
                "exit_ev_c": result["exit_ev_c"],
            })

    return exit_orders


# =============================================================================
# Bankroll Reservation and Rebalancing Functions
# =============================================================================


def compute_available_deployment(
    bankroll: float,
    deployed_capital: float,
    day_of_week: int,
) -> tuple[float, float]:
    """
    Compute available vs reserved capital for day.

    Args:
        bankroll: Total bankroll
        deployed_capital: Already deployed amount
        day_of_week: 0=Monday through 6=Sunday

    Returns:
        (available_for_deployment, reserved_for_later)

    Mon-Wed: Max 60% of bankroll deployable
    Thu+: Full bankroll available (minus already deployed)
    """
    from src.config import DEFAULT_WEEKLY_TRADING_CONFIG as cfg

    if day_of_week >= cfg.full_deployment_day:
        max_deployment = bankroll
    else:
        max_deployment = bankroll * cfg.early_week_max_deployment

    available = max(0.0, max_deployment - deployed_capital)
    reserved = max(0.0, bankroll - max_deployment)

    return available, reserved


def compute_remaining_edge(
    p_model_yes: float,
    side: str,
    current_bid_c: float,
) -> tuple[float, float]:
    """
    Compute remaining edge for a position.

    Args:
        p_model_yes: Model probability P(YES)
        side: "YES" or "NO"
        current_bid_c: Current market bid for the held side in cents

    Returns:
        (remaining_edge_pp, fair_value_c)
    """
    p_side = p_model_yes if side == "YES" else (1.0 - p_model_yes)
    fair_c = p_side * 100.0
    edge_pp = fair_c - current_bid_c
    return edge_pp, fair_c


@dataclass
class ReinvestmentAnalysis:
    """Analysis of whether to exit position A and enter position B."""

    exit_ticker: str
    exit_side: str
    enter_ticker: str
    enter_side: str

    # Exit analysis
    exit_contracts: int
    exit_proceeds_c: float  # VWAP bid * contracts (gross)
    exit_fee_c: float
    exit_net_proceeds_c: float
    exit_remaining_ev_c: float

    # Entry analysis
    enter_contracts: int
    enter_cost_c: float  # VWAP ask * contracts + fees
    enter_ev_c: float

    # Decision metrics
    net_capital_change_dollars: float  # (enter_cost - exit_net_proceeds) / 100
    ev_improvement_c: float  # enter_ev - exit_remaining_ev
    fee_drag_c: float  # Total round-trip fees

    # Recommendation
    should_reinvest: bool
    reason: str


def analyze_reinvestment(
    exit_ticker: str,
    exit_side: str,
    exit_contracts: int,
    exit_p_model_yes: float,
    exit_bid_c: float,
    enter_ticker: str,
    enter_side: str,
    enter_contracts: int,
    enter_p_model_yes: float,
    enter_ask_c: float,
    fee_schedule: FeeSchedule = FeeSchedule(),
) -> ReinvestmentAnalysis:
    """
    Analyze whether to exit one position and enter another.

    Decision criteria:
    - new_EV > old_remaining_EV + round_trip_fees + buffer
    """
    from src.config import DEFAULT_WEEKLY_TRADING_CONFIG as cfg

    # Exit side proceeds
    exit_gross_c = exit_bid_c * exit_contracts
    exit_fee = kalshi_fee_total_dollars(
        price_dollars=exit_bid_c / 100.0,
        contracts=exit_contracts,
        is_maker=False,
        fee_schedule=fee_schedule,
    )
    exit_fee_c = exit_fee * 100.0
    exit_net_c = exit_gross_c - exit_fee_c

    # Remaining EV of current position
    exit_p_side = exit_p_model_yes if exit_side == "YES" else (1.0 - exit_p_model_yes)
    exit_remaining_ev_c = (exit_p_side * 100.0 - exit_bid_c) * exit_contracts

    # Entry side cost
    enter_gross_c = enter_ask_c * enter_contracts
    enter_fee = kalshi_fee_total_dollars(
        price_dollars=enter_ask_c / 100.0,
        contracts=enter_contracts,
        is_maker=False,
        fee_schedule=fee_schedule,
    )
    enter_fee_c = enter_fee * 100.0
    enter_cost_c = enter_gross_c + enter_fee_c

    # Entry EV
    enter_p_side = enter_p_model_yes if enter_side == "YES" else (1.0 - enter_p_model_yes)
    enter_ev_c = (enter_p_side * 100.0 - enter_ask_c - enter_fee_c / enter_contracts) * enter_contracts

    # Decision metrics
    fee_drag_c = exit_fee_c + enter_fee_c
    buffer_c = cfg.reinvest_buffer_cents * max(exit_contracts, enter_contracts)
    ev_improvement = enter_ev_c - exit_remaining_ev_c
    threshold_c = fee_drag_c + buffer_c

    should_reinvest = ev_improvement > threshold_c

    if should_reinvest:
        reason = f"EV improvement {ev_improvement:.1f}c > threshold {threshold_c:.1f}c (fees {fee_drag_c:.1f}c + buffer {buffer_c:.1f}c)"
    else:
        reason = f"EV improvement {ev_improvement:.1f}c <= threshold {threshold_c:.1f}c (not worth churn)"

    return ReinvestmentAnalysis(
        exit_ticker=exit_ticker,
        exit_side=exit_side,
        enter_ticker=enter_ticker,
        enter_side=enter_side,
        exit_contracts=exit_contracts,
        exit_proceeds_c=exit_gross_c,
        exit_fee_c=exit_fee_c,
        exit_net_proceeds_c=exit_net_c,
        exit_remaining_ev_c=exit_remaining_ev_c,
        enter_contracts=enter_contracts,
        enter_cost_c=enter_cost_c,
        enter_ev_c=enter_ev_c,
        net_capital_change_dollars=(enter_cost_c - exit_net_c) / 100.0,
        ev_improvement_c=ev_improvement,
        fee_drag_c=fee_drag_c,
        should_reinvest=should_reinvest,
        reason=reason,
    )
