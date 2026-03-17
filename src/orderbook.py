"""
Order book utilities for computing effective costs with liquidity constraints.

Kalshi orderbook returns YES bids and NO bids only.
In a binary market:
- YES bid at P¢ = NO ask at (100-P)¢
- NO bid at P¢ = YES ask at (100-P)¢

So to BUY YES contracts, you take the NO bids (which are YES asks).
To BUY NO contracts, you take the YES bids (which are NO asks).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal
import requests

logger = logging.getLogger(__name__)

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Module-level session for connection pooling
_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Get or create a module-level requests session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure connection pool for parallel requests
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=25,
            max_retries=1,
        )
        _session.mount("https://", adapter)
    return _session


@dataclass
class FillResult:
    """Result of computing cost to fill an order."""

    contracts_requested: int
    contracts_fillable: int
    total_cost_cents: int
    vwap_cents: float  # Volume-weighted average price
    best_price_cents: int
    worst_price_cents: int
    fully_fillable: bool
    levels_used: int  # How many price levels needed


@dataclass
class EVOptimalFillResult:
    """Result of computing EV-optimal fill walking the orderbook."""

    contracts_requested: int
    contracts_to_buy: int  # Contracts at +EV levels
    total_cost_cents: int
    vwap_cents: float
    best_price_cents: int
    worst_price_cents: int  # Worst price we'd actually fill at
    levels_used: int
    stopped_at_price: int | None  # Price where EV went negative (if any)
    stopped_reason: str  # "max_contracts", "ev_negative", "liquidity_exhausted"
    level_details: list[dict]  # Details for each level


def compute_ev_optimal_fill(
    orderbook: dict,
    side: Literal["YES", "NO"],
    p_side: float,
    max_contracts: int,
    min_ev_c: float = 0.5,
) -> EVOptimalFillResult:
    """
    Walk orderbook level-by-level, buying at each +EV level until limit.

    Unlike compute_fill_cost which fills a fixed quantity, this function
    stops when hitting a price level where EV goes negative.

    Args:
        orderbook: Dict with 'yes' and 'no' bid arrays
        side: "YES" or "NO" - which side you want to BUY
        p_side: Model probability for this side (0-1)
        max_contracts: Maximum contracts to acquire
        min_ev_c: Minimum EV in cents to consider +EV (default 0.5c)

    Returns:
        EVOptimalFillResult with contracts to buy at +EV levels only
    """
    # To BUY YES, we take NO bids (which become YES asks when inverted)
    # To BUY NO, we take YES bids (which become NO asks when inverted)
    if side == "YES":
        bids = orderbook.get("no") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])
    else:
        bids = orderbook.get("yes") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])

    if not asks:
        return EVOptimalFillResult(
            contracts_requested=max_contracts,
            contracts_to_buy=0,
            total_cost_cents=0,
            vwap_cents=0.0,
            best_price_cents=0,
            worst_price_cents=0,
            levels_used=0,
            stopped_at_price=None,
            stopped_reason="no_liquidity",
            level_details=[],
        )

    # Taker fee = 7% * p_side * (1 - p_side)
    taker_fee_c = 7 * p_side * (1 - p_side)

    total_cost = 0
    total_contracts = 0
    levels_used = 0
    best_price = asks[0][0]
    worst_price = best_price
    stopped_at_price = None
    stopped_reason = "liquidity_exhausted"
    level_details = []

    remaining = max_contracts
    for price, qty in asks:
        # Calculate EV at this price level
        ev_at_level = (p_side * 100) - price - taker_fee_c

        level_info = {
            "price_c": price,
            "qty_available": qty,
            "ev_c": round(ev_at_level, 2),
        }

        # Check if this level is +EV
        if ev_at_level < min_ev_c:
            stopped_at_price = price
            stopped_reason = "ev_negative"
            level_info["action"] = "stopped"
            level_info["qty_taken"] = 0
            level_details.append(level_info)
            break

        # Take contracts at this level
        take_qty = min(remaining, qty)
        total_cost += take_qty * price
        total_contracts += take_qty
        remaining -= take_qty
        levels_used += 1
        worst_price = price

        level_info["action"] = "bought"
        level_info["qty_taken"] = take_qty
        level_details.append(level_info)

        # Check if we hit max contracts
        if remaining <= 0:
            stopped_reason = "max_contracts"
            break

    vwap = total_cost / total_contracts if total_contracts > 0 else 0.0

    return EVOptimalFillResult(
        contracts_requested=max_contracts,
        contracts_to_buy=total_contracts,
        total_cost_cents=total_cost,
        vwap_cents=vwap,
        best_price_cents=best_price,
        worst_price_cents=worst_price,
        levels_used=levels_used,
        stopped_at_price=stopped_at_price,
        stopped_reason=stopped_reason,
        level_details=level_details,
    )


def fetch_orderbook(ticker: str, session: requests.Session | None = None) -> dict | None:
    """
    Fetch order book for a market.

    Args:
        ticker: Market ticker
        session: Optional requests session for connection pooling

    Returns:
        dict with 'yes' and 'no' keys, each containing list of [price, quantity] bids
        or None if fetch fails
    """
    url = f"{KALSHI_BASE_URL}/markets/{ticker}/orderbook"
    sess = session or get_session()
    try:
        resp = sess.get(url, timeout=10)
        if resp.status_code != 200:
            logger.debug("Orderbook fetch returned status %d for %s", resp.status_code, ticker)
            return None
        data = resp.json()
        # Support both legacy ("orderbook" with cent ints) and current
        # ("orderbook_fp" with dollar strings) API response formats
        ob = data.get("orderbook") or {}
        ob_fp = data.get("orderbook_fp") or {}
        if ob.get("yes") or ob.get("no"):
            # Legacy format: [[price_cents, quantity], ...]
            return {
                "yes": ob.get("yes", []),
                "no": ob.get("no", []),
            }
        # Current format: yes_dollars/no_dollars with [["0.55", "500.00"], ...]
        # Convert to cent integers: [55, 500]
        def _parse_dollar_levels(levels: list) -> list:
            parsed = []
            for price_str, qty_str in levels:
                price_c = round(float(price_str) * 100)
                qty = int(float(qty_str))
                parsed.append([price_c, qty])
            return parsed

        return {
            "yes": _parse_dollar_levels(ob_fp.get("yes_dollars", [])),
            "no": _parse_dollar_levels(ob_fp.get("no_dollars", [])),
        }
    except requests.exceptions.Timeout:
        logger.debug("Timeout fetching orderbook for %s", ticker)
        return None
    except requests.exceptions.RequestException as e:
        logger.debug("Network error fetching orderbook for %s: %s", ticker, e)
        return None
    except (ValueError, KeyError) as e:
        logger.debug("Parse error for orderbook %s: %s", ticker, e)
        return None


def fetch_orderbooks_parallel(
    tickers: list[str],
    max_workers: int = 10,
) -> dict[str, dict | None]:
    """
    Fetch orderbooks for multiple tickers in parallel using threads.

    Args:
        tickers: List of market tickers
        max_workers: Max concurrent requests (default 10)

    Returns:
        Dict mapping ticker -> orderbook (or None if fetch failed)
    """
    session = get_session()
    results: dict[str, dict | None] = {}

    def _fetch(ticker: str) -> tuple[str, dict | None]:
        return ticker, fetch_orderbook(ticker, session=session)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, orderbook = future.result()
            results[ticker] = orderbook

    return results


def compute_fill_cost(
    orderbook: dict,
    side: Literal["YES", "NO"],
    contracts: int,
) -> FillResult:
    """
    Compute effective cost to acquire N contracts by walking the order book.

    Args:
        orderbook: Dict with 'yes' and 'no' bid arrays
        side: "YES" or "NO" - which side you want to BUY
        contracts: Number of contracts you want to acquire

    Returns:
        FillResult with effective cost and VWAP
    """
    # To BUY YES, we take NO bids (which become YES asks when inverted)
    # To BUY NO, we take YES bids (which become NO asks when inverted)

    if side == "YES":
        # NO bids are YES asks: bid at P means ask at (100-P)
        bids = orderbook.get("no") or []
        # Convert to asks: [price, qty] -> [100-price, qty], sorted ascending
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])
    else:
        # YES bids are NO asks: bid at P means ask at (100-P)
        bids = orderbook.get("yes") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])

    if not asks:
        return FillResult(
            contracts_requested=contracts,
            contracts_fillable=0,
            total_cost_cents=0,
            vwap_cents=0.0,
            best_price_cents=0,
            worst_price_cents=0,
            fully_fillable=False,
            levels_used=0,
        )

    total_cost = 0
    total_filled = 0
    levels_used = 0
    best_price = asks[0][0]
    worst_price = best_price

    remaining = contracts
    for price, qty in asks:
        if remaining <= 0:
            break

        fill_qty = min(remaining, qty)
        total_cost += fill_qty * price
        total_filled += fill_qty
        remaining -= fill_qty
        levels_used += 1
        worst_price = price

    vwap = total_cost / total_filled if total_filled > 0 else 0.0

    return FillResult(
        contracts_requested=contracts,
        contracts_fillable=total_filled,
        total_cost_cents=total_cost,
        vwap_cents=vwap,
        best_price_cents=best_price,
        worst_price_cents=worst_price,
        fully_fillable=(total_filled >= contracts),
        levels_used=levels_used,
    )


@dataclass
class SellResult:
    """Result of computing proceeds from selling contracts."""

    contracts_requested: int
    contracts_fillable: int
    total_proceeds_cents: int
    vwap_cents: float  # Volume-weighted average sale price
    best_price_cents: int  # Highest bid (best for seller)
    worst_price_cents: int  # Lowest bid you'd hit
    fully_fillable: bool
    levels_used: int


def compute_sell_proceeds(
    orderbook: dict,
    side: Literal["YES", "NO"],
    contracts: int,
) -> SellResult:
    """
    Compute effective proceeds from selling N contracts by walking the bid book.

    For exiting a position (selling what you own):
    - To SELL YES, you hit YES bids (people wanting to buy YES from you)
    - To SELL NO, you hit NO bids (people wanting to buy NO from you)

    Args:
        orderbook: Dict with 'yes' and 'no' bid arrays
        side: "YES" or "NO" - which side you want to SELL
        contracts: Number of contracts you want to sell

    Returns:
        SellResult with effective proceeds and VWAP
    """
    # Get bids for the side we're selling
    if side == "YES":
        bids = orderbook.get("yes") or []
    else:
        bids = orderbook.get("no") or []

    # Sort descending by price (best bids first for seller)
    bids = sorted(bids, key=lambda x: -x[0])

    if not bids:
        return SellResult(
            contracts_requested=contracts,
            contracts_fillable=0,
            total_proceeds_cents=0,
            vwap_cents=0.0,
            best_price_cents=0,
            worst_price_cents=0,
            fully_fillable=False,
            levels_used=0,
        )

    total_proceeds = 0
    total_sold = 0
    levels_used = 0
    best_price = bids[0][0]
    worst_price = best_price

    remaining = contracts
    for price, qty in bids:
        if remaining <= 0:
            break

        sell_qty = min(remaining, qty)
        total_proceeds += sell_qty * price
        total_sold += sell_qty
        remaining -= sell_qty
        levels_used += 1
        worst_price = price

    vwap = total_proceeds / total_sold if total_sold > 0 else 0.0

    return SellResult(
        contracts_requested=contracts,
        contracts_fillable=total_sold,
        total_proceeds_cents=total_proceeds,
        vwap_cents=vwap,
        best_price_cents=best_price,
        worst_price_cents=worst_price,
        fully_fillable=(total_sold >= contracts),
        levels_used=levels_used,
    )


def compute_available_liquidity(
    orderbook: dict,
    side: Literal["YES", "NO"],
    max_price_cents: int,
) -> tuple[int, float]:
    """
    Compute total contracts available up to a max price.

    Args:
        orderbook: Dict with 'yes' and 'no' bid arrays
        side: "YES" or "NO" - which side you want to BUY
        max_price_cents: Maximum price you're willing to pay

    Returns:
        (total_contracts, vwap_cents) available at or below max_price
    """
    if side == "YES":
        bids = orderbook.get("no") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])
    else:
        bids = orderbook.get("yes") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])

    total_qty = 0
    total_cost = 0

    for price, qty in asks:
        if price > max_price_cents:
            break
        total_qty += qty
        total_cost += price * qty

    vwap = total_cost / total_qty if total_qty > 0 else 0.0
    return total_qty, vwap


def format_orderbook_depth(orderbook: dict, side: Literal["YES", "NO"], levels: int = 5) -> str:
    """Format top N levels of order book for display."""
    if side == "YES":
        bids = orderbook.get("no") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])
    else:
        bids = orderbook.get("yes") or []
        asks = sorted([(100 - p, q) for p, q in bids], key=lambda x: x[0])

    if not asks:
        return "  (no liquidity)"

    lines = []
    cumulative = 0
    for i, (price, qty) in enumerate(asks[:levels]):
        cumulative += qty
        lines.append(f"  {price:3d}¢: {qty:4d} contracts (cum: {cumulative:4d})")

    total_available = sum(q for _, q in asks)
    if len(asks) > levels:
        lines.append(f"  ... {len(asks) - levels} more levels ({total_available} total)")

    return "\n".join(lines)


def analyze_trade_with_depth(
    ticker: str,
    side: Literal["YES", "NO"],
    desired_contracts: int,
    p_model: float,
) -> dict:
    """
    Analyze a potential trade including order book depth.

    Args:
        ticker: Market ticker
        side: "YES" or "NO"
        desired_contracts: How many contracts you want
        p_model: Model probability for YES (0-1)

    Returns:
        dict with analysis including effective cost, edge at VWAP, etc.
    """
    ob = fetch_orderbook(ticker)
    if not ob:
        return {"error": "Could not fetch orderbook"}

    fill = compute_fill_cost(ob, side, desired_contracts)

    if fill.contracts_fillable == 0:
        return {
            "error": "No liquidity available",
            "side": side,
            "desired_contracts": desired_contracts,
        }

    # Compute edge using VWAP instead of best price
    if side == "YES":
        p_side = p_model
    else:
        p_side = 1.0 - p_model

    # Edge at best price vs edge at VWAP
    edge_at_best = (p_side * 100) - fill.best_price_cents
    edge_at_vwap = (p_side * 100) - fill.vwap_cents

    # Slippage cost
    slippage_cents = fill.vwap_cents - fill.best_price_cents
    slippage_total = slippage_cents * fill.contracts_fillable

    return {
        "ticker": ticker,
        "side": side,
        "desired_contracts": desired_contracts,
        "fillable_contracts": fill.contracts_fillable,
        "fully_fillable": fill.fully_fillable,
        "best_price_c": fill.best_price_cents,
        "worst_price_c": fill.worst_price_cents,
        "vwap_c": round(fill.vwap_cents, 2),
        "total_cost_c": fill.total_cost_cents,
        "levels_used": fill.levels_used,
        "p_model_pct": round(p_side * 100, 1),
        "edge_at_best_c": round(edge_at_best, 2),
        "edge_at_vwap_c": round(edge_at_vwap, 2),
        "slippage_per_contract_c": round(slippage_cents, 2),
        "slippage_total_c": round(slippage_total, 2),
        "depth_summary": format_orderbook_depth(ob, side),
    }


def analyze_all_markets(
    tickers: list[str],
    side: Literal["YES", "NO"],
    contracts: int,
    p_models: dict[str, float] | None = None,
) -> list[dict]:
    """
    Analyze liquidity across multiple markets.

    Args:
        tickers: List of market tickers
        side: "YES" or "NO"
        contracts: Desired number of contracts
        p_models: Optional dict of ticker -> model probability (0-1)

    Returns:
        List of analysis dicts, one per ticker
    """
    results = []

    for ticker in tickers:
        ob = fetch_orderbook(ticker)
        if not ob:
            results.append({"ticker": ticker, "error": "Could not fetch orderbook"})
            continue

        fill = compute_fill_cost(ob, side, contracts)

        result = {
            "ticker": ticker,
            "side": side,
            "desired": contracts,
            "fillable": fill.contracts_fillable,
            "best_c": fill.best_price_cents,
            "vwap_c": round(fill.vwap_cents, 2),
            "slippage_c": round(fill.vwap_cents - fill.best_price_cents, 2),
            "levels": fill.levels_used,
        }

        # Add edge calculation if we have model probability
        if p_models and ticker in p_models:
            p = p_models[ticker]
            p_side = p if side == "YES" else (1 - p)
            result["p_model"] = round(p_side * 100, 1)
            result["edge_at_best_c"] = round(p_side * 100 - fill.best_price_cents, 2)
            result["edge_at_vwap_c"] = round(p_side * 100 - fill.vwap_cents, 2)

        results.append(result)

    return results


def print_liquidity_report(results: list[dict], title: str = "Liquidity Report"):
    """Pretty-print liquidity analysis results."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    # Header
    print(f"{'Ticker':<28} {'Fill':>8} {'Best':>6} {'VWAP':>7} {'Slip':>6} {'Lvls':>5}", end="")
    if results and "edge_at_best_c" in results[0]:
        print(f" {'P(%)':>6} {'Edge@Best':>10} {'Edge@VWAP':>10}")
    else:
        print()

    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['ticker']:<28} {r['error']}")
            continue

        fillable = f"{r['fillable']}/{r['desired']}"
        print(f"{r['ticker']:<28} {fillable:>8} {r['best_c']:>5}¢ {r['vwap_c']:>6.1f}¢ {r['slippage_c']:>5.1f}¢ {r['levels']:>5}", end="")

        if "edge_at_best_c" in r:
            print(f" {r['p_model']:>5.1f}% {r['edge_at_best_c']:>9.1f}¢ {r['edge_at_vwap_c']:>9.1f}¢")
        else:
            print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.orderbook <ticker> [side] [contracts]")
        print("  python -m src.orderbook --event <event_ticker> [side] [contracts]")
        print()
        print("Examples:")
        print("  python -m src.orderbook KXTSAW-26JAN11-A2.2 YES 50")
        print("  python -m src.orderbook --event KXTSAW-26JAN11 YES 100")
        sys.exit(1)

    # Check for --event mode
    if sys.argv[1] == "--event":
        if len(sys.argv) < 3:
            print("Error: --event requires event ticker")
            sys.exit(1)

        event_ticker = sys.argv[2]
        side = sys.argv[3] if len(sys.argv) > 3 else "YES"
        contracts = int(sys.argv[4]) if len(sys.argv) > 4 else 50

        # Fetch all markets for this event
        url = f"{KALSHI_BASE_URL}/events/{event_ticker}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"Failed to fetch event: {resp.status_code}")
                sys.exit(1)
            markets = resp.json().get("markets", [])
        except Exception as e:
            print(f"Error fetching event: {e}")
            sys.exit(1)

        if not markets:
            print(f"No markets found for event {event_ticker}")
            sys.exit(1)

        tickers = [m["ticker"] for m in markets if m.get("ticker")]
        print(f"\nAnalyzing {len(tickers)} markets for {event_ticker}...")
        print(f"Side: {side}, Contracts: {contracts}")

        results = analyze_all_markets(tickers, side, contracts)
        print_liquidity_report(results, f"Liquidity Report: {event_ticker} ({side} {contracts})")

    else:
        # Single ticker mode
        ticker = sys.argv[1]
        side = sys.argv[2] if len(sys.argv) > 2 else "YES"
        contracts = int(sys.argv[3]) if len(sys.argv) > 3 else 50

        print(f"\nFetching orderbook for {ticker}...")
        ob = fetch_orderbook(ticker)

        if not ob:
            print("Failed to fetch orderbook")
            sys.exit(1)

        print(f"\n{side} side depth:")
        print(format_orderbook_depth(ob, side, levels=10))

        print(f"\nCost to fill {contracts} {side} contracts:")
        fill = compute_fill_cost(ob, side, contracts)
        print(f"  Fillable: {fill.contracts_fillable}/{fill.contracts_requested}")
        print(f"  Best price: {fill.best_price_cents}¢")
        print(f"  Worst price: {fill.worst_price_cents}¢")
        print(f"  VWAP: {fill.vwap_cents:.2f}¢")
        print(f"  Total cost: ${fill.total_cost_cents / 100:.2f}")
        print(f"  Levels used: {fill.levels_used}")
