"""Main forecast CLI for gas price predictions with Kalshi market integration.

Usage:
    python -m src.cli_predict --auto-week --auto-asof
    python -m src.cli_predict --auto-week --auto-asof --maker
    python -m src.cli_predict --auto-week --auto-asof --save
    python -m src.cli_predict --week 2026-03-17 --asof-day 3
    python -m src.cli_predict --week 2026-03-17 --asof-day 0 --thresholds 3.50 3.55 3.60
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_TRADING_CONFIG, DEFAULT_MAKER_CONFIG
from src.orderbook import fetch_orderbook, fetch_orderbooks_parallel, get_session
from src.predict_utils import (
    auto_asof_day,
    auto_week_start,
    generate_predictions_table,
    save_predictions,
    weekly_avg_distribution,
)
from src.trade import (
    FeeSchedule,
    add_trade_metrics,
    generate_maker_orders,
    kalshi_fee_total_dollars,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES_TICKER = "KXAAAGASW"


# ============================================================
# KALSHI API
# ============================================================
def kalshi_get(path: str, params: dict | None = None, session: requests.Session | None = None):
    """Make GET request to Kalshi API with optional session for connection pooling."""
    sess = session or get_session()
    r = sess.get(f"{KALSHI_BASE_URL}{path}", params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def candidate_event_tickers_for_week(week_start: date) -> list[str]:
    """Generate event ticker for a given week.

    Kalshi KXAAAGASW events settle on the Monday AFTER the trading week.
    Event ticker uses the settlement Monday date: KXAAAGASW-YYMONDD.
    """
    settlement_monday = week_start + timedelta(days=7)
    yy = settlement_monday.year % 100
    mon = settlement_monday.strftime("%b").upper()
    dd = settlement_monday.day
    ticker = f"{KALSHI_SERIES_TICKER}-{yy:02d}{mon}{dd:02d}"
    print(f"Week {week_start} -> Settlement {settlement_monday} -> Event: {ticker}")
    return [ticker]


def _normalize_market(m: dict) -> dict:
    """Convert Kalshi dollar-string fields to legacy cent-int fields."""
    out = dict(m)
    for field in ("yes_bid", "yes_ask", "no_bid", "no_ask"):
        dollar_key = f"{field}_dollars"
        if field not in out and dollar_key in out and out[dollar_key] is not None:
            try:
                out[field] = round(float(out[dollar_key]) * 100)
            except (ValueError, TypeError):
                pass
    return out


def fetch_kalshi_event_markets(event_ticker: str):
    """Fetch all markets for a Kalshi event."""
    resp = kalshi_get(f"/events/{event_ticker}")
    if not resp:
        return None
    markets = resp.get("markets")
    if markets:
        markets = [_normalize_market(m) for m in markets]
    return markets


def find_kalshi_buckets_for_week(week_start: date):
    """Find Kalshi event and markets for a given week."""
    candidates = candidate_event_tickers_for_week(week_start)
    print(f"\nTrying event tickers: {candidates}")

    for ev in candidates:
        print(f"Checking {ev}...", end=" ")
        mkts = fetch_kalshi_event_markets(ev)
        if mkts:
            print(f"Found {len(mkts)} markets")
            return ev, mkts
        print("not found")

    return None, None


def _needs_enrichment(markets: list[dict]) -> bool:
    """Check if markets need quote enrichment (missing yes_ask/no_ask)."""
    if not markets:
        return False
    m = markets[0]
    has_quotes = (m.get("yes_ask") is not None or m.get("yes_ask_dollars") is not None)
    return not has_quotes


def enrich_markets_with_quotes(markets: list[dict]) -> list[dict]:
    """Enrich markets with quote data from /markets/{ticker} endpoint in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not _needs_enrichment(markets):
        return markets

    session = get_session()
    tickers = [m.get("ticker") for m in markets if m.get("ticker")]

    def _fetch_market(ticker: str) -> tuple[str, dict | None]:
        resp = kalshi_get(f"/markets/{ticker}", session=session)
        if resp and "market" in resp:
            return ticker, resp["market"]
        return ticker, None

    market_details: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_market, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, details = future.result()
            if details:
                market_details[ticker] = details

    enriched = []
    for m in markets:
        t = m.get("ticker")
        if not t:
            continue
        merged = dict(m)
        if t in market_details:
            details = market_details[t]
            merged.update(details)
            for field in ("yes_bid", "yes_ask", "no_bid", "no_ask"):
                dollar_key = f"{field}_dollars"
                if dollar_key in details and details[dollar_key] is not None:
                    merged[field] = round(float(details[dollar_key]) * 100)
        enriched.append(merged)
    return enriched


# ============================================================
# BUCKET / PROBABILITY HELPERS
# ============================================================
def bucket_label(m: dict) -> str:
    """Create human-readable bucket label from market data."""
    st = m.get("strike_type")
    f = m.get("floor_strike")

    if st == "greater" and f is not None:
        return f">${f:.3f}"
    return m.get("subtitle") or m.get("title") or m.get("ticker") or "bucket"


def model_prob_for_bucket(m: dict, weekly_avgs: np.ndarray) -> float:
    """Compute model probability for a market bucket from simulated weekly averages."""
    st = m.get("strike_type")
    f = m.get("floor_strike")

    if st == "greater" and f is not None:
        return float(np.mean(weekly_avgs > float(f)))
    return float("nan")


def extract_strike_for_sorting(m: dict) -> float:
    """Extract strike price for sorting markets."""
    f = m.get("floor_strike")
    return float(f) if f is not None else 0.0


# ============================================================
# KELLY
# ============================================================
def kelly_fraction_binary(p_win: float, cost: float) -> float:
    """Full Kelly fraction for a $1 binary contract."""
    if not (np.isfinite(p_win) and np.isfinite(cost)):
        return 0.0
    if cost <= 0.0 or cost >= 1.0:
        return 0.0
    f = (p_win - cost) / (1.0 - cost)
    return float(np.clip(f, 0.0, 1.0))


def _side_metrics_row(r: pd.Series):
    """Return trade-relevant metrics for the row's best_side."""
    side = r["best_side"]
    p_yes = float(r["P_model"])

    if side == "YES":
        _yes_vwap = r.get("yes_vwap") if "yes_vwap" in r.index else None
        _yes_ask = r.get("yes_ask")
        ask = float(_yes_vwap) if pd.notna(_yes_vwap) else (float(_yes_ask) if pd.notna(_yes_ask) else np.nan)
        p_mkt = float(r["P_mkt_yes"])
        fee_c = float(r["fee_yes_per"]) * 100.0
        ev_c = float(r["ev_yes_c"])
        p_side = p_yes
        edge_pp = (p_yes - p_mkt) * 100.0
    else:
        _no_vwap = r.get("no_vwap") if "no_vwap" in r.index else None
        _no_ask = r.get("no_ask")
        ask = float(_no_vwap) if pd.notna(_no_vwap) else (float(_no_ask) if pd.notna(_no_ask) else np.nan)
        p_mkt = float(r["P_mkt_no"])
        fee_c = float(r["fee_no_per"]) * 100.0
        ev_c = float(r["ev_no_c"])
        p_side = 1.0 - p_yes
        edge_pp = ((1.0 - p_yes) - p_mkt) * 100.0

    cost_c = ask + fee_c
    cost = cost_c / 100.0
    roi = (ev_c / cost_c) if cost_c > 0 else np.nan
    kelly_full = kelly_fraction_binary(p_side, cost)

    return {
        "side": side,
        "ask_c": ask,
        "fee_c": fee_c,
        "cost_c": cost_c,
        "p_mkt": p_mkt,
        "p_side": p_side,
        "edge_pp": edge_pp,
        "ev_c": ev_c,
        "roi": roi,
        "kelly_full": kelly_full,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================
def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def default_thresholds() -> list[float]:
    """Actual Kalshi KXAAAGASW market strikes.

    10c increments from $2.90 to $4.50, plus 5c exceptions at $3.35 and $3.45.
    """
    from src.backtest import kalshi_thresholds
    return kalshi_thresholds()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gas price forecast CLI")
    parser.add_argument("--week", type=str, help="Week start date (Monday, YYYY-MM-DD)")
    parser.add_argument("--auto-week", action="store_true", help="Auto-detect current week")
    parser.add_argument("--asof-day", type=int, help="Number of observed days (0-7)")
    parser.add_argument("--auto-asof", action="store_true", help="Auto-detect observed days")
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        help="Price thresholds ($/gal). Default: use Kalshi market strikes."
    )
    parser.add_argument("--n-sims", type=int, default=25_000, help="Monte Carlo simulations")
    parser.add_argument("--data-path", type=str, default="data/aaa_daily.csv")
    parser.add_argument("--save", action="store_true", help="Save predictions to files")
    parser.add_argument("--maker", action="store_true", help="Generate maker order suggestions")
    parser.add_argument("--bankroll", type=float, default=DEFAULT_TRADING_CONFIG.bankroll)
    parser.add_argument("--auto-bankroll", action="store_true",
                        help="Fetch bankroll from Kalshi API")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ----------------------------
    # Bankroll
    # ----------------------------
    bankroll = args.bankroll
    cash_balance = bankroll
    if args.auto_bankroll:
        try:
            from src.kalshi_client import KalshiClient
            client = KalshiClient()
            balance_data = client.get_balance()
            cash_balance = balance_data.get("balance", 0) / 100.0
            raw_positions = client.get_positions()
            position_exposure = sum(
                abs(p.get("market_exposure", 0)) / 100.0
                for p in raw_positions if p.get("position", 0) != 0
            )
            bankroll = cash_balance + position_exposure
            print(f"[auto-bankroll] Cash: ${cash_balance:.2f} + Positions: ${position_exposure:.2f} = Bankroll: ${bankroll:.2f}")
        except Exception as e:
            print(f"[auto-bankroll] ERROR: {e}")
            print(f"[auto-bankroll] Falling back to --bankroll={bankroll}")

    # ----------------------------
    # Determine week
    # ----------------------------
    if args.auto_week:
        week_start = auto_week_start()
    elif args.week:
        week_start = datetime.strptime(args.week, "%Y-%m-%d").date()
        if week_start.weekday() != 0:
            logger.warning(f"{week_start} is not a Monday. Adjusting to Monday.")
            week_start = week_start - timedelta(days=week_start.weekday())
    else:
        print("Error: specify --week YYYY-MM-DD or --auto-week")
        sys.exit(1)

    week_end = week_start + timedelta(days=6)
    print(f"\n{'='*70}")
    print(f"Gas Price Forecast — Week of {week_start} to {week_end}")
    print(f"{'='*70}")

    # ----------------------------
    # Determine asof day
    # ----------------------------
    if args.auto_asof:
        from src.io import load_daily_prices
        try:
            df_daily = load_daily_prices(args.data_path)
            asof_day = auto_asof_day(week_start, df_daily)
        except ValueError:
            from src.io import load_aaa_csv
            df_all = load_aaa_csv(args.data_path)
            asof_day = auto_asof_day(week_start, df_all)
    elif args.asof_day is not None:
        asof_day = args.asof_day
    else:
        asof_day = 0

    print(f"Observed days: {asof_day}")

    # ----------------------------
    # Run forecast
    # ----------------------------
    config = DEFAULT_MODEL_CONFIG
    if args.n_sims != 25_000:
        config = type(config)(n_sims=args.n_sims)

    result = weekly_avg_distribution(
        week_start=week_start,
        asof_day=asof_day,
        thresholds=[],
        config=config,
        data_path=args.data_path,
    )

    weekly_avgs = result["weekly_avgs"]

    # ----------------------------
    # Display model summary
    # ----------------------------
    print(f"\nModel: AR(1) on daily diffs")
    print(f"  phi = {result['ar1'].phi:.4f}")
    print(f"  sigma = {result['ar1'].sigma:.6f}")
    print(f"  R² = {result['ar1'].r_squared:.4f}")
    print(f"  Training obs: {result['ar1'].n_obs}")
    print(f"  Last price: ${result['last_price']:.3f} ({result['last_obs_date']})")

    if result["observed_prices"] is not None:
        obs_str = ", ".join(f"${p:.3f}" for p in result["observed_prices"])
        print(f"  Observed this week: [{obs_str}]")

    settlement_date = result.get("settlement_date", week_start + timedelta(days=7))
    print(f"\nForecast Settlement Price ({settlement_date}, next Monday):")
    print(f"  Mean:   ${result['mean']:.3f}")
    print(f"  Median: ${result['median']:.3f}")
    print(f"  Std:    ${result['std']:.4f}")
    print(f"  90% CI: [${result['p5']:.3f}, ${result['p95']:.3f}]")

    # ----------------------------
    # Kalshi market discovery
    # ----------------------------
    event_ticker, markets = find_kalshi_buckets_for_week(week_start)

    if not markets:
        print("\nNo Kalshi markets found for this week.")
        # Fall back to model-only thresholds
        if args.thresholds:
            thresholds = sorted(args.thresholds)
        else:
            thresholds = default_thresholds()

        from src.model import compute_threshold_probs
        result["probs"] = compute_threshold_probs(weekly_avgs, thresholds)

        print(f"\n{'Threshold':>10}  {'P(above)':>10}  {'P(below)':>10}")
        print("-" * 35)
        for t in thresholds:
            p = result["probs"].get(t, 0.5)
            print(f"  ${t:.3f}    {p*100:7.1f}%    {(1-p)*100:7.1f}%")

        predictions_df = generate_predictions_table(result, thresholds)
        if args.save:
            save_predictions(predictions_df, result)
            print(f"\nPredictions saved.")
        print()
        return 0

    # Enrich with quotes
    markets = enrich_markets_with_quotes(markets)

    # Sort markets by strike
    markets = sorted(markets, key=extract_strike_for_sorting)

    print(f"\nKalshi event: {event_ticker}")
    print(f"Markets found: {len(markets)}")

    # ----------------------------
    # Build predictions table matched to Kalshi markets
    # ----------------------------
    rows = []
    for m in markets:
        ticker = m.get("ticker", "")
        label = bucket_label(m)
        p_model = model_prob_for_bucket(m, weekly_avgs)
        strike = m.get("floor_strike", 0)

        # Market quotes (cent ints)
        yes_bid = m.get("yes_bid")
        yes_ask = m.get("yes_ask")
        no_bid = m.get("no_bid")
        no_ask = m.get("no_ask")

        # Derive NO from YES if not available
        if no_bid is None and yes_ask is not None:
            no_bid = 100 - yes_ask
        if no_ask is None and yes_bid is not None:
            no_ask = 100 - yes_bid

        # Market-implied probabilities
        p_mkt_yes = (yes_bid / 100.0) if yes_bid is not None else np.nan
        p_mkt_no = (no_bid / 100.0) if no_bid is not None else np.nan

        rows.append({
            "bucket": label,
            "ticker": ticker,
            "threshold": strike,
            "P_model": round(p_model, 4),
            "P_above": round(p_model, 4),
            "P_below": round(1.0 - p_model, 4),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "P_mkt_yes": round(p_mkt_yes, 4) if np.isfinite(p_mkt_yes) else np.nan,
            "P_mkt_no": round(p_mkt_no, 4) if np.isfinite(p_mkt_no) else np.nan,
            "open_interest": m.get("open_interest_fp", ""),
        })

    tbl = pd.DataFrame(rows)

    # ----------------------------
    # Add trade metrics (fees, EV, best side)
    # ----------------------------
    tbl = add_trade_metrics(tbl, is_maker=True)

    # ----------------------------
    # Fetch orderbooks in parallel for VWAP
    # ----------------------------
    tickers = tbl["ticker"].tolist()
    print(f"\nFetching orderbooks for {len(tickers)} markets...")
    orderbooks = fetch_orderbooks_parallel(tickers)

    # Add VWAP from orderbooks
    from src.orderbook import compute_fill_cost
    for idx, row in tbl.iterrows():
        ticker = row["ticker"]
        ob = orderbooks.get(ticker)
        if not ob:
            tbl.loc[idx, "yes_vwap"] = np.nan
            tbl.loc[idx, "no_vwap"] = np.nan
            continue

        # Compute VWAP for 100 contracts on each side
        yes_fill = compute_fill_cost(ob, "YES", 100)
        no_fill = compute_fill_cost(ob, "NO", 100)
        tbl.loc[idx, "yes_vwap"] = yes_fill.vwap_cents if yes_fill.contracts_fillable > 0 else np.nan
        tbl.loc[idx, "no_vwap"] = no_fill.vwap_cents if no_fill.contracts_fillable > 0 else np.nan

    # Re-add trade metrics with VWAP
    tbl = add_trade_metrics(tbl, is_maker=True)

    # ----------------------------
    # Display predictions with market data
    # ----------------------------
    print(f"\n{'='*100}")
    print(f"PREDICTIONS vs MARKET — {event_ticker}")
    print(f"{'='*100}")

    header = f"{'Bucket':<12} {'P(model)':>9} {'P(mkt)':>8} {'Edge':>7} {'Side':>5} {'Bid':>5} {'Ask':>5} {'EV(c)':>7} {'Ticker'}"
    print(header)
    print("-" * 100)

    for _, row in tbl.iterrows():
        p_model = float(row["P_model"])
        best_side = row.get("best_side", "")
        if not best_side or pd.isna(best_side):
            best_side = ""

        if best_side and best_side != "":
            metrics = _side_metrics_row(row)
            side_str = metrics["side"]
            edge_str = f"{metrics['edge_pp']:+.1f}pp"
            ev_str = f"{metrics['ev_c']:.1f}"
            bid_c = int(row["yes_bid"]) if side_str == "YES" else int(row["no_bid"])
            ask_c = int(row["yes_ask"]) if side_str == "YES" else int(row["no_ask"])
        else:
            side_str = ""
            edge_str = ""
            ev_str = ""
            bid_c = ""
            ask_c = ""

        print(f"{row['bucket']:<12} {_pct(p_model):>9} {_pct(row.get('P_mkt_yes', 0)):>8} "
              f"{edge_str:>7} {side_str:>5} {bid_c!s:>5} {ask_c!s:>5} {ev_str:>7} {row['ticker']}")

    # ----------------------------
    # Maker orders
    # ----------------------------
    if args.maker:
        print(f"\n{'='*70}")
        print(f"MAKER ORDERS — Bankroll: ${bankroll:.2f}")
        print(f"{'='*70}")

        maker_df = generate_maker_orders(
            tbl,
            bankroll=bankroll,
        )

        if maker_df.empty:
            print("\nNo valid maker orders found.")
        else:
            print(f"\n{'Bucket':<12} {'Side':>5} {'Price':>6} {'EV':>6} {'Edge':>7} {'Ctrs':>6} {'Kelly':>7} {'Ticker'}")
            print("-" * 80)
            for _, mo in maker_df.iterrows():
                print(f"{mo['bucket']:<12} {mo['side']:>5} {mo['limit_price_c']:>5}c "
                      f"{mo['ev_if_filled_c']:>5.1f}c {mo['edge_pp']:>5.1f}pp "
                      f"{mo['contracts']:>6} {mo['kelly_full']:>6.3f} {mo['ticker']}")

            total_exposure = sum(
                mo["limit_price_c"] * mo["contracts"] / 100.0
                for _, mo in maker_df.iterrows()
            )
            print(f"\nTotal maker exposure: ${total_exposure:.2f} ({len(maker_df)} orders)")

            # Save maker orders
            if args.save:
                maker_orders = maker_df.to_dict(orient="records")
                orders_path = Path("data/latest_maker_orders.json")
                orders_path.parent.mkdir(parents=True, exist_ok=True)
                with open(orders_path, "w") as f:
                    json.dump(maker_orders, f, indent=2, default=str)
                print(f"Maker orders saved to {orders_path}")

    # ----------------------------
    # Save predictions
    # ----------------------------
    if args.save:
        # Build save-compatible predictions from Kalshi-matched table
        thresholds = tbl["threshold"].tolist()
        from src.model import compute_threshold_probs
        result["probs"] = compute_threshold_probs(weekly_avgs, thresholds)
        predictions_df = generate_predictions_table(result, thresholds)

        # Add ticker column
        predictions_df["ticker"] = tbl["ticker"].values

        # Save with Kalshi-specific filename
        week_str = week_start.strftime("%Y%m%d")
        csv_path = f"data/predict_{event_ticker}_{week_str}.csv"
        meta_path = f"data/predict_meta_{event_ticker}_{week_str}.json"
        save_predictions(predictions_df, result, output_csv=csv_path, output_meta=meta_path)

        # Also save as latest
        save_predictions(predictions_df, result)
        print(f"\nPredictions saved to {csv_path} and data/latest_predict.csv")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
