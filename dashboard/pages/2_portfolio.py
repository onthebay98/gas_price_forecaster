"""
Portfolio Page

View positions and open orders fetched directly from Kalshi API.
"""
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
import streamlit as st
import pandas as pd
from datetime import datetime

from dashboard.utils.formatters import fmt_dollars, fmt_dollars_signed

st.set_page_config(page_title="Portfolio", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.title("Portfolio")

# Check if API credentials are available
has_api_credentials = bool(
    os.environ.get("KALSHI_API_KEY_ID") and
    (os.environ.get("KALSHI_PRIVATE_KEY") or os.environ.get("KALSHI_PRIVATE_KEY_PATH"))
)

if not has_api_credentials:
    st.warning(
        "Kalshi API credentials not found. Set KALSHI_API_KEY_ID and either "
        "KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH in your .env file."
    )
    st.stop()

# Fetch balance, positions, and open orders from API
with st.spinner("Fetching data from Kalshi..."):
    from src.kalshi_client import KalshiClient
    try:
        client = KalshiClient()
        balance_data = client.get_balance()
        cash_balance = balance_data.get("balance", 0) / 100.0  # cents to dollars
        portfolio_value = balance_data.get("portfolio_value", 0) / 100.0
        open_orders = client.get_open_orders()
    except Exception as e:
        st.warning(f"Could not fetch data: {e}")
        cash_balance = 0.0
        portfolio_value = 0.0
        open_orders = []

    # Fetch positions
    positions_df = pd.DataFrame()
    try:
        positions_raw = client.get_positions()
        if positions_raw:
            rows = []
            for p in positions_raw:
                ticker = p.get("ticker", "")
                # Kalshi positions: market_exposure object or direct fields
                side = "YES"  # default
                contracts = 0
                entry_price_c = 0

                # Handle different API response formats
                if "position" in p:
                    contracts = abs(p["position"])
                    side = "YES" if p["position"] > 0 else "NO"
                elif "yes_amount" in p:
                    yes_amt = p.get("yes_amount", 0)
                    no_amt = p.get("no_amount", 0)
                    if yes_amt > 0:
                        side = "YES"
                        contracts = yes_amt
                    elif no_amt > 0:
                        side = "NO"
                        contracts = no_amt

                if "market_exposure" in p:
                    entry_price_c = p["market_exposure"]

                if contracts > 0:
                    rows.append({
                        "ticker": ticker,
                        "side": side,
                        "contracts": contracts,
                        "entry_price_c": entry_price_c,
                    })

            if rows:
                positions_df = pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"Could not fetch positions: {e}")

# Portfolio Summary
total_account_value = cash_balance + portfolio_value

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cash", fmt_dollars(cash_balance))

with col2:
    st.metric("Positions Value", fmt_dollars(portfolio_value))

with col3:
    st.metric("Account Total", fmt_dollars(total_account_value))

with col4:
    st.metric("Positions", len(positions_df) if not positions_df.empty else 0)

st.divider()

# Current Positions
st.subheader("Current Positions")

col1, col2 = st.columns([3, 1])

with col2:
    if st.button("Refresh"):
        st.rerun()

if positions_df.empty:
    st.info("No open positions in your Kalshi account.")
else:
    # Load model probabilities for context
    predict_path = PROJECT_ROOT / "data" / "latest_predict.csv"
    prob_lookup = {}
    if predict_path.exists():
        try:
            predict_df = pd.read_csv(predict_path)
            for _, row in predict_df.iterrows():
                ticker = row.get("ticker", "")
                p_model = row.get("P_model", None)
                if ticker and pd.notna(p_model):
                    prob_lookup[ticker] = p_model
        except Exception:
            pass

    display_df = positions_df.copy()

    # Add model probability column
    def get_model_prob(row):
        ticker = row.get("ticker", "")
        p_yes = prob_lookup.get(ticker)
        if p_yes is None:
            return None
        side = row.get("side", "").upper()
        return p_yes if side == "YES" else (1 - p_yes)

    display_df["model_p"] = display_df.apply(get_model_prob, axis=1)

    # Format ticker as bucket (e.g., "KXAAAGASW-26MAR23-T3.832" -> ">3.83")
    if "ticker" in display_df.columns:
        def format_bucket(t):
            try:
                parts = str(t).split("-")
                # Look for threshold part (starts with T or A)
                for part in reversed(parts):
                    if part.startswith("T") or part.startswith("A"):
                        return f">{part[1:]}"
                return str(t)
            except Exception:
                return str(t)

        display_df["bucket"] = display_df["ticker"].apply(format_bucket)

    # Sort by bucket
    def extract_bucket_val(b):
        try:
            return float(str(b).replace(">", "").strip())
        except (ValueError, AttributeError):
            return 0.0

    display_df["_sort"] = display_df.get("bucket", display_df["ticker"]).apply(extract_bucket_val)
    display_df = display_df.sort_values("_sort", ascending=True)
    display_df = display_df.drop(columns=["_sort"])

    # Format for display
    if "entry_price_c" in display_df.columns:
        display_df["entry_price_c"] = display_df["entry_price_c"].apply(
            lambda x: f"{x:.1f}c" if pd.notna(x) and x > 0 else "-"
        )
    if "model_p" in display_df.columns:
        display_df["model_p"] = display_df["model_p"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"
        )

    show_cols = ["bucket", "side", "contracts", "entry_price_c", "model_p"]
    available_show = [c for c in show_cols if c in display_df.columns]

    row_height = 35
    header_height = 38
    table_height = len(display_df) * row_height + header_height
    st.dataframe(display_df[available_show], width="stretch", hide_index=True, height=table_height)

st.divider()

# Open Orders Section
st.subheader("Open Orders")

if not open_orders:
    st.info("No open (resting) orders.")
else:
    orders_data = []
    total_order_exposure = 0.0

    for o in open_orders:
        ticker = o.get("ticker", "")
        side = o.get("side", "").upper()
        remaining = o.get("remaining_count", 0)
        yes_price = o.get("yes_price", 0)

        # Calculate side price (what you pay)
        if side == "NO":
            side_price = 100 - yes_price
        else:
            side_price = yes_price

        exposure = (side_price * remaining) / 100.0
        total_order_exposure += exposure

        # Format ticker as bucket
        def format_order_bucket(t):
            try:
                parts = str(t).split("-")
                for part in reversed(parts):
                    if part.startswith("T") or part.startswith("A"):
                        return f">{part[1:]}"
                return str(t)
            except Exception:
                return str(t)

        orders_data.append({
            "bucket": format_order_bucket(ticker),
            "Full Ticker": ticker,
            "Side": side,
            "Qty": remaining,
            "Price": f"{side_price}c",
            "Exposure": fmt_dollars(exposure),
        })

    orders_df = pd.DataFrame(orders_data)

    # Sort by bucket
    def extract_bucket_value(bucket_str):
        try:
            return float(str(bucket_str).replace(">", "").strip())
        except (ValueError, TypeError, AttributeError):
            return 0.0

    orders_df["_bucket_sort"] = orders_df["bucket"].apply(extract_bucket_value)
    orders_df = orders_df.sort_values("_bucket_sort", ascending=True)
    orders_df = orders_df.drop(columns=["_bucket_sort"])

    # Show summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Open Orders", len(open_orders))
    with col2:
        st.metric("Total Order Exposure", fmt_dollars(total_order_exposure))

    # Show table
    display_order_cols = ["bucket", "Side", "Qty", "Price", "Exposure"]
    row_height = 35
    header_height = 38
    table_height = len(orders_df) * row_height + header_height
    st.dataframe(orders_df[display_order_cols], width="stretch", hide_index=True, height=table_height)

# Footer
st.caption(f"Last synced: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
