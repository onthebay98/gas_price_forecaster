"""
Predictions Page

Display model predictions, run new predictions, and view +EV trades.
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

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from dashboard.services.prediction_service import (
    load_prediction_meta,
    load_prediction_data,
    load_maker_orders,
    run_prediction,
)
from dashboard.utils.formatters import fmt_pct, fmt_cents, fmt_dollars, fmt_edge_pp, fmt_price

st.set_page_config(page_title="Predictions", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.title("Model Predictions")

# Run Predictions Section
st.subheader("Run Predictions")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("Refresh Prices", type="primary", width="stretch", help="Re-fetch market prices and recalculate EV"):
        with st.spinner("Refreshing market prices..."):
            success, output = run_prediction()
            if success:
                st.success("Prices refreshed!")
                st.rerun()
            else:
                st.error(f"Refresh failed: {output}")

with col2:
    if st.button("Fresh Refit", type="secondary", width="stretch", help="Bypass model cache and refit from scratch"):
        with st.spinner("Refitting model (this takes longer)..."):
            success, output = run_prediction(force_refit=True)
            if success:
                st.success("Refit complete!")
                st.rerun()
            else:
                st.error(f"Refit failed: {output}")

with col3:
    preview_meta = load_prediction_meta()
    if preview_meta:
        try:
            from dateutil import parser as dateutil_parser
            lines = []
            gen_at = preview_meta.get("generated_at_utc")
            if gen_at:
                t = dateutil_parser.parse(gen_at).astimezone()
                lines.append(f"Prices: {t.strftime('%Y-%m-%d %H:%M')}")
            if lines:
                st.caption(" | ".join(lines))
        except Exception:
            pass

st.divider()

# Load current prediction data
meta = load_prediction_meta()
data = load_prediction_data()

if meta is None or data is None:
    st.warning("No prediction data found. Click 'Run Predictions' to generate forecasts.")
    st.stop()

# Settlement Info
st.subheader("Settlement Info")

col1, col2, col3, col4 = st.columns(4)

with col1:
    settlement = meta.get("settlement_date", meta.get("week_start", "N/A"))
    st.metric("Settlement Date", settlement)

with col2:
    last_price = meta.get("last_price")
    st.metric("Last Price", fmt_price(last_price) if last_price else "N/A")

with col3:
    n_observed = meta.get("n_observed", 0)
    st.metric("Observed Days", n_observed)

with col4:
    last_obs = meta.get("last_obs_date", "N/A")
    st.metric("Last Obs Date", last_obs)

st.divider()

# Model Output
st.subheader("Model Output")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    forecast_mean = meta.get("forecast_mean")
    st.metric("Mean", fmt_price(forecast_mean) if forecast_mean else "N/A")

with col2:
    forecast_std = meta.get("forecast_std")
    st.metric("Std Dev", f"${forecast_std:.4f}" if forecast_std else "N/A")

with col3:
    forecast_median = meta.get("forecast_median")
    st.metric("Median", fmt_price(forecast_median) if forecast_median else "N/A")

with col4:
    forecast_p5 = meta.get("forecast_p5")
    st.metric("P5", fmt_price(forecast_p5) if forecast_p5 else "N/A")

with col5:
    forecast_p95 = meta.get("forecast_p95")
    st.metric("P95", fmt_price(forecast_p95) if forecast_p95 else "N/A")

# AR(1) model parameters
col1, col2, col3, col4 = st.columns(4)

with col1:
    phi = meta.get("ar1_phi")
    st.metric("AR(1) phi", f"{phi:.4f}" if phi else "N/A")

with col2:
    sigma = meta.get("ar1_sigma")
    st.metric("AR(1) sigma", f"{sigma:.4f}" if sigma else "N/A")

with col3:
    r2 = meta.get("ar1_r_squared")
    st.metric("R-squared", f"{r2:.4f}" if r2 else "N/A")

with col4:
    n_obs = meta.get("ar1_n_obs")
    st.metric("Training Obs", n_obs if n_obs else "N/A")

st.divider()

# +EV Trades Section
st.subheader("+EV Trades")

if data is not None and not data.empty:
    # Check if trade metric columns exist
    has_ev_cols = "ev_yes_c" in data.columns or "ev_no_c" in data.columns

    if has_ev_cols:
        ev_yes = pd.to_numeric(data.get("ev_yes_c", pd.Series(dtype=float)), errors="coerce").fillna(0)
        ev_no = pd.to_numeric(data.get("ev_no_c", pd.Series(dtype=float)), errors="coerce").fillna(0)
        best_ev = np.maximum(ev_yes, ev_no)

        # Filter to +EV trades
        plus_ev_mask = best_ev > 0
        plus_ev_data = data[plus_ev_mask].copy()

        if not plus_ev_data.empty:
            display_rows = []
            for _, row in plus_ev_data.iterrows():
                ev_y = float(row.get("ev_yes_c", 0) or 0)
                ev_n = float(row.get("ev_no_c", 0) or 0)

                if ev_y > ev_n:
                    side = "YES"
                    ev_c = ev_y
                    ask_c = row.get("yes_ask", 0)
                    edge = float(row.get("edge_yes", 0) or 0) * 100
                    p_model = float(row.get("P_model", 0) or 0)
                else:
                    side = "NO"
                    ev_c = ev_n
                    ask_c = row.get("no_ask", 0)
                    edge = float(row.get("edge_no", 0) or 0) * 100
                    p_model = 1 - float(row.get("P_model", 0) or 0)

                display_rows.append({
                    "bucket": row.get("bucket", ""),
                    "side": side,
                    "ask_c": ask_c,
                    "p_model(side)": p_model,
                    "edge_pp": edge,
                    "ev_c": ev_c,
                })

            display_df = pd.DataFrame(display_rows)

            # Sort by bucket (extract numeric value)
            def extract_bucket_val(b):
                try:
                    return float(str(b).replace(">", "").strip())
                except (ValueError, AttributeError):
                    return 0.0
            display_df["_sort"] = display_df["bucket"].apply(extract_bucket_val)
            display_df = display_df.sort_values("_sort", ascending=True)
            display_df = display_df.drop(columns=["_sort"])

            # Apply formatting
            display_df["p_model(side)"] = display_df["p_model(side)"].apply(lambda x: fmt_pct(x) if pd.notna(x) else "-")
            display_df["edge_pp"] = display_df["edge_pp"].apply(lambda x: fmt_edge_pp(x) if pd.notna(x) else "-")
            display_df["ev_c"] = display_df["ev_c"].apply(lambda x: f"{x:.2f}c" if pd.notna(x) else "-")
            display_df["ask_c"] = display_df["ask_c"].apply(lambda x: f"{int(x)}c" if pd.notna(x) else "-")

            st.dataframe(display_df, width="stretch", hide_index=True)
            st.caption("Showing all buckets with positive expected value.")
        else:
            st.info("No +EV trades found in current market conditions.")
    else:
        st.info("No market data columns (ev_yes_c, ev_no_c) in predictions. Run with --maker to include market prices.")
else:
    st.info("No prediction data available.")

st.divider()

# All Threshold Buckets
st.subheader("All Threshold Buckets")

# Show core prediction columns
bucket_display_cols = [
    "bucket", "P_above", "P_below", "P_model",
]
# Add market columns if available
for col in ["yes_bid", "yes_ask", "no_bid", "no_ask", "edge_yes", "edge_no", "best_side", "ev_yes_c", "ev_no_c"]:
    if col in data.columns:
        bucket_display_cols.append(col)

available_bucket_cols = [c for c in bucket_display_cols if c in data.columns]

if available_bucket_cols:
    bucket_df = data[available_bucket_cols].copy()

    # Store raw EV values for styling before formatting
    ev_yes_raw = pd.to_numeric(bucket_df.get("ev_yes_c", pd.Series(dtype=float)), errors="coerce").fillna(0)
    ev_no_raw = pd.to_numeric(bucket_df.get("ev_no_c", pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Format P_above and P_below as percentages
    if "P_above" in bucket_df.columns:
        bucket_df["P_above"] = bucket_df["P_above"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    if "P_below" in bucket_df.columns:
        bucket_df["P_below"] = bucket_df["P_below"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")

    # Compute combined "Side" column if market data present
    if "P_model" in bucket_df.columns and "best_side" in bucket_df.columns:
        def format_side(row):
            p = row["P_model"]
            side = row["best_side"]
            if pd.isna(p) or pd.isna(side):
                return "-"
            p_side = p if side == "YES" else (1 - p)
            return f"{side} {p_side:.1%}"

        bucket_df["Side"] = bucket_df.apply(format_side, axis=1)
        bucket_df = bucket_df.drop(columns=["P_model", "best_side"], errors="ignore")
        cols = list(bucket_df.columns)
        if "Side" in cols:
            cols.remove("Side")
            cols.insert(1, "Side")
            bucket_df = bucket_df[cols]
    elif "P_model" in bucket_df.columns:
        bucket_df["P_model"] = bucket_df["P_model"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")

    if "edge_yes" in bucket_df.columns:
        bucket_df["edge_yes"] = bucket_df["edge_yes"].apply(lambda x: fmt_edge_pp(x * 100) if pd.notna(x) else "-")
    if "edge_no" in bucket_df.columns:
        bucket_df["edge_no"] = bucket_df["edge_no"].apply(lambda x: fmt_edge_pp(x * 100) if pd.notna(x) else "-")
    if "ev_yes_c" in bucket_df.columns:
        bucket_df["ev_yes_c"] = bucket_df["ev_yes_c"].apply(lambda x: f"{x:.1f}c" if pd.notna(x) else "-")
    if "ev_no_c" in bucket_df.columns:
        bucket_df["ev_no_c"] = bucket_df["ev_no_c"].apply(lambda x: f"{x:.1f}c" if pd.notna(x) else "-")

    # Apply green highlighting for positive EV
    def highlight_positive_ev(row):
        styles = [""] * len(row)
        col_names = list(row.index)

        row_idx = bucket_df.index.get_loc(row.name)

        if "ev_yes_c" in col_names and ev_yes_raw.iloc[row_idx] > 0:
            styles[col_names.index("ev_yes_c")] = "background-color: #90EE90; color: black"

        if "ev_no_c" in col_names and ev_no_raw.iloc[row_idx] > 0:
            styles[col_names.index("ev_no_c")] = "background-color: #90EE90; color: black"

        return styles

    styled_df = bucket_df.style.apply(highlight_positive_ev, axis=1)

    # Show all rows without scrolling
    row_height = 35
    header_height = 38
    table_height = len(bucket_df) * row_height + header_height
    st.dataframe(styled_df, width="stretch", hide_index=True, height=table_height)
else:
    st.warning("Prediction data missing expected columns.")

st.divider()

# Market Making Orders Section
st.subheader("Market Making Orders")

maker_data = load_maker_orders()

if maker_data is None:
    st.info("No maker orders data found. Run predictions with --maker to generate maker orders.")
elif not maker_data.get("maker_orders"):
    st.info("No valid maker orders found for current market conditions.")
else:
    observed_days = maker_data.get("observed_days", 0)

    # Get current day name for min_prob display
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    today_weekday = datetime.now().weekday()
    today_name = day_names[today_weekday]

    # Import min_prob from config
    try:
        from src.config import get_maker_min_prob_for_day
        today_min_prob = int(get_maker_min_prob_for_day(today_weekday) * 100)
    except Exception:
        today_min_prob = 75

    st.caption(
        f"**Filters:** min_prob={today_min_prob}% ({today_name}), "
        f"EV robustness (EV>=0 after +1 tick slippage)"
    )

    maker_orders = maker_data["maker_orders"]
    maker_df = pd.DataFrame(maker_orders)

    # Sort by bucket (extract numeric value)
    def extract_bucket_value(bucket_str):
        try:
            return float(str(bucket_str).replace(">", "").strip())
        except (ValueError, AttributeError):
            return 0.0

    maker_df["_bucket_sort"] = maker_df["bucket"].apply(extract_bucket_value)
    maker_df = maker_df.sort_values("_bucket_sort", ascending=True)
    maker_df = maker_df.drop(columns=["_bucket_sort"])

    display_cols = ["bucket", "side", "limit_price_c", "contracts", "current_bid_c", "current_ask_c", "p_model", "edge_pp", "ev_if_filled_c"]
    available_cols = [c for c in display_cols if c in maker_df.columns]

    if available_cols:
        display_df = maker_df[available_cols].copy()

        col_rename = {
            "limit_price_c": "Limit",
            "contracts": "Qty",
            "current_bid_c": "Bid",
            "current_ask_c": "Ask",
            "p_model": "Model P",
            "edge_pp": "Edge",
            "ev_if_filled_c": "EV/fill",
        }
        display_df = display_df.rename(columns=col_rename)

        if "Limit" in display_df.columns:
            display_df["Limit"] = display_df["Limit"].apply(lambda x: f"{int(x)}c" if pd.notna(x) else "-")
        if "Bid" in display_df.columns:
            display_df["Bid"] = display_df["Bid"].apply(lambda x: f"{int(x)}c" if pd.notna(x) else "-")
        if "Ask" in display_df.columns:
            display_df["Ask"] = display_df["Ask"].apply(lambda x: f"{int(x)}c" if pd.notna(x) else "-")
        if "Model P" in display_df.columns:
            display_df["Model P"] = display_df["Model P"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
        if "Edge" in display_df.columns:
            display_df["Edge"] = display_df["Edge"].apply(lambda x: f"{x:+.1f}pp" if pd.notna(x) else "-")
        if "EV/fill" in display_df.columns:
            display_df["EV/fill"] = display_df["EV/fill"].apply(lambda x: f"{x:+.2f}c" if pd.notna(x) else "-")

        # Show all rows without scrolling
        row_height = 35
        header_height = 38
        table_height = len(display_df) * row_height + header_height
        st.dataframe(display_df, width="stretch", hide_index=True, height=table_height)
    else:
        st.info("Maker orders data available but missing expected columns.")
