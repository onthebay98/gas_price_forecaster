"""
Gas Price Forecaster Dashboard

Main page showing AAA gas price history and year-over-year comparison.
"""
import streamlit as st
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.constants import AAA_DAILY_CSV

st.set_page_config(page_title="Gas Price Forecaster", layout="wide", initial_sidebar_state="expanded")

st.title("Gas Price Forecaster")

# --- Sleep Toggle ---
def get_sleep_disabled() -> bool:
    """Check if sleep is currently disabled via pmset."""
    try:
        result = subprocess.run(["pmset", "-g"], stdin=subprocess.DEVNULL, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "SleepDisabled" in line:
                return line.strip().endswith("1")
    except Exception:
        pass
    return False

def toggle_sleep(disable: bool):
    """Toggle disablesleep via sudo pmset."""
    val = "1" if disable else "0"
    subprocess.run(["sudo", "-n", "pmset", "-a", "disablesleep", val], stdin=subprocess.DEVNULL, check=True)

col_title, col_sleep = st.columns([4, 1])
with col_sleep:
    sleep_disabled = get_sleep_disabled()
    label = "Sleep: OFF" if sleep_disabled else "Sleep: ON"
    icon = ":no_entry_sign:" if sleep_disabled else ":zzz:"
    if st.button(f"{icon} {label}", use_container_width=True):
        try:
            toggle_sleep(not sleep_disabled)
            st.rerun()
        except subprocess.CalledProcessError:
            st.error("sudo failed -- add passwordless sudoers entry (see below)")
            st.code(
                'echo "%admin ALL=(root) NOPASSWD: /usr/bin/pmset -a disablesleep 0, /usr/bin/pmset -a disablesleep 1" | sudo tee /etc/sudoers.d/pmset-sleep',
                language="bash",
            )

try:
    import altair as alt

    prices_df = pd.read_csv(AAA_DAILY_CSV, parse_dates=["date"])
    prices_df = prices_df.sort_values("date")

    # --- Summary Metrics ---
    st.subheader("Current Prices")

    latest_price = prices_df["price"].iloc[-1]
    latest_date = prices_df["date"].iloc[-1]

    # 1-week change
    one_week_ago = latest_date - timedelta(days=7)
    week_mask = prices_df["date"] <= one_week_ago
    if week_mask.any():
        week_ago_price = prices_df.loc[week_mask, "price"].iloc[-1]
        week_change = latest_price - week_ago_price
    else:
        week_ago_price = None
        week_change = None

    # 1-month change
    one_month_ago = latest_date - timedelta(days=30)
    month_mask = prices_df["date"] <= one_month_ago
    if month_mask.any():
        month_ago_price = prices_df.loc[month_mask, "price"].iloc[-1]
        month_change = latest_price - month_ago_price
    else:
        month_ago_price = None
        month_change = None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            f"${latest_price:.3f}",
            help=f"As of {latest_date.strftime('%Y-%m-%d')}",
        )

    with col2:
        if week_change is not None:
            st.metric(
                "1-Week Change",
                f"${latest_price:.3f}",
                delta=f"${week_change:+.3f}",
            )
        else:
            st.metric("1-Week Change", "-")

    with col3:
        if month_change is not None:
            st.metric(
                "1-Month Change",
                f"${latest_price:.3f}",
                delta=f"${month_change:+.3f}",
            )
        else:
            st.metric("1-Month Change", "-")

    st.divider()

    # --- Chart 1: Recent 90-day price history ---
    st.subheader("Recent Price History (Last 90 Days)")

    cutoff_90 = latest_date - timedelta(days=90)
    recent_df = prices_df[prices_df["date"] >= cutoff_90].copy()

    recent_chart = alt.Chart(recent_df).mark_line(
        color="#D55E00",
        strokeWidth=2,
    ).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d")),
        y=alt.Y("price:Q", title="Price ($/gal)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("price:Q", title="Price", format="$.3f"),
        ],
    ).properties(height=400)

    st.altair_chart(recent_chart, use_container_width=True)

    st.divider()

    # --- Chart 2: Year-over-Year Comparison (last 5 years) ---
    st.subheader("Year-over-Year Comparison")
    st.caption("Prices normalized to same calendar dates for overlay. Last 5 years shown.")

    # Add year and day-of-year columns
    prices_df["year"] = prices_df["date"].dt.year
    prices_df["day_of_year"] = prices_df["date"].dt.dayofyear

    current_year = datetime.now().year
    current_day_of_year = datetime.now().timetuple().tm_yday

    # Filter to last 5 years
    recent_years = sorted(prices_df["year"].unique())[-5:]
    yoy_df = prices_df[prices_df["year"].isin(recent_years)].copy()

    # For current year, only show up to today
    yoy_df = yoy_df[
        (yoy_df["year"] != current_year) |
        (yoy_df["day_of_year"] <= current_day_of_year)
    ]

    # Create a normalized date column for x-axis alignment (use 2024 as base year)
    yoy_df["plot_date"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(yoy_df["day_of_year"] - 1, unit="D")
    yoy_df["year_str"] = yoy_df["year"].astype(str)

    # Color mapping
    all_year_colors = {
        2019: "#999999",
        2020: "#F0E442",
        2021: "#CC79A7",
        2022: "#56B4E9",
        2023: "#009E73",
        2024: "#E69F00",
        2025: "#0072B2",
        2026: "#D55E00",
    }
    years_sorted = sorted(recent_years)
    colors = [all_year_colors.get(y, "#666666") for y in years_sorted]
    color_scale = alt.Scale(
        domain=[str(y) for y in years_sorted],
        range=colors,
    )

    yoy_chart = alt.Chart(yoy_df).mark_line().encode(
        x=alt.X("plot_date:T", title="Date", axis=alt.Axis(format="%b %d")),
        y=alt.Y("price:Q", title="Price ($/gal)", scale=alt.Scale(zero=False)),
        color=alt.Color("year_str:N", title="Year", scale=color_scale, sort=years_sorted),
        strokeWidth=alt.condition(
            alt.datum.year == current_year,
            alt.value(3),
            alt.value(1.5),
        ),
        opacity=alt.condition(
            alt.datum.year == current_year,
            alt.value(1.0),
            alt.value(0.6),
        ),
        tooltip=[
            alt.Tooltip("year_str:N", title="Year"),
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("price:Q", title="Price", format="$.3f"),
        ],
    ).properties(height=400)

    st.altair_chart(yoy_chart, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load gas price data: {e}")
