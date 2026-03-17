"""
History Page

Track portfolio value over time with weekly snapshots.
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
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from dashboard.utils.formatters import fmt_dollars, fmt_dollars_signed, fmt_pct
from dashboard.utils.constants import PORTFOLIO_HISTORY_JSON

st.set_page_config(page_title="History", page_icon="", layout="wide", initial_sidebar_state="collapsed")

st.title("Portfolio History")


def load_portfolio_history() -> dict:
    """Load portfolio history from JSON."""
    if not PORTFOLIO_HISTORY_JSON.exists():
        return {
            "starting_bankroll": 700.0,
            "start_date": "2026-03-16",
            "weekly_snapshots": [],
            "cash_flows": [],
        }

    with open(PORTFOLIO_HISTORY_JSON) as f:
        data = json.load(f)
        # Ensure cash_flows exists for backwards compatibility
        if "cash_flows" not in data:
            data["cash_flows"] = []
        return data


def get_net_cash_flows(cash_flows: list) -> float:
    """Calculate net deposits (positive = deposited, negative = withdrawn)."""
    return sum(cf.get("amount", 0) for cf in cash_flows)


def get_cash_flows_between(cash_flows: list, start_date: str, end_date: str) -> float:
    """Get net cash flows between two dates (exclusive start, inclusive end)."""
    total = 0.0
    for cf in cash_flows:
        cf_date = cf.get("date", "")
        if cf_date and start_date < cf_date <= end_date:
            total += cf.get("amount", 0)
    return total


def save_portfolio_history(data: dict) -> None:
    """Save portfolio history to JSON."""
    import os
    with open(PORTFOLIO_HISTORY_JSON, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def get_last_completed_week() -> tuple[datetime, datetime]:
    """
    Get the last completed trading week (Monday-Sunday).

    Returns: (week_start, week_end) as datetime objects
    """
    today = datetime.now().date()

    # Find the most recent Sunday (end of last week)
    days_since_sunday = (today.weekday() + 1) % 7
    if days_since_sunday == 0:
        last_sunday = today - timedelta(days=7)
    else:
        last_sunday = today - timedelta(days=days_since_sunday)

    # Week starts on Monday
    last_monday = last_sunday - timedelta(days=6)

    return datetime.combine(last_monday, datetime.min.time()), datetime.combine(last_sunday, datetime.min.time())


def get_current_portfolio_value() -> tuple[float, float, float, float]:
    """
    Fetch current portfolio value from Kalshi API.

    Returns: (cash_balance, positions_value, positions_cost, total_value)
    """
    has_api_credentials = bool(
        os.environ.get("KALSHI_API_KEY_ID") and
        (os.environ.get("KALSHI_PRIVATE_KEY") or os.environ.get("KALSHI_PRIVATE_KEY_PATH"))
    )

    if not has_api_credentials:
        return 0.0, 0.0, 0.0, 0.0

    try:
        from src.kalshi_client import KalshiClient

        client = KalshiClient()
        balance_data = client.get_balance()
        cash_balance = balance_data.get("balance", 0) / 100.0
        portfolio_value = balance_data.get("portfolio_value", 0) / 100.0

        total_value = cash_balance + portfolio_value
        return cash_balance, portfolio_value, 0.0, total_value
    except Exception as e:
        st.warning(f"Could not fetch current value: {e}")
        return 0.0, 0.0, 0.0, 0.0


# Load history
history = load_portfolio_history()
starting_bankroll = history["starting_bankroll"]
start_date = history["start_date"]
snapshots = history.get("weekly_snapshots", [])
cash_flows = history.get("cash_flows", [])

# Fetch current portfolio value
with st.spinner("Fetching current portfolio value..."):
    cash_balance, positions_value, positions_cost, current_value = get_current_portfolio_value()

# If we couldn't get current value, use last snapshot
if current_value == 0 and snapshots:
    current_value = snapshots[-1]["ending_value"]

# Calculate Trading P&L
net_deposits = get_net_cash_flows(cash_flows)
total_capital_contributed = starting_bankroll + net_deposits
trading_pnl = current_value - total_capital_contributed
total_return_pct = (trading_pnl / total_capital_contributed * 100) if total_capital_contributed > 0 else 0

# Last recorded week value for delta display
if snapshots:
    last_week_value = snapshots[-1]["ending_value"]
else:
    last_week_value = starting_bankroll

# Portfolio Value Summary
st.subheader("Portfolio Value")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Starting Bankroll", fmt_dollars(starting_bankroll), help=f"As of {start_date}")

with col2:
    st.metric("Cash", fmt_dollars(cash_balance))

with col3:
    st.metric("Positions Value", fmt_dollars(positions_value))

with col4:
    current_vs_last_week = current_value - last_week_value
    st.metric(
        "Current Value",
        fmt_dollars(current_value),
        delta=fmt_dollars_signed(current_vs_last_week),
        help="Delta shows change vs last recorded week",
    )

# Second row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Trading P&L",
        fmt_dollars_signed(trading_pnl),
        help="Current value minus total capital contributed",
    )

with col2:
    st.metric("Return", f"{total_return_pct:+.1f}%", help="Trading P&L as % of total capital contributed")

with col3:
    st.metric(
        "Net Deposits",
        fmt_dollars_signed(net_deposits),
        help="Total deposits minus withdrawals since start",
    )

with col4:
    weeks_active = len(snapshots)
    st.metric("Weeks Recorded", weeks_active)

st.divider()

# Record Week Section
st.subheader("Record Week")

last_week_start, last_week_end = get_last_completed_week()
last_week_start_str = last_week_start.strftime("%Y-%m-%d")
last_week_end_str = last_week_end.strftime("%Y-%m-%d")

# Check if this week is already recorded
already_recorded = any(
    snap.get("week_start") == last_week_start_str
    for snap in snapshots
)

# Calculate start-of-week value for recording
if snapshots:
    last_recorded_value = snapshots[-1]["ending_value"]
    last_recorded_date = snapshots[-1]["week_end"]
else:
    last_recorded_value = starting_bankroll
    last_recorded_date = start_date

period_deposits = get_cash_flows_between(cash_flows, last_recorded_date, last_week_end_str)
record_start_value = last_recorded_value + period_deposits

if already_recorded:
    st.success(f"Week of {last_week_start.strftime('%b %d')} - {last_week_end.strftime('%b %d')} already recorded.")
else:
    week_pnl = current_value - record_start_value

    st.info(
        f"**Ready to record week of {last_week_start.strftime('%b %d')} - {last_week_end.strftime('%b %d')}**\n\n"
        f"Start of week: \\${record_start_value:,.2f} "
        f"(prev: \\${last_recorded_value:,.2f} + deposits: \\${period_deposits:,.2f})\n\n"
        f"End of week: **\\${current_value:,.2f}** (P&L: {fmt_dollars_signed(week_pnl)})"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Record This Week", type="primary", width="stretch"):
            new_snapshot = {
                "week_start": last_week_start_str,
                "week_end": last_week_end_str,
                "ending_value": round(current_value, 2),
                "pnl": round(week_pnl, 2),
                "notes": "",
            }

            history["weekly_snapshots"].append(new_snapshot)
            save_portfolio_history(history)

            st.success(f"Recorded: {fmt_dollars(current_value)} (P&L: {fmt_dollars_signed(week_pnl)})")
            st.rerun()

    with col2:
        st.caption("Records current portfolio value as ending value for this week.")

st.divider()

# Record Deposit/Withdrawal Section
st.subheader("Record Deposit or Withdrawal")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    flow_type = st.selectbox("Type", ["Deposit", "Withdrawal"])

with col2:
    flow_amount = st.number_input("Amount ($)", min_value=0.01, value=100.0, step=10.0)

with col3:
    flow_notes = st.text_input("Notes (optional)", placeholder="e.g., Added trading capital")

if st.button("Record Cash Flow"):
    amount = flow_amount if flow_type == "Deposit" else -flow_amount
    new_flow = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "amount": round(amount, 2),
        "type": flow_type.lower(),
        "notes": flow_notes,
    }
    history["cash_flows"].append(new_flow)
    save_portfolio_history(history)
    st.success(f"Recorded {flow_type.lower()}: {fmt_dollars(abs(amount))}")
    st.rerun()

# Show existing cash flows
if cash_flows:
    with st.expander(f"Cash Flow History ({len(cash_flows)} entries)"):
        flow_df = pd.DataFrame(cash_flows)
        flow_df["amount_fmt"] = flow_df["amount"].apply(lambda x: fmt_dollars_signed(x))
        st.dataframe(
            flow_df[["date", "type", "amount_fmt", "notes"]].rename(
                columns={"amount_fmt": "Amount", "date": "Date", "type": "Type", "notes": "Notes"}
            ),
            width="stretch",
            hide_index=True,
        )

st.divider()

# Portfolio Value Chart
st.subheader("Portfolio Value Over Time")

chart_data = [{"date": start_date, "value": starting_bankroll, "label": "Start"}]
for snap in snapshots:
    chart_data.append({
        "date": snap["week_end"],
        "value": snap["ending_value"],
        "label": f"Week of {snap['week_start']}",
    })
# Add current value point
chart_data.append({
    "date": datetime.now().strftime("%Y-%m-%d"),
    "value": current_value,
    "label": "Current",
})

chart_df = pd.DataFrame(chart_data)
chart_df["date"] = pd.to_datetime(chart_df["date"])

if len(chart_df) > 1:
    import altair as alt

    chart = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Portfolio Value ($)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("label:N", title="Period"),
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("value:Q", title="Value", format="$.2f"),
        ],
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Not enough data points for chart yet.")

st.divider()

# Weekly Snapshots Table
st.subheader("Weekly History")

if snapshots or current_value > 0:
    table_data = []
    running_value = starting_bankroll
    prev_end_date = start_date

    for snap in snapshots:
        week_end = snap.get("week_end", snap["week_start"])

        period_cash_flows = get_cash_flows_between(cash_flows, prev_end_date, week_end)
        adjusted_starting = running_value + period_cash_flows
        week_trading_pnl = snap["ending_value"] - adjusted_starting
        pnl_pct = (week_trading_pnl / adjusted_starting * 100) if adjusted_starting > 0 else 0

        table_data.append({
            "Week": f"{snap['week_start']} to {week_end}",
            "Starting": fmt_dollars(adjusted_starting),
            "Ending": fmt_dollars(snap["ending_value"]),
            "P&L": fmt_dollars_signed(week_trading_pnl),
            "Return": f"{pnl_pct:+.1f}%",
            "Notes": snap.get("notes", ""),
        })
        running_value = snap["ending_value"]
        prev_end_date = week_end

    # Add preview row for current week if not yet recorded
    if not already_recorded and current_value > 0:
        preview_pnl = current_value - record_start_value
        preview_pnl_pct = (preview_pnl / record_start_value * 100) if record_start_value > 0 else 0

        table_data.append({
            "Week": f"{last_week_start_str} to {last_week_end_str} (pending)",
            "Starting": fmt_dollars(record_start_value),
            "Ending": fmt_dollars(current_value),
            "P&L": fmt_dollars_signed(preview_pnl),
            "Return": f"{preview_pnl_pct:+.1f}%",
            "Notes": "Not yet recorded",
        })

    st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)
else:
    st.info("No weekly snapshots recorded yet.")

# Manual entry for edge cases
with st.expander("Manual Entry (for corrections or missed weeks)"):
    st.caption("Use this if you need to record a different week or correct a value.")

    with st.form("manual_snapshot_form"):
        col1, col2 = st.columns(2)

        with col1:
            manual_week_start = st.date_input("Week Start (Monday)", value=last_week_start)
            manual_week_end = st.date_input("Week End (Sunday)", value=last_week_end)

        with col2:
            manual_value = st.number_input(
                "Ending Portfolio Value ($)",
                min_value=0.0,
                value=current_value,
                step=0.01,
            )
            manual_notes = st.text_input("Notes (optional)")

        submitted = st.form_submit_button("Add Manual Snapshot")

        if submitted:
            if snapshots:
                prev_value = snapshots[-1]["ending_value"]
            else:
                prev_value = starting_bankroll

            manual_pnl = manual_value - prev_value

            new_snapshot = {
                "week_start": manual_week_start.strftime("%Y-%m-%d"),
                "week_end": manual_week_end.strftime("%Y-%m-%d"),
                "ending_value": round(manual_value, 2),
                "pnl": round(manual_pnl, 2),
                "notes": manual_notes,
            }

            history["weekly_snapshots"].append(new_snapshot)
            save_portfolio_history(history)

            st.success(f"Added: {fmt_dollars(manual_value)} (P&L: {fmt_dollars_signed(manual_pnl)})")
            st.rerun()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
