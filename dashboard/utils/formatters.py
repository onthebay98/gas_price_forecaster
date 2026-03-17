"""Display formatting utilities."""
import pandas as pd
import numpy as np


def fmt_pct(x: float, decimals: int = 1) -> str:
    """Format as percentage."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"{100 * x:.{decimals}f}%"


def fmt_pct_signed(x: float, decimals: int = 1) -> str:
    """Format as signed percentage (with +/-)."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"{100 * x:+.{decimals}f}%"


def fmt_cents(x: float, decimals: int = 0) -> str:
    """Format as cents."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"{x:.{decimals}f}c"


def fmt_dollars(x: float, decimals: int = 2) -> str:
    """Format as dollars."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"${x:,.{decimals}f}"


def fmt_dollars_signed(x: float, decimals: int = 2) -> str:
    """Format as signed dollars (with +/-)."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    sign = "+" if x >= 0 else ""
    return f"{sign}${x:,.{decimals}f}"


def fmt_number(x: float, decimals: int = 0) -> str:
    """Format as number with commas."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"{x:,.{decimals}f}"


def fmt_edge_pp(x: float) -> str:
    """Format edge in percentage points."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"{x:+.1f}pp"


def fmt_price(x: float) -> str:
    """Format gas price (e.g., $3.832)."""
    if pd.isna(x) or not np.isfinite(x):
        return "-"
    return f"${x:.3f}"


def color_pnl(val: float) -> str:
    """Return CSS color for P&L value."""
    if pd.isna(val) or not np.isfinite(val):
        return ""
    if val > 0:
        return "color: green"
    elif val < 0:
        return "color: red"
    return ""
