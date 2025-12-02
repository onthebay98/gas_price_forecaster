from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/eia_weekly_gas_raw.xls")  # or .xlsx
OUT_PATH = Path("data/eia_weekly.csv")
AAA_OUT_PATH = Path("data/aaa_daily.csv")


def find_header_row(raw_path: Path, sheet_name: str = "Data 1") -> int:
    """
    Scan the 'Data 1' sheet without a header and detect which row contains
    the real header (e.g. 'Date', 'Weekly U.S....').
    Returns the row index (0-based).
    """
    df = pd.read_excel(raw_path, sheet_name=sheet_name, header=None)

    for idx, row in df.iterrows():
        texts = [str(v).strip().lower() for v in row if isinstance(v, str)]
        if not texts:
            continue

        has_date = any(t == "date" or "date" in t for t in texts)
        has_price = any(
            "price" in t
            or "dollar" in t
            or "weekly u.s. regular" in t
            or "gasoline" in t
            for t in texts
        )

        if has_date and has_price:
            return idx # type: ignore

    # Fallback: assume row 2 if detection fails (common for EIA sheets)
    return 2


def ingest_eia_xls(
    raw_path: Path = RAW_PATH,
    out_path: Path = OUT_PATH,
    sheet_name: str = "Data 1",
) -> pd.DataFrame:
    """
    Reads the downloaded EIA weekly XLS/XLSX file from the 'Data 1' sheet,
    detects the header row, identifies the date and value columns, and
    writes a clean weekly CSV.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found")

    header_row = find_header_row(raw_path, sheet_name=sheet_name)

    # Re-read with detected header row
    df = pd.read_excel(raw_path, sheet_name=sheet_name, header=header_row)

    # Normalize columns for easier detection
    cols_lower = {c.lower(): c for c in df.columns if isinstance(c, str)}

    # Find date-like column
    date_col = None
    for key, col in cols_lower.items():
        if "date" in key or "week" in key or "period" in key:
            date_col = col
            break

    # Find price/value column
    value_col = None
    for key, col in cols_lower.items():
        if "price" in key or "value" in key or "dollars per gallon" in key:
            value_col = col
            break

    if date_col is None or value_col is None:
        raise RuntimeError(
            f"Could not find date/value columns. Columns: {list(df.columns)}"
        )

    df = df[[date_col, value_col]].rename(
        columns={date_col: "date", value_col: "price"}
    )

    # Parse and clean
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = (
        df.dropna(subset=["date", "price"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote clean weekly data to {out_path} ({len(df)} rows).")

    return df


def write_unified_csv(df: pd.DataFrame, out_path: Path = AAA_OUT_PATH) -> None:
    """
    Writes df to aaa_daily.csv so your model can use it immediately.
    """
    df.to_csv(out_path, index=False)
    print(f"[OK] Unified gas price series written to {out_path}.")


def main() -> int:
    try:
        df = ingest_eia_xls()
        write_unified_csv(df)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
