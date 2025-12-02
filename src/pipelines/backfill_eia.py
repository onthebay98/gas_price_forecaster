import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = Path("data/aaa_daily.csv")

# EIA gasoline daily retail price series (US, regular, all formulations)
# This is a reasonable proxy for AAA national average.
EIA_SERIES_ID = "PET.EMM_EPMRR_PTE_NUS_DPG.D"
EIA_API_URL = "https://api.eia.gov/series/"


def fetch_eia_series(api_key: str, series_id: str) -> pd.DataFrame:
    """
    Fetch daily gasoline price series from EIA API.

    Returns a DataFrame with columns: ['date', 'price']
    where 'date' is datetime64[ns], 'price' is float (USD/gal).
    """
    params = {
        "api_key": api_key,
        "series_id": series_id,
    }
    resp = requests.get(EIA_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "series" not in data or not data["series"]:
        raise RuntimeError(f"Unexpected EIA response structure: {data}")

    series_data = data["series"][0]
    # EIA returns like: [["20251201", 3.001], ["20251130", 2.995], ...]
    raw = series_data.get("data")
    if not raw:
        raise RuntimeError("No data found in EIA response")

    rows = []
    for date_str, value in raw:
        # EIA dates as YYYYMMDD
        try:
            dt = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            # some series use YYYYMM, but ours should be YYYYMMDD
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({"date": dt, "price": price})

    if not rows:
        raise RuntimeError("No valid rows parsed from EIA data")

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def backfill_eia_to_csv(
    api_key: str,
    out_path: Path = DATA_PATH,
    overwrite: bool = False,
) -> None:
    """
    Fetch EIA daily gasoline price series and write to CSV in the same
    schema as aaa_daily.csv: columns ['date', 'price'].

    If overwrite=False and file exists, merges and deduplicates.
    """
    df_new = fetch_eia_series(api_key, EIA_SERIES_ID)
    df_new["date"] = pd.to_datetime(df_new["date"])

    if out_path.exists() and not overwrite:
        df_old = pd.read_csv(out_path, parse_dates=["date"])
        # merge, preferring existing data for duplicate dates
        df = (
            pd.concat([df_old, df_new], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="first")
            .sort_values("date")
            .reset_index(drop=True)
        )
    else:
        df = df_new

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


def main() -> int:
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        print(
            "EIA_API_KEY environment variable not set. "
            "Get a key from https://api.eia.gov/ and set EIA_API_KEY.",
        )
        return 1

    try:
        backfill_eia_to_csv(api_key, overwrite=False)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
