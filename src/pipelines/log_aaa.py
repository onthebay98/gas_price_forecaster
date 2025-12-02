import csv
from datetime import date
from pathlib import Path

import pandas as pd

from .fetch_aaa import fetch_aaa_national_regular

DATA_PATH = Path("data/aaa_daily.csv")


def append_daily_price():
    """Append today's AAA price to data/aaa_daily.csv (no duplicates)."""

    today = date.today().isoformat()
    price = fetch_aaa_national_regular()

    # Ensure data directory exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, dtype={"date": str})
        # normalize date column to ISO strings
        df["date"] = df["date"].astype(str)
        if (df["date"] == today).any():
            print(f"{today} already logged in aaa_daily.csv")
            return
    else:
        # Create with header if it doesn't exist
        with open(DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "price"])
        df = pd.DataFrame(columns=["date", "price"])

    # Append new row
    new_row = {"date": today, "price": price}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(DATA_PATH, index=False)
    print(f"Logged {today} - ${price:.3f} to {DATA_PATH}")


def main():
    append_daily_price()


if __name__ == "__main__":
    main()
