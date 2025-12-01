import csv
import os
from datetime import date
from pathlib import Path

from .fetch_aaa import fetch_aaa_national_regular


DATA_PATH = Path("data/aaa_daily.csv")


def append_daily_price():
    """Append today's AAA price to data/aaa_daily.csv (no duplicates)."""

    today = date.today().isoformat()
    price = fetch_aaa_national_regular()

    # Ensure data directory exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist, create with header
    if not DATA_PATH.exists():
        with open(DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "price"])

    # Check if today's date already exists
    with open(DATA_PATH, "r") as f:
        lines = f.read().splitlines()

    if any(line.startswith(today) for line in lines[1:]):
        print(f"{today} already logged in aaa_daily.csv")
        return

    # Append today's row
    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([today, price])

    print(f"Logged {today} - ${price:.3f} to {DATA_PATH}")


def main():
    append_daily_price()


if __name__ == "__main__":
    main()
