import re
import sys
from datetime import datetime

import requests

AAA_URL = "https://gasprices.aaa.com/"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_aaa_national_regular() -> float:
    """
    Fetch the AAA national average price for regular gasoline by scraping
    the public HTML page.

    Returns
    -------
    float
        Current national average price for regular gas (USD per gallon).

    Raises
    ------
    RuntimeError
        If the page cannot be fetched or parsed.
    """
    try:
        resp = requests.get(AAA_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Error fetching AAA page: {e}") from e

    text = resp.text

    # Pattern 1: "Today’s AAA National Average $3.001"
    match = re.search(
        r"Today.?s AAA National Average\s*\$([0-9]+\.[0-9]{3})",
        text,
        flags=re.IGNORECASE,
    )

    # Pattern 2: "Current Avg. $3.001 $3.477 ..." (first number is Regular)
    if not match:
        match = re.search(
            r"Current Avg\.\s*\$([0-9]+\.[0-9]{3})",
            text,
            flags=re.IGNORECASE,
        )

    if not match:
        raise RuntimeError("Could not parse AAA national average from page HTML")

    value_str = match.group(1)
    try:
        return float(value_str)
    except ValueError as e:
        raise RuntimeError(f"Parsed value is not a float: {value_str!r}") from e


def main() -> int:
    try:
        price = fetch_aaa_national_regular()
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    today = datetime.utcnow().date().isoformat()
    print(f"{today} - AAA national regular gas price: ${price:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
