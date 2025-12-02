# src/live/kalshi_client.py

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Any, Dict, List

import requests

from .config import settings
from .rate_limiter import RateLimiter
from .metrics import metrics

logger = logging.getLogger(__name__)


@dataclass
class KalshiContract:
    market_id: str
    contract_type: Literal["above", "range"]
    threshold: Optional[float]  # for "above"
    range_low: Optional[float]  # for "range"
    range_high: Optional[float]
    target_date: date

    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float


class KalshiClient:
    """
    Minimal Kalshi API client.

    Notes:
    - You MUST fill in the real endpoints according to Kalshi's docs.
    - This client:
        * Uses a shared Session.
        * Applies self-imposed rate limiting.
        * Handles 429 with backoff.
        * Exposes metrics.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or settings.kalshi_api_key
        if not self.api_key:
            raise RuntimeError("KALSHI_API_KEY not set")

        self.base_url = settings.kalshi_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
        )
        self.rate_limiter = RateLimiter(
            max_requests=settings.max_requests_per_minute, period_seconds=60
        )

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Low-level HTTP wrapper with:
        - rate limiting
        - basic retries
        - metrics
        """
        url = f"{self.base_url}{path}"
        endpoint = path
        retries = 0
        max_retries = 5

        while True:
            self.rate_limiter.acquire()

            start = time.time()
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=10,
                )
                status = str(resp.status_code)

                if metrics.enabled:
                    metrics.api_requests_total.labels(
                        endpoint=endpoint, status=status
                    ).inc()
                    metrics.api_latency_seconds.labels(endpoint=endpoint).observe(
                        time.time() - start
                    )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        sleep_for = float(retry_after)
                    else:
                        sleep_for = min(60.0, 2.0**retries)
                    logger.warning(
                        "Rate limited by Kalshi on %s, sleeping for %.2fs",
                        endpoint,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    retries += 1
                    if retries > max_retries:
                        raise RuntimeError("Too many 429 responses from Kalshi")
                    continue

                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                retries += 1
                if retries > max_retries:
                    logger.error("API request failed after retries: %s", e)
                    raise
                sleep_for = min(60.0, 2.0**retries)
                logger.warning(
                    "API request error on %s (%s), retrying in %.2fs",
                    endpoint,
                    e,
                    sleep_for,
                )
                time.sleep(sleep_for)

    # ------------- HIGH-LEVEL METHODS (YOU MUST WIRE TO REAL API) -------------

    def list_markets(self, **params: Any) -> Any:
        """
        Example: fetch market list.

        TODO: Update path and params according to Kalshi documentation.
        """
        return self._request("GET", "/markets", params=params)

    def get_order_book(self, market_id: str) -> Any:
        """
        Example: fetch order book for a specific market/contract.

        TODO: Update path/params based on Kalshi docs.
        """
        return self._request("GET", f"/markets/{market_id}/orderbook")

    def fetch_gas_contracts(self) -> List[KalshiContract]:
        """
        Fetch all 'Gas prices in the US this month?' contracts and parse them.

        YOU MUST:
        - Identify the correct market symbol / ID for gas.
        - Map Kalshi's JSON to this dataclass.

        This method is intentionally left as a skeleton so you can adapt it
        directly to their official API schema.
        """
        raise NotImplementedError(
            "Implement fetch_gas_contracts using Kalshi's official API schema."
        )

        # Example of what this method would do once wired:
        # markets = self.list_markets(category="gas")
        # gas_market = next(m for m in markets["markets"] if m["title"] == "Gas prices in the US this month?")
        # contracts = []
        # for contract in gas_market["contracts"]:
        #     ob = self.get_order_book(contract["id"])
        #     contracts.append(
        #         KalshiContract(
        #             market_id=contract["id"],
        #             contract_type="above",
        #             threshold=float(contract["strike"]),  # example mapping
        #             range_low=None,
        #             range_high=None,
        #             target_date=date.fromisoformat(contract["expiry_date"]),
        #             yes_bid=float(ob["yes"]["bid"]),
        #             yes_ask=float(ob["yes"]["ask"]),
        #             no_bid=float(ob["no"]["bid"]),
        #             no_ask=float(ob["no"]["ask"]),
        #         )
        #     )
        # return contracts
