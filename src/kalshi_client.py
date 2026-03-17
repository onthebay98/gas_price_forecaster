"""Kalshi authenticated API client for order management.

Provides methods for:
- Checking account balance
- Listing open orders and positions
- Creating limit orders (maker)
- Cancelling orders (including kill switch)
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Optional

import requests

from src.kalshi_auth import KalshiCredentials, load_credentials_from_env, sign_request

logger = logging.getLogger(__name__)

# Production trading API (different from public market data API)
KALSHI_TRADING_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"


@dataclass
class OrderResult:
    """Result of order submission."""

    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[int] = None


@dataclass
class CancelResult:
    """Result of order cancellation."""

    success: bool
    order_id: str
    error_message: Optional[str] = None


def _normalize_order(o: dict) -> dict:
    """Normalize Kalshi order response to legacy cent-integer fields.

    The API migrated from cent-integer fields (yes_price, remaining_count, count)
    to dollar-string fields (yes_price_dollars, remaining_count_fp, initial_count_fp).
    This function adds the legacy fields so downstream code keeps working.
    """
    out = dict(o)
    # yes_price / no_price: dollar string → cent int
    if "yes_price" not in out and "yes_price_dollars" in out:
        out["yes_price"] = round(float(out["yes_price_dollars"]) * 100)
    if "no_price" not in out and "no_price_dollars" in out:
        out["no_price"] = round(float(out["no_price_dollars"]) * 100)
    # remaining_count: string → int
    if "remaining_count" not in out and "remaining_count_fp" in out:
        out["remaining_count"] = int(float(out["remaining_count_fp"]))
    # count (initial): string → int
    if "count" not in out and "initial_count_fp" in out:
        out["count"] = int(float(out["initial_count_fp"]))
    # fill_count: string → int
    if "fill_count" not in out and "fill_count_fp" in out:
        out["fill_count"] = int(float(out["fill_count_fp"]))
    return out


def _normalize_position(p: dict) -> dict:
    """Normalize Kalshi position response to legacy fields."""
    out = dict(p)
    if "position" not in out and "position_fp" in out:
        out["position"] = int(float(out["position_fp"]))
    if "market_exposure" not in out and "market_exposure_dollars" in out:
        out["market_exposure"] = round(float(out["market_exposure_dollars"]) * 100)
    return out


class KalshiClient:
    """Authenticated Kalshi API client for order management."""

    def __init__(
        self,
        credentials: Optional[KalshiCredentials] = None,
        use_demo: bool = False,
        timeout: int = 30,
    ):
        """
        Initialize the Kalshi client.

        Args:
            credentials: API credentials; loads from env if None
            use_demo: Use demo/sandbox API instead of production
            timeout: Request timeout in seconds
        """
        self.credentials = credentials or load_credentials_from_env()
        self.base_url = KALSHI_DEMO_URL if use_demo else KALSHI_TRADING_URL
        self.timeout = timeout
        self._session = requests.Session()
        self._is_demo = use_demo

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        max_retries: int = 3,
    ) -> requests.Response:
        """Make authenticated request to Kalshi API with rate limit retry."""
        full_path = f"/trade-api/v2{path}"
        url = f"{self.base_url}{path}"

        for attempt in range(max_retries):
            headers = sign_request(self.credentials, method, full_path)
            headers["Content-Type"] = "application/json"

            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                params=params,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                logger.warning(
                    f"Rate limited (429) on {method} {path}, "
                    f"retrying in {retry_after:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_after)
                continue
            return response

        # Exhausted retries — return the last 429 response
        return response

    def get_balance(self) -> dict:
        """
        Get account balance.

        Returns:
            dict with 'balance' (cents), 'payout_available' (cents), etc.

        Raises:
            requests.HTTPError: If request fails
        """
        resp = self._request("GET", "/portfolio/balance")
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> list[dict]:
        """
        Get all open positions.

        Returns:
            List of position dicts with ticker, position (+ for yes, - for no), etc.
        """
        resp = self._request("GET", "/portfolio/positions")
        resp.raise_for_status()
        return [_normalize_position(p) for p in resp.json().get("market_positions", [])]

    def get_open_orders(self, ticker: Optional[str] = None) -> list[dict]:
        """
        Get open (resting) orders.

        Args:
            ticker: Optional filter by market ticker

        Returns:
            List of order dicts
        """
        params = {"status": "resting"}
        if ticker:
            params["ticker"] = ticker
        resp = self._request("GET", "/portfolio/orders", params=params)
        resp.raise_for_status()
        return [_normalize_order(o) for o in resp.json().get("orders", [])]

    def create_order(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        action: Literal["buy", "sell"],
        count: int,
        yes_price: int,
        client_order_id: Optional[str] = None,
        post_only: bool = True,
    ) -> OrderResult:
        """
        Submit a limit order to Kalshi.

        Args:
            ticker: Market ticker (e.g., "KXTSAW-26JAN18-A2.10")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            yes_price: Price in cents (1-99) - always specified as YES price
            client_order_id: Optional unique ID for deduplication
            post_only: If True, reject order if it would execute immediately
                      (ensures maker fee; recommended for market making)

        Returns:
            OrderResult with success status and order details
        """
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        # API migrated to dollar-string fields (March 2026 fixed-point migration).
        # Legacy integer fields (yes_price, count) removed March 12, 2026.
        yes_price_dollars = f"{yes_price / 100:.2f}"
        count_fp = f"{count:.2f}"
        payload = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count_fp": count_fp,
            "type": "limit",
            "yes_price_dollars": yes_price_dollars,
            "client_order_id": client_order_id,
            "post_only": post_only,
        }

        logger.debug(f"Creating order: {payload}")

        try:
            resp = self._request("POST", "/portfolio/orders", json_data=payload)

            if resp.status_code == 201:
                data = resp.json()
                order = _normalize_order(data.get("order", {}))
                return OrderResult(
                    success=True,
                    order_id=order.get("order_id"),
                    client_order_id=order.get("client_order_id"),
                    status=order.get("status"),
                )
            else:
                error_msg = resp.text
                try:
                    error_data = resp.json()
                    error_msg = error_data.get("error", {}).get("message", resp.text)
                except Exception:
                    pass

                return OrderResult(
                    success=False,
                    error_code=resp.status_code,
                    error_message=error_msg,
                    client_order_id=client_order_id,
                )
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return OrderResult(
                success=False,
                error_message=str(e),
                client_order_id=client_order_id,
            )

    def cancel_order(self, order_id: str) -> CancelResult:
        """
        Cancel a specific order by order_id.

        Args:
            order_id: The order ID to cancel

        Returns:
            CancelResult with success status
        """
        try:
            resp = self._request("DELETE", f"/portfolio/orders/{order_id}")
            if resp.status_code in (200, 204):
                return CancelResult(success=True, order_id=order_id)
            else:
                return CancelResult(
                    success=False,
                    order_id=order_id,
                    error_message=resp.text,
                )
        except requests.RequestException as e:
            return CancelResult(
                success=False,
                order_id=order_id,
                error_message=str(e),
            )

    def cancel_all_orders(self, ticker: Optional[str] = None) -> list[CancelResult]:
        """
        Cancel all open orders (kill switch).

        Args:
            ticker: Optional - only cancel orders for this market

        Returns:
            List of CancelResult for each order
        """
        orders = self.get_open_orders(ticker=ticker)
        results = []

        for order in orders:
            order_id = order.get("order_id")
            if order_id:
                logger.info(f"Cancelling order {order_id}")
                result = self.cancel_order(order_id)
                results.append(result)

        return results

    def get_market(self, ticker: str) -> Optional[dict]:
        """
        Get market details by ticker.

        Args:
            ticker: Market ticker

        Returns:
            Market dict or None if not found
        """
        try:
            resp = self._request("GET", f"/markets/{ticker}")
            if resp.status_code == 200:
                return resp.json().get("market")
            return None
        except requests.RequestException:
            return None
