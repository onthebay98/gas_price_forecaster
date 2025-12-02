# src/live/metrics.py

import logging
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, start_http_server
except ImportError:
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    start_http_server = None  # type: ignore


class Metrics:
    def __init__(self) -> None:
        if not settings.enable_prometheus or Counter is None or Histogram is None:
            self.enabled = False
            if settings.enable_prometheus and Counter is None:
                logger.warning(
                    "Prometheus metrics requested but prometheus_client is not installed."
                )
            return

        self.enabled = True

        self.api_requests_total = Counter(
            "kalshi_api_requests_total",
            "Total Kalshi API requests",
            ["endpoint", "status"],
        )
        self.api_latency_seconds = Histogram(
            "kalshi_api_latency_seconds",
            "Kalshi API latency in seconds",
            ["endpoint"],
        )
        self.watcher_loop_duration_seconds = Histogram(
            "kalshi_watcher_loop_duration_seconds",
            "Watcher loop duration in seconds",
        )

    def start_server(self) -> None:
        if not self.enabled or start_http_server is None:
            return
        logger.info(
            "Starting Prometheus metrics server on port %s", settings.prometheus_port
        )
        start_http_server(settings.prometheus_port)


metrics = Metrics()
