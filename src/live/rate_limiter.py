# src/live/rate_limiter.py

import time
from collections import deque
from typing import Deque


class RateLimiter:
    """
    Simple sliding-window rate limiter.

    max_requests: allowed requests per period_seconds.
    """

    def __init__(self, max_requests: int, period_seconds: int) -> None:
        self.max_requests = max_requests
        self.period = period_seconds
        self._timestamps: Deque[float] = deque()

    def acquire(self) -> None:
        now = time.time()

        # drop old timestamps
        while self._timestamps and now - self._timestamps[0] > self.period:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_requests:
            sleep_time = self.period - (now - self._timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._timestamps.append(time.time())
