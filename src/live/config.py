# src/live/config.py

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

def _get_bool(env_var: str, default: bool) -> bool:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_int(env_var: str, default: int) -> int:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _get_float(env_var: str, default: float) -> float:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

@dataclass(frozen=True)
class Settings:
    kalshi_api_key: str
    kalshi_base_url: str
    poll_seconds: int
    ev_threshold: float
    fee_per_contract: float
    max_requests_per_minute: int
    log_level: str
    environment: str
    enable_prometheus: bool
    prometheus_port: int
    alert_webhook_url: str | None

def _load_settings() -> Settings:
    return Settings(
        kalshi_api_key=os.getenv("KALSHI_API_KEY", ""),
        kalshi_base_url=os.getenv("KALSHI_BASE_URL", "https://api.kalshi.com").rstrip("/"),
        poll_seconds=_get_int("KALSHI_POLL_SECONDS", 60),
        ev_threshold=_get_float("KALSHI_EV_THRESHOLD", 0.03),
        fee_per_contract=_get_float("KALSHI_FEE_PER_CONTRACT", 0.0),
        max_requests_per_minute=_get_int("KALSHI_MAX_REQUESTS_PER_MIN", 60),
        log_level=os.getenv("KALSHI_LOG_LEVEL", "INFO"),
        environment=os.getenv("ENVIRONMENT", "local"),
        enable_prometheus=_get_bool("ENABLE_PROMETHEUS", False),
        prometheus_port=_get_int("PROMETHEUS_PORT", 8000),
        alert_webhook_url=os.getenv("KALSHI_ALERT_WEBHOOK_URL"),
    )

settings = _load_settings()
