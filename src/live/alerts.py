# src/live/alerts.py

import logging
import os
import textwrap
from typing import Optional

import requests

from .config import settings

logger = logging.getLogger(__name__)


def send_console_alert(message: str) -> None:
    """
    Base alert sink: logs a warning.
    """
    logger.warning("ALERT %s", message)


def send_webhook_alert(message: str) -> None:
    """
    Optional webhook alert (Slack/Discord/custom).
    Uses KALSHI_ALERT_WEBHOOK_URL if set.
    """
    url = settings.alert_webhook_url
    if not url:
        return

    try:
        payload = {"text": message}
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to send webhook alert: %s", e)


def ping(alert_msg: str) -> None:
    """
    High-level alert function used by watcher.
    """
    msg = textwrap.shorten(alert_msg, width=500, placeholder=" ...")
    send_console_alert(msg)
    send_webhook_alert(msg)

    # Example macOS notification (optional, local only)
    if os.name == "posix" and "Darwin" in os.uname().sysname:
        try:
            os.system(
                f"""osascript -e 'display notification "{msg}" with title "Kalshi Alert"'"""
            )
        except Exception as e:
            logger.debug("macOS notification failed: %s", e)
