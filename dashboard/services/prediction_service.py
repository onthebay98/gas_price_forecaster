"""Service for running predictions and loading prediction data."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from dashboard.utils.constants import (
    PROJECT_ROOT,
    LATEST_PREDICT_CSV,
    LATEST_PREDICT_META_JSON,
    LATEST_MAKER_ORDERS_JSON,
    AAA_DAILY_CSV,
)


def load_prediction_meta() -> dict[str, Any] | None:
    """Load prediction metadata from latest_predict_meta.json."""
    if not LATEST_PREDICT_META_JSON.exists():
        return None
    try:
        return json.loads(LATEST_PREDICT_META_JSON.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def load_prediction_data() -> pd.DataFrame | None:
    """Load prediction data from latest_predict.csv."""
    if not LATEST_PREDICT_CSV.exists():
        return None
    try:
        return pd.read_csv(LATEST_PREDICT_CSV)
    except Exception:
        return None


def load_maker_orders() -> dict[str, Any] | None:
    """Load maker orders from latest_maker_orders.json.

    Returns:
        Dict with maker_orders list and metadata, or None.
    """
    if not LATEST_MAKER_ORDERS_JSON.exists():
        return None
    try:
        return json.loads(LATEST_MAKER_ORDERS_JSON.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def run_prediction(force_refit: bool = False) -> tuple[bool, str]:
    """
    Run cli_predict as subprocess.

    Args:
        force_refit: If True, bypass model cache and refit from scratch.

    Returns:
        Tuple of (success: bool, output_or_error: str)
    """
    cmd = [
        sys.executable, "-m", "src.cli_predict",
        "--auto-week",
        "--auto-asof",
        "--maker",
    ]
    if force_refit:
        cmd.append("--force-refit")

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr or result.stdout or "Unknown error"

    except subprocess.TimeoutExpired:
        return False, "Prediction timed out after 5 minutes"
    except Exception as e:
        return False, str(e)


def get_prediction_summary() -> dict[str, Any]:
    """
    Get a summary of the current prediction state.

    Returns dict with:
        - has_data: bool
        - meta: dict or None
        - bucket_count: int
    """
    meta = load_prediction_meta()
    data = load_prediction_data()

    return {
        "has_data": meta is not None and data is not None,
        "meta": meta,
        "bucket_count": len(data) if data is not None else 0,
    }
