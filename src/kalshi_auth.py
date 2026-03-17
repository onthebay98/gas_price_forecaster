"""Kalshi API authentication using RSA-PSS signing.

Kalshi requires RSA-PSS signatures for authenticated API requests.
Generate API keys at: Kalshi Dashboard > Settings > API
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Auto-load .env file if it exists
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


@dataclass(frozen=True)
class KalshiCredentials:
    """Kalshi API credentials."""

    api_key_id: str  # UUID from Settings > API
    private_key_pem: str  # PEM-formatted RSA private key


def load_credentials_from_env() -> KalshiCredentials:
    """
    Load Kalshi API credentials from environment variables.

    Environment variables:
        KALSHI_API_KEY_ID: Your API key ID (UUID format)
        KALSHI_PRIVATE_KEY: PEM-formatted private key (inline)
        KALSHI_PRIVATE_KEY_PATH: Path to .key file (alternative to inline)

    Returns:
        KalshiCredentials with loaded values

    Raises:
        ValueError: If required credentials are missing
    """
    api_key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()

    # Private key can be inline (PEM string) or path to .key file
    private_key = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()

    if not private_key and key_path:
        key_file = Path(key_path)
        if not key_file.exists():
            raise ValueError(f"Private key file not found: {key_path}")
        private_key = key_file.read_text()

    if not api_key_id:
        raise ValueError(
            "Missing KALSHI_API_KEY_ID environment variable. "
            "Generate an API key at: Kalshi Dashboard > Settings > API"
        )

    if not private_key:
        raise ValueError(
            "Missing Kalshi private key. Set either:\n"
            "  KALSHI_PRIVATE_KEY (PEM string) or\n"
            "  KALSHI_PRIVATE_KEY_PATH (path to .key file)"
        )

    return KalshiCredentials(api_key_id=api_key_id, private_key_pem=private_key)


def sign_request(
    credentials: KalshiCredentials,
    method: str,
    path: str,
    timestamp_ms: Optional[int] = None,
) -> dict[str, str]:
    """
    Generate authentication headers for a Kalshi API request.

    Kalshi uses RSA-PSS signing with SHA256. The message format is:
        {timestamp_ms}{METHOD}{path}

    Args:
        credentials: API credentials with key ID and private key
        method: HTTP method (GET, POST, DELETE)
        path: Full request path including /trade-api/v2 prefix
        timestamp_ms: Optional timestamp in milliseconds; uses current time if None

    Returns:
        dict with authentication headers:
        - KALSHI-ACCESS-KEY: API key ID
        - KALSHI-ACCESS-TIMESTAMP: Timestamp in milliseconds
        - KALSHI-ACCESS-SIGNATURE: Base64-encoded RSA-PSS signature

    Raises:
        ValueError: If private key is invalid or cannot be loaded
    """
    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)

    # Message format: timestamp + method + path (strip query params)
    path_clean = path.split("?")[0]
    message = f"{timestamp_ms}{method.upper()}{path_clean}"

    # Load private key
    try:
        private_key = serialization.load_pem_private_key(
            credentials.private_key_pem.encode(),
            password=None,
            backend=default_backend(),
        )
    except Exception as e:
        raise ValueError(f"Failed to load private key: {e}") from e

    # Sign with RSA-PSS + SHA256
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    signature_b64 = base64.b64encode(signature).decode()

    return {
        "KALSHI-ACCESS-KEY": credentials.api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        "KALSHI-ACCESS-SIGNATURE": signature_b64,
    }
