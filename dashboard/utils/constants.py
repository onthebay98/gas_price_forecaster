"""Dashboard constants."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
AAA_DAILY_CSV = DATA_DIR / "aaa_daily.csv"
LATEST_PREDICT_CSV = DATA_DIR / "latest_predict.csv"
LATEST_PREDICT_META_JSON = DATA_DIR / "latest_predict_meta.json"
LATEST_MAKER_ORDERS_JSON = DATA_DIR / "latest_maker_orders.json"
PORTFOLIO_HISTORY_JSON = DATA_DIR / "portfolio_history.json"
LIVE_MAKER_LOG = DATA_DIR / "live_maker.log"
LIVE_MAKER_PID = DATA_DIR / "live_maker.pid"

# Kalshi API
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
