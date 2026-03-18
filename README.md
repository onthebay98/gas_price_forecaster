# Gas Price Forecaster & Market Maker

Probabilistic forecasting system for AAA national average gas prices, built to trade on [Kalshi](https://kalshi.com) prediction markets. Fits an AR(1) model on daily price changes, runs 25,000-path Monte Carlo simulations conditioned on partial-week observations, and generates automated maker orders with Kelly criterion sizing.

## How It Works

```
AAA daily prices → AR(1) on daily diffs with 12 exogenous features
                 → condition on observed days (Mon → next Mon)
                 → 25k Monte Carlo simulation (8-day horizon)
                 → threshold probabilities for Kalshi strike grid
                 → Kelly sizing → maker order generation
```

**Market**: Kalshi's weekly gas price series (`KXAAAGASW`) settles on a single Monday's AAA national average price. The model forecasts 8 days forward and extracts the settlement-day endpoint distribution.

### Model

- **AR(1) on differenced prices** with Fourier seasonality (K=2), day-of-week dummies, and a 35-year seasonal norm
- **Partial-week conditioning**: as intra-week prices are observed, the simulation fixes known days — Brier score improves from 0.068 (day 0) to 0.009 (day 7) across a 12-week walk-forward backtest
- **Variance calibration**: global sigma multiplier (3.8x) calibrated against 1,850 historical Monday-to-Monday observations (1990-2025)

### Trading

- **6-gate order validation**: probability floors, EV recalculation with real fees, continuous EV thresholds, robustness checks (+1 tick slippage), contract clipping
- **Order management**: submit, update (outbid repricing), resize (Kelly adjustment), prune (cancel negative EV), place exit orders
- **All orders are `post_only`** to ensure maker fee rates (1.75% vs 7% taker)

## Architecture

```
src/
├── model.py              # AR(1) fitting, Monte Carlo simulation, conditioning
├── features.py           # Fourier, DOW dummies, seasonal norm
├── predict_utils.py      # Forecast orchestration (8-day sim → settlement)
├── trade.py              # EV calculation, Kelly sizing, maker orders
├── backtest.py           # Walk-forward backtesting, threshold grid
├── cli_predict.py        # Main forecast CLI with Kalshi market integration
├── cli_maker_submit.py   # Order validation, submission, management
├── kalshi_client.py      # Authenticated API client
├── orderbook.py          # Orderbook fetching, fill cost computation
├── config.py             # Config dataclasses, variance scalers
└── io.py                 # Data loading (handles weekly→daily transition)

scripts/
├── fetch_aaa.py          # AAA price scraper
├── live_maker.py         # Real-time maker loop with adverse selection protection
├── run_backtest.py       # CLI for walk-forward backtests
└── calibrate_variance.py # Calibration diagnostics

dashboard/
├── app.py                # Streamlit home (price charts, model metrics)
└── pages/
    ├── 1_predictions.py  # Model predictions, +EV trades, maker orders
    ├── 2_portfolio.py    # Live portfolio from Kalshi API
    ├── 3_history.py      # Portfolio history and P&L
    └── 4_live_maker.py   # Start/stop live maker, log viewer
```

## Automation

Runs on GitHub Actions:

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `gas_monitor.yml` | Every 15 min, 7 AM-10 PM ET | Full pipeline: fetch → predict → manage orders → Discord alerts |
| `cancel-makers-sunday.yml` | Sunday 11:55 PM ET | Cancel all orders at market close |
| `emergency-cancel.yml` | Manual dispatch | Kill switch |

Discord alerts fire only when order state changes (placed, cancelled, resized, updated).

## Backtest Results

12-week walk-forward backtest (Dec 2025 - Mar 2026), 25k simulations per week, expanding training window:

| Obs Days | MAE | MAPE | Brier Score | vs Naive |
|----------|-----|------|-------------|----------|
| 0 (unconditional) | $0.156 | 4.83% | 0.068 | baseline |
| 2 | $0.138 | 4.33% | 0.058 | -15% |
| 4 | $0.104 | 3.30% | 0.042 | -39% |
| 6 | $0.055 | 1.78% | 0.019 | -72% |
| 7 (day before settle) | $0.034 | 1.12% | 0.009 | -87% |

The model's unconditional forecasts (days 0-3) don't beat a random walk — its edge comes from conditioning on partial-week observations (day 4+), where the trend regime is revealed and variance collapses.

## Data

- **Source**: AAA national average regular gas price
- **History**: Weekly observations 1990-2025 (1,835 rows), daily observations Dec 2025-present
- **The AR(1) model trains exclusively on daily data** (~103 obs as of Mar 2026). Weekly data cannot contribute to daily phi estimation — a 7-day lag autocorrelation is fundamentally different from a 1-day lag.

## Usage

```bash
# Forecast with Kalshi market matching
python -m src.cli_predict --auto-week --auto-asof

# Generate and save maker orders
python -m src.cli_predict --auto-week --auto-asof --save --maker

# Submit orders (dry run first)
python -m src.cli_maker_submit --dry-run
python -m src.cli_maker_submit

# Run backtest
PYTHONPATH=. python scripts/run_backtest.py --all

# Launch dashboard
streamlit run dashboard/app.py

# Tests
pytest tests/ -v
```

## Tech Stack

Python, NumPy, SciPy, scikit-learn, statsmodels, Streamlit, Kalshi API, GitHub Actions, Discord webhooks
