# CLAUDE.md

Gas price forecasting system for Kalshi prediction market trading.

## Data Flow

```
aaa_daily.csv → fit AR(1) on daily diffs (daily data only)
              → condition on partial-week observations
              → 25k Monte Carlo simulations (8 days: Mon through next Mon)
              → extract settlement day (next Monday) price distribution
              → threshold probabilities → Kelly sizing → maker orders
```

## Kalshi Market Structure

**Series**: `KXAAAGASW` — "US gas prices this week"
**Settlement**: AAA national average regular gas price on `<date>` (a Monday).
**Contract rules** (from AAAGAS rules PDF):
- Underlying: "the average gas price for regular gas in `<area>` on `<date>`"
- Source Agency: AAA
- Payout: "strictly greater than `<price>`"
- Last Trading Date: day before `<date>`, 11:59 PM ET

**Event ticker pattern**: `KXAAAGASW-YYMONDD` where DD is the settlement Monday.
Example: `KXAAAGASW-26MAR23` → opens Mar 16, closes Mar 22, settles on Mar 23 AAA price.

**Market ticker pattern**: `KXAAAGASW-26MAR23-3.670` (event + strike price in $/gal).

**CRITICAL**: The market settles on a single Monday's AAA price, NOT a weekly average. The simulation must forecast 8 days (Mon through next Mon) and the settlement target is `sims[7, :]`.

## Data

**Source**: AAA national average regular gas price, scraped daily from gasprices.aaa.com.
- 1990-Nov 2025: Weekly observations (every Monday), 1,835 rows
- Dec 2025-present: Daily observations (7 days/week), ~106 rows
- Total: ~1,940 observations, current price ~$3.72

**Frequency Handling**: The data has two regimes. `io.py` provides three loaders:
- `load_aaa_csv()`: Raw data as-is (mixed weekly + daily)
- `load_daily_prices()`: Daily-frequency portion only (Dec 2025+)
- `load_interpolated_daily()`: Full history with linear interpolation to daily (for seasonal_norm ONLY)

**CRITICAL**: The AR(1) model trains exclusively on daily data (~103 obs). The weekly data CANNOT contribute to phi estimation — a 7-day lag autocorrelation is fundamentally different from a 1-day lag. For daily AR(1) with phi=0.56, the implied 7-day autocorrelation is phi^7 ≈ 0.017 (essentially zero). The interpolated daily loader is used only for seasonal_norm computation, never for AR(1) training — interpolated values would inject artificial autocorrelation.

## Model

**Specification**: AR(1) on differenced prices with exogenous features

```
diff_t = c + phi * diff_{t-1} + X_t @ beta + epsilon_t
where diff_t = price_t - price_{t-1} (consecutive daily observations only)
```

**Current Parameters** (as of 2026-03-16):
- phi = 0.56 (strong daily momentum)
- sigma = 0.0136 (daily innovation std in $/gal)
- R² = 0.67
- Training obs: 103 (daily data, 2025-12-03 to 2026-03-16)

**Training Sample Limitation**: 103 observations is thin. Phi and sigma estimates will stabilize as more daily data accumulates. The daily scraper adds one observation per day. By Jun 2026: ~280 obs. By Dec 2026: ~460 obs.

**Phi Validation**: The build plan cited 0.39 as the lag-1 autocorrelation of raw diffs. The OLS phi of 0.56 is higher because (a) it's conditioned on exogenous features that absorb some variance, and (b) the Dec 2025-Mar 2026 period saw an unusually strong upward trend (prices rose from $2.98 to $3.72, a regime with high momentum). The Phase 0 rolling-window check showed phi stable at 0.5-0.6 across most 5-year windows on weekly data, with one outlier (2001-2006: 0.32). The daily-only phi (0.75 from Phase 0 autocorrelation check) vs OLS phi (0.56) gap is explained by the exogenous features absorbing some serial correlation.

**Exogenous Features** (12 total):
- **Fourier K=2** (4 features): sin_y1, cos_y1, sin_y2, cos_y2 (annual seasonality)
- **DOW dummies** (7 features): dow_0 through dow_6 (all 7 days)
- **seasonal_norm** (1 feature): smoothed DOY median of log-prices (computed from full 35-year history)

**Settlement Target**: Next Monday's AAA price (single-day endpoint, not weekly average). The simulation runs 8 days (Mon through next Mon). Conditioning on partial-week observations (Mon-Sat) fixes those days in the simulation, but the edge structure is different from weekly averages — early-week observations primarily reveal the trend regime rather than mechanically resolving portions of the target.

**Conditioning value** (detrended, 14 daily weeks Dec 2025-Mar 2026):
- Day 1 (Mon only): 0% within-week info (phi^6 ≈ 0.03)
- Day 2 (through Tue): 95% of within-week variance resolved (trend revealed)
- Day 4 (through Thu): 98% resolved
- Note: Strong results partly reflect persistent uptrend period; may weaken in sideways markets.

**Variance Scaling**: `GLOBAL_SIGMA_MULTIPLIER = 3.8` calibrated from Part 0 historical sanity check (1,850 weekly Mon-to-Mon observations, 1990-2026). Model-implied 8-day std was $0.016, empirical was $0.063 (ratio 0.26). All other scalers (DOW, seasonal, conditioning, mean bias) remain **placeholder 1.0/0.0 values** — N=12 backtest weeks is insufficient for reliable calibration.

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/model.py` | AR(1) fitting (daily data only), Monte Carlo simulation, conditioning |
| `src/features.py` | Calendar features (Fourier, DOW, seasonal_norm) |
| `src/config.py` | All config dataclasses, variance scalers, GLOBAL_SIGMA_MULTIPLIER |
| `src/backtest.py` | Walk-forward backtesting, Kalshi threshold grid, metrics |
| `src/predict_utils.py` | Orchestrates forecast pipeline (8-day sim, next-Monday settlement) |
| `src/io.py` | Data loading (handles weekly→daily transition) |
| `src/trade.py` | EV calculation, Kelly sizing, maker order generation |
| `src/kalshi_client.py` | Authenticated API client (copied from TSA) |
| `src/kalshi_auth.py` | RSA-PSS API authentication (copied from TSA) |
| `src/orderbook.py` | Orderbook fetching, fill cost computation |
| `src/cli_predict.py` | Main forecast CLI with Kalshi market integration |
| `src/cli_maker_submit.py` | Maker order validation, submission, update, resize, prune, exit orders |
| `scripts/fetch_aaa.py` | AAA price scraper |
| `scripts/log_aaa.py` | Daily price logging (appends to CSV) |
| `scripts/live_maker.py` | Real-time maker loop with adverse selection protection |
| `scripts/phase0_validation.py` | Feasibility checks (conditioning value, phi stability) |
| `scripts/run_backtest.py` | CLI for backtests (--all runs 0-7 with naive baseline) |
| `scripts/calibrate_variance.py` | Calibration diagnostics (Parts 0-5) |
| `dashboard/app.py` | Streamlit home page (gas price charts, metrics) |
| `dashboard/pages/1_predictions.py` | Model predictions, +EV trades, maker orders |
| `dashboard/pages/2_portfolio.py` | Live portfolio from Kalshi API |
| `dashboard/pages/3_history.py` | Portfolio history and P&L tracking |
| `dashboard/pages/4_live_maker.py` | Start/stop live_maker, log viewer |

## Commands

```bash
# Forecast (with Kalshi market matching)
python -m src.cli_predict --auto-week --auto-asof
python -m src.cli_predict --auto-week --auto-asof --save
python -m src.cli_predict --auto-week --auto-asof --maker

# Maker orders
python -m src.cli_maker_submit --dry-run            # Preview
python -m src.cli_maker_submit                       # Submit validated orders
python -m src.cli_maker_submit --update              # Reprice if outbid
python -m src.cli_maker_submit --resize              # Adjust to Kelly
python -m src.cli_maker_submit --prune               # Cancel negative EV
python -m src.cli_maker_submit --place-exits         # Exit sell orders
python -m src.cli_maker_submit --cancel-all          # Kill switch
python -m src.cli_maker_submit --list-open           # Open orders
python -m src.cli_maker_submit --positions           # Positions
python -m src.cli_maker_submit --balance             # Account balance

# Live maker
python scripts/live_maker.py              # Production
python scripts/live_maker.py --dry-run     # Monitor only
python scripts/live_maker.py --cancel-on-exit  # Cancel all on Ctrl+C

# Dashboard
streamlit run dashboard/app.py

# Phase 0 validation
PYTHONPATH=. python scripts/phase0_validation.py

# Tests
pytest tests/ -v
```

## Key Files

| File | Purpose |
|------|---------|
| `data/aaa_daily.csv` | Raw AAA price data (weekly 1990-2025, daily 2025+) |
| `data/latest_predict.csv` | Current predictions |
| `data/latest_predict_meta.json` | Forecast metadata |
| `data/latest_maker_orders.json` | Generated maker orders |

## Differences from TSA

| Aspect | TSA | Gas |
|--------|-----|-----|
| Model | SARIMAX(1,1,1)×(1,0,0,7) | AR(1) on diffs |
| Settlement | Weekly average (Mon-Sun) | Single-day AAA price (next Monday) |
| Seasonality | 28 features (holidays, Fourier K=1-4) | 12 features (Fourier K=2, DOW, seasonal_norm) |
| Data | Daily 7/week since 2019 (~1,882 obs) | Daily Dec 2025+ (~103 obs) |
| Simulation | 7 days, extract mean | 8 days, extract endpoint |
| Holiday blocking | Thanksgiving, Christmas, Early Jan | None (TBD) |
| Unit root | No (stationary after log) | Yes (non-stationary levels, stationary diffs) |
| Variance calibration | Calibrated from 112-week backtest | Uncalibrated placeholders |

## Trading

**Series ticker**: `KXAAAGASW`
**Fees**: Maker 1.75%, Taker 7% (same as TSA).

`cli_predict.py` handles full Kalshi integration:
- Event ticker generation (`KXAAAGASW-YYMONDD` based on settlement Monday)
- Market discovery via `/events/{ticker}` API
- Quote enrichment and orderbook fetching (parallel)
- Model probability matching to market strikes
- VWAP computation, trade metrics, maker order generation

## Maker Order Pipeline

### 1. Order Generation (`trade.py` via `cli_predict.py --maker`)

Same engine as TSA. `generate_maker_orders()` evaluates YES/NO for each bucket, picks highest EV side. Kelly sizing with 3x oversize multiplier, bankroll split equally across N valid buckets. 40-60% probability region skipped.

### 2. Order Validation (`cli_maker_submit.py`)

`validate_orders()` — unified single-pass validation:

| Filter | Threshold | Notes |
|--------|-----------|-------|
| 1. Temp blocklist | `TEMP_BLOCKED_THRESHOLDS` | Currently empty (no holiday blocking) |
| 2. Min probability | Day-of-week based | From `get_maker_min_prob_for_day()` |
| 3. EV recalculation | `calc_ev_if_filled_c()` | Always recalculates with real fees |
| 4. Continuous EV threshold | `0.5 * (100 - p)` cents | Scales with uncertainty |
| 5. EV robustness | EV after +1 tick slippage | >= 0.0c |
| 6. Contract clipping | `max_contracts` param | Clips, doesn't reject |

### 3. Order Submission

- `post_only=True` on all orders (ensures maker fee)
- NO prices converted to YES prices for API: `yes_price = 100 - no_price`
- Deduplication: cancels weaker-priced duplicates for same ticker/side
- Cash balance check and contract reduction if insufficient

### 4. Order Management Modes

| Mode | Function | Purpose |
|------|----------|---------|
| `--update` | `update_orders()` | Refresh if outbid (move to bid+1 if EV still positive) |
| `--resize` | `resize_orders()` | Adjust contract sizes to match current Kelly |
| `--prune` | `prune_orders()` | Cancel orders where EV went negative |
| `--place-exits` | `place_exit_orders()` | Place maker sell limits for held positions at +EV exit prices |
| `--cancel-all` | — | Kill switch, cancel all open orders |

### 5. Pre-Flight Checks

- Data freshness: `is_model_data_fresh()` blocks if `last_obs_date` > 1 day stale (catches silent AAA scraper failures)
- Minimum cash balance >= $500 (revisit after first live dry-run — gas contract sizes may differ from TSA)
- Opposite-position blocking (won't place BUY opposite to held position)
- Existing order deduplication (won't duplicate ticker/side)
- No TSA-style holiday blocking (gas has no equivalent)

## Automation

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `gas_monitor.yml` | Every 15 min, 7 AM–10 PM ET | Fetch AAA, predict, manage maker orders (prune→resize→update→place→exits), Discord alerts |
| `log_aaa.yml` | Daily 9 AM ET | Backup AAA price scraper (idempotent with gas_monitor) |
| `cancel-makers-sunday.yml` | Sunday ~11:55 PM ET | Cancel all maker orders at market close |
| `emergency-cancel.yml` | Manual dispatch | Kill switch — cancel all open orders |

**gas_monitor.yml pipeline order**: fetch AAA → predict (--save --maker --auto-bankroll) → prune (-EV orders) → resize (Kelly adjustment) → update (outbid repricing) → place new → place exits → build Discord message → post to Discord → commit data.

**Discord alerts**: Only sent when order changes occur (placed, cancelled, resized, updated, or exit orders). Includes forecast header (settlement date, last price, mean, std, obs count) and order change details.

**Data commits**: Workflow auto-commits `aaa_daily.csv`, `latest_predict.csv`, `latest_predict_meta.json`, `latest_maker_orders.json` after each run.

## Phase Status

- [x] Phase 0: Market validation (conditioning value strong, phi stable on weekly windows)
- [x] Phase 1: Cleanup & scaffold
- [x] Phase 2: Core forecast engine (AR(1), features, simulation, conditioning)
- [x] Phase 3: Kalshi integration (market discovery, orderbooks, predictions vs market)
- [ ] Phase 2.5: GasBuddy integration (BLOCKED — ToS prohibits scraping, no public API)
- [x] Phase 4: Order management (cli_maker_submit.py with validation gates)
- [x] Phase 5: Automation (GitHub Actions, Discord alerts)
- [x] Phase 6: Backtesting & calibration (completed — see results below)

## Backtesting

**Walk-forward backtest**: 12 weeks (2025-12-22 to 2026-03-09), 25k sims, expanding training window, actual Kalshi strikes (19 thresholds).

**Thresholds**: 10c increments $2.90-$4.50 + 5c exceptions at $3.35, $3.45 (19 strikes total). Defined in `backtest.py:kalshi_thresholds()`.

| Obs Days | MAE | MAPE | Brier | vs Naive | Dir Acc | Mean Z | Std Z |
|----------|-----|------|-------|----------|---------|--------|-------|
| 0 | $0.156 | 4.83% | 0.0677 | — | 41.7% | +0.89 | 3.28 |
| 1 | $0.151 | 4.69% | 0.0653 | -3.6% | 50.0% | +0.92 | 3.46 |
| 2 | $0.138 | 4.33% | 0.0575 | -15.2% | 50.0% | +1.13 | 3.51 |
| 3 | $0.121 | 3.84% | 0.0502 | -25.9% | 50.0% | +1.33 | 3.62 |
| 4 | $0.104 | 3.30% | 0.0416 | -38.6% | 50.0% | +1.26 | 3.62 |
| 5 | $0.080 | 2.58% | 0.0310 | -54.2% | 41.7% | +1.21 | 3.64 |
| 6 | $0.055 | 1.78% | 0.0191 | -71.8% | 75.0% | +1.23 | 3.40 |
| 7 | $0.034 | 1.12% | 0.0091 | -86.6% | 83.3% | +0.87 | 2.84 |

**Naive baseline**: AR(1) at observed_days=0 (zero conditioning).

**vs Random Walk baseline** (N(current_price, empirical_weekly_σ)):
- Day 0-3: AR(1) is **worse** than random walk (+49% to +10% worse Brier)
- Day 4: AR(1) matches random walk (-10%)
- Day 5-7: AR(1) beats random walk (-34% to -81%) — mostly from conditioning

**Calibration applied**:
- `GLOBAL_SIGMA_MULTIPLIER = 3.8` — from 1,850 Mon-to-Mon observations (independent of backtest)
- All other scalers left at 1.0/0.0 — N=12 too thin for reliable per-day calibration

**Calibration NOT applied** (insufficient data):
- DOW variance ratios (N≈15 per DOW, too noisy)
- Seasonal multipliers (single season in sample)
- Conditioning variance scaler (Std Z ~3.3 driven by 3 outlier weeks)
- Mean bias corrections (risk of overfitting to 12 data points)

**Honest assessment**: The AR(1) model does not demonstrate unconditional predictive power over a random walk for days 0-3. Its value comes from conditioning on observed intra-week prices (day 4+). With only 12 backtest weeks during a specific regime (flat→sharp uptrend), all conclusions have wide confidence intervals. Revisit calibration when N>30 weeks (~Aug 2026).

## Conventions

Same as TSA project:
- Config is source of truth for thresholds/parameters
- Plan before implementing for complex changes
- Accept strategic corrections immediately
