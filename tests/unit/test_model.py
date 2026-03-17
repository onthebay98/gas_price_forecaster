"""Tests for AR(1) model fitting and simulation."""
import numpy as np
import pandas as pd
import pytest

from src.model import (
    AR1Result,
    compute_threshold_probs,
    compute_weekly_avg_distribution,
    fit_ar1,
    simulate_future_days,
)
from src.features import make_features


def _make_synthetic_prices(n=200, seed=42):
    """Generate synthetic daily price series with AR(1) structure."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-12-01", periods=n, freq="D")
    prices = np.zeros(n)
    prices[0] = 3.50
    for i in range(1, n):
        diff = 0.001 + 0.3 * (prices[i - 1] - prices[i - 2] if i >= 2 else 0) + rng.normal(0, 0.01)
        prices[i] = prices[i - 1] + diff
    return pd.Series(prices, index=dates, name="price")


class TestFitAR1:
    def test_basic_fit(self):
        prices = _make_synthetic_prices()
        features = make_features(prices.index)
        result = fit_ar1(prices, features)

        assert isinstance(result, AR1Result)
        assert result.n_obs > 100
        assert -1 < result.phi < 1  # Stationary
        assert result.sigma > 0
        assert 0 <= result.r_squared <= 1
        assert len(result.beta) == len(features.columns)

    def test_fits_on_non_daily_data(self):
        """AR(1) should still fit on non-daily data (with warning logged)."""
        # Create weekly data (7-day gaps)
        dates = pd.date_range("2020-01-06", periods=100, freq="7D")
        prices = pd.Series(3.0 + np.cumsum(np.random.normal(0, 0.05, 100)), index=dates)
        result = fit_ar1(prices)
        # Should still fit but n_obs reflects the data we gave it
        assert result.n_obs > 50

    def test_fit_without_features(self):
        """fit_ar1 should work without exogenous features."""
        prices = _make_synthetic_prices()
        result = fit_ar1(prices)

        assert isinstance(result, AR1Result)
        assert len(result.beta) == 0
        assert len(result.feature_names) == 0
        assert result.n_obs > 100

    def test_train_date_range(self):
        """AR1Result should report training date range."""
        prices = _make_synthetic_prices()
        result = fit_ar1(prices)
        assert result.train_start == "2025-12-03"  # First valid date after diff + lag (need 2 prior days)
        assert result.train_end is not None

    def test_residuals_zero_mean(self):
        prices = _make_synthetic_prices()
        features = make_features(prices.index)
        result = fit_ar1(prices, features)

        # OLS residuals should be mean-zero
        assert abs(result.residuals.mean()) < 0.001


class TestSimulateFutureDays:
    def test_unconditional_simulation(self):
        prices = _make_synthetic_prices()
        features = make_features(prices.index)
        ar1 = fit_ar1(prices, features)

        # Forecast 7 days ahead
        future_dates = pd.date_range(prices.index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
        future_features = make_features(future_dates)

        sims = simulate_future_days(
            ar1=ar1,
            last_price=float(prices.iloc[-1]),
            last_daily_diff=float(prices.iloc[-1] - prices.iloc[-2]),
            future_features=future_features,
            n_sims=1000,
        )

        assert sims.shape == (7, 1000)
        # Prices should be roughly in the right range
        assert np.all(sims > 0)  # No negative prices
        assert np.mean(sims) > 1.0  # Above $1/gal

    def test_conditioned_simulation(self):
        prices = _make_synthetic_prices()
        features = make_features(prices.index)
        ar1 = fit_ar1(prices, features)

        last_price = float(prices.iloc[-1])
        future_dates = pd.date_range(prices.index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
        future_features = make_features(future_dates)

        # Condition on 3 observed days
        observed = np.array([last_price + 0.01, last_price + 0.02, last_price + 0.03])

        sims = simulate_future_days(
            ar1=ar1,
            last_price=last_price,
            last_daily_diff=0.005,
            future_features=future_features,
            n_sims=1000,
            observed_prices=observed,
            n_observed=3,
        )

        # First 3 days should be exact
        np.testing.assert_array_almost_equal(sims[0, :], observed[0])
        np.testing.assert_array_almost_equal(sims[1, :], observed[1])
        np.testing.assert_array_almost_equal(sims[2, :], observed[2])

        # Remaining days should have variance
        assert np.std(sims[3, :]) > 0
        assert np.std(sims[6, :]) > 0

    def test_conditioning_reduces_variance(self):
        prices = _make_synthetic_prices()
        features = make_features(prices.index)
        ar1 = fit_ar1(prices, features)

        last_price = float(prices.iloc[-1])
        future_dates = pd.date_range(prices.index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
        future_features = make_features(future_dates)

        # Unconditional
        sims_uncond = simulate_future_days(
            ar1=ar1, last_price=last_price, last_daily_diff=0.005,
            future_features=future_features, n_sims=5000,
        )
        avg_uncond = compute_weekly_avg_distribution(sims_uncond)

        # Conditioned on 4 days
        observed = np.array([last_price + 0.01, last_price + 0.02, last_price + 0.03, last_price + 0.04])
        sims_cond = simulate_future_days(
            ar1=ar1, last_price=last_price, last_daily_diff=0.005,
            future_features=future_features, n_sims=5000,
            observed_prices=observed, n_observed=4,
        )
        avg_cond = compute_weekly_avg_distribution(sims_cond)

        # Conditioned should have less variance in weekly average
        assert np.std(avg_cond) < np.std(avg_uncond)


class TestThresholdProbs:
    def test_basic_probs(self):
        avgs = np.array([3.40, 3.45, 3.50, 3.55, 3.60])
        probs = compute_threshold_probs(avgs, [3.45, 3.50])

        assert probs[3.45] == 0.6  # 3 out of 5 above 3.45
        assert probs[3.50] == 0.4  # 2 out of 5 above 3.50

    def test_extreme_thresholds(self):
        avgs = np.random.normal(3.50, 0.05, 10000)
        probs = compute_threshold_probs(avgs, [0.0, 10.0])

        assert probs[0.0] > 0.99  # Almost all above $0
        assert probs[10.0] < 0.01  # Almost none above $10
