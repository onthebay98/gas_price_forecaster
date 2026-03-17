"""Tests for feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, add_seasonal_norm, make_features


class TestMakeFeatures:
    def test_output_shape(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        features = make_features(dates)

        assert features.shape == (30, 11)  # 4 Fourier + 7 DOW

    def test_fourier_range(self):
        dates = pd.date_range("2026-01-01", periods=365, freq="D")
        features = make_features(dates)

        for col in ["sin_y1", "cos_y1", "sin_y2", "cos_y2"]:
            assert features[col].min() >= -1.0
            assert features[col].max() <= 1.0

    def test_dow_dummies_exclusive(self):
        dates = pd.date_range("2026-01-01", periods=14, freq="D")
        features = make_features(dates)

        dow_cols = [f"dow_{d}" for d in range(7)]
        # Each row should have exactly one DOW dummy = 1
        row_sums = features[dow_cols].sum(axis=1)
        np.testing.assert_array_equal(row_sums.values, np.ones(14))

    def test_monday_is_dow_0(self):
        # 2026-01-05 is Monday
        dates = pd.DatetimeIndex([pd.Timestamp("2026-01-05")])
        features = make_features(dates)
        assert features["dow_0"].iloc[0] == 1.0
        assert features["dow_6"].iloc[0] == 0.0


class TestSeasonalNorm:
    def test_adds_column(self):
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        features = make_features(dates)

        # Create a fake price history
        history_dates = pd.date_range("2020-01-01", periods=2000, freq="D")
        history = pd.Series(
            3.0 + 0.5 * np.sin(2 * np.pi * history_dates.dayofyear / 365),
            index=history_dates,
        )

        features = add_seasonal_norm(features, history)
        assert "seasonal_norm" in features.columns
        assert features["seasonal_norm"].notna().all()

    def test_seasonal_pattern(self):
        """Seasonal norm should reflect summer vs winter price levels."""
        history_dates = pd.date_range("2020-01-01", periods=2000, freq="D")
        # Summer high, winter low
        history = pd.Series(
            3.0 + 0.5 * np.sin(2 * np.pi * (history_dates.dayofyear - 90) / 365),
            index=history_dates,
        )

        # January dates vs July dates
        jan_dates = pd.date_range("2026-01-01", periods=7, freq="D")
        jul_dates = pd.date_range("2026-07-01", periods=7, freq="D")

        jan_feats = add_seasonal_norm(make_features(jan_dates), history)
        jul_feats = add_seasonal_norm(make_features(jul_dates), history)

        # July seasonal_norm should be higher than January
        assert jul_feats["seasonal_norm"].mean() > jan_feats["seasonal_norm"].mean()

    def test_empty_history(self):
        dates = pd.date_range("2026-01-01", periods=7, freq="D")
        features = make_features(dates)
        empty = pd.Series(dtype=float)
        features = add_seasonal_norm(features, empty)
        assert (features["seasonal_norm"] == 0.0).all()
