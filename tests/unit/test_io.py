"""Tests for data loading."""
import pandas as pd
import pytest

from src.io import load_aaa_csv, load_daily_prices, load_interpolated_daily


class TestLoadAAA:
    def test_loads_successfully(self):
        df = load_aaa_csv()
        assert "price" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) > 1000  # Should have at least 1000 weekly obs

    def test_no_negative_prices(self):
        df = load_aaa_csv()
        assert (df["price"].dropna() > 0).all()

    def test_date_range(self):
        df = load_aaa_csv()
        assert df.index.min().year == 1990
        assert df.index.max().year >= 2026


class TestLoadDailyPrices:
    def test_loads_daily_portion(self):
        df = load_daily_prices()
        assert isinstance(df.index, pd.DatetimeIndex)
        # Daily data should have consecutive dates
        gaps = df.index.to_series().diff().dt.days.dropna()
        assert (gaps <= 1).all()
        # Should have at least some observations
        assert len(df) > 50

    def test_prices_positive(self):
        df = load_daily_prices()
        assert (df["price"].dropna() > 0).all()


class TestLoadInterpolatedDaily:
    def test_no_gaps(self):
        df = load_interpolated_daily()
        gaps = df.index.to_series().diff().dt.days.dropna()
        assert (gaps == 1).all()

    def test_interpolated_values_reasonable(self):
        df = load_interpolated_daily()
        # All values should be positive (gas prices)
        assert (df["price"] > 0).all()
        # No NaN after interpolation
        assert df["price"].notna().all()
