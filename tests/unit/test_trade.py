"""Tests for trading logic (fees, EV, Kelly)."""
import pytest

from src.trade import (
    FeeSchedule,
    calc_ev_if_filled_c,
    kalshi_fee_total_dollars,
)


class TestFees:
    def test_maker_fee(self):
        fee = kalshi_fee_total_dollars(
            price_dollars=0.50, contracts=100, is_maker=True
        )
        # 0.0175 * 100 * 0.5 * 0.5 = 0.4375 → rounded up to $0.44
        assert fee == 0.44

    def test_taker_fee(self):
        fee = kalshi_fee_total_dollars(
            price_dollars=0.50, contracts=100, is_maker=False
        )
        # 0.07 * 100 * 0.5 * 0.5 = 1.75 → exactly $1.75
        assert fee == 1.75

    def test_zero_contracts(self):
        fee = kalshi_fee_total_dollars(
            price_dollars=0.50, contracts=0, is_maker=True
        )
        assert fee == 0.0

    def test_extreme_price(self):
        # At 99c (0.99), fee should be very small
        fee = kalshi_fee_total_dollars(
            price_dollars=0.99, contracts=100, is_maker=True
        )
        # 0.0175 * 100 * 0.99 * 0.01 = 0.01733 → $0.02
        assert fee == 0.02


class TestEV:
    def test_positive_ev_yes(self):
        # Model says 90% YES, buying at 80c → should be positive EV
        ev = calc_ev_if_filled_c(p_model=0.90, side="YES", price_paid_c=80, is_maker=True)
        assert ev > 0

    def test_negative_ev(self):
        # Model says 50% YES, buying at 60c → should be negative EV
        ev = calc_ev_if_filled_c(p_model=0.50, side="YES", price_paid_c=60, is_maker=True)
        assert ev < 0

    def test_no_side(self):
        # Model says 10% YES → 90% NO, buying NO at 80c → should be positive
        ev = calc_ev_if_filled_c(p_model=0.10, side="NO", price_paid_c=80, is_maker=True)
        assert ev > 0
