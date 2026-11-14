"""Synthetic, hand-specified candles; fixtures never call production loaders."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from market_trader.domain import Candle


@pytest.fixture
def minute_candles() -> tuple[Candle, ...]:
    """F-RESAMPLE from TEST_CASES.md, also used for historical prefix tests."""
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = (
        ("100", "101", "99", "100.5", "1.5"),
        ("101", "102", "100", "101.5", "2"),
        ("102", "103", "101", "102.5", "0"),
        ("103", "104", "102", "103.5", "3"),
        ("104", "105", "103", "104.5", "0.5"),
    )
    return tuple(
        Candle(
            instrument="TEST",
            interval=timedelta(minutes=1),
            open_time=start + timedelta(minutes=index),
            close_time=start + timedelta(minutes=index + 1),
            available_at=start + timedelta(minutes=index + 1),
            open=Decimal(open_),
            high=Decimal(high),
            low=Decimal(low),
            close=Decimal(close),
            volume=Decimal(volume),
        )
        for index, (open_, high, low, close, volume) in enumerate(values)
    )
