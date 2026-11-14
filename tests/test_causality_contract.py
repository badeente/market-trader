"""Contract tests for CAU-001 and the numeric part of CAU-003."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from market_trader.data import rolling_windows
from market_trader.domain import Candle
from market_trader.representations import MeanCloseFeature, mean_close_features


@pytest.mark.case("CAU-001")
@pytest.mark.parametrize(
    "available_count,expected_starts",
    [
        pytest.param(0, (), id="empty_history"),
        pytest.param(2, (), id="warmup_incomplete"),
        pytest.param(3, (0,), id="first_complete_window"),
        pytest.param(4, (0, 1), id="advance_by_one_candle"),
    ],
)
def test_cau_001_only_full_historical_windows_are_yielded(
    minute_candles: tuple[Candle, ...],
    available_count: int,
    expected_starts: tuple[int, ...],
) -> None:
    windows = list(rolling_windows(minute_candles[:available_count], size=3))

    assert windows == [minute_candles[start : start + 3] for start in expected_starts]


@pytest.mark.case("CAU-003")
def test_cau_003_future_prices_do_not_change_historical_numeric_features(
    minute_candles: tuple[Candle, ...],
) -> None:
    # Only candles after the common three-candle prefix change.
    changed_future = minute_candles[:3] + (
        replace(
            minute_candles[3],
            open=Decimal("3"),
            high=Decimal("4"),
            low=Decimal("2"),
            close=Decimal("3.5"),
        ),
        replace(
            minute_candles[4],
            open=Decimal("4"),
            high=Decimal("5"),
            low=Decimal("3"),
            close=Decimal("4.5"),
        ),
    )
    original = mean_close_features(minute_candles, lookback=2)
    changed = mean_close_features(changed_future, lookback=2)
    prefix_only = mean_close_features(minute_candles[:3], lookback=2)

    def expected(values: tuple[str, ...]) -> tuple[MeanCloseFeature, ...]:
        return tuple(
            MeanCloseFeature(
                instrument="TEST",
                interval=timedelta(minutes=1),
                close_time=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                available_at=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                value=Decimal(value),
            )
            for minute, value in enumerate(values, start=2)
        )

    # Independent expected values also reject empty, constant or stale output.
    assert original == expected(("101", "102", "103", "104"))
    assert changed == expected(("101", "102", "53", "4"))
    assert original[:2] == changed[:2] == prefix_only == expected(("101", "102"))
