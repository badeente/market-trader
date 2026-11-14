"""Causal numeric representations of validated historical candles."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from market_trader._numeric import mean_decimals
from market_trader.data import rolling_windows
from market_trader.domain import Candle


@dataclass(frozen=True, slots=True)
class MeanCloseFeature:
    """Trailing mean with its instrument, period and information boundary."""

    instrument: str
    interval: timedelta
    close_time: datetime
    available_at: datetime
    value: Decimal


def mean_close_features(
    candles: Sequence[Candle], *, lookback: int
) -> tuple[MeanCloseFeature, ...]:
    """Return one causal trailing mean per full lookback, without warmup rows."""
    return tuple(
        MeanCloseFeature(
            instrument=window[-1].instrument,
            interval=window[-1].interval,
            close_time=window[-1].close_time,
            available_at=max(candle.available_at for candle in window),
            value=mean_decimals(tuple(candle.close for candle in window)),
        )
        for window in rolling_windows(candles, size=lookback)
    )
