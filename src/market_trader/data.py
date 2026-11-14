"""Canonical candle import, validation, aggregation and historical windows."""

import csv
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from itertools import groupby
from pathlib import Path
from typing import Never

from market_trader._numeric import sum_decimals
from market_trader.domain import Candle, CandleValidationError

_NUMBERS = ("open", "high", "low", "close", "volume")
_TIMES = ("open_time", "close_time", "available_at")
_COLUMNS = ("instrument", "interval_seconds", *_TIMES, *_NUMBERS)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _fail(row_number: int, field: str, message: str) -> Never:
    raise CandleValidationError(row_number=row_number, field=field, message=message)


def _required(row: dict, row_number: int, field: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        _fail(row_number, field, "required value is missing")
    return value.strip()


def _decimal(row: dict, row_number: int, field: str) -> Decimal:
    value = _required(row, row_number, field)
    try:
        return Decimal(value)
    except InvalidOperation:
        _fail(row_number, field, "expected a decimal number")


def _timestamp(row: dict, row_number: int, field: str) -> datetime:
    value = _required(row, row_number, field)
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.utcoffset() is None:
            _fail(row_number, field, "an explicit timezone is required")
        return timestamp.astimezone(UTC)
    except (ValueError, OverflowError) as error:
        # Preserve an already classified data error rather than wrapping it.
        if isinstance(error, CandleValidationError):
            raise
        _fail(row_number, field, "expected an ISO timestamp with timezone")


def _interval(row: dict, row_number: int) -> timedelta:
    value = _required(row, row_number, "interval_seconds")
    try:
        seconds = int(value)
        if seconds <= 0:
            raise ValueError
        return timedelta(seconds=seconds)
    except (ValueError, OverflowError):
        _fail(row_number, "interval_seconds", "expected positive whole seconds")


def _validate_candle(candle: Candle, row_number: int) -> None:
    if not isinstance(candle, Candle):
        _fail(row_number, "row", "expected a Candle")
    if (not isinstance(candle.instrument, str) or not candle.instrument.strip()
            or candle.instrument != candle.instrument.strip()):
        _fail(row_number, "instrument", "expected a nonempty instrument without padding")
    if not isinstance(candle.interval, timedelta) or candle.interval <= timedelta(0):
        _fail(row_number, "interval", "expected a positive duration")
    for field in _NUMBERS:
        value = getattr(candle, field)
        if not isinstance(value, Decimal) or not value.is_finite():
            _fail(row_number, field, "expected a finite Decimal")
    for field in _TIMES:
        value = getattr(candle, field)
        if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
            _fail(row_number, field, "expected a timezone-aware UTC timestamp")
    if candle.close_time - candle.open_time != candle.interval:
        _fail(row_number, "close_time", "must equal open_time plus interval")
    if candle.available_at < candle.close_time:
        _fail(row_number, "available_at", "cannot precede close_time")
    if candle.high < max(candle.open, candle.close):
        _fail(row_number, "high", "cannot be below open or close")
    if candle.low > min(candle.open, candle.close):
        _fail(row_number, "low", "cannot be above open or close")
    if candle.volume < 0:
        _fail(row_number, "volume", "cannot be negative")


def _history(candles: Sequence[Candle]) -> tuple[Candle, ...]:
    history = tuple(candles)
    validate_candles(history)
    if not history:
        return history
    first = previous = history[0]
    for row_number, candle in enumerate(history[1:], start=2):
        if candle.instrument != first.instrument:
            _fail(row_number, "instrument", "history must contain one instrument")
        if candle.interval != first.interval:
            _fail(row_number, "interval", "history must contain one source interval")
        if candle.open_time < previous.close_time:
            _fail(row_number, "open_time", "history must be ordered and non-overlapping")
        previous = candle
    return history


def read_candles(path: Path) -> tuple[Candle, ...]:
    """Read canonical CSV with decimal values and validate its candle rows."""
    candles = []
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        header = None
        try:
            header = reader.fieldnames
            if header is None:
                _fail(0, "header", "CSV header is missing")
            seen = set()
            for field in header:
                if field in seen:
                    _fail(0, field, "duplicate CSV column")
                if field not in _COLUMNS:
                    _fail(0, field, "unexpected CSV column")
                seen.add(field)
            for field in _COLUMNS:
                if field not in seen:
                    _fail(0, field, "required CSV column is missing")
            for row_number, row in enumerate(reader, start=1):
                if None in row:
                    _fail(row_number, "row", "more values than CSV columns")
                candle = Candle(
                    instrument=_required(row, row_number, "instrument"),
                    interval=_interval(row, row_number),
                    **{field: _timestamp(row, row_number, field) for field in _TIMES},
                    **{field: _decimal(row, row_number, field) for field in _NUMBERS},
                )
                _validate_candle(candle, row_number)
                candles.append(candle)
        except csv.Error as error:
            if header is None:
                _fail(0, "header", f"invalid CSV header: {error}")
            _fail(len(candles) + 1, "row", f"invalid CSV: {error}")
    return tuple(candles)


def validate_candles(candles: Sequence[Candle]) -> None:
    """Raise CandleValidationError for invalid numeric or temporal values."""
    for row_number, candle in enumerate(candles, start=1):
        _validate_candle(candle, row_number)


def resample_candles(
    candles: Sequence[Candle], *, interval: timedelta
) -> tuple[Candle, ...]:
    """Aggregate complete UTC-epoch-aligned intervals; reject incomplete blocks."""
    if not isinstance(interval, timedelta) or interval <= timedelta(0):
        raise ValueError("interval must be a positive timedelta")
    history = _history(candles)
    if not history:
        return ()
    source_interval = history[0].interval
    if interval % source_interval:
        raise ValueError("interval must be a whole multiple of the source interval")
    expected_count = interval // source_interval
    result = []

    def bucket(item: tuple[int, Candle]) -> datetime:
        return _EPOCH + ((item[1].open_time - _EPOCH) // interval) * interval

    for start, items in groupby(enumerate(history, start=1), key=bucket):
        entries = tuple(items)
        group = tuple(candle for _, candle in entries)
        if len(group) != expected_count or any(
            candle.open_time != start + index * source_interval
            for index, candle in enumerate(group)
        ):
            _fail(entries[0][0], "open_time", f"incomplete target interval starting {start.isoformat()}")
        result.append(Candle(
            instrument=group[0].instrument,
            interval=interval,
            open_time=start,
            close_time=start + interval,
            available_at=max(candle.available_at for candle in group),
            open=group[0].open,
            high=max(candle.high for candle in group),
            low=min(candle.low for candle in group),
            close=group[-1].close,
            volume=sum_decimals(tuple(candle.volume for candle in group)),
        ))
    return tuple(result)


def rolling_windows(
    candles: Sequence[Candle], *, size: int
) -> Iterator[tuple[Candle, ...]]:
    """Yield full historical windows in order, advancing by one candle."""
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    history = _history(candles)
    return (history[end - size : end] for end in range(size, len(history) + 1))
