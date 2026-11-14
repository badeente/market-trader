"""DAT-006: explicit synthetic sessions, closed periods and active candle gaps."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.domain import CandleValidationError
from market_trader.normalization import (
    CandleGap, DataGapError, NormalizationConfig, normalize_candles,
)

MINUTE = timedelta(minutes=1)


def at(day, hour=0, minute=0):
    return datetime(2026, 1, day, hour, minute, tzinfo=UTC)


def schedule(*sessions):
    return SessionCalendar("synthetic-v1", at(1), at(6), tuple(sessions))


def bar(template, timestamp, *, interval=MINUTE, instrument="TEST"):
    return replace(
        template, instrument=instrument, interval=interval, open_time=timestamp,
        close_time=timestamp + interval, available_at=timestamp + interval,
    )


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize(("second_day", "second_hour", "second_minute"), [
    (2, 9, 4), (3, 9, 0), (5, 9, 0),
], ids=["intraday-break", "overnight", "closed-days"])
def test_dat_006_closed_periods_are_not_missing_active_candles(
    minute_candles, second_day, second_hour, second_minute,
):
    second_open = at(second_day, second_hour, second_minute)
    calendar = schedule(
        TradingSession(at(2, 9), at(2, 9, 2)),
        TradingSession(second_open, second_open + 2 * MINUTE),
    )
    before = bar(minute_candles[0], at(2, 9, 1))
    after = bar(minute_candles[0], second_open)
    config = NormalizationConfig(calendar=calendar, gap_policy="reject")
    dataset = normalize_candles((after, before), config=config).datasets[0]
    assert dataset.candles == (before, after)
    assert dataset.quality.gaps == ()
    assert dataset.quality.source_rows == (2, 1)
    assert dataset.config.calendar == calendar
    assert dataset.config.calendar.name == "synthetic-v1"


@pytest.mark.case("DAT-005")
def test_dat_005_only_active_parts_of_a_gap_are_reported(minute_candles):
    calendar = schedule(
        TradingSession(at(2, 9), at(2, 9, 3)),
        TradingSession(at(2, 9, 5), at(2, 9, 8)),
    )
    first = bar(minute_candles[0], at(2, 9))
    last = bar(minute_candles[0], at(2, 9, 7))
    source = [last, first, first]
    dataset = normalize_candles(source, config=NormalizationConfig(calendar=calendar)).datasets[0]
    assert dataset.candles == (first, last)
    assert dataset.quality.gaps == (
        CandleGap(at(2, 9, 1), at(2, 9, 3), 2),
        CandleGap(at(2, 9, 5), at(2, 9, 7), 2),
    )
    assert dataset.quality.input_count == 3
    assert dataset.quality.output_count == 2
    assert source == [last, first, first]


@pytest.mark.case("DAT-005")
def test_dat_005_strict_calendar_rejects_active_gaps_with_original_row(minute_candles):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 4)))
    first = bar(minute_candles[0], at(2, 9))
    last = bar(minute_candles[0], at(2, 9, 3))
    with pytest.raises(DataGapError) as raised:
        normalize_candles((last, first), config=NormalizationConfig(calendar=calendar, gap_policy="reject"))
    assert raised.value.row_number == 1
    assert raised.value.dataset.quality.gaps == (CandleGap(at(2, 9, 1), at(2, 9, 3), 2),)
    assert raised.value.dataset.config.calendar == calendar


@pytest.mark.case("DAT-005")
def test_dat_005_completely_missing_session_between_observed_candles_is_reported(minute_candles):
    calendar = schedule(*(TradingSession(at(day, 9), at(day, 9, 2)) for day in (2, 3, 5)))
    before = bar(minute_candles[0], at(2, 9, 1))
    after = bar(minute_candles[0], at(5, 9))
    dataset = normalize_candles((before, after), config=NormalizationConfig(calendar=calendar)).datasets[0]
    assert dataset.quality.gaps == (CandleGap(at(3, 9), at(3, 9, 2), 2),)


@pytest.mark.case("DAT-006")
def test_dat_006_each_session_has_its_own_interval_grid_and_close_boundary(minute_candles):
    calendar = schedule(
        TradingSession(at(2, 9), at(2, 9, 4)),
        TradingSession(at(2, 9, 5), at(2, 9, 9)),
    )
    candles = tuple(bar(minute_candles[0], at(2, 9, minute), interval=2 * MINUTE) for minute in (0, 2, 5, 7))
    candles = (*candles[:-1], replace(candles[-1], available_at=at(2, 10)))
    dataset = normalize_candles(candles, config=NormalizationConfig(calendar=calendar, gap_policy="reject")).datasets[0]
    assert dataset.quality.gaps == ()
    assert dataset.candles == candles
    assert dataset.candles[-1].close_time == calendar.sessions[-1].close_time
    assert dataset.candles[-1].available_at == at(2, 10)


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize("timestamp", [at(2, 8, 59), at(2, 9, 2), at(2, 9, 3), at(2, 9, 6)])
def test_dat_006_candles_in_closed_periods_are_rejected_not_dropped(minute_candles, timestamp):
    calendar = schedule(
        TradingSession(at(2, 9), at(2, 9, 2)),
        TradingSession(at(2, 9, 4), at(2, 9, 6)),
    )
    valid = bar(minute_candles[0], at(2, 9))
    invalid = bar(minute_candles[0], timestamp)
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((valid, invalid), config=NormalizationConfig(calendar=calendar))
    assert raised.value.row_number == 2
    assert raised.value.field == "open_time"


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize("timestamp", [at(1) - MINUTE, at(6)])
def test_dat_006_candles_outside_declared_coverage_are_rejected(minute_candles, timestamp):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 4)))
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((bar(minute_candles[0], timestamp),), config=NormalizationConfig(calendar=calendar))
    assert raised.value.row_number == 1
    assert raised.value.field == "open_time"


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize(("minute", "field"), [(1, "open_time"), (3, "close_time")])
def test_dat_006_off_grid_or_session_crossing_candles_are_rejected(minute_candles, minute, field):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 4)))
    candle = bar(minute_candles[0], at(2, 9, minute), interval=2 * MINUTE)
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((candle,), config=NormalizationConfig(calendar=calendar))
    assert raised.value.row_number == 1
    assert raised.value.field == field


@pytest.mark.case("DAT-006")
def test_dat_006_partial_session_candles_are_not_silently_truncated(minute_candles):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 3)))
    candle = bar(minute_candles[0], at(2, 9), interval=2 * MINUTE)
    with pytest.raises(ValueError) as raised:
        normalize_candles((candle,), config=NormalizationConfig(calendar=calendar))
    assert not isinstance(raised.value, CandleValidationError)  # Unsupported configuration.


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize("invalid", [
    "empty-name", "padded-name", "wrong-name-type", "naive-coverage", "non-utc-coverage",
    "reversed-coverage", "zero-coverage", "mutable-sessions", "wrong-session-type",
    "naive-session", "non-utc-session", "reversed-session", "zero-session",
    "overlapping-sessions", "unordered-sessions", "duplicate-session",
    "session-before-coverage", "session-after-coverage",
])
def test_dat_006_invalid_schedules_fail_even_without_candles(invalid):
    session = TradingSession(at(2, 9), at(2, 9, 4))
    base = schedule(session)
    offset = timezone(timedelta(hours=1))
    changes = {
        "empty-name": {"name": ""},
        "padded-name": {"name": " synthetic "},
        "wrong-name-type": {"name": 7},
        "naive-coverage": {"coverage_start": at(1).replace(tzinfo=None)},
        "non-utc-coverage": {"coverage_start": at(1).astimezone(offset)},
        "reversed-coverage": {"coverage_end": at(1) - MINUTE},
        "zero-coverage": {"coverage_end": at(1)},
        "mutable-sessions": {"sessions": [session]},
        "wrong-session-type": {"sessions": ("invalid",)},
        "naive-session": {"sessions": (replace(session, open_time=at(2, 9).replace(tzinfo=None)),)},
        "non-utc-session": {"sessions": (replace(session, close_time=at(2, 9, 4).astimezone(offset)),)},
        "reversed-session": {"sessions": (replace(session, close_time=at(2, 8)),)},
        "zero-session": {"sessions": (replace(session, close_time=at(2, 9)),)},
        "overlapping-sessions": {"sessions": (session, TradingSession(at(2, 9, 3), at(2, 9, 5)))},
        "unordered-sessions": {"sessions": (session, TradingSession(at(2, 8), at(2, 8, 2)))},
        "duplicate-session": {"sessions": (session, session)},
        "session-before-coverage": {"sessions": (TradingSession(at(1) - MINUTE, at(1)),)},
        "session-after-coverage": {"sessions": (TradingSession(at(6), at(6) + MINUTE),)},
    }
    calendar = replace(base, **changes[invalid])
    with pytest.raises(ValueError):
        normalize_candles((), config=NormalizationConfig(calendar=calendar))


@pytest.mark.case("DAT-006")
def test_dat_006_empty_calendar_is_a_valid_explicitly_closed_schedule(minute_candles):
    config = NormalizationConfig(calendar=schedule())
    assert normalize_candles((), config=config).datasets == ()
    with pytest.raises(CandleValidationError):
        normalize_candles((bar(minute_candles[0], at(2, 9)),), config=config)


@pytest.mark.case("DAT-006")
def test_dat_006_adjacent_sessions_are_allowed(minute_candles):
    calendar = schedule(
        TradingSession(at(2, 9), at(2, 9, 2)),
        TradingSession(at(2, 9, 2), at(2, 9, 4)),
    )
    candles = tuple(bar(minute_candles[0], at(2, 9, minute)) for minute in range(4))
    assert normalize_candles(candles, config=NormalizationConfig(calendar=calendar)).datasets[0].quality.gaps == ()


@pytest.mark.case("DAT-006")
def test_dat_006_no_unrequested_boundary_gaps_are_inferred(minute_candles):
    calendar = schedule(TradingSession(at(2, 9), at(2, 10)))
    candle = bar(minute_candles[0], at(2, 9, 30))
    dataset = normalize_candles((candle,), config=NormalizationConfig(calendar=calendar, gap_policy="reject")).datasets[0]
    assert dataset.quality.gaps == ()


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("changed", ["name", "session-open", "session-close", "new-session", "coverage-start", "coverage-end"])
def test_dat_009_calendar_rules_are_part_of_identity(minute_candles, changed):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 3)))
    modifications = {
        "name": replace(calendar, name="synthetic-v2"),
        "session-open": replace(calendar, sessions=(TradingSession(at(2, 8, 59), at(2, 9, 3)),)),
        "session-close": replace(calendar, sessions=(TradingSession(at(2, 9), at(2, 9, 4)),)),
        "new-session": replace(calendar, sessions=(*calendar.sessions, TradingSession(at(3, 9), at(3, 10)))),
        "coverage-start": replace(calendar, coverage_start=at(1) - timedelta(days=1)),
        "coverage-end": replace(calendar, coverage_end=at(6) + timedelta(days=1)),
    }
    candles = (bar(minute_candles[0], at(2, 9, 1)),)
    original = normalize_candles(candles, config=NormalizationConfig(calendar=calendar)).datasets[0]
    different = normalize_candles(candles, config=NormalizationConfig(calendar=modifications[changed])).datasets[0]
    assert original.candles == different.candles
    assert original.dataset_id != different.dataset_id


@pytest.mark.case("DAT-009")
def test_dat_009_calendar_identity_is_repeatable_and_metadata_immutable(minute_candles):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 3)))
    candles = (bar(minute_candles[0], at(2, 9, 1)),)
    first = normalize_candles(candles, config=NormalizationConfig(calendar=calendar)).datasets[0]
    equivalent = replace(calendar, sessions=tuple(replace(s) for s in calendar.sessions))
    repeated = normalize_candles(candles, config=NormalizationConfig(calendar=equivalent)).datasets[0]
    continuous = normalize_candles(candles, config=NormalizationConfig(calendar="continuous")).datasets[0]
    assert first.dataset_id == repeated.dataset_id
    assert first.dataset_id != continuous.dataset_id
    with pytest.raises(FrozenInstanceError):
        first.config.calendar.sessions = ()
    with pytest.raises(FrozenInstanceError):
        first.config.calendar.sessions[0].open_time = at(2, 8)


@pytest.mark.case("DAT-009")
def test_dat_009_existing_continuous_identity_stays_compatible(minute_candles):
    dataset = normalize_candles(minute_candles[:1], config=NormalizationConfig(calendar="continuous")).datasets[0]
    assert dataset.dataset_id == "sha256:84f63adf46991b6b8e2d3969b95744ae72ca9ba5fc887fd882cb247aba4987ff"


@pytest.mark.case("DAT-010")
def test_dat_010_calendar_gaps_are_isolated_between_instruments_and_intervals(minute_candles):
    calendar = schedule(TradingSession(at(2, 9), at(2, 9, 4)))
    template = minute_candles[0]
    source = (
        bar(template, at(2, 9)), bar(template, at(2, 9, 2)),
        bar(template, at(2, 9), interval=2 * MINUTE),
        bar(template, at(2, 9, 2), interval=2 * MINUTE),
        *(bar(template, at(2, 9, minute), instrument="OTHER") for minute in range(4)),
    )
    result = normalize_candles(source, config=NormalizationConfig(calendar=calendar))
    assert len(result.datasets) == 3
    assert result.select(instrument="TEST", interval=MINUTE).quality.gaps == (CandleGap(at(2, 9, 1), at(2, 9, 2), 1),)
    assert result.select(instrument="TEST", interval=2 * MINUTE).quality.gaps == ()
    assert result.select(instrument="OTHER", interval=MINUTE).quality.gaps == ()
