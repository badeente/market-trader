"""Requested UTC ranges have explicit boundary and completely absent-series checks."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.coverage import DatasetRequest
from market_trader.domain import CandleValidationError
from market_trader.normalization import (
    CandleGap, DataGapError, NormalizationConfig, normalize_candles,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
REQUEST = DatasetRequest("TEST", MINUTE, START, START + 5 * MINUTE)
CONTINUOUS = NormalizationConfig(calendar="continuous")
SESSIONS = SessionCalendar(
    "synthetic-v1", START, START + timedelta(days=1),
    (TradingSession(START, START + 2 * MINUTE),
     TradingSession(START + 4 * MINUTE, START + 6 * MINUTE)),
)


@pytest.mark.case("DAT-005")
def test_dat_005_request_reports_leading_and_trailing_missing_candles(minute_candles):
    source = [minute_candles[3], minute_candles[2], minute_candles[3]]
    dataset = normalize_candles(source, config=CONTINUOUS, requests=[REQUEST]).datasets[0]
    assert dataset.candles == minute_candles[2:4]
    assert dataset.request == REQUEST
    assert dataset.quality.expected_count == 5
    assert dataset.quality.input_count == 3
    assert dataset.quality.output_count == 2
    assert dataset.quality.source_rows == (2, 1)
    assert dataset.quality.gaps == (
        CandleGap(START, START + 2 * MINUTE, 2),
        CandleGap(START + 4 * MINUTE, START + 5 * MINUTE, 1),
    )
    assert source == [minute_candles[3], minute_candles[2], minute_candles[3]]


@pytest.mark.case("DAT-005")
def test_dat_005_request_includes_internal_gaps_once(minute_candles):
    dataset = normalize_candles(minute_candles[::2], config=CONTINUOUS, requests=(REQUEST,)).datasets[0]
    assert dataset.quality.gaps == (
        CandleGap(START + MINUTE, START + 2 * MINUTE, 1),
        CandleGap(START + 3 * MINUTE, START + 4 * MINUTE, 1),
    )
    assert dataset.quality.output_count + sum(g.missing_count for g in dataset.quality.gaps) == 5


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize(("indices", "row_number", "missing"), [
    ((2, 3, 4), 1, CandleGap(START, START + 2 * MINUTE, 2)),
    ((0, 1, 2), None, CandleGap(START + 3 * MINUTE, START + 5 * MINUTE, 2)),
    ((), None, CandleGap(START, START + 5 * MINUTE, 5)),
    ((4, 2, 0), 2, CandleGap(START + MINUTE, START + 2 * MINUTE, 1)),
], ids=["head", "tail", "empty", "internal"])
def test_dat_005_strict_request_rejects_missing_data_with_honest_source_location(
    minute_candles, indices, row_number, missing,
):
    config = replace(CONTINUOUS, gap_policy="reject")
    with pytest.raises(DataGapError) as raised:
        normalize_candles(tuple(minute_candles[i] for i in indices), config=config, requests=(REQUEST,))
    assert raised.value.row_number == row_number
    assert raised.value.field == "open_time"
    assert raised.value.dataset.request == REQUEST
    assert raised.value.dataset.quality.gaps[0] == missing
    assert raised.value.dataset.quality.expected_count == 5


@pytest.mark.case("DAT-005")
def test_dat_005_full_request_accepts_exact_half_open_boundaries(minute_candles):
    dataset = normalize_candles(minute_candles, config=replace(CONTINUOUS, gap_policy="reject"), requests=(REQUEST,)).datasets[0]
    assert dataset.candles[0].open_time == REQUEST.start_time
    assert dataset.candles[-1].close_time == REQUEST.end_time
    assert dataset.quality.gaps == ()
    assert dataset.quality.expected_count == 5


@pytest.mark.case("DAT-005")
def test_dat_005_continuous_request_sets_grid_even_when_first_candle_is_missing(minute_candles):
    original = minute_candles[0]
    shifted = replace(original, open_time=original.open_time + timedelta(seconds=30),
                      close_time=original.close_time + timedelta(seconds=30),
                      available_at=original.available_at + timedelta(seconds=30))
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((shifted,), config=CONTINUOUS, requests=(REQUEST,))
    assert raised.value.row_number == 1
    assert raised.value.field == "open_time"


@pytest.mark.case("DAT-005")
def test_dat_005_microsecond_intervals_are_counted_without_rounding(minute_candles):
    tick = timedelta(microseconds=1)
    request = replace(REQUEST, interval=tick, end_time=START + 3 * tick)
    candle = replace(minute_candles[0], interval=tick, open_time=START + tick,
                     close_time=START + 2 * tick, available_at=START + 2 * tick)
    dataset = normalize_candles((candle,), config=CONTINUOUS, requests=(request,)).datasets[0]
    assert dataset.quality.expected_count == 3
    assert dataset.quality.gaps == (
        CandleGap(START, START + tick, 1), CandleGap(START + 2 * tick, START + 3 * tick, 1),
    )


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize("invalid", [
    "empty-requests", "wrong-item", "duplicate-series", "different-ranges-same-series",
    "empty-instrument", "padded-instrument", "wrong-instrument-type", "zero-interval",
    "negative-interval", "wrong-interval-type", "naive-start", "naive-end",
    "non-utc-start", "non-utc-end", "wrong-time-type", "zero-range", "reversed-range", "partial-interval",
])
def test_dat_005_invalid_requests_fail_before_empty_data_can_hide_them(invalid):
    offset = timezone(timedelta(hours=1))
    changes = {
        "empty-instrument": {"instrument": ""}, "padded-instrument": {"instrument": " TEST "},
        "wrong-instrument-type": {"instrument": 1}, "zero-interval": {"interval": timedelta(0)},
        "negative-interval": {"interval": -MINUTE}, "wrong-interval-type": {"interval": True},
        "naive-start": {"start_time": START.replace(tzinfo=None)},
        "naive-end": {"end_time": REQUEST.end_time.replace(tzinfo=None)},
        "non-utc-start": {"start_time": START.astimezone(offset)},
        "non-utc-end": {"end_time": REQUEST.end_time.astimezone(offset)},
        "wrong-time-type": {"start_time": "2026-01-01"},
        "zero-range": {"end_time": START}, "reversed-range": {"end_time": START - MINUTE},
        "partial-interval": {"end_time": START + timedelta(seconds=90)},
    }
    if invalid == "empty-requests":
        requests = ()
    elif invalid == "wrong-item":
        requests = (None,)
    elif invalid == "duplicate-series":
        requests = (REQUEST, REQUEST)
    elif invalid == "different-ranges-same-series":
        requests = (REQUEST, replace(REQUEST, end_time=START + 4 * MINUTE))
    else:
        requests = (replace(REQUEST, **changes[invalid]),)
    with pytest.raises(ValueError) as raised:
        normalize_candles((), config=CONTINUOUS, requests=requests)
    assert not isinstance(raised.value, CandleValidationError)


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize(("start_time", "end_time"), [
    (START - MINUTE, START), (REQUEST.end_time, REQUEST.end_time + MINUTE),
    (REQUEST.end_time - timedelta(seconds=30), REQUEST.end_time + timedelta(seconds=30)),
])
def test_dat_005_rows_outside_request_are_rejected_not_discarded(minute_candles, start_time, end_time):
    candle = replace(minute_candles[0], open_time=start_time, close_time=end_time, available_at=end_time)
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((minute_candles[1], candle), config=CONTINUOUS, requests=(REQUEST,))
    assert raised.value.row_number == 2
    assert raised.value.field == ("open_time" if start_time < START or start_time >= REQUEST.end_time else "close_time")


@pytest.mark.case("DAT-006")
def test_dat_006_requested_boundary_gaps_exclude_closed_periods(minute_candles):
    request = replace(REQUEST, end_time=START + 6 * MINUTE)
    dataset = normalize_candles((minute_candles[1], minute_candles[4]), config=NormalizationConfig(calendar=SESSIONS), requests=(request,)).datasets[0]
    assert dataset.quality.expected_count == 4
    assert dataset.quality.gaps == (
        CandleGap(START, START + MINUTE, 1),
        CandleGap(START + 5 * MINUTE, START + 6 * MINUTE, 1),
    )


@pytest.mark.case("DAT-006")
def test_dat_006_fully_missing_session_series_reports_only_active_spans():
    request = replace(REQUEST, end_time=START + 6 * MINUTE)
    dataset = normalize_candles((), config=NormalizationConfig(calendar=SESSIONS), requests=(request,)).datasets[0]
    assert dataset.candles == ()
    assert dataset.quality.expected_count == 4
    assert dataset.quality.gaps == (
        CandleGap(START, START + 2 * MINUTE, 2),
        CandleGap(START + 4 * MINUTE, START + 6 * MINUTE, 2),
    )


@pytest.mark.case("DAT-006")
def test_dat_006_request_can_clip_a_session_at_valid_candle_boundaries(minute_candles):
    request = replace(REQUEST, start_time=START + MINUTE)
    dataset = normalize_candles((minute_candles[1], minute_candles[4]), config=NormalizationConfig(calendar=SESSIONS, gap_policy="reject"), requests=(request,)).datasets[0]
    assert dataset.quality.expected_count == 2
    assert dataset.quality.gaps == ()


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize(("start", "end"), [(2 * MINUTE, 4 * MINUTE), (timedelta(seconds=150), timedelta(seconds=210))])
def test_dat_006_closed_only_request_is_complete_with_zero_expected_rows(start, end):
    request = replace(REQUEST, start_time=START + start, end_time=START + end)
    dataset = normalize_candles((), config=NormalizationConfig(calendar=SESSIONS, gap_policy="reject"), requests=(request,)).datasets[0]
    assert dataset.quality.expected_count == 0
    assert dataset.quality.gaps == ()
    assert dataset.candles == ()


@pytest.mark.case("DAT-006")
@pytest.mark.parametrize(("start", "end"), [
    (START - MINUTE, START + MINUTE), (START, SESSIONS.coverage_end + MINUTE),
    (START + timedelta(seconds=30), START + MINUTE),
    (START, START + timedelta(seconds=90)),
])
def test_dat_006_request_requires_known_calendar_coverage_and_aligned_active_edges(start, end):
    request = replace(REQUEST, start_time=start, end_time=end)
    with pytest.raises(ValueError) as raised:
        normalize_candles((), config=NormalizationConfig(calendar=SESSIONS), requests=(request,))
    assert not isinstance(raised.value, CandleValidationError)


@pytest.mark.case("DAT-006")
def test_dat_006_requested_session_grids_can_restart_after_closed_time(minute_candles):
    calendar = replace(SESSIONS, sessions=(
        TradingSession(START, START + 4 * MINUTE),
        TradingSession(START + 5 * MINUTE, START + 9 * MINUTE),
    ))
    request = replace(REQUEST, interval=2 * MINUTE, start_time=START + 4 * MINUTE, end_time=START + 9 * MINUTE)
    candles = tuple(replace(minute_candles[0], interval=2 * MINUTE, open_time=START + i * MINUTE,
                            close_time=START + (i + 2) * MINUTE, available_at=START + (i + 2) * MINUTE) for i in (5, 7))
    dataset = normalize_candles(candles, config=NormalizationConfig(calendar=calendar, gap_policy="reject"), requests=(request,)).datasets[0]
    assert dataset.quality.expected_count == 2
    assert dataset.quality.gaps == ()


@pytest.mark.case("DAT-010")
def test_dat_010_missing_instrument_is_selectable_and_not_hidden_by_another_series(minute_candles):
    absent = replace(REQUEST, instrument="ABSENT")
    result = normalize_candles(minute_candles, config=CONTINUOUS, requests=(REQUEST, absent))
    assert [(d.instrument, d.interval) for d in result.datasets] == [("ABSENT", MINUTE), ("TEST", MINUTE)]
    empty = result.select(instrument="ABSENT", interval=MINUTE)
    assert empty.candles == ()
    assert empty.request == absent
    assert empty.quality.input_count == empty.quality.output_count == 0
    assert empty.quality.source_rows == empty.quality.duplicates == ()
    assert empty.quality.was_reordered is False
    assert empty.quality.expected_count == 5
    assert empty.quality.gaps == (CandleGap(START, START + 5 * MINUTE, 5),)
    assert result.select(instrument="TEST", interval=MINUTE).quality.gaps == ()


@pytest.mark.case("DAT-010")
def test_dat_010_missing_interval_uses_its_own_range_and_expected_count(minute_candles):
    two_minutes = replace(REQUEST, interval=2 * MINUTE, end_time=START + 4 * MINUTE)
    result = normalize_candles(minute_candles, config=CONTINUOUS, requests=(two_minutes, REQUEST))
    assert result.select(instrument="TEST", interval=2 * MINUTE).quality.expected_count == 2
    assert result.select(instrument="TEST", interval=2 * MINUTE).quality.gaps == (CandleGap(START, START + 4 * MINUTE, 2),)
    assert result.select(instrument="TEST", interval=MINUTE).quality.expected_count == 5


@pytest.mark.case("DAT-010")
@pytest.mark.parametrize(("instrument", "interval"), [("OTHER", MINUTE), ("TEST", 5 * MINUTE)])
def test_dat_010_unrequested_input_series_are_explicit_errors(minute_candles, instrument, interval):
    original = minute_candles[0]
    extra = replace(original, instrument=instrument, interval=interval,
                    close_time=START + interval, available_at=START + interval)
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((original, extra), config=CONTINUOUS, requests=(REQUEST,))
    assert raised.value.row_number == 2
    assert raised.value.field == "series"


@pytest.mark.case("DAT-009")
def test_dat_009_request_identity_is_repeatable_and_independent_of_request_order(minute_candles):
    absent = replace(REQUEST, instrument="ABSENT")
    first = normalize_candles(minute_candles, config=CONTINUOUS, requests=(REQUEST, absent))
    second = normalize_candles((*reversed(minute_candles), minute_candles[0]), config=CONTINUOUS, requests=(absent, REQUEST))
    assert [d.dataset_id for d in first.datasets] == [d.dataset_id for d in second.datasets]


@pytest.mark.case("DAT-009")
def test_dat_009_identical_content_with_different_completeness_requirements_is_distinguishable(minute_candles):
    observed = normalize_candles(minute_candles, config=CONTINUOUS).datasets[0]
    exact = normalize_candles(minute_candles, config=CONTINUOUS, requests=(REQUEST,)).datasets[0]
    longer = normalize_candles(minute_candles, config=CONTINUOUS, requests=(replace(REQUEST, end_time=START + 6 * MINUTE),)).datasets[0]
    earlier = normalize_candles(minute_candles, config=CONTINUOUS, requests=(replace(REQUEST, start_time=START - MINUTE),)).datasets[0]
    assert len({d.dataset_id for d in (observed, exact, longer, earlier)}) == 4
    assert observed.request is None
    assert observed.quality.expected_count is None
    assert exact.quality.expected_count == 5
    assert longer.quality.expected_count == earlier.quality.expected_count == 6


@pytest.mark.case("DAT-009")
def test_dat_009_empty_series_still_has_a_content_identity_and_immutable_request():
    requests = [REQUEST]
    first = normalize_candles((), config=CONTINUOUS, requests=requests).datasets[0]
    requests.clear()
    assert first.request == REQUEST
    with pytest.raises(FrozenInstanceError):
        first.request.end_time = START
    other = normalize_candles((), config=CONTINUOUS, requests=(replace(REQUEST, instrument="OTHER"),)).datasets[0]
    different_range = normalize_candles((), config=CONTINUOUS, requests=(replace(REQUEST, end_time=START + 6 * MINUTE),)).datasets[0]
    strict = replace(CONTINUOUS, gap_policy="reject")
    with pytest.raises(DataGapError) as raised:
        normalize_candles((), config=strict, requests=(REQUEST,))
    assert len({first.dataset_id, other.dataset_id, different_range.dataset_id, raised.value.dataset.dataset_id}) == 4
