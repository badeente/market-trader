"""Hand-specified normalization expectations, independent of implementation helpers."""

from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from decimal import Decimal, localcontext

import pytest

from market_trader.data import read_candles, resample_candles, rolling_windows
from market_trader.domain import CandleValidationError
from market_trader.normalization import (
    CandleGap, CandleKey, DataGapError, DuplicateConflictError, DuplicateRemoval,
    NormalizationConfig, normalize_candles,
)

CONTINUOUS = NormalizationConfig(calendar="continuous", gap_policy="report")
MINUTE = timedelta(minutes=1)


def shifted(candle, delta):
    return replace(
        candle, open_time=candle.open_time + delta,
        close_time=candle.close_time + delta,
        available_at=candle.available_at + delta,
    )


@pytest.mark.case("DAT-004")
def test_dat_004_identical_duplicates_are_audited_with_original_rows(minute_candles):
    a, b = minute_candles[:2]
    source = [b, a, replace(b), replace(a), replace(b)]
    dataset = normalize_candles(source, config=CONTINUOUS).datasets[0]

    assert dataset.candles == (a, b)
    assert dataset.quality.input_count == 5
    assert dataset.quality.output_count == 2
    assert dataset.quality.source_rows == (2, 1)
    assert dataset.quality.was_reordered is True
    assert dataset.quality.duplicates == (
        DuplicateRemoval(CandleKey("TEST", MINUTE, b.open_time), 1, 3),
        DuplicateRemoval(CandleKey("TEST", MINUTE, a.open_time), 2, 4),
        DuplicateRemoval(CandleKey("TEST", MINUTE, b.open_time), 1, 5),
    )
    assert source == [b, a, b, a, b]


@pytest.mark.case("DAT-004")
@pytest.mark.parametrize(("field", "value"), [
    ("open", Decimal("100.25")), ("high", Decimal("102")),
    ("low", Decimal("98")), ("close", Decimal("100.75")),
    ("volume", Decimal("2")), ("available_at", "delayed"),
])
def test_dat_004_conflicts_reject_prices_volume_and_availability(minute_candles, field, value):
    first = minute_candles[0]
    if value == "delayed":
        value = first.available_at + timedelta(seconds=1)
    source = (minute_candles[1], first, replace(first), replace(first, **{field: value}))
    with pytest.raises(DuplicateConflictError) as raised:
        normalize_candles(source, config=CONTINUOUS)

    error = raised.value
    assert isinstance(error, CandleValidationError)
    assert error.key == CandleKey("TEST", MINUTE, first.open_time)
    assert error.first_row == 2
    assert error.row_number == 4
    assert error.field == "duplicate"
    assert error.differing_fields == (field,)


@pytest.mark.case("DAT-004")
def test_dat_004_all_conflicting_fields_are_reported(minute_candles):
    first = minute_candles[0]
    conflict = replace(first, high=Decimal("102"), volume=Decimal("9"))
    with pytest.raises(DuplicateConflictError) as raised:
        normalize_candles((first, conflict), config=CONTINUOUS)
    assert raised.value.differing_fields == ("high", "volume")


@pytest.mark.case("DAT-005")
def test_dat_005_sorting_reports_each_gap_without_filling(minute_candles):
    a, _, _, d, e = minute_candles
    h = shifted(e, timedelta(minutes=3))
    dataset = normalize_candles((h, e, a, d), config=CONTINUOUS).datasets[0]
    assert dataset.candles == (a, d, e, h)
    assert dataset.quality.source_rows == (3, 4, 2, 1)
    assert dataset.quality.was_reordered is True
    assert dataset.quality.gaps == (
        CandleGap(a.close_time, d.open_time, 2),
        CandleGap(e.close_time, h.open_time, 2),
    )
    assert dataset.quality.input_count == dataset.quality.output_count == 4
    assert dataset.config == CONTINUOUS


@pytest.mark.case("DAT-005")
def test_dat_005_strict_policy_rejects_gaps_and_retains_diagnostics(minute_candles):
    a, _, _, d, _ = minute_candles
    with pytest.raises(DataGapError) as raised:
        normalize_candles((d, a), config=replace(CONTINUOUS, gap_policy="reject"))
    error = raised.value
    assert error.row_number == 1  # Original row after the gap, before sorting.
    assert error.field == "open_time"
    assert error.dataset.candles == (a, d)
    assert error.dataset.quality.gaps == (CandleGap(a.close_time, d.open_time, 2),)
    assert error.dataset.config.gap_policy == "reject"


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize("length", [0, 1, 5])
def test_dat_005_empty_single_and_contiguous_histories_have_no_invented_boundary_gaps(
    minute_candles, length,
):
    result = normalize_candles(minute_candles[:length], config=CONTINUOUS)
    if length == 0:
        assert result.datasets == ()
    else:
        dataset = result.datasets[0]
        assert dataset.quality.gaps == ()
        assert dataset.quality.was_reordered is False
        assert dataset.quality.source_rows == tuple(range(1, length + 1))


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize("seconds", [30, 90])
def test_dat_005_overlaps_and_off_grid_starts_are_rejected(minute_candles, seconds):
    first = minute_candles[0]
    later = shifted(first, timedelta(seconds=seconds))
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles((later, first), config=CONTINUOUS)
    assert raised.value.row_number == 1
    assert raised.value.field == "open_time"


@pytest.mark.case("DAT-005")
def test_dat_005_timeline_is_anchored_to_first_candle_not_a_guessed_exchange_phase(minute_candles):
    source = tuple(shifted(c, timedelta(seconds=17)) for c in minute_candles)
    assert normalize_candles(source, config=CONTINUOUS).datasets[0].quality.gaps == ()


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize("config", [
    NormalizationConfig(calendar="sessions"),
    NormalizationConfig(calendar="continuous", gap_policy="fill"),
    None,
])
def test_dat_005_unknown_calendar_or_policy_is_rejected_even_for_empty_input(config):
    with pytest.raises(ValueError):
        normalize_candles((), config=config)


@pytest.mark.case("DAT-003")
def test_dat_003_normalization_validates_before_sorting_and_preserves_error_row(minute_candles):
    source = (minute_candles[4], minute_candles[0], replace(minute_candles[1], volume=Decimal("NaN")))
    with pytest.raises(CandleValidationError) as raised:
        normalize_candles(source, config=CONTINUOUS)
    assert raised.value.row_number == 3
    assert raised.value.field == "volume"


@pytest.mark.case("DAT-009")
def test_dat_009_identity_depends_on_normalized_content_not_order_or_duplicates(minute_candles):
    original = normalize_candles(minute_candles, config=CONTINUOUS).datasets[0]
    repeated = normalize_candles(minute_candles, config=CONTINUOUS).datasets[0]
    reordered = normalize_candles((*reversed(minute_candles), minute_candles[0]), config=CONTINUOUS).datasets[0]
    assert original.dataset_id == repeated.dataset_id == reordered.dataset_id
    assert original.quality != reordered.quality
    algorithm, digest = original.dataset_id.split(":")
    assert algorithm == "sha256"
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize(("field", "value"), [
    ("open", Decimal("100.25")), ("high", Decimal("102")),
    ("low", Decimal("98")), ("close", Decimal("100.75")),
    ("volume", Decimal("2")), ("available_at", "delayed"),
    ("instrument", "OTHER"), ("open_time", "shifted"),
])
def test_dat_009_each_content_change_changes_identity(minute_candles, field, value):
    first = minute_candles[0]
    if value == "delayed":
        changed = replace(first, available_at=first.available_at + timedelta(microseconds=1))
    elif value == "shifted":
        changed = shifted(first, timedelta(microseconds=1))
    else:
        changed = replace(first, **{field: value})
    before = normalize_candles((first,), config=CONTINUOUS).datasets[0]
    after = normalize_candles((changed,), config=CONTINUOUS).datasets[0]
    assert before.dataset_id != after.dataset_id


@pytest.mark.case("DAT-009")
def test_dat_009_processing_policy_and_resampling_are_distinguishable(minute_candles):
    original = normalize_candles(minute_candles, config=CONTINUOUS).datasets[0]
    strict = normalize_candles(minute_candles, config=replace(CONTINUOUS, gap_policy="reject")).datasets[0]
    five_minutes = resample_candles(minute_candles, interval=timedelta(minutes=5))
    resampled = normalize_candles(five_minutes, config=CONTINUOUS).datasets[0]
    assert len({original.dataset_id, strict.dataset_id, resampled.dataset_id}) == 3


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("precision", [2, 50])
def test_dat_009_decimal_spelling_signed_zero_and_context_do_not_change_identity(minute_candles, precision):
    first = replace(minute_candles[0], volume=Decimal("0"))
    same = replace(first, open=Decimal("1.0000E+2"), close=Decimal("100.50000"), volume=Decimal("-0.000"))
    expected = normalize_candles((first,), config=CONTINUOUS).datasets[0].dataset_id
    with localcontext() as context:
        context.prec = precision
        result = normalize_candles((same, first), config=CONTINUOUS).datasets[0]
    assert result.dataset_id == expected
    assert result.quality.output_count == 1
    assert len(result.quality.duplicates) == 1


@pytest.mark.case("DAT-009")
def test_dat_009_small_changes_are_not_lost_by_rounding_or_float_conversion(minute_candles):
    first = replace(minute_candles[0], volume=Decimal("1000000000000000000000000000000"))
    changed = replace(first, volume=Decimal("1000000000000000000000000000000.000001"))
    with localcontext() as context:
        context.prec = 2
        before = normalize_candles((first,), config=CONTINUOUS).datasets[0]
        after = normalize_candles((changed,), config=CONTINUOUS).datasets[0]
    assert before.dataset_id != after.dataset_id


@pytest.mark.case("DAT-009")
def test_dat_009_raw_csv_is_unchanged_and_results_snapshot_input(tmp_path):
    path = tmp_path / "raw.csv"
    raw = (
        b"instrument,interval_seconds,open_time,close_time,available_at,open,high,low,close,volume\r\n"
        b"TEST,60,2026-01-01T00:01:00Z,2026-01-01T00:02:00Z,2026-01-01T00:02:00Z,101,102,100,101.5,2\r\n"
        b"TEST,60,2026-01-01T00:00:00Z,2026-01-01T00:01:00Z,2026-01-01T00:01:00Z,100,101,99,100.5,0.125\r\n"
    )
    path.write_bytes(raw)
    source = list(read_candles(path))
    snapshot = tuple(source)
    dataset = normalize_candles(source, config=CONTINUOUS).datasets[0]
    assert path.read_bytes() == raw
    assert source == list(snapshot)
    source.clear()
    assert dataset.candles == (snapshot[1], snapshot[0])
    assert dataset.candles[0].volume == Decimal("0.125")
    with pytest.raises(FrozenInstanceError):
        dataset.quality.output_count = 0
    with pytest.raises(FrozenInstanceError):
        dataset.candles = ()


@pytest.mark.case("DAT-010")
def test_dat_010_instruments_and_intervals_are_grouped_and_selected_separately(minute_candles):
    a, b = minute_candles[:2]
    other = tuple(replace(c, instrument="OTHER") for c in (a, b))
    five = replace(a, interval=timedelta(minutes=5), close_time=a.open_time + timedelta(minutes=5), available_at=a.open_time + timedelta(minutes=5))
    source = (b, other[1], five, a, other[0], replace(a))
    result = normalize_candles(source, config=CONTINUOUS)
    assert [(d.instrument, d.interval) for d in result.datasets] == [
        ("OTHER", MINUTE), ("TEST", MINUTE), ("TEST", timedelta(minutes=5)),
    ]
    selected = result.select(instrument="TEST", interval=MINUTE)
    assert selected.candles == (a, b)
    assert selected.quality.input_count == 3
    assert selected.quality.source_rows == (4, 1)
    assert selected.quality.duplicates == (DuplicateRemoval(CandleKey("TEST", MINUTE, a.open_time), 4, 6),)
    assert result.select(instrument="OTHER", interval=MINUTE).candles == other
    assert result.select(instrument="TEST", interval=timedelta(minutes=5)).candles == (five,)
    assert len({d.dataset_id for d in result.datasets}) == 3
    assert list(rolling_windows(selected.candles, size=2)) == [(a, b)]


@pytest.mark.case("DAT-010")
@pytest.mark.parametrize(("instrument", "interval"), [
    ("MISSING", MINUTE), ("TEST", timedelta(minutes=5)),
])
def test_dat_010_selection_never_falls_back_to_another_series(minute_candles, instrument, interval):
    result = normalize_candles(minute_candles, config=CONTINUOUS)
    with pytest.raises(KeyError):
        result.select(instrument=instrument, interval=interval)


@pytest.mark.case("DAT-010")
def test_dat_010_empty_result_has_no_implicit_default_series():
    result = normalize_candles((), config=CONTINUOUS)
    with pytest.raises(KeyError):
        result.select(instrument="TEST", interval=MINUTE)


@pytest.mark.case("DAT-010")
def test_dat_010_other_series_do_not_change_identity_or_gap_detection(minute_candles):
    a, b = minute_candles[:2]
    standalone = normalize_candles((a, b), config=CONTINUOUS).datasets[0]
    others = (replace(a, instrument="OTHER"), replace(minute_candles[4], instrument="OTHER"))
    result = normalize_candles((others[1], b, others[0], a), config=CONTINUOUS)
    selected = result.select(instrument="TEST", interval=MINUTE)
    assert selected.dataset_id == standalone.dataset_id
    assert selected.quality.gaps == ()
    assert result.select(instrument="OTHER", interval=MINUTE).quality.gaps == (
        CandleGap(a.close_time, minute_candles[4].open_time, 3),
    )
