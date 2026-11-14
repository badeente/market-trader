"""Boundary cases for the five data functions, specified before their bodies."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_DOWN, localcontext
from pathlib import Path

import pytest

from market_trader.data import (
    read_candles,
    resample_candles,
    rolling_windows,
    validate_candles,
)
from market_trader.domain import Candle, CandleValidationError
from market_trader.representations import mean_close_features

FIXTURE = Path(__file__).parent / "fixtures" / "fractional_volume.csv"


def csv_input(
    tmp_path: Path, changes: dict[str, str], *, copies: int = 1, omit: str = ""
) -> Path:
    with FIXTURE.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = [name for name in reader.fieldnames if name != omit]
        original = next(reader)
    rows = [original.copy() for _ in range(copies)]
    rows[-1].update(changes)
    path = tmp_path / "candles.csv"
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.case("DAT-002")
def test_dat_002_explicit_offsets_are_normalized_to_utc(tmp_path: Path) -> None:
    path = csv_input(
        tmp_path,
        {
            "open_time": "2026-01-01T01:00:00+01:00",
            "close_time": "2026-01-01T01:01:00+01:00",
            "available_at": "2026-01-01T01:01:00+01:00",
        },
    )
    candle, = read_candles(path)
    assert candle.open_time == datetime(2026, 1, 1, tzinfo=UTC)
    assert candle.close_time == datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    assert candle.available_at == candle.close_time
    assert all(
        time.tzinfo is UTC
        for time in (candle.open_time, candle.close_time, candle.available_at)
    )


@pytest.mark.case("DAT-002")
@pytest.mark.parametrize("field", ["open_time", "close_time", "available_at"])
@pytest.mark.parametrize("value", ["2026-01-01T00:01:00", "not-a-timestamp"])
def test_dat_002_naive_or_invalid_timestamps_are_rejected(
    tmp_path: Path, field: str, value: str
) -> None:
    with pytest.raises(CandleValidationError) as error:
        read_candles(csv_input(tmp_path, {field: value}))
    assert error.value.row_number == 1
    assert error.value.field == field


@pytest.mark.case("DAT-003")
def test_dat_003_csv_error_identifies_the_actual_data_row(tmp_path: Path) -> None:
    with pytest.raises(CandleValidationError) as error:
        read_candles(csv_input(tmp_path, {"close": "broken"}, copies=2))
    assert error.value.row_number == 2
    assert error.value.field == "close"


@pytest.mark.case("DAT-003")
def test_dat_003_missing_header_column_is_a_schema_error(tmp_path: Path) -> None:
    with pytest.raises(CandleValidationError) as error:
        read_candles(csv_input(tmp_path, {}, omit="close"))
    assert error.value.row_number == 0
    assert error.value.field == "close"


@pytest.mark.case("DAT-003")
@pytest.mark.parametrize("problem", ["duplicate_header", "extra_value", "empty_file"])
def test_dat_003_malformed_csv_does_not_silently_drop_data(
    tmp_path: Path, problem: str
) -> None:
    path = tmp_path / "malformed.csv"
    lines = FIXTURE.read_text().splitlines()
    if problem == "duplicate_header":
        path.write_text(lines[0] + ",close\n" + lines[1] + ",999\n")
        row_number, field = 0, "close"
    elif problem == "extra_value":
        path.write_text(lines[0] + "\n" + lines[1] + ",999\n")
        row_number, field = 1, "row"
    else:
        path.write_text("")
        row_number, field = 0, "header"
    with pytest.raises(CandleValidationError) as error:
        read_candles(path)
    assert (error.value.row_number, error.value.field) == (row_number, field)


@pytest.mark.case("DAT-003")
def test_dat_003_invalid_header_quoting_is_reported_as_a_header_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_header.csv"
    path.write_text('"unterminated header\n')
    with pytest.raises(CandleValidationError) as error:
        read_candles(path)
    assert (error.value.row_number, error.value.field) == (0, "header")


@pytest.mark.case("CAU-004")
def test_cau_004_missing_file_remains_a_technical_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_candles(tmp_path / "does_not_exist.csv")


@pytest.mark.case("DAT-003")
def test_dat_003_valid_candles_are_accepted(minute_candles: tuple[Candle, ...]) -> None:
    assert validate_candles(minute_candles) is None


@pytest.mark.case("DAT-003")
def test_dat_003_negative_prices_and_zero_volume_are_not_implicitly_forbidden(
    minute_candles: tuple[Candle, ...],
) -> None:
    candle = replace(
        minute_candles[0], open=Decimal("-100"), high=Decimal("-99"),
        low=Decimal("-102"), close=Decimal("-101"), volume=Decimal("0"),
    )
    assert validate_candles((candle,)) is None


@pytest.mark.case("DAT-003")
@pytest.mark.parametrize(
    "changes,field",
    [
        ({"interval": timedelta(0)}, "interval"),
        ({"close_time": datetime(2026, 1, 1, 0, 1, 1, tzinfo=UTC)}, "close_time"),
        ({"available_at": datetime(2026, 1, 1, tzinfo=UTC)}, "available_at"),
        ({"instrument": ""}, "instrument"),
    ],
)
def test_dat_003_invalid_candle_metadata_is_rejected(
    minute_candles: tuple[Candle, ...], changes: dict, field: str
) -> None:
    with pytest.raises(CandleValidationError) as error:
        validate_candles((replace(minute_candles[0], **changes),))
    assert error.value.row_number == 1
    assert error.value.field == field


@pytest.mark.case("DAT-007")
def test_dat_007_boundary_candle_belongs_to_the_next_aggregate(
    minute_candles: tuple[Candle, ...],
) -> None:
    following = tuple(
        replace(
            candle, open_time=candle.open_time + timedelta(minutes=5),
            close_time=candle.close_time + timedelta(minutes=5),
            available_at=candle.available_at + timedelta(minutes=5),
            open=candle.open + 10, high=candle.high + 10,
            low=candle.low + 10, close=candle.close + 10,
        )
        for candle in minute_candles
    )
    result = resample_candles(minute_candles + following, interval=timedelta(minutes=5))
    assert [(c.open, c.high, c.low, c.close, c.volume) for c in result] == [
        tuple(map(Decimal, ("100", "105", "99", "104.5", "7"))),
        tuple(map(Decimal, ("110", "115", "109", "114.5", "7"))),
    ]
    assert [c.open_time.minute for c in result] == [0, 5]
    assert [c.close_time.minute for c in result] == [5, 10]


@pytest.mark.case("DAT-008")
@pytest.mark.parametrize("indices", [(0, 1, 2, 3), (1, 2, 3, 4), (0, 1, 3, 4)])
def test_dat_008_incomplete_or_gapped_aggregates_fail_explicitly(
    minute_candles: tuple[Candle, ...], indices: tuple[int, ...]
) -> None:
    with pytest.raises(CandleValidationError) as error:
        resample_candles(
            tuple(minute_candles[index] for index in indices),
            interval=timedelta(minutes=5),
        )
    assert error.value.field == "open_time"


@pytest.mark.case("DAT-007")
@pytest.mark.parametrize("seconds", [0, 30, 90])
def test_dat_007_resampling_requires_a_positive_multiple_of_source_interval(
    minute_candles: tuple[Candle, ...], seconds: int
) -> None:
    with pytest.raises(ValueError):
        resample_candles(minute_candles, interval=timedelta(seconds=seconds))


@pytest.mark.case("DAT-007")
def test_dat_007_aggregation_preserves_small_volume_beside_large_volume(
    minute_candles: tuple[Candle, ...],
) -> None:
    candles = tuple(replace(candle, volume=Decimal("0")) for candle in minute_candles)
    candles = (
        replace(candles[0], volume=Decimal("1000000000000000000000000000000")),
        *candles[1:-1], replace(candles[-1], volume=Decimal("0.125")),
    )
    result, = resample_candles(candles, interval=timedelta(minutes=5))
    assert result.volume == Decimal("1000000000000000000000000000000.125")


@pytest.mark.case("CAU-001")
@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_cau_001_invalid_window_sizes_are_rejected(
    minute_candles: tuple[Candle, ...], size: object
) -> None:
    with pytest.raises(ValueError):
        list(rolling_windows(minute_candles, size=size))


@pytest.mark.case("CAU-001")
@pytest.mark.parametrize(
    "kind,field",
    [("instrument", "instrument"), ("interval", "interval"),
     ("reversed", "open_time"), ("duplicate", "open_time")],
)
def test_cau_001_windows_require_a_single_ordered_history(
    minute_candles: tuple[Candle, ...], kind: str, field: str
) -> None:
    first, second = minute_candles[:2]
    if kind == "instrument":
        history = (first, replace(second, instrument="OTHER"))
    elif kind == "interval":
        history = (first, replace(
            second, interval=timedelta(minutes=2),
            close_time=second.close_time + timedelta(minutes=1),
            available_at=second.available_at + timedelta(minutes=1),
        ))
    elif kind == "reversed":
        history = (second, first)
    else:
        history = (first, first)
    with pytest.raises(CandleValidationError) as error:
        list(rolling_windows(history, size=2))
    assert error.value.field == field


@pytest.mark.case("CAU-003")
@pytest.mark.parametrize("lookback", [0, -1, True, 1.5])
def test_cau_003_invalid_feature_lookbacks_are_rejected(
    minute_candles: tuple[Candle, ...], lookback: object
) -> None:
    with pytest.raises(ValueError):
        mean_close_features(minute_candles, lookback=lookback)


@pytest.mark.case("CAU-005")
def test_cau_005_window_iterator_snapshots_its_input(
    minute_candles: tuple[Candle, ...],
) -> None:
    mutable = list(minute_candles)
    windows = rolling_windows(mutable, size=3)
    mutable[0] = replace(mutable[0], close=Decimal("99.5"))
    assert next(windows) == minute_candles[:3]


@pytest.mark.case("CAU-002")
def test_cau_002_feature_availability_includes_delayed_source_candles(
    minute_candles: tuple[Candle, ...],
) -> None:
    delayed = replace(
        minute_candles[0], available_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
    )
    feature, = mean_close_features((delayed, minute_candles[1]), lookback=2)
    assert feature.close_time == datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
    assert feature.available_at == datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
    assert feature.value == Decimal("101")


@pytest.mark.case("CAU-003")
def test_cau_003_features_do_not_depend_on_callers_decimal_context(
    minute_candles: tuple[Candle, ...],
) -> None:
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_DOWN
        features = mean_close_features(minute_candles, lookback=2)
    assert [feature.value for feature in features] == list(map(Decimal, ("101", "102", "103", "104")))
