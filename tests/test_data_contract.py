"""Contract tests for DAT-001, DAT-003 and DAT-007."""

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from market_trader.data import read_candles, resample_candles, validate_candles
from market_trader.domain import Candle, CandleValidationError

FIXTURE = Path(__file__).parent / "fixtures" / "fractional_volume.csv"


@pytest.mark.case("DAT-001")
def test_dat_001_csv_import_preserves_values_units_and_timestamps() -> None:
    candles = read_candles(FIXTURE)

    assert candles == (
        Candle(
            instrument="TEST",
            interval=timedelta(minutes=1),
            open_time=datetime(2026, 1, 1, tzinfo=UTC),
            close_time=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            available_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100.5"),
            volume=Decimal("0.125"),
        ),
    )


@pytest.mark.case("DAT-003")
def test_dat_003_missing_close_reports_source_row_and_field(tmp_path: Path) -> None:
    with FIXTURE.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        row = next(reader)
    assert fieldnames is not None
    row["close"] = ""
    invalid_csv = tmp_path / "missing_close.csv"
    with invalid_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(CandleValidationError) as error:
        read_candles(invalid_csv)

    assert error.value.row_number == 1
    assert error.value.field == "close"


INVALID_NUMBERS = [
    pytest.param({field: Decimal(value)}, field, id=f"{field}_{value}")
    for field in ("open", "high", "low", "close", "volume")
    for value in ("NaN", "Infinity", "-Infinity")
]
INVALID_NUMBERS += [
    pytest.param({"volume": Decimal("-0.125")}, "volume", id="negative_volume"),
    pytest.param(
        {"close": Decimal("99.5"), "high": Decimal("99.75")},
        "high",
        id="high_below_open",
    ),
    pytest.param({"high": Decimal("100.25")}, "high", id="high_below_close"),
    pytest.param({"low": Decimal("100.25")}, "low", id="low_above_open"),
    pytest.param(
        {"close": Decimal("99.5"), "low": Decimal("99.75")},
        "low",
        id="low_above_close",
    ),
]


@pytest.mark.case("DAT-003")
@pytest.mark.parametrize("changes,field", INVALID_NUMBERS)
def test_dat_003_invalid_numeric_values_are_rejected(
    minute_candles: tuple[Candle, ...], changes: dict[str, Decimal], field: str
) -> None:
    invalid = replace(minute_candles[0], **changes)

    with pytest.raises(CandleValidationError) as error:
        validate_candles((invalid,))

    assert error.value.row_number == 1
    assert error.value.field == field


@pytest.mark.case("DAT-003")
@pytest.mark.parametrize("changes,field", INVALID_NUMBERS)
def test_dat_003_csv_import_rejects_invalid_numbers(
    tmp_path: Path, changes: dict[str, Decimal], field: str
) -> None:
    with FIXTURE.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        row = next(reader)
    assert fieldnames is not None
    row.update({name: str(value) for name, value in changes.items()})
    invalid_csv = tmp_path / "invalid_numbers.csv"
    with invalid_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(CandleValidationError) as error:
        read_candles(invalid_csv)

    assert error.value.row_number == 1
    assert error.value.field == field


@pytest.mark.case("DAT-007")
def test_dat_007_resampling_has_hand_calculated_ohlcv_and_time_bounds(
    minute_candles: tuple[Candle, ...],
) -> None:
    candles = resample_candles(minute_candles, interval=timedelta(minutes=5))

    assert candles == (
        Candle(
            instrument="TEST",
            interval=timedelta(minutes=5),
            open_time=datetime(2026, 1, 1, tzinfo=UTC),
            close_time=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
            available_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("104.5"),
            volume=Decimal("7"),
        ),
    )
