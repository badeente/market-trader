"""Dependency-free data types for the first test-first contracts."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Candle:
    """One completed candle; validation belongs to the data boundary."""

    instrument: str
    interval: timedelta
    open_time: datetime
    close_time: datetime
    available_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class CandleValidationError(ValueError):
    """Data error: one-based row, zero for CSV headers, or None without a source row."""

    def __init__(self, *, row_number: int | None, field: str, message: str) -> None:
        self.row_number = row_number
        self.field = field
        location = "dataset" if row_number is None else f"row {row_number}"
        super().__init__(f"{location}, {field}: {message}")
