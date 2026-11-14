"""Strict v1 JSON encoding using an explicit allowlist of dataset records."""

from dataclasses import fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
import json

from market_trader.artifacts import DatasetProvenance, DatasetStorageError, SourceFingerprint
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.coverage import DatasetRequest
from market_trader.domain import Candle
from market_trader.normalization import (
    CandleGap, CandleKey, DataQualityReport, DuplicateRemoval,
    NormalizationConfig, NormalizedDataset,
)
from market_trader.training_data import TrainingConfig, TrainingExample


def json_bytes(value: object, *, default=None) -> bytes:
    try:
        return json.dumps(value, default=default, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except UnicodeError as error:
        raise DatasetStorageError("artifact text must contain valid UTF-8 characters") from error


def _reject_number(value: str):
    raise DatasetStorageError(f"non-integer JSON number is unsupported: {value}")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise DatasetStorageError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def parse_json(text: str, *, allow_floats: bool = False) -> object:
    try:
        return json.loads(text, object_pairs_hook=_unique_keys,
                          parse_float=float if allow_floats else _reject_number,
                          parse_constant=_reject_number)
    except (ValueError, RecursionError) as error:
        raise DatasetStorageError(f"invalid artifact JSON: {error}") from error


def record_object(value: object, names) -> dict:
    if not isinstance(value, dict) or set(value) != set(names):
        raise DatasetStorageError(f"expected exactly the fields {tuple(names)}")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise DatasetStorageError("expected a string")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        raise DatasetStorageError("expected a whole integer, not a boolean or float")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise DatasetStorageError("expected a boolean")
    return value


def _timestamp(value: object) -> datetime:
    timestamp = datetime.fromisoformat(_text(value))
    if timestamp.utcoffset() != timedelta(0):
        raise DatasetStorageError("expected a timezone-aware UTC timestamp")
    return timestamp.astimezone(UTC)


def _duration(value: object) -> timedelta:
    return timedelta(microseconds=_integer(value))


def _decimal(value: object) -> Decimal:
    number = Decimal(_text(value))
    if not number.is_finite():
        raise DatasetStorageError("expected a finite decimal string")
    return number


def _sequence(value: object, reader) -> tuple:
    if not isinstance(value, list):
        raise DatasetStorageError("expected a JSON array")
    return tuple(reader(item) for item in value)


def decode_record(value: object, kind: type):
    schema = _SCHEMAS[kind]
    data = record_object(value, schema)
    try:
        return kind(**{name: reader(data[name]) for name, reader in schema.items()})
    except (ValueError, OverflowError, InvalidOperation, RecursionError) as error:
        raise DatasetStorageError(f"invalid {kind.__name__}: {error}") from error


def _calendar(value: object):
    return value if value == "continuous" else decode_record(value, SessionCalendar)


def encode(value: object) -> object:
    if type(value) in _SCHEMAS:
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if isinstance(value, datetime):
        if value.utcoffset() != timedelta(0):
            raise DatasetStorageError("expected a timezone-aware UTC timestamp")
        return value.astimezone(UTC).isoformat(timespec="microseconds")
    if isinstance(value, timedelta):
        return value // timedelta(microseconds=1)
    if isinstance(value, Decimal):
        return str(value)
    if value is None or type(value) in (str, int, bool):
        return value
    raise DatasetStorageError(f"unsupported artifact value: {type(value).__name__}")


_SCHEMAS = {
    Candle: {
        "instrument": _text, "interval": _duration,
        **{field: _timestamp for field in ("open_time", "close_time", "available_at")},
        **{field: _decimal for field in ("open", "high", "low", "close", "volume")},
    },
    TradingSession: {"open_time": _timestamp, "close_time": _timestamp},
    SessionCalendar: {
        "name": _text, "coverage_start": _timestamp, "coverage_end": _timestamp,
        "sessions": lambda value: _sequence(value, lambda item: decode_record(item, TradingSession)),
    },
    DatasetRequest: {"instrument": _text, "interval": _duration,
                     "start_time": _timestamp, "end_time": _timestamp},
    NormalizationConfig: {"calendar": _calendar, "gap_policy": _text},
    CandleKey: {"instrument": _text, "interval": _duration, "open_time": _timestamp},
    DuplicateRemoval: {"key": lambda value: decode_record(value, CandleKey),
                       "kept_row": _integer, "removed_row": _integer},
    CandleGap: {"start_time": _timestamp, "end_time": _timestamp, "missing_count": _integer},
    DataQualityReport: {
        "input_count": _integer, "output_count": _integer,
        "source_rows": lambda value: _sequence(value, _integer), "was_reordered": _boolean,
        "duplicates": lambda value: _sequence(value, lambda item: decode_record(item, DuplicateRemoval)),
        "gaps": lambda value: _sequence(value, lambda item: decode_record(item, CandleGap)),
        "expected_count": lambda value: None if value is None else _integer(value),
    },
    NormalizedDataset: {
        "instrument": _text, "interval": _duration,
        "candles": lambda value: _sequence(value, lambda item: decode_record(item, Candle)),
        "config": lambda value: decode_record(value, NormalizationConfig), "dataset_id": _text,
        "quality": lambda value: decode_record(value, DataQualityReport),
        "request": lambda value: None if value is None else decode_record(value, DatasetRequest),
    },
    SourceFingerprint: {"name": _text, "sha256": _text, "size_bytes": _integer},
    DatasetProvenance: {
        "description": _text,
        "sources": lambda value: _sequence(value, lambda item: decode_record(item, SourceFingerprint)),
        "parent_dataset_ids": lambda value: _sequence(value, _text),
    },
    TrainingConfig: {
        "lookback_bars": _integer, "horizon_bars": _integer,
        "validation_start": _timestamp, "test_start": _timestamp,
    },
    TrainingExample: {
        **{field: _timestamp for field in ("decision_at", "entry_at", "exit_at", "target_available_at")},
        "features": lambda value: _sequence(value, _decimal), "target_return": _decimal,
    },
}
