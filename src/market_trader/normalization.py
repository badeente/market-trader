"""Explicit candle normalization, quality diagnostics and dataset identity."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from market_trader.calendars import SessionCalendar, calendar_metadata, validate_calendar
from market_trader.coverage import DatasetRequest, validate_requests
from market_trader.data import validate_candles
from market_trader.domain import Candle, CandleValidationError

_VALUE_FIELDS = ("close_time", "available_at", "open", "high", "low", "close", "volume")
_NUMBER_FIELDS = ("open", "high", "low", "close", "volume")
NORMALIZATION_FORMAT = "market-trader.normalized-candles.v1"


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Choose the expected timeline explicitly; market sessions are not inferred."""

    calendar: Literal["continuous"] | SessionCalendar
    gap_policy: Literal["report", "reject"] = "report"


@dataclass(frozen=True, slots=True)
class CandleKey:
    instrument: str
    interval: timedelta
    open_time: datetime


@dataclass(frozen=True, slots=True)
class DuplicateRemoval:
    key: CandleKey
    kept_row: int
    removed_row: int


@dataclass(frozen=True, slots=True)
class CandleGap:
    """Missing candle starts in [start_time, end_time), with no fabricated rows."""

    start_time: datetime
    end_time: datetime
    missing_count: int


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    input_count: int
    output_count: int
    source_rows: tuple[int, ...]
    was_reordered: bool
    duplicates: tuple[DuplicateRemoval, ...]
    gaps: tuple[CandleGap, ...]
    expected_count: int | None = None


@dataclass(frozen=True, slots=True)
class NormalizedDataset:
    instrument: str
    interval: timedelta
    candles: tuple[Candle, ...]
    config: NormalizationConfig
    dataset_id: str
    quality: DataQualityReport
    request: DatasetRequest | None = None


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    datasets: tuple[NormalizedDataset, ...]

    def select(self, *, instrument: str, interval: timedelta) -> NormalizedDataset:
        """Select one exact instrument/interval pair; raise KeyError if absent."""
        for dataset in self.datasets:
            if dataset.instrument == instrument and dataset.interval == interval:
                return dataset
        raise KeyError((instrument, interval))


class DuplicateConflictError(CandleValidationError):
    """A repeated key differs in values or availability; no dataset is accepted."""

    def __init__(
        self, *, key: CandleKey, first_row: int, row_number: int,
        differing_fields: tuple[str, ...],
    ) -> None:
        self.key = key
        self.first_row = first_row
        self.differing_fields = differing_fields
        super().__init__(
            row_number=row_number, field="duplicate",
            message=f"conflict with row {first_row} for {key}: {differing_fields}",
        )


class DataGapError(CandleValidationError):
    """Strict normalization rejected gaps; the failed group's report is inspectable."""

    def __init__(self, *, dataset: NormalizedDataset, row_number: int | None) -> None:
        self.dataset = dataset
        super().__init__(
            row_number=row_number, field="open_time",
            message=f"missing candles for {dataset.instrument}, {dataset.interval}",
        )


def normalize_candles(
    candles: Sequence[Candle], *, config: NormalizationConfig,
    requests: Sequence[DatasetRequest] | None = None,
) -> NormalizationResult:
    """Validate, group, sort and deduplicate; report or reject calendar-active gaps."""
    if not isinstance(config, NormalizationConfig):
        raise ValueError("config must be a NormalizationConfig")
    validate_calendar(config.calendar)
    if config.gap_policy not in ("report", "reject"):
        raise ValueError("gap_policy must be report or reject")
    requested = None if requests is None else validate_requests(requests, calendar=config.calendar)

    history = tuple(candles)
    validate_candles(history)
    groups: dict[tuple[str, timedelta], list[tuple[int, Candle]]] = {
        key: [] for key in requested
    } if requested is not None else {}
    for row_number, candle in enumerate(history, start=1):
        key = (candle.instrument, candle.interval)
        if requested is not None and key not in requested:
            raise CandleValidationError(
                row_number=row_number, field="series", message=f"unrequested instrument/interval: {key}",
            )
        groups.setdefault(key, []).append((row_number, candle))
    return NormalizationResult(tuple(
        _normalize_group(
            groups[key], instrument=key[0], interval=key[1], config=config,
            request=None if requested is None else requested[key],
        ) for key in sorted(groups)
    ))


def _normalize_group(
    entries: Sequence[tuple[int, Candle]], *, instrument: str, interval: timedelta,
    config: NormalizationConfig, request: DatasetRequest | None,
) -> NormalizedDataset:
    calendar = config.calendar
    if request is not None:
        for row_number, candle in entries:
            _validate_requested_candle(candle, request=request, calendar=calendar, row_number=row_number)
    if isinstance(calendar, SessionCalendar):
        if request is None:
            calendar.validate_interval(interval)
        for row_number, candle in entries:
            _validate_session_candle(candle, calendar=calendar, row_number=row_number)
    unique: dict[datetime, tuple[int, Candle]] = {}
    duplicates = []
    for row_number, candle in entries:
        previous = unique.get(candle.open_time)
        if previous is None:
            unique[candle.open_time] = (row_number, candle)
            continue
        first_row, first = previous
        key = CandleKey(candle.instrument, candle.interval, candle.open_time)
        differences = tuple(
            field for field in _VALUE_FIELDS if getattr(first, field) != getattr(candle, field)
        )
        if differences:
            raise DuplicateConflictError(
                key=key, first_row=first_row, row_number=row_number,
                differing_fields=differences,
            )
        duplicates.append(DuplicateRemoval(key, first_row, row_number))

    ordered = sorted(unique.values(), key=lambda entry: entry[1].open_time)
    candles = tuple(candle for _, candle in ordered)
    source_rows = tuple(row for row, _ in ordered)
    gaps = []
    first_gap_row = None
    if request is not None:
        gaps.extend(_gap_ranges(
            request.start_time, candles[0].open_time if candles else request.end_time,
            calendar=calendar, interval=interval,
        ))
        if gaps and source_rows:
            first_gap_row = source_rows[0]
    for (_, previous), (row_number, current) in zip(ordered, ordered[1:]):
        distance = current.open_time - previous.close_time
        if distance < timedelta(0) or (calendar == "continuous" and distance % current.interval):
            raise CandleValidationError(
                row_number=row_number, field="open_time",
                message="overlap or start outside the first candle's interval grid",
            )
        missing = _gap_ranges(previous.close_time, current.open_time, calendar=calendar, interval=interval)
        if missing and not gaps:
            first_gap_row = row_number
        gaps.extend(missing)
    if request is not None and candles:
        gaps.extend(_gap_ranges(candles[-1].close_time, request.end_time, calendar=calendar, interval=interval))
    expected_count = None if request is None else sum(
        gap.missing_count for gap in _gap_ranges(
            request.start_time, request.end_time, calendar=calendar, interval=interval,
        )
    )

    dataset = NormalizedDataset(
        instrument=instrument,
        interval=interval,
        candles=candles,
        config=config,
        dataset_id=_dataset_id(candles, instrument=instrument, interval=interval, config=config, request=request),
        quality=DataQualityReport(
            input_count=len(entries), output_count=len(candles), source_rows=source_rows,
            was_reordered=source_rows != tuple(row for row, _ in unique.values()),
            duplicates=tuple(duplicates), gaps=tuple(gaps),
            expected_count=expected_count,
        ),
        request=request,
    )
    if config.gap_policy == "reject" and gaps:
        raise DataGapError(dataset=dataset, row_number=first_gap_row)
    return dataset


def _gap_ranges(
    start: datetime, end: datetime, *, calendar: Literal["continuous"] | SessionCalendar,
    interval: timedelta,
) -> tuple[CandleGap, ...]:
    if start == end:
        return ()
    spans = calendar.active_spans(start, end) if isinstance(calendar, SessionCalendar) else ((start, end),)
    return tuple(CandleGap(left, right, (right - left) // interval) for left, right in spans)


def _validate_requested_candle(
    candle: Candle, *, request: DatasetRequest,
    calendar: Literal["continuous"] | SessionCalendar, row_number: int,
) -> None:
    if not request.start_time <= candle.open_time < request.end_time:
        raise CandleValidationError(
            row_number=row_number, field="open_time", message="outside requested data range",
        )
    if candle.close_time > request.end_time:
        raise CandleValidationError(
            row_number=row_number, field="close_time", message="crosses the requested range end",
        )
    if calendar == "continuous" and (candle.open_time - request.start_time) % request.interval:
        raise CandleValidationError(
            row_number=row_number, field="open_time", message="outside the requested interval grid",
        )


def _validate_session_candle(
    candle: Candle, *, calendar: SessionCalendar, row_number: int,
) -> None:
    if not calendar.coverage_start <= candle.open_time < calendar.coverage_end:
        raise CandleValidationError(
            row_number=row_number, field="open_time", message="outside declared calendar coverage",
        )
    if candle.close_time > calendar.coverage_end:
        raise CandleValidationError(
            row_number=row_number, field="close_time", message="outside declared calendar coverage",
        )
    session = calendar.session_at(candle.open_time)
    if session is None:
        raise CandleValidationError(
            row_number=row_number, field="open_time", message="outside an active session",
        )
    if candle.close_time > session.close_time:
        raise CandleValidationError(
            row_number=row_number, field="close_time", message="candle crosses the session close",
        )
    if (candle.open_time - session.open_time) % candle.interval:
        raise CandleValidationError(
            row_number=row_number, field="open_time", message="outside the session's interval grid",
        )


def _decimal_text(value: Decimal) -> str:
    """Canonical numeric equality without rounding or expanding large exponents."""
    if value.is_zero():
        return "0"
    sign, digits, exponent = value.as_tuple()
    end = len(digits)
    while digits[end - 1] == 0:
        end -= 1
    coefficient = "".join(str(digit) for digit in digits[:end])
    return f"{'-' if sign else ''}{coefficient}e{exponent + len(digits) - end}"


def _dataset_id(
    candles: tuple[Candle, ...], *, instrument: str, interval: timedelta,
    config: NormalizationConfig, request: DatasetRequest | None,
) -> str:
    """SHA-256 of versioned JSON lines; provenance rows are deliberately separate."""
    digest = hashlib.sha256()

    def append(value: object) -> None:
        digest.update(json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")

    header = {
        "format": NORMALIZATION_FORMAT,
        "calendar": calendar_metadata(config.calendar),
        "gap_policy": config.gap_policy,
        "instrument": instrument,
        "interval_us": interval // timedelta(microseconds=1),
    }
    if request is not None:
        header["request"] = {
            "start_time": request.start_time.astimezone(UTC).isoformat(timespec="microseconds"),
            "end_time": request.end_time.astimezone(UTC).isoformat(timespec="microseconds"),
        }
    append(header)
    for candle in candles:
        append([
            *(getattr(candle, field).astimezone(UTC).isoformat(timespec="microseconds")
              for field in ("open_time", "close_time", "available_at")),
            *(_decimal_text(getattr(candle, field)) for field in _NUMBER_FIELDS),
        ])
    return f"sha256:{digest.hexdigest()}"
