"""Private Dukascopy CSV to the fixed AP1 research-session dataset."""

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

from market_trader.artifacts import SourceFingerprint
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.coverage import DatasetRequest, validate_requests
from market_trader.data import validate_candles
from market_trader.domain import Candle, CandleValidationError
from market_trader.normalization import NormalizationConfig, NormalizedDataset, normalize_candles
from market_trader.storage import fingerprint_source

_INTERVAL = timedelta(minutes=15)
_BERLIN = ZoneInfo("Europe/Berlin")
_NAME = "AP1 complete observed weekdays 09:00-17:30 Europe/Berlin v1"


@dataclass(frozen=True, slots=True)
class ReferenceImport:
    dataset: NormalizedDataset
    audit: dict
    source: SourceFingerprint


def import_dukascopy_reference(
    path: Path, *, instrument: str, start: datetime, end: datetime,
    expected_sha256: str | None = None,
) -> ReferenceImport:
    """Import complete research dates; the source is never rewritten."""
    request = DatasetRequest(instrument, _INTERVAL, start, end)
    # Explicit-session requests allow arbitrary UTC coverage boundaries outside
    # sessions; validate their scalar contract without imposing a continuous grid.
    validate_requests((request,), calendar=SessionCalendar(_NAME, start, end, ()))
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be canonical sha256:<64 lowercase hex>")
    # Parse only the immutable snapshot whose digest is checked below. The shared
    # root-file fingerprint must agree; concurrent changes cannot misattribute it.
    raw = Path(path).read_bytes()
    source = fingerprint_source(path, name="Dukascopy 15-minute bid ISO UTC source")
    snapshot_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    if source.sha256 != snapshot_sha256 or source.size_bytes != len(raw):
        raise ValueError("sha256 source changed while taking import snapshot")
    if expected_sha256 is not None and snapshot_sha256 != expected_sha256:
        raise ValueError("sha256 source hash mismatch")
    rows = []
    with io.StringIO(raw.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            header = reader.fieldnames
        except csv.Error as error:
            raise CandleValidationError(row_number=0, field="header", message=f"invalid CSV: {error}") from error
        columns = ("timestamp", "open", "high", "low", "close", "volume")
        if header is None or len(header) != len(columns) or set(header) != set(columns):
            raise CandleValidationError(row_number=0, field="header", message="expected unique timestamp,open,high,low,close,volume columns")
        try:
            for row_number, row in enumerate(reader, 1):
                if None in row or any(value is None or not value.strip() for value in row.values()):
                    raise CandleValidationError(row_number=row_number, field="row", message="expected all six values")
                try:
                    opened = datetime.fromisoformat(row["timestamp"])
                    if opened.utcoffset() != timedelta(0):
                        raise ValueError("timestamp must be explicit UTC")
                    opened = opened.astimezone(UTC)
                    if (opened - datetime(1970, 1, 1, tzinfo=UTC)) % _INTERVAL:
                        raise ValueError("timestamp must be on the 15-minute UTC grid")
                    closed = opened + _INTERVAL
                    candle = Candle(instrument, _INTERVAL, opened, closed, closed,
                                    *(Decimal(row[field]) for field in columns[1:]))
                    validate_candles((candle,))
                    if any(getattr(candle, field) <= 0 for field in columns[1:5]):
                        raise ValueError("prices must be positive")
                except (ValueError, InvalidOperation, OverflowError) as error:
                    raise CandleValidationError(row_number=row_number, field="row", message=str(error)) from error
                if rows and candle.open_time <= rows[-1].open_time:
                    raise CandleValidationError(row_number=row_number, field="timestamp", message="raw timestamps must be strictly increasing; duplicate or descending row")
                rows.append(candle)
        except csv.Error as error:
            raise CandleValidationError(row_number=len(rows) + 1, field="row", message=f"invalid CSV: {error}") from error
    sessions = []
    selected = []
    dates = []
    outside = 0
    excluded_partial = 0
    by_date = {}
    for candle in rows:
        if not start <= candle.open_time < end:
            continue
        local = candle.open_time.astimezone(_BERLIN)
        if local.weekday() >= 5 or not time(9) <= local.time() < time(17, 30):
            outside += 1
        else:
            by_date.setdefault(local.date(), []).append(candle)
    day = start.astimezone(_BERLIN).date()
    last = (end - timedelta(microseconds=1)).astimezone(_BERLIN).date()
    while day <= last:
        opened = datetime.combine(day, time(9), _BERLIN).astimezone(UTC)
        closed = datetime.combine(day, time(17, 30), _BERLIN).astimezone(UTC)
        observed = by_date.get(day, [])
        if day.weekday() < 5 and opened < end and closed > start:
            starts = {c.open_time for c in observed}
            missing = [opened + i * _INTERVAL for i in range(34)
                       if opened + i * _INTERVAL not in starts]
            status = "no_rows_in_window" if not observed else "partial_session"
            if not missing and opened >= start and closed <= end:
                status = "admitted"
                sessions.append(TradingSession(opened, closed))
                selected.extend(observed)
            else:
                excluded_partial += len(observed)
            dates.append({"date": day.isoformat(), "status": status,
                          "observed_starts_count": len(observed),
                          "missing_starts_count": len(missing),
                          "missing_starts": [t.isoformat() for t in missing]})
        day += timedelta(days=1)
    if not sessions:
        raise ValueError("requested range contains no complete research sessions")
    calendar = SessionCalendar(_NAME, start, end, tuple(sessions))
    dataset = normalize_candles(selected, config=NormalizationConfig(calendar, "reject"), requests=(request,)).datasets[0]
    audit = {
        "format": "market-trader.ap1-reference-import.v1",
        "dates": dates, "raw_rows": len(rows),
        "outside_window_rows": outside,
        "outside_requested_range_rows": sum(not start <= c.open_time < end for c in rows),
        "excluded_partial_session_rows": excluded_partial,
        "admitted_rows": len(selected),
        "caveats": [
            "Research window only; not an official provider or exchange calendar.",
            "Complete observed dates only may introduce selection bias; not provider completeness certification.",
            "Dukascopy bid prices; historical ask fills are not reconstructed.",
            "Volume meaning is not independently confirmed; not a model feature.",
            "Internal research only; raw source must remain private.",
        ],
    }
    return ReferenceImport(dataset, audit, source)



def slice_session_dataset(
    dataset: NormalizedDataset, start: datetime, end: datetime,
) -> NormalizedDataset:
    """Crop coverage and candles to whole accepted sessions, never expand them."""
    calendar = dataset.config.calendar
    if not isinstance(calendar, SessionCalendar):
        raise ValueError("slice requires an explicit SessionCalendar")
    request = DatasetRequest(dataset.instrument, dataset.interval, start, end)
    validate_requests((request,), calendar=calendar)
    for boundary in (start, end):
        session = calendar.session_at(boundary)
        if session is not None and boundary != session.open_time:
            raise ValueError("slice boundary cuts a session")
    sessions = tuple(s for s in calendar.sessions if start <= s.open_time and s.close_time <= end)
    if not sessions:
        raise ValueError("requested range contains no complete research sessions")
    cropped = SessionCalendar(calendar.name, start, end, sessions)
    candles = tuple(c for c in dataset.candles if start <= c.open_time < end)
    return normalize_candles(
        candles, config=NormalizationConfig(cropped, "reject"), requests=(request,),
    ).datasets[0]
