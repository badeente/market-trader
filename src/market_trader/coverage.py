"""Explicit completeness requirements for individual candle series."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from market_trader.calendars import SessionCalendar


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """Require one instrument/interval over the UTC range [start_time, end_time)."""

    instrument: str
    interval: timedelta
    start_time: datetime
    end_time: datetime


def validate_requests(
    requests: Sequence[DatasetRequest], *, calendar: Literal["continuous"] | SessionCalendar,
) -> dict[tuple[str, timedelta], DatasetRequest]:
    """Validate and snapshot unique series requests, including empty input series."""
    if not isinstance(requests, Sequence) or not requests:
        raise ValueError("requests must be a nonempty sequence of DatasetRequest values")
    result = {}
    for request in tuple(requests):
        if not isinstance(request, DatasetRequest):
            raise ValueError("requests must contain DatasetRequest values")
        if (not isinstance(request.instrument, str) or not request.instrument
                or request.instrument != request.instrument.strip()):
            raise ValueError("request instrument must be nonempty and without padding")
        if not isinstance(request.interval, timedelta) or request.interval <= timedelta(0):
            raise ValueError("request interval must be a positive timedelta")
        for field in ("start_time", "end_time"):
            value = getattr(request, field)
            if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
                raise ValueError(f"request {field} must be a timezone-aware UTC timestamp")
        if request.start_time >= request.end_time:
            raise ValueError("request range must have positive duration")
        if isinstance(calendar, SessionCalendar):
            calendar.validate_interval(request.interval)
            if (request.start_time < calendar.coverage_start
                    or request.end_time > calendar.coverage_end):
                raise ValueError("request is outside declared calendar coverage")
            for boundary in (request.start_time, request.end_time):
                session = calendar.session_at(boundary)
                if session is not None and (boundary - session.open_time) % request.interval:
                    raise ValueError("request boundary cuts an active candle interval")
        elif (request.end_time - request.start_time) % request.interval:
            raise ValueError("continuous request duration must be a whole multiple of its interval")
        key = (request.instrument, request.interval)
        if key in result:
            raise ValueError(f"only one request per instrument/interval is supported: {key}")
        result[key] = request
    return result
