"""Finite, explicit UTC session schedules; no exchange rules are inferred."""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal


@dataclass(frozen=True, slots=True)
class TradingSession:
    """An active half-open UTC interval [open_time, close_time)."""

    open_time: datetime
    close_time: datetime


@dataclass(frozen=True, slots=True)
class SessionCalendar:
    """A complete explicit schedule inside a declared UTC coverage range."""

    name: str
    coverage_start: datetime
    coverage_end: datetime
    sessions: tuple[TradingSession, ...]

    def validate_interval(self, interval: timedelta) -> None:
        """Reject sessions that would require partial candles."""
        if not isinstance(interval, timedelta) or interval <= timedelta(0):
            raise ValueError("interval must be a positive timedelta")
        if any((session.close_time - session.open_time) % interval for session in self.sessions):
            raise ValueError("every session duration must be a whole multiple of the candle interval")

    def session_at(self, timestamp: datetime) -> TradingSession | None:
        """Find a session containing a candle's open time."""
        index = bisect_right(self.sessions, timestamp, key=lambda session: session.open_time) - 1
        if index >= 0 and timestamp < self.sessions[index].close_time:
            return self.sessions[index]
        return None

    def active_spans(
        self, start_time: datetime, end_time: datetime,
    ) -> tuple[tuple[datetime, datetime], ...]:
        """Intersect an observed gap with active sessions."""
        spans = []
        index = max(0, bisect_right(
            self.sessions, start_time, key=lambda session: session.open_time,
        ) - 1)
        while index < len(self.sessions) and self.sessions[index].open_time < end_time:
            session = self.sessions[index]
            start = max(start_time, session.open_time)
            end = min(end_time, session.close_time)
            if start < end:
                spans.append((start, end))
            index += 1
        return tuple(spans)


def _validate_utc(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be a timezone-aware UTC timestamp")


def validate_calendar(calendar: Literal["continuous"] | SessionCalendar) -> None:
    """Validate a schedule even if no candle rows are supplied."""
    if calendar == "continuous":
        return
    if not isinstance(calendar, SessionCalendar):
        raise ValueError("calendar must be continuous or an explicit SessionCalendar")
    if not isinstance(calendar.name, str) or not calendar.name or calendar.name != calendar.name.strip():
        raise ValueError("calendar name must be nonempty and without padding")
    _validate_utc(calendar.coverage_start, "coverage_start")
    _validate_utc(calendar.coverage_end, "coverage_end")
    if calendar.coverage_start >= calendar.coverage_end:
        raise ValueError("calendar coverage must have positive duration")
    if not isinstance(calendar.sessions, tuple):
        raise ValueError("sessions must be an immutable tuple")
    previous = None
    for session in calendar.sessions:
        if not isinstance(session, TradingSession):
            raise ValueError("sessions must contain TradingSession values")
        _validate_utc(session.open_time, "session.open_time")
        _validate_utc(session.close_time, "session.close_time")
        if session.open_time >= session.close_time:
            raise ValueError("session must have positive duration")
        if session.open_time < calendar.coverage_start or session.close_time > calendar.coverage_end:
            raise ValueError("session is outside declared calendar coverage")
        if previous is not None and session.open_time < previous.close_time:
            raise ValueError("sessions must be ordered and non-overlapping")
        previous = session


def calendar_metadata(calendar: Literal["continuous"] | SessionCalendar) -> str | dict:
    """Canonical calendar configuration for dataset identity."""
    if calendar == "continuous":
        return calendar

    def timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="microseconds")

    return {
        "format": "market-trader.session-calendar.v1",
        "name": calendar.name,
        "coverage_start": timestamp(calendar.coverage_start),
        "coverage_end": timestamp(calendar.coverage_end),
        "sessions": [
            [timestamp(session.open_time), timestamp(session.close_time)]
            for session in calendar.sessions
        ],
    }
