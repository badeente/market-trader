"""Synthetic-only AP1 reference source boundary contracts."""
import hashlib
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from market_trader.calendars import SessionCalendar

BERLIN = ZoneInfo("Europe/Berlin")
HEADER = "timestamp,open,high,low,close,volume"


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def session_rows(day):
    start = datetime.fromisoformat(day + "T09:00:00").replace(tzinfo=BERLIN).astimezone(UTC)
    return [f"{(start + i * timedelta(minutes=15)).isoformat()},100,102,99,101,3" for i in range(34)]


def source(tmp_path, rows, header=HEADER):
    path = tmp_path / "synthetic.csv"
    path.write_bytes((header + "\r\n" + "\r\n".join(rows) + "\r\n").encode())
    return path


def load(path, start, end, **kwargs):
    from market_trader.dukascopy_import import import_dukascopy_reference
    return import_dukascopy_reference(path, instrument="deuidxeur", start=utc(start), end=utc(end), **kwargs)


@pytest.mark.parametrize("day,open_hour", [("2024-01-08", 8), ("2024-07-08", 7)])
def test_complete_session_uses_berlin_and_preserves_source(tmp_path, day, open_hour):
    path = source(tmp_path, session_rows(day))
    before = path.read_bytes()
    result = load(path, day + "T00:00:00", day + "T23:00:00")
    dataset = result.dataset
    assert isinstance(dataset.config.calendar, SessionCalendar)
    assert dataset.request is not None
    assert len(dataset.candles) == 34
    assert dataset.candles[0].open_time.hour == open_hour
    assert dataset.candles[-1].close_time == dataset.candles[0].open_time + timedelta(hours=8, minutes=30)
    assert all(c.available_at == c.close_time for c in dataset.candles)
    assert dataset.config.gap_policy == "reject"
    assert dataset.quality.gaps == ()
    assert dataset.quality.expected_count == 34
    assert len(dataset.config.calendar.sessions) == 1
    assert dataset.request.start_time == utc(day + "T00:00:00")
    assert result.source.sha256 == "sha256:" + hashlib.sha256(before).hexdigest()
    assert result.source.size_bytes == len(before)
    assert "Dukascopy" in result.source.name
    assert path.read_bytes() == before
    again = load(path, day + "T00:00:00", day + "T23:00:00")
    assert again == result


def test_audit_lists_every_weekday_and_excludes_partial_and_weekend(tmp_path):
    rows = session_rows("2024-01-05") + session_rows("2024-01-06")
    rows += session_rows("2024-01-08")[:-1]
    rows += ["2024-01-08T17:00:00+00:00,100,102,99,101,3"]
    result = load(source(tmp_path, rows), "2024-01-05T00:00:00", "2024-01-10T00:00:00")
    assert len(result.dataset.candles) == 34
    assert [(d["date"], d["status"]) for d in result.audit["dates"]] == [
        ("2024-01-05", "admitted"), ("2024-01-08", "partial_session"),
        ("2024-01-09", "no_rows_in_window")]
    partial = result.audit["dates"][1]
    assert partial["missing_starts_count"] == 1
    assert partial["missing_starts"] == ["2024-01-08T16:15:00+00:00"]
    assert result.audit["outside_window_rows"] == 35
    assert result.audit["excluded_partial_session_rows"] == 33
    assert result.audit["raw_rows"] == 102
    assert "selection bias" in " ".join(result.audit["caveats"])
    assert "holiday" not in str(result.audit["dates"])


@pytest.mark.parametrize("start,end,day", [
    ("2025-12-31T00:00:00", "2026-01-01T00:00:00", "2025-12-31"),
    ("2024-01-05T19:00:00", "2024-01-09T00:00:00", "2024-01-08"),
])
def test_audit_excludes_local_dates_whose_research_window_is_outside_request(tmp_path, start, end, day):
    result = load(source(tmp_path, session_rows(day)), start, end)
    assert [row["date"] for row in result.audit["dates"]] == [day]


@pytest.mark.parametrize("header", [
    "", HEADER + ",extra", HEADER.replace("volume", "open"),
    HEADER.replace(",volume", ""), HEADER.replace("timestamp", "Timestamp"),
])
def test_rejects_nonexact_header(tmp_path, header):
    from market_trader.domain import CandleValidationError
    with pytest.raises(CandleValidationError, match="header"):
        load(source(tmp_path, session_rows("2024-01-08"), header),
             "2024-01-08T00:00:00", "2024-01-09T00:00:00")



@pytest.mark.parametrize("bad", [
    "2020-01-01T08:00:00,100,102,99,101,3",
    "2020-01-01T09:00:00+01:00,100,102,99,101,3",
    "2020-01-01T08:01:00+00:00,100,102,99,101,3",
    "2020-01-01T08:00:00+00:00,NaN,102,99,101,3",
    "2020-01-01T08:00:00+00:00,100,Infinity,99,101,3",
    "2020-01-01T08:00:00+00:00,100,102,99,101,-1",
    "2020-01-01T08:00:00+00:00,-100,102,-101,101,3",
    "2020-01-01T08:00:00+00:00,100,100,99,101,3",
    "2020-01-01T08:00:00+00:00,100,102,102,101,3",
    "2020-01-01T08:00:00+00:00,100,102,99,101,",
    "2020-01-01T08:00:00+00:00,100,102,99,101",
    "2020-01-01T08:00:00+00:00,100,102,99,101,3,extra",
    "2020-01-01T08:00:00+00:00,garbage,102,99,101,3",
    '"2020-01-01T08:00:00+00:00,100,102,99,101,3',
])
def test_validates_raw_rows_even_outside_requested_range(tmp_path, bad):
    from market_trader.domain import CandleValidationError
    with pytest.raises(CandleValidationError) as caught:
        load(source(tmp_path, [bad] + session_rows("2024-01-08")),
             "2024-01-08T00:00:00", "2024-01-09T00:00:00")
    assert caught.value.row_number == 1


@pytest.mark.parametrize("prefix", [
    ["2020-01-01T08:00:00+00:00,100,102,99,101,3"] * 2,
    ["2020-01-01T08:15:00+00:00,100,102,99,101,3",
     "2020-01-01T08:00:00+00:00,100,102,99,101,3"],
])
def test_rejects_duplicate_or_descending_raw_rows_before_selection(tmp_path, prefix):
    from market_trader.domain import CandleValidationError
    with pytest.raises(CandleValidationError, match="strictly increasing") as caught:
        load(source(tmp_path, prefix + session_rows("2024-01-08")),
             "2024-01-08T00:00:00", "2024-01-09T00:00:00")
    assert caught.value.row_number == 2


@pytest.mark.parametrize("expected", ["sha256:" + "0" * 64, "0" * 64, "SHA256:" + "a" * 64, "sha256:" + "A" * 64, "sha256:" + "a" * 63, " sha256:" + "a" * 64])
def test_expected_hash_is_strict_canonical_and_must_match(tmp_path, expected):
    with pytest.raises(ValueError, match="sha256"):
        load(source(tmp_path, session_rows("2024-01-08")),
             "2024-01-08T00:00:00", "2024-01-09T00:00:00", expected_sha256=expected)


def test_matching_hash_accepts_exact_bytes(tmp_path):
    path = source(tmp_path, session_rows("2024-01-08"))
    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    result = load(path, "2024-01-08T00:00:00", "2024-01-09T00:00:00", expected_sha256=expected)
    assert result.source.sha256 == expected


def test_parser_uses_same_snapshot_as_fingerprint(tmp_path, monkeypatch):
    import market_trader.dukascopy_import as importer
    path = source(tmp_path, session_rows("2024-01-08"))
    original = path.read_bytes()
    fingerprint = importer.fingerprint_source

    def mutate_after_fingerprinting(path, *, name):
        result = fingerprint(path, name=name)
        path.write_bytes(original.replace(b",100,", b",101,"))
        return result

    monkeypatch.setattr(importer, "fingerprint_source", mutate_after_fingerprinting)
    result = load(path, "2024-01-08T00:00:00", "2024-01-09T00:00:00")
    assert result.source.sha256 == "sha256:" + hashlib.sha256(original).hexdigest()
    assert result.dataset.candles[0].open == 100


@pytest.mark.parametrize("rows", [[], session_rows("2024-01-08")[:-1], session_rows("2024-01-06")])
def test_no_admitted_sessions_errors_clearly(tmp_path, rows):
    with pytest.raises(ValueError, match="no complete research sessions"):
        load(source(tmp_path, rows), "2024-01-05T00:00:00", "2024-01-09T00:00:00")


@pytest.mark.parametrize("start,end", [
    (datetime(2024, 1, 8), utc("2024-01-09T00:00:00")),
    (datetime.fromisoformat("2024-01-08T00:00:00+01:00"), utc("2024-01-09T00:00:00")),
])
def test_import_requires_explicit_utc_boundaries(tmp_path, start, end):
    from market_trader.dukascopy_import import import_dukascopy_reference
    with pytest.raises(ValueError, match="UTC"):
        import_dukascopy_reference(source(tmp_path, session_rows("2024-01-08")),
                                  instrument="deuidxeur", start=start, end=end)



def test_safe_slices_crop_calendar_request_and_all_future_candles(tmp_path):
    from market_trader.dukascopy_import import slice_session_dataset
    rows = session_rows("2023-12-29") + session_rows("2024-01-02") + session_rows("2025-01-02")
    original = load(source(tmp_path, rows), "2023-01-01T00:00:00", "2026-01-01T00:00:00").dataset
    fit = slice_session_dataset(original, utc("2023-01-01T00:00:00"), utc("2025-01-01T00:00:00"))
    validation = slice_session_dataset(fit, utc("2024-01-01T00:00:00"), utc("2025-01-01T00:00:00"))
    assert original.request is not None
    assert fit.request is not None
    assert validation.request is not None
    assert isinstance(validation.config.calendar, SessionCalendar)
    assert len(original.candles) == 102
    assert len(fit.candles) == 68
    assert len(validation.candles) == 34
    assert all(c.open_time.year == 2024 for c in validation.candles)
    assert all(c.close_time < utc("2025-01-01T00:00:00") for c in fit.candles)
    assert validation.config.calendar.coverage_start == validation.request.start_time == utc("2024-01-01T00:00:00")
    assert validation.config.calendar.coverage_end == validation.request.end_time == utc("2025-01-01T00:00:00")
    assert len(validation.config.calendar.sessions) == 1
    assert validation.quality.expected_count == 34
    assert validation.quality.gaps == ()
    assert len({original.dataset_id, fit.dataset_id, validation.dataset_id}) == 3
    assert slice_session_dataset(original, original.request.start_time, original.request.end_time) == original
    assert slice_session_dataset(fit, validation.request.start_time, validation.request.end_time) == validation


@pytest.mark.parametrize("start,end,reason", [
    ("2024-01-08T08:15:00", "2024-01-09T00:00:00", "cuts a session"),
    ("2024-01-08T00:00:00", "2024-01-08T16:15:00", "cuts a session"),
    ("2024-01-07T23:00:00", "2024-01-09T00:00:00", "coverage"),
    ("2024-01-08T00:00:00", "2024-01-09T01:00:00", "coverage"),
    ("2024-01-08T17:00:00", "2024-01-09T00:00:00", "no complete research sessions"),
])
def test_slice_rejects_session_cuts_expansion_or_empty(tmp_path, start, end, reason):
    from market_trader.dukascopy_import import slice_session_dataset
    original = load(source(tmp_path, session_rows("2024-01-08")), "2024-01-08T00:00:00", "2024-01-09T00:00:00").dataset
    with pytest.raises(ValueError, match=reason):
        slice_session_dataset(original, utc(start), utc(end))



def test_malformed_quoted_header_is_classified(tmp_path):
    from market_trader.domain import CandleValidationError
    path = tmp_path / "bad.csv"
    path.write_bytes(b'"timestamp,open,high,low,close,volume\n')
    with pytest.raises(CandleValidationError, match="header") as caught:
        load(path, "2024-01-08T00:00:00", "2024-01-09T00:00:00")
    assert caught.value.row_number == 0



def test_snapshot_change_before_root_fingerprint_is_rejected(tmp_path, monkeypatch):
    import market_trader.dukascopy_import as importer
    path = source(tmp_path, session_rows("2024-01-08"))
    fingerprint = importer.fingerprint_source

    def mutate_before_fingerprinting(path, *, name):
        path.write_bytes(path.read_bytes().replace(b",100,", b",101,"))
        return fingerprint(path, name=name)

    monkeypatch.setattr(importer, "fingerprint_source", mutate_before_fingerprinting)
    with pytest.raises(ValueError, match="source changed"):
        load(path, "2024-01-08T00:00:00", "2024-01-09T00:00:00")


def test_dst_weekend_changes_utc_session_open_without_inventing_holidays(tmp_path):
    rows = session_rows("2024-03-29") + session_rows("2024-04-01")
    result = load(source(tmp_path, rows), "2024-03-29T00:00:00", "2024-04-01T22:00:00")
    assert isinstance(result.dataset.config.calendar, SessionCalendar)
    assert [s.open_time.hour for s in result.dataset.config.calendar.sessions] == [8, 7]
    assert [(d["date"], d["status"]) for d in result.audit["dates"]] == [
        ("2024-03-29", "admitted"), ("2024-04-01", "admitted")]


def test_requested_range_is_half_open_and_partial_edge_dates_are_not_admitted(tmp_path):
    rows = session_rows("2024-01-08") + session_rows("2024-01-09") + session_rows("2024-01-10")
    result = load(source(tmp_path, rows), "2024-01-08T08:15:00", "2024-01-10T16:15:00")
    assert len(result.dataset.candles) == 34
    assert [d["status"] for d in result.audit["dates"]] == ["partial_session", "admitted", "partial_session"]
    assert result.audit["dates"][0]["missing_starts"] == ["2024-01-08T08:00:00+00:00"]
    assert result.audit["dates"][2]["missing_starts"] == ["2024-01-10T16:15:00+00:00"]
    assert result.audit["outside_requested_range_rows"] == 2
    assert result.audit["raw_rows"] == (result.audit["outside_requested_range_rows"]
        + result.audit["outside_window_rows"] + result.audit["excluded_partial_session_rows"]
        + result.audit["admitted_rows"])
