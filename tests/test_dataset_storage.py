"""Persistence must retain data and provenance and reject inconsistent artifacts."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json

import pytest

from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.coverage import DatasetRequest
from market_trader.data import resample_candles
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader import storage
from market_trader.storage import (
    DatasetProvenance, DatasetStorageError, SourceFingerprint,
    fingerprint_source, load_dataset, save_dataset, verify_sources,
)

CONFIG = NormalizationConfig(calendar="continuous")
SYNTHETIC = DatasetProvenance(description="hand-specified synthetic candles")
FORMAT = "market-trader.dataset-artifact.v1"


@pytest.fixture
def dataset(minute_candles):
    return normalize_candles(minute_candles, config=CONFIG).datasets[0]


def rewrite(path, payload):
    """Recompute the public checksum to exercise semantic checks beyond that hash."""
    body = {key: value for key, value in payload.items() if key != "artifact_id"}
    encoded = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    payload["artifact_id"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.mark.case("DAT-009")
def test_dat_009_source_fingerprint_uses_exact_bytes_and_logical_name(tmp_path):
    raw = tmp_path / "private-local-name.csv"
    raw.write_bytes(b"abc")
    fingerprint = fingerprint_source(raw, name="raw-page-1")
    assert fingerprint == SourceFingerprint(
        "raw-page-1", "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", 3,
    )
    assert raw.read_bytes() == b"abc"


@pytest.mark.case("DAT-009")
def test_dat_009_roundtrip_retains_calendar_request_and_original_batch_audit(tmp_path, minute_candles):
    start = minute_candles[0].open_time
    calendar = SessionCalendar("synthetic-v1", start, start + timedelta(days=1), (
        TradingSession(start, start + timedelta(minutes=2)),
        TradingSession(start + timedelta(minutes=3), start + timedelta(minutes=5)),
    ))
    request = DatasetRequest("TEST", timedelta(minutes=1), start, start + timedelta(minutes=5))
    batch = (replace(minute_candles[4], instrument="OTHER"), minute_candles[3],
             minute_candles[1], minute_candles[3], minute_candles[4])
    original = normalize_candles(batch, config=NormalizationConfig(calendar=calendar),
                                 requests=(request, replace(request, instrument="OTHER"))).select(
                                     instrument="TEST", interval=timedelta(minutes=1))
    raw = tmp_path / "seed.txt"
    raw.write_bytes(b"synthetic source description\r\n")
    provenance = DatasetProvenance("synthetic batch", (fingerprint_source(raw, name="seed"),))
    path = tmp_path / "dataset.json"
    manifest = save_dataset(path, original, provenance=provenance)
    restored = load_dataset(path)
    assert restored.dataset == original
    assert restored.manifest == manifest
    assert manifest.format == FORMAT
    assert manifest.normalization_format == "market-trader.normalized-candles.v1"
    assert manifest.dataset_id == original.dataset_id
    assert manifest.provenance == provenance
    assert original.quality.source_rows == (3, 2, 5)  # Global rows may exceed this group's input count.
    assert original.quality.input_count == 4
    assert restored.dataset.quality.duplicates[0].removed_row == 4
    assert raw.read_bytes() == b"synthetic source description\r\n"
    with pytest.raises(FrozenInstanceError):
        restored.manifest.provenance.description = "changed"


@pytest.mark.case("DAT-009")
def test_dat_009_roundtrip_preserves_decimal_and_microsecond_precision(tmp_path, minute_candles):
    first = minute_candles[0]
    candle = replace(first, interval=timedelta(microseconds=1), close_time=first.open_time + timedelta(microseconds=1),
                     available_at=first.open_time + timedelta(microseconds=2),
                     volume=Decimal("1000000000000000000000000000000.000001"))
    original = normalize_candles((candle,), config=CONFIG).datasets[0]
    with localcontext() as context:
        context.prec = 2
        save_dataset(tmp_path / "dataset.json", original, provenance=SYNTHETIC)
        restored = load_dataset(tmp_path / "dataset.json").dataset
    assert restored == original
    assert restored.candles[0].volume.as_tuple() == candle.volume.as_tuple()


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("closed_only", [False, True])
def test_dat_009_empty_requested_series_can_be_stored(tmp_path, closed_only):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    request = DatasetRequest("MISSING", timedelta(minutes=1), start, start + timedelta(minutes=5))
    config = NormalizationConfig(calendar=SessionCalendar("closed", start, start + timedelta(days=1), ()), gap_policy="reject") if closed_only else CONFIG
    original = normalize_candles((), config=config, requests=(request,)).datasets[0]
    path = tmp_path / "empty.json"
    save_dataset(path, original, provenance=SYNTHETIC)
    assert load_dataset(path).dataset == original


@pytest.mark.case("DAT-009")
def test_dat_009_writes_are_deterministic_and_do_not_embed_local_paths(tmp_path, dataset):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    a = save_dataset(first, dataset, provenance=SYNTHETIC)
    b = save_dataset(second, dataset, provenance=SYNTHETIC)
    assert a == b
    assert first.read_bytes() == second.read_bytes()
    assert str(tmp_path).encode() not in first.read_bytes()


@pytest.mark.case("DAT-009")
def test_dat_009_artifact_identity_includes_audit_and_provenance(tmp_path, dataset, minute_candles):
    reordered = normalize_candles((*reversed(minute_candles), minute_candles[0]), config=CONFIG).datasets[0]
    assert reordered.dataset_id == dataset.dataset_id
    a = save_dataset(tmp_path / "a.json", dataset, provenance=SYNTHETIC)
    b = save_dataset(tmp_path / "b.json", reordered, provenance=SYNTHETIC)
    c = save_dataset(tmp_path / "c.json", dataset, provenance=replace(SYNTHETIC, description="another origin"))
    assert len({a.artifact_id, b.artifact_id, c.artifact_id}) == 3
    assert a.dataset_id == b.dataset_id == c.dataset_id


@pytest.mark.case("DAT-009")
def test_dat_009_resampling_interval_changes_identity_and_retains_parent(tmp_path, dataset, minute_candles):
    interval = timedelta(minutes=5)
    candles = resample_candles(minute_candles, interval=interval)
    derived = normalize_candles(candles, config=CONFIG).datasets[0]
    repeated = normalize_candles(
        resample_candles(minute_candles, interval=interval), config=CONFIG,
    ).datasets[0]
    unchanged_interval = normalize_candles(
        resample_candles(minute_candles, interval=timedelta(minutes=1)), config=CONFIG,
    ).datasets[0]
    provenance = DatasetProvenance("resample to five minutes", parent_dataset_ids=(dataset.dataset_id,))
    path = tmp_path / "derived.json"
    save_dataset(path, derived, provenance=provenance)
    restored = load_dataset(path)
    assert restored.dataset == derived
    assert restored.dataset.interval == interval
    assert restored.dataset.dataset_id == repeated.dataset_id
    assert unchanged_interval.dataset_id == dataset.dataset_id
    assert restored.dataset.dataset_id != unchanged_interval.dataset_id
    assert restored.manifest.provenance.parent_dataset_ids == (dataset.dataset_id,)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("change", ["none", "same-size", "truncate", "missing-label", "extra-label"])
def test_dat_009_source_verification_detects_changed_bytes_and_wrong_mapping(tmp_path, dataset, change):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"abc")
    provenance = DatasetProvenance("canonical import", (fingerprint_source(raw, name="page"),))
    manifest = save_dataset(tmp_path / "dataset.json", dataset, provenance=provenance)
    paths = {"page": raw}
    if change == "none":
        assert verify_sources(manifest, paths=paths) is None
        return
    if change == "same-size":
        raw.write_bytes(b"abd")
    elif change == "truncate":
        raw.write_bytes(b"a")
    elif change == "missing-label":
        paths = {}
    else:
        paths["extra"] = raw
    with pytest.raises(DatasetStorageError):
        verify_sources(manifest, paths=paths)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("change", ["candle", "quality", "provenance", "identity"])
def test_dat_009_unchecked_artifact_changes_fail_checksum(tmp_path, dataset, change):
    path = tmp_path / "dataset.json"
    save_dataset(path, dataset, provenance=SYNTHETIC)
    payload = json.loads(path.read_text())
    if change == "candle":
        payload["dataset"]["candles"][0]["close"] = "100.75"
    elif change == "quality":
        payload["dataset"]["quality"]["input_count"] = 999
    elif change == "provenance":
        payload["provenance"]["description"] = "changed"
    else:
        payload["dataset"]["dataset_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(DatasetStorageError):
        load_dataset(path)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("change", [
    "candle", "policy", "identity", "gap", "count", "bool-count", "source-rows",
    "reordered", "unknown-format", "unknown-field", "missing-field", "float-price",
    "nan-price", "naive-time", "bool-interval", "provenance",
])
def test_dat_009_semantic_validation_is_required_even_with_recomputed_checksum(tmp_path, dataset, change):
    path = tmp_path / "dataset.json"
    save_dataset(path, dataset, provenance=SYNTHETIC)
    payload = json.loads(path.read_text())
    data = payload["dataset"]
    if change == "candle": data["candles"][0]["close"] = "100.75"
    elif change == "policy": data["config"]["gap_policy"] = "reject"
    elif change == "identity": data["dataset_id"] = "sha256:" + "0" * 64
    elif change == "gap": data["quality"]["gaps"] = [{"start_time": data["candles"][0]["open_time"], "end_time": data["candles"][0]["close_time"], "missing_count": 1}]
    elif change == "count": data["quality"]["output_count"] = 999
    elif change == "bool-count": data["quality"]["input_count"] = True
    elif change == "source-rows": data["quality"]["source_rows"][1] = 1
    elif change == "reordered": data["quality"]["was_reordered"] = True
    elif change == "unknown-format": payload["format"] = "market-trader.dataset-artifact.v999"
    elif change == "unknown-field": data["surprise"] = 1
    elif change == "missing-field": del data["request"]
    elif change == "float-price": data["candles"][0]["volume"] = 0.125
    elif change == "nan-price": data["candles"][0]["volume"] = "NaN"
    elif change == "naive-time": data["candles"][0]["open_time"] = "2026-01-01T00:00:00"
    elif change == "bool-interval": data["interval"] = True
    else: payload["provenance"]["description"] = ""
    rewrite(path, payload)
    with pytest.raises(DatasetStorageError):
        load_dataset(path)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("content", ["{", '{"format":1,"format":2}', '{"number":NaN}', '{"number":Infinity}'])
def test_dat_009_malformed_or_ambiguous_json_is_rejected(tmp_path, content):
    path = tmp_path / "broken.json"
    path.write_text(content)
    with pytest.raises(DatasetStorageError):
        load_dataset(path)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("operation", ["save", "load"])
def test_dat_009_invalid_unicode_is_an_artifact_error(tmp_path, dataset, operation):
    path = tmp_path / "dataset.json"
    if operation == "save":
        with pytest.raises(DatasetStorageError):
            save_dataset(path, dataset, provenance=DatasetProvenance("invalid\ud800"))
        assert not path.exists()
    else:
        save_dataset(path, dataset, provenance=SYNTHETIC)
        payload = json.loads(path.read_text())
        payload["provenance"]["description"] = "invalid\ud800"
        path.write_text(json.dumps(payload))
        with pytest.raises(DatasetStorageError):
            load_dataset(path)


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("change", ["identity", "invalid-candle", "reversed-candles", "list-candles", "count", "expected", "row", "flag"])
def test_dat_009_invalid_in_memory_datasets_are_rejected_before_writing(tmp_path, dataset, change):
    if change == "identity": changed = replace(dataset, dataset_id="sha256:" + "0" * 64)
    elif change == "invalid-candle": changed = replace(dataset, candles=(replace(dataset.candles[0], high=Decimal("0")), *dataset.candles[1:]))
    elif change == "reversed-candles": changed = replace(dataset, candles=tuple(reversed(dataset.candles)))
    elif change == "list-candles": changed = replace(dataset, candles=list(dataset.candles))
    elif change == "count": changed = replace(dataset, quality=replace(dataset.quality, input_count=99))
    elif change == "expected": changed = replace(dataset, quality=replace(dataset.quality, expected_count=5))
    elif change == "row": changed = replace(dataset, quality=replace(dataset.quality, source_rows=(1, 1, 3, 4, 5)))
    else: changed = replace(dataset, quality=replace(dataset.quality, was_reordered=True))
    with pytest.raises(DatasetStorageError):
        save_dataset(tmp_path / "invalid.json", changed, provenance=SYNTHETIC)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.case("DAT-009")
@pytest.mark.parametrize("provenance", [
    DatasetProvenance(""), DatasetProvenance(" padded "), DatasetProvenance("x", sources=[]),
    DatasetProvenance("x", sources=(SourceFingerprint("raw", "bad", 1),)),
    DatasetProvenance("x", sources=(SourceFingerprint("raw", "sha256:" + "a" * 64, -1),)),
    DatasetProvenance("x", parent_dataset_ids=("bad",)),
])
def test_dat_009_invalid_provenance_is_rejected(tmp_path, dataset, provenance):
    with pytest.raises(DatasetStorageError):
        save_dataset(tmp_path / "invalid.json", dataset, provenance=provenance)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.case("CAU-004")
def test_cau_004_existing_target_and_raw_data_are_never_overwritten(tmp_path, dataset):
    path = tmp_path / "existing.json"
    path.write_bytes(b"existing raw or normalized contents")
    with pytest.raises(FileExistsError):
        save_dataset(path, dataset, provenance=SYNTHETIC)
    assert path.read_bytes() == b"existing raw or normalized contents"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.case("CAU-004")
@pytest.mark.parametrize("operation", ["fsync", "link"])
def test_cau_004_failed_publication_leaves_no_partial_artifact(tmp_path, dataset, monkeypatch, operation):
    def fail(*args, **kwargs):
        raise OSError("injected filesystem failure")
    monkeypatch.setattr(storage.os, operation, fail)
    with pytest.raises(OSError, match="injected filesystem failure"):
        save_dataset(tmp_path / "dataset.json", dataset, provenance=SYNTHETIC)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.case("CAU-004")
def test_cau_004_missing_artifact_and_source_remain_technical_errors(tmp_path, dataset):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        fingerprint_source(tmp_path / "missing.csv", name="raw")
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"abc")
    provenance = DatasetProvenance("raw snapshot", (fingerprint_source(raw, name="page"),))
    path = tmp_path / "dataset.json"
    manifest = save_dataset(path, dataset, provenance=provenance)
    raw.unlink()
    moved = tmp_path / "moved.json"
    path.rename(moved)
    assert load_dataset(moved).dataset == dataset  # Loading never follows source labels as paths.
    with pytest.raises(FileNotFoundError):
        verify_sources(manifest, paths={"page": raw})
