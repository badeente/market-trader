"""Local versioned dataset artifacts with provenance and integrity verification."""

import os
import hashlib
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from market_trader._dataset_codec import decode_record, encode, json_bytes, parse_json, record_object
from market_trader.artifacts import (
    ARTIFACT_FORMAT, DatasetManifest, DatasetProvenance, DatasetStorageError,
    SourceFingerprint, StoredDataset,
)
from market_trader.normalization import NORMALIZATION_FORMAT, CandleKey, NormalizedDataset, normalize_candles

_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _label(value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DatasetStorageError("expected a nonempty label without padding")


def _hash(value: str) -> None:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise DatasetStorageError("expected a lowercase sha256 identifier")


def _validate_provenance(provenance: DatasetProvenance) -> None:
    _label(provenance.description)
    names = set()
    for source in provenance.sources:
        _label(source.name)
        _hash(source.sha256)
        if source.size_bytes < 0 or source.name in names:
            raise DatasetStorageError("source sizes must be nonnegative and source names unique")
        names.add(source.name)
    for parent in provenance.parent_dataset_ids:
        _hash(parent)
    if len(set(provenance.parent_dataset_ids)) != len(provenance.parent_dataset_ids):
        raise DatasetStorageError("parent dataset identifiers must be unique")


def _validate_dataset(dataset: NormalizedDataset) -> None:
    _hash(dataset.dataset_id)
    try:
        result = normalize_candles(
            dataset.candles, config=dataset.config,
            requests=None if dataset.request is None else (dataset.request,),
        )
    except ValueError as error:
        raise DatasetStorageError(f"dataset violates normalization rules: {error}") from error
    if len(result.datasets) != 1:
        raise DatasetStorageError("artifact must contain exactly one normalized series")
    expected = result.datasets[0]
    if (dataset.dataset_id != expected.dataset_id or dataset.instrument != expected.instrument
            or dataset.interval != expected.interval or dataset.candles != expected.candles):
        raise DatasetStorageError("normalized dataset identity, ordering or series metadata mismatch")
    quality = dataset.quality
    if (quality.output_count != len(dataset.candles)
            or quality.input_count != quality.output_count + len(quality.duplicates)
            or quality.gaps != expected.quality.gaps
            or quality.expected_count != expected.quality.expected_count):
        raise DatasetStorageError("quality counts or gaps do not match normalized data")
    rows = quality.source_rows
    if (len(rows) != len(dataset.candles) or len(set(rows)) != len(rows)
            or any(row <= 0 for row in rows)
            or quality.was_reordered != (rows != tuple(sorted(rows)))):
        raise DatasetStorageError("invalid original source rows or reordering flag")
    by_row = dict(zip(rows, dataset.candles))
    removed = set()
    previous_row = 0
    for duplicate in quality.duplicates:
        candle = by_row.get(duplicate.kept_row)
        if (candle is None or duplicate.removed_row in by_row or duplicate.removed_row in removed
                or duplicate.removed_row <= duplicate.kept_row or duplicate.removed_row <= previous_row
                or duplicate.key != CandleKey(dataset.instrument, dataset.interval, candle.open_time)):
            raise DatasetStorageError("invalid duplicate audit or retained candle reference")
        removed.add(duplicate.removed_row)
        previous_row = duplicate.removed_row


def _checksum(body: dict) -> str:
    return "sha256:" + hashlib.sha256(json_bytes(body)).hexdigest()


def fingerprint_source(path: Path, *, name: str) -> SourceFingerprint:
    """Hash raw bytes without modifying them; store only an explicit logical name."""
    _label(name)
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return SourceFingerprint(name, "sha256:" + digest.hexdigest(), size)


def publish_new_file(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically in an existing directory."""
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".market-trader-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        # A same-filesystem hard link publishes complete bytes atomically without overwriting.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_dataset(
    path: Path, dataset: NormalizedDataset, *, provenance: DatasetProvenance,
) -> DatasetManifest:
    """Validate and atomically publish a new artifact; never replace an existing path."""
    if not isinstance(dataset, NormalizedDataset) or not isinstance(provenance, DatasetProvenance):
        raise DatasetStorageError("expected a NormalizedDataset and DatasetProvenance")
    # The codec checks immutable shapes and exact scalar types on both boundaries.
    data = encode(dataset)
    origin = encode(provenance)
    _validate_dataset(decode_record(data, NormalizedDataset))
    _validate_provenance(decode_record(origin, DatasetProvenance))
    body = {"format": ARTIFACT_FORMAT, "normalization_format": NORMALIZATION_FORMAT,
            "dataset": data, "provenance": origin}
    artifact_id = _checksum(body)
    publish_new_file(path, json_bytes({**body, "artifact_id": artifact_id}) + b"\n")
    return DatasetManifest(dataset.dataset_id, artifact_id, provenance)


def load_dataset(path: Path) -> StoredDataset:
    """Verify artifact checksum and normalized identity before returning typed data."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except UnicodeError as error:
        raise DatasetStorageError("artifact must be UTF-8 JSON") from error
    payload = record_object(parse_json(content), ("format", "normalization_format", "artifact_id", "dataset", "provenance"))
    if payload["format"] != ARTIFACT_FORMAT:
        raise DatasetStorageError("unsupported dataset artifact format")
    if payload["normalization_format"] != NORMALIZATION_FORMAT:
        raise DatasetStorageError("unsupported normalization format")
    _hash(payload["artifact_id"])
    body = {key: value for key, value in payload.items() if key != "artifact_id"}
    if payload["artifact_id"] != _checksum(body):
        raise DatasetStorageError("artifact checksum mismatch")
    dataset = decode_record(payload["dataset"], NormalizedDataset)
    provenance = decode_record(payload["provenance"], DatasetProvenance)
    _validate_dataset(dataset)
    _validate_provenance(provenance)
    return StoredDataset(dataset, DatasetManifest(dataset.dataset_id, payload["artifact_id"], provenance))


def verify_sources(manifest: DatasetManifest, *, paths: Mapping[str, Path]) -> None:
    """Check recorded source bytes using an explicit logical-name/path mapping."""
    _validate_provenance(manifest.provenance)
    if set(paths) != {source.name for source in manifest.provenance.sources}:
        raise DatasetStorageError("source path mapping must match the recorded logical names")
    for source in manifest.provenance.sources:
        if fingerprint_source(paths[source.name], name=source.name) != source:
            raise DatasetStorageError(f"source bytes no longer match: {source.name}")
