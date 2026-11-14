"""Immutable types for local dataset persistence; no filesystem access."""

from dataclasses import dataclass

from market_trader.normalization import NORMALIZATION_FORMAT, NormalizedDataset

ARTIFACT_FORMAT = "market-trader.dataset-artifact.v1"


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    name: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    description: str
    sources: tuple[SourceFingerprint, ...] = ()
    parent_dataset_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    artifact_id: str
    provenance: DatasetProvenance
    format: str = ARTIFACT_FORMAT
    normalization_format: str = NORMALIZATION_FORMAT


@dataclass(frozen=True, slots=True)
class StoredDataset:
    dataset: NormalizedDataset
    manifest: DatasetManifest


class DatasetStorageError(ValueError):
    """Invalid artifact, inconsistent dataset, or mismatched source evidence."""
