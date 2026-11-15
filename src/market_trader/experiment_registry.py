"""Private filesystem experiment runs, independent of any research backend.

The executor owns creation of ``artifacts/`` and must publish ``summary.json``
and a complete ``manifest.json`` (``artifacts`` maps relative filenames to
``sha256:<hex>``). Registry snapshots are never rewritten. A killed process
may leave a pending/running run: recovery/queues are deliberately out of scope.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
import re
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4


_SUMMARY_FIELDS = (
    "runtime", "training_rows", "validation_rows", "test_metrics_computed",
    "model_report_id", "training_data_id", "validation_dataset_id",
    "comparison_signature", "validation_forecast", "trading",
)


def _json_value(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("JSON object keys must be strings")
        for item in value.values():
            _json_value(item)
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
    elif value is not None and type(value) not in (str, int, bool):
        raise TypeError(f"not a strict JSON value: {type(value).__name__}")


def _encode(value):
    _json_value(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(data):
    return "sha256:" + sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _check_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symbolic link is not allowed: {part}")
    return path


def _read_bytes(path):
    path = _check_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"expected a regular file: {path}")
        return stream.read()


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    _json_value(value)
    return value


def _read_json(path):
    return _decode(_read_bytes(path))


def _write_record(directory, record):
    _check_path(directory / "run.json")
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, prefix=".run-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_encode(record))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "run.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _transition(directory, record, status):
    record["status"] = status
    _write_record(directory, record)
    _check_path(directory / "run.log")
    with (directory / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{_now()} {status}\n")


def _selected_summary(summary):
    if not isinstance(summary, dict) or not set(_SUMMARY_FIELDS) <= summary.keys():
        raise ValueError("summary is missing required result fields")
    _encode(summary)
    if summary["test_metrics_computed"] is not False:
        raise ValueError("summary must not contain held-out test metrics")
    for key in ("training_rows", "validation_rows"):
        if type(summary[key]) is not int or summary[key] < 0:
            raise ValueError(f"summary {key} must be a nonnegative integer")
    for key in ("model_report_id", "training_data_id", "validation_dataset_id"):
        if not isinstance(summary[key], str) or not summary[key].strip():
            raise ValueError(f"summary {key} must be a nonempty identity")
    for key in ("runtime", "validation_forecast", "trading"):
        if not isinstance(summary[key], dict):
            raise ValueError(f"summary {key} must be a JSON object")
    if not isinstance(summary["comparison_signature"], (dict, str)) or not summary["comparison_signature"]:
        raise ValueError("summary comparison_signature must contain comparable conditions")
    return {key: summary[key] for key in _SUMMARY_FIELDS}


def _verify_artifacts(directory, *, record=None, returned=None):
    output = directory / "artifacts"
    try:
        manifest_bytes = _read_bytes(output / "manifest.json")
        if record is not None and _digest(manifest_bytes) != record.get("manifest_sha256"):
            raise ValueError("completion manifest digest mismatch")
        manifest = _decode(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise ValueError("artifact manifest status must be complete")
        files = manifest.get("artifacts")
        if not isinstance(files, dict) or "summary.json" not in files:
            raise ValueError("artifact manifest must include summary.json")
        for name, expected in files.items():
            if (not isinstance(name, str) or not name or "\\" in name
                    or name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))
                    or name == "manifest.json"):
                raise ValueError(f"unsafe artifact filename: {name!r}")
            if _digest(_read_bytes(output / name)) != expected:
                raise ValueError(f"artifact checksum mismatch: {name}")
        actual = set()
        for path in output.rglob("*"):
            _check_path(path)
            if not path.is_dir():
                if not path.is_file():
                    raise ValueError(f"artifact is not a regular file: {path}")
                actual.add(path.relative_to(output).as_posix())
        if actual != set(files) | {"manifest.json"}:
            raise ValueError("artifact files do not match manifest inventory")
        summary = _read_json(output / "summary.json")
        selected = _selected_summary(summary)
        if returned is not None and _encode(summary) != _encode(returned):
            raise ValueError("executor summary does not match summary.json identity")
        if record is not None and _encode(selected) != _encode(record.get("summary")):
            raise ValueError("record summary does not match artifact summary identity")
        return selected, _digest(manifest_bytes)
    except OSError as error:
        raise ValueError(f"missing or unreadable executor artifact/manifest: {error}") from error


def register_run(root: Path, payload: dict, *, executor: Callable[[Path], dict],
                 parent_id: str | None = None) -> dict:
    """Claim a new run, snapshot the opaque config, execute and persist results."""
    if type(payload) is not dict:
        raise TypeError("config payload must be a JSON object")
    snapshot = _encode(payload)
    payload = _decode(snapshot)
    root = _check_path(root)
    if parent_id is not None:
        read_run(root, parent_id)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_id = "run-" + uuid4().hex
    directory = root / run_id
    directory.mkdir(mode=0o700)
    with (directory / "config.json").open("xb") as stream:
        stream.write(snapshot)
    record = {
        "run_id": run_id, "parent_id": parent_id, "status": "pending",
        "created_at": _now(), "started_at": None, "finished_at": None,
        "name": payload.get("name"), "hypothesis": payload.get("hypothesis"),
        "seed": payload.get("seed"), "config_sha256": _digest(snapshot),
        "config_path": "config.json", "artifacts_path": "artifacts", "log_path": "run.log",
    }
    _transition(directory, record, "pending")
    record["started_at"] = _now()
    _transition(directory, record, "running")
    try:
        summary = executor(directory / "artifacts")
        _selected_summary(summary)
        record["summary"], record["manifest_sha256"] = _verify_artifacts(directory, returned=summary)
        _verified_record(root, run_id)
        record["finished_at"] = _now()
        _transition(directory, record, "completed")
    except BaseException as error:
        record["finished_at"] = _now()
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            _transition(directory, record, "failed")
            with (directory / "run.log").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record["error"]) + "\n")
        except BaseException as persistence_error:
            error.add_note(f"Could not persist failed run: {persistence_error}")
        raise
    return record


def read_run(root: Path, run_id: str) -> dict:
    """Read a run record and its verified immutable snapshot."""
    record, _ = _verified_record(root, run_id)
    return record


def _verified_record(root, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"run-[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid run identifier; expected run-<32 lowercase hex>")
    directory = _check_path(Path(root) / run_id)
    record = _read_json(directory / "run.json")
    if not isinstance(record, dict):
        raise ValueError("run record must be a JSON object")
    if record.get("run_id") != run_id:
        raise ValueError("run record ID does not match directory identity")
    snapshot = _read_bytes(directory / "config.json")
    if _digest(snapshot) != record.get("config_sha256"):
        raise ValueError("config snapshot digest mismatch")
    if record.get("status") == "completed":
        _verify_artifacts(directory, record=record)
    return record, _decode(snapshot)


def list_runs(root: Path) -> list[dict]:
    """List per-run records; no mutable centralized index is required."""
    root = _check_path(root)
    if not root.exists():
        return []
    runs = []
    for path in sorted(root.iterdir()):
        if re.fullmatch(r"run-[0-9a-f]{32}", path.name) is None:
            continue
        _check_path(path)
        # An exclusively claimed directory may not have published its first
        # atomic record yet. It cannot be advertised as completed.
        if not (path / "run.json").exists():
            _check_path(path / "run.json")
            continue
        runs.append(read_run(root, path.name))
    return runs


def load_config_snapshot(root: Path, run_id: str) -> dict:
    """Return a detached snapshot suitable for registering a new child run."""
    _, snapshot = _verified_record(root, run_id)
    return snapshot


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            field = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                difference = {"field": field, "left": left.get(key), "right": right.get(key)}
                difference["left_missing" if key not in left else "right_missing"] = True
                result.append(difference)
            else:
                result.extend(_differences(left[key], right[key], field))
        return result
    if _encode(left) == _encode(right):
        return []
    return [{"field": prefix, "left": left, "right": right}]


def compare_runs(root: Path, left_id: str, right_id: str) -> dict:
    """Compare verified completed runs; signatures define comparable conditions.

    Differences use recursive dotted field paths with explicit missing markers.
    Names/hypotheses are separated from model/config parameters. Compatibility
    is not an endorsement or a claim of statistical significance.
    """
    left, left_config = _verified_record(root, left_id)
    right, right_config = _verified_record(root, right_id)
    if left.get("status") != "completed" or right.get("status") != "completed":
        raise ValueError("comparison requires two completed runs")
    differences = _differences(left["summary"]["comparison_signature"], right["summary"]["comparison_signature"])
    config_differences = _differences(left_config, right_config)
    metadata = {"name", "hypothesis"}
    return {
        "left_id": left_id, "right_id": right_id, "compatible": not differences,
        "reason": ("equal comparison_signature; comparable conditions" if not differences
                   else "different comparison_signature; conditions are not directly comparable"),
        "comparison_differences": differences, "config_differences": config_differences,
        "metadata_differences": [d for d in config_differences if d["field"].split(".")[0] in metadata],
        "parameter_differences": [d for d in config_differences if d["field"].split(".")[0] not in metadata],
        "left_summary": left["summary"], "right_summary": right["summary"],
    }