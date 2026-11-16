"""Filesystem contracts using synthetic artifacts only; never execute research data."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import stat

import pytest


def registry():
    from market_trader import experiment_registry
    return experiment_registry


def payload(**changes):
    return {"version": 1, "name": "synthetic", "hypothesis": "fixture only", "seed": 7,
            "source": {"sha256": "synthetic-source"}, "model": {"alpha": "0.1"},
            "features": {"lookback": 8}, "execution": {"spread": "2.0"}, **changes}


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return "sha256:" + sha256(data).hexdigest()


def synthetic_executor(output):
    assert not output.exists(), "registry must leave artifact creation to executor"
    output.mkdir()
    summary = {
        "runtime": {"code_id": "fixture-code", "python": "fixture-python", "dependencies": {}},
        "training_rows": 3, "validation_rows": 2, "test_metrics_computed": False,
        "model_report_id": "fixture-model", "training_data_id": "fixture-training",
        "validation_dataset_id": "fixture-validation",
        "comparison_signature": {"source": "fixture", "costs": {"spread": "2.0"}},
        "validation_forecast": {"mae": "0.01"}, "trading": {"ridge": {"trades": 0}},
        "data_audit": {"large": "not copied into run.json"},
    }
    (output / "summary.json").write_bytes(encode(summary))
    (output / "model.json").write_bytes(encode({"synthetic": True}))
    manifest = {"status": "complete", "artifacts": {
        path.name: digest(path.read_bytes()) for path in output.iterdir()}}
    (output / "manifest.json").write_bytes(encode(manifest))
    return summary


def test_register_persists_private_snapshot_lifecycle_and_small_summary(tmp_path):
    api = registry()
    root = tmp_path / "runs"
    original = payload()

    def execute(output):
        current = api.read_run(root, output.parent.name)
        assert current["status"] == "running"
        assert current["started_at"] is not None and current["finished_at"] is None
        assert (output.parent / "config.json").read_bytes() == encode(original)
        return synthetic_executor(output)

    run = api.register_run(root, original, executor=execute)
    assert re.fullmatch(r"run-[0-9a-f]{32}", run["run_id"])
    directory = root / run["run_id"]
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert run["status"] == "completed" and run["parent_id"] is None
    assert run["config_sha256"] == digest((directory / "config.json").read_bytes())
    assert run["manifest_sha256"] == digest((directory / "artifacts/manifest.json").read_bytes())
    assert "data_audit" not in run["summary"]
    assert run["summary"]["runtime"]["code_id"] == "fixture-code"
    for key in ("created_at", "started_at", "finished_at"):
        assert datetime.fromisoformat(run[key]).utcoffset().total_seconds() == 0
    log = (directory / "run.log").read_text()
    assert all(status in log for status in ("pending", "running", "completed"))
    assert api.read_run(root, run["run_id"]) == run
    assert api.list_runs(root) == [run]
    assert api.load_config_snapshot(root, run["run_id"]) == original
    original["model"]["alpha"] = "99"
    assert api.load_config_snapshot(root, run["run_id"])["model"]["alpha"] == "0.1"


def test_allocation_hook_precedes_all_publication_and_preserves_executor_api(tmp_path):
    api = registry()
    claimed = []
    def allocated(directory):
        assert directory.is_dir()
        assert list(directory.iterdir()) == []
        assert api.list_runs(tmp_path) == []
        claimed.append(directory)
    def execute(output):
        assert output == claimed[0] / 'artifacts'
        assert api.read_run(tmp_path, output.parent.name)['status'] == 'running'
        return synthetic_executor(output)
    run = api.register_run(tmp_path, payload(), executor=execute, on_allocated=allocated)
    assert run['status'] == 'completed'
    assert len(claimed) == 1 and claimed[0].name == run['run_id']


def test_failed_allocation_hook_cannot_publish_unowned_unfinished_run(tmp_path):
    api = registry()
    claimed = []
    def allocated(directory):
        claimed.append(directory)
        raise RuntimeError('ownership persistence failed')
    with pytest.raises(RuntimeError, match='ownership persistence failed'):
        api.register_run(tmp_path, payload(), executor=synthetic_executor, on_allocated=allocated)
    assert len(claimed) == 1
    assert list(claimed[0].iterdir()) == []
    assert api.list_runs(tmp_path) == []


@pytest.mark.parametrize("error", [ValueError("executor failed"), KeyboardInterrupt(), SystemExit(9)])
def test_executor_failure_persists_status_log_and_reraises_original(tmp_path, error):
    api = registry()

    def execute(output):
        raise error

    with pytest.raises(type(error)) as caught:
        api.register_run(tmp_path, payload(), executor=execute)
    assert caught.value is error
    run, = api.list_runs(tmp_path)
    assert run["status"] == "failed"
    assert run["finished_at"] is not None
    assert run["error"]["type"] == type(error).__name__
    assert "failed" in (tmp_path / run["run_id"] / "run.log").read_text()
    assert type(error).__name__ in (tmp_path / run["run_id"] / "run.log").read_text()


def test_parent_replay_is_new_run_and_parent_remains_unchanged(tmp_path):
    api = registry()
    first = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    directory = tmp_path / first["run_id"]
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    replay = api.register_run(tmp_path, api.load_config_snapshot(tmp_path, first["run_id"]),
                              executor=synthetic_executor, parent_id=first["run_id"])
    assert replay["parent_id"] == first["run_id"]
    assert replay["run_id"] != first["run_id"]
    assert replay["config_sha256"] == first["config_sha256"]
    assert before == {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("bad_id", ["../escape", "run-abc", "run-" + "A" * 32,
                                      "run-" + "0" * 32, "/absolute"])
def test_invalid_parent_fails_before_allocating(tmp_path, bad_id):
    with pytest.raises((ValueError, FileNotFoundError)):
        registry().register_run(tmp_path, payload(), executor=synthetic_executor, parent_id=bad_id)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["read_run", "load_config_snapshot"])
def test_snapshot_digest_is_verified(tmp_path, operation):
    api = registry()
    run = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    (tmp_path / run["run_id"] / "config.json").write_bytes(encode(payload(seed=8)))
    with pytest.raises(ValueError, match="config.*(digest|checksum)"):
        getattr(api, operation)(tmp_path, run["run_id"])
    with pytest.raises(ValueError, match="config.*(digest|checksum)"):
        api.list_runs(tmp_path)


@pytest.mark.parametrize("operation", ["read_run", "load_config_snapshot"])
@pytest.mark.parametrize("run_id", ["../escape", "run-" + "A" * 32, "bad", "run-" + "0" * 31])
def test_malformed_ids_are_rejected(tmp_path, operation, run_id):
    with pytest.raises(ValueError, match="run.*(ID|identifier)"):
        getattr(registry(), operation)(tmp_path, run_id)


def test_record_id_must_match_directory(tmp_path):
    api = registry()
    run = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    record_path = tmp_path / run["run_id"] / "run.json"
    record_path.write_bytes(encode({**run, "run_id": "run-" + "0" * 32}))
    with pytest.raises(ValueError, match="(ID|identity)"):
        api.read_run(tmp_path, run["run_id"])


@pytest.mark.parametrize("damage", ["stub", "no_manifest", "incomplete", "missing", "wrong_hash",
                                     "unlisted", "summary_mismatch", "no_summary", "empty_artifacts"])
def test_incomplete_or_inconsistent_executor_output_never_completes(tmp_path, damage):
    api = registry()

    def execute(output):
        if damage == "stub":
            return {}
        summary = synthetic_executor(output)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if damage == "no_manifest":
            manifest_path.unlink()
        elif damage == "incomplete":
            manifest["status"] = "running"
        elif damage == "missing":
            (output / "model.json").unlink()
        elif damage == "wrong_hash":
            (output / "model.json").write_text("tampered")
        elif damage == "unlisted":
            (output / "unlisted.json").write_text("{}")
        elif damage == "summary_mismatch":
            summary["training_rows"] = 900
        elif damage == "no_summary":
            del manifest["artifacts"]["summary.json"]
        elif damage == "empty_artifacts":
            manifest["artifacts"] = {}
        if manifest_path.exists():
            manifest_path.write_bytes(encode(manifest))
        return summary

    with pytest.raises(ValueError, match="(artifact|manifest|summary)"):
        api.register_run(tmp_path, payload(), executor=execute)
    run, = api.list_runs(tmp_path)
    assert run["status"] == "failed"


@pytest.mark.parametrize("target", ["manifest.json", "summary.json", "model.json"])
def test_completed_artifacts_are_verified_on_read_and_compare_prerequisites(tmp_path, target):
    api = registry()
    run = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    path = tmp_path / run["run_id"] / "artifacts" / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="(digest|checksum)"):
        api.read_run(tmp_path, run["run_id"])

@pytest.mark.parametrize("target", ["root", "run", "config.json", "run.json", "artifacts", "manifest.json", "model.json"])
def test_symlinks_are_never_followed(tmp_path, target):
    api = registry()
    root = tmp_path / "runs"
    run = api.register_run(root, payload(), executor=synthetic_executor)
    directory = root / run["run_id"]
    path = {"root": root, "run": directory, "artifacts": directory / "artifacts"}.get(target)
    if path is None:
        path = directory / (target if target in ("config.json", "run.json") else "artifacts/" + target)
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="symbolic|symlink"):
        api.read_run(root, run["run_id"])
    if target in ("root", "run"):
        with pytest.raises(ValueError, match="symbolic|symlink"):
            api.register_run(root, payload(), executor=synthetic_executor, parent_id=run["run_id"])


@pytest.mark.parametrize("name", ["../outside", "/etc/passwd", "sub/../../outside", "./model.json",
                                  "a//b", "a\\b", "manifest.json", ""])
def test_manifest_paths_cannot_escape_or_alias(tmp_path, name):
    def execute(output):
        summary = synthetic_executor(output)
        manifest = json.loads((output / "manifest.json").read_bytes())
        manifest["artifacts"][name] = digest(b"anything")
        (output / "manifest.json").write_bytes(encode(manifest))
        return summary

    with pytest.raises(ValueError, match="unsafe artifact"):
        registry().register_run(tmp_path, payload(), executor=execute)


def test_symlink_root_fails_before_allocating(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic|symlink"):
        registry().register_run(link, payload(), executor=synthetic_executor)
    assert list(real.iterdir()) == []

def executor_with_summary(**changes):
    def execute(output):
        summary = synthetic_executor(output)
        summary.update(changes)
        (output / "summary.json").write_bytes(encode(summary))
        manifest = json.loads((output / "manifest.json").read_bytes())
        manifest["artifacts"]["summary.json"] = digest((output / "summary.json").read_bytes())
        (output / "manifest.json").write_bytes(encode(manifest))
        return summary
    return execute


def test_compare_compatible_runs_separates_metadata_and_parameter_differences(tmp_path):
    api = registry()
    left = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    right = api.register_run(tmp_path, payload(name="second", hypothesis="changed", model={"alpha": "0.2"}),
                             executor=synthetic_executor)
    comparison = api.compare_runs(tmp_path, left["run_id"], right["run_id"])
    assert comparison["compatible"] is True
    assert comparison["comparison_differences"] == []
    assert comparison["parameter_differences"] == [{"field": "model.alpha", "left": "0.1", "right": "0.2"}]
    assert {d["field"] for d in comparison["metadata_differences"]} == {"name", "hypothesis"}
    assert {d["field"] for d in comparison["config_differences"]} == {"model.alpha", "name", "hypothesis"}
    assert comparison["left_summary"] == left["summary"]
    assert comparison["right_summary"] == right["summary"]


def test_compare_incompatible_conditions_reports_nested_field_differences(tmp_path):
    api = registry()
    left = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    right = api.register_run(tmp_path, payload(), executor=executor_with_summary(
        comparison_signature={"source": "different", "costs": {"spread": "3.0"}, "extra": None}))
    comparison = api.compare_runs(tmp_path, left["run_id"], right["run_id"])
    assert comparison["compatible"] is False
    assert "comparison_signature" in comparison["reason"]
    assert {d["field"] for d in comparison["comparison_differences"]} == {"costs.spread", "source", "extra"}
    extra = next(d for d in comparison["comparison_differences"] if d["field"] == "extra")
    assert extra["left_missing"] is True and extra["right"] is None


def test_compare_requires_completed_verified_runs(tmp_path):
    api = registry()
    complete = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    def fail(output):
        raise ValueError("fixture")
    with pytest.raises(ValueError):
        api.register_run(tmp_path, payload(), executor=fail)
    failed = next(run for run in api.list_runs(tmp_path) if run["status"] == "failed")
    with pytest.raises(ValueError, match="completed"):
        api.compare_runs(tmp_path, complete["run_id"], failed["run_id"])
    (tmp_path / complete["run_id"] / "artifacts/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        api.compare_runs(tmp_path, complete["run_id"], complete["run_id"])

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), (1, 2), {1: "integer key"}, Path("x")])
def test_config_requires_strict_json_without_coercion_before_allocation(tmp_path, bad):
    with pytest.raises((ValueError, TypeError), match="JSON|json|finite|compliant"):
        registry().register_run(tmp_path, payload(model={"bad": bad}), executor=synthetic_executor)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("changes", [{"test_metrics_computed": True}, {"training_rows": True},
    {"validation_rows": -1}, {"model_report_id": ""}, {"comparison_signature": {}},
    {"runtime": None}, {"validation_forecast": []}, {"trading": []}])
def test_summary_requires_validation_only_typed_identity_fields(tmp_path, changes):
    with pytest.raises(ValueError, match="summary"):
        registry().register_run(tmp_path, payload(), executor=executor_with_summary(**changes))
    assert registry().list_runs(tmp_path)[0]["status"] == "failed"


@pytest.mark.parametrize("damage", ["nonfinite", "duplicate"])
def test_manifest_json_rejects_nonfinite_values_and_duplicate_keys(tmp_path, damage):
    def execute(output):
        summary = synthetic_executor(output)
        text = (output / "manifest.json").read_bytes().rstrip()
        extra = b',"x":NaN}' if damage == "nonfinite" else b',"status":"complete"}'
        (output / "manifest.json").write_bytes(text[:-1] + extra)
        return summary
    with pytest.raises(ValueError, match="JSON|duplicate|finite"):
        registry().register_run(tmp_path, payload(), executor=execute)


def test_snapshot_tampering_during_execution_never_completes(tmp_path):
    def execute(output):
        (output.parent / "config.json").write_bytes(encode(payload(seed=999)))
        return synthetic_executor(output)
    with pytest.raises(ValueError, match="config.*digest"):
        registry().register_run(tmp_path, payload(), executor=execute)
    directory, = tmp_path.iterdir()
    assert json.loads((directory / "run.json").read_bytes())["status"] == "failed"


def test_concurrent_claims_keep_every_run_separate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    api = registry()
    barrier = Barrier(8)
    def register(index):
        def execute(output):
            barrier.wait(timeout=30)
            return synthetic_executor(output)
        return api.register_run(tmp_path, payload(seed=index), executor=execute)
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(register, range(8)))
    assert len({run["run_id"] for run in runs}) == 8
    assert len(api.list_runs(tmp_path)) == 8
    assert {api.load_config_snapshot(tmp_path, run["run_id"])["seed"] for run in runs} == set(range(8))


def test_uuid_collision_cannot_overwrite_existing_run(tmp_path, monkeypatch):
    from uuid import UUID
    api = registry()
    first = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    monkeypatch.setattr(api, "uuid4", lambda: UUID(first["run_id"][4:]))
    with pytest.raises(FileExistsError):
        api.register_run(tmp_path, payload(seed=8), executor=synthetic_executor)
    assert api.read_run(tmp_path, first["run_id"]) == first
    assert len(api.list_runs(tmp_path)) == 1


def test_list_ignores_uninitialized_claims_and_unrelated_entries(tmp_path):
    api = registry()
    (tmp_path / "run-".__add__("0" * 32)).mkdir()
    (tmp_path / "unrelated").write_text("not a run")
    assert api.list_runs(tmp_path) == []


def test_atomic_record_publication_exposes_pending_running_and_completed(tmp_path, monkeypatch):
    api = registry()
    replace = api.os.replace
    seen = []
    def observe(source, target):
        new = json.loads(Path(source).read_bytes())
        assert Path(source).parent == Path(target).parent
        if Path(target).exists():
            old = json.loads(Path(target).read_bytes())
            assert old["status"] in ("pending", "running")
        result = replace(source, target)
        visible = api.read_run(tmp_path, new["run_id"])
        seen.append(visible["status"])
        return result
    monkeypatch.setattr(api.os, "replace", observe)
    api.register_run(tmp_path, payload(), executor=synthetic_executor)
    assert seen == ["pending", "running", "completed"]
    directory, = tmp_path.iterdir()
    assert not list(directory.glob(".run-*"))


@pytest.mark.parametrize("target", ["manifest.json", "model.json"])
def test_executor_symbolic_artifact_is_failed_not_completed(tmp_path, target):
    def execute(output):
        summary = synthetic_executor(output)
        path = output / target
        original = output.parent / ("original-" + target)
        path.rename(original)
        path.symlink_to(original)
        return summary
    with pytest.raises(ValueError, match="symbolic"):
        registry().register_run(tmp_path, payload(), executor=execute)
    run, = registry().list_runs(tmp_path)
    assert run["status"] == "failed"


def test_nested_artifact_files_are_supported_and_hashed(tmp_path):
    def execute(output):
        summary = synthetic_executor(output)
        (output / "nested").mkdir()
        (output / "nested/data.bin").write_bytes(b"synthetic nested artifact")
        manifest = json.loads((output / "manifest.json").read_bytes())
        manifest["artifacts"]["nested/data.bin"] = digest((output / "nested/data.bin").read_bytes())
        (output / "manifest.json").write_bytes(encode(manifest))
        return summary
    run = registry().register_run(tmp_path, payload(), executor=execute)
    assert registry().read_run(tmp_path, run["run_id"])["status"] == "completed"


@pytest.mark.parametrize("operation", ["read_run", "load_config_snapshot", "list_runs"])
def test_record_summary_identity_cannot_be_replaced(tmp_path, operation):
    api = registry()
    run = api.register_run(tmp_path, payload(), executor=synthetic_executor)
    run["summary"]["model_report_id"] = "different-model"
    (tmp_path / run["run_id"] / "run.json").write_bytes(encode(run))
    with pytest.raises(ValueError, match="summary.*identity"):
        if operation == "list_runs":
            api.list_runs(tmp_path)
        else:
            getattr(api, operation)(tmp_path, run["run_id"])
