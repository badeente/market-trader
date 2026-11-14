import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from market_trader import baseline
from market_trader.baseline import baseline_report, held_out_report
from market_trader.training_data import FEATURE_NAMES, TARGET_NAME, TRAINING_DATA_FORMAT

START = datetime(2026, 1, 1, tzinfo=UTC)


def at(minute):
    return (START + timedelta(minutes=minute)).isoformat()


def row(minute, feature, target):
    return {
        "decision_at": at(minute),
        "entry_at": at(minute),
        "exit_at": at(minute + 2),
        "target_available_at": at(minute + 2),
        "features": [str(feature), "4", "5", "6"],
        "target_return": str(target),
    }


def canonical(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def write_export(path, body):
    checksum = "sha256:" + sha256(canonical(body)).hexdigest()
    path.write_bytes(canonical({**body, "training_data_id": checksum}) + b"\n")


def write_model_report(path, report):
    body = {key: value for key, value in report.items() if key != "report_id"}
    checksum = "sha256:" + sha256(canonical(body)).hexdigest()
    path.write_bytes(canonical({**body, "report_id": checksum}) + b"\n")


@pytest.fixture
def export(tmp_path):
    path = tmp_path / "training-data.json"
    body = {
        "format": TRAINING_DATA_FORMAT,
        "dataset_id": "sha256:" + "a" * 64,
        "dataset_artifact_id": "sha256:" + "b" * 64,
        "instrument": "TEST",
        "interval_microseconds": 60_000_000,
        "target_name": TARGET_NAME,
        "feature_names": list(FEATURE_NAMES),
        "config": {
            "lookback_bars": 2, "horizon_bars": 2, "validation_start": at(10), "test_start": at(20),
        },
        "splits": {
            "train": [row(3, 1, 1), row(4, 3, 5)],
            "validation": [row(10, 100, 6)],
            "test": [row(20, 200, 500)],
        },
    }
    write_export(path, body)
    return path, body


@pytest.mark.case("ML-002")
def test_ridge_uses_training_only_scaling_and_handles_constant_features(export):
    report = baseline_report(export[0])
    model = report["model"]
    assert model["means"] == (2.0, 4.0, 5.0, 6.0)
    assert model["scales"] == (1.0, 1.0, 1.0, 1.0)
    assert model["intercept"] == 3.0
    assert model["coefficients"] == pytest.approx([2 / 1.1, 0, 0, 0])
    assert report["training"]["row_count"] == 2
    assert report["validation"]["row_count"] == 1
    expected_error = abs(3 + 98 * (2 / 1.1) - 6)
    assert report["validation"]["ridge"] == pytest.approx({"mae": expected_error, "rmse": expected_error})
    assert report["validation"]["zero_return"] == {"mae": 6.0, "rmse": 6.0}


@pytest.mark.case("ML-002")
def test_ridge_solves_jointly_for_correlated_features(export):
    path, body = export
    for sample in body["splits"]["train"]:
        sample["features"][1] = sample["features"][0]
    write_export(path, body)

    model = baseline_report(path)["model"]
    assert model["means"] == (2.0, 2.0, 5.0, 6.0)
    assert model["coefficients"] == pytest.approx([2 / 2.1, 2 / 2.1, 0, 0])


@pytest.mark.case("ML-006")
def test_validation_and_test_rows_never_change_the_fit(export):
    path, body = export
    initial = baseline_report(path)
    body["splits"]["test"] = [{"held_out": True}]
    write_export(path, body)
    changed_test = baseline_report(path)
    assert changed_test["training_data_id"] != initial["training_data_id"]
    assert changed_test["model"] == initial["model"]
    assert changed_test["validation"] == initial["validation"]

    body["splits"]["validation"][0]["target_return"] = "1"
    write_export(path, body)
    changed_validation = baseline_report(path)
    assert changed_validation["model"] == initial["model"]
    assert changed_validation["validation"] != initial["validation"]


@pytest.mark.parametrize(("split", "index", "field", "value", "diagnostic"), [
    ("train", 1, "target_available_at", at(10), "train split cutoff"),
    ("validation", 0, "target_available_at", at(20), "validation split cutoff"),
    ("train", 1, "decision_at", at(3), "chronological"),
    ("validation", 0, "features", ["1"], "feature count"),
    ("validation", 0, "target_return", "NaN", "finite decimal"),
])
def test_rejects_invalid_scored_rows(export, split, index, field, value, diagnostic):
    path, body = export
    body["splits"][split][index][field] = value
    write_export(path, body)
    with pytest.raises(ValueError, match=diagnostic):
        baseline_report(path)


def test_cli_writes_reproducible_model_and_metrics_without_overwriting(export, tmp_path):
    path, _ = export

    def run(output):
        return subprocess.run(
            [sys.executable, "-m", "market_trader.cli", "model", "train-baseline",
             "--training-data", str(path), "--output", str(output)],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )

    first, second = tmp_path / "models" / "first.json", tmp_path / "second.json"
    for output in (first, second):
        result = run(output)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["validation"]["row_count"] == 1
    before = first.read_bytes()
    assert second.read_bytes() == before
    payload = json.loads(before)
    assert payload["format"] == "market-trader.ridge-validation.v1"
    report_id = payload.pop("report_id")
    assert report_id == "sha256:" + sha256(canonical(payload)).hexdigest()
    assert payload["training_data_id"] == json.loads(path.read_bytes())["training_data_id"]

    result = run(first)
    assert result.returncode == 1
    assert "save baseline report" in result.stderr
    assert first.read_bytes() == before

    path.write_bytes(path.read_bytes().replace(b'"TEST"', b'"EDITED"'))
    failed_output = tmp_path / "failed.json"
    result = run(failed_output)
    assert result.returncode == 1
    assert "training data checksum mismatch" in result.stderr
    assert not failed_output.exists()


@pytest.fixture
def saved_model(export, tmp_path):
    path = tmp_path / "ridge-validation.json"
    write_model_report(path, baseline_report(export[0]))
    return path


@pytest.mark.case("ML-006")
def test_held_out_scores_use_saved_parameters(export, saved_model, monkeypatch):
    def fail_fit(*args):
        pytest.fail("held-out evaluation attempted to fit a model")

    monkeypatch.setattr(baseline, "_fit_ridge", fail_fit)
    report = held_out_report(saved_model, export[0])
    expected_error = abs(3 + 198 * (2 / 1.1) - 500)
    assert report["test"]["ridge"] == pytest.approx({"mae": expected_error, "rmse": expected_error})
    assert report["test"]["zero_return"] == {"mae": 500.0, "rmse": 500.0}
    assert report["test"]["row_count"] == 1
    assert report["test"]["first_decision_at"] == at(20)
    assert report["model_report_id"] == json.loads(saved_model.read_bytes())["report_id"]


@pytest.mark.case("ML-004")
def test_evaluation_rejects_a_different_training_export(export, saved_model):
    path, body = export
    body["splits"]["test"][0]["target_return"] = "600"
    write_export(path, body)
    with pytest.raises(ValueError, match="training data identifier"):
        held_out_report(saved_model, path)


@pytest.mark.case("ML-004")
@pytest.mark.parametrize(("change", "diagnostic"), [
    ({"model": {"scales": [0.0, 1.0, 1.0, 1.0]}}, "positive scales"),
    ({"model": {"coefficients": [True, 0, 0, 0]}}, "finite number"),
    ({"training": {"last_target_available_at": at(20)}}, "training timestamps"),
    ({"feature_names": list(reversed(FEATURE_NAMES))}, "metadata"),
])
def test_evaluation_rejects_invalid_model_even_with_valid_checksum(export, saved_model, change, diagnostic):
    report = json.loads(saved_model.read_bytes())
    for key, value in change.items():
        report[key] = {**report[key], **value} if isinstance(value, dict) else value
    write_model_report(saved_model, report)
    with pytest.raises(ValueError, match=diagnostic):
        held_out_report(saved_model, export[0])


@pytest.mark.case("ML-001")
def test_evaluation_rejects_held_out_rows_before_test_start(export, saved_model):
    path, body = export
    body["splits"]["test"] = [row(19, 200, 500)]
    write_export(path, body)
    write_model_report(saved_model, baseline_report(path))
    with pytest.raises(ValueError, match="test split cutoff"):
        held_out_report(saved_model, path)


def test_cli_writes_held_out_report_and_rejects_tampering(export, saved_model, tmp_path):
    path, _ = export

    def run(output):
        return subprocess.run(
            [sys.executable, "-m", "market_trader.cli", "model", "evaluate-baseline",
             "--model", str(saved_model), "--training-data", str(path), "--output", str(output)],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )

    output = tmp_path / "reports" / "test.json"
    before = saved_model.read_bytes(), path.read_bytes()
    result = run(output)
    assert result.returncode == 0, result.stderr
    assert (saved_model.read_bytes(), path.read_bytes()) == before
    payload = json.loads(output.read_bytes())
    assert json.loads(result.stdout) == payload
    assert payload["format"] == "market-trader.ridge-test.v1"
    report_id = payload.pop("report_id")
    assert report_id == "sha256:" + sha256(canonical(payload)).hexdigest()

    existing = output.read_bytes()
    result = run(output)
    assert result.returncode == 1
    assert "save held-out report" in result.stderr
    assert output.read_bytes() == existing

    saved_model.write_bytes(saved_model.read_bytes().replace(b'"alpha":0.1', b'"alpha":0.2'))
    failed_output = tmp_path / "failed.json"
    result = run(failed_output)
    assert result.returncode == 1
    assert "model report checksum mismatch" in result.stderr
    assert not failed_output.exists()
