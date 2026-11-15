"""Synthetic-only contract checks for the versioned AP1 target."""
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json

import pytest

from market_trader import baseline, training_data as td
from market_trader._dataset_codec import encode, json_bytes
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.domain import Candle
from market_trader.normalization import NormalizationConfig, normalize_candles

START = datetime(2024, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def at(minute):
    return START + minute * MINUTE


def bars(minutes=range(18)):
    return tuple(Candle(
        instrument="TEST", interval=MINUTE, open_time=at(i), close_time=at(i + 1),
        available_at=at(i + 1), open=Decimal(100 + i), high=Decimal(110 + i),
        low=Decimal(90 + i), close=Decimal(101 + i), volume=Decimal(1),
    ) for i in minutes)


def dataset(candles=None, calendar="continuous"):
    return normalize_candles(bars() if candles is None else candles,
                             config=NormalizationConfig(calendar)).datasets[0]


def config():
    return td.TrainingConfig(3, 2, at(8), at(13))


def test_open_target_hand_calculation_extra_bar_and_legacy_unchanged():
    assert getattr(td, "OPEN_TARGET_NAME", None) == "forward_return_next_open_to_horizon_open"
    assert td.OPEN_TRAINING_DATA_FORMAT == "market-trader.training-data.v2"
    old = td.build_training_data(dataset(), config())
    new = td.build_open_training_data(dataset(), config())
    row = new.train[0]
    assert row.features == old.train[0].features
    assert row.target_return == Decimal(2) / 103
    assert old.train[0].target_return == Decimal(2) / 103
    changed = list(bars())
    changed[4] = replace(changed[4], close=Decimal(109))
    changed[5] = replace(changed[5], open=Decimal(107))
    altered = td.build_open_training_data(dataset(changed), config()).train[0]
    assert altered.target_return == Decimal(4) / 103
    assert altered.features == row.features
    assert td.build_training_data(dataset(changed), config()).train[0].target_return == Decimal(6) / 103
    assert (row.decision_at, row.entry_at, row.exit_at, row.target_available_at) == (
        at(3), at(3), at(5), at(6))
    assert new.test[-1].decision_at == at(15)
    assert old.test[-1].decision_at == at(16)
    assert len(asdict(config())) == 4


def test_include_test_false_never_builds_reserved_features(monkeypatch):
    real_features = td.candle_features
    decisions = []

    def checked_features(history):
        decisions.append(history[-1].close_time)
        assert history[-1].close_time < config().test_start
        return real_features(history)

    monkeypatch.setattr(td, "candle_features", checked_features)
    result = td.build_open_training_data(dataset(), config(), include_test=False)
    assert result.test == ()
    assert decisions
    assert result.train and result.validation
    # The reserved split need not even be present in the supplied candles.
    truncated = td.build_open_training_data(dataset(bars(range(13))), config(), include_test=False)
    assert truncated == result
    assert [r.decision_at for r in result.train] == [at(3), at(4)]
    assert [r.decision_at for r in result.validation] == [at(8), at(9)]
    with pytest.raises(ValueError, match="validation split"):
        td.build_open_training_data(dataset(bars(range(9))), config(), include_test=False)


def write_checked(path, body, key):
    body = {k: v for k, v in body.items() if k != key}
    path.write_bytes(json_bytes({**body, key: "sha256:" + sha256(json_bytes(body)).hexdigest()}))


def export(path, *, legacy=False, include_test=False):
    data = dataset()
    rows = td.build_training_data(data, config()) if legacy else td.build_open_training_data(
        data, config(), include_test=include_test)
    body = {
        "format": td.TRAINING_DATA_FORMAT if legacy else td.OPEN_TRAINING_DATA_FORMAT,
        "dataset_id": data.dataset_id, "dataset_artifact_id": "sha256:" + "b" * 64,
        "instrument": "TEST", "interval_microseconds": 60_000_000,
        "config": encode(config()), "feature_names": list(td.FEATURE_NAMES),
        "target_name": td.TARGET_NAME if legacy else td.OPEN_TARGET_NAME,
        "splits": {name: encode(getattr(rows, name)) for name in ("train", "validation", "test")},
    }
    write_checked(path, body, "training_data_id")
    return body, rows


def test_new_ridge_fit_reload_and_legacy_default(tmp_path):
    path, model_path = tmp_path / "training.json", tmp_path / "model.json"
    body, rows = export(path)
    report = baseline.baseline_report(path)
    assert report["format"] == baseline.OPEN_BASELINE_FORMAT == "market-trader.ridge-validation.v2"
    assert report["target_name"] == td.OPEN_TARGET_NAME
    assert report["training"]["row_count"] == len(rows.train)
    assert report["validation"]["row_count"] == len(rows.validation)
    assert baseline.baseline_report(path) == report
    write_checked(model_path, report, "report_id")
    saved = baseline.load_ridge_model(model_path)
    assert saved.target_name == td.OPEN_TARGET_NAME
    predictions = tuple(saved.model.predict(row.features) for row in rows.validation)
    assert report["validation"]["ridge"] == baseline._error_metrics(tuple(
        pred - float(row.target_return) for pred, row in zip(predictions, rows.validation)))
    export(path, legacy=True)
    legacy_report = baseline.baseline_report(path)
    assert legacy_report["format"] == baseline.BASELINE_FORMAT
    write_checked(model_path, legacy_report, "report_id")
    legacy_model = baseline.load_ridge_model(model_path)
    assert legacy_model.target_name == td.TARGET_NAME
    assert baseline.SavedRidgeModel(
        legacy_model.report_id, legacy_model.model, legacy_model.instrument, legacy_model.interval,
        legacy_model.config, legacy_model.last_target_available_at).target_name == td.TARGET_NAME


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_open_ridge_rejects_label_available_before_exit_bar_close(tmp_path, split):
    path, model_path = tmp_path / "data.json", tmp_path / "model.json"
    body, _ = export(path, include_test=True)
    row = body["splits"][split][0]
    row["target_available_at"] = row["exit_at"]
    write_checked(path, body, "training_data_id")
    if split == "test":
        write_checked(model_path, baseline.baseline_report(path), "report_id")
        with pytest.raises(ValueError, match="target horizon"):
            baseline.held_out_evaluation(model_path, path)
    else:
        with pytest.raises(ValueError, match="target horizon"):
            baseline.baseline_report(path)


@pytest.mark.parametrize("legacy", [False, True])
def test_validation_evaluation_is_frozen_and_validation_only(tmp_path, monkeypatch, legacy):
    path, model_path = tmp_path / "data.json", tmp_path / "model.json"
    body, rows = export(path, legacy=legacy)
    fitted = baseline.baseline_report(path)
    # Invalid training/test rows prove the evaluator never decodes or scores them.
    body["splits"]["train"] = [{"not_scored": True}]
    body["splits"]["test"] = [{"not_scored": True}]
    write_checked(path, body, "training_data_id")
    fitted["training_data_id"] = json.loads(path.read_bytes())["training_data_id"]
    write_checked(model_path, fitted, "report_id")
    before = path.read_bytes(), model_path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("validation evaluation refit or called held-out evaluation")

    monkeypatch.setattr(baseline, "_fit_ridge", forbidden)
    monkeypatch.setattr(baseline, "held_out_evaluation", forbidden)
    report, points = baseline.validation_evaluation(model_path, path)
    assert report["format"] == "market-trader.ridge-validation-evaluation.v1"
    assert "test" not in report
    assert report["model_report_id"] == baseline.load_ridge_model(model_path).report_id
    assert report["training_data_id"] == fitted["training_data_id"]
    assert report["validation"] == {
        **fitted["validation"], "first_decision_at": rows.validation[0].decision_at.isoformat(),
        "last_decision_at": rows.validation[-1].decision_at.isoformat(),
    }
    assert points == [{"at": row.decision_at.isoformat(),
                       "predicted": baseline.load_ridge_model(model_path).model.predict(row.features),
                       "actual": float(row.target_return)} for row in rows.validation]
    assert report["report_id"] == "sha256:" + sha256(json_bytes(
        {k: v for k, v in report.items() if k != "report_id"})).hexdigest()
    assert baseline.validation_evaluation(model_path, path) == (report, points)
    assert before == (path.read_bytes(), model_path.read_bytes())


@pytest.mark.parametrize("legacy", [False, True])
def test_model_reader_rejects_crossed_version_target_pair(tmp_path, legacy):
    path, model_path = tmp_path / "data.json", tmp_path / "model.json"
    export(path, legacy=legacy)
    report = baseline.baseline_report(path)
    report["target_name"] = td.OPEN_TARGET_NAME if legacy else td.TARGET_NAME
    write_checked(model_path, report, "report_id")
    with pytest.raises(ValueError, match="target"):
        baseline._read_model_report(model_path)
    with pytest.raises(ValueError, match="target"):
        baseline.load_ridge_model(model_path)


@pytest.mark.parametrize("starts", [(0, 6, 12), (0, 10, 20)])
def test_open_windows_require_exit_bar_inside_same_session(starts):
    calendar = SessionCalendar("TEST", at(0), at(starts[-1] + 6), tuple(
        TradingSession(at(start), at(start + 6)) for start in starts))
    candles = bars(i for start in starts for i in range(start, start + 6))
    cfg = td.TrainingConfig(3, 2, at(starts[1]) + timedelta(microseconds=1),
                            at(starts[2]) + timedelta(microseconds=1))
    result = td.build_open_training_data(dataset(candles, calendar), cfg)
    for split, start in zip((result.train, result.validation, result.test), starts):
        assert len(split) == 1
        assert split[0].decision_at == at(start + 3)
        assert split[0].exit_at == at(start + 5)
        assert split[0].target_available_at == at(start + 6)


def test_open_feature_availability_and_all_future_availability_purge():
    candles = list(bars())
    candles[1] = replace(candles[1], available_at=at(4))
    result = td.build_open_training_data(dataset(candles), config())
    assert [row.decision_at for row in result.train] == [at(4)]
    candles = list(bars())
    candles[6] = replace(candles[6], available_at=at(8))
    result = td.build_open_training_data(dataset(candles), config())
    assert [row.decision_at for row in result.train] == [at(3)]
    candles = list(bars())
    candles[4] = replace(candles[4], available_at=at(7))
    result = td.build_open_training_data(dataset(candles), config())
    assert result.train[0].target_available_at == at(7)  # not just the exit candle
    candles[3] = replace(candles[3], available_at=at(8))
    with pytest.raises(ValueError, match="train split"):
        td.build_open_training_data(dataset(candles), config())


def test_open_missing_tail_gaps_and_nonpositive_prices():
    with pytest.raises(ValueError, match="test split"):
        td.build_open_training_data(dataset(bars(range(13))), config())
    candles = bars()
    with pytest.raises(ValueError, match="gap"):
        td.build_open_training_data(dataset(candles[:3] + candles[4:]), config())
    candles = list(candles)
    candles[0] = replace(candles[0], open=Decimal(0), low=Decimal(0))
    with pytest.raises(ValueError, match="positive"):
        td.build_open_training_data(dataset(candles), config())


@pytest.mark.parametrize("legacy", [False, True])
def test_training_reader_rejects_crossed_version_target_pairs(tmp_path, legacy):
    path = tmp_path / "data.json"
    body, _ = export(path, legacy=legacy)
    body["target_name"] = td.OPEN_TARGET_NAME if legacy else td.TARGET_NAME
    write_checked(path, body, "training_data_id")
    with pytest.raises(ValueError, match="target"):
        baseline.baseline_report(path)


def test_open_heldout_requires_actual_nonempty_test(tmp_path):
    path, model_path = tmp_path / "data.json", tmp_path / "model.json"
    export(path)
    write_checked(model_path, baseline.baseline_report(path), "report_id")
    with pytest.raises(ValueError, match="test split must be a nonempty array"):
        baseline.held_out_evaluation(model_path, path)
    _, rows = export(path, include_test=True)
    write_checked(model_path, baseline.baseline_report(path), "report_id")
    report, points = baseline.held_out_evaluation(model_path, path)
    assert report["test"]["row_count"] == len(rows.test)
    assert report["target_name"] == td.OPEN_TARGET_NAME
    assert len(points) == len(rows.test)


@pytest.mark.parametrize("change,diagnostic", [
    ("training_checksum", "training data checksum"),
    ("model_checksum", "model report checksum"),
    ("identifier", "training data identifier"),
    ("metadata", "metadata"),
    ("availability", "target horizon"),
    ("purge", "validation split cutoff"),
    ("before_start", "validation split cutoff"),
    ("empty", "validation split must be a nonempty"),
])
def test_validation_evaluation_checks_integrity_metadata_and_boundaries(tmp_path, change, diagnostic):
    path, model_path = tmp_path / "data.json", tmp_path / "model.json"
    body, _ = export(path)
    report = baseline.baseline_report(path)
    if change == "training_checksum":
        write_checked(path, body, "training_data_id")
        path.write_bytes(path.read_bytes().replace(b'"TEST"', b'"EDITED"'))
    elif change == "identifier":
        report["training_data_id"] = "sha256:" + "a" * 64
    elif change == "metadata":
        report["instrument"] = "OTHER"
    elif change in ("availability", "purge", "before_start", "empty"):
        row = body["splits"]["validation"][0]
        if change == "availability":
            row["target_available_at"] = row["exit_at"]
        elif change == "purge":
            row["target_available_at"] = config().test_start.isoformat()
        elif change == "before_start":
            row["decision_at"] = at(7).isoformat()
        else:
            body["splits"]["validation"] = []
        write_checked(path, body, "training_data_id")
        report["training_data_id"] = json.loads(path.read_bytes())["training_data_id"]
    write_checked(model_path, report, "report_id")
    if change == "model_checksum":
        model_path.write_bytes(model_path.read_bytes().replace(b'"alpha":0.1', b'"alpha":0.2'))
    with pytest.raises(ValueError, match=diagnostic):
        baseline.validation_evaluation(model_path, path)
