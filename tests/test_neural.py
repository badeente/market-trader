"""Synthetic-only AP3 tests; inference remains usable without Torch."""
import builtins
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import importlib.util
import json

import pytest

from market_trader._dataset_codec import json_bytes
from market_trader.training_data import FEATURE_NAMES, OPEN_TARGET_NAME, OPEN_TRAINING_DATA_FORMAT


def seal(body, key="report_id"):
    body = {k: v for k, v in body.items() if k != key}
    return {**body, key: "sha256:" + sha256(json_bytes(body)).hexdigest()}


def write(path, body):
    path.write_bytes(json_bytes(body))
    return path


def settings():
    return {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "batch_size": 128, "max_epochs": 100, "patience": 10, "min_delta": 1e-6,
            "seed": 0, "device": "cpu", "threads": 1, "dtype": "float64", "deterministic": True}


def saved_body():
    return seal({
        "format": "market-trader.mlp-validation.v1", "training_data_id": "sha256:" + "c" * 64,
        "dataset_id": "sha256:" + "a" * 64, "dataset_artifact_id": "sha256:" + "b" * 64,
        "instrument": "SYNTHETIC", "interval_microseconds": 900_000_000,
        "feature_names": list(FEATURE_NAMES), "target_name": OPEN_TARGET_NAME,
        "config": {"lookback_bars": 8, "horizon_bars": 2,
                   "validation_start": "2024-01-01T00:00:00+00:00", "test_start": "2025-01-01T00:00:00+00:00"},
        "model": {"kind": "mlp", "architecture": [4, 16, 8, 1], "parameter_count": 225,
                  "means": [1., 2., 3., 4.], "scales": [1., 1., 1., 1.],
                  "target_mean": 2., "target_scale": 3.,
                  "layers": [{"weights": [[0.1] * n for _ in range(m)], "biases": [0.2] * m}
                             for n, m in [(4, 16), (16, 8), (8, 1)]]},
        "training": {"row_count": 2, "first_decision_at": "2021-01-01T00:00:00+00:00",
                     "last_target_available_at": "2023-12-31T00:00:00+00:00",
                     "fit_settings": settings(), "runtime_versions": {"python": "3.14.9", "torch": "2.15.0+cpu", "numpy": "2.5.4"},
                     "epoch_history": [{"epoch": i, "training_mse": 1. / i, "validation_mse": 1. / i} for i in range(1, 101)],
                     "best_epoch": 100, "stopping_reason": "max_epochs"},
        "validation": {"row_count": 1, "mlp": {"mae": 1., "rmse": 1.},
                       "zero_return": {"mae": 1., "rmse": 1.}}})


def test_loader_predicts_pure_python_and_checks_checksum(tmp_path):
    from market_trader.neural import MLP_FORMAT, SavedMLPModel, load_mlp_model
    body = saved_body()
    path = write(tmp_path / "model.json", body)
    loaded = load_mlp_model(path)
    assert isinstance(loaded, SavedMLPModel)
    assert MLP_FORMAT == body["format"]
    assert loaded.report_id == body["report_id"]
    assert loaded.target_name == OPEN_TARGET_NAME
    assert loaded.instrument == "SYNTHETIC"
    assert loaded.interval == timedelta(minutes=15)
    # Standardized input zeros: hidden1=.2, hidden2=.52, output=.616.
    assert loaded.model.predict(tuple(map(Decimal, [1, 2, 3, 4]))) == pytest.approx(3.848, abs=1e-14)
    body["model"]["target_mean"] = 5.
    write(path, body)
    with pytest.raises(ValueError, match="checksum"):
        load_mlp_model(path)


@pytest.mark.parametrize(("keys", "value"), [
    (("format",), "market-trader.mlp-validation.v2"),
    (("target_name",), "forward_return_next_open_to_horizon_close"),
    (("feature_names",), ["wrong"] * 4),
    (("instrument",), " bad "),
    (("interval_microseconds",), 60_000_000),
    (("config", "lookback_bars"), 2),
    (("config", "test_start"), "2026-01-01T00:00:00+00:00"),
    (("model", "kind"), "ridge"),
    (("model", "architecture"), [4, 8, 16, 1]),
    (("model", "parameter_count"), 224),
    (("model", "means"), [0.] * 3),
    (("model", "scales"), [1., 1., 0., 1.]),
    (("model", "target_scale"), -1.),
    (("model", "target_mean"), True),
    (("model", "layers", 0, "weights", 0), [1.] * 3),
    (("model", "layers", 1, "biases"), [0.] * 7),
    (("model", "layers", 2, "weights", 0, 0), "nan"),
    (("model", "unknown"), 1),
    (("unknown",), 1),
    (("training", "unknown"), 1),
    (("training", "row_count"), True),
    (("training", "last_target_available_at"), "2024-01-01T00:00:00+00:00"),
    (("training", "fit_settings", "max_epochs"), 101),
    (("training", "epoch_history", 0, "validation_mse"), -1.),
    (("training", "best_epoch"), 101),
    (("validation", "mlp", "mae"), -1.),
    (("training_data_id",), "not-a-hash"),
])
def test_loader_rejects_resigned_invalid_schema(tmp_path, keys, value):
    from market_trader.neural import load_mlp_model
    body = saved_body()
    target = body
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = write(tmp_path / "bad.json", seal(body))
    with pytest.raises(ValueError):
        load_mlp_model(path)


def test_inference_rejects_nonfinite_features_and_never_imports(tmp_path, monkeypatch):
    from market_trader.neural import load_mlp_model
    path = write(tmp_path / "model.json", saved_body())
    def forbidden(*args, **kwargs):
        raise AssertionError("loader must not import modules dynamically")
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", forbidden)
        loaded = load_mlp_model(path)
        prediction = loaded.model.predict((Decimal(1),) * 4)
    assert prediction == pytest.approx(3.08)
    with pytest.raises(ValueError):
        loaded.model.predict((Decimal("NaN"),) * 4)
    with pytest.raises(ValueError):
        loaded.model.predict((Decimal("Infinity"),) * 4)
    with pytest.raises(ValueError):
        loaded.model.predict((Decimal(1),) * 3)


requires_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="training requires optional Torch")


def sample(at, index):
    x = (index % 17 - 8) / 8
    return {"decision_at": at.isoformat(), "entry_at": at.isoformat(),
            "exit_at": (at + timedelta(minutes=30)).isoformat(),
            "target_available_at": (at + timedelta(minutes=45)).isoformat(),
            "features": [str(x), str(x * x), "4", str(-x)],
            "target_return": str(0.01 * x + 0.02)}


@pytest.fixture
def synthetic(tmp_path):
    body = {k: v for k, v in saved_body().items() if k in (
        "dataset_id", "dataset_artifact_id", "instrument", "interval_microseconds", "config", "feature_names", "target_name")}
    body.update(format=OPEN_TRAINING_DATA_FORMAT, splits={
        "train": [sample(datetime(2021, 1, 1, tzinfo=UTC) + timedelta(hours=i), i) for i in range(257)],
        "validation": [sample(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i), i) for i in range(19)],
        "test": []})
    path = tmp_path / "synthetic.json"
    write(path, seal(body, "training_data_id"))
    return path, body


@requires_torch
def test_real_training_scalers_native_parity_reproducibility_and_global_restore(synthetic, tmp_path, monkeypatch):
    import torch
    from statistics import fmean, pstdev
    from market_trader.neural import mlp_report, load_mlp_model
    path, body = synthetic
    networks = []
    original = torch.nn.Sequential
    def capture(*args, **kwargs):
        network = original(*args, **kwargs)
        networks.append(network)
        return network
    monkeypatch.setattr(torch.nn, "Sequential", capture)
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_dtype = torch.get_default_dtype()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    first = mlp_report(path)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.get_num_threads() == before_threads
    assert torch.get_default_dtype() == before_dtype
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == before_warn_only
    assert first == mlp_report(path)
    assert sum(parameter.numel() for parameter in networks[0].parameters()) == 225
    assert all(parameter.dtype == torch.float64 and parameter.device.type == "cpu"
               for parameter in networks[0].parameters())
    assert first["format"] == "market-trader.mlp-validation.v1"
    assert first == seal(first)
    assert first["training"]["fit_settings"] == settings()
    assert first["training"]["runtime_versions"]["torch"] == torch.__version__
    assert first["training"]["row_count"] == 257
    history = first["training"]["epoch_history"]
    assert 1 < len(history) <= 100
    assert history[-1]["training_mse"] < history[0]["training_mse"]
    assert first["training"]["best_epoch"] == min(history, key=lambda e: e["validation_mse"])["epoch"]
    columns = list(zip(*[list(map(float, r["features"])) for r in body["splits"]["train"]]))
    assert first["model"]["means"] == pytest.approx(list(map(fmean, columns)))
    assert first["model"]["scales"] == pytest.approx([pstdev(c) or 1. for c in columns])
    ys = [float(r["target_return"]) for r in body["splits"]["train"]]
    assert first["model"]["target_mean"] == fmean(ys)
    assert first["model"]["target_scale"] == pstdev(ys)
    model_path = write(tmp_path / "model.json", first)
    saved = load_mlp_model(model_path)
    features = [tuple(map(Decimal, r["features"])) for r in body["splits"]["validation"]]
    exported = [saved.model.predict(f) for f in features]
    data = first["model"]
    native_x = torch.tensor([[float(v) for v in f] for f in features], dtype=torch.float64)
    native_x = (native_x - torch.tensor(data["means"], dtype=torch.float64)) / torch.tensor(data["scales"], dtype=torch.float64)
    with torch.no_grad():
        standardized = networks[0](native_x).flatten().tolist()
    native = [v * data["target_scale"] + data["target_mean"] for v in standardized]
    assert exported == pytest.approx(native, abs=1e-14, rel=1e-13)
    targets = [float(r["target_return"]) for r in body["splits"]["validation"]]
    assert first["validation"]["mlp"]["mae"] == fmean(abs(p - y) for p, y in zip(exported, targets))
    best_loss = fmean(((p - y) / data["target_scale"]) ** 2 for p, y in zip(native, targets))
    assert best_loss == pytest.approx(history[first["training"]["best_epoch"] - 1]["validation_mse"], abs=1e-13)


@pytest.mark.parametrize(("keys", "value"), [
    (("splits", "test"), [{"reserved": "must not be inspected"}]),
    (("splits", "test"), None),
    (("format",), "market-trader.training-data.v1"),
    (("target_name",), "forward_return_next_open_to_horizon_close"),
])
def test_training_rejects_non_open_or_nonempty_test_before_backend(synthetic, monkeypatch, keys, value):
    from market_trader.neural import mlp_report
    path, body = synthetic
    target = body
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    # v1/close is internally valid baseline data, but forbidden for this MLP.
    if keys == ("format",):
        body["target_name"] = "forward_return_next_open_to_horizon_close"
    write(path, seal(body, "training_data_id"))
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("invalid artifact reached optional training backend")
        return original(name, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", guard)
        with pytest.raises(ValueError):
            mlp_report(path)


@pytest.mark.parametrize("seed", [True, -1, 0.5, 2**63, "0"])
def test_invalid_seed_rejected_before_backend(synthetic, monkeypatch, seed):
    import market_trader.neural as neural
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid seed reached trainer")
    monkeypatch.setattr(neural, "_fit", forbidden)
    with pytest.raises(ValueError, match="seed"):
        neural.mlp_report(synthetic[0], seed=seed)


def linked_model(tmp_path, training_path, body):
    saved = saved_body()
    saved["training_data_id"] = json.loads(training_path.read_text())["training_data_id"]
    saved["training"].update(row_count=len(body["splits"]["train"]),
                              first_decision_at=body["splits"]["train"][0]["decision_at"],
                              last_target_available_at=body["splits"]["train"][-1]["target_available_at"])
    return write(tmp_path / "linked-model.json", seal(saved))


def test_validation_evaluation_frozen_predictor_identity_metrics_and_no_backend(synthetic, tmp_path, monkeypatch):
    import market_trader.neural as neural
    from statistics import fmean
    training_path, body = synthetic
    model_path = linked_model(tmp_path, training_path, body)
    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must never import modules or fit")
    monkeypatch.setattr(neural, "_fit", forbidden)
    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", forbidden)
        report, points = neural.mlp_validation_evaluation(model_path, training_path)
        model = neural.load_mlp_model(model_path)
    assert report["format"] == "market-trader.mlp-validation-evaluation.v1"
    assert report["report_id"] == seal(report)["report_id"]
    assert report["model_report_id"] == model.report_id
    assert report["training_data_id"] == json.loads(training_path.read_text())["training_data_id"]
    assert report["validation"]["row_count"] == len(points) == 19
    assert report["validation"]["first_decision_at"] == points[0]["at"]
    assert report["validation"]["last_decision_at"] == points[-1]["at"]
    assert [p["at"] for p in points] == [r["decision_at"] for r in body["splits"]["validation"]]
    for point, row in zip(points, body["splits"]["validation"]):
        assert point["predicted"] == model.model.predict(tuple(map(Decimal, row["features"])))
        assert point["actual"] == float(row["target_return"])
    assert report["validation"]["mlp"]["mae"] == fmean(abs(p["predicted"] - p["actual"]) for p in points)
    # An entirely valid but different export must not reuse a fitted artifact.
    body["splits"]["validation"][0]["target_return"] = "0.9"
    write(training_path, seal(body, "training_data_id"))
    with pytest.raises(ValueError, match="identifier"):
        neural.mlp_validation_evaluation(model_path, training_path)


@pytest.mark.parametrize(("keys", "value"), [
    (("instrument",), "OTHER"),
    (("training", "row_count"), 2),
    (("training", "first_decision_at"), "2021-01-02T00:00:00+00:00"),
    (("training", "last_target_available_at"), "2023-12-31T00:00:00+00:00"),
])
def test_evaluation_rejects_resigned_training_metadata_mismatch(synthetic, tmp_path, keys, value):
    from market_trader.neural import mlp_validation_evaluation
    path, body = synthetic
    model_path = linked_model(tmp_path, path, body)
    saved = json.loads(model_path.read_text())
    target = saved
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    write(model_path, seal(saved))
    with pytest.raises(ValueError, match="metadata"):
        mlp_validation_evaluation(model_path, path)


@requires_torch
def test_early_stop_restores_actual_best_and_validation_never_fits_scalers(synthetic, tmp_path, monkeypatch):
    import torch
    from statistics import fmean
    from market_trader.neural import mlp_report, mlp_validation_evaluation
    path, body = synthetic
    ordinary = mlp_report(path)
    for row in body["splits"]["validation"]:
        row["target_return"] = str(Decimal("0.04") - Decimal(row["target_return"]))
    write(path, seal(body, "training_data_id"))
    networks, snapshots = [], []
    original_network = torch.nn.Sequential
    original_step = torch.optim.AdamW.step
    def capture(*args, **kwargs):
        net = original_network(*args, **kwargs)
        networks.append(net)
        return net
    count = 0
    def step(optimizer, *args, **kwargs):
        nonlocal count
        assert all(group["lr"] == 0.001 and group["weight_decay"] == 0.0001 for group in optimizer.param_groups)
        result = original_step(optimizer, *args, **kwargs)
        count += 1
        if count % 3 == 0:
            snapshots.append({name: tensor.clone() for name, tensor in networks[-1].state_dict().items()})
        return result
    monkeypatch.setattr(torch.nn, "Sequential", capture)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    report = mlp_report(path)
    for field in ("means", "scales", "target_mean", "target_scale"):
        assert report["model"][field] == ordinary["model"][field]
    training = report["training"]
    assert training["stopping_reason"] == "early_stopping"
    assert training["best_epoch"] < len(training["epoch_history"]) < 100
    actual_best = snapshots[training["best_epoch"] - 1]
    assert all(torch.equal(actual_best[k], v) for k, v in networks[-1].state_dict().items())
    assert any(not torch.equal(actual_best[k], v) for k, v in snapshots[-1].items())
    model_path = write(tmp_path / "early.json", report)
    evaluation, points = mlp_validation_evaluation(model_path, path)
    assert evaluation["validation"]["mlp"] == report["validation"]["mlp"]
    assert report["validation"]["mlp"]["mae"] == fmean(abs(p["predicted"] - p["actual"]) for p in points)


def test_evaluation_rejects_nonobject_training_block(synthetic, tmp_path):
    from market_trader.neural import mlp_validation_evaluation
    path, body = synthetic
    model_path = linked_model(tmp_path, path, body)
    saved = json.loads(model_path.read_text())
    saved["training"] = []
    write(model_path, seal(saved))
    with pytest.raises(ValueError):
        mlp_validation_evaluation(model_path, path)


@pytest.mark.parametrize("bad", [10**400, float("inf"), float("nan"), -float("inf")])
def test_loader_rejects_numbers_outside_finite_float_range(tmp_path, bad):
    from market_trader.neural import load_mlp_model
    body = saved_body()
    body["model"]["layers"][0]["weights"][0][0] = bad
    # Raw JSON permits exercising NaN/Infinity rejection before checksum parsing.
    if isinstance(bad, int):
        path = write(tmp_path / "overflow.json", seal(body))
    else:
        path = tmp_path / "nonfinite.json"
        path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        load_mlp_model(path)


@pytest.mark.parametrize(("split", "index", "field", "value"), [
    ("train", 0, "decision_at", "2020-01-01T00:00:00+00:00"),
    ("train", 1, "decision_at", "2021-01-01T00:00:00+00:00"),
    ("validation", 0, "decision_at", "2025-01-01T00:00:00+00:00"),
    ("validation", 0, "target_available_at", "2025-01-01T00:00:00+00:00"),
    ("train", 0, "target_available_at", "2021-01-01T00:30:00+00:00"),
    ("validation", 0, "target_available_at", "2024-01-01T00:30:00+00:00"),
    ("train", 0, "features", ["1"]),
    ("validation", 0, "target_return", "NaN"),
])
def test_temporal_and_schema_guards_precede_fitting(synthetic, monkeypatch, split, index, field, value):
    import market_trader.neural as neural
    path, body = synthetic
    body["splits"][split][index][field] = value
    write(path, seal(body, "training_data_id"))
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid split reached fitting")
    monkeypatch.setattr(neural, "_fit", forbidden)
    with pytest.raises(ValueError):
        neural.mlp_report(path)


@requires_torch
def test_training_failure_restores_rng_and_settings(synthetic, monkeypatch):
    import torch
    from market_trader.neural import mlp_report
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    dtype = torch.get_default_dtype()
    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 1
        assert torch.are_deterministic_algorithms_enabled()
        raise RuntimeError("synthetic optimizer failure")
    monkeypatch.setattr(torch.optim.AdamW, "step", fail)
    with pytest.raises(RuntimeError, match="synthetic optimizer failure"):
        mlp_report(synthetic[0])
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert torch.get_default_dtype() == dtype
