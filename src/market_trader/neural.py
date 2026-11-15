"""Frozen AP3 MLP: safe JSON and stdlib inference; Torch is training-only."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from math import fsum, isfinite
from pathlib import Path
from statistics import fmean, pstdev
import platform
import re

from market_trader._dataset_codec import decode_record, json_bytes, parse_json, record_object
from market_trader.baseline import (
    _TRAINING_METADATA, _checked_split, _error_metrics, _number, _read_training_data,
)
from market_trader.training_data import (
    FEATURE_NAMES, OPEN_TARGET_NAME, OPEN_TRAINING_DATA_FORMAT, TrainingConfig,
)

MLP_FORMAT = "market-trader.mlp-validation.v1"
_ARCHITECTURE = [4, 16, 8, 1]
_START = datetime(2021, 1, 1, tzinfo=UTC)
_VALIDATION_START = datetime(2024, 1, 1, tzinfo=UTC)
_TEST_START = datetime(2025, 1, 1, tzinfo=UTC)
_INTERVAL = 900_000_000
_SETTINGS = {
    "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
    "batch_size": 128, "max_epochs": 100, "patience": 10, "min_delta": 1e-6,
    "device": "cpu", "threads": 1, "dtype": "float64", "deterministic": True,
}
_TRAINING_FIELDS = ("row_count", "first_decision_at", "last_target_available_at", "fit_settings",
                    "runtime_versions", "epoch_history", "best_epoch", "stopping_reason")


@dataclass(frozen=True, slots=True)
class MLPModel:
    """Immutable, float64-exported dense layers with standard-library inference."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    target_mean: float
    target_scale: float
    layers: tuple

    def predict(self, features: tuple[Decimal, ...]) -> float:
        numbers = tuple(float(x) for x in features)
        if len(numbers) != 4 or not all(isfinite(x) for x in numbers):
            raise ValueError("model requires four finite features")
        try:
            values = tuple((x - mean) / scale for x, mean, scale in
                           zip(numbers, self.means, self.scales, strict=True))
            for index, (weights, biases) in enumerate(self.layers):
                values = tuple(bias + fsum(w * x for w, x in zip(row, values, strict=True))
                               for row, bias in zip(weights, biases, strict=True))
                if not all(isfinite(x) for x in values):
                    raise ValueError("nonfinite dense layer output")
                if index < 2:
                    values = tuple(max(0., x) for x in values)
            prediction = self.target_mean + self.target_scale * values[0]
        except (OverflowError, ValueError) as error:
            raise ValueError("model prediction exceeds the floating-point range") from error
        if not isfinite(prediction):
            raise ValueError("model prediction exceeds the floating-point range")
        return prediction


@dataclass(frozen=True, slots=True)
class SavedMLPModel:
    """Verified predictor and the metadata required for offline inference."""

    report_id: str
    model: MLPModel
    instrument: str
    interval: timedelta
    config: TrainingConfig
    last_target_available_at: datetime
    target_name: str = OPEN_TARGET_NAME


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _identifier(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("expected a sha256 identifier")


def _config(config: TrainingConfig, interval: object) -> None:
    if (type(interval) is not int or interval != _INTERVAL or config.lookback_bars != 8
            or config.horizon_bars != 2 or config.validation_start != _VALIDATION_START
            or config.test_start != _TEST_START):
        raise ValueError("MLP requires the frozen eight-bar/30-minute 2021–2024 configuration")


def _model_number(value: object, field: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"model {field} must contain a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"model {field} exceeds the floating-point range") from error
    if not isfinite(result):
        raise ValueError(f"model {field} must contain a finite number")
    return result


def _vector(value: object, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"model {field} has an invalid shape")
    return tuple(_model_number(x, field) for x in value)


def _decode_model(value: object) -> MLPModel:
    data = record_object(value, ("kind", "architecture", "parameter_count", "means", "scales",
                                 "target_mean", "target_scale", "layers"))
    if (data["kind"] != "mlp" or data["architecture"] != _ARCHITECTURE
            or not isinstance(data["architecture"], list)
            or any(type(x) is not int for x in data["architecture"])
            or type(data["parameter_count"]) is not int or data["parameter_count"] != 225):
        raise ValueError("unsupported MLP architecture; expected 225 parameters")
    means = _vector(data["means"], 4, "means")
    scales = _vector(data["scales"], 4, "scales")
    target_mean = _model_number(data["target_mean"], "target_mean")
    target_scale = _model_number(data["target_scale"], "target_scale")
    if min((*scales, target_scale)) <= 0:
        raise ValueError("model requires positive scales")
    if not isinstance(data["layers"], list) or len(data["layers"]) != 3:
        raise ValueError("model must contain three dense layers")
    layers = []
    for layer, inputs, outputs in zip(data["layers"], _ARCHITECTURE[:-1], _ARCHITECTURE[1:], strict=True):
        layer = record_object(layer, ("weights", "biases"))
        if not isinstance(layer["weights"], list) or len(layer["weights"]) != outputs:
            raise ValueError("model weights have an invalid shape")
        layers.append((tuple(_vector(row, inputs, "weights") for row in layer["weights"]),
                       _vector(layer["biases"], outputs, "biases")))
    return MLPModel(means, scales, target_mean, target_scale, tuple(layers))


def _validate_training(value: object) -> datetime:
    training = record_object(value, _TRAINING_FIELDS)
    _positive_integer(training["row_count"], "training row_count")
    stamps = []
    for field in ("first_decision_at", "last_target_available_at"):
        if not isinstance(training[field], str):
            raise ValueError("training timestamps must be UTC strings")
        stamp = datetime.fromisoformat(training[field])
        if stamp.utcoffset() != timedelta(0):
            raise ValueError("training timestamps must be UTC")
        stamps.append(stamp)
    if not _START <= stamps[0] <= stamps[1] < _VALIDATION_START:
        raise ValueError("training timestamps must lie in 2021–2023")
    settings = record_object(training["fit_settings"], (*_SETTINGS, "seed"))
    if type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    for key, expected in _SETTINGS.items():
        if type(settings[key]) is not type(expected) or settings[key] != expected:
            raise ValueError(f"unsupported frozen fit setting: {key}")
    versions = record_object(training["runtime_versions"], ("python", "torch", "numpy"))
    if any(not isinstance(x, str) or not x or x != x.strip() for x in versions.values()):
        raise ValueError("runtime versions must be nonempty strings")
    history = training["epoch_history"]
    if not isinstance(history, list) or not 1 <= len(history) <= 100:
        raise ValueError("epoch history must contain 1–100 epochs")
    losses = []
    significant_best, stale = float("inf"), 0
    for index, epoch in enumerate(history, 1):
        epoch = record_object(epoch, ("epoch", "training_mse", "validation_mse"))
        if type(epoch["epoch"]) is not int or epoch["epoch"] != index:
            raise ValueError("epoch history must be sequential")
        for key in ("training_mse", "validation_mse"):
            if _model_number(epoch[key], key) < 0:
                raise ValueError("losses must be nonnegative")
        loss = epoch["validation_mse"]
        losses.append(loss)
        if loss < significant_best - _SETTINGS["min_delta"]:
            significant_best, stale = loss, 0
        else:
            stale += 1
        if stale >= 10 and index != len(history):
            raise ValueError("history continues past early stopping")
    best = _positive_integer(training["best_epoch"], "best_epoch")
    if best != losses.index(min(losses)) + 1:
        raise ValueError("best_epoch must identify the actual best validation checkpoint")
    expected_reason = "early_stopping" if stale >= 10 else "max_epochs"
    if training["stopping_reason"] != expected_reason or (stale < 10 and len(history) != 100):
        raise ValueError("stopping reason does not match the epoch history")
    return stamps[1]


def _validate_metrics(value: object) -> None:
    data = record_object(value, ("row_count", "mlp", "zero_return"))
    _positive_integer(data["row_count"], "validation row_count")
    for field in ("mlp", "zero_return"):
        errors = record_object(data[field], ("mae", "rmse"))
        if any(_model_number(x, field) < 0 for x in errors.values()):
            raise ValueError("metrics must be nonnegative")


def _read_report(path: Path) -> dict:
    saved = record_object(parse_json(path.read_text(encoding="utf-8"), allow_floats=True),
                          ("format", "report_id", "training_data_id", "model", "training", "validation", *_TRAINING_METADATA))
    if saved["format"] != MLP_FORMAT:
        raise ValueError("unsupported MLP report format")
    body = {key: value for key, value in saved.items() if key != "report_id"}
    if saved["report_id"] != "sha256:" + sha256(json_bytes(body)).hexdigest():
        raise ValueError("model report checksum mismatch")
    return saved


def _decode_saved(saved: dict) -> SavedMLPModel:
    if saved["feature_names"] != list(FEATURE_NAMES) or saved["target_name"] != OPEN_TARGET_NAME:
        raise ValueError("model features or target do not match the inference schema")
    for field in ("report_id", "training_data_id", "dataset_id", "dataset_artifact_id"):
        _identifier(saved[field])
    instrument = saved["instrument"]
    if not isinstance(instrument, str) or not instrument or instrument != instrument.strip():
        raise ValueError("model instrument must be nonempty and unpadded")
    config = decode_record(saved["config"], TrainingConfig)
    _config(config, saved["interval_microseconds"])
    last = _validate_training(saved["training"])
    _validate_metrics(saved["validation"])
    return SavedMLPModel(saved["report_id"], _decode_model(saved["model"]), instrument,
                         timedelta(microseconds=saved["interval_microseconds"]), config, last, saved["target_name"])


def load_mlp_model(path: Path) -> SavedMLPModel:
    """Load a checksummed JSON predictor without importing any runtime backend."""
    return _decode_saved(_read_report(path))


def _seal(report: dict) -> dict:
    return {**report, "report_id": "sha256:" + sha256(json_bytes(report)).hexdigest()}


def _score(model: MLPModel, rows: tuple) -> dict:
    targets = tuple(_number(row.target_return) for row in rows)
    predictions = tuple(model.predict(row.features) for row in rows)
    return {"row_count": len(rows),
            "mlp": _error_metrics(tuple(p - y for p, y in zip(predictions, targets, strict=True))),
            "zero_return": _error_metrics(targets)}


def _prepare(path: Path) -> tuple:
    artifact, config, duration = _read_training_data(path)
    if artifact["format"] != OPEN_TRAINING_DATA_FORMAT or artifact["target_name"] != OPEN_TARGET_NAME:
        raise ValueError("MLP requires Open-to-Open training data")
    if artifact["splits"]["test"] != []:
        raise ValueError("MLP requires an empty test split; exclude 2025 before feature preparation")
    _config(config, artifact["interval_microseconds"])
    train = _checked_split(artifact["splits"]["train"], name="train", start=_START,
                           cutoff=config.validation_start, duration=duration,
                           availability_delay=timedelta(minutes=15))
    validation = _checked_split(artifact["splits"]["validation"], name="validation", start=config.validation_start,
                                cutoff=config.test_start, duration=duration,
                                availability_delay=timedelta(minutes=15))
    return artifact, train, validation


def _fit(train: tuple, validation: tuple, seed: int) -> tuple[dict, dict]:
    # Importing this module, loading JSON, and predicting never reach this import.
    import torch
    from importlib.metadata import version

    columns = tuple(zip(*(tuple(map(_number, row.features)) for row in train), strict=True))
    means = [fmean(column) for column in columns]
    scales = [pstdev(column) or 1. for column in columns]
    targets = tuple(_number(row.target_return) for row in train)
    target_mean, target_scale = fmean(targets), pstdev(targets) or 1.
    scalers = {"means": means, "scales": scales, "target_mean": target_mean, "target_scale": target_scale}

    def tensors(rows):
        x = [[(_number(value) - mean) / scale for value, mean, scale in
              zip(row.features, means, scales, strict=True)] for row in rows]
        y = [[(_number(row.target_return) - target_mean) / target_scale] for row in rows]
        tensors = (torch.tensor(x, dtype=torch.float64, device="cpu"),
                   torch.tensor(y, dtype=torch.float64, device="cpu"))
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("standardized training values must be finite")
        return tensors

    x, y = tensors(train)
    vx, vy = tensors(validation)
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.random.default_generator.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(4, 16, dtype=torch.float64, device="cpu"), torch.nn.ReLU(),
            torch.nn.Linear(16, 8, dtype=torch.float64, device="cpu"), torch.nn.ReLU(),
            torch.nn.Linear(8, 1, dtype=torch.float64, device="cpu"),
        )
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.0001)
        history, best_state = [], None
        best_loss, significant_best, stale, best_epoch = float("inf"), float("inf"), 0, 0
        reason = "max_epochs"
        for epoch in range(1, 101):
            network.train()
            permutation = torch.randperm(len(train), generator=generator, device="cpu")
            for batch in permutation.split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(network(x[batch]), y[batch])
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                training_loss = float(torch.nn.functional.mse_loss(network(x), y))
                validation_loss = float(torch.nn.functional.mse_loss(network(vx), vy))
            if not isfinite(training_loss) or not isfinite(validation_loss):
                raise ValueError("MLP training produced nonfinite losses")
            history.append({"epoch": epoch, "training_mse": training_loss, "validation_mse": validation_loss})
            # Keep the actual minimum checkpoint, even for sub-min_delta gains.
            # Only significant improvement resets the patience counter.
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = {name: tensor.detach().clone() for name, tensor in network.state_dict().items()}
            if validation_loss < significant_best - 1e-6:
                significant_best, stale = validation_loss, 0
            else:
                stale += 1
            if stale >= 10:
                reason = "early_stopping"
                break
        network.load_state_dict(best_state)
        network.eval()
        layers = [{"weights": layer.weight.detach().tolist(), "biases": layer.bias.detach().tolist()}
                  for layer in network if isinstance(layer, torch.nn.Linear)]
        model = {"kind": "mlp", "architecture": list(_ARCHITECTURE), "parameter_count": 225,
                 **scalers, "layers": layers}
        training = {"fit_settings": {**_SETTINGS, "seed": seed}, "epoch_history": history,
                    "best_epoch": best_epoch, "stopping_reason": reason,
                    "runtime_versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                                         "numpy": version("numpy")}}
        return model, training
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def mlp_report(training_path: Path, *, seed: int = 0) -> dict:
    """Fit the frozen CPU MLP on training only; select and score on validation."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    artifact, train, validation = _prepare(training_path)
    model_data, fit = _fit(train, validation, seed)
    # Official metrics run through exactly the JSON decoder used at reload.
    predictor = _decode_model(model_data)
    report = {"format": MLP_FORMAT, "training_data_id": artifact["training_data_id"],
              **{key: artifact[key] for key in _TRAINING_METADATA}, "model": model_data,
              "training": {"row_count": len(train), "first_decision_at": train[0].decision_at.isoformat(),
                           "last_target_available_at": max(row.target_available_at for row in train).isoformat(), **fit},
              "validation": _score(predictor, validation)}
    sealed = _seal(report)
    _decode_saved(sealed)
    return sealed


def mlp_validation_evaluation(model_path: Path, training_path: Path) -> tuple[dict, list[dict]]:
    """Score validation only with verified JSON weights; never fit or expose held-out data."""
    artifact, train, validation = _prepare(training_path)
    saved = _read_report(model_path)
    predictor = _decode_saved(saved)
    if saved["training_data_id"] != artifact["training_data_id"]:
        raise ValueError("model training data identifier does not match the input artifact")
    if any(saved[key] != artifact[key] for key in _TRAINING_METADATA):
        raise ValueError("model metadata does not match the training data")
    expected_training = {"row_count": len(train), "first_decision_at": train[0].decision_at.isoformat(),
                         "last_target_available_at": max(row.target_available_at for row in train).isoformat()}
    if any(saved["training"].get(key) != value for key, value in expected_training.items()):
        raise ValueError("model training metadata does not match the training data")
    report = {"format": "market-trader.mlp-validation-evaluation.v1", "model_report_id": predictor.report_id,
              "training_data_id": artifact["training_data_id"],
              **{key: artifact[key] for key in _TRAINING_METADATA},
              "validation": {**_score(predictor.model, validation),
                             "first_decision_at": validation[0].decision_at.isoformat(),
                             "last_decision_at": validation[-1].decision_at.isoformat()}}
    points = [{"at": row.decision_at.isoformat(), "predicted": predictor.model.predict(row.features),
               "actual": _number(row.target_return)} for row in validation]
    return _seal(report), points
