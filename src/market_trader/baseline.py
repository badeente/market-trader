"""A fixed CPU ridge model and forecast evaluation against zero return."""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from math import fsum, isfinite, sqrt
from pathlib import Path
from statistics import fmean, pstdev

from market_trader._dataset_codec import decode_record, json_bytes, parse_json, record_object
from market_trader.training_data import (
    FEATURE_NAMES, TARGET_NAME, TRAINING_DATA_FORMAT, OPEN_TARGET_NAME,
    OPEN_TRAINING_DATA_FORMAT, TrainingConfig, TrainingExample,
)

BASELINE_FORMAT = "market-trader.ridge-validation.v1"
OPEN_BASELINE_FORMAT = "market-trader.ridge-validation.v2"
_TRAINING_TARGETS = {TRAINING_DATA_FORMAT: TARGET_NAME, OPEN_TRAINING_DATA_FORMAT: OPEN_TARGET_NAME}
_MODEL_TARGETS = {BASELINE_FORMAT: TARGET_NAME, OPEN_BASELINE_FORMAT: OPEN_TARGET_NAME}
HELD_OUT_FORMAT = "market-trader.ridge-test.v1"
RIDGE_ALPHA = 0.1
_TRAINING_METADATA = (
    "dataset_id", "dataset_artifact_id", "instrument", "interval_microseconds",
    "config", "feature_names", "target_name",
)


def _number(value: Decimal) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("feature or target exceeds the model's floating-point range")
    return result


@dataclass(frozen=True, slots=True)
class RidgeModel:
    """The fitted scaler and coefficients in FEATURE_NAMES order."""

    alpha: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]

    def contributions(self, features: tuple[Decimal, ...]) -> tuple[float, ...]:
        return tuple(
            coefficient * ((_number(value) - mean) / scale)
            for value, mean, scale, coefficient in zip(
                features, self.means, self.scales, self.coefficients, strict=True,
            )
        )

    def predict(self, features: tuple[Decimal, ...]) -> float:
        prediction = self.intercept + fsum(self.contributions(features))
        if not isfinite(prediction):
            raise ValueError("model prediction exceeds the floating-point range")
        return prediction


@dataclass(frozen=True, slots=True)
class SavedRidgeModel:
    """Verified predictor and the metadata required for offline inference."""

    report_id: str
    model: RidgeModel
    instrument: str
    interval: timedelta
    config: TrainingConfig
    last_target_available_at: datetime
    target_name: str = TARGET_NAME


def _solve_positive_definite(matrix: list[list[float]], rhs: list[float]) -> tuple[float, ...]:
    """Solve a small positive-definite system using its Cholesky factor."""
    size = len(rhs)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            remainder = matrix[row][column] - fsum(
                lower[row][index] * lower[column][index] for index in range(column)
            )
            if row == column:
                lower[row][column] = sqrt(remainder)
            else:
                lower[row][column] = remainder / lower[column][column]

    intermediate = [0.0] * size
    for row in range(size):
        subtotal = fsum(lower[row][index] * intermediate[index] for index in range(row))
        intermediate[row] = (rhs[row] - subtotal) / lower[row][row]

    solution = [0.0] * size
    for row in reversed(range(size)):
        subtotal = fsum(lower[index][row] * solution[index] for index in range(row + 1, size))
        solution[row] = (intermediate[row] - subtotal) / lower[row][row]
    return tuple(solution)


def _fit_ridge(rows: tuple[TrainingExample, ...]) -> RidgeModel:
    """Fit mean squared error with a ridge penalty on standardized slopes."""
    columns = tuple(zip(*(tuple(map(_number, row.features)) for row in rows), strict=True))
    deviations = tuple(pstdev(column) for column in columns)
    means = tuple(fmean(column) if deviation else column[0]
                  for column, deviation in zip(columns, deviations, strict=True))
    scales = tuple(deviation or 1.0 for deviation in deviations)
    standardized = tuple(
        tuple((value - mean) / scale for value in column)
        for column, mean, scale in zip(columns, means, scales, strict=True)
    )
    targets = tuple(_number(row.target_return) for row in rows)
    intercept = fmean(targets)
    centered_targets = tuple(target - intercept for target in targets)
    matrix = [
        [fmean(x * y for x, y in zip(left, right, strict=True))
         for right in standardized]
        for left in standardized
    ]
    for index, row in enumerate(matrix):
        row[index] += RIDGE_ALPHA
    rhs = [fmean(x * y for x, y in zip(column, centered_targets, strict=True))
           for column in standardized]
    coefficients = _solve_positive_definite(matrix, rhs)
    return RidgeModel(RIDGE_ALPHA, means, scales, intercept, coefficients)


def _error_metrics(errors: tuple[float, ...]) -> dict[str, float]:
    return {
        "mae": fmean(map(abs, errors)),
        "rmse": sqrt(fmean(error * error for error in errors)),
    }


def _score_predictions(predictions: tuple[float, ...], targets: tuple[float, ...]) -> dict:
    errors = tuple(prediction - target for prediction, target in zip(predictions, targets, strict=True))
    return {
        "row_count": len(targets),
        "ridge": _error_metrics(errors),
        "zero_return": _error_metrics(targets),
    }


def _checked_split(
    rows: object, *, name: str, start: datetime | None, cutoff: datetime | None, duration: timedelta,
    availability_delay: timedelta = timedelta(0),
) -> tuple[TrainingExample, ...]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} split must be a nonempty array")
    examples = tuple(decode_record(row, TrainingExample) for row in rows)
    previous = None
    for row in examples:
        if len(row.features) != len(FEATURE_NAMES):
            raise ValueError(f"{name} split feature count does not match the model schema")
        if previous is not None and row.decision_at <= previous:
            raise ValueError(f"{name} split decisions must be strictly chronological")
        if ((start is not None and row.decision_at < start)
                or (cutoff is not None and (
                    row.decision_at >= cutoff or row.target_available_at >= cutoff
                ))):
            raise ValueError(f"{name} split cutoff is inconsistent with the row timestamps")
        if (row.entry_at != row.decision_at or row.exit_at - row.entry_at != duration
                or row.target_available_at < row.exit_at + availability_delay):
            raise ValueError(f"{name} split row timestamps do not match the configured target horizon")
        previous = row.decision_at
    return examples


def _read_training_data(path: Path) -> tuple[dict, TrainingConfig, timedelta]:
    artifact = record_object(
        parse_json(path.read_text(encoding="utf-8")),
        ("format", "training_data_id", "splits", *_TRAINING_METADATA),
    )
    if artifact["format"] not in _TRAINING_TARGETS:
        raise ValueError("unsupported training data format")
    training_data_id = artifact.pop("training_data_id")
    if training_data_id != "sha256:" + sha256(json_bytes(artifact)).hexdigest():
        raise ValueError("training data checksum mismatch")
    artifact["training_data_id"] = training_data_id
    if artifact["feature_names"] != list(FEATURE_NAMES) or artifact["target_name"] != _TRAINING_TARGETS[artifact["format"]]:
        raise ValueError("training data features or target do not match the model schema")
    interval = artifact["interval_microseconds"]
    if type(interval) is not int or interval <= 0:
        raise ValueError("interval_microseconds must be a positive integer")
    config = decode_record(artifact["config"], TrainingConfig)
    duration = timedelta(microseconds=interval * config.horizon_bars)
    record_object(artifact["splits"], ("train", "validation", "test"))
    return artifact, config, duration


def baseline_report(path: Path) -> dict:
    """Fit the training split and score the validation split of a saved export."""
    artifact, config, duration = _read_training_data(path)
    splits = artifact["splits"]
    availability_delay = (timedelta(microseconds=artifact["interval_microseconds"])
                          if artifact["target_name"] == OPEN_TARGET_NAME else timedelta(0))
    train = _checked_split(
        splits["train"], name="train", start=None, cutoff=config.validation_start, duration=duration,
        availability_delay=availability_delay,
    )
    validation = _checked_split(
        splits["validation"], name="validation", start=config.validation_start,
        cutoff=config.test_start, duration=duration, availability_delay=availability_delay,
    )
    model = _fit_ridge(train)
    report = {
        "format": OPEN_BASELINE_FORMAT if artifact["target_name"] == OPEN_TARGET_NAME else BASELINE_FORMAT,
        "training_data_id": artifact["training_data_id"],
        **{key: artifact[key] for key in _TRAINING_METADATA},
        "model": {"kind": "ridge", **asdict(model)},
        "training": {
            "row_count": len(train),
            "first_decision_at": train[0].decision_at.isoformat(),
            "last_target_available_at": max(row.target_available_at for row in train).isoformat(),
        },
        "validation": _score_predictions(
            tuple(model.predict(row.features) for row in validation),
            tuple(_number(row.target_return) for row in validation),
        ),
    }
    return {**report, "report_id": "sha256:" + sha256(json_bytes(report)).hexdigest()}


def _model_number(value: object, field: str) -> float:
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError(f"model {field} must contain a finite number")
    return float(value)


def _decode_model(value: object) -> RidgeModel:
    data = record_object(value, ("kind", "alpha", "means", "scales", "intercept", "coefficients"))
    alpha = _model_number(data["alpha"], "alpha")
    if data["kind"] != "ridge" or alpha != RIDGE_ALPHA:
        raise ValueError("unsupported baseline model kind or alpha")
    vectors = {}
    for field in ("means", "scales", "coefficients"):
        values = data[field]
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise ValueError(f"model {field} must match the feature count")
        vectors[field] = tuple(_model_number(item, field) for item in values)
    if any(scale <= 0 for scale in vectors["scales"]):
        raise ValueError("model requires positive scales")
    return RidgeModel(alpha=alpha, intercept=_model_number(data["intercept"], "intercept"), **vectors)


def _read_model_report(model_path: Path) -> dict:
    saved = record_object(
        parse_json(model_path.read_text(encoding="utf-8"), allow_floats=True),
        ("format", "report_id", "training_data_id", "model", "training", "validation", *_TRAINING_METADATA),
    )
    if saved["format"] not in _MODEL_TARGETS:
        raise ValueError("unsupported model report format")
    body = {key: value for key, value in saved.items() if key != "report_id"}
    if saved["report_id"] != "sha256:" + sha256(json_bytes(body)).hexdigest():
        raise ValueError("model report checksum mismatch")
    if saved["target_name"] != _MODEL_TARGETS[saved["format"]]:
        raise ValueError("model target does not match the report format")
    return saved


def _decode_saved_model(saved: dict) -> SavedRidgeModel:
    if saved["feature_names"] != list(FEATURE_NAMES) or saved["target_name"] != _MODEL_TARGETS[saved["format"]]:
        raise ValueError("model features or target do not match the inference schema")
    instrument = saved["instrument"]
    if not isinstance(instrument, str) or not instrument or instrument != instrument.strip():
        raise ValueError("model instrument must be a nonempty string without padding")
    interval = saved["interval_microseconds"]
    if type(interval) is not int or interval <= 0:
        raise ValueError("model interval_microseconds must be a positive integer")
    config = decode_record(saved["config"], TrainingConfig)

    training = record_object(saved["training"], ("row_count", "first_decision_at", "last_target_available_at"))
    if type(training["row_count"]) is not int or training["row_count"] <= 0:
        raise ValueError("model training row count must be a positive integer")
    first, last = training["first_decision_at"], training["last_target_available_at"]
    if not isinstance(first, str) or not isinstance(last, str):
        raise ValueError("model training timestamps must be UTC strings")
    first_at, last_at = datetime.fromisoformat(first), datetime.fromisoformat(last)
    if (first_at.utcoffset() != timedelta(0) or last_at.utcoffset() != timedelta(0)
            or not first_at <= last_at < config.validation_start):
        raise ValueError("model training timestamps must precede the validation cutoff")

    return SavedRidgeModel(
        saved["report_id"], _decode_model(saved["model"]), instrument,
        timedelta(microseconds=interval), config, last_at, saved["target_name"],
    )


def load_ridge_model(path: Path) -> SavedRidgeModel:
    """Verify and load a saved predictor for inference."""
    return _decode_saved_model(_read_model_report(path))


def validation_evaluation(model_path: Path, training_data_path: Path) -> tuple[dict, list[dict]]:
    """Score only validation with a verified frozen predictor; never fit or inspect test rows."""
    artifact, config, duration = _read_training_data(training_data_path)
    saved = _read_model_report(model_path)
    if saved["training_data_id"] != artifact["training_data_id"]:
        raise ValueError("model training data identifier does not match the input artifact")
    if any(saved[key] != artifact[key] for key in _TRAINING_METADATA):
        raise ValueError("model metadata does not match the training data")
    predictor = _decode_saved_model(saved)
    validation = _checked_split(
        artifact["splits"]["validation"], name="validation", start=config.validation_start,
        cutoff=config.test_start, duration=duration,
        availability_delay=predictor.interval if predictor.target_name == OPEN_TARGET_NAME else timedelta(0),
    )
    targets = tuple(_number(row.target_return) for row in validation)
    predictions = tuple(predictor.model.predict(row.features) for row in validation)
    report = {
        "format": "market-trader.ridge-validation-evaluation.v1",
        "model_report_id": predictor.report_id,
        "training_data_id": artifact["training_data_id"],
        **{key: artifact[key] for key in _TRAINING_METADATA},
        "validation": {
            **_score_predictions(predictions, targets),
            "first_decision_at": validation[0].decision_at.isoformat(),
            "last_decision_at": validation[-1].decision_at.isoformat(),
        },
    }
    points = [
        {"at": row.decision_at.isoformat(), "predicted": predicted, "actual": target}
        for row, predicted, target in zip(validation, predictions, targets, strict=True)
    ]
    return {**report, "report_id": "sha256:" + sha256(json_bytes(report)).hexdigest()}, points


def held_out_report(model_path: Path, training_data_path: Path) -> dict:
    """Score a saved model on its training artifact's held-out split."""
    report, _ = held_out_evaluation(model_path, training_data_path)
    return report


def held_out_evaluation(model_path: Path, training_data_path: Path) -> tuple[dict, list[dict]]:
    """Return the held-out report and the individual forecasts behind its metrics."""
    artifact, config, duration = _read_training_data(training_data_path)
    saved = _read_model_report(model_path)
    if saved["training_data_id"] != artifact["training_data_id"]:
        raise ValueError("model training data identifier does not match the input artifact")
    if any(saved[key] != artifact[key] for key in _TRAINING_METADATA):
        raise ValueError("model metadata does not match the training data")

    predictor = _decode_saved_model(saved)
    test = _checked_split(
        artifact["splits"]["test"], name="test", start=config.test_start, cutoff=None, duration=duration,
        availability_delay=predictor.interval if predictor.target_name == OPEN_TARGET_NAME else timedelta(0),
    )
    targets = tuple(_number(row.target_return) for row in test)
    predictions = tuple(predictor.model.predict(row.features) for row in test)
    report = {
        "format": HELD_OUT_FORMAT,
        "model_report_id": predictor.report_id,
        "training_data_id": artifact["training_data_id"],
        **{key: artifact[key] for key in _TRAINING_METADATA},
        "test": {
            **_score_predictions(predictions, targets),
            "first_decision_at": test[0].decision_at.isoformat(),
            "last_decision_at": test[-1].decision_at.isoformat(),
        },
    }
    points = [
        {"at": row.decision_at.isoformat(), "predicted": predicted, "actual": target}
        for row, predicted, target in zip(test, predictions, targets, strict=True)
    ]
    return {**report, "report_id": "sha256:" + sha256(json_bytes(report)).hexdigest()}, points
