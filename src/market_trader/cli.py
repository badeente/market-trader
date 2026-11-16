"""Command-line entry point for offline datasets and R0 backtests."""

import argparse
import json
import os
import re
import shutil
import sys
import tomllib
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from enum import Enum
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from market_trader._config import config_decimal, require_fields
from market_trader._dataset_codec import json_bytes
from market_trader.artifacts import DatasetProvenance, StoredDataset
from market_trader.backtesting import ExecutionProfile, run_backtest
from market_trader.baseline import baseline_report, held_out_report
from market_trader.coverage import DatasetRequest, validate_requests
from market_trader.data import read_candles
from market_trader.normalization import NormalizationConfig, NormalizedDataset, normalize_candles
from market_trader.reporting import BACKTEST_REPORT_FORMAT, render_report
from market_trader.report_server import create_report_server
from market_trader.storage import fingerprint_source, load_dataset, publish_new_file, save_dataset
from market_trader.strategies import configure_strategy
from market_trader.training_data import (
    TARGET_NAME, TRAINING_DATA_FORMAT, TrainingConfig, build_training_data,
)


class _CommandError(Exception):
    """An expected failure with its processing step."""


@dataclass(frozen=True, slots=True)
class _ImportConfig:
    source_path: Path
    source_name: str
    description: str
    request: DatasetRequest
    normalization: NormalizationConfig


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without padding")
    return value


def _read_config(path: Path) -> _ImportConfig:
    with path.open("rb") as stream:
        config = require_fields(tomllib.load(stream), "config", "version", "source", "dataset")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("config version must be the integer 1")
    source = require_fields(config["source"], "source", "path", "name", "description")
    dataset = require_fields(
        config["dataset"], "dataset",
        "instrument", "interval_seconds", "start_time", "end_time", "calendar", "gap_policy",
    )
    if dataset["calendar"] != "continuous":
        raise ValueError("dataset.calendar must be continuous in config version 1")
    gap_policy = dataset["gap_policy"]
    if gap_policy not in ("report", "reject"):
        raise ValueError("dataset.gap_policy must be report or reject")
    interval_seconds = dataset["interval_seconds"]
    if type(interval_seconds) is not int or interval_seconds <= 0:
        raise ValueError("dataset.interval_seconds must be a positive whole integer")
    try:
        interval = timedelta(seconds=interval_seconds)
    except OverflowError as error:
        raise ValueError("dataset.interval_seconds is too large") from error
    request = DatasetRequest(dataset["instrument"], interval, dataset["start_time"], dataset["end_time"])
    normalization = NormalizationConfig(calendar="continuous", gap_policy=gap_policy)
    validate_requests((request,), calendar=normalization.calendar)
    return _ImportConfig(
        source_path=path.parent / _text(source["path"], "source.path"),
        source_name=_text(source["name"], "source.name"),
        description=_text(source["description"], "source.description"),
        request=request,
        normalization=normalization,
    )


def _verify_artifact(path: Path) -> StoredDataset:
    try:
        return load_dataset(path)
    except (OSError, ValueError) as error:
        raise _CommandError(f"verify artifact {path}: {error}") from error


def _read_input(config_path: Path) -> tuple[NormalizedDataset, DatasetProvenance]:
    try:
        config = _read_config(config_path)
    except (OSError, ValueError) as error:
        raise _CommandError(f"read config {config_path}: {error}") from error

    try:
        with TemporaryDirectory(prefix="market-trader-source-") as directory:
            snapshot = Path(directory) / "source.csv"
            # Hash and parse the same bytes even if the original path is replaced.
            shutil.copyfile(config.source_path, snapshot)
            source = fingerprint_source(snapshot, name=config.source_name)
            candles = read_candles(snapshot)
    except (OSError, ValueError) as error:
        raise _CommandError(f"read CSV {config.source_path}: {error}") from error

    try:
        result = normalize_candles(candles, config=config.normalization, requests=(config.request,))
        dataset, = result.datasets
    except ValueError as error:
        raise _CommandError(f"normalize CSV {config.source_path}: {error}") from error

    provenance = DatasetProvenance(description=config.description, sources=(source,))
    return dataset, provenance


def _import_dataset(config_path: Path, output: Path) -> StoredDataset:
    dataset, provenance = _read_input(config_path)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".market-trader-", dir=output.parent) as directory:
            staged = Path(directory) / "dataset.json"
            save_dataset(staged, dataset, provenance=provenance)
            restored = _verify_artifact(staged)
            # Publish the verified bytes without replacing an existing artifact.
            os.link(staged, output)
    except (OSError, ValueError) as error:
        raise _CommandError(f"save artifact {output}: {error}") from error
    return restored


def _execution_profile(table: object) -> ExecutionProfile:
    defaults = ExecutionProfile()
    names = {field.name for field in fields(defaults)}
    if not isinstance(table, dict):
        raise ValueError("execution must be a table")
    if unknown := table.keys() - names:
        raise ValueError(f"unsupported execution fields: {', '.join(sorted(unknown))}")
    values = {}
    for name, value in table.items():
        if isinstance(getattr(defaults, name), bool):
            if type(value) is not bool:
                raise ValueError(f"execution.{name} must be a boolean")
            values[name] = value
        else:
            values[name] = config_decimal(value, f"execution.{name}")
    return ExecutionProfile(**values)


def _source_digest() -> str:
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\x00")
        digest.update(sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _json_value(value: object) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.name
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _prepare_training(config_path: Path, output: Path) -> dict:
    try:
        with config_path.open("rb") as stream:
            config = require_fields(
                tomllib.load(stream), "training config", "version", "dataset_artifact",
                "lookback_bars", "horizon_bars", "validation_start", "test_start",
            )
        if type(config["version"]) is not int or config["version"] != 1:
            raise ValueError("training config version must be the integer 1")
        input_path = config_path.parent / _text(config["dataset_artifact"], "dataset_artifact")
        settings = TrainingConfig(
            lookback_bars=config["lookback_bars"], horizon_bars=config["horizon_bars"],
            validation_start=config["validation_start"], test_start=config["test_start"],
        )
    except (OSError, ValueError) as error:
        raise _CommandError(f"read training config {config_path}: {error}") from error

    stored = _verify_artifact(input_path)
    try:
        training = build_training_data(stored.dataset, settings)
    except (ValueError, DecimalException) as error:
        raise _CommandError(f"prepare training data {input_path}: {error}") from error

    splits = {
        "train": training.train,
        "validation": training.validation,
        "test": training.test,
    }
    source = {
        "dataset_id": stored.manifest.dataset_id,
        "dataset_artifact_id": stored.manifest.artifact_id,
    }
    payload = {
        **source,
        "format": TRAINING_DATA_FORMAT,
        "instrument": stored.dataset.instrument,
        "interval_microseconds": stored.dataset.interval // timedelta(microseconds=1),
        "config": asdict(training.config),
        "feature_names": training.feature_names,
        "target_name": TARGET_NAME,
        "splits": {name: [asdict(row) for row in rows] for name, rows in splits.items()},
    }
    training_data_id = "sha256:" + sha256(json_bytes(payload, default=_json_value)).hexdigest()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        publish_new_file(
            output, json_bytes({**payload, "training_data_id": training_data_id}, default=_json_value) + b"\n",
        )
    except OSError as error:
        raise _CommandError(f"save training data {output}: {error}") from error
    return {
        **source, "training_data_id": training_data_id,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
    }


def _save_model_report(report: dict, output: Path, name: str) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        publish_new_file(output, json_bytes(report) + b"\n")
    except OSError as error:
        raise _CommandError(f"save {name} report {output}: {error}") from error


def _train_baseline(training_data: Path, output: Path) -> dict:
    try:
        report = baseline_report(training_data)
    except (OSError, ValueError, ArithmeticError) as error:
        raise _CommandError(f"train baseline {training_data}: {error}") from error
    _save_model_report(report, output, "baseline")
    return {
        "report_id": report["report_id"],
        "training_data_id": report["training_data_id"],
        "training_rows": report["training"]["row_count"],
        "validation": report["validation"],
    }


def _evaluate_baseline(model_path: Path, training_data: Path, output: Path) -> dict:
    try:
        report = held_out_report(model_path, training_data)
    except (OSError, ValueError, ArithmeticError) as error:
        raise _CommandError(f"evaluate baseline {model_path} on {training_data}: {error}") from error
    _save_model_report(report, output, "held-out")
    return report


def _backtest_report(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream, parse_float=Decimal)
        inputs = config.keys() & {"data_config", "dataset_artifact"}
        if len(inputs) != 1:
            raise ValueError("backtest config requires exactly one of data_config and dataset_artifact")
        input_key, = inputs
        require_fields(config, "backtest config", "version", input_key, "strategy", "execution")
        if type(config["version"]) is not int or config["version"] != 1:
            raise ValueError("backtest config version must be the integer 1")
        input_path = path.parent / _text(config[input_key], input_key)
        profile = _execution_profile(config["execution"])
        strategy = configure_strategy(config["strategy"], price_tick=profile.price_tick, config_dir=path.parent)
    except (OSError, ValueError, ArithmeticError) as error:
        raise _CommandError(f"read backtest config {path}: {error}") from error

    stored = None
    if input_key == "dataset_artifact":
        stored = _verify_artifact(input_path)
        dataset, provenance = stored.dataset, stored.manifest.provenance
    else:
        dataset, provenance = _read_input(input_path)
    try:
        result = run_backtest(dataset, strategy.create(dataset), lookback=strategy.lookback, profile=profile)
        runtime = {
            "python": sys.version.split()[0],
            "market_trader": version("market-trader"),
            "code_id": _source_digest(),
        }
    except (OSError, ValueError, ArithmeticError) as error:
        raise _CommandError(f"run backtest {path}: {error}") from error
    report = asdict(result)
    report["execution"] = report.pop("profile")
    report["strategy"] = {"name": strategy.name, "lookback": strategy.lookback, **dict(strategy.parameters)}
    report["provenance"] = asdict(provenance)
    if stored is not None:
        report["dataset_artifact_id"] = stored.manifest.artifact_id
    report["runtime"] = runtime
    report["format"] = BACKTEST_REPORT_FORMAT
    return report


def _write_html_report(args: argparse.Namespace) -> None:
    try:
        payload = render_report(
            args.backtest, compare=args.compare, evaluation=args.evaluation,
            model=args.model, training_data=args.training_data, dataset=args.dataset,
        )
    except (OSError, ValueError, ArithmeticError, KeyError, TypeError) as error:
        raise _CommandError(f"render HTML report for {args.backtest}: {error}") from error
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        publish_new_file(args.output, payload)
    except OSError as error:
        raise _CommandError(f"save HTML report {args.output}: {error}") from error


def _report_registries(tokens: list[str]) -> dict[str, Path] | None:
    """Parse exact source mappings; never strip or normalize supplied tokens."""
    if not tokens:
        return None
    registries = {}
    for token in tokens:
        alias, separator, value = token.partition("=")
        if (not separator or re.fullmatch(r"[a-z][a-z0-9_]*", alias) is None
                or alias in registries or not value or value != value.strip()
                or any(ord(char) < 32 or ord(char) == 127 for char in value)
                or not Path(value).is_absolute() or str(Path(value)) != value
                or any(part in (".", "..") for part in value.split("/"))):
            raise ValueError("--registry requires unique lowercase alias=/absolute/path without padding or traversal")
        registries[alias] = Path(value)
    return registries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="market-trader")
    commands = parser.add_subparsers(required=True)
    data = commands.add_parser("data", help="Import, verify and prepare local datasets")
    data_commands = data.add_subparsers(dest="action", required=True)

    import_data = data_commands.add_parser("import", help="Import a canonical CSV")
    import_data.add_argument("--config", type=Path, required=True)
    import_data.add_argument("--output", type=Path, required=True)

    verify_data = data_commands.add_parser("verify", help="Verify a stored dataset")
    verify_data.add_argument("--artifact", type=Path, required=True)

    prepare_training = data_commands.add_parser("prepare-training", help="Build supervised examples")
    prepare_training.add_argument("--config", type=Path, required=True)
    prepare_training.add_argument("--output", type=Path, required=True)

    model = commands.add_parser("model", help="Train and evaluate offline prediction models")
    model_commands = model.add_subparsers(dest="action", required=True)
    train_baseline = model_commands.add_parser("train-baseline", help="Fit and validate a CPU ridge baseline")
    train_baseline.add_argument("--training-data", type=Path, required=True)
    train_baseline.add_argument("--output", type=Path, required=True)
    evaluate_baseline = model_commands.add_parser("evaluate-baseline", help="Score a saved baseline on held-out data")
    evaluate_baseline.add_argument("--model", type=Path, required=True)
    evaluate_baseline.add_argument("--training-data", type=Path, required=True)
    evaluate_baseline.add_argument("--output", type=Path, required=True)

    backtest = commands.add_parser("backtest", help="Run an offline R0 backtest")
    backtest.add_argument("--config", type=Path, required=True)
    backtest.set_defaults(action="backtest")

    report = commands.add_parser("report", help="Build an offline HTML report from saved results")
    report.add_argument("--backtest", type=Path, required=True, help="Primary backtest JSON")
    report.add_argument("--compare", type=Path, help="Second backtest JSON on the same data and costs")
    report.add_argument("--evaluation", type=Path, help="Saved Ridge held-out metrics")
    report.add_argument("--model", type=Path, help="Frozen Ridge model for forecast details")
    report.add_argument("--training-data", type=Path, help="Training export for individual held-out forecasts")
    report.add_argument("--dataset", type=Path, help="Backtest dataset artifact for Ridge trade explanations")
    report.add_argument("--output", type=Path, required=True, help="New HTML file")
    report.set_defaults(action="report")

    serve = commands.add_parser("serve-reports", help="Read-only internal HTML report viewer")
    serve.add_argument("--directory", type=Path, required=True, help="Dedicated directory of published HTML files")
    serve.add_argument("--bind", default="127.0.0.1", help="Explicit private IPv4 address (default: loopback)")
    serve.add_argument("--port", type=int, default=8086)
    serve.add_argument("--allow-network", action="append", default=[], help="Allowed private IPv4 subnet; repeatable")
    serve.add_argument("--registry", action="append", default=[], metavar="ALIAS=/ABSOLUTE/PATH",
                       help="Read-only experiment source; repeatable, enables the workbench")
    serve.set_defaults(action="serve-reports")

    reference = commands.add_parser("reference", help="Run the frozen AP1 reference protocol; reserve 2025")
    reference.add_argument("--config", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True, help="New private artifact directory")
    reference.set_defaults(action="reference")

    experiment = commands.add_parser("experiment", help="Run, inspect, replay and compare private AP2 experiments")
    experiment.set_defaults(action="experiment")
    experiment_commands = experiment.add_subparsers(dest="experiment_action", required=True)
    for action, help_text in (
        ("run", "Train, validate, backtest and register a configured experiment"),
        ("list", "List registered run records"),
        ("show", "Inspect one run and verify its stored metadata"),
        ("replay", "Repeat an immutable configuration as a new child run"),
        ("compare", "Check fair conditions and show configuration differences"),
    ):
        command = experiment_commands.add_parser(action, help=help_text)
        command.add_argument("--registry", type=Path, required=True, help="Private run directory, never a served directory")
        if action == "run":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--parent", help="Optional existing parent run ID")
        elif action in ("show", "replay"):
            command.add_argument("--run-id", required=True)
        elif action == "compare":
            command.add_argument("--left", required=True, help="First completed run ID")
            command.add_argument("--right", required=True, help="Second completed run ID")

    args = parser.parse_args(argv)
    try:
        if args.action == "experiment":
            from market_trader.experiments import dispatch_experiment

            try:
                result = dispatch_experiment(args)
                encoded = json.dumps(result, sort_keys=True, allow_nan=False)
            except (OSError, ValueError, ArithmeticError, KeyError, TypeError) as error:
                raise _CommandError(f"experiment {args.experiment_action}: {error}") from error
            print(encoded)
            return 0
        if args.action == "reference":
            from market_trader.reference import run_reference

            try:
                summary = run_reference(args.config, args.output)
            except (OSError, ValueError, ArithmeticError) as error:
                raise _CommandError(f"AP1 reference: {error}") from error
            print(json.dumps(summary, sort_keys=True))
            return 0
        if args.action == "serve-reports":
            try:
                with create_report_server(
                    args.directory, bind=args.bind, port=args.port,
                    allowed_networks=("127.0.0.0/8", *args.allow_network),
                    registries=_report_registries(args.registry),
                ) as server:
                    print(f"Report viewer listening on {args.bind}:{server.server_address[1]}", flush=True)
                    server.serve_forever()
            except KeyboardInterrupt:
                return 0
            except (OSError, ValueError) as error:
                raise _CommandError(f"serve reports: {error}") from error
            return 0
        if args.action == "report":
            _write_html_report(args)
            print(args.output)
            return 0
        if args.action == "backtest":
            print(json.dumps(_backtest_report(args.config), default=_json_value, sort_keys=True))
            return 0
        if args.action == "prepare-training":
            print(json.dumps(_prepare_training(args.config, args.output), sort_keys=True))
            return 0
        if args.action == "train-baseline":
            print(json.dumps(_train_baseline(args.training_data, args.output), sort_keys=True))
            return 0
        if args.action == "evaluate-baseline":
            print(json.dumps(_evaluate_baseline(args.model, args.training_data, args.output), sort_keys=True))
            return 0
        if args.action == "import":
            stored = _import_dataset(args.config, args.output)
        else:
            stored = _verify_artifact(args.artifact)
    except _CommandError as error:
        print(f"market-trader: {error}", file=sys.stderr)
        return 1

    print(json.dumps({
        "artifact_id": stored.manifest.artifact_id,
        "dataset_id": stored.manifest.dataset_id,
        "candle_count": len(stored.dataset.candles),
        "missing_candles": sum(gap.missing_count for gap in stored.dataset.quality.gaps),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
