"""One bounded AP1 pipeline; no run queue, tuning, networking or held-out scoring."""

from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from market_trader._dataset_codec import json_bytes
from market_trader.artifacts import DatasetProvenance
from market_trader.backtesting import ExecutionProfile, run_backtest
from market_trader.baseline import baseline_report, validation_evaluation, load_ridge_model
from market_trader.dukascopy_import import import_dukascopy_reference, slice_session_dataset
from market_trader.reference import EXPERIMENT_ID, _START, _VALIDATION, _TEST, _END, read_reference_config, trading_summary
from market_trader.reference_reporting import enrich_reference_report
from market_trader.reporting import BACKTEST_REPORT_FORMAT, render_report
from market_trader.storage import save_dataset, load_dataset, publish_new_file
from market_trader.strategies import configure_strategy
from market_trader.training_data import (
    OPEN_TARGET_NAME, OPEN_TRAINING_DATA_FORMAT, TrainingConfig, build_open_training_data,
)


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.name
    raise TypeError(f"unsupported reference value: {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    publish_new_file(path, json_bytes(value, default=_json_value) + b"\n")


def _source_id() -> str:
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\x00")
        digest.update(sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _runtime() -> dict:
    result = {"python": sys.version.split()[0], "market_trader": version("market-trader"),
              "code_id": _source_id(), "git_head": None, "git_dirty": None}
    root = Path(__file__).resolve().parents[2]
    if (root / ".git").exists():
        result["git_head"] = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
        result["git_dirty"] = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True,
        ).stdout)
    return result


def _policy() -> dict:
    return {
        "format": "market-trader.ap1-policy.v1", "experiment": EXPERIMENT_ID,
        "instrument": "deuidxeur", "provider": "Dukascopy", "source_price_type": "bid",
        "source_start": _START.isoformat(), "source_end": _END.isoformat(),
        "research_timezone": "Europe/Berlin", "research_window": "Mon-Fri 09:00-17:30",
        "calendar_status": "research filter, not certified provider hours", "complete_sessions_only": True,
        "validation_start": _VALIDATION.isoformat(), "reserved_test_start": _TEST.isoformat(),
        "target_name": OPEN_TARGET_NAME, "lookback_bars": 8, "horizon_bars": 2,
        "label_availability": "maximum availability including the exit-open candle",
        "features": "existing four causal price features; no volume feature",
        "ridge_alpha": "0.1", "entry_threshold": "0", "sma_fast_period": 4, "sma_slow_period": 8,
        "quantity": "1", "spread": "2.0", "slippage_per_fill": "0.1", "commission_rate": "0",
        "price_tick": "0.001", "initial_capital": "100000", "account_unit": "synthetic Q; not EUR",
        "costs_status": "hypothetical research costs, not confirmed broker conditions",
        "seed": "not used by deterministic ridge solver", "test_metrics_computed": False,
        "selection_limit": "complete-observed-session selection may introduce bias; no forward fill",
        "data_rights": "private research source; no public raw-data redistribution or licence certification",
    }


def _label_checks(training, development) -> list[dict]:
    """Independently recalculate representative targets using exact rational prices."""
    by_open = {candle.open_time: candle for candle in development.candles}
    checks = []
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        for split, examples in (("train", training.train), ("validation", training.validation)):
            for index in sorted({0, len(examples) // 2}):
                row = examples[index]
                entry, exit_bar = by_open[row.entry_at], by_open[row.exit_at]
                expected = Fraction(exit_bar.open - entry.open) / Fraction(entry.open)
                value = Decimal(expected.numerator) / Decimal(expected.denominator)
                future = [by_open[row.entry_at + step*development.interval]
                          for step in range(training.config.horizon_bars + 1)]
                available = max(candle.available_at for candle in future)
                if (row.target_return != value or row.target_available_at != available
                        or row.entry_at != row.decision_at
                        or row.exit_at - row.entry_at != 2*development.interval):
                    raise ValueError("independent reference label check failed")
                checks.append({"split": split, "decision_at": row.decision_at,
                               "entry_at": row.entry_at, "entry_reference_open": entry.open,
                               "exit_at": row.exit_at, "exit_reference_open": exit_bar.open,
                               "target_available_at": available, "target_return": value,
                               "rational_recalculation_matches": True})
    return checks


def _assert_trade_alignment(report: dict, rows) -> None:
    eligible = {row.decision_at: row for row in rows}
    for trade in report["trades"]:
        entry, exit_fill = trade["entry"], trade["exit"]
        decision = datetime.fromisoformat(entry["order"]["decision_at"])
        row = eligible.get(decision)
        if (row is None or datetime.fromisoformat(entry["candle_open_time"]) != row.entry_at
                or datetime.fromisoformat(exit_fill["candle_open_time"]) != row.exit_at
                or entry["side"] != "buy" or exit_fill["side"] != "sell"
                or Decimal(entry["quantity"]) != 1):
            raise ValueError("reference trade does not match the common open-to-open decision cohort")


def _backtest(stage: Path, name: str, config: dict, dataset, manifest, provenance, runtime, profile) -> dict:
    strategy = configure_strategy(config, price_tick=profile.price_tick, config_dir=stage)
    result = run_backtest(dataset, strategy.create(dataset), lookback=strategy.lookback, profile=profile)
    report = asdict(result)
    report["execution"] = report.pop("profile")
    report.update({"format": BACKTEST_REPORT_FORMAT, "strategy": {
        "name": strategy.name, "lookback": strategy.lookback, **dict(strategy.parameters),
    }, "provenance": asdict(provenance), "dataset_artifact_id": manifest.artifact_id, "runtime": runtime})
    path = stage / f"{name}-backtest.json"
    _write_json(path, report)
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_directory(stage: Path, output: Path) -> None:
    """Claim a new private directory; publish the complete manifest last, no overwrite."""
    output.mkdir(mode=0o700)
    created = []
    try:
        for path in sorted(stage.iterdir(), key=lambda path: (path.name == "manifest.json", path.name)):
            target = output / path.name
            os.link(path, target)
            created.append(target)
    except BaseException:
        for path in created:
            path.unlink()
        output.rmdir()
        raise


def run_reference(config_path: Path, output: Path) -> dict:
    """Run the approved AP1 protocol once into a new private artifact directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"reference output already exists: {output}")
    settings = read_reference_config(config_path)
    runtime = _runtime()
    imported = import_dukascopy_reference(
        settings.source_csv, instrument="deuidxeur", start=_START, end=_END,
        expected_sha256=settings.source_sha256,
    )
    # No 2025 candle is handed to a feature builder, model or backtest.
    development = slice_session_dataset(imported.dataset, _START, _TEST)
    validation = slice_session_dataset(imported.dataset, _VALIDATION, _TEST)
    if any(candle.close_time > _TEST for candle in development.candles):
        raise ValueError("reserved test candle reached the development dataset")
    config = TrainingConfig(8, 2, _VALIDATION, _TEST)
    training = build_open_training_data(development, config, include_test=False)
    if training.test or any(row.decision_at >= _TEST for row in (*training.train, *training.validation)):
        raise ValueError("reserved test rows reached the AP1 feature pipeline")
    policy = _policy()
    profile = ExecutionProfile(initial_capital=Decimal(100000), spread=Decimal("2.0"),
        slippage=Decimal("0.1"), price_tick=Decimal("0.001"), quantity_step=Decimal(1),
        commission_rate=Decimal(0), allow_short=False, liquidate_at_end=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".market-trader-ap1-", dir=output.parent) as directory:
        stage = Path(directory)
        origin = DatasetProvenance("Dukascopy DEUIDXEUR bid; AP1 complete-session research filter", (imported.source,))
        full_manifest = save_dataset(stage / "research-dataset.json", imported.dataset, provenance=origin)
        derived = DatasetProvenance("AP1 development excludes reserved 2025; source/session assumptions in policy",
                                    (imported.source,), (full_manifest.dataset_id,))
        dev_manifest = save_dataset(stage / "development-dataset.json", development, provenance=derived)
        val_origin = DatasetProvenance("AP1 validation 2024 only; hypothetical research execution costs",
                                       (imported.source,), (dev_manifest.dataset_id,))
        val_manifest = save_dataset(stage / "validation-dataset.json", validation, provenance=val_origin)
        # Read the exact staged target back through the public artifact verifier.
        for name in ("research", "development", "validation"):
            load_dataset(stage / f"{name}-dataset.json")
        _write_json(stage / "import-audit.json", imported.audit)
        _write_json(stage / "reference-policy.json", policy)
        _write_json(stage / "label-checks.json", _label_checks(training, development))
        body = {
            "format": OPEN_TRAINING_DATA_FORMAT, "dataset_id": dev_manifest.dataset_id,
            "dataset_artifact_id": dev_manifest.artifact_id, "instrument": development.instrument,
            "interval_microseconds": development.interval // timedelta(microseconds=1),
            "config": asdict(training.config), "feature_names": training.feature_names,
            "target_name": OPEN_TARGET_NAME,
            "splits": {"train": [asdict(row) for row in training.train],
                       "validation": [asdict(row) for row in training.validation], "test": []},
        }
        training_id = "sha256:" + sha256(json_bytes(body, default=_json_value)).hexdigest()
        training_path = stage / "training-data.json"
        _write_json(training_path, {**body, "training_data_id": training_id})
        model_report = baseline_report(training_path)
        model_path = stage / "ridge-model.json"
        _write_json(model_path, model_report)
        saved = load_ridge_model(model_path)
        if saved.target_name != OPEN_TARGET_NAME:
            raise ValueError("AP1 model has an incompatible target version")
        evaluation, points = validation_evaluation(model_path, training_path)
        if any(not _VALIDATION <= datetime.fromisoformat(point["at"]) < _TEST for point in points):
            raise ValueError("validation forecasts escaped their reserved time range")
        _write_json(stage / "validation-evaluation.json", evaluation)
        _write_json(stage / "validation-predictions.json", points)
        ridge = _backtest(stage, "ridge", {"name": "ridge_long", "model_report": "ridge-model.json",
            "entry_threshold": Decimal(0), "quantity": Decimal(1)}, validation, val_manifest, val_origin, runtime, profile)
        sma = _backtest(stage, "sma", {"name": "sma_timed", "fast_period": 4, "slow_period": 8,
            "holding_bars": 2, "quantity": Decimal(1)}, validation, val_manifest, val_origin, runtime, profile)
        for report in (ridge, sma):
            _assert_trade_alignment(report, training.validation)
        summary = {
            "format": "market-trader.ap1-summary.v1", "experiment": EXPERIMENT_ID,
            "training_rows": len(training.train), "validation_rows": len(training.validation),
            "test_metrics_computed": False, "reserved_test_start": _TEST.isoformat(),
            "reserved_test_end": _END.isoformat(), "target_name": OPEN_TARGET_NAME,
            "training_data_id": training_id, "model_report_id": model_report["report_id"],
            "development_dataset_id": development.dataset_id, "validation_dataset_id": validation.dataset_id,
            "source_sha256": imported.source.sha256, "validation_forecast": evaluation["validation"],
            "trading": {"ridge": trading_summary(ridge), "sma_timed": trading_summary(sma)},
            "data_audit": imported.audit, "policy": policy, "runtime": runtime,
        }
        _write_json(stage / "summary.json", summary)
        comparison = render_report(stage / "ridge-backtest.json", compare=stage / "sma-backtest.json",
                                   model=model_path, dataset=stage / "validation-dataset.json")
        publish_new_file(stage / "report.html", enrich_reference_report(comparison, summary=summary))
        if _source_id() != runtime["code_id"]:
            raise ValueError("source changed during reference run; retry from a stable checkout")
        files = {path.name: "sha256:" + sha256(path.read_bytes()).hexdigest() for path in sorted(stage.iterdir())}
        _write_json(stage / "manifest.json", {"format": "market-trader.ap1-manifest.v1", "status": "complete",
            "experiment": EXPERIMENT_ID, "test_metrics_computed": False, "runtime": runtime, "artifacts": files})
        _publish_directory(stage, output)
    return {"experiment": EXPERIMENT_ID, "training_rows": len(training.train),
            "validation_rows": len(training.validation), "test_metrics_computed": False,
            "model_report_id": model_report["report_id"], "validation_dataset_id": validation.dataset_id}
