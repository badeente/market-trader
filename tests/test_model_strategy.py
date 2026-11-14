"""Synthetic execution checks for the frozen ridge strategy."""

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

import pytest

from market_trader import baseline
from market_trader.artifacts import DatasetProvenance
from market_trader.backtesting import run_backtest
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.domain import Candle
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.storage import save_dataset
from market_trader.strategies import configure_strategy
from market_trader.training_data import FEATURE_NAMES, TARGET_NAME

START = datetime(2026, 1, 2, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def at(index):
    return START + index * MINUTE


def bars(prices):
    return tuple(
        Candle(
            "TEST", MINUTE, at(index), at(index + 1), at(index + 1),
            Decimal(opened), Decimal(max(opened, closed)), Decimal(min(opened, closed)),
            Decimal(closed), Decimal(1),
        )
        for index, (opened, closed) in enumerate(prices)
    )


@pytest.fixture
def candles():
    return bars([(100, 100), (100, 102), (103, 103), (103, 102), (106, 102), (110, 102)])


def dataset(candles, calendar="continuous"):
    result, = normalize_candles(candles, config=NormalizationConfig(calendar)).datasets
    return result


def write_model(path, payload):
    body = {key: value for key, value in payload.items() if key != "report_id"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    report_id = "sha256:" + sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps({**body, "report_id": report_id}), encoding="utf-8")
    return report_id


@pytest.fixture
def frozen_report(tmp_path):
    report = {
        "format": baseline.BASELINE_FORMAT,
        "dataset_id": "sha256:" + "a" * 64,
        "dataset_artifact_id": "sha256:" + "b" * 64,
        "training_data_id": "sha256:" + "c" * 64,
        "instrument": "TEST",
        "interval_microseconds": 60_000_000,
        "feature_names": list(FEATURE_NAMES),
        "target_name": TARGET_NAME,
        "config": {
            "lookback_bars": 2, "horizon_bars": 2,
            "validation_start": at(-60).isoformat(), "test_start": at(0).isoformat(),
        },
        "model": {
            "kind": "ridge", "alpha": 0.1, "means": [0.005, 0, 0, 0],
            "scales": [0.5, 1, 1, 1], "intercept": 0, "coefficients": [0.5, 0, 0, 0],
        },
        "training": {
            "row_count": 5, "first_decision_at": at(-1440).isoformat(),
            "last_target_available_at": at(-120).isoformat(),
        },
        "validation": {"row_count": 1, "ridge": {"mae": 0, "rmse": 0},
                       "zero_return": {"mae": 0, "rmse": 0}},
    }
    path = tmp_path / "frozen.json"
    write_model(path, report)
    return path, report


def configured(path):
    return configure_strategy({
        "name": "ridge_long", "model_report": path.name,
        "entry_threshold": Decimal("0.01"), "quantity": 2,
    }, config_dir=path.parent)


@pytest.mark.case("ML-005")
def test_model_backtest_cli_uses_frozen_report_and_charges_costs(candles, frozen_report, tmp_path):
    path, _ = frozen_report
    calendar = SessionCalendar("test session", at(0), at(6), (TradingSession(at(0), at(6)),))
    manifest = save_dataset(
        tmp_path / "dataset.json", dataset(candles, calendar), provenance=DatasetProvenance("synthetic"),
    )
    config = tmp_path / "backtest.toml"
    config.write_text(
        'version = 1\ndataset_artifact = "dataset.json"\n'
        '[strategy]\nname = "ridge_long"\nmodel_report = "frozen.json"\n'
        'entry_threshold = 0.01\nquantity = 2\n'
        '[execution]\nspread = 0.2\nslippage = 0.1\ncommission_rate = 0.001\n',
        encoding="utf-8",
    )
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    before = path.read_bytes()
    command = [
        sys.executable, "-m", "market_trader.cli", "backtest", "--config", str(config),
    ]
    run = subprocess.run(command, cwd=other_cwd, capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    assert path.read_bytes() == before
    report = json.loads(run.stdout)
    assert report["dataset_artifact_id"] == manifest.artifact_id
    assert report["strategy"]["model_report_id"] == json.loads(before)["report_id"]
    assert report["strategy"]["lookback"] == 2
    assert report["strategy"]["horizon_bars"] == 2
    entry, exit_fill = report["fills"]
    assert [fill["candle_open_time"] for fill in (entry, exit_fill)] == [at(2).isoformat(), at(4).isoformat()]
    assert [fill["order"]["decision_at"] for fill in (entry, exit_fill)] == [at(2).isoformat(), at(4).isoformat()]
    assert [(fill["phase"], fill["reason"]) for fill in (entry, exit_fill)] == [("open", "entry"), ("open", "exit")]
    assert [Decimal(fill["price"]) for fill in (entry, exit_fill)] == [Decimal("103.2"), Decimal("105.8")]
    assert [Decimal(fill["fee"]) for fill in (entry, exit_fill)] == [Decimal("0.2064"), Decimal("0.2116")]
    assert Decimal(report["trades"][0]["net_pnl"]) == Decimal("4.782")
    assert Decimal(report["accounts"][-1]["equity"]) == Decimal("1004.782")
    assert report["unfilled_order"] is None and not report["rejections"]


@pytest.mark.case("ML-005")
def test_inference_uses_only_available_candles_and_never_fits(candles, frozen_report, monkeypatch):
    def fail_fit(*args):
        pytest.fail("backtest attempted to fit the model")

    monkeypatch.setattr(baseline, "_fit_ridge", fail_fit)
    path, _ = frozen_report
    strategy = configured(path)
    data = dataset(candles)
    original = run_backtest(data, strategy.create(data), lookback=strategy.lookback)
    assert original.trades[0].entry.candle_open_time == at(2)
    assert original == run_backtest(data, strategy.create(data), lookback=strategy.lookback)

    delayed = list(candles)
    delayed[1] = replace(delayed[1], available_at=at(3))
    delayed_data = dataset(delayed)
    assert not run_backtest(
        delayed_data, strategy.create(delayed_data), lookback=strategy.lookback,
    ).fills


def test_time_exit_waits_for_next_eligible_decision(candles, frozen_report):
    delayed = list(candles)
    delayed[3] = replace(delayed[3], available_at=at(5))
    data = dataset(delayed)
    strategy = configured(frozen_report[0])
    result = run_backtest(data, strategy.create(data), lookback=strategy.lookback)
    trade, = result.trades
    assert trade.entry.candle_open_time == at(2)
    assert trade.exit.candle_open_time == at(5)


def test_entry_requires_a_session_with_room_for_the_scheduled_exit(frozen_report):
    source = bars([(100, 100), (100, 102), (102, 102), (102, 100), (100, 102)] + [(102, 102)] * 5)
    calendar = SessionCalendar(
        "adjacent sessions", at(0), at(10), (TradingSession(at(0), at(4)), TradingSession(at(4), at(10))),
    )
    data = dataset(source, calendar)
    strategy = configured(frozen_report[0])
    assert not run_backtest(data, strategy.create(data), lookback=strategy.lookback).fills


def test_forecast_must_exceed_the_configured_threshold(candles, frozen_report):
    path, report = frozen_report
    report["model"]["intercept"] = 0.01
    report["model"]["coefficients"] = [0, 0, 0, 0]
    write_model(path, report)
    data = dataset(candles)
    strategy = configured(path)
    assert not run_backtest(data, strategy.create(data), lookback=strategy.lookback).fills


@pytest.mark.case("ML-004")
def test_backtest_rejects_a_different_feature_lookback(candles, frozen_report):
    data = dataset(candles)
    strategy = configured(frozen_report[0])
    with pytest.raises(ValueError, match="model lookback"):
        run_backtest(data, strategy.create(data), lookback=3)


@pytest.mark.case("ML-004")
@pytest.mark.parametrize(("field", "value", "diagnostic"), [
    ("instrument", "OTHER", "instrument"),
    ("interval_microseconds", 120_000_000, "interval"),
    ("overlap", True, "training cutoff"),
])
def test_backtest_rejects_incompatible_model(candles, frozen_report, field, value, diagnostic):
    path, report = frozen_report
    if field == "overlap":
        report["training"]["last_target_available_at"] = at(0).isoformat()
        report["config"]["validation_start"] = at(60).isoformat()
        report["config"]["test_start"] = at(120).isoformat()
    else:
        report[field] = value
    write_model(path, report)
    strategy = configured(path)
    with pytest.raises(ValueError, match=diagnostic):
        strategy.create(dataset(candles))
