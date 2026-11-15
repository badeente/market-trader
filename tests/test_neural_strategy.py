"""Synthetic frozen-MLP integration tests; no real-data training."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from market_trader.strategies import configure_strategy
from market_trader.backtesting import ExecutionProfile, run_backtest
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.training_data import FEATURE_NAMES, OPEN_TARGET_NAME
from tests.test_model_strategy import bars as minute_bars, dataset, write_model


def at(index):
    return datetime(2024, 1, 2, tzinfo=UTC) + index * timedelta(minutes=15)


def bars(prices):
    return tuple(replace(bar, interval=timedelta(minutes=15), open_time=at(i),
                         close_time=at(i + 1), available_at=at(i + 1))
                 for i, bar in enumerate(minute_bars(prices)))


def learned_bars():
    return bars([(100, 100)] * 7 + [(100, 102), (103, 103), (103, 102),
                                  (106, 102), (110, 102)])


def configured(path, **changes):
    return configure_strategy({
        "name": "mlp_long", "model_report": path.name,
        "entry_threshold": Decimal("0.01"), "quantity": 2, **changes,
    }, config_dir=path.parent)


@pytest.mark.parametrize("content", ['{"format": "pickle"}', '{invalid', '[]', '{"format": []}'])
def test_mlp_strategy_rejects_unknown_or_invalid_artifact(tmp_path, content):
    path = tmp_path / "mlp.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="model.*(format|JSON|object)"):
        configured(path)


@pytest.fixture
def mlp_report(tmp_path):
    # Explicit synthetic weights: forecast = ReLU(return_1 - 0.005).
    layers = []
    for inputs, outputs in ((4, 16), (16, 8), (8, 1)):
        weights = [[0.0] * inputs for _ in range(outputs)]
        weights[0][0] = 1.0
        layers.append({"weights": weights, "biases": [0.0] * outputs})
    layers[0]["biases"][0] = -0.005
    body = {
        "format": "market-trader.mlp-validation.v1",
        "dataset_id": "sha256:" + "a" * 64,
        "dataset_artifact_id": "sha256:" + "b" * 64,
        "training_data_id": "sha256:" + "c" * 64,
        "instrument": "TEST", "interval_microseconds": 900_000_000,
        "feature_names": list(FEATURE_NAMES), "target_name": OPEN_TARGET_NAME,
        "config": {"lookback_bars": 8, "horizon_bars": 2,
                   "validation_start": "2024-01-01T00:00:00+00:00", "test_start": "2025-01-01T00:00:00+00:00"},
        "model": {"kind": "mlp", "architecture": [4, 16, 8, 1], "parameter_count": 225,
                  "means": [0.] * 4, "scales": [1.] * 4,
                  "target_mean": 0., "target_scale": 1., "layers": layers},
        "training": {"row_count": 5, "first_decision_at": "2021-01-04T00:00:00+00:00",
                     "last_target_available_at": "2023-12-29T00:00:00+00:00",
                     "fit_settings": {"optimizer": "AdamW", "learning_rate": 0.001,
                         "weight_decay": 0.0001, "batch_size": 128, "max_epochs": 100,
                         "patience": 10, "min_delta": 1e-6, "device": "cpu", "threads": 1,
                         "dtype": "float64", "deterministic": True, "seed": 0},
                     "runtime_versions": {"python": "synthetic", "torch": "synthetic", "numpy": "synthetic"},
                     "epoch_history": [{"epoch": i, "training_mse": 1., "validation_mse": 1.}
                                       for i in range(1, 12)],
                     "best_epoch": 1, "stopping_reason": "early_stopping"},
        "validation": {"row_count": 1, "mlp": {"mae": 0., "rmse": 0.},
                       "zero_return": {"mae": 0., "rmse": 0.}},
    }
    path = tmp_path / "mlp.json"
    write_model(path, body)
    return path, body


@pytest.mark.parametrize("field,value", [("target_name", "wrong"), ("kind", "ridge")])
def test_mlp_model_rejects_wrong_target_or_kind(mlp_report, field, value):
    path, body = mlp_report
    if field == "kind":
        body["model"][field] = value
    else:
        body[field] = value
    write_model(path, body)
    with pytest.raises(ValueError, match="target|kind"):
        configured(path)


def test_mlp_entry_timed_exit_uses_existing_cost_ledger_and_fresh_factory(mlp_report):
    strategy = configured(mlp_report[0])
    data = dataset(learned_bars())
    profile = ExecutionProfile(spread=Decimal("0.2"), slippage=Decimal("0.1"),
                               commission_rate=Decimal("0.001"))
    result = run_backtest(data, strategy.create(data), lookback=strategy.lookback, profile=profile)
    assert result == run_backtest(data, strategy.create(data), lookback=strategy.lookback, profile=profile)
    trade, = result.trades
    assert trade.entry.candle_open_time == at(8)
    assert trade.exit.candle_open_time == at(10)
    assert trade.entry.price == Decimal("103.2")
    assert trade.exit.price == Decimal("105.8")
    assert trade.net_pnl == Decimal("4.782")
    assert dict(strategy.parameters)["horizon_bars"] == 2


def test_shared_learned_clock_uses_saved_horizon_not_a_hardcoded_two(mlp_report):
    from market_trader.neural import load_mlp_model
    from market_trader.strategies import _learned_long
    from market_trader.backtesting import Signal, Action
    saved = load_mlp_model(mlp_report[0])
    saved = replace(saved, config=replace(saved.config, horizon_bars=3))
    data = dataset(learned_bars())
    decide = _learned_long(saved, Decimal("0.01"), Signal(Action.ENTER_LONG, quantity=Decimal(2)), data)
    result = run_backtest(data, decide, lookback=8)
    assert result.trades[0].exit.candle_open_time == at(11)


def test_mlp_entry_session_gate_and_candle_availability(mlp_report):
    strategy = configured(mlp_report[0])
    source = learned_bars()
    calendar = SessionCalendar("synthetic", at(0), at(12),
                               (TradingSession(at(0), at(10)), TradingSession(at(10), at(12))))
    for data in (dataset(source, calendar),
                 dataset(tuple(replace(bar, available_at=at(9)) if i == 7 else bar
                               for i, bar in enumerate(source)))):
        assert not run_backtest(data, strategy.create(data), lookback=strategy.lookback).fills


@pytest.mark.parametrize("problem", ["instrument", "interval", "overlap"])
def test_mlp_strategy_checks_bound_dataset_identity_and_training_boundary(mlp_report, problem):
    source = learned_bars()
    if problem == "instrument":
        source = tuple(replace(bar, instrument="OTHER") for bar in source)
    elif problem == "interval":
        source = minute_bars([(100, 100)] * 12)
    else:
        source = tuple(replace(bar, open_time=bar.open_time.replace(year=2023),
                               close_time=bar.close_time.replace(year=2023),
                               available_at=bar.available_at.replace(year=2023)) for bar in source)
    with pytest.raises(ValueError, match="instrument|interval|training cutoff"):
        configured(mlp_report[0]).create(dataset(source))


@pytest.mark.parametrize("changes", [{"extra": 1}, {"quantity": 0}, {"entry_threshold": -1},
                                      {"model_report": " mlp.json"}])
def test_mlp_strategy_requires_exact_configuration(mlp_report, changes):
    with pytest.raises(ValueError):
        configured(mlp_report[0], **changes)
