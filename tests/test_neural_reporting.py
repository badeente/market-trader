"""Saved synthetic MLP trade reconstruction, not held-out evaluation."""
import json
from dataclasses import replace
import subprocess
import sys

import pytest

from market_trader.artifacts import DatasetProvenance
from market_trader.reporting import render_report
from market_trader.storage import save_dataset
from market_trader.training_data import candle_features
from tests.test_model_strategy import dataset
from tests.test_neural_strategy import mlp_report, learned_bars, at
from market_trader.calendars import SessionCalendar, TradingSession
from tests.test_report_cli import saved_run, report_data


@pytest.fixture
def mlp_run(tmp_path, mlp_report):
    path, body = mlp_report
    data = dataset(learned_bars())
    stored = tmp_path / "dataset.json"
    save_dataset(stored, data, provenance=DatasetProvenance("synthetic MLP test only"))
    config = tmp_path / "learned.toml"
    config.write_text('version = 1\ndataset_artifact = "dataset.json"\n'
                      '[strategy]\nname = "mlp_long"\nmodel_report = "mlp.json"\n'
                      'entry_threshold = 0.01\nquantity = 2\n'
                      '[execution]\nspread = 0.2\nslippage = 0.1\ncommission_rate = 0.001\n')
    return saved_run(tmp_path, config), stored, path, data


def test_report_reconstructs_nonlinear_forecast_without_fake_contributions(tmp_path, mlp_run):
    run, stored, path, data = mlp_run
    output = tmp_path / "mlp.html"
    output.write_bytes(render_report(run, model=path, dataset=stored))
    html, report = report_data(output)
    prediction = report["runs"][0]["trades"][0]["prediction"]
    assert prediction["forecast"] == pytest.approx(0.015)
    assert prediction["threshold"] == 0.01
    assert prediction["at"] == data.candles[7].close_time.isoformat()
    assert prediction["attribution"] == "nonlinear_no_additive_attribution"
    assert [row["value"] for row in prediction["inputs"]] == list(map(str, candle_features(data.candles[:8])))
    assert all("contribution" not in row for row in prediction["inputs"])
    assert "Nonlinear model: no additive feature attribution" in html
    assert report["forecast"] is None


def test_neural_report_never_invokes_legacy_heldout_evaluation(mlp_run, monkeypatch):
    run, stored, path, data = mlp_run
    def forbidden(*args):
        pytest.fail("AP3 must not call held-out evaluation")
    monkeypatch.setattr("market_trader.reporting.held_out_evaluation", forbidden)
    with pytest.raises(ValueError, match="MLP.*held-out"):
        render_report(run, model=path, dataset=stored, training_data=path)


@pytest.mark.parametrize("field,value", [("horizon_bars", 3), ("name", "ridge_long")])
def test_report_rejects_mlp_strategy_metadata_mismatch(mlp_run, field, value):
    run, stored, path, data = mlp_run
    body = json.loads(run.read_text())
    body["strategy"][field] = value
    run.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="model.*(horizon|kind)"):
        render_report(run, model=path, dataset=stored)


def test_loader_strategy_and_report_inference_do_not_import_torch_or_numpy(mlp_run):
    run, stored, path, data = mlp_run
    code = r"""
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        raise AssertionError('optional training dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from pathlib import Path
from decimal import Decimal
from market_trader.neural import load_mlp_model
from market_trader.strategies import configure_strategy
from market_trader.storage import load_dataset
from market_trader.backtesting import run_backtest
from market_trader.reporting import render_report
run, dataset, model = map(Path, sys.argv[1:])
saved = load_mlp_model(model)
data = load_dataset(dataset).dataset
strategy = configure_strategy({'name': 'mlp_long', 'model_report': model.name,
                               'entry_threshold': Decimal('0.01'), 'quantity': 2}, config_dir=model.parent)
result = run_backtest(data, strategy.create(data), lookback=strategy.lookback)
assert len(result.trades) == 1
assert b'nonlinear_no_additive_attribution' in render_report(run, model=model, dataset=dataset)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code, str(run), str(stored), str(path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field,value,diagnostic", [
    ("model_report_id", "sha256:" + "0" * 64, "model does not match"),
    ("entry_threshold", "0.02", "forecast"),
    ("lookback", 7, "lookback"),
    ("dataset_id", "sha256:" + "0" * 64, "dataset does not match"),
])
def test_report_rejects_identity_or_forecast_mismatch(mlp_run, field, value, diagnostic):
    run, stored, path, data = mlp_run
    body = json.loads(run.read_text())
    target = body if field == "dataset_id" else body["strategy"]
    target[field] = value
    run.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=diagnostic):
        render_report(run, model=path, dataset=stored)


@pytest.mark.parametrize("content", ["[]", "{invalid", '{"format": []}',
                                     '{"format":"market-trader.mlp-validation.v1"}'])
def test_report_invalid_model_json_is_a_value_error(mlp_run, content):
    run, stored, path, data = mlp_run
    path.write_text(content)
    with pytest.raises(ValueError):
        render_report(run, model=path, dataset=stored)


def test_report_rejects_unavailable_inputs(mlp_run):
    run, stored, path, data = mlp_run
    source = tuple(replace(bar, available_at=at(9)) if i == 7 else bar
                   for i, bar in enumerate(data.candles))
    bound = dataset(source)
    stored.unlink()
    manifest = save_dataset(stored, bound, provenance=DatasetProvenance("synthetic delayed inputs"))
    body = json.loads(run.read_text())
    body["dataset_id"] = bound.dataset_id
    body["dataset_artifact_id"] = manifest.artifact_id
    run.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="unavailable candles"):
        render_report(run, model=path, dataset=stored)


def test_report_rejects_session_ineligible_entry(mlp_run):
    run, stored, path, data = mlp_run
    calendar = SessionCalendar("synthetic", at(0), at(12),
                               (TradingSession(at(0), at(10)), TradingSession(at(10), at(12))))
    bound = dataset(data.candles, calendar)
    stored.unlink()
    manifest = save_dataset(stored, bound, provenance=DatasetProvenance("synthetic session gate"))
    body = json.loads(run.read_text())
    body["dataset_id"] = bound.dataset_id
    body["dataset_artifact_id"] = manifest.artifact_id
    run.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="session"):
        render_report(run, model=path, dataset=stored)


def test_report_rejects_training_overlap_even_with_matching_dataset_ids(mlp_run):
    run, stored, path, data = mlp_run
    shifted = tuple(replace(bar, open_time=bar.open_time.replace(year=2023),
                            close_time=bar.close_time.replace(year=2023),
                            available_at=bar.available_at.replace(year=2023)) for bar in data.candles)
    old = dataset(shifted)
    stored.unlink()
    manifest = save_dataset(stored, old, provenance=DatasetProvenance("synthetic overlap"))
    body = json.loads(run.read_text().replace("2024-01-02", "2023-01-02"))
    body["dataset_id"] = old.dataset_id
    body["dataset_artifact_id"] = manifest.artifact_id
    run.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="training cutoff"):
        render_report(run, model=path, dataset=stored)
