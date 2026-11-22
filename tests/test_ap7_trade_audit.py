"""Synthetic ledger/image audit; no fitting or optional framework imports."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from tests.test_neural_reporting import mlp_run
from tests.test_neural_strategy import mlp_report


@pytest.fixture
def cnn_run(mlp_run, monkeypatch):
    from market_trader import image_neural
    run, stored, model_path, data = mlp_run
    report = json.loads(run.read_text())
    report['strategy']['name'] = 'cnn_long'
    run.write_text(json.dumps(report))
    calls = []
    def predict(window, *, decision_at):
        calls.append((tuple(window), decision_at))
        return 0.015
    model = image_neural.SavedCNNModel(report_id=report['strategy']['model_report_id'], instrument=data.instrument,
        interval=data.interval, config=SimpleNamespace(lookback_bars=8, horizon_bars=2), model=None,
        last_target_available_at=data.candles[0].open_time - timedelta(days=1))
    monkeypatch.setattr(image_neural.SavedCNNModel, 'predict_window', lambda self, window, *, decision_at: predict(window, decision_at=decision_at), raising=False)
    monkeypatch.setattr(image_neural, 'load_cnn_model', lambda path: model, raising=False)
    at = data.candles[7].close_time.isoformat()
    predictions = stored.with_name('predictions.json')
    predictions.write_text(json.dumps([{'at': at, 'predicted': .015, 'actual': .02}]))
    return run, stored, model_path, predictions, data, calls


def test_first_executed_trade_is_reconstructed_bound_and_rendered(cnn_run):
    from market_trader.image_trade_audit import build_trade_image_audit, render_trade_image_audit
    from market_trader.chart_images import image_key
    run, stored, model, predictions, data, calls = cnn_run
    audit = build_trade_image_audit(run, model, stored, predictions)
    assert audit['mode'] == 'trades'
    assert audit['audited_count'] == audit['total_trades'] == 1
    sample, = audit['samples']
    assert sample['image_key'] == image_key(data.candles[:8], decision_at=data.candles[7].close_time)
    assert calls == [(data.candles[:8], data.candles[7].close_time)]
    assert sample['trade'] == json.loads(run.read_text())['trades'][0]
    assert sample['forecast'] == sample['saved_forecast'] == .015
    assert sample['image_data_uri'].startswith('data:image/png;base64,')
    assert 'actual' not in sample['window']
    html = render_trade_image_audit(audit)
    assert '1 of 1' in html and '<img ' in html and 'data:image/png;base64,' in html
    assert 'Outcomes are outside input pixels' in html


def test_saved_forecast_mismatch_is_not_published(cnn_run):
    from market_trader.image_trade_audit import build_trade_image_audit
    run, stored, model, predictions, data, calls = cnn_run
    points = json.loads(predictions.read_text()); points[0]['predicted'] = .016
    predictions.write_text(json.dumps(points))
    with pytest.raises(ValueError, match='saved forecast'):
        build_trade_image_audit(run, model, stored, predictions)


def test_cnn_ledger_report_uses_actual_window_not_numerical_features(cnn_run, monkeypatch):
    from market_trader import reporting, image_neural
    from tests.test_report_cli import report_data
    run, stored, model, predictions, data, calls = cnn_run
    monkeypatch.setattr(reporting, 'load_frozen_model', image_neural.load_cnn_model)
    def forbidden(*args):
        pytest.fail('CNN must never call numerical candle_features')
    monkeypatch.setattr(reporting, 'candle_features', forbidden)
    html = reporting.render_report(run, model=model, dataset=stored)
    out = run.with_suffix('.html'); out.write_bytes(html)
    _, report = report_data(out)
    prediction = report['runs'][0]['trades'][0]['prediction']
    assert prediction['model_kind'] == 'cnn'
    assert prediction['forecast'] == .015 and prediction['inputs'] == []
    assert prediction['attribution'] == 'nonlinear_no_additive_attribution'
    assert b'CNN entry signal' in html


@pytest.mark.parametrize('mutation', ['image_key', 'window', 'uri', 'count'])
def test_renderer_rejects_incoherent_or_external_images(cnn_run, mutation):
    from market_trader.image_trade_audit import build_trade_image_audit, render_trade_image_audit
    run, stored, model, predictions, data, calls = cnn_run
    audit = build_trade_image_audit(run, model, stored, predictions)
    sample = audit['samples'][0]
    if mutation == 'image_key':
        sample['image_key'] = 'sha256:' + '0'*64
    elif mutation == 'window':
        sample['window']['candles'][0]['close'] = '1'
    elif mutation == 'uri':
        sample['image_data_uri'] = 'file:///private/image.png'
    else:
        audit['audited_count'] = 0
    with pytest.raises(ValueError):
        render_trade_image_audit(audit)


def test_zero_trade_audit_shows_evaluated_forecast_without_inventing_trade(cnn_run):
    from market_trader.image_trade_audit import build_trade_image_audit, render_trade_image_audit
    from tests.test_report_cli import saved_run
    run, stored, model, predictions, data, calls = cnn_run
    config = run.with_suffix('.toml')
    config.write_text(config.read_text().replace('entry_threshold = 0.01', 'entry_threshold = 1'))
    run = saved_run(run.parent, config)
    body = json.loads(run.read_text()); body['strategy']['name'] = 'cnn_long'
    run.write_text(json.dumps(body))
    audit = build_trade_image_audit(run, model, stored, predictions)
    assert audit['total_trades'] == 0 and audit['total_decisions'] == 1
    assert audit['mode'] == 'decisions' and audit['audited_count'] == 1
    assert audit['samples'][0]['trade'] is None
    html = render_trade_image_audit(audit)
    assert 'No executed CNN trades' in html and '1 of 1' in html


@pytest.mark.parametrize('limit', [True, 0, 9, 1.5])
def test_audit_rejects_invalid_limits_before_loading(limit):
    from market_trader.image_trade_audit import build_trade_image_audit
    with pytest.raises(ValueError, match='limit'):
        build_trade_image_audit('missing', 'missing', 'missing', 'missing', limit=limit)


def test_cnn_report_refuses_legacy_heldout_evaluation_explicitly(cnn_run, monkeypatch):
    from market_trader import reporting, image_neural
    run, stored, model, predictions, data, calls = cnn_run
    monkeypatch.setattr(reporting, 'load_frozen_model', image_neural.load_cnn_model)
    with pytest.raises(ValueError, match='CNN.*held-out'):
        reporting.render_report(run, model=model, training_data=model)


def test_real_stdlib_saved_cnn_first_eight_of_many_synthetic_trades(tmp_path):
    from market_trader._dataset_codec import json_bytes
    from market_trader.artifacts import DatasetProvenance
    from market_trader.storage import save_dataset
    from market_trader.image_neural import load_cnn_model
    from market_trader.image_trade_audit import build_trade_image_audit, render_trade_image_audit
    from market_trader.reporting import render_report
    from tests.test_image_neural import saved_body, seal
    from tests.test_model_strategy import dataset
    from tests.test_neural_strategy import bars
    from tests.test_report_cli import saved_run
    body = saved_body(); body['model']['target_mean'] = .015
    model = tmp_path / 'cnn.json'; model.write_bytes(json_bytes(seal(body)))
    data = dataset(tuple(replace(bar, instrument='deuidxeur') for bar in bars([(100, 101)]*50)))
    stored = tmp_path / 'dataset.json'
    save_dataset(stored, data, provenance=DatasetProvenance('synthetic CNN audit only'))
    config = tmp_path / 'cnn-run.toml'
    config.write_text('version = 1\ndataset_artifact = "dataset.json"\n'
        '[strategy]\nname = "cnn_long"\nmodel_report = "cnn.json"\nentry_threshold = 0.01\nquantity = 1\n[execution]\nspread = 0\n')
    run = saved_run(tmp_path, config)
    ledger = json.loads(run.read_text())
    saved = load_cnn_model(model)
    predictions = tmp_path / 'predictions.json'
    predictions.write_text(json.dumps([dict(at=data.candles[i].close_time.isoformat(),
        predicted=saved.predict_window(data.candles[i-7:i+1], decision_at=data.candles[i].close_time), actual=0.)
        for i in range(7, len(data.candles)-3)]))
    audit = build_trade_image_audit(run, model, stored, predictions)
    assert audit['total_trades'] == len(ledger['trades']) > 8
    assert audit['audited_count'] == 8
    assert [sample['trade'] for sample in audit['samples']] == ledger['trades'][:8]
    assert len(render_trade_image_audit(audit).encode()) < 4*1024*1024
    assert b'"model_kind":"cnn"' in render_report(run, model=model, dataset=stored)


def test_audit_rejects_unavailable_input_even_when_dataset_ids_rebound(cnn_run):
    from market_trader.artifacts import DatasetProvenance
    from market_trader.storage import save_dataset
    from market_trader.image_trade_audit import build_trade_image_audit
    from tests.test_model_strategy import dataset
    run, stored, model, predictions, data, calls = cnn_run
    changed = dataset(tuple(replace(c, available_at=c.available_at + timedelta(minutes=15))
        if i == 7 else c for i,c in enumerate(data.candles)))
    stored.unlink(); manifest = save_dataset(stored, changed, provenance=DatasetProvenance('synthetic delayed input'))
    ledger = json.loads(run.read_text()); ledger['dataset_id'] = changed.dataset_id
    ledger['dataset_artifact_id'] = manifest.artifact_id; run.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match='causal window'):
        build_trade_image_audit(run, model, stored, predictions)
    assert calls == []


def test_audit_has_explicit_zero_based_ledger_index(cnn_run):
    from market_trader.image_trade_audit import build_trade_image_audit
    run, stored, model, predictions, data, calls = cnn_run
    audit = build_trade_image_audit(run, model, stored, predictions)
    assert audit['samples'][0]['ledger_trade_index'] == 0


def test_audit_ledger_validation_uses_frozen_decimal_context(cnn_run):
    from decimal import localcontext, ROUND_DOWN
    from market_trader.image_trade_audit import build_trade_image_audit
    run, stored, model, predictions, data, calls = cnn_run
    expected = build_trade_image_audit(run, model, stored, predictions)
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN
        actual = build_trade_image_audit(run, model, stored, predictions)
    assert actual == expected


def test_zero_completed_trades_does_not_hide_executed_open_position(cnn_run):
    from market_trader.artifacts import DatasetProvenance
    from market_trader.storage import save_dataset
    from market_trader.image_trade_audit import build_trade_image_audit, render_trade_image_audit
    from tests.test_model_strategy import dataset
    from tests.test_report_cli import saved_run
    run, stored, model, predictions, data, calls = cnn_run
    stored.unlink()
    save_dataset(stored, dataset(data.candles[:9]), provenance=DatasetProvenance('synthetic open entry'))
    run = saved_run(run.parent, run.with_suffix('.toml'))
    ledger = json.loads(run.read_text()); ledger['strategy']['name'] = 'cnn_long'
    run.write_text(json.dumps(ledger))
    assert ledger['trades'] == [] and ledger['position'] is not None
    audit = build_trade_image_audit(run, model, stored, predictions)
    assert audit['open_position'] == ledger['position']
    html = render_trade_image_audit(audit)
    assert 'executed open entry' in html and 'No executed CNN trades' not in html
