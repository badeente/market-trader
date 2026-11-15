"""Synthetic AP2 execution through the actual AP1-compatible model/ledger stack."""
import json
from hashlib import sha256
from importlib.util import find_spec

import pytest

from tests.test_reference import reference_fixture


def settings(tmp_path, extra='', heldout_shift=0, name='trial'):
    from market_trader.experiment_config import read_experiment_config
    source, old = reference_fixture(tmp_path, heldout_shift=heldout_shift)
    path = tmp_path / 'ap2.toml'
    path.write_text('version=1\nname=' + json.dumps(name) + '\nhypothesis="causal test"\nseed=0\n'
                    'source_csv="source.csv"\nsource_sha256="sha256:' + sha256(source.read_bytes()).hexdigest() + '"\n' + extra)
    return read_experiment_config(path), old


def read(output, name):
    return json.loads((output / name).read_text())


def test_defaults_are_actual_ap1_equivalent_training_and_accounting(tmp_path, monkeypatch):
    assert find_spec('market_trader.experiment_pipeline'), 'AP2 pipeline missing'
    from market_trader.experiment_pipeline import execute_experiment
    from market_trader.reference import run_reference
    from market_trader import baseline
    monkeypatch.setattr(baseline, 'held_out_evaluation', lambda *a: pytest.fail('reserved evaluation called'))
    config, ap1 = settings(tmp_path)
    first, second, reference = (tmp_path / name for name in ('first', 'second', 'reference'))
    summary = execute_experiment(config, first)
    assert summary == read(first, 'summary.json')
    assert summary == execute_experiment(config, second)
    run_reference(ap1, reference)
    old = read(reference, 'summary.json')
    for key in ('training_rows', 'validation_rows', 'validation_forecast', 'trading'):
        assert summary[key] == old[key]
    assert summary['training_rows'] == summary['validation_rows'] == 48
    assert summary['test_metrics_computed'] is False
    assert read(first, 'training-data.json')['splits']['test'] == []
    assert read(first, 'ridge-model.json')['model'] == read(reference, 'ridge-model.json')['model']
    assert summary['comparison_signature']['execution']['spread'] == '2.0'
    manifest = read(first, 'manifest.json')
    assert manifest['status'] == 'complete'
    assert set(manifest['artifacts']) == {p.name for p in first.iterdir()} - {'manifest.json'}
    for name, checksum in manifest['artifacts'].items():
        assert checksum == 'sha256:' + sha256((first / name).read_bytes()).hexdigest()
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / 'manifest.json').read_bytes() == (second / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        execute_experiment(config, first)


@pytest.mark.parametrize('lookback,horizon,fast', [(6, 3, 2), (2, 31, 1)])
def test_configured_horizon_labels_and_exits(tmp_path, lookback, horizon, fast):
    from datetime import datetime
    from decimal import Decimal, localcontext, Context, ROUND_HALF_EVEN
    from market_trader.experiment_pipeline import execute_experiment
    config, _ = settings(tmp_path,
        f'[training]\nlookback_bars={lookback}\nhorizon_bars={horizon}\n'
        f'[strategy]\nsma_fast_period={fast}\nsma_slow_period={lookback}\n'
        '[execution]\nspread=0\nslippage=0\n')
    output = tmp_path / 'result'
    summary = execute_experiment(config, output)
    assert summary['training_rows'] == summary['validation_rows'] == 2 * (34-lookback-horizon)
    training = read(output, 'training-data.json')
    for row in training['splits']['validation']:
        assert (datetime.fromisoformat(row['exit_at']) - datetime.fromisoformat(row['entry_at'])).total_seconds() == horizon*900
    for check in read(output, 'label-checks.json'):
        with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
            assert Decimal(check['target_return']) == ((Decimal(check['exit_reference_open']) - Decimal(check['entry_reference_open'])) / Decimal(check['entry_reference_open']))
        assert datetime.fromisoformat(check['target_available_at']) > datetime.fromisoformat(check['exit_at'])
    rows = {r['decision_at']: r for r in training['splits']['validation']}
    for name in ('ridge', 'sma'):
        backtest = read(output, f'{name}-backtest.json')
        for trade in backtest['trades']:
            row = rows[trade['entry']['order']['decision_at']]
            assert trade['entry']['candle_open_time'] == row['entry_at']
            assert trade['exit']['candle_open_time'] == row['exit_at']
        assert Decimal(summary['trading']['ridge' if name == 'ridge' else 'sma_timed']['modeled_cost_points']) == 0
    report = (output / 'report.html').read_text()
    assert f'horizon {horizon} bars ({horizon*15} minutes)' in report
    assert f'SMA {fast}/{lookback}' in report


def test_costs_are_real_fills_and_comparison_conditions(tmp_path):
    from dataclasses import replace
    from decimal import Decimal
    from market_trader.experiment_pipeline import execute_experiment
    config, _ = settings(tmp_path)
    original = execute_experiment(config, tmp_path / 'default')
    changed = execute_experiment(replace(config, spread=Decimal('4'), slippage=Decimal('0.25'),
        commission_rate=Decimal('0.0001'), price_tick=Decimal('0.01'), initial_capital=Decimal('120000')), tmp_path / 'costs')
    assert original['validation_forecast'] == changed['validation_forecast']
    assert original['comparison_signature'] != changed['comparison_signature']
    assert original['comparison_signature']['execution']['quantity_step'] == '1'
    assert changed['trading']['ridge']['modeled_cost_points'] != original['trading']['ridge']['modeled_cost_points']
    fills = read(tmp_path / 'costs', 'ridge-backtest.json')['trades']
    assert fills and Decimal(fills[0]['entry']['fee']) > 0
    assert Decimal(fills[0]['entry']['price']) - Decimal(fills[0]['entry']['reference_price']) >= Decimal('2.25')
    threshold = execute_experiment(replace(config, entry_threshold=Decimal('1')), tmp_path / 'threshold')
    assert threshold['comparison_signature'] == original['comparison_signature']
    assert threshold['trading']['ridge']['trade_count'] == 0
    assert threshold['config']['strategy']['entry_threshold'] == '1'


def test_reserved_mutation_never_reaches_features_fit_or_validation(tmp_path, monkeypatch):
    from datetime import datetime, UTC
    from market_trader import baseline, training_data
    from market_trader.experiment_pipeline import execute_experiment
    original_features, original_fit = training_data.candle_features, baseline._fit_ridge
    fits = []
    def checked_features(history):
        assert all(bar.open_time < datetime(2025, 1, 1, tzinfo=UTC) for bar in history)
        return original_features(history)
    def checked_fit(rows):
        assert all(row.target_available_at < datetime(2024, 1, 1, tzinfo=UTC) for row in rows)
        fits.append(len(rows))
        return original_fit(rows)
    monkeypatch.setattr(training_data, 'candle_features', checked_features)
    monkeypatch.setattr(baseline, '_fit_ridge', checked_fit)
    monkeypatch.setattr(baseline, 'held_out_evaluation', lambda *a: pytest.fail('reserved evaluation called'))
    outputs = []
    for name, shift in (('original', 0), ('mutated', 10000)):
        folder = tmp_path / name
        folder.mkdir()
        config, _ = settings(folder, heldout_shift=shift)
        output = folder / 'result'
        execute_experiment(config, output)
        outputs.append(output)
    left, right = outputs
    assert fits == [48, 48]
    assert read(left, 'ridge-model.json')['model'] == read(right, 'ridge-model.json')['model']
    for key in ('validation_forecast', 'trading'):
        assert read(left, 'summary.json')[key] == read(right, 'summary.json')[key]
    assert read(left, 'validation-predictions.json') == read(right, 'validation-predictions.json')
    for name in ('ridge', 'sma'):
        assert read(left, f'{name}-backtest.json')['trades'] == read(right, f'{name}-backtest.json')['trades']


def test_metadata_is_escaped_and_runtime_tracks_report_template(tmp_path):
    from dataclasses import replace
    from decimal import Decimal
    from market_trader.experiment_pipeline import execute_experiment
    config, _ = settings(tmp_path, name='<script>alert(1)</script>')
    config = replace(config, hypothesis='<img src=x onerror=alert(2)>', spread=Decimal('5.5'))
    output = tmp_path / 'result'
    summary = execute_experiment(config, output)
    report = (output / 'report.html').read_text()
    assert '<script>alert(1)</script>' not in report
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in report
    assert '<img src=x' not in report
    assert '&lt;img src=x' in report
    assert '5.5' in report
    assert 'AP1' not in report
    assert summary['runtime']['report_template_sha256'].startswith('sha256:')


def test_bad_source_does_not_publish_output(tmp_path):
    from market_trader.experiment_pipeline import execute_experiment
    config, _ = settings(tmp_path)
    config.source_csv.write_text(config.source_csv.read_text().replace('20000', '20001'))
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        execute_experiment(config, output)
    assert not output.exists()
