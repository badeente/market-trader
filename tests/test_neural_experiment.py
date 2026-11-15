"""AP3 versioned configuration and real synthetic experiment integration."""
from copy import deepcopy
from dataclasses import replace
import json
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import subprocess
import sys

import pytest

from market_trader.experiment_config import (
    experiment_config_from_payload, experiment_config_payload, read_experiment_config,
)
from tests.test_experiment_pipeline import settings, read


def v2_config(tmp_path, kind='mlp', extra=''):
    config, _ = settings(tmp_path)
    path = tmp_path / 'ap3.toml'
    path.write_text('version=2\nname="neural trial"\nhypothesis="causal test"\nseed=0\n'
                    'source_csv="source.csv"\nsource_sha256=' + json.dumps(config.source_sha256)
                    + '\n[model]\nkind=' + json.dumps(kind) + '\n' + extra)
    return read_experiment_config(path)


def test_v2_explicit_model_and_complete_frozen_snapshot(tmp_path):
    config = v2_config(tmp_path)
    assert config.model_kind == 'mlp'
    assert (config.lookback_bars, config.horizon_bars, config.sma_fast_period, config.sma_slow_period) == (8, 2, 4, 8)
    payload = experiment_config_payload(config)
    model = payload['model']
    assert model == {
        'kind': 'mlp', 'layers': [4, 16, 8, 1], 'hidden_activation': 'relu', 'output_activation': 'linear',
        'optimizer': 'AdamW', 'learning_rate': '0.001', 'weight_decay': '0.0001',
        'batch_size': 128, 'max_epochs': 100, 'patience': 10, 'min_delta': '0.000001',
        'loss': 'standardized_target_mse', 'input_scaler': 'train_only_standard',
        'target_scaler': 'train_only_standard', 'device': 'cpu', 'dtype': 'float64',
        'threads': 1, 'deterministic': True,
    }
    assert experiment_config_from_payload(json.loads(json.dumps(payload))) == config
    changed = deepcopy(payload)
    changed['model']['max_epochs'] = 99
    with pytest.raises(ValueError):
        experiment_config_from_payload(changed)
    incomplete = deepcopy(payload)
    del incomplete['model']['patience']
    with pytest.raises(ValueError):
        experiment_config_from_payload(incomplete)
    ridge = v2_config(tmp_path, 'ridge')
    assert experiment_config_payload(ridge)['model'] == {'kind': 'ridge', 'alpha': '0.1'}
    assert experiment_config_from_payload(experiment_config_payload(ridge)) == ridge
    for field, invalid in (('threads', True), ('batch_size', 128.0), ('layers', [4.0, 16, 8, 1])):
        malformed = deepcopy(payload)
        malformed['model'][field] = invalid
        with pytest.raises(ValueError):
            experiment_config_from_payload(malformed)
    legacy, _ = settings(tmp_path)
    assert legacy.model_kind == 'ridge'
    assert 'model' not in experiment_config_payload(legacy)


@pytest.mark.parametrize('replacement,extra', [
    ('', ''), ('[model]\nkind="forest"', ''), ('[model]', ''),
    ('[model]\nkind="mlp"\nmax_epochs=5', ''),
    ('[model]\nkind="ridge"\nalpha=0.2', ''),
    ('[model]\nkind="mlp"', '[training]\nlookback_bars=6\nhorizon_bars=3\n[strategy]\nsma_fast_period=2\nsma_slow_period=6\n'),
    ('[model]\nkind="mlp"', '[strategy]\nsma_fast_period=2\n'),
])
def test_v2_strict_model_contract(tmp_path, replacement, extra):
    old, _ = settings(tmp_path)
    path = tmp_path / 'invalid.toml'
    path.write_text('version=2\nname="trial"\nhypothesis="test"\nseed=0\nsource_csv="source.csv"\n'
                    'source_sha256=' + json.dumps(old.source_sha256) + '\n' + replacement + '\n' + extra)
    with pytest.raises(ValueError):
        read_experiment_config(path)


def test_ridge_runtime_records_neural_distributions_without_importing_them(tmp_path):
    config = v2_config(tmp_path, 'ridge')
    snapshot = tmp_path / 'snapshot.json'
    snapshot.write_text(json.dumps(experiment_config_payload(config)))
    script = '''
import builtins, json, sys
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy'):
        raise AssertionError('Ridge must not import numerical training dependencies')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from market_trader.experiment_config import experiment_config_from_payload
from market_trader.experiment_pipeline import execute_experiment
snapshot, output = map(Path, sys.argv[1:])
execute_experiment(experiment_config_from_payload(json.loads(snapshot.read_text())), output)
'''
    result = subprocess.run([sys.executable, '-c', script, str(snapshot), str(tmp_path / 'ridge')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    runtime = read(tmp_path / 'ridge', 'summary.json')['runtime']
    report = (tmp_path / 'ridge' / 'report.html').read_text()
    assert 'inverse target-transformed' not in report
    assert '(early stopping)' not in report
    distributions = {}
    for name in ('torch', 'numpy'):
        try:
            distributions[name] = version(name)
        except PackageNotFoundError:
            pass
    assert runtime.get('distributions', {}) == distributions
    from hashlib import sha256
    from pathlib import Path
    lock = Path(__file__).resolve().parents[1] / 'requirements' / 'ap3-cpu-lock.txt'
    assert runtime['dependency_lock_sha256'] == 'sha256:' + sha256(lock.read_bytes()).hexdigest()


@pytest.mark.skipif(find_spec('torch') is None, reason='AP3 training extra not installed')
def test_mlp_experiment_real_artifacts_reload_and_reference_cohort(tmp_path, monkeypatch):
    from datetime import datetime, UTC
    from hashlib import sha256
    from market_trader import training_data
    from market_trader.experiment_pipeline import execute_experiment
    config = v2_config(tmp_path)
    original = training_data.candle_features
    def guarded(history):
        assert all(bar.open_time < datetime(2025, 1, 1, tzinfo=UTC) for bar in history)
        return original(history)
    monkeypatch.setattr(training_data, 'candle_features', guarded)
    output = tmp_path / 'mlp'
    summary = execute_experiment(config, output)
    assert set(summary['validation_forecast']) == {'mlp', 'ridge', 'zero', 'zero_return', 'row_count', 'first_decision_at', 'last_decision_at'}
    assert summary['validation_forecast']['zero'] == summary['validation_forecast']['zero_return']
    assert set(summary['trading']) == {'mlp', 'ridge', 'sma_timed'}
    model = read(output, 'mlp-model.json')
    assert summary['model_report_id'] == model['report_id']
    assert summary['training_data_id'] == model['training_data_id']
    assert model['format'] == 'market-trader.mlp-validation.v1'
    history = read(output, 'training-history.json')
    assert history['format'] == 'market-trader.mlp-training-history.v1'
    assert history['model_report_id'] == model['report_id']
    assert history['training_data_id'] == model['training_data_id']
    assert history['epoch_history'] == model['training']['epoch_history']
    assert summary['training'] == model['training']
    assert 1 <= history['best_epoch'] <= len(history['epoch_history']) <= 100
    assert [e['epoch'] for e in history['epoch_history']] == list(range(1, len(history['epoch_history']) + 1))
    assert history['stopping_reason'] in ('early_stopping', 'max_epochs')
    rows = read(output, 'training-data.json')['splits']
    assert rows['test'] == []
    assert len(rows['train']) == len(rows['validation']) == 48
    from decimal import Decimal
    from market_trader.neural import load_mlp_model, mlp_validation_evaluation
    loaded = load_mlp_model(output / 'mlp-model.json')
    evaluation, points = mlp_validation_evaluation(output / 'mlp-model.json', output / 'training-data.json')
    assert read(output, 'mlp-validation-evaluation.json') == evaluation
    assert read(output, 'mlp-validation-predictions.json') == points
    assert read(output, 'validation-evaluation.json') == evaluation
    assert read(output, 'validation-predictions.json') == points
    assert summary['validation_forecast']['mlp'] == evaluation['validation']['mlp']
    ridge_evaluation = read(output, 'ridge-validation-evaluation.json')
    assert summary['validation_forecast']['ridge'] == ridge_evaluation['validation']['ridge']
    assert len(points) == 48
    for row, point in zip(rows['validation'], points, strict=True):
        assert point['at'] == row['decision_at']
        assert point['predicted'] == loaded.model.predict(tuple(map(Decimal, row['features'])))
    by_decision = {r['decision_at']: r for r in rows['validation']}
    for name in ('mlp', 'ridge', 'sma'):
        for trade in read(output, f'{name}-backtest.json')['trades']:
            row = by_decision[trade['entry']['order']['decision_at']]
            assert trade['entry']['candle_open_time'] == row['entry_at']
            assert trade['exit']['candle_open_time'] == row['exit_at']
    manifest = read(output, 'manifest.json')
    assert set(manifest['artifacts']) == {p.name for p in output.iterdir()} - {'manifest.json'}
    for name, checksum in manifest['artifacts'].items():
        assert checksum == 'sha256:' + sha256((output / name).read_bytes()).hexdigest()


def test_ap3_report_readable_units_dynamic_costs_history_and_escaping(tmp_path):
    from market_trader.experiment_reporting import enrich_experiment_report
    config = replace(v2_config(tmp_path), name='<script>name</script>',
                     hypothesis='<img src=x>', spread=__import__('decimal').Decimal('5.5'))
    # Deliberately small reporting fixture; these are not published experiment results.
    summary = {
        'config': experiment_config_payload(config),
        'comparison_signature': {'interval_microseconds': 900_000_000},
        'training_rows': 48, 'validation_rows': 48,
        'validation_forecast': {'mlp': {'mae': .0002, 'rmse': .0003},
            'ridge': {'mae': .0004, 'rmse': .0005}, 'zero': {'mae': .0006, 'rmse': .0007},
            'row_count': 48, 'first_decision_at': '2024-01-02T09:00:00+00:00',
            'last_decision_at': '2024-01-03T16:00:00+00:00'},
        'trading': {name: {'trade_count': 2, 'net_points': '7.25', 'reference_points': '12.75',
            'modeled_cost_points': '5.5', 'max_drawdown_points': '3.25'} for name in ('mlp', 'ridge', 'sma_timed')},
        'training': {'row_count': 48, 'best_epoch': 1, 'stopping_reason': 'early_stopping',
            'fit_settings': {'seed': 0, 'learning_rate': .001, 'max_epochs': 100},
            'epoch_history': [{'epoch': 1, 'training_mse': .75, 'validation_mse': 1.25}]},
        'training_data_id': 'data-id', 'model_report_id': 'model-id', 'validation_dataset_id': 'validation-id',
        'runtime': {'code_id': 'code-id'},
    }
    report = enrich_experiment_report(b'<html><body><p>verified ledger</p></body></html>', summary=summary).decode()
    assert 'AP3' in report and 'verified ledger' in report
    assert '<script>name</script>' not in report and '&lt;script&gt;name&lt;/script&gt;' in report
    assert '<img src=x>' not in report and '&lt;img src=x&gt;' in report
    assert '<table id="ap3-forecast"' in report and '<table id="ap3-trading"' in report
    assert '<th>MAE (bp)</th>' in report and '<th>RMSE (bp)</th>' in report
    assert '<td>2.0000</td>' in report and '<td>3.0000</td>' in report
    assert '<td>MLP</td>' in report and '<td>Ridge</td>' in report and '<td>Zero</td>' in report
    assert 'Net points' in report and '<td>7.25</td>' in report and 'synthetic Q, not EUR' in report
    assert 'spread 5.5' in report and 'horizon 2 bars (30 minutes)' in report and 'SMA 4/8' in report
    assert '2024 is development/selection data' in report and 'not an independent final test' in report
    assert '<table id="ap3-training-history"' in report
    assert 'Best epoch 1' in report and 'early_stopping' in report and 'standardized target MSE' in report
    assert 'learning_rate' in report and '0.001' in report and '100' in report
    assert '<td>0.75</td>' in report and '<td>1.25</td>' in report


@pytest.mark.skipif(find_spec('torch') is None, reason='AP3 training extra not installed')
def test_registered_ridge_mlp_conditions_and_same_seed_replay(tmp_path):
    from decimal import Decimal
    from market_trader.experiment_pipeline import execute_experiment
    from market_trader.experiment_registry import register_run, compare_runs, load_config_snapshot, read_run
    config = v2_config(tmp_path)
    root = tmp_path / 'runs'
    def run(config, parent_id=None):
        return register_run(root, experiment_config_payload(config),
            executor=lambda path: execute_experiment(config, path), parent_id=parent_id)
    ridge = run(replace(config, model_kind='ridge'))
    mlp = run(config)
    comparison = compare_runs(root, ridge['run_id'], mlp['run_id'])
    assert comparison['compatible'] is True
    assert comparison['comparison_differences'] == []
    assert any(item['field'] == 'model.kind' for item in comparison['parameter_differences'])
    assert 'mlp' in mlp['summary']['validation_forecast']
    snapshot = load_config_snapshot(root, mlp['run_id'])
    replay = run(experiment_config_from_payload(snapshot), mlp['run_id'])
    assert replay['summary'] == mlp['summary']
    assert replay['parent_id'] == mlp['run_id']
    assert read_run(root, replay['run_id']) == replay
    first, second = (root / item['run_id'] / 'artifacts' for item in (mlp, replay))
    first_files, second_files = sorted(p.name for p in first.iterdir()), sorted(p.name for p in second.iterdir())
    assert first_files == second_files
    for name in first_files:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    costly = run(replace(config, spread=Decimal('5.5'), slippage=Decimal('0.25'), commission_rate=Decimal('.0001')))
    assert not compare_runs(root, mlp['run_id'], costly['run_id'])['compatible']
    assert costly['summary']['validation_forecast'] == mlp['summary']['validation_forecast']
    assert costly['summary']['trading']['mlp']['modeled_cost_points'] != mlp['summary']['trading']['mlp']['modeled_cost_points']
    html = (root / costly['run_id'] / 'artifacts' / 'report.html').read_text()
    assert '<table id="ap3-forecast"' in html and '<table id="ap3-training-history"' in html
    assert 'spread 5.5' in html and 'slippage per fill 0.25' in html and 'commission rate 0.0001' in html
    assert 'horizon 2 bars (30 minutes)' in html


