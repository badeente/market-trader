"""AP5 synthetic-only configurable core contract."""
from copy import deepcopy
from pathlib import Path
from decimal import Decimal
import builtins
import json
import pytest
from tests.test_neural import synthetic, requires_torch, seal, write, saved_body


def canonical():
    return {**_model_settings('mlp'), **config_body()['model']}


def v2_body():
    body = saved_body()
    body['format'] = 'market-trader.mlp-validation.v2'
    settings = canonical()
    body['training']['fit_settings'] = {**settings, 'seed': 7}
    body['training']['epoch_history'] = body['training']['epoch_history'][:3]
    body['training']['best_epoch'] = 3
    body['model']['architecture'] = settings['layers']
    body['model']['parameter_count'] = 81
    body['model']['layers'] = [dict(weights=[[0.1]*n for _ in range(m)], biases=[0.2]*m)
                               for n,m in zip(settings['layers'][:-1], settings['layers'][1:])]
    return seal(body)


def test_v2_reload_and_evaluation_without_dynamic_imports(tmp_path, monkeypatch):
    from market_trader.neural import load_mlp_model
    path = write(tmp_path / 'model.json', v2_body())
    def forbidden(*args, **kwargs):
        raise AssertionError('inference must not import')
    with monkeypatch.context() as scope:
        scope.setattr(builtins, '__import__', forbidden)
        model = load_mlp_model(path)
        assert model.model.predict((Decimal(1),)*4) == pytest.approx(2.9)


@requires_torch
def test_actual_custom_fit_deterministic_reload_replay(synthetic, tmp_path):
    from market_trader.neural import mlp_report, load_mlp_model, mlp_validation_evaluation
    path, _ = synthetic
    report = mlp_report(path, seed=7, settings=canonical())
    assert report == mlp_report(path, seed=7, settings=canonical())
    assert report['format'] == 'market-trader.mlp-validation.v2'
    assert report['model']['architecture'] == [4, 7, 5, 1]
    assert report['model']['parameter_count'] == 81
    assert report['training']['fit_settings'] == {**canonical(), 'seed': 7}
    assert len(report['training']['epoch_history']) == 3
    model_path = write(tmp_path / 'fit.json', report)
    assert load_mlp_model(model_path).report_id == report['report_id']
    evaluation, points = mlp_validation_evaluation(model_path, path)
    assert evaluation['validation']['mlp'] == report['validation']['mlp']
    assert len(points) == 19


@pytest.mark.parametrize('keys,value', [
    (('model','architecture'), [4,8,5,1]), (('model','parameter_count'), 80),
    (('training','fit_settings','threads'), True),
    (('training','fit_settings','max_epochs'), 4),
    (('training','fit_settings','layers'), [4,8,5,1]),
    (('training','fit_settings','learning_rate'), 'NaN'),
    (('training','best_epoch'), 2), (('training','stopping_reason'), 'early_stopping'),
])
def test_v2_resigned_malformed_metadata(tmp_path, keys, value):
    from market_trader.neural import load_mlp_model
    body = v2_body()
    target = body
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError):
        load_mlp_model(write(tmp_path / 'bad.json', seal(body)))

from market_trader.experiment_config import (
    _model_settings, _parse, experiment_config_payload, experiment_config_from_payload,
)


def config_body():
    return dict(version=3, name='synthetic', hypothesis='width test', seed=7,
                source_csv='/synthetic/never-read.csv', source_sha256='sha256:' + 'a' * 64,
                model=dict(kind='mlp', layers=[4, 7, 5, 1], learning_rate='0.002',
                           weight_decay='0.01', batch_size=16, max_epochs=3, patience=2, min_delta='0'))


@requires_torch
def test_v3_pipeline_custom_causal_backtest_report_replay(tmp_path):
    from dataclasses import replace
    from tests.test_experiment_pipeline import settings, read
    from market_trader.experiment_pipeline import execute_experiment
    config, _ = settings(tmp_path)
    config = replace(config, version=3, model_kind='mlp', model_settings=canonical(),
                     lookback_bars=6, horizon_bars=3, sma_fast_period=2, sma_slow_period=6)
    first, second = tmp_path / 'first', tmp_path / 'second'
    summary = execute_experiment(config, first)
    assert read(first, 'mlp-model.json')['format'] == 'market-trader.mlp-validation.v2'
    assert summary['training']['fit_settings'] == {**canonical(), 'seed': config.seed}
    assert read(first, 'training-data.json')['splits']['test'] == []
    assert summary['test_metrics_computed'] is False
    assert summary == execute_experiment(config, second)
    assert (first / 'manifest.json').read_bytes() == (second / 'manifest.json').read_bytes()
    assert 'MLP' in (first / 'report.html').read_text()


def test_v3_canonical_settings_and_variable_causal_config():
    body = config_body()
    body['training'] = dict(lookback_bars=10, horizon_bars=3)
    body['strategy'] = dict(sma_fast_period=4, sma_slow_period=10)
    config = _parse(body, Path('/'))
    payload = experiment_config_payload(config)
    assert payload['model'] == {**_model_settings('mlp'), **body['model']}
    assert experiment_config_payload(experiment_config_from_payload(payload)) == payload
    assert config.lookback_bars == 10 and config.horizon_bars == 3


@pytest.mark.parametrize('key,value', [
    ('layers', [4, True, 5, 1]), ('layers', [4, 3, 5, 1]), ('layers', [4, 65, 5, 1]),
    ('layers', [4, 7, 1]), ('learning_rate', True), ('learning_rate', 'NaN'),
    ('learning_rate', '0.000001'), ('learning_rate', '0.011'), ('weight_decay', '0.11'),
    ('batch_size', True), ('batch_size', 15), ('batch_size', 513), ('max_epochs', 101),
    ('patience', 21), ('min_delta', '0.011'), ('device', 'cuda'),
])
def test_v3_rejects_invalid_model_values(key, value):
    body = config_body()
    body['model'][key] = value
    with pytest.raises(ValueError):
        _parse(body, Path('/'))


def test_v3_seed_upper_bound_rejected_before_output():
    body = config_body()
    body['seed'] = 2**63
    with pytest.raises(ValueError, match='seed'):
        _parse(body, Path('/'))


def test_v3_snapshot_is_complete_strict_and_legacy_stays_frozen():
    payload = experiment_config_payload(_parse(config_body(), Path('/')))
    for key in payload['model']:
        bad = deepcopy(payload)
        del bad['model'][key]
        with pytest.raises(ValueError):
            experiment_config_from_payload(bad)
    for key, value in [('threads', True), ('deterministic', 1), ('learning_rate', 0.002)]:
        bad = deepcopy(payload)
        bad['model'][key] = value
        with pytest.raises(ValueError):
            experiment_config_from_payload(bad)
    old = config_body()
    old['version'] = 2
    with pytest.raises(ValueError):
        _parse(old, Path('/'))
