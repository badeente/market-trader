"""Strict AP2 v1, frozen AP3 v2 and configurable AP5 v3 configuration.

Root TOML keys: version, name, hypothesis, seed, source_csv, source_sha256.
Optional tables: training (lookback_bars, horizon_bars), strategy
(entry_threshold, sma_fast_period, sma_slow_period), execution (spread,
slippage, commission_rate, price_tick, initial_capital). Version 2 requires
model (kind only: ridge or mlp); snapshots record its complete fixed settings.
Version 1 still rejects model. Numeric execution/threshold inputs may be TOML
numbers or decimal strings; snapshots encode them as exact decimal strings.
Dates and size are fixed; MLP also fixes eight-bar/two-bar features/target and
SMA 4/8 to the approved first reference in v2. Explicit v3 supports bounded
model/training controls and causal lookback/horizon; snapshots remain complete.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    version: int
    name: str
    hypothesis: str
    seed: int
    source_csv: Path
    source_sha256: str
    lookback_bars: int = 8
    horizon_bars: int = 2
    entry_threshold: Decimal = Decimal('0')
    sma_fast_period: int = 4
    sma_slow_period: int = 8
    spread: Decimal = Decimal('2.0')
    slippage: Decimal = Decimal('0.1')
    commission_rate: Decimal = Decimal('0')
    price_tick: Decimal = Decimal('0.001')
    initial_capital: Decimal = Decimal('100000')
    model_kind: str = 'ridge'
    model_settings: dict | None = None


def _model_settings(kind):
    """Frozen AP3 policy, not a user hyperparameter surface."""
    if kind == 'ridge':
        return {'kind': 'ridge', 'alpha': '0.1'}
    return {'kind': 'mlp', 'layers': [4, 16, 8, 1],
            'hidden_activation': 'relu', 'output_activation': 'linear',
            'optimizer': 'AdamW', 'learning_rate': '0.001', 'weight_decay': '0.0001',
            'batch_size': 128, 'max_epochs': 100, 'patience': 10, 'min_delta': '0.000001',
            'loss': 'standardized_target_mse', 'input_scaler': 'train_only_standard',
            'target_scaler': 'train_only_standard', 'device': 'cpu', 'dtype': 'float64',
            'threads': 1, 'deterministic': True}


def canonical_model_settings(value, *, snapshot=False):
    """Validate AP5 settings and return the complete canonical model policy."""
    if not isinstance(value, dict) or value.get('kind') not in ('mlp', 'ridge'):
        raise ValueError('model.kind must be ridge or mlp')
    expected = _model_settings(value['kind'])
    adjustable = {'layers', 'learning_rate', 'weight_decay', 'batch_size',
                  'max_epochs', 'patience', 'min_delta'} if value['kind'] == 'mlp' else set()
    _table(value, 'model', expected if snapshot else adjustable | {'kind'})
    result = {**expected, **value}
    if value['kind'] == 'mlp':
        layers = result['layers']
        if (type(layers) is not list or len(layers) != 4
                or any(type(x) is not int for x in layers)
                or layers[0] != 4 or layers[-1] != 1
                or any(not 4 <= x <= 64 for x in layers[1:3])):
            raise ValueError('model.layers must be [4, H1, H2, 1], widths 4–64')
        for key, low, high in [('batch_size', 16, 512), ('max_epochs', 1, 100), ('patience', 1, 20)]:
            number = _integer(result[key], key, low)
            if number > high:
                raise ValueError(f'{key} must not exceed {high}')
        for key, low, high in [('learning_rate', '0.00001', '0.01'),
                               ('weight_decay', '0', '0.1'), ('min_delta', '0', '0.01')]:
            number = _decimal(result[key], key)
            if not Decimal(low) <= number <= Decimal(high):
                raise ValueError(f'{key} out of range')
            result[key] = str(number)
    if snapshot and (value.keys() != expected.keys()
                     or any(type(value[k]) is not type(result[k]) for k in result)
                     or value != result):
        raise ValueError('snapshot must contain the complete canonical model settings')
    for key in expected.keys() - adjustable:
        if type(result[key]) is not type(expected[key]) or result[key] != expected[key]:
            raise ValueError(f'unsupported fixed model setting: {key}')
    return result


def _table(value, label, allowed):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a table')
    unknown = value.keys() - set(allowed)
    if unknown:
        raise ValueError(f'unexpected {label} fields: {sorted(unknown)}')
    return value


def _integer(value, label, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _decimal(value, label, *, positive=False):
    if type(value) not in (str, int, float, Decimal):
        raise ValueError(f'{label} must be a finite number')
    if isinstance(value, str) and (not value or value != value.strip()):
        raise ValueError(f'{label} must be an unpadded decimal')
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f'{label} must be a finite number') from None
    if not result.is_finite() or result < 0 or (positive and not result):
        raise ValueError(f'{label} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _parse(body, base, *, snapshot=False):
    if not isinstance(body, dict):
        raise ValueError('experiment must be a table')
    version = body.get('version')
    if type(version) is not int or version not in (1, 2, 3, 4):
        raise ValueError('experiment version must be the integer 1, 2, 3 or 4')
    _table(body, 'experiment', ('version', 'name', 'hypothesis', 'seed', 'source_csv',
                               'source_sha256', 'training', 'strategy', 'execution',
                               *(('model',) if version >= 2 else ())))
    required = {'version', 'name', 'hypothesis', 'seed', 'source_csv', 'source_sha256'}
    if required - body.keys():
        raise ValueError(f'missing experiment fields: {sorted(required - body.keys())}')
    kind = 'ridge'
    model = None
    if version == 2:
        model = _table(body.get('model'), 'model',
                       _model_settings('mlp').keys() | _model_settings('ridge').keys() if snapshot else ('kind',))
        kind = model.get('kind')
        if kind not in ('ridge', 'mlp'):
            raise ValueError('model.kind must be ridge or mlp')
        expected = _model_settings(kind)
        if snapshot and (model != expected
                or any(type(model[key]) is not type(value) for key, value in expected.items())
                or (kind == 'mlp' and any(type(value) is not int for value in model['layers']))):
            raise ValueError('snapshot must contain the complete fixed model settings')
    if version == 3:
        model = canonical_model_settings(body.get('model'), snapshot=snapshot)
        kind = model['kind']
    if version == 4:
        from market_trader.cnn_settings import canonical_cnn_settings
        if not isinstance(body.get('model'), dict) or body['model'].get('kind') != 'cnn':
            raise ValueError('AP7 v4 requires explicit model.kind cnn')
        model = canonical_cnn_settings(body['model'], snapshot=snapshot)
        kind = 'cnn'
    for key in ('name', 'hypothesis', 'source_csv'):
        value = body[key]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f'{key} must be nonempty text without padding')
    checksum = body['source_sha256']
    if not isinstance(checksum, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', checksum):
        raise ValueError('source_sha256 must be an exact lowercase sha256 identifier')
    seed = _integer(body['seed'], 'seed', 0)
    if version == 4 and seed != 0:
        raise ValueError('AP7 reference CNN requires seed 0')
    if version in (3, 4) and seed >= 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    training = _table(body.get('training', {}), 'training', ('lookback_bars', 'horizon_bars'))
    strategy = _table(body.get('strategy', {}), 'strategy',
                      ('entry_threshold', 'sma_fast_period', 'sma_slow_period'))
    execution = _table(body.get('execution', {}), 'execution',
                       ('spread', 'slippage', 'commission_rate', 'price_tick', 'initial_capital'))
    lookback = _integer(training.get('lookback_bars', 8), 'lookback_bars', 2)
    horizon = _integer(training.get('horizon_bars', 2), 'horizon_bars', 1)
    fast = _integer(strategy.get('sma_fast_period', 4), 'sma_fast_period', 1)
    slow = _integer(strategy.get('sma_slow_period', 8), 'sma_slow_period', 2)
    if lookback + horizon + 1 > 34:
        raise ValueError('lookback + horizon + 1 must not exceed 34 session bars')
    if not fast < slow == lookback:
        raise ValueError('SMA periods must satisfy fast < slow == lookback')
    defaults = {'spread': '2.0', 'slippage': '0.1', 'commission_rate': '0',
                'price_tick': '0.001', 'initial_capital': '100000'}
    costs = {key: _decimal(execution.get(key, value), key,
                          positive=key in ('price_tick', 'initial_capital'))
             for key, value in defaults.items()}
    if costs['commission_rate'] >= 1:
        raise ValueError('commission_rate must be less than 1')
    if version == 2 and kind == 'mlp' and (lookback, horizon, fast, slow) != (8, 2, 4, 8):
        raise ValueError('AP3 mlp requires lookback 8, horizon 2 and SMA 4/8')
    if version == 4 and (lookback, horizon, fast, slow) != (8, 2, 4, 8):
        raise ValueError('AP7 cnn requires lookback 8, horizon 2 and SMA 4/8')
    return ExperimentConfig(version, body['name'], body['hypothesis'], seed,
                            (base / body['source_csv']).resolve(), checksum,
                            lookback, horizon, _decimal(strategy.get('entry_threshold', '0'), 'entry_threshold'),
                            fast, slow, **costs, model_kind=kind,
                            model_settings=model if version in (3, 4) else None)


def read_experiment_config(path: Path) -> ExperimentConfig:
    """Read TOML, resolving its relative CSV path against the TOML directory."""
    path = Path(path).resolve()
    with path.open('rb') as stream:
        return _parse(tomllib.load(stream), path.parent)


def experiment_config_payload(config: ExperimentConfig) -> dict:
    """Canonical JSON-friendly configuration, including every configurable default."""
    if config.model_kind not in (('cnn',) if config.version == 4 else ('ridge', 'mlp')) or (config.version == 1 and config.model_kind != 'ridge'):
        raise ValueError('model kind is incompatible with experiment version')
    payload = {'version': config.version, 'name': config.name, 'hypothesis': config.hypothesis,
            'seed': config.seed, 'source_csv': str(config.source_csv), 'source_sha256': config.source_sha256,
            'training': {'lookback_bars': config.lookback_bars, 'horizon_bars': config.horizon_bars},
            'strategy': {'entry_threshold': str(config.entry_threshold),
                         'sma_fast_period': config.sma_fast_period, 'sma_slow_period': config.sma_slow_period},
            'execution': {key: str(getattr(config, key)) for key in
                          ('spread', 'slippage', 'commission_rate', 'price_tick', 'initial_capital')}}
    if config.version >= 2:
        if config.version == 4:
            from market_trader.cnn_settings import canonical_cnn_settings
            payload['model'] = canonical_cnn_settings(
                config.model_settings if config.model_settings is not None else canonical_cnn_settings(), snapshot=True)
        else:
            payload['model'] = (canonical_model_settings(config.model_settings if config.model_settings is not None else _model_settings(config.model_kind), snapshot=True)
                                if config.version == 3 else _model_settings(config.model_kind))
        if payload['model']['kind'] != config.model_kind:
            raise ValueError('model settings kind mismatch')
    return payload


def experiment_config_from_payload(body: dict) -> ExperimentConfig:
    """Load a canonical snapshot without silently resolving relative paths/defaults."""
    if not isinstance(body, dict) or not isinstance(body.get('source_csv'), str) or not Path(body['source_csv']).is_absolute():
        raise ValueError('snapshot source_csv must be absolute')
    config = _parse(body, Path('/'), snapshot=True)
    if experiment_config_payload(config) != body:
        raise ValueError('snapshot must contain the complete canonical configuration')
    return config
