"""Strict versioned AP2 configuration; AP1's policy is not changed.

Root TOML keys: version, name, hypothesis, seed, source_csv, source_sha256.
Optional tables: training (lookback_bars, horizon_bars), strategy
(entry_threshold, sma_fast_period, sma_slow_period), execution (spread,
slippage, commission_rate, price_tick, initial_capital). All other keys fail.
Numeric execution/threshold inputs may be TOML numbers or decimal strings;
snapshots encode them as exact decimal strings. Dates, model and size are fixed.
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


def _parse(body, base):
    _table(body, 'experiment', ('version', 'name', 'hypothesis', 'seed', 'source_csv',
                               'source_sha256', 'training', 'strategy', 'execution'))
    required = {'version', 'name', 'hypothesis', 'seed', 'source_csv', 'source_sha256'}
    if required - body.keys():
        raise ValueError(f'missing experiment fields: {sorted(required - body.keys())}')
    if type(body['version']) is not int or body['version'] != 1:
        raise ValueError('experiment version must be the integer 1')
    for key in ('name', 'hypothesis', 'source_csv'):
        value = body[key]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f'{key} must be nonempty text without padding')
    checksum = body['source_sha256']
    if not isinstance(checksum, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', checksum):
        raise ValueError('source_sha256 must be an exact lowercase sha256 identifier')
    seed = _integer(body['seed'], 'seed', 0)
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
    return ExperimentConfig(1, body['name'], body['hypothesis'], seed,
                            (base / body['source_csv']).resolve(), checksum,
                            lookback, horizon, _decimal(strategy.get('entry_threshold', '0'), 'entry_threshold'),
                            fast, slow, **costs)


def read_experiment_config(path: Path) -> ExperimentConfig:
    """Read TOML, resolving its relative CSV path against the TOML directory."""
    path = Path(path).resolve()
    with path.open('rb') as stream:
        return _parse(tomllib.load(stream), path.parent)


def experiment_config_payload(config: ExperimentConfig) -> dict:
    """Canonical JSON-friendly configuration, including every configurable default."""
    return {'version': config.version, 'name': config.name, 'hypothesis': config.hypothesis,
            'seed': config.seed, 'source_csv': str(config.source_csv), 'source_sha256': config.source_sha256,
            'training': {'lookback_bars': config.lookback_bars, 'horizon_bars': config.horizon_bars},
            'strategy': {'entry_threshold': str(config.entry_threshold),
                         'sma_fast_period': config.sma_fast_period, 'sma_slow_period': config.sma_slow_period},
            'execution': {key: str(getattr(config, key)) for key in
                          ('spread', 'slippage', 'commission_rate', 'price_tick', 'initial_capital')}}


def experiment_config_from_payload(body: dict) -> ExperimentConfig:
    """Load a canonical snapshot without silently resolving relative paths/defaults."""
    if not isinstance(body, dict) or not isinstance(body.get('source_csv'), str) or not Path(body['source_csv']).is_absolute():
        raise ValueError('snapshot source_csv must be absolute')
    config = _parse(body, Path('/'))
    if experiment_config_payload(config) != body:
        raise ValueError('snapshot must contain the complete canonical configuration')
    return config
