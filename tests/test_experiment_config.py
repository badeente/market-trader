"""Strict AP2 input and canonical immutable snapshots."""
import json
from importlib.util import find_spec

import pytest


def config_file(tmp_path, extra=""):
    path = tmp_path / "experiment.toml"
    path.write_text('version=1\nname="trial"\nhypothesis="causal baseline"\nseed=0\n'
                    'source_csv="source.csv"\nsource_sha256="sha256:' + 'a'*64 + '"\n' + extra)
    return path


def test_minimal_config_roundtrip(tmp_path):
    assert find_spec('market_trader.experiment_config'), 'AP2 config reader missing'
    from market_trader.experiment_config import (
        read_experiment_config, experiment_config_payload, experiment_config_from_payload,
    )
    config = read_experiment_config(config_file(tmp_path))
    payload = experiment_config_payload(config)
    assert payload['source_csv'] == str(tmp_path / 'source.csv')
    assert payload['training'] == {'lookback_bars': 8, 'horizon_bars': 2}
    assert payload['execution']['spread'] == '2.0'
    assert payload['version'] == 1 and payload['seed'] == 0
    assert experiment_config_from_payload(json.loads(json.dumps(payload))) == config
    assert experiment_config_payload(experiment_config_from_payload(payload)) == payload
    with pytest.raises(Exception):
        config.name = 'changed'


@pytest.mark.parametrize('old,new', [
    ('version=1', 'version=true'), ('version=1', 'version=2'),
    ('seed=0', 'seed=true'), ('seed=0', 'seed=-1'), ('seed=0', 'seed=0.5'),
    ('name="trial"', 'name=""'), ('hypothesis="causal baseline"', 'hypothesis=" padded"'),
    ('source_csv="source.csv"', 'source_csv=true'), ('source.csv', ' source.csv'),
    ('sha256:', 'SHA256:'),
])
def test_invalid_root_values(tmp_path, old, new):
    from market_trader.experiment_config import read_experiment_config
    path = config_file(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        read_experiment_config(path)


@pytest.mark.parametrize('extra', [
    'test_start="2024-01-01"', 'model_kind="forest"', 'quantity=2',
    '[model]\nkind="forest"', '[splits]\ntest_start="2024-01-01"',
    '[training]\nlookback_bars=true', '[training]\nlookback_bars=1',
    '[training]\nhorizon_bars=0', '[training]\nhorizon_bars=2.0',
    '[training]\nhorizon_bars=26', '[training]\nunknown=1',
    '[strategy]\nsma_fast_period=8', '[strategy]\nsma_slow_period=9',
    '[strategy]\nsma_fast_period=true', '[strategy]\nentry_threshold=-0.1',
    '[strategy]\nentry_threshold=nan', '[strategy]\nentry_threshold=true',
    '[execution]\nspread=inf', '[execution]\nslippage=-1',
    '[execution]\ncommission_rate=1', '[execution]\ncommission_rate=true',
    '[execution]\nprice_tick=0', '[execution]\ninitial_capital=0',
    '[execution]\nspread="bad"', '[execution]\nallow_short=true',
    '[execution]\nliquidate_at_end=true',
])
def test_invalid_nested_and_reserved_values(tmp_path, extra):
    from market_trader.experiment_config import read_experiment_config
    with pytest.raises(ValueError):
        read_experiment_config(config_file(tmp_path, extra))


def test_variant_config_and_snapshot_strictness(tmp_path):
    from market_trader.experiment_config import (
        read_experiment_config, experiment_config_payload, experiment_config_from_payload,
    )
    config = read_experiment_config(config_file(tmp_path,
        '[training]\nlookback_bars=6\nhorizon_bars=3\n'
        '[strategy]\nsma_fast_period=2\nsma_slow_period=6\nentry_threshold=0.001\n'
        '[execution]\nspread=4\nslippage="0.25"\ncommission_rate=0.0001\n'
        'price_tick="0.01"\ninitial_capital=120000\n'))
    payload = experiment_config_payload(config)
    assert payload['training'] == {'lookback_bars': 6, 'horizon_bars': 3}
    assert payload['execution']['commission_rate'] == '0.0001'
    assert experiment_config_payload(experiment_config_from_payload(payload)) == payload
    for change in ({'source_csv': 'relative.csv'}, {'training': {}}, {'execution': {}},
                   {'version': True}, {'model_kind': 'forest'}):
        with pytest.raises(ValueError):
            experiment_config_from_payload({**payload, **change})
