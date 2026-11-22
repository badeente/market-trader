"""Frozen AP7 settings: no tuning surface, including deep type aliases."""
from copy import deepcopy
import pytest


def test_frozen_settings_are_fresh_and_kind_only():
    from market_trader.cnn_settings import canonical_cnn_settings
    settings = canonical_cnn_settings()
    assert settings == canonical_cnn_settings({'kind': 'cnn'})
    assert settings['parameter_count'] == 5409
    assert settings['input_shape'] == [3, 24, 32]
    assert settings['architecture'] == 'tiny_cnn_v1'
    assert settings['max_epochs'] == 20
    assert canonical_cnn_settings(settings, snapshot=True) == settings
    settings['input_shape'][0] = 99
    assert canonical_cnn_settings()['input_shape'][0] == 3
    with pytest.raises(ValueError):
        canonical_cnn_settings(canonical_cnn_settings())


@pytest.mark.parametrize('key,value', [('threads', True), ('max_epochs', 20.0),
    ('input_shape', [3.0,24,32]), ('betas', [0.9,0.999]), ('fused', 0),
    ('learning_rate', 0.001), ('parameter_count', 417), ('unknown', 1)])
def test_snapshot_rejects_deep_type_or_value_changes(key, value):
    from market_trader.cnn_settings import canonical_cnn_settings
    data = canonical_cnn_settings()
    data[key] = value
    with pytest.raises(ValueError):
        canonical_cnn_settings(data, snapshot=True)


def test_complete_snapshot_required():
    from market_trader.cnn_settings import canonical_cnn_settings
    with pytest.raises(ValueError):
        canonical_cnn_settings({'kind': 'cnn'}, snapshot=True)
