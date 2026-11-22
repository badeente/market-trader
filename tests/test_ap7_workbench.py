"""AP7 read-only workbench projections over verified synthetic registries."""
import json
import pytest
from pathlib import Path

from market_trader.experiment_config import ExperimentConfig, experiment_config_payload
from tests.test_workbench_data import make_run, api, encode


def test_v4_cnn_projection_preserves_signature_and_real_training_history(tmp_path):
    config = experiment_config_payload(ExperimentConfig(version=4, name='synthetic CNN',
        hypothesis='no market fit', seed=0, source_csv=Path('/private/synthetic.csv'),
        source_sha256='sha256:' + 'a'*64, model_kind='cnn'))
    history = dict(best_epoch=2, stopping_reason='maximum epochs synthetic', epoch_history=[
        dict(epoch=1, train_mse=1.1, validation_mse=1.2),
        dict(epoch=2, train_mse=.8, validation_mse=.9)])
    signature = dict(policy='ap7-v4', common_cohort='synthetic index', representations=['image', 'tabular'])
    _, key = make_run(tmp_path, config_changes=config, signature=signature,
        summary_changes=dict(validation_forecast=dict(cnn=dict(mae=.0002, rmse=.0003),
            ridge=dict(mae=.0003, rmse=.0004), zero_return=dict(mae=.0005, rmse=.0006)),
            trading=dict(cnn=dict(net_points='0', modeled_cost_points='0', max_drawdown_points='0', trade_count=0)),
            training=history), extra={'training-history.json': encode(history),
                'cnn-backtest.json': encode(dict(execution=dict(initial_capital='100000'),
                    accounts=[dict(at=None, equity='100000')]))})
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['error'] is None
    assert detail['run']['model_kind'] == 'cnn'
    assert detail['signature'] == signature
    assert detail['metrics_by_model']['cnn']['mae_bp'] == '2.0000'
    assert detail['trading_by_model']['cnn']['trade_count'] == 0
    assert detail['training']['history'][1] == dict(epoch=2, training_mse=.8, validation_mse=.9)
    assert detail['training']['best_epoch'] == 2
    warnings = ' '.join(detail['warnings'])
    assert '2024' in warnings and 'selection' in warnings and 'representation' in warnings.lower()
    assert 'wick/detail' in warnings and '2025' in warnings


@pytest.mark.parametrize('damage', ['malformed_epoch', 'wrong_minimum', 'summary_mismatch'])
def test_cnn_workbench_refuses_nonactual_or_malformed_training_history(tmp_path, damage):
    config = experiment_config_payload(ExperimentConfig(version=4, name='synthetic',
        hypothesis='no fit', seed=0, source_csv=Path('/private/synthetic.csv'),
        source_sha256='sha256:' + 'a'*64, model_kind='cnn'))
    history = dict(best_epoch=2, stopping_reason='synthetic', epoch_history=[
        dict(epoch=1, training_mse=1.1, validation_mse=1.2),
        dict(epoch=2, training_mse=.8, validation_mse=.9)])
    if damage == 'malformed_epoch':
        history['epoch_history'] = ['not an epoch']
    if damage == 'wrong_minimum':
        history['best_epoch'] = 1
    summary_history = {**history}
    if damage == 'summary_mismatch':
        summary_history['stopping_reason'] = 'different actual history'
    _, key = make_run(tmp_path, config_changes=config,
        summary_changes=dict(validation_forecast=dict(cnn=dict(mae=.1, rmse=.2)),
            trading=dict(cnn=dict(net_points='0', modeled_cost_points='0', max_drawdown_points='0', trade_count=0)),
            training=summary_history), extra={'training-history.json': encode(history),
                'cnn-backtest.json': encode(dict(execution=dict(initial_capital='100000'),
                    accounts=[dict(at=None, equity='100000')]))})
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert detail['training'] is None
