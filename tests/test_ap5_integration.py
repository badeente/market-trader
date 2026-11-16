"""Integration contracts for AP5 model dispatch and honest report labels."""
import json
from hashlib import sha256
from market_trader._dataset_codec import json_bytes
from tests.test_neural_strategy import mlp_report


def test_configurable_model_format_reaches_existing_strategy_loader(mlp_report):
    from market_trader.strategies import load_frozen_model
    path, body = mlp_report
    body = {k: v for k,v in body.items() if k != 'report_id'}
    body['format'] = 'market-trader.mlp-validation.v2'
    from market_trader.experiment_config import _model_settings
    body['training']['fit_settings'] = {**_model_settings('mlp'), 'seed': body['training']['fit_settings']['seed']}
    body['report_id'] = 'sha256:' + sha256(json_bytes(body)).hexdigest()
    path.write_bytes(json_bytes(body))
    saved = load_frozen_model(path, expected_kind='mlp')
    assert saved.report_id == body['report_id']


def test_v3_experiment_report_shows_actual_training_settings():
    from market_trader.experiment_reporting import enrich_experiment_report
    config = {'version': 3, 'name': 'synthetic AP5', 'hypothesis': 'layout only', 'seed': 0,
              'model': {'kind': 'mlp', 'layers': [4,8,4,1], 'learning_rate': '0.0005'},
              'training': {'lookback_bars': 8, 'horizon_bars': 2},
              'strategy': {'sma_fast_period': 4, 'sma_slow_period': 8, 'entry_threshold': '0'},
              'execution': {'spread': '2', 'slippage': '.1', 'commission_rate': '0',
                            'price_tick': '.001', 'initial_capital': '100000'}}
    metrics = {'mae': 0.0, 'rmse': 0.0}
    trade = dict(trade_count=0, net_points='0', reference_points='0', modeled_cost_points='0', max_drawdown_points='0')
    summary = {'config': config, 'comparison_signature': {'interval_microseconds': 900000000},
               'validation_forecast': {'row_count': 1, 'first_decision_at': '2024-01-02',
                    'last_decision_at': '2024-01-02', 'mlp': metrics, 'ridge': metrics, 'zero': metrics},
               'trading': {'mlp': trade, 'ridge': trade, 'sma_timed': trade},
               'training': {'best_epoch': 1, 'stopping_reason': 'max_epochs', 'fit_settings': {'learning_rate': 0.0005},
                    'epoch_history': [{'epoch': 1, 'training_mse': 0.0, 'validation_mse': 0.0}]},
               'training_data_id': 'synthetic', 'model_report_id': 'synthetic',
               'validation_dataset_id': 'synthetic', 'runtime': {}}
    html = enrich_experiment_report(b'<html><head></head><body><main>ledger</main></body></html>',summary=summary).decode()
    assert 'AP5 — synthetic AP5' in html
    assert 'ap3-forecast' in html and '0.0005' in html
    architecture = config['model']['layers']
    count = sum((inputs+1)*outputs for inputs, outputs in zip(architecture[:-1], architecture[1:], strict=True))
    assert 'MLP 4 → 8 → 4 → 1' in html
    assert f'{count} trainable parameters' in html
    assert 'MLP 4 → 16 → 8 → 1; 225 trainable parameters' not in html


def test_v3_sma_condition_change_prevents_best_run_selection(tmp_path):
    from dataclasses import replace
    from market_trader.experiment_registry import register_run
    from market_trader.experiment_config import experiment_config_payload
    from market_trader.experiment_pipeline import execute_experiment
    from market_trader.experiment_variants import summarize_runs
    from tests.test_experiment_pipeline import settings
    base, _ = settings(tmp_path)
    configs = [replace(base, version=3, sma_fast_period=fast) for fast in (4,2)]
    runs = [register_run(tmp_path/'runs',experiment_config_payload(config),
                        executor=lambda output, config=config:execute_experiment(config,output)) for config in configs]
    result = summarize_runs(tmp_path/'runs',[run['run_id'] for run in runs])
    assert result['compatible'] is False
    assert result['best_run_id'] is None
    assert any('sma_fast_period' in diff['field'] for pair in result['condition_differences'] for diff in pair['differences'])
