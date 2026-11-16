"""AP5 derivation contracts; synthetic parent data is explicitly not market evidence."""
from pathlib import Path
import json
import pytest


def parent(tmp_path):
    from market_trader.experiment_config import ExperimentConfig, experiment_config_payload
    from market_trader.experiment_registry import register_run
    from tests.test_experiment_registry import synthetic_executor
    payload = experiment_config_payload(ExperimentConfig(2, 'parent', 'fixture', 0,
        tmp_path/'missing.csv', 'sha256:'+'0'*64, model_kind='mlp'))
    return register_run(tmp_path/'runs', payload, executor=synthetic_executor)


def test_derive_seed_retains_parent_and_records_changes(tmp_path):
    from market_trader import experiment_variants
    record = parent(tmp_path)
    root = tmp_path/'runs'
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = experiment_variants.derive_variant(root, record['run_id'], {'seed': 1},
        name='one', hypothesis='A seed is a controlled training variant, not an optimum')
    assert result['format'] == 'market-trader.experiment-variant.v1'
    assert result['parent_id'] == record['run_id']
    assert result['parent_config_sha256'] == record['config_sha256']
    assert result['config']['seed'] == 1
    assert result['config']['version'] == 2
    assert result['changes'] == [{'field': 'seed', 'left': 0, 'right': 1, 'category': 'training'}]
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('changes', [{}, {'seed': 0}, {'seed': True}, {'source_csv': '/other.csv'}, {'model.device': 'cuda'}])
def test_invalid_or_empty_changes_never_write(tmp_path, changes):
    from market_trader.experiment_variants import derive_variant
    record = parent(tmp_path)
    with pytest.raises(ValueError):
        derive_variant(tmp_path/'runs', record['run_id'], changes, name='x', hypothesis='y')


def test_trading_and_condition_changes_are_distinct(tmp_path):
    from market_trader.experiment_variants import derive_variant
    record = parent(tmp_path)
    result = derive_variant(tmp_path/'runs', record['run_id'],
        {'strategy.entry_threshold': 0.0005, 'execution.spread': '3.0'}, name='x', hypothesis='y')
    assert result['config']['strategy']['entry_threshold'] == '0.0005'
    assert [x['category'] for x in result['changes']] == ['conditions', 'trading_rule']


def test_adjustable_neural_parameters_explicitly_promote_legacy_config(tmp_path):
    from market_trader.experiment_variants import derive_variant
    record = parent(tmp_path)
    result = derive_variant(tmp_path/'runs', record['run_id'],
        {'model.learning_rate': 0.0005, 'model.layers': [4,8,4,1], 'model.max_epochs': 5},
        name='small', hypothesis='bounded architecture and optimizer variation')
    assert result['config']['version'] == 3
    assert result['config']['model']['learning_rate'] == '0.0005'
    assert result['config']['model']['layers'] == [4,8,4,1]
    assert result['config']['model']['max_epochs'] == 5
    assert {x['category'] for x in result['changes']} == {'model', 'training'}


def test_causal_window_variation_is_a_condition_change_not_a_model_tweak(tmp_path):
    from market_trader.experiment_variants import derive_variant
    record = parent(tmp_path)
    result = derive_variant(tmp_path/'runs', record['run_id'],
        {'training.lookback_bars': 6, 'strategy.sma_slow_period': 6, 'training.horizon_bars': 3},
        name='window', hypothesis='different information window, not a fair pure model comparison')
    assert result['config']['version'] == 3
    assert {x['category'] for x in result['changes']} == {'conditions'}


def test_sweep_is_explicit_bounded_and_validates_every_variant(tmp_path):
    from market_trader.experiment_variants import prepare_sweep
    record = parent(tmp_path)
    plan = prepare_sweep(tmp_path/'runs', record['run_id'], {'seed': [1, 2]},
        name='seeds', hypothesis='Two seeds are not a statistical robustness study')
    assert [x['config']['seed'] for x in plan['variants']] == [1, 2]
    assert plan['objective'] == {'split': 'validation_2024', 'metric': 'rmse', 'direction': 'minimize'}
    for grid in ({'seed': [1,2,3,4,5]}, {'seed': [1,1]}, {}, {'seed': [1,True]}):
        with pytest.raises(ValueError):
            prepare_sweep(tmp_path/'runs', record['run_id'], grid, name='x', hypothesis='y')


def test_saved_plan_is_revalidated_against_its_parent(tmp_path):
    from market_trader.experiment_variants import derive_variant, read_plan
    record = parent(tmp_path)
    result = derive_variant(tmp_path/'runs', record['run_id'], {'seed': 1}, name='x', hypothesis='y')
    path = tmp_path/'variant.json'
    path.write_text(json.dumps(result))
    assert read_plan(path, tmp_path/'runs') == result
    result['config']['seed'] = 2
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        read_plan(path, tmp_path/'runs')


def test_plan_rejects_boolean_numeric_aliases_in_provenance(tmp_path):
    from market_trader.experiment_variants import derive_variant, read_plan
    record = parent(tmp_path)
    result = derive_variant(tmp_path/'runs', record['run_id'], {'seed': 1}, name='x', hypothesis='y')
    result['changes'][0]['left'] = False
    path = tmp_path/'bool-alias.json'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        read_plan(path, tmp_path/'runs')


def test_search_selection_uses_validation_rmse_not_trading_profit(tmp_path):
    from market_trader.experiment_variants import summarize_runs
    from market_trader.experiment_registry import register_run, _encode, _digest
    from tests.test_experiment_registry import synthetic_executor
    from market_trader.experiment_config import ExperimentConfig, experiment_config_payload
    payload = experiment_config_payload(ExperimentConfig(2,'fixture','not market data',0,
        tmp_path/'unused.csv','sha256:'+'0'*64,model_kind='mlp'))
    def execute(rmse, profit):
        def runner(output):
            summary = synthetic_executor(output)
            summary['validation_forecast'] = {'mlp': {'mae': rmse, 'rmse': rmse}}
            summary['trading'] = {'mlp': {'net_points': str(profit)}}
            (output/'summary.json').write_bytes(_encode(summary))
            files = {p.name:_digest(p.read_bytes()) for p in output.iterdir() if p.name != 'manifest.json'}
            (output/'manifest.json').write_bytes(_encode({'status':'complete','artifacts':files}))
            return summary
        return runner
    left = register_run(tmp_path/'runs',payload,executor=execute(.02,100))
    right = register_run(tmp_path/'runs',payload,executor=execute(.01,-100))
    result = summarize_runs(tmp_path/'runs',[left['run_id'],right['run_id']])
    assert result['compatible'] is True
    assert result['best_run_id'] == right['run_id']
    assert result['objective']['metric'] == 'rmse'
    assert result['test_metrics_computed'] is False
