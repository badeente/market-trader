"""Synthetic filesystem projections: no fitting, Torch, or real runs."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from market_trader import experiment_registry as registry
from market_trader.experiment_config import ExperimentConfig, experiment_config_payload


def encode(value):
    return registry._encode(value)


def make_run(root, *, kind='ridge', config_changes=None, signature=None, accounts=None,
             report=b'<html>checked report</html>', extra=None, summary_changes=None):
    config = experiment_config_payload(ExperimentConfig(
        version=1 if kind == 'ridge' else 2, name='<script>alert(1)</script>',
        hypothesis='synthetic only', seed=7, source_csv=Path('/private/input.csv'),
        source_sha256='sha256:' + 'a' * 64, model_kind=kind))
    config.update(config_changes or {})
    times = [None, '2024-01-02T08:00:00+00:00', '2024-01-02T08:15:00+00:00',
             '2024-01-02T08:30:00+00:00']
    accounts = accounts or [{'at': at, 'equity': str(value)} for at, value in
                            zip(times, [100000, 100010, 99995, 100003])]
    def execute(output):
        output.mkdir()
        summary = {
            'runtime': {'code_id': 'fixture-code', 'python': '3.11', 'dependencies': {}},
            'training_rows': 3, 'validation_rows': 2, 'test_metrics_computed': False,
            'model_report_id': 'fixture-model', 'training_data_id': 'fixture-training',
            'validation_dataset_id': 'fixture-validation',
            'comparison_signature': signature or {'source_sha256': 'fixture', 'execution': {'spread': '2.0'}},
            'validation_forecast': {kind: {'mae': 0.0002, 'rmse': 0.0003},
                                    'zero_return': {'mae': 0.0005, 'rmse': 0.0006}},
            'trading': {kind: {'net_points': '3', 'modeled_cost_points': '2.2',
                              'max_drawdown_points': '15', 'trade_count': 2}},
        }
        summary.update(summary_changes or {})
        files = {'summary.json': encode(summary), 'report.html': report,
                 f'{kind}-backtest.json': encode({'execution': {'initial_capital': '100000'},
                                                  'accounts': accounts}),
                 'raw-dataset.json': b'PRIVATE RAW INPUT', 'weights.bin': b'PRIVATE WEIGHTS'}
        if kind == 'mlp':
            files['training-history.json'] = encode({'epoch_history': [
                {'epoch': 1, 'training_mse': 1.1, 'validation_mse': 1.2},
                {'epoch': 2, 'training_mse': 0.8, 'validation_mse': 0.9}],
                'best_epoch': 2, 'stopping_reason': 'patience'})
        files.update(extra or {})
        for name, data in files.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (output / 'manifest.json').write_bytes(encode({'status': 'complete', 'artifacts': {
            name: registry._digest(data) for name, data in files.items()}}))
        return summary
    record = registry.register_run(root, config, executor=execute)
    return record, 'ap2:' + record['run_id']


def api(roots, **kwargs):
    from market_trader.workbench_data import WorkbenchData
    return WorkbenchData(roots, **kwargs)


def test_overview_empty_and_verified_primary_units(tmp_path):
    data = api({'ap2': tmp_path})
    assert data.overview() == {'runs': [], 'sources': [{'id': 'ap2'}], 'errors': [], 'read_only': True}
    record, key = make_run(tmp_path)
    overview = data.overview()
    row, = overview['runs']
    assert set(row) == {'key', 'source', 'run_id', 'name', 'hypothesis', 'model_kind', 'status',
                        'created_at', 'parent_id', 'selectable', 'training_rows', 'validation_rows',
                        'metrics', 'report_url', 'dataset_id', 'source_sha256', 'period'}
    assert row['key'] == key and row['run_id'] == record['run_id']
    assert row['model_kind'] == 'ridge' and row['status'] == 'completed' and row['selectable']
    assert row['name'] == '<script>alert(1)</script>'
    assert row['metrics'] == {'mae_bp': '2.0000', 'rmse_bp': '3.0000', 'net_points': '3',
                              'cost_points': '2.2', 'drawdown_points': '15', 'trade_count': 2}
    assert row['report_url'] == '/experiment-report?run=' + key.replace(':', '%3A')


def test_detail_separates_forecast_training_and_real_account_curves(tmp_path):
    _, key = make_run(tmp_path, kind='mlp')
    detail = api({'ap2': tmp_path}).detail(key)
    assert set(detail) == {'run', 'config', 'signature', 'runtime', 'metrics_by_model',
                           'trading_by_model', 'training', 'curves', 'warnings', 'error'}
    assert detail['run']['model_kind'] == 'mlp'
    assert detail['config']['model']['kind'] == 'mlp'
    assert detail['config']['source_csv'] == '[redacted]'
    assert detail['metrics_by_model']['mlp'] == {'mae_bp': '2.0000', 'rmse_bp': '3.0000'}
    assert detail['metrics_by_model']['zero_return']['mae_bp'] == '5.0000'
    assert detail['trading_by_model']['mlp']['modeled_cost_points'] == '2.2'
    assert detail['training'] == {'history': [
        {'epoch': 1, 'training_mse': 1.1, 'validation_mse': 1.2},
        {'epoch': 2, 'training_mse': 0.8, 'validation_mse': 0.9}],
        'best_epoch': 2, 'stopping_reason': 'patience'}
    curves = detail['curves']
    assert curves['source_count'] == 4 and curves['sampled'] is False
    assert [p[1] for p in curves['equity']] == ['0', '10', '-5', '3']
    assert [p[1] for p in curves['drawdown']] == ['0', '0', '-15', '-7']
    assert curves['equity'][0][0] is None
    assert any('initial' in warning.lower() for warning in detail['warnings'])
    text = json.dumps(detail)
    assert '/private' not in text and str(tmp_path) not in text
    assert 'PRIVATE RAW' not in text and 'PRIVATE WEIGHTS' not in text
    assert detail['error'] is None


def test_row_projects_verified_dataset_and_actual_periods(tmp_path):
    period = {'training_start': '2023-01-02', 'training_end': '2023-12-29',
              'validation_start': '2024-01-02', 'validation_end': '2024-12-30',
              'reserved_test_start': '2025-01-01', 'reserved_test_end': '2026-01-01'}
    _, key = make_run(tmp_path, summary_changes={
        'training': {'first_decision_at': period['training_start'],
                     'last_target_available_at': period['training_end']},
        'validation_forecast': {'ridge': {'mae': 0.0002, 'rmse': 0.0003},
                               'first_decision_at': period['validation_start'],
                               'last_decision_at': period['validation_end']},
        'reserved_test_start': period['reserved_test_start'],
        'reserved_test_end': period['reserved_test_end']})
    data = api({'ap2': tmp_path})
    row = data.detail(key)['run']
    assert row['dataset_id'] == 'fixture-validation'
    assert row['source_sha256'] == 'sha256:' + 'a' * 64
    assert row['period'] == period
    assert data.overview()['runs'][0] == row


@pytest.mark.parametrize('changes', [
    {'validation_forecast': {'ridge': {'mae': 'NaN', 'rmse': 0.1}}},
    {'validation_forecast': {'ridge': {'mae': -0.1, 'rmse': 0.1}}},
    {'validation_forecast': {'zero_return': {'mae': 0.1, 'rmse': 0.2}}},
    {'trading': {'ridge': []}},
    {'training': []},
    {'trading': {'ridge': {'net_points': 'NaN', 'modeled_cost_points': '1',
                          'max_drawdown_points': '1', 'trade_count': 1}}},
])
def test_integrity_valid_but_invalid_result_schema_is_not_selectable(tmp_path, changes):
    _, key = make_run(tmp_path, summary_changes=changes)
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert all(value is None for value in detail['run']['metrics'].values())


@pytest.mark.parametrize('accounts', [
    [{'at': None, 'equity': 'NaN'}],
    [{'at': None, 'equity': 'Infinity'}],
    [{'at': None}],
    ['not-an-account'],
])
def test_invalid_account_schema_cannot_publish_completed_curves(tmp_path, accounts):
    _, key = make_run(tmp_path, accounts=accounts)
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert detail['curves']['equity'] == []


@pytest.mark.parametrize('value', ['-Infinity', 'Infinity', 'NaN'])
@pytest.mark.parametrize('at', [None, '2024-01-02T08:00:00+00:00'])
def test_nonfinite_equity_invalidates_run_without_zero_substitution(tmp_path, value, at):
    _, key = make_run(tmp_path, accounts=[{'at': at, 'equity': value}])
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert detail['run']['selectable'] is False
    assert all(value is None for value in detail['run']['metrics'].values())
    assert detail['curves']['equity'] == [] and detail['curves']['drawdown'] == []


@pytest.mark.parametrize('capital', ['-Infinity', 'Infinity', 'NaN', '0', '-1'])
def test_initial_capital_must_be_finite_and_positive(tmp_path, capital):
    backtest = {'execution': {'initial_capital': capital},
                'accounts': [{'at': None, 'equity': '100000'}]}
    _, key = make_run(tmp_path, extra={'ridge-backtest.json': encode(backtest)})
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert detail['run']['selectable'] is False
    assert all(value is None for value in detail['run']['metrics'].values())
    assert detail['curves']['equity'] == [] and detail['curves']['drawdown'] == []


@pytest.mark.parametrize('history', [
    {'epoch_history': [], 'best_epoch': 1, 'stopping_reason': 'patience'},
    {'epoch_history': [{'epoch': 1, 'training_mse': 'NaN', 'validation_mse': 1}],
     'best_epoch': 1, 'stopping_reason': 'patience'},
    {'epoch_history': [{'epoch': 1, 'training_mse': 1, 'validation_mse': 1}],
     'best_epoch': 3, 'stopping_reason': 'patience'},
])
def test_invalid_mlp_history_is_not_published_as_valid_run(tmp_path, history):
    _, key = make_run(tmp_path, kind='mlp', extra={'training-history.json': encode(history)})
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert detail['training'] is None


@pytest.mark.parametrize('text', ['input=/private/input.csv', 'failed(/private/input.csv)',
                                  '"/private/input.csv"', r'input=C:\private\input.csv'])
def test_embedded_private_paths_are_redacted_without_interpreting_html(tmp_path, text):
    _, key = make_run(tmp_path, config_changes={'hypothesis': text})
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'completed'
    assert detail['run']['hypothesis'] == '[redacted]'
    assert detail['config']['hypothesis'] == '[redacted]'
    assert '/private' not in json.dumps(detail)


def test_forecast_zero_alias_is_one_baseline_not_two_independent_models(tmp_path):
    baseline = {'mae': 0.0005, 'rmse': 0.0006}
    _, key = make_run(tmp_path, summary_changes={'validation_forecast': {
        'ridge': {'mae': 0.0002, 'rmse': 0.0003}, 'zero': baseline, 'zero_return': baseline}})
    forecasts = api({'ap2': tmp_path}).detail(key)['metrics_by_model']
    assert set(forecasts) == {'ridge', 'zero_return'}
    assert forecasts['zero_return'] == {'mae_bp': '5.0000', 'rmse_bp': '6.0000'}


def test_curve_timestamps_cannot_expose_private_paths(tmp_path):
    _, key = make_run(tmp_path, accounts=[{'at': '/private/input.csv', 'equity': '100000'}])
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'invalid'
    assert '/private' not in json.dumps(detail)


def test_forecast_model_keys_cannot_expose_private_paths(tmp_path):
    _, key = make_run(tmp_path, summary_changes={'validation_forecast': {
        'ridge': {'mae': 0.0002, 'rmse': 0.0003},
        '/private/model.json': {'mae': 0.0005, 'rmse': 0.0006}}})
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == 'completed'
    assert '/private' not in json.dumps(detail)


def test_ridge_detail_has_no_training_history(tmp_path):
    _, key = make_run(tmp_path)
    assert api({'ap2': tmp_path}).detail(key)['training'] is None


@pytest.mark.parametrize('alias', ['', 'AP2', '../ap2', 'a:b', 'a b', 'a/b', 'ap2\n'])
def test_alias_validation(alias, tmp_path):
    with pytest.raises(ValueError):
        api({alias: tmp_path})


@pytest.mark.parametrize('key', ['../etc/passwd', 'ap2:run-abc', 'ap2:run-' + 'A' * 32,
                                  'ap2:run-' + '0' * 32 + '\n', 'ap2:../x', None])
def test_strict_key_validation(key, tmp_path):
    with pytest.raises(ValueError):
        api({'ap2': tmp_path}).detail(key)


def test_unknown_valid_keys_have_public_typed_error(tmp_path):
    from market_trader.workbench_data import UnknownRunError
    for key in ['missing:run-' + '0' * 32, 'ap2:run-' + '0' * 32]:
        with pytest.raises(UnknownRunError):
            api({'ap2': tmp_path}).detail(key)


@pytest.mark.parametrize('damage', ['record', 'config', 'manifest', 'artifact', 'missing', 'unknown_status'])
def test_one_corrupt_run_never_hides_good_run(tmp_path, damage):
    good, good_key = make_run(tmp_path)
    bad, bad_key = make_run(tmp_path)
    directory = tmp_path / bad['run_id']
    target = {'record': 'run.json', 'config': 'config.json', 'manifest': 'artifacts/manifest.json',
              'artifact': 'artifacts/weights.bin', 'missing': 'artifacts/raw-dataset.json'}
    if damage == 'missing':
        (directory / target[damage]).unlink()
    elif damage == 'unknown_status':
        (directory / 'run.json').write_bytes(encode({**bad, 'status': 'pretend'}))
    else:
        (directory / target[damage]).write_text('corrupt /private/credential password=private')
    data = api({'ap2': tmp_path, 'empty': tmp_path / 'absent'})
    rows = {row['key']: row for row in data.overview()['runs']}
    assert rows[good_key]['selectable'] is True
    assert rows[bad_key]['status'] == 'invalid' and rows[bad_key]['selectable'] is False
    assert all(value is None for value in rows[bad_key]['metrics'].values())
    detail = data.detail(bad_key)
    assert detail['error'] and detail['curves']['equity'] == []
    assert detail['metrics_by_model'] == {} and detail['training'] is None
    assert '/private' not in json.dumps(detail) and str(tmp_path) not in json.dumps(detail)


@pytest.mark.parametrize('status', ['failed', 'pending', 'running'])
def test_noncompleted_runs_have_null_metrics_and_no_error_disclosure(tmp_path, status):
    record, key = make_run(tmp_path)
    record.update(status=status, error={'message': '/private/pass token=secret'})
    (tmp_path / record['run_id'] / 'run.json').write_bytes(encode(record))
    detail = api({'ap2': tmp_path}).detail(key)
    assert detail['run']['status'] == status
    assert detail['run']['selectable'] is False
    assert all(value is None for value in detail['run']['metrics'].values())
    assert detail['metrics_by_model'] == {} and detail['trading_by_model'] == {}
    assert detail['curves']['source_count'] == 0 and detail['training'] is None
    assert 'secret' not in json.dumps(detail) and '/private' not in json.dumps(detail)


def test_missing_root_reports_only_alias_and_preserves_other_rows(tmp_path):
    root = tmp_path / 'good'
    record, _ = make_run(root)
    empty = tmp_path / 'empty'
    empty.mkdir()
    missing = tmp_path / 'private-unavailable-root'
    overview = api({'good': root, 'empty': empty, 'missing': missing}).overview()
    assert overview['errors'] == ['Registry missing is unavailable.']
    assert len(overview['runs']) == 1
    assert overview['runs'][0]['run_id'] == record['run_id']
    assert overview['runs'][0]['selectable'] is True
    assert str(tmp_path) not in json.dumps(overview)
    assert missing.name not in json.dumps(overview)


def test_bad_source_does_not_hide_other_sources_or_expose_paths(tmp_path):
    root = tmp_path / 'good'
    _, key = make_run(root)
    bad = tmp_path / 'file-not-directory'
    bad.write_text('private')
    overview = api({'good': root, 'bad': bad}).overview()
    assert len(overview['runs']) == 1 and overview['errors']
    assert str(tmp_path) not in json.dumps(overview)


def test_compare_cross_source_actual_conditions_and_path_diffs(tmp_path):
    _, first = make_run(tmp_path / 'a')
    _, second = make_run(tmp_path / 'b', kind='mlp', config_changes={'source_csv': '/other/input.csv'})
    second = second.replace('ap2:', 'ap3:')
    data = api({'ap2': tmp_path / 'a', 'ap3': tmp_path / 'b'})
    result = data.compare([first, second])
    assert set(result) == {'compatible', 'runs', 'condition_differences', 'parameter_differences',
                           'details', 'warnings'}
    assert result['compatible'] is True
    assert [r['key'] for r in result['runs']] == [first, second]
    assert result['condition_differences'] == [{'left': first, 'right': second, 'differences': []}]
    diff, = [d for d in result['parameter_differences'][0]['differences'] if d['field'] == 'source_csv']
    assert diff == {'field': 'source_csv', 'left': '[redacted]', 'right': '[redacted]'}
    assert any(d['field'] == 'model' for d in result['parameter_differences'][0]['differences'])
    assert str(tmp_path) not in json.dumps(result) and '/other' not in json.dumps(result)


def test_compare_incompatible_cost_and_runtime_conditions(tmp_path):
    keys = []
    for signature in [{'costs': {'spread': '2'}, 'runtime': 'a'},
                      {'costs': {'spread': '3'}, 'runtime': 'b'},
                      {'costs': {'spread': '4'}, 'runtime': 'b'}]:
        _, key = make_run(tmp_path, signature=signature)
        keys.append(key)
    result = api({'ap2': tmp_path}).compare(keys)
    assert result['compatible'] is False and result['warnings']
    assert len(result['condition_differences']) == 3
    assert {d['field'] for d in result['condition_differences'][0]['differences']} == {'costs.spread', 'runtime'}


@pytest.mark.parametrize('keys', [[], ['x'], ['x', 'x'], ['x'] * 5, 'not-a-list', [None, 'x']])
def test_comparison_rejects_invalid_selections(tmp_path, keys):
    with pytest.raises(ValueError):
        api({'ap2': tmp_path}).compare(keys)


def test_compare_rejects_failed_and_corrupt_completed(tmp_path):
    from market_trader.workbench_data import InvalidRunError
    _, left = make_run(tmp_path)
    record, right = make_run(tmp_path)
    data = api({'ap2': tmp_path})
    (tmp_path / record['run_id'] / 'artifacts/weights.bin').write_bytes(b'tampered')
    with pytest.raises(InvalidRunError):
        data.compare([left, right])
    record['status'] = 'failed'
    (tmp_path / record['run_id'] / 'run.json').write_bytes(encode(record))
    with pytest.raises(InvalidRunError):
        data.compare([left, right])


def test_compare_rejects_first_run_failed_during_later_entry_load(tmp_path, monkeypatch):
    from market_trader.workbench_data import InvalidRunError
    record, first = make_run(tmp_path)
    _, second = make_run(tmp_path)
    data = api({'ap2': tmp_path})
    original = data._load
    calls = 0

    def load(*args):
        nonlocal calls
        entry = original(*args)
        calls += 1
        if calls == 2:
            (tmp_path / record['run_id'] / 'run.json').write_bytes(
                encode({**record, 'status': 'failed'}))
        return entry

    monkeypatch.setattr(data, '_load', load)
    with pytest.raises(InvalidRunError, match='completed valid runs'):
        data.compare([first, second])


@pytest.mark.parametrize('phase', ['later_load', 'differences'])
@pytest.mark.parametrize('rewrite', ['failed', 'valid'])
def test_compare_revalidates_snapshots_through_computation(tmp_path, monkeypatch, phase, rewrite):
    from market_trader.workbench_data import InvalidRunError
    record, first = make_run(tmp_path)
    _, second = make_run(tmp_path)
    data = api({'ap2': tmp_path})
    directory = tmp_path / record['run_id']
    changed = False

    def mutate():
        nonlocal changed
        if changed:
            return
        changed = True
        if rewrite == 'failed':
            record['status'] = 'failed'
        else:
            summary_path = directory / 'artifacts/summary.json'
            summary = json.loads(summary_path.read_bytes())
            summary['comparison_signature']['execution']['spread'] = '9.0'
            summary['validation_forecast']['ridge']['mae'] = 0.123
            summary_path.write_bytes(encode(summary))
            manifest_path = directory / 'artifacts/manifest.json'
            manifest = json.loads(manifest_path.read_bytes())
            manifest['artifacts']['summary.json'] = registry._digest(summary_path.read_bytes())
            manifest_path.write_bytes(encode(manifest))
            record['manifest_sha256'] = registry._digest(manifest_path.read_bytes())
            record['summary'] = summary
        (directory / 'run.json').write_bytes(encode(record))
        registry._verified_record(tmp_path, record['run_id'])

    if phase == 'later_load':
        original = data._load

        def load(*args):
            entry = original(*args)
            if args[2] == second.split(':')[1]:
                mutate()
            return entry

        monkeypatch.setattr(data, '_load', load)
    else:
        original = registry._differences

        def differences(*args, **kwargs):
            mutate()
            return original(*args, **kwargs)

        monkeypatch.setattr(registry, '_differences', differences)

    with pytest.raises(InvalidRunError, match='completed valid runs'):
        data.compare([first, second])
    assert changed
    if rewrite == 'valid':
        result = data.compare([first, second])
        assert result['compatible'] is False
        assert result['details'][0]['signature']['execution']['spread'] == '9.0'
        assert result['details'][0]['run']['metrics']['mae_bp'] == '1230.000'
        diff, = result['condition_differences'][0]['differences']
        assert diff == {'field': 'execution.spread', 'left': '9.0', 'right': '2.0'}


def test_verified_cache_avoids_raw_reads_and_returns_detached_data(tmp_path, monkeypatch):
    _, key = make_run(tmp_path)
    data = api({'ap2': tmp_path})
    calls = []
    original = registry._read_bytes
    def observe(path):
        calls.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(registry, '_read_bytes', observe)
    first = data.detail(key)
    assert 'raw-dataset.json' in calls and 'weights.bin' in calls
    calls.clear()
    first['run']['name'] = 'mutated caller'
    assert data.overview()['runs'][0]['name'] != 'mutated caller'
    assert data.detail(key)['run']['selectable']
    assert calls == []
    _, other = make_run(tmp_path)
    data.compare([key, other])
    calls.clear()
    data.compare([key, other])
    assert calls == []


@pytest.mark.parametrize('target', ['run.json', 'config.json', 'artifacts/manifest.json',
                                    'artifacts/raw-dataset.json', 'artifacts/weights.bin'])
@pytest.mark.parametrize('mutation', ['in_place_preserve_mtime', 'replace_preserve_mtime'])
def test_cache_detects_all_artifact_record_and_config_mutations(tmp_path, target, mutation):
    record, key = make_run(tmp_path)
    data = api({'ap2': tmp_path})
    assert data.detail(key)['run']['selectable']
    path = tmp_path / record['run_id'] / target
    before = path.stat()
    original = path.read_bytes()
    damaged = bytes([original[0] ^ 1]) + original[1:]
    if mutation == 'replace_preserve_mtime':
        replacement = path.with_name('replacement')
        replacement.write_bytes(damaged)
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
    else:
        path.write_bytes(damaged)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    detail = data.detail(key)
    assert detail['run']['status'] == 'invalid'
    assert all(value is None for value in detail['run']['metrics'].values())
    path.write_bytes(original)
    assert data.detail(key)['run']['selectable']


@pytest.mark.parametrize('target', ['root', 'run', 'artifacts', 'config.json', 'artifacts/weights.bin'])
def test_cache_rejects_symlink_replacement(tmp_path, target):
    root = tmp_path / 'runs'
    record, key = make_run(root)
    data = api({'ap2': root})
    assert data.detail(key)['run']['selectable']
    directory = root / record['run_id']
    path = {'root': root, 'run': directory}.get(target, directory / target)
    moved = tmp_path / 'moved'
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    assert data.detail(key)['run']['status'] == 'invalid'


def test_cache_source_directory_replacement_is_reverified(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / 'runs'
    _, key = make_run(root)
    data = api({'ap2': root})
    data.detail(key)
    moved = tmp_path / 'old'
    root.rename(moved)
    shutil.copytree(moved, root)
    calls = []
    original = registry._verified_record
    def observe(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(registry, '_verified_record', observe)
    assert data.detail(key)['run']['selectable'] and len(calls) == 1


def test_cache_serializes_heavy_fills_and_bounds_entries(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time
    _, key = make_run(tmp_path)
    data = api({'ap2': tmp_path})
    active = maximum = fills = 0
    guard = Lock()
    original = registry._verified_record
    def observe(*args):
        nonlocal active, maximum, fills
        with guard:
            active += 1
            maximum = max(maximum, active)
            fills += 1
        time.sleep(0.01)
        try:
            return original(*args)
        finally:
            with guard:
                active -= 1
    monkeypatch.setattr(registry, '_verified_record', observe)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(data.detail, [key] * 8))
    assert maximum == 1 and fills == 1
    assert all(r['run']['selectable'] for r in results)
    data._cache_limit = 2
    for _ in range(3):
        _, another = make_run(tmp_path)
        data.detail(another)
    assert len(data._cache) <= 2


def test_report_returns_only_fixed_checked_bytes(tmp_path):
    from market_trader.workbench_data import InvalidRunError
    record, key = make_run(tmp_path, report=b'<html>REAL CHECKED</html>')
    data = api({'ap2': tmp_path})
    assert data.report(key) == b'<html>REAL CHECKED</html>'
    with pytest.raises(TypeError):
        data.report(key, 'weights.bin')
    path = tmp_path / record['run_id'] / 'artifacts/report.html'
    before = path.stat()
    path.write_bytes(b'<html>TAMPERED!!!!</html>')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(InvalidRunError):
        data.report(key)
    assert data.detail(key)['run']['status'] == 'invalid'


def test_display_sampling_preserves_endpoints_and_equity_drawdown_extrema(tmp_path):
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    accounts = [{'at': (start + timedelta(minutes=i)).isoformat(), 'equity': str(100000 + i % 17)}
                for i in range(3001)]
    accounts[555]['equity'] = '101000'
    accounts[1234]['equity'] = '98000'
    _, key = make_run(tmp_path, accounts=accounts)
    data = api({'ap2': tmp_path})
    detail = data.detail(key)
    curves = detail['curves']
    assert curves['sampled'] is True and curves['source_count'] == 3001
    assert 4 <= len(curves['equity']) <= 800
    assert [p[0] for p in curves['equity']] == [p[0] for p in curves['drawdown']]
    assert curves['equity'][0] == [accounts[0]['at'], '0']
    assert curves['equity'][-1] == [accounts[-1]['at'], str(int(accounts[-1]['equity']) - 100000)]
    assert [accounts[555]['at'], '1000'] in curves['equity']
    assert [accounts[1234]['at'], '-2000'] in curves['equity']
    assert [accounts[1234]['at'], '-3000'] in curves['drawdown']
    assert any('sampl' in warning.lower() for warning in detail['warnings'])
    assert detail['run']['metrics']['drawdown_points'] == '15'  # untouched full ledger summary
    small = api({'ap2': tmp_path}, curve_limit=12).detail(key)['curves']
    assert len(small['equity']) <= 12 and [accounts[1234]['at'], '-3000'] in small['drawdown']


@pytest.mark.parametrize('limit', [0, -1, True, 3, 801, 10.5])
def test_invalid_curve_limits_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        api({'ap2': tmp_path}, curve_limit=limit)


@pytest.mark.parametrize('operation', ['report', 'detail', 'overview'])
def test_oversized_report_preflight_precedes_cold_verification(tmp_path, monkeypatch, operation):
    from market_trader.workbench_data import InvalidRunError
    record, key = make_run(tmp_path)
    report = tmp_path / record['run_id'] / 'artifacts/report.html'
    with report.open('r+b') as stream:
        stream.truncate(32 * 1024 * 1024 + 1)  # Sparse; never materialize the oversized HTML.

    def forbidden_verification(*args):
        pytest.fail('Oversized report reached unbounded registry verification')

    monkeypatch.setattr(registry, '_verified_record', forbidden_verification)
    data = api({'ap2': tmp_path})
    if operation == 'report':
        with pytest.raises(InvalidRunError):
            data.report(key)
    else:
        row = data.detail(key)['run'] if operation == 'detail' else data.overview()['runs'][0]
        assert row['status'] == 'invalid' and row['selectable'] is False
    assert not data._cache


def test_report_enforces_32mib_limit(tmp_path):
    from market_trader.workbench_data import InvalidRunError
    _, key = make_run(tmp_path, report=b'X' * (32 * 1024 * 1024 + 1))
    with pytest.raises(InvalidRunError):
        api({'ap2': tmp_path}).report(key)


def test_report_no_fallback_for_missing_fixed_report_or_failed_run(tmp_path):
    from market_trader.workbench_data import InvalidRunError
    record, key = make_run(tmp_path)
    path = tmp_path / record['run_id']
    (path / 'artifacts/report.html').unlink()
    with pytest.raises(InvalidRunError):
        api({'ap2': tmp_path}).report(key)
    record['status'] = 'failed'
    (path / 'run.json').write_bytes(encode(record))
    with pytest.raises(InvalidRunError):
        api({'ap2': tmp_path}).report(key)
