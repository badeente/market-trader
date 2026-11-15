"""The experiment workbench is an offline CLI, not a remote training endpoint."""
import json
import subprocess
import sys

import pytest


def test_experiment_cli_runs_replays_and_inspects_real_artifacts(tmp_path):
    from importlib.util import find_spec
    from tests.test_reference import reference_fixture

    assert find_spec('market_trader.experiments') is not None, 'AP2 orchestration is missing'
    _, reference = reference_fixture(tmp_path)
    config = tmp_path/'experiment.toml'
    config.write_text(reference.read_text().replace(
        'experiment = "ap1-deuidxeur-open30-v1"',
        'name = "synthetic integration"\nhypothesis = "replay exact settings"\nseed = 0'))
    registry = tmp_path/'registry'
    def cli(*args):
        result = subprocess.run([sys.executable, '-m', 'market_trader.cli', 'experiment',
                                 *args, '--registry', str(registry)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    first = cli('run', '--config', str(config))
    assert first['status'] == 'completed'
    replay = cli('replay', '--run-id', first['run_id'])
    assert replay['status'] == 'completed'
    assert replay['run_id'] != first['run_id']
    assert replay['parent_id'] == first['run_id']
    assert len(cli('list')) == 2
    assert cli('show', '--run-id', first['run_id'])['run_id'] == first['run_id']
    comparison = cli('compare', '--left', first['run_id'], '--right', replay['run_id'])
    assert comparison['compatible'] is True
    first_artifacts = registry/first['run_id']/'artifacts'
    replay_artifacts = registry/replay['run_id']/'artifacts'
    assert (first_artifacts/'ridge-model.json').read_bytes() == (replay_artifacts/'ridge-model.json').read_bytes()
    summary = json.loads((first_artifacts/'summary.json').read_text())
    assert summary['training_rows'] == 48 and summary['validation_rows'] == 48
    assert summary['test_metrics_computed'] is False
    result = subprocess.run([sys.executable, '-m', 'market_trader.cli', 'experiment', 'show', '--registry',
                             str(registry), '--run-id', '../escape'], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and 'Traceback' not in result.stderr


@pytest.mark.parametrize('body', [[], None, 'not an object', 7])
def test_experiment_cli_rejects_nonobject_run_record_without_traceback(tmp_path, body):
    registry = tmp_path/'registry'
    run_id = 'run-' + 'a'*32
    directory = registry/run_id
    directory.mkdir(parents=True)
    (directory/'run.json').write_text(json.dumps(body))
    result = subprocess.run([sys.executable, '-m', 'market_trader.cli', 'experiment', 'show',
                             '--registry', str(registry), '--run-id', run_id],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    assert 'run record must be a JSON object' in result.stderr


@pytest.mark.parametrize('command,options', [
    ('run', ('--config', '--registry', '--parent')),
    ('list', ('--registry',)),
    ('show', ('--registry', '--run-id')),
    ('replay', ('--registry', '--run-id')),
    ('compare', ('--registry', '--left', '--right')),
])
def test_experiment_commands_document_their_contract(command, options):
    result = subprocess.run([sys.executable, '-m', 'market_trader.cli', 'experiment', command, '--help'],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for option in options:
        assert option in result.stdout
