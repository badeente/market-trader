"""AP5 command-line entrypoints; no network mutation or real market fixtures."""
import json
from pathlib import Path
from market_trader.cli import main
from tests.test_experiment_variants import parent


def test_cli_derives_verified_new_document_without_overwrite(tmp_path, capsys):
    record = parent(tmp_path)
    output = tmp_path/'variant.json'
    args = ['experiment', 'derive', '--registry', str(tmp_path/'runs'), '--run-id', record['run_id'],
            '--name', 'seed variant', '--hypothesis', 'controlled seed difference',
            '--set', 'seed=1', '--output', str(output)]
    try:
        status = main(args)
    except SystemExit as error:
        status = error.code
    assert status == 0
    body = json.loads(output.read_text())
    assert body['config']['seed'] == 1 and body['parent_id'] == record['run_id']
    assert main(args) == 1
    assert 'already exists' in capsys.readouterr().err


def test_cli_sweep_preflights_and_publishes_a_plan(tmp_path, capsys):
    record = parent(tmp_path)
    grid = tmp_path/'grid.json'
    grid.write_text('{"seed":[1,2]}')
    output = tmp_path/'sweep.json'
    args = ['experiment', 'sweep', '--registry', str(tmp_path/'runs'), '--run-id', record['run_id'],
            '--name', 'two seeds', '--hypothesis', 'finite example', '--grid', str(grid), '--output', str(output)]
    try:
        status = main(args)
    except SystemExit as error:
        status = error.code
    assert status == 0
    assert len(json.loads(output.read_text())['variants']) == 2
    assert not (tmp_path/'queue').exists()


def test_local_job_commands_are_discoverable(capsys):
    try:
        status = main(['experiment','job','--help'])
    except SystemExit as error:
        status = error.code
    assert status == 0
    text = capsys.readouterr().out
    assert all(word in text for word in ['submit','list','show','cancel','work','results'])


def test_cli_queue_submission_cancel_and_real_worker_failure(tmp_path, capsys):
    from market_trader.experiment_jobs import read_job
    config = tmp_path/'missing.toml'
    config.write_text('version=1\nname="synthetic failure"\nhypothesis="missing source is a failure, not market evidence"\nseed=0\nsource_csv="missing.csv"\nsource_sha256="sha256:'+'0'*64+'"\n')
    queue, registry = tmp_path/'queue', tmp_path/'runs'
    args = ['experiment','job','submit','--queue',str(queue),'--registry',str(registry),'--config',str(config)]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first['status'] == 'queued'
    assert main(['experiment','job','cancel','--queue',str(queue),'--job-id',first['job_id']]) == 0
    capsys.readouterr()
    assert read_job(queue,first['job_id'])['status'] == 'cancelled'
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert main(['experiment','job','work','--queue',str(queue),'--max-jobs','1','--total-seconds','20']) == 0
    assert read_job(queue,second['job_id'])['status'] == 'failed'


def test_replay_is_queued_with_exact_original_snapshot(tmp_path, capsys):
    record = parent(tmp_path)
    queue = tmp_path/'queue'
    args = ['experiment','job','submit','--queue',str(queue),'--registry',str(tmp_path/'runs'),
            '--replay-run',record['run_id']]
    try:
        status = main(args)
    except SystemExit as error:
        status = error.code
    assert status == 0
    job = json.loads(capsys.readouterr().out)
    assert job['config_sha256'] == record['config_sha256']
    assert job['parent_id'] == record['run_id']
    assert job['metadata']['operation'] == 'replay'
