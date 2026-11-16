"""Synthetic bounded-queue lifecycle tests; no research data fits."""
import copy
import json
import os
import time
from pathlib import Path
import stat

import pytest


def api():
    from market_trader import experiment_jobs
    return experiment_jobs


def payload(tmp_path):
    from market_trader.experiment_config import ExperimentConfig, experiment_config_payload
    return experiment_config_payload(ExperimentConfig(1, 'synthetic', 'lifecycle only', 0,
                                      tmp_path / 'invalid.csv', 'sha256:' + '0' * 64))


def test_submit_private_immutable_detached_snapshot(tmp_path):
    jobs = api()
    config = payload(tmp_path)
    submitted = copy.deepcopy(config)
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', config, metadata={'changes': []})
    assert job['status'] == 'queued'
    assert job['budget'] == {'wall_seconds': 300, 'cpu_seconds': 180, 'memory_mib': 4096}
    assert stat.S_IMODE(queue.stat().st_mode) == 0o700
    assert stat.S_IMODE((queue / job['job_id']).stat().st_mode) == 0o700
    config['name'] = 'mutated'
    assert json.loads((queue / job['job_id'] / 'config.json').read_text()) == submitted
    assert jobs.list_jobs(queue) == [job]
    assert jobs.read_job(queue, job['job_id']) == job


def test_cancel_is_request_only_and_events_are_append_only(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    event_path = queue / job['job_id'] / 'events.jsonl'
    before = event_path.read_bytes()
    cancelled = jobs.cancel_job(queue, job['job_id'])
    assert cancelled['cancel_requested'] is True
    assert cancelled['status'] == 'cancelled'
    assert cancelled['run_id'] is None
    assert event_path.read_bytes().startswith(before)
    assert jobs.cancel_job(queue, job['job_id']) == cancelled
    assert [json.loads(line)['event'] for line in event_path.read_text().splitlines()] == ['submitted', 'cancel_requested', 'cancelled']


def test_actual_fixed_module_subprocess_links_failed_run(tmp_path):
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    Path(config['source_csv']).write_text('invalid synthetic CSV\n')
    job = jobs.submit_job(queue, runs, config)
    result = jobs.work_jobs(queue, max_jobs=1, total_seconds=15)
    assert result['processed'] == 1
    failed = jobs.read_job(queue, job['job_id'])
    assert failed['status'] == 'failed'
    assert failed['started_at'] and failed['finished_at']
    assert failed['run_id'].startswith('run-')
    from market_trader.experiment_registry import read_run
    assert read_run(runs, failed['run_id'])['status'] == 'failed'
    assert [json.loads(line)['event'] for line in (queue / job['job_id'] / 'events.jsonl').read_text().splitlines()] == ['submitted', 'running', 'run_linked', 'failed']


def test_actual_subprocess_wall_timeout_fails_unfinished_run(tmp_path, monkeypatch):
    import threading
    jobs = api()
    original = jobs.subprocess.Popen
    timers = []
    def bounded_launch(*args, **kwargs):
        process = original(*args, **kwargs)
        timer = threading.Timer(4, lambda: process.kill() if process.poll() is None else None)
        timer.start()
        timers.append(timer)
        return process
    monkeypatch.setattr(jobs.subprocess, 'Popen', bounded_launch)
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    os.mkfifo(config['source_csv'])  # A genuine blocked input, not a fake worker.
    job = jobs.submit_job(queue, runs, config, budget={'wall_seconds': 1})
    start = time.monotonic()
    try:
        jobs.work_jobs(queue, total_seconds=10)
    finally:
        for timer in timers:
            timer.cancel()
    record = jobs.read_job(queue, job['job_id'])
    assert record['status'] == 'timed_out'
    assert time.monotonic() - start < 6  # Includes durable record fsync under parallel test load.
    from market_trader.experiment_registry import read_run
    assert read_run(runs, record['run_id'])['status'] == 'failed'


def test_actual_running_cancel_and_worker_exclusion(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    os.mkfifo(config['source_csv'])
    job = jobs.submit_job(queue, runs, config, budget={'wall_seconds': 4})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(jobs.work_jobs, queue, total_seconds=6)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            record = jobs.read_job(queue, job['job_id'])
            if record['run_id']:
                break
            time.sleep(0.02)
        assert record['run_id']
        with pytest.raises(ValueError, match='worker'):
            jobs.work_jobs(queue)
        jobs.cancel_job(queue, job['job_id'])
        result = future.result(timeout=6)
    assert result['jobs'][0]['status'] == 'cancelled'
    from market_trader.experiment_registry import read_run
    cancelled_run = read_run(runs, record['run_id'])
    assert cancelled_run['status'] == 'failed'
    assert cancelled_run['error']['type'] == 'JobTerminated'


def test_queued_cancel_does_not_create_registry(tmp_path):
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    job = jobs.submit_job(queue, runs, payload(tmp_path))
    jobs.cancel_job(queue, job['job_id'])
    result = jobs.work_jobs(queue)
    assert result['processed'] == 0
    assert jobs.read_job(queue, job['job_id'])['status'] == 'cancelled'
    assert not runs.exists()


def test_batch_preflights_all_then_submits_detached_records(tmp_path):
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    bad = dict(config, unexpected='not allowed')
    with pytest.raises(ValueError):
        jobs.submit_jobs(queue, runs, [{'payload': config}, {'payload': bad}])
    assert not queue.exists()
    records = jobs.submit_jobs(queue, runs, [{'payload': config}, {'payload': dict(config, name='variant')}])
    assert len(records) == 2
    assert {r['job_id'] for r in jobs.list_jobs(queue)} == {r['job_id'] for r in records}


def test_batch_rolls_back_new_directories_on_late_io_error(tmp_path, monkeypatch):
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    original = jobs._write
    writes = 0
    def failing_write(path, value, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 5:
            raise OSError('synthetic disk failure')
        return original(path, value, **kwargs)
    monkeypatch.setattr(jobs, '_write', failing_write)
    with pytest.raises(OSError, match='disk failure'):
        jobs.submit_jobs(queue, runs, [{'payload': payload(tmp_path)}] * 2)
    assert jobs.list_jobs(queue) == []
    assert not list(queue.glob('job-*'))


def test_safety_installs_real_unix_caps_and_cooperative_term(tmp_path):
    import subprocess
    import sys
    from market_trader import experiment_job_worker as worker
    assert hasattr(worker, '_install_safety')
    code = '''
import json, os, resource, signal
from market_trader.experiment_job_worker import _install_safety, JobTerminated
_install_safety({'cpu_seconds': 2, 'memory_mib': 256}, os.getppid())
print(json.dumps([resource.getrlimit(resource.RLIMIT_CPU), resource.getrlimit(resource.RLIMIT_AS)]), flush=True)
try:
    os.kill(os.getpid(), signal.SIGTERM)
except JobTerminated:
    print('cooperative', flush=True)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    limits, message = result.stdout.splitlines()
    assert json.loads(limits) == [[2, 3], [256 * 1024 * 1024, 256 * 1024 * 1024]]
    assert message == 'cooperative'


def test_stale_running_is_interrupted_without_pid_signals_or_retry(tmp_path, monkeypatch):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    path = queue / job['job_id'] / 'job.json'
    record = jobs.read_job(queue, job['job_id'])
    record.update(status='running', pid=os.getpid())  # Deliberately untrusted persisted PID.
    jobs._write(path, record)
    def forbidden(*args, **kwargs):
        pytest.fail('recovery must never signal a persisted PID')
    monkeypatch.setattr(jobs.os, 'killpg', forbidden)
    result = jobs.work_jobs(queue)
    assert result['processed'] == 0
    assert jobs.read_job(queue, job['job_id'])['status'] == 'interrupted'
    assert result['recovered'][0]['job_id'] == job['job_id']


def test_advertised_completion_requires_verified_committed_run(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    record = dict(job, status='completed')
    jobs._write(queue / job['job_id'] / 'job.json', record)
    with pytest.raises(ValueError, match='completed'):
        jobs.read_job(queue, job['job_id'])


def test_cpu_exhaustion_exit_is_timed_out(tmp_path, monkeypatch):
    jobs = api()
    class FinishedCpuWorker:
        returncode = 124
        def __init__(self, *args, **kwargs):
            pass
        def poll(self):
            return 124
        def wait(self, **kwargs):
            return 124
    monkeypatch.setattr(jobs.subprocess, 'Popen', FinishedCpuWorker)
    queue = tmp_path / 'queue'
    jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    assert jobs.work_jobs(queue)['jobs'][0]['status'] == 'timed_out'


@pytest.mark.parametrize('target', ['config.json', 'job.json', 'submission.json', 'events.jsonl', 'directory', 'queue', '.queue.lock', '.worker.lock'])
def test_symlinks_are_rejected_without_touching_targets(tmp_path, target):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    outside = tmp_path / 'outside'
    outside.write_text('DO NOT TOUCH')
    if target == 'queue':
        moved = tmp_path / 'original-queue'
        queue.rename(moved)
        queue.symlink_to(moved, target_is_directory=True)
    elif target == 'directory':
        directory = queue / job['job_id']
        moved = tmp_path / 'original-job'
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    else:
        location = queue / target if target.startswith('.') else queue / job['job_id'] / target
        location.unlink(missing_ok=True)
        location.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        if target == '.worker.lock':
            jobs.work_jobs(queue)
        elif target == 'events.jsonl':
            jobs.cancel_job(queue, job['job_id'])
        else:
            jobs.read_job(queue, job['job_id'])
    assert outside.read_text() == 'DO NOT TOUCH'


@pytest.mark.parametrize('metadata', [{'x': float('nan')}, {'x': 'z' * 8192}, {1: 'wrong-key'}, {'x': Path('/x')}])
def test_bounded_strict_metadata_rejected_before_allocation(tmp_path, metadata):
    with pytest.raises((ValueError, TypeError)):
        api().submit_job(tmp_path / 'queue', tmp_path / 'runs', payload(tmp_path), metadata=metadata)
    assert not (tmp_path / 'queue').exists()


def test_snapshot_tamper_duplicate_json_and_public_queue_are_rejected(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    record_path = queue / job['job_id'] / 'job.json'
    original = record_path.read_bytes()
    record_path.write_text('{"status":"queued","status":"completed"}')
    with pytest.raises(ValueError, match='duplicate'):
        jobs.read_job(queue, job['job_id'])
    record_path.write_bytes(original)
    config_path = queue / job['job_id'] / 'config.json'
    config = json.loads(config_path.read_text())
    config['name'] = 'tampered'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='digest|canonical'):
        jobs.read_job(queue, job['job_id'])
    queue.chmod(0o755)
    with pytest.raises(ValueError, match='private'):
        jobs.list_jobs(queue)


def test_concurrent_batch_capacity_is_atomic(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    jobs = api()
    # fsync timing is orthogonal to flock/capacity correctness.
    monkeypatch.setattr(jobs.os, 'fsync', lambda fd: None)
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    submission = {'payload': payload(tmp_path)}
    for _ in range(7):
        jobs.submit_jobs(queue, runs, [submission] * 4)
    jobs.submit_jobs(queue, runs, [submission] * 2)
    barrier = threading.Barrier(2)
    def submit():
        barrier.wait()
        try:
            return len(jobs.submit_jobs(queue, runs, [submission] * 2))
        except ValueError as error:
            assert 'capacity' in str(error)
            return 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(results) == [0, 2]
    assert len(jobs.list_jobs(queue)) == 32
    assert len(list(queue.glob('job-*'))) == 32


@pytest.mark.parametrize('publication', ['allocated', 'running'])
def test_controller_death_kills_real_worker_and_recovers_linked_run(tmp_path, publication):
    import signal
    import subprocess
    import sys
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    os.mkfifo(config['source_csv'])
    job = jobs.submit_job(queue, runs, config, budget={'wall_seconds': 20})
    # Trusted literal controller fixture. The production child command is
    # still exclusively the fixed module entry point.
    script = '''
import sys
from pathlib import Path
from market_trader import experiment_jobs as jobs
original = jobs.subprocess.Popen
def launch(command, **kwargs):
    if sys.argv[2] == 'allocated':
        # Hold after durable ownership, before config/record publication.
        code = """
import time
from market_trader import experiment_jobs as jobs
from market_trader.experiment_job_worker import main
original = jobs._link_run
def link(*args):
    original(*args)
    time.sleep(30)
jobs._link_run = link
raise SystemExit(main())
"""
        command = [sys.executable, '-c', code, *command[3:]]
    child = original(command, **kwargs)
    print(child.pid, flush=True)
    return child
jobs.subprocess.Popen = launch
jobs.work_jobs(Path(sys.argv[1]), total_seconds=25)
'''
    controller = subprocess.Popen([sys.executable, '-c', script, str(queue), publication],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        import select
        assert select.select([controller.stdout], [], [], 5)[0]
        child_pid = int(controller.stdout.readline())
        deadline = time.monotonic() + 5
        record = jobs.read_job(queue, job['job_id'])
        def ready(record):
            return record['run_id'] and (publication == 'allocated' or
                (runs / record['run_id'] / 'run.json').exists())
        while not ready(record) and time.monotonic() < deadline:
            time.sleep(0.02)
            record = jobs.read_job(queue, job['job_id'])
        assert ready(record)
        controller.kill()
        controller.wait(timeout=3)
        # Zombie is already dead and cannot fit, retain locks, or receive work.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                status = Path(f'/proc/{child_pid}/stat').read_text().split()[2]
            except FileNotFoundError:
                break
            if status == 'Z':
                break
            time.sleep(0.02)
        else:
            pytest.fail('worker survived controller death')
        result = jobs.work_jobs(queue, total_seconds=5)
        assert result['processed'] == 0
        assert jobs.read_job(queue, job['job_id'])['status'] == 'interrupted'
        from market_trader.experiment_registry import read_run
        if publication == 'running':
            assert read_run(runs, record['run_id'])['status'] == 'failed'
        else:
            assert not (runs / record['run_id'] / 'run.json').exists()
            assert jobs.registry_api.list_runs(runs) == []
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=3)
        controller.stdout.close()
        controller.stderr.close()


def _synthetic_completed_run(queue, job):
    jobs = api()
    from market_trader import experiment_registry as registry
    snapshot = jobs._read(queue / job['job_id'] / 'config.json')
    def execute(output):
        jobs._link_run(queue, job['job_id'], output.parent.name)
        output.mkdir()
        summary = {'runtime': {}, 'training_rows': 0, 'validation_rows': 0,
            'test_metrics_computed': False, 'model_report_id': 'synthetic-model',
            'training_data_id': 'synthetic-training', 'validation_dataset_id': 'synthetic-validation',
            'comparison_signature': 'synthetic', 'validation_forecast': {}, 'trading': {}}
        (output / 'summary.json').write_bytes(registry._encode(summary))
        manifest = {'status': 'complete', 'artifacts': {
            'summary.json': registry._digest((output / 'summary.json').read_bytes())}}
        (output / 'manifest.json').write_bytes(registry._encode(manifest))
        return summary
    return registry.register_run(Path(job['registry']), snapshot, executor=execute)


def test_committed_run_wins_over_late_cancel_and_crash_recovery(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    with jobs._lock(queue):
        jobs._transition(queue, job, 'running')
    run = _synthetic_completed_run(queue, job)
    jobs.cancel_job(queue, job['job_id'])
    result = jobs.work_jobs(queue)
    assert result['processed'] == 0
    record = jobs.read_job(queue, job['job_id'])
    assert record['status'] == 'completed'
    assert record['run_id'] == run['run_id']
    before = (queue / job['job_id'] / 'events.jsonl').read_bytes()
    assert jobs.cancel_job(queue, job['job_id']) == record
    assert (queue / job['job_id'] / 'events.jsonl').read_bytes() == before


def test_real_owned_group_escalates_term_to_kill():
    import subprocess
    import sys
    jobs = api()
    child = subprocess.Popen([sys.executable, '-c',
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(10)"],
        stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert child.stdout is not None
    try:
        assert child.stdout.readline().strip() == 'ready'
        start = time.monotonic()
        jobs._stop_owned(child)
        assert child.returncode == -9
        assert time.monotonic() - start < 3
        jobs._stop_owned(child)  # Already reaped: must not signal/reuse its PID.
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        child.stdout.close()


def test_total_worker_budget_leaves_later_job_queued(tmp_path):
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    os.mkfifo(config['source_csv'])
    records = jobs.submit_jobs(queue, runs, [{'payload': config}] * 2)
    result = jobs.work_jobs(queue, max_jobs=2, total_seconds=1)
    assert result['processed'] == 1
    assert result['jobs'][0]['status'] == 'timed_out'
    assert jobs.read_job(queue, records[1]['job_id'])['status'] == 'queued'


@pytest.mark.parametrize('options', [{'max_jobs': True}, {'max_jobs': 0}, {'max_jobs': 33},
                                      {'total_seconds': 0}, {'total_seconds': 3601}, {'total_seconds': 1.5}])
def test_worker_bounds_rejected_before_allocation(tmp_path, options):
    with pytest.raises(ValueError):
        api().work_jobs(tmp_path / 'queue', **options)
    assert not (tmp_path / 'queue').exists()


def test_absolute_cpu_hard_cap_is_never_above_six_hundred(monkeypatch):
    from market_trader import experiment_job_worker as worker
    calls = []
    monkeypatch.setattr(worker, '_parent_guard', lambda pid: None)
    monkeypatch.setattr(worker.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(worker.resource, 'setrlimit', lambda kind, value: calls.append((kind, value)))
    worker._install_safety({'cpu_seconds': 600, 'memory_mib': 8192}, os.getpid())
    assert (worker.resource.RLIMIT_CPU, (600, 600)) in calls


def test_event_hardlinks_are_rejected(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    target = tmp_path / 'not-an-event-log'
    target.write_text('untouched')
    target.chmod(0o600)
    log = queue / job['job_id'] / 'events.jsonl'
    log.unlink()
    os.link(target, log)
    with pytest.raises(ValueError):
        jobs.cancel_job(queue, job['job_id'])
    assert target.read_text() == 'untouched'


def test_snapshot_digest_protects_exact_bytes_not_just_json_values(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path))
    config = queue / job['job_id'] / 'config.json'
    config.write_bytes(config.read_bytes() + b' ')
    with pytest.raises(ValueError, match='digest|canonical'):
        jobs.read_job(queue, job['job_id'])


def test_immutable_metadata_distinguishes_boolean_from_integer(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    job = jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path), metadata={'flag': True})
    job['metadata']['flag'] = 1
    jobs._write(queue / job['job_id'] / 'job.json', job)
    with pytest.raises(ValueError, match='immutable'):
        jobs.read_job(queue, job['job_id'])


def test_real_cpu_limit_interrupts_busy_subprocess():
    import subprocess
    import sys
    code = '''
import os
from market_trader.experiment_job_worker import _install_safety, JobCpuLimit
_install_safety({'cpu_seconds': 1, 'memory_mib': 256}, os.getppid())
try:
    while True:
        pass
except JobCpuLimit:
    print('CPU budget exhausted')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'CPU budget exhausted'


def test_invalid_submission_allocates_nothing(tmp_path):
    jobs = api()
    queue = tmp_path / 'queue'
    bad = payload(tmp_path)
    bad['command'] = 'unsafe'
    with pytest.raises((ValueError, TypeError)):
        jobs.submit_job(queue, tmp_path / 'runs', bad)
    assert not queue.exists()
    with pytest.raises((ValueError, FileNotFoundError)):
        jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path), parent_id='run-' + '0' * 32)
    assert not queue.exists()


@pytest.mark.parametrize('budget', [
    {'wall_seconds': True}, {'wall_seconds': 0}, {'wall_seconds': 901},
    {'cpu_seconds': 601}, {'memory_mib': 255}, {'memory_mib': 8193},
    {'command': 'anything'}, {'cpu_seconds': 1.5},
])
def test_strict_budget_before_allocation(tmp_path, budget):
    with pytest.raises((ValueError, TypeError)):
        api().submit_job(tmp_path / 'queue', tmp_path / 'runs', payload(tmp_path), budget=budget)
    assert not (tmp_path / 'queue').exists()


@pytest.mark.parametrize('native', [True, False])
def test_actual_native_cpu_hard_kill_and_unrelated_sigkill(tmp_path, monkeypatch, native):
    import subprocess
    import sys
    jobs = api()
    original = subprocess.Popen
    # Native OpenSSL work cannot dispatch the Python SIGXCPU handler before
    # the hard limit. The other literal fixture is a deliberate non-CPU kill.
    code = '''
import os, hashlib, signal
from market_trader.experiment_job_worker import _install_safety
_install_safety({'cpu_seconds':1,'memory_mib':256},os.getppid())
'''+ ("hashlib.pbkdf2_hmac('sha256', b'a', b'b', 200000000)" if native
       else "os.kill(os.getpid(), signal.SIGKILL)")
    children = []
    def launch(command, **kwargs):
        child = original([sys.executable, '-c', code], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(jobs.subprocess, 'Popen', launch)
    queue = tmp_path / 'queue'
    jobs.submit_job(queue, tmp_path / 'runs', payload(tmp_path),
                    budget={'cpu_seconds': 1, 'wall_seconds': 12, 'memory_mib': 256})
    record = jobs.work_jobs(queue, total_seconds=15)['jobs'][0]
    assert children[0].returncode == -9
    assert record['status'] == ('timed_out' if native else 'failed')
    evidence = record['termination']
    assert evidence['returncode'] == -9
    assert evidence['cpu_hard_limit_seconds'] == 2
    if native:
        soft, hard = evidence['observed_cpu_limit_seconds']
        assert hard == 2 and 1 <= soft <= hard  # Kernel may advance soft on XCPU.
    assert evidence['cpu_limit_exhausted'] is native
    assert (evidence['user_cpu_seconds'] + evidence['system_cpu_seconds'] >= 1.95) is native


@pytest.mark.parametrize('phase', ['allocated', 'before_pending', 'pending', 'running'])
def test_actual_worker_killed_before_executor_has_durable_owner(tmp_path, monkeypatch, phase):
    import subprocess
    import sys
    jobs = api()
    registry = jobs.registry_api
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    config = payload(tmp_path)
    job = jobs.submit_job(queue, runs, config)
    # An unrelated same-config unfinished run must never be selected by a scan.
    def fail(output):
        raise RuntimeError('unrelated fixture')
    with pytest.raises(RuntimeError):
        registry.register_run(runs, config, executor=fail)
    unrelated = registry.list_runs(runs)[0]
    unrelated['status'] = 'running'
    registry._write_record(runs / unrelated['run_id'], unrelated)
    unrelated_bytes = (runs / unrelated['run_id'] / 'run.json').read_bytes()
    code = '''
import os, signal, sys
from market_trader import experiment_registry as registry, experiment_jobs as jobs
from market_trader.experiment_job_worker import main
phase = sys.argv.pop(1)
original_transition = registry._transition
def transition(directory, record, status):
    if phase == 'before_pending' and status == 'pending':
        os.kill(os.getpid(), signal.SIGKILL)
    original_transition(directory, record, status)
    if status == phase:
        os.kill(os.getpid(), signal.SIGKILL)
registry._transition = transition
original_link = jobs._link_run
def link(*args):
    original_link(*args)
    if phase == 'allocated':
        os.kill(os.getpid(), signal.SIGKILL)
jobs._link_run = link
raise SystemExit(main())
'''
    original = subprocess.Popen
    children = []
    def launch(command, **kwargs):
        child = original([sys.executable, '-c', code, phase, *command[3:]], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(jobs.subprocess, 'Popen', launch)
    result = jobs.work_jobs(queue, total_seconds=15)
    assert children[0].returncode == -9
    record = result['jobs'][0]
    assert record['status'] == 'failed'
    assert record['run_id'] and record['run_id'] != unrelated['run_id']
    owned = runs / record['run_id']
    if phase in {'allocated', 'before_pending'}:
        # A claimed but unpublished empty directory is deliberately not a run.
        assert not (owned / 'run.json').exists()
        assert (owned / 'config.json').exists() is (phase == 'before_pending')
    else:
        assert registry.read_run(runs, record['run_id'])['status'] == 'failed'
    jobs.work_jobs(queue, total_seconds=2)  # Recovery/draining is idempotent.
    assert (runs / unrelated['run_id'] / 'run.json').read_bytes() == unrelated_bytes
    assert all(r['status'] not in {'pending', 'running'} for r in registry.list_runs(runs)
               if r['run_id'] != unrelated['run_id'])


def test_allocated_job_run_link_is_idempotent_and_cannot_be_reassigned(tmp_path):
    jobs = api()
    queue, runs = tmp_path / 'queue', tmp_path / 'runs'
    job = jobs.submit_job(queue, runs, payload(tmp_path))
    with jobs._lock(queue):
        jobs._transition(queue, job, 'running')
    runs.mkdir(mode=0o700)
    first, other = 'run-' + '1' * 32, 'run-' + '2' * 32
    for run_id in (first, other):
        (runs / run_id).mkdir(mode=0o700)
    jobs._link_run(queue, job['job_id'], first)
    events = (queue / job['job_id'] / 'events.jsonl').read_bytes()
    jobs._link_run(queue, job['job_id'], first)
    assert (queue / job['job_id'] / 'events.jsonl').read_bytes() == events
    with pytest.raises(ValueError, match='eligible'):
        jobs._link_run(queue, job['job_id'], other)
    assert jobs.read_job(queue, job['job_id'])['run_id'] == first
    result = jobs.work_jobs(queue)
    assert result['recovered'][0]['status'] == 'interrupted'
    assert jobs.registry_api.list_runs(runs) == []


@pytest.mark.parametrize('usage,observed,expected', [
    (600.0, (600, 600), True), (10.0, (600, 600), False),
    (600.0, None, False), (600.0, (600, 601), False),
])
def test_hard_cap_equal_soft_requires_actual_usage_and_observed_cap(usage, observed, expected):
    from types import SimpleNamespace
    process = SimpleNamespace(returncode=-9, child_cpu_limit=observed,
                              child_rusage=SimpleNamespace(ru_utime=usage, ru_stime=0.0))
    evidence = api()._termination_evidence(process, {'cpu_seconds': 600})
    assert evidence['cpu_hard_limit_seconds'] == 600
    assert evidence['cpu_limit_exhausted'] is expected
