"""Private, explicitly drained local experiment queue (no service or command API).

The queue is trusted local state owned by the current Unix user, not a sandbox
against that same user changing Python code. No persisted PID is ever signalled.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import stat
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
from uuid import uuid4

from . import experiment_registry as registry_api
from .experiment_config import experiment_config_from_payload

_DEFAULT_BUDGET = {'wall_seconds': 300, 'cpu_seconds': 180, 'memory_mib': 4096}
_BOUNDS = {'wall_seconds': (1, 900), 'cpu_seconds': (1, 600), 'memory_mib': (256, 8192)}
_TERMINAL = {'completed', 'failed', 'cancelled', 'timed_out', 'interrupted'}
_IMMUTABLE = ('job_id', 'registry', 'parent_id', 'budget', 'metadata', 'config_sha256', 'created_at')
_MAX_RECORD = 131072


def _bounded(value, limit=65536):
    def check(item, depth=0):
        if depth > 12:
            raise ValueError('JSON nesting exceeds queue limit')
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError('JSON keys must be strings')
                check(child, depth + 1)
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
    check(value)
    encoded = registry_api._encode(value)
    if len(encoded) > limit:
        raise ValueError('JSON exceeds queue size limit')
    return encoded


def _budget(value):
    if value is None:
        value = {}
    if type(value) is not dict or value.keys() - _BOUNDS.keys():
        raise ValueError('unknown budget fields')
    result = dict(_DEFAULT_BUDGET, **value)
    for key, (low, high) in _BOUNDS.items():
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'{key} must be an integer in {low}..{high}')
    return result


def _path(path):
    return registry_api._check_path(Path(path))


def _private(path):
    path = _path(path)
    info = path.stat(follow_symlinks=False)
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError(f'queue state must be private and owned by current user: {path}')
    return path


def _root(queue, create=False):
    queue = _path(queue)
    if create:
        queue.mkdir(mode=0o700, parents=True, exist_ok=True)
    if queue.exists():
        _private(queue)
        if not queue.is_dir():
            raise ValueError('queue must be a directory')
    return queue


def _read(path):
    path = _private(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECORD:
            raise ValueError('invalid or oversized queue record')
        data = stream.read(_MAX_RECORD + 1)
    if len(data) > _MAX_RECORD:
        raise ValueError('oversized queue record')
    result = registry_api._decode(data)
    if _bounded(result, _MAX_RECORD) != data:
        raise ValueError('queue JSON must use canonical bytes')
    return result


def _write(path, value, *, exclusive=False):
    path = _path(path)
    data = _bounded(value, _MAX_RECORD)
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix='.record-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _lock(queue, name='.queue.lock', *, nonblocking=False):
    path = _path(queue / name)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('unsafe queue lock')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError:
            raise ValueError('a worker is already active for this queue') from None
        yield fd
    finally:
        # Do not LOCK_UN: a subprocess may retain this flock on parent death.
        os.close(fd)


def _identifier(job_id):
    if type(job_id) is not str or re.fullmatch(r'job-[0-9a-f]{32}', job_id) is None:
        raise ValueError('invalid job identifier')
    return job_id


def _read_job(queue, job_id):
    directory = _private(queue / _identifier(job_id))
    record = _read(directory / 'job.json')
    submitted = _read(directory / 'submission.json')
    config = _read(directory / 'config.json')
    if type(record) is not dict or type(submitted) is not dict:
        raise ValueError('job record must be a JSON object')
    if (record.get('job_id') != job_id
            or _bounded({key: record.get(key) for key in _IMMUTABLE}) != _bounded(submitted)):
        raise ValueError('job immutable submission mismatch')
    if registry_api._digest(_bounded(config)) != record.get('config_sha256'):
        raise ValueError('config snapshot digest mismatch')
    experiment_config_from_payload(config)
    if record.get('status') not in _TERMINAL | {'queued', 'running'}:
        raise ValueError('unknown job status')
    if _budget(record.get('budget')) != record['budget']:
        raise ValueError('invalid stored budget')
    if record['status'] == 'completed':
        if not record.get('run_id'):
            raise ValueError('completed job must link a verified completed run')
        run = registry_api.read_run(_path(record['registry']), record['run_id'])
        if (run['status'] != 'completed' or run['config_sha256'] != record['config_sha256']
                or run['parent_id'] != record['parent_id']):
            raise ValueError('completed job does not match its committed run')
    return record


def _list(queue):
    records = []
    for directory in sorted(queue.iterdir()):
        if re.fullmatch(r'job-[0-9a-f]{32}', directory.name):
            _private(directory)
            if not (directory / 'job.json').exists():
                _path(directory / 'job.json')
                continue
            records.append(_read_job(queue, directory.name))
    return sorted(records, key=lambda r: (r['created_at'], r['job_id']))


def _prepare(registry, submission):
    if type(submission) is not dict or submission.keys() - {'payload', 'parent_id', 'budget', 'metadata'}:
        raise ValueError('invalid submission fields')
    snapshot = _bounded(submission.get('payload'))
    detached = registry_api._decode(snapshot)
    experiment_config_from_payload(detached)
    bounds = _budget(submission.get('budget'))
    metadata = submission.get('metadata')
    if metadata is None:
        metadata = {}
    if type(metadata) is not dict:
        raise ValueError('metadata must be a JSON object')
    metadata = registry_api._decode(_bounded(metadata, 8192))
    parent_id = submission.get('parent_id')
    if parent_id is not None:
        registry_api.read_run(registry, parent_id)
    return detached, {'registry': str(registry), 'parent_id': parent_id,
                      'budget': bounds, 'metadata': metadata,
                      'config_sha256': registry_api._digest(snapshot)}


def _allocate(queue, prepared, allocated):
    detached, fields = prepared
    job_id = 'job-' + uuid4().hex
    directory = queue / job_id
    directory.mkdir(mode=0o700)
    allocated.append(directory)
    immutable = dict(fields, job_id=job_id, created_at=registry_api._now())
    _write(directory / 'config.json', detached, exclusive=True)
    _write(directory / 'submission.json', immutable, exclusive=True)
    record = dict(immutable, status='queued', run_id=None, started_at=None,
                  finished_at=None, cancel_requested=False, error=None)
    _write(directory / 'job.json', record, exclusive=True)
    _event(directory, 'submitted')
    return record


def submit_job(queue: Path, registry: Path, payload: dict, *, parent_id: str | None = None,
               budget: dict | None = None, metadata: dict | None = None) -> dict:
    """Validate completely before allocating one immutable queued snapshot."""
    return submit_jobs(queue, registry, [{'payload': payload, 'parent_id': parent_id,
                        'budget': budget, 'metadata': metadata}])[0]


def submit_jobs(queue: Path, registry: Path, submissions: list[dict]) -> list[dict]:
    """Atomically preflight and enqueue a batch of one to four local variants."""
    if type(submissions) is not list or not 1 <= len(submissions) <= 4:
        raise ValueError('submission batch must contain 1..4 jobs')
    registry = _path(registry)
    prepared = [_prepare(registry, item) for item in submissions]
    queue = _root(queue, create=True)
    with _lock(queue):
        if sum(r['status'] == 'queued' for r in _list(queue)) + len(prepared) > 32:
            raise ValueError('queue capacity is 32 queued jobs')
        allocated = []
        try:
            return [_allocate(queue, item, allocated) for item in prepared]
        except BaseException as error:
            for directory in reversed(allocated):
                try:
                    shutil.rmtree(_path(directory))
                except BaseException as rollback_error:
                    error.add_note(f'batch rollback failed for {directory.name}: {rollback_error}')
            raise


def _event(directory, event, **details):
    path = _path(directory / 'events.jsonl')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, 'ab') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_size > 65536 or info.st_mode & 0o077
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise ValueError('unsafe or oversized event log')
        stream.write(_bounded({'at': registry_api._now(), 'event': event, **details}, 4096))
        stream.flush()
        os.fsync(stream.fileno())


def cancel_job(queue: Path, job_id: str) -> dict:
    """Cancel queued work atomically; request running cancellation without signals."""
    _identifier(job_id)
    queue = _root(queue)
    with _lock(queue):
        record = _read_job(queue, job_id)
        if record['status'] not in _TERMINAL and not record['cancel_requested']:
            record['cancel_requested'] = True
            directory = queue / job_id
            _write(directory / 'job.json', record)
            _event(directory, 'cancel_requested')
        if record['status'] == 'queued':
            _transition(queue, record, 'cancelled')
        return record


def list_jobs(queue: Path) -> list[dict]:
    queue = _root(queue)
    if not queue.exists():
        return []
    with _lock(queue):
        return _list(queue)


def _transition(queue, record, status, error=None):
    directory = queue / record['job_id']
    record['status'] = status
    if status == 'running':
        record['started_at'] = registry_api._now()
    if status in _TERMINAL:
        record['finished_at'] = registry_api._now()
    record['error'] = error
    _write(directory / 'job.json', record)
    _event(directory, status)
    return record


def _link_run(queue, job_id, run_id):
    if type(run_id) is not str or re.fullmatch(r'run-[0-9a-f]{32}', run_id) is None:
        raise ValueError('invalid run identifier')
    with _lock(queue):
        record = _read_job(queue, job_id)
        if record['status'] != 'running' or record['run_id'] not in (None, run_id):
            raise ValueError('job is not eligible to link a run')
        directory = _private(Path(record['registry']) / run_id)
        if not directory.is_dir():
            raise ValueError('allocated run must be a directory')
        if _path(directory / 'run.json').exists():
            run = registry_api.read_run(Path(record['registry']), run_id)
            if run['config_sha256'] != record['config_sha256'] or run['parent_id'] != record['parent_id']:
                raise ValueError('linked run does not belong to job submission')
        if record['run_id'] == run_id:
            return
        record['run_id'] = run_id
        _write(queue / job_id / 'job.json', record)
        _event(queue / job_id, 'run_linked', run_id=run_id)


def _poll_owned(process):
    """Reap only this Popen child, retaining its kernel CPU accounting.

    Never mix Popen.poll/wait with wait4: either may consume the one exit
    status. Keep Popen's lock/returncode so destructor and ownership semantics
    still work. No RUSAGE_CHILDREN deltas (which include unrelated children).
    """
    if process.returncode is not None:
        return process.returncode
    with process._waitpid_lock:
        if process.returncode is None:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                process.child_rusage = usage
                process.returncode = os.waitstatus_to_exitcode(status)
            else:
                try:
                    process.child_cpu_limit = resource.prlimit(process.pid, resource.RLIMIT_CPU)
                except ProcessLookupError:
                    pass  # Exited since wait4; the next poll will reap it.
    return process.returncode


def _wait_owned(process, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while _poll_owned(process) is None:
        if timeout is not None and deadline is not None and time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired('owned worker', timeout)
        time.sleep(0.01)
    return process.returncode


def _termination_evidence(process, budget):
    usage = getattr(process, 'child_rusage', None)
    hard = min(budget['cpu_seconds'] + 1, 600)
    user = usage.ru_utime if usage is not None else None
    system = usage.ru_stime if usage is not None else None
    observed = getattr(process, 'child_cpu_limit', None)
    # Linux CPU timers can fire just below the integral hard limit due to
    # accounting granularity; allow one accounting tick, not wall-clock time.
    # Require SIGKILL, an observed kernel cap, and hard-limit CPU usage;
    # a generic kill/OOM below that boundary remains an ordinary failure.
    tolerance = 1 / os.sysconf('SC_CLK_TCK')
    exhausted = (process.returncode in (124, -signal.SIGXCPU)
                 or (process.returncode == -signal.SIGKILL and usage is not None
                     # Linux advances the soft cap after sending SIGXCPU.
                     and observed is not None and observed[1] == hard
                     and budget['cpu_seconds'] <= observed[0] <= hard
                     and usage.ru_utime + usage.ru_stime >= hard - tolerance))
    return {'returncode': process.returncode, 'user_cpu_seconds': user,
            'system_cpu_seconds': system, 'cpu_hard_limit_seconds': hard,
            'observed_cpu_limit_seconds': list(observed) if observed is not None else None,
            'cpu_accounting_tolerance_seconds': tolerance,
            'cpu_limit_exhausted': exhausted}


def _stop_owned(process):
    """Signal only the unreaped child group created in this invocation."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        _wait_owned(process, timeout=0.5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_owned(process)


def _settle_run(record, reason):
    """Verify a committed result or fail this job's unfinished linked run."""
    if not record['run_id']:
        return False
    root = _path(record['registry'])
    if re.fullmatch(r'run-[0-9a-f]{32}', record['run_id']) is None:
        raise ValueError('invalid linked run identifier')
    directory = _private(root / record['run_id'])
    if not _path(directory / 'run.json').exists():
        # Ownership was persisted before the first atomic record. Nothing was
        # published, so leave this exact claimed directory inert, never invent
        # a completed run and never search other runs by config hash.
        return False
    run = registry_api.read_run(root, record['run_id'])
    if run['config_sha256'] != record['config_sha256'] or run['parent_id'] != record['parent_id']:
        raise ValueError('linked run does not belong to job submission')
    if run['status'] == 'completed':
        return True
    if run['status'] in {'pending', 'running'}:
        run['status'] = 'failed'
        run['finished_at'] = registry_api._now()
        run['error'] = {'type': 'JobInterrupted', 'message': reason[:1024]}
        _write(root / record['run_id'] / 'run.json', run)
    return False


def work_jobs(queue: Path, *, max_jobs: int = 1, total_seconds: int = 900) -> dict:
    """Drain a finite queue with one fixed Python subprocess at a time."""
    if type(max_jobs) is not int or not 1 <= max_jobs <= 32:
        raise ValueError('max_jobs must be an integer in 1..32')
    if type(total_seconds) is not int or not 1 <= total_seconds <= 3600:
        raise ValueError('total_seconds must be an integer in 1..3600')
    queue = _root(queue, create=True)
    start = time.monotonic()
    results = []
    recovered = []
    with _lock(queue, '.worker.lock', nonblocking=True) as worker_fd:
        # Children inherit this flock until exit, so acquiring it proves there
        # is no surviving old worker. Never trust/signals PIDs read from disk.
        with _lock(queue):
            for stale in _list(queue):
                if stale['status'] == 'running':
                    completed = _settle_run(stale, 'controller interrupted; no automatic retry')
                    recovered.append(_transition(queue, stale,
                        'completed' if completed else 'interrupted',
                        None if completed else 'previous controller exited before recording an outcome'))
        while len(results) < max_jobs and time.monotonic() - start < total_seconds:
            with _lock(queue):
                queued = [r for r in _list(queue) if r['status'] == 'queued']
                if not queued:
                    break
                if queued[0]['cancel_requested']:
                    results.append(_transition(queue, queued[0], 'cancelled'))
                    continue
                record = _transition(queue, queued[0], 'running')
            env = os.environ.copy()
            env['MARKET_TRADER_WORKER_FD'] = str(worker_fd)
            env['MARKET_TRADER_WORKER_PARENT'] = str(os.getpid())
            command = [sys.executable, '-m', 'market_trader.experiment_job_worker',
                       '--queue', str(queue), '--job-id', record['job_id']]
            process = None
            termination = None
            outcome = 'failed'
            deadline = min(start + total_seconds, time.monotonic() + record['budget']['wall_seconds'])
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True, close_fds=True, pass_fds=(worker_fd,), env=env)
                while _poll_owned(process) is None:
                    current = read_job(queue, record['job_id'])
                    if current['cancel_requested']:
                        outcome = 'cancelled'
                        _stop_owned(process)
                        break
                    if time.monotonic() >= deadline:
                        outcome = 'timed_out'
                        _stop_owned(process)
                        break
                    time.sleep(0.025)
                code = _wait_owned(process)
                termination = _termination_evidence(process, record['budget'])
                if outcome == 'failed' and termination['cpu_limit_exhausted']:
                    outcome = 'timed_out'
                error = None if code == 0 else f'worker exited with status {code}'
            except OSError as exc:
                error = str(exc)[:1024]
            finally:
                if process is not None and process.returncode is None:
                    _stop_owned(process)
            with _lock(queue):
                record = _read_job(queue, record['job_id'])
                if termination is not None:
                    record['termination'] = termination
                completed = _settle_run(record, outcome)
                results.append(_transition(queue, record, 'completed' if completed else outcome,
                                           None if completed else error))
    return {'processed': len(results), 'jobs': results, 'recovered': recovered,
            'elapsed_seconds': time.monotonic() - start}


def read_job(queue: Path, job_id: str) -> dict:
    _identifier(job_id)
    queue = _root(queue)
    with _lock(queue):
        return _read_job(queue, job_id)
