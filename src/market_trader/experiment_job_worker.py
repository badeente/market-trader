"""Fixed internal experiment subprocess. Not an arbitrary command runner."""
import argparse
import ctypes
import os
from pathlib import Path
import resource
import signal
import sys


class JobTerminated(BaseException):
    """Cooperative controller cancellation/timeout; registry records failure."""


class JobCpuLimit(BaseException):
    """The kernel-enforced per-job CPU allowance was exhausted."""


def _parent_guard(parent_pid):
    if sys.platform != 'linux':
        raise RuntimeError('safe local workers require Linux parent-death signalling')
    libc = ctypes.CDLL(None, use_errno=True)
    # KILL, not TERM: even a stuck native fit must not outlive its controller.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'could not install parent-death signal')
    if os.getppid() != parent_pid:
        raise JobTerminated('controller exited before worker startup')


def _install_safety(budget, parent_pid):
    _parent_guard(parent_pid)
    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise JobTerminated('controller requested termination')
    def cpu_exhausted(signum, frame):
        signal.signal(signal.SIGXCPU, signal.SIG_IGN)
        raise JobCpuLimit('CPU budget exhausted')
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGXCPU, cpu_exhausted)
    cpu = budget['cpu_seconds']
    memory = budget['memory_mib'] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, min(cpu + 1, 600)))
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args(argv)
    parent_pid = int(os.environ['MARKET_TRADER_WORKER_PARENT'])
    _parent_guard(parent_pid)
    from . import experiment_jobs as jobs
    queue = jobs._root(args.queue)
    # Only an explicit local controller can launch this internal entry point.
    fd = int(os.environ['MARKET_TRADER_WORKER_FD'])
    actual = os.fstat(fd)
    expected = (queue / '.worker.lock').stat(follow_symlinks=False)
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('worker lock descriptor mismatch')
    record = jobs.read_job(queue, args.job_id)
    if record['status'] != 'running':
        raise ValueError('job is not running')
    config = jobs.experiment_config_from_payload(jobs._read(queue / args.job_id / 'config.json'))
    _install_safety(record['budget'], parent_pid)
    from .experiment_pipeline import execute_experiment

    def execute(output):
        return execute_experiment(config, output)

    try:
        jobs.registry_api.register_run(Path(record['registry']),
            jobs._read(queue / args.job_id / 'config.json'), executor=execute,
            parent_id=record['parent_id'],
            on_allocated=lambda directory: jobs._link_run(queue, args.job_id, directory.name))
    except JobCpuLimit:
        return 124
    except BaseException:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
