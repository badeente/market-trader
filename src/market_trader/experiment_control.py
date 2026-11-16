"""Local-only AP5 CLI orchestration. Never imported by HTTP mutation routes."""
from pathlib import Path


def dispatch_job(args):
    from market_trader.experiment_jobs import submit_job, list_jobs, read_job, cancel_job, work_jobs
    if args.job_action == 'list':
        return list_jobs(args.queue)
    if args.job_action == 'show':
        return read_job(args.queue, args.job_id)
    if args.job_action == 'cancel':
        return cancel_job(args.queue, args.job_id)
    if args.job_action == 'work':
        return work_jobs(args.queue, max_jobs=args.max_jobs, total_seconds=args.total_seconds)
    if args.job_action == 'submit':
        budget = dict(wall_seconds=args.wall_seconds, cpu_seconds=args.cpu_seconds, memory_mib=args.memory_mib)
        if args.replay_run is not None:
            from market_trader.experiment_registry import load_config_snapshot
            if args.parent is not None:
                raise ValueError('Replay parent is the selected original run; do not override --parent')
            return submit_job(args.queue, args.registry, load_config_snapshot(args.registry, args.replay_run),
                              parent_id=args.replay_run, budget=budget, metadata={'operation': 'replay'})
        if args.config is not None:
            from market_trader.experiment_config import read_experiment_config, experiment_config_payload
            payload = experiment_config_payload(read_experiment_config(args.config))
            return submit_job(args.queue, args.registry, payload, parent_id=args.parent, budget=budget)
        from market_trader.experiment_variants import read_plan
        plan = read_plan(args.plan, args.registry)
        if args.parent is not None:
            raise ValueError('Plan parent is immutable; do not override it with --parent')
        variants = plan['variants'] if plan['format'] == 'market-trader.sweep-plan.v1' else [plan]
        submissions = [dict(payload=item['config'], parent_id=item['parent_id'], budget=budget,
                            metadata=dict(origin_parent_id=item['parent_id'],
                                          origin_config_sha256=item['parent_config_sha256'],
                                          changes=item['changes'])) for item in variants]
        if len(submissions) == 1:
            return submit_job(args.queue, args.registry, **submissions[0])
        from market_trader.experiment_jobs import submit_jobs
        return submit_jobs(args.queue, args.registry, submissions)
    if args.job_action == 'results':
        from market_trader.experiment_variants import summarize_runs
        keys = args.job_id
        if not 2 <= len(keys) <= 4 or len(set(keys)) != len(keys):
            raise ValueError('Results require 2–4 unique completed jobs')
        jobs = [read_job(args.queue, key) for key in keys]
        if any(job['status'] != 'completed' or not job.get('run_id') for job in jobs):
            raise ValueError('Results require completed jobs; inspect list/show for failures and cancellation')
        registries = {str(Path(job['registry']).absolute()) for job in jobs}
        if len(registries) != 1:
            raise ValueError('Select jobs from the same private registry')
        result = summarize_runs(Path(next(iter(registries))), [job['run_id'] for job in jobs])
        result['job_ids'] = keys
        return result
    raise ValueError('Unknown local job operation')
