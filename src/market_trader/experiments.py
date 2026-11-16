"""Thin offline orchestration between experiment configuration and run registry."""
from pathlib import Path


def run_experiment(config_path: Path, registry: Path, *, parent_id: str | None = None) -> dict:
    from market_trader.experiment_config import read_experiment_config, experiment_config_payload
    from market_trader.experiment_pipeline import execute_experiment
    from market_trader.experiment_registry import register_run

    config = read_experiment_config(config_path)
    return register_run(registry, experiment_config_payload(config),
                        executor=lambda output: execute_experiment(config, output), parent_id=parent_id)


def replay_experiment(registry: Path, run_id: str) -> dict:
    from market_trader.experiment_config import experiment_config_from_payload, experiment_config_payload
    from market_trader.experiment_pipeline import execute_experiment
    from market_trader.experiment_registry import load_config_snapshot, register_run

    snapshot = load_config_snapshot(registry, run_id)
    config = experiment_config_from_payload(snapshot)
    if experiment_config_payload(config) != snapshot:
        raise ValueError('stored experiment configuration cannot be reproduced by this config version')
    return register_run(registry, snapshot, executor=lambda output: execute_experiment(config, output),
                        parent_id=run_id)


def dispatch_experiment(args) -> dict | list:
    from market_trader.experiment_registry import list_runs, read_run, compare_runs

    if args.experiment_action == 'job':
        from market_trader.experiment_control import dispatch_job
        return dispatch_job(args)
    if args.experiment_action in ('derive', 'sweep'):
        from market_trader.experiment_variants import derive_variant, prepare_sweep, read_document
        from market_trader.experiment_registry import _decode, _encode
        from market_trader.storage import publish_new_file
        changes = {}
        for assignment in args.changes:
            field, separator, value = assignment.partition('=')
            if not separator or field in changes:
                raise ValueError('Changes require unique FIELD=JSON assignments')
            changes[field] = _decode(value.encode('utf-8'))
        if args.experiment_action == 'derive':
            result = derive_variant(args.registry, args.run_id, changes, name=args.name, hypothesis=args.hypothesis)
        else:
            result = prepare_sweep(args.registry, args.run_id, read_document(args.grid),
                                   name=args.name, hypothesis=args.hypothesis, fixed_changes=changes)
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError('Variant output already exists')
        publish_new_file(args.output, _encode(result))
        return result
    if args.experiment_action == 'run':
        return run_experiment(args.config, args.registry, parent_id=args.parent)
    if args.experiment_action == 'replay':
        return replay_experiment(args.registry, args.run_id)
    if args.experiment_action == 'list':
        return list_runs(args.registry)
    if args.experiment_action == 'show':
        return read_run(args.registry, args.run_id)
    if args.experiment_action == 'compare':
        return compare_runs(args.registry, args.left, args.right)
    raise ValueError('unknown experiment operation')
