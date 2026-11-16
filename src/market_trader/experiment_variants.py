"""Explicit, validated variants derived from verified immutable experiments."""
from copy import deepcopy
from pathlib import Path

from market_trader.experiment_config import experiment_config_from_payload, experiment_config_payload
from market_trader.experiment_registry import _verified_record


_FIELDS = {
    'seed': 'training',
    'model.layers': 'model',
    **{f'model.{field}': 'training' for field in
       ('learning_rate', 'weight_decay', 'batch_size', 'max_epochs', 'patience', 'min_delta')},
    'strategy.entry_threshold': 'trading_rule',
    **{f'training.{field}': 'conditions' for field in ('lookback_bars', 'horizon_bars')},
    **{f'strategy.{field}': 'conditions' for field in ('sma_fast_period', 'sma_slow_period')},
    **{f'execution.{field}': 'conditions' for field in
       ('spread', 'slippage', 'commission_rate', 'price_tick', 'initial_capital')},
}
_DECIMALS = {'strategy.entry_threshold', 'model.learning_rate', 'model.weight_decay', 'model.min_delta',
    *(f'execution.{field}' for field in
    ('spread', 'slippage', 'commission_rate', 'price_tick', 'initial_capital'))}


def derive_variant(registry: Path, run_id: str, changes: dict, *, name: str, hypothesis: str) -> dict:
    """Produce a new snapshot and provenance without executing or rewriting a run."""
    if type(changes) is not dict or not changes or any(type(key) is not str or key not in _FIELDS for key in changes):
        raise ValueError('Expected nonempty supported variant fields')
    record, original = _verified_record(Path(registry), run_id)
    experiment_config_from_payload(original)
    payload = deepcopy(original)
    model_changes = any(field.startswith('model.') for field in changes)
    mlp_window = (payload.get('model', {}).get('kind') == 'mlp'
                  and any(field.startswith('training.') or field.startswith('strategy.sma_') for field in changes))
    if model_changes or mlp_window:
        if payload.get('model', {}).get('kind') != 'mlp':
            raise ValueError('Neural training variants require an MLP parent')
        payload['version'] = 3
    payload['name'], payload['hypothesis'] = name, hypothesis
    differences = []
    for field, value in sorted(changes.items()):
        parts = field.split('.')
        table = payload if len(parts) == 1 else payload[parts[0]]
        key = parts[-1]
        if field in _DECIMALS and type(value) in (float, int):
            value = str(value)
        differences.append(dict(field=field, left=table[key], right=value, category=_FIELDS[field]))
        table[key] = value
    config = experiment_config_from_payload(payload)
    if all(diff['left'] == diff['right'] for diff in differences):
        raise ValueError('Variant must change at least one parameter; use replay for an unchanged configuration')
    return dict(format='market-trader.experiment-variant.v1', source_registry=str(Path(registry).absolute()),
                parent_id=run_id, parent_config_sha256=record['config_sha256'],
                config=experiment_config_payload(config), changes=differences)


def prepare_sweep(registry: Path, run_id: str, grid: dict, *, name: str,
                  hypothesis: str, fixed_changes: dict | None = None) -> dict:
    """Preflight a finite declarative grid; do not queue or execute anything."""
    from itertools import product
    from math import prod
    from market_trader.experiment_registry import _encode
    if (type(grid) is not dict or not 1 <= len(grid) <= 4
            or any(type(key) is not str or key not in _FIELDS or type(values) is not list
                   or not 1 <= len(values) <= 4 for key, values in grid.items())):
        raise ValueError('Grid requires 1–4 supported axes with finite lists')
    fixed_changes = {} if fixed_changes is None else fixed_changes
    if type(fixed_changes) is not dict or grid.keys() & fixed_changes.keys():
        raise ValueError('Fixed changes and grid axes must be distinct')
    if prod(map(len, grid.values())) > 4:
        raise ValueError('Sweep must contain at most four variants')
    fields = sorted(grid)
    variants, identities = [], set()
    for index, values in enumerate(product(*(grid[key] for key in fields)), 1):
        item = derive_variant(registry, run_id, {**fixed_changes, **dict(zip(fields, values, strict=True))},
                              name=f'{name} [{index}]', hypothesis=hypothesis)
        identity = {key: value for key, value in item['config'].items() if key not in ('name', 'hypothesis')}
        encoded = _encode(identity)
        if encoded in identities:
            raise ValueError('Sweep configurations must be unique')
        identities.add(encoded)
        variants.append(item)
    return dict(format='market-trader.sweep-plan.v1', source_registry=str(Path(registry).absolute()),
                parent_id=run_id, name=name, hypothesis=hypothesis, grid=deepcopy(grid),
                fixed_changes=deepcopy(fixed_changes), variants=variants,
                objective=dict(split='validation_2024', metric='rmse', direction='minimize'))


def read_document(path: Path) -> dict:
    """Read bounded, strict JSON from a regular local file, never a symlink."""
    import os
    import stat
    from market_trader.experiment_registry import _check_path, _decode
    path = _check_path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise ValueError('Plan must be a regular JSON file of at most 1 MiB')
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('Plan exceeds 1 MiB')
    try:
        body = _decode(raw)
    except RecursionError as error:
        raise ValueError('JSON nesting is too deep') from error
    if type(body) is not dict:
        raise ValueError('Plan must be an object')
    return body


def read_plan(path: Path, registry: Path) -> dict:
    """Read a bounded local plan and reconstruct it from the verified parent."""
    try:
        body = read_document(path)
        if body.get('format') == 'market-trader.experiment-variant.v1':
            differences = body['changes']
            if (type(differences) is not list or len(differences) > len(_FIELDS)
                    or any(type(diff) is not dict or type(diff.get('field')) is not str for diff in differences)
                    or len({diff['field'] for diff in differences}) != len(differences)):
                raise ValueError('Invalid plan changes')
            expected = derive_variant(registry, body['parent_id'],
                {diff['field']: diff['right'] for diff in differences},
                name=body['config']['name'], hypothesis=body['config']['hypothesis'])
        elif body.get('format') == 'market-trader.sweep-plan.v1':
            expected = prepare_sweep(registry, body['parent_id'], body['grid'],
                name=body['name'], hypothesis=body['hypothesis'], fixed_changes=body['fixed_changes'])
        else:
            raise ValueError('Unsupported plan format')
        from market_trader.experiment_registry import _encode
        if _encode(body) != _encode(expected):
            raise ValueError('Plan does not match the verified parent and declared changes')
        return expected
    except (KeyError, TypeError, RecursionError) as error:
        raise ValueError('Invalid experiment plan') from error


def summarize_runs(registry: Path, run_ids: list[str]) -> dict:
    """Compare captured completed results; select by development RMSE, never profit."""
    from decimal import Decimal, InvalidOperation
    from itertools import combinations
    from market_trader.experiment_registry import _differences
    if (type(run_ids) is not list or not 2 <= len(run_ids) <= 4
            or any(type(key) is not str for key in run_ids) or len(set(run_ids)) != len(run_ids)):
        raise ValueError('Select 2–4 unique completed run IDs')
    entries, rows = [], []
    for run_id in run_ids:
        record, config = _verified_record(registry, run_id)
        experiment_config_from_payload(config)
        if record['status'] != 'completed' or record['summary']['test_metrics_computed'] is not False:
            raise ValueError('Results require completed validation-only runs')
        kind = config.get('model', {}).get('kind', 'ridge')
        try:
            raw = record['summary']['validation_forecast'][kind]['rmse']
            if type(raw) not in (str, float, int):
                raise ValueError('Invalid RMSE type')
            score = Decimal(str(raw))
            if not score.is_finite() or score < 0:
                raise ValueError('Invalid RMSE range')
        except (KeyError, InvalidOperation) as error:
            raise ValueError('Missing finite validation RMSE') from error
        entries.append(record)
        rows.append(dict(run_id=run_id, model_kind=kind, validation_rmse=str(score),
                         trading=deepcopy(record['summary']['trading'][kind])))
    conditions = [dict(left=left['run_id'], right=right['run_id'],
                       differences=_differences(left['summary']['comparison_signature'], right['summary']['comparison_signature']))
                  for left, right in combinations(entries, 2)]
    compatible = not any(pair['differences'] for pair in conditions)
    for captured in entries:
        if _verified_record(registry, captured['run_id'])[0] != captured:
            raise ValueError('Run changed during result selection')
    ordered = sorted(rows, key=lambda row: Decimal(row['validation_rmse'])) if compatible else rows
    return dict(compatible=compatible, condition_differences=conditions, rows=ordered,
                best_run_id=ordered[0]['run_id'] if compatible else None,
                objective=dict(split='validation_2024', metric='rmse', direction='minimize', unit='return'),
                test_metrics_computed=False,
                warning='2024 is development/selection data, not an independent final test; lowest RMSE is not a profitability claim.')
