"""Read-only, private-registry projections. No trainers or optional dependencies."""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import os
import stat
from threading import RLock
import re
from pathlib import Path
from urllib.parse import quote

from market_trader import experiment_registry as registry


def _fingerprint(root, directory):
    """Stat every run entry, not its bytes; ctime/inode defeat restored mtime.

    Include source identity and all ancestor identities, reject every symlink and
    special file. Tree inventory also catches new/unlisted artifacts on cache hits.
    Atime is deliberately excluded because our own verification reads change it.
    """
    result = []
    for path in (root, *root.parents):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('Unsafe registry directory')
        result.append((str(path), info.st_dev, info.st_ino, info.st_mode))
    def visit(path):
        info = path.lstat()
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError('Unsafe run entry')
        result.append((str(path), info.st_dev, info.st_ino, info.st_mode,
                       info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        if stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir()):
                visit(child)
    info = root.lstat()
    result.append(('source', info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns))
    visit(directory)
    return tuple(result)


def _safe(value):
    """Redact private paths/secrets while leaving untrusted text as JSON data."""
    if isinstance(value, dict):
        return {('[redacted]' if re.search(r'(^/|[A-Za-z]:[\\\\/])', key) else key):
                ('[redacted]' if re.search(r'password|secret|token|credential|api.?key|source_csv', key, re.I)
                 else _safe(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        if re.search(r'(?<![\w<])/(?!>)[^\s<>]+|[A-Za-z]:[\\/]|\w+://', value):
            return '[redacted]'
    return value


def _curves(report, limit):
    initial = _number(report['execution']['initial_capital'])
    if initial <= 0:
        raise ValueError('Initial capital must be positive')
    peak = initial
    equity, drawdown = [], []
    for account in report['accounts']:
        if account['at'] is not None:
            if not isinstance(account['at'], str):
                raise ValueError('Invalid account timestamp')
            datetime.fromisoformat(account['at'])
        value = _number(account['equity'])
        peak = max(peak, value)
        equity.append([account['at'], str(value - initial)])
        drawdown.append([account['at'], str(value - peak)])
    count = len(equity)
    indexes = set(range(count))
    if count > limit:
        indexes = {0, count - 1}
        for series in (equity, drawdown):
            indexes.add(min(range(count), key=lambda i: Decimal(series[i][1])))
            indexes.add(max(range(count), key=lambda i: Decimal(series[i][1])))
        buckets = (limit - len(indexes)) // 4
        for bucket in range(buckets):
            start = 1 + bucket * (count - 2) // buckets
            end = 1 + (bucket + 1) * (count - 2) // buckets
            for series in (equity, drawdown):
                indexes.add(min(range(start, end), key=lambda i: Decimal(series[i][1])))
                indexes.add(max(range(start, end), key=lambda i: Decimal(series[i][1])))
    ordered = sorted(indexes)
    return dict(equity=[equity[i] for i in ordered], drawdown=[drawdown[i] for i in ordered],
                source_count=count, sampled=count > limit)


_METRICS = ('mae_bp', 'rmse_bp', 'net_points', 'cost_points', 'drawdown_points', 'trade_count')


def _number(value, *, nonnegative=False):
    if type(value) not in (str, int, float):
        raise ValueError('Expected a finite result number')
    result = Decimal(str(value))
    if not result.is_finite() or (nonnegative and result < 0):
        raise ValueError('Expected a finite result number')
    return result


def _validate_summary(summary, kind):
    """Integrity proves identity, not the shape or numerical validity of results."""
    if not isinstance(summary.get('training', {}), dict):
        raise ValueError('Invalid training metadata')
    forecasts, trading = summary['validation_forecast'], summary['trading']
    if not isinstance(forecasts.get(kind), dict) or not isinstance(trading.get(kind), dict):
        raise ValueError('Missing primary model results')
    for values in forecasts.values():
        if isinstance(values, dict):
            for field in ('mae', 'rmse'):
                _number(values[field], nonnegative=True)
    for values in trading.values():
        if not isinstance(values, dict):
            raise ValueError('Invalid trading results')
        _number(values['net_points'])
        for field in ('modeled_cost_points', 'max_drawdown_points'):
            _number(values[field], nonnegative=True)
        if type(values['trade_count']) is not int or values['trade_count'] < 0:
            raise ValueError('Invalid trade count')


def _forecast(summary):
    return {name: {'mae_bp': str(Decimal(str(values['mae'])) * 10000),
                   'rmse_bp': str(Decimal(str(values['rmse'])) * 10000)}
            for name, values in summary.get('validation_forecast', {}).items()
            if isinstance(values, dict) and 'mae' in values and 'rmse' in values
            and not (name == 'zero' and 'zero_return' in summary.get('validation_forecast', {}))}


def _row(source, record, config, summary=None):
    run_id = record['run_id']
    key = source + ':' + run_id
    kind = config.get('model', {}).get('kind', 'ridge')
    complete = record['status'] == 'completed'
    summary = (record.get('summary', {}) if summary is None else summary) if complete else {}
    training_period = summary.get('training', {})
    validation_period = summary.get('validation_forecast', {})
    forecast = _forecast(summary).get(kind, {})
    trading = summary.get('trading', {}).get(kind, {})
    metrics = dict.fromkeys(_METRICS)
    if complete:
        metrics.update(forecast)
        metrics.update(net_points=trading.get('net_points'), cost_points=trading.get('modeled_cost_points'),
                       drawdown_points=trading.get('max_drawdown_points'), trade_count=trading.get('trade_count'))
    return dict(key=key, source=source, run_id=run_id, name=record.get('name'),
                hypothesis=record.get('hypothesis'), model_kind=kind, status=record['status'],
                created_at=record.get('created_at'), parent_id=record.get('parent_id'), selectable=complete,
                training_rows=summary.get('training_rows'), validation_rows=summary.get('validation_rows'),
                dataset_id=summary.get('validation_dataset_id'), source_sha256=config.get('source_sha256'),
                period=dict(training_start=training_period.get('first_decision_at'),
                            training_end=training_period.get('last_target_available_at'),
                            validation_start=validation_period.get('first_decision_at'),
                            validation_end=validation_period.get('last_decision_at'),
                            reserved_test_start=summary.get('reserved_test_start'),
                            reserved_test_end=summary.get('reserved_test_end')),
                metrics=metrics, report_url='/experiment-report?run=' + quote(key, safe='') if complete else None)


class UnknownRunError(ValueError):
    """A well-formed key does not exist (HTTP 404)."""


class InvalidRunError(ValueError):
    """A run is not valid and completed (HTTP 409)."""


_RUN_ID = re.compile(r'run-[0-9a-f]{32}')
_ALIAS = re.compile(r'[a-z][a-z0-9_]*')


class WorkbenchData:
    def __init__(self, registries: dict[str, Path], *, curve_limit: int = 800):
        if not isinstance(registries, dict) or any(
                not isinstance(source, str) or _ALIAS.fullmatch(source) is None for source in registries):
            raise ValueError('Invalid registry aliases')
        if type(curve_limit) is not int or not 6 <= curve_limit <= 800:
            raise ValueError('Curve limit must be an integer between 6 and 800')
        self._roots = {source: Path(root).absolute() for source, root in registries.items()}
        self._curve_limit = curve_limit
        self._lock = RLock()
        self._cache = OrderedDict()
        self._cache_limit = 64

    def _load(self, source, root, run_id):
        key = source + ':' + run_id
        with self._lock:
            try:
                fingerprint = _fingerprint(root, root / run_id)
                cached = self._cache.get(key)
                if cached is not None and cached['fingerprint'] == fingerprint:
                    self._cache.move_to_end(key)
                    return cached
                self._cache.pop(key, None)
                # Reject oversized HTML before the legacy verifier hashes its
                # bytes. Failed/pending runs need not have a report at all.
                try:
                    report_info = (root / run_id / 'artifacts/report.html').lstat()
                except FileNotFoundError:
                    pass
                else:
                    if not stat.S_ISREG(report_info.st_mode) or report_info.st_size > 32 * 1024 * 1024:
                        raise ValueError('Report is not a bounded regular file')
                record, config = registry._verified_record(root, run_id)
                projection = self._verified_projection(source, root, run_id, record, config)
                if _fingerprint(root, root / run_id) != fingerprint:
                    raise ValueError('Run changed during verification')
                entry = dict(fingerprint=fingerprint, record=record, config=config, projection=projection)
                self._cache[key] = entry
                while len(self._cache) > self._cache_limit:
                    self._cache.popitem(last=False)
                return entry
            except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
                self._cache.pop(key, None)
                raise

    def _resolve(self, key):
        if not isinstance(key, str) or key.count(':') != 1:
            raise ValueError('Expected source:run-<32 lowercase hex>')
        source, run_id = key.split(':')
        if _ALIAS.fullmatch(source) is None or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError('Expected source:run-<32 lowercase hex>')
        if source not in self._roots:
            raise UnknownRunError('Unknown run')
        root = self._roots[source]
        directory = root / run_id
        if not directory.exists() and not directory.is_symlink():
            raise UnknownRunError('Unknown run')
        return source, root, run_id

    @staticmethod
    def _empty(row, config=None, error=None):
        return dict(run=row, config=_safe(config or {}), signature=None, runtime={},
                    metrics_by_model={}, trading_by_model={}, training=None,
                    curves=dict(equity=[], drawdown=[], source_count=0, sampled=False),
                    warnings=[], error=error)

    def _project(self, source, root, run_id):
        try:
            return deepcopy(self._load(source, root, run_id)['projection'])
        except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
            row = _row(source, {'run_id': run_id, 'status': 'invalid'}, {})
            return self._empty(row, error='Run integrity or data validation failed.')

    def detail(self, key: str) -> dict:
        source, root, run_id = self._resolve(key)
        return self._project(source, root, run_id)

    def _verified_projection(self, source, root, run_id, record, config):
        if record.get('status') not in ('pending', 'running', 'failed', 'completed'):
            raise ValueError('Unknown run status')
        from market_trader.experiment_config import experiment_config_from_payload
        experiment_config_from_payload(config)
        output = root / run_id / 'artifacts'
        summary = registry._read_json(output / 'summary.json') if record['status'] == 'completed' else {}
        if record['status'] == 'completed':
            _validate_summary(summary, config.get('model', {}).get('kind', 'ridge'))
        row = _safe(_row(source, record, config, summary))
        row['report_url'] = '/experiment-report?run=' + quote(row['key'], safe='') if row['selectable'] else None
        if record['status'] != 'completed':
            return self._empty(row, config, 'Run failed.' if record['status'] == 'failed' else None)
        curves = _curves(registry._read_json(output / (row['model_kind'] + '-backtest.json')), self._curve_limit)
        warnings = []
        if curves['sampled']:
            warnings.append('Display curves sampled; full-source summary metrics are unchanged.')
        if any(point[0] is None for point in curves['equity']):
            warnings.append('Initial account snapshot has no timestamp (initial capital).')
        training = None
        if row['model_kind'] == 'mlp':
            history = registry._read_json(output / 'training-history.json')
            epochs = history['epoch_history']
            if not isinstance(epochs, list) or not epochs:
                raise ValueError('Missing actual epoch history')
            for index, epoch in enumerate(epochs, 1):
                if not isinstance(epoch, dict) or type(epoch.get('epoch')) is not int or epoch['epoch'] != index:
                    raise ValueError('Invalid epoch sequence')
                for field in ('training_mse', 'validation_mse'):
                    _number(epoch[field], nonnegative=True)
            if type(history['best_epoch']) is not int or not 1 <= history['best_epoch'] <= len(epochs):
                raise ValueError('Invalid best epoch')
            if not isinstance(history['stopping_reason'], str):
                raise ValueError('Invalid stopping reason')
            training = dict(history=[{field: epoch[field] for field in
                                     ('epoch', 'training_mse', 'validation_mse')} for epoch in epochs],
                            best_epoch=history['best_epoch'], stopping_reason=_safe(history['stopping_reason']))
        return dict(run=row, config=_safe(config), signature=_safe(summary.get('comparison_signature')),
                    runtime=_safe(summary.get('runtime', {})), metrics_by_model=_safe(_forecast(summary)),
                    trading_by_model=_safe(summary.get('trading', {})), training=training,
                    curves=curves, warnings=warnings, error=None)

    def report(self, key: str) -> bytes:
        source, root, run_id = self._resolve(key)
        with self._lock:
            try:
                entry = self._load(source, root, run_id)
                if not entry['projection']['run']['selectable']:
                    raise ValueError('Run is not completed')
                output = root / run_id / 'artifacts'
                manifest_bytes = registry._read_bytes(output / 'manifest.json')
                if registry._digest(manifest_bytes) != entry['record'].get('manifest_sha256'):
                    raise ValueError('Manifest changed')
                manifest = registry._decode(manifest_bytes)
                expected = manifest['artifacts']['report.html']
                path = registry._check_path(output / 'report.html')
                limit = 32 * 1024 * 1024
                with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                        raise ValueError('Report is not a bounded regular file')
                    data = stream.read(limit + 1)
                if len(data) > limit or registry._digest(data) != expected:
                    raise ValueError('Report changed')
                if _fingerprint(root, root / run_id) != entry['fingerprint']:
                    raise ValueError('Run changed')
                return data
            except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
                self._cache.pop(key, None)
                raise InvalidRunError('Verified report is unavailable') from None

    def compare(self, keys: list[str]) -> dict:
        if (type(keys) is not list or not 2 <= len(keys) <= 4
                or any(type(key) is not str for key in keys) or len(set(keys)) != len(keys)):
            raise ValueError('Select 2–4 unique completed runs')
        entries, resolved = [], []
        for key in keys:
            source, root, run_id = self._resolve(key)
            try:
                entry = self._load(source, root, run_id)
                if not entry['projection']['run']['selectable']:
                    raise ValueError('Run is not completed')
            except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
                raise InvalidRunError('Comparison requires completed valid runs') from None
            entries.append(entry)
            resolved.append((root, run_id))
        details = [deepcopy(entry['projection']) for entry in entries]
        originals = [(entry['record'], entry['config']) for entry in entries]
        conditions, parameters = [], []
        for i, (left, left_config) in enumerate(originals):
            for j in range(i + 1, len(keys)):
                right, right_config = originals[j]
                conditions.append(dict(left=keys[i], right=keys[j], differences=_safe(registry._differences(
                    left['summary']['comparison_signature'], right['summary']['comparison_signature']))))
                parameters.append(dict(left=keys[i], right=keys[j], differences=_safe([
                    diff for diff in registry._differences(left_config, right_config)
                    if diff['field'].split('.')[0] not in ('name', 'hypothesis')])))
        compatible = not any(pair['differences'] for pair in conditions)
        result = dict(compatible=compatible, runs=[detail['run'] for detail in details],
                      condition_differences=conditions, parameter_differences=parameters, details=details,
                      warnings=[] if compatible else ['Different conditions: these runs are not directly comparable.'])
        # Filesystem writers do not share our cache lock. Validate every captured
        # version after collection AND computation, without independently reloading.
        try:
            for entry, (root, run_id) in zip(entries, resolved):
                if _fingerprint(root, root / run_id) != entry['fingerprint']:
                    raise ValueError('Run changed during comparison')
        except (OSError, ValueError, TypeError, KeyError, ArithmeticError):
            raise InvalidRunError('Comparison requires completed valid runs') from None
        return result

    def overview(self) -> dict:
        rows, errors = [], []
        for source, root in self._roots.items():
            try:
                registry._check_path(root)
                entries = sorted(root.iterdir())
            except (OSError, ValueError):
                errors.append('Registry ' + source + ' is unavailable.')
                continue
            for path in entries:
                if _RUN_ID.fullmatch(path.name) is not None:
                    rows.append(self._project(source, root, path.name)['run'])
        return dict(runs=rows, sources=[{'id': source} for source in self._roots], errors=errors, read_only=True)
