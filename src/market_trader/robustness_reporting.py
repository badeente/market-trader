"""Framework-free, development-only AP8A study rendering from supplied data.

This module does not load artifacts, run models, or infer attribution. Validation
is deliberately separate from rendering so invalid complete studies never render.
"""
from decimal import Context, Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from html import escape
import math

_ERRORS = ('mae', 'rmse', 'zero_mae', 'zero_rmse')
_COSTS = ('trade_count', 'net_points', 'reference_points', 'modeled_cost_points',
          'max_drawdown_points', 'max_drawdown_fraction', 'win_fraction')
_MAX_BYTES = 8 * 1024 * 1024


def _text(value):
    return escape('None (empty slice)' if value is None else str(value), quote=True)


def _table(caption, columns, rows, *, css_class=''):
    return ('<div class="table-scroll" role="region" tabindex="0" aria-label="' + _text(caption) + '">'
            '<table' + (' class="' + _text(css_class) + '"' if css_class else '') + '><caption>' + _text(caption) + '</caption><thead><tr>' +
            ''.join('<th scope="col">' + _text(c) + '</th>' for c in columns) +
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + _text(v) + '</td>' for v in row) +
            '</tr>' for row in rows) + '</tbody></table></div>')


def _object(value, fields):
    if type(value) is not dict or not set(fields) <= value.keys():
        raise ValueError('missing required object fields: ' + ', '.join(fields))
    return value


def _integer(value, minimum=0, maximum=50000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('invalid integer count or seed')
    return value


def _label(value):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ValueError('expected bounded nonempty text')
    try:
        value.encode('utf-8')
    except UnicodeError:
        raise ValueError('text must be valid UTF-8') from None
    return value


def _number(value) -> float:
    if type(value) not in (int, float):
        raise ValueError('expected finite numeric value, not bool')
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError('invalid numeric value') from None
    if not math.isfinite(number):
        raise ValueError('nonfinite metric')
    return number


def _cost_number(value) -> Decimal:
    # Existing trading_summary deliberately serializes Decimal points as strings.
    if type(value) not in (int, float, str):
        raise ValueError('expected finite cost number, not bool')
    if type(value) is str and (not value or len(value) > 128 or value != value.strip()):
        raise ValueError('invalid numeric string')
    try:
        number = Decimal(str(value))
    except (ValueError, InvalidOperation, OverflowError):
        raise ValueError('invalid numeric value') from None
    if not number.is_finite() or (number and not -308 <= number.adjusted() <= 308):
        raise ValueError('nonfinite or unbounded cost number')
    return number


def _metrics(value):
    _object(value, ('row_count',) + _ERRORS)
    count = _integer(value['row_count'])
    for key in _ERRORS:
        if count == 0:
            if value[key] is not None:
                raise ValueError('empty slices must have None metrics')
        elif _number(value[key]) < 0:
            raise ValueError('negative forecast error')
    if count and (value['rmse'] < value['mae'] or value['zero_rmse'] < value['zero_mae']):
        raise ValueError('RMSE cannot be smaller than MAE')
    return count


def _rows(value, keys, field='key'):
    if type(value) is not list or len(value) != len(keys):
        raise ValueError('invalid bounded calendar rows')
    for row, key in zip(value, keys):
        _object(row, (field,))
        if row[field] != key:
            raise ValueError('invalid calendar key/order')
    return value


def _slices(run):
    forecast = run['forecast']
    if _metrics(forecast) != run['validation_rows']:
        raise ValueError('forecast count must match validation rows')
    slices = _object(run['slices'], ('format','timezone','overall','months','quarters','sessions'))
    if slices['format'] != 'market-trader.ap8-slices.v1' or slices['timezone'] != 'Europe/Berlin':
        raise ValueError('invalid slice identity')
    _metrics(slices['overall'])
    if any(slices['overall'][k] != forecast[k] for k in ('row_count',) + _ERRORS):
        raise ValueError('overall slices and forecast metrics disagree')
    groups = (('months', [f'2024-{i:02}' for i in range(1,13)]),
              ('quarters', [f'2024-Q{i}' for i in range(1,5)]),
              ('sessions', ['09:00-12:00','12:00-15:00','15:00-17:30']))
    for group, keys in groups:
        rows = _rows(slices[group], keys)
        if sum(_metrics(row) for row in rows) != run['validation_rows']:
            raise ValueError('slice counts must reconcile')
    for i, quarter in enumerate(slices['quarters']):
        if sum(m['row_count'] for m in slices['months'][i*3:i*3+3]) != quarter['row_count']:
            raise ValueError('quarter and month counts disagree')


def _close(a, b):
    return abs(a-b) <= max(Decimal('1e-20'), max(abs(a),abs(b)) * Decimal('1e-12'))


def _cost_summary(summary):
    _object(summary, _COSTS + ('months',))
    count = _integer(summary['trade_count'])
    nums = {k:_cost_number(summary[k]) for k in _COSTS if k not in ('trade_count','win_fraction')}
    if any(nums[k] < 0 for k in ('modeled_cost_points','max_drawdown_points')):
        raise ValueError('negative costs/drawdown points')
    # Shared ledger fractions are signed (equity - running peak) / peak;
    # trading_summary preserves that nonpositive value, unlike point magnitude.
    if nums['max_drawdown_fraction'] > 0:
        raise ValueError('positive drawdown fraction')
    if count == 0:
        if summary['win_fraction'] is not None:
            raise ValueError('empty trade win fraction must be None')
    elif not 0 <= _cost_number(summary['win_fraction']) <= 1:
        raise ValueError('invalid win fraction')
    rows = _rows(summary['months'], [f'2024-{i:02}' for i in range(1,13)], 'month')
    totals = {k:Decimal(0) for k in ('net_points','reference_points','modeled_cost_points')}
    trades = 0
    for row in rows:
        _object(row, ('trade_count', *totals))
        row_count = _integer(row['trade_count'])
        trades += row_count
        values = {k:_cost_number(row[k]) for k in totals}
        if values['modeled_cost_points'] < 0 or not _close(values['reference_points']-values['modeled_cost_points'], values['net_points']):
            raise ValueError('monthly point totals do not reconcile')
        if row_count == 0 and any(values.values()):
            raise ValueError('empty month cannot contain trade points')
        for k in totals:
            totals[k] += values[k]
    if trades != count or any(not _close(totals[k], nums[k]) for k in totals):
        raise ValueError('trade and point totals must match monthly ledger summary')
    if not _close(nums['reference_points']-nums['modeled_cost_points'],nums['net_points']):
        raise ValueError('point totals do not reconcile')


def _history(run):
    training = _object(run['training'], ('epoch_history','best_epoch','stopping_reason'))
    history = training['epoch_history']
    patience = 5 if run['model_kind'] == 'cnn' else 10
    if type(history) is not list or not 1 <= len(history) <= 20:
        raise ValueError('epoch history must contain 1–20 actual epochs')
    if 'fit_settings' in training:
        settings = _object(training['fit_settings'], ('max_epochs','patience','min_delta'))
        if (_integer(settings['max_epochs'], 1) != 20 or
                _integer(settings['patience'], 1) != patience or
                _cost_number(settings['min_delta']) != Decimal('0.000001')):
            raise ValueError('history policy contradicts frozen family settings')
        if 'seed' in settings and _integer(settings['seed'], maximum=2) != run['seed']:
            raise ValueError('training seed mismatch')
    if 'row_count' in training and _integer(training['row_count'], 1, 1000000) != run['training_rows']:
        raise ValueError('training row count mismatch')
    losses, significant, stale = [], float('inf'), 0
    for index, item in enumerate(history, 1):
        _object(item, ('epoch','validation_mse'))
        if _integer(item['epoch'], 1, 20) != index:
            raise ValueError('epoch history must be sequential')
        aliases = [k for k in ('training_mse','train_mse') if k in item]
        if not aliases or any(_number(item[k]) < 0 for k in aliases):
            raise ValueError('missing/negative training MSE')
        if len(aliases) == 2 and item['training_mse'] != item['train_mse']:
            raise ValueError('contradictory training MSE aliases')
        loss = _number(item['validation_mse'])
        if loss < 0:
            raise ValueError('negative validation MSE')
        losses.append(loss)
        if loss < significant - 1e-6:
            significant, stale = loss, 0
        else:
            stale += 1
        if stale >= patience and index != len(history):
            raise ValueError('history continues past early stopping')
    if _integer(training['best_epoch'], 1, 20) != losses.index(min(losses)) + 1:
        raise ValueError('best checkpoint must be actual earliest minimum validation MSE')
    reason = 'early_stopping' if stale >= patience else 'max_epochs'
    if training['stopping_reason'] != reason or (reason == 'max_epochs' and len(history) != 20):
        raise ValueError('stopping reason must match actual history')


def _validate(study):
    _object(study, ('format','status','protocol','test_metrics_computed','reserved_test_accessed',
                    'period','new_fit_count','reference_reuse_count','runs','limitations'))
    for key, expected in (('format','market-trader.ap8-study.v1'),('status','complete'),
                          ('protocol','ap8a-development-robustness-v1')):
        if study[key] != expected:
            raise ValueError('invalid study ' + key)
    if study['test_metrics_computed'] is not False or study['reserved_test_accessed'] is not False:
        raise ValueError('reserved test must remain closed')
    if study['period'] != dict(train='2021-2023', validation='2024', reserved='2025'):
        raise ValueError('invalid development period')
    if _integer(study['new_fit_count']) != 4 or _integer(study['reference_reuse_count']) != 2:
        raise ValueError('expected four new fits and two references')
    runs = study['runs']
    if type(runs) is not list or len(runs) != 6:
        raise ValueError('complete study requires six runs')
    pairs = set()
    for run in runs:
        _object(run, ('model_kind','seed','reused','run_id','model_report_id','training_rows',
                      'validation_rows','training','forecast','slices','costs'))
        if 'status' in run and run['status'] != 'complete':
            raise ValueError('cannot render failed or incomplete run')
        kind, seed = run['model_kind'], _integer(run['seed'], maximum=2)
        if type(kind) is not str or kind not in ('cnn','mlp') or (kind,seed) in pairs:
            raise ValueError('expected six unique CNN/MLP seed pairs')
        pairs.add((kind,seed))
        if run['reused'] is not (seed == 0):
            raise ValueError('only seed zero references must be reused')
        _label(run['run_id']); _label(run['model_report_id'])
        _integer(run['training_rows'], 1, 1000000); _integer(run['validation_rows'], 1)
        _history(run)
        _slices(run)
        _object(run['costs'], ('half','baseline','double'))
        for scenario in ('half','baseline','double'):
            _cost_summary(run['costs'][scenario])
    if type(study['limitations']) is not list or len(study['limitations']) > 100:
        raise ValueError('invalid limitations')
    for item in study['limitations']:
        _label(item)


def _mean(values):
    # Decimal aggregation avoids overflow for valid finite floats near 1e308.
    return sum((Decimal(str(value)) for value in values), Decimal(0)) / len(values)


def _cost_ranges(runs):
    for scenario in ('half','baseline','double'):
        for key in _COSTS:
            values = [_cost_number(r['costs'][scenario][key]) for r in runs
                      if r['costs'][scenario][key] is not None]
            yield (scenario, key, len(values),
                   min(values) if values else 'None (no trades)',
                   max(values) if values else 'None (no trades)',
                   _mean(values) if values else 'None (no trades)')


def render_robustness_report(study: dict) -> bytes:
    """Return a bounded standalone HTML report; invalid studies raise ValueError."""
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        return _render(study)


def _render(study):
    _validate(study)
    parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AP8A development robustness</title><style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#e7eef5;font:16px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:1.5rem}h1,h2,h3,p,li,td,th,summary{overflow-wrap:anywhere}
section,details{margin:1rem 0;padding:1rem;border:1px solid #698195;border-radius:.4rem}
.table-scroll{display:block;min-width:0;max-width:100%;overflow-x:auto;margin:1rem 0}
.table-scroll table{border-collapse:collapse;min-width:55rem;width:100%}
.table-scroll th{min-width:7rem;white-space:normal;word-break:normal;overflow-wrap:normal}
.table-scroll td{white-space:nowrap;word-break:normal;overflow-wrap:normal}
.ap8-individual td:nth-child(3),.ap8-individual td:nth-child(4){min-width:15rem;max-width:20rem;white-space:normal;overflow-wrap:anywhere}
td,th{padding:.6rem;text-align:left;vertical-align:top;border-bottom:1px solid #526776}
caption{text-align:left;font-weight:bold;padding:.5rem}summary{cursor:pointer;font-weight:bold}
:focus-visible{outline:3px solid #ffd479;outline-offset:3px}.warning{border-left:5px solid #ffd479;padding:1rem}
@media(max-width:640px){main{padding:.6rem}section,details{padding:.6rem}td,th{padding:.4rem;font-size:.85rem}}
</style></head><body><main><h1>AP8A development robustness</h1>
<section aria-label="Research limitations" class="warning"><h2>Development-only interpretation</h2>
<p>All results are 2024 checkpoint-selected development diagnostics, NOT an independent 2025 or rolling test.
2025 remains closed: no test metrics computed and no reserved test accessed. Training: 2021-2023; validation: 2024.</p>
<p>2 reused references (seed 0), 4 new fits (seeds 1 and 2). Every seed is shown, including unfavorable results.
No automatic best-seed selection, deployment ranking, or significance claim from three seeds.</p>
<p>CNN: max 20 epochs, patience 5. MLP: max 20 epochs, patience 10. Both use min_delta 0.000001 for patience;
the saved checkpoint is the earliest actual minimum validation MSE, including sub-min_delta improvements.</p>
<p>CNN and MLP stopping policies differ; image and numeric representations are not equivalent.
RNG variation is distinct from cross-platform replay. Calendar/time slices are not a regime classifier.
Forecast errors do not establish profitability. Nonlinear models: no additive feature attribution.</p>
<p>Cost scenarios 0.5 / 1 / 2 scale spread, slippage and commission only using frozen forecasts and the existing ledger;
quantity, price tick and trading decisions remain fixed. Tick rounding need not scale proportionally.
These are supplied modeled trade totals, not inferred profit claims; calendar slices do not reset a trade engine.
The renderer checks summary consistency, not source integrity or actual ledger replay; those require upstream verification.</p></section>''']
    limitations = ''.join('<li>' + _text(x) + '</li>' for x in study['limitations']) or '<li>None declared; protocol limitations still apply.</li>'
    parts.append('<section><h2>Declared limitations</h2><ul>' + limitations + '</ul></section>')
    for family in ('cnn', 'mlp'):
        runs = sorted((r for r in study['runs'] if r['model_kind'] == family), key=lambda r:r['seed'])
        parts.append('<section><h2>' + family.upper() + ' — all seeds</h2>')
        parts.append(_table('Individual forecasts', ('seed','reference status','run ID','model report ID','training rows','validation rows') + _ERRORS,
                     ((r['seed'], 'reused' if r['reused'] else 'new fit',r['run_id'],r['model_report_id'],r['training_rows'],r['validation_rows'],
                       *(r['forecast'][k] for k in _ERRORS)) for r in runs), css_class='ap8-individual'))
        parts.append(_table('Descriptive range / mean (not seed selection)', ('metric','minimum','maximum','mean'),
                     ((k,min(values),max(values),_mean(values)) for k in _ERRORS for values in [[r['forecast'][k] for r in runs]])))
        parts.append(_table('Cost descriptive range / mean (not seed selection; None is not imputed)',
                            ('scenario','metric','available seeds of 3','minimum','maximum','mean'), _cost_ranges(runs)))
        for r in runs:
            parts.append('<details><summary>' + family.upper() + ' seed ' + str(r['seed']) + ' — ' + _text(r['run_id']) + '</summary>')
            t=r['training']
            parts.append('<h3>Actual training history</h3><p>Best checkpoint: ' + _text(t['best_epoch']) + '; stopping reason: ' + _text(t['stopping_reason']) + '</p>')
            parts.append(_table('Actual epoch losses (best checkpoint uses actual minimum validation MSE)', ('epoch','training MSE','validation MSE','checkpoint'),
                         ((h['epoch'],h.get('training_mse',h.get('train_mse')),h['validation_mse'], 'best' if h['epoch']==t['best_epoch'] else '') for h in t['epoch_history'])))
            parts.append(_table('Cost and trade totals', ('scenario','factor')+_COSTS,
                         ((s,f,*(r['costs'][s][k] if r['costs'][s][k] is not None else 'None (no trades)' for k in _COSTS)) for s,f in (('half',.5),('baseline',1),('double',2)))))
            for scenario in ('half','baseline','double'):
                parts.append(_table(scenario+' monthly trade totals (continuous ledger)', ('month','trade_count','net_points','reference_points','modeled_cost_points'),
                             ((m[k] for k in ('month','trade_count','net_points','reference_points','modeled_cost_points')) for m in r['costs'][scenario]['months'])))
            for group in ('months','quarters','sessions'):
                parts.append(_table('Calendar/time slices — '+group+' (Europe/Berlin)', ('key','row_count')+_ERRORS,
                             ((m['key'],m['row_count'],*(m[k] for k in _ERRORS)) for m in r['slices'][group])))
            parts.append('</details>')
        parts.append('</section>')
    parts.append('</main></body></html>')
    result = ''.join(parts).encode('utf-8')
    if len(result) > _MAX_BYTES:
        raise ValueError('report exceeds 8 MiB')
    return result
