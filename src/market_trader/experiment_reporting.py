"""Dynamic, escaped AP2 metadata around the verified existing ledger report."""
from decimal import Decimal, localcontext, Context, ROUND_HALF_EVEN
from html import escape
import json


def enrich_experiment_report(report: bytes, *, summary: dict) -> bytes:
    """Show the actual snapshot, forecast and reconciled trade metrics, not AP1 labels."""
    config = summary['config']
    if config['version'] in (2, 3):
        return _enrich_ap3_report(report, summary)
    minutes = summary['comparison_signature']['interval_microseconds'] // 60_000_000
    heading = f"AP2 — {config['name']}"
    metadata = (
        '<section id="ap2-experiment"><h1>' + escape(heading) + '</h1><p>'
        + escape(config['hypothesis']) + '</p><p>2025 bleibt reserviert; test metrics not computed. '
        'Training 2021–2023; validation 2024 only. Hypothetical research costs; '
        'synthetic Q, not EUR. Complete-observed-session selection may introduce bias.</p><p>'
        + escape(f"Lookback {config['training']['lookback_bars']} bars; horizon "
                 f"{config['training']['horizon_bars']} bars ({config['training']['horizon_bars'] * minutes} minutes); "
                 f"SMA {config['strategy']['sma_fast_period']}/{config['strategy']['sma_slow_period']}; "
                 'quantity 1; Ridge alpha 0.1. Bid source with symmetric spread offsets, not historical ask fills.')
        + '</p><h2>Configuration and execution costs</h2><pre>'
        + escape(json.dumps(config, ensure_ascii=True, sort_keys=True, indent=2))
        + '</pre><h2>Validation forecast</h2><pre>'
        + escape(json.dumps(summary['validation_forecast'], sort_keys=True, indent=2))
        + '</pre><h2>Reconciled trading (points per unit)</h2><pre>'
        + escape(json.dumps(summary['trading'], sort_keys=True, indent=2))
        + '</pre><h2>Reproduction identities</h2><pre>'
        + escape(json.dumps({key: summary[key] for key in
                             ('training_data_id', 'model_report_id', 'validation_dataset_id', 'runtime')},
                            sort_keys=True, indent=2)) + '</pre></section>'
    )
    html = report.decode('utf-8')
    if '<body>' not in html:
        raise ValueError('existing report has no body insertion point')
    return html.replace('<body>', '<body>' + metadata, 1).encode('utf-8')


def _table(identifier, headers, rows):
    """Escape every cell, including provenance and other user-supplied metadata."""
    return ('<div class="ap3-table-scroll"><table id="' + escape(identifier, quote=True) + '"><thead><tr>'
            + ''.join('<th>' + escape(str(value)) + '</th>' for value in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in row)
                      + '</tr>' for row in rows) + '</tbody></table></div>')


def _settings_table(identifier, settings):
    return _table(identifier, ('Parameter', 'Value'),
                  ((key, json.dumps(value, ensure_ascii=True, sort_keys=True)
                    if isinstance(value, (dict, list)) else value) for key, value in settings.items()))


def _bp(value):
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        return format(Decimal(str(value)) * 10000, '.4f')


def _enrich_ap3_report(report, summary):
    """Readable AP3 comparison; the embedded ledger remains the verified renderer's."""
    config = summary['config']
    forecast = summary['validation_forecast']
    neural = config['model']['kind'] == 'mlp'
    selection = ('2024 is development/selection data (early stopping), not an independent final test. '
                 if neural else '2024 validation is not the reserved independent final test. ')
    prediction_units = ('MLP predictions are inverse target-transformed before scoring or trading.'
                        if neural else 'Ridge predictions and errors use original return units.')
    minutes = summary['comparison_signature']['interval_microseconds'] // 60_000_000
    training, strategy, costs = (config[key] for key in ('training', 'strategy', 'execution'))
    stage = 'AP5' if config['version'] == 3 else 'AP3'
    metadata = ('<section class="panel" id="ap3-experiment"><h1>' + escape(stage + ' — ' + config['name'])
        + '</h1><p>' + escape(config['hypothesis'])
        + '</p><p>Training 2021–2023; validation 2024 only. ' + selection
        + '2025 bleibt reserviert; '
          'test metrics not computed. Hypothetical research costs; synthetic Q, not EUR. '
          'Complete-observed-session selection may introduce bias.</p><p>'
        + escape(f"Lookback {training['lookback_bars']} bars; horizon {training['horizon_bars']} bars "
                 f"({training['horizon_bars'] * minutes} minutes); "
                 f"SMA {strategy['sma_fast_period']}/{strategy['sma_slow_period']}; quantity 1; "
                 f"entry threshold {strategy['entry_threshold']}; spread {costs['spread']}; "
                 f"slippage per fill {costs['slippage']}; commission rate {costs['commission_rate']}; "
                 f"price tick {costs['price_tick']}; initial capital {costs['initial_capital']}.")
        + '</p><p>Bid source with symmetric spread offsets, not historical ask fills. '
          'One unit and contract multiplier one; point P&amp;L equals synthetic Q, not EUR.</p>'
          '<h2>Validation forecast — original return units, displayed in basis points</h2><p>'
        + escape(f"Common cohort: {forecast['row_count']} rows; first {forecast['first_decision_at']}; "
                 f"last {forecast['last_decision_at']}. 1 bp = 0.0001 return. "
                 + prediction_units) + '</p>')
    forecast_rows = []
    for key, label in (('mlp', 'MLP'), ('ridge', 'Ridge'), ('zero', 'Zero')):
        metrics = forecast.get(key, forecast.get('zero_return') if key == 'zero' else None)
        if metrics is not None:
            forecast_rows.append((label, _bp(metrics['mae']), _bp(metrics['rmse'])))
    metadata += _table('ap3-forecast', ('Model', 'MAE (bp)', 'RMSE (bp)'), forecast_rows)
    metadata += '<h2>Reconciled trading — points per unit</h2>'
    metadata += _table('ap3-trading',
        ('Strategy', 'Trades', 'Net points', 'Reference points', 'Modeled cost points', 'Max drawdown points'),
        ((label, *(summary['trading'][key][field] for field in
                   ('trade_count', 'net_points', 'reference_points', 'modeled_cost_points', 'max_drawdown_points')))
         for key, label in (('mlp', 'MLP'), ('ridge', 'Ridge'), ('sma_timed', 'SMA timed'))
         if key in summary['trading']))
    title = 'Model parameters' if config['version'] == 3 else 'Fixed model parameters'
    metadata += '<h2>' + title + '</h2>' + _settings_table('ap3-model-settings', config['model'])
    if config['model']['kind'] == 'mlp':
        fitted = summary['training']
        architecture = config['model']['layers'] if config['version'] == 3 else [4,16,8,1]
        parameters = sum((inputs + 1) * outputs for inputs, outputs in zip(architecture[:-1], architecture[1:], strict=True))
        description = 'MLP ' + ' → '.join(map(str, architecture)) + f'; {parameters} trainable parameters. '
        metadata += ('<p>' + escape(description) +
            'Input and target scalers fitted only on training rows. Ridge reference alpha 0.1.</p>'
            '<h2>Actual training and selection</h2><p>'
            + escape(f"Best epoch {fitted['best_epoch']}; epochs run {len(fitted['epoch_history'])}; "
                     f"stopping reason {fitted['stopping_reason']}; seed {config['seed']}.")
            + '</p>' + _settings_table('ap3-fit-settings', fitted['fit_settings'])
            + '<h3>Epoch history — standardized target MSE</h3>'
            + _table('ap3-training-history', ('Epoch', 'Training MSE', 'Validation MSE'),
                     ((row['epoch'], row['training_mse'], row['validation_mse'])
                      for row in fitted['epoch_history'])))
    metadata += ('<details><summary>Canonical configuration and reproduction identities</summary><pre>'
        + escape(json.dumps({'config': config, **{key: summary[key] for key in
            ('training_data_id', 'model_report_id', 'validation_dataset_id', 'runtime')}},
            ensure_ascii=True, sort_keys=True, indent=2)) + '</pre></details></section>')
    html = report.decode('utf-8')
    if '<body>' not in html:
        raise ValueError('existing report has no body insertion point')
    style = ('<style>#ap3-experiment h2,#ap3-experiment h3{margin-top:1.2rem}'
             '#ap3-experiment p{margin-top:.7rem}'
             '#ap3-experiment .ap3-table-scroll{overflow-x:auto;max-width:100%;margin-top:.7rem}'
             '#ap3-experiment table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}'
             '#ap3-experiment th,#ap3-experiment td{text-align:left;vertical-align:top;'
             'padding:.45rem .65rem;border-bottom:1px solid var(--border);white-space:nowrap}</style>')
    if '<main>' in html:
        html = html.replace('<main>', '<main>' + style + metadata, 1)
    else:
        html = html.replace('<body>', '<body><main>' + style + metadata + '</main>', 1)
    return html.encode('utf-8')
