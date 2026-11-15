"""Dynamic, escaped AP2 metadata around the verified existing ledger report."""
from html import escape
import json


def enrich_experiment_report(report: bytes, *, summary: dict) -> bytes:
    """Show the actual snapshot, forecast and reconciled trade metrics, not AP1 labels."""
    config = summary['config']
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
