"""AP7 report projections use synthetic metadata, never market-fit claims."""
import importlib.util
import json

import pytest


def cnn_summary():
    metric = dict(mae=0.0002, rmse=0.0003)
    trade = dict(trade_count=0, net_points='0', reference_points='0', modeled_cost_points='0', max_drawdown_points='0')
    return dict(config=dict(version=4, name='<img onerror=x>', hypothesis='<script>x</script>', seed=0,
                model=dict(kind='cnn', architecture='tiny_cnn_v1'), training=dict(lookback_bars=8, horizon_bars=2),
                strategy=dict(sma_fast_period=4, sma_slow_period=8, entry_threshold='0'),
                execution=dict(spread='2', slippage='.1', commission_rate='0', price_tick='.001', initial_capital='100000')),
            validation_forecast=dict(cnn=metric, ridge=metric, zero=metric, zero_return=metric, row_count=2,
                                     first_decision_at='2024-01-02', last_decision_at='2024-01-03'),
            trading=dict(cnn=trade, ridge=trade, sma_timed=trade),
            training=dict(best_epoch=1, stopping_reason='synthetic only', fit_settings={},
                          epoch_history=[dict(epoch=1, training_mse=1.0, validation_mse=1.1)]),
            comparison_signature=dict(interval_microseconds=900000000), training_data_id='synthetic image index',
            source_training_data_id='synthetic numerical training', model_report_id='synthetic',
            validation_dataset_id='synthetic', runtime={})


def test_v4_enrichment_explicit_cnn_representation_selection_and_no_tabular_claim():
    from market_trader.experiment_reporting import enrich_experiment_report
    html = enrich_experiment_report(b'<html><body><main>verified ledger</main></body></html>', summary=cnn_summary()).decode()
    for text in ('AP7', 'CNN', '5409', 'Flatten660', '256×192', '32×24', '8×8', '2024 is development/selection',
                 'not an independent final test', 'wick/detail', 'no additive', '2025', 'ap7-training-history',
                 'synthetic numerical training', 'verified ledger'):
        assert text in html
    assert 'MLP' not in html and 'Input and target scalers' not in html
    assert '<script>x</script>' not in html and '<img onerror=x>' not in html
    assert html.index('<main>') < html.index('id="ap7-experiment"') < html.index('</main>')


def test_zero_trade_formatter_is_honest_escaped_and_bounded():
    assert importlib.util.find_spec('market_trader.image_trade_audit') is not None, 'missing trade audit formatter'
    from market_trader.image_trade_audit import render_trade_image_audit
    audit = dict(format='market-trader.trade-image-audit.v1', model_report_id='<script>alert(1)</script>',
                 dataset_id='synthetic', dataset_artifact_id='synthetic', mode='decisions',
                 total_trades=0, total_decisions=0, audited_count=0, limit=8, samples=[])
    html = render_trade_image_audit(audit)
    assert 'No executed CNN trades' in html
    assert '0 of 0' in html
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
    assert 'href=' not in html and 'src=' not in html
    assert len(html.encode()) < 4096
