"""AP3 comparison tables stay inside the existing readable report layout."""
from market_trader.experiment_reporting import enrich_experiment_report


def test_neural_summary_uses_main_panel_and_scrollable_tables():
    config = {'version': 2, 'name': 'synthetic layout', 'hypothesis': 'layout only', 'seed': 0,
              'model': {'kind': 'mlp'}, 'training': {'lookback_bars': 8, 'horizon_bars': 2},
              'strategy': {'sma_fast_period': 4, 'sma_slow_period': 8, 'entry_threshold': '0'},
              'execution': {'spread': '2', 'slippage': '.1', 'commission_rate': '0',
                            'price_tick': '.001', 'initial_capital': '100000'}}
    metrics = {'mae': 0.0, 'rmse': 0.0}
    trade = dict(trade_count=0, net_points='0', reference_points='0', modeled_cost_points='0', max_drawdown_points='0')
    summary = {'config': config, 'comparison_signature': {'interval_microseconds': 900000000},
               'validation_forecast': {'row_count': 1, 'first_decision_at': '2024-01-02',
                    'last_decision_at': '2024-01-02', 'mlp': metrics, 'ridge': metrics, 'zero': metrics},
               'trading': {'mlp': trade, 'ridge': trade, 'sma_timed': trade},
               'training': {'best_epoch': 1, 'stopping_reason': 'synthetic', 'fit_settings': {},
                    'epoch_history': [{'epoch': 1, 'training_mse': 0.0, 'validation_mse': 0.0}]},
               'training_data_id': 'synthetic', 'model_report_id': 'synthetic',
               'validation_dataset_id': 'synthetic', 'runtime': {}}
    original = b'<html><head></head><body><main><p>verified ledger</p></main></body></html>'
    html = enrich_experiment_report(original, summary=summary).decode()
    panel = '<section class="panel" id="ap3-experiment">'
    assert panel in html
    assert html.index('<main>') < html.index(panel) < html.index('</main>')
    assert html.count('class="ap3-table-scroll"') == html.count('<table ')
    assert '#ap3-experiment' in html and 'overflow-x:auto' in html
    assert 'verified ledger' in html
