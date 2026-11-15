"""AP1 summaries must distinguish validation from reserved final testing."""
from importlib.util import find_spec
import json

import pytest


def test_reference_report_is_validation_only_and_escapes_metadata():
    assert find_spec("market_trader.reference_reporting") is not None, "AP1 report summary is missing"
    from market_trader.reference_reporting import enrich_reference_report
    from market_trader.reference import trading_summary
    from tests.test_report_validation import report_body
    from tests.test_backtesting import LONG, EXIT

    body = json.loads(json.dumps(report_body(LONG, EXIT)).replace("2026-01-01", "2024-01-02"))
    trade_metrics = trading_summary(body)
    marker = '</pre><script>alert(1)</script>'
    summary = {
        "training_rows": 2, "validation_rows": 1, "test_metrics_computed": False,
        "target_name": "forward_return_next_open_to_horizon_open",
        "validation_forecast": {"ridge": {"mae": 0.01, "rmse": 0.02},
                                "zero_return": {"mae": 0.02, "rmse": 0.03}},
        "trading": {"ridge": trade_metrics, "sma_timed": trade_metrics},
        "data_audit": {"notes": marker}, "model_report_id": "sha256:example",
    }
    original = b'<!doctype html><html><head><title>Market Trader</title></head><body><main><p>interactive report</p></main></body></html>'
    result = enrich_reference_report(original, summary=summary).decode()
    assert 'AP1' in result and 'Validierung 2024' in result
    assert '2025 bleibt reserviert' in result
    assert 'MAE' in result and 'RMSE' in result and 'Monat' in result
    assert 'interactive report' in result
    assert marker not in result
    assert '&lt;script&gt;' in result
    assert 'Keine bestätigten Brokerkonditionen' in result
    summary["test_metrics_computed"] = True
    with pytest.raises(ValueError, match="reserved"):
        enrich_reference_report(original, summary=summary)
