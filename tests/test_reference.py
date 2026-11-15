"""The AP1 entry point is a frozen reference protocol, not a tuning service."""

import subprocess
import sys
from importlib.util import find_spec

import pytest


def test_reference_config_is_versioned_and_resolves_source_relative_to_itself(tmp_path):
    assert find_spec("market_trader.reference") is not None, "AP1 config reader is missing"
    from market_trader.reference import read_reference_config

    config = tmp_path / "reference.toml"
    config.write_text('version = 1\nexperiment = "ap1-deuidxeur-open30-v1"\n'
                      'source_csv = "data.csv"\nsource_sha256 = "sha256:' + 'a'*64 + '"\n')
    settings = read_reference_config(config)
    assert settings.source_csv == tmp_path / "data.csv"
    assert settings.source_sha256 == "sha256:" + "a"*64
    for change in ('version = true', 'version = 2'):
        config.write_text(config.read_text().replace('version = 1', change))
        with pytest.raises(ValueError):
            read_reference_config(config)
        config.write_text(config.read_text().replace(change, 'version = 1'))
    config.write_text(config.read_text() + 'test_start = "2024-01-01"\n')
    with pytest.raises(ValueError, match="unexpected"):
        read_reference_config(config)


def test_reference_trading_summary_reconciles_points_and_months():
    import json
    from decimal import Decimal
    from tests.test_report_validation import report_body
    from tests.test_backtesting import LONG, EXIT
    from market_trader.backtesting import ExecutionProfile
    import market_trader.reference as reference

    assert hasattr(reference, "trading_summary"), "AP1 ledger summary is missing"
    body = report_body(LONG, EXIT, profile=ExecutionProfile(initial_capital=Decimal(100000),
                       spread=Decimal(2), slippage=Decimal("0.1")))
    body = json.loads(json.dumps(body).replace("2026-01-01", "2024-01-02"))
    summary = reference.trading_summary(body)
    assert Decimal(summary["net_points"]) == Decimal("7.8")
    assert Decimal(summary["reference_points"]) == 10
    assert Decimal(summary["modeled_cost_points"]) == Decimal("2.2")
    assert summary["trade_count"] == 1
    assert len(summary["months"]) == 12
    assert sum(month["trade_count"] for month in summary["months"]) == 1
    assert sum(Decimal(month["net_points"]) for month in summary["months"]) == Decimal("7.8")
    body["trades"][0]["net_pnl"] = "999"
    with pytest.raises(ValueError):
        reference.trading_summary(body)


def reference_fixture(tmp_path, *, heldout_shift=0):
    import csv
    from datetime import datetime, timedelta, UTC
    from decimal import Decimal
    from hashlib import sha256
    from zoneinfo import ZoneInfo

    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for day_index, day in enumerate(("2021-01-04", "2021-06-01", "2024-01-02", "2024-06-03", "2025-01-02")):
            start = datetime.fromisoformat(day + "T09:00:00").replace(tzinfo=ZoneInfo("Europe/Berlin"))
            for index in range(34):
                opened = Decimal(20000 + day_index*100 + (index*7)%13 + (heldout_shift if day.startswith("2025") else 0))
                closed = opened + Decimal(index%5 - 2)/10
                writer.writerow([(start + index*timedelta(minutes=15)).astimezone(UTC).isoformat(),
                                 opened, max(opened, closed)+1, min(opened, closed)-1, closed, 10])
    config = tmp_path / "ap1.toml"
    config.write_text('version = 1\nexperiment = "ap1-deuidxeur-open30-v1"\nsource_csv = "source.csv"\n'
                      'source_sha256 = "sha256:' + sha256(source.read_bytes()).hexdigest() + '"\n')
    return source, config


def test_reference_pipeline_is_reproducible_and_does_not_evaluate_reserved_test(tmp_path, monkeypatch):
    import json
    from hashlib import sha256
    import market_trader.reference as reference
    from market_trader import baseline

    assert hasattr(reference, "run_reference"), "AP1 reference runner is missing"
    monkeypatch.setattr(baseline, "held_out_evaluation", lambda *args: pytest.fail("AP1 evaluated reserved test"))
    source, config = reference_fixture(tmp_path)
    before = source.read_bytes()
    first, second = tmp_path/"first", tmp_path/"second"
    results = [reference.run_reference(config, out) for out in (first, second)]
    assert results[0] == results[1]
    assert results[0]["test_metrics_computed"] is False
    for path in first.iterdir():
        assert path.read_bytes() == (second/path.name).read_bytes(), path.name
    training = json.loads((first/"training-data.json").read_text())
    assert training["format"] == "market-trader.training-data.v2"
    assert training["splits"]["test"] == []
    assert len(training["splits"]["train"]) == 48
    assert len(training["splits"]["validation"]) == 48
    points = json.loads((first/"validation-predictions.json").read_text())
    assert len(points) == 48
    assert all(point["at"].startswith("2024-") for point in points)
    summary = json.loads((first/"summary.json").read_text())
    assert summary["test_metrics_computed"] is False
    assert summary["validation_rows"] == 48
    assert '2025 bleibt reserviert' in (first/"report.html").read_text()
    manifest = json.loads((first/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["artifacts"].items():
        assert "sha256:" + sha256((first/name).read_bytes()).hexdigest() == expected
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        reference.run_reference(config, first)


def test_reference_reserved_prices_cannot_change_fit_or_validation(tmp_path, monkeypatch):
    import json
    from datetime import datetime, UTC
    from market_trader import baseline, training_data
    from market_trader.reference import run_reference

    fits = []
    original_fit = baseline._fit_ridge
    original_features = training_data.candle_features
    def checked_fit(rows):
        assert all(row.decision_at < datetime(2024, 1, 1, tzinfo=UTC) for row in rows)
        assert all(row.target_available_at < datetime(2024, 1, 1, tzinfo=UTC) for row in rows)
        fits.append(len(rows))
        return original_fit(rows)
    def checked_features(history):
        assert all(bar.open_time < datetime(2025, 1, 1, tzinfo=UTC) for bar in history)
        return original_features(history)
    monkeypatch.setattr(baseline, "_fit_ridge", checked_fit)
    monkeypatch.setattr(training_data, "candle_features", checked_features)
    outputs = []
    for name, shift in (("original", 0), ("changed-reserved", 10000)):
        folder = tmp_path/name
        folder.mkdir()
        _, config = reference_fixture(folder, heldout_shift=shift)
        output = folder/"result"
        run_reference(config, output)
        outputs.append(output)
    first, second = outputs
    assert fits == [48, 48]
    assert json.loads((first/"ridge-model.json").read_text())["model"] == json.loads((second/"ridge-model.json").read_text())["model"]
    assert json.loads((first/"summary.json").read_text())["validation_forecast"] == json.loads((second/"summary.json").read_text())["validation_forecast"]
    for name in ("ridge", "sma"):
        assert json.loads((first/f"{name}-backtest.json").read_text())["trades"] == json.loads((second/f"{name}-backtest.json").read_text())["trades"]


def test_reference_cli_rejects_source_mismatch_without_publishing(tmp_path):
    source, config = reference_fixture(tmp_path)
    source.write_text(source.read_text().replace("20000", "20001"))
    output = tmp_path/"invalid"
    result = subprocess.run([sys.executable, "-m", "market_trader.cli", "reference", "--config", str(config),
                             "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_reference_cli_documents_its_offline_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "market_trader.cli", "reference", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout and "--output" in result.stdout
