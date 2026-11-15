import json
import re
from decimal import Decimal

import pytest

from market_trader.artifacts import DatasetProvenance
from market_trader.baseline import baseline_report, held_out_report
from market_trader.storage import save_dataset
from tests.test_baseline import export, saved_model, write_export, write_model_report
from tests.test_data_cli import BACKTEST, BACKTEST_SMA, ROOT, command
from tests.test_model_strategy import at, bars, dataset


def saved_run(tmp_path, config):
    run = command(ROOT, "backtest", "--config", config)
    assert run.returncode == 0, run.stderr
    path = tmp_path / f"{config.stem}.json"
    path.write_text(run.stdout, encoding="utf-8")
    return path


def report_data(output):
    html = output.read_text(encoding="utf-8")
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html)
    assert match, html[:200]
    return html, json.loads(match[1])


@pytest.mark.case("REP-006")
@pytest.mark.case("REP-007")
def test_report_replays_saved_runs_and_preserves_inputs(tmp_path):
    first = saved_run(tmp_path, BACKTEST)
    second = saved_run(tmp_path, BACKTEST_SMA)
    before = first.read_bytes(), second.read_bytes()
    one, two = tmp_path / "one.html", tmp_path / "two.html"
    args = ("report", "--backtest", first, "--compare", second, "--output")
    for cwd, path in ((ROOT, one), (tmp_path, two)):
        result = command(cwd, *args, path)
        assert result.returncode == 0, result.stderr
    assert one.read_bytes() == two.read_bytes()
    html, report = report_data(one)
    assert "<!doctype html>" in html
    assert 'src="http' not in html
    single, sma = report["runs"]
    assert single["name"] == "single_long"
    assert sma["name"] == "sma_crossover"
    assert Decimal(single["trades"][0]["net_pnl"]) == Decimal("13.185")
    assert Decimal(single["trades"][0]["entry"]["fee"]) == Decimal("0.3006")
    assert single["drawdown"][-1] <= 0
    assert report["forecast"] is None
    existing = command(ROOT, *args, one)
    assert existing.returncode == 1
    assert "save HTML report" in existing.stderr
    assert one.read_bytes() == two.read_bytes()
    assert (first.read_bytes(), second.read_bytes()) == before


@pytest.mark.case("REP-007")
@pytest.mark.case("REP-009")
@pytest.mark.parametrize("problem", ["format", "costs", "account", "model"])
def test_report_rejects_mismatched_inputs_without_creating_html(tmp_path, problem):
    first = saved_run(tmp_path, BACKTEST)
    edited = json.loads(first.read_bytes())
    if problem == "format":
        edited["format"] = "unknown"
    elif problem == "costs":
        edited["execution"]["spread"] = "20"
    elif problem == "account":
        edited["accounts"][0]["at"] = None
    else:
        edited["strategy"]["model_report_id"] = "unrelated"
    bad = tmp_path / "edited.json"
    bad.write_text(json.dumps(edited), encoding="utf-8")
    output = tmp_path / "report.html"
    args = ["--backtest", first, "--compare", bad]
    if problem == "model":
        args = ["--backtest", first, "--evaluation", bad]
    result = command(ROOT, "report", *args, "--output", output)
    assert result.returncode == 1
    assert "render HTML report" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.case("REP-009")
def test_report_escapes_data_and_supports_a_run_without_trades(tmp_path):
    from tests.test_report_validation import report_body, save_report

    report = report_body()
    marker = "</script><script>alert(1)</script>"
    report["strategy"]["name"] = marker
    source = save_report(tmp_path, report)
    output = tmp_path / "report.html"
    result = command(ROOT, "report", "--backtest", source, "--output", output)
    assert result.returncode == 0, result.stderr
    html, data = report_data(output)
    assert marker not in html
    assert data["runs"][0]["name"] == marker
    assert data["runs"][0]["trades"] == []


@pytest.mark.case("REP-008")
def test_report_adds_forecast_and_trade_inputs(tmp_path, export, saved_model):
    export_path, body = export
    for row, feature in zip(body["splits"]["train"], ("-1", "1"), strict=True):
        row["features"][0] = feature
    write_export(export_path, body)
    write_model_report(saved_model, baseline_report(export_path))
    model_report = json.loads(saved_model.read_bytes())
    stored = tmp_path / "dataset.json"
    save_dataset(stored, dataset(bars([(100, 100), (100, 102), (103, 103), (103, 102), (106, 102)])),
                 provenance=DatasetProvenance("report fixture"))
    config = tmp_path / "ridge.toml"
    config.write_text(
        'version = 1\ndataset_artifact = "dataset.json"\n'
        '[strategy]\nname = "ridge_long"\nmodel_report = "ridge-validation.json"\n'
        'entry_threshold = 0.01\nquantity = 2\n'
        '[execution]\nspread = 0.2\nslippage = 0.1\ncommission_rate = 0.001\n',
        encoding="utf-8",
    )
    source = saved_run(tmp_path, config)
    evaluation = tmp_path / "evaluation.json"
    score = held_out_report(saved_model, export[0])
    evaluation.write_text(json.dumps(score), encoding="utf-8")
    output = tmp_path / "ridge.html"
    result = command(ROOT, "report", "--backtest", source, "--output", output,
                     "--evaluation", evaluation, "--training-data", export[0],
                     "--model", saved_model, "--dataset", stored)
    assert result.returncode == 0, result.stderr
    _, report = report_data(output)
    forecast = report["forecast"]
    assert forecast["test"] == score["test"]
    assert forecast["points"][0]["at"] == score["test"]["first_decision_at"]
    assert forecast["points"][0]["actual"] == 500
    assert forecast["points"][0]["predicted"] == pytest.approx(3 + 200 * (2 / 1.1))
    trade, = report["runs"][0]["trades"]
    assert Decimal(trade["net_pnl"]) == Decimal("4.782")
    explanation = trade["prediction"]
    assert explanation["at"] == at(2).isoformat()
    assert explanation["threshold"] == 0.01
    assert explanation["forecast"] == pytest.approx(sum(row["contribution"] for row in explanation["inputs"]))
    assert report["runs"][0]["strategy"]["model_report_id"] == model_report["report_id"]

    evaluation.write_text(json.dumps({**score, "model_report_id": "different"}), encoding="utf-8")
    result = command(ROOT, "report", "--backtest", source, "--evaluation", evaluation,
                     "--output", tmp_path / "mismatch.html")
    assert result.returncode == 1
    assert "evaluation checksum" in result.stderr
    assert not (tmp_path / "mismatch.html").exists()
