"""Saved reports must reconcile, not merely contain plausible summary numbers."""

from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import json

import pytest

from market_trader.backtesting import ExecutionProfile, Signal, Action, run_backtest
from market_trader.reporting import BACKTEST_REPORT_FORMAT, render_report
from tests.test_backtesting import FLAT, LONG, EXIT, candle, dataset, scripted
from tests.test_data_cli import ROOT, command


def report_body(*signals, profile=None, prices=None):
    bars = prices or [FLAT, FLAT, ("110", "111", "109", "110")]
    result = run_backtest(dataset(*(candle(i, row) for i, row in enumerate(bars))),
                          scripted(*signals), profile=profile or ExecutionProfile(
                              commission_rate=Decimal("0.001")))
    body = asdict(result)
    body["execution"] = body.pop("profile")
    body.update(format=BACKTEST_REPORT_FORMAT, strategy={"name": "test", "lookback": 1},
                runtime={}, provenance={})
    return json.loads(json.dumps(body, default=lambda value: value.isoformat()
                                if hasattr(value, "isoformat") else str(value)))


def save_report(tmp_path, body):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.parametrize("field,value", [
    ("spread", "-1"), ("price_tick", "0"), ("quantity_step", "0"),
    ("allow_short", "yes"), ("liquidate_at_end", "false"),
])
def test_report_rejects_invalid_execution_profiles_even_without_trades(tmp_path, field, value):
    body = report_body()
    body["execution"][field] = value
    with pytest.raises(ValueError):
        render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("field,value", [("quantity_step", "2"), ("minimum_quantity", "2"), ("allow_short", False)])
def test_report_rejects_fills_that_violate_execution_profile(tmp_path, field, value):
    body = report_body(Signal(Action.ENTER_SHORT), EXIT, profile=ExecutionProfile(allow_short=True))
    body["execution"][field] = value
    with pytest.raises(ValueError):
        render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("field,value", [("price", "999"), ("gross_pnl", "999")])
def test_report_rejects_fabricated_trade_profit(tmp_path, field, value):
    body = report_body(LONG, EXIT)
    trade = body["trades"][0]
    if field == "price":
        trade["entry"][field] = value
    else:
        trade[field] = value
        trade["net_pnl"] = str(Decimal(value) - Decimal(trade["entry"]["fee"])
                                - Decimal(trade["exit"]["fee"]))
    source = save_report(tmp_path, body)
    with pytest.raises(ValueError):
        render_report(source)
    output = tmp_path / "invalid.html"
    result = command(ROOT, "report", "--backtest", source, "--output", output)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("problem", ["side", "quantity", "negative_price", "reference", "fee",
                                      "missing_fill", "extra_fill", "changed_fill", "missing_trade"])
def test_report_rejects_inconsistent_execution_ledger(tmp_path, problem):
    body = report_body(LONG, EXIT)
    trade = body["trades"][0]
    if problem == "side":
        trade["exit"]["side"] = trade["entry"]["side"]
    elif problem == "quantity":
        trade["exit"]["quantity"] = "2"
    elif problem == "negative_price":
        trade["entry"]["price"] = "-100"
        trade["exit"]["price"] = "-90"
    elif problem == "reference":
        trade["entry"]["reference_price"] = "0"
    elif problem == "fee":
        trade["entry"]["fee"] = "0"
        trade["net_pnl"] = str(Decimal(trade["gross_pnl"]) - Decimal(trade["exit"]["fee"]))
    elif problem == "missing_fill":
        body["fills"].pop()
    elif problem == "extra_fill":
        body["fills"].append(body["fills"][-1])
    elif problem == "changed_fill":
        body["fills"][0]["price"] = "999"
    else:
        body["trades"] = []
    with pytest.raises(ValueError):
        render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("field", ["balance", "realized_pnl", "unrealized_pnl", "equity"])
@pytest.mark.parametrize("index", [0, 2, 3])
def test_report_reconciles_each_account_snapshot(tmp_path, field, index):
    body = report_body(LONG, EXIT)
    body["accounts"][index][field] = str(Decimal(body["accounts"][index][field]) + 1)
    if field == "equity":
        # Keep the old summary checks satisfied; ledger validation must still reject it.
        initial = Decimal(body["execution"]["initial_capital"])
        peak = initial
        drawdowns = []
        for account in body["accounts"]:
            equity = Decimal(account["equity"])
            peak = max(peak, equity)
            drawdowns.append((equity - peak) / peak)
        body["total_return"] = str((Decimal(body["accounts"][-1]["equity"]) - initial) / initial)
        body["max_drawdown"] = str(min(drawdowns))
    with pytest.raises(ValueError):
        render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("signals,profile,prices", [
    ((), ExecutionProfile(), None),
    ((LONG,), ExecutionProfile(commission_rate=Decimal("0.001")), None),
    ((Signal(Action.ENTER_SHORT), EXIT), ExecutionProfile(allow_short=True,
        commission_rate=Decimal("0.001")), None),
    ((LONG,), ExecutionProfile(liquidate_at_end=True, commission_rate=Decimal("0.001")), None),
    ((Signal(Action.ENTER_LONG, stop=Decimal("99.5")),), ExecutionProfile(), None),
    ((Signal(Action.ENTER_LONG, target=Decimal("100.5")),), ExecutionProfile(), None),
])
def test_report_accepts_valid_engine_ledgers(tmp_path, signals, profile, prices):
    body = report_body(*signals, profile=profile, prices=prices)
    assert render_report(save_report(tmp_path, body)).startswith(b"<!doctype html>")


@pytest.mark.parametrize("problem", ["entry_phase", "exit_phase", "entry_reason", "exit_reason",
                                      "same_open_exit", "before_history", "early_liquidation",
                                      "liquidation_disabled", "unliquidated"])
def test_report_rejects_impossible_execution_chronology(tmp_path, problem):
    liquidates = problem in ("early_liquidation", "liquidation_disabled")
    body = report_body(LONG, *(() if liquidates else (EXIT,)),
                      profile=ExecutionProfile(liquidate_at_end=liquidates))
    if problem == "entry_phase":
        body["trades"][0]["entry"]["phase"] = "intrabar"
    elif problem == "exit_phase":
        body["trades"][0]["exit"]["phase"] = "close"
    elif problem == "entry_reason":
        body["trades"][0]["entry"]["reason"] = "exit"
    elif problem == "exit_reason":
        body["trades"][0]["exit"]["reason"] = "entry"
    elif problem == "same_open_exit":
        body["trades"][0]["exit"]["candle_open_time"] = body["trades"][0]["entry"]["candle_open_time"]
        # No fee/mark changes: move realized profit into the entry candle account.
        body["accounts"][2].update(body["accounts"][3], at=body["accounts"][2]["at"])
    elif problem == "before_history":
        body["accounts"][0]["at"] = body["accounts"][2]["at"]
        body["accounts"] = body["accounts"][0:1] + body["accounts"][3:]
        body["accounts"][0].update(body["accounts"][-1], at=body["accounts"][0]["at"])
        body["trades"][0]["exit"]["candle_open_time"] = body["trades"][0]["entry"]["candle_open_time"]
    elif problem == "early_liquidation":
        # Extend account history after a purported terminal close.
        body["accounts"].append({**body["accounts"][-1], "at": "2026-01-01T00:04:00+00:00"})
    elif problem == "liquidation_disabled":
        body["execution"]["liquidate_at_end"] = False
    else:
        body = report_body(LONG)
        body["execution"]["liquidate_at_end"] = True
    body["fills"] = [fill.copy() for trade in body["trades"] for fill in (trade["entry"], trade["exit"])]
    if body["position"] is not None:
        body["fills"].append(body["position"]["entry"].copy())
    with pytest.raises(ValueError):
        render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("field,value", [("price", "999"), ("quantity", "0"), ("reference_price", "0")])
def test_report_checks_open_position_execution(tmp_path, field, value):
    body = report_body(LONG, profile=ExecutionProfile())
    body["position"]["entry"][field] = value
    body["fills"][0][field] = value
    with pytest.raises(ValueError):
        render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("liquidate", [False, True])
def test_report_preserves_engine_precision_28_arithmetic(tmp_path, short, liquidate):
    profile = ExecutionProfile(allow_short=short, liquidate_at_end=liquidate,
        commission_rate=Decimal("0.001234567890123456789"),
        quantity_step=Decimal("0.00000000000000000001"),
        price_tick=Decimal("0.00000000000000000001"),
        spread=Decimal("0.1234567890123456789"), slippage=Decimal("0.01234567890123456789"))
    entry = Signal(Action.ENTER_SHORT if short else Action.ENTER_LONG,
                   quantity=Decimal("0.1234567890123456789"))
    prices = [("100.1234567890123456789", "111.1234567890123456789", "99", "105.1234567890123456789"),
              ("103.1234567890123456789", "111.1234567890123456789", "99", "108.1234567890123456789")]
    body = report_body(entry, EXIT, entry, EXIT, entry, prices=prices * 4, profile=profile)
    with localcontext() as context:
        context.prec = 6
        assert render_report(save_report(tmp_path, body)).startswith(b"<!doctype html>")


def test_report_accepts_delayed_marks_and_same_open_reentry(tmp_path):
    # Protection can close a previous trade before a pending entry at this open.
    signals = (Signal(Action.ENTER_LONG, stop=Decimal("98")), LONG)
    prices = [FLAT, FLAT, ("95", "97", "94", "96")]
    body = report_body(*signals, prices=prices)
    assert len(body["trades"]) == 1 and body["position"] is not None
    assert render_report(save_report(tmp_path, body))
    # A delayed candle's close is not necessarily the mark used by the engine.
    from tests.test_backtesting import START, MINUTE
    delayed = replace(candle(1, ("100", "110", "99", "109")), available_at=START + 3 * MINUTE)
    result = run_backtest(dataset(candle(0, FLAT), delayed), scripted(LONG),
                          profile=ExecutionProfile(commission_rate=Decimal("0.001")))
    saved = asdict(result)
    saved["execution"] = saved.pop("profile")
    body.update(saved)
    body = json.loads(json.dumps(body, default=lambda value: value.isoformat()
                                if hasattr(value, "isoformat") else str(value)))
    assert render_report(save_report(tmp_path, body))


@pytest.mark.parametrize("path,value", [
    (("accounts",), None), (("trades",), {}), (("position",), []),
    (("trades", 0, "entry", "price"), "not-a-number"),
    (("trades", 0, "entry", "price"), None),
    (("trades", 0, "exit"), None), (("execution",), []),
    (("strategy",), None), (("rejections",), None), (("runtime",), "MISSING"),
])
def test_report_malformed_input_is_a_value_error_without_cli_output(tmp_path, path, value):
    body = report_body(LONG, EXIT)
    target = body
    for key in path[:-1]:
        target = target[key]
    if value == "MISSING":
        del target[path[-1]]
    else:
        target[path[-1]] = value
    source = save_report(tmp_path, body)
    with pytest.raises(ValueError):
        render_report(source)
    output = tmp_path / "invalid.html"
    result = command(ROOT, "report", "--backtest", source, "--output", output)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert not output.exists()
