"""Build a standalone HTML report from saved offline results."""

from datetime import datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from hashlib import sha256
from math import isfinite
from pathlib import Path

from market_trader._dataset_codec import json_bytes, parse_json
from market_trader.baseline import HELD_OUT_FORMAT, SavedRidgeModel, held_out_evaluation, load_ridge_model
from market_trader.storage import load_dataset
from market_trader.training_data import FEATURE_NAMES, candle_features

BACKTEST_REPORT_FORMAT = "market-trader.backtest-report.v1"


def _read_report(path: Path, expected_format: str) -> dict:
    data = parse_json(path.read_text(encoding="utf-8"), allow_floats=True)
    if not isinstance(data, dict) or data.get("format") != expected_format:
        raise ValueError(f"{path} must contain a {expected_format} report")
    return data


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("expected a UTC timestamp in saved results")
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("report timestamps must be in UTC")
    return timestamp


def _number(value: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("expected a decimal string in backtest results")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("backtest amounts must be finite")
    return result


def _run_summary(report: dict) -> dict:
    strategy = report["strategy"]
    if not isinstance(strategy["name"], str) or not strategy["name"]:
        raise ValueError("backtest strategy must have a name")
    accounts = report["accounts"]
    trades = report["trades"]
    if not isinstance(accounts, list) or not accounts or not isinstance(trades, list):
        raise ValueError("backtest requires account snapshots and a trade list")
    initial = _number(report["execution"]["initial_capital"])
    if initial <= 0:
        raise ValueError("backtest initial capital must be positive")
    peak = initial
    returns, drawdown = [], []
    previous = None
    for account in accounts:
        at = account["at"]
        if at is None and len(accounts) == 1:
            timestamp = None
        else:
            timestamp = _timestamp(at)
            if previous is not None and timestamp <= previous:
                raise ValueError("account timestamps must be strictly chronological")
        previous = timestamp
        equity = _number(account["equity"])
        peak = max(peak, equity)
        returns.append((equity - initial) / initial)
        drawdown.append((equity - peak) / peak)
    if returns[-1] != _number(report["total_return"]) or min(drawdown) != _number(report["max_drawdown"]):
        raise ValueError("backtest return or drawdown does not match account history")
    for trade in trades:
        entry, exit_fill = trade["entry"], trade["exit"]
        for fill in (entry, exit_fill):
            _timestamp(fill["candle_open_time"])
            _number(fill["price"])
        side = entry["side"]
        if side not in ("buy", "sell"):
            raise ValueError("entry fill side must be buy or sell")
        direction = 1 if side == "buy" else -1
        reference_pnl = direction * (
            _number(exit_fill["reference_price"]) - _number(entry["reference_price"])
        ) * _number(entry["quantity"])
        entry_fee, exit_fee = _number(entry["fee"]), _number(exit_fill["fee"])
        gross = _number(trade["gross_pnl"])
        if gross - entry_fee - exit_fee != _number(trade["net_pnl"]):
            raise ValueError("trade net P&L does not match gross P&L and fees")
        trade["reference_pnl"] = str(reference_pnl)
        trade["fill_price_impact"] = str(reference_pnl - gross)
        trade["fees"] = str(entry_fee + exit_fee)
    return {
        "name": strategy["name"], "strategy": strategy, "dataset_id": report["dataset_id"],
        "dataset_artifact_id": report.get("dataset_artifact_id"), "execution": report["execution"],
        "runtime": report["runtime"], "provenance": report["provenance"], "returns": list(map(float, returns)),
        "drawdown": list(map(float, drawdown)), "equity_change": str(equity - initial),
        "trades": trades, "rejections": len(report["rejections"]),
        "position_open": report["position"] is not None,
        "order_pending": report["unfilled_order"] is not None,
    }


def _account_times(primary: dict, comparison: dict | None) -> list:
    times = [row["at"] for row in primary["accounts"]]
    if comparison is not None:
        for key in ("dataset_id", "execution"):
            if primary[key] != comparison[key]:
                raise ValueError("comparison requires the same dataset and execution profile")
        if times != [row["at"] for row in comparison["accounts"]]:
            raise ValueError("comparison requires the same account timestamps")
    return times


def _forecast_summary(evaluation: Path | None, model: Path | None, training_data: Path | None,
                      model_id: str | None) -> dict | None:
    if training_data is not None and model is None:
        raise ValueError("--training-data requires --model")
    saved = None
    if evaluation is not None:
        saved = _read_report(evaluation, HELD_OUT_FORMAT)
        body = {key: value for key, value in saved.items() if key != "report_id"}
        if saved["report_id"] != "sha256:" + sha256(json_bytes(body)).hexdigest():
            raise ValueError("evaluation checksum mismatch")
        if not model_id or saved["model_report_id"] != model_id:
            raise ValueError("evaluation model does not match the primary backtest")
    points = []
    if training_data is not None:
        computed, points = held_out_evaluation(model, training_data)
        if computed["model_report_id"] != model_id:
            raise ValueError("evaluation model does not match the primary backtest")
        if saved is not None and saved != computed:
            raise ValueError("evaluation does not match the model and training data")
        saved = computed
    if saved is None:
        return None
    test = saved["test"]
    if type(test["row_count"]) is not int or test["row_count"] <= 0:
        raise ValueError("evaluation must have a positive test row count")
    if _timestamp(test["first_decision_at"]) > _timestamp(test["last_decision_at"]):
        raise ValueError("evaluation test dates are out of order")
    for baseline in ("ridge", "zero_return"):
        for metric in ("mae", "rmse"):
            score = test[baseline][metric]
            if type(score) not in (int, float) or not isfinite(score) or score < 0:
                raise ValueError(f"invalid evaluation metric: {baseline}.{metric}")
    return {"instrument": saved["instrument"], "target": saved["target_name"],
            "test": test, "points": points, "report_id": saved["report_id"]}


def _explain_ridge_trades(run: dict, model: SavedRidgeModel, dataset_path: Path) -> None:
    stored = load_dataset(dataset_path)
    dataset = stored.dataset
    strategy = run["strategy"]
    if (dataset.dataset_id != run["dataset_id"]
            or (run["dataset_artifact_id"] is not None
                and stored.manifest.artifact_id != run["dataset_artifact_id"])):
        raise ValueError("dataset does not match the primary backtest")
    if dataset.instrument != model.instrument or dataset.interval != model.interval:
        raise ValueError("model instrument or interval does not match the backtest dataset")
    lookback = model.config.lookback_bars
    if strategy["lookback"] != lookback:
        raise ValueError("model lookback does not match the primary backtest")
    threshold = _number(strategy["entry_threshold"])
    candle_index = {candle.close_time: index for index, candle in enumerate(dataset.candles)}
    for trade in run["trades"]:
        at = _timestamp(trade["entry"]["order"]["decision_at"])
        end = candle_index.get(at, -1) + 1
        if end < lookback:
            raise ValueError("trade has no complete decision window in the backtest dataset")
        window = dataset.candles[end - lookback:end]
        if any(candle.available_at > at for candle in window):
            raise ValueError("trade decision window contains unavailable candles")
        values = candle_features(window)
        forecast = model.model.predict(values)
        if Decimal(str(forecast)) <= threshold:
            raise ValueError("reconstructed Ridge forecast does not exceed the trade's entry threshold")
        inputs = [{"name": "intercept", "value": None, "contribution": model.model.intercept}]
        inputs.extend(
            {"name": name, "value": str(value), "contribution": contribution}
            for name, value, contribution in zip(
                FEATURE_NAMES, values, model.model.contributions(values), strict=True,
            )
        )
        trade["prediction"] = {
            "at": at.isoformat(), "forecast": forecast, "threshold": float(threshold), "inputs": inputs,
        }


def render_report(backtest: Path, *, compare: Path | None = None, evaluation: Path | None = None,
                  model: Path | None = None, training_data: Path | None = None,
                  dataset: Path | None = None) -> bytes:
    """Render backtest results and optional Ridge forecast details without fitting a model."""
    primary = _read_report(backtest, BACKTEST_REPORT_FORMAT)
    comparison = _read_report(compare, BACKTEST_REPORT_FORMAT) if compare is not None else None
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        runs = [_run_summary(primary)]
        if comparison is not None:
            runs.append(_run_summary(comparison))
        times = _account_times(primary, comparison)
        model_id = runs[0]["strategy"].get("model_report_id")
        saved_model = load_ridge_model(model) if model is not None else None
        if saved_model is not None and saved_model.report_id != model_id:
            raise ValueError("model does not match the primary backtest")
        if dataset is not None:
            if saved_model is None:
                raise ValueError("--dataset requires --model")
            _explain_ridge_trades(runs[0], saved_model, dataset)
        forecast = _forecast_summary(evaluation, model, training_data, model_id)
    template = Path(__file__).with_name("report_template.html.in").read_text(encoding="utf-8")
    data = json_bytes({"times": times, "runs": runs, "forecast": forecast}).decode("utf-8")
    return template.replace("__REPORT_DATA__", data.replace("<", r"\u003c"), 1).encode("utf-8")
