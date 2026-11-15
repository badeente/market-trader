"""Frozen AP1 data/label/baseline protocol, without tuning or held-out scoring."""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext

from pathlib import Path
import re

import tomllib
from zoneinfo import ZoneInfo

from market_trader._config import require_fields
from market_trader.report_validation import number, timestamp, validate_ledger

EXPERIMENT_ID = "ap1-deuidxeur-open30-v1"
_START = datetime(2021, 1, 1, tzinfo=UTC)
_VALIDATION = datetime(2024, 1, 1, tzinfo=UTC)
_TEST = datetime(2025, 1, 1, tzinfo=UTC)
_END = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    source_csv: Path
    source_sha256: str


def read_reference_config(path: Path) -> ReferenceConfig:
    """Read source location/identity; research parameters are frozen by protocol."""
    with Path(path).open("rb") as stream:
        config = require_fields(tomllib.load(stream), "AP1 config", "version", "experiment", "source_csv", "source_sha256")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("AP1 config version must be the integer 1")
    if config["experiment"] != EXPERIMENT_ID:
        raise ValueError("unsupported AP1 experiment protocol")
    source = config["source_csv"]
    if not isinstance(source, str) or not source or source != source.strip():
        raise ValueError("source_csv must be a nonempty path without padding")
    checksum = config["source_sha256"]
    if not isinstance(checksum, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", checksum) is None:
        raise ValueError("source_sha256 must be an exact lowercase sha256 identifier")
    return ReferenceConfig((Path(path).parent / source).resolve(), checksum)


def run_reference(config_path: Path, output: Path) -> dict:
    """Execute AP1 through the existing importer/model/ledger/report contracts."""
    from market_trader.reference_pipeline import run_reference as execute

    return execute(config_path, output)


def trading_summary(report: dict, *, year: int = 2024) -> dict:
    """Derive point/cost/month metrics from a reconciled closed-position ledger."""
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        validate_ledger(report)
        if report["position"] is not None or report["unfilled_order"] is not None or report["rejections"]:
            raise ValueError("reference run requires no open positions, pending orders or rejections")
        berlin = ZoneInfo("Europe/Berlin")
        for account in report["accounts"]:
            if account["at"] is None or timestamp(account["at"]).astimezone(berlin).year != year:
                raise ValueError("reference account history must stay within the validation year")
        months = {f"{year}-{month:02}": {"month": f"{year}-{month:02}", "trade_count": 0,
                  "net_points": Decimal(0), "reference_points": Decimal(0), "modeled_cost_points": Decimal(0)}
                  for month in range(1, 13)}
        wins = 0
        for trade in report["trades"]:
            entry, exit_fill = trade["entry"], trade["exit"]
            direction = 1 if entry["side"] == "buy" else -1
            quantity = number(entry["quantity"])
            gross = direction * (number(exit_fill["price"]) - number(entry["price"])) * quantity
            net = gross - number(entry["fee"]) - number(exit_fill["fee"])
            if gross != number(trade["gross_pnl"]) or net != number(trade["net_pnl"]):
                raise ValueError("reference trade P&L must reconcile with its fills")
            reference = direction * (number(exit_fill["reference_price"]) - number(entry["reference_price"])) * quantity
            key = timestamp(exit_fill["candle_open_time"]).astimezone(berlin).strftime("%Y-%m")
            if key not in months:
                raise ValueError("reference trade is outside the validation year")
            month = months[key]
            month["trade_count"] += 1
            month["net_points"] += net
            month["reference_points"] += reference
            month["modeled_cost_points"] += reference - net
            wins += net > 0
        net = sum(month["net_points"] for month in months.values())
        initial = number(report["execution"]["initial_capital"])
        if number(report["accounts"][-1]["equity"]) - initial != net:
            raise ValueError("reference trade totals must match final equity")
        peak = initial
        drawdown = Decimal(0)
        for account in report["accounts"]:
            equity = number(account["equity"])
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        count = len(report["trades"])
        return {
            "trade_count": count, "net_points": str(net),
            "reference_points": str(sum(month["reference_points"] for month in months.values())),
            "modeled_cost_points": str(sum(month["modeled_cost_points"] for month in months.values())),
            "max_drawdown_points": str(drawdown), "max_drawdown_fraction": report["max_drawdown"],
            "win_fraction": str(Decimal(wins) / count) if count else None,
            "months": [{key: str(value) if isinstance(value, Decimal) else value for key, value in month.items()}
                       for month in months.values()],
        }
