"""Internal consistency checks for saved engine results (not a profitability audit)."""


from datetime import datetime, timedelta
from decimal import Decimal

from market_trader.backtesting import ExecutionProfile, _execution_price, _validate_profile


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("expected a UTC timestamp in saved results")
    result = datetime.fromisoformat(value)
    if result.utcoffset() != timedelta(0):
        raise ValueError("report timestamps must be in UTC")
    return result


def number(value: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError("expected a decimal string in backtest results")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("backtest amounts must be finite")
    return result


def _validate_fill(fill: dict, profile: ExecutionProfile) -> None:
    timestamp(fill["candle_open_time"])
    if fill["side"] not in ("buy", "sell"):
        raise ValueError("fill side must be buy or sell")
    for field in ("quantity", "price", "reference_price"):
        if number(fill[field]) <= 0:
            raise ValueError(f"fill {field} must be positive")
    quantity = number(fill["quantity"])
    if quantity % profile.quantity_step or quantity < profile.minimum_quantity:
        raise ValueError("fill quantity violates execution quantity grid or minimum")
    if fill["reason"] == "entry" and fill["side"] == "sell" and not profile.allow_short:
        raise ValueError("short entry is disabled by the execution profile")
    # Reuse engine cost/grid arithmetic rather than a subtly different formula.
    price = _execution_price(profile, fill["side"], number(fill["reference_price"]))
    if number(fill["price"]) != price:
        raise ValueError("fill price does not match reference price and execution costs")
    fee = number(fill["price"]) * number(fill["quantity"]) * profile.commission_rate
    if number(fill["fee"]) != fee:
        raise ValueError("fill fee does not match execution commission rate")


def validate_ledger(report: dict) -> None:
    """Check the execution ledger before deriving any displayed metrics."""
    execution = report["execution"]
    profile = ExecutionProfile(
        initial_capital=number(execution["initial_capital"]),
        commission_rate=number(execution["commission_rate"]),
        price_tick=number(execution["price_tick"]), quantity_step=number(execution["quantity_step"]),
        spread=number(execution["spread"]), slippage=number(execution["slippage"]),
        minimum_quantity=number(execution["minimum_quantity"]),
        allow_short=execution["allow_short"], liquidate_at_end=execution["liquidate_at_end"],
    )
    _validate_profile(profile)
    expected = []
    for trade in report["trades"]:
        entry, exit_fill = trade["entry"], trade["exit"]
        _validate_fill(entry, profile)
        _validate_fill(exit_fill, profile)
        if entry["side"] == exit_fill["side"]:
            raise ValueError("trade entry and exit sides must oppose")
        if number(entry["quantity"]) != number(exit_fill["quantity"]):
            raise ValueError("trade entry and exit quantities must match")
        expected.extend((entry, exit_fill))
    if report["position"] is not None:
        entry = report["position"]["entry"]
        _validate_fill(entry, profile)
        expected.append(entry)
    if not isinstance(report["fills"], list) or report["fills"] != expected:
        raise ValueError("top-level fills do not match trades and open position")
    _validate_chronology(report)
    _validate_accounts(report)


def _fill_clock(fill: dict) -> tuple:
    phases = {"open": 0, "intrabar": 1, "close": 2}
    if fill["phase"] not in phases:
        raise ValueError("invalid fill phase")
    return timestamp(fill["candle_open_time"]), phases[fill["phase"]]


def _validate_chronology(report: dict) -> None:
    accounts = report["accounts"]
    times = [timestamp(row["at"]) for row in accounts] if any(
        row["at"] is not None for row in accounts) else []
    if not times:
        if len(accounts) != 1 or report["fills"]:
            raise ValueError("undated history must be one empty initial snapshot")
    elif any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("account timestamps must be strictly chronological")
    previous = None
    for fill in report["fills"]:
        clock = _fill_clock(fill)
        if not times or not times[0] <= clock[0] < times[-1]:
            raise ValueError("fill is outside the account history")
        if previous is not None and clock < previous:
            raise ValueError("fills must be chronological by candle and execution phase")
        previous = clock
    for trade in report["trades"]:
        entry, exit_fill = trade["entry"], trade["exit"]
        _validate_entry(entry)
        if _fill_clock(exit_fill) <= _fill_clock(entry):
            raise ValueError("trade exit must follow entry")
        reason, phase = exit_fill["reason"], exit_fill["phase"]
        if reason in ("stop", "target"):
            valid = phase in ("open", "intrabar")
        elif reason == "exit":
            valid = phase == "open"
        elif reason == "liquidation":
            valid = (phase == "close" and report["execution"]["liquidate_at_end"] is True
                     and exit_fill == report["fills"][-1] and len(times) >= 2
                     and times[-2] <= _fill_clock(exit_fill)[0] < times[-1])
        else:
            valid = False
        if not valid:
            raise ValueError("invalid exit reason, phase, or terminal liquidation")
    if report["position"] is not None:
        _validate_entry(report["position"]["entry"])
        if report["execution"]["liquidate_at_end"]:
            raise ValueError("terminal liquidation cannot leave an open position")


def _validate_entry(entry: dict) -> None:
    if entry["phase"] != "open" or entry["reason"] != "entry":
        raise ValueError("entries must execute at the open with reason entry")


def _validate_accounts(report: dict) -> None:
    accounts = report["accounts"]
    if not isinstance(accounts, list) or not accounts:
        raise ValueError("backtest requires account snapshots")
    initial = number(report["execution"]["initial_capital"])
    realized = Decimal(0)
    position = None
    fill_index = trade_index = 0
    fills = report["fills"]
    for account in accounts:
        at = timestamp(account["at"]) if account["at"] is not None else None
        # A candle open equals the PREVIOUS close snapshot. Even an intrabar
        # fill belongs to the next close, not to that previous snapshot.
        while (fill_index < len(fills) and at is not None
               and timestamp(fills[fill_index]["candle_open_time"]) < at):
            fill = fills[fill_index]
            if position is None:
                position = fill
            else:
                realized += number(report["trades"][trade_index]["net_pnl"])
                trade_index += 1
                position = None
            fill_index += 1
        balance = initial + realized
        if position is not None:
            balance -= number(position["fee"])
        unrealized = number(account["unrealized_pnl"])
        # Saved reports do not contain the available candle mark. With an open
        # position we can reconcile equity, but cannot reconstruct that mark.
        if position is None and unrealized != 0:
            raise ValueError("flat account cannot have unrealized P&L")
        if (number(account["realized_pnl"]) != realized
                or number(account["balance"]) != balance
                or number(account["equity"]) != balance + unrealized):
            raise ValueError("account snapshot does not reconcile with the execution ledger")
    if fill_index != len(fills):
        raise ValueError("fills extend beyond the account history")
