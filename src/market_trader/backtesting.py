"""Deterministic offline execution and accounting for one normalized series."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import Enum, auto
from fractions import Fraction
from typing import Literal

from market_trader._numeric import round_to_grid, sum_decimals
from market_trader.domain import Candle
from market_trader.normalization import NormalizedDataset, normalize_candles

_ZERO = Decimal(0)
_ONE = Decimal(1)
_Side = Literal["buy", "sell"]
_Phase = Literal["open", "intrabar", "close"]
_Reason = Literal["entry", "exit", "stop", "target", "liquidation"]


class Action(Enum):
    HOLD = auto()
    ENTER_LONG = auto()
    ENTER_SHORT = auto()
    EXIT = auto()


@dataclass(frozen=True, slots=True)
class Signal:
    action: Action
    quantity: Decimal | None = None
    stop: Decimal | None = None
    target: Decimal | None = None
    stop_distance: Decimal | None = None
    risk_fraction: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    initial_capital: Decimal = Decimal(1000)
    commission_rate: Decimal = _ZERO
    price_tick: Decimal = Decimal("0.1")
    quantity_step: Decimal = _ONE
    allow_short: bool = False
    spread: Decimal = _ZERO
    slippage: Decimal = _ZERO
    minimum_quantity: Decimal = _ZERO
    liquidate_at_end: bool = False


@dataclass(frozen=True, slots=True)
class Fill:
    """Identify the fill candle and distinguish its open from unknown intrabar timing."""

    candle_open_time: datetime
    phase: _Phase
    side: _Side
    reason: _Reason
    quantity: Decimal
    price: Decimal
    fee: Decimal
    reference_price: Decimal
    intrabar_ambiguous: bool = False
    order: Order | None = None


@dataclass(frozen=True, slots=True)
class Position:
    entry: Fill
    stop: Decimal | None
    target: Decimal | None


@dataclass(frozen=True, slots=True)
class Trade:
    entry: Fill
    exit: Fill
    gross_pnl: Decimal
    net_pnl: Decimal


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Balance includes paid entry fees, and equity includes unrealized price P&L."""

    at: datetime | None
    balance: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class Decision:
    at: datetime
    window: tuple[Candle, ...]
    account: AccountSnapshot
    position: Position | None


@dataclass(frozen=True, slots=True)
class EntryPlan:
    quantity: Decimal
    entry_price: Decimal
    stop: Decimal | None
    planned_stop_risk: Decimal | None
    risk_budget: Decimal | None


@dataclass(frozen=True, slots=True)
class Order:
    decision_at: datetime
    signal: Signal
    plan: EntryPlan | None = None


@dataclass(frozen=True, slots=True)
class OrderRejection:
    order: Order
    reason: str


@dataclass(frozen=True, slots=True)
class BacktestResult:
    dataset_id: str
    profile: ExecutionProfile
    fills: tuple[Fill, ...]
    trades: tuple[Trade, ...]
    accounts: tuple[AccountSnapshot, ...]
    position: Position | None
    rejections: tuple[OrderRejection, ...]
    unfilled_order: Order | None
    total_return: Decimal
    max_drawdown: Decimal

    @property
    def account(self) -> AccountSnapshot:
        return self.accounts[-1]


def _positive_amount(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value > 0


def _validate_profile(profile: ExecutionProfile) -> None:
    for name in ("initial_capital", "price_tick", "quantity_step"):
        if not _positive_amount(getattr(profile, name)):
            raise ValueError(f"{name} must be a positive finite Decimal")
    for name in ("commission_rate", "spread", "slippage", "minimum_quantity"):
        value = getattr(profile, name)
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            raise ValueError(f"{name} must be a nonnegative finite Decimal")
    for name in ("allow_short", "liquidate_at_end"):
        if type(getattr(profile, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")


def _validated_candles(dataset: NormalizedDataset) -> tuple[Candle, ...]:
    # Recompute coverage so a hand-built dataset cannot bypass the gap check.
    normalized = normalize_candles(
        dataset.candles, config=dataset.config,
        requests=None if dataset.request is None else (dataset.request,),
    )
    if len(normalized.datasets) != 1:
        raise ValueError("a backtest requires exactly one normalized candle series")
    expected = normalized.datasets[0]
    if expected.quality.gaps:
        raise ValueError("backtest dataset has missing candles in active intervals")
    if (expected.candles != dataset.candles or expected.dataset_id != dataset.dataset_id
            or expected.instrument != dataset.instrument or expected.interval != dataset.interval):
        raise ValueError("backtest dataset does not match its normalized identity or series")
    if any(candle.low <= 0 for candle in expected.candles):
        raise ValueError("R0 candle prices must be positive")
    return expected.candles


def _cost_offset(profile: ExecutionProfile) -> Decimal:
    with localcontext() as context:
        context.prec = max(context.prec, len(profile.spread.as_tuple().digits) + 1)
        half_spread = profile.spread / 2
    return sum_decimals((half_spread, profile.slippage))


def _round_price(price: Decimal | Fraction, tick: Decimal, side: _Side) -> Decimal:
    return round_to_grid(price, tick, upward=side == "buy")


def _execution_price(profile: ExecutionProfile, side: _Side, reference: Decimal) -> Decimal:
    offset = _cost_offset(profile)
    adjusted = sum_decimals((reference, offset if side == "buy" else offset.copy_negate()))
    price = _round_price(adjusted, profile.price_tick, side)
    if price <= 0:
        raise ValueError("execution price must be positive after costs and price-grid rounding")
    return price


def _protection_error(side: _Side, price: Decimal, stop: Decimal | None, target: Decimal | None) -> str | None:
    if stop is not None and (stop >= price if side == "buy" else stop <= price):
        return "stop must be on the loss side of the entry price"
    if target is not None and (target <= price if side == "buy" else target >= price):
        return "target must be on the profit side of the entry price"
    return None


def _prepare_order(decision: Decision, signal: Signal, profile: ExecutionProfile) -> Order:
    entry_fields = (signal.quantity, signal.stop, signal.target, signal.stop_distance, signal.risk_fraction)
    if signal.action in (Action.HOLD, Action.EXIT):
        if any(value is not None for value in entry_fields):
            raise ValueError("HOLD and EXIT cannot specify entry parameters")
        return Order(decision.at, signal)
    side = "buy" if signal.action is Action.ENTER_LONG else "sell"
    if side == "sell" and not profile.allow_short:
        raise ValueError("profile does not allow short positions")
    for price in (signal.stop, signal.target):
        if price is not None and (not _positive_amount(price) or price % profile.price_tick):
            raise ValueError("stop and target prices must be positive and on the price grid")

    reference = decision.window[-1].close
    stop = signal.stop
    if signal.stop_distance is not None:
        if stop is not None or not _positive_amount(signal.stop_distance):
            raise ValueError("stop_distance must be positive and cannot be combined with an absolute stop")
        reference_ratio = Fraction(reference)
        distance = reference_ratio * Fraction(signal.stop_distance)
        stop = (
            _round_price(reference_ratio - distance, profile.price_tick, "sell") if side == "buy"
            else _round_price(reference_ratio + distance, profile.price_tick, "buy")
        )
        if stop <= 0:
            raise ValueError("relative stop must produce a positive stop price")
    entry_price = _execution_price(profile, side, reference)
    protection_error = (
        _protection_error(side, reference, stop, signal.target)
        or _protection_error(side, entry_price, stop, signal.target)
    )
    if protection_error and (signal.risk_fraction is not None or signal.stop_distance is not None):
        raise ValueError(protection_error)

    per_unit_risk = None
    if stop is not None and protection_error is None:
        closing_side = "sell" if side == "buy" else "buy"
        stop_price = _execution_price(profile, closing_side, stop)
        fees = (entry_price + stop_price) * profile.commission_rate
        per_unit_risk = abs(entry_price - stop_price) + fees

    budget = None
    quantity = _ONE if signal.quantity is None else signal.quantity
    if signal.risk_fraction is not None:
        if signal.quantity is not None:
            raise ValueError("risk_fraction and fixed quantity are mutually exclusive")
        if not _positive_amount(signal.risk_fraction) or signal.risk_fraction > _ONE:
            raise ValueError("risk_fraction must be greater than zero and at most one")
        if per_unit_risk is None or per_unit_risk <= 0:
            raise ValueError("risk-based sizing requires a stop with positive modeled risk")
        if decision.account.equity <= 0:
            raise ValueError("risk-based sizing requires positive equity")
        budget = decision.account.equity * signal.risk_fraction
        step = profile.quantity_step
        quantity = (budget // (per_unit_risk * step)) * step
    if not _positive_amount(quantity) or quantity < profile.minimum_quantity:
        raise ValueError("entry quantity is below the minimum")
    if quantity % profile.quantity_step:
        raise ValueError("entry quantity must be on the quantity grid")
    planned_risk = quantity * per_unit_risk if per_unit_risk is not None else None
    if budget is not None and planned_risk is not None and planned_risk > budget:
        raise ValueError("rounded quantity exceeds the planned risk budget")
    return Order(decision.at, signal, EntryPlan(quantity, entry_price, stop, planned_risk, budget))


@dataclass(slots=True)
class _Ledger:
    profile: ExecutionProfile
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    position: Position | None = None
    realized_pnl: Decimal = _ZERO

    def snapshot(self, at: datetime | None, mark: Decimal | None) -> AccountSnapshot:
        balance = self.profile.initial_capital + self.realized_pnl
        unrealized = _ZERO
        if self.position is not None:
            entry = self.position.entry
            balance -= entry.fee
            if mark is not None:
                unrealized = self._gross_pnl(entry, mark)
        return AccountSnapshot(at, balance, balance + unrealized, self.realized_pnl, unrealized)

    def execute(self, order: Order, candle: Candle) -> str | None:
        signal = order.signal
        if signal.action is Action.EXIT:
            if self.position is None:
                return "no open position at execution"
            self._close(candle, "open", "exit", candle.open, order=order)
            return None
        if self.position is not None:
            return "a position is already open at execution"

        side = "buy" if signal.action is Action.ENTER_LONG else "sell"
        assert order.plan is not None
        quantity, stop, target = order.plan.quantity, order.plan.stop, signal.target
        if reason := _protection_error(side, candle.open, stop, target):
            return reason
        entry = self._fill(candle, "open", side, "entry", quantity, candle.open, order=order)
        if reason := _protection_error(side, entry.price, stop, target):
            return reason
        if entry.price * quantity + entry.fee > self.profile.initial_capital + self.realized_pnl:
            return "insufficient capital at execution"
        self.fills.append(entry)
        self.position = Position(entry, stop, target)
        return None

    def protect(self, candle: Candle, phase: _Phase) -> None:
        position = self.position
        if position is None:
            return
        stop, target = position.stop, position.target
        if phase == "open":
            low = high = candle.open
        else:
            low, high = candle.low, candle.high
        offset = _cost_offset(self.profile)
        if position.entry.side == "buy":
            stop_hit = stop is not None and low <= stop
            target_reference = sum_decimals((target, offset)) if target is not None else None
            target_hit = target_reference is not None and high >= target_reference
        else:
            stop_hit = stop is not None and high >= stop
            target_reference = sum_decimals((target, offset.copy_negate())) if target is not None else None
            target_hit = target_reference is not None and low <= target_reference
        if stop is not None and stop_hit:
            reference = candle.open if phase == "open" else stop
            self._close(candle, phase, "stop", reference, ambiguous=phase == "intrabar" and target_hit)
        elif target_reference is not None and target_hit:
            reference = candle.open if phase == "open" else target_reference
            self._close(candle, phase, "target", reference)

    def liquidate(self, candle: Candle, reference: Decimal) -> None:
        self._close(candle, "close", "liquidation", reference)

    def _close(
        self, candle: Candle, phase: _Phase, reason: _Reason, reference: Decimal, *,
        ambiguous: bool = False, order: Order | None = None,
    ) -> None:
        assert self.position is not None
        entry = self.position.entry
        side = "sell" if entry.side == "buy" else "buy"
        exit_fill = self._fill(
            candle, phase, side, reason, entry.quantity, reference, ambiguous=ambiguous, order=order,
        )
        gross = self._gross_pnl(entry, exit_fill.price)
        net = gross - entry.fee - exit_fill.fee
        self.fills.append(exit_fill)
        self.trades.append(Trade(entry, exit_fill, gross, net))
        self.realized_pnl += net
        self.position = None

    def _fill(
        self, candle: Candle, phase: _Phase, side: _Side, reason: _Reason,
        quantity: Decimal, reference: Decimal, *, ambiguous: bool = False, order: Order | None = None,
    ) -> Fill:
        price = _execution_price(self.profile, side, reference)
        fee = price * quantity * self.profile.commission_rate
        return Fill(candle.open_time, phase, side, reason, quantity, price, fee, reference, ambiguous, order)

    @staticmethod
    def _gross_pnl(entry: Fill, exit_price: Decimal) -> Decimal:
        difference = exit_price - entry.price if entry.side == "buy" else entry.price - exit_price
        return difference * entry.quantity


def run_backtest(
    dataset: NormalizedDataset,
    strategy: Callable[[Decision], Signal],
    *,
    lookback: int = 1,
    profile: ExecutionProfile = ExecutionProfile(),
) -> BacktestResult:
    """Run a strategy on available candle closes and simulate fills on the market clock."""
    if type(lookback) is not int or lookback <= 0:
        raise ValueError("lookback must be a positive integer")
    _validate_profile(profile)
    candles = _validated_candles(dataset)
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        return _run(candles, dataset.dataset_id, strategy, lookback, profile)


def _run(
    candles: tuple[Candle, ...], dataset_id: str, strategy: Callable[[Decision], Signal],
    lookback: int, profile: ExecutionProfile,
) -> BacktestResult:
    ledger = _Ledger(profile)
    accounts = [ledger.snapshot(candles[0].open_time if candles else None, None)]
    pending = None
    rejections = []

    # Consume release events as of each close and mark from the newest visible candle.
    releases = iter(sorted(range(len(candles)), key=lambda index: candles[index].available_at))
    release = next(releases, None)
    last_visible = -1
    for index, candle in enumerate(candles):
        ledger.protect(candle, "open")
        if pending is not None:
            if reason := ledger.execute(pending, candle):
                rejections.append(OrderRejection(pending, reason))
            pending = None
        ledger.protect(candle, "intrabar")

        now = candle.close_time
        while release is not None and candles[release].available_at <= now:
            last_visible = max(last_visible, release)
            release = next(releases, None)
        mark = candles[last_visible].close if last_visible >= 0 else None
        account = ledger.snapshot(now, mark)
        accounts.append(account)

        start = index + 1 - lookback
        if start < 0:
            continue
        window = candles[start : index + 1]
        if any(bar.available_at > now for bar in window):
            continue
        # Keep strategy arithmetic from mutating the simulator's decimal context.
        decision = Decision(now, window, account, ledger.position)
        with localcontext():
            signal = strategy(decision)
        if not isinstance(signal, Signal) or not isinstance(signal.action, Action):
            raise ValueError("strategy must return a Signal with a valid Action")
        try:
            order = _prepare_order(decision, signal, profile)
        except ValueError as error:
            rejections.append(OrderRejection(Order(now, signal), str(error)))
            continue
        if signal.action is not Action.HOLD:
            pending = order

    if profile.liquidate_at_end and ledger.position is not None:
        assert last_visible >= 0
        last = candles[-1]
        ledger.liquidate(last, candles[last_visible].close)
        accounts[-1] = ledger.snapshot(last.close_time, None)

    peak = profile.initial_capital
    max_drawdown = _ZERO
    for point in accounts:
        peak = max(peak, point.equity)
        max_drawdown = min(max_drawdown, (point.equity - peak) / peak)
    total_return = (accounts[-1].equity - profile.initial_capital) / profile.initial_capital
    return BacktestResult(
        dataset_id, profile, tuple(ledger.fills), tuple(ledger.trades), tuple(accounts),
        ledger.position, tuple(rejections), pending, total_return, max_drawdown,
    )
