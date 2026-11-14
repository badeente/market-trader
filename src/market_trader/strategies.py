"""Built-in backtest strategies selected by their configuration names."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from market_trader._config import config_decimal, require_fields
from market_trader._numeric import mean_decimals, round_to_grid
from market_trader.backtesting import Action, Decision, ExecutionProfile, Signal
from market_trader.baseline import SavedRidgeModel, load_ridge_model
from market_trader.calendars import SessionCalendar
from market_trader.normalization import NormalizedDataset
from market_trader.training_data import candle_features

_HOLD = Signal(Action.HOLD)
_EXIT = Signal(Action.EXIT)
_RISK_FIELDS = ("risk_fraction", "stop_distance")
_PULLBACK_LOOKBACK = 4


@dataclass(frozen=True, slots=True)
class ConfiguredStrategy:
    """Validated parameters and a factory called once for each backtest."""

    name: str
    lookback: int
    parameters: tuple[tuple[str, int | Decimal | str], ...]
    _factory: Callable[[NormalizedDataset | None], Callable[[Decision], Signal]]

    def create(self, dataset: NormalizedDataset | None = None) -> Callable[[Decision], Signal]:
        """Bind a fresh strategy instance to the evaluation dataset when required."""
        return self._factory(dataset)


def _risk_fraction(value: object) -> Decimal:
    risk = config_decimal(value, "strategy.risk_fraction")
    if not 0 < risk <= 1:
        raise ValueError("strategy.risk_fraction must be greater than zero and at most one")
    return risk


def _risk_parameters(table: dict) -> dict[str, Decimal]:
    risk = _risk_fraction(table["risk_fraction"])
    distance = config_decimal(table["stop_distance"], "strategy.stop_distance")
    if not 0 < distance < 1:
        raise ValueError("strategy.stop_distance must be between zero and one")
    return {"risk_fraction": risk, "stop_distance": distance}


def _fixed_quantity(value: object) -> Decimal:
    quantity = config_decimal(value, "strategy.quantity")
    if quantity <= 0:
        raise ValueError("strategy.quantity must be positive")
    return quantity


def _single_long(entry: Signal) -> Callable[[Decision], Signal]:
    submitted = False

    def decide(decision: Decision) -> Signal:
        nonlocal submitted
        if submitted:
            return _HOLD
        submitted = True
        return entry

    return decide


def _configure_single_long(table: dict, _price_tick: Decimal, _config_dir: Path) -> ConfiguredStrategy:
    require_fields(table, "strategy", "name", *_RISK_FIELDS)
    parameters = _risk_parameters(table)
    entry = Signal(Action.ENTER_LONG, **parameters)
    return ConfiguredStrategy("single_long", 1, tuple(parameters.items()), lambda _: _single_long(entry))


def _sma_crossover(entry: Signal, fast: int, slow: int) -> Callable[[Decision], Signal]:
    def decide(decision: Decision) -> Signal:
        if len(decision.window) < slow + 1:
            raise ValueError("sma_crossover requires slow_period + 1 candles")
        closes = tuple(candle.close for candle in decision.window[-slow - 1 :])
        previous_fast = mean_decimals(closes[-fast - 1 : -1])
        previous_slow = mean_decimals(closes[:-1])
        current_fast = mean_decimals(closes[-fast:])
        current_slow = mean_decimals(closes[1:])
        position = decision.position
        if position is None and previous_fast <= previous_slow and current_fast > current_slow:
            return entry
        if (position is not None and position.entry.side == "buy"
                and previous_fast >= previous_slow and current_fast < current_slow):
            return _EXIT
        return _HOLD

    return decide


def _configure_sma_crossover(table: dict, _price_tick: Decimal, _config_dir: Path) -> ConfiguredStrategy:
    fixed_size = "quantity" in table
    sizing_fields = ("quantity",) if fixed_size else _RISK_FIELDS
    require_fields(table, "strategy", "name", "fast_period", "slow_period", *sizing_fields)
    fast, slow = table["fast_period"], table["slow_period"]
    if type(fast) is not int or fast <= 0:
        raise ValueError("strategy.fast_period must be a positive integer")
    if type(slow) is not int or slow <= 0:
        raise ValueError("strategy.slow_period must be a positive integer")
    if fast >= slow:
        raise ValueError("strategy.fast_period must be smaller than slow_period")
    sizing = {"quantity": _fixed_quantity(table["quantity"])} if fixed_size else _risk_parameters(table)
    parameters = {"fast_period": fast, "slow_period": slow, **sizing}
    entry = Signal(Action.ENTER_LONG, **sizing)
    return ConfiguredStrategy(
        "sma_crossover", slow + 1, tuple(parameters.items()), lambda _: _sma_crossover(entry, fast, slow),
    )


def _two_legged_pullback(
    *, risk_fraction: Decimal, movement_threshold: Decimal, bounce_threshold: Decimal,
    stop_buffer: Decimal, reward_multiple: Decimal, price_tick: Decimal,
) -> Callable[[Decision], Signal]:
    drop_factor = 1 - Fraction(movement_threshold)
    bounce_factor = 1 + Fraction(bounce_threshold)
    stop_factor = 1 - Fraction(stop_buffer)
    reward = Fraction(reward_multiple)

    def decide(decision: Decision) -> Signal:
        if len(decision.window) < _PULLBACK_LOOKBACK:
            raise ValueError("two_legged_pullback requires four candles")
        if decision.position is not None:
            return _HOLD
        peak, first_trough, bounce, second_trough = decision.window[-_PULLBACK_LOOKBACK:]
        first_drop = (
            first_trough.low < peak.low
            and Fraction(first_trough.low) < Fraction(peak.high) * drop_factor
        )
        rebound = (
            bounce.low >= first_trough.low
            and bounce.close > first_trough.close
            and bounce.high < peak.high
            and Fraction(bounce.high) > Fraction(first_trough.low) * bounce_factor
        )
        lower_low = second_trough.low < first_trough.low
        reference = Fraction(second_trough.close)
        net_decline = reference < Fraction(peak.close) * drop_factor
        if not (first_drop and rebound and lower_low and net_decline):
            return _HOLD

        stop = round_to_grid(Fraction(second_trough.low) * stop_factor, price_tick, upward=False)
        target = round_to_grid(reference + reward * (reference - Fraction(stop)), price_tick, upward=True)
        return Signal(Action.ENTER_LONG, stop=stop, target=target, risk_fraction=risk_fraction)

    return decide


def _configure_two_legged_pullback(table: dict, price_tick: Decimal, _config_dir: Path) -> ConfiguredStrategy:
    fields = ("movement_threshold", "bounce_threshold", "stop_buffer", "reward_multiple")
    require_fields(table, "strategy", "name", "risk_fraction", *fields)
    parameters = {key: config_decimal(table[key], f"strategy.{key}") for key in fields}
    parameters["risk_fraction"] = _risk_fraction(table["risk_fraction"])
    if not 0 < parameters["movement_threshold"] < 1:
        raise ValueError("strategy.movement_threshold must be between zero and one")
    if parameters["bounce_threshold"] <= 0:
        raise ValueError("strategy.bounce_threshold must be positive")
    if not 0 <= parameters["stop_buffer"] < 1:
        raise ValueError("strategy.stop_buffer must be nonnegative and less than one")
    if parameters["reward_multiple"] <= 0:
        raise ValueError("strategy.reward_multiple must be positive")
    if not isinstance(price_tick, Decimal) or not price_tick.is_finite() or price_tick <= 0:
        raise ValueError("execution.price_tick must be a positive finite Decimal")
    return ConfiguredStrategy(
        "two_legged_pullback", _PULLBACK_LOOKBACK, tuple(parameters.items()),
        lambda _: _two_legged_pullback(price_tick=price_tick, **parameters),
    )


def _ridge_long(
    saved: SavedRidgeModel, threshold: Decimal, entry: Signal, dataset: NormalizedDataset | None,
) -> Callable[[Decision], Signal]:
    if dataset is None:
        raise ValueError("ridge_long requires the evaluation dataset")
    if dataset.instrument != saved.instrument:
        raise ValueError("ridge model instrument does not match the evaluation dataset")
    if dataset.interval != saved.interval:
        raise ValueError("ridge model candle interval does not match the evaluation dataset")
    if dataset.candles and dataset.candles[0].open_time <= saved.last_target_available_at:
        raise ValueError("ridge evaluation candles must start after the model training cutoff")
    duration = saved.interval * saved.config.horizon_bars
    calendar = dataset.config.calendar

    def decide(decision: Decision) -> Signal:
        if len(decision.window) != saved.config.lookback_bars:
            raise ValueError("ridge_long requires the exact model lookback")
        if decision.position is not None:
            exit_at = decision.position.entry.candle_open_time + duration
            return _EXIT if decision.at >= exit_at else _HOLD

        if isinstance(calendar, SessionCalendar):
            session = calendar.session_at(decision.window[0].open_time)
            # The exit order needs a later candle in the same session to fill at its open.
            if session is None or decision.at + duration >= session.close_time:
                return _HOLD
        prediction = saved.model.predict(candle_features(decision.window))
        return entry if Decimal(str(prediction)) > threshold else _HOLD

    return decide


def _configure_ridge_long(table: dict, _price_tick: Decimal, config_dir: Path) -> ConfiguredStrategy:
    require_fields(table, "strategy", "name", "model_report", "entry_threshold", "quantity")
    model_report = table["model_report"]
    if not isinstance(model_report, str) or not model_report or model_report != model_report.strip():
        raise ValueError("strategy.model_report must be a nonempty path without padding")
    threshold = config_decimal(table["entry_threshold"], "strategy.entry_threshold")
    quantity = _fixed_quantity(table["quantity"])
    if threshold < 0:
        raise ValueError("strategy.entry_threshold must be nonnegative")
    saved = load_ridge_model(config_dir / model_report)
    entry = Signal(Action.ENTER_LONG, quantity=quantity)
    parameters = (
        ("model_report_id", saved.report_id), ("entry_threshold", threshold),
        ("quantity", quantity), ("horizon_bars", saved.config.horizon_bars),
    )
    return ConfiguredStrategy(
        "ridge_long", saved.config.lookback_bars, parameters,
        lambda dataset: _ridge_long(saved, threshold, entry, dataset),
    )


_REGISTRY = {
    "single_long": _configure_single_long,
    "sma_crossover": _configure_sma_crossover,
    "two_legged_pullback": _configure_two_legged_pullback,
    "ridge_long": _configure_ridge_long,
}


def configure_strategy(
    table: object, *, price_tick: Decimal = ExecutionProfile().price_tick,
    config_dir: Path = Path("."),
) -> ConfiguredStrategy:
    """Validate a strategy table and prepare a factory for independent runs."""
    if not isinstance(table, dict):
        raise ValueError("strategy must be a table")
    if "name" not in table:
        raise ValueError("missing strategy fields: name")
    name = table["name"]
    if not isinstance(name, str) or name not in _REGISTRY:
        raise ValueError(f"unknown strategy {name!r}")
    return _REGISTRY[name](table, price_tick, config_dir)
