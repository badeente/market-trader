"""Built-in backtest strategies selected by their configuration names."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from market_trader._config import config_decimal, require_fields
from market_trader._dataset_codec import parse_json
from market_trader._numeric import mean_decimals, round_to_grid
from market_trader.backtesting import Action, Decision, ExecutionProfile, Signal
from market_trader.baseline import BASELINE_FORMAT, OPEN_BASELINE_FORMAT, load_ridge_model
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


def _sma_timed(
    entry: Signal, fast: int, slow: int, holding_bars: int, dataset: NormalizedDataset | None,
) -> Callable[[Decision], Signal]:
    """Bind the fixed holding clock; all trading/accounting stays in the engine."""
    if dataset is None:
        raise ValueError("sma_timed requires the evaluation dataset")
    interval = dataset.interval
    if not isinstance(interval, timedelta) or interval <= timedelta(0):
        raise ValueError("sma_timed requires a positive dataset interval")
    duration = interval * holding_bars
    calendar = dataset.config.calendar

    def decide(decision: Decision) -> Signal:
        window = decision.window
        if len(window) != slow:
            raise ValueError("sma_timed requires the exact slow_period lookback")
        if (decision.at != window[-1].close_time
                or any(bar.interval != interval or bar.instrument != dataset.instrument
                       or bar.close_time - bar.open_time != interval
                       or bar.available_at > decision.at for bar in window)):
            raise ValueError("sma_timed decision window does not match the bound dataset or close clock")
        if decision.position is not None:
            if decision.position.entry.side != "buy":
                return _HOLD
            exit_at = decision.position.entry.candle_open_time + duration
            return _EXIT if decision.at >= exit_at else _HOLD
        if isinstance(calendar, SessionCalendar):
            session = calendar.session_at(window[0].open_time)
            # No boundary entry: the next-open exit must actually exist in this session.
            if (session is None or decision.at + duration >= session.close_time
                    or any(calendar.session_at(bar.open_time) != session
                           or bar.close_time > session.close_time for bar in window)):
                return _HOLD
        if any(left.close_time != right.open_time for left, right in zip(window, window[1:])):
            raise ValueError("sma_timed decision window must contain consecutive candles")
        closes = tuple(candle.close for candle in window)
        return entry if mean_decimals(closes[-fast:]) > mean_decimals(closes) else _HOLD

    return decide


def _configure_sma_timed(table: dict, _price_tick: Decimal, _config_dir: Path) -> ConfiguredStrategy:
    require_fields(table, "strategy", "name", "fast_period", "slow_period", "holding_bars", "quantity")
    for field in ("fast_period", "slow_period", "holding_bars"):
        if type(table[field]) is not int or table[field] <= 0:
            raise ValueError(f"strategy.{field} must be a positive integer")
    fast, slow, holding = table["fast_period"], table["slow_period"], table["holding_bars"]
    if fast >= slow:
        raise ValueError("strategy.fast_period must be smaller than slow_period")
    quantity = _fixed_quantity(table["quantity"])
    parameters = {"fast_period": fast, "slow_period": slow,
                  "holding_bars": holding, "quantity": quantity}
    entry = Signal(Action.ENTER_LONG, quantity=quantity)
    return ConfiguredStrategy(
        "sma_timed", slow, tuple(parameters.items()),
        lambda dataset: _sma_timed(entry, fast, slow, holding, dataset),
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


def _learned_long(
    saved, threshold: Decimal, entry: Signal, dataset: NormalizedDataset | None,
    *, name: str = "ridge_long",
) -> Callable[[Decision], Signal]:
    if dataset is None:
        raise ValueError(f"{name} requires the evaluation dataset")
    model_name = name.removesuffix("_long")
    if dataset.instrument != saved.instrument:
        raise ValueError(f"{model_name} model instrument does not match the evaluation dataset")
    if dataset.interval != saved.interval:
        raise ValueError(f"{model_name} model candle interval does not match the evaluation dataset")
    if dataset.candles and dataset.candles[0].open_time <= saved.last_target_available_at:
        raise ValueError(f"{model_name} evaluation candles must start after the model training cutoff")
    duration = saved.interval * saved.config.horizon_bars
    calendar = dataset.config.calendar

    def decide(decision: Decision) -> Signal:
        if len(decision.window) != saved.config.lookback_bars:
            raise ValueError(f"{name} requires the exact model lookback")
        if decision.position is not None:
            exit_at = decision.position.entry.candle_open_time + duration
            return _EXIT if decision.at >= exit_at else _HOLD

        if isinstance(calendar, SessionCalendar):
            session = calendar.session_at(decision.window[0].open_time)
            # The exit order needs a later candle in the same session to fill at its open.
            if session is None or decision.at + duration >= session.close_time:
                return _HOLD
        prediction = (saved.predict_window(decision.window, decision_at=decision.at)
                      if name == "cnn_long" else saved.model.predict(candle_features(decision.window)))
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
        lambda dataset: _learned_long(saved, threshold, entry, dataset),
    )


def load_frozen_model(path: Path, *, expected_kind: str | None = None):
    """Dispatch only explicitly supported JSON artifacts; inference needs no Torch."""
    if expected_kind == 'cnn':
        from market_trader.image_neural import load_cnn_model
        return load_cnn_model(path)
    try:
        artifact = parse_json(path.read_text(encoding="utf-8"), allow_floats=True)
    except ValueError as error:
        raise ValueError(f"invalid model JSON: {error}") from None
    if not isinstance(artifact, dict):
        raise ValueError("model JSON must be an object")
    formats = {BASELINE_FORMAT: "ridge", OPEN_BASELINE_FORMAT: "ridge",
               "market-trader.mlp-validation.v1": "mlp",
               "market-trader.mlp-validation.v2": "mlp",
               "market-trader.cnn-validation.v1": "cnn"}
    artifact_format = artifact.get("format")
    if not isinstance(artifact_format, str):
        raise ValueError("model artifact format must be a string")
    kind = formats.get(artifact_format)
    if kind is None or (expected_kind is not None and kind != expected_kind):
        raise ValueError("unsupported model artifact format")
    if (not isinstance(artifact.get("model"), dict)
            or artifact["model"].get("kind") != kind):
        raise ValueError("model kind does not match the artifact format")
    if kind == "ridge":
        return load_ridge_model(path)
    if kind == "cnn":
        from market_trader.image_neural import load_cnn_model
        return load_cnn_model(path)
    from market_trader.training_data import OPEN_TARGET_NAME
    if artifact.get("target_name") != OPEN_TARGET_NAME:
        raise ValueError("MLP model target must be next-open to horizon-open")
    from market_trader.neural import load_mlp_model
    return load_mlp_model(path)


def _configure_mlp_long(table: dict, _price_tick: Decimal, config_dir: Path) -> ConfiguredStrategy:
    require_fields(table, "strategy", "name", "model_report", "entry_threshold", "quantity")
    model_report = table["model_report"]
    if not isinstance(model_report, str) or not model_report or model_report != model_report.strip():
        raise ValueError("strategy.model_report must be a nonempty path without padding")
    threshold = config_decimal(table["entry_threshold"], "strategy.entry_threshold")
    quantity = _fixed_quantity(table["quantity"])
    if threshold < 0:
        raise ValueError("strategy.entry_threshold must be nonnegative")
    saved = load_frozen_model(config_dir / model_report, expected_kind="mlp")
    entry = Signal(Action.ENTER_LONG, quantity=quantity)
    parameters = (("model_report_id", saved.report_id), ("entry_threshold", threshold),
                  ("quantity", quantity), ("horizon_bars", saved.config.horizon_bars))
    return ConfiguredStrategy("mlp_long", saved.config.lookback_bars, parameters,
                              lambda dataset: _learned_long(saved, threshold, entry, dataset, name="mlp_long"))


def _configure_cnn_long(table: dict, _price_tick: Decimal, config_dir: Path) -> ConfiguredStrategy:
    require_fields(table, "strategy", "name", "model_report", "entry_threshold", "quantity")
    model_report = table['model_report']
    if not isinstance(model_report, str) or not model_report or model_report != model_report.strip():
        raise ValueError('strategy.model_report must be a nonempty path without padding')
    threshold = config_decimal(table['entry_threshold'], 'strategy.entry_threshold')
    quantity = _fixed_quantity(table['quantity'])
    if threshold < 0:
        raise ValueError('strategy.entry_threshold must be nonnegative')
    saved = load_frozen_model(config_dir/model_report, expected_kind='cnn')
    entry = Signal(Action.ENTER_LONG, quantity=quantity)
    parameters = (('model_report_id', saved.report_id), ('entry_threshold', threshold),
                  ('quantity', quantity), ('horizon_bars', saved.config.horizon_bars))
    return ConfiguredStrategy('cnn_long', saved.config.lookback_bars, parameters,
                              lambda dataset: _learned_long(saved, threshold, entry, dataset, name='cnn_long'))


_REGISTRY = {
    "cnn_long": _configure_cnn_long,
    "mlp_long": _configure_mlp_long,
    "single_long": _configure_single_long,
    "sma_crossover": _configure_sma_crossover,
    "sma_timed": _configure_sma_timed,
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
