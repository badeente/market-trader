"""Causal feature rows and chronological splits for offline supervised training."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, localcontext

from market_trader._numeric import mean_decimals
from market_trader.calendars import SessionCalendar
from market_trader.data import rolling_windows
from market_trader.domain import Candle
from market_trader.normalization import NormalizedDataset

TRAINING_DATA_FORMAT = "market-trader.training-data.v1"
FEATURE_NAMES = ("return_1", "window_return", "range_fraction", "close_vs_mean")
TARGET_NAME = "forward_return_next_open_to_horizon_close"
OPEN_TARGET_NAME = "forward_return_next_open_to_horizon_open"
OPEN_TRAINING_DATA_FORMAT = "market-trader.training-data.v2"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Set lookback, forward horizon, and chronological decision cutoffs."""

    lookback_bars: int
    horizon_bars: int
    validation_start: datetime
    test_start: datetime

    def __post_init__(self) -> None:
        if type(self.lookback_bars) is not int or self.lookback_bars < 2:
            raise ValueError("lookback_bars must be an integer of at least 2")
        if type(self.horizon_bars) is not int or self.horizon_bars < 1:
            raise ValueError("horizon_bars must be a positive integer")
        for field in ("validation_start", "test_start"):
            timestamp = getattr(self, field)
            if not isinstance(timestamp, datetime) or timestamp.utcoffset() != timedelta(0):
                raise ValueError(f"{field} must be a timezone-aware UTC timestamp")
        if self.validation_start >= self.test_start:
            raise ValueError("validation_start must precede test_start")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    decision_at: datetime
    entry_at: datetime
    exit_at: datetime
    target_available_at: datetime
    features: tuple[Decimal, ...]
    target_return: Decimal


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    config: TrainingConfig
    feature_names: tuple[str, ...]
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]


def candle_features(history: tuple[Candle, ...]) -> tuple[Decimal, ...]:
    """Compute the model features from an eligible, ordered candle window."""
    first, previous, last = history[0], history[-2], history[-1]
    mean = mean_decimals(tuple(candle.close for candle in history))
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)):
        return (
            (last.close - previous.close) / previous.close,
            (last.close - first.close) / first.close,
            (last.high - last.low) / last.close,
            (last.close - mean) / mean,
        )


def build_training_data(dataset: NormalizedDataset, config: TrainingConfig) -> TrainingDataset:
    """Build full, same-session windows with targets purged at split cutoffs."""
    if dataset.quality.gaps:
        raise ValueError("training data requires a candle series without active gaps")
    window_size = config.lookback_bars + config.horizon_bars
    windows = rolling_windows(dataset.candles, size=window_size)
    if any(candle.open <= 0 or candle.close <= 0 for candle in dataset.candles):
        raise ValueError("training data requires positive open and close prices")
    calendar = dataset.config.calendar
    splits: dict[str, list[TrainingExample]] = {"train": [], "validation": [], "test": []}

    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)):
        for window in windows:
            history, future = window[:config.lookback_bars], window[config.lookback_bars:]
            first, entry, exit_bar = history[0], future[0], future[-1]
            if isinstance(calendar, SessionCalendar):
                session = calendar.session_at(first.open_time)
                if session is None or exit_bar.close_time > session.close_time:
                    continue
            if exit_bar.close_time - first.open_time != window_size * first.interval:
                raise ValueError("training windows cannot contain active gaps")

            decision_at = history[-1].close_time
            if any(candle.available_at > decision_at for candle in history):
                continue
            target_available_at = max(candle.available_at for candle in future)
            if decision_at < config.validation_start:
                if target_available_at >= config.validation_start:
                    continue
                split = "train"
            elif decision_at < config.test_start:
                if target_available_at >= config.test_start:
                    continue
                split = "validation"
            else:
                split = "test"
            splits[split].append(TrainingExample(
                decision_at=decision_at,
                entry_at=entry.open_time,
                exit_at=exit_bar.close_time,
                target_available_at=target_available_at,
                features=candle_features(history),
                target_return=(exit_bar.close - entry.open) / entry.open,
            ))

    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"{split} split has no eligible examples")
    return TrainingDataset(
        config, FEATURE_NAMES, tuple(splits["train"]), tuple(splits["validation"]), tuple(splits["test"]),
    )


def build_open_training_data(
    dataset: NormalizedDataset, config: TrainingConfig, *, include_test: bool = True,
) -> TrainingDataset:
    """Build next-open to horizon-open labels, including the exit bar's availability."""
    if dataset.quality.gaps:
        raise ValueError("training data requires a candle series without active gaps")
    window_size = config.lookback_bars + config.horizon_bars + 1
    windows = rolling_windows(dataset.candles, size=window_size)
    if any(candle.open <= 0 or candle.close <= 0 for candle in dataset.candles):
        raise ValueError("training data requires positive open and close prices")
    calendar = dataset.config.calendar
    splits: dict[str, list[TrainingExample]] = {"train": [], "validation": [], "test": []}
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)):
        for window in windows:
            history, future = window[:config.lookback_bars], window[config.lookback_bars:]
            first, entry, exit_bar = history[0], future[0], future[config.horizon_bars]
            decision_at = history[-1].close_time
            if not include_test and decision_at >= config.test_start:
                continue
            if isinstance(calendar, SessionCalendar):
                session = calendar.session_at(first.open_time)
                if session is None or exit_bar.close_time > session.close_time:
                    continue
            if exit_bar.close_time - first.open_time != window_size * first.interval:
                raise ValueError("training windows cannot contain active gaps")
            decision_at = history[-1].close_time
            if any(candle.available_at > decision_at for candle in history):
                continue
            target_available_at = max(candle.available_at for candle in future)
            if decision_at < config.validation_start:
                if target_available_at >= config.validation_start:
                    continue
                split = "train"
            elif decision_at < config.test_start:
                if target_available_at >= config.test_start:
                    continue
                split = "validation"
            else:
                split = "test"
            splits[split].append(TrainingExample(
                decision_at=decision_at, entry_at=entry.open_time, exit_at=exit_bar.open_time,
                target_available_at=target_available_at, features=candle_features(history),
                target_return=(exit_bar.open - entry.open) / entry.open,
            ))
    for split, rows in splits.items():
        if not rows and (split != "test" or include_test):
            raise ValueError(f"{split} split has no eligible examples")
    return TrainingDataset(
        config, FEATURE_NAMES, tuple(splits["train"]), tuple(splits["validation"]), tuple(splits["test"]),
    )
