"""Synthetic AP1 timed-SMA contracts; no market data or model fitting."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

import pytest

from market_trader.backtesting import (
    AccountSnapshot, Action, Decision, ExecutionProfile, Fill, Position, Signal, run_backtest,
)
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.domain import Candle
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.strategies import configure_strategy

START = datetime(2024, 1, 2, tzinfo=UTC)
INTERVAL = timedelta(minutes=15)


def at(index):
    return START + index * INTERVAL


def bars(closes):
    return tuple(
        Candle("TEST", INTERVAL, at(i), at(i + 1), at(i + 1),
               Decimal(close), Decimal(close), Decimal(close), Decimal(close), Decimal(1))
        for i, close in enumerate(closes)
    )


def dataset(candles, calendar: SessionCalendar | Literal["continuous"] = "continuous"):
    result, = normalize_candles(candles, config=NormalizationConfig(calendar)).datasets
    return result


def configured(**overrides):
    return configure_strategy({"name": "sma_timed", "fast_period": 4, "slow_period": 8,
                               "holding_bars": 2, "quantity": 1, **overrides})


def decision(window, *, position=None, when=None):
    return Decision(window[-1].close_time if when is None else when, tuple(window),
                    AccountSnapshot(None, Decimal(1000), Decimal(1000), Decimal(0), Decimal(0)),
                    position)


def test_registry_uses_slow_lookback_and_level_signal_not_crossover():
    source = bars(range(100, 109))
    config = configured()
    assert config.name == "sma_timed" and config.lookback == 8
    assert dict(config.parameters) == {"fast_period": 4, "slow_period": 8,
                                       "holding_bars": 2, "quantity": Decimal(1)}
    strategy = config.create(dataset(source))
    # Fast was already above slow in the preceding window: no crossover is required.
    assert strategy(decision(source[:8])) == Signal(Action.ENTER_LONG, quantity=Decimal(1))
    assert strategy(decision(source[1:])) == Signal(Action.ENTER_LONG, quantity=Decimal(1))
    assert config.create(dataset(source)) is not strategy


@pytest.mark.parametrize("closes", [[100] * 8, list(range(108, 100, -1))])
def test_equal_or_lower_fast_mean_holds(closes):
    source = bars(closes)
    assert configured().create(dataset(source))(decision(source)).action is Action.HOLD


@pytest.mark.parametrize(("offset", "expected"), [(1, Action.HOLD), (2, Action.EXIT), (3, Action.EXIT)])
def test_exit_uses_entry_open_clock_not_signal_or_sma(offset, expected):
    source = bars(range(100, 112))
    strategy = configured().create(dataset(source))
    position = Position(Fill(at(8), "open", "buy", "entry", Decimal(1), Decimal(108),
                             Decimal(0), Decimal(108)), None, None)
    window = source[offset:offset + 8]
    query = decision(window, position=position)
    assert strategy(query).action is expected
    assert strategy(query) == strategy(query)  # Pure: repeated evaluation does not advance a counter.


def test_engine_fills_next_open_at_same_regular_clock_and_charges_costs():
    source = bars(range(100, 112))
    data = dataset(source)
    config = configured()
    profile = ExecutionProfile(spread=Decimal("2"), slippage=Decimal("0.1"),
                               commission_rate=Decimal("0.001"))
    result = run_backtest(data, config.create(data), lookback=config.lookback, profile=profile)
    trade, = result.trades
    assert [fill.candle_open_time for fill in (trade.entry, trade.exit)] == [at(8), at(10)]
    assert [fill.order.decision_at for fill in (trade.entry, trade.exit)] == [at(8), at(10)]
    assert [fill.phase for fill in (trade.entry, trade.exit)] == ["open", "open"]
    assert [fill.price for fill in (trade.entry, trade.exit)] == [Decimal("109.1"), Decimal("108.9")]
    assert [fill.fee for fill in (trade.entry, trade.exit)] == [Decimal("0.1091"), Decimal("0.1089")]
    assert trade.net_pnl == Decimal("-0.418")
    assert result == run_backtest(data, config.create(data), lookback=config.lookback, profile=profile)


@pytest.mark.parametrize("field", ["fast_period", "slow_period", "holding_bars"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2", Decimal(2), None])
def test_rejects_non_positive_integer_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        configured(**{field: value})


@pytest.mark.parametrize("fast,slow", [(8, 8), (9, 8)])
def test_fast_must_be_smaller_than_slow(fast, slow):
    with pytest.raises(ValueError, match="smaller"):
        configured(fast_period=fast, slow_period=slow)


@pytest.mark.parametrize("quantity", [True, False, 0, -1, "1", None, float("inf"), Decimal("NaN")])
def test_invalid_fixed_quantity(quantity):
    with pytest.raises(ValueError, match="quantity"):
        configured(quantity=quantity)


@pytest.mark.parametrize("extra", ["risk_fraction", "stop_distance", "stop", "target", "unknown"])
def test_rejects_additional_parameters(extra):
    with pytest.raises(ValueError, match="unexpected strategy fields"):
        configured(**{extra: Decimal("0.1")})


@pytest.mark.parametrize("missing", ["fast_period", "slow_period", "holding_bars", "quantity"])
def test_requires_every_parameter(missing):
    table = {"name": "sma_timed", "fast_period": 4, "slow_period": 8,
             "holding_bars": 2, "quantity": 1}
    del table[missing]
    with pytest.raises(ValueError, match="missing strategy fields"):
        configure_strategy(table)


def test_factory_freezes_parameters_not_the_mutable_input_table():
    table = {"name": "sma_timed", "fast_period": 4, "slow_period": 8,
             "holding_bars": 2, "quantity": 1}
    config = configure_strategy(table)
    table["holding_bars"] = 99
    source = bars(range(100, 112))
    data = dataset(source)
    result = run_backtest(data, config.create(data), lookback=config.lookback)
    assert result.trades[0].exit.candle_open_time == at(10)


def test_ap1_full_session_has_exactly_same_24_allowed_decision_windows():
    source = bars(range(100, 134))
    calendar = SessionCalendar("research", at(0), at(34), (TradingSession(at(0), at(34)),))
    strategy = configured().create(dataset(source, calendar))
    allowed = [end for end in range(8, 35)
               if strategy(decision(source[end - 8:end])).action is Action.ENTER_LONG]
    assert allowed == list(range(8, 32))


def test_entire_lookback_must_belong_to_same_session():
    source = bars(range(100, 124))
    calendar = SessionCalendar("adjacent", at(0), at(24),
                               (TradingSession(at(0), at(12)), TradingSession(at(12), at(24))))
    strategy = configured().create(dataset(source, calendar))
    for end in range(12, 20):
        assert strategy(decision(source[end - 8:end])).action is Action.HOLD
    assert strategy(decision(source[12:20])).action is Action.ENTER_LONG


def test_regular_sessions_close_positions_before_overnight_without_boundary_entries():
    first = bars(range(100, 112))
    second = tuple(replace(bar, open_time=bar.open_time + timedelta(days=1),
                           close_time=bar.close_time + timedelta(days=1),
                           available_at=bar.available_at + timedelta(days=1)) for bar in first)
    calendar = SessionCalendar("two days", at(0), at(12) + timedelta(days=1), (
        TradingSession(at(0), at(12)),
        TradingSession(at(0) + timedelta(days=1), at(12) + timedelta(days=1)),
    ))
    data = dataset(first + second, calendar)
    config = configured()
    result = run_backtest(data, config.create(data), lookback=config.lookback)
    assert [(t.entry.candle_open_time, t.exit.candle_open_time) for t in result.trades] == [
        (at(8), at(10)), (at(8) + timedelta(days=1), at(10) + timedelta(days=1)),
    ]
    assert result.position is None and result.unfilled_order is None


def test_delayed_availability_exits_at_first_eligible_close():
    source = list(bars(range(100, 113)))
    source[9] = replace(source[9], available_at=at(11))
    data = dataset(source)
    config = configured()
    result = run_backtest(data, config.create(data), lookback=config.lookback)
    assert result.trades[0].entry.candle_open_time == at(8)
    assert result.trades[0].exit.candle_open_time == at(11)


def test_foreign_short_is_not_traded_by_long_only_strategy():
    source = bars(range(100, 112))
    position = Position(Fill(at(8), "open", "sell", "entry", Decimal(1), Decimal(108),
                             Decimal(0), Decimal(108)), None, None)
    assert configured().create(dataset(source))(
        decision(source[2:10], position=position),
    ).action is Action.HOLD


@pytest.mark.parametrize("quantity", [2, Decimal("1.25"), Decimal("0.5")])
def test_valid_quantity_is_fixed_without_protection_or_risk_sizing(quantity):
    source = bars(range(100, 108))
    assert configured(quantity=quantity).create(dataset(source))(decision(source)) == Signal(
        Action.ENTER_LONG, quantity=Decimal(str(quantity)),
    )


@pytest.mark.parametrize("interval", [timedelta(minutes=1), timedelta(hours=1)])
def test_holding_clock_uses_bound_interval_not_hardcoded_fifteen_minutes(interval):
    source = tuple(replace(bar, interval=interval, open_time=START + i * interval,
                           close_time=START + (i + 1) * interval,
                           available_at=START + (i + 1) * interval)
                   for i, bar in enumerate(bars(range(100, 111))))
    data = dataset(source)
    config = configured()
    result = run_backtest(data, config.create(data), lookback=config.lookback)
    trade, = result.trades
    assert trade.exit.candle_open_time - trade.entry.candle_open_time == 2 * interval


def test_factory_requires_dataset_interval_binding():
    with pytest.raises(ValueError, match="evaluation dataset"):
        configured().create()


@pytest.mark.parametrize("interval", [timedelta(0), -INTERVAL, None])
def test_factory_rejects_invalid_bound_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        configured().create(replace(dataset(bars(range(100, 112))), interval=interval))


@pytest.mark.parametrize("length", [7, 9])
def test_decision_requires_exact_slow_lookback(length):
    source = bars(range(100, 112))
    with pytest.raises(ValueError, match="lookback"):
        configured().create(dataset(source))(decision(source[:length]))


@pytest.mark.parametrize("corruption", ["interval", "instrument", "gap", "future", "clock"])
def test_rejects_decision_windows_incompatible_with_binding(corruption):
    source = bars(range(100, 112))
    window = list(source[:8])
    when = at(8)
    if corruption == "interval":
        window[0] = replace(window[0], interval=INTERVAL * 2)
    elif corruption == "instrument":
        window[0] = replace(window[0], instrument="OTHER")
    elif corruption == "gap":
        window[0] = replace(window[0], open_time=at(-1), close_time=at(0))
    elif corruption == "future":
        window[0] = replace(window[0], available_at=at(9))
    else:
        when = at(9)
    with pytest.raises(ValueError, match="window"):
        configured().create(dataset(source))(decision(window, when=when))
