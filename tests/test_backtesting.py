"""Hand-calculated acceptance cases for the first R0 backtest slice."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from market_trader.backtesting import Action, ExecutionProfile, Signal, run_backtest
from market_trader.domain import Candle
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.strategies import configure_strategy

START = datetime(2026, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
HOLD = Signal(Action.HOLD)
EXIT = Signal(Action.EXIT)
LONG = Signal(Action.ENTER_LONG)
FLAT = ("100", "101", "99", "100")


def candle(index, prices):
    open_time = START + index * MINUTE
    close_time = open_time + MINUTE
    return Candle(
        "TEST", MINUTE, open_time, close_time, close_time,
        *(Decimal(price) for price in prices), Decimal(1),
    )


def dataset(*candles):
    result, = normalize_candles(candles, config=NormalizationConfig(calendar="continuous")).datasets
    return result


def scripted(*signals):
    remaining = iter(signals)
    return lambda decision: next(remaining, HOLD)


@pytest.mark.case("ACC-001")
def test_acc_001_without_orders_account_stays_flat():
    data = dataset(candle(0, FLAT), candle(1, ("103", "110", "102", "109")))
    result = run_backtest(data, scripted())
    assert not result.fills
    assert not result.trades
    assert [point.equity for point in result.accounts] == [1000, 1000, 1000]
    assert result.account.balance == 1000
    assert result.total_return == 0


@pytest.mark.case("ACC-002")
def test_acc_002_long_profit_and_return_use_account_capital():
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("110", "111", "109", "110")))
    result = run_backtest(data, scripted(LONG, EXIT))
    assert [(fill.side, fill.price) for fill in result.fills] == [("buy", 100), ("sell", 110)]
    trade, = result.trades
    assert (trade.gross_pnl, trade.net_pnl) == (10, 10)
    assert result.account.realized_pnl == 10
    assert result.account.unrealized_pnl == 0
    assert result.account.balance == result.account.equity == 1010
    assert result.total_return == Decimal("0.01")


@pytest.mark.case("ACC-003")
@pytest.mark.case("EXE-003")
def test_acc_003_long_stop_loss_fills_on_the_trigger_candle():
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("100", "101", "97", "99")))
    result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, stop=Decimal(98))))
    entry, stop = result.fills
    assert (entry.price, stop.price) == (100, 98)
    assert (stop.reason, stop.phase, stop.candle_open_time) == ("stop", "intrabar", START + 2 * MINUTE)
    assert result.account.realized_pnl == -2
    assert result.account.balance == result.account.equity == 998
    assert result.total_return == Decimal("-0.002")


@pytest.mark.case("ACC-004")
@pytest.mark.parametrize(("bar", "pnl", "price", "reason"), [
    (("90", "91", "89", "90"), 10, 90, "exit"),
    pytest.param(("100", "103", "99", "101"), -2, 102, "stop", marks=pytest.mark.case("EXE-004")),
])
def test_acc_004_short_profit_and_stop_loss_have_the_correct_sign(bar, pnl, price, reason):
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, bar))
    short = Signal(Action.ENTER_SHORT, stop=Decimal(102))
    signal = EXIT if reason == "exit" else HOLD
    result = run_backtest(data, scripted(short, signal), profile=ExecutionProfile(allow_short=True))
    entry, exit_fill = result.fills
    assert (entry.side, entry.price) == ("sell", 100)
    assert (exit_fill.side, exit_fill.price, exit_fill.reason) == ("buy", price, reason)
    assert result.trades[0].gross_pnl == pnl
    assert result.account.realized_pnl == pnl
    assert result.account.equity == 1000 + pnl


@pytest.mark.case("ACC-005")
def test_acc_005_commission_is_charged_on_each_executed_notional_once():
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("110", "111", "109", "110")))
    profile = ExecutionProfile(commission_rate=Decimal("0.001"))
    with localcontext() as context:
        context.prec = 2
        result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, quantity=Decimal(2)), EXIT),
                              profile=profile)
    assert [fill.fee for fill in result.fills] == [Decimal("0.20"), Decimal("0.22")]
    assert result.accounts[2].balance == Decimal("999.80")
    trade, = result.trades
    assert trade.gross_pnl == 20
    assert trade.net_pnl == result.account.realized_pnl == Decimal("19.58")
    assert result.account.equity == Decimal("1019.58")


@pytest.mark.case("EXE-001")
@pytest.mark.case("ACC-006")
def test_exe_001_market_order_uses_next_open_and_default_keeps_position_open():
    data = dataset(candle(0, FLAT), candle(1, ("103", "106", "102", "105")))
    result = run_backtest(data, scripted(LONG, HOLD))
    entry, = result.fills
    assert (entry.candle_open_time, entry.phase, entry.price, entry.fee) == (START + MINUTE, "open", 103, 0)
    assert result.position.entry == entry
    assert not result.trades
    assert result.account.realized_pnl == 0
    assert result.account.unrealized_pnl == 2
    assert result.account.equity == 1002


@pytest.mark.case("EXE-002")
def test_exe_002_signal_on_last_close_remains_unfilled():
    result = run_backtest(dataset(candle(0, FLAT)), scripted(LONG))
    assert not result.fills
    assert result.unfilled_order.decision_at == START + MINUTE
    assert result.unfilled_order.signal == LONG


@pytest.mark.case("EXE-005")
def test_exe_005_stop_gap_fills_at_worse_open():
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("95", "97", "94", "96")))
    result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, stop=Decimal(98))))
    stop = result.fills[-1]
    assert (stop.reason, stop.phase, stop.price, stop.intrabar_ambiguous) == ("stop", "open", 95, False)
    assert result.account.realized_pnl == -5
    assert result.account.equity == 995


@pytest.mark.parametrize(("bar", "reason", "phase", "price", "ambiguous"), [
    pytest.param(("100", "105", "97", "101"), "stop", "intrabar", 98, True, marks=pytest.mark.case("EXE-006")),
    pytest.param(("100", "105", "99", "103"), "target", "intrabar", 104, False, marks=pytest.mark.case("EXE-007")),
    pytest.param(("105", "106", "97", "103"), "target", "open", 105, False, marks=pytest.mark.case("EXE-007")),
])
def test_protection_honors_open_then_uses_conservative_intrabar_order(bar, reason, phase, price, ambiguous):
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, bar))
    result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, stop=Decimal(98), target=Decimal(104))))
    _, exit_fill = result.fills
    assert (exit_fill.reason, exit_fill.phase, exit_fill.price, exit_fill.intrabar_ambiguous) == (
        reason, phase, price, ambiguous,
    )
    assert result.account.realized_pnl == price - 100


@pytest.mark.case("CAU-001")
@pytest.mark.case("CAU-002")
def test_cau_002_a_delayed_candle_blocks_the_window_until_its_availability():
    bars = [candle(index, FLAT) for index in range(2, 7)]
    bars[2] = replace(bars[2], available_at=START + 6 * MINUTE)
    decisions = []

    def strategy(decision):
        decisions.append((decision.at.minute, tuple(bar.close_time.minute for bar in decision.window)))
        return LONG if decision.at.minute == 6 else HOLD

    result = run_backtest(dataset(*bars), strategy, lookback=2)
    assert decisions == [(4, (3, 4)), (6, (5, 6)), (7, (6, 7))]
    entry, = result.fills
    assert entry.candle_open_time == START + 6 * MINUTE


@pytest.mark.case("CAU-002")
def test_cau_002_open_position_is_marked_only_with_an_available_close():
    delayed = replace(candle(1, ("100", "110", "99", "109")), available_at=START + 3 * MINUTE)
    data = dataset(candle(0, FLAT), delayed)
    result = run_backtest(data, scripted(LONG))
    assert result.position is not None
    assert result.account.equity == 1000


@pytest.mark.case("DAT-005")
def test_dat_005_runner_rejects_an_active_gap_before_invoking_strategy():
    data = dataset(candle(0, FLAT), candle(2, FLAT))
    with pytest.raises(ValueError, match="missing candles"):
        run_backtest(data, lambda decision: pytest.fail("invalid input reached the strategy"))


@pytest.mark.parametrize(("signal", "message"), [
    pytest.param(Signal(Action.ENTER_SHORT), "short", marks=pytest.mark.case("SIG-002")),
    pytest.param(Signal(Action.ENTER_LONG, quantity=Decimal(11)), "capital", marks=pytest.mark.case("RSK-003")),
])
def test_runner_records_a_rejection_without_changing_account(signal, message):
    data = dataset(candle(0, FLAT), candle(1, FLAT))
    result = run_backtest(data, scripted(signal))
    assert not result.fills
    assert result.account.equity == 1000
    rejection, = result.rejections
    assert rejection.order.signal == signal
    assert message in rejection.reason


@pytest.mark.case("ACC-007")
def test_acc_007_explicit_liquidation_fills_at_close_after_regular_orders():
    data = dataset(candle(0, FLAT), candle(1, ("100", "106", "99", "105")))
    result = run_backtest(data, scripted(LONG, EXIT), profile=ExecutionProfile(liquidate_at_end=True))
    _, exit_fill = result.fills
    assert (exit_fill.phase, exit_fill.reason, exit_fill.price) == ("close", "liquidation", 105)
    assert result.position is None
    assert result.account.realized_pnl == 5
    assert result.account.unrealized_pnl == 0
    assert result.account.equity == 1005
    assert result.unfilled_order.signal == EXIT


@pytest.mark.case("ACC-007")
@pytest.mark.case("CAU-002")
def test_liquidation_uses_the_last_available_close_without_backdating_the_fill():
    delayed = replace(candle(2, ("105", "111", "104", "110")), available_at=START + 4 * MINUTE)
    data = dataset(candle(0, FLAT), candle(1, ("100", "106", "99", "105")), delayed)
    result = run_backtest(data, scripted(LONG), profile=ExecutionProfile(liquidate_at_end=True))
    _, exit_fill = result.fills
    assert (exit_fill.candle_open_time, exit_fill.phase, exit_fill.price) == (START + 2 * MINUTE, "close", 105)
    assert result.account.equity == 1005


@pytest.mark.case("ACC-008")
def test_acc_008_drawdown_is_relative_to_running_peak():
    data = dataset(
        candle(0, FLAT), candle(1, ("100", "110", "99", "110")),
        candle(2, ("110", "111", "99", "99")), candle(3, ("99", "109", "98", "108.9")),
    )
    result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, quantity=Decimal(10))))
    assert [point.equity for point in result.accounts] == [1000, 1000, 1100, 990, 1089]
    assert result.total_return == Decimal("0.089")
    assert result.max_drawdown == Decimal("-0.1")


@pytest.mark.parametrize(("profile", "entry", "exit_price", "pnl"), [
    pytest.param(ExecutionProfile(slippage=Decimal("0.5")), "100.5", "109.5", "9",
                 marks=pytest.mark.case("EXE-008")),
    pytest.param(ExecutionProfile(spread=Decimal("0.2")), "100.1", "109.9", "9.8",
                 marks=pytest.mark.case("EXE-009")),
])
def test_execution_price_applies_costs_once_per_side(profile, entry, exit_price, pnl):
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("110", "111", "109", "110")))
    result = run_backtest(data, scripted(LONG, EXIT), profile=profile)
    assert [fill.reference_price for fill in result.fills] == [100, 110]
    assert [fill.price for fill in result.fills] == [Decimal(entry), Decimal(exit_price)]
    assert result.account.realized_pnl == Decimal(pnl)


@pytest.mark.case("EXE-007")
@pytest.mark.parametrize(("high", "should_fill"), [("104", False), ("104.2", True)])
def test_target_requires_an_executable_price_at_its_limit(high, should_fill):
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("100", high, "99", "103")))
    signal = Signal(Action.ENTER_LONG, target=Decimal(104))
    result = run_backtest(data, scripted(signal), profile=ExecutionProfile(spread=Decimal("0.4")))
    assert len(result.fills) == (2 if should_fill else 1)
    if should_fill:
        assert result.fills[-1].price == 104
        assert result.fills[-1].reason == "target"


@pytest.mark.parametrize(("step", "quantity", "risk"), [
    pytest.param("1", "5", "10", marks=pytest.mark.case("RSK-001")),
    pytest.param("0.3", "4.8", "9.6", marks=pytest.mark.case("RSK-002")),
])
def test_risk_sizing_floors_to_grid_without_exceeding_budget(step, quantity, risk):
    data = dataset(candle(0, FLAT), candle(1, FLAT))
    signal = Signal(Action.ENTER_LONG, stop=Decimal(98), risk_fraction=Decimal("0.01"))
    result = run_backtest(
        data, scripted(signal), profile=ExecutionProfile(quantity_step=Decimal(step), minimum_quantity=Decimal(step)),
    )
    fill, = result.fills
    assert fill.quantity == Decimal(quantity)
    assert fill.order.plan.stop == 98
    assert fill.order.plan.risk_budget == 10
    assert fill.order.plan.planned_stop_risk == Decimal(risk)


@pytest.mark.case("RSK-003")
@pytest.mark.case("RSK-004")
@pytest.mark.parametrize(("stop", "minimum", "diagnostic"), [
    (None, "0", "stop"),
    ("100", "0", "stop"),
    ("102", "0", "stop"),
    ("98", "6", "minimum"),
])
def test_risk_sizing_rejects_invalid_stop_or_below_minimum(stop, minimum, diagnostic):
    data = dataset(candle(0, FLAT), candle(1, FLAT))
    signal = Signal(Action.ENTER_LONG, stop=None if stop is None else Decimal(stop),
                    risk_fraction=Decimal("0.01"))
    result = run_backtest(data, scripted(signal), profile=ExecutionProfile(minimum_quantity=Decimal(minimum)))
    assert not result.fills
    rejection, = result.rejections
    assert diagnostic in rejection.reason
    assert result.account.equity == 1000


@pytest.mark.case("RSK-005")
@pytest.mark.parametrize(("distance", "stop"), [("0.005", "99.5"), ("0.0051", "99.4"), ("0.02", "98")])
def test_relative_stop_is_an_explicit_fraction_and_rounds_away_from_entry(distance, stop):
    safe = candle(1, ("100", "101", "100", "100"))
    data = dataset(candle(0, FLAT), safe)
    result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, stop_distance=Decimal(distance))))
    assert result.position.stop == Decimal(stop)


@pytest.mark.case("RSK-006")
def test_cost_adjusted_risk_plan_is_preserved_when_a_stop_gap_increases_loss():
    data = dataset(candle(0, FLAT), candle(1, FLAT), candle(2, ("95", "97", "94", "96")))
    profile = ExecutionProfile(commission_rate=Decimal("0.001"), slippage=Decimal("0.1"))
    signal = Signal(Action.ENTER_LONG, stop=Decimal(98), risk_fraction=Decimal("0.01"))
    result = run_backtest(data, scripted(signal), profile=profile)
    entry, stop = result.fills
    assert entry.quantity == 4
    assert entry.order.plan.planned_stop_risk == Decimal("9.592")
    assert entry.order.plan.risk_budget == 10
    assert stop.reference_price == 95
    assert stop.price == Decimal("94.9")
    assert result.account.realized_pnl == Decimal("-21.58")


@pytest.mark.case("EXE-007")
def test_short_target_waits_for_the_executable_ask():
    data = dataset(
        candle(0, FLAT), candle(1, FLAT), candle(2, ("100", "101", "96", "97")),
        candle(3, ("97", "99", "95.8", "97")),
    )
    profile = ExecutionProfile(allow_short=True, spread=Decimal("0.4"))
    short = Signal(Action.ENTER_SHORT, target=Decimal(96))
    result = run_backtest(data, scripted(short), profile=profile)
    _, fill = result.fills
    assert (fill.price, fill.reference_price, fill.candle_open_time) == (
        96, Decimal("95.8"), START + 3 * MINUTE,
    )


@pytest.mark.case("EXE-001")
def test_adverse_tick_rounding_preserves_small_off_grid_differences():
    above_tick = "100.0000000000000000000000000001"
    data = dataset(candle(0, FLAT), candle(1, (above_tick, "101", "99", "100")))
    result = run_backtest(data, scripted(LONG))
    fill, = result.fills
    assert fill.reference_price == Decimal(above_tick)
    assert fill.price == Decimal("100.1")


@pytest.mark.case("SIG-001")
@pytest.mark.case("ACC-009")
@pytest.mark.case("REP-001")
def test_winning_losing_and_open_trades_reconcile_with_fees():
    data = dataset(
        candle(0, FLAT), candle(1, FLAT), candle(2, ("110", "111", "109", "110")),
        candle(3, FLAT), candle(4, ("95", "96", "94", "95")),
        candle(5, ("100", "106", "99", "105")), candle(6, ("105", "106", "104", "105")),
    )
    profile = ExecutionProfile(commission_rate=Decimal("0.001"))
    signals = (LONG, EXIT, LONG, EXIT, LONG, HOLD, HOLD)
    result = run_backtest(data, scripted(*signals), profile=profile)
    assert result == run_backtest(data, scripted(*signals), profile=profile)
    assert [fill.reason for fill in result.fills] == ["entry", "exit", "entry", "exit", "entry"]
    assert [trade.net_pnl for trade in result.trades] == [Decimal("9.79"), Decimal("-5.195")]
    assert result.account.realized_pnl == Decimal("4.595")
    assert result.position.entry.fee == Decimal("0.1")
    assert result.account.unrealized_pnl == 5
    assert result.account.balance == Decimal("1004.495")
    assert result.account.equity == Decimal("1009.495")


@pytest.mark.case("CAU-004")
def test_strategy_errors_are_propagated_to_the_caller():
    def broken_strategy(decision):
        raise RuntimeError("feature computation failed")

    with pytest.raises(RuntimeError, match="feature computation failed"):
        run_backtest(dataset(candle(0, FLAT)), broken_strategy)


@pytest.mark.case("CAU-003")
def test_future_prices_do_not_change_signals_on_the_shared_prefix():
    common = (candle(0, FLAT), candle(1, FLAT))

    def signals_for(future):
        signals = []

        def strategy(decision):
            signal = LONG if decision.window[-1].close >= 100 else HOLD
            if decision.at <= common[-1].close_time:
                signals.append((decision.at, signal))
            return signal

        run_backtest(dataset(*common, future), strategy)
        return signals

    assert signals_for(candle(2, ("110", "112", "109", "111"))) == signals_for(
        candle(2, ("90", "92", "89", "91")),
    ) == [(START + MINUTE, LONG), (START + 2 * MINUTE, LONG)]


@pytest.mark.case("EXE-001")
def test_fixed_size_absolute_stop_is_validated_against_the_actual_open():
    data = dataset(candle(0, FLAT), candle(1, ("105", "106", "103", "104")))
    result = run_backtest(data, scripted(Signal(Action.ENTER_LONG, stop=Decimal(102))))
    fill, = result.fills
    assert fill.price == 105
    assert result.position.stop == 102


@pytest.mark.case("SIG-005")
def test_configured_strategy_creates_fresh_state_for_each_run():
    settings = configure_strategy({
        "name": "single_long", "risk_fraction": Decimal("0.01"), "stop_distance": Decimal("0.02"),
    })
    a = dataset(candle(0, FLAT), candle(1, FLAT))
    other = ("200", "201", "199", "200")
    b = dataset(candle(0, other), candle(1, other))
    first = run_backtest(a, settings.create(), lookback=settings.lookback)
    second = run_backtest(b, settings.create(), lookback=settings.lookback)
    separate = run_backtest(b, settings.create(), lookback=settings.lookback)
    assert [fill.price for fill in first.fills] == [100]
    assert [fill.price for fill in second.fills] == [200]
    assert second == separate


@pytest.mark.case("SIG-004")
@pytest.mark.parametrize(("sizing", "quantity"), [
    ({"risk_fraction": Decimal("0.02"), "stop_distance": Decimal("0.10")}, 1),
    ({"quantity": 2}, 2),
])
def test_crossover_enters_on_the_next_open_and_exits_on_the_down_cross(sizing, quantity):
    settings = configure_strategy({
        "name": "sma_crossover", "fast_period": 1, "slow_period": 2,
        **sizing,
    })
    data = dataset(
        candle(0, FLAT), candle(1, ("100", "101", "98", "99")),
        candle(2, ("99", "105", "98", "104")), candle(3, ("103", "106", "102", "105")),
        candle(4, ("105", "106", "100", "101")), candle(5, ("100", "102", "98", "99")),
    )
    result = run_backtest(data, settings.create(), lookback=settings.lookback)
    assert settings.lookback == 3
    assert [(fill.reason, fill.candle_open_time, fill.price) for fill in result.fills] == [
        ("entry", START + 3 * MINUTE, 103),
        ("exit", START + 5 * MINUTE, 100),
    ]
    assert [fill.quantity for fill in result.fills] == [quantity, quantity]
    assert (result.fills[0].order.plan.stop is None) == ("quantity" in sizing)
    assert result.position is None


@pytest.mark.case("SIG-004")
@pytest.mark.parametrize(("sizing", "diagnostic"), [
    ({"quantity": 0}, "quantity must be positive"),
    ({"quantity": True}, "quantity must be a finite TOML number"),
    ({"quantity": 1, "risk_fraction": Decimal("0.01"), "stop_distance": Decimal("0.02")},
     "unexpected strategy fields"),
])
def test_sma_fixed_quantity_rejects_invalid_or_mixed_sizing(sizing, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        configure_strategy({"name": "sma_crossover", "fast_period": 1, "slow_period": 2, **sizing})


@pytest.fixture
def pullback_parameters():
    return {
        "name": "two_legged_pullback",
        "risk_fraction": Decimal("0.01"),
        "movement_threshold": Decimal("0.003"),
        "bounce_threshold": Decimal("0.002"),
        "stop_buffer": Decimal("0.002"),
        "reward_multiple": 2,
    }


@pytest.fixture
def pullback_bars():
    return (
        candle(0, ("103", "104", "102", "103")),
        candle(1, ("103", "103", "100", "100.5")),
        candle(2, ("100.5", "102", "100.2", "101.5")),
        candle(3, ("101.5", "102", "99", "100")),
        candle(4, ("100.2", "101", "99.5", "100.4")),
    )


@pytest.mark.case("SIG-004")
@pytest.mark.parametrize(("last_prices", "reason", "price", "equity"), [
    (("100.4", "103", "100", "102.8"), "target", "102.4", "1017.6"),
    (("98", "100", "97", "98.5"), "stop", "98", "982.4"),
])
def test_pullback_entry_and_protection_are_executed_by_r0(
    pullback_parameters, pullback_bars, last_prices, reason, price, equity,
):
    settings = configure_strategy(pullback_parameters)
    data = dataset(*pullback_bars, candle(5, last_prices))
    result = run_backtest(data, settings.create(), lookback=settings.lookback)
    assert settings.lookback == 4
    entry, exit_fill = result.fills
    assert entry.order.decision_at == START + 4 * MINUTE
    assert entry.candle_open_time == START + 4 * MINUTE
    assert (entry.price, entry.quantity) == (Decimal("100.2"), 8)
    assert entry.order.signal.stop == Decimal("98.8")
    assert entry.order.signal.target == Decimal("102.4")
    assert entry.order.plan.planned_stop_risk == Decimal("9.6")
    assert (exit_fill.reason, exit_fill.price) == (reason, Decimal(price))
    assert result.account.equity == Decimal(equity)
    assert result.position is None


@pytest.mark.case("SIG-004")
@pytest.mark.parametrize(("index", "prices"), [
    (0, ("103", "104", "100", "103")),  # No lower first trough.
    (2, ("100.5", "102", "99.5", "101.5")),  # Bounce candle itself breaks the first low.
    (2, ("100.5", "104", "100.2", "101.5")),  # Bounce reaches the first peak.
    (2, ("100.5", "102", "100.2", "100.5")),  # Bounce fails to close above the first trough candle.
    (3, ("101.5", "102", "100", "100")),  # Second trough ties the first low.
    (3, ("101.5", "103", "99", "103")),  # No net decline in closing prices.
])
def test_pullback_holds_when_a_pattern_condition_is_missing(pullback_parameters, pullback_bars, index, prices):
    bars = list(pullback_bars)
    bars[index] = candle(index, prices)
    settings = configure_strategy(pullback_parameters)
    result = run_backtest(dataset(*bars), settings.create(), lookback=settings.lookback)
    assert result.fills == result.rejections == ()
    assert result.unfilled_order is None


@pytest.mark.case("SIG-004")
def test_pullback_rejects_a_bounce_that_only_equals_the_threshold(pullback_parameters, pullback_bars):
    settings = configure_strategy({**pullback_parameters, "bounce_threshold": Decimal("0.02")})
    result = run_backtest(dataset(*pullback_bars), settings.create(), lookback=settings.lookback)
    assert result.fills == result.rejections == ()
    assert result.unfilled_order is None


@pytest.mark.case("CAU-002")
def test_pullback_cannot_signal_from_a_delayed_second_trough(pullback_parameters, pullback_bars):
    bars = list(pullback_bars)
    bars[3] = replace(bars[3], available_at=START + 5 * MINUTE)
    settings = configure_strategy(pullback_parameters)
    result = run_backtest(dataset(*bars), settings.create(), lookback=settings.lookback)
    assert result.fills == result.rejections == ()
    assert result.unfilled_order is None
