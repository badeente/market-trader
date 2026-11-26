"""AP8 synthetic diagnostics and real shared-ledger replay, never real inputs."""
import json
import math
import sys
from datetime import UTC, datetime, timedelta

import pytest


def analysis():
    from market_trader import robustness_analysis
    return robustness_analysis


def point(at="2024-01-02T08:00:00+00:00", predicted=2.0, actual=1.0):
    return dict(at=at, predicted=predicted, actual=actual)


def test_slices_calendar_clock_and_empty_metrics():
    rows = [point(), point("2024-04-02T10:00:00Z", 0.0, 3.0),
            point("2024-10-28T14:00:00+00:00", 4.0, 0.0)]
    result = analysis().forecast_slices(rows)
    assert result["format"] == "market-trader.ap8-slices.v1"
    assert result["timezone"] == "Europe/Berlin"
    assert result["overall"] == dict(row_count=3, mae=8/3, rmse=math.sqrt(26/3),
                                     zero_mae=4/3, zero_rmse=math.sqrt(10/3))
    assert [r["key"] for r in result["months"]] == [f"2024-{m:02}" for m in range(1, 13)]
    assert [r["key"] for r in result["quarters"]] == [f"2024-Q{q}" for q in range(1, 5)]
    assert [r["key"] for r in result["sessions"]] == ["09:00-12:00", "12:00-15:00", "15:00-17:30"]
    for key in ("months", "quarters", "sessions"):
        assert sum(r["row_count"] for r in result[key]) == 3
    assert [r["row_count"] for r in result["sessions"]] == [1, 1, 1]
    assert result["months"][1] == dict(key="2024-02", row_count=0, mae=None, rmse=None,
                                      zero_mae=None, zero_rmse=None)
    assert result == analysis().forecast_slices(rows)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("rows", [
    [], (), None, [point()]*50001, [point(), point()],
    [point("2024-04-02T10:00:00Z"), point()],
    [point(predicted=True)], [point(actual=False)], [point(predicted="1")],
    [point(actual=float("nan"))], [point(predicted=float("inf"))],
    [point("2025-01-02T08:00:00Z")], [point("2023-12-29T08:00:00Z")],
    [point("2024-01-02T08:00:00")], [point("2024-01-02T09:00:00+01:00")],
    [point("2024-01-02T07:59:59Z")], [point("2024-01-02T16:30:00Z")],
    [point("2024-01-06T08:00:00Z")], [point("not a date")],
    [dict(at=True, predicted=1., actual=1.)], [dict(at=point()["at"], predicted=1.)],
    [dict(**point(), future=1.)], [None], [point(predicted=1e308, actual=-1e308)],
])
def test_slices_reject_invalid_rows(rows):
    with pytest.raises(ValueError):
        analysis().forecast_slices(rows)


def test_slice_clock_boundaries_cover_every_eligible_row():
    # DST transition: UTC 07 becomes Berlin 09, UTC 10 becomes Berlin 12.
    rows = [point("2024-03-28T08:00:00Z"), point("2024-04-02T07:00:00Z"),
            point("2024-04-02T09:59:59Z"), point("2024-04-02T10:00:00Z"),
            point("2024-04-02T12:59:59Z"), point("2024-04-02T13:00:00Z"),
            point("2024-04-02T15:29:59Z")]
    result = analysis().forecast_slices(rows)
    assert [r["row_count"] for r in result["sessions"]] == [3, 2, 2]
    assert [r["row_count"] for r in result["quarters"]] == [1, 6, 0, 0]
    assert [r["row_count"] for r in result["months"]][2:4] == [1, 6]


@pytest.fixture
def cohort(tmp_path):
    from decimal import Decimal as D
    from market_trader.artifacts import DatasetProvenance
    from market_trader.backtesting import ExecutionProfile
    from market_trader.calendars import SessionCalendar, TradingSession
    from market_trader.domain import Candle
    from market_trader.normalization import NormalizationConfig, normalize_candles
    from market_trader.storage import save_dataset
    starts = [datetime(2024, 1, 2, 8, tzinfo=UTC), datetime(2024, 4, 2, 7, tzinfo=UTC)]
    interval = timedelta(minutes=15)
    sessions = tuple(TradingSession(at, at+34*interval) for at in starts)
    calendar = SessionCalendar("synthetic Berlin", starts[0], sessions[-1].close_time, sessions)
    candles = tuple(Candle("deuidxeur", interval, at+i*interval, at+(i+1)*interval,
                           at+(i+1)*interval, D(100+i), D(101+i), D(99+i), D(100+i), D(1))
                    for at in starts for i in range(34))
    data, = normalize_candles(candles, config=NormalizationConfig(calendar)).datasets
    path = tmp_path / "synthetic-validation.json"
    manifest = save_dataset(path, data, provenance=DatasetProvenance("synthetic 2024 only"))
    expected = tuple(at+i*interval for at in starts for i in range(8, 32))
    points = [point(at.isoformat(), .01, .02) for at in expected]
    profile = ExecutionProfile(initial_capital=D(100000), spread=D(".06"), slippage=D(".01"),
                               commission_rate=D(".001"), price_tick=D(".1"), minimum_quantity=D(1))
    return path, data, manifest, expected, points, profile


def replay(cohort, **kwargs):
    path, data, manifest, expected, points, profile = cohort
    options = dict(expected_decisions=expected, model_report_id="sha256:"+"a"*64, base_profile=profile)
    options.update(kwargs)
    return analysis().cost_backtests(path, points, **options)


def test_cost_replay_shared_ledger_factor_one_parity(cohort):
    from dataclasses import asdict, replace
    from decimal import Decimal as D
    from types import SimpleNamespace
    from market_trader.backtesting import Action, Signal, run_backtest
    from market_trader.reference import trading_summary
    from market_trader.reference_pipeline import _json_value
    from market_trader.reporting import BACKTEST_REPORT_FORMAT
    from market_trader.strategies import _learned_long
    path, data, manifest, expected, points, profile = cohort
    saved = SimpleNamespace(instrument=data.instrument, interval=data.interval,
                            last_target_available_at=datetime(2023, 12, 29, tzinfo=UTC),
                            config=SimpleNamespace(lookback_bars=8, horizon_bars=2),
                            model=SimpleNamespace(predict=lambda features: .01))
    observed = []
    for (scenario, report, summary), factor in zip(replay(cohort), (D(".5"), D(1), D(2)), strict=True):
        observed.append(scenario)
        scaled = replace(profile, spread=profile.spread*factor, slippage=profile.slippage*factor,
                         commission_rate=profile.commission_rate*factor)
        direct = run_backtest(data, _learned_long(saved, D(0), Signal(Action.ENTER_LONG, quantity=D(1)), data),
                              lookback=8, profile=scaled)
        raw = json.loads(json.dumps(asdict(direct), default=_json_value))
        for key in ("fills", "trades", "accounts", "position", "unfilled_order", "rejections",
                    "total_return", "max_drawdown", "dataset_id"):
            assert report[key] == raw[key]
        assert report["execution"] == raw["profile"]
        assert "profile" not in report
        assert report["format"] == BACKTEST_REPORT_FORMAT
        assert report["dataset_artifact_id"] == manifest.artifact_id
        assert report["provenance"] == json.loads(json.dumps(asdict(manifest.provenance)))
        assert report["runtime"]["python"]
        assert report["strategy"]["name"] == "ap8_frozen_forecasts_long"
        assert report["strategy"]["model_report_id"] == "sha256:"+"a"*64
        assert report["strategy"]["lookback"] == 8
        assert report["strategy"]["horizon_bars"] == 2
        assert D(str(report["strategy"]["entry_threshold"])) == 0
        assert D(str(report["strategy"]["quantity"])) == 1
        assert report["strategy"]["scenario"] == scenario
        assert D(str(report["strategy"]["factor"])) == factor
        assert report["strategy"]["forecast_digest"].startswith("sha256:")
        assert summary == trading_summary(report)
        assert summary["trade_count"] == 16
        assert sum(m["trade_count"] for m in summary["months"]) == 16
        assert [m["trade_count"] for m in summary["months"]][0:4] == [8, 0, 0, 8]
        for trade in report["trades"]:
            entry, exit = trade["entry"], trade["exit"]
            assert entry["order"]["signal"]["stop"] is None
            assert entry["order"]["signal"]["target"] is None
            assert datetime.fromisoformat(exit["candle_open_time"])-datetime.fromisoformat(entry["candle_open_time"]) == 2*data.interval
        json.dumps((report, summary), allow_nan=False)
    assert observed == ["half", "baseline", "double"]


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "order", "actual_bool", "nonfinite",
                                  "expected_list", "expected_duplicate", "expected_2025", "expected_naive",
                                  "cohort_shift", "cohort_early", "cohort_session_exit", "model_id"])
def test_cost_invalid_cohort_fails_before_ledger(cohort, monkeypatch, case):
    path, data, manifest, expected, points, profile = cohort
    options = {}
    if case == "missing": points = points[:-1]
    if case == "extra": points = points+[point((expected[-1]+data.interval).isoformat())]
    if case == "duplicate": points = points+[points[-1]]
    if case == "order": points = list(reversed(points))
    if case == "actual_bool": points = [dict(p, actual=True) for p in points]
    if case == "nonfinite": points = [dict(p, predicted=float("nan")) for p in points]
    if case == "expected_list": options["expected_decisions"] = list(expected)
    if case == "expected_duplicate": options["expected_decisions"] = expected+(expected[-1],)
    if case == "expected_2025": options["expected_decisions"] = (datetime(2025, 1, 2, 8, tzinfo=UTC),)
    if case == "expected_naive": options["expected_decisions"] = tuple(at.replace(tzinfo=None) for at in expected)
    if case.startswith("cohort_"):
        at = (expected[0]+timedelta(seconds=1) if case == "cohort_shift" else
              data.candles[0].open_time+data.interval if case == "cohort_early" else
              data.config.calendar.sessions[0].close_time-2*data.interval)
        points, options["expected_decisions"] = [point(at.isoformat())], (at,)
    if case == "model_id": options["model_report_id"] = "sha256:bogus"
    altered = path, data, manifest, expected, points, profile
    def no_engine(*args, **kwargs):
        pytest.fail("invalid cohort reached execution")
    monkeypatch.setattr(analysis(), "run_backtest", no_engine)
    with pytest.raises(ValueError):
        next(replay(altered, **options))


@pytest.mark.parametrize("field,value", [
    ("spread", -1), ("spread", True), ("slippage", "sNaN"),
    ("commission_rate", "NaN"), ("price_tick", 0), ("quantity_step", "2"),
    ("minimum_quantity", "2"), ("initial_capital", "Infinity"),
    ("allow_short", True), ("liquidate_at_end", True),
])
def test_cost_invalid_profile_fails_before_ledger(cohort, monkeypatch, field, value):
    from dataclasses import replace
    from decimal import Decimal
    profile = replace(cohort[-1], **{field: Decimal(value) if isinstance(value, str) or type(value) is int else value})
    def no_engine(*args, **kwargs):
        pytest.fail("invalid profile reached execution")
    monkeypatch.setattr(analysis(), "run_backtest", no_engine)
    with pytest.raises(ValueError):
        next(replay(cohort, base_profile=profile))


@pytest.mark.parametrize("case", ["instrument", "interval", "continuous", "2025_extra", "2023_extra",
                                  "2025_availability", "out_of_research", "coverage_2025"])
def test_dataset_scope_rejected_before_execution(cohort, tmp_path, monkeypatch, case):
    from dataclasses import replace
    from market_trader.artifacts import DatasetProvenance
    from market_trader.calendars import TradingSession
    from market_trader.normalization import NormalizationConfig, normalize_candles
    from market_trader.storage import save_dataset
    path, data, manifest, expected, points, profile = cohort
    bars, calendar = data.candles, data.config.calendar
    if case == "instrument":
        bars = tuple(replace(b, instrument="other") for b in bars)
    if case == "interval":
        bars = tuple(replace(b, interval=timedelta(minutes=30), close_time=b.close_time+data.interval,
                             available_at=b.available_at+data.interval) for b in bars[::2])
        expected = tuple(s.open_time+i*timedelta(minutes=30) for s in calendar.sessions for i in range(8, 15))
        points = [point(at.isoformat()) for at in expected]
    if case == "continuous":
        # One session avoids an unrelated continuous-calendar gap.
        bars, calendar = bars[:34], "continuous"
        expected, points = expected[:24], points[:24]
    if case in ("2025_extra", "2023_extra"):
        extra_at = datetime(2025, 1, 2, 8, tzinfo=UTC) if case == "2025_extra" else datetime(2023, 12, 29, 8, tzinfo=UTC)
        shift = extra_at-bars[0].open_time
        extra = tuple(replace(b, open_time=b.open_time+shift, close_time=b.close_time+shift,
                              available_at=b.available_at+shift) for b in bars[:34])
        bars = tuple(sorted((*bars, *extra), key=lambda b: b.open_time))
        sessions = tuple(sorted((*calendar.sessions, TradingSession(extra_at, extra_at+34*data.interval)),
                                key=lambda s: s.open_time))
        calendar = replace(calendar, coverage_start=sessions[0].open_time,
                           coverage_end=sessions[-1].close_time, sessions=sessions)
    if case == "2025_availability":
        bars = (*bars[:-1], replace(bars[-1], available_at=datetime(2025, 1, 2, tzinfo=UTC)))
        # Drop rows depending on this delayed target; still reject the dataset itself.
        expected, points = expected[:-1], points[:-1]
    if case == "out_of_research":
        session = calendar.sessions[0]
        calendar = replace(calendar, coverage_start=session.open_time-timedelta(hours=1),
                           sessions=(replace(session, open_time=session.open_time-timedelta(hours=1)), *calendar.sessions[1:]))
    if case == "coverage_2025":
        calendar = replace(calendar, coverage_end=datetime(2025, 1, 3, tzinfo=UTC))
    altered, = normalize_candles(bars, config=NormalizationConfig(calendar)).datasets
    target = tmp_path / "bad-scope.json"
    save_dataset(target, altered, provenance=DatasetProvenance("synthetic scope rejection"))
    def no_engine(*args, **kwargs):
        pytest.fail("invalid dataset scope reached execution")
    monkeypatch.setattr(analysis(), "run_backtest", no_engine)
    with pytest.raises(ValueError):
        next(replay((target, altered, manifest, expected, points, profile)))


def test_actual_targets_cannot_change_any_replay_output(cohort):
    path, data, manifest, expected, points, profile = cohort
    altered = [dict(p, actual=(-1000.0 if i % 2 else 1000.0)) for i, p in enumerate(points)]
    for original, changed in zip(replay(cohort), replay((path, data, manifest, expected, altered, profile)), strict=True):
        assert original == changed  # Includes frozen-input digest, every fill/account and summary.
    iterator = replay(cohort)
    next(iterator)
    assert "points" not in iterator.gi_frame.f_locals
    assert "rows" not in iterator.gi_frame.f_locals
    assert all(type(value) is float for value in iterator.gi_frame.f_locals["forecasts"].values())
    iterator.close()


def test_strategy_reads_only_current_clock_and_prioritizes_timed_exit(cohort):
    from decimal import Decimal
    from market_trader.backtesting import Action, Decision, run_backtest
    path, data, manifest, expected, points, profile = cohort
    seen = []
    class LookupOnly:
        def get(self, at):
            seen.append(at)
            return .01 if at in expected else None
    strategy = analysis()._frozen_strategy(data, LookupOnly())
    direct = run_backtest(data, strategy, lookback=8, profile=profile)
    assert len(direct.trades) == 16
    # Entry clocks only; while positioned the strategy exits/holds without reading forecasts.
    assert seen == [trade.entry.order.decision_at for trade in direct.trades]
    assert set(seen) <= set(expected)
    trade = direct.trades[0]
    window = data.candles[:8]
    account = direct.accounts[0]
    # A missing clock does not fall back to a previous/future forecast.
    missing = expected[0]+timedelta(seconds=1)
    signal = analysis()._frozen_strategy(data, {})(Decision(missing, window, account, None))
    assert signal.action is Action.HOLD
    for prediction in (0., -.01):
        assert analysis()._frozen_strategy(data, {expected[0]: prediction})(Decision(expected[0], window, account, None)).action is Action.HOLD
    # Session boundary disallows entry even with a positive cached prediction.
    last_at = data.config.calendar.sessions[0].close_time-2*data.interval
    assert analysis()._frozen_strategy(data, {last_at: 1.})(Decision(last_at, data.candles[24:32], account, None)).action is Action.HOLD
    # Exit eligibility survives a missing forecast and wins at the scheduled horizon.
    from market_trader.backtesting import Position
    position = Position(trade.entry, None, None)
    exit_at = trade.entry.candle_open_time+2*data.interval
    signal = analysis()._frozen_strategy(data, {})(Decision(exit_at, window, account, position))
    assert signal.action is Action.EXIT
    with pytest.raises(ValueError):
        strategy(Decision(expected[0], window[:-1], account, None))


def test_tick_rounding_costs_are_reexecuted_not_proportional(cohort):
    from decimal import Decimal as D
    outcomes = list(replay(cohort))
    # All three raw offsets (.02/.04/.08) round adversely to the same .1 grid step.
    prices = [[fill["price"] for fill in report["fills"]] for _, report, _ in outcomes]
    assert prices[0] == prices[1] == prices[2]
    assert all(D(report["fills"][0]["price"]) == D("108.1") for _, report, _ in outcomes)
    costs = [D(summary["modeled_cost_points"]) for _, _, summary in outcomes]
    assert costs[0] < costs[1] < costs[2]
    assert costs[0]*2 != costs[1]
    assert costs[1]*2 != costs[2]
    for _, report, summary in outcomes:
        assert D(summary["reference_points"]) - D(summary["modeled_cost_points"]) == D(summary["net_points"])
        assert D(report["accounts"][-1]["equity"])-D(report["execution"]["initial_capital"]) == D(summary["net_points"])


def test_generator_releases_previous_ledger_before_next_simulation(cohort, monkeypatch):
    module = analysis()
    real_engine = module.run_backtest
    calls = []
    iterator = replay(cohort)
    def observe(*args, **kwargs):
        assert "report" not in iterator.gi_frame.f_locals
        assert "summary" not in iterator.gi_frame.f_locals
        assert "result" not in iterator.gi_frame.f_locals
        calls.append(kwargs["profile"])
        return real_engine(*args, **kwargs)
    monkeypatch.setattr(module, "run_backtest", observe)
    assert calls == []
    for expected_scenario in ("half", "baseline", "double"):
        scenario, report, summary = next(iterator)
        assert scenario == expected_scenario
        assert summary["trade_count"] == 16
        del report, summary
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(calls) == 3


def test_cohort_subset_holds_all_other_eligible_clocks(cohort):
    # expected_decisions is the verified source cohort, not inferred from signals/positions.
    path, data, manifest, expected, points, profile = cohort
    subset = (expected[1], expected[-1])
    selected = [point(at.isoformat(), 1.0, -1.0) for at in subset]
    for _, report, summary in replay((path, data, manifest, subset, selected, profile)):
        assert summary["trade_count"] == 2
        assert [datetime.fromisoformat(t["entry"]["order"]["decision_at"]) for t in report["trades"]] == list(subset)


def test_50000_point_bound_is_inclusive_and_finite():
    at = datetime(2024, 1, 2, 8, tzinfo=UTC)
    points = [point((at+timedelta(milliseconds=i)).isoformat(), .125, .25) for i in range(50000)]
    result = analysis().forecast_slices(points)
    assert result["overall"] == dict(row_count=50000, mae=.125, rmse=.125, zero_mae=.25, zero_rmse=.25)
    for group in ("months", "quarters", "sessions"):
        assert sum(r["row_count"] for r in result[group]) == 50000


def test_large_finite_forecasts_produce_finite_metrics():
    result = analysis().forecast_slices([point(predicted=1e200, actual=0.0)])
    assert result["overall"]["mae"] == 1e200
    assert result["overall"]["rmse"] == 1e200
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("count", [3, 12])
@pytest.mark.parametrize("predicted,actual,expected_mae,expected_zero_mae", [
    (sys.float_info.max, 0.0, sys.float_info.max, 0.0),
    (0.0, sys.float_info.max, sys.float_info.max, sys.float_info.max),
    # Perfect forecasts isolate the zero comparator's aggregation from MAE.
    (sys.float_info.max, sys.float_info.max, 0.0, sys.float_info.max),
], ids=["max-error", "max-target", "zero-comparator-only"])
def test_repeated_max_finite_forecasts_have_representable_metrics(
        count, predicted, actual, expected_mae, expected_zero_mae):
    at = datetime(2024, 1, 2, 8, tzinfo=UTC)
    rows = [point((at+timedelta(seconds=i)).isoformat(), predicted, actual)
            for i in range(count)]
    result = analysis().forecast_slices(rows)
    expected = dict(row_count=count, mae=expected_mae, rmse=expected_mae,
                    zero_mae=expected_zero_mae, zero_rmse=expected_zero_mae)
    assert result["overall"] == expected
    for group in ("months", "quarters", "sessions"):
        assert sum(row["row_count"] for row in result[group]) == count
        for row in result[group]:
            if row["row_count"]:
                assert {key: row[key] for key in expected} == expected
    json.dumps(result, allow_nan=False)


def test_cost_scaling_is_independent_of_callers_decimal_context(cohort):
    from dataclasses import replace
    from decimal import Decimal, localcontext, ROUND_DOWN
    profile = replace(cohort[-1], spread=Decimal(".06123456789"), slippage=Decimal(".0123456789"),
                      commission_rate=Decimal(".00123456789"))
    regular = list(replay(cohort, base_profile=profile))
    with localcontext() as context:
        context.prec, context.rounding = 4, ROUND_DOWN
        altered = list(replay(cohort, base_profile=profile))
        assert context.prec == 4 and context.rounding == ROUND_DOWN
    assert altered == regular


def test_factor_one_matches_verified_saved_ridge_strategy_with_nonconstant_forecasts(cohort, tmp_path):
    from dataclasses import asdict
    from decimal import Decimal
    from hashlib import sha256
    from market_trader._dataset_codec import json_bytes
    from market_trader.baseline import OPEN_BASELINE_FORMAT, load_ridge_model
    from market_trader.backtesting import run_backtest
    from market_trader.reference_pipeline import _json_value
    from market_trader.strategies import configure_strategy
    from market_trader.training_data import FEATURE_NAMES, OPEN_TARGET_NAME, candle_features
    path, data, manifest, expected, points, profile = cohort
    body = dict(format=OPEN_BASELINE_FORMAT, dataset_id=data.dataset_id,
                dataset_artifact_id=manifest.artifact_id, training_data_id="sha256:"+"b"*64,
                instrument="deuidxeur", interval_microseconds=900000000,
                feature_names=list(FEATURE_NAMES), target_name=OPEN_TARGET_NAME,
                config=dict(lookback_bars=8, horizon_bars=2, validation_start="2024-01-01T00:00:00+00:00",
                            test_start="2025-01-01T00:00:00+00:00"),
                model=dict(kind="ridge", alpha=.1, means=[0.]*4, scales=[1.]*4,
                           intercept=-.0025, coefficients=[.3, 0., 0., 0.]),
                training=dict(row_count=1, first_decision_at="2023-12-29T10:00:00+00:00",
                              last_target_available_at="2023-12-29T10:45:00+00:00"), validation={})
    model_id = "sha256:"+sha256(json_bytes(body)).hexdigest()
    model_path = tmp_path / "frozen-synthetic-ridge.json"
    model_path.write_bytes(json_bytes(dict(body, report_id=model_id)))
    saved = load_ridge_model(model_path)
    by_at = {bar.close_time: i for i, bar in enumerate(data.candles)}
    frozen = [point(at.isoformat(), saved.model.predict(candle_features(data.candles[by_at[at]-7:by_at[at]+1])), -123.)
              for at in expected]
    assert any(p["predicted"] > 0 for p in frozen) and any(p["predicted"] < 0 for p in frozen)
    configured = configure_strategy(dict(name="ridge_long", model_report=model_path.name,
                                         entry_threshold=Decimal(0), quantity=Decimal(1)), config_dir=tmp_path)
    direct = run_backtest(data, configured.create(data), lookback=8, profile=profile)
    assert direct.trades
    raw = json.loads(json.dumps(asdict(direct), default=_json_value))
    iterator = replay((path, data, manifest, expected, frozen, profile), model_report_id=model_id)
    next(iterator)
    scenario, baseline, summary = next(iterator)
    assert scenario == "baseline"
    for key in raw.keys()-{"profile"}:
        assert baseline[key] == raw[key]
    assert baseline["execution"] == raw["profile"]
    iterator.close()


def test_inference_and_real_replay_do_not_import_optional_frameworks(cohort):
    from dataclasses import asdict
    import subprocess
    import sys
    path, data, manifest, expected, points, profile = cohort
    script = '''
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy'):
            raise AssertionError('optional framework import: '+fullname)
sys.meta_path.insert(0, Block())
from datetime import datetime
from decimal import Decimal
from market_trader.backtesting import ExecutionProfile
from market_trader.robustness_analysis import forecast_slices, cost_backtests
points, raw_profile = json.loads(sys.argv[2]), json.loads(sys.argv[3])
profile = ExecutionProfile(**{k: Decimal(v) if isinstance(v, str) else v for k,v in raw_profile.items()})
assert forecast_slices(points)['overall']['row_count'] == len(points)
expected = tuple(datetime.fromisoformat(p['at']) for p in points)
for scenario, report, summary in cost_backtests(sys.argv[1], points, expected_decisions=expected,
    model_report_id='sha256:'+'a'*64, base_profile=profile):
    assert summary['trade_count'] == 16
    json.dumps((report, summary), allow_nan=False)
assert not any(n.split('.')[0] in ('torch', 'numpy') for n in sys.modules)
print('stdlib-only: three real ledger scenarios validated')
'''
    result = subprocess.run([sys.executable, "-c", script, str(path), json.dumps(points),
                             json.dumps(asdict(profile), default=str)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stdlib-only: three real ledger scenarios validated"


def test_baseline_preserves_exact_source_costs_not_context_rounded(cohort):
    from dataclasses import replace
    from decimal import Decimal
    profile = replace(cohort[-1], spread=Decimal(".061234567890123456789012345678901"))
    iterator = replay(cohort, base_profile=profile)
    next(iterator)
    _, report, _ = next(iterator)
    assert report["execution"]["spread"] == str(profile.spread)
    iterator.close()


def test_zero_predictions_hold_and_return_valid_empty_trade_ledgers(cohort):
    path, data, manifest, expected, points, profile = cohort
    frozen = [dict(p, predicted=0.) for p in points]
    for _, report, summary in replay((path, data, manifest, expected, frozen, profile)):
        assert not report["trades"] and not report["fills"]
        assert report["position"] is None and report["unfilled_order"] is None
        assert summary["trade_count"] == 0
        assert summary["win_fraction"] is None
        assert all(m["trade_count"] == 0 for m in summary["months"])


def test_delayed_history_is_not_an_eligible_source_decision(cohort, tmp_path, monkeypatch):
    from dataclasses import replace
    from market_trader.artifacts import DatasetProvenance
    from market_trader.normalization import normalize_candles
    from market_trader.storage import save_dataset
    path, data, manifest, expected, points, profile = cohort
    bars = list(data.candles)
    bars[7] = replace(bars[7], available_at=expected[0]+data.interval)
    delayed, = normalize_candles(bars, config=data.config).datasets
    target = tmp_path / "delayed.json"
    save_dataset(target, delayed, provenance=DatasetProvenance("synthetic delayed availability"))
    def no_engine(*args, **kwargs):
        pytest.fail("unavailable source history reached ledger")
    monkeypatch.setattr(analysis(), "run_backtest", no_engine)
    with pytest.raises(ValueError, match="ineligible"):
        next(replay((target, delayed, manifest, expected, points, profile)))
