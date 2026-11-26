"""Bounded AP8A 2024 development diagnostics; never independent test metrics."""
from collections.abc import Iterator
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

from market_trader._dataset_codec import json_bytes
from market_trader.backtesting import Action, ExecutionProfile, Signal, _validate_profile, run_backtest
from market_trader.calendars import SessionCalendar
from market_trader.reference import trading_summary
from market_trader.reporting import BACKTEST_REPORT_FORMAT
from market_trader.storage import load_dataset

_BERLIN = ZoneInfo("Europe/Berlin")
_BINS = ("09:00-12:00", "12:00-15:00", "15:00-17:30")


def _utc_2024(at):
    if (not isinstance(at, datetime) or at.utcoffset() != timedelta(0)
            or at.year != 2024):
        raise ValueError("expected a UTC 2024 timestamp")
    local = at.astimezone(_BERLIN)
    clock = (local.hour, local.minute, local.second, local.microsecond)
    if local.weekday() >= 5 or not (9, 0, 0, 0) <= clock < (17, 30, 0, 0):
        raise ValueError("timestamp is outside Berlin weekday research bins")
    return at.astimezone(UTC)


def _validated_points(points):
    if type(points) is not list or not 1 <= len(points) <= 50000:
        raise ValueError("points must be a nonempty list of at most 50000 rows")
    rows = []
    previous = None
    for point in points:
        if type(point) is not dict or set(point) != {"at", "predicted", "actual"}:
            raise ValueError("forecast rows require exactly at, predicted, actual")
        text = point["at"]
        if (not isinstance(text, str) or re.fullmatch(
                r"2024-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", text) is None):
            raise ValueError("forecast timestamp must be an ISO UTC 2024 string")
        at = _utc_2024(datetime.fromisoformat(text))
        if previous is not None and at <= previous:
            raise ValueError("forecasts must be unique and chronological")
        values = []
        for field in ("predicted", "actual"):
            value = point[field]
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("forecast values must be finite numbers, not bools")
            values.append(float(value))
        if not math.isfinite(values[0] - values[1]):
            raise ValueError("forecast error must be finite")
        rows.append((at, *values))
        previous = at
    return rows


def _metrics(rows):
    if not rows:
        return dict(row_count=0, mae=None, rmse=None, zero_mae=None, zero_rmse=None)
    count = len(rows)
    errors = [abs(row[1] - row[2]) for row in rows]
    zeros = [abs(row[2]) for row in rows]
    # Scale before aggregating: finite errors need not have finite float sums/squares.
    scale, zero_scale = max(errors), max(zeros)
    # Bound roundoff in each normalized mean before rescaling at float max.
    return dict(row_count=count,
                mae=scale*min(1.0, math.fsum(x/scale for x in errors)/count) if scale else 0.0,
                rmse=scale*math.sqrt(math.fsum((x/scale)**2/count for x in errors)) if scale else 0.0,
                zero_mae=zero_scale*min(1.0, math.fsum(x/zero_scale for x in zeros)/count) if zero_scale else 0.0,
                zero_rmse=zero_scale*math.sqrt(math.fsum((x/zero_scale)**2/count for x in zeros)) if zero_scale else 0.0)


def forecast_slices(points) -> dict:
    """Score fixed calendar and Berlin decision-time bins on all forecast rows."""
    rows = [(at.astimezone(_BERLIN), predicted, actual)
            for at, predicted, actual in _validated_points(points)]
    return dict(format="market-trader.ap8-slices.v1", timezone="Europe/Berlin",
                overall=_metrics(rows),
                months=[dict(key=f"2024-{m:02}", **_metrics([r for r in rows if r[0].month == m]))
                        for m in range(1, 13)],
                quarters=[dict(key=f"2024-Q{q}", **_metrics([r for r in rows if (r[0].month-1)//3+1 == q]))
                          for q in range(1, 5)],
                sessions=[dict(key=key, **_metrics([r for r in rows if
                          (0 if r[0].hour < 12 else 1 if r[0].hour < 15 else 2) == i]))
                          for i, key in enumerate(_BINS)])


def _frozen_strategy(dataset, forecasts):
    """Bind only (UTC decision, prediction), never a scoring target or row."""
    duration = dataset.interval * 2
    calendar = dataset.config.calendar
    hold, exit_signal = Signal(Action.HOLD), Signal(Action.EXIT)
    entry = Signal(Action.ENTER_LONG, quantity=Decimal(1))

    def decide(decision):
        if len(decision.window) != 8:
            raise ValueError("ap8_frozen_forecasts_long requires lookback8")
        if decision.position is not None:
            exit_at = decision.position.entry.candle_open_time + duration
            return exit_signal if decision.at >= exit_at else hold
        session = calendar.session_at(decision.window[0].open_time)
        if session is None or decision.at + duration >= session.close_time:
            return hold
        prediction = forecasts.get(decision.at)
        return entry if prediction is not None and prediction > 0 else hold

    return decide


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.name
    raise TypeError(f"unsupported AP8 report value: {type(value).__name__}")


def _validate_dataset_scope(dataset):
    calendar = dataset.config.calendar
    start, end = datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
    if (dataset.instrument != "deuidxeur" or dataset.interval != timedelta(minutes=15)
            or not isinstance(calendar, SessionCalendar)):
        raise ValueError("AP8 requires deuidxeur m15 with a SessionCalendar")
    if not start <= calendar.coverage_start < calendar.coverage_end <= end:
        raise ValueError("calendar coverage must be within 2024")
    if not 1 <= len(dataset.candles) <= 50000 or dataset.quality.gaps:
        raise ValueError("AP8 requires a bounded nonempty dataset without active gaps")
    for session in calendar.sessions:
        _utc_2024(session.open_time)
        _utc_2024(session.close_time-timedelta(microseconds=1))
        if session.open_time.astimezone(_BERLIN).date() != session.close_time.astimezone(_BERLIN).date():
            raise ValueError("research session must stay within one Berlin date")
    for bar in dataset.candles:
        _utc_2024(bar.open_time)
        _utc_2024(bar.close_time-timedelta(microseconds=1))
        if not start <= bar.available_at < end:
            raise ValueError("candle availability must stay within 2024")


def _eligible_decisions(dataset):
    """Check open-target window eligibility, without features or scoring labels."""
    calendar = dataset.config.calendar
    if not isinstance(calendar, SessionCalendar):
        raise ValueError("AP8 requires a SessionCalendar")
    eligible = set()
    bars = dataset.candles
    for index in range(8, len(bars)-2):
        window = bars[index-8:index+3]
        at = window[7].close_time
        session = calendar.session_at(window[0].open_time)
        if (session is None or window[-1].close_time > session.close_time
                or window[-1].close_time-window[0].open_time != 11*dataset.interval
                or any(bar.available_at > at for bar in window[:8])
                or any(bar.available_at.year != 2024 for bar in window[8:])):
            continue
        eligible.add(at)
    return eligible


def _validate_inputs(rows, expected_decisions, model_report_id, base_profile):
    if type(expected_decisions) is not tuple or not 1 <= len(expected_decisions) <= 50000:
        raise ValueError("expected_decisions must be a bounded nonempty tuple")
    previous = None
    for at in expected_decisions:
        _utc_2024(at)
        if previous is not None and at <= previous:
            raise ValueError("expected decisions must be unique and chronological")
        previous = at
    if tuple(row[0] for row in rows) != expected_decisions:
        raise ValueError("forecast decisions must exactly match the verified source cohort")
    if not isinstance(model_report_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", model_report_id) is None:
        raise ValueError("model_report_id must be a lowercase sha256 identifier")
    if not isinstance(base_profile, ExecutionProfile):
        raise ValueError("base_profile must be the source ExecutionProfile")
    _validate_profile(base_profile)
    if (base_profile.allow_short or base_profile.liquidate_at_end
            or Fraction(1) % Fraction(base_profile.quantity_step)
            or base_profile.minimum_quantity > 1):
        raise ValueError("AP8 requires long-only quantity1 with timed exits, not liquidation")


def cost_backtests(dataset_path, points, *, expected_decisions, model_report_id,
                   base_profile: ExecutionProfile) -> Iterator[tuple[str, dict, dict]]:
    """Yield half/baseline/double reports and validated summaries, one at a time.

    The caller must verify model/source lineage, supply the saved source execution
    profile and the exact source validation decision tuple, and reconcile baseline
    against its reference ledger before accepting comparisons. Here every forecast
    must match that cohort exactly and every cohort clock must be dataset-eligible;
    other clocks hold. No model is loaded/fitted and no scoring target is captured.
    Validation runs on first iteration, before any execution ledger is created.
    """
    rows = _validated_points(points)
    _validate_inputs(rows, expected_decisions, model_report_id, base_profile)
    forecasts = {at: predicted for at, predicted, _actual in rows}
    del rows, points  # Scoring targets cannot enter any strategy closure.
    digest = "sha256:" + sha256(json_bytes([
        {"at": at.isoformat(), "predicted": predicted} for at, predicted in forecasts.items()
    ])).hexdigest()
    stored = load_dataset(Path(dataset_path))
    data = stored.dataset
    _validate_dataset_scope(data)
    if not set(expected_decisions) <= _eligible_decisions(data):
        raise ValueError("source cohort contains ineligible dataset decisions")
    runtime = {"python": sys.version.split()[0], "market_trader": version("market-trader"),
               "execution": "existing run_backtest; frozen forecasts; no fitting"}
    for scenario, factor in (("half", Decimal("0.5")), ("baseline", Decimal(1)), ("double", Decimal(2))):
        # Do not hold a decimal context open across yield (it would leak to callers).
        precision = max(28, *(len(cost.as_tuple().digits)+1 for cost in
                             (base_profile.spread, base_profile.slippage, base_profile.commission_rate)))
        with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN)):
            profile = replace(base_profile, spread=base_profile.spread*factor,
                              slippage=base_profile.slippage*factor,
                              commission_rate=base_profile.commission_rate*factor)
        result = run_backtest(data, _frozen_strategy(data, forecasts), lookback=8, profile=profile)
        report = asdict(result)
        del result
        report["execution"] = report.pop("profile")
        report.update(format=BACKTEST_REPORT_FORMAT, strategy={
            "name": "ap8_frozen_forecasts_long", "lookback": 8, "horizon_bars": 2,
            "entry_threshold": Decimal(0), "quantity": Decimal(1),
            "model_report_id": model_report_id, "scenario": scenario, "factor": factor,
            "forecast_digest": digest,
        }, provenance=asdict(stored.manifest.provenance),
            dataset_artifact_id=stored.manifest.artifact_id, runtime=runtime)
        report = json.loads(json_bytes(report, default=_json_value))
        summary = trading_summary(report)
        yield scenario, report, summary
        del report, summary
