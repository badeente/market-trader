"""Bounded, self-contained causal CNN input/ledger audits (no serving paths)."""
import base64
from copy import deepcopy
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from hashlib import sha256
from html import escape
import json
from math import isfinite, isclose
import os
from pathlib import Path
import stat

from market_trader._dataset_codec import parse_json
from market_trader.report_validation import timestamp

AUDIT_FORMAT = 'market-trader.trade-image-audit.v1'
MAX_HTML_BYTES = 4 * 1024 * 1024


def _read(path):
    from market_trader.image_dataset import _directory
    path = Path(path)
    with _directory(path.parent) as parent:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 32 * 1024 * 1024:
                raise ValueError('audit source must be a bounded regular file')
            payload = stream.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError('audit source exceeds bound')
    return parse_json(payload.decode('utf-8'), allow_floats=True), 'sha256:' + sha256(payload).hexdigest()


def build_trade_image_audit(backtest_path, model_path, dataset_path, predictions_path, *, limit=8):
    """Regenerate first executed trade inputs, or explicitly evaluated forecasts.

    Prediction comparisons use an explicit 1e-12 absolute/relative tolerance;
    full ledger validation is reused, not replaced with a second simulator.
    """
    from market_trader.image_neural import load_cnn_model
    from market_trader.storage import load_dataset
    from market_trader.reporting import BACKTEST_REPORT_FORMAT, _run_summary
    from market_trader.chart_images import image_key, render_candles, window_payload
    from market_trader.calendars import SessionCalendar
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('audit limit must be an integer from 1 to 8')
    report, backtest_id = _read(backtest_path)
    if report.get('format') != BACKTEST_REPORT_FORMAT:
        raise ValueError('invalid audit backtest format')
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        _run_summary(deepcopy(report))
    model = load_cnn_model(Path(model_path))
    stored = load_dataset(Path(dataset_path))
    data, strategy = stored.dataset, report['strategy']
    if (strategy['name'] != 'cnn_long' or strategy.get('model_report_id') != model.report_id
            or strategy.get('lookback') != model.config.lookback_bars
            or strategy.get('horizon_bars') != model.config.horizon_bars):
        raise ValueError('CNN model/strategy binding mismatch')
    if (report['dataset_id'] != data.dataset_id
            or report.get('dataset_artifact_id') != stored.manifest.artifact_id
            or data.instrument != model.instrument or data.interval != model.interval):
        raise ValueError('audit dataset binding mismatch')
    if model.config.lookback_bars != 8 or model.config.horizon_bars != 2:
        raise ValueError('audit requires frozen eight/two policy')
    if any(c.open_time.year != 2024 or c.close_time.year != 2024 or c.available_at.year != 2024
           or c.open_time <= model.last_target_available_at for c in data.candles):
        raise ValueError('audit requires validation-only candles after training cutoff; no 2025')
    points, predictions_id = _read(predictions_path)
    if not isinstance(points, list) or len(points) > 50000:
        raise ValueError('invalid bounded saved forecasts')
    forecasts = {}
    previous = None
    for point in points:
        at = timestamp(point['at'])
        value = point['predicted']
        if (type(value) not in (int, float) or not isfinite(value)
                or previous is not None and at <= previous):
            raise ValueError('saved forecasts must be finite and chronological')
        forecasts[at] = value
        previous = at
    trades = report['trades']
    selected = [(timestamp(t['entry']['order']['decision_at']), t) for t in trades[:limit]]
    if not trades:
        selected = [(at, None) for at in list(forecasts)[:limit]]
    by_close = {c.close_time: i for i, c in enumerate(data.candles)}
    by_open = {c.open_time: c for c in data.candles}
    samples = []
    threshold = Decimal(strategy['entry_threshold'])
    if not threshold.is_finite():
        raise ValueError('invalid entry threshold')
    for at, trade in selected:
        end = by_close.get(at, -1) + 1
        window = tuple(data.candles[end-8:end]) if end >= 8 else ()
        if (len(window) != 8 or any(c.available_at > at for c in window)
                or any(b.open_time != a.close_time for a,b in zip(window, window[1:]))):
            raise ValueError('audit decision has no complete causal window')
        calendar = data.config.calendar
        if isinstance(calendar, SessionCalendar):
            session = calendar.session_at(window[0].open_time)
            if session is None or at + model.interval * 2 >= session.close_time:
                raise ValueError('audit decision crosses session')
        forecast = model.predict_window(window, decision_at=at)
        if (type(forecast) not in (int, float) or not isfinite(forecast)
                or at not in forecasts or not isclose(forecast, forecasts[at], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('re-prediction does not match saved forecast')
        if trade is not None:
            entry, exit_fill = trade['entry'], trade['exit']
            if (Decimal(str(forecast)) <= threshold or entry['side'] != 'buy'
                    or timestamp(entry['candle_open_time']) != at
                    or timestamp(entry['order']['decision_at']) != at
                    or timestamp(exit_fill['candle_open_time']) != at + model.interval * 2
                    or exit_fill['reason'] != 'exit'):
                raise ValueError('trade is not bound to CNN decision/timed exit')
            for fill in (entry, exit_fill):
                candle = by_open.get(timestamp(fill['candle_open_time']))
                if candle is None or Decimal(fill['reference_price']) != candle.open:
                    raise ValueError('ledger fill reference does not match dataset open')
        png = render_candles(window, decision_at=at)
        samples.append(dict(decision_at=at.isoformat(), forecast=forecast, saved_forecast=forecasts[at],
            prediction_tolerance=dict(relative=1e-12, absolute=1e-12), threshold=str(threshold),
            image_key=image_key(window, decision_at=at), image_sha256='sha256:' + sha256(png).hexdigest(),
            image_data_uri='data:image/png;base64,' + base64.b64encode(png).decode('ascii'),
            ledger_trade_index=len(samples) if trade is not None else None,
            window=window_payload(window, decision_at=at), trade=deepcopy(trade),
            ledger_decision='executed_trade' if trade is not None else 'evaluated_forecast_no_executed_trade'))
    audit = dict(format=AUDIT_FORMAT, model_report_id=model.report_id, dataset_id=data.dataset_id,
        dataset_artifact_id=stored.manifest.artifact_id, backtest_sha256=backtest_id,
        predictions_sha256=predictions_id, mode='trades' if trades else 'decisions',
        total_trades=len(trades), total_decisions=len(points), audited_count=len(samples), limit=limit, samples=samples,
        open_position=deepcopy(report['position']), pending_order=deepcopy(report['unfilled_order']))
    render_trade_image_audit(audit)
    return audit


def render_trade_image_audit(audit: dict) -> str:
    """Escape metadata; accept only bounded canonical inline AP6 PNGs."""
    from market_trader.image_pixels_v1 import block_sums
    if not isinstance(audit, dict) or audit.get('format') != AUDIT_FORMAT:
        raise ValueError('invalid trade image audit format')
    for field in ('total_trades', 'total_decisions', 'audited_count', 'limit'):
        if type(audit.get(field)) is not int or audit[field] < 0:
            raise ValueError('invalid audit count')
    total = audit['total_trades'] if audit['total_trades'] else audit['total_decisions']
    if (not 1 <= audit['limit'] <= 8 or audit['mode'] != ('trades' if audit['total_trades'] else 'decisions')
            or not isinstance(audit['samples'], list)
            or audit['audited_count'] != len(audit['samples'])
            or audit['audited_count'] != min(audit['limit'], total)):
        raise ValueError('audit count/mode mismatch')
    metadata = {key: value for key,value in audit.items() if key != 'samples'}
    html = ('<section class="panel" id="cnn-trade-image-audit"><h2>CNN causal image audit</h2>'
        + ('<p>No completed CNN trades; ledger contains an executed open entry. Evaluated forecasts are not completed trades.</p>'
           if not audit['total_trades'] and audit.get('open_position') is not None
           else '<p>No executed CNN trades; evaluated forecasts are not trades.</p>' if not audit['total_trades']
           else '<p>First chronological completed executed CNN trades; no profit selection. Any terminal open position is listed separately.</p>')
        + f"<p>{audit['audited_count']} of {total} audited. Outcomes are outside input pixels.</p><pre>"
        + escape(json.dumps(metadata, ensure_ascii=True, sort_keys=True)) + '</pre>')
    for sample in audit['samples']:
        uri = sample['image_data_uri']
        prefix = 'data:image/png;base64,'
        if not isinstance(uri, str) or not uri.startswith(prefix) or len(uri) > 300000:
            raise ValueError('only bounded inline PNG images are allowed')
        try:
            png = base64.b64decode(uri[len(prefix):], validate=True)
        except ValueError:
            raise ValueError('invalid inline PNG encoding') from None
        block_sums(png)
        if sample['image_sha256'] != 'sha256:' + sha256(png).hexdigest():
            raise ValueError('inline PNG checksum mismatch')
        from market_trader._dataset_codec import decode_record
        from market_trader.domain import Candle
        from market_trader.chart_images import image_key, render_candles, window_payload
        window = sample['window']
        rows = window['candles']
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError('invalid audit window')
        candles = tuple(decode_record({**row, 'instrument': window['instrument'],
            'interval': window['interval_microseconds'], 'volume': '0'}, Candle) for row in rows)
        at = timestamp(sample['decision_at'])
        if (window_payload(candles, decision_at=at) != window
                or image_key(candles, decision_at=at) != sample['image_key']
                or render_candles(candles, decision_at=at) != png):
            raise ValueError('inline PNG/input identity does not match causal window')
        details = {key: value for key,value in sample.items() if key != 'image_data_uri'}
        html += ('<article><img width="256" height="192" alt="Historical eight completed candles only" src="'
            + escape(uri, quote=True) + '"><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
            + escape(json.dumps(details, ensure_ascii=True, sort_keys=True, indent=2)) + '</pre></article>')
        if len(html.encode('utf-8')) > MAX_HTML_BYTES:
            raise ValueError('trade image audit exceeds HTML bound')
    html += '</section>'
    if len(html.encode('utf-8')) > MAX_HTML_BYTES:
        raise ValueError('trade image audit exceeds HTML bound')
    return html
