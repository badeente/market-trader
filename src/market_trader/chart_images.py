"""Deterministic candle-only RGB8 PNGs; standard library, exact rational pixels.

PNG structure follows https://www.w3.org/TR/png/ (IHDR/IDAT/IEND only).
No target, volume, source dataset, model, font, or browser dependencies.
"""
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import struct
import zlib

from market_trader.domain import Candle

_WIDTH, _HEIGHT, _MARGIN, _BODY = 256, 192, 12, 12
_BACKGROUND = (17, 24, 39)
_UP, _DOWN, _DOJI = (46, 204, 113), (239, 83, 80), (200, 200, 200)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field}: expected a timezone-aware UTC datetime")
    return value.astimezone(UTC)


def validate_chart_price(value: object) -> None:
    """Bound decimal expansion/rational arithmetic before consuming a price."""
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError('expected a positive finite Decimal')
    parts = value.as_tuple()
    exponent = parts.exponent
    if not isinstance(exponent, int):
        raise ValueError('expected a finite Decimal exponent')
    if (len(parts.digits) > 1000 or abs(exponent) > 1000
            or len(parts.digits) + exponent > 1000):
        raise ValueError('price resource bounds exceeded')


def _decimal_text(value: Decimal) -> str:
    # normalize() would round under the caller's Decimal context.
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def window_payload(candles: tuple[Candle, ...], *, decision_at: datetime) -> dict:
    """Validate and serialize only consumed input; volume is deliberately ignored.

    Accept tuple/list windows, reject coercions of all candle fields. UTC ISO
    strings always include six fractional digits. Decimal strings have no
    exponent or insignificant fractional trailing zeroes. Calendar years and
    research sessions are the builder's policy, not this generic renderer's.
    """
    decision = _utc(decision_at, "decision_at")
    if not isinstance(candles, (tuple, list)) or len(candles) != 8:
        raise ValueError("expected exactly eight candles")
    interval = timedelta(minutes=15)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    rows = []
    instrument = None
    previous = None
    for candle in candles:
        if not isinstance(candle, Candle):
            raise ValueError("expected a Candle")
        if (not isinstance(candle.instrument, str) or not candle.instrument
                or candle.instrument != candle.instrument.strip()):
            raise ValueError("instrument must be nonempty unpadded text")
        if instrument is None:
            instrument = candle.instrument
        if candle.instrument != instrument:
            raise ValueError("candles must have the same instrument")
        if not isinstance(candle.interval, timedelta) or candle.interval != interval:
            raise ValueError("interval must be 15 minutes")
        times = {field: _utc(getattr(candle, field), field)
                 for field in ("open_time", "close_time", "available_at")}
        opened, closed, available = (times[f] for f in
                                    ("open_time", "close_time", "available_at"))
        if (opened - epoch) % interval or closed - opened != interval:
            raise ValueError("candles must be UTC-grid-aligned completed 15-minute bars")
        if previous is not None and opened != previous:
            raise ValueError("candles must be ordered and contiguous")
        if not closed <= available <= decision:
            raise ValueError("candle availability must be between close and decision")
        previous = closed
        for field in ("open", "high", "low", "close"):
            price = getattr(candle, field)
            validate_chart_price(price)
        if not candle.low <= min(candle.open, candle.close) <= max(candle.open, candle.close) <= candle.high:
            raise ValueError("invalid OHLC ordering")
        rows.append({**{field: value.isoformat(timespec="microseconds") for field, value in times.items()},
                     **{field: _decimal_text(getattr(candle, field))
                        for field in ("open", "high", "low", "close")}})
    if previous != decision:
        raise ValueError("last candle must end exactly at decision_at")
    return {"instrument": instrument, "interval_microseconds": 900000000,
            "decision_at": decision.isoformat(timespec="microseconds"), "candles": rows}


def chart_recipe() -> dict:
    """Return a fresh complete recipe bound to installed renderer source bytes.

    recipe_id hashes canonical JSON of this object without recipe_id. The source
    digest is deliberately read from the installed .py file (not its path, pyc,
    package version or mtime), so an unchanged wheel has checkout identity.
    """
    body = {
        "format": "market-trader.chart-recipe.v1",
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry": {
            "width": _WIDTH, "height": _HEIGHT, "margin": _MARGIN,
            "body_width": _BODY, "wick_width": 1, "minimum_body_height": 1,
            "count": 8, "x_centres": [12 + (2 * i + 1) * 232 // 16 for i in range(8)],
            "x_rule": "floor(12+(2*i+1)*232/16), i=0..7",
            "body_x_rule": "inclusive centre-6 through centre+5",
            "body_y_rule": "inclusive min(y(open),y(close)) through max; doji one pixel",
            "wick_rule": "inclusive y(high) through y(low) at centre; draw before body",
            "price_y_rule": "12+round_half_up((upper-price)*167/(upper-lower))",
            "rounding": "exact-rational nearest integer; ties toward positive infinity",
        },
        "colors": {"background": list(_BACKGROUND), "up": list(_UP),
                   "down": list(_DOWN), "doji": list(_DOJI)},
        "color_rule": "compare exact close to open; up >, down <, doji =",
        "scale": {"scope": "only eight input lows/highs", "padding_fraction": "0.05",
                  "flat_absolute_padding": "0.001", "arithmetic": "exact fractions",
                  "lower": "min(low)-padding", "upper": "max(high)+padding"},
        "input": {"candles": 8, "interval_microseconds": 900000000,
                  "order": "UTC-epoch-grid-aligned contiguous; last close equals decision",
                  "availability": "close_time<=available_at<=decision_at",
                  "prices": "positive finite Decimal; low<=open,close<=high",
                  "max_price_coefficient_digits": 1000, "max_price_abs_exponent": 1000,
                  "max_price_integer_digits": 1000,
                  "instrument": "same nonempty unpadded instrument",
                  "timestamps": "UTC ISO8601 with microseconds and +00:00",
                  "decimals": "fixed-point; remove insignificant fractional zeroes",
                  "excluded": ["volume", "targets", "labels", "future", "dataset_ids"],
                  "research_years_and_sessions": "enforced by builder, not renderer"},
        "encoding": {"format": "PNG", "bit_depth": 8, "color_type": 2,
                     "compression_method": 0, "filter_method": 0, "interlace": 0,
                     "chunks": ["IHDR", "IDAT", "IEND"], "scanline_filter": 0,
                     "zlib_header_hex": "7801", "deflate": "stored blocks of at most 65535 bytes",
                     "deflate_alignment_bits": 0, "idat_chunks": 1,
                     "checksums": "PNG CRC32 and zlib Adler32", "metadata": False},
        "canonical_json": "UTF-8; sorted keys; compact separators; ensure_ascii=False; allow_nan=False",
        "image_identity": "sha256(canonical_json({recipe_id,window:window_payload}))",
        "content": "candles only; no axes, text, volume, indicators, labels, results or antialiasing",
    }
    return {**body, "recipe_id": "sha256:" + hashlib.sha256(_canonical(body)).hexdigest()}


def image_key(candles: tuple[Candle, ...], *, decision_at: datetime) -> str:
    """SHA256 identity of the recipe and consumed causal input, never labels."""
    window = window_payload(candles, decision_at=decision_at)
    body = {"recipe_id": chart_recipe()["recipe_id"], "window": window}
    return "sha256:" + hashlib.sha256(_canonical(body)).hexdigest()


def _nearest(value: Fraction) -> int:
    """Round a nonnegative rational to nearest integer; ties toward +infinity."""
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload)))


def _png(raw: bytes) -> bytes:
    # Explicit stored blocks avoid version-dependent compressor heuristics.
    stream = bytearray(b"\x78\x01")
    for offset in range(0, len(raw), 65535):
        block = raw[offset:offset + 65535]
        stream.append(int(offset + len(block) == len(raw)))
        stream.extend(struct.pack("<HH", len(block), len(block) ^ 65535))
        stream.extend(block)
    stream.extend(struct.pack(">I", zlib.adler32(raw)))
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", _WIDTH, _HEIGHT, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", bytes(stream)) + _chunk(b"IEND", b""))


def render_candles(candles: tuple[Candle, ...], *, decision_at: datetime) -> bytes:
    """Render a causal eight-candle window; validation is input-only."""
    window_payload(candles, decision_at=decision_at)
    low = min(Fraction(c.low) for c in candles)
    high = max(Fraction(c.high) for c in candles)
    padding = (high - low) / 20 if high != low else Fraction(1, 1000)
    lower, upper = low - padding, high + padding

    def y(price):
        return _MARGIN + _nearest((upper - Fraction(price)) * 167 / (upper - lower))

    rows = [bytearray(bytes(_BACKGROUND) * _WIDTH) for _ in range(_HEIGHT)]
    for index, candle in enumerate(candles):
        centre = _MARGIN + (2 * index + 1) * 232 // 16
        color = bytes(_UP if candle.close > candle.open else
                      _DOWN if candle.close < candle.open else _DOJI)
        for row in range(y(candle.high), y(candle.low) + 1):
            rows[row][centre * 3:centre * 3 + 3] = color
        top, bottom = sorted((y(candle.open), y(candle.close)))
        for row in range(top, bottom + 1):
            rows[row][(centre - _BODY // 2) * 3:(centre + _BODY // 2) * 3] = color * _BODY
    return _png(b"".join(b"\x00" + row for row in rows))
