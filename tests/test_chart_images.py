"""AP6 independent RGB decoder and causal renderer contracts."""
import hashlib
import json
import struct
import zlib
from dataclasses import replace
import pytest
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from market_trader.domain import Candle

START = datetime(2024, 1, 2, 8, tzinfo=UTC)
INTERVAL = timedelta(minutes=15)
DECISION = START + 8 * INTERVAL
BG, UP, DOWN, DOJI = (17, 24, 39), (46, 204, 113), (239, 83, 80), (200, 200, 200)


def bars():
    return tuple(Candle("DEUIDXEUR", INTERVAL, START + i * INTERVAL,
                        START + (i + 1) * INTERVAL, START + (i + 1) * INTERVAL,
                        Decimal(100), Decimal(103), Decimal(99), Decimal(102), Decimal(1))
                 for i in range(8))


def decode(png):
    """Independent struct/CRC/zlib decoder, not renderer's helpers."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, []
    while offset < len(png):
        size, = struct.unpack_from(">I", png, offset)
        kind = png[offset + 4:offset + 8]
        body = png[offset + 8:offset + 8 + size]
        crc, = struct.unpack_from(">I", png, offset + 8 + size)
        assert crc == zlib.crc32(kind + body)
        chunks.append((kind, body))
        offset += size + 12
    assert offset == len(png)
    assert [kind for kind, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[0][1]) == (256, 192, 8, 2, 0, 0, 0)
    assert chunks[-1][1] == b""
    compressed = chunks[1][1]
    assert compressed[:2] == b"\x78\x01"
    pos, stored = 2, bytearray()
    while True:
        header = compressed[pos]
        assert header in (0, 1)  # BTYPE=00, zero alignment bits
        length, complement = struct.unpack_from("<HH", compressed, pos + 1)
        assert complement == length ^ 65535
        stored.extend(compressed[pos + 5:pos + 5 + length])
        pos += 5 + length
        if header:
            break
    assert pos + 4 == len(compressed)
    assert struct.unpack_from(">I", compressed, pos)[0] == zlib.adler32(stored)
    raw = zlib.decompress(compressed)
    assert raw == stored
    stride = 256 * 3 + 1
    assert len(raw) == 192 * stride
    assert all(raw[y * stride] == 0 for y in range(192))
    return [[tuple(raw[y * stride + 1 + x * 3:y * stride + 4 + x * 3])
             for x in range(256)] for y in range(192)]


def test_extreme_price_representation_is_rejected_before_expansion():
    from market_trader.chart_images import window_payload
    for value in (Decimal('1e1001'), Decimal('1e-1001'), Decimal('1.'+'1'*1000)):
        data=tuple(replace(c,open=value,high=value,low=value,close=value) for c in bars())
        with pytest.raises(ValueError,match='resource'):
            window_payload(data,decision_at=DECISION)


def test_billion_exponent_does_not_reach_formatting(monkeypatch):
    import market_trader.chart_images as renderer
    def forbidden(value):
        pytest.fail('unbounded decimal reached fixed-point formatting')
    monkeypatch.setattr(renderer,'_decimal_text',forbidden)
    value=Decimal('1e1000000000')
    data=tuple(replace(c,open=value,high=value,low=value,close=value) for c in bars())
    with pytest.raises(ValueError,match='resource'):
        renderer.window_payload(data,decision_at=DECISION)


def test_png_is_rgb8_stored_deflate_with_only_candles():
    from market_trader.chart_images import render_candles
    pixels = decode(render_candles(bars(), decision_at=DECISION))
    assert {p for row in pixels for p in row} == {BG, UP}
    assert all(pixels[y][x] == BG for y in range(192) for x in range(256)
               if x < 12 or x >= 244 or y < 12 or y >= 180)
    # Frozen equally spaced slot centres: floor(12 + (2*i+1)*232/16).
    for centre in (26, 55, 84, 113, 142, 171, 200, 229):
        assert pixels[20][centre] == UP  # high wick
        assert pixels[171][centre] == UP  # low wick
        assert pixels[58][centre - 6:centre + 6] == [UP] * 12
        assert pixels[58][centre - 7] == BG
        assert pixels[58][centre + 6] == BG
        assert pixels[19][centre] == BG
        assert pixels[172][centre] == BG


def test_payload_is_exact_causal_allowlist_canonical_and_volume_free():
    from market_trader.chart_images import window_payload
    result = window_payload(bars(), decision_at=DECISION)
    assert result == {
        "instrument": "DEUIDXEUR", "interval_microseconds": 900000000,
        "decision_at": DECISION.isoformat(timespec="microseconds"),
        "candles": [{"open_time": c.open_time.isoformat(timespec="microseconds"),
                     "close_time": c.close_time.isoformat(timespec="microseconds"),
                     "available_at": c.available_at.isoformat(timespec="microseconds"),
                     "open": "100", "high": "103", "low": "99", "close": "102"}
                    for c in bars()],
    }
    changed = tuple(replace(c, volume=None, open=Decimal("100.000")) for c in bars())
    assert window_payload(changed, decision_at=DECISION) == result
    assert window_payload(list(bars()), decision_at=DECISION) == result


@pytest.mark.parametrize("field,value", [
    ("instrument", "OTHER"), ("instrument", True), ("instrument", " DEUIDXEUR"),
    ("interval", True), ("interval", timedelta(minutes=1)),
    ("open", True), ("open", 100), ("open", "100"), ("open", 100.0),
    ("high", Decimal("NaN")), ("low", Decimal("sNaN")),
    ("close", Decimal("Infinity")), ("low", Decimal(0)),
    ("low", Decimal(-1)), ("low", Decimal(101)), ("high", Decimal(101)),
    ("open_time", START.replace(tzinfo=None)), ("close_time", True),
    ("available_at", DECISION + INTERVAL), ("available_at", START),
    ("open_time", START + timedelta(seconds=1)),
])
def test_invalid_candle_rejected_by_all_public_boundaries(field, value):
    from market_trader.chart_images import image_key, render_candles, window_payload
    changed = (replace(bars()[0], **{field: value}),) + bars()[1:]
    for boundary in (window_payload, image_key, render_candles):
        with pytest.raises(ValueError):
            boundary(changed, decision_at=DECISION)


@pytest.mark.parametrize("candles,decision", [
    ((), DECISION), (bars()[:7], DECISION), (bars() + bars()[:1], DECISION),
    (bars()[:3] + bars()[4:] + bars()[:1], DECISION),
    (bars()[::-1], DECISION), ((True,) + bars()[1:], DECISION),
    (bars(), True), (bars(), DECISION.replace(tzinfo=None)),
    (bars(), DECISION + timedelta(seconds=1)), (bars(), DECISION - INTERVAL),
    (None, DECISION), (True, DECISION),
])
def test_invalid_windows_rejected(candles, decision):
    from market_trader.chart_images import window_payload
    with pytest.raises(ValueError):
        window_payload(candles, decision_at=decision)


def test_non_utc_and_off_grid_windows_rejected_but_calendar_year_unrestricted():
    from datetime import timezone
    from market_trader.chart_images import window_payload
    offset = timezone(timedelta(hours=1))
    changed = tuple(replace(c, open_time=c.open_time.astimezone(offset),
                            close_time=c.close_time.astimezone(offset),
                            available_at=c.available_at.astimezone(offset)) for c in bars())
    with pytest.raises(ValueError):
        window_payload(changed, decision_at=DECISION)
    shifted = tuple(replace(c, open_time=c.open_time + timedelta(seconds=1),
                            close_time=c.close_time + timedelta(seconds=1),
                            available_at=c.available_at + timedelta(seconds=1)) for c in bars())
    with pytest.raises(ValueError):
        window_payload(shifted, decision_at=DECISION + timedelta(seconds=1))
    later = tuple(replace(c, open_time=c.open_time.replace(year=2025),
                          close_time=c.close_time.replace(year=2025),
                          available_at=c.available_at.replace(year=2025)) for c in bars())
    assert window_payload(later, decision_at=DECISION.replace(year=2025))["instrument"] == "DEUIDXEUR"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def test_recipe_hash_binds_actual_module_bytes_and_image_key_binds_recipe():
    from pathlib import Path
    import market_trader.chart_images as renderer
    recipe = renderer.chart_recipe()
    recipe_id = recipe.pop("recipe_id")
    assert recipe_id == "sha256:" + hashlib.sha256(canonical(recipe)).hexdigest()
    assert recipe["renderer_sha256"] == hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest()
    assert recipe["format"] == "market-trader.chart-recipe.v1"
    assert recipe["geometry"]["width"] == 256
    assert recipe["geometry"]["height"] == 192
    assert recipe["colors"] == {"background": list(BG), "up": list(UP), "down": list(DOWN), "doji": list(DOJI)}
    assert renderer.image_key(bars(), decision_at=DECISION) == "sha256:" + hashlib.sha256(canonical({
        "recipe_id": recipe_id, "window": renderer.window_payload(bars(), decision_at=DECISION),
    })).hexdigest()
    recipe["colors"]["up"][0] = 0
    assert renderer.chart_recipe()["colors"]["up"] == list(UP)


def test_down_doji_and_up_have_exact_vertical_body_and_wick_pixels():
    from market_trader.chart_images import render_candles
    candles = list(bars())
    candles[1] = replace(candles[1], open=Decimal(102), close=Decimal(100))
    candles[2] = replace(candles[2], close=Decimal(100))
    pixels = decode(render_candles(tuple(candles), decision_at=DECISION))
    # Independent hand-derived y: high=20, low=171, open100=133, close102=58.
    expected = [[BG] * 256 for _ in range(192)]
    for i, centre in enumerate((26, 55, 84, 113, 142, 171, 200, 229)):
        color = (UP, DOWN, DOJI)[i] if i < 3 else UP
        for y in range(20, 172):
            expected[y][centre] = color
        for y in ([133] if i == 2 else range(58, 134)):
            expected[y][centre - 6:centre + 6] = [color] * 12
    assert pixels == expected


@pytest.mark.parametrize("price", ["0.00000001", "100", "10000000000000000000000000000.000000000000001"])
def test_flat_window_is_one_pixel_doji_with_absolute_padding(price):
    from market_trader.chart_images import render_candles
    flat = tuple(replace(c, open=Decimal(price), high=Decimal(price),
                         low=Decimal(price), close=Decimal(price)) for c in bars())
    pixels = decode(render_candles(flat, decision_at=DECISION))
    expected = [[BG] * 256 for _ in range(192)]
    for centre in (26, 55, 84, 113, 142, 171, 200, 229):
        expected[96][centre - 6:centre + 6] = [DOJI] * 12
    assert pixels == expected


def test_subpixel_nonzero_body_remains_one_pixel_and_up_colored():
    from market_trader.chart_images import render_candles
    candles = (replace(bars()[0], close=Decimal("100.0000000000000000000000001")),) + bars()[1:]
    pixels = decode(render_candles(candles, decision_at=DECISION))
    assert pixels[133][20:32] == [UP] * 12
    assert pixels[132][20:26] == [BG] * 6
    assert pixels[134][20:26] == [BG] * 6


def test_rational_mapping_and_payload_ignore_decimal_context_and_spelling():
    from decimal import Context, Inexact, Rounded, ROUND_DOWN, ROUND_UP, localcontext
    from market_trader.chart_images import chart_recipe, image_key, render_candles, window_payload
    candles = tuple(replace(c, open=Decimal("100.00000000000000000001"),
                           close=Decimal("100.00000000000000000002")) for c in bars())
    def result():
        return (chart_recipe(), window_payload(candles, decision_at=DECISION),
                image_key(candles, decision_at=DECISION), render_candles(candles, decision_at=DECISION))
    baseline = result()
    for precision, rounding in ((2, ROUND_DOWN), (6, ROUND_UP), (80, ROUND_DOWN)):
        with localcontext(Context(prec=precision, rounding=rounding, traps=[Inexact, Rounded])):
            assert result() == baseline
    spelled = tuple(replace(c, open=Decimal("1E+2"), high=Decimal("103.000"),
                            low=Decimal("99.0"), close=Decimal("102.00")) for c in bars())
    assert image_key(spelled, decision_at=DECISION) == image_key(bars(), decision_at=DECISION)
    assert render_candles(spelled, decision_at=DECISION) == render_candles(bars(), decision_at=DECISION)


def test_volume_future_and_targets_are_not_renderer_inputs():
    import inspect
    from market_trader.chart_images import image_key, render_candles, window_payload
    source = list(bars()) + [replace(bars()[-1], open_time=DECISION,
                                   close_time=DECISION + INTERVAL, available_at=DECISION + INTERVAL)]
    target = {"return": Decimal("1"), "dataset_id": "first"}
    before = (image_key(tuple(source[:8]), decision_at=DECISION),
              render_candles(tuple(source[:8]), decision_at=DECISION))
    source[8] = replace(source[8], high=Decimal("999999"), close=Decimal("999998"))
    target.update({"return": Decimal("-123"), "dataset_id": "different"})
    for i in range(8):
        source[i] = replace(source[i], volume=Decimal("NaN"))
    assert (image_key(tuple(source[:8]), decision_at=DECISION),
            render_candles(tuple(source[:8]), decision_at=DECISION)) == before
    for function in (image_key, render_candles, window_payload):
        assert set(inspect.signature(function).parameters) == {"candles", "decision_at"}
        with pytest.raises(TypeError):
            function(bars(), decision_at=DECISION, target=target)


def test_all_consumed_fields_affect_identity_even_when_pixels_do_not():
    from market_trader.chart_images import image_key
    before = image_key(bars(), decision_at=DECISION)
    changed = (replace(bars()[0], available_at=bars()[0].available_at + timedelta(seconds=1)),) + bars()[1:]
    assert image_key(changed, decision_at=DECISION) != before
    changed = (replace(bars()[0], close=Decimal("102.00000000000001")),) + bars()[1:]
    assert image_key(changed, decision_at=DECISION) != before
    changed = tuple(replace(c, instrument="OTHER") for c in bars())
    assert image_key(changed, decision_at=DECISION) != before
    changed = tuple(replace(c, open_time=c.open_time + INTERVAL, close_time=c.close_time + INTERVAL,
                            available_at=c.available_at + INTERVAL) for c in bars())
    assert image_key(changed, decision_at=DECISION + INTERVAL) != before


def test_fresh_process_recipe_png_and_key_are_stable_without_optional_imports():
    import os
    from pathlib import Path
    import subprocess
    import sys
    from market_trader.chart_images import chart_recipe, image_key, render_candles
    script = '''import hashlib,json,runpy,sys
from market_trader.chart_images import chart_recipe,image_key,render_candles
ns=runpy.run_path(sys.argv[1]); bars=ns["bars"](); decision=ns["DECISION"]
print(json.dumps([chart_recipe(),image_key(bars,decision_at=decision),hashlib.sha256(render_candles(bars,decision_at=decision)).hexdigest(),sorted(set(sys.modules)&{"torch","numpy","PIL","matplotlib"})],sort_keys=True))'''
    expected = [chart_recipe(), image_key(bars(), decision_at=DECISION),
                hashlib.sha256(render_candles(bars(), decision_at=DECISION)).hexdigest(), []]
    for seed in ("1", "42"):
        result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve())],
                                env={**os.environ, "PYTHONHASHSEED": seed},
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout) == expected


def test_source_policy_change_conservatively_invalidates_recipe_and_key(tmp_path):
    import importlib.util
    from pathlib import Path
    import market_trader.chart_images as renderer
    copied = tmp_path / "chart_images.py"
    copied.write_bytes(Path(renderer.__file__).read_bytes() + b"\n# policy revision\n")
    spec = importlib.util.spec_from_file_location("chart_images_revision", copied)
    revision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(revision)
    assert revision.chart_recipe()["recipe_id"] != renderer.chart_recipe()["recipe_id"]
    assert revision.image_key(bars(), decision_at=DECISION) != renderer.image_key(bars(), decision_at=DECISION)
    assert revision.render_candles(bars(), decision_at=DECISION) == renderer.render_candles(bars(), decision_at=DECISION)
