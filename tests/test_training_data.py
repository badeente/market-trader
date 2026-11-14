import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Context, Decimal, localcontext

import pytest

from market_trader.artifacts import DatasetProvenance
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.domain import Candle
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.storage import save_dataset
from market_trader.training_data import TrainingConfig, build_training_data

START = datetime(2026, 1, 1, tzinfo=UTC)
MINUTE = timedelta(minutes=1)


def at(minute):
    return START + minute * MINUTE


def bars(minutes):
    return tuple(
        Candle(
            instrument="TEST", interval=MINUTE, open_time=at(minute),
            close_time=at(minute + 1), available_at=at(minute + 1),
            open=Decimal(100 + minute), high=Decimal(101 + minute),
            low=Decimal(99 + minute), close=Decimal(100 + minute), volume=Decimal(1),
        )
        for minute in minutes
    )


def normalized(candles, calendar="continuous"):
    dataset, = normalize_candles(candles, config=NormalizationConfig(calendar)).datasets
    return dataset


@pytest.fixture
def config():
    return TrainingConfig(3, 2, at(7), at(10))


@pytest.mark.case("ML-003")
def test_causal_features_and_forward_return_from_next_open(config):
    candles = list(bars(range(14)))
    candles[3] = replace(candles[3], close=Decimal(104))
    candles[4] = replace(candles[4], open=Decimal(105))
    result = build_training_data(normalized(candles), config)
    row = result.train[0]

    assert result.feature_names == ("return_1", "window_return", "range_fraction", "close_vs_mean")
    with localcontext(Context(prec=28)):
        assert row.features == (
            Decimal(1) / 101, Decimal(2) / 100, Decimal(2) / 102, Decimal(1) / 101,
        )
        assert row.target_return == Decimal(1) / 103
    assert (row.decision_at, row.entry_at, row.exit_at, row.target_available_at) == (
        at(3), at(3), at(5), at(5),
    )

    candles[4] = replace(candles[4], close=Decimal(105))
    changed = build_training_data(normalized(candles), config).train[0]
    assert changed.features == row.features
    assert changed.target_return != row.target_return

    with localcontext(Context(prec=6)):
        repeated = build_training_data(normalized(candles), config).train[0]
    assert repeated == changed


@pytest.mark.case("ML-001")
def test_strict_split_boundaries_and_tail(config):
    candles = list(bars(range(14)))
    result = build_training_data(normalized(candles), config)

    assert [row.decision_at for row in result.train] == [at(3), at(4)]
    assert [row.decision_at for row in result.validation] == [at(7)]
    assert [row.decision_at for row in result.test] == [at(10), at(11), at(12)]
    assert all(row.target_available_at < config.validation_start for row in result.train)
    assert all(row.target_available_at < config.test_start for row in result.validation)

    candles[10] = replace(candles[10], close=Decimal(111))
    changed = build_training_data(normalized(candles), config)
    assert changed.train == result.train
    assert changed.validation == result.validation
    assert changed.test != result.test


@pytest.mark.case("ML-001")
def test_availability_controls_features_and_target_purging(config):
    candles = list(bars(range(14)))
    candles[1] = replace(candles[1], available_at=at(4))
    result = build_training_data(normalized(candles), config)
    assert [row.decision_at for row in result.train] == [at(4)]

    candles[1] = replace(candles[1], available_at=at(2))
    candles[4] = replace(candles[4], available_at=at(5))
    candles[5] = replace(candles[5], available_at=at(7))
    result = build_training_data(normalized(candles), config)
    assert [row.decision_at for row in result.train] == [at(3)]

    candles[3] = replace(candles[3], available_at=at(7))
    with pytest.raises(ValueError, match="train split"):
        build_training_data(normalized(candles), config)


@pytest.mark.parametrize(("second", "third"), [(6, 12), (10, 20)])
def test_windows_do_not_cross_explicit_sessions(second, third):
    starts = (0, second, third)
    calendar = SessionCalendar(
        "TEST", at(0), at(third + 6),
        tuple(TradingSession(at(start), at(start + 6)) for start in starts),
    )
    candles = bars(minute for start in starts for minute in range(start, start + 6))
    offset = timedelta(microseconds=1)
    result = build_training_data(
        normalized(candles, calendar), TrainingConfig(3, 2, at(second) + offset, at(third) + offset),
    )

    assert [row.decision_at for row in result.train] == [at(3), at(4)]
    assert [row.decision_at for row in result.validation] == [at(second + 3), at(second + 4)]
    assert [row.decision_at for row in result.test] == [at(third + 3), at(third + 4)]
    assert result.train[-1].exit_at == at(6)


@pytest.mark.parametrize(("overrides", "diagnostic"), [
    ({"lookback_bars": 1}, "lookback_bars"),
    ({"horizon_bars": True}, "horizon_bars"),
    ({"horizon_bars": 0}, "horizon_bars"),
    ({"validation_start": at(7).replace(tzinfo=None)}, "validation_start"),
    ({"test_start": at(10).astimezone(timezone(timedelta(hours=1)))}, "test_start"),
    ({"validation_start": at(10)}, "validation_start"),
    ({"validation_start": at(9)}, "validation split"),
    ({"test_start": at(13)}, "test split"),
])
def test_invalid_config_or_empty_split(config, overrides, diagnostic):
    with pytest.raises(ValueError, match=diagnostic):
        build_training_data(normalized(bars(range(14))), replace(config, **overrides))


def test_reject_active_gaps_and_nonpositive_prices(config):
    candles = list(bars(range(14)))
    with pytest.raises(ValueError, match="gap"):
        build_training_data(normalized(candles[:3] + candles[4:]), config)
    candles[0] = replace(candles[0], open=Decimal(0), low=Decimal(0))
    with pytest.raises(ValueError, match="positive"):
        build_training_data(normalized(candles), config)


def test_cli_exports_reproducible_training_data(tmp_path):
    source = tmp_path / "dataset.json"
    manifest = save_dataset(
        source, normalized(bars(range(14))), provenance=DatasetProvenance("synthetic fixture", ()),
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'version = 1\ndataset_artifact = "dataset.json"\nlookback_bars = 3\nhorizon_bars = 2\n'
        'validation_start = 2026-01-01T00:07:00Z\ntest_start = 2026-01-01T00:10:00Z\n',
        encoding="utf-8",
    )
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()

    def run(output):
        return subprocess.run(
            [sys.executable, "-m", "market_trader.cli", "data", "prepare-training",
             "--config", str(config_path), "--output", str(output)],
            cwd=other_cwd, capture_output=True, text=True, timeout=10,
        )

    first, second = tmp_path / "export" / "first.json", tmp_path / "second.json"
    for output in (first, second):
        command = run(output)
        assert command.returncode == 0, command.stderr
        assert json.loads(command.stdout)["split_counts"] == {"train": 2, "validation": 1, "test": 3}
    before = first.read_bytes()
    assert second.read_bytes() == before
    payload = json.loads(before)
    assert payload["format"] == "market-trader.training-data.v1"
    assert payload["dataset_id"] == manifest.dataset_id
    assert payload["dataset_artifact_id"] == manifest.artifact_id
    assert payload["splits"]["test"][-1]["target_available_at"] == at(14).isoformat()
    assert isinstance(payload["splits"]["train"][0]["target_return"], str)
    artifact_id = payload.pop("training_data_id")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert artifact_id == "sha256:" + hashlib.sha256(canonical).hexdigest()

    command = run(first)
    assert command.returncode == 1
    assert "save training data" in command.stderr
    assert first.read_bytes() == before

    source.write_text("{}", encoding="utf-8")
    failed_output = tmp_path / "failed.json"
    command = run(failed_output)
    assert command.returncode == 1
    assert "verify artifact" in command.stderr
    assert not failed_output.exists()
