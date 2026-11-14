import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from market_trader.artifacts import DatasetProvenance
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.coverage import DatasetRequest
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.storage import fingerprint_source, load_dataset, save_dataset

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "data-example.toml"
FIXTURE = ROOT / "tests" / "fixtures" / "fractional_volume.csv"
BACKTEST = ROOT / "configs" / "backtest-example.toml"
BACKTEST_SMA = ROOT / "configs" / "backtest-sma-example.toml"
BACKTEST_PULLBACK = ROOT / "configs" / "backtest-pullback-example.toml"


def command(cwd, *args):
    return subprocess.run(
        [sys.executable, "-m", "market_trader.cli", *map(str, args)],
        cwd=cwd, capture_output=True, text=True, timeout=10,
    )


def cli(cwd, *args):
    return command(cwd, "data", *args)


@pytest.fixture
def local_example(tmp_path):
    config = tmp_path / "config.toml"
    source = tmp_path / FIXTURE.name
    shutil.copyfile(FIXTURE, source)
    config.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace("../tests/fixtures/", ""),
        encoding="utf-8",
    )
    return config, source


@pytest.mark.case("SET-004")
@pytest.mark.case("DAT-009")
def test_offline_import_and_verify_are_reproducible_from_another_directory(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    raw = FIXTURE.read_bytes()
    runs = [
        cli(ROOT, "import", "--config", EXAMPLE.relative_to(ROOT), "--output", first),
        cli(tmp_path, "import", "--config", EXAMPLE, "--output", second),
        cli(tmp_path, "verify", "--artifact", first.name),
    ]
    for run in runs:
        assert run.returncode == 0, run.stderr
        assert not run.stderr
    assert first.read_bytes() == second.read_bytes()
    stored = load_dataset(first)
    assert len(stored.dataset.candles) == 1
    assert stored.dataset.candles[0].volume == Decimal("0.125")
    assert stored.dataset.quality.expected_count == 1
    source, = stored.manifest.provenance.sources
    assert source.name == "fractional-volume"
    assert source.size_bytes == len(raw)
    assert source.sha256 == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert FIXTURE.read_bytes() == raw
    for run in runs:
        assert json.loads(run.stdout) == {
            "artifact_id": stored.manifest.artifact_id,
            "dataset_id": stored.manifest.dataset_id,
            "candle_count": 1,
            "missing_candles": 0,
        }


@pytest.mark.case("CAU-004")
@pytest.mark.parametrize(("old", "new", "diagnostic"), [
    ("version = 1", "version = 2", "version"),
    ("version = 1", "version = true", "version"),
    ('calendar = "continuous"', 'calendar = "market"', "calendar"),
    ('gap_policy = "reject"', 'gap_policy = "ignore"', "gap_policy"),
    ("interval_seconds = 60", "interval_seconds = 0", "interval_seconds"),
    ("interval_seconds = 60", "interval_seconds = 99999999999999999", "interval_seconds"),
    ("end_time = 2026-01-01T00:01:00Z", "end_time = 2026-01-01T00:01:00", "end_time"),
    ('gap_policy = "reject"', 'gap_poilcy = "reject"', "gap_poilcy"),
    ("version = 1", "version =", "Invalid value"),
])
def test_invalid_config_fails_before_import(local_example, old, new, diagnostic):
    config, source = local_example
    config.write_text(config.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    source.unlink()
    output = config.parent / "dataset.json"

    run = cli(config.parent, "import", "--config", config, "--output", output)

    assert run.returncode == 1
    assert not run.stdout
    assert "config" in run.stderr and diagnostic in run.stderr
    assert not output.exists()


@pytest.mark.case("CAU-004")
@pytest.mark.parametrize("contents", [None, b"bad,header\n", b"\xff"])
def test_invalid_source_fails_without_publishing(local_example, contents):
    config, source = local_example
    if contents is None:
        source.unlink()
    else:
        source.write_bytes(contents)
    output = config.parent / "dataset.json"

    run = cli(config.parent, "import", "--config", config, "--output", output)

    assert run.returncode == 1
    assert not run.stdout
    assert "read CSV" in run.stderr and source.name in run.stderr
    assert "Traceback" not in run.stderr
    assert not output.exists()


@pytest.mark.case("DAT-005")
@pytest.mark.parametrize("gap_policy", ["report", "reject"])
def test_configured_request_applies_gap_policy(local_example, gap_policy):
    config, _ = local_example
    config.write_text(
        config.read_text(encoding="utf-8").replace("00:01:00Z", "00:02:00Z")
        .replace('gap_policy = "reject"', f'gap_policy = "{gap_policy}"'),
        encoding="utf-8",
    )
    output = config.parent / "dataset.json"

    run = cli(config.parent, "import", "--config", config, "--output", output)

    if gap_policy == "report":
        assert run.returncode == 0, run.stderr
        assert json.loads(run.stdout)["missing_candles"] == 1
        assert load_dataset(output).dataset.quality.expected_count == 2
    else:
        assert run.returncode == 1
        assert "normalize" in run.stderr and "missing candles" in run.stderr
        assert not run.stdout
        assert not output.exists()


@pytest.mark.case("CAU-004")
def test_import_preserves_existing_output(local_example):
    config, source = local_example
    raw = source.read_bytes()
    output = config.parent / "dataset.json"
    output.write_bytes(b"keep me")

    run = cli(config.parent, "import", "--config", config, "--output", output)

    assert run.returncode == 1
    assert "save" in run.stderr and output.name in run.stderr
    assert not run.stdout
    assert output.read_bytes() == b"keep me"
    assert source.read_bytes() == raw


@pytest.mark.case("CAU-004")
def test_verify_detects_changed_artifact(tmp_path):
    artifact = tmp_path / "dataset.json"
    imported = cli(tmp_path, "import", "--config", EXAMPLE, "--output", artifact)
    assert imported.returncode == 0, imported.stderr
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["provenance"]["description"] = "changed"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    run = cli(tmp_path, "verify", "--artifact", artifact)

    assert run.returncode == 1
    assert not run.stdout
    assert "verify" in run.stderr and "checksum" in run.stderr


@pytest.mark.case("SET-004")
@pytest.mark.case("ACC-009")
@pytest.mark.case("REP-001")
@pytest.mark.case("REP-002")
@pytest.mark.case("REP-005")
def test_backtest_example_is_reproducible_and_the_report_reconciles(tmp_path):
    first = command(ROOT, "backtest", "--config", BACKTEST.relative_to(ROOT))
    second = command(tmp_path, "backtest", "--config", BACKTEST)
    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["format"] == "market-trader.backtest-report.v1"
    assert report["dataset_id"].startswith("sha256:")
    assert "dataset_artifact_id" not in report
    assert report["strategy"]["name"] == "single_long"
    assert report["provenance"]["sources"][0]["name"] == "backtest-candles"
    assert report["execution"]["spread"] == "0.2"
    assert report["runtime"]["code_id"].startswith("sha256:")
    entry, close = report["fills"]
    assert (Decimal(entry["price"]), Decimal(close["price"])) == (Decimal("100.2"), Decimal("104.8"))
    assert Decimal(entry["quantity"]) == Decimal(close["quantity"]) == 3
    assert close["reason"] == "liquidation"
    gross = (Decimal(close["price"]) - Decimal(entry["price"])) * Decimal(entry["quantity"])
    fees = Decimal(entry["fee"]) + Decimal(close["fee"])
    assert gross - fees == Decimal("13.185") == Decimal(report["trades"][0]["net_pnl"])
    assert Decimal(report["accounts"][-1]["equity"]) == 1000 + gross - fees
    assert Decimal(report["max_drawdown"]) == Decimal("-0.0039006")


@pytest.mark.case("REP-003")
def test_backtest_report_records_changed_strategy_and_execution(tmp_path):
    text = BACKTEST.read_text(encoding="utf-8").replace(
        '"backtest-data.toml"', f'"{(BACKTEST.parent / "backtest-data.toml").as_posix()}"',
    )
    config = tmp_path / "backtest.toml"
    config.write_text(text.replace("risk_fraction = 0.01", "risk_fraction = 0.02")
                      .replace("spread = 0.2", "spread = 0.4"), encoding="utf-8")
    original = command(ROOT, "backtest", "--config", BACKTEST)
    changed = command(tmp_path, "backtest", "--config", config)
    assert original.returncode == changed.returncode == 0
    first, second = json.loads(original.stdout), json.loads(changed.stdout)
    assert first["dataset_id"] == second["dataset_id"]
    assert first["strategy"]["risk_fraction"] != second["strategy"]["risk_fraction"]
    assert first["execution"]["spread"] != second["execution"]["spread"]
    assert first["fills"] != second["fills"]


@pytest.mark.case("REP-004")
@pytest.mark.parametrize(("before", "after", "diagnostic"), [
    ('name = "single_long"', 'name = "other"', "strategy"),
    pytest.param("spread = 0.2", "funding = 0.2", "funding", marks=pytest.mark.case("EXE-010")),
    ('"backtest-data.toml"', '"missing.toml"', "missing.toml"),
    ("risk_fraction = 0.01", "risk_fraction = 2", "risk_fraction"),
])
def test_backtest_cli_errors_do_not_emit_success_report(tmp_path, before, after, diagnostic):
    text = BACKTEST.read_text(encoding="utf-8").replace(before, after)
    text = text.replace('"backtest-data.toml"', f'"{(BACKTEST.parent / "backtest-data.toml").as_posix()}"')
    config = tmp_path / "backtest.toml"
    config.write_text(text, encoding="utf-8")
    run = command(tmp_path, "backtest", "--config", config)
    assert run.returncode == 1
    assert not run.stdout
    assert diagnostic in run.stderr
    assert "Traceback" not in run.stderr


@pytest.mark.case("SIG-003")
@pytest.mark.case("SIG-004")
@pytest.mark.case("REP-005")
def test_backtest_cli_switches_strategies_using_only_the_strategy_table(tmp_path):
    configs = [tomllib.loads(path.read_text(encoding="utf-8")) for path in (BACKTEST, BACKTEST_SMA)]
    assert configs[0].pop("strategy") != configs[1].pop("strategy")
    assert configs[0] == configs[1]

    single = command(ROOT, "backtest", "--config", BACKTEST.relative_to(ROOT))
    crossover = command(tmp_path, "backtest", "--config", BACKTEST_SMA)
    assert single.returncode == 0, single.stderr
    assert crossover.returncode == 0, crossover.stderr
    first, second = json.loads(single.stdout), json.loads(crossover.stdout)
    assert first.keys() == second.keys()
    for key in ("format", "dataset_id", "execution", "provenance", "runtime"):
        assert first[key] == second[key]
    assert first["strategy"]["name"] == "single_long"
    assert second["strategy"] == {
        "name": "sma_crossover",
        "fast_period": 1,
        "slow_period": 2,
        "risk_fraction": "0.01",
        "stop_distance": "0.02",
        "lookback": 3,
    }
    entry, close = second["fills"]
    assert entry["order"]["decision_at"] == "2026-01-01T00:03:00+00:00"
    assert entry["candle_open_time"] == "2026-01-01T00:03:00+00:00"
    assert Decimal(entry["price"]) == Decimal("103.2")
    assert Decimal(close["price"]) == Decimal("104.8")
    assert Decimal(second["accounts"][-1]["equity"]) == Decimal("1004.176")


@pytest.mark.case("SIG-003")
@pytest.mark.parametrize(("old", "new", "diagnostic"), [
    ('"sma_crossover"', '"unknown"', "unknown strategy"),
    ("fast_period = 1", "fast_period = true", "fast_period must be a positive integer"),
    ("fast_period = 1", "fast_period = 0", "fast_period must be a positive integer"),
    ("slow_period = 2", "slow_period = 1", "fast_period must be smaller than slow_period"),
    ("slow_period = 2", "slow_period = 2.0", "slow_period must be a positive integer"),
    ("slow_period = 2", "", "missing strategy fields: slow_period"),
    ("slow_period = 2", "slow_period = 2\nlookback = 1", "unexpected strategy fields: lookback"),
])
def test_invalid_strategy_config_is_rejected_before_opening_data(tmp_path, old, new, diagnostic):
    text = BACKTEST_SMA.read_text(encoding="utf-8").replace(old, new)
    config = tmp_path / "strategy.toml"
    # The relative data config does not exist here. Strategy validation should win.
    config.write_text(text, encoding="utf-8")
    run = command(tmp_path, "backtest", "--config", config)
    assert run.returncode == 1
    assert run.stdout == ""
    assert diagnostic in run.stderr
    assert "backtest-data.toml" not in run.stderr


@pytest.mark.case("SIG-004")
@pytest.mark.case("REP-005")
def test_pullback_example_uses_the_execution_profile_tick(tmp_path):
    run = command(tmp_path, "backtest", "--config", BACKTEST_PULLBACK)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["format"] == "market-trader.backtest-report.v1"
    assert report["strategy"]["name"] == "two_legged_pullback"
    assert report["strategy"]["lookback"] == 4
    assert report["execution"]["price_tick"] == "0.25"
    entry, exit_fill = report["fills"]
    assert Decimal(entry["price"]) == Decimal("100.25")
    assert Decimal(entry["order"]["plan"]["stop"]) == Decimal("98.75")
    assert Decimal(entry["order"]["signal"]["target"]) == Decimal("102.50")
    assert (exit_fill["reason"], Decimal(exit_fill["price"])) == ("target", Decimal("102.50"))
    assert Decimal(report["accounts"][-1]["equity"]) == 1018


@pytest.mark.case("SIG-003")
@pytest.mark.parametrize(("old", "new", "field"), [
    ("movement_threshold = 0.003", "movement_threshold = 0", "movement_threshold"),
    ("bounce_threshold = 0.002", "bounce_threshold = true", "bounce_threshold"),
    ("stop_buffer = 0.002", "stop_buffer = 1", "stop_buffer"),
    ("reward_multiple = 2", "reward_multiple = -1", "reward_multiple"),
])
def test_invalid_pullback_parameters_fail_before_loading_data(tmp_path, old, new, field):
    config = tmp_path / "pullback.toml"
    config.write_text(BACKTEST_PULLBACK.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    run = command(tmp_path, "backtest", "--config", config)
    assert run.returncode == 1
    assert not run.stdout
    assert field in run.stderr
    assert "backtest-pullback-data.toml" not in run.stderr


def artifact_backtest_config(path, inputs):
    path.write_text(f"""version = 1
{inputs}

[strategy]
name = "single_long"
risk_fraction = 0.01
stop_distance = 0.02

[execution]
liquidate_at_end = true
""", encoding="utf-8")
    return path


@pytest.fixture
def session_artifact(tmp_path, minute_candles):
    first, second, _, fourth, fifth = minute_candles
    start, end = first.open_time, fifth.close_time
    calendar = SessionCalendar("two synthetic sessions", start, end, (
        TradingSession(start, second.close_time),
        TradingSession(fourth.open_time, end),
    ))
    request = DatasetRequest("TEST", timedelta(minutes=1), start, end)
    dataset, = normalize_candles(
        (first, second, fourth, fifth),
        config=NormalizationConfig(calendar=calendar, gap_policy="reject"),
        requests=(request,),
    ).datasets
    raw = tmp_path / "source.csv"
    raw.write_bytes(b"synthetic source snapshot\n")
    provenance = DatasetProvenance("synthetic sessions", (fingerprint_source(raw, name=raw.name),))
    path = tmp_path / "dataset.json"
    manifest = save_dataset(path, dataset, provenance=provenance)
    raw.unlink()
    return path, manifest


@pytest.mark.case("SET-004")
@pytest.mark.case("REP-002")
@pytest.mark.case("REP-003")
def test_backtest_uses_the_artifact_calendar_and_records_its_identity(tmp_path, session_artifact):
    artifact, manifest = session_artifact
    directory = tmp_path / "configs"
    directory.mkdir()
    config = artifact_backtest_config(directory / "run.toml", 'dataset_artifact = "../dataset.json"')
    first = command(ROOT, "backtest", "--config", config)
    second = command(tmp_path, "backtest", "--config", config.relative_to(tmp_path))
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""
    report = json.loads(first.stdout)
    assert report["dataset_id"] == manifest.dataset_id
    assert report["dataset_artifact_id"] == manifest.artifact_id
    assert report["provenance"]["sources"][0]["name"] == "source.csv"
    assert len(report["accounts"]) == 5
    entry, close = report["fills"]
    assert entry["candle_open_time"] == "2026-01-01T00:01:00+00:00"
    assert (Decimal(entry["price"]), Decimal(close["price"])) == (101, Decimal("104.5"))
    assert Decimal(report["accounts"][-1]["equity"]) == 1014
    assert load_dataset(artifact).manifest == manifest


@pytest.mark.case("REP-004")
@pytest.mark.parametrize("failure", ["missing", "checksum"])
def test_backtest_reports_artifact_verification_errors(tmp_path, session_artifact, failure):
    artifact, _ = session_artifact
    if failure == "missing":
        artifact.unlink()
    else:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["provenance"]["description"] = "changed after publication"
        artifact.write_text(json.dumps(payload), encoding="utf-8")
    config = artifact_backtest_config(tmp_path / "run.toml", 'dataset_artifact = "dataset.json"')
    run = command(tmp_path, "backtest", "--config", config)
    assert run.returncode == 1
    assert not run.stdout
    assert "verify artifact" in run.stderr and "dataset.json" in run.stderr
    if failure == "checksum":
        assert "checksum mismatch" in run.stderr
    assert "Traceback" not in run.stderr


@pytest.mark.case("REP-004")
@pytest.mark.parametrize(("inputs", "diagnostic"), [
    ("", "exactly one of data_config and dataset_artifact"),
    ('data_config = "missing.toml"\ndataset_artifact = "missing.json"',
     "exactly one of data_config and dataset_artifact"),
    ("dataset_artifact = 1", "dataset_artifact must be a nonempty string"),
])
def test_backtest_requires_one_valid_input_path(tmp_path, inputs, diagnostic):
    config = artifact_backtest_config(tmp_path / "run.toml", inputs)
    run = command(tmp_path, "backtest", "--config", config)
    assert run.returncode == 1
    assert not run.stdout
    assert "read backtest config" in run.stderr and diagnostic in run.stderr


@pytest.mark.case("DAT-005")
@pytest.mark.case("REP-004")
def test_backtest_rejects_active_gaps_in_a_verified_artifact(tmp_path, minute_candles):
    dataset, = normalize_candles(
        (minute_candles[0], minute_candles[2]),
        config=NormalizationConfig(calendar="continuous", gap_policy="report"),
    ).datasets
    artifact = tmp_path / "dataset.json"
    save_dataset(artifact, dataset, provenance=DatasetProvenance("synthetic series with a gap"))
    assert load_dataset(artifact).dataset.quality.gaps
    config = artifact_backtest_config(tmp_path / "run.toml", 'dataset_artifact = "dataset.json"')
    run = command(tmp_path, "backtest", "--config", config)
    assert run.returncode == 1
    assert not run.stdout
    assert "run backtest" in run.stderr and "missing candles" in run.stderr
