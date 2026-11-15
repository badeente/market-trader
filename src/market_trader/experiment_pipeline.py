"""Actual AP2 training/validation pipeline using the existing model and ledger.

No held-out scoring, run identity, parameter search or alternate accounting.
Shared AP1 helpers only serialize, backtest, verify alignment and publish; all
features, fitting, predictions and artifacts are produced afresh for this config.
"""
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from market_trader._dataset_codec import json_bytes
from market_trader.artifacts import DatasetProvenance
from market_trader.backtesting import ExecutionProfile
from market_trader.baseline import baseline_report, load_ridge_model, validation_evaluation
from market_trader.dukascopy_import import import_dukascopy_reference, slice_session_dataset
from market_trader.experiment_config import (
    ExperimentConfig, experiment_config_from_payload, experiment_config_payload,
)
from market_trader.experiment_reporting import enrich_experiment_report
from market_trader.reference import _START, _VALIDATION, _TEST, _END, trading_summary
from market_trader.reference_pipeline import (
    _assert_trade_alignment, _backtest, _json_value, _publish_directory, _runtime, _source_id, _write_json,
)
from market_trader.reporting import render_report
from market_trader.storage import load_dataset, publish_new_file, save_dataset
from market_trader.training_data import OPEN_TARGET_NAME, OPEN_TRAINING_DATA_FORMAT, TrainingConfig, build_open_training_data


def _label_checks(training, development):
    """Independently verify open-based labels and availability for configured horizon."""
    by_open = {bar.open_time: bar for bar in development.candles}
    checks = []
    with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
        for split, rows in (('train', training.train), ('validation', training.validation)):
            for index in sorted({0, len(rows) // 2, len(rows) - 1}):
                row = rows[index]
                entry, exit_bar = by_open[row.entry_at], by_open[row.exit_at]
                rational = Fraction(exit_bar.open - entry.open) / Fraction(entry.open)
                expected = Decimal(rational.numerator) / Decimal(rational.denominator)
                availability = max(by_open[row.entry_at + step * development.interval].available_at
                                   for step in range(training.config.horizon_bars + 1))
                if (row.target_return != expected or row.target_available_at != availability
                        or row.entry_at != row.decision_at
                        or row.exit_at - row.entry_at != training.config.horizon_bars * development.interval):
                    raise ValueError('independent AP2 label/availability check failed')
                checks.append({'split': split, 'decision_at': row.decision_at, 'entry_at': row.entry_at,
                               'exit_at': row.exit_at, 'entry_reference_open': entry.open,
                               'exit_reference_open': exit_bar.open, 'target_return': expected,
                               'target_available_at': availability, 'rational_recalculation_matches': True})
    return checks


def execute_experiment(config: ExperimentConfig, output: Path) -> dict:
    """Train and evaluate a config into a new private, checksum-complete directory."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'experiment output already exists: {output}')
    # Reject manually constructed invalid configs too, before import or output writes.
    config = experiment_config_from_payload(experiment_config_payload(config))
    snapshot = experiment_config_payload(config)
    runtime = _runtime()
    template = Path(__file__).with_name('report_template.html.in')
    runtime['report_template_sha256'] = 'sha256:' + sha256(template.read_bytes()).hexdigest()
    imported = import_dukascopy_reference(config.source_csv, instrument='deuidxeur',
        start=_START, end=_END, expected_sha256=config.source_sha256)
    # Raw 2025 is structurally audited only; physically exclude it before features.
    development = slice_session_dataset(imported.dataset, _START, _TEST)
    validation = slice_session_dataset(imported.dataset, _VALIDATION, _TEST)
    if any(bar.close_time > _TEST for bar in development.candles):
        raise ValueError('reserved candle reached development')
    training = build_open_training_data(development,
        TrainingConfig(config.lookback_bars, config.horizon_bars, _VALIDATION, _TEST), include_test=False)
    if training.test or any(row.decision_at >= _TEST for row in (*training.train, *training.validation)):
        raise ValueError('reserved rows reached AP2 model features')
    profile = ExecutionProfile(initial_capital=config.initial_capital, spread=config.spread,
        slippage=config.slippage, price_tick=config.price_tick, quantity_step=Decimal(1),
        commission_rate=config.commission_rate, allow_short=False, liquidate_at_end=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.market-trader-ap2-', dir=output.parent) as directory:
        stage = Path(directory)
        origin = DatasetProvenance('Dukascopy DEUIDXEUR bid; complete-session research filter', (imported.source,))
        full = save_dataset(stage / 'research-dataset.json', imported.dataset, provenance=origin)
        dev_origin = DatasetProvenance('AP2 development excludes reserved 2025', (imported.source,), (full.dataset_id,))
        dev_manifest = save_dataset(stage / 'development-dataset.json', development, provenance=dev_origin)
        val_origin = DatasetProvenance('AP2 validation 2024; hypothetical research costs',
                                       (imported.source,), (dev_manifest.dataset_id,))
        val_manifest = save_dataset(stage / 'validation-dataset.json', validation, provenance=val_origin)
        for name in ('research', 'development', 'validation'):
            load_dataset(stage / f'{name}-dataset.json')
        _write_json(stage / 'import-audit.json', imported.audit)
        _write_json(stage / 'config.json', snapshot)
        _write_json(stage / 'label-checks.json', _label_checks(training, development))
        body = {'format': OPEN_TRAINING_DATA_FORMAT, 'dataset_id': dev_manifest.dataset_id,
            'dataset_artifact_id': dev_manifest.artifact_id, 'instrument': development.instrument,
            'interval_microseconds': development.interval // timedelta(microseconds=1),
            'config': asdict(training.config), 'feature_names': training.feature_names, 'target_name': OPEN_TARGET_NAME,
            'splits': {'train': [asdict(row) for row in training.train],
                       'validation': [asdict(row) for row in training.validation], 'test': []}}
        training_id = 'sha256:' + sha256(json_bytes(body, default=_json_value)).hexdigest()
        training_path, model_path = stage / 'training-data.json', stage / 'ridge-model.json'
        _write_json(training_path, {**body, 'training_data_id': training_id})
        model_report = baseline_report(training_path)
        _write_json(model_path, model_report)
        if load_ridge_model(model_path).target_name != OPEN_TARGET_NAME:
            raise ValueError('incompatible AP2 model target')
        evaluation, points = validation_evaluation(model_path, training_path)
        if any(not _VALIDATION <= datetime.fromisoformat(point['at']) < _TEST for point in points):
            raise ValueError('validation forecasts escaped 2024')
        _write_json(stage / 'validation-evaluation.json', evaluation)
        _write_json(stage / 'validation-predictions.json', points)
        ridge = _backtest(stage, 'ridge', {'name': 'ridge_long', 'model_report': 'ridge-model.json',
            'entry_threshold': config.entry_threshold, 'quantity': Decimal(1)},
            validation, val_manifest, val_origin, runtime, profile)
        sma = _backtest(stage, 'sma', {'name': 'sma_timed', 'fast_period': config.sma_fast_period,
            'slow_period': config.sma_slow_period, 'holding_bars': config.horizon_bars, 'quantity': Decimal(1)},
            validation, val_manifest, val_origin, runtime, profile)
        for report in (ridge, sma):
            _assert_trade_alignment(report, training.validation)
        signature = {'source_sha256': imported.source.sha256,
            'development_dataset_id': development.dataset_id, 'validation_dataset_id': validation.dataset_id,
            'target_name': OPEN_TARGET_NAME, 'feature_names': list(training.feature_names),
            'lookback_bars': config.lookback_bars, 'horizon_bars': config.horizon_bars,
            'interval_microseconds': body['interval_microseconds'],
            'splits': {'train_start': _START.isoformat(), 'validation_start': _VALIDATION.isoformat(),
                       'test_start': _TEST.isoformat(), 'test_end': _END.isoformat()},
            'execution': asdict(profile), 'quantity': '1', 'contract_multiplier': '1',
            'runtime': runtime}
        summary = {'format': 'market-trader.ap2-summary.v1', 'name': config.name,
            'hypothesis': config.hypothesis, 'seed': config.seed, 'config': snapshot,
            'training_rows': len(training.train), 'validation_rows': len(training.validation),
            'test_metrics_computed': False, 'reserved_test_start': _TEST.isoformat(), 'reserved_test_end': _END.isoformat(),
            'target_name': OPEN_TARGET_NAME, 'training_data_id': training_id,
            'model_report_id': model_report['report_id'], 'development_dataset_id': development.dataset_id,
            'validation_dataset_id': validation.dataset_id, 'source_sha256': imported.source.sha256,
            'runtime': runtime, 'comparison_signature': signature,
            'validation_forecast': evaluation['validation'],
            'trading': {'ridge': trading_summary(ridge), 'sma_timed': trading_summary(sma)}}
        # Normalize once, so the API and stored summary have identical JSON semantics.
        summary = json.loads(json_bytes(summary, default=_json_value))
        _write_json(stage / 'summary.json', summary)
        report = render_report(stage / 'ridge-backtest.json', compare=stage / 'sma-backtest.json',
                               model=model_path, dataset=stage / 'validation-dataset.json')
        publish_new_file(stage / 'report.html', enrich_experiment_report(report, summary=summary))
        if (_source_id() != runtime['code_id']
                or 'sha256:' + sha256(template.read_bytes()).hexdigest() != runtime['report_template_sha256']):
            raise ValueError('source changed during experiment; retry from a stable checkout')
        files = {path.name: 'sha256:' + sha256(path.read_bytes()).hexdigest() for path in sorted(stage.iterdir())}
        _write_json(stage / 'manifest.json', {'format': 'market-trader.ap2-manifest.v1', 'status': 'complete',
            'test_metrics_computed': False, 'runtime': runtime, 'artifacts': files})
        _publish_directory(stage, output)
    return summary
