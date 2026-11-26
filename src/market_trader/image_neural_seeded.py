"""AP8 seed1/2 CNNs; the AP7 network, recipe and training policy are unchanged.

Inference is immutable stdlib-only JSON. Optional Torch/NumPy are used only
inside the shared bounded fitting kernel, never while loading or scoring.
"""
from pathlib import Path
import tempfile

from market_trader.cnn_settings import _exact
from market_trader.image_neural import (
    SavedCNNModel, _bindings, _decode_saved, _fit_for_seed, _metrics, _number,
    _points, _read_report, _seal, _training_rows,
)

SEEDED_CNN_FORMAT = 'market-trader.cnn-validation.v2'


def _seed(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError('AP8 CNN requires seed 1 or 2')
    return value


def _decode_seeded(saved):
    training = saved['training']
    if type(training) is not dict or type(training.get('fit_settings')) is not dict:
        raise ValueError('invalid frozen CNN training settings')
    seed = _seed(training['fit_settings'].get('seed'))
    return _decode_saved(saved, expected_seed=seed)


def load_seeded_cnn_model(path: Path) -> SavedCNNModel:
    """Load only bounded v2 JSON with an actual recorded seed1/2 policy."""
    return _decode_seeded(_read_report(path, expected_format=SEEDED_CNN_FORMAT))


def seeded_cnn_report(dataset_path: Path, training_path: Path, *, seed: int,
                      cache_dir=None) -> dict:
    """Fit the unchanged frozen CNN policy with actual seed1/2 initialization."""
    seed = _seed(seed)
    from market_trader.image_training_data import prepare_image_cohort, build_input_cache
    cohort = prepare_image_cohort(dataset_path, training_path)

    def run(directory):
        cache = build_input_cache(cohort, Path(directory), accelerated=True)
        predictions = []
        model, training = _fit_for_seed(
            cache, cohort.training.train, cohort.training.validation, seed,
            expected_seed=seed, validation_predictions=predictions)
        points = [dict(at=row.decision_at.isoformat(), predicted=predicted,
                       actual=_number(float(row.target_return)))
                  for row, predicted in zip(cohort.training.validation, predictions, strict=True)]
        report = _seal(dict(format=SEEDED_CNN_FORMAT, **_bindings(cohort), model=model,
                           training={**_training_rows(cohort), **training}, validation=_metrics(points)))
        _decode_seeded(report)
        return report

    if cache_dir is not None:
        return run(cache_dir)
    with tempfile.TemporaryDirectory(prefix='market-trader-cnn-seeded-') as private:
        return run(Path(private)/'cache')


def seeded_cnn_validation_evaluation(model_path: Path, dataset_path: Path,
                                     training_path: Path, *, cache_dir=None):
    """Score every validation row, with exact frozen source/cohort bindings."""
    from market_trader.image_training_data import prepare_image_cohort, build_input_cache
    saved = _read_report(model_path, expected_format=SEEDED_CNN_FORMAT)
    predictor = _decode_seeded(saved)
    cohort = prepare_image_cohort(dataset_path, training_path)
    bindings = _bindings(cohort)
    if any(not _exact(saved[k], value) for k, value in bindings.items()):
        raise ValueError('CNN source/cohort/model binding mismatch')
    if any(not _exact(saved['training'][k], value) for k, value in _training_rows(cohort).items()):
        raise ValueError('CNN training row binding mismatch')
    if saved['validation']['row_count'] != len(cohort.training.validation):
        raise ValueError('CNN validation row binding mismatch')

    def run(directory):
        cache = build_input_cache(cohort, Path(directory), accelerated=False)
        points = _points(predictor.model, cohort, cache)
        validation = dict(**_metrics(points), first_decision_at=points[0]['at'],
                          last_decision_at=points[-1]['at'])
        evaluation = _seal(dict(format='market-trader.cnn-seeded-validation-evaluation.v1',
                               model_kind='cnn', model_report_id=predictor.report_id,
                               **bindings, validation=validation))
        return evaluation, points

    if cache_dir is not None:
        return run(cache_dir)
    with tempfile.TemporaryDirectory(prefix='market-trader-cnn-seeded-evaluate-') as private:
        return run(Path(private)/'cache')
