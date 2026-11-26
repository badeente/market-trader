"""Frozen AP7 CNN. Safe JSON and stdlib inference; optional Torch fits only."""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import fsum, isfinite, prod
from pathlib import Path
from statistics import fmean, pstdev
import os
import platform
import re
import stat
import tempfile

from market_trader._dataset_codec import decode_record, encode, json_bytes, parse_json, record_object
from market_trader.baseline import _error_metrics
from market_trader.cnn_settings import canonical_cnn_settings, _exact
from market_trader.training_data import OPEN_TARGET_NAME, TrainingConfig

CNN_FORMAT = 'market-trader.cnn-validation.v1'
_SHAPES = {'conv.weight':[4,3,3,3], 'conv.bias':[4],
           'hidden.weight':[8,660], 'hidden.bias':[8],
           'output.weight':[1,8], 'output.bias':[1]}
_START = datetime(2021,1,1,tzinfo=UTC)
_VALIDATION = datetime(2024,1,1,tzinfo=UTC)
_END = datetime(2025,1,1,tzinfo=UTC)
_MAX_BYTES = 2 * 1024 * 1024


def _number(value):
    if type(value) not in (int,float):
        raise ValueError('expected finite model number')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('model number outside finite range') from error
    if not isfinite(result):
        raise ValueError('expected finite model number')
    return result


@dataclass(frozen=True, slots=True)
class CNNModel:
    target_mean: float
    target_scale: float
    tensors: tuple

    def predict_sums(self, sums) -> float:
        if not isinstance(sums, (tuple,list)) or len(sums) != 2304 or any(
                type(x) is not int or not 0 <= x <= 16320 for x in sums):
            raise ValueError('expected 2304 bounded integer RGB block sums')
        pixels = tuple(x / 64 / 255 for x in sums)
        cw, cb, hw, hb, ow, ob = self.tensors
        try:
            pooled = []
            for channel in range(4):
                weights = cw[channel*27:(channel+1)*27]
                # CHW contiguous convolution and pool; no padding, stride1/2.
                convolution = []
                for y in range(22):
                    for x in range(30):
                        values = (pixels[c*768+(y+dy)*32+x+dx]
                                  for c in range(3) for dy in range(3) for dx in range(3))
                        value = cb[channel] + fsum(w*v for w,v in zip(weights,values,strict=True))
                        if not isfinite(value):
                            raise ValueError('nonfinite convolution output')
                        convolution.append(max(0.,value))
                for y in range(0,22,2):
                    for x in range(0,30,2):
                        pooled.append(fsum((convolution[y*30+x],convolution[y*30+x+1],
                                           convolution[(y+1)*30+x],convolution[(y+1)*30+x+1]))/4)
            hidden = tuple(hb[i]+fsum(w*v for w,v in
                           zip(hw[i*660:(i+1)*660],pooled,strict=True)) for i in range(8))
            if not all(isfinite(x) for x in hidden):
                raise ValueError('nonfinite hidden output')
            hidden = tuple(max(0.,x) for x in hidden)
            result = self.target_mean + self.target_scale*(ob[0]+fsum(w*v for w,v in zip(ow,hidden,strict=True)))
        except (OverflowError,ValueError) as error:
            raise ValueError('CNN prediction outside finite range') from error
        if not isfinite(result):
            raise ValueError('CNN prediction outside finite range')
        return result


def _decode_model(value) -> CNNModel:
    data = record_object(value, ('kind','architecture','parameter_count','target_mean','target_scale','tensors'))
    if (data['kind'] != 'cnn' or data['architecture'] != 'tiny_cnn_v1'
            or type(data['parameter_count']) is not int or data['parameter_count'] != 5409):
        raise ValueError('unsupported CNN architecture; expected 5409 parameters')
    tensors = record_object(data['tensors'], tuple(_SHAPES))
    decoded = []
    for name,shape in _SHAPES.items():
        tensor = record_object(tensors[name], ('shape','values'))
        if not _exact(tensor['shape'],shape):
            raise ValueError('invalid CNN tensor shape')
        values = tensor['values']
        if type(values) is not list or len(values) != prod(shape):
            raise ValueError('invalid CNN tensor values shape')
        decoded.append(tuple(_number(x) for x in values))
    mean,scale = _number(data['target_mean']),_number(data['target_scale'])
    if scale <= 0:
        raise ValueError('target scale must be positive')
    return CNNModel(mean,scale,tuple(decoded))


@dataclass(frozen=True, slots=True)
class SavedCNNModel:
    report_id: str
    model: CNNModel
    instrument: str
    interval: timedelta
    config: TrainingConfig
    last_target_available_at: datetime
    target_name: str = OPEN_TARGET_NAME

    def predict_window(self, candles, *, decision_at) -> float:
        from market_trader.chart_images import render_candles
        from market_trader.image_pixels_v1 import block_sums
        if (not isinstance(decision_at, datetime) or decision_at.utcoffset() != timedelta(0)
                or not _START <= decision_at < _END):
            raise ValueError('CNN decision must be in 2021–2024 UTC')
        if (not isinstance(candles,(tuple,list)) or len(candles) != 8
                or any(getattr(c,'instrument',None) != self.instrument for c in candles)):
            raise ValueError('CNN window instrument or length mismatch')
        return self.model.predict_sums(block_sums(render_candles(candles,decision_at=decision_at)))


def _seal(body):
    return {**body, 'report_id':'sha256:'+sha256(json_bytes(body)).hexdigest()}


def _read_report(path, *, expected_format=CNN_FORMAT):
    # Open nonblocking before inspecting type: FIFOs/devices cannot stall loading.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
            raise ValueError('expected bounded regular CNN JSON file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(_MAX_BYTES+1)
        if len(raw) > _MAX_BYTES:
            raise ValueError('expected bounded CNN JSON file')
    finally:
        os.close(fd)
    saved = record_object(parse_json(raw.decode('utf-8'), allow_floats=True), (
        'format','report_id','training_data_id','input_index_id','source_training_data_id',
        'dataset_id','dataset_artifact_id','instrument','interval_microseconds','target_name',
        'config','input_recipe','model','training','validation'))
    if saved['format'] != expected_format:
        raise ValueError('unsupported CNN report format')
    if saved['report_id'] != _seal({k:v for k,v in saved.items() if k != 'report_id'})['report_id']:
        raise ValueError('CNN report checksum mismatch')
    return saved


def _identifier(value):
    if type(value) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}', value) is None:
        raise ValueError('expected sha256 identifier')


def _count(value):
    if type(value) is not int or not 1 <= value <= 50000:
        raise ValueError('expected bounded positive row count')
    return value


def _timestamp(value):
    if type(value) is not str:
        raise ValueError('expected UTC timestamp string')
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() != timedelta(0):
        raise ValueError('expected UTC timestamp')
    return stamp


def _history(value, policy):
    if type(value) is not list or not 1 <= len(value) <= policy['max_epochs']:
        raise ValueError('invalid epoch history length')
    significant, stale, losses = float('inf'), 0, []
    for index, item in enumerate(value, 1):
        epoch = record_object(item, ('epoch','training_mse','validation_mse'))
        if type(epoch['epoch']) is not int or epoch['epoch'] != index:
            raise ValueError('epoch history must be sequential')
        if any(_number(epoch[k]) < 0 for k in ('training_mse','validation_mse')):
            raise ValueError('epoch history losses must be nonnegative')
        loss = _number(epoch['validation_mse'])
        losses.append(loss)
        if loss < significant - float(policy['min_delta']):
            significant, stale = loss, 0
        else:
            stale += 1
        if stale >= policy['patience'] and index != len(value):
            raise ValueError('epoch history continues past early stopping')
    reason = 'early_stopping' if stale >= policy['patience'] else 'max_epochs'
    if reason == 'max_epochs' and len(value) != policy['max_epochs']:
        raise ValueError('epoch history ends before stopping condition')
    return losses.index(min(losses))+1, reason


def _validate_training(value, *, expected_seed=0):
    data = record_object(value, ('row_count','first_decision_at','last_target_available_at',
                                'fit_settings','runtime_versions','epoch_history','best_epoch','stopping_reason','inference_parity'))
    _count(data['row_count'])
    first, last = (_timestamp(data[k]) for k in ('first_decision_at','last_target_available_at'))
    if not _START <= first <= last < _VALIDATION:
        raise ValueError('training cutoff must be 2021–2023')
    policy = canonical_cnn_settings()
    if not _exact(data['fit_settings'], {**policy, 'seed':expected_seed}):
        raise ValueError('invalid frozen CNN fit settings or seed')
    versions = record_object(data['runtime_versions'], ('python','torch','numpy'))
    if any(type(v) is not str or not v or v != v.strip() or len(v) > 128 for v in versions.values()):
        raise ValueError('invalid runtime versions')
    best, reason = _history(data['epoch_history'], policy)
    if type(data['best_epoch']) is not int or data['best_epoch'] != best:
        raise ValueError('best_epoch must identify actual minimum validation checkpoint')
    if data['stopping_reason'] != reason:
        raise ValueError('stopping reason inconsistent with history')
    return last


def _decode_saved(saved, *, expected_seed=0):
    from market_trader.image_pixels_v1 import image_input_recipe
    for field in ('report_id','training_data_id','input_index_id','source_training_data_id','dataset_id','dataset_artifact_id'):
        _identifier(saved[field])
    if saved['training_data_id'] != saved['input_index_id']:
        raise ValueError('image index binding mismatch')
    if not _exact(saved['input_recipe'], image_input_recipe()):
        raise ValueError('input recipe does not match installed frozen implementation')
    if saved['instrument'] != 'deuidxeur' or saved['target_name'] != OPEN_TARGET_NAME:
        raise ValueError('CNN requires literal deuidxeur and open target')
    if type(saved['interval_microseconds']) is not int or saved['interval_microseconds'] != 900000000:
        raise ValueError('CNN requires m15 interval')
    config = decode_record(saved['config'], TrainingConfig)
    if (config.lookback_bars != 8 or config.horizon_bars != 2
            or config.validation_start != _VALIDATION or config.test_start != _END):
        raise ValueError('CNN requires frozen 8/2 2021–2024 configuration')
    last = _validate_training(saved['training'], expected_seed=expected_seed)
    metrics = record_object(saved['validation'], ('row_count','cnn','zero_return'))
    _count(metrics['row_count'])
    parity = record_object(saved['training']['inference_parity'], ('row_count','max_abs_error',
                           'max_tolerance_ratio','absolute_tolerance','relative_tolerance'))
    if (_count(parity['row_count']) != metrics['row_count']
            or _number(parity['max_abs_error']) < 0
            or not 0 <= _number(parity['max_tolerance_ratio']) <= 1
            or parity['absolute_tolerance'] != '0.00000000000001'
            or parity['relative_tolerance'] != '0.0000000000001'):
        raise ValueError('invalid all-validation inference parity metadata')
    if saved['training']['row_count'] + metrics['row_count'] > 50000:
        raise ValueError('cohort exceeds bounded row count')
    for key in ('cnn','zero_return'):
        errors = record_object(metrics[key], ('mae','rmse'))
        if any(_number(x) < 0 for x in errors.values()):
            raise ValueError('metrics must be nonnegative')
    return SavedCNNModel(saved['report_id'], _decode_model(saved['model']), saved['instrument'],
                         timedelta(microseconds=saved['interval_microseconds']), config,
                         last, saved['target_name'])


def load_cnn_model(path: Path) -> SavedCNNModel:
    """Load only bounded versioned JSON; never invokes a framework loader."""
    return _decode_saved(_read_report(path))


def _fit(cache, train, validation, seed=0, settings=None, *, validation_predictions=None):
    """Frozen AP7 seed0 entry point; nonzero seeds require the AP8 API."""
    if type(seed) is not int or seed != 0:
        raise ValueError('CNN requires frozen seed 0')
    return _fit_for_seed(cache, train, validation, seed, settings, expected_seed=0,
                         validation_predictions=validation_predictions)


def _fit_for_seed(cache, train, validation, seed, settings=None, *, expected_seed,
                  validation_predictions=None):
    """Same bounded AP7 kernel, with an explicitly checked initialization seed."""
    if (type(seed) is not int or type(expected_seed) is not int
            or expected_seed not in (0, 1, 2) or seed != expected_seed):
        raise ValueError('CNN seed does not match the requested frozen run')
    policy = canonical_cnn_settings() if settings is None else canonical_cnn_settings(settings, snapshot=True)
    if not train or not validation or len(train)+len(validation) > 50000:
        raise ValueError('CNN requires bounded nonempty train/validation')
    if cache.row_count != len(train)+len(validation) or cache.train_count != len(train):
        raise ValueError('cache split dimensions do not match training rows')
    import random
    python_rng = random.getstate()
    import numpy as np
    import torch
    numpy_rng = np.random.get_state()
    rng, threads, dtype = torch.get_rng_state().clone(), torch.get_num_threads(), torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    mapped = None
    try:
        targets = tuple(_number(float(row.target_return)) for row in train)
        mean, scale = fmean(targets), pstdev(targets) or 1.
        if not isfinite(mean) or not isfinite(scale) or scale <= 0:
            raise ValueError('nonfinite target scaler')
        labels = tuple((_number(float(row.target_return))-mean)/scale for row in (*train,*validation))
        if not all(isfinite(y) for y in labels):
            raise ValueError('standardized targets must be finite')
        mapped = np.memmap(cache.path, dtype='<u2', mode='r', shape=(cache.row_count,3,24,32))
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.random.default_generator.manual_seed(seed)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        network = torch.nn.Sequential(
            torch.nn.Conv2d(3,4,3,padding=0,dtype=torch.float64,device='cpu'), torch.nn.ReLU(),
            torch.nn.AvgPool2d(2), torch.nn.Flatten(),
            torch.nn.Linear(660,8,dtype=torch.float64,device='cpu'), torch.nn.ReLU(),
            torch.nn.Linear(8,1,dtype=torch.float64,device='cpu'))
        if sum(p.numel() for p in network.parameters()) != 5409:
            raise ValueError('runtime CNN parameter count mismatch')
        optimizer = torch.optim.AdamW(network.parameters(), lr=float(policy['learning_rate']),
            weight_decay=float(policy['weight_decay']), betas=tuple(map(float,policy['betas'])),
            eps=float(policy['eps']),amsgrad=False,foreach=False,fused=False,maximize=False)

        def batch(indices):
            # Advanced indexing copies at most 128 compact rows; conversion is
            # likewise minibatch-only. np.memmap never becomes a cohort tensor.
            values = mapped[indices]
            if bool((values > 16320).any()):
                raise ValueError('cache contains invalid block sums')
            x = torch.tensor(values, dtype=torch.float64, device='cpu') / 64 / 255
            y = torch.tensor([[labels[i]] for i in indices], dtype=torch.float64, device='cpu')
            return x, y

        def mse(start, stop):
            total = 0.
            with torch.no_grad():
                for begin in range(start,stop,policy['batch_size']):
                    indices = list(range(begin,min(begin+policy['batch_size'],stop)))
                    x, y = batch(indices)
                    total += float(torch.nn.functional.mse_loss(network(x),y,reduction='sum'))
            return total / (stop-start)

        history, best_state = [], None
        best, significant, stale, best_epoch = float('inf'),float('inf'),0,0
        reason = 'max_epochs'
        for epoch in range(1,policy['max_epochs']+1):
            network.train()
            permutation = torch.randperm(len(train),generator=generator,device='cpu')
            for indices in permutation.split(policy['batch_size']):
                x, y = batch(indices.tolist())
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(network(x),y)
                loss.backward()
                optimizer.step()
            network.eval()
            training_loss, validation_loss = mse(0,len(train)),mse(len(train),cache.row_count)
            if not isfinite(training_loss) or not isfinite(validation_loss):
                raise ValueError('CNN training produced nonfinite losses')
            history.append(dict(epoch=epoch,training_mse=training_loss,validation_mse=validation_loss))
            if validation_loss < best:
                best, best_epoch = validation_loss, epoch
                best_state = {name:value.detach().clone() for name,value in network.state_dict().items()}
            if validation_loss < significant-float(policy['min_delta']):
                significant, stale = validation_loss, 0
            else:
                stale += 1
            if stale >= policy['patience']:
                reason = 'early_stopping'
                break
        network.load_state_dict(best_state)
        network.eval()
        layers = (network[0].weight,network[0].bias,network[4].weight,network[4].bias,network[6].weight,network[6].bias)
        model = dict(kind='cnn',architecture='tiny_cnn_v1',parameter_count=5409,
            target_mean=mean,target_scale=scale,
            tensors={name:dict(shape=list(value.shape),values=value.detach().flatten().tolist())
                     for name,value in zip(_SHAPES,layers,strict=True)})
        # Reject nonfinite exported weights even if losses happened to be finite.
        predictor = _decode_model(model)
        max_error, max_ratio = 0., 0.
        with torch.no_grad():
            for begin in range(len(train),cache.row_count,policy['batch_size']):
                indices = list(range(begin,min(begin+policy['batch_size'],cache.row_count)))
                x, _ = batch(indices)
                native = (network(x).flatten()*scale+mean).tolist()
                for index, expected in zip(indices,native,strict=True):
                    sums = tuple(int(v) for v in mapped[index].ravel())
                    actual = predictor.predict_sums(sums)
                    error = abs(actual-expected)
                    ratio = error/(1e-14+1e-13*abs(expected))
                    if not isfinite(error) or not isfinite(ratio) or ratio > 1:
                        raise ValueError('CNN stdlib/Torch inference parity exceeds fixed tolerance')
                    max_error, max_ratio = max(max_error,error),max(max_ratio,ratio)
                    if validation_predictions is not None:
                        validation_predictions.append(actual)
        parity = dict(row_count=len(validation),max_abs_error=max_error,max_tolerance_ratio=max_ratio,
                      absolute_tolerance='0.00000000000001',relative_tolerance='0.0000000000001')
        return model, dict(fit_settings={**policy,'seed':seed},epoch_history=history,
            best_epoch=best_epoch,stopping_reason=reason,inference_parity=parity,
            runtime_versions=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=str(np.__version__)))
    finally:
        try:
            if mapped is not None:
                mapped._mmap.close()
        finally:
            # Cleanup I/O failure must not leak RNG or process-wide settings.
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(rng)
            torch.set_num_threads(threads)
            torch.set_default_dtype(dtype)
            torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)


def _bindings(cohort):
    index = cohort.index_document()
    return dict(training_data_id=index['input_index_id'], input_index_id=index['input_index_id'],
        source_training_data_id=cohort.training_data_id, dataset_id=cohort.data.dataset_id,
        dataset_artifact_id=cohort.dataset_artifact_id, instrument=cohort.data.instrument,
        interval_microseconds=cohort.data.interval // timedelta(microseconds=1),
        config=encode(cohort.training.config), target_name=OPEN_TARGET_NAME,
        input_recipe=index['input_recipe'])


def _training_rows(cohort):
    rows = cohort.training.train
    return dict(row_count=len(rows),first_decision_at=rows[0].decision_at.isoformat(),
                last_target_available_at=max(row.target_available_at for row in rows).isoformat())


def _points(model, cohort, cache):
    return [dict(at=row.decision_at.isoformat(),
                 predicted=model.predict_sums(cache.row(cache.train_count+i)),
                 actual=_number(float(row.target_return)))
            for i,row in enumerate(cohort.training.validation)]


def _metrics(points):
    return dict(row_count=len(points),
                cnn=_error_metrics(tuple(p['predicted']-p['actual'] for p in points)),
                zero_return=_error_metrics(tuple(p['actual'] for p in points)))


def cnn_report(dataset_path: Path, training_path: Path, *, seed=0, settings=None, cache_dir=None) -> dict:
    """Fit only when explicitly called; the cache is temporary unless supplied."""
    if type(seed) is not int or seed != 0:
        raise ValueError('CNN requires frozen seed 0')
    policy = canonical_cnn_settings() if settings is None else canonical_cnn_settings(settings,snapshot=True)
    from market_trader.image_training_data import prepare_image_cohort, build_input_cache
    cohort = prepare_image_cohort(dataset_path,training_path)

    def run(directory):
        cache = build_input_cache(cohort,Path(directory),accelerated=True)
        predictions = []
        model, training = _fit(cache,cohort.training.train,cohort.training.validation,seed,policy,
                               validation_predictions=predictions)
        points = [dict(at=row.decision_at.isoformat(),predicted=predicted,actual=_number(float(row.target_return)))
                  for row,predicted in zip(cohort.training.validation,predictions,strict=True)]
        report = _seal(dict(format=CNN_FORMAT,**_bindings(cohort),model=model,
            training={**_training_rows(cohort),**training},validation=_metrics(points)))
        _decode_saved(report)
        return report

    if cache_dir is not None:
        return run(cache_dir)
    with tempfile.TemporaryDirectory(prefix='market-trader-cnn-') as private:
        return run(Path(private)/'cache')


def cnn_validation_evaluation(model_path: Path, dataset_path: Path, training_path: Path, *, cache_dir=None):
    """Score every eligible validation row with the stdlib JSON predictor."""
    from market_trader.image_training_data import prepare_image_cohort, build_input_cache
    saved = _read_report(model_path)
    predictor = _decode_saved(saved)
    cohort = prepare_image_cohort(dataset_path,training_path)
    bindings = _bindings(cohort)
    if any(not _exact(saved[k],v) for k,v in bindings.items()):
        raise ValueError('CNN source/cohort/model binding mismatch')
    if any(not _exact(saved['training'][k],v) for k,v in _training_rows(cohort).items()):
        raise ValueError('CNN training row binding mismatch')
    if saved['validation']['row_count'] != len(cohort.training.validation):
        raise ValueError('CNN validation row binding mismatch')

    def run(directory):
        cache = build_input_cache(cohort,Path(directory),accelerated=False)
        points = _points(predictor.model,cohort,cache)
        validation = dict(**_metrics(points),first_decision_at=points[0]['at'],last_decision_at=points[-1]['at'])
        evaluation = _seal(dict(format='market-trader.cnn-validation-evaluation.v1',
            model_kind='cnn',model_report_id=predictor.report_id,**bindings,validation=validation))
        return evaluation, points

    if cache_dir is not None:
        return run(cache_dir)
    with tempfile.TemporaryDirectory(prefix='market-trader-cnn-evaluate-') as private:
        return run(Path(private)/'cache')
