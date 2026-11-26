"""AP8 synthetic-only seeded CNN and frozen AP7 compatibility evidence."""
from dataclasses import FrozenInstanceError
import importlib.util
import pytest
from market_trader._dataset_codec import json_bytes
from tests.test_image_neural import saved_body, seal


def seeded_body(seed=1):
    body = saved_body()
    body['format'] = 'market-trader.cnn-validation.v2'
    body['training']['fit_settings']['seed'] = seed
    return seal(body)


@pytest.mark.parametrize('seed', [1, 2])
def test_seeded_safe_loader_immutable_and_ap7_stays_strict(tmp_path, seed):
    from market_trader.image_neural_seeded import load_seeded_cnn_model
    from market_trader.image_neural import load_cnn_model, _fit, cnn_report
    path = tmp_path/'model.json'
    body = seeded_body(seed)
    path.write_bytes(json_bytes(body))
    saved = load_seeded_cnn_model(path)
    assert saved.report_id == body['report_id']
    assert saved.model.predict_sums((0,)*2304) == 0.01
    with pytest.raises(FrozenInstanceError):
        saved.model.target_mean = 1
    with pytest.raises(ValueError, match='format'):
        load_cnn_model(path)
    body['format'] = 'market-trader.cnn-validation.v1'
    path.write_bytes(json_bytes(seal(body)))
    with pytest.raises(ValueError, match='seed'):
        load_cnn_model(path)
    with pytest.raises(ValueError, match='format'):
        load_seeded_cnn_model(path)
    with pytest.raises(ValueError, match='seed'):
        _fit(None, (), (), seed=seed)
    with pytest.raises(ValueError, match='seed'):
        cnn_report(None, None, seed=seed)


@pytest.mark.parametrize('keys,value', [
    (('training','fit_settings','seed'), 0),
    (('training','fit_settings','seed'), 3),
    (('training','fit_settings','seed'), True),
    (('training','fit_settings','seed'), 1.0),
    (('training','fit_settings','threads'), True),
    (('training','best_epoch'), 1),
    (('training','epoch_history',0,'training_mse'), -1),
    (('training','epoch_history',0,'epoch'), True),
    (('training','stopping_reason'), 'early_stopping'),
    (('training','inference_parity','row_count'), 2),
    (('training','inference_parity','max_tolerance_ratio'), 1.01),
    (('training','inference_parity','max_abs_error'), True),
    (('training','inference_parity','relative_tolerance'), '1'),
    (('training','runtime_versions','torch'), ''),
    (('training','last_target_available_at'), '2024-01-01T00:00:00+00:00'),
    (('source_training_data_id',), 'bad'),
    (('training_data_id',), 'sha256:'+'e'*64),
    (('input_recipe','max_sum'), 16321),
    (('input_recipe','fitted_statistics'), 0),
    (('model','parameter_count'), 5409.0),
    (('model','target_scale'), 0),
    (('model','target_mean'), True),
    (('model','tensors','conv.weight','shape'), [4,3,3,4]),
    (('model','tensors','output.bias','values'), [True]),
    (('validation','row_count'), False),
    (('validation','cnn','mae'), -1),
    (('config','lookback_bars'), 9),
    (('interval_microseconds',), 900000000.0),
])
def test_seeded_resigned_invalid_schema_history_recipe_and_source(tmp_path, keys, value):
    from market_trader.image_neural_seeded import load_seeded_cnn_model
    body = seeded_body()
    target = body
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = tmp_path/'bad.json'
    path.write_bytes(json_bytes(seal(body)))
    with pytest.raises(ValueError):
        load_seeded_cnn_model(path)


def test_seeded_bounded_json_checksum_and_special_files(tmp_path):
    import os
    from market_trader.image_neural_seeded import load_seeded_cnn_model
    path = tmp_path/'model.json'
    body = seeded_body()
    body['model']['target_mean'] = 3
    path.write_bytes(json_bytes(body))
    with pytest.raises(ValueError, match='checksum'):
        load_seeded_cnn_model(path)
    path.write_bytes(b' '*(2*1024*1024+1))
    with pytest.raises(ValueError, match='bounded'):
        load_seeded_cnn_model(path)
    fifo = tmp_path/'fifo'
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match='bounded'):
        load_seeded_cnn_model(fifo)
    link = tmp_path/'link'
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_seeded_cnn_model(link)
    for raw in (b'{"format":1,"format":2}', b'{"x":NaN}', b'[]'):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            load_seeded_cnn_model(path)


requires_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None,
                                  reason='optional training runtime')


def runtime_state():
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    return (random.getstate(), (state[0], state[1].tobytes(), *state[2:]),
            bytes(torch.get_rng_state().tolist()), torch.get_num_threads(),
            torch.get_default_dtype(), torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())


@requires_torch
def test_actual_seed1_seed2_training_replay_reload_allval_native_parity(tmp_path, monkeypatch):
    from statistics import fmean, pstdev
    import numpy as np
    import torch
    from tests.test_image_dataset import sources
    from market_trader.cnn_settings import canonical_cnn_settings
    from market_trader.image_training_data import prepare_image_cohort, build_input_cache
    from market_trader.image_neural_seeded import seeded_cnn_report, load_seeded_cnn_model
    dataset, training = sources(tmp_path, size=14)
    cohort = prepare_image_cohort(dataset, training)
    original = torch.nn.Sequential
    networks = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        networks.append(result)
        return result
    monkeypatch.setattr(torch.nn, 'Sequential', capture)
    state = runtime_state()
    reports = []
    for seed in (1, 2):
        report = seeded_cnn_report(dataset, training, seed=seed, cache_dir=tmp_path/'cache')
        assert runtime_state() == state
        assert report == seeded_cnn_report(dataset, training, seed=seed, cache_dir=tmp_path/'cache')
        assert runtime_state() == state
        assert report['format'] == 'market-trader.cnn-validation.v2'
        assert report == seal(report)
        fit = report['training']
        assert fit['fit_settings'] == {**canonical_cnn_settings(), 'seed':seed}
        assert fit['inference_parity']['row_count'] == len(cohort.training.validation)
        assert fit['inference_parity']['max_tolerance_ratio'] <= 1
        assert fit['best_epoch'] == min(fit['epoch_history'], key=lambda e:e['validation_mse'])['epoch']
        targets = [float(row.target_return) for row in cohort.training.train]
        assert report['model']['target_mean'] == fmean(targets)
        assert report['model']['target_scale'] == (pstdev(targets) or 1.)
        path = tmp_path/f'seed{seed}.json'
        path.write_bytes(json_bytes(report))
        saved = load_seeded_cnn_model(path)
        from market_trader.image_neural_seeded import seeded_cnn_validation_evaluation
        evaluation, points = seeded_cnn_validation_evaluation(path, dataset, training, cache_dir=tmp_path/'cache')
        assert evaluation['validation']['cnn'] == report['validation']['cnn']
        assert evaluation['validation']['zero_return'] == report['validation']['zero_return']
        assert len(points) == report['validation']['row_count']
        import json
        import os
        import subprocess
        import sys
        code = '''
import builtins,json,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy'):
        raise AssertionError('optional framework imported in fresh inference process')
    return original(name,*args,**kwargs)
builtins.__import__=guard
from market_trader.image_neural_seeded import load_seeded_cnn_model,seeded_cnn_validation_evaluation
saved=load_seeded_cnn_model(sys.argv[1])
evaluation,points=seeded_cnn_validation_evaluation(*sys.argv[1:4])
print(json.dumps(points))
'''
        child = subprocess.run([sys.executable,'-c',code,str(path),str(dataset),str(training)],
                               capture_output=True,text=True,check=True,env=os.environ.copy())
        assert json.loads(child.stdout) == points
        network = networks[-2]
        assert sum(p.numel() for p in network.parameters()) == 5409
        assert all(p.dtype == torch.float64 and p.device.type == 'cpu' for p in network.parameters())
        cache = build_input_cache(cohort, tmp_path/'cache', accelerated=False)
        values = [cache.row(cache.train_count+i) for i in range(len(cohort.training.validation))]
        with torch.no_grad():
            x = torch.tensor(np.array(values).reshape(-1,3,24,32), dtype=torch.float64)/64/255
            native = (network(x).flatten()*saved.model.target_scale+saved.model.target_mean).tolist()
        actual = [saved.model.predict_sums(row) for row in values]
        for a, b in zip(actual, native, strict=True):
            assert abs(a-b) <= 1e-14+1e-13*abs(b)
        maximum = max(abs(a-b) for a,b in zip(actual,native,strict=True))
        assert maximum == fit['inference_parity']['max_abs_error']
        print('seed',seed,'all-validation native/stdlib max_abs_error',maximum)
        for row, prediction in zip(cohort.training.validation, actual, strict=True):
            assert saved.predict_window(cohort.window(row), decision_at=row.decision_at) == prediction
        standardized_mse = fmean(((p-float(row.target_return))/saved.model.target_scale)**2
                                for p,row in zip(native,cohort.training.validation,strict=True))
        assert standardized_mse == pytest.approx(fit['epoch_history'][fit['best_epoch']-1]['validation_mse'], rel=1e-12)
        reports.append(report)
    assert reports[0]['model']['tensors'] != reports[1]['model']['tensors']


@pytest.mark.parametrize('seed', [True, False, 0, -1, 3, 1.0, '1', None])
def test_seeded_report_rejects_invalid_seed_before_io_or_optional_import(tmp_path, seed):
    from market_trader.image_neural_seeded import seeded_cnn_report
    with pytest.raises(ValueError, match='seed'):
        seeded_cnn_report(tmp_path/'missing', tmp_path/'missing', seed=seed)


@pytest.mark.parametrize('seed', [1, 2])
def test_seeded_core_evaluation_bindings_points_and_optional_isolation(tmp_path, monkeypatch, seed):
    import builtins
    from tests.test_image_dataset import sources
    from market_trader.image_neural import _bindings, _training_rows
    from market_trader.image_training_data import prepare_image_cohort
    from market_trader.image_neural_seeded import (
        load_seeded_cnn_model, seeded_cnn_validation_evaluation,
    )
    dataset, training = sources(tmp_path, size=13)
    cohort = prepare_image_cohort(dataset, training)
    body = seeded_body(seed)
    body.update(_bindings(cohort))
    body['training'].update(_training_rows(cohort))
    body = seal(body)
    path = tmp_path/'model.json'
    path.write_bytes(json_bytes(body))
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'torch'):
            raise AssertionError('optional framework imported during inference')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    saved = load_seeded_cnn_model(path)
    evaluation, points = seeded_cnn_validation_evaluation(path, dataset, training)
    assert evaluation['format'] == 'market-trader.cnn-seeded-validation-evaluation.v1'
    assert evaluation['model_kind'] == 'cnn'
    assert evaluation['model_report_id'] == body['report_id']
    assert evaluation == seal(evaluation)
    assert evaluation['validation']['row_count'] == len(points) == 3
    assert evaluation['validation']['first_decision_at'] == points[0]['at']
    assert evaluation['validation']['last_decision_at'] == points[-1]['at']
    for row, point in zip(cohort.training.validation, points, strict=True):
        assert set(point) == {'at', 'predicted', 'actual'}
        assert type(point['predicted']) is float and type(point['actual']) is float
        assert point == {'at':row.decision_at.isoformat(), 'predicted':0.01,
                         'actual':float(row.target_return)}
        assert saved.predict_window(cohort.window(row), decision_at=row.decision_at) == point['predicted']
    for field in ('source_training_data_id', 'dataset_id', 'dataset_artifact_id'):
        bad = {**body, field:'sha256:'+'f'*64}
        path.write_bytes(json_bytes(seal(bad)))
        with pytest.raises(ValueError, match='binding'):
            seeded_cnn_validation_evaluation(path, dataset, training)
    body['training']['row_count'] += 1
    path.write_bytes(json_bytes(seal(body)))
    with pytest.raises(ValueError, match='binding'):
        seeded_cnn_validation_evaluation(path, dataset, training)
    body['training']['row_count'] -= 1
    body['validation']['row_count'] += 1
    body['training']['inference_parity']['row_count'] += 1
    path.write_bytes(json_bytes(seal(body)))
    with pytest.raises(ValueError, match='binding'):
        seeded_cnn_validation_evaluation(path, dataset, training)


@requires_torch
@pytest.mark.parametrize('seed', [1, 2])
def test_seeded_fit_restores_state_even_when_memmap_close_fails(tmp_path, monkeypatch, seed):
    import numpy as np
    from types import SimpleNamespace
    from market_trader.image_neural import _fit_for_seed
    path = tmp_path/'cache.u16'
    data = np.memmap(path, dtype='<u2', mode='w+', shape=(3,3,24,32))
    data[:] = 1234
    data.flush()
    data._mmap.close()
    original = np.memmap
    class FailingClose:
        def __init__(self, mapped):
            self.mapped = mapped
            self._mmap = self
        def __getitem__(self, index):
            return self.mapped[index]
        def close(self):
            self.mapped._mmap.close()
            raise OSError('synthetic mmap close failure')
    def mapped(*args, **kwargs):
        return FailingClose(original(*args, **kwargs))
    cache = SimpleNamespace(path=path, row_count=3, train_count=2)
    rows = [SimpleNamespace(target_return=value) for value in (0.01,0.02,0.5)]
    state = runtime_state()
    monkeypatch.setattr(np, 'memmap', mapped)
    try:
        with pytest.raises(OSError, match='close failure'):
            _fit_for_seed(cache, rows[:2], rows[2:], seed, expected_seed=seed)
        assert runtime_state() == state
    finally:
        # The RED run must not contaminate later tests in this process.
        import random
        import torch
        random.setstate(state[0])
        npstate = state[1]
        np.random.set_state((npstate[0],np.frombuffer(npstate[1],dtype='uint32').copy(),*npstate[2:]))
        torch.set_rng_state(torch.tensor(list(state[2]),dtype=torch.uint8))
        torch.set_num_threads(state[3])
        torch.set_default_dtype(state[4])
        torch.use_deterministic_algorithms(state[5],warn_only=state[6])


@requires_torch
@pytest.mark.parametrize('seed', [1, 2])
def test_seeded_shared_kernel_bounded_batches_allval_and_failure_paths(tmp_path, monkeypatch, seed):
    """Regression coverage of the inherited AP7 kernel under each new seed."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    from market_trader import image_neural
    path = tmp_path/'bounded.u16'
    data = np.memmap(path, dtype='<u2', mode='w+', shape=(258,3,24,32))
    data[:] = 1234
    data.flush()
    cache = SimpleNamespace(path=path, row_count=258, train_count=129)
    rows = [SimpleNamespace(target_return=(i%7)/100) for i in range(129)]
    validation = [SimpleNamespace(target_return=1+i/100) for i in range(129)]
    original_tensor, original_adam = torch.tensor, torch.optim.AdamW
    sizes, optimizer_settings = [], []
    def bounded(values, *args, **kwargs):
        if isinstance(values,np.ndarray) and values.ndim == 4:
            assert values.dtype == np.dtype('<u2')
            sizes.append(len(values))
            assert len(values) <= 128
        if isinstance(values,list) and values and isinstance(values[0],list):
            assert len(values) <= 128
        return original_tensor(values,*args,**kwargs)
    def optimizer(*args, **kwargs):
        optimizer_settings.append(kwargs)
        return original_adam(*args, **kwargs)
    monkeypatch.setattr(torch,'tensor',bounded)
    monkeypatch.setattr(torch.optim,'AdamW',optimizer)
    state = runtime_state()
    predictions = []
    model, training = image_neural._fit_for_seed(cache,rows,validation,seed,
                                               expected_seed=seed,validation_predictions=predictions)
    assert runtime_state() == state
    assert max(sizes) == 128 and 1 in sizes
    assert len(predictions) == training['inference_parity']['row_count'] == 129
    assert training['inference_parity']['max_tolerance_ratio'] <= 1
    assert model['target_mean'] < 0.1
    assert optimizer_settings == [dict(lr=0.001,weight_decay=0.0001,betas=(0.9,0.999),
                                     eps=1e-8,amsgrad=False,foreach=False,fused=False,maximize=False)]
    data._mmap.close()
    # Genuine failing I/O, invalid targets/cache, backend initialization and
    # export/parity paths all share the same process-state cleanup guarantee.
    for stage in ('cache_open', 'targets', 'cache_values', 'optimizer', 'export', 'parity'):
        with monkeypatch.context() as scoped:
            badcache = SimpleNamespace(path=path,row_count=258,train_count=129)
            badrows = rows
            if stage == 'cache_open':
                badcache.path = tmp_path/'missing'
            elif stage == 'targets':
                badrows = [SimpleNamespace(target_return=float('inf')), *rows[1:]]
            elif stage == 'cache_values':
                mapped = np.memmap(path,dtype='<u2',mode='r+',shape=(258,3,24,32))
                mapped[0,0,0,0] = 16321
                mapped.flush()
                mapped._mmap.close()
            elif stage == 'optimizer':
                def fail_optimizer(*args, **kwargs):
                    raise RuntimeError('synthetic optimizer failure')
                scoped.setattr(torch.optim,'AdamW',fail_optimizer)
            elif stage == 'export':
                def fail_export(*args, **kwargs):
                    raise ValueError('synthetic export failure')
                scoped.setattr(image_neural,'_decode_model',fail_export)
            elif stage == 'parity':
                class BadParity:
                    def predict_sums(self, sums):
                        return 1e9
                scoped.setattr(image_neural,'_decode_model',lambda model:BadParity())
            before = runtime_state()
            with pytest.raises((ValueError, OSError, RuntimeError)):
                image_neural._fit_for_seed(badcache,badrows,validation,seed,expected_seed=seed)
            assert runtime_state() == before
            if stage == 'cache_values':
                mapped = np.memmap(path,dtype='<u2',mode='r+',shape=(258,3,24,32))
                mapped[0,0,0,0] = 1234
                mapped.flush()
                mapped._mmap.close()
