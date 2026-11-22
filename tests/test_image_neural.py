"""Synthetic-only fixed visual model tests; optional framework is training-only."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
from market_trader._dataset_codec import json_bytes


def seal(body):
    body = {k:v for k,v in body.items() if k != 'report_id'}
    return {**body, 'report_id':'sha256:'+sha256(json_bytes(body)).hexdigest()}


def model_body():
    return {'kind':'cnn','architecture':'tiny_cnn_v1','parameter_count':5409,
        'target_mean':0.01, 'target_scale':0.02,
        'tensors': {name:{'shape':shape, 'values':[0.0]*size} for name,shape,size in [
            ('conv.weight',[4,3,3,3],108),('conv.bias',[4],4),
            ('hidden.weight',[8,660],5280),('hidden.bias',[8],8),
            ('output.weight',[1,8],8),('output.bias',[1],1)]}}


def saved_body():
    from market_trader.cnn_settings import canonical_cnn_settings
    from market_trader.image_pixels_v1 import image_input_recipe
    from market_trader.training_data import OPEN_TARGET_NAME
    return seal({'format':'market-trader.cnn-validation.v1',
        'training_data_id':'sha256:'+'c'*64, 'input_index_id':'sha256:'+'c'*64,
        'source_training_data_id':'sha256:'+'d'*64,
        'dataset_id':'sha256:'+'a'*64, 'dataset_artifact_id':'sha256:'+'b'*64,
        'instrument':'deuidxeur', 'interval_microseconds':900000000,
        'target_name':OPEN_TARGET_NAME,
        'config':{'lookback_bars':8,'horizon_bars':2,
                  'validation_start':'2024-01-01T00:00:00+00:00','test_start':'2025-01-01T00:00:00+00:00'},
        'input_recipe':image_input_recipe(), 'model':model_body(),
        'training':{'row_count':2,'first_decision_at':'2021-01-01T02:00:00+00:00',
            'last_target_available_at':'2023-12-31T12:00:00+00:00',
            'fit_settings':{**canonical_cnn_settings(),'seed':0},
            'runtime_versions':{'python':'3.11','torch':'2.5','numpy':'2.1'},
            'epoch_history':[{'epoch':i,'training_mse':1/i,'validation_mse':1/i} for i in range(1,21)],
            'best_epoch':20,'stopping_reason':'max_epochs',
            'inference_parity':{'row_count':3,'max_abs_error':0.,'max_tolerance_ratio':0.,
                'absolute_tolerance':'0.00000000000001','relative_tolerance':'0.0000000000001'}},
        'validation':{'row_count':3,'cnn':{'mae':0.1,'rmse':0.2},'zero_return':{'mae':0.1,'rmse':0.2}}})


def test_bounded_safe_loader_immutable_stdlib_prediction(tmp_path):
    from dataclasses import FrozenInstanceError
    from market_trader.image_neural import load_cnn_model
    body = saved_body()
    path = tmp_path/'model.json'
    path.write_bytes(json_bytes(body))
    saved = load_cnn_model(path)
    assert saved.report_id == body['report_id']
    assert saved.model.predict_sums((0,)*2304) == 0.01
    with pytest.raises(FrozenInstanceError):
        saved.model.target_mean = 3
    body['model']['target_mean'] = 3
    path.write_bytes(json_bytes(body))
    with pytest.raises(ValueError,match='checksum'):
        load_cnn_model(path)
    path.write_bytes(b' '*(2*1024*1024+1))
    with pytest.raises(ValueError,match='bounded'):
        load_cnn_model(path)


@pytest.mark.parametrize('keys,value', [
    (('instrument',),'DEUIDXEUR'), (('target_name',),'wrong'),
    (('interval_microseconds',),900000000.0), (('config','lookback_bars'),9),
    (('training_data_id',),'sha256:'+'e'*64), (('source_training_data_id',),'bad'),
    (('input_recipe','max_sum'),16321), (('input_recipe','fitted_statistics'),0),
    (('training','row_count'),True), (('training','row_count'),50001),
    (('training','last_target_available_at'),'2024-01-01T00:00:00+00:00'),
    (('training','fit_settings','seed'),1), (('training','fit_settings','threads'),True),
    (('training','runtime_versions','torch'),''), (('training','best_epoch'),1),
    (('training','epoch_history',0,'training_mse'),-1),
    (('training','epoch_history',0,'epoch'),True),
    (('training','stopping_reason'),'early_stopping'),
    (('validation','row_count'),False), (('validation','cnn','mae'),-1),
    (('training','inference_parity','row_count'),2),
    (('training','inference_parity','max_abs_error'),True),
    (('training','inference_parity','max_tolerance_ratio'),1.01),
    (('training','inference_parity','relative_tolerance'),'1'),
])
def test_loader_rejects_resigned_schema_history_and_bindings(tmp_path, keys, value):
    from market_trader.image_neural import load_cnn_model
    body = saved_body()
    target = body
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = tmp_path/'bad.json'
    path.write_bytes(json_bytes(seal(body)))
    with pytest.raises(ValueError):
        load_cnn_model(path)


def test_checkpoint_actual_minimum_independent_of_patience(tmp_path):
    from market_trader.image_neural import load_cnn_model
    body = saved_body()
    training = body['training']
    training['epoch_history'] = [dict(epoch=i,training_mse=1.,validation_mse=1-i*1e-8) for i in range(1,7)]
    training.update(best_epoch=6,stopping_reason='early_stopping')
    path = tmp_path/'subdelta.json'
    path.write_bytes(json_bytes(seal(body)))
    load_cnn_model(path)
    training['epoch_history'].append(dict(epoch=7,training_mse=1.,validation_mse=0.5))
    training.update(best_epoch=7,stopping_reason='max_epochs')
    path.write_bytes(json_bytes(seal(body)))
    with pytest.raises(ValueError,match='history'):
        load_cnn_model(path)


requires_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='optional training runtime')


@requires_torch
def test_synthetic_memmap_fit_replay_scaler_parity_and_global_restore(tmp_path, monkeypatch):
    import random
    import numpy as np
    import torch
    from statistics import fmean, pstdev
    from types import SimpleNamespace
    from decimal import Decimal
    from market_trader.image_neural import _fit, _decode_model
    rows = tuple(SimpleNamespace(target_return=Decimal(str(v))) for v in [0.01,0.02,0.04,0.03,1.0,2.0,3.0])
    inputs = np.memmap(tmp_path/'sums.u16',dtype='<u2',mode='w+',shape=(7,3,24,32))
    for i in range(7):
        inputs[i] = (np.arange(2304).reshape(3,24,32)*(i+1)+i*331)%16321
    inputs.flush()
    cache = SimpleNamespace(path=tmp_path/'sums.u16',row_count=7,train_count=4)
    networks = []
    original = torch.nn.Sequential
    def capture(*args,**kwargs):
        result = original(*args,**kwargs)
        networks.append(result)
        return result
    monkeypatch.setattr(torch.nn,'Sequential',capture)
    rng, threads, dtype = torch.get_rng_state().clone(), torch.get_num_threads(), torch.get_default_dtype()
    det, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    first, history = _fit(cache,rows[:4],rows[4:])
    assert torch.equal(rng,torch.get_rng_state())
    assert (threads,dtype,det,warn) == (torch.get_num_threads(),torch.get_default_dtype(),
        torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    assert random.getstate() == py_rng
    assert np_rng[0] == np.random.get_state()[0] and np.array_equal(np_rng[1],np.random.get_state()[1])
    assert (first,history) == _fit(cache,rows[:4],rows[4:])
    assert sum(p.numel() for p in networks[0].parameters()) == 5409
    assert all(p.dtype == torch.float64 and p.device.type == 'cpu' for p in networks[0].parameters())
    assert first['target_mean'] == fmean([0.01,0.02,0.04,0.03])
    assert first['target_scale'] == pstdev([0.01,0.02,0.04,0.03])
    parity = history['inference_parity']
    assert parity['row_count'] == 3
    assert parity['max_abs_error'] < 1e-14
    assert parity['absolute_tolerance'] == '0.00000000000001'
    assert parity['relative_tolerance'] == '0.0000000000001'
    epochs = history['epoch_history']
    assert 1 <= len(epochs) <= 20
    assert history['best_epoch'] == min(epochs,key=lambda e:e['validation_mse'])['epoch']
    exported = _decode_model(first)
    with torch.no_grad():
        native = (networks[0](torch.tensor(inputs[4:].copy(),dtype=torch.float64)/64/255).flatten()
                  *first['target_scale']+first['target_mean']).tolist()
    actual = [exported.predict_sums(tuple(int(v) for v in row.flatten())) for row in inputs[4:]]
    maximum = max(abs(a-b) for a,b in zip(actual,native))
    print('synthetic Torch/stdlb max_abs_error:',maximum)
    assert maximum < 1e-14
    expected = fmean(((p-float(row.target_return))/first['target_scale'])**2 for p,row in zip(native,rows[4:]))
    assert expected == pytest.approx(epochs[history['best_epoch']-1]['validation_mse'],rel=1e-12)


@requires_torch
def test_synthetic_min_delta_checkpoint_bounded_batches_optimizer_and_failure_restore(tmp_path, monkeypatch):
    import torch
    import numpy as np
    from decimal import Decimal
    from types import SimpleNamespace
    from market_trader.image_neural import _fit
    targets = tuple(SimpleNamespace(target_return=Decimal(i%7)/100) for i in range(260))
    path = tmp_path/'bounded.u16'
    mapped = np.memmap(path,dtype='<u2',mode='w+',shape=(260,3,24,32))
    mapped[:] = 1234
    mapped.flush()
    cache = SimpleNamespace(path=path,row_count=260,train_count=257)
    tensor, mse, adam = torch.tensor, torch.nn.functional.mse_loss, torch.optim.AdamW
    batch_sizes, optimizer_settings, eval_count = [], [], []
    def bounded_tensor(values,*args,**kwargs):
        if isinstance(values,np.ndarray) and values.ndim == 4:
            batch_sizes.append(len(values))
            assert len(values) <= 128
        return tensor(values,*args,**kwargs)
    def scripted_loss(predicted,actual,*args,**kwargs):
        if kwargs.get('reduction') == 'sum' and len(actual) == 3:
            eval_count.append(1)
            return tensor(3*(1-len(eval_count)*1e-8),dtype=torch.float64)
        return mse(predicted,actual,*args,**kwargs)
    def optimizer(*args,**kwargs):
        optimizer_settings.append(kwargs)
        return adam(*args,**kwargs)
    monkeypatch.setattr(torch,'tensor',bounded_tensor)
    monkeypatch.setattr(torch.nn.functional,'mse_loss',scripted_loss)
    monkeypatch.setattr(torch.optim,'AdamW',optimizer)
    model, training = _fit(cache,targets[:257],targets[257:])
    assert training['best_epoch'] == 6
    assert training['stopping_reason'] == 'early_stopping'
    assert len(training['epoch_history']) == 6
    assert max(batch_sizes) == 128 and 1 in batch_sizes
    assert optimizer_settings == [dict(lr=0.001,weight_decay=0.0001,betas=(0.9,0.999),
        eps=1e-8,amsgrad=False,foreach=False,fused=False,maximize=False)]
    mapped[0,0,0,0] = 16321
    mapped.flush()
    rng, threads, dtype = torch.get_rng_state().clone(),torch.get_num_threads(),torch.get_default_dtype()
    deterministic, warn = torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(ValueError,match='block sums'):
        _fit(cache,targets[:257],targets[257:])
    assert torch.equal(rng,torch.get_rng_state())
    assert (threads,dtype,deterministic,warn) == (torch.get_num_threads(),torch.get_default_dtype(),
        torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())


@requires_torch
def test_synthetic_report_all_validation_and_window_reload_without_backend(tmp_path):
    import os
    from tests.test_image_dataset import sources
    from market_trader.image_training_data import prepare_image_cohort
    from market_trader.image_neural import cnn_report, load_cnn_model, cnn_validation_evaluation
    dataset, training = sources(tmp_path,size=14)
    report = cnn_report(dataset,training,cache_dir=tmp_path/'cache')
    assert report == seal(report)
    path = tmp_path/'cnn.json'
    path.write_bytes(json_bytes(report))
    saved = load_cnn_model(path)
    cohort = prepare_image_cohort(dataset,training)
    assert report['training_data_id'] == cohort.index_document()['input_index_id']
    assert report['source_training_data_id'] == cohort.training_data_id
    evaluation, points = cnn_validation_evaluation(path,dataset,training,cache_dir=tmp_path/'cache')
    assert len(points) == len(cohort.training.validation) == 4
    assert evaluation['validation']['row_count'] == 4
    assert evaluation['validation']['cnn'] == report['validation']['cnn']
    for point,row in zip(points,cohort.training.validation,strict=True):
        assert point['at'] == row.decision_at.isoformat()
        assert point['actual'] == float(row.target_return)
        assert point['predicted'] == saved.predict_window(cohort.window(row),decision_at=row.decision_at)
    code = '''
import builtins,json,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy'):
        raise AssertionError('optional framework imported by inference')
    return original(name,*args,**kwargs)
builtins.__import__=guard
from market_trader.image_neural import load_cnn_model,cnn_validation_evaluation
model=load_cnn_model(sys.argv[1])
result,points=cnn_validation_evaluation(*sys.argv[1:4],cache_dir=__import__('pathlib').Path(sys.argv[4]))
print(json.dumps(points))
'''
    result = subprocess.run([sys.executable,'-c',code,str(path),str(dataset),str(training),str(tmp_path/'cache')],
        capture_output=True,text=True,check=True,env=os.environ.copy())
    assert json.loads(result.stdout) == points
    bad = json.loads(json_bytes(report))
    bad['source_training_data_id'] = 'sha256:'+'f'*64
    path.write_bytes(json_bytes(seal(bad)))
    with pytest.raises(ValueError,match='binding'):
        cnn_validation_evaluation(path,dataset,training,cache_dir=tmp_path/'cache')


def test_core_allval_evaluation_and_window_are_framework_free(tmp_path, monkeypatch):
    import builtins
    from dataclasses import replace
    from datetime import UTC, datetime
    from tests.test_image_dataset import sources
    from market_trader.image_training_data import prepare_image_cohort
    from market_trader.image_neural import (load_cnn_model, cnn_validation_evaluation,
                                           _bindings, _training_rows)
    dataset, training = sources(tmp_path,size=13)
    cohort = prepare_image_cohort(dataset,training)
    body = saved_body()
    body.update(_bindings(cohort))
    body['training'].update(_training_rows(cohort))
    path = tmp_path/'model.json'
    path.write_bytes(json_bytes(seal(body)))
    original = builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in ('torch','numpy'):
            raise AssertionError('optional backend used in core inference')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    saved = load_cnn_model(path)
    evaluation, points = cnn_validation_evaluation(path,dataset,training)
    assert len(points) == 3 and evaluation['validation']['row_count'] == 3
    assert evaluation['model_kind'] == 'cnn'
    assert [point['predicted'] for point in points] == [0.01]*3
    assert evaluation['validation']['first_decision_at'] == points[0]['at']
    assert evaluation['validation']['last_decision_at'] == points[-1]['at']
    row = cohort.training.validation[0]
    window = cohort.window(row)
    assert saved.predict_window(window,decision_at=row.decision_at) == 0.01
    for invalid in ((), [replace(c,instrument='DEUIDXEUR') for c in window]):
        with pytest.raises(ValueError):
            saved.predict_window(invalid,decision_at=row.decision_at)
    with pytest.raises(ValueError,match='2021–2024'):
        saved.predict_window(window,decision_at=datetime(2025,1,1,tzinfo=UTC))


@pytest.mark.parametrize('seed',[True,1,-1,0.0,'0'])
def test_seed_rejected_before_backend_and_io(seed,tmp_path):
    from market_trader.image_neural import cnn_report
    with pytest.raises(ValueError,match='seed'):
        cnn_report(tmp_path/'missing',tmp_path/'missing',seed=seed)


def test_inference_rejects_hidden_overflow_before_relu_can_hide_it():
    from market_trader.image_neural import _decode_model
    body = model_body()
    body['tensors']['conv.bias']['values'] = [1.]*4
    body['tensors']['hidden.weight']['values'][0] = -1e308
    body['tensors']['hidden.bias']['values'][0] = -1e308
    model = _decode_model(body)
    with pytest.raises(ValueError,match='finite range'):
        model.predict_sums((0,)*2304)


def test_stdlib_predict_sums_zero_weights_and_input_validation():
    from market_trader.image_neural import _decode_model
    model = _decode_model(model_body())
    assert model.predict_sums((16320,)*2304) == 0.01
    assert isinstance(model.tensors, tuple)
    for bad in [(0,)*2303, (False,)*2304, (-1,)*2304, (16321,)*2304, (1.0,)*2304]:
        with pytest.raises(ValueError):
            model.predict_sums(bad)


@pytest.mark.parametrize('key,value', [('kind','mlp'),('parameter_count',417),
    ('parameter_count',5409.0),('target_mean',True),('target_scale',0),('target_scale',1e309)])
def test_model_schema_rejects_invalid_values(key,value):
    from market_trader.image_neural import _decode_model
    body = model_body()
    body[key] = value
    with pytest.raises(ValueError):
        _decode_model(body)
