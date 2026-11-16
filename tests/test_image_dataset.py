"""Synthetic AP6 exports: no real data, training or network."""
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import pytest

from market_trader._dataset_codec import encode, json_bytes
from market_trader.artifacts import DatasetProvenance
from market_trader.calendars import SessionCalendar, TradingSession
from market_trader.domain import Candle
from market_trader.normalization import NormalizationConfig, normalize_candles
from market_trader.storage import save_dataset
from market_trader.training_data import (TrainingConfig, build_open_training_data,
    FEATURE_NAMES, OPEN_TARGET_NAME, OPEN_TRAINING_DATA_FORMAT)


def api():
    from market_trader import image_dataset
    return image_dataset


def test_future_price_resource_bounds_are_enforced_before_regeneration():
    from market_trader.image_dataset import _candles
    rows=[{'open_time':'2024-01-02T08:00:00+00:00','close_time':'2024-01-02T08:15:00+00:00',
           'available_at':'2024-01-02T08:15:00+00:00','open':'1e1001','high':'1e1001','low':'1e1001','close':'1e1001'}]
    for index in (1,2):
        start=datetime(2024,1,2,8,tzinfo=UTC)+index*timedelta(minutes=15)
        rows.append({**rows[0],'open_time':start.isoformat(),'close_time':(start+timedelta(minutes=15)).isoformat(),
                     'available_at':(start+timedelta(minutes=15)).isoformat()})
    with pytest.raises(ValueError,match='resource'):
        _candles(rows)


@pytest.mark.parametrize("count", [0, 65, True, 1.0, -1])
def test_budget_rejected_before_any_io(tmp_path, count):
    with pytest.raises(ValueError, match="per_split"):
        api().build_image_dataset(tmp_path/'missing', tmp_path/'missing', tmp_path/'out',
                                  cache=tmp_path/'cache', per_split=count)
    assert list(tmp_path.iterdir()) == []


def sealed(path, body, key):
    body = {k:v for k,v in body.items() if k != key}
    path.write_bytes(json_bytes({**body, key:'sha256:'+sha256(json_bytes(body)).hexdigest()}))


def sources(tmp_path, *, year=2023, size=20, days=1):
    starts = [base+timedelta(days=day) for base in
        (datetime(year, 6, 5, 7, tzinfo=UTC),datetime(2024, 6, 3, 7, tzinfo=UTC))
        for day in range(days)]
    step = timedelta(minutes=15)
    calendar = SessionCalendar('synthetic research', starts[0], starts[-1]+timedelta(days=1),
        tuple(TradingSession(s, s+timedelta(hours=8, minutes=30)) for s in starts))
    bars = tuple(Candle('deuidxeur', step, s+i*step, s+(i+1)*step, s+(i+1)*step,
                       Decimal(100+i), Decimal(104+i), Decimal(98+i), Decimal(101+i), Decimal(1))
                 for s in starts for i in range(size))
    # Complete sessions avoid normalization's active-gap audit.
    calendar = SessionCalendar(calendar.name, calendar.coverage_start, calendar.coverage_end,
        tuple(TradingSession(s, s+size*step) for s in starts))
    dataset = normalize_candles(bars, config=NormalizationConfig(calendar)).datasets[0]
    dataset_path = tmp_path/'development.json'
    manifest = save_dataset(dataset_path, dataset, provenance=DatasetProvenance('synthetic', ()))
    cfg = TrainingConfig(8, 2, datetime(2024,1,1,tzinfo=UTC), datetime(2025,1,1,tzinfo=UTC))
    rows = build_open_training_data(dataset, cfg, include_test=False)
    training_path = tmp_path/'training.json'
    body = {'format':OPEN_TRAINING_DATA_FORMAT, 'dataset_id':dataset.dataset_id,
            'dataset_artifact_id':manifest.artifact_id, 'instrument':dataset.instrument,
            'interval_microseconds':900000000, 'config':encode(cfg),
            'feature_names':list(FEATURE_NAMES), 'target_name':OPEN_TARGET_NAME,
            'splits':{name:encode(getattr(rows,name)) for name in ('train','validation','test')}}
    sealed(training_path, body, 'training_data_id')
    return dataset_path, training_path


def bundle_bytes(path):
    return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}


def test_export_is_standalone_deterministic_and_warm_cache(tmp_path):
    dataset, training = sources(tmp_path)
    cold = api().build_image_dataset(dataset, training, tmp_path/'one', cache=tmp_path/'cache', per_split=2)
    warm = api().build_image_dataset(dataset, training, tmp_path/'two', cache=tmp_path/'cache', per_split=2)
    assert cold['sample_count'] == warm['sample_count'] == 4
    assert cold['manifest']['image_dataset_id'].startswith('sha256:')
    assert 'dataset_id' not in cold['manifest']
    assert cold['cache_hits'] == warm['cache_misses'] == 0
    assert cold['cache_misses'] == warm['cache_hits'] == 4
    assert bundle_bytes(tmp_path/'one') == bundle_bytes(tmp_path/'two')
    sample_doc = json.loads((tmp_path/'one'/'samples.json').read_bytes())
    samples = sample_doc['samples']
    assert len(samples) == 4
    assert all(set(row) == {'sample_id','split','decision_at','image_key','image_path','image_sha256','window'} for row in samples)
    assert all('volume' not in json.dumps(row) and 'target' not in json.dumps(row) for row in samples)
    labels = json.loads((tmp_path/'one'/'labels.json').read_bytes())['labels']
    assert all(len(row['future']) == 3 for row in labels)
    training_rows = json.loads(training.read_bytes())['splits']
    assert [r['decision_at'] for r in samples] == [training_rows[s][i]['decision_at']
        for s in ('train','validation') for i in (0,-1)]
    dataset.unlink()
    training.unlink()
    assert api().verify_image_dataset(tmp_path/'one') == cold['manifest']


def resign_bundle(output):
    manifest = json.loads((output/'manifest.json').read_bytes())
    manifest['artifacts'] = {name:'sha256:'+sha256((output/name).read_bytes()).hexdigest()
                             for name in manifest['artifacts']}
    sealed(output/'manifest.json', manifest, 'image_dataset_id')


@pytest.mark.parametrize('fault', ['extra','missing','corrupt','symlink','hardlink','directory','traversal'])
def test_bundle_inventory_and_regular_file_boundaries(tmp_path, fault):
    import os
    dataset, training = sources(tmp_path)
    output = tmp_path/'out'
    api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    image = next((output/'images').iterdir())
    if fault == 'extra':
        (output/'extra').write_bytes(b'not inventoried')
    elif fault == 'missing':
        image.unlink()
    elif fault == 'corrupt':
        image.write_bytes(b'bad')
    elif fault in ('symlink','hardlink','directory'):
        saved = tmp_path/'saved.png'
        image.rename(saved)
        if fault == 'symlink': image.symlink_to(saved)
        elif fault == 'hardlink': os.link(saved,image)
        else: image.mkdir()
    else:
        manifest = json.loads((output/'manifest.json').read_bytes())
        outside = tmp_path/'outside'
        outside.write_bytes(b'private')
        manifest['artifacts']['../outside'] = 'sha256:'+sha256(outside.read_bytes()).hexdigest()
        sealed(output/'manifest.json',manifest,'image_dataset_id')
    with pytest.raises((ValueError,OSError)):
        api().verify_image_dataset(output)


@pytest.mark.parametrize('fault', ['sample_schema','sample_format','label_schema','label_format',
    'manifest_schema','split','sample_id','future_time','future_ohlc','future_availability',
    'future_2025','future_count','wrong_return','wrong_image','unused_image','label_order','recipe'])
def test_resigned_semantic_contradictions_are_rejected(tmp_path, fault):
    dataset, training = sources(tmp_path)
    output = tmp_path/'out'
    api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    samples = json.loads((output/'samples.json').read_bytes())
    labels = json.loads((output/'labels.json').read_bytes())
    manifest = json.loads((output/'manifest.json').read_bytes())
    sample, label = samples['samples'][0], labels['labels'][0]
    if fault == 'sample_schema': sample['target_return'] = label['target_return']
    elif fault == 'sample_format': samples['format'] = 'wrong'
    elif fault == 'label_schema': label['extra'] = 1
    elif fault == 'label_format': labels['format'] = 'wrong'
    elif fault == 'manifest_schema': manifest['extra'] = 1
    elif fault == 'split': sample['split'] = 'validation'
    elif fault == 'sample_id':
        old = sample['sample_id']
        sample['sample_id'] = label['sample_id'] = 'sha256:'+'a'*64
        manifest['sessions'][sample['sample_id']] = manifest['sessions'].pop(old)
    elif fault == 'future_time': label['future'][1]['open_time'] = label['future'][0]['open_time']
    elif fault == 'future_ohlc': label['future'][1]['low'] = '-1'
    elif fault == 'future_availability': label['future'][0]['available_at'] = label['future'][0]['open_time']
    elif fault == 'future_2025':
        label['future'][0]['available_at'] = label['target_available_at'] = '2025-01-01T00:00:00.000000+00:00'
    elif fault == 'future_count': label['future'].append(label['future'][-1])
    elif fault == 'wrong_return': label['target_return'] = '0.12345'
    elif fault == 'wrong_image':
        path = output/sample['image_path']
        path.write_bytes(next((output/'images').glob('*')).read_bytes()[::-1])
        sample['image_sha256'] = 'sha256:'+sha256(path.read_bytes()).hexdigest()
    elif fault == 'unused_image':
        name = 'images/'+'b'*64+'.png'
        (output/name).write_bytes((output/sample['image_path']).read_bytes())
        manifest['artifacts'][name] = 'sha256:'+sha256((output/name).read_bytes()).hexdigest()
    elif fault == 'label_order': labels['labels'].reverse()
    else:
        recipe = json.loads((output/'recipe.json').read_bytes())
        recipe['width'] = 17
        (output/'recipe.json').write_bytes(json_bytes(recipe))
    (output/'samples.json').write_bytes(json_bytes(samples))
    (output/'labels.json').write_bytes(json_bytes(labels))
    sealed(output/'manifest.json',manifest,'image_dataset_id')
    resign_bundle(output)
    with pytest.raises((ValueError,OSError)):
        api().verify_image_dataset(output)


@pytest.mark.parametrize('fault', ['symlink','hardlink','directory','corrupt','root_symlink','ancestor_symlink','public'])
def test_cache_rejects_unsafe_hits_without_repair_or_partial_output(tmp_path, fault):
    import os
    dataset, training = sources(tmp_path)
    cache = tmp_path/'cache'
    api().build_image_dataset(dataset,training,tmp_path/'first',cache=cache,per_split=1)
    entry = next(cache.iterdir())
    if fault in ('symlink','hardlink','directory'):
        saved = tmp_path/'saved'
        entry.rename(saved)
        if fault == 'symlink': entry.symlink_to(saved)
        elif fault == 'hardlink': os.link(saved,entry)
        else: entry.mkdir()
    elif fault == 'corrupt': entry.write_bytes(b'corrupt')
    elif fault == 'root_symlink':
        real = tmp_path/'real'
        cache.rename(real)
        cache.symlink_to(real,target_is_directory=True)
    elif fault == 'ancestor_symlink':
        alias = tmp_path/'alias'
        alias.symlink_to(tmp_path,target_is_directory=True)
        cache = alias/'cache'
    else: cache.chmod(0o755)
    before = (tmp_path/'saved').read_bytes() if (tmp_path/'saved').exists() else None
    with pytest.raises((ValueError,OSError)):
        api().build_image_dataset(dataset,training,tmp_path/'second',cache=cache,per_split=1)
    assert not (tmp_path/'second').exists()
    assert not list(tmp_path.glob('.image-dataset-*'))
    if before is not None: assert (tmp_path/'saved').read_bytes() == before
    if fault == 'corrupt': assert entry.read_bytes() == b'corrupt'


@pytest.mark.parametrize('where', ['cache','publish'])
def test_filesystem_failure_never_publishes_partial_bundle(tmp_path, monkeypatch, where):
    from market_trader import storage
    dataset, training = sources(tmp_path)
    before = dataset.read_bytes(),training.read_bytes()
    def fail(*args,**kwargs): raise OSError('injected filesystem failure')
    if where == 'cache': monkeypatch.setattr(storage,'publish_new_file',fail)
    else: monkeypatch.setattr(api(),'_publish_directory',fail)
    with pytest.raises(OSError,match='injected'):
        api().build_image_dataset(dataset,training,tmp_path/'out',cache=tmp_path/'cache',per_split=1)
    assert not (tmp_path/'out').exists()
    assert not list(tmp_path.glob('.image-dataset-*'))
    assert before == (dataset.read_bytes(),training.read_bytes())


@pytest.mark.parametrize('fault', ['wrong_dataset','wrong_artifact','features','target','deleted_row',
    'boundary','test_rows','horizon','lookback','target_name','checksum'])
def test_build_reconciles_every_sealed_row_and_source_identity(tmp_path, fault):
    dataset, training = sources(tmp_path)
    body = json.loads(training.read_bytes())
    row = body['splits']['validation'][3]  # not selected by per_split=1
    if fault == 'wrong_dataset': body['dataset_id'] = 'sha256:'+'a'*64
    elif fault == 'wrong_artifact': body['dataset_artifact_id'] = 'sha256:'+'b'*64
    elif fault == 'features': row['features'][0] = '0.987'
    elif fault == 'target': row['target_return'] = '0.987'
    elif fault == 'deleted_row': body['splits']['validation'].pop(3)
    elif fault == 'boundary': body['config']['test_start'] = '2026-01-01T00:00:00.000000+00:00'
    elif fault == 'test_rows': body['splits']['test'] = [row]
    elif fault == 'horizon': body['config']['horizon_bars'] = 3
    elif fault == 'lookback': body['config']['lookback_bars'] = 7
    elif fault == 'target_name': body['target_name'] = 'wrong'
    sealed(training,body,'training_data_id')
    if fault == 'checksum': training.write_bytes(training.read_bytes().replace(b'deuidxeur',b'OTHER'))
    with pytest.raises((ValueError,OSError)):
        api().build_image_dataset(dataset,training,tmp_path/'out',cache=tmp_path/'cache',per_split=1)
    assert not (tmp_path/'out').exists()
    assert not (tmp_path/'cache').exists()


def rewrite_sources(dataset_path,training_path,transform):
    from dataclasses import replace
    from market_trader.storage import load_dataset
    old = load_dataset(dataset_path).dataset
    bars = transform(list(old.candles))
    data = normalize_candles(tuple(bars),config=old.config).datasets[0]
    new = dataset_path.with_name('changed.json')
    manifest = save_dataset(new,data,provenance=DatasetProvenance('synthetic changed',()))
    body = json.loads(training_path.read_bytes())
    body.update(dataset_id=data.dataset_id,dataset_artifact_id=manifest.artifact_id)
    config = TrainingConfig(8,2,datetime(2024,1,1,tzinfo=UTC),datetime(2025,1,1,tzinfo=UTC))
    rows = build_open_training_data(data,config,include_test=False)
    body['splits'] = {name:encode(getattr(rows,name)) for name in ('train','validation','test')}
    changed_training = training_path.with_name('changed-training.json')
    sealed(changed_training,body,'training_data_id')
    return new,changed_training


def test_future_mutation_changes_labels_but_never_input_cache_identity(tmp_path):
    from dataclasses import replace
    dataset,training = sources(tmp_path)
    before = api().build_image_dataset(dataset,training,tmp_path/'before',cache=tmp_path/'cache',per_split=1)
    def transform(bars):
        bars[10] = replace(bars[10],open=bars[10].open+1)
        return bars
    changed,changed_training = rewrite_sources(dataset,training,transform)
    after = api().build_image_dataset(changed,changed_training,tmp_path/'after',cache=tmp_path/'cache',per_split=1)
    assert before['manifest']['image_dataset_id'] != after['manifest']['image_dataset_id']
    assert (tmp_path/'before'/'samples.json').read_bytes() == (tmp_path/'after'/'samples.json').read_bytes()
    assert (tmp_path/'before'/'labels.json').read_bytes() != (tmp_path/'after'/'labels.json').read_bytes()
    assert after['cache_hits'] == 2 and after['cache_misses'] == 0
    assert bundle_bytes(tmp_path/'before'/'images') == bundle_bytes(tmp_path/'after'/'images')


def test_decimal_context_cannot_change_bundle_bytes(tmp_path):
    from decimal import localcontext
    dataset,training = sources(tmp_path)
    with localcontext() as ctx:
        ctx.prec = 6
        ctx.rounding = 'ROUND_UP'
        api().build_image_dataset(dataset,training,tmp_path/'one',cache=tmp_path/'cache',per_split=1)
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = 'ROUND_FLOOR'
        api().build_image_dataset(dataset,training,tmp_path/'two',cache=tmp_path/'cache',per_split=1)
    assert bundle_bytes(tmp_path/'one') == bundle_bytes(tmp_path/'two')


def test_sampling_upper_boundary_and_insufficient_rows(tmp_path):
    dataset,training = sources(tmp_path,size=34,days=3)
    result = api().build_image_dataset(dataset,training,tmp_path/'out',cache=tmp_path/'cache',per_split=64)
    assert result['sample_count'] == 128
    assert api().verify_image_dataset(tmp_path/'out') == result['manifest']
    short = tmp_path/'short'
    short.mkdir(mode=0o700)
    dataset,training = sources(short)
    with pytest.raises(ValueError,match='insufficient'):
        api().build_image_dataset(dataset,training,short/'out',cache=short/'cache',per_split=16)
    assert not (short/'cache').exists()


def test_cache_and_output_cannot_overlap(tmp_path):
    dataset,training = sources(tmp_path)
    output = tmp_path/'out'
    with pytest.raises(ValueError,match='overlap'):
        api().build_image_dataset(dataset,training,output,cache=output,per_split=1)
    assert not output.exists()


def test_builder_rejects_resigned_source_with_2025_candles(tmp_path):
    from dataclasses import replace
    from market_trader.storage import load_dataset
    dataset,training = sources(tmp_path)
    original = load_dataset(dataset).dataset
    new_start = datetime(2025,6,2,7,tzinfo=UTC)
    old_start = original.candles[0].open_time
    shift = new_start-old_start
    extra = tuple(replace(c,open_time=c.open_time+shift,close_time=c.close_time+shift,
                           available_at=c.available_at+shift) for c in original.candles[:20])
    old_calendar = original.config.calendar
    assert isinstance(old_calendar,SessionCalendar)
    calendar = replace(old_calendar,coverage_end=new_start+timedelta(days=1),
        sessions=(*old_calendar.sessions,TradingSession(new_start,new_start+20*timedelta(minutes=15))))
    data = normalize_candles((*original.candles,*extra),config=NormalizationConfig(calendar)).datasets[0]
    source2025 = tmp_path/'research.json'
    saved = save_dataset(source2025,data,provenance=DatasetProvenance('synthetic research with 2025',()))
    body = json.loads(training.read_bytes())
    body.update(dataset_id=data.dataset_id,dataset_artifact_id=saved.artifact_id)
    sealed(training,body,'training_data_id')
    with pytest.raises(ValueError,match='no 2025'):
        api().build_image_dataset(source2025,training,tmp_path/'out',cache=tmp_path/'cache',per_split=1)
    assert not (tmp_path/'out').exists() and not (tmp_path/'cache').exists()


@pytest.mark.parametrize('fault', ['unavailable','session','date2025','cutoff'])
def test_resigned_time_and_session_breaches(tmp_path,fault):
    dataset,training = sources(tmp_path)
    output = tmp_path/'out'
    api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    samples = json.loads((output/'samples.json').read_bytes())
    labels = json.loads((output/'labels.json').read_bytes())
    manifest = json.loads((output/'manifest.json').read_bytes())
    sample = samples['samples'][0]
    if fault == 'unavailable':
        sample['window']['candles'][0]['available_at'] = labels['labels'][0]['exit_at']
    elif fault == 'session':
        manifest['sessions'][sample['sample_id']]['close_time'] = labels['labels'][0]['exit_at']
    elif fault == 'date2025':
        sample['window']['candles'][0]['open_time'] = '2025-01-01T00:00:00.000000+00:00'
    else:
        labels['labels'][0]['future'][0]['available_at'] = labels['labels'][0]['target_available_at'] = '2024-01-01T00:00:00.000000+00:00'
    (output/'samples.json').write_bytes(json_bytes(samples))
    (output/'labels.json').write_bytes(json_bytes(labels))
    sealed(output/'manifest.json',manifest,'image_dataset_id')
    resign_bundle(output)
    with pytest.raises(ValueError): api().verify_image_dataset(output)


@pytest.mark.parametrize('kind',['file','directory','symlink'])
def test_output_is_new_and_never_overwrites(tmp_path,kind):
    dataset,training = sources(tmp_path)
    output = tmp_path/'out'
    if kind == 'file': output.write_bytes(b'untouched')
    elif kind == 'directory': output.mkdir()
    else: output.symlink_to(tmp_path/'absent')
    with pytest.raises(FileExistsError):
        api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    assert not (tmp_path/'cache').exists()
    if kind == 'file': assert output.read_bytes() == b'untouched'
    if kind == 'symlink': assert output.is_symlink()


def test_flat_bars_and_standalone_verification_without_trainers(tmp_path):
    from dataclasses import replace
    import subprocess
    import sys
    dataset,training = sources(tmp_path)
    def flat(bars):
        return [replace(c,open=Decimal('100.000'),high=Decimal('100.000'),low=Decimal('100.000'),
                        close=Decimal('100.000')) for c in bars]
    dataset,training = rewrite_sources(dataset,training,flat)
    result = api().build_image_dataset(dataset,training,tmp_path/'out',cache=tmp_path/'cache',per_split=1)
    dataset.unlink()
    training.unlink()
    script = '''import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy') or fullname in ('market_trader.baseline','market_trader.neural','market_trader.storage'):
            raise AssertionError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0,Deny())
from market_trader.image_dataset import verify_image_dataset
from pathlib import Path
print(verify_image_dataset(Path(sys.argv[1]))['image_dataset_id'])
'''
    checked = subprocess.run([sys.executable,'-c',script,str(tmp_path/'out')],capture_output=True,text=True)
    assert checked.returncode == 0,checked.stderr
    assert checked.stdout.strip() == result['manifest']['image_dataset_id']


def test_publication_race_refuses_even_empty_directory(tmp_path,monkeypatch):
    dataset,training = sources(tmp_path)
    publish = api()._publish_directory
    def race(stage,output,**kwargs):
        output.mkdir()
        (output/'sentinel').write_bytes(b'other writer')
        return publish(stage,output,**kwargs)
    monkeypatch.setattr(api(),'_publish_directory',race)
    with pytest.raises(FileExistsError):
        api().build_image_dataset(dataset,training,tmp_path/'out',cache=tmp_path/'cache',per_split=1)
    assert bundle_bytes(tmp_path/'out') == {'sentinel':b'other writer'}
    assert not list(tmp_path.glob('.image-dataset-*'))


@pytest.mark.parametrize('kind',['input_future','future_future'])
@pytest.mark.parametrize('field',['high','available_at'])
def test_repeated_candle_evidence_must_agree_across_all_samples(tmp_path,kind,field):
    dataset,training=sources(tmp_path)
    output=tmp_path/'out'
    count=2 if kind=='input_future' else 10
    api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=count)
    doc=json.loads((output/'labels.json').read_text())
    bar=doc['labels'][0]['future'][1] if kind=='input_future' else doc['labels'][8]['future'][2]
    if field=='high':
        bar[field]='300'
    else:
        bar[field]=(datetime.fromisoformat(bar[field])+timedelta(seconds=1)).isoformat(timespec='microseconds')
        # Preserve this sample's own target-availability consistency.
        if kind=='future_future':
            doc['labels'][8]['target_available_at']=bar[field]
    (output/'labels.json').write_bytes(json_bytes(doc))
    resign_bundle(output)
    with pytest.raises(ValueError,match='inconsistent repeated candle'):
        api().verify_image_dataset(output)


def test_high_precision_target_matches_frozen_training_subtract_then_divide(tmp_path):
    from dataclasses import replace
    dataset,training=sources(tmp_path)
    entry=Decimal('100.4035559837902845253152181850569148055868')
    exit=Decimal('101.1953312134647943162666978556017425890430')
    def transform(bars):
        return [replace(c,open=entry if i%20==8 else exit if i%20==10 else c.open,
                        high=Decimal('300'),low=Decimal('50')) for i,c in enumerate(bars)]
    dataset,training=rewrite_sources(dataset,training,transform)
    output=tmp_path/'out'
    result=api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    labels=json.loads((output/'labels.json').read_text())['labels']
    assert labels[0]['target_return']=='0.007885928161770868381470367754'
    assert api().verify_image_dataset(output)==result['manifest']


def test_publication_ancestor_swap_never_publishes_attacker_stage(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path)
    parent = tmp_path/'parent'
    parent.mkdir(mode=0o700)
    moved = tmp_path/'moved'
    attacker = tmp_path/'attacker'
    attacker.mkdir()
    publish = api()._publish_directory
    forged = []
    def race(stage, output, **kwargs):
        stage_name = stage.name
        parent.rename(moved)
        parent.symlink_to(attacker, target_is_directory=True)
        fake = attacker/stage_name
        fake.mkdir()
        (fake/'sentinel').write_bytes(b'attacker')
        forged.append(fake)
        return publish(stage, output, **kwargs)
    monkeypatch.setattr(api(), '_publish_directory', race)
    with pytest.raises((OSError, ValueError)):
        api().build_image_dataset(dataset, training, parent/'out',
                                  cache=tmp_path/'cache', per_split=1)
    assert not (attacker/'out').exists()
    assert not (moved/'out').exists()
    assert list(moved.iterdir()) == []
    assert bundle_bytes(forged[0]) == {'sentinel': b'attacker'}


def test_cache_missing_child_under_symlink_parent_is_never_created(tmp_path):
    dataset,training = sources(tmp_path)
    real = tmp_path/'real'
    real.mkdir()
    alias = tmp_path/'alias'
    alias.symlink_to(real,target_is_directory=True)
    with pytest.raises((ValueError,OSError)):
        api().build_image_dataset(dataset,training,tmp_path/'out',cache=alias/'new-cache',per_split=1)
    assert list(real.iterdir()) == []
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('fault',['policy_bool','recipe_bool','session_span','sample_reference_type'])
def test_strict_json_types_and_session_boundaries(tmp_path,fault):
    dataset,training = sources(tmp_path)
    output = tmp_path/'out'
    api().build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    manifest = json.loads((output/'manifest.json').read_bytes())
    if fault == 'policy_bool': manifest['policy']['test_count'] = False
    elif fault == 'recipe_bool':
        recipe = json.loads((output/'recipe.json').read_bytes())
        recipe['geometry']['wick_width'] = True
        (output/'recipe.json').write_bytes(json_bytes(recipe))
    elif fault == 'session_span':
        session = next(iter(manifest['sessions'].values()))
        session['open_time'] = '2021-01-01T00:00:00.000000+00:00'
        session['close_time'] = '2024-12-31T23:00:00.000000+00:00'
    else:
        samples = json.loads((output/'samples.json').read_bytes())
        samples['samples'][0]['image_path'] = {'bad':'reference'}
        (output/'samples.json').write_bytes(json_bytes(samples))
    sealed(output/'manifest.json',manifest,'image_dataset_id')
    resign_bundle(output)
    with pytest.raises(ValueError): api().verify_image_dataset(output)
