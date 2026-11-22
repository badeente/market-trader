"""AP7 full synthetic cohort and compact cache; never real fits/data."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
import os
import struct

import pytest

from tests.test_image_dataset import sources, sealed, rewrite_sources
from market_trader._dataset_codec import json_bytes


def api():
    from market_trader import image_training_data
    return image_training_data


def test_full_cohort_reconciles_all_rows_and_separates_labels(tmp_path):
    dataset, training=sources(tmp_path,size=34,days=3)
    cohort=api().prepare_image_cohort(dataset,training)
    original=json.loads(training.read_bytes())
    assert cohort.training_data_id==original['training_data_id']
    assert cohort.dataset_artifact_id==original['dataset_artifact_id']
    assert len(cohort.training.train)==len(cohort.training.validation)==72
    assert cohort.training.test==()
    with pytest.raises(FrozenInstanceError): cohort.training_data_id='changed'
    index=cohort.index_document()
    labels=cohort.labels_document()
    assert index['format']=='market-trader.image-index.v1'
    assert len(index['samples'])==144
    assert [s['split'] for s in index['samples']]==['train']*72+['validation']*72
    assert index['input_index_id']=='sha256:'+sha256(json_bytes({k:v for k,v in index.items() if k!='input_index_id'})).hexdigest()
    assert labels['source_training_data_id']==cohort.training_data_id
    assert len(labels['labels'])==144
    assert not any(x in json.dumps(index) for x in ('target_return','entry_at','dataset_id','training_data_id'))
    row=cohort.training.train[0]
    window=cohort.window(row)
    assert len(window)==8 and window[-1].close_time==row.decision_at
    assert all(c.available_at<=row.decision_at for c in window)
    index['samples'].clear()
    assert len(cohort.index_document()['samples'])==144


def test_compact_cache_exact_rows_cold_warm_no_png_corpus(tmp_path):
    from market_trader.chart_images import render_candles
    from market_trader.image_pixels_v1 import block_sums
    dataset, training = sources(tmp_path, size=12)
    cohort = api().prepare_image_cohort(dataset, training)
    cold = api().build_input_cache(cohort, tmp_path/'cache')
    assert cold.row_count == 4 and cold.train_count == 2
    assert cold.input_index_id == cohort.index_document()['input_index_id']
    assert cold.path.stat().st_size == 4*2304*2
    assert cold.sha256 == 'sha256:'+sha256(cold.path.read_bytes()).hexdigest()
    for i, row in enumerate((*cohort.training.train, *cohort.training.validation)):
        assert cold.row(i) == block_sums(render_candles(cohort.window(row), decision_at=row.decision_at))
        assert struct.unpack('<2304H', cold.path.read_bytes()[i*4608:(i+1)*4608]) == cold.row(i)
    warm = api().build_input_cache(cohort, tmp_path/'cache')
    assert warm == cold
    assert sorted(p.suffix for p in (tmp_path/'cache').iterdir()) == ['.json', '.u16']
    for bad in (-1, 4, True, 1.0):
        with pytest.raises(ValueError): cold.row(bad)
    metadata = json.loads(next((tmp_path/'cache').glob('*.json')).read_bytes())
    assert not any(s in json.dumps(metadata) for s in ('target', 'training_data_id', 'dataset_id'))


def test_sample_resource_bound_precedes_rebuilding(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path, size=12)
    monkeypatch.setattr(api(), 'MAX_SAMPLES', 3)
    import market_trader.training_data as td
    def forbidden(*args, **kwargs):
        pytest.fail('sample bound must be checked before building features')
    monkeypatch.setattr(td, 'build_open_training_data', forbidden)
    with pytest.raises(ValueError, match='resource|50000'):
        api().prepare_image_cohort(dataset, training)


@pytest.mark.parametrize('which', ['dataset', 'training'])
def test_source_byte_resource_bound_precedes_json_parsing(tmp_path, which):
    dataset, training = sources(tmp_path, size=12)
    target = dataset if which == 'dataset' else training
    with target.open('wb') as f:
        f.truncate(96*1024*1024+1)
    with pytest.raises(ValueError, match='resource|budget|bounded'):
        api().prepare_image_cohort(dataset, training)


@pytest.mark.parametrize('fault', ['wrong_dataset','wrong_artifact','features','target','deleted_row',
    'boundary','test_rows','horizon','lookback','target_name','checksum','instrument'])
def test_all_original_rows_metadata_and_fixed_policy_are_reconciled(tmp_path, fault):
    dataset, training = sources(tmp_path)
    body = json.loads(training.read_bytes())
    row = body['splits']['validation'][3]
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
    elif fault == 'instrument': body['instrument'] = 'DEUIDXEUR'
    sealed(training, body, 'training_data_id')
    if fault == 'checksum': training.write_bytes(training.read_bytes().replace(b'deuidxeur', b'OTHER'))
    with pytest.raises((ValueError, OSError)):
        api().prepare_image_cohort(dataset, training)


def test_future_only_mutation_preserves_full_input_index_and_cache(tmp_path):
    dataset, training = sources(tmp_path, size=12)
    before = api().prepare_image_cohort(dataset, training)
    cold = api().build_input_cache(before, tmp_path/'cache')
    def transform(bars):
        bars[-1] = replace(bars[-1], open=bars[-1].open+1)
        return bars
    changed, changed_training = rewrite_sources(dataset, training, transform)
    after = api().prepare_image_cohort(changed, changed_training)
    assert before.index_document() == after.index_document()
    assert before.labels_document() != after.labels_document()
    assert before.training_data_id != after.training_data_id
    assert api().build_input_cache(after, tmp_path/'cache') == cold


@pytest.mark.parametrize('fault', ['data_corrupt','data_oversize','data_symlink','data_hardlink',
    'data_directory','metadata_corrupt','metadata_symlink','metadata_hardlink','metadata_oversize',
    'incomplete','root_symlink','ancestor_symlink','public','public_parent','file_public','fifo'])
def test_unsafe_or_corrupt_cache_hits_are_not_repaired(tmp_path, fault):
    dataset, training = sources(tmp_path, size=11)
    cohort = api().prepare_image_cohort(dataset, training)
    root = tmp_path/'cache'
    cache = api().build_input_cache(cohort, root)
    data = cache.path
    metadata = next(root.glob('*.json'))
    target = metadata if fault.startswith('metadata') else data
    if fault.endswith(('symlink','hardlink','directory')) and not fault.startswith(('root','ancestor')):
        saved = tmp_path/'saved'
        target.rename(saved)
        if fault.endswith('symlink'): target.symlink_to(saved)
        elif fault.endswith('hardlink'): os.link(saved, target)
        else: target.mkdir()
    elif fault.endswith('corrupt'): target.write_bytes(b'bad')
    elif fault.endswith('oversize'):
        with target.open('ab') as f: f.truncate(10000 if target == metadata else 250000000)
    elif fault == 'incomplete': metadata.unlink()
    elif fault == 'public': root.chmod(0o755)
    elif fault == 'public_parent': tmp_path.chmod(0o755)
    elif fault == 'file_public': data.chmod(0o644)
    elif fault == 'fifo':
        data.unlink()
        os.mkfifo(data)
    elif fault == 'root_symlink':
        root.rename(tmp_path/'saved-root')
        root.symlink_to(tmp_path/'saved-root', target_is_directory=True)
    elif fault == 'ancestor_symlink':
        link = tmp_path/'link'
        link.symlink_to(tmp_path, target_is_directory=True)
        root = link/'cache'
    before = [(p.name, p.lstat()) for p in root.iterdir()]
    with pytest.raises((ValueError, OSError)):
        api().build_input_cache(cohort, root)
    assert before == [(p.name, p.lstat()) for p in root.iterdir()]


def test_verified_cache_row_rejects_post_verification_mutation(tmp_path):
    dataset, training = sources(tmp_path, size=11)
    cache = api().build_input_cache(api().prepare_image_cohort(dataset, training), tmp_path/'cache')
    with cache.path.open('r+b') as f: f.write(b'\0\0')
    with pytest.raises(ValueError, match='changed'):
        cache.row(0)


def test_no_2025_source_reaches_feature_builder(tmp_path, monkeypatch):
    from datetime import timedelta
    from market_trader.artifacts import DatasetProvenance
    from market_trader.calendars import SessionCalendar
    from market_trader.normalization import normalize_candles
    from market_trader.storage import load_dataset, save_dataset
    dataset, training = sources(tmp_path, size=12)
    old = load_dataset(dataset).dataset
    shift = timedelta(days=365)
    bars = old.candles[:12]+tuple(replace(c, open_time=c.open_time+shift,
        close_time=c.close_time+shift, available_at=c.available_at+shift) for c in old.candles[12:])
    calendar = old.config.calendar
    sessions = (calendar.sessions[0], replace(calendar.sessions[1],
        open_time=calendar.sessions[1].open_time+shift, close_time=calendar.sessions[1].close_time+shift))
    calendar = replace(calendar, coverage_end=calendar.coverage_end+shift, sessions=sessions)
    changed = normalize_candles(bars, config=replace(old.config, calendar=calendar)).datasets[0]
    path = tmp_path/'with-2025.json'
    manifest = save_dataset(path, changed, provenance=DatasetProvenance('synthetic', ()))
    body = json.loads(training.read_bytes())
    body.update(dataset_id=changed.dataset_id, dataset_artifact_id=manifest.artifact_id)
    sealed(training, body, 'training_data_id')
    import market_trader.training_data as td
    def forbidden(*args, **kwargs): pytest.fail('2025 reached feature construction')
    monkeypatch.setattr(td, 'build_open_training_data', forbidden)
    with pytest.raises(ValueError, match='no 2025'):
        api().prepare_image_cohort(path, training)


def test_candle_count_resource_boundary(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path, size=12)
    monkeypatch.setattr(api(), 'MAX_CANDLES', 23)
    with pytest.raises(ValueError, match='candle resource'):
        api().prepare_image_cohort(dataset, training)


def test_cache_owner_and_cold_parent_checks_precede_creation(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path, size=11)
    cohort = api().prepare_image_cohort(dataset, training)
    root = tmp_path/'cache'
    uid = os.getuid()
    monkeypatch.setattr(os, 'getuid', lambda: uid+1)
    with pytest.raises(ValueError, match='owner'):
        api().build_input_cache(cohort, root)
    assert not root.exists()


@pytest.mark.parametrize('fault', ['checksum','value','identity','shape','extra','duplicate_key'])
def test_cache_coherent_metadata_faults_and_hash_mismatch_rejected(tmp_path, fault):
    dataset, training = sources(tmp_path, size=11)
    cohort = api().prepare_image_cohort(dataset, training)
    root = tmp_path/'cache'
    cache = api().build_input_cache(cohort, root)
    metadata = next(root.glob('*.json'))
    body = json.loads(metadata.read_bytes())
    if fault in ('checksum','value'):
        with cache.path.open('r+b') as f: f.write(struct.pack('<H', 16321 if fault == 'value' else 0))
        if fault == 'value': body['sha256'] = 'sha256:'+sha256(cache.path.read_bytes()).hexdigest()
    elif fault == 'identity': body['input_index_id'] = 'sha256:'+'a'*64
    elif fault == 'shape': body['shape'] = [2,3,32,24]
    elif fault == 'extra': body['target_return'] = 'not an input'
    metadata.write_bytes(json_bytes(body))
    if fault == 'duplicate_key': metadata.write_bytes(b'{"format":"bad",'+metadata.read_bytes()[1:])
    before = cache.path.read_bytes(), metadata.read_bytes()
    with pytest.raises(ValueError): api().build_input_cache(cohort, root)
    assert before == (cache.path.read_bytes(), metadata.read_bytes())


def test_failed_new_cache_generation_leaves_no_partial_published_data(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path, size=11)
    cohort = api().prepare_image_cohort(dataset, training)
    import market_trader.chart_images as images
    def fail(*args, **kwargs): raise OSError('injected renderer failure')
    monkeypatch.setattr(images, 'render_candles', fail)
    with pytest.raises(OSError, match='injected'):
        api().build_input_cache(cohort, tmp_path/'cache')
    assert list((tmp_path/'cache').iterdir()) == []


def test_rebuilt_full_cohort_also_cannot_exceed_sample_resource_limit(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path, size=11)
    cohort = api().prepare_image_cohort(dataset, training)
    import market_trader.training_data as td
    oversized = replace(cohort.training, train=cohort.training.train*25001,
                        validation=cohort.training.validation*25000)
    monkeypatch.setattr(td, 'build_open_training_data', lambda *a, **kw: oversized)
    with pytest.raises(ValueError, match='resource|50000'):
        api().prepare_image_cohort(dataset, training)
    with pytest.raises(ValueError, match='resource|50000'):
        api().build_input_cache(replace(cohort, training=oversized), tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()


def test_numpy_accelerated_cache_matches_every_stdlib_byte(tmp_path):
    pytest.importorskip('numpy')
    dataset, training = sources(tmp_path, size=12)
    cohort = api().prepare_image_cohort(dataset, training)
    slow = api().build_input_cache(cohort, tmp_path/'slow')
    fast = api().build_input_cache(cohort, tmp_path/'fast', accelerated=True)
    assert slow.path.read_bytes() == fast.path.read_bytes()
    assert slow.sha256 == fast.sha256
    assert slow.input_index_id == fast.input_index_id
