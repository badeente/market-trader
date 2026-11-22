"""AP7 full causal cohort and compact input-only cache; no framework imports.

Private caches trust their current-UID owner, not malicious same-UID writers.
Source artifacts and AP6 audit bundles retain their original schema and limits.
"""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from market_trader._dataset_codec import encode, json_bytes
from market_trader.chart_images import image_key, window_payload
from market_trader.domain import Candle
from market_trader.image_dataset import START, VALIDATION, END, STEP, _session
from market_trader.image_pixels_v1 import image_input_recipe
from market_trader.normalization import NormalizedDataset
from market_trader.training_data import TrainingDataset

MAX_SAMPLES = 50_000
MAX_SOURCE_BYTES = 96*1024*1024
MAX_CANDLES = 100_000


def _source_bytes(path):
    from market_trader.image_dataset import _directory, _read_at
    path = Path(path)
    with _directory(path.parent) as directory:
        return _read_at(directory, path.name, limit=MAX_SOURCE_BYTES)


@dataclass(frozen=True, slots=True)
class _TrainingSnapshot:
    payload: bytes

    def read_text(self, encoding):
        return self.payload.decode(encoding)


def _source_dataset(path):
    # Apply the existing checked storage codec to a bounded immutable snapshot.
    from market_trader._dataset_codec import parse_json, record_object, decode_record
    from market_trader.artifacts import ARTIFACT_FORMAT, DatasetManifest, DatasetProvenance, StoredDataset
    from market_trader.normalization import NORMALIZATION_FORMAT
    from market_trader.storage import _validate_dataset, _validate_provenance, _hash
    document = record_object(parse_json(_source_bytes(path).decode('utf-8')),
                             ('format', 'normalization_format', 'artifact_id', 'dataset', 'provenance'))
    if document['format'] != ARTIFACT_FORMAT or document['normalization_format'] != NORMALIZATION_FORMAT:
        raise ValueError('unsupported source dataset artifact format')
    _hash(document['artifact_id'])
    if document['artifact_id'] != _digest(json_bytes({k:v for k,v in document.items() if k != 'artifact_id'})):
        raise ValueError('source dataset artifact checksum mismatch')
    raw = document['dataset']
    if (not isinstance(raw, dict) or not isinstance(raw.get('candles'), list)
            or len(raw['candles']) > MAX_CANDLES):
        raise ValueError('source candle resource bound exceeded')
    data = decode_record(raw, NormalizedDataset)
    provenance = decode_record(document['provenance'], DatasetProvenance)
    _validate_dataset(data)
    _validate_provenance(provenance)
    return StoredDataset(data, DatasetManifest(data.dataset_id, document['artifact_id'], provenance))


def _digest(payload):
    return 'sha256:' + sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ImageCohort:
    data: NormalizedDataset
    training: TrainingDataset
    training_data_id: str
    dataset_artifact_id: str

    def window(self, row) -> tuple[Candle, ...]:
        # Binary search avoids a copied all-candle mapping for every input.
        from bisect import bisect_left
        end = bisect_left(self.data.candles, row.decision_at, key=lambda c: c.close_time)
        history = self.data.candles[max(0, end-7):end+1]
        window_payload(history, decision_at=row.decision_at)
        session = self.data.config.calendar.session_at(history[0].open_time)
        if session is None:
            raise ValueError('missing research session')
        if history[-1].close_time > session.close_time:
            raise ValueError('window crosses session')
        return history

    def index_document(self) -> dict:
        samples = []
        for split in ('train', 'validation'):
            for row in getattr(self.training, split):
                body = {'split': split, 'decision_at': encode(row.decision_at),
                        'image_key': image_key(self.window(row), decision_at=row.decision_at)}
                samples.append({'sample_id': _digest(json_bytes(body)), **body})
        body = {'format': 'market-trader.image-index.v1',
                'input_recipe': image_input_recipe(), 'samples': samples}
        return {**body, 'input_index_id': _digest(json_bytes(body))}

    def labels_document(self) -> dict:
        index = self.index_document()
        rows = (*self.training.train, *self.training.validation)
        labels = [{'sample_id': sample['sample_id'],
                   **{field: encode(getattr(row, field)) for field in
                      ('entry_at', 'exit_at', 'target_available_at', 'target_return')}}
                  for sample, row in zip(index['samples'], rows, strict=True)]
        body = {'format': 'market-trader.image-training-labels.v1',
                'source_training_data_id': self.training_data_id, 'labels': labels}
        return {**body, 'labels_id': _digest(json_bytes(body))}


def prepare_image_cohort(dataset_path: Path, training_path: Path) -> ImageCohort:
    from market_trader.baseline import _read_training_data, _checked_split
    from market_trader.training_data import build_open_training_data, OPEN_TRAINING_DATA_FORMAT
    from market_trader.calendars import SessionCalendar
    from market_trader.chart_images import validate_chart_price

    stored = _source_dataset(dataset_path)
    data = stored.dataset
    source, config, duration = _read_training_data(_TrainingSnapshot(_source_bytes(training_path)))
    if (any(not isinstance(source['splits'][s], list) for s in ('train', 'validation', 'test'))
            or sum(len(source['splits'][s]) for s in ('train', 'validation', 'test')) > MAX_SAMPLES):
        raise ValueError('source exceeds maximum50000 sample resource bound')
    if (source['format'] != OPEN_TRAINING_DATA_FORMAT or config.lookback_bars != 8
            or config.horizon_bars != 2 or config.validation_start != VALIDATION
            or config.test_start != END or source['splits']['test'] != []
            or data.instrument != 'deuidxeur' or data.interval != STEP
            or source['instrument'] != data.instrument
            or source['interval_microseconds'] != 900000000
            or source['dataset_id'] != data.dataset_id
            or source['dataset_artifact_id'] != stored.manifest.artifact_id
            or not isinstance(data.config.calendar, SessionCalendar)):
        raise ValueError('source dataset/training metadata or fixed policy mismatch')
    if any(not START <= c.open_time < c.close_time < END or c.available_at >= END
           for c in data.candles):
        raise ValueError('source candles must be development-only 2021-2024; no 2025')
    for candle in data.candles:
        for field in ('open', 'high', 'low', 'close'):
            validate_chart_price(getattr(candle, field))
    rebuilt = build_open_training_data(data, config, include_test=False)
    if len(rebuilt.train)+len(rebuilt.validation) > MAX_SAMPLES:
        raise ValueError('rebuilt cohort exceeds maximum50000 sample resource bound')
    for split, start, cutoff in (('train', START, VALIDATION), ('validation', VALIDATION, END)):
        checked = _checked_split(source['splits'][split], name=split, start=start,
                                 cutoff=cutoff, duration=duration, availability_delay=STEP)
        if checked != getattr(rebuilt, split):
            raise ValueError('sealed training rows do not reconcile with source candles')
    cohort = ImageCohort(data, rebuilt, source['training_data_id'], stored.manifest.artifact_id)
    # Validate same-session research policy for every eligible input AND target.
    by_open = {c.open_time: c for c in data.candles}
    for row in (*rebuilt.train, *rebuilt.validation):
        history = cohort.window(row)
        future = tuple(by_open[row.decision_at+i*STEP] for i in range(3))
        session = data.config.calendar.session_at(history[0].open_time)
        _session(history, future, encode(session))
    return cohort


# Headerless rows permit verified NumPy memmap access without float64 copies.
_ROW_BYTES = 2304*2


def _file_state(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_regular(directory, name, expected_size=None):
    import os
    import stat
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077
                or (expected_size is not None and info.st_size != expected_size)):
            raise ValueError('cache requires private current-owner bounded regular unlinked files')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _private_directory(fd):
    import os
    import stat
    info = os.fstat(fd)
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('cache directory and parent must be private current-owner mode0700')


@dataclass(frozen=True, slots=True)
class InputCache:
    path: Path
    row_count: int
    train_count: int
    input_index_id: str
    sha256: str
    _state: tuple

    def row(self, index) -> tuple[int, ...]:
        import os
        import struct
        from market_trader.image_dataset import _directory
        if type(index) is not int or not 0 <= index < self.row_count:
            raise ValueError('cache row index outside bounded integer range')
        with _directory(self.path.parent.parent) as parent:
            _private_directory(parent)
        with _directory(self.path.parent) as directory:
            _private_directory(directory)
            fd = _open_regular(directory, self.path.name, self.row_count*_ROW_BYTES)
            try:
                if _file_state(os.fstat(fd)) != self._state:
                    raise ValueError('verified cache changed; refusing repair')
                payload = os.pread(fd, _ROW_BYTES, index*_ROW_BYTES)
                if len(payload) != _ROW_BYTES or _file_state(os.fstat(fd)) != self._state:
                    raise ValueError('cache row changed or truncated')
                values = struct.unpack('<2304H', payload)
                if any(v > 16320 for v in values):
                    raise ValueError('cache row exceeds block sum bounds')
                return values
            finally:
                os.close(fd)


def _publish_stream(directory, name, chunks):
    import os
    import secrets
    temporary = '.input-'+secrets.token_hex(16)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=directory)
    try:
        with os.fdopen(fd, 'wb') as output:
            for chunk in chunks:
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=directory)


def build_input_cache(cohort, cache_dir: Path, *, accelerated=False) -> InputCache:
    import os
    import struct
    from market_trader.chart_images import render_candles
    from market_trader.image_dataset import _directory, _read_at
    from market_trader._dataset_codec import parse_json
    from market_trader.image_pixels_v1 import block_sums
    if type(accelerated) is not bool:
        raise ValueError('accelerated must be bool')
    count = len(cohort.training.train)+len(cohort.training.validation)
    if not 0 < count <= MAX_SAMPLES:
        raise ValueError('cohort exceeds maximum50000 sample resource bound')
    cache_dir = Path(cache_dir).absolute()
    with _directory(cache_dir.parent) as parent:
        _private_directory(parent)
        try:
            os.mkdir(cache_dir.name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
    with _directory(cache_dir) as directory:
        _private_directory(directory)
        index = cohort.index_document()
        identity = index['input_index_id']
        stem = identity[7:]
        data_name, meta_name = stem+'.u16', stem+'.json'
        expected = {'format': 'market-trader.image-input-cache.v1',
                    'input_index_id': identity, 'input_recipe_id': index['input_recipe']['recipe_id'],
                    'shape': [count, 3, 24, 32], 'dtype': '<u2', 'byte_length': count*_ROW_BYTES,
                    'train_count': len(cohort.training.train)}
        present = [os.path.lexists(Path('/proc/self/fd')/str(directory)/name)
                   for name in (data_name, meta_name)]
        if any(present) and not all(present):
            raise ValueError('incomplete cache hit; refusing repair')
        if not any(present):
            digest = sha256()
            def rows():
                for row in (*cohort.training.train, *cohort.training.validation):
                    png = render_candles(cohort.window(row), decision_at=row.decision_at)
                    payload = struct.pack('<2304H', *block_sums(png, accelerated=accelerated))
                    digest.update(payload)
                    yield payload
            _publish_stream(directory, data_name, rows())
            metadata = {**expected, 'sha256': 'sha256:'+digest.hexdigest()}
            _publish_stream(directory, meta_name, (json_bytes(metadata),))
        metadata = parse_json(_read_at(directory, meta_name, limit=8192, private=True).decode('utf-8'))
        if (not isinstance(metadata, dict) or set(metadata) != set(expected)|{'sha256'}
                or any(type(metadata[k]) is not type(v) or metadata[k] != v for k,v in expected.items())):
            raise ValueError('cache metadata mismatch; refusing repair')
        fd = _open_regular(directory, data_name, count*_ROW_BYTES)
        try:
            before = _file_state(os.fstat(fd))
            digest = sha256()
            for _ in range(count):
                payload = os.read(fd, _ROW_BYTES)
                if len(payload) != _ROW_BYTES or any(v > 16320 for v in struct.unpack('<2304H', payload)):
                    raise ValueError('corrupt cache values; refusing repair')
                digest.update(payload)
            if os.read(fd, 1) or _file_state(os.fstat(fd)) != before:
                raise ValueError('cache changed during verification')
            checksum = 'sha256:'+digest.hexdigest()
            if metadata['sha256'] != checksum:
                raise ValueError('corrupt cache checksum; refusing repair')
        finally:
            os.close(fd)
    return InputCache(cache_dir/data_name, count, len(cohort.training.train), identity, checksum, before)
