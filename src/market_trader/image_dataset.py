"""Bounded, verified causal image bundles; no trainer runtime required.

Checksums prove internal integrity, not authenticity against a party able to
rewrite every artifact. Build reconciles all sealed source rows; standalone
verification regenerates images and labels from the separately stored evidence.

Build output parents must be private (mode0700, no group/other permissions),
owned by the current UID, and managed by a trusted owner. Malicious same-UID
writers and root are outside this contract; this is not a sandbox for them.
"""
import ctypes
from contextlib import contextmanager
import stat
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext, MIN_EMIN, MAX_EMAX
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile
from zoneinfo import ZoneInfo

from market_trader._dataset_codec import (decode_record, encode, json_bytes,
                                          parse_json, record_object)
from market_trader.calendars import SessionCalendar
from market_trader.domain import Candle

FORMAT = 'market-trader.image-dataset.v1'
SAMPLES_FORMAT = 'market-trader.image-samples.v1'
LABELS_FORMAT = 'market-trader.image-labels.v1'
STEP = timedelta(minutes=15)
START = datetime(2021, 1, 1, tzinfo=UTC)
VALIDATION = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2025, 1, 1, tzinfo=UTC)
_HASH = re.compile(r'sha256:[0-9a-f]{64}')
POLICY = {'lookback_bars':8, 'horizon_bars':2, 'interval_microseconds':900000000,
          'instrument':'deuidxeur', 'start':encode(START),
          'validation_start':encode(VALIDATION), 'test_start':encode(END),
          'test_count':0, 'sampling':'floor(i*(n-1)/(k-1)); k=1 selects first',
          'research_session':'Europe/Berlin Mon-Fri 09:00-17:30'}
BAR_FIELDS = ('open_time','close_time','available_at','open','high','low','close')


def _digest(data):
    return 'sha256:' + sha256(data).hexdigest()


def _identity(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError('expected lowercase sha256 identifier')
    return value


def _time(value):
    if not isinstance(value, str):
        raise ValueError('expected UTC timestamp string')
    result = datetime.fromisoformat(value)
    if result.utcoffset() != timedelta(0):
        raise ValueError('expected UTC timestamp')
    return result


def _indices(count, budget):
    if count < budget:
        raise ValueError('insufficient rows for per_split')
    return [0] if budget == 1 else [i*(count-1)//(budget-1) for i in range(budget)]


def _bar(candle):
    def price(value):
        text = format(value,'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    return {field:(price(getattr(candle,field)) if field in ('open','high','low','close')
                   else encode(getattr(candle,field))) for field in BAR_FIELDS}


def _candles(rows):
    if not isinstance(rows, list) or len(rows) not in (3,8):
        raise ValueError('expected bounded candle array')
    candles = tuple(decode_record({**record_object(row, BAR_FIELDS), 'instrument':'deuidxeur',
                               'interval':900000000, 'volume':'0'}, Candle) for row in rows)
    from market_trader.chart_images import validate_chart_price
    for index, bar in enumerate(candles):
        prices = (bar.open,bar.high,bar.low,bar.close)
        for price in prices:
            validate_chart_price(price)
        if (any(price <= 0 for price in prices)
                or not bar.low <= min(bar.open,bar.close) <= max(bar.open,bar.close) <= bar.high
                or not START <= bar.open_time < bar.close_time < END or bar.available_at >= END
                or bar.close_time-bar.open_time != STEP or bar.available_at < bar.close_time
                or (bar.open_time-datetime(1970,1,1,tzinfo=UTC)) % STEP
                or (index and bar.open_time != candles[index-1].close_time)):
            raise ValueError('invalid candle prices, chronology or development dates')
    return candles


def _session(history, future, session):
    record_object(session, ('open_time','close_time'))
    opening, closing = _time(session['open_time']), _time(session['close_time'])
    if not opening <= history[0].open_time < closing or future[-1].close_time > closing:
        raise ValueError('window or target crosses session')
    berlin = ZoneInfo('Europe/Berlin')
    local_open, local_close = opening.astimezone(berlin), closing.astimezone(berlin)
    if (local_open.date() != local_close.date() or local_open.weekday() >= 5
            or local_open.hour*60+local_open.minute < 540
            or local_close.hour*60+local_close.minute > 1050
            or (opening-datetime(1970,1,1,tzinfo=UTC)) % STEP
            or (closing-opening) % STEP):
        raise ValueError('invalid research session bounds')
    for bar in (*history, *future):
        local = bar.open_time.astimezone(berlin)
        close = bar.close_time.astimezone(berlin)
        if (local.weekday() >= 5 or close.date() != local.date()
                or local.hour*60+local.minute < 540 or close.hour*60+close.minute > 1050
                or local.date() != history[0].open_time.astimezone(berlin).date()):
            raise ValueError('candle outside eligible research session')


def _parent_binding(path, descriptor):
    with _directory(path) as current:
        _private(current, output_parent=True)
        _private(descriptor, output_parent=True)
        if not os.path.samestat(os.fstat(current), os.fstat(descriptor)):
            raise ValueError('output parent changed during build')


def _publish_directory(stage, output, *, parent_fd, stage_fd, container_fd):
    # The source child is protected by its private holder, not an outer name.
    _parent_binding(output.parent, parent_fd)
    _private(container_fd)
    if not os.path.samestat(os.stat('bundle', dir_fd=container_fd, follow_symlinks=False), os.fstat(stage_fd)):
        raise ValueError('verified stage changed before publication')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(container_fd, b'bundle', parent_fd, os.fsencode(output.name), 1)
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))
    try:
        if not os.path.samestat(os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False), os.fstat(stage_fd)):
            raise ValueError('published stage identity changed')
        _parent_binding(output.parent, parent_fd)
    except (OSError, ValueError):
        # Recurse only inside our pinned bundle, never through a mutable outer
        # destination name. The private parent has a trusted owner; hostile
        # same-UID swaps are outside the contract. Outer removal is nonrecursive.
        try:
            _clear_owned_directory(stage_fd)
            if os.path.samestat(os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False), os.fstat(stage_fd)):
                os.rmdir(output.name, dir_fd=parent_fd)
        except OSError:
            pass  # Do not replace the initiating publication failure.
        raise


def _clear_owned_directory(descriptor):
    """Clear contents of a pinned, private directory controlled by a trusted UID.

    Names below this descriptor are trusted under the same-UID owner contract;
    do not use this helper for an untrusted outer destination or namespace.
    """
    for name in os.listdir(descriptor):
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)


def build_image_dataset(dataset_path: Path, training_path: Path, output: Path, *,
                        cache: Path, per_split: int = 16) -> dict:
    """Export into a current-UID private mode0700 parent managed by a trusted owner.

    Reject nonprivate/wrong-owner parents before cache or output side effects.
    Malicious same-UID writers are explicitly outside the publication contract.
    """
    if type(per_split) is not int or not 1 <= per_split <= 64:
        raise ValueError('per_split must be an integer from 1 to 64')
    # Source checks are deliberately local imports; verification needs neither.
    from market_trader.baseline import _read_training_data, _checked_split
    from market_trader.storage import load_dataset
    from market_trader.training_data import build_open_training_data, OPEN_TRAINING_DATA_FORMAT

    output, cache = Path(output), Path(cache)
    absolute_output, absolute_cache = output.absolute(), cache.absolute()
    if (absolute_output == absolute_cache or absolute_output in absolute_cache.parents
            or absolute_cache in absolute_output.parents):
        raise ValueError('cache and output must not overlap')
    if os.path.lexists(output):
        raise FileExistsError(str(output))
    with _directory(output.parent) as output_parent:
        _private(output_parent, output_parent=True)
    stored = load_dataset(dataset_path)
    data = stored.dataset
    training, config, duration = _read_training_data(Path(training_path))
    if (training['format'] != OPEN_TRAINING_DATA_FORMAT or config.lookback_bars != 8
            or config.horizon_bars != 2 or config.validation_start != VALIDATION
            or config.test_start != END or training['splits']['test'] != []
            or data.instrument != 'deuidxeur' or data.interval != STEP
            or training['instrument'] != data.instrument
            or training['interval_microseconds'] != 900000000
            or training['dataset_id'] != data.dataset_id
            or training['dataset_artifact_id'] != stored.manifest.artifact_id
            or not isinstance(data.config.calendar, SessionCalendar)):
        raise ValueError('source dataset/training metadata or fixed policy mismatch')
    if any(not START <= c.open_time < c.close_time < END or c.available_at >= END
           for c in data.candles):
        raise ValueError('source candles must be development-only 2021-2024; no 2025')
    from market_trader.chart_images import validate_chart_price
    for candle in data.candles:
        for field in ('open', 'high', 'low', 'close'):
            validate_chart_price(getattr(candle, field))
    rebuilt = build_open_training_data(data, config, include_test=False)
    for split, start, cutoff in (('train',START,VALIDATION),('validation',VALIDATION,END)):
        checked = _checked_split(training['splits'][split], name=split, start=start,
            cutoff=cutoff, duration=duration, availability_delay=STEP)
        if checked != getattr(rebuilt,split):
            raise ValueError('sealed training rows do not reconcile with source candles')
        _indices(len(checked), per_split)
    with _directory(output.parent) as output_parent:
        _private(output_parent, output_parent=True)
        return _build_image_bundle(data, rebuilt, training, stored, output, cache, per_split, output_parent)


def _build_image_bundle(data, rebuilt, training, stored, output, cache, per_split, output_parent):
    with _directory(cache.parent) as parent:
        try:
            os.mkdir(cache.name,mode=0o700,dir_fd=parent)
        except FileExistsError:
            pass
    with _directory(cache) as descriptor:
        _private(descriptor)
    with _staging(output_parent) as (stage, container_fd, stage_fd):
        return _write_image_bundle(data, rebuilt, training, stored, output, cache, per_split,
                                   output_parent, stage, container_fd, stage_fd)


def _write_image_bundle(data, rebuilt, training, stored, output, cache, per_split,
                        output_parent, stage, container_fd, stage_fd):
    from market_trader.chart_images import chart_recipe, window_payload, image_key, render_candles
    by_open = {c.open_time:c for c in data.candles}
    stage_files = Path(f'/proc/self/fd/{stage_fd}')
    hits = misses = 0
    samples, labels, sessions = [], [], {}
    artifacts = {}
    (stage_files/'images').mkdir(mode=0o700)
    def write(name, payload):
        (stage_files/name).write_bytes(payload)
        artifacts[name] = _digest(payload)
    recipe = chart_recipe()
    write('recipe.json', json_bytes(recipe))
    split_counts = {}
    for split in ('train','validation'):
        rows = getattr(rebuilt,split)
        split_counts[split] = len(rows)
        for index in _indices(len(rows), per_split):
            row = rows[index]
            history = tuple(by_open[row.decision_at-(8-i)*STEP] for i in range(8))
            future = tuple(by_open[row.decision_at+i*STEP] for i in range(3))
            session = data.config.calendar.session_at(history[0].open_time)
            if session is None:
                raise ValueError('missing research session')
            session_data = encode(session)
            _session(history,future,session_data)
            window = window_payload(history, decision_at=row.decision_at)
            key = image_key(history, decision_at=row.decision_at)
            png = render_candles(history, decision_at=row.decision_at)
            if _cached_png(cache,key,png):
                hits += 1
            else:
                misses += 1
            name = 'images/'+key[7:]+'.png'
            write(name,png)
            sample_id = _digest(json_bytes({'split':split,'decision_at':encode(row.decision_at),
                                           'image_key':key}))
            samples.append({'sample_id':sample_id,'split':split,'decision_at':encode(row.decision_at),
                            'image_key':key,'image_path':name,'image_sha256':_digest(png),'window':window})
            labels.append({'sample_id':sample_id, **{f:encode(getattr(row,f)) for f in
                ('entry_at','exit_at','target_available_at','target_return')},
                'future':[_bar(c) for c in future]})
            sessions[sample_id] = session_data
    write('samples.json',json_bytes({'format':SAMPLES_FORMAT,'samples':samples}))
    write('labels.json',json_bytes({'format':LABELS_FORMAT,'labels':labels}))
    body = {'format':FORMAT,'recipe_id':recipe['recipe_id'],
            'source_dataset_id':data.dataset_id,'source_dataset_artifact_id':stored.manifest.artifact_id,
            'source_training_data_id':training['training_data_id'], 'policy':POLICY,
            'per_split':per_split,'split_counts':split_counts,'sessions':sessions,'artifacts':artifacts}
    manifest = {**body,'image_dataset_id':_digest(json_bytes(body))}
    (stage_files/'manifest.json').write_bytes(json_bytes(manifest))
    _verify_at(stage_fd)
    _publish_directory(stage,output,parent_fd=output_parent,stage_fd=stage_fd,container_fd=container_fd)
    return {'manifest':manifest,'cache_hits':hits,'cache_misses':misses,'sample_count':len(samples)}


@contextmanager
def _staging(parent_fd):
    """Keep the rename source entry inside a private, descriptor-pinned holder.

    A writer of outer entries can rename/replace the holder but cannot redirect
    its pinned child. This is not a sandbox against root or hostile same-UID code.
    """
    stage = Path(tempfile.mkdtemp(prefix='.image-dataset-', dir=f'/proc/self/fd/{parent_fd}'))
    container_fd = None
    stage_fd = None
    private = False
    child_created = False
    failed = False
    try:
        container_fd = os.open(stage.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        _private(container_fd)
        private = True
        os.mkdir('bundle', mode=0o700, dir_fd=container_fd)
        child_created = True
        stage_fd = os.open('bundle', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=container_fd)
        yield stage, container_fd, stage_fd
    except BaseException:
        failed = True
        raise
    finally:
        try:
            if container_fd is None:
                # Acquisition failed before any contents could be written.
                # Nonrecursive removal cannot destroy a replacement's contents.
                os.rmdir(stage.name, dir_fd=parent_fd)
            elif private:
                if stage_fd is not None:
                    try:
                        remaining = os.stat('bundle', dir_fd=container_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        if os.path.samestat(remaining, os.fstat(stage_fd)):
                            _clear_owned_directory(stage_fd)
                            os.rmdir('bundle', dir_fd=container_fd)
                elif child_created:
                    # mkdir succeeded, but open failed (e.g. EMFILE). The empty
                    # child is ours even though no child descriptor was obtained.
                    os.rmdir('bundle', dir_fd=container_fd)
                try:
                    remaining = os.stat(stage.name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if os.path.samestat(remaining, os.fstat(container_fd)):
                        os.rmdir(stage.name, dir_fd=parent_fd)
        except OSError:
            if not failed:
                raise
        finally:
            if stage_fd is not None:
                os.close(stage_fd)
            if container_fd is not None:
                os.close(container_fd)


@contextmanager
def _directory(path):
    """Pin a directory without following any ancestor symlinks."""
    parts = Path(path).absolute().parts
    if '..' in parts:
        raise ValueError('parent traversal is forbidden')
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _private(descriptor, *, output_parent=False):
    info = os.fstat(descriptor)
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        if output_parent:
            raise ValueError('output parent must be owned by current user and private mode0700 '
                             '(no group/other permissions); trusted owner required; '
                             'malicious same-UID writers are outside the contract')
        raise ValueError('cache must be private and owned by current user')


def _cached_png(cache, key, png):
    from market_trader.storage import publish_new_file
    name = _identity(key)[7:]+'.png'
    with _directory(cache) as descriptor:
        _private(descriptor)
        try:
            cached = _read_at(descriptor,name,limit=200_000,private=True)
        except FileNotFoundError:
            # Pin the private directory across exclusive, atomic publication.
            publish_new_file(Path('/proc/self/fd')/str(descriptor)/name,png)
            cached = _read_at(descriptor,name,limit=200_000,private=True)
            if cached != png:
                raise ValueError('cache publication verification failed')
            return False
        if cached != png:
            raise ValueError('corrupt cache image; refusing repair')
        return True


def _read_at(descriptor, name, *, limit=2_000_000, private=False):
    if '/' in name:
        if not re.fullmatch(r'images/[0-9a-f]{64}\.png', name):
            raise ValueError('invalid artifact path')
        child = os.open('images', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor)
        try:
            return _read_at(child,name[7:],limit=200_000)
        finally:
            os.close(child)
    if name in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]+',name):
        raise ValueError('invalid artifact path')
    file = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    try:
        info = os.fstat(file)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError('artifact must be a bounded, unlinked regular file')
        if private:
            _private(file)
        chunks, remaining = [], limit+1
        while remaining:
            chunk = os.read(file,min(65536,remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if not remaining:
            raise ValueError('artifact exceeds byte budget')
        return b''.join(chunks)
    finally:
        os.close(file)


def verify_image_dataset(output: Path) -> dict:
    """Verify internal semantics and regenerate PNGs/labels without source files."""
    with _directory(output) as descriptor:
        return _verify_at(descriptor)


def _verify_at(descriptor):
    from market_trader.chart_images import chart_recipe, window_payload, image_key, render_candles
    def read(name):
        return _read_at(descriptor,name)
    manifest = record_object(parse_json(read('manifest.json').decode('utf-8')),
        ('format','recipe_id','source_dataset_id','source_dataset_artifact_id',
         'source_training_data_id','policy','per_split','split_counts','sessions','artifacts','image_dataset_id'))
    budget = manifest['per_split']
    if type(budget) is not int or not 1 <= budget <= 64:
        raise ValueError('invalid per_split budget')
    counts = record_object(manifest['split_counts'],('train','validation'))
    for count in counts.values():
        if type(count) is not int or count < budget:
            raise ValueError('invalid split row counts')
    body = {k:v for k,v in manifest.items() if k != 'image_dataset_id'}
    if manifest['format'] != FORMAT or manifest['image_dataset_id'] != _digest(json_bytes(body)):
        raise ValueError('manifest identity mismatch')
    if json_bytes(manifest['policy']) != json_bytes(POLICY):
        raise ValueError('fixed image policy mismatch')
    for field in ('source_dataset_id','source_dataset_artifact_id','source_training_data_id'):
        _identity(manifest[field])
    inventory = manifest['artifacts']
    if (not isinstance(inventory,dict) or len(inventory) > 131
            or not {'recipe.json','samples.json','labels.json'} <= set(inventory)
            or any(name not in ('recipe.json','samples.json','labels.json')
                   and not re.fullmatch(r'images/[0-9a-f]{64}\.png',name) for name in inventory)):
        raise ValueError('invalid artifact inventory')
    if set(os.listdir(descriptor)) != {'manifest.json','recipe.json','samples.json','labels.json','images'}:
        raise ValueError('missing or extra bundle artifacts')
    images_fd = os.open('images',os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,dir_fd=descriptor)
    try:
        if set(os.listdir(images_fd)) != {name[7:] for name in inventory if name.startswith('images/')}:
            raise ValueError('missing or extra images')
    finally:
        os.close(images_fd)
    for name, digest in inventory.items():
        if _digest(read(name)) != _identity(digest):
            raise ValueError('artifact checksum mismatch')
    recipe = parse_json(read('recipe.json').decode('utf-8'))
    if json_bytes(recipe) != json_bytes(chart_recipe()) or manifest['recipe_id'] != recipe['recipe_id']:
        raise ValueError('recipe mismatch')
    sample_doc = record_object(parse_json(read('samples.json').decode('utf-8')),('format','samples'))
    label_doc = record_object(parse_json(read('labels.json').decode('utf-8')),('format','labels'))
    if sample_doc['format'] != SAMPLES_FORMAT or label_doc['format'] != LABELS_FORMAT:
        raise ValueError('sample/label format mismatch')
    samples, labels = sample_doc['samples'], label_doc['labels']
    if (not isinstance(samples,list) or not isinstance(labels,list)
            or len(samples) != len(labels) or len(samples) != 2*budget):
        raise ValueError('sample/label count mismatch')
    if not isinstance(manifest['sessions'],dict):
        raise ValueError('expected session mapping')
    seen, image_paths, previous = set(), set(), {}
    # At most 128*(8+3) occurrences; source-volume is deliberately not consumed.
    candle_evidence = {}
    for ordinal,(sample,label) in enumerate(zip(samples,labels,strict=True)):
        record_object(sample,('sample_id','split','decision_at','image_key','image_path','image_sha256','window'))
        record_object(label,('sample_id','entry_at','exit_at','target_available_at','target_return','future'))
        for field in ('sample_id','image_key','image_sha256'):
            _identity(sample[field])
        if not isinstance(sample['image_path'],str):
            raise ValueError('image_path must be a string')
        split = 'train' if ordinal < budget else 'validation'
        decision = _time(sample['decision_at'])
        start, cutoff = (START,VALIDATION) if split == 'train' else (VALIDATION,END)
        if (sample['split'] != split or not start <= decision < cutoff
                or (split in previous and decision <= previous[split])):
            raise ValueError('sample split or chronological ordering mismatch')
        previous[split] = decision
        expected_id = _digest(json_bytes({'split':split,'decision_at':encode(decision),
                                         'image_key':sample['image_key']}))
        if sample['sample_id'] != expected_id or sample['sample_id'] in seen:
            raise ValueError('sample identity or uniqueness mismatch')
        seen.add(sample['sample_id'])
        image_paths.add(sample['image_path'])
        window = record_object(sample['window'],('instrument','interval_microseconds','decision_at','candles'))
        bars = _candles(window['candles'])
        decision = _time(sample['decision_at'])
        if window != window_payload(bars, decision_at=decision):
            raise ValueError('window metadata mismatch')
        key = image_key(bars,decision_at=decision)
        png = render_candles(bars,decision_at=decision)
        if (sample['image_key'] != key or sample['image_path'] != 'images/'+key[7:]+'.png'
                or sample['image_sha256'] != _digest(png)
                or read(sample['image_path']) != png):
            raise ValueError('image regeneration mismatch')
        future = _candles(label['future'])
        if (len(future) != 3 or future[0].open_time != decision
                or future[-1].open_time != decision+2*STEP
                or label['future'] != [_bar(c) for c in future]):
            raise ValueError('target future alignment or canonical metadata mismatch')
        for candle in (*bars, *future):
            identity = (candle.instrument, candle.interval, candle.open_time)
            evidence = _bar(candle)
            if identity in candle_evidence and candle_evidence[identity] != evidence:
                raise ValueError('inconsistent repeated candle evidence')
            candle_evidence[identity] = evidence
        if sample['sample_id'] not in manifest['sessions']:
            raise ValueError('missing sample session')
        _session(bars,future,manifest['sessions'][sample['sample_id']])
        # Preserve the frozen training target's two rounded Decimal operations.
        with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=MIN_EMIN, Emax=MAX_EMAX)):
            result = (future[-1].open-future[0].open)/future[0].open
        if (label['sample_id'] != sample['sample_id'] or not isinstance(label['target_return'],str)
                or label['target_return'] != str(result)
                or _time(label['entry_at']) != decision
                or _time(label['exit_at']) != decision+2*STEP
                or _time(label['target_available_at']) != max(c.available_at for c in future)
                or _time(label['target_available_at']) >= cutoff):
            raise ValueError('label regeneration mismatch')
    if (set(manifest['sessions']) != seen
            or set(inventory) != {'recipe.json','samples.json','labels.json',*image_paths}):
        raise ValueError('unreferenced or missing artifacts/session metadata')
    return manifest
