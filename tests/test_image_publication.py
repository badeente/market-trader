"""Defense-in-depth race probes; hostile same-UID writers are not a sandbox contract."""
import ctypes
import errno
import os
from pathlib import Path
import pytest
from tests.test_image_dataset import sources
from market_trader import image_dataset as api


def test_wrong_owner_output_parent_rejected_before_side_effects(tmp_path, monkeypatch):
    from types import SimpleNamespace
    dataset, training = sources(tmp_path)
    parent = tmp_path/'parent'
    parent.mkdir(mode=0o700)
    identity = parent.stat()
    real_fstat = os.fstat
    def metadata(descriptor):
        info = real_fstat(descriptor)
        if os.path.samestat(info, identity):
            return SimpleNamespace(st_uid=os.getuid()+1, st_mode=info.st_mode)
        return info
    monkeypatch.setattr(api.os, 'fstat', metadata)
    with pytest.raises(ValueError, match='output parent.*current user.*0700'):
        api.build_image_dataset(dataset, training, parent/'out', cache=tmp_path/'cache', per_split=1)
    assert list(parent.iterdir()) == []
    assert not (tmp_path/'cache').exists()


def test_trusted_private_output_parent_accepted(tmp_path):
    dataset, training = sources(tmp_path)
    parent = tmp_path/'parent'
    parent.mkdir(mode=0o700)
    result = api.build_image_dataset(dataset, training, parent/'out', cache=tmp_path/'cache', per_split=1)
    assert api.verify_image_dataset(parent/'out') == result['manifest']
    assert sorted(p.name for p in parent.iterdir()) == ['out']


def test_rollback_recursion_uses_owned_bundle_descriptor_not_output_name(tmp_path, monkeypatch):
    dataset, training = sources(tmp_path)
    parent = tmp_path/'parent'
    parent.mkdir(mode=0o700)
    moved = tmp_path/'moved'
    real_cdll = ctypes.CDLL
    real_rmtree = api.shutil.rmtree
    visited = []
    def library(*args, **kwargs):
        real = real_cdll(*args, **kwargs)
        class Proxy:
            def renameat2(self, source_fd, source, destination_fd, destination, flags):
                result = real.renameat2(source_fd, source, destination_fd, destination, flags)
                assert result == 0
                parent.rename(moved)
                parent.mkdir(mode=0o700)
                return result
        return Proxy()
    def owned_recursion(path, *args, **kwargs):
        assert path != 'out', 'rollback must not recursively delete mutable outer output name'
        descriptor = kwargs['dir_fd']
        assert os.path.samestat(os.fstat(descriptor), (moved/'out').stat())
        visited.append(path)
        return real_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(api.ctypes, 'CDLL', library)
    monkeypatch.setattr(api.shutil, 'rmtree', owned_recursion)
    with pytest.raises(ValueError, match='output parent changed'):
        api.build_image_dataset(dataset, training, parent/'out', cache=tmp_path/'cache', per_split=1)
    assert visited == ['images']
    assert list(parent.iterdir()) == list(moved.iterdir()) == []


@pytest.mark.parametrize('where', ['holder', 'bundle'])
def test_staging_acquisition_emfile_preserves_original_error_and_leaves_nothing(tmp_path, monkeypatch, where):
    dataset, training = sources(tmp_path)
    parent = tmp_path/'parent'
    parent.mkdir(mode=0o700)
    real_open = os.open
    initiating = OSError(errno.EMFILE, 'injected staging acquisition exhaustion')
    fired = []
    def exhaust(path, flags, *args, **kwargs):
        if ((where == 'bundle' and path == 'bundle') or
                (where == 'holder' and str(path).startswith('.image-dataset-'))):
            fired.append(path)
            raise initiating
        return real_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(api.os, 'open', exhaust)
    with pytest.raises(OSError) as caught:
        api.build_image_dataset(dataset, training, parent/'out', cache=tmp_path/'cache', per_split=1)
    assert fired
    assert caught.value is initiating
    assert caught.value.errno == errno.EMFILE
    assert list(parent.iterdir()) == []
    assert list((tmp_path/'cache').iterdir()) == []


@pytest.mark.parametrize('mode', [0o755, 0o777])
@pytest.mark.parametrize('existing_cache', [False, True])
def test_untrusted_writable_or_readable_output_parent_rejected_before_side_effects(tmp_path, mode, existing_cache):
    """Reject the old rollback probe's untrusted namespace, not sandbox its owner."""
    dataset, training = sources(tmp_path)
    parent = tmp_path/'parent'
    parent.mkdir(mode=mode)
    parent.chmod(mode)
    sentinel = parent/'sentinel'
    sentinel.write_bytes(b'foreign data')
    cache = tmp_path/'cache'
    if existing_cache:
        cache.mkdir(mode=0o700)
        (cache/'sentinel').write_bytes(b'unchanged cache')
    before = {str(p.relative_to(tmp_path)): p.read_bytes()
              for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='output parent.*current user.*0700.*trusted.*same-UID'):
        api.build_image_dataset(dataset, training, parent/'out', cache=cache, per_split=1)
    assert sorted(p.name for p in parent.iterdir()) == ['sentinel']
    assert cache.exists() == existing_cache
    assert {str(p.relative_to(tmp_path)): p.read_bytes()
            for p in tmp_path.rglob('*') if p.is_file()} == before


def test_outer_stage_name_swap_at_syscall_cannot_redirect_verified_source(tmp_path,monkeypatch):
    dataset,training=sources(tmp_path)
    parent=tmp_path/'parent';parent.mkdir(mode=0o700)
    output=parent/'out'
    real_cdll=ctypes.CDLL
    fired=[]
    def library(*args,**kwargs):
        real=real_cdll(*args,**kwargs)
        class Proxy:
            def renameat2(self,source_fd,source,destination_fd,destination,flags):
                holder=next(name for name in os.listdir(destination_fd) if name.startswith('.image-dataset-'))
                os.rename(holder,'displaced-verified-stage',src_dir_fd=destination_fd,dst_dir_fd=destination_fd)
                os.mkdir(holder,mode=0o700,dir_fd=destination_fd)
                fake=parent/holder
                (fake/'sentinel').write_bytes(b'UNVERIFIED ATTACKER STAGE')
                (fake/'bundle').mkdir()
                (fake/'bundle'/'sentinel').write_bytes(b'UNVERIFIED ATTACKER STAGE')
                fired.append((source_fd,destination_fd,holder))
                return real.renameat2(source_fd,source,destination_fd,destination,flags)
        return Proxy()
    monkeypatch.setattr(api.ctypes,'CDLL',library)
    result=api.build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    assert fired and fired[0][0]!=fired[0][1]
    assert api.verify_image_dataset(output)==result['manifest']
    assert not (output/'sentinel').exists()
    assert list((parent/'displaced-verified-stage').iterdir())==[]
    assert (parent/fired[0][2]/'sentinel').read_bytes()==b'UNVERIFIED ATTACKER STAGE'


@pytest.mark.parametrize('kind',['symlink','directory'])
@pytest.mark.parametrize('when',['staging','before_syscall','after_syscall'])
def test_ancestor_replacement_is_rejected_and_owned_data_cleaned(tmp_path,monkeypatch,kind,when):
    dataset,training=sources(tmp_path)
    parent=tmp_path/'parent';parent.mkdir(mode=0o700)
    moved=tmp_path/'moved'
    attacker=tmp_path/'attacker';attacker.mkdir()
    output=parent/'out'
    fired=[]
    def swap():
        if fired:return
        fired.append(True)
        parent.rename(moved)
        if kind=='symlink':parent.symlink_to(attacker,target_is_directory=True)
        else:parent.mkdir(mode=0o700)
    real_cdll=ctypes.CDLL
    real_cache=api._cached_png
    def cached(*args,**kwargs):
        if when=='staging':swap()
        return real_cache(*args,**kwargs)
    def library(*args,**kwargs):
        real=real_cdll(*args,**kwargs)
        class Proxy:
            def renameat2(self,source_fd,source,destination_fd,destination,flags):
                assert source_fd!=destination_fd
                info=os.fstat(source_fd)
                assert info.st_uid==os.getuid() and info.st_mode&0o077==0
                if when=='before_syscall':swap()
                result=real.renameat2(source_fd,source,destination_fd,destination,flags)
                if when=='after_syscall':swap()
                return result
        return Proxy()
    monkeypatch.setattr(api,'_cached_png',cached)
    monkeypatch.setattr(api.ctypes,'CDLL',library)
    with pytest.raises((ValueError,OSError)):
        api.build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    assert fired
    assert not output.exists() and list(moved.iterdir())==[] and list(attacker.iterdir())==[]


@pytest.mark.parametrize('kind',['empty_directory','file','symlink'])
def test_competing_destination_at_syscall_is_preserved(tmp_path,monkeypatch,kind):
    dataset,training=sources(tmp_path)
    output=tmp_path/'out'
    sentinel=tmp_path/'untouched';sentinel.mkdir()
    real_cdll=ctypes.CDLL
    def library(*args,**kwargs):
        real=real_cdll(*args,**kwargs)
        class Proxy:
            def renameat2(self,source_fd,source,destination_fd,destination,flags):
                if kind=='empty_directory':output.mkdir()
                elif kind=='file':output.write_bytes(b'other writer')
                else:output.symlink_to(sentinel,target_is_directory=True)
                return real.renameat2(source_fd,source,destination_fd,destination,flags)
        return Proxy()
    monkeypatch.setattr(api.ctypes,'CDLL',library)
    with pytest.raises(FileExistsError):
        api.build_image_dataset(dataset,training,output,cache=tmp_path/'cache',per_split=1)
    assert not list(tmp_path.glob('.image-dataset-*'))
    if kind=='file':assert output.read_bytes()==b'other writer'
    elif kind=='symlink':assert output.is_symlink() and output.resolve()==sentinel
    else:assert output.is_dir() and list(output.iterdir())==[]
