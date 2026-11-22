"""AP7 exact, bounded stdlib preprocessing (synthetic PNG only)."""
import hashlib
import struct
import zlib

import pytest

from market_trader.chart_images import _png, _chunk
from market_trader._dataset_codec import json_bytes


def api():
    from market_trader import image_pixels_v1
    return image_pixels_v1


def rgb_fixture():
    return b''.join(b'\0'+bytes(v for x in range(256) for v in (x,y,(x+y)%256)) for y in range(192))


def test_exact_chw_integer_sums_normalization_and_source_recipe():
    sums=api().block_sums(_png(rgb_fixture()))
    expected=tuple(sum(((x if c==0 else y if c==1 else (x+y)%256))
                       for y in range(by*8,by*8+8) for x in range(bx*8,bx*8+8))
                   for c in range(3) for by in range(24) for bx in range(32))
    assert sums==expected and len(sums)==2304
    assert all(type(x) is int for x in sums)
    assert api().normalized_input(sums)==tuple(x/64/255 for x in expected)
    from pathlib import Path
    recipe=api().image_input_recipe()
    assert recipe['implementation_sha256']==hashlib.sha256(Path(api().__file__).read_bytes()).hexdigest()
    assert recipe['recipe_id']=='sha256:'+hashlib.sha256(json_bytes({k:v for k,v in recipe.items() if k!='recipe_id'})).hexdigest()
    assert recipe['input_shape']==[3,24,32]


@pytest.mark.parametrize('fault', ['signature','crc','dimensions','metadata','trailing','truncated',
                                  'compressed','adler','filter','deflate','oversize','bytearray'])
def test_decoder_rejects_noncanonical_and_corrupt_pngs(fault):
    raw = rgb_fixture()
    png = _png(raw)
    if fault == 'signature': png = b'bad'+png[3:]
    elif fault == 'crc': png = png[:29]+bytes([png[29]^1])+png[30:]
    elif fault == 'dimensions':
        png = png[:8]+_chunk(b'IHDR',struct.pack('>IIBBBBB',255,192,8,2,0,0,0))+png[33:]
    elif fault == 'metadata': png = png[:33]+_chunk(b'tEXt',b'no labels')+png[33:]
    elif fault == 'trailing': png += b'\0'
    elif fault == 'truncated': png = png[:-3]
    elif fault == 'compressed':
        png = png[:33]+_chunk(b'IDAT',zlib.compress(raw))+_chunk(b'IEND',b'')
    elif fault == 'adler':
        stream = png[41:-16]
        png = png[:33]+_chunk(b'IDAT',stream[:-1]+bytes([stream[-1]^1]))+_chunk(b'IEND',b'')
    elif fault == 'filter': png = _png(b'\1'+raw[1:])
    elif fault == 'deflate':
        stream = png[41:-16]
        png = png[:33]+_chunk(b'IDAT',stream[:2]+b'\7'+stream[3:])+_chunk(b'IEND',b'')
    elif fault == 'oversize': png += b'x'*200000
    elif fault == 'bytearray': png = bytearray(png)
    with pytest.raises(ValueError): api().block_sums(png)


@pytest.mark.parametrize('bad', [(), (0,)*2303, (0,)*2305, (True,)+(0,)*2303,
                                (1.0,)+(0,)*2303, (-1,)+(0,)*2303, (16321,)+(0,)*2303])
def test_normalization_rejects_coercions_and_invalid_sums(bad):
    with pytest.raises(ValueError): api().normalized_input(bad)


def test_explicit_numpy_accelerator_matches_exact_python_and_default_is_framework_free():
    pytest.importorskip('numpy')
    png = _png(rgb_fixture())
    assert api().block_sums(png, accelerated=True) == api().block_sums(png)
    assert api().normalized_input((16320,)*2304) == (1.0,)*2304
    for bad in (1, 'yes', None):
        with pytest.raises(ValueError): api().block_sums(png, accelerated=bad)


def test_default_decoder_and_recipe_import_without_numpy_or_torch():
    import subprocess
    import sys
    script = '''
import sys
class NoFramework:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch'):
            raise AssertionError('unexpected framework import')
sys.meta_path.insert(0, NoFramework())
from market_trader.image_pixels_v1 import block_sums, image_input_recipe
from market_trader.chart_images import _png
assert len(block_sums(_png((b'\\0'+b'\\xff'*768)*192))) == 2304
assert image_input_recipe()['max_sum'] == 16320
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=20)
