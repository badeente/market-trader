"""Frozen input-only AP6 PNG -> exact CHW block sums; stdlib by default."""
from hashlib import sha256
from pathlib import Path
import struct
import zlib

from market_trader._dataset_codec import json_bytes
from market_trader.chart_images import chart_recipe

SIZE = 2304
MAX_SUM = 16320
_RAW_SIZE = 192 * (1 + 256 * 3)


def image_input_recipe() -> dict:
    body = {'format':'market-trader.image-input-recipe.v1',
            'ap6_recipe_id':chart_recipe()['recipe_id'],
            'implementation_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'source':{'format':'PNG','width':256,'height':192,'color':'RGB8'},
            'block_shape':[8,8], 'reduction':'exact integer block sums',
            'input_shape':[3,24,32], 'layout':'CHW', 'storage_dtype':'<u2',
            'max_sum':MAX_SUM, 'normalization':'float64 sum/64/255',
            'fitted_statistics':False}
    return {**body, 'recipe_id':'sha256:'+sha256(json_bytes(body)).hexdigest()}


def _decode(png: bytes) -> bytes:
    if type(png) is not bytes or len(png) > 200_000 or not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('expected bounded canonical AP6 PNG')
    offset, payloads = 8, []
    for kind in (b'IHDR', b'IDAT', b'IEND'):
        if offset + 12 > len(png):
            raise ValueError('truncated PNG chunk')
        length = struct.unpack_from('>I',png,offset)[0]
        end = offset + 12 + length
        if end > len(png) or png[offset+4:offset+8] != kind:
            raise ValueError('invalid canonical PNG chunks')
        payload = png[offset+8:end-4]
        if zlib.crc32(kind+payload) != struct.unpack_from('>I',png,end-4)[0]:
            raise ValueError('PNG CRC mismatch')
        payloads.append(payload)
        offset = end
    if (offset != len(png) or payloads[0] != struct.pack('>IIBBBBB',256,192,8,2,0,0,0)
            or payloads[2]):
        raise ValueError('invalid PNG dimensions, metadata or trailing bytes')
    stream = payloads[1]
    # Parse the renderer's stored-block encoding explicitly. No general-purpose
    # decompressor can allocate from attacker-controlled expansion ratios here.
    if stream[:2] != b'\x78\x01':
        raise ValueError('noncanonical AP6 zlib stream')
    raw, offset = bytearray(), 2
    while len(raw) < _RAW_SIZE:
        remaining = _RAW_SIZE-len(raw)
        length = min(remaining,65535)
        final = int(remaining <= 65535)
        header = bytes([final])+struct.pack('<HH',length,length ^ 65535)
        if stream[offset:offset+5] != header or offset+5+length > len(stream):
            raise ValueError('noncanonical or truncated AP6 stored block')
        raw.extend(stream[offset+5:offset+5+length])
        offset += 5+length
    if (len(stream) != offset+4 or zlib.adler32(raw) != struct.unpack_from('>I',stream,offset)[0]
            or any(raw[y*769] != 0 for y in range(192))):
        raise ValueError('invalid PNG Adler, length or scanline filter')
    return bytes(raw)


def block_sums(png: bytes, *, accelerated: bool = False) -> tuple[int, ...]:
    if type(accelerated) is not bool:
        raise ValueError('accelerated must be bool')
    raw = _decode(png)
    if accelerated:
        import numpy as np
        rgb = np.frombuffer(raw,dtype=np.uint8).reshape(192,769)[:,1:].reshape(24,8,32,8,3)
        return tuple(int(v) for v in rgb.sum(axis=(1,3),dtype=np.uint16).transpose(2,0,1).ravel())
    result = [0]*SIZE
    for y in range(192):
        base = y*769+1
        row = (y//8)*32
        for x in range(256):
            cell = row+x//8
            offset = base+x*3
            result[cell] += raw[offset]
            result[768+cell] += raw[offset+1]
            result[1536+cell] += raw[offset+2]
    return tuple(result)


def normalized_input(sums) -> tuple[float, ...]:
    if not isinstance(sums,(tuple,list)) or len(sums) != SIZE or any(
            type(v) is not int or not 0 <= v <= MAX_SUM for v in sums):
        raise ValueError('expected exactly2304 integer block sums in [0,16320]')
    return tuple(v/64/255 for v in sums)
