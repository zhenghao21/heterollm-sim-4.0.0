"""Read-only identity diagnostics: independent OpenSSL and Windows CNG SHA256."""
import _hashlib
import ctypes as C
import datetime
import json
import os
import ssl
import sys
from pathlib import Path

MAX_CHUNK = 1024 * 1024
PROVIDER = 'Microsoft Primitive Provider'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def openssl_sha256():
    value = _hashlib.openssl_sha256()
    if type(value).__module__ != '_hashlib':
        raise RuntimeError('Independent OpenSSL implementation unavailable; no fallback')
    return value


class CNGSHA256:
    """BCrypt SHA256 with explicit Microsoft Primitive Provider, no OpenSSL fallback."""
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('Windows BCrypt/CNG unavailable; no substitute permitted')
        self.dll_path = str(Path(os.environ.get('SystemRoot', r'C:\Windows'))/'System32/bcrypt.dll')
        self.dll = C.WinDLL(self.dll_path)
        self.alg = C.c_void_p()
        self.handle = C.c_void_p()
        self.finished = False
        signatures = {
            'BCryptOpenAlgorithmProvider': [C.POINTER(C.c_void_p), C.c_wchar_p, C.c_wchar_p, C.c_uint32],
            'BCryptGetProperty': [C.c_void_p, C.c_wchar_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_uint32), C.c_uint32],
            'BCryptCreateHash': [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32, C.c_uint32],
            'BCryptHashData': [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32],
            'BCryptFinishHash': [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32],
            'BCryptDestroyHash': [C.c_void_p],
            'BCryptCloseAlgorithmProvider': [C.c_void_p, C.c_uint32],
        }
        for name, args in signatures.items():
            fn = getattr(self.dll, name); fn.argtypes = args; fn.restype = C.c_int32
        try:
            self.check(self.dll.BCryptOpenAlgorithmProvider(C.byref(self.alg), 'SHA256', PROVIDER, 0), 'open')
            size, written = C.c_uint32(), C.c_uint32()
            self.check(self.dll.BCryptGetProperty(self.alg, 'ObjectLength', C.byref(size), 4, C.byref(written), 0), 'object length')
            if written.value != 4 or size.value <= 0:
                raise RuntimeError('Invalid CNG hash object size')
            self.object = C.create_string_buffer(size.value)
            self.check(self.dll.BCryptCreateHash(self.alg, C.byref(self.handle), self.object, size.value, None, 0, 0), 'create')
        except BaseException:
            self.close(); raise

    @staticmethod
    def check(status, operation):
        if status != 0:
            raise RuntimeError('BCrypt '+operation+' failed NTSTATUS=0x%08x' % (status & 0xffffffff))

    def update(self, data):
        if self.finished or not isinstance(data, bytes) or len(data) > MAX_CHUNK:
            raise ValueError('CNG requires live hash and immutable bytes <=1MiB')
        # c_char_p points at this exact Python bytes object, including embedded NUL.
        # Explicit length prevents C-string truncation; no transformed/re-encoded copy.
        pointer = C.c_char_p(data)
        self.check(self.dll.BCryptHashData(self.handle, pointer, len(data), 0), 'update')

    def hexdigest(self):
        if self.finished:
            raise ValueError('CNG finalized already')
        output = (C.c_ubyte * 32)()
        self.check(self.dll.BCryptFinishHash(self.handle, output, 32, 0), 'finish')
        self.finished = True
        return bytes(output).hex()

    def close(self):
        if getattr(self, 'handle', None) and self.handle.value:
            self.dll.BCryptDestroyHash(self.handle); self.handle = C.c_void_p()
        if getattr(self, 'alg', None) and self.alg.value:
            self.dll.BCryptCloseAlgorithmProvider(self.alg, 0); self.alg = C.c_void_p()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def stat_record(st):
    return {k: getattr(st, k, None) for k in
            ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_atime_ns', 'st_file_attributes')}


def same_file_stat(a, b):
    # Access time may legitimately change during this read-only experiment.
    return all(a[k] == b[k] for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def write_new(path, payload):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def self_test(include_large=False):
    vectors = [
        (b'', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
        (b'abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'),
    ]
    if include_large:
        vectors.append((b'a'*1000000, 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0'))
    for data, expected in vectors:
        left = openssl_sha256()
        with CNGSHA256() as right:
            for offset in range(0, len(data), MAX_CHUNK):
                block = data[offset:offset+MAX_CHUNK]
                left.update(block); right.update(block)
            a, b = left.hexdigest(), right.hexdigest()
            if a != expected or b != expected or a != b:
                raise RuntimeError('Independent implementations failed a known vector')
    return {'known_vectors_passed': len(vectors), 'openssl_version': ssl.OPENSSL_VERSION,
            'openssl_module': _hashlib.__file__, 'CNG_provider': PROVIDER,
            'independent_algorithms': ['_hashlib.openssl_sha256', 'BCrypt SHA256'],
            'model_file_reads': 0, 'million_a_control': include_large, 'maximum_update_bytes':1000000 if include_large else 3, 'fallback_used': False}


def scan_file(path, output, expected_size, expected_sha256, *, chunk_bytes=MAX_CHUNK,
              cng_factory=CNGSHA256, openssl_factory=openssl_sha256):
    """One file open, each returned immutable chunk feeds both engines and two chunk digests.
    Disagreements are retained and rejected. No reread, repair, alternate winner or retry.
    """
    if not 1 <= chunk_bytes <= MAX_CHUNK or expected_size < 0:
        raise ValueError('Invalid bounded read geometry')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    path = Path(path)
    result = {'schema':'dual-sha256-file-pass/v1', 'path':str(path.resolve()), 'pid':os.getpid(),
              'started_utc':now(), 'status':'rejected', 'bytes_read':0, 'chunk_count':0,
              'chunk_bytes':chunk_bytes, 'expected_size':expected_size, 'expected_sha256':expected_sha256,
              'chunk_digest_disagreements':[], 'full_openssl_sha256':None, 'full_cng_sha256':None,
              'read_only':True, 'extra_EOF_probe_bytes':0, 'file_open_count':0,
              'stat_unchanged':False, 'errors':[], 'algorithm_winner_selected':False,
              'model_latency_measured':False, 'cost_parameters':0}
    try:
        # Instantiate both algorithms before touching model content.
        left = openssl_factory()
        with cng_factory() as right, (output/'chunks.jsonl').open('x', encoding='utf-8') as log:
            result['path_stat_before'] = stat_record(path.stat())
            if result['path_stat_before']['st_size'] != expected_size:
                raise ValueError('Size differs before read; zero model bytes read')
            with path.open('rb', buffering=0) as stream:
                result['file_open_count'] = 1
                result['handle_stat_before'] = stat_record(os.fstat(stream.fileno()))
                if not same_file_stat(result['path_stat_before'], result['handle_stat_before']):
                    raise ValueError('Path and open handle identify different file')
                while result['bytes_read'] < expected_size:
                    offset = result['bytes_read']
                    wanted = min(chunk_bytes, expected_size-offset)
                    data = stream.read(wanted)
                    if not data:
                        raise EOFError('Early EOF; no retry')
                    result['bytes_read'] += len(data)
                    left.update(data); right.update(data)
                    chunk_left = openssl_factory(); chunk_left.update(data)
                    with cng_factory() as chunk_right:
                        chunk_right.update(data)
                        a, b = chunk_left.hexdigest(), chunk_right.hexdigest()
                    identical = a == b
                    if not identical:
                        result['chunk_digest_disagreements'].append(result['chunk_count'])
                    log.write(json.dumps({'index':result['chunk_count'], 'offset':offset, 'bytes':len(data),
                                          'openssl_sha256':a, 'cng_sha256':b, 'equal':identical})+'\n')
                    log.flush()
                    result['chunk_count'] += 1
                    if len(data) != wanted:
                        raise IOError('Short read retained; stop without retry')
                result['handle_stat_after'] = stat_record(os.fstat(stream.fileno()))
            result['path_stat_after'] = stat_record(path.stat())
            result['full_openssl_sha256'] = left.hexdigest()
            result['full_cng_sha256'] = right.hexdigest()
            result['stat_unchanged'] = all(same_file_stat(result['path_stat_before'], result[k]) for k in
                                       ('handle_stat_before','handle_stat_after','path_stat_after'))
            if not result['stat_unchanged']:
                result['errors'].append('file identity/size/mtime/ctime changed')
            if result['chunk_digest_disagreements']:
                result['errors'].append('chunk algorithms disagree; neither implementation accepted')
            if result['full_openssl_sha256'] != result['full_cng_sha256']:
                result['errors'].append('full algorithms disagree; neither implementation accepted')
            if result['full_openssl_sha256'] != expected_sha256 or result['full_cng_sha256'] != expected_sha256:
                result['errors'].append('expected identity mismatch; no winning implementation selected')
            if not result['errors'] and result['bytes_read'] == expected_size:
                result['status'] = 'agreed_expected_identity'
    except BaseException as error:
        result['errors'].append(type(error).__name__+': '+str(error))
        # Preserve the path observation even when full hashing could not complete.
        try: result['path_stat_after'] = stat_record(path.stat())
        except OSError: pass
    finally:
        result['finished_utc'] = now()
        write_new(output/'finish.json', result)
    return result
