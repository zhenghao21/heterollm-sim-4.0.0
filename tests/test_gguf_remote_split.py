from dataclasses import replace
import io
import re
import struct

import pytest

from heterollm_sim.gguf_parity import (
    GGUFError, _unique_tensor_bytes, build_model_from_gguf, merge_gguf_shards,
    parse_gguf_directory, read_gguf_metadata_only, read_gguf_metadata_cache,
    validate_gguf_inventory, write_gguf_metadata_cache, read_gguf_metadata,
)
from heterollm_sim.gguf_remote import read_remote_gguf_metadata


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def fixture(name='tensor', *, split=None, total=2, metadata=None):
    values = {'general.architecture': 'fixture', **(metadata or {})}
    if split is not None:
        values.update({'split.no': split, 'split.count': 2, 'split.tensors.count': total})
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(values))
    for key, value in values.items():
        data += string(key)
        data += struct.pack('<I', 8 if isinstance(value, str) else 4)
        data += string(value) if isinstance(value, str) else struct.pack('<I', value)
    data += string(name) + struct.pack('<IQIQ', 1, 8192, 0, 0)
    data += bytes((-len(data)) % 32)
    return data + bytes(8192 * 4)


def parsed(data, path):
    return parse_gguf_directory(io.BytesIO(data), path=path, file_size=len(data))


class Response:
    def __init__(self, body, start, end, size, *, status=206, content_range=None, content_length=None):
        self.status = status
        self.headers = {'Content-Range': content_range or f'bytes {start}-{end}/{size}',
                        'Content-Length': str(len(body)) if content_length is None else content_length}
        self.stream = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return self.stream.read(size)


def source(names):
    return {'repo': 'test/model', 'revision': 'a' * 40, 'files': [
        {'rfilename': name, 'size': len(data), 'lfs': {'size': len(data), 'sha256': 'b' * 64}}
        for name, data in names.items()
    ]}


def remote(names, **response_overrides):
    requests = []

    def opener(request, timeout):
        start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
        data = names[request.full_url.rsplit('/', 1)[1]]
        requests.append((start, end, len(data)))
        return Response(data[start:end + 1], start, end, len(data), **response_overrides)

    return opener, requests


def test_remote_directory_uses_bounded_ranges_and_preserves_publisher_identity():
    names = {'model.gguf': fixture()}
    opener, requests = remote(names)
    gguf = read_remote_gguf_metadata(source(names), opener=opener, read_ahead=256)
    validate_gguf_inventory(gguf)
    assert gguf.sha256 == 'b' * 64
    assert gguf.sources[0]['sha256_origin'] == 'publisher_lfs'
    assert gguf.tensor_directory[0].physical_offset == gguf.sources[0]['data_start']
    assert requests[0][:2] == (0, 23)
    assert sum(end - start + 1 for start, end, _ in requests) < len(names['model.gguf']) // 10
    assert all(end - start + 1 <= 256 for start, end, _ in requests)


def test_remote_and_local_share_directory_geometry(tmp_path):
    data = fixture()
    path = tmp_path / 'model.gguf'
    path.write_bytes(data)
    local = read_gguf_metadata_only(path)
    opener, _ = remote({'model.gguf': data})
    distant = read_remote_gguf_metadata(source({'model.gguf': data}), opener=opener)
    assert distant.metadata == local.metadata
    assert distant.tensor_count == local.tensor_count
    assert [replace(t, source_path='') for t in distant.tensor_directory] == [
        replace(t, source_path='') for t in local.tensor_directory]


def test_remote_refuses_short_body_and_wrong_publisher_total():
    names = {'model.gguf': fixture()}
    entry = source(names)
    data = names['model.gguf']
    def short(request, timeout):
        return Response(data[:23], 0, 23, len(data), content_length='24')
    with pytest.raises(GGUFError, match='body length'):
        read_remote_gguf_metadata(entry, opener=short)
    opener, _ = remote(names)
    entry['files'][0]['size'] += 32
    entry['files'][0]['lfs']['size'] += 32
    with pytest.raises(GGUFError, match='Content-Range'):
        read_remote_gguf_metadata(entry, opener=opener)


def test_parser_validates_payload_range_without_reading_weights():
    data = fixture()
    gguf = parsed(data, 'model.gguf')
    with pytest.raises(GGUFError, match='truncated GGUF tensor payload'):
        parse_gguf_directory(io.BytesIO(data), path='model.gguf',
                             file_size=gguf.sources[0]['data_start'] + 16)


@pytest.mark.parametrize('overrides', [
    {'status': 200}, {'content_range': 'bytes 1-24/1'}, {'content_length': '99'},
])
def test_remote_refuses_ignored_or_invalid_ranges(overrides):
    names = {'model.gguf': fixture()}
    opener, _ = remote(names, **overrides)
    with pytest.raises(GGUFError):
        read_remote_gguf_metadata(source(names), opener=opener)


def test_split_remote_same_offsets_are_distinct_physical_payloads():
    names = {'model-00001-of-00002.gguf': fixture('first', split=0),
             'model-00002-of-00002.gguf': fixture('second', split=1)}
    opener, _ = remote(names)
    gguf = read_remote_gguf_metadata(source(names), opener=opener)
    assert gguf.tensor_count == 2 and len(gguf.sources) == 2
    assert gguf.sha256 == ''
    assert _unique_tensor_bytes(gguf.tensor_directory) == 65536
    assert gguf.tensor_directory[0].offset == gguf.tensor_directory[1].offset == 0
    assert gguf.tensor_directory[0].source_path != gguf.tensor_directory[1].source_path


def test_local_split_auto_discovery_and_missing_file(tmp_path):
    first = tmp_path / 'model-00001-of-00002.gguf'
    second = tmp_path / 'model-00002-of-00002.gguf'
    first.write_bytes(fixture('first', split=0))
    with pytest.raises(GGUFError, match='missing GGUF shard'):
        read_gguf_metadata_only(first)
    second.write_bytes(fixture('second', split=1))
    gguf = read_gguf_metadata_only(second)
    assert [t.name for t in gguf.tensor_directory] == ['first', 'second']
    assert _unique_tensor_bytes(gguf.tensor_directory) == 65536
    hashed = read_gguf_metadata(second)
    assert hashed.sha256 == ''
    assert all(len(item['sha256']) == 64 for item in hashed.sources)
    with pytest.raises(GGUFError, match='caches are unsupported'):
        write_gguf_metadata_cache(first)


@pytest.mark.parametrize('case', ['missing', 'duplicate_index', 'duplicate_tensor', 'metadata', 'count'])
def test_split_merge_fail_closed(case):
    a = parsed(fixture('first', split=0), 'first.gguf')
    b = parsed(fixture('second', split=1), 'second.gguf')
    if case == 'missing':
        shards = [a]
    elif case == 'duplicate_index':
        shards = [a, replace(b, metadata={**b.metadata, 'split.no': 0})]
    elif case == 'duplicate_tensor':
        shards = [a, parsed(fixture('first', split=1), 'second.gguf')]
    elif case == 'metadata':
        shards = [a, parsed(fixture('second', split=1, metadata={'general.architecture': 'conflict'}), 'second.gguf')]
    else:
        shards = [a, parsed(fixture('second', split=1, total=3), 'second.gguf')]
    with pytest.raises(GGUFError):
        merge_gguf_shards(shards)


def test_serialized_first_shard_cannot_build_graph():
    a = parsed(fixture('first', split=0), 'first.gguf')
    with pytest.raises(GGUFError, match='incomplete GGUF split inventory'):
        build_model_from_gguf(a)


def test_serialized_inventory_rejects_tensor_without_physical_source():
    data = fixture()
    gguf = parsed(data, 'model.gguf')
    malformed = replace(gguf, tensor_directory=(replace(gguf.tensor_directory[0], source_path='elsewhere.gguf'),))
    with pytest.raises(GGUFError, match='physical tensor sources'):
        validate_gguf_inventory(malformed)


def test_single_file_cache_keeps_physical_inventory(tmp_path):
    path = tmp_path / 'model.gguf'
    path.write_bytes(fixture())
    write_gguf_metadata_cache(path)
    cached = read_gguf_metadata_cache(path, strict=True)
    assert cached.sources[0]['path'] == str(path)
    assert cached.tensor_directory[0].source_path == str(path)
    validate_gguf_inventory(cached)


def test_remote_refuses_floating_revision():
    names = {'model.gguf': fixture()}
    entry = source(names)
    entry['revision'] = 'main'
    with pytest.raises(GGUFError, match='fixed commit'):
        read_remote_gguf_metadata(entry)
