"""The geometry used by simulation must be bound to the exact hashed bytes."""
import io
import struct
from hashlib import sha256
from pathlib import Path

from heterollm_sim.gguf_parity import read_gguf_metadata


def text(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def fixture(*, layers=2, large_array=False, tensor=False, tail=b""):
    entries = [
        text("general.architecture") + struct.pack("<I", 8) + text("llama"),
        text("general.file_type") + struct.pack("<II", 4, 0),
        text("llama.block_count") + struct.pack("<II", 4, layers),
        text("llama.embedding_length") + struct.pack("<II", 4, 4),
        text("llama.attention.head_count") + struct.pack("<II", 4, 1),
    ]
    if large_array:
        entries.append(text("unused.numeric_array") + struct.pack("<IIQ", 9, 4, 4097)
                       + struct.pack("<" + "I" * 4097, *range(4097)))
    payload = b"GGUF" + struct.pack("<IQQ", 3, int(tensor), len(entries)) + b"".join(entries)
    if tensor:
        payload += text("token_embd.weight") + struct.pack("<IQQIQ", 2, 4, 2, 0, 0)
        payload += b"\x00" * (-len(payload) % 32)
        payload += struct.pack("<8f", *[float(i) for i in range(8)])
    return payload + tail


class TracedRead:
    """A same-handle/stat read that would return another header on rewind."""
    def __init__(self, handle, first, after_rewind=None):
        self.handle = handle
        self.first = first
        self.after_rewind = after_rewind
        self.stream = io.BytesIO(first)
        self.total_read = 0
        self.rewinds = []

    def __enter__(self):
        self.handle.__enter__()
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def fileno(self): return self.handle.fileno()
    def tell(self): return self.stream.tell()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.total_read += len(data)
        return data

    def seek(self, offset, whence=0):
        self.rewinds.append((offset, whence))
        if self.after_rewind is not None and offset == 0 and whence == 0:
            self.stream = io.BytesIO(self.after_rewind)
        return self.stream.seek(offset, whence)


def read_traced(path, payload, monkeypatch, after_rewind=None):
    path.write_bytes(payload)
    original_open = Path.open
    traces = []
    def open_file(target, *args, **kwargs):
        raw = original_open(target, *args, **kwargs)
        if target != path: return raw
        trace = TracedRead(raw, payload, after_rewind)
        traces.append(trace)
        return trace
    monkeypatch.setattr(Path, "open", open_file)
    metadata = read_gguf_metadata(path)
    assert len(traces) == 1
    return metadata, traces[0]


def test_geometry_and_hash_cannot_come_from_two_different_reads(tmp_path, monkeypatch):
    original, changed = fixture(layers=2), fixture(layers=3)
    assert len(original) == len(changed)
    metadata, trace = read_traced(tmp_path / "model.gguf", original, monkeypatch, changed)
    assert metadata.n_layer == 2
    assert metadata.sha256 == sha256(original).hexdigest()
    assert metadata.sha256 != sha256(changed).hexdigest()
    assert trace.rewinds == []
    assert trace.total_read == len(original)


def test_alignment_directory_and_multichunk_payload_are_hashed_once(tmp_path, monkeypatch):
    payload = fixture(tensor=True, tail=bytes(range(256)) * (3 * 4096) + b"final-tail")
    metadata, trace = read_traced(tmp_path / "payload.gguf", payload, monkeypatch)
    tensor = metadata.tensor_directory[0]
    assert (tensor.name, tensor.shape, tensor.type_name, tensor.n_bytes, tensor.offset) == (
        "token_embd.weight", (4, 2), "F32", 32, 0)
    assert metadata.sha256 == sha256(payload).hexdigest()
    assert trace.total_read == len(payload) and trace.rewinds == []


def test_truncated_metadata_array_values_still_participate_in_identity(tmp_path, monkeypatch):
    payload = fixture(large_array=True, tensor=True)
    metadata, trace = read_traced(tmp_path / "array.gguf", payload, monkeypatch)
    assert metadata.metadata["unused.numeric_array"] == {"count": 4097, "truncated": True}
    assert metadata.sha256 == sha256(payload).hexdigest()
    assert trace.total_read == len(payload) and trace.rewinds == []
