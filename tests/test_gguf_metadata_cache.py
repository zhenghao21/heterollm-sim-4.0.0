import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from heterollm_sim.gguf_parity import (
    GGUFError,
    read_gguf_metadata_cache,
    write_gguf_metadata_cache,
)


def _text(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _fixture(tail=b"payload"):
    entries = [
        _text("general.architecture") + struct.pack("<I", 8) + _text("llama"),
        _text("llama.block_count") + struct.pack("<II", 4, 1),
        _text("llama.embedding_length") + struct.pack("<II", 4, 4),
        _text("llama.attention.head_count") + struct.pack("<II", 4, 1),
    ]
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries) + tail


def test_sidecar_build_and_load_does_not_reopen_source(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    sidecar = write_gguf_metadata_cache(source)
    assert sidecar == Path(str(source) + ".metadata.json")
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert json_sha(sidecar) == sidecar.read_text(encoding="utf-8").split('"content_sha256": "')[1].split('"', 1)[0]
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if Path(path).resolve() == source.resolve():
            raise AssertionError("sidecar load must not reopen GGUF payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    loaded = read_gguf_metadata_cache(source)
    assert loaded.sha256 == expected_sha
    assert loaded.tensor_count == 0


def json_sha(path):
    import json
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = document.pop("content_sha256")
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest() if expected else ""


def test_source_identity_and_strict_hash_reject_changes(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    sidecar = write_gguf_metadata_cache(source)
    source.write_bytes(_fixture(b"changed"))
    with pytest.raises(GGUFError, match="source identity mismatch"):
        read_gguf_metadata_cache(source, sidecar)


def test_tampered_sidecar_is_rejected(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    sidecar = write_gguf_metadata_cache(source)
    text = sidecar.read_text(encoding="utf-8").replace('"tensor_count": 0', '"tensor_count": 1')
    sidecar.write_text(text, encoding="utf-8")
    with pytest.raises(GGUFError, match="content hash mismatch"):
        read_gguf_metadata_cache(source, sidecar)


def test_schema_version_and_json_shape_are_rejected(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    sidecar = write_gguf_metadata_cache(source)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["parser"]["schema_version"] = 2
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GGUFError, match="parser identity"):
        read_gguf_metadata_cache(source, sidecar)
    sidecar.write_text("[]", encoding="utf-8")
    with pytest.raises(GGUFError, match="JSON object"):
        read_gguf_metadata_cache(source, sidecar)


def test_strict_mode_rejects_rewritten_geometry_even_with_recomputed_sidecar_hash(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    sidecar = write_gguf_metadata_cache(source)
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    document["gguf"]["n_layer"] = 99
    payload = dict(document)
    payload.pop("content_sha256")
    document["content_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GGUFError, match="geometry mismatch"):
        read_gguf_metadata_cache(source, sidecar, strict=True)


def test_strict_mode_rehashes_same_stat_source(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    sidecar = write_gguf_metadata_cache(source)
    original_stat = source.stat()
    source.write_bytes(_fixture(b"changed"))
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(GGUFError, match="source SHA256 mismatch"):
        read_gguf_metadata_cache(source, sidecar, strict=True)


def test_cache_output_cannot_overwrite_gguf(tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    with pytest.raises(GGUFError, match="must not overwrite"):
        write_gguf_metadata_cache(source, source)


def test_cache_build_binds_pre_and_post_source_stat(tmp_path, monkeypatch):
    source = tmp_path / "model.gguf"
    source.write_bytes(_fixture())
    from heterollm_sim import gguf_parity
    original = gguf_parity.read_gguf_metadata

    def mutate_after_read(path):
        metadata = original(path)
        source.write_bytes(_fixture(b"mutated"))
        return metadata

    monkeypatch.setattr(gguf_parity, "read_gguf_metadata", mutate_after_read)
    with pytest.raises(GGUFError, match="changed during metadata cache creation"):
        write_gguf_metadata_cache(source)
