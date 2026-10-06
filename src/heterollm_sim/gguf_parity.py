"""Minimal GGUF metadata reader and native/simulator parity checks."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
import os
from typing import Any, BinaryIO, Mapping
import json


class GGUFError(ValueError):
    pass


# Sidecars are deliberately versioned independently from R0 evidence.  A
# parser change must invalidate an old directory rather than silently reusing
# geometry produced by a different parser.
GGUF_METADATA_CACHE_SCHEMA = "gguf-metadata-cache/v1"
GGUF_METADATA_PARSER_SOURCE = "heterollm_sim.gguf_parity:gguf-directory-parser/v1"
GGUF_MODEL_PRESET_SCHEMA = "heterollm.gguf-model-preset/v1"


_FILE_TYPE_NAMES = {
    # llama_ftype file-type enum used by GGUF metadata, distinct from the
    # ggml tensor-type enum below.  The native 0f3a71b baseline uses 15=Q4_K_M.
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
    4: "MOSTLY_Q4_1_SOME_F16", 7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S",
    27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S", 29: "MOSTLY_IQ2_M",
    30: "MOSTLY_IQ3_M", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
    # Newer llama.cpp uses 30 for the mixed IQ3_M file type.
    30: "MOSTLY_IQ3_M",
}
_TENSOR_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    # Importance-matrix quantizers used by recent Qwen GGUF releases.
    # Block geometry follows ggml_type_traits; keeping these here allows
    # metadata/shape parity and physical-byte accounting before kernel-level
    # support is added to the planner.
    # IDs are ggml_type values (not llama file-type values).  Keep the enum
    # aligned with ggml.h: IQ3_S is 21 and IQ4_XS is 23.
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    type_name: str
    # ``None`` means the GGML type is newer than this parser's physical-size
    # registry.  Keeping the tensor opaque is safer than inventing a byte
    # count and then claiming an exact model artifact.
    block_size: int | None
    n_bytes: int | None
    offset: int


@dataclass(frozen=True)
class GGUFMetadata:
    path: str
    sha256: str
    version: int
    tensor_count: int
    metadata_kv_count: int
    architecture: str | None
    n_layer: int | None
    n_embd: int | None
    n_head: int | None
    n_head_kv: int | None
    vocab_size: int | None
    context_length: int | None
    quantization: str | None
    file_type: int | None
    metadata: Mapping[str, Any]
    tensor_directory: tuple[GGUFTensor, ...] = ()

    @property
    def n_layer_nextn(self) -> int:
        """Auxiliary MTP layers excluded from the executable trunk graph."""
        if not self.architecture:
            return 0
        try:
            return max(0, int(self.metadata.get(f"{self.architecture}.nextn_predict_layers", 0)))
        except (TypeError, ValueError):
            return 0

    @property
    def n_layer_all(self) -> int | None:
        return self.n_layer + self.n_layer_nextn if self.n_layer is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "version": self.version,
                "tensor_count": self.tensor_count, "metadata_kv_count": self.metadata_kv_count,
                "architecture": self.architecture, "n_layer": self.n_layer,
                "n_layer_all": self.n_layer_all, "n_layer_nextn": self.n_layer_nextn,
                "n_embd": self.n_embd, "n_head": self.n_head, "n_head_kv": self.n_head_kv,
                "vocab_size": self.vocab_size, "context_length": self.context_length,
                "quantization": self.quantization, "file_type": self.file_type,
                "metadata": dict(self.metadata),
                "tensor_directory": [t.__dict__ for t in self.tensor_directory]}


class _HashedReadStream:
    """Hash the exact bytes returned to the parser and the unread payload once."""

    def __init__(self, raw: BinaryIO):
        self.raw = raw
        self.digest = sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data

    def tell(self) -> int:
        return self.raw.tell()

    def fileno(self) -> int:
        return self.raw.fileno()


def _read_string(f: BinaryIO) -> str:
    raw = f.read(8)
    if len(raw) != 8:
        raise GGUFError("truncated GGUF string length")
    (length,) = struct.unpack("<Q", raw)
    if length > 16 * 1024 * 1024:
        raise GGUFError("GGUF string exceeds 16 MiB safety limit")
    value = f.read(length)
    if len(value) != length:
        raise GGUFError("truncated GGUF string")
    return value.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, value_type: int) -> Any:
    formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if value_type in formats:
        size = struct.calcsize(formats[value_type])
        raw = f.read(size)
        if len(raw) != size:
            raise GGUFError("truncated GGUF metadata value")
        return struct.unpack(formats[value_type], raw)[0]
    if value_type == 8:
        return _read_string(f)
    if value_type == 9:
        raw = f.read(12)
        if len(raw) != 12:
            raise GGUFError("truncated GGUF array header")
        subtype, count = struct.unpack("<IQ", raw)
        if count > 10_000_000:
            raise GGUFError("GGUF metadata array exceeds safety limit")
        # Keep exported metadata bounded; token types/scores can contain one
        # entry per vocabulary item and are not needed for parity.
        if count > 4096:
            for _ in range(count):
                _read_value(f, subtype)
            return {"count": count, "truncated": True}
        if subtype == 8:
            for _ in range(count):
                _read_string(f)
            return {"count": count}
        return [_read_value(f, subtype) for _ in range(count)]
    raise GGUFError(f"unsupported GGUF metadata type {value_type}")


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _read_gguf_metadata(
    path: str | Path,
    *,
    hash_payload: bool = True,
    allow_unknown_types: bool = False,
) -> GGUFMetadata:
    p = Path(path)
    with p.open("rb") as raw:
        f = _HashedReadStream(raw) if hash_payload else raw
        initial_stat = os.fstat(f.fileno())
        head = f.read(24)
        if len(head) != 24 or head[:4] != b"GGUF":
            raise GGUFError(f"not a GGUF file: {p}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", head[4:])
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            type_raw = f.read(4)
            if len(type_raw) != 4:
                raise GGUFError("truncated GGUF metadata type")
            metadata[key] = _read_value(f, struct.unpack("<I", type_raw)[0])
        alignment = _as_int(metadata.get("general.alignment")) or 32
        directory = []
        names_seen: set[str] = set()
        for _ in range(int(tensor_count)):
            name = _read_string(f)
            if name in names_seen:
                raise GGUFError(f"duplicate GGUF tensor name: {name}")
            names_seen.add(name)
            raw = f.read(4)
            if len(raw) != 4:
                raise GGUFError("truncated GGUF tensor rank")
            (rank,) = struct.unpack("<I", raw)
            if rank > 8:
                raise GGUFError("invalid GGUF tensor rank")
            dims_raw = f.read(8 * rank)
            type_raw = f.read(4); offset_raw = f.read(8)
            if len(dims_raw) != 8 * rank or len(type_raw) != 4 or len(offset_raw) != 8:
                raise GGUFError("truncated GGUF tensor directory")
            dims = tuple(int(x) for x in struct.unpack("<" + "Q" * rank, dims_raw))
            type_id = struct.unpack("<I", type_raw)[0]
            offset = struct.unpack("<Q", offset_raw)[0]
            spec = _TENSOR_TYPES.get(type_id)
            if spec is None:
                if not allow_unknown_types:
                    raise GGUFError(f"unsupported GGUF tensor type {type_id}")
                directory.append(GGUFTensor(
                    name, dims, type_id, f"GGML_TYPE_{type_id}", None, None, offset
                ))
                continue
            type_name, block_size, block_bytes = spec
            elements = 1
            for dim in dims: elements *= dim
            if block_size > 1 and elements % block_size:
                raise GGUFError(
                    f"GGUF tensor {name} has {elements} elements, not divisible by {block_size}"
                )
            n_bytes = (elements // block_size) * block_bytes
            directory.append(GGUFTensor(name, dims, type_id, type_name, block_size, n_bytes, offset))
        data_start = ((f.tell() + alignment - 1) // alignment) * alignment
        file_size = initial_stat.st_size
        ordered = sorted(directory, key=lambda tensor: tensor.offset)
        for index, tensor in enumerate(ordered):
            start = data_start + tensor.offset
            if start >= file_size:
                raise GGUFError(f"truncated GGUF tensor payload: {tensor.name}")
            next_start = (
                data_start + ordered[index + 1].offset
                if index + 1 < len(ordered) else file_size
            )
            if next_start <= start:
                raise GGUFError(f"overlapping GGUF tensor payload: {tensor.name}")
            if tensor.n_bytes is not None and start + tensor.n_bytes > file_size:
                raise GGUFError(f"truncated GGUF tensor payload: {tensor.name}")
            if tensor.n_bytes is not None and start + tensor.n_bytes > next_start:
                raise GGUFError(f"overlapping GGUF tensor payload: {tensor.name}")
        # Full identity mode hashes the exact bytes returned to the parser.
        # Metadata-only mode stops after the directory and never touches the
        # tensor payload; the sidecar loader validates size/mtime/file-id.
        if hash_payload:
            for _chunk in iter(lambda: f.read(1024 * 1024), b""):
                pass
            digest = f.digest.hexdigest()
            bytes_read = f.bytes_read
        else:
            digest = ""
            bytes_read = None
        final_stat = os.fstat(f.fileno())
        path_stat = p.stat()
        # Windows fstat/stat can differ in timestamp semantics; ctime
        # is not a portable content-change clock. Check identity, size and mtime.
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size,
                                 stat.st_mtime_ns)
        if ((hash_payload and bytes_read != file_size)
                or identity(initial_stat) != identity(final_stat)
                or identity(final_stat) != identity(path_stat)):
            raise GGUFError(
                f"GGUF changed during metadata/hash read: {p}; "
                f"bytes_read={bytes_read}, expected_bytes={file_size}; "
                f"initial={identity(initial_stat)}, final={identity(final_stat)}, "
                f"path={identity(path_stat)}"
            )
    arch = metadata.get("general.architecture")
    prefix = str(arch) if arch else ""
    pick = lambda suffix: metadata.get(f"{prefix}.{suffix}") if prefix else None
    vocab = metadata.get("tokenizer.ggml.tokens")
    if isinstance(vocab, dict):
        vocab = vocab.get("count")
    elif vocab is None:
        vocab = metadata.get("general.vocab_size")
    file_type = metadata.get("general.file_type")
    raw_layers = _as_int(pick("block_count"))
    # llama.cpp exposes n_layer() as n_layer_all - n_layer_nextn.  MTP/nextn
    # blocks are auxiliary prediction heads and must not enter the simulator's
    # decoder graph or timing geometry.
    nextn = _as_int(pick("nextn_predict_layers")) or 0
    trunk_layers = raw_layers - nextn if raw_layers is not None else None
    if trunk_layers is not None and trunk_layers <= 0:
        raise GGUFError("GGUF block_count is not larger than nextn_predict_layers")
    return GGUFMetadata(str(p.resolve()), digest, int(version), int(tensor_count), int(kv_count),
                        str(arch) if arch is not None else None, trunk_layers,
                        _as_int(pick("embedding_length")), _as_int(pick("attention.head_count")),
                        _as_int(pick("attention.head_count_kv")), _as_int(vocab), _as_int(pick("context_length")),
                        _FILE_TYPE_NAMES.get(int(file_type)) if file_type is not None else None,
                        _as_int(file_type), metadata, tuple(directory))


def read_gguf_metadata(
    path: str | Path, *, allow_unknown_types: bool = False
) -> GGUFMetadata:
    """Read and hash a GGUF in the historical, strict default mode."""
    return _read_gguf_metadata(
        path, hash_payload=True, allow_unknown_types=allow_unknown_types
    )


def read_gguf_metadata_only(
    path: str | Path, *, allow_unknown_types: bool = False
) -> GGUFMetadata:
    """Read header/metadata/tensor directory without reading tensor payload."""
    return _read_gguf_metadata(
        path, hash_payload=False, allow_unknown_types=allow_unknown_types
    )


def _cache_file_id(stat: os.stat_result) -> str:
    # st_ino is the Windows file index exposed by Python; include st_dev for
    # files moved between volumes and keep the JSON representation portable.
    return f"{int(stat.st_dev)}:{int(stat.st_ino)}"


def _cache_source_stat(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _cache_payload(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _parser_source_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _metadata_cache_document(gguf: GGUFMetadata, stat: os.stat_result,
                             *, parser_source: str = GGUF_METADATA_PARSER_SOURCE) -> dict[str, Any]:
    return {
        "schema": GGUF_METADATA_CACHE_SCHEMA,
        "parser": {"schema_version": 1, "source_identity": parser_source,
                    "source_sha256": _parser_source_sha256()},
        "source": {"path": str(Path(gguf.path).resolve()), "size_bytes": int(stat.st_size),
                   "mtime_ns": int(stat.st_mtime_ns), "file_id": _cache_file_id(stat),
                   "sha256": gguf.sha256},
        "gguf": {"path": gguf.path, "sha256": gguf.sha256, "version": gguf.version,
                 "tensor_count": gguf.tensor_count, "metadata_kv_count": gguf.metadata_kv_count,
                 "architecture": gguf.architecture, "n_layer": gguf.n_layer,
                 "n_embd": gguf.n_embd, "n_head": gguf.n_head, "n_head_kv": gguf.n_head_kv,
                 "vocab_size": gguf.vocab_size, "context_length": gguf.context_length,
                 "quantization": gguf.quantization, "file_type": gguf.file_type,
                 "metadata": dict(gguf.metadata),
                 "tensor_directory": [dict(name=t.name, shape=list(t.shape), type_id=t.type_id,
                    type_name=t.type_name, block_size=t.block_size, n_bytes=t.n_bytes, offset=t.offset)
                    for t in gguf.tensor_directory]},
    }


def write_gguf_metadata_cache(gguf_path: str | Path, cache_path: str | Path | None = None) -> Path:
    """Build a sidecar, hashing the source exactly once during creation."""
    source = Path(gguf_path).resolve(strict=True)
    target = Path(cache_path) if cache_path is not None else Path(str(source) + ".metadata.json")
    if target.resolve() == source or (target.exists() and target.samefile(source)):
        raise GGUFError("GGUF metadata cache output must not overwrite source GGUF")
    before = source.stat()
    gguf = read_gguf_metadata(source)
    stat = source.stat()
    if _cache_source_stat(before) != _cache_source_stat(stat):
        raise GGUFError("GGUF changed during metadata cache creation")
    document = _metadata_cache_document(gguf, stat)
    document["content_sha256"] = sha256(_cache_payload(document)).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def gguf_metadata_digest(gguf: GGUFMetadata) -> str:
    """Digest only the metadata and tensor directory used by the simulator."""
    payload = {
        "metadata": dict(gguf.metadata),
        "tensor_directory": [
            {"name": tensor.name, "shape": list(tensor.shape), "type_id": tensor.type_id,
             "type_name": tensor.type_name, "block_size": tensor.block_size,
             "n_bytes": tensor.n_bytes, "offset": tensor.offset}
            for tensor in gguf.tensor_directory
        ],
    }
    return sha256(_cache_payload(payload)).hexdigest()


def _metadata_from_cache(document: Mapping[str, Any], source: Path) -> GGUFMetadata:
    data = document.get("gguf")
    if not isinstance(data, Mapping):
        raise GGUFError("GGUF metadata cache missing gguf section")
    try:
        if (not isinstance(data.get("metadata"), dict)
                or not isinstance(data.get("tensor_directory"), list)
                or data.get("path") != str(source)):
            raise ValueError("invalid metadata, directory or source path")
        for field in ("version", "tensor_count", "metadata_kv_count"):
            if type(data.get(field)) is not int:
                raise ValueError("invalid " + field)
        for field in ("n_layer", "n_embd", "n_head", "n_head_kv", "vocab_size", "context_length", "file_type"):
            if data.get(field) is not None and type(data[field]) is not int:
                raise ValueError("invalid " + field)
        for item in data["tensor_directory"]:
            if (not isinstance(item, dict) or not isinstance(item.get("shape"), list)
                    or any(type(x) is not int for x in item["shape"])
                    or any(type(item.get(field)) is not int for field in ("type_id", "offset"))
                    or any(item.get(field) is not None and type(item.get(field)) is not int
                           for field in ("block_size", "n_bytes"))
                    or not isinstance(item.get("name"), str) or not isinstance(item.get("type_name"), str)):
                raise ValueError("invalid tensor")
        tensors = tuple(GGUFTensor(str(item["name"]), tuple(int(x) for x in item["shape"]), int(item["type_id"]),
            str(item["type_name"]), _as_int(item.get("block_size")), _as_int(item.get("n_bytes")), int(item["offset"]))
            for item in data["tensor_directory"])
        if len(tensors) != data["tensor_count"]:
            raise ValueError("tensor count mismatch")
        return GGUFMetadata(str(source), str(data["sha256"]), int(data["version"]), int(data["tensor_count"]),
            int(data["metadata_kv_count"]), data.get("architecture"), _as_int(data.get("n_layer")),
            _as_int(data.get("n_embd")), _as_int(data.get("n_head")), _as_int(data.get("n_head_kv")),
            _as_int(data.get("vocab_size")), _as_int(data.get("context_length")), data.get("quantization"),
            _as_int(data.get("file_type")), dict(data.get("metadata") or {}), tensors)
    except (KeyError, TypeError, ValueError) as exc:
        raise GGUFError("invalid GGUF metadata cache geometry") from exc


def read_gguf_metadata_cache(gguf_path: str | Path, cache_path: str | Path | None = None,
                             *, strict: bool = False) -> GGUFMetadata:
    """Load a validated sidecar; strict mode rehashes the GGUF payload."""
    source = Path(gguf_path).resolve(strict=True)
    before = source.stat()
    target = Path(cache_path) if cache_path is not None else Path(str(source) + ".metadata.json")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GGUFError(f"cannot read GGUF metadata cache: {target}") from exc
    if not isinstance(document, dict):
        raise GGUFError("GGUF metadata cache must be a JSON object")
    if document.get("schema") != GGUF_METADATA_CACHE_SCHEMA:
        raise GGUFError("unsupported GGUF metadata cache schema")
    parser = document.get("parser")
    if (not isinstance(parser, Mapping)
            or type(parser.get("schema_version")) is not int or parser.get("schema_version") != 1
            or parser.get("source_identity") != GGUF_METADATA_PARSER_SOURCE
            or parser.get("source_sha256") != _parser_source_sha256()):
        raise GGUFError("GGUF metadata cache parser identity mismatch")
    actual_content = document.get("content_sha256")
    unsigned = dict(document); unsigned.pop("content_sha256", None)
    try:
        expected_content = sha256(_cache_payload(unsigned)).hexdigest()
    except (ValueError, TypeError) as exc:
        raise GGUFError("invalid GGUF metadata cache JSON values") from exc
    if actual_content != expected_content:
        raise GGUFError("GGUF metadata cache content hash mismatch")
    source_info = document.get("source")
    if not isinstance(source_info, Mapping):
        raise GGUFError("GGUF metadata cache missing source identity")
    if (any(type(source_info.get(field)) is not int for field in ("size_bytes", "mtime_ns"))
            or any(not isinstance(source_info.get(field), str) for field in ("path", "file_id", "sha256"))):
        raise GGUFError("invalid GGUF metadata cache source identity")
    stat = source.stat()
    expected = (str(source), int(stat.st_size), int(stat.st_mtime_ns), _cache_file_id(stat))
    actual = (str(source_info.get("path")), int(source_info.get("size_bytes", -1)),
              int(source_info.get("mtime_ns", -1)), str(source_info.get("file_id")))
    if expected != actual:
        raise GGUFError(f"GGUF metadata cache source identity mismatch: {source}")
    gguf = _metadata_from_cache(document, source)
    if str(source_info.get("sha256")) != gguf.sha256:
        raise GGUFError("GGUF metadata cache source SHA256 binding mismatch")
    if strict:
        fresh = read_gguf_metadata(source)
        if fresh.sha256 != gguf.sha256:
            raise GGUFError("GGUF metadata cache source SHA256 mismatch")
        if _cache_payload(document["gguf"]) != _cache_payload(_metadata_cache_document(fresh, source.stat())["gguf"]):
            raise GGUFError("GGUF metadata cache geometry mismatch")
    if _cache_source_stat(before) != _cache_source_stat(source.stat()):
        raise GGUFError("GGUF changed during metadata cache load")
    return gguf


def build_gguf_model_preset(gguf: GGUFMetadata) -> dict[str, Any]:
    """Build a conservative, architecture-agnostic GGUF model preset.

    This is an artifact preset, not an executable :class:`ModelSpec`.  It
    preserves facts proved by the file and records unknown physical types as
    opaque instead of inventing a layer graph or byte count.  Architecture
    adapters can later consume the same inventory when they can prove an
    executable mapping.
    """
    if not isinstance(gguf, GGUFMetadata):
        raise TypeError("gguf must be GGUFMetadata")
    opaque = [tensor for tensor in gguf.tensor_directory if tensor.n_bytes is None]
    known_bytes = sum(
        tensor.n_bytes for tensor in gguf.tensor_directory
        if tensor.n_bytes is not None
    )
    geometry_fields = (
        gguf.architecture, gguf.n_layer, gguf.n_embd, gguf.n_head,
        gguf.n_head_kv, gguf.vocab_size, gguf.context_length,
    )
    file_id = gguf.sha256 or gguf_metadata_digest(gguf)
    preset_id = "gguf-{}".format(file_id[:16])
    limitations = [
        "GGUF metadata and tensor inventory are exact file evidence; execution graph semantics require an architecture adapter.",
        "Hardware topology, kernel performance, and llama.cpp runtime options are not part of a GGUF file.",
    ]
    if opaque:
        limitations.append(
            "{} tensor type(s) are not in the local physical-size registry; their payload bytes remain opaque.".format(
                len(opaque)
            )
        )
    # Keep the artifact graph displayable by the frontend while refusing to
    # invent operators from tensor names.  This mirrors the existing
    # display-only preset contract used for out-of-domain architectures.
    graph_dtype = "unknown"
    graph_shape = ["B", "T", "H"]
    input_tensor = {
        "tensor_id": "input.hidden",
        "role": "input",
        "logical_bytes": None,
        "producer_operator_id": "input",
        "consumer_operator_ids": ["unsupported-artifact"],
        "dtype": graph_dtype,
        "shape": graph_shape,
        "layout": "logical",
        "attributes": {},
        "provenance": [],
    }
    output_tensor = {
        "tensor_id": "output.hidden",
        "role": "output",
        "logical_bytes": None,
        "producer_operator_id": "unsupported-artifact",
        "consumer_operator_ids": ["output"],
        "dtype": graph_dtype,
        "shape": graph_shape,
        "layout": "logical",
        "attributes": {},
        "provenance": [],
    }
    graph = {
        "graph_id": preset_id,
        "operators": [
            {
                "operator_id": "input",
                "op_kind": "model_input",
                "sequence_index": 0,
                "layer_id": None,
                "input_tensor_ids": [],
                "output_tensor_ids": ["input.hidden"],
                "weight_tensor_ids": [],
                "ports": [{"port_id": "out0", "direction": "output", "tensor_id": "input.hidden", "dtype": graph_dtype, "shape": graph_shape, "layout": "logical", "attributes": {}}],
                "parameters": {},
                "attributes": {},
                "provenance": [],
            },
            {
                "operator_id": "unsupported-artifact",
                "op_kind": "unsupported_component_group",
                "sequence_index": 1,
                "layer_id": None,
                "input_tensor_ids": ["input.hidden"],
                "output_tensor_ids": ["output.hidden"],
                "weight_tensor_ids": [],
                "ports": [
                    {"port_id": "in0", "direction": "input", "tensor_id": "input.hidden", "dtype": graph_dtype, "shape": graph_shape, "layout": "logical", "attributes": {}},
                    {"port_id": "out0", "direction": "output", "tensor_id": "output.hidden", "dtype": graph_dtype, "shape": graph_shape, "layout": "logical", "attributes": {}},
                ],
                "parameters": {"architecture": gguf.architecture, "layer_count": gguf.n_layer, "tensor_count": gguf.tensor_count},
                "attributes": {"unsupported": True, "reason": "GGUF artifact inventory has no proven execution graph"},
                "provenance": [],
            },
            {
                "operator_id": "output",
                "op_kind": "model_output",
                "sequence_index": 2,
                "layer_id": None,
                "input_tensor_ids": ["output.hidden"],
                "output_tensor_ids": [],
                "weight_tensor_ids": [],
                "ports": [{"port_id": "in0", "direction": "input", "tensor_id": "output.hidden", "dtype": graph_dtype, "shape": graph_shape, "layout": "logical", "attributes": {}}],
                "parameters": {},
                "attributes": {},
                "provenance": [],
            },
        ],
        "tensors": [input_tensor, output_tensor],
        "source_operators": [],
        "sub_operators": [],
        "transforms": [],
        "executable": False,
        "attributes": {
            "authoritative": False,
            "derivation": "gguf_artifact_inventory",
            "architecture": gguf.architecture or "unknown",
            "support_level": "metadata_only",
            "execution_graph": "unproven",
            "symbols": {
                "B": "batch",
                "T": "sequence",
                "H": gguf.n_embd or "hidden",
                "V": gguf.vocab_size or 0,
            },
            "max_sequence_length": gguf.context_length or 0,
            "evidence": {"file_identity": "exact" if gguf.sha256 else "metadata_digest_only", "geometry": "exact" if all(value is not None for value in geometry_fields) else "partial"},
            "ui": {"collapsed_groups": ["unsupported-artifact"]},
            "limitations": limitations,
        },
        "provenance": [],
    }
    return {
        "schema_version": GGUF_MODEL_PRESET_SCHEMA,
        "preset": {
            "id": preset_id,
            "name": Path(gguf.path).stem or "GGUF model",
            "family": gguf.architecture or "unknown",
            "source": "gguf_import",
            "source_sha": gguf.sha256 or None,
            "support_level": "metadata_only",
            "coverage": "gguf_artifact",
            "generation_allowed": False,
            "license": None,
            "openness": None,
            "limitations": limitations,
        },
        "model": {
            "architecture": gguf.architecture,
            "n_layer": gguf.n_layer,
            "n_layer_all": gguf.n_layer_all,
            "n_layer_nextn": gguf.n_layer_nextn,
            "n_embd": gguf.n_embd,
            "n_head": gguf.n_head,
            "n_head_kv": gguf.n_head_kv,
            "vocab_size": gguf.vocab_size,
            "context_length": gguf.context_length,
            "quantization": gguf.quantization,
            "file_type": gguf.file_type,
        },
        "evidence": {
            "file_identity": "exact" if gguf.sha256 else "metadata_digest_only",
            "geometry": "exact" if all(value is not None for value in geometry_fields) else "partial",
            "tensor_directory": "exact",
            "physical_tensor_bytes": "exact" if not opaque else "partial",
            "execution_graph": "unproven",
        },
        "weights": {
            "tensor_count": gguf.tensor_count,
            "known_physical_bytes": known_bytes,
            "opaque_tensor_count": len(opaque),
            "tensor_inventory": [
                {
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "type_id": tensor.type_id,
                    "type": tensor.type_name,
                    "block_size": tensor.block_size,
                    "n_bytes": tensor.n_bytes,
                    "offset": tensor.offset,
                    "bytes_status": "exact" if tensor.n_bytes is not None else "opaque",
                }
                for tensor in gguf.tensor_directory
            ],
        },
        "gguf": gguf.as_dict(),
        "graph": graph,
    }


def import_gguf_model_preset(path: str | Path) -> dict[str, Any]:
    """Import any parseable GGUF as a conservative generic preset."""
    return build_gguf_model_preset(
        read_gguf_metadata(path, allow_unknown_types=True)
    )


def _expected_geometry(model: Any) -> dict[str, int | str | None]:
    from .ir import model_graph_execution_view
    view = model_graph_execution_view(model.graph, schema_version=model.schema_version)
    layer = view.layer_instances[0].layer
    return {"architecture": view.architecture, "n_layer": len(view.layer_instances), "n_embd": layer.hidden_size,
            "n_head": layer.attention_heads, "n_head_kv": layer.effective_kv_heads,
            "vocab_size": view.vocabulary_size, "context_length": view.max_sequence_length}


def compare_gguf_to_model(gguf: GGUFMetadata, model: Any, *, context_length: int | None = None,
                          expected_prompt_tokens: int | None = None, actual_prompt_tokens: int | None = None,
                          expected_output_tokens: int | None = None, actual_output_tokens: int | None = None) -> dict[str, Any]:
    expected = _expected_geometry(model)
    mismatches: list[dict[str, Any]] = []
    family = str(getattr(model, "metadata", {}).get("family", "")).lower()
    if not gguf.architecture:
        mismatches.append({"field": "architecture", "expected": family or expected["architecture"], "actual": None})
    elif family and not any(part in gguf.architecture.lower() for part in family.replace(".", " ").split()):
        # Generic graph architecture is accepted; concrete family mismatches fail closed.
        if not (family.startswith("qwen") and gguf.architecture.lower().startswith("qwen")):
            mismatches.append({"field": "architecture", "expected": family or expected["architecture"], "actual": gguf.architecture})
    for field in ("n_layer", "n_embd", "n_head", "n_head_kv", "vocab_size"):
        actual, want = getattr(gguf, field), expected[field]
        if actual != want:
            mismatches.append({"field": field, "expected": want, "actual": actual})
    # Runtime context is a limit; GGUF context_length is the model maximum.
    # A smaller runtime context is valid, while exceeding the model maximum
    # must fail closed.
    if context_length is not None and gguf.context_length is not None and context_length > gguf.context_length:
        mismatches.append({"field": "context_length", "expected_max": gguf.context_length, "actual": context_length})
    for field, want, actual in (("prompt_tokens", expected_prompt_tokens, actual_prompt_tokens), ("output_tokens", expected_output_tokens, actual_output_tokens)):
        if want is not None and actual is not None and want != actual:
            mismatches.append({"field": field, "expected": want, "actual": actual})
    return {"ok": not mismatches, "status": "pass" if not mismatches else "fail", "mismatches": mismatches,
            "expected": expected, "gguf": gguf.as_dict()}


def assert_gguf_parity(report: Mapping[str, Any]) -> None:
    if not report.get("ok"):
        fields = ", ".join(str(item.get("field")) for item in report.get("mismatches", ()))
        raise ValueError(f"GGUF/native parity gate failed: {fields or 'unknown mismatch'}")


def _gguf_architecture_prefixes(gguf: GGUFMetadata) -> tuple[str, ...]:
    """Return metadata namespaces in precedence order."""
    arch = str(gguf.architecture or "").strip().lower()
    if not arch:
        return ()
    return (arch, "qwen35") if arch == "qwen35moe" else (arch,)


def _gguf_positive_metadata_int(
    metadata: Mapping[str, Any], prefixes: tuple[str, ...], suffixes: tuple[str, ...],
    *, label: str, required: bool = False,
) -> int | None:
    """Read one positive integer from formal GGUF metadata namespaces."""
    values: list[tuple[str, int]] = []
    for prefix in prefixes:
        namespace_values: list[tuple[str, int]] = []
        for suffix in suffixes:
            key = f"{prefix}.{suffix}"
            if key not in metadata:
                continue
            value = _as_int(metadata.get(key))
            if value is None or value <= 0:
                raise GGUFError(f"GGUF {key} must be a positive integer")
            namespace_values.append((key, value))
        if namespace_values:
            values.extend(namespace_values)
            break  # architecture namespace takes precedence over alias
    if not values:
        if required:
            raise GGUFError(f"GGUF is missing required {label} metadata")
        return None
    first = values[0][1]
    if any(value != first for _, value in values[1:]):
        keys = ", ".join(f"{key}={value}" for key, value in values)
        raise GGUFError(f"conflicting GGUF {label} metadata: {keys}")
    return first


def _gguf_attention_metadata_head_dim(gguf: GGUFMetadata) -> int | None:
    """Resolve formal attention head dimension metadata, if declared."""
    prefixes = _gguf_architecture_prefixes(gguf)
    if not prefixes:
        return None
    values: list[tuple[str, int]] = []
    for label, suffixes in (
        ("key_length", ("attention.key_length", "attention.head_dim", "head_dim")),
        ("value_length", ("attention.value_length", "attention.head_dim", "head_dim")),
    ):
        value = _gguf_positive_metadata_int(gguf.metadata, prefixes, suffixes, label=label)
        if value is not None:
            values.append((label, value))
    if not values:
        return None
    first = values[0][1]
    if any(value != first for _, value in values[1:]):
        raise GGUFError("GGUF key/value attention head dimensions conflict")
    return first


def _matrix_extent(tensor: GGUFTensor) -> tuple[int, int]:
    """Return GGUF matrix ``(input K, output N)`` dimensions."""
    if len(tensor.shape) != 2 or any(int(dimension) <= 0 for dimension in tensor.shape):
        raise GGUFError(f"tensor {tensor.name} is not a positive-rank-2 matrix")
    return int(tensor.shape[0]), int(tensor.shape[1])


def _resolve_attention_head_dim(
    gguf: GGUFMetadata,
    q: GGUFTensor,
    k: GGUFTensor,
    v: GGUFTensor,
    o: GGUFTensor,
    *,
    qwen35_full: bool = False,
    declared_head_dim: int | None = None,
) -> int:
    """Resolve and cross-check Q/K/V/O attention geometry."""
    hidden = int(gguf.n_embd or 0)
    heads = int(gguf.n_head or 0)
    kv_heads = int(gguf.n_head_kv or 0)
    if hidden <= 0 or heads <= 0 or kv_heads <= 0:
        raise GGUFError("GGUF attention geometry is missing positive head counts")
    q_in, q_width = _matrix_extent(q)
    k_in, k_width = _matrix_extent(k)
    v_in, v_width = _matrix_extent(v)
    o_width, o_out = _matrix_extent(o)
    if q_in != hidden or k_in != hidden or v_in != hidden or o_out != hidden:
        raise GGUFError(f"GGUF Q/K/V/O width conflicts with hidden size {hidden}")
    metadata_dim = declared_head_dim if declared_head_dim is not None else _gguf_attention_metadata_head_dim(gguf)
    if metadata_dim is None:
        divisor = heads * (2 if qwen35_full else 1)
        if q_width % divisor:
            raise GGUFError("GGUF Q projection width is not divisible by head count")
        metadata_dim = q_width // divisor
    if metadata_dim <= 0:
        raise GGUFError("GGUF attention head dimension must be positive")
    multiplier = 2 if qwen35_full else 1
    expected = (
        heads * metadata_dim * multiplier,
        kv_heads * metadata_dim,
        kv_heads * metadata_dim,
        heads * metadata_dim,
    )
    actual = (q_width, k_width, v_width, o_width)
    if actual != expected:
        raise GGUFError(
            "GGUF Q/K/V/O tensor widths conflict with declared/inferred head_dim "
            f"(got q={q_width}, k={k_width}, v={v_width}, o_in={o_width}; expected {expected})"
        )
    return int(metadata_dim)


def _unique_tensor_bytes(tensors: tuple[GGUFTensor, ...] | list[GGUFTensor]) -> int:
    """Count physical payload bytes once for tensors sharing a GGUF offset."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors:
        if tensor.n_bytes is None:
            continue
        identity = (int(tensor.offset), int(tensor.n_bytes))
        if identity in seen:
            continue
        seen.add(identity)
        total += int(tensor.n_bytes)
    return total


def _build_model_from_gguf_registered(gguf: GGUFMetadata):
    """Bind a supported GGUF directory to an executable ``ModelSpec``.

    Architecture selection follows the same boundary as llama.cpp: the
    ``general.architecture`` metadata value is resolved through a registry,
    and only then is the selected graph implementation allowed to interpret
    tensor names. Multiple architecture IDs may share one implementation.
    """
    from .ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    from .architecture_adapters import resolve_gguf_architecture_adapter
    import re
    adapter = resolve_gguf_architecture_adapter(gguf.architecture)
    if adapter is None:
        raise GGUFError(f"unsupported GGUF architecture: {gguf.architecture}")
    required = (gguf.n_layer, gguf.n_embd, gguf.n_head, gguf.n_head_kv, gguf.vocab_size, gguf.context_length)
    if any(value is None or value <= 0 for value in required):
        raise GGUFError("GGUF is missing required model geometry")
    if any(tensor.n_bytes is None for tensor in gguf.tensor_directory):
        raise GGUFError(
            "GGUF contains unregistered tensor types; use the generic artifact preset"
        )
    by_layer: dict[int, list[GGUFTensor]] = {}
    for tensor in gguf.tensor_directory:
        match = re.search(r"(?:blk|layers)\.(\d+)\.", tensor.name)
        if match:
            by_layer.setdefault(int(match.group(1)), []).append(tensor)
    if not set(range(gguf.n_layer)).issubset(by_layer):
        raise GGUFError("GGUF tensor directory does not contain every decoder layer")

    def binding(t: GGUFTensor) -> dict[str, Any]:
        return {"name": t.name, "shape": list(t.shape), "type": t.type_name,
                "block_size": t.block_size, "n_bytes": t.n_bytes, "offset": t.offset}
    layers = []
    # Qwen3.5/3.8 GGUFs encode a 3-linear/1-full hybrid sequence.  The
    # metadata also carries the final ``nextn`` block; llama.cpp reports it in
    # block_count, so retain it as an executable full-attention layer for
    # geometry parity while recording the auxiliary tensors in bindings.
    is_qwen35 = adapter.hybrid
    qwen35_metadata_prefix = gguf.architecture if gguf.architecture in {"qwen35", "qwen35moe"} else "qwen35"
    qwen35_prefixes = _gguf_architecture_prefixes(gguf)
    declared_head_dim = _gguf_attention_metadata_head_dim(gguf) if is_qwen35 else None
    has_qwen35_linear_blocks = bool(
        is_qwen35 and any(
            any(tensor.name.lower().endswith(suffix) for suffix in ("attn_qkv.weight", "ssm_out.weight"))
            for values in by_layer.values() for tensor in values
        )
    )
    linear_attention = None
    if is_qwen35:
        from .ir import LinearAttentionSpec
        md = gguf.metadata
        if gguf.architecture == "qwen35moe":
            expert_count = _as_int(md.get(f"{qwen35_metadata_prefix}.expert_count"))
            if expert_count is None:
                # Some converters retain the qwen35 metadata namespace even
                # when general.architecture is qwen35moe.  Accept that alias
                # only when it proves a dense model; missing evidence stays
                # fail-closed.
                expert_count = _as_int(md.get("qwen35.expert_count"))
            if expert_count is None or expert_count > 1:
                raise GGUFError(
                    "Qwen35 MoE GGUF lacks a proven dense mapping; use the generic artifact preset"
                )
        elif _as_int(md.get(f"{qwen35_metadata_prefix}.expert_count")) not in (None, 1):
            raise GGUFError(
                "Qwen35 GGUF contains experts; use the generic artifact preset"
            )
        def ssm_int(suffix: str) -> int | None:
            return _gguf_positive_metadata_int(
                md, qwen35_prefixes, (f"ssm.{suffix}",), label=f"ssm.{suffix}"
            )
        ssm_groups = ssm_int("group_count") if has_qwen35_linear_blocks else None
        ssm_inner = ssm_int("inner_size") if has_qwen35_linear_blocks else None
        ssm_state = ssm_int("state_size") if has_qwen35_linear_blocks else None
        ssm_conv = ssm_int("conv_kernel") if has_qwen35_linear_blocks else None
        # These are architecture geometry fields, not calibration defaults.
        # Missing values are handled below from a fused QKV tensor where that
        # width proves the inner size; state/group/conv remain unproven and
        # therefore fail closed rather than silently selecting 16/128/4.
        if has_qwen35_linear_blocks and (ssm_groups is None or ssm_state is None or ssm_conv is None):
            raise GGUFError("Qwen3.5 GGUF is missing formal SSM geometry metadata")
        if has_qwen35_linear_blocks and ssm_inner is None:
            for candidate in (tensor for values in by_layer.values() for tensor in values):
                if candidate.name.lower().endswith("attn_qkv.weight"):
                    _, width = _matrix_extent(candidate)
                    if width % 3 == 0:
                        ssm_inner = width // 3
                        break
        if has_qwen35_linear_blocks and (ssm_inner is None or ssm_inner % ssm_state):
            raise GGUFError("Qwen3.5 GGUF SSM inner_size is not divisible by state_size")
        if has_qwen35_linear_blocks:
            linear_attention = LinearAttentionSpec(
                key_heads=ssm_groups,
                value_heads=ssm_inner // ssm_state,
                key_head_dim=ssm_state,
                value_head_dim=ssm_state,
                conv_kernel_size=ssm_conv,
                state_dtype="fp32",
                output_gate=True,
                gate_activation="silu",
            )
    for index in range(gguf.n_layer):
        tensors = by_layer[index]
        names = {t.name.lower(): t for t in tensors}
        def find(*parts):
            # Match the projection component as a path component.  A loose
            # substring search makes ``attn_q`` accidentally select
            # ``attn_q_norm`` on directory order changes.
            for part in parts:
                suffix = part.lower()
                for t in tensors:
                    name = t.name.lower()
                    if name.endswith(suffix + ".weight") or name.endswith(suffix):
                        return t
            return None
        q, k, v = find("attn_q", "attention.wq"), find("attn_k", "attention.wk"), find("attn_v", "attention.wv")
        qkv = find("attn_qkv")
        o = find("attn_output", "attention.wo", "self_attn.o_proj")
        ssm_out = find("ssm_out")
        gate, up, down = find("ffn_gate", "mlp.gate_proj"), find("ffn_up", "mlp.up_proj"), find("ffn_down", "mlp.down_proj")
        if is_qwen35 and qkv is not None and ssm_out is not None:
            # Linear-attention blocks use a fused QKV input projection and an
            # SSM output projection instead of separate Q/K/V tensors.
            q = k = v = qkv
            o = ssm_out
        if not all((q, k, v, o, down)) or (gate is None and up is None):
            raise GGUFError(f"layer {index} is missing QKV/O/FFN tensors")
        def extent(t):
            return _matrix_extent(t)
        hidden = int(gguf.n_embd)
        # GGUF matrices are stored as [input (K), output (N)].  The gated/up
        # projection therefore exposes the FFN width on dimension 1; using
        # dimension 0 silently collapsed every imported model's MLP to the
        # hidden size and made simulator timings dramatically too small.
        intermediate = int((gate or up).shape[1])
        is_linear_block = bool(is_qwen35 and q is k is v)
        resolved_head_dim: int | None = None
        if not is_linear_block:
            resolved_head_dim = _resolve_attention_head_dim(
                gguf, q, k, v, o, qwen35_full=is_qwen35,
                declared_head_dim=declared_head_dim,
            )
        elif declared_head_dim is None:
            # A Qwen3.5 file containing only linear blocks cannot prove the
            # full-attention query geometry from the fused SSM projection.
            # Refuse the executable adapter instead of reviving the old 256
            # hard-coded calibration geometry.
            raise GGUFError("Qwen3.5 linear-only GGUF lacks formal attention head_dim metadata")
        else:
            resolved_head_dim = declared_head_dim
        metadata: dict[str, Any] = {"gguf_tensor_bindings": [binding(t) for t in tensors], "gguf_physical_weight_bytes": _unique_tensor_bytes(tensors)}
        projections: dict[str, Any] = {}
        def add_projection(pid, ts, shard_axis="n", segment_ids=None):
            segments = []
            for j, t in enumerate(ts):
                if t.type_name not in {"Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q5_0", "Q8_0", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "IQ2_S", "IQ1_S", "IQ1_M"}:
                    return
                kk, nn = extent(t)
                segment_id = (tuple(segment_ids)[j] if segment_ids is not None else f"{pid.replace('.', '-')}-{j}")
                segments.append({"segment_id": segment_id, "physical_tensor_name": t.name, "k": kk, "n": nn, "format": t.type_name, "physical_bytes": t.n_bytes, "tp_shard_axis": shard_axis})
            if segments: projections[pid] = {"segments": segments}
        if is_qwen35 and q is k is v:
            add_projection("linear_attention.qkv", (q,))
            add_projection("linear_attention.output", (o,), "k")
            gate_tensor = find("attn_gate")
            if gate_tensor is not None:
                add_projection("linear_attention.output_gate", (gate_tensor,))
        else:
            add_projection("attention.qkv", (q, k, v), segment_ids=("q", "k", "v")); add_projection("attention.output", (o,), "k", segment_ids=("output",))
            if is_qwen35:
                # llama.cpp's Qwen3.5 full-attention graph uses q/k RMS norms,
                # a sigmoid query gate, and a rotary section.  The dimension
                # comes from formal metadata or the Q/K/V/O tensor evidence.
                q_width = int(gguf.n_head) * int(resolved_head_dim)
                rotary_dim = _gguf_positive_metadata_int(
                    gguf.metadata, qwen35_prefixes, ("rope.dimension_count",),
                    label="rope.dimension_count",
                )
                if rotary_dim is None:
                    rotary_dim = int(resolved_head_dim)
                if rotary_dim > resolved_head_dim or rotary_dim % 2:
                    raise GGUFError("Qwen3.5 rotary dimension conflicts with attention head_dim")
                metadata["attention_execution_descriptor"] = {
                    "schema_version": "heterollm.attention-execution/v1",
                    "query_heads": int(gguf.n_head), "kv_heads": int(gguf.n_head_kv),
                    "head_dim": int(resolved_head_dim), "query_width": q_width, "gate_width": q_width,
                    "q_projection_width": 2 * q_width, "rotary_dim": int(rotary_dim),
                    "qk_scale": 1.0 / (float(resolved_head_dim) ** 0.5), "qk_norm": True,
                    "gate_activation": "sigmoid",
                }
        if is_qwen35 and q is k is v:
            # Gated-delta-net alpha/beta projections are real matrix multiplies;
            # omit scalar state vectors (ssm_a/dt) from weight projections.
            alpha = find("ssm_alpha")
            beta = find("ssm_beta")
            if alpha is not None: add_projection("linear_attention.alpha", (alpha,))
            if beta is not None: add_projection("linear_attention.beta", (beta,))
        if gate and up:
            # Keep both views: ``mlp.up_gate`` describes a fused logical
            # workload, while llama.cpp's CPU graph executes the two physical
            # matrices as separate MUL_MAT invocations.  Exposing the
            # single-segment descriptors lets a runtime with explicit
            # physical-call evidence lower those invocations without guessing
            # or splitting a byte count heuristically.
            add_projection("mlp.gate", (gate,), segment_ids=("mlp-up_gate-0",))
            add_projection("mlp.up", (up,), segment_ids=("mlp-up_gate-1",))
            add_projection("mlp.up_gate", (gate, up),
                           segment_ids=("mlp-up_gate-0", "mlp-up_gate-1"))
        elif up: add_projection("mlp.up", (up,))
        add_projection("mlp.down", (down,))
        if projections:
            metadata["weight_projection_descriptors"] = {"schema_version": "heterollm.weight-projections/v1", "projections": projections}
        if is_qwen35:
            mixer = "full_attention" if qkv is None else "linear_attention"
            la = linear_attention if mixer == "linear_attention" else None
            metadata["qwen35_block_kind"] = mixer
            metadata["qwen35_full_attention_interval"] = gguf.metadata.get(f"{qwen35_metadata_prefix}.full_attention_interval")
        else:
            mixer, la = "full_attention", None
        layers.append(LayerSpec(layer_id=f"layer-{index:03d}", kind="dense", hidden_size=hidden, intermediate_size=intermediate,
                                attention_heads=int(gguf.n_head), kv_heads=int(gguf.n_head_kv), attention_head_dim=int(resolved_head_dim or 0),
                                sequence_mixer=mixer, linear_attention=la, dtype="fp16", weight_bytes=sum(t.n_bytes for t in tensors), metadata=metadata))
    embedding = next((t for t in gguf.tensor_directory if "token_embd.weight" in t.name or "embed_tokens.weight" in t.name), None)
    if embedding is None: raise GGUFError("GGUF embedding tensor is missing")
    output = next((t for t in gguf.tensor_directory if t.name.lower() in {"output.weight", "lm_head.weight"}), None)
    # Several tied-embedding exports (including Qwen3.5 GGUFs from Unsloth)
    # omit a separate output/lm_head tensor.  llama.cpp reuses token_embd for
    # the final projection in that case, so bind the same physical tensor
    # instead of rejecting an otherwise valid model.
    output_norm = next((t for t in gguf.tensor_directory if t.name == "output_norm.weight"), None)
    # Missing output/lm_head means a tied logical matrix in llama.cpp.  A
    # present tensor is independent unless it aliases the embedding payload at
    # the same GGUF offset and byte length.
    output_tied_to_embedding = output is None
    if output_tied_to_embedding:
        output = embedding
    elif (output.offset, output.n_bytes) == (embedding.offset, embedding.n_bytes):
        output_tied_to_embedding = True
    tie_values: list[tuple[str, bool]] = []
    tie_keys = [
        *(f"{prefix}.tie_word_embeddings" for prefix in _gguf_architecture_prefixes(gguf)),
        "general.tie_word_embeddings",
    ]
    for key in tie_keys:
        if key not in gguf.metadata:
            continue
        raw = gguf.metadata[key]
        if not isinstance(raw, bool):
            raise GGUFError(f"GGUF {key} must be boolean")
        tie_values.append((key, raw))
    if tie_values and any(value != tie_values[0][1] for _, value in tie_values[1:]):
        raise GGUFError("GGUF tie_word_embeddings metadata conflicts across namespaces")
    tie_metadata = tie_values[0][1] if tie_values else None
    if tie_metadata is True and not output_tied_to_embedding:
        raise GGUFError("GGUF tie_word_embeddings metadata conflicts with independent output.weight")
    if tie_metadata is False and output_tied_to_embedding:
        raise GGUFError("GGUF tie_word_embeddings=false but output.weight is absent")
    graph_architecture = adapter.graph_family if is_qwen35 else gguf.architecture
    graph = build_model_graph_from_layer_specs("GGUF-" + gguf.architecture, layers, architecture=graph_architecture,
        vocabulary_size=int(gguf.vocab_size), max_sequence_length=int(gguf.context_length), embedding_weight_bytes=embedding.n_bytes,
        output_weight_bytes=output.n_bytes,
        tie_word_embeddings=bool(output_tied_to_embedding),
        # Keep the file-level label under an audit-only key.  The planner's
        # artifact dispatcher consumes per-segment formats from
        # weight_projection_descriptors; feeding a mixed file label such as
        # Q4_K_M into its single-format registry would incorrectly reject the
        # otherwise valid mixed tensor graph.
        metadata={"gguf_sha256": gguf.sha256, "gguf_file_quantization": gguf.quantization, "gguf_tensor_count": len(gguf.tensor_directory),
                  # This is the simulator equivalent of llama.cpp's
                  # architecture enum.  Keep both the implementation family
                  # and the raw GGUF ID in the artifact for auditability.
                  "gguf_architecture_adapter": adapter.adapter_id,
                  "gguf_architecture_id": gguf.architecture,
                  # Keep the native loading-unit geometry beside the executable
                  # graph.  llama.cpp counts nextn/MTP blocks in -ngl while the
                  # simulator intentionally keeps them out of the main graph.
                  # This is evidence for an explicit mapping, never a reason to
                  # silently clamp an arbitrary gpu_layers value.
                  "gguf_declared_block_count": gguf.n_layer_all,
                  "gguf_imported_executable_layers": gguf.n_layer,
                  "gguf_mtp_layer_count": gguf.n_layer_nextn,
                  "gguf_output_tied_to_embedding": output_tied_to_embedding,
                  "gguf_physical_weight_bytes": _unique_tensor_bytes(gguf.tensor_directory),
                  # Keep graph-level bindings compact; tensor directory owns
                  # block geometry and the planner must not interpret these
                  # audit fields as one global artifact format.
                  "gguf_embedding_binding": {k: v for k, v in binding(embedding).items() if k != "block_size"},
                  "gguf_output_binding": {k: v for k, v in binding(output).items() if k != "block_size"},
                  # Static input only; these bytes already belong to the model
                  # artifact and must not create a second capacity allocation.
                  "gguf_output_norm_binding": ({k: v for k, v in binding(output_norm).items() if k != "block_size"}
                                               if output_norm is not None else None),
                  "gguf_norm_epsilon": gguf.metadata.get(
                      f"{qwen35_metadata_prefix if is_qwen35 else gguf.architecture}.attention.layer_norm_rms_epsilon")})
    return ModelSpec(name="GGUF-" + gguf.architecture, graph=graph, metadata=graph.attributes)


def build_model_from_gguf(gguf: GGUFMetadata):
    """Select the GGUF graph implementation through the architecture registry.

    ``_build_model_from_gguf_registered`` performs the exact metadata lookup
    and fail-closed validation.  Keeping this public entry point as a thin
    call avoids a second, competing architecture registry.
    """

    return _build_model_from_gguf_registered(gguf)


__all__ = ["GGUFError", "GGUFTensor", "GGUFMetadata", "GGUF_METADATA_CACHE_SCHEMA",
           "GGUF_METADATA_PARSER_SOURCE", "GGUF_MODEL_PRESET_SCHEMA",
           "read_gguf_metadata", "read_gguf_metadata_only",
           "write_gguf_metadata_cache", "read_gguf_metadata_cache", "gguf_metadata_digest",
           "build_gguf_model_preset", "import_gguf_model_preset", "compare_gguf_to_model",
           "assert_gguf_parity", "build_model_from_gguf"]
