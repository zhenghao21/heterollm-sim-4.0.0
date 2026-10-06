import struct

from heterollm_sim import import_gguf_model_preset
from heterollm_sim.schema_v1 import model_graph_from_dict


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<II", 4, value)


def test_generic_preset_keeps_unknown_tensor_types_opaque(tmp_path):
    metadata = b"".join(
        (
            _kv_string("general.architecture", "qwen35"),
            _kv_u32("general.alignment", 32),
            _kv_u32("qwen35.block_count", 1),
            _kv_u32("qwen35.embedding_length", 5120),
            _kv_u32("qwen35.attention.head_count", 24),
            _kv_u32("qwen35.attention.head_count_kv", 4),
            _kv_u32("qwen35.context_length", 262144),
            _kv_u32("general.vocab_size", 248320),
        )
    )
    tensor = _gguf_string("opaque.weight") + struct.pack("<I", 1)
    tensor += struct.pack("<Q", 1) + struct.pack("<IQ", 999, 0)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 8)
    path = tmp_path / "qwen3.8-27b.gguf"
    directory_end = len(header) + len(metadata) + len(tensor)
    padding = b"\x00" * ((32 - directory_end % 32) % 32)
    path.write_bytes(header + metadata + tensor + padding + b"\x00\x01\x02\x03")

    preset = import_gguf_model_preset(path)

    assert preset["preset"]["support_level"] == "metadata_only"
    assert preset["preset"]["source_sha"]
    assert preset["evidence"]["physical_tensor_bytes"] == "partial"
    assert preset["weights"]["opaque_tensor_count"] == 1
    assert preset["weights"]["tensor_inventory"][0]["bytes_status"] == "opaque"
    assert preset["graph"]["executable"] is False
    graph = preset["graph"]
    assert [item["op_kind"] for item in graph["operators"]] == [
        "model_input", "unsupported_component_group", "model_output"
    ]
    assert graph["operators"][1]["parameters"]["tensor_count"] == 1
    assert graph["operators"][1]["attributes"]["unsupported"] is True
    assert len(graph["operators"][1]["ports"]) == 2
    assert graph["attributes"]["execution_graph"] == "unproven"
    assert graph["attributes"]["symbols"]["V"] == 248320
    model_graph_from_dict(graph)
