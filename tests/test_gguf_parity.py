import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from hashlib import sha256
from pathlib import Path

from heterollm_sim.config import model_from_dict
from heterollm_sim.gguf_parity import (
    GGUFError,
    assert_gguf_parity,
    compare_gguf_to_model,
    read_gguf_metadata,
    build_model_from_gguf,
)
from heterollm_sim.model_presets import materialize_model_payload


def _str(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key, typ, value):
    return _str(key) + struct.pack("<I", typ) + value


class GGUFParityTests(unittest.TestCase):
    def _fixture(self, *, layers=24, vocab=151936):
        items = [
            _kv("general.architecture", 8, _str("qwen2")),
            _kv("general.file_type", 4, struct.pack("<I", 15)),
            _kv("qwen2.block_count", 4, struct.pack("<I", layers)),
            _kv("qwen2.embedding_length", 4, struct.pack("<I", 896)),
            _kv("qwen2.attention.head_count", 4, struct.pack("<I", 14)),
            _kv("qwen2.attention.head_count_kv", 4, struct.pack("<I", 2)),
            _kv("qwen2.context_length", 4, struct.pack("<I", 32768)),
            _kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, vocab) + b"".join(_str("x") for _ in range(vocab))),
        ]
        return b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)) + b"".join(items)

    def test_reads_header_geometry_hash_and_quantization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.gguf"
            path.write_bytes(self._fixture())
            got = read_gguf_metadata(path)
        self.assertEqual(got.architecture, "qwen2")
        self.assertEqual(got.n_layer, 24)
        self.assertEqual(got.n_embd, 896)
        self.assertEqual(got.n_head_kv, 2)
        self.assertEqual(got.vocab_size, 151936)
        # llama.h's ``llama_ftype`` enum used by the native baseline encodes
        # file type 15 as Q4_K_M; this is distinct from ggml tensor type 15
        # (Q8_K), which is validated separately below.
        self.assertEqual(got.quantization, "MOSTLY_Q4_K_M")
        self.assertEqual(len(got.sha256), 64)

    def test_metadata_and_hash_use_one_open_handle(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.gguf"
            payload = self._fixture(vocab=8)
            path.write_bytes(payload)
            original_open = Path.open
            calls = []
            def tracked_open(target, *args, **kwargs):
                calls.append(target)
                return original_open(target, *args, **kwargs)
            with patch.object(Path, "open", tracked_open):
                got = read_gguf_metadata(path)
            self.assertEqual(calls, [path])
            self.assertEqual(got.sha256, sha256(payload).hexdigest())

    def test_changed_handle_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.gguf"
            path.write_bytes(self._fixture(vocab=8))
            original = path.stat()
            changed = SimpleNamespace(**{key: getattr(original, key) for key in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")})
            changed.st_mtime_ns += 1
            with patch("heterollm_sim.gguf_parity.os.fstat", side_effect=[original, changed]):
                with self.assertRaisesRegex(GGUFError, "changed during metadata/hash"):
                    read_gguf_metadata(path)

    def test_short_hash_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.gguf"
            path.write_bytes(self._fixture(vocab=8))
            original = path.stat()
            enlarged = SimpleNamespace(**{key: getattr(original, key) for key in
                ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")})
            enlarged.st_size += 1
            with patch("heterollm_sim.gguf_parity.os.fstat", return_value=enlarged):
                with self.assertRaisesRegex(GGUFError, "bytes_read="):
                    read_gguf_metadata(path)

    def test_geometry_match_passes_and_layer_mismatch_fails_closed(self):
        model = model_from_dict(materialize_model_payload("qwen2_5-0_5b"))
        with tempfile.TemporaryDirectory() as td:
            good = Path(td) / "good.gguf"; good.write_bytes(self._fixture())
            bad = Path(td) / "bad.gguf"; bad.write_bytes(self._fixture(layers=23))
            ok = compare_gguf_to_model(read_gguf_metadata(good), model)
            mismatch = compare_gguf_to_model(read_gguf_metadata(bad), model)
        self.assertTrue(ok["ok"], ok)
        self.assertFalse(mismatch["ok"])
        self.assertIn("n_layer", {item["field"] for item in mismatch["mismatches"]})
        with self.assertRaises(ValueError):
            assert_gguf_parity(mismatch)

    def test_qwen35_hybrid_gguf_binds_linear_and_full_blocks(self):
        path = Path("artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf")
        if not path.exists():
            self.skipTest("Qwen3.8-27B GGUF unavailable")
        gguf = read_gguf_metadata(path)
        model = build_model_from_gguf(gguf)
        report = compare_gguf_to_model(gguf, model, context_length=512)
        self.assertTrue(report["ok"], report)
        # qwen35 block_count includes one nextn/MTP head; llama.cpp's trunk
        # graph exposes n_layer() == n_layer_all - n_layer_nextn.
        self.assertEqual(model.num_layers, 64)
        mixers = [item.layer.sequence_mixer for item in model._execution_view.layer_instances]
        self.assertEqual(mixers.count("linear_attention"), 48)
        self.assertEqual(mixers.count("full_attention"), 16)
        self.assertEqual(model._execution_view.layer_instances[0].layer.linear_attention.value_heads, 48)
        self.assertEqual(model._execution_view.layer_instances[0].layer.linear_attention.key_heads, 16)
        projections = model._execution_view.layer_instances[0].layer.metadata["weight_projection_descriptors"]["projections"]
        self.assertTrue({"linear_attention.qkv", "linear_attention.output", "linear_attention.output_gate"}.issubset(projections))
        self.assertTrue({"mlp.gate", "mlp.up", "mlp.up_gate"}.issubset(projections))
        self.assertEqual(projections["mlp.gate"]["segments"][0]["physical_bytes"], 38297600)
        self.assertEqual(projections["mlp.up"]["segments"][0]["physical_bytes"], 38297600)

    def test_qwen35_tensor_enum_and_block_bytes_match_ggml(self):
        path = Path("artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf")
        if not path.exists():
            self.skipTest("Qwen3.8-27B GGUF unavailable")
        gguf = read_gguf_metadata(path)
        tensors = {tensor.name: tensor for tensor in gguf.tensor_directory}
        # ggml.h: 21=IQ3_S (110 bytes/256 weights), 23=IQ4_XS
        # (136 bytes/256), 20=IQ4_NL (18 bytes/32), 13=Q5_K (176/256).
        self.assertEqual((tensors["blk.0.ffn_gate.weight"].type_id,
                          tensors["blk.0.ffn_gate.weight"].type_name,
                          tensors["blk.0.ffn_gate.weight"].n_bytes), (21, "IQ3_S", 38297600))
        self.assertEqual((tensors["blk.0.attn_gate.weight"].type_id,
                          tensors["blk.0.attn_gate.weight"].type_name,
                          tensors["blk.0.attn_gate.weight"].n_bytes), (23, "IQ4_XS", 16711680))
        self.assertEqual(tensors["blk.0.ssm_alpha.weight"].type_name, "IQ4_XS")

    def test_qwen35_tied_embedding_binds_missing_output_tensor(self):
        path = Path("artifacts/multimodel_20260913/models/Qwen3.5-0.8B-Q4_K_M.gguf")
        if not path.exists():
            self.skipTest("Qwen3.5-0.8B GGUF unavailable")
        gguf = read_gguf_metadata(path)
        self.assertFalse(any(t.name.lower() in {"output.weight", "lm_head.weight"}
                             for t in gguf.tensor_directory))
        model = build_model_from_gguf(gguf)
        graph_metadata = model.metadata["metadata"]
        self.assertTrue(graph_metadata["gguf_output_tied_to_embedding"])
        self.assertEqual(model.output_weight_bytes, model.embedding_weight_bytes)

    def test_invalid_magic_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.gguf"; path.write_bytes(b"nope")
            with self.assertRaises(GGUFError):
                read_gguf_metadata(path)

    def test_bundled_qwen_tensor_directory_binds_physical_weights(self):
        path = Path("artifacts/native_benchmark_20260912/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
        if not path.exists():
            self.skipTest("bundled benchmark GGUF unavailable")
        gguf = read_gguf_metadata(path)
        model = build_model_from_gguf(gguf)
        self.assertEqual(model.num_layers, gguf.n_layer)
        self.assertEqual(model.max_sequence_length, gguf.context_length)
        # GGUF matrices use [input, output]; the FFN width must come from the
        # projection output dimension (Qwen2.5-0.5B is 4864), not hidden size.
        self.assertEqual(
            model._execution_view.layer_instances[0].layer.intermediate_size,
            gguf.metadata["qwen2.feed_forward_length"],
        )
        graph_metadata = model.metadata["metadata"]
        self.assertEqual(graph_metadata["gguf_embedding_binding"]["n_bytes"], model.embedding_weight_bytes)
        self.assertEqual(graph_metadata["gguf_output_binding"]["n_bytes"], model.output_weight_bytes)
        self.assertGreaterEqual(len(gguf.tensor_directory), 290)
        for descriptor in model._execution_view.layer_instances:
            metadata = descriptor.layer.metadata
            bindings = metadata["gguf_tensor_bindings"]
            self.assertEqual(metadata["gguf_physical_weight_bytes"], sum(item["n_bytes"] for item in bindings))
            projections = metadata["weight_projection_descriptors"]["projections"]
            bound = {item["name"]: item["n_bytes"] for item in bindings}
            for projection in projections.values():
                self.assertEqual(
                    sum(segment["physical_bytes"] for segment in projection["segments"]),
                    sum(bound[segment["physical_tensor_name"]] for segment in projection["segments"]),
                )
