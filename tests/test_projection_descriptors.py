import math
import unittest

from heterollm_sim import planner
from heterollm_sim.projection_descriptors import (
    ARTIFACT_QUANTIZATION_REGISTRY,
    WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA,
    canonical_artifact_quantization,
    materialize_weight_projection,
)


def test_iq3_s_compute_bits_match_ggml_family():
    # IQ3_S is a 3-bit importance-matrix format (3.4375 bpw on disk), not a
    # 4-bit kernel artifact.  The distinction feeds bandwidth/dequant costs.
    assert ARTIFACT_QUANTIZATION_REGISTRY["IQ3_S"].compute_weight_bits == 3


_OBSERVED_TENSORS = {
    # From artifacts/bionic_generalization_20260905/sources/qwen05_gguf.json.
    "qwen05_ffn_down": {
        "allocated_bytes": 3_575_040,
        "dimensions": (4864, 896),
        "name": "blk.0.ffn_down.weight",
    },
    "qwen05_ffn_gate": {
        "allocated_bytes": 2_996_224,
        "dimensions": (896, 4864),
        "name": "blk.0.ffn_gate.weight",
    },
    # From artifacts/bionic_generalization_20260905/sources/smol17_gguf.json.
    "smol17_attn_q": {
        "allocated_bytes": 2_359_296,
        "dimensions": (2048, 2048),
        "name": "blk.0.attn_q.weight",
    },
}


def _descriptor(tensor, format_name, physical_bytes=None):
    k, n = tensor["dimensions"]
    return {
        "weight_projection_descriptors": {
            "schema_version": WEIGHT_PROJECTION_DESCRIPTOR_SCHEMA,
            "projections": {
                "p": {
                    "segments": [
                        {
                            "segment_id": "s",
                            "physical_tensor_name": tensor["name"],
                            "k": k,
                            "n": n,
                            "format": format_name,
                            "physical_bytes": (
                                tensor["allocated_bytes"]
                                if physical_bytes is None
                                else physical_bytes
                            ),
                            "tp_shard_axis": "replicated",
                        }
                    ]
                }
            },
        }
    }


class PackedProjectionDescriptorTests(unittest.TestCase):
    def test_registry_matches_real_packed_inventory_bytes(self):
        cases = (
            (
                _OBSERVED_TENSORS["qwen05_ffn_down"],
                "Q6_K",
                (256, 192, 18, 6),
            ),
            (
                _OBSERVED_TENSORS["qwen05_ffn_gate"],
                "Q5_0",
                (32, 20, 2, 5),
            ),
            (
                _OBSERVED_TENSORS["smol17_attn_q"],
                "Q4_K",
                (256, 128, 16, 4),
            ),
        )
        for tensor, format_name, contract in cases:
            with self.subTest(tensor=tensor["name"], format=format_name):
                k, n = tensor["dimensions"]
                spec = ARTIFACT_QUANTIZATION_REGISTRY[format_name]
                self.assertEqual(
                    (
                        spec.block_size,
                        spec.payload_bytes,
                        spec.metadata_bytes,
                        spec.compute_weight_bits,
                    ),
                    contract,
                )
                blocks = n * int(math.ceil(k / float(spec.block_size)))
                self.assertEqual(
                    tensor["allocated_bytes"],
                    blocks * (spec.payload_bytes + spec.metadata_bytes),
                )

                projection = materialize_weight_projection(
                    _descriptor(tensor, format_name),
                    "p",
                )

                self.assertEqual(
                    projection.weight_storage_bytes,
                    blocks * spec.payload_bytes,
                )
                self.assertEqual(
                    projection.weight_metadata_bytes,
                    blocks * spec.metadata_bytes,
                )
                self.assertEqual(
                    projection.full_physical_bytes,
                    tensor["allocated_bytes"],
                )
                self.assertEqual(projection.weight_bits, spec.compute_weight_bits)
                self.assertEqual(
                    projection.fused_dequant_operations,
                    blocks * spec.block_size,
                )

    def test_mislabeled_physical_bytes_are_rejected(self):
        tensor = _OBSERVED_TENSORS["smol17_attn_q"]
        packed_payload_only = 2048 * 8 * 128

        with self.assertRaisesRegex(ValueError, "Q4_K block contract 2359296"):
            materialize_weight_projection(
                _descriptor(tensor, "Q4_K", packed_payload_only),
                "p",
            )

    def test_q4_k_m_is_not_a_single_primitive_format(self):
        tensor = _OBSERVED_TENSORS["smol17_attn_q"]

        self.assertIsNone(canonical_artifact_quantization("Q4_K_M"))
        with self.assertRaisesRegex(ValueError, "unsupported format Q4_K_M"):
            materialize_weight_projection(
                _descriptor(tensor, "Q4_K_M"),
                "p",
            )

    def test_planner_uses_descriptor_aliases_for_primitive_formats(self):
        self.assertEqual(
            planner._artifact_label_from_metadata(
                None,
                ({"weight_format": "q8-0"},),
            ),
            "Q8_0",
        )

        with self.assertRaisesRegex(
            ValueError,
            "unsupported artifact quantization Q4_K_M",
        ):
            planner._artifact_label_from_metadata(
                None,
                ({"quantization": "Q4_K_M"},),
            )


if __name__ == "__main__":
    unittest.main()
