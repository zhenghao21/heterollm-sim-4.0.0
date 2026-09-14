"""Storage-only Query coverage for fused attention."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.cost_models import (
    FusedAttentionKVPhysicalContract,
    FusedAttentionWorkload,
    estimate_gpu_fused_attention,
)
from heterollm_sim.reference import build_reference_scenario


def _profiles():
    scenario = build_reference_scenario()
    return (
        scenario.resolve_component_profile("gpu0"),
        scenario.resolve_component_profile("hbm0"),
    )


def _workload(**updates):
    values = {
        "batch_tokens": 2,
        "context_tokens": 7,
        "hidden_size": 16,
        "input_bits": 16,
        "output_bits": 16,
        "kv_hidden_size": 4,
        "kv_input_bits": 8,
        "kv_read_tokens": 11,
    }
    values.update(updates)
    return FusedAttentionWorkload(**values)


class AttentionQueryStorageTests(unittest.TestCase):
    def test_default_is_legacy_and_f32_query_only_changes_backing_query_read(self):
        legacy = _workload()
        f32_query = replace(legacy, query_storage_bits=32)
        gpu, hbm = _profiles()
        legacy_estimate = estimate_gpu_fused_attention(gpu, hbm, legacy)
        f32_estimate = estimate_gpu_fused_attention(gpu, hbm, f32_query)

        self.assertIsNone(legacy.query_storage_bits)
        self.assertEqual(legacy.effective_query_storage_bits, 16)
        self.assertEqual(legacy.query_read_bytes, 64)
        self.assertEqual(f32_query.query_read_bytes, 128)
        self.assertEqual(f32_query.kv_read_bytes, legacy.kv_read_bytes)
        self.assertEqual(f32_query.write_bytes, legacy.write_bytes)
        self.assertEqual(
            f32_query.onchip_working_set_bytes,
            legacy.onchip_working_set_bytes,
        )
        self.assertEqual(
            f32_estimate.metadata["effective_query_storage_bits"], 32
        )
        self.assertNotIn("effective_query_storage_bits", legacy_estimate.metadata)
        self.assertEqual(
            f32_estimate.metadata["tensor_service_ns"],
            legacy_estimate.metadata["tensor_service_ns"],
        )

    def test_query_storage_bits_rejects_non_positive_or_non_integer_values(self):
        for value in (0, -1, True, False, 16.0, "32"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "query_storage_bits"
            ):
                _workload(query_storage_bits=value)

    def test_q4_materialized_attention_preserves_f32_query_and_kv_contract(self):
        q4 = _workload(
            batch_tokens=3,
            context_tokens=7,
            hidden_size=256,
            kv_hidden_size=128,
            kv_input_bits=4,
            kv_read_tokens=7,
            kv_physical_contract=FusedAttentionKVPhysicalContract(
                payload_bytes_per_token=128,
                metadata_bytes_per_token=16,
                dequant_operations_per_token=256,
                artifact_format="Q4_0",
            ),
            q4_mma_view_tokens_lower_bound=256,
            q4_mma_head_dim=64,
            query_storage_bits=32,
        )
        legacy = replace(q4, query_storage_bits=None)
        gpu, hbm = _profiles()
        estimate = estimate_gpu_fused_attention(gpu, hbm, q4)

        self.assertEqual(q4.query_read_bytes, 3 * 256 * 4)
        self.assertEqual(q4.kv_read_bytes, legacy.kv_read_bytes)
        self.assertEqual(q4.kv_payload_bytes, legacy.kv_payload_bytes)
        self.assertEqual(q4.kv_metadata_bytes, legacy.kv_metadata_bytes)
        self.assertEqual(estimate.metadata["query_read_bytes"], 3 * 256 * 4)
        self.assertEqual(estimate.metadata["effective_query_storage_bits"], 32)
        self.assertTrue(estimate.metadata["q4_mma_materialization_applied"])


if __name__ == "__main__":
    unittest.main()
