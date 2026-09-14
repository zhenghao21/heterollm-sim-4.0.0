"""Opt-in physical-row dispatch retained from the round-eight Q5_K audit."""

from dataclasses import replace
import unittest

from heterollm_sim.config import _gpu_quantized_matmul_capability_from_dict
from heterollm_sim.cost_models import GemmWorkload, HBMProfile, estimate_gpu_gemm
from heterollm_sim.serde import to_primitive
from tests.test_cost_models import gpu_profile, gpu_quantized_matmul_capability


class CombinedCostRefinementTests(unittest.TestCase):
    def test_q5_dispatch_uses_physical_rows_and_preserves_other_work(self):
        capability = replace(
            gpu_quantized_matmul_capability(supported_weight_formats=("Q5_K",)),
            min_m=6,
        )
        baseline = gpu_profile()
        enabled = replace(baseline, quantized_matmul_capabilities=(capability,))
        memory = HBMProfile(bandwidth_gb_s=1000)
        for rows, formats, expected_dtype in (
            (1, ("Q5_K",), "fp16"),
            (5, ("Q5_K",), "fp16"),
            (6, ("Q5_K",), "int8"),
            (128, ("Q5_K",), "int8"),
            (128, ("Q5_K", "IQ4_XS"), "fp16"),
        ):
            with self.subTest(rows=rows, formats=formats):
                workload = GemmWorkload(
                    m=rows, k=5120, n=10240, activation_bits=16, weight_bits=5,
                    packed_weight_formats=formats, weight_storage_bytes=36044800,
                )
                before = estimate_gpu_gemm(baseline, memory, workload)
                after = estimate_gpu_gemm(enabled, memory, workload)
                self.assertEqual(after.metadata["tensor_dtype"], expected_dtype)
                for key in (
                    "activation_bytes", "weight_bytes", "output_bytes",
                    "issued_operations", "hbm_bandwidth",
                ):
                    self.assertEqual(before.metadata[key], after.metadata[key])
                if expected_dtype == "fp16":
                    self.assertEqual(before, after)
                else:
                    self.assertEqual(
                        after.metadata["quantized_matmul_capability"]["min_m"], 6
                    )

    def test_optional_dispatch_contract_round_trips_and_validates(self):
        capability = gpu_quantized_matmul_capability()
        self.assertEqual(capability.min_m, 1)
        legacy_payload = to_primitive(capability)
        legacy_payload.pop("min_m")
        self.assertEqual(
            _gpu_quantized_matmul_capability_from_dict(legacy_payload), capability
        )
        payload = dict(legacy_payload, min_m=6)
        self.assertEqual(
            to_primitive(_gpu_quantized_matmul_capability_from_dict(payload)),
            payload,
        )
        for invalid in (0, -1, True, 1.5, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _gpu_quantized_matmul_capability_from_dict(
                    dict(legacy_payload, min_m=invalid)
                )


if __name__ == "__main__":
    unittest.main()
