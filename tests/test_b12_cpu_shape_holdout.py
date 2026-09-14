import unittest

from tools.audit_cpu_shape_holdout_b12 import _cpu, _estimate, _workload


class B12CpuShapeHoldoutTests(unittest.TestCase):
    def test_generic_dequant_scales_with_m_k_n(self):
        cpu = _cpu(capability=False)
        for axis, first, second in (
            ("m", _workload(1, 256, 16), _workload(2, 256, 16)),
            ("k", _workload(4, 256, 16), _workload(4, 512, 16)),
            ("n", _workload(4, 512, 16), _workload(4, 512, 32)),
        ):
            with self.subTest(axis=axis):
                a = _estimate(cpu, first)
                b = _estimate(cpu, second)
                self.assertEqual(
                    b["packed_weight_transform_instructions"],
                    2 * a["packed_weight_transform_instructions"],
                )
                self.assertGreater(b["auxiliary_instructions"], 0)

    def test_generic_path_counts_dequant_once_and_capability_gate_is_fail_closed(self):
        generic = _estimate(_cpu(capability=False), _workload(8, 512, 32))
        self.assertEqual(generic["quantized_format_coverage"], "generic_quantized_fallback")
        self.assertEqual(
            generic["auxiliary_instructions"],
            generic["packed_weight_transform_issue_instructions"],
        )
        bounded = _estimate(_cpu(capability=True, maximum_m=4), _workload(8, 512, 32))
        self.assertEqual(bounded["quantized_format_coverage"], "generic_quantized_fallback")

    def test_declared_isa_path_uses_source_blocks(self):
        row = _estimate(_cpu(capability=True), _workload(8, 512, 32))
        self.assertEqual(row["quantized_format_coverage"], "declared_quantized_capability")
        self.assertEqual(row["source_dot_blocks"], 8 * 32 * 2)

    def test_generic_schedule_accounts_each_instruction_once(self):
        row = _estimate(_cpu(capability=False), _workload(8, 512, 32))
        self.assertEqual(
            row["total_instructions"],
            row["compute_instructions"]
            + row["auxiliary_instructions"]
            + row["special_function_instructions"]
            + row["load_instructions"]
            + row["store_instructions"],
        )

    def test_thread_budget_reduces_or_preserves_service(self):
        rows = [
            _estimate(_cpu(capability=False, thread_count=t), _workload(8, 512, 32))
            for t in (1, 2, 4)
        ]
        self.assertGreaterEqual(rows[0]["service_ns"], rows[1]["service_ns"])
        self.assertGreaterEqual(rows[1]["service_ns"], rows[2]["service_ns"])


if __name__ == "__main__":
    unittest.main()
