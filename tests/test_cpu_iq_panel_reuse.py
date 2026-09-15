"""Structural IQ panel tests. Standard library only; never native or full sim."""
from __future__ import annotations
import unittest
from dataclasses import replace

from heterollm_sim import cost_models as model
from heterollm_sim.reference import build_reference_scenario


def workload(m=64, n=17408, k=5120, fmt="IQ4_XS", bits=16):
    return model.GemmWorkload(m=m, n=n, k=k, activation_bits=bits, weight_bits=4 if fmt != "IQ3_S" else 3,
        output_bits=bits, packed_weight_formats=(fmt,), packed_weight_transform_operations=n*k,
        packed_weight_format_segments=((fmt, n, n*k),), weight_storage_bytes=n*((k+255)//256)*136)


def contract(**changes):
    value = model.CPUIQPanelDispatch(compiled_avx2=True, source_activation_dtype="F32", source_output_dtype="F32",
        source_weight_layout="ordinary_contiguous_2d", no_iq_panel_environment_state="unset",
        source_sha256="a"*64, cpu_backend_sha256="b"*64, source_refs=("synthetic-structural-test-not-a-native-measurement",))
    return replace(value, **changes)


def compute_phase(estimate):
    return next(p for p in estimate.phases if p.name == "cpu_gemm")


def schedule(estimate):
    return compute_phase(estimate).metadata["instruction_schedule"]


def demand_signature(estimate):
    return [(p.name, [(d.resource_id, d.service_ns, d.energy_pj) for d in p.demands]) for p in estimate.phases]


class PanelReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref = build_reference_scenario()
        cpu = ref.component_profiles["cpu"]["legacy-cpu"]
        cls.cpu = replace(cpu, pipeline=replace(cpu.pipeline, core_count=16, frequency_ghz=4.3))
        cls.memory = replace(ref.component_profiles["host_memory"]["legacy-host-memory"], bandwidth_gb_s=89.6)

    def estimate(self, w, dispatch=None, cpu=None):
        return model.estimate_cpu_gemm(cpu or self.cpu, self.memory, w, iq_panel_dispatch=dispatch)

    def assert_preserved(self, w, dispatch):
        original = self.estimate(w)
        candidate = self.estimate(w, dispatch)
        self.assertEqual(demand_signature(original), demand_signature(candidate))
        self.assertFalse(schedule(candidate)["iq_panel_weight_reuse"]["applied"])
        return schedule(candidate)["iq_panel_weight_reuse"]

    def test_no_contract_preserves_generic_transform_and_has_no_audit(self):
        for fmt in ("IQ3_S", "IQ4_XS"):
            for m in (1, 4, 7, 8, 64):
                with self.subTest(fmt=fmt, m=m):
                    w = workload(m=m, fmt=fmt)
                    estimate = self.estimate(w)
                    generic = schedule(estimate)
                    self.assertEqual(generic["packed_weight_transform_instructions"], m * w.packed_weight_transform_operations)
                    self.assertNotIn("iq_panel_weight_reuse", generic)

    def test_m1_m4_m7_never_change(self):
        for m in (1, 4, 7):
            audit = self.assert_preserved(workload(m=m), contract())
            self.assertIn("native_iq_panel_minimum_m_is_eight", audit["rejection_reasons"])

    def test_m8_m16_m64_transform_once_preserves_dot_and_conversion(self):
        for fmt in ("IQ3_S", "IQ4_XS"):
            for m in (8, 16, 64):
                with self.subTest(fmt=fmt, m=m):
                    w = workload(m=m, fmt=fmt)
                    a, b = self.estimate(w), self.estimate(w, contract())
                    sa, sb = schedule(a), schedule(b)
                    audit = sb["iq_panel_weight_reuse"]
                    self.assertTrue(audit["applied"])
                    self.assertEqual(audit["generic_transform_operations"], m*w.n*w.k)
                    self.assertEqual(audit["effective_transform_operations"], w.n*w.k)
                    self.assertEqual(audit["panel_superblock_decodes"], (w.n//8)*(w.k//256))
                    self.assertEqual(audit["dot_operations_unchanged"], 2*m*w.n*w.k)
                    self.assertEqual(audit["native_activation_conversion_elements_unchanged"], m*w.k)
                    for key in ("compute_instructions", "activation_quantization_blocks", "activation_quantization_instructions", "activation_quantization_accounted"):
                        self.assertEqual((key in sa, sa.get(key)), (key in sb, sb.get(key)), key)
                    self.assertLess(sb["service_ns"], sa["service_ns"])

    def test_storage_demands_do_not_expand_to_float_model(self):
        w = workload()
        a, b = compute_phase(self.estimate(w)), compute_phase(self.estimate(w, contract()))
        for key in ("read_bytes", "write_bytes", "memory_service_ns", "cache"):
            self.assertEqual(a.metadata[key], b.metadata[key])
        self.assertEqual([(d.resource_id, d.service_ns) for d in a.demands if d.resource_id != self.cpu.compute_resource_id],
                         [(d.resource_id, d.service_ns) for d in b.demands if d.resource_id != self.cpu.compute_resource_id])

    def test_scratch_is_one_local_panel_per_thread(self):
        records=[]
        for m,n in ((8,17408),(64,17408),(64,34816)):
            records.append(schedule(self.estimate(workload(m=m,n=n),contract()))["iq_panel_weight_reuse"])
        for r in records:
            self.assertEqual(r["scratch_bytes_per_thread"], (5120//256)*2240)
            self.assertEqual(r["scratch_total_bytes"], 16*(5120//256)*2240)
            self.assertFalse(r["scratch_new_backing_memory_demand_applied"])

    def test_unknown_history_is_not_unset(self):
        audit=self.assert_preserved(workload(),contract(no_iq_panel_environment_state="unknown"))
        self.assertIn("historical_ggml_no_iq_panel_unknown",audit["rejection_reasons"])
        self.assertFalse(audit["native_dispatch_proven"])

    def test_default_unset_assumption_is_explicit_conditional(self):
        r=schedule(self.estimate(workload(),contract(no_iq_panel_environment_state="unknown",assume_default_unset=True)))["iq_panel_weight_reuse"]
        self.assertTrue(r["applied"])
        self.assertTrue(r["default_unset_assumption_used"])
        self.assertFalse(r["native_dispatch_proven"])
        self.assertEqual(r["evaluation_scope"],"conditional_default_unset_ablation")

    def test_recorded_presence_never_overridden_by_assumption(self):
        for assume in (False,True):
            r=self.assert_preserved(workload(),contract(no_iq_panel_environment_state="set",assume_default_unset=assume))
            self.assertIn("ggml_no_iq_panel_present_even_if_value_is_zero",r["rejection_reasons"])

    def test_native_dtype_layout_isa_and_reference_counterexamples(self):
        cases=({"compiled_avx2":False},{"source_activation_dtype":"F16"},{"source_activation_dtype":"I32"},
               {"source_output_dtype":"BF16"},{"source_weight_layout":"strided_2d"},
               {"source_weight_layout":"contiguous_3d"},{"source_activation_ne3":2},{"use_reference_kernel":True})
        for changes in cases:
            with self.subTest(**changes): self.assert_preserved(workload(),contract(**changes))

    def test_non_aligned_n_and_k_unchanged(self):
        for n,k in ((17409,5120),(17408,5136),(17409,5136)):
            with self.subTest(n=n,k=k): self.assert_preserved(workload(n=n,k=k),contract())

    def test_unsupported_quantization_unchanged(self):
        for fmt in ("Q4_K","Q8_0","IQ4_NL","F16"):
            with self.subTest(fmt=fmt): self.assert_preserved(workload(fmt=fmt),contract())

    def test_mixed_formats_or_unaligned_physical_segments_unchanged(self):
        w=workload(n=16)
        mixed=replace(w,packed_weight_formats=("IQ3_S","IQ4_XS"),packed_weight_format_segments=(("IQ3_S",8,8*w.k),("IQ4_XS",8,8*w.k)))
        self.assert_preserved(mixed,contract())
        split=replace(w,packed_weight_format_segments=(("IQ4_XS",4,4*w.k),("IQ4_XS",12,12*w.k)))
        self.assert_preserved(split,contract())

    def test_missing_source_identity_unchanged(self):
        for changes in ({"source_refs":()},{"source_sha256":""},{"cpu_backend_sha256":"bad"}):
            with self.subTest(**changes): self.assert_preserved(workload(),contract(**changes))

    def test_no_throughput_constants_changed(self):
        w=workload();a=compute_phase(self.estimate(w));b=compute_phase(self.estimate(w,contract()))
        self.assertEqual(a.metadata["throughput_compute_service_ns"],b.metadata["throughput_compute_service_ns"])
        self.assertFalse(b.metadata["instruction_schedule"]["iq_panel_weight_reuse"]["throughput_constants_changed"])

    def test_does_not_stack_with_existing_quantized_dot_profile(self):
        cap=model.CPUQuantizedDotCapability(name="synthetic",supported_weight_formats=("IQ4_XS",),source_activation_bits=(16,),
            dot_activation_bits=8,dot_weight_bits=8,accumulator_bits=32,effective_ops_per_instruction=32.0,
            dot_issue_instructions_per_cycle_per_core=2.0,auxiliary_ops_per_instruction=32.0,
            activation_quantization_block_elements=256,evidence="synthetic structural test, not an old calibration")
        cpu=replace(self.cpu,quantized_dot_capabilities=(cap,));w=workload()
        a=self.estimate(w,cpu=cpu);b=self.estimate(w,contract(),cpu=cpu)
        self.assertEqual(demand_signature(a),demand_signature(b))
        self.assertIn("existing_quantized_dot_capability_outside_this_ablation",schedule(b)["iq_panel_weight_reuse"]["rejection_reasons"])

    def test_native_dtype_evidence_does_not_rewrite_logical_bit_widths(self):
        # Dispatch is qualified by native tensor dtypes; numeric cost widths
        # retain their baseline meanings in this one-mechanism ablation.
        for bits in (16,32):
            w=workload(bits=bits);a=self.estimate(w);b=self.estimate(w,contract())
            self.assertEqual(schedule(a)["compute_instructions"],schedule(b)["compute_instructions"])
            self.assertEqual(w.activation_bits,bits)
            self.assertEqual(w.output_bits,bits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
