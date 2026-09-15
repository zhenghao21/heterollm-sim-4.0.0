"""Planner coverage of explicit, source-bound CPU IQ panel reuse."""
from copy import deepcopy
from dataclasses import fields, replace
import unittest
from unittest.mock import patch

from heterollm_sim import planner
from heterollm_sim.cost_models import CPUIQPanelDispatch, FusedAttentionWorkload, GemmWorkload
from heterollm_sim.serde import to_primitive
from heterollm_sim.serving import BatchCohort, BatchItem
from tests.model_helpers import replace_model_layer_specs
from tests.test_mmq_planner import scenario as projection_scenario


CONTRACT = {
    "enabled": True, "compiled_avx2": True,
    "no_iq_panel_environment_state": "unknown", "assume_default_unset": True,
    "source_sha256": "a" * 64, "cpu_backend_sha256": "b" * 64,
    "source_refs": ["synthetic-source-contract-for-structural-tests"],
}


def scenario(*, enabled=True, tokens=8, fmt="IQ4_XS", f32=True, bindings=True):
    base = projection_scenario(enabled=None, tokens=tokens, weight_format=fmt)
    layers = []
    for layer in planner._execution_layers(base):
        metadata = deepcopy(layer.metadata)
        unique = {}
        for projection in metadata["weight_projection_descriptors"]["projections"].values():
            for seg in projection["segments"]:
                unique[seg["physical_tensor_name"]] = {
                    "name": seg["physical_tensor_name"], "shape": [seg["k"], seg["n"]],
                    "type": fmt, "block_size": 256, "n_bytes": seg["physical_bytes"], "offset": 0,
                }
        if bindings:
            metadata["gguf_tensor_bindings"] = list(unique.values())
        layers.append(replace(layer, metadata=metadata))
    model = replace_model_layer_specs(base.model, tuple(layers))
    model = replace(model, metadata={**model.metadata, "gguf_sha256": "c" * 64})
    flags = {**base.workload.metadata, "llama_cpp_f32_hidden_storage": f32}
    if enabled is not None:
        flags["llama_cpp_cpu_iq_panel_reuse"] = {**CONTRACT, "enabled": enabled}
    placement = replace(
        base.placement,
        op_to_component={key: "cpu0" for key in base.placement.op_to_component},
        tensor_to_component={"kv_cache": "hostmem0", "linear_state": "hostmem0"},
        parallel=base.placement.parallel,
        kv_policy=replace(base.placement.kv_policy, cache_component="hostmem0"),
    )
    return replace(base, model=model, workload=replace(base.workload, metadata=flags), placement=placement)


def contract_scenario(base, **changes):
    flags = dict(base.workload.metadata)
    flags["llama_cpp_cpu_iq_panel_reuse"] = {**flags["llama_cpp_cpu_iq_panel_reuse"], **changes}
    return replace(base, workload=replace(base.workload, metadata=flags))


def binding_scenario(base, **changes):
    layers = []
    for layer in planner._execution_layers(base):
        metadata = deepcopy(layer.metadata)
        for binding in metadata["gguf_tensor_bindings"]:
            binding.update(changes)
        layers.append(replace(layer, metadata=metadata))
    return replace(base, model=replace_model_layer_specs(base.model, tuple(layers)))


def physical_tasks(schedule):
    return [task for task in schedule.tasks if task.metadata.get("phase") == "cpu_gemm"]


def demand_signature(schedule):
    return [(task.task_id, task.dependencies, to_primitive(task.demands)) for task in schedule.tasks]


def direct_work(base, *, m=8, fmt="IQ4_XS"):
    spec = planner._ARTIFACT_QUANTIZATION_REGISTRY[fmt]
    k = n = 256
    payload, extra = n * spec.payload_bytes, n * spec.metadata_bytes
    workload = GemmWorkload(
        m=m, k=k, n=n, activation_bits=16, output_bits=16,
        weight_bits=spec.compute_weight_bits, weight_storage_bytes=payload, weight_metadata_bytes=extra,
        activation_storage_bytes=4*m*k, output_storage_bytes=4*m*n,
        packed_weight_formats=(fmt,), packed_weight_format_segments=((fmt, n, n*k),),
        packed_weight_transform_operations=n*k,
    )
    metadata = {
        "weight_projection_descriptor_applied": True, "projection_segment_count": 1,
        "projection_segments": [{
            "physical_tensor_name": "blk.0.attn_q.weight", "format": fmt,
            "global_k": k, "global_n": n, "local_k": k, "local_n": n,
            "physical_bytes": payload+extra, "local_physical_bytes": payload+extra,
        }],
    }
    return workload, metadata


class CPUIQPanelPlannerTests(unittest.TestCase):
    def qualify(self, base=None, workload=None, metadata=None, **flags):
        base = base or scenario()
        default_work, default_metadata = direct_work(base)
        return planner._llama_source_cpu_iq_panel_dispatch(
            base, workload or default_work, planner._component(base, "cpu0"),
            default_metadata if metadata is None else metadata,
            model_weight_read=flags.get("model_weight_read", True),
            rhs_is_activation=flags.get("rhs_is_activation", False),
        )

    def test_absent_false_and_missing_enabled_preserve_exact_tasks(self):
        absent = planner.compile_scenario(scenario(enabled=None))
        disabled_base = scenario(enabled=False)
        disabled = planner.compile_scenario(disabled_base)
        flags = {**disabled_base.workload.metadata, "llama_cpp_cpu_iq_panel_reuse": {}}
        unspecified = planner.compile_scenario(replace(disabled_base, workload=replace(disabled_base.workload, metadata=flags)))
        self.assertEqual([to_primitive(t) for t in absent.tasks], [to_primitive(t) for t in disabled.tasks])
        self.assertEqual([to_primitive(t) for t in absent.tasks], [to_primitive(t) for t in unspecified.tasks])
        self.assertFalse(any("cpu_iq_panel_reuse" in t.metadata for t in absent.tasks))
        self.assertNotIn("cpu_iq_panel_reuse", absent.manifest.metadata)

    def test_imported_iq_matrices_apply_conditionally_and_count_once(self):
        for fmt in ("IQ3_S", "IQ4_XS"):
            with self.subTest(fmt=fmt):
                compiled = planner.compile_scenario(scenario(fmt=fmt))
                mains = physical_tasks(compiled)
                applied = [t for t in mains if t.metadata["cpu_iq_panel_reuse"].get("applied")]
                self.assertEqual(len(applied), 5)
                # The fixture's K/V output storage is not 4*M*N; a global
                # F32 flag must never erase this per-invocation mismatch.
                rejected = [t.metadata["cpu_iq_panel_reuse"] for t in mains if not t.metadata["cpu_iq_panel_reuse"].get("applied")]
                self.assertEqual(len(rejected), 2)
                self.assertTrue(all(a["reason"] == "ordinary_f32_input_output_not_proven" for a in rejected))
                for task in applied:
                    audit = task.metadata["cpu_iq_panel_reuse"]
                    self.assertTrue(audit["default_unset_assumption_used"])
                    self.assertFalse(audit["native_dispatch_proven"])
                    self.assertEqual(audit["generic_transform_operations"], audit["m"] * audit["effective_transform_operations"])
                    self.assertEqual(audit["native_source_activation_dtype"], "F32")
                    self.assertEqual(audit["native_source_weight_layout"], "ordinary_contiguous_2d")
                self.assertTrue(all("cpu_iq_panel_reuse" not in t.metadata for t in compiled.tasks if t.metadata.get("phase") != "cpu_gemm"))
                summary = planner.summarize_cpu_iq_panel_reuse(compiled.tasks)
                self.assertEqual(summary, compiled.manifest.metadata["cpu_iq_panel_reuse"])
                self.assertEqual(summary["cpu_gemm_tasks"], len(mains))
                self.assertEqual(summary["audited_tasks"], len(mains))
                self.assertEqual(summary["applied_tasks"], len(applied))
                self.assertEqual(summary["conditional_tasks"], len(applied))
                self.assertEqual(summary["native_dispatch_proven_tasks"], 0)
                self.assertEqual(summary["uncovered_tasks"], len(mains)-len(applied))

    def test_missing_tensor_evidence_preserves_demands_and_reports_counts(self):
        base = scenario(bindings=False)
        compiled = planner.compile_scenario(base)
        original = planner.compile_scenario(contract_scenario(base, enabled=False))
        self.assertEqual(demand_signature(compiled), demand_signature(original))
        summary = compiled.manifest.metadata["cpu_iq_panel_reuse"]
        self.assertEqual(summary["applied_tasks"], 0)
        self.assertGreaterEqual(summary["uncovered_reason_counts"]["source_tensor_binding_missing_or_ambiguous"], 7)

    def test_f32_storage_contract_required(self):
        base = scenario(f32=False)
        compiled = planner.compile_scenario(base)
        original = planner.compile_scenario(contract_scenario(base, enabled=False))
        self.assertEqual(demand_signature(compiled), demand_signature(original))
        self.assertEqual(compiled.manifest.metadata["cpu_iq_panel_reuse"]["applied_tasks"], 0)
        self.assertIn("ordinary_f32_input_output_not_proven", compiled.manifest.metadata["cpu_iq_panel_reuse"]["uncovered_reason_counts"])

    def test_native_dispatch_contract_unknown_or_set_keeps_original_cost(self):
        for state, assume in (("unknown", False), ("set", False), ("set", True)):
            base = contract_scenario(scenario(), no_iq_panel_environment_state=state, assume_default_unset=assume)
            candidate = planner.compile_scenario(base)
            original = planner.compile_scenario(contract_scenario(base, enabled=False))
            self.assertEqual(demand_signature(candidate), demand_signature(original))
            self.assertEqual(candidate.manifest.metadata["cpu_iq_panel_reuse"]["applied_tasks"], 0)

    def test_source_model_and_source_tensor_counterexamples(self):
        base = scenario()
        for changes in ({"shape": [256,256,1]}, {"shape": [512,256]}, {"type": "Q4_K"},
                        {"n_bytes": 1}, {"offset": -1}, {"block_size": 32}):
            with self.subTest(changes=changes):
                dispatch, audit = self.qualify(binding_scenario(base, **changes))
                self.assertIsNone(dispatch)
                self.assertEqual(audit["status"], "uncovered")
        self.assertIsNotNone(self.qualify(base)[0])
        self.assertIsNone(self.qualify(replace(base, model=replace(base.model, metadata={})))[0])

    def test_dynamic_rhs_experts_fusion_and_shards_are_excluded(self):
        base = scenario()
        work, metadata = direct_work(base)
        cases = [{"dynamic_rhs": True}, {"rhs_operand_kind": "activation"}, {"expert_index": 0},
                 {"coverage_component": "routed_expert"}, {"ffn_path": "shared"},
                 {"fused": True}, {"fusion_group": "epilogue"}]
        for extra in cases:
            with self.subTest(extra=extra):
                self.assertIsNone(self.qualify(base, metadata={**metadata, **extra})[0])
        self.assertIsNotNone(self.qualify(base, metadata={
            **metadata, "fusion_group": "qkv_rope", "fusion_enabled": False,
            "fusion_decision": "target_is_not_gpu",
        })[0])
        self.assertIsNone(self.qualify(base, model_weight_read=False)[0])
        self.assertIsNone(self.qualify(base, rhs_is_activation=True)[0])
        self.assertIsNone(self.qualify(base, workload=replace(work, epilogue_operations=1, epilogue_name="synthetic"))[0])
        for field in ("global_k", "global_n", "local_k", "local_n", "physical_bytes"):
            bad = deepcopy(metadata)
            bad["projection_segments"][0][field] += 1
            self.assertIsNone(self.qualify(base, metadata=bad)[0])
        self.assertIsNone(self.qualify(base, metadata={**metadata, "projection_segment_count": 2})[0])

    def test_source_dtype_declarations_cannot_override_actual_storage(self):
        base = contract_scenario(scenario(), source_activation_dtype="F32", source_output_dtype="F32",
                                 source_weight_layout="ordinary_contiguous_2d")
        work, _ = direct_work(base)
        for changes in ({"activation_storage_bytes": 2*work.m*work.k},
                        {"output_storage_bytes": 2*work.m*work.n}, {"accumulator_bits": 16}):
            dispatch, audit = self.qualify(base, workload=replace(work, **changes))
            self.assertIsNone(dispatch)
            self.assertEqual(audit["reason"], "ordinary_f32_input_output_not_proven")
        # Adapter dtype/layout hints are ignored even when wrong: the tensor evidence wins.
        actual = contract_scenario(scenario(), source_activation_dtype="I32", source_weight_layout="strided_2d")
        dispatch, _ = self.qualify(actual)
        self.assertEqual(dispatch.source_activation_dtype, "F32")
        self.assertEqual(dispatch.source_weight_layout, "ordinary_contiguous_2d")

    def test_cache_key_covers_every_dispatch_field_and_preserves_none_key(self):
        work, _ = direct_work(scenario())
        dispatch, _ = self.qualify()
        key = planner._cpu_gemm_cost_key("cpu0", work, dispatch)
        self.assertEqual(planner._cpu_gemm_cost_key("cpu0", work), ("cpu_gemm", "cpu0", work))
        values = {"compiled_avx2": False, "source_activation_dtype": "F16", "source_output_dtype": "BF16",
                  "source_weight_layout": "strided_2d", "source_activation_ne3": 2, "use_reference_kernel": True,
                  "no_iq_panel_environment_state": "set", "assume_default_unset": False,
                  "source_sha256": "d"*64, "cpu_backend_sha256": "e"*64, "source_refs": ("another-source",)}
        self.assertEqual(set(values), {field.name for field in fields(CPUIQPanelDispatch)})
        for name, value in values.items():
            with self.subTest(field=name):
                self.assertNotEqual(key, planner._cpu_gemm_cost_key("cpu0", work, replace(dispatch, **{name: value})))

    def test_memoized_cost_does_not_leak_applied_dispatch_to_generic_or_disabled(self):
        base = scenario()
        workload, _ = direct_work(base)
        dispatch, _ = self.qualify(base)
        cpu, memory = planner._cpu_profiles(base, "cpu0")
        context = planner.CompilationContext(base)
        with planner._compilation_scope(base, context), patch.object(
            planner, "estimate_cpu_gemm", wraps=planner.estimate_cpu_gemm
        ) as estimate:
            results = []
            for facts in (dispatch, dispatch, replace(dispatch, no_iq_panel_environment_state="set"), None, None):
                results.append(planner._memoized_cost_estimate(
                    base, planner._cpu_gemm_cost_key("cpu0", workload, facts),
                    lambda facts=facts: planner.estimate_cpu_gemm(cpu, memory, workload, iq_panel_dispatch=facts),
                ))
            self.assertEqual(estimate.call_count, 3)
            self.assertIs(results[0], results[1])
            self.assertIs(results[3], results[4])
            self.assertEqual(results[2].phases[-1].demands, results[3].phases[-1].demands)
            self.assertNotEqual(results[0].phases[-1].demands, results[3].phases[-1].demands)

    def test_dynamic_attention_replay_refreshes_shape_and_never_dispatches_iq_panel(self):
        base = scenario()
        layer = planner._execution_layers(base)[0]
        rank = planner._parallel_plan(base).ranks[0]
        work = GemmWorkload(m=8, n=16, k=256, activation_bits=16, weight_bits=16, output_bits=16)
        payload = planner._DynamicAttentionCostTaskReplayPayload(
            role="qk", workload=work,
            fused_workload=FusedAttentionWorkload(batch_tokens=8, context_tokens=16, hidden_size=256),
            fusion_targets=(("qk", "cpu0"), ("pv", "cpu0")), layer=layer, rank=rank,
            tp_degree=1, token_batch=8, score_heads=8, target_component_id="cpu0",
            phase_index=1, phase_name="cpu_gemm",
        )
        for tokens in (32, 64):
            replay = planner._TaskSegmentDynamicReplayContext(base, tokens, tokens)
            overrides = planner._task_segment_dynamic_task_overrides((payload,), replay)
            self.assertIsNotNone(overrides)
            demands, metadata = overrides[0]
            audit = metadata["cpu_iq_panel_reuse"]
            self.assertEqual((audit["m"], audit["n"], audit["k"]), (8, tokens, 256))
            self.assertEqual(audit["reason"], "runtime_rhs_is_not_a_physical_model_weight")
            self.assertNotIn("iq_panel_weight_reuse", metadata["phase_metadata"]["instruction_schedule"])
            disabled = contract_scenario(base, enabled=False)
            old = planner._task_segment_dynamic_task_overrides(
                (payload,), planner._TaskSegmentDynamicReplayContext(disabled, tokens, tokens),
            )
            self.assertEqual(demands, old[0][0])
            self.assertNotIn("cpu_iq_panel_reuse", old[0][1])

    def test_ambiguous_source_tensor_does_not_qualify(self):
        base = scenario()
        layers = []
        for layer in planner._execution_layers(base):
            metadata = deepcopy(layer.metadata)
            metadata["gguf_tensor_bindings"].append({
                **metadata["gguf_tensor_bindings"][0], "n_bytes": 1,
            })
            layers.append(replace(layer, metadata=metadata))
        base = replace(base, model=replace_model_layer_specs(base.model, tuple(layers)))
        dispatch, audit = self.qualify(base)
        self.assertIsNone(dispatch)
        self.assertEqual(audit["reason"], "source_tensor_binding_missing_or_ambiguous")

    def test_serving_cohort_exposes_physical_counts_in_manifest_and_cost_metadata(self):
        base = scenario()
        context = planner.CompilationContext(base)
        with planner._compilation_scope(base, context):
            for index in (0, 1):
                cohort = BatchCohort("cpu-iqp-" + str(index), "prefill", 0.0, (
                    BatchItem("request-" + str(index), "prefill", 8, index * 8,
                              kv_append_tokens=8, kv_materialized_tokens=8, logit_tokens=0),
                ))
                lowered = planner._lower_serving_cohort(base, cohort)
                counts = planner.summarize_cpu_iq_panel_reuse(lowered.schedule.tasks)
                self.assertGreater(counts["applied_tasks"], 0)
                self.assertEqual(counts, lowered.extra_metadata["cpu_iq_panel_reuse"])
                self.assertEqual(counts, lowered.schedule.manifest.metadata["cpu_iq_panel_reuse"])
                self.assertEqual(counts["audited_tasks"], len(physical_tasks(lowered.schedule)))
        disabled = contract_scenario(base, enabled=False)
        lowered = planner._lower_serving_cohort(disabled, cohort)
        self.assertNotIn("cpu_iq_panel_reuse", lowered.extra_metadata)
        self.assertNotIn("cpu_iq_panel_reuse", lowered.schedule.manifest.metadata)

    def test_invalid_source_contract_preserves_original_work(self):
        for changes in ({"compiled_avx2": "yes"}, {"no_iq_panel_environment_state": "unset_today"},
                        {"assume_default_unset": "yes"}, {"source_refs": []}, {"source_sha256": "bad"}):
            with self.subTest(changes=changes):
                self.assertIsNone(self.qualify(contract_scenario(scenario(), **changes))[0])


if __name__ == "__main__":
    unittest.main()
