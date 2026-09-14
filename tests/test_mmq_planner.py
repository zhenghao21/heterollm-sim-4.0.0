"""Behavior checks for source-qualified conversion/main/fixup integration."""
from dataclasses import replace
import unittest

from heterollm_sim import planner
from heterollm_sim.cost_models import GemmWorkload
from heterollm_sim.serde import to_primitive
from tests.test_physical_projection_invocations import _descriptors, _full_layer, _scenario
from heterollm_sim.projection_descriptors import ARTIFACT_QUANTIZATION_REGISTRY


CONTRACT = {
    "backend_commit": "0f3a71be15af836d277c9f918adfafb45732677e",
    "compiled_int8_mma": True,
    "force_cublas": False,
    "ordinary_contiguous_2d": True,
    "max_shared_memory_per_block_optin_bytes": 101376,
}


KV_CONTRACT = {
    "backend_commit": CONTRACT["backend_commit"], "rope_type": "neox",
    "unified_non_transposed_cache": True, "ordinary_internal_k": True,
    "fusion_environment_enabled": True, "explicit_flash_attention": True,
}


def scenario(*, enabled=True, tokens=128, weight_format="Q4_K", contract=None,
             cache_format="f16", kv_contract=None):
    metadata = _descriptors()
    spec = next(s for s in ARTIFACT_QUANTIZATION_REGISTRY.values() if s.name == weight_format)
    for projection in metadata["weight_projection_descriptors"]["projections"].values():
        for segment in projection["segments"]:
            segment["k"] *= 2
            segment["n"] *= 2
            segment["format"] = weight_format
            segment["physical_bytes"] = (
                segment["n"] * (segment["k"] // spec.block_size)
                * (spec.payload_bytes + spec.metadata_bytes)
            )
    layer = replace(_full_layer(metadata), hidden_size=256, intermediate_size=512)
    base = _scenario(layer, tokens=tokens, capability=True, conversion=True)
    flags = {**base.workload.metadata, "llama_cpp_f32_hidden_storage": True}
    if enabled is not None:
        flags["llama_cpp_mmq_source_work"] = enabled
    hardware = replace(base.hardware, components=tuple(
        replace(c, metadata={**c.metadata, "llama_cpp_mmq_contract": dict(CONTRACT if contract is None else contract)})
        if c.component_id == "gpu0" else c for c in base.hardware.components
    ))
    profiles = {**base.component_profiles, "gpu": {
        key: replace(value, tensor_core=replace(value.tensor_core, sm_count=84))
        for key, value in base.component_profiles["gpu"].items()
    }}
    placement_metadata = dict(base.placement.metadata)
    if cache_format in {"Q4_0", "Q8_0"}:
        placement_metadata["kv_artifact_quantization"] = cache_format
    return replace(base, hardware=hardware, component_profiles=profiles,
                   model=replace(base.model, metadata={**base.model.metadata,
                       "llama_cpp_native_kv_writeback": dict(KV_CONTRACT if kv_contract is None else kv_contract)}),
                   placement=replace(base.placement, metadata=placement_metadata,
                       kv_policy=replace(base.placement.kv_policy,
                           dtype={"Q4_0": "int4", "Q8_0": "int8"}.get(cache_format, cache_format),
                           cache_component="hbm0", offload_component=None, offload_ratio=0.0)),
                   workload=replace(base.workload, metadata=flags))


def source_work(task):
    return task.metadata.get("mmq_source_work", {})


class MMQPlannerTests(unittest.TestCase):
    def test_missing_and_false_flag_preserve_identical_tasks(self):
        absent = planner.compile_scenario(scenario(enabled=None))
        disabled = planner.compile_scenario(scenario(enabled=False))
        self.assertEqual([to_primitive(t) for t in absent.tasks], [to_primitive(t) for t in disabled.tasks])
        self.assertFalse(any(source_work(t) for t in absent.tasks))

    def test_conversion_main_fixup_dependency_and_payloads(self):
        schedule = planner.compile_scenario(scenario())
        by_id = {t.task_id: t for t in schedule.tasks}
        mains = [t for t in schedule.tasks if t.metadata.get("phase") == "gpu_gemm"
                 and source_work(t).get("status") == "applied"]
        self.assertGreaterEqual(len(mains), 7)
        for main in mains:
            work = source_work(main)
            projection = main.metadata["projection_id"]
            stages = {stage: [t for t in schedule.tasks if t.metadata.get("projection_id") == projection
                              and source_work(t).get("stage") == stage]
                      for stage in ("conversion", "fixup")}
            self.assertEqual(sum(t.metadata.get("phase") == "kernel_launch" for t in stages["conversion"]), 1)
            self.assertEqual(sum(t.metadata.get("phase") == "kernel_launch" for t in stages["fixup"]), int(work["fixup_launch"]))
            self.assertEqual(main.metadata["cost_model"]["output_bytes"], 4 * work["m"] * work["n"])
            self.assertEqual(main.metadata["cost_model"]["activation_bytes"], work["consumer_unique_bytes"])
            self.assertEqual(main.metadata["cost_model"]["internal_tensor_dtype"], "int8")
            self.assertEqual(main.metadata["cost_model"]["source_tensor_dtype"], "fp16")

            def ancestors(task):
                found, pending = set(), list(task.dependencies)
                while pending:
                    key = pending.pop()
                    if key in found:
                        continue
                    found.add(key)
                    pending.extend(by_id[key].dependencies)
                return found

            self.assertTrue(all(t.task_id in ancestors(main) for t in stages["conversion"]))
            self.assertTrue(all(main.task_id in ancestors(t) for t in stages["fixup"]))
            self.assertTrue(all(t.metadata["target_component"] == "gpu0" for stage in stages.values() for t in stage))
        old = planner.compile_scenario(scenario(enabled=False))
        count = lambda s: sum(t.metadata.get("event_kind") == "model_weight_access" for t in s.tasks)
        self.assertEqual(count(old), count(schedule))

    def test_dispatch_keeps_mmvq_and_extends_only_declared_range(self):
        for fmt, rows in (("Q4_K", 5), ("IQ4_XS", 8), ("Q6_K", 7)):
            with self.subTest(fmt=fmt):
                schedule = planner.compile_scenario(scenario(tokens=rows, weight_format=fmt))
                self.assertFalse(any(source_work(t).get("status") == "applied" for t in schedule.tasks))
                self.assertTrue(any(source_work(t).get("status") == "mmvq_precedes_mmq" for t in schedule.tasks))
                self.assertTrue(any(t.metadata.get("activation_conversion_applied") for t in schedule.tasks))

    def test_wrong_source_or_shared_limit_is_explicitly_uncovered(self):
        for change in ({"backend_commit": "unverified"}, {"force_cublas": True},
                       {"max_shared_memory_per_block_optin_bytes": 32768}):
            schedule = planner.compile_scenario(scenario(contract={**CONTRACT, **change}))
            audits = [source_work(t) for t in schedule.tasks if source_work(t)]
            self.assertTrue(audits)
            self.assertFalse(any(a.get("status") == "applied" for a in audits))
            self.assertTrue(all(a.get("reason") for a in audits))

    def test_invalid_opt_in_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            planner.compile_scenario(scenario(enabled="true"))

    def test_source_high_water_is_uncovered_without_clamping(self):
        case = scenario(tokens=9, weight_format="IQ4_XS")
        gpu = next(c for c in case.hardware.components if c.component_id == "gpu0")
        profile = next(iter(case.component_profiles["gpu"].values()))
        workload = GemmWorkload(m=9, k=1024, n=128, activation_bits=16,
            weight_bits=4, output_bits=32, packed_weight_formats=("IQ4_XS",),
            activation_storage_bytes=4*9*1024)
        metadata = {"projection_segment_count": 1, "projection_segments": ({
            "local_k": 1024, "local_n": 128, "format": "IQ4_XS", "physical_tensor_name": "test.weight"},)}
        work, audit, limit = planner._declared_mmq_work(
            case, workload, gpu, profile, metadata, model_weight_read=True, rhs_is_activation=False)
        self.assertIsNone(work)
        self.assertEqual(limit, 0)
        self.assertEqual(audit["status"], "uncovered")
        self.assertIn("high-water", audit["reason"])


if __name__ == "__main__":
    unittest.main()
