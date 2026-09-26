from dataclasses import replace
import unittest
from copy import deepcopy

from heterollm_sim.config import ScenarioConfig, scenario_from_dict
from heterollm_sim.cost_models import (
    CPUProfile,
    CPUQuantizedDotCapability,
    GPUProfile,
    GPUQuantizedMatmulCapability,
    HostGemmOffloadCapability,
    HostRecurrentOffloadCapability,
)
from heterollm_sim.ir import SCHEMA_VERSION
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import hardware_input_payload


def reference_payload():
    scenario = build_reference_scenario()
    profiles = {
        "components": to_primitive(scenario.component_profiles),
        "host_orchestration": to_primitive(scenario.host_orchestration_profile),
        "fusion": to_primitive(scenario.fusion_policy),
        "cim_interconnect": to_primitive(scenario.cim_interconnect),
        "runtime": to_primitive(scenario.runtime_profile),
    }
    return {
        "schema_version": scenario.schema_version,
        "name": scenario.name,
        "hardware": to_primitive(scenario.hardware),
        "model": to_primitive(scenario.model),
        "placement": to_primitive(scenario.placement),
        "workload": to_primitive(scenario.workload),
        "profiles": profiles,
        "weights_resident": scenario.weights_resident,
        "assumptions": list(scenario.assumptions),
    }


def quantized_dot_capability_payload():
    return {
        "name": "avx_vnni_q8_dot",
        "supported_weight_formats": ["q4_0", "q5_k"],
        "source_activation_bits": [16, 32],
        "dot_activation_bits": 8,
        "dot_weight_bits": 8,
        "accumulator_bits": 32,
        "effective_ops_per_instruction": 64.0,
        "dot_issue_instructions_per_cycle_per_core": 2.0,
        "auxiliary_ops_per_instruction": 32.0,
        "activation_quantization_block_elements": 64,
        "activation_quantization_instructions_per_block": 3.5,
        "evidence": "AVX-VNNI kernel contract",
        "source_dot_work": {},
        "source_dot_work_max_m": None,
        "source_dot_work_all_m": False,
        "source_dot_work_max_k": None,
    }


def gpu_quantized_matmul_capability_payload():
    return {
        "name": "cuda_mmq_q8",
        "supported_weight_formats": ["IQ3_S", "IQ4_XS"],
        "source_activation_bits": [16],
        "internal_activation_bits": 8,
        "accumulator_bits": 32,
        "kernel_family": "cuda_mmq",
        "tensor_core_dtype": "int8",
        "evidence": "llama.cpp CUDA MMQ source contract",
        "provenance": "llama.cpp@18443257a",
        "min_m": 1,
    }


class ScenarioProfileSchemaV400Tests(unittest.TestCase):
    def test_hardware_input_is_authoritative_over_legacy_scenario_mirrors(self):
        payload = reference_payload()
        hardware_input = hardware_input_payload(build_reference_scenario())
        self.assertEqual(hardware_input["contract_version"], "2")
        self.assertNotIn("profiles", hardware_input)
        self.assertIn("execution_profile", hardware_input["hardware"]["components"][0])
        hardware_input["hardware"]["name"] = "authoritative-hardware"
        payload["hardware"]["name"] = "stale-legacy-mirror"
        payload["hardware_input"] = hardware_input
        scenario = scenario_from_dict(payload)
        self.assertEqual(scenario.hardware.name, "authoritative-hardware")

    def test_hardware_input_rejects_unknown_kind_and_version(self):
        for field, value in (("kind", "scenario"), ("schema_version", "3.0")):
            payload = reference_payload()
            payload["hardware_input"] = hardware_input_payload(build_reference_scenario())
            payload["hardware_input"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                scenario_from_dict(payload)

    def test_tied_weight_runtime_copy_policy_is_explicit_boolean(self):
        payload = reference_payload()
        payload["placement"]["metadata"]["control_plane"] = {
            "policy": {"options": {"gpu_loadable_layers": 1, "tied_weight_runtime_copies": True}}
        }
        scenario = scenario_from_dict(payload)
        self.assertTrue(scenario.placement.metadata["control_plane"]["policy"]["options"]["tied_weight_runtime_copies"])
        for value in (1, "true", None):
            payload["placement"]["metadata"]["control_plane"]["policy"]["options"]["tied_weight_runtime_copies"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                scenario_from_dict(payload)

    def test_optional_prefill_stop_offsets_roundtrip_and_validation(self):
        from heterollm_sim.config import scheduler_from_dict
        scheduler = scheduler_from_dict({"prefill_stop_offsets": [128, 4]})
        self.assertEqual(scheduler.prefill_stop_offsets, (128, 4))
        self.assertEqual(scheduler_from_dict(to_primitive(scheduler)), scheduler)
        self.assertEqual(scheduler_from_dict({}).prefill_stop_offsets, ())
        for bad in ([0], [-1], [True], [4, 4], "4"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                scheduler_from_dict({"prefill_stop_offsets": bad})

    def test_only_v400_is_accepted(self):
        self.assertEqual(SCHEMA_VERSION, "4.0.0")
        self.assertEqual(scenario_from_dict(reference_payload()).schema_version, "4.0.0")
        for version in ("0.1", "0.5", "2.0.0", "3.0.0"):
            payload = reference_payload()
            payload["schema_version"] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "exactly 4.0.0"):
                scenario_from_dict(payload)

    def test_v3_profiles_use_explicit_nested_types(self):
        scenario = scenario_from_dict(reference_payload())
        self.assertIsInstance(scenario.gpu_profile, GPUProfile)
        self.assertIsInstance(scenario.cpu_profile, CPUProfile)
        self.assertEqual(scenario.gpu_profile.tensor_core.sm_count, 120)
        self.assertEqual(len(scenario.gpu_profile.cache_hierarchy.levels), 2)
        self.assertEqual(scenario.cpu_profile.pipeline.core_count, 16)
        self.assertEqual(len(scenario.cpu_profile.cache_hierarchy.levels), 3)
        self.assertEqual(scenario.host_orchestration_profile.cpu_component_id, "cpu0")
        self.assertTrue(scenario.fusion_policy.flash_attention)
        self.assertEqual(scenario.gpu_profile.quantized_matmul_capabilities, ())

    def test_host_control_instruction_counts_are_explicit_v4_profile_fields(self):
        payload = reference_payload()
        host = payload["profiles"]["host_orchestration"]
        expected = {
            "capacity_fixed_instructions": 96,
            "capacity_instructions_per_request": 64,
            "schedule_fixed_instructions": 192,
            "schedule_instructions_per_request": 48,
            "schedule_instructions_per_token": 8,
            "command_build_fixed_instructions": 128,
            "command_build_instructions_per_invocation": 12,
            "dma_queue_submission_ns": 62.5,
        }
        self.assertEqual(
            {name: host[name] for name in expected},
            expected,
        )

        parsed = scenario_from_dict(payload).host_orchestration_profile
        self.assertEqual(parsed.capacity_instruction_count(2), 224)
        self.assertEqual(parsed.schedule_instruction_count(2, 3), 312)
        self.assertEqual(parsed.command_build_instruction_count(4), 176)

        host["command_build_instructions_per_invocation"] = 1.5
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            scenario_from_dict(payload)

    def test_gpu_quantized_matmul_capability_parses_and_round_trips(self):
        payload = reference_payload()
        capability_payload = gpu_quantized_matmul_capability_payload()
        payload["profiles"]["components"]["gpu"]["legacy-gpu"][
            "quantized_matmul_capabilities"
        ] = [capability_payload]

        scenario = scenario_from_dict(payload)
        capability = scenario.gpu_profile.quantized_matmul_capabilities[0]

        self.assertIsInstance(capability, GPUQuantizedMatmulCapability)
        self.assertEqual(
            capability.supported_weight_formats, ("IQ3_S", "IQ4_XS")
        )
        self.assertEqual(capability.source_activation_bits, (16,))
        self.assertEqual(to_primitive(capability), capability_payload)

        round_trip_payload = reference_payload()
        round_trip_payload["profiles"]["components"] = to_primitive(
            scenario.component_profiles
        )
        round_trip = scenario_from_dict(round_trip_payload)
        self.assertEqual(
            round_trip.gpu_profile.quantized_matmul_capabilities,
            scenario.gpu_profile.quantized_matmul_capabilities,
        )

    def test_gpu_quantized_matmul_capability_rejects_invalid_contracts(self):
        invalid_capabilities = (
            {"supported_weight_formats": []},
            {"source_activation_bits": []},
            {"source_activation_bits": [0]},
            {"internal_activation_bits": 16},
            {"accumulator_bits": 0},
            {"kernel_family": ""},
            {"tensor_core_dtype": "unsupported"},
            {"evidence": ""},
            {"provenance": ""},
            {"undocumented_kernel_magic": 1},
        )

        for override in invalid_capabilities:
            payload = reference_payload()
            capability_payload = gpu_quantized_matmul_capability_payload()
            capability_payload.update(override)
            payload["profiles"]["components"]["gpu"]["legacy-gpu"][
                "quantized_matmul_capabilities"
            ] = [capability_payload]
            with self.subTest(override=override), self.assertRaises(ValueError):
                scenario_from_dict(payload)

    def test_cpu_quantized_dot_capability_parses_and_round_trips(self):
        payload = reference_payload()
        capability_payload = quantized_dot_capability_payload()
        payload["profiles"]["components"]["cpu"]["legacy-cpu"][
            "quantized_dot_capabilities"
        ] = [capability_payload]

        scenario = scenario_from_dict(payload)
        capability = scenario.cpu_profile.quantized_dot_capabilities[0]

        self.assertIsInstance(capability, CPUQuantizedDotCapability)
        self.assertEqual(capability.supported_weight_formats, ("q4_0", "q5_k"))
        self.assertEqual(capability.source_activation_bits, (16, 32))
        self.assertEqual(
            to_primitive(capability),
            capability_payload,
        )

        round_trip_payload = reference_payload()
        round_trip_payload["profiles"]["components"] = to_primitive(
            scenario.component_profiles
        )
        round_trip = scenario_from_dict(round_trip_payload)
        self.assertEqual(
            round_trip.cpu_profile.quantized_dot_capabilities,
            scenario.cpu_profile.quantized_dot_capabilities,
        )

    def test_cpu_source_work_budget_roundtrip_and_validation(self):
        payload = reference_payload()
        capability = quantized_dot_capability_payload()
        capability.update(source_dot_work_max_m=7, source_dot_work={
            "q4_0": {"block_elements": 256, "auxiliary_vector_ops": 86,
                     "vector_loads": 16, "scalar_lut_loads": 0}})
        payload["profiles"]["components"]["cpu"]["legacy-cpu"]["quantized_dot_capabilities"] = [capability]
        parsed = scenario_from_dict(payload).cpu_profile.quantized_dot_capabilities[0]
        self.assertEqual(to_primitive(parsed), capability)
        for overrides in ({"source_dot_work_max_m": 0},
                          {"source_dot_work_max_m": True},
                          {"source_dot_work_max_m": None},
                          {"evidence": ""},
                          {"source_dot_work": {"unknown": capability["source_dot_work"]["q4_0"]}},
                          {"source_dot_work": {"q4_0": {"block_elements": 256}}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(parsed, **overrides)

    def test_cpu_fixed_row_budget_and_explicit_scope_roundtrip(self):
        payload = reference_payload()
        capability = quantized_dot_capability_payload()
        budget = {
            "block_elements": 256, "auxiliary_vector_ops": 52,
            "vector_loads": 14, "scalar_lut_loads": 2,
            "row_auxiliary_vector_ops": 16, "row_vector_loads": 9,
            "row_scalar_lut_loads": 0, "row_store_issue_ops": 8,
        }
        capability.update(source_dot_work_all_m=True, source_dot_work_max_k=2**31 - 1,
                          source_dot_work={"q4_0": budget})
        payload["profiles"]["components"]["cpu"]["legacy-cpu"]["quantized_dot_capabilities"] = [capability]
        parsed = scenario_from_dict(payload).cpu_profile.quantized_dot_capabilities[0]
        self.assertEqual(to_primitive(parsed), capability)
        invalid = [
            {"source_dot_work_all_m": value} for value in (1, "true", None)
        ] + [
            {"source_dot_work_max_k": value} for value in (0, -1, True, 256.0)
        ] + [
            {"source_dot_work_max_m": 7}, {"source_dot_work_all_m": False},
            {"evidence": ""}, {"source_dot_work": {}},
        ]
        for key in budget:
            invalid.append({"source_dot_work": {"q4_0": {**budget, key: True}}})
        for key in ("row_auxiliary_vector_ops", "row_vector_loads", "row_scalar_lut_loads", "row_store_issue_ops"):
            invalid.append({"source_dot_work": {"q4_0": {k: v for k, v in budget.items() if k != key}}})
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(parsed, **overrides)
        with self.assertRaises(ValueError):
            replace(parsed, source_dot_work={}, source_dot_work_all_m=False)

    def test_gpu_host_gemm_offload_capability_parses_and_round_trips(self):
        payload = reference_payload()
        capability_payload = {
            "minimum_m": 32,
            "evidence": "llama.cpp runtime source audit",
        }
        payload["profiles"]["components"]["gpu"]["legacy-gpu"][
            "host_gemm_offload"
        ] = capability_payload

        scenario = scenario_from_dict(payload)
        capability = scenario.gpu_profile.host_gemm_offload

        self.assertIsInstance(capability, HostGemmOffloadCapability)
        self.assertEqual(capability.minimum_m, 32)
        self.assertEqual(to_primitive(capability), capability_payload)

        round_trip_payload = reference_payload()
        round_trip_payload["profiles"]["components"] = to_primitive(
            scenario.component_profiles
        )
        round_trip = scenario_from_dict(round_trip_payload)
        self.assertEqual(
            round_trip.gpu_profile.host_gemm_offload,
            capability,
        )

    def test_gpu_host_gemm_offload_capability_rejects_invalid_contracts(self):
        invalid_capabilities = (
            {"minimum_m": 0, "evidence": "source audit"},
            {"minimum_m": 1.5, "evidence": "source audit"},
            {"minimum_m": 32, "evidence": ""},
            {"minimum_m": 32, "evidence": 42},
            {
                "minimum_m": 32,
                "evidence": "source audit",
                "unsupported_operator": "elementwise",
            },
        )

        for capability_payload in invalid_capabilities:
            payload = reference_payload()
            payload["profiles"]["components"]["gpu"]["legacy-gpu"][
                "host_gemm_offload"
            ] = capability_payload
            with self.subTest(capability=capability_payload):
                with self.assertRaises(ValueError):
                    scenario_from_dict(payload)

    def test_gpu_host_recurrent_offload_parses_and_round_trips(self):
        payload = reference_payload()
        capability_payload = {
            "op_offload": True,
            "minimum_m": 32,
            "architecture": "qwen3_5_hybrid_transformer",
            "supported_ops": [
                "rms_norm",
                "l2_norm",
                "silu",
                "sigmoid",
                "softplus",
                "ssm_conv",
                "gated_delta_net",
                "gate",
            ],
            "query_width": 2048,
            "key_width": 2048,
            "value_width": 6144,
            "conv_kernel_size": 4,
            "state_dtype": "fp32",
            "evidence": "llama.cpp source audit",
            "provenance": "llama.cpp@revision",
        }
        payload["profiles"]["components"]["gpu"]["legacy-gpu"][
            "host_recurrent_offload"
        ] = capability_payload

        scenario = scenario_from_dict(payload)
        capability = scenario.gpu_profile.host_recurrent_offload
        self.assertIsInstance(capability, HostRecurrentOffloadCapability)
        self.assertEqual(capability.minimum_m, 32)
        self.assertEqual(capability.supported_ops, tuple(
            capability_payload["supported_ops"]
        ))
        self.assertEqual(to_primitive(capability), capability_payload)

        round_trip_payload = reference_payload()
        round_trip_payload["profiles"]["components"] = to_primitive(
            scenario.component_profiles
        )
        round_trip = scenario_from_dict(round_trip_payload)
        self.assertEqual(
            round_trip.gpu_profile.host_recurrent_offload,
            capability,
        )

    def test_gpu_host_recurrent_offload_rejects_unknown_fields(self):
        payload = reference_payload()
        payload["profiles"]["components"]["gpu"]["legacy-gpu"][
            "host_recurrent_offload"
        ] = {
            "op_offload": True,
            "minimum_m": 32,
            "architecture": "qwen3_5_hybrid_transformer",
            "supported_ops": ["ssm_conv"],
            "query_width": 2048,
            "key_width": 2048,
            "value_width": 6144,
            "conv_kernel_size": 4,
            "state_dtype": "fp32",
            "evidence": "source audit",
            "provenance": "revision",
            "invented_efficiency": 0.8,
        }
        with self.assertRaisesRegex(
            ValueError, "contains unknown fields: invented_efficiency"
        ):
            scenario_from_dict(payload)

    def test_cpu_quantized_dot_capability_rejects_nested_unknown_field(self):
        payload = reference_payload()
        capability_payload = quantized_dot_capability_payload()
        capability_payload["undocumented_lane_magic"] = 1
        payload["profiles"]["components"]["cpu"]["legacy-cpu"][
            "quantized_dot_capabilities"
        ] = [capability_payload]

        with self.assertRaisesRegex(
            ValueError,
            "contains unknown fields: undocumented_lane_magic",
        ):
            scenario_from_dict(payload)

    def test_cpu_quantized_dot_capability_rejects_invalid_arrays(self):
        invalid_values = (
            ("supported_weight_formats", []),
            ("supported_weight_formats", [""]),
            ("supported_weight_formats", [4]),
            ("source_activation_bits", []),
            ("source_activation_bits", [0]),
        )

        for field_name, invalid_value in invalid_values:
            payload = reference_payload()
            capability_payload = quantized_dot_capability_payload()
            capability_payload[field_name] = invalid_value
            payload["profiles"]["components"]["cpu"]["legacy-cpu"][
                "quantized_dot_capabilities"
            ] = [capability_payload]
            with self.subTest(field_name=field_name, value=invalid_value):
                with self.assertRaises(ValueError):
                    scenario_from_dict(payload)

    def test_all_required_profiles_are_mandatory(self):
        for field in ("components", "host_orchestration", "fusion"):
            payload = reference_payload()
            payload["profiles"].pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "required"):
                scenario_from_dict(payload)

        for kind in ("gpu", "hbm", "cpu", "host_memory", "cim"):
            payload = reference_payload()
            payload["profiles"]["components"].pop(kind)
            with self.subTest(kind=kind), self.assertRaisesRegex(
                ValueError, "unknown {} cost profile".format(kind)
            ):
                scenario_from_dict(payload)

    def test_old_flat_gpu_and_cpu_profiles_are_rejected(self):
        payload = reference_payload()
        payload["profiles"]["gpu"] = {"peak_tops": 120, "attainable_efficiency": 0.6}
        with self.assertRaises((TypeError, ValueError)):
            scenario_from_dict(payload)

    def test_host_orchestration_references_real_cpu_and_gpu_components(self):
        payload = reference_payload()
        payload["profiles"]["host_orchestration"]["cpu_component_id"] = "removed-cpu"
        with self.assertRaisesRegex(ValueError, "absent from hardware"):
            scenario_from_dict(payload)

        payload = reference_payload()
        payload["profiles"]["host_orchestration"]["cpu_component_id"] = "gpu0"
        with self.assertRaisesRegex(ValueError, "must be a cpu"):
            scenario_from_dict(payload)

        payload = reference_payload()
        payload["profiles"]["host_orchestration"]["gpu_component_id"] = "cpu0"
        with self.assertRaisesRegex(ValueError, "must be a gpu"):
            scenario_from_dict(payload)
        payload = reference_payload()
        payload["profiles"]["components"]["cpu"]["legacy-cpu"] = {
            "gemm_gops": 100,
            "compute_resource_id": "cpu0.compute",
        }
        with self.assertRaises((TypeError, ValueError)):
            scenario_from_dict(payload)

    def test_unknown_fields_are_rejected_at_every_v3_authoring_boundary(self):
        targets = (
            ("scenario", ()),
            ("profiles", ("profiles",)),
            ("hardware", ("hardware",)),
            ("component", ("hardware", "components", 0)),
            ("port", ("hardware", "components", 0, "ports", 0)),
            ("link", ("hardware", "links", 0)),
            ("rank mapping", ("placement", "parallel", "rank_mapping", 0)),
            ("request", ("workload", "requests", 0)),
            ("gpu profile", ("profiles", "components", "gpu", "legacy-gpu")),
            ("tensor core", ("profiles", "components", "gpu", "legacy-gpu", "tensor_core")),
            ("cache level", ("profiles", "components", "gpu", "legacy-gpu", "cache_hierarchy", "levels", 0)),
            ("cpu pipeline", ("profiles", "components", "cpu", "legacy-cpu", "pipeline")),
            ("fusion", ("profiles", "fusion")),
        )
        for label, path in targets:
            payload = reference_payload()
            target = payload
            for part in path:
                target = target[part]
            target["removed_v2_field"] = True
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError,
                "contains unknown fields: removed_v2_field",
            ):
                scenario_from_dict(payload)

    def test_v4_control_plane_policy_and_options_reject_unknown_fields(self):
        payload = reference_payload()
        payload["placement"]["metadata"]["control_plane"] = {
            "policy": {"gpu_loadable_layers": 0}
        }
        with self.assertRaisesRegex(
            ValueError,
            "placement.metadata.control_plane.policy.*gpu_loadable_layers",
        ):
            scenario_from_dict(payload)

        payload = reference_payload()
        payload["placement"]["metadata"]["control_plane"] = {
            "policy": {"options": {"gpu_loadable_layer": 0}}
        }
        with self.assertRaisesRegex(
            ValueError,
            "placement.metadata.control_plane.policy.options.*gpu_loadable_layer",
        ):
            scenario_from_dict(payload)

    def test_scenario_constructor_requires_explicit_v3_profile_types(self):
        base = build_reference_scenario()
        invalid = {
            kind: dict(registry)
            for kind, registry in base.component_profiles.items()
        }
        invalid["cpu"]["legacy-cpu"] = "not-a-profile"
        with self.assertRaisesRegex(ValueError, "must be a CPUProfile"):
            replace(base, component_profiles=invalid)
        with self.assertRaises((TypeError, ValueError)):
            ScenarioConfig(
                base.name,
                base.hardware,
                base.model,
                base.placement,
                base.workload,
                "not-a-registry",
                base.host_orchestration_profile,
                base.fusion_policy,
                base.cim_interconnect,
                base.weights_resident,
                base.schema_version,
                base.assumptions,
            )


if __name__ == "__main__":
    unittest.main()
