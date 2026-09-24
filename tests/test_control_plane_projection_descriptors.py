import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from heterollm_sim.control_plane_planner import (
    PlacementPolicy,
    _ExecutionRank,
    _cim_rank_padded_weight_bytes,
    _derive_fusion_opportunities,
    _derive_requirements,
    _operator_cost,
)
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.ir import LayerSpec, LinearAttentionSpec
from heterollm_sim.control_plane_state import mapping_input_fingerprint
from heterollm_sim.projection_descriptors import (
    ARTIFACT_QUANTIZATION_REGISTRY,
)
from heterollm_sim.reference import build_reference_scenario
from tests.model_helpers import model_from_layer_specs


def _segment(segment_id, k, n, format_name, tp_shard_axis):
    spec = ARTIFACT_QUANTIZATION_REGISTRY[format_name]
    physical_bytes = (
        n
        * int(math.ceil(k / float(spec.block_size)))
        * (spec.payload_bytes + spec.metadata_bytes)
    )
    return {
        "segment_id": segment_id,
        "physical_tensor_name": "blk.0.{}.weight".format(segment_id),
        "k": k,
        "n": n,
        "format": format_name,
        "physical_bytes": physical_bytes,
        "tp_shard_axis": tp_shard_axis,
    }


def _projection(*segments):
    return {"segments": list(segments)}


def _mlp_projections():
    return {
        "mlp.up_gate": _projection(
            _segment("gate", 5120, 17408, "IQ3_S", "n"),
            _segment("up", 5120, 17408, "IQ3_S", "n"),
        ),
        "mlp.down": _projection(
            _segment("down", 17408, 5120, "IQ4_XS", "k")
        ),
    }


def _full_attention_metadata():
    projections = {
        "attention.qkv": _projection(
            _segment("q", 5120, 12288, "IQ4_XS", "n"),
            _segment("k", 5120, 1024, "IQ4_XS", "n"),
            _segment("v", 5120, 1024, "Q5_K", "n"),
        ),
        "attention.output": _projection(
            _segment("output", 6144, 5120, "IQ4_XS", "k")
        ),
        **_mlp_projections(),
    }
    return {
        "weight_projection_descriptors": {
            "schema_version": "heterollm.weight-projections/v1",
            "projections": projections,
        },
        "attention_execution_descriptor": {
            "schema_version": "heterollm.attention-execution/v1",
            "query_heads": 24,
            "kv_heads": 4,
            "head_dim": 256,
            "query_width": 6144,
            "gate_width": 6144,
            "q_projection_width": 12288,
            "rotary_dim": 64,
            "qk_scale": 0.0625,
            "qk_norm": True,
            "gate_activation": "sigmoid",
        },
    }


def _linear_attention_metadata():
    projections = {
        "linear_attention.qkv": _projection(
            _segment("qkv", 5120, 10240, "Q5_K", "n")
        ),
        "linear_attention.output_gate": _projection(
            _segment("output_gate", 5120, 6144, "IQ4_XS", "n")
        ),
        "linear_attention.output": _projection(
            _segment("output", 6144, 5120, "IQ4_XS", "k")
        ),
        **_mlp_projections(),
    }
    return {
        "weight_projection_descriptors": {
            "schema_version": "heterollm.weight-projections/v1",
            "projections": projections,
        }
    }


def _scenario_for_layer(layer):
    scenario = build_reference_scenario()
    model = model_from_layer_specs("descriptor-model", (layer,))
    parallel = replace(
        scenario.placement.parallel,
        layer_to_stage={layer.layer_id: 0},
    )
    placement = replace(
        scenario.placement,
        model_name=model.name,
        op_to_component={},
        tensor_to_component={},
        tensor_bytes={},
        parallel=parallel,
    )
    return replace(scenario, model=model, placement=placement)


def _full_layer(metadata=None):
    return LayerSpec(
        "full0",
        "dense",
        hidden_size=5120,
        intermediate_size=17408,
        attention_heads=24,
        kv_heads=4,
        attention_head_dim=256,
        dtype="fp16",
        quantization="w4a16",
        weight_bytes=180_512_768,
        metadata={} if metadata is None else metadata,
    )


class ControlPlaneProjectionDescriptorTests(unittest.TestCase):
    def test_full_attention_and_mlp_matrices_use_projection_descriptors(self):
        scenario = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        by_id = {
            item.item_id: item for item in _derive_requirements(scenario)
        }

        attention = by_id["full0.attention"]
        qkv, output = attention.matrices
        self.assertEqual((qkv.k, qkv.n), (5120, 14336))
        self.assertEqual((output.k, output.n), (6144, 5120))
        self.assertEqual(
            qkv.weight_storage_bytes + qkv.weight_metadata_bytes,
            39_813_120,
        )
        self.assertEqual(
            output.weight_storage_bytes + output.weight_metadata_bytes,
            16_711_680,
        )
        self.assertEqual(qkv.weight_bits, 5)
        self.assertEqual(qkv.packed_weight_formats, ("IQ4_XS", "Q5_K"))
        self.assertEqual(
            qkv.packed_weight_transform_operations,
            qkv.descriptor_audit["fused_dequant_operations"],
        )
        self.assertEqual(qkv.projection_id, "attention.qkv")
        self.assertTrue(
            qkv.descriptor_audit["weight_projection_descriptor_applied"]
        )

        up_gate, down = by_id["full0.mlp"].matrices
        self.assertEqual((up_gate.k, up_gate.n), (5120, 34816))
        self.assertEqual((down.k, down.n), (17408, 5120))
        self.assertEqual(
            up_gate.weight_storage_bytes + up_gate.weight_metadata_bytes,
            76_595_200,
        )
        self.assertEqual(
            down.weight_storage_bytes + down.weight_metadata_bytes,
            47_349_760,
        )

    def test_declared_owner_bytes_are_unchanged_by_descriptors(self):
        baseline = _scenario_for_layer(_full_layer())
        described = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        baseline_weights = {
            item.item_id: item.tensor_bytes
            for item in _derive_requirements(baseline)
            if item.tensor_id and item.item_id.startswith("full0.")
        }
        described_weights = {
            item.item_id: item.tensor_bytes
            for item in _derive_requirements(described)
            if item.tensor_id and item.item_id.startswith("full0.")
        }

        self.assertEqual(described_weights, baseline_weights)
        self.assertEqual(sum(described_weights.values()), 180_512_768)

    def test_linear_attention_matrices_use_projection_descriptors(self):
        layer = replace(
            _full_layer(_linear_attention_metadata()),
            layer_id="linear0",
            sequence_mixer="linear_attention",
            linear_attention=LinearAttentionSpec(
                key_heads=16,
                value_heads=48,
                key_head_dim=128,
                value_head_dim=128,
                conv_kernel_size=4,
                output_gate=True,
            ),
            weight_bytes=193_879_936,
        )
        requirement = next(
            item
            for item in _derive_requirements(_scenario_for_layer(layer))
            if item.item_id == "linear0.linear_attention"
        )

        self.assertEqual(
            [(matrix.k, matrix.n) for matrix in requirement.matrices],
            [(5120, 10240), (6144, 5120), (5120, 6144)],
        )
        self.assertEqual(
            [
                matrix.weight_storage_bytes + matrix.weight_metadata_bytes
                for matrix in requirement.matrices
            ],
            [36_044_800, 16_711_680, 16_711_680],
        )

    def test_gpu_and_cpu_use_gguf_bytes_but_packed_cim_requires_conversion(self):
        scenario = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        requirement = next(
            item
            for item in _derive_requirements(scenario)
            if item.item_id == "full0.attention"
        )
        router = TopologyRouter(scenario.hardware)
        estimate = type("Estimate", (), {"service_ns": 1.0})()

        with patch(
            "heterollm_sim.control_plane_planner.estimate_gpu_gemm",
            return_value=estimate,
        ) as gpu_estimator:
            _operator_cost(
                scenario,
                "ttft",
                requirement,
                scenario.hardware.component_map()["gpu0"],
                None,
                router,
            )
        gpu_workloads = [call.args[2] for call in gpu_estimator.call_args_list]
        self.assertEqual(
            [workload.weight_bytes for workload in gpu_workloads],
            [39_813_120, 16_711_680],
        )
        self.assertEqual(
            [workload.weight_bits for workload in gpu_workloads], [5, 4]
        )
        self.assertEqual(
            [workload.packed_weight_formats for workload in gpu_workloads],
            [("IQ4_XS", "Q5_K"), ("IQ4_XS",)],
        )

        with patch(
            "heterollm_sim.control_plane_planner.estimate_cpu_gemm",
            return_value=estimate,
        ) as cpu_estimator, patch(
            "heterollm_sim.control_plane_planner._route_cost", return_value=0.0
        ):
            _operator_cost(
                scenario,
                "ttft",
                requirement,
                scenario.hardware.component_map()["cpu0"],
                None,
                router,
            )
        self.assertEqual(
            [call.args[2].weight_bytes for call in cpu_estimator.call_args_list],
            [39_813_120, 16_711_680],
        )
        cpu_workloads = [
            call.args[2] for call in cpu_estimator.call_args_list
        ]
        self.assertEqual(
            [workload.packed_weight_formats for workload in cpu_workloads],
            [("IQ4_XS", "Q5_K"), ("IQ4_XS",)],
        )
        self.assertEqual(
            [
                workload.packed_weight_transform_operations
                for workload in cpu_workloads
            ],
            [
                matrix.packed_weight_transform_operations
                for matrix in requirement.matrices
            ],
        )

        rank = _ExecutionRank(0, 0, 0, 0, "gpu0", "hbm0", "cim0")
        cim_requirement = replace(
            requirement,
            layer=replace(
                requirement.layer, dtype="int8", quantization="w4a8"
            ),
        )
        # A nominal int8/w4a8 label does not decode the packed GGUF layout.
        # Array-padding tests use actual dense operands; this source must fail
        # closed until a supported, explicit conversion contract is supplied.
        with self.assertRaisesRegex(ValueError, "explicit decoded conversion contract"):
            _cim_rank_padded_weight_bytes(
                scenario, cim_requirement, rank, "cim0"
            )
        backing_changed = replace(
            cim_requirement,
            matrices=tuple(
                replace(
                    matrix,
                    weight_storage_bytes=1,
                    weight_metadata_bytes=999_999_999,
                )
                for matrix in requirement.matrices
            ),
        )
        with self.assertRaisesRegex(ValueError, "explicit decoded conversion contract"):
            _cim_rank_padded_weight_bytes(
                scenario, backing_changed, rank, "cim0"
            )

    def test_dynamic_rhs_and_plain_w4a16_keep_legacy_cpu_schedule(self):
        described = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        described_requirements = {
            item.item_id: item for item in _derive_requirements(described)
        }
        plain = _scenario_for_layer(_full_layer())
        plain_requirement = next(
            item
            for item in _derive_requirements(plain)
            if item.item_id == "full0.attention"
        )
        estimate = type("Estimate", (), {"service_ns": 1.0})()

        captured = []
        for scenario, requirement in (
            (described, described_requirements["full0.attention.qk"]),
            (plain, plain_requirement),
        ):
            with patch(
                "heterollm_sim.control_plane_planner.estimate_cpu_gemm",
                return_value=estimate,
            ) as estimator, patch(
                "heterollm_sim.control_plane_planner._route_cost", return_value=0.0
            ):
                _operator_cost(
                    scenario,
                    "ttft",
                    requirement,
                    scenario.hardware.component_map()["cpu0"],
                    None,
                    TopologyRouter(scenario.hardware),
                )
            captured.extend(
                call.args[2] for call in estimator.call_args_list
            )

        self.assertTrue(captured)
        self.assertTrue(all(not workload.packed_weight_formats for workload in captured))
        self.assertTrue(
            all(
                workload.packed_weight_transform_operations == 0
                for workload in captured
            )
        )

    def test_full_attention_primitives_follow_execution_descriptor(self):
        scenario = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        by_id = {
            item.item_id: item for item in _derive_requirements(scenario)
        }

        rope = by_id["full0.attention.rope"]
        self.assertEqual(rope.elements_per_token, 7168)
        self.assertEqual(rope.operation_elements_per_token, 1792)
        self.assertEqual(rope.input_count, 3)
        self.assertEqual(by_id["full0.attention.qk"].matrices[0].k, 6144)
        self.assertEqual(by_id["full0.attention.pv"].matrices[0].n, 6144)
        self.assertEqual(
            by_id["full0.attention.qk"].attention_score_heads, 24
        )
        self.assertEqual(
            by_id["full0.attention.softmax.reduce"].elements_per_token, 24
        )
        for suffix in (
            "attention.q_norm.reduce",
            "attention.q_norm.apply",
            "attention.k_norm.reduce",
            "attention.k_norm.apply",
            "attention.qk_scale",
            "attention.gate",
        ):
            self.assertIn("full0." + suffix, by_id)

        router = TopologyRouter(scenario.hardware)
        gpu = scenario.hardware.component_map()["gpu0"]
        estimate = type("Estimate", (), {"service_ns": 1.0})()
        options = PlacementPolicy(design_prefill_tokens=32)
        captured = {}
        for suffix in ("qk", "pv"):
            with patch(
                "heterollm_sim.control_plane_planner.estimate_gpu_gemm",
                return_value=estimate,
            ) as estimator:
                _operator_cost(
                    scenario,
                    "tpot",
                    by_id["full0.attention." + suffix],
                    gpu,
                    None,
                    router,
                    placement_policy=options,
                )
            captured[suffix] = estimator.call_args.args[2]
        self.assertEqual(
            (captured["qk"].m, captured["qk"].k, captured["qk"].n),
            (1, 6144, 32),
        )
        self.assertEqual(captured["qk"].output_storage_bytes, 1536)
        self.assertEqual(
            (captured["pv"].m, captured["pv"].k, captured["pv"].n),
            (1, 32, 6144),
        )
        self.assertEqual(captured["pv"].activation_storage_bytes, 1536)

    def test_qk_norm_descriptor_does_not_offer_qkv_rope_fusion(self):
        scenario = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        requirements = _derive_requirements(scenario)
        opportunities = _derive_fusion_opportunities(
            scenario,
            PlacementPolicy(),
            requirements,
        )

        self.assertNotIn(
            "qkv_rope",
            {
                opportunity.group
                for opportunity in opportunities
                if opportunity.layer_id == "full0"
            },
        )
        self.assertIn(
            "flash_attention",
            {
                opportunity.group
                for opportunity in opportunities
                if opportunity.layer_id == "full0"
            },
        )

    def test_descriptor_metadata_naturally_changes_mapping_fingerprint(self):
        baseline = _scenario_for_layer(_full_layer())
        described = _scenario_for_layer(
            _full_layer(_full_attention_metadata())
        )
        self.assertNotEqual(
            mapping_input_fingerprint(baseline),
            mapping_input_fingerprint(described),
        )


if __name__ == "__main__":
    unittest.main()
