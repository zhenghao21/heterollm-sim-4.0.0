"""Behavior boundaries for declared llama.cpp physical projection calls."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.ir import (
    LayerSpec,
    LinearAttentionSpec,
    ParallelSpec,
    RankMappingSpec,
    WorkloadSpec,
)
from heterollm_sim.cost_models import HostGemmOffloadCapability
from heterollm_sim.planner import compile_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive
from tests.model_helpers import model_from_layer_specs


def _segment(segment_id, k, n, *, axis="n", format_name="Q8_0"):
    block_size, payload_bytes, metadata_bytes = {
        "Q4_0": (32, 16, 2),
        "Q8_0": (32, 32, 2),
    }[format_name]
    return {
        "segment_id": segment_id,
        "physical_tensor_name": "blk.0.{}.weight".format(segment_id),
        "k": k,
        "n": n,
        "format": format_name,
        "physical_bytes": (
            n
            * ((k + block_size - 1) // block_size)
            * (payload_bytes + metadata_bytes)
        ),
        "tp_shard_axis": axis,
    }


def _descriptors(*, include_alpha_beta=True):
    projections = {
        "attention.qkv": {
            "segments": [
                _segment("attn_q", 128, 128),
                _segment("attn_k", 128, 64),
                _segment("attn_v", 128, 64),
            ],
        },
        "attention.q": {
            "segments": [_segment("attn_q", 128, 128)],
            "alias_of": "attention.qkv",
        },
        "attention.k": {
            "segments": [_segment("attn_k", 128, 64)],
            "alias_of": "attention.qkv",
        },
        "attention.v": {
            "segments": [_segment("attn_v", 128, 64)],
            "alias_of": "attention.qkv",
        },
        "attention.output": {
            "segments": [_segment("attn_output", 128, 128, axis="k")],
        },
        "linear_attention.qkv": {
            "segments": [_segment("linear_qkv", 128, 256)],
        },
        "linear_attention.output_gate": {
            "segments": [_segment("linear_gate", 128, 128)],
        },
        "linear_attention.output": {
            "segments": [_segment("linear_output", 128, 128, axis="k")],
        },
        "mlp.up_gate": {
            "segments": [
                _segment("ffn_gate", 128, 256),
                _segment("ffn_up", 128, 256),
            ],
        },
        "mlp.gate": {
            "segments": [_segment("ffn_gate", 128, 256)],
            "alias_of": "mlp.up_gate",
        },
        "mlp.up": {
            "segments": [_segment("ffn_up", 128, 256)],
            "alias_of": "mlp.up_gate",
        },
        "mlp.down": {
            "segments": [_segment("ffn_down", 256, 128, axis="k")],
        },
    }
    if include_alpha_beta:
        projections.update(
            {
                "linear_attention.alpha": {
                    "segments": [_segment("linear_alpha", 128, 8)],
                },
                "linear_attention.beta": {
                    "segments": [_segment("linear_beta", 128, 8)],
                },
            }
        )
    return {
        "weight_projection_descriptors": {
            "schema_version": "heterollm.weight-projections/v1",
            "projections": projections,
        }
    }


def _attention_execution_descriptors():
    metadata = _descriptors()
    projections = metadata["weight_projection_descriptors"]["projections"]
    projections["attention.qkv"]["segments"] = [
        _segment("q", 128, 256),
        _segment("k", 128, 64),
        _segment("v", 128, 64),
    ]
    projections["attention.q"]["segments"][0] = _segment("q", 128, 256)
    projections["attention.k"]["segments"][0] = _segment("k", 128, 64)
    projections["attention.v"]["segments"][0] = _segment("v", 128, 64)
    metadata["attention_execution_descriptor"] = {
        "schema_version": "heterollm.attention-execution/v1",
        "query_heads": 8,
        "kv_heads": 4,
        "head_dim": 16,
        "query_width": 128,
        "gate_width": 128,
        "q_projection_width": 256,
        "rotary_dim": 16,
        "qk_scale": 0.25,
        "qk_norm": False,
        "gate_activation": "sigmoid",
    }
    return metadata


def _full_layer(metadata):
    return LayerSpec(
        "full0",
        "dense",
        hidden_size=128,
        intermediate_size=256,
        attention_heads=8,
        kv_heads=4,
        dtype="fp16",
        metadata=metadata,
    )


def _linear_layer(metadata):
    return LayerSpec(
        "linear0",
        "dense",
        hidden_size=128,
        intermediate_size=256,
        attention_heads=8,
        kv_heads=4,
        sequence_mixer="linear_attention",
        linear_attention=LinearAttentionSpec(
            key_heads=4,
            value_heads=8,
            key_head_dim=16,
            value_head_dim=16,
            conv_kernel_size=4,
            output_gate=True,
        ),
        dtype="fp16",
        metadata=metadata,
    )


def _scenario(
    layer,
    *,
    tokens=1,
    capability=None,
    conversion=False,
    vocabulary=0,
    gpu_cc=1200,
    tp_degree=1,
):
    base = build_reference_scenario()
    model = model_from_layer_specs(
        "physical-projection-test",
        (layer,),
        vocabulary_size=vocabulary,
    )
    metadata = {}
    if capability is not None:
        metadata["llama_cpp_physical_projection_invocations"] = capability
    if conversion:
        metadata["llama_cpp_f32_q8_1_mmvq"] = True
    placement = replace(
        base.placement,
        model_name=model.name,
        op_to_component={
            "attention": "gpu0",
            "linear_attention": "gpu0",
            "mlp": "gpu0",
            "lm_head": "gpu0",
        },
        tensor_to_component={"kv_cache": "hbm0", "linear_state": "hbm0"},
        tensor_bytes={"kv_cache": 64 * 1024, "linear_state": 64 * 1024},
        parallel=ParallelSpec(
            tp_degree=tp_degree,
            pp_degree=1,
            ep_degree=1,
            rank_mapping=tuple(
                RankMappingSpec(
                    rank=rank,
                    component_id="gpu0",
                    tp_rank=rank,
                    pp_rank=0,
                    ep_rank=0,
                    memory_component_id="hbm0",
                )
                for rank in range(tp_degree)
            ),
        ),
    )
    hardware = base.hardware
    if gpu_cc is not None:
        hardware = replace(
            hardware,
            components=tuple(
                replace(
                    component,
                    metadata={
                        **dict(component.metadata),
                        "cuda_compute_capability": gpu_cc,
                    },
                )
                if component.component_id == "gpu0"
                else component
                for component in hardware.components
            ),
        )
    return replace(
        base,
        model=model,
        hardware=hardware,
        placement=placement,
        workload=WorkloadSpec(
            "physical-projection-test",
            request_count=1,
            prompt_tokens=tokens,
            output_tokens=0,
            metadata=metadata,
        ),
    )


def _gemms(schedule, *projection_ids):
    wanted = set(projection_ids)
    return [
        task
        for task in schedule.tasks
        if task.metadata.get("phase") == "gpu_gemm"
        and task.metadata.get("projection_id") in wanted
    ]


class PhysicalProjectionInvocationTests(unittest.TestCase):
    def test_default_and_false_capability_keep_the_legacy_schedule(self):
        layer = _full_layer(_descriptors())
        default = compile_scenario(_scenario(layer))
        disabled = compile_scenario(_scenario(layer, capability=False))

        self.assertEqual(
            [to_primitive(task) for task in default.tasks],
            [to_primitive(task) for task in disabled.tasks],
        )
        legacy = _gemms(
            default,
            "attention.q",
            "attention.k",
            "attention.v",
            "mlp.gate",
            "mlp.up",
        )
        self.assertEqual(legacy, [])

    def test_complete_qkv_aliases_lower_one_call_per_tensor_without_byte_duplication(self):
        layer = _full_layer(_descriptors())
        schedule = compile_scenario(_scenario(layer, capability=True))
        projections = _gemms(
            schedule,
            "attention.q",
            "attention.k",
            "attention.v",
        )

        self.assertEqual(
            [task.metadata["projection_id"] for task in projections],
            ["attention.q", "attention.k", "attention.v"],
        )
        self.assertEqual(
            [task.metadata["weight_bytes"] for task in projections],
            [17_408, 8_704, 8_704],
        )
        tensors = [
            task.metadata["projection_segments"][0]["physical_tensor_name"]
            for task in projections
        ]
        self.assertEqual(tensors, ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"])
        self.assertEqual(len(tensors), len(set(tensors)))
        self.assertEqual(sum(task.metadata["weight_bytes"] for task in projections), 34_816)

    def test_mlp_fuses_only_single_row_homogeneous_gate_up_and_splits_multirow(self):
        layer = _full_layer(_descriptors())
        single_row = compile_scenario(_scenario(layer, tokens=1, capability=True))
        multirow = compile_scenario(_scenario(layer, tokens=4, capability=True))

        fused = _gemms(single_row, "mlp.up_gate", "mlp.gate", "mlp.up")
        self.assertEqual([task.metadata["projection_id"] for task in fused], ["mlp.up_gate"])
        self.assertTrue(fused[0].metadata["fusion_enabled"])

        split = _gemms(multirow, "mlp.up_gate", "mlp.gate", "mlp.up")
        self.assertEqual(
            [task.metadata["projection_id"] for task in split],
            ["mlp.gate", "mlp.up"],
        )
        self.assertTrue(
            any(task.metadata.get("event_kind") == "mlp_activation" for task in multirow.tasks)
        )
        self.assertTrue(all(not task.metadata["fusion_enabled"] for task in split))

        two_row = compile_scenario(_scenario(layer, tokens=2, capability=True))
        self.assertEqual(
            [
                task.metadata["projection_id"]
                for task in _gemms(two_row, "mlp.up_gate", "mlp.gate", "mlp.up")
            ],
            ["mlp.gate", "mlp.up"],
        )

        mixed = _descriptors()
        projections = mixed["weight_projection_descriptors"]["projections"]
        projections["mlp.up_gate"]["segments"][1] = _segment(
            "ffn_up", 128, 256, format_name="Q4_0"
        )
        projections["mlp.up"]["segments"][0] = _segment(
            "ffn_up", 128, 256, format_name="Q4_0"
        )
        mixed_schedule = compile_scenario(
            _scenario(_full_layer(mixed), tokens=1, capability=True)
        )
        self.assertEqual(
            [
                task.metadata["projection_id"]
                for task in _gemms(
                    mixed_schedule, "mlp.up_gate", "mlp.gate", "mlp.up"
                )
            ],
            ["mlp.gate", "mlp.up"],
        )

    def test_mlp_single_row_quantized_fusion_requires_declared_newer_cuda(self):
        layer = _full_layer(_descriptors())
        for gpu_cc in (None, 600):
            with self.subTest(gpu_cc=gpu_cc):
                schedule = compile_scenario(
                    _scenario(layer, capability=True, gpu_cc=gpu_cc)
                )
                projections = _gemms(schedule, "mlp.up_gate", "mlp.gate", "mlp.up")
                self.assertEqual(
                    [task.metadata["projection_id"] for task in projections],
                    ["mlp.gate", "mlp.up"],
                )
                self.assertTrue(
                    all(not task.metadata["fusion_enabled"] for task in projections)
                )

        legacy_without_capability = compile_scenario(
            _scenario(layer, gpu_cc=None)
        )
        self.assertEqual(
            [
                task.metadata["projection_id"]
                for task in _gemms(
                    legacy_without_capability,
                    "mlp.up_gate",
                    "mlp.gate",
                    "mlp.up",
                )
            ],
            ["mlp.up_gate"],
        )

    def test_partial_alias_group_fails_closed_when_capability_is_declared(self):
        metadata = _descriptors()
        del metadata["weight_projection_descriptors"]["projections"]["attention.v"]

        with self.assertRaisesRegex(ValueError, "physical projection group is incomplete"):
            compile_scenario(_scenario(_full_layer(metadata), capability=True))

    def test_aliases_must_partition_the_combined_tensor_without_duplication(self):
        mismatch = _descriptors()
        mismatch["weight_projection_descriptors"]["projections"]["attention.k"][
            "segments"
        ][0]["physical_tensor_name"] = "blk.0.not_attn_k.weight"
        with self.assertRaises(ValueError):
            compile_scenario(_scenario(_full_layer(mismatch), capability=True))

        duplicate = _descriptors()
        projections = duplicate["weight_projection_descriptors"]["projections"]
        projections["attention.qkv"]["segments"][1] = _segment("attn_q", 128, 64)
        projections["attention.k"]["segments"][0] = _segment("attn_q", 128, 64)
        with self.assertRaises(ValueError):
            compile_scenario(_scenario(_full_layer(duplicate), capability=True))

    def test_alpha_beta_follow_declared_geometry_and_are_absent_without_declaration(self):
        declared = compile_scenario(
            _scenario(_linear_layer(_descriptors()), capability=True)
        )
        controls = _gemms(
            declared,
            "linear_attention.alpha",
            "linear_attention.beta",
        )
        self.assertEqual(
            [task.metadata["projection_id"] for task in controls],
            ["linear_attention.alpha", "linear_attention.beta"],
        )
        for task in controls:
            self.assertEqual(task.metadata["cost_model"]["gemm_k"], 128)
            self.assertEqual(task.metadata["cost_model"]["gemm_n"], 8)
            self.assertEqual(task.metadata["weight_bytes"], 1_088)
            self.assertEqual(task.metadata["runtime_output_storage_bytes"], 32)
            self.assertEqual(task.metadata["runtime_input_storage_bits"], 32)
            self.assertEqual(task.metadata["runtime_output_storage_bits"], 32)

        legacy = compile_scenario(
            _scenario(_linear_layer(_descriptors(include_alpha_beta=False)), capability=True)
        )
        self.assertEqual(
            _gemms(legacy, "linear_attention.alpha", "linear_attention.beta"),
            [],
        )

        partial = _descriptors()
        del partial["weight_projection_descriptors"]["projections"][
            "linear_attention.beta"
        ]
        with self.assertRaisesRegex(ValueError, "physical projection group is incomplete"):
            compile_scenario(_scenario(_linear_layer(partial), capability=True))

        wrong_geometry = _descriptors()
        wrong_geometry["weight_projection_descriptors"]["projections"][
            "linear_attention.alpha"
        ]["segments"][0] = _segment("linear_alpha", 128, 16)
        with self.assertRaisesRegex(ValueError, "must map hidden width to value heads"):
            compile_scenario(
                _scenario(_linear_layer(wrong_geometry), capability=True)
            )

    def test_host_offloaded_controls_transfer_to_cpu_recurrent_scan(self):
        scenario = _scenario(_linear_layer(_descriptors()), capability=True)
        placement = replace(
            scenario.placement,
            op_to_component={
                **dict(scenario.placement.op_to_component),
                "linear_attention": "cpu0",
            },
            tensor_to_component={
                **dict(scenario.placement.tensor_to_component),
                "linear0.linear_attention_weights": "hostmem0",
            },
        )
        profiles = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        gpu_profiles = dict(profiles["gpu"])
        profile_id = next(iter(gpu_profiles))
        gpu_profiles[profile_id] = replace(
            gpu_profiles[profile_id],
            host_gemm_offload=HostGemmOffloadCapability(
                minimum_m=1, evidence="test host GEMM control offload"
            ),
            host_recurrent_offload=None,
        )
        profiles["gpu"] = gpu_profiles
        schedule = compile_scenario(
            replace(
                scenario,
                placement=placement,
                component_profiles=profiles,
            )
        )
        controls = _gemms(
            schedule, "linear_attention.alpha", "linear_attention.beta"
        )
        self.assertEqual(len(controls), 2)
        self.assertTrue(
            all(task.metadata["host_gemm_offload_applied"] for task in controls)
        )
        self.assertTrue(
            all(task.metadata["placement_component"] == "cpu0" for task in controls)
        )
        scan = next(
            task
            for task in schedule.tasks
            if task.metadata.get("linear_op") == "scan_recurrent_update"
            and task.metadata.get("phase") == "cpu_elementwise"
        )
        self.assertEqual(scan.metadata["target_component"], "cpu0")
        transfers = [
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "operator_input_transfer"
            and task.metadata.get("operator_id")
            == "linear0.linear_attention.state_update"
        ]
        self.assertEqual(
            [task.metadata["input_component"] for task in transfers],
            ["gpu0"],
        )
        self.assertEqual(
            [task.metadata["input_bytes"] for task in transfers], [64]
        )

    def test_tp_padding_uses_each_split_descriptor_local_output_width(self):
        schedule = compile_scenario(
            _scenario(
                _full_layer(_attention_execution_descriptors()),
                capability=True,
                tp_degree=3,
            )
        )
        projections = _gemms(schedule, "attention.q", "attention.k", "attention.v")
        self.assertEqual(len(projections), 9)
        expected = {
            "attention.q": (86, 172),
            "attention.k": (22, 44),
            "attention.v": (22, 44),
        }
        for task in projections:
            n, output_bytes = expected[task.metadata["projection_id"]]
            self.assertEqual(task.metadata["cost_model"]["gemm_n"], n)
            self.assertEqual(
                task.metadata["modeled_memory_write_bytes"], output_bytes
            )

    def test_q8_conversion_counts_only_declared_projection_calls_and_excludes_head(self):
        schedule = compile_scenario(
            _scenario(
                _full_layer(_descriptors()),
                capability=True,
                conversion=True,
                vocabulary=256,
            )
        )
        conversions = [
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "activation_quantization"
            and task.metadata.get("phase") == "kernel_launch"
        ]
        covered = {
            "attention.q",
            "attention.k",
            "attention.v",
            "attention.output",
            "mlp.up_gate",
            "mlp.down",
        }
        self.assertEqual(
            [task.metadata["projection_id"] for task in conversions],
            [
                "attention.q",
                "attention.k",
                "attention.v",
                "attention.output",
                "mlp.up_gate",
                "mlp.down",
            ],
        )
        self.assertEqual({task.metadata["projection_id"] for task in conversions}, covered)
        head = next(
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "lm_head_projection"
        )
        self.assertNotIn(head.task_id, {task.task_id for task in conversions})
        self.assertFalse(head.metadata.get("activation_conversion_applied", False))


if __name__ == "__main__":
    unittest.main()
