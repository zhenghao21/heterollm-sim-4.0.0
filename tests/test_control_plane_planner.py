import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from heterollm_sim import control_plane_planner as control_plane_planner_module
from heterollm_sim import planner as planner_module
from heterollm_sim import serving as serving_module
from heterollm_sim.control_plane_planner import (
    PlacementPolicy,
    _Candidate,
    _derive_requirements,
    _operator_cost,
    _rank_local_weight_candidates,
    _solve_builtin,
    plan_runtime_placement,
)
from heterollm_sim.communication import TopologyRouter
from heterollm_sim.ir import (
    ComponentSpec,
    HardwareSpec,
    KVCachePolicy,
    LayerSpec,
    LinearAttentionSpec,
    LinkSpec,
    MTPBranchSpec,
    ParallelSpec,
    PortSpec,
    RankMappingSpec,
)
from heterollm_sim.control_plane_state import mapping_fingerprint_status
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.planner import compile_scenario
from tests.model_helpers import model_from_layer_specs


def _replace_component_profile(scenario, kind, profile_id, **changes):
    registries = {
        registry_kind: dict(registry)
        for registry_kind, registry in scenario.component_profiles.items()
    }
    registries[kind][profile_id] = replace(
        registries[kind][profile_id], **changes
    )
    return replace(scenario, component_profiles=registries)


def _with_weight_logical_bytes(scenario, logical_bytes_by_tensor):
    graph = scenario.model.graph
    return replace(
        scenario,
        model=replace(
            scenario.model,
            graph=replace(
                graph,
                tensors=tuple(
                    replace(
                        tensor,
                        logical_bytes=logical_bytes_by_tensor[tensor.tensor_id],
                    )
                    if tensor.tensor_id in logical_bytes_by_tensor
                    else tensor
                    for tensor in graph.tensors
                ),
            ),
        ),
    )


def _two_gpu_explicit_scenario():
    scenario = build_reference_scenario()
    component_map = scenario.hardware.component_map()
    gpu0 = component_map["gpu0"]
    nv0 = PortSpec(
        "nv0",
        "NVLink",
        "endpoint",
        version="4.0",
        lanes=8,
        bandwidth_gbps=400.0,
    )
    gpu1 = ComponentSpec(
        "gpu1",
        "gpu",
        ports=(nv0,),
        package_id="package0",
        die_id="gpu1_die",
        capacity_bytes=gpu0.capacity_bytes,
        peak_ops_per_s=gpu0.peak_ops_per_s,
        cost_profile_id=gpu0.cost_profile_id,
    )
    hardware = replace(
        scenario.hardware,
        components=tuple(
            replace(component, ports=component.ports + (nv0,))
            if component.component_id == "gpu0"
            else component
            for component in scenario.hardware.components
        )
        + (gpu1,),
        links=scenario.hardware.links
        + (
            LinkSpec(
                "gpu0-gpu1",
                "gpu0",
                "nv0",
                "gpu1",
                "nv0",
                "NVLink",
                version="4.0",
                lanes=8,
                bandwidth_gbps=400.0,
                latency_ns=25.0,
                metadata={"shared_bidirectional": True},
            ),
        ),
    )
    parallel = ParallelSpec(
        tp_degree=2,
        rank_mapping=(
            RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0", "cim0"),
            RankMappingSpec(1, "gpu1", 1, 0, 0, "hbm0", "cim0"),
        ),
        layer_to_stage={"dense0": 0, "moe1": 0},
    )
    placement = replace(scenario.placement, parallel=parallel)
    return replace(scenario, hardware=hardware, placement=placement)


def _two_gpu_local_hbm_scenario():
    scenario = _two_gpu_explicit_scenario()
    components = scenario.hardware.component_map()
    gpu0_hbm_port = next(
        port for port in components["gpu0"].ports if port.protocol == "HBM"
    )
    hbm8_port = replace(gpu0_hbm_port, port_id="hbm8")
    hbm8 = replace(components["hbm0"], component_id="hbm8")
    gpu1 = replace(
        components["gpu1"],
        ports=components["gpu1"].ports + (hbm8_port,),
    )
    template_link = next(
        link for link in scenario.hardware.links if link.link_id == "gpu-hbm0"
    )
    hardware = replace(
        scenario.hardware,
        components=tuple(
            gpu1 if component.component_id == "gpu1" else component
            for component in scenario.hardware.components
        )
        + (hbm8,),
        links=scenario.hardware.links
        + (
            replace(
                template_link,
                link_id="gpu1-hbm8",
                source_component="gpu1",
                source_port="hbm8",
                target_component="hbm8",
            ),
        ),
    )
    hardware = replace(
        hardware,
        components=tuple(
            replace(component, capacity_bytes=1)
            if component.is_storage
            and "cim" not in component.kind.lower()
            and component.component_id not in {"hbm0", "hbm8"}
            else component
            for component in hardware.components
        ),
    )
    parallel = replace(
        scenario.placement.parallel,
        rank_mapping=(
            replace(
                scenario.placement.parallel.rank_mapping[0],
                memory_component_id="hbm0",
            ),
            replace(
                scenario.placement.parallel.rank_mapping[1],
                memory_component_id="hbm8",
            ),
        ),
    )
    placement = replace(
        scenario.placement,
        parallel=parallel,
    )
    return replace(scenario, hardware=hardware, placement=placement)


def _two_gpu_direct_hbf_candidate_scenario():
    scenario = _two_gpu_explicit_scenario()
    components = scenario.hardware.component_map()
    gpu1 = replace(
        components["gpu1"],
        ports=components["gpu1"].ports
        + (
            PortSpec(
                "hbf0",
                "UCIe",
                "endpoint",
                bandwidth_gbps=256.0,
                payload="streaming",
            ),
        ),
    )
    hbf0 = ComponentSpec(
        "hbf0",
        "hbf",
        ports=(
            PortSpec(
                "ucie",
                "UCIe",
                "endpoint",
                bandwidth_gbps=256.0,
                payload="streaming",
            ),
        ),
        package_id="package0",
        die_id="hbf0_die",
        capacity_bytes=16 * 1024**3,
        read_bandwidth_gbps=128.0,
        metadata={"writable": True},
    )
    hardware = replace(
        scenario.hardware,
        components=tuple(
            gpu1
            if component.component_id == "gpu1"
            else component
            for component in scenario.hardware.components
        )
        + (hbf0,),
        links=scenario.hardware.links
        + (
            LinkSpec(
                "gpu1-hbf0",
                "gpu1",
                "hbf0",
                "hbf0",
                "ucie",
                "UCIe",
                bandwidth_gbps=256.0,
                latency_ns=25.0,
                payload="streaming",
            ),
        ),
    )
    parallel = replace(
        scenario.placement.parallel,
        rank_mapping=tuple(
            replace(rank, memory_component_id=None)
            for rank in scenario.placement.parallel.rank_mapping
        ),
    )
    placement = replace(
        scenario.placement,
        parallel=parallel,
    )
    return replace(
        scenario,
        hardware=hardware,
        placement=placement,
    )


class ControlPlanePlannerTests(unittest.TestCase):
    def test_options_and_result_contract_are_json_serializable(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            PlacementPolicy(mode="unknown")
        with self.assertRaisesRegex(ValueError, "objective"):
            PlacementPolicy(objective="energy")
        with self.assertRaisesRegex(ValueError, "time_limit_s"):
            PlacementPolicy(time_limit_s=0)
        with self.assertRaisesRegex(ValueError, "tied_weight_runtime_copies"):
            PlacementPolicy(tied_weight_runtime_copies="yes")

        scenario = build_reference_scenario()
        result = plan_runtime_placement(scenario)
        payload = result.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["placement"]["parallel"]["tp_degree"], 1)
        self.assertEqual(
            payload["placement"]["kv_policy"]["offload_component"], "hbm1"
        )
        self.assertFalse(payload["surrogate"]["is_full_simulation"])
        self.assertEqual(
            result.placement.metadata["control_plane"]["evidence"]["fingerprint_schema"],
            "runtime-control-plane-v4",
        )
        self.assertFalse(hasattr(result, "patch"))

    def test_heuristic_is_deterministic_and_preserves_fixed_configuration(self):
        scenario = build_reference_scenario()
        first = plan_runtime_placement(scenario)
        second = plan_runtime_placement(scenario)
        self.assertEqual(first.placement, second.placement)
        self.assertEqual(first.decisions, second.decisions)
        self.assertEqual(first.unplaced, second.unplaced)
        self.assertEqual(first.objective_value, second.objective_value)
        self.assertIs(first.placement.parallel, scenario.placement.parallel)
        self.assertIs(first.placement.kv_policy, scenario.placement.kv_policy)
        self.assertEqual(
            first.placement.parallel,
            scenario.placement.parallel,
        )
        self.assertEqual(
            first.placement.kv_policy.offload_component,
            scenario.placement.kv_policy.offload_component,
        )
        applied = first.apply(scenario)
        self.assertIs(applied.hardware, scenario.hardware)
        self.assertIs(applied.workload, scenario.workload)

    def test_runtime_mtp_policy_does_not_change_mapping_fingerprint(self):
        scenario = build_reference_scenario()
        baseline = plan_runtime_placement(scenario)
        policy = scenario.workload.mtp
        self.assertIsNotNone(policy)
        changed_workload = replace(
            scenario.workload,
            mtp=replace(
                policy,
                candidate_tokens=policy.candidate_tokens + 3,
                acceptance_rate=0.125,
                acceptance_trace=(0.0, 1.0),
            ),
        )

        changed = plan_runtime_placement(
            replace(scenario, workload=changed_workload)
        )

        self.assertEqual(
            baseline.input_fingerprint, changed.input_fingerprint
        )
        self.assertEqual(
            baseline.placement.op_to_component,
            changed.placement.op_to_component,
        )
        self.assertEqual(
            baseline.placement.tensor_to_component,
            changed.placement.tensor_to_component,
        )

    def test_generated_kv_mapping_does_not_mirror_into_kv_policy(self):
        scenario = build_reference_scenario()
        result = plan_runtime_placement(scenario)
        self.assertTrue(result.fully_placed)
        generated_cache = result.placement.tensor_to_component["kv_cache"]
        self.assertEqual(generated_cache, scenario.placement.kv_policy.cache_component)

        frontend_placement = replace(
            result.placement,
            kv_policy=replace(
                result.placement.kv_policy,
                cache_component="hbm1",
            ),
        )
        frontend_scenario = replace(scenario, placement=frontend_placement)
        status = mapping_fingerprint_status(frontend_scenario)

        self.assertEqual(status["input_fingerprint"], result.input_fingerprint)
        self.assertNotEqual(status["current_input_fingerprint"], result.input_fingerprint)
        self.assertTrue(status["mapping_stale"])

    def test_replan_relocates_generated_state_after_topology_change(self):
        scenario = build_reference_scenario()
        authored = replace(
            scenario,
            placement=replace(
                scenario.placement,
                op_to_component={},
                tensor_to_component={},
                tensor_bytes={},
                kv_policy=replace(
                    scenario.placement.kv_policy,
                    cache_component=None,
                    offload_component="hostmem0",
                ),
            ),
        )
        first = plan_runtime_placement(authored)
        self.assertTrue(first.fully_placed, first.unplaced)
        mapped = first.apply(authored)
        self.assertEqual(mapped.placement.tensor_to_component["kv_cache"], "hbm0")
        self.assertIn(
            "kv_cache",
            mapped.placement.metadata["control_plane"]["decision"][
                "generated_tensor_ids"
            ],
        )

        components = tuple(
            component
            for component in mapped.hardware.components
            if component.component_id != "hbm0"
        )
        links = tuple(
            link
            for link in mapped.hardware.links
            if link.source_component != "hbm0"
            and link.target_component != "hbm0"
        )
        rank = replace(
            mapped.placement.parallel.rank_mapping[0],
            memory_component_id="hbm1",
        )
        changed = replace(
            mapped,
            hardware=replace(mapped.hardware, components=components, links=links),
            placement=replace(
                mapped.placement,
                parallel=replace(
                    mapped.placement.parallel,
                    rank_mapping=(rank,),
                ),
            ),
        )

        replanned = plan_runtime_placement(changed)

        self.assertTrue(replanned.fully_placed, replanned.unplaced)
        self.assertEqual(
            replanned.placement.tensor_to_component["kv_cache"], "hbm1"
        )

    def test_generated_kv_mapping_is_valid_when_policy_target_is_unspecified(self):
        scenario = build_reference_scenario()
        tensor_mapping = dict(scenario.placement.tensor_to_component)
        tensor_mapping.pop("kv_cache", None)
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes.pop("kv_cache", None)
        placement = replace(
            scenario.placement,
            tensor_to_component=tensor_mapping,
            tensor_bytes=tensor_bytes,
            kv_policy=replace(
                scenario.placement.kv_policy,
                cache_component=None,
            ),
        )

        result = plan_runtime_placement(replace(scenario, placement=placement))

        self.assertTrue(result.fully_placed, result.unplaced)
        self.assertIsNone(result.placement.kv_policy.cache_component)
        self.assertIn("kv_cache", result.placement.tensor_to_component)
        self.assertTrue(
            planner_module.validate_scenario(
                result.apply(replace(scenario, placement=placement))
            ).is_valid
        )

    def test_cim_gemm_weights_are_colocated_and_capacity_checked(self):
        scenario = build_reference_scenario()
        result = plan_runtime_placement(scenario)
        cims = scenario.hardware.component_map()
        cim_decisions = [
            decision
            for decision in result.decisions
            if decision.cim_eligible
            and "cim" in cims[decision.component_id].kind.lower()
        ]
        self.assertTrue(cim_decisions)
        for decision in cim_decisions:
            self.assertEqual(
                decision.component_id, decision.tensor_component_id
            )
            self.assertGreaterEqual(
                decision.padded_weight_bytes, decision.tensor_bytes
            )
            self.assertLessEqual(
                decision.padded_weight_bytes,
                scenario.cim_profile.weight_capacity_bytes,
            )
        physical = sum(
            (
                decision.padded_weight_bytes
                if decision.padded_weight_bytes
                else decision.tensor_bytes
            )
            for decision in result.decisions
            if decision.tensor_component_id == "cim0"
            and decision.tensor_id
            and "weight" in decision.tensor_id
        )
        self.assertLessEqual(physical, scenario.cim_profile.weight_capacity_bytes)
        schedule = compile_scenario(result.apply(scenario))
        self.assertTrue(
            any(
                any(
                    demand.resource_id == scenario.cim_profile.array_resource_id
                    for demand in task.demands
                )
                for task in schedule.tasks
            )
        )

    def test_rank_local_shards_do_not_materialize_aggregate_backing(self):
        scenario = _two_gpu_local_hbm_scenario()
        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=4),
        )
        metadata = result.placement.metadata["control_plane"]["decision"]
        attention = next(
            decision
            for decision in result.decisions
            if decision.item_id == "dense0.attention"
        )
        self.assertEqual(
            {shard.storage_component_id for shard in attention.rank_tensor_shards},
            {"hbm0", "hbm8"},
        )
        self.assertNotIn(attention.tensor_id, metadata["logical_weight_views"])
        self.assertEqual(
            {entry["rank_id"] for entry in metadata["rank_weight_shards"][attention.tensor_id]},
            {0, 1},
        )
        self.assertNotIn("model_weights", result.placement.tensor_to_component)
        self.assertNotIn("model_weights", result.placement.tensor_bytes)
        self.assertNotIn("aggregate_weight_backing", metadata)

    def test_rank_local_capacity_shortage_remains_fail_closed(self):
        scenario = _two_gpu_local_hbm_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=1)
                if component.component_id in {"hbm0", "hbm8"}
                else component
                for component in scenario.hardware.components
            ),
        )
        configured = replace(
            scenario,
            hardware=hardware,
        )

        result = plan_runtime_placement(
            configured,
            PlacementPolicy(gpu_loadable_layers=4),
        )
        attention = next(
            item
            for item in result.unplaced
            if item.item_id == "dense0.attention"
        )
        self.assertIn("rank-local", attention.reason)
        self.assertFalse(result.fully_placed)

    def test_hbf_is_not_a_resident_rank_local_weight_candidate(self):
        scenario = _two_gpu_direct_hbf_candidate_scenario()
        requirement = next(
            item
            for item in _derive_requirements(scenario)
            if item.item_id == "dense0.attention"
        )
        plan = build_parallel_plan(scenario)
        stores = tuple(
            component
            for component in scenario.hardware.components
            if component.component_id in {"hbm0", "hbf0"}
        )

        candidates = _rank_local_weight_candidates(
            scenario,
            requirement,
            0,
            TopologyRouter(scenario.hardware),
            plan.ranks_for_layer(requirement.layer),
            stores,
        )

        self.assertTrue(candidates)
        self.assertNotIn(
            "hbf0",
            {
                shard.storage_component_id
                for shards, _usage, _cost in candidates
                for shard in shards
            },
        )

    def test_resident_rank_weight_shards_on_hbf_fail_validation(self):
        scenario = _two_gpu_direct_hbf_candidate_scenario()
        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=4),
        )
        self.assertTrue(result.fully_placed, result.unplaced)
        mapped = result.apply(scenario)
        metadata = dict(mapped.placement.metadata["control_plane"]["decision"])
        rank_weight_shards = {
            tensor_id: [dict(entry) for entry in entries]
            for tensor_id, entries in metadata["rank_weight_shards"].items()
        }
        entry = rank_weight_shards["dense0.attention_weights"][0]
        entry["component_id"] = "hbf0"
        entry["storage_component_id"] = "hbf0"
        metadata["rank_weight_shards"] = rank_weight_shards
        tampered = replace(
            mapped,
            placement=replace(
                mapped.placement,
                metadata={
                    **mapped.placement.metadata,
                    "control_plane": {
                        **mapped.placement.metadata["control_plane"],
                        "decision": metadata,
                    },
                },
            ),
        )

        validation = planner_module.validate_scenario(tampered)

        self.assertFalse(validation.is_valid)
        self.assertTrue(
            any(
                "resident rank weight shard" in error and "hbf" in error
                for error in validation.errors_en
            ),
            validation.errors_en,
        )

    def test_cold_gpu_load_is_valid_and_cim_streaming_requires_option(self):
        scenario = build_reference_scenario()
        cold = replace(
            scenario,
            weights_resident=False,
        )
        profiles = {
            kind: dict(registry)
            for kind, registry in cold.component_profiles.items()
        }
        gpu_profile = profiles["gpu"]["legacy-gpu"]
        profiles["gpu"]["legacy-gpu"] = replace(
            gpu_profile,
            tensor_core=replace(
                gpu_profile.tensor_core,
                cycles_per_mma=1_000_000_000.0,
            ),
        )
        cpu_profile = profiles["cpu"]["legacy-cpu"]
        profiles["cpu"]["legacy-cpu"] = replace(
            cpu_profile,
            pipeline=replace(
                cpu_profile.pipeline,
                core_count=1,
                frequency_ghz=0.001,
            ),
        )
        cold = replace(cold, component_profiles=profiles)

        implicit = plan_runtime_placement(cold)
        self.assertTrue(implicit.fully_placed, implicit.unplaced)
        self.assertFalse(
            implicit.placement.metadata["control_plane"]["decision"][
                "cold_cim_streaming_tensors"
            ]
        )
        explicit = plan_runtime_placement(
            cold,
            PlacementPolicy(allow_cold_cim_streaming=True),
        )
        self.assertTrue(
            explicit.placement.metadata["control_plane"]["decision"][
                "cold_cim_streaming_tensors"
            ]
        )
        self.assertTrue(explicit.fully_placed, explicit.unplaced)
        metadata = explicit.placement.metadata["control_plane"]["decision"]
        tensor_id = next(iter(metadata["cold_cim_streaming_tensors"]))
        detail = metadata["weight_tensor_details"][tensor_id]
        self.assertEqual(detail["residency"], "cold_cim_transient")
        self.assertEqual(detail["backing_component_id"], "hostmem0")
        self.assertEqual(detail["backing_tensor_id"], tensor_id)
        self.assertEqual(detail["total_physical_bytes"], 0)
        self.assertTrue(metadata["rank_weight_shards"][tensor_id])
        self.assertEqual(
            {entry["physical_bytes"] for entry in metadata["rank_weight_shards"][tensor_id]},
            {0},
        )
        self.assertTrue(compile_scenario(explicit.apply(cold)).tasks)

    def test_cold_cim_backing_requires_artifact_capacity_and_route(self):
        cold = replace(build_reference_scenario(), weights_resident=False)
        requirement = next(
            item
            for item in _derive_requirements(cold)
            if item.item_id == "dense0.attention"
        )

        def selected(scenario):
            return control_plane_planner_module._select_cold_cim_backing_component(
                scenario,
                requirement,
                "cim0",
                TopologyRouter(scenario.hardware),
                ("cim0",),
                control_plane_planner_module._component_capacities(scenario),
                {},
            )

        self.assertEqual(selected(cold).component_id, "hostmem0")
        undersized = replace(
            cold,
            hardware=replace(
                cold.hardware,
                components=tuple(
                    replace(
                        component,
                        capacity_bytes=cold.model.total_declared_weight_bytes - 1,
                    )
                    if component.component_id == "hostmem0"
                    else component
                    for component in cold.hardware.components
                ),
            ),
        )
        self.assertIsNone(selected(undersized))
        disconnected = replace(
            cold,
            hardware=replace(
                cold.hardware,
                links=tuple(
                    link
                    for link in cold.hardware.links
                    if link.link_id != "cpu-hostmem-ddr"
                ),
                require_connected=False,
            ),
        )
        self.assertIsNone(selected(disconnected))

    def test_larger_model_refreshes_derived_bytes_without_aggregate_backing(self):
        scenario = build_reference_scenario()
        embedding_tensor_id = next(
            operator.weight_tensor_ids[0]
            for operator in scenario.model.graph.operators
            if operator.op_kind == "embedding"
        )
        embedding_bytes = scenario.model.embedding_weight_bytes + 8 * 1024 * 1024
        tensors = tuple(
            replace(tensor, logical_bytes=embedding_bytes)
            if tensor.tensor_id == embedding_tensor_id
            else tensor
            for tensor in scenario.model.graph.tensors
        )
        larger_model = replace(
            scenario.model,
            graph=replace(scenario.model.graph, tensors=tensors),
        )
        configured = replace(
            scenario,
            model=larger_model,
            placement=replace(
                scenario.placement,
                model_name=larger_model.name,
            ),
        )

        result = plan_runtime_placement(configured)
        metadata = result.placement.metadata["control_plane"]["decision"]

        self.assertNotIn("model_weights", result.placement.tensor_to_component)
        self.assertNotIn("model_weights", result.placement.tensor_bytes)
        self.assertNotIn("aggregate_weight_backing_bytes", metadata)
        self.assertNotIn("model_weights", metadata["physical_tensor_bytes"])
        self.assertEqual(
            metadata["derived_logical_model_weight_bytes"],
            larger_model.total_declared_weight_bytes,
        )
        self.assertEqual(
            metadata["declared_model_weight_bytes"],
            larger_model.total_declared_weight_bytes,
        )

    def test_disconnected_io_returns_clear_partial_mapping(self):
        scenario = build_reference_scenario()
        disconnected = replace(
            scenario.hardware,
            name="disconnected",
            links=(),
            require_connected=False,
        )
        configured = replace(
            scenario,
            hardware=disconnected,
            placement=replace(
                scenario.placement,
                hardware_name="disconnected",
            ),
        )
        result = plan_runtime_placement(configured)
        self.assertFalse(result.fully_placed)
        self.assertEqual(result.status, "partial")
        self.assertTrue(result.unplaced)
        reasons = " ".join(item.reason for item in result.unplaced)
        self.assertRegex(reasons, r"路由|并行计划|拓扑")

    def test_capacity_shortage_never_claims_full_placement(self):
        scenario = build_reference_scenario()
        components = tuple(
            replace(component, capacity_bytes=1)
            if component.is_storage or "cim" in component.kind.lower()
            else component
            for component in scenario.hardware.components
        )
        configured = replace(
            scenario,
            hardware=replace(scenario.hardware, components=components),
        )
        configured = _replace_component_profile(
            configured,
            "cim",
            "legacy-cim",
            weight_capacity_bytes=1,
        )
        result = plan_runtime_placement(configured)
        self.assertFalse(result.fully_placed)
        self.assertEqual(result.status, "partial")
        self.assertTrue(
            any("缺少" in item.reason or "容量" in item.reason for item in result.unplaced)
        )

    def test_derives_dense_moe_linear_attention_and_mtp(self):
        scenario = build_reference_scenario()
        linear = LayerSpec(
            "linear0",
            "dense",
            hidden_size=64,
            intermediate_size=128,
            attention_heads=4,
            kv_heads=2,
            sequence_mixer="linear_attention",
            linear_attention=LinearAttentionSpec(
                key_heads=2,
                value_heads=4,
                key_head_dim=16,
                value_head_dim=16,
                conv_kernel_size=4,
                state_dtype="fp32",
            ),
            dtype="int8",
            quantization="w8a8",
            weight_bytes=64 * 1024,
        )
        moe = LayerSpec(
            "moe1",
            "moe",
            hidden_size=64,
            intermediate_size=96,
            attention_heads=4,
            kv_heads=2,
            num_experts=4,
            experts_per_token=2,
            shared_expert_intermediate_size=32,
            shared_expert_gate=True,
            dtype="int8",
            quantization="w8a8",
            weight_bytes=256 * 1024,
        )
        model = model_from_layer_specs(
            "auto-hybrid",
            (linear, moe),
            vocabulary_size=256,
            embedding_weight_bytes=16 * 1024,
            mtp=MTPBranchSpec(
                prediction_layers=1,
                auxiliary_head=True,
                prediction_layer_weight_bytes=8 * 1024,
                auxiliary_head_weight_bytes=16 * 1024,
            ),
        )
        configured = replace(
            scenario,
            model=model,
            placement=replace(
                scenario.placement,
                model_name=model.name,
                parallel=replace(
                    scenario.placement.parallel,
                    layer_to_stage={"linear0": 0, "moe1": 0},
                ),
            ),
        )
        result = plan_runtime_placement(configured)
        kinds = {decision.kind for decision in result.decisions}
        self.assertTrue(
            {
                "embedding",
                "linear_attention",
                "linear_state_update",
                "attention",
                "experts",
                "router",
                "shared_expert",
                "shared_expert_gate",
                "lm_head",
                "mtp_prediction_layer",
                "mtp_aux_head",
                "state_tensor",
            }.issubset(kinds)
        )
        self.assertIn("linear_state", result.placement.tensor_to_component)
        self.assertNotIn("linear_state", result.placement.tensor_bytes)

    def test_mtp_prediction_and_aux_head_bytes_fall_back_independently(self):
        scenario = build_reference_scenario()
        layer = LayerSpec(
            "mtp0",
            "dense",
            hidden_size=64,
            intermediate_size=128,
            attention_heads=4,
            kv_heads=2,
            dtype="int8",
            quantization="w8a8",
            weight_bytes=64 * 1024,
        )

        declared_prediction = model_from_layer_specs(
            "mtp-declared-prediction",
            (layer,),
            vocabulary_size=32,
            embedding_weight_bytes=2_048,
            mtp=MTPBranchSpec(
                prediction_layers=2,
                auxiliary_head=True,
                prediction_layer_weight_bytes=1_024,
            ),
        )
        declared_aux = model_from_layer_specs(
            "mtp-declared-aux",
            (layer,),
            vocabulary_size=32,
            embedding_weight_bytes=2_048,
            mtp=MTPBranchSpec(
                prediction_layers=2,
                auxiliary_head=True,
                auxiliary_head_weight_bytes=4_096,
            ),
        )

        def mtp_bytes(model):
            configured = replace(
                scenario,
                model=model,
                placement=replace(
                    scenario.placement,
                    model_name=model.name,
                    parallel=replace(
                        scenario.placement.parallel,
                        layer_to_stage={"mtp0": 0},
                    ),
                ),
            )
            return {
                item.tensor_id: item.tensor_bytes
                for item in plan_runtime_placement(configured).decisions
                if item.kind in {"mtp_prediction_layer", "mtp_aux_head"}
            }

        self.assertEqual(
            mtp_bytes(declared_prediction),
            {
                "mtp.prediction_layer.000.weights": 1_024,
                "mtp.prediction_layer.001.weights": 1_024,
                "mtp.aux_head.weights": 64 * 32,
            },
        )
        self.assertEqual(
            mtp_bytes(declared_aux),
            {
                "mtp.prediction_layer.000.weights": 64 * 64,
                "mtp.prediction_layer.001.weights": 64 * 64,
                "mtp.aux_head.weights": 4_096,
            },
        )

    def test_state_mapping_is_workload_independent_and_runtime_sized(self):
        scenario = build_reference_scenario()
        huge_request = replace(
            scenario.workload.requests[0],
            prompt_tokens=1_100_000,
            output_tokens=1,
        )
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=1 * 1024**3)
                if component.component_id == "hbm0"
                else component
                for component in scenario.hardware.components
            ),
        )
        configured = replace(
            scenario,
            hardware=hardware,
            workload=replace(
                scenario.workload, requests=(huge_request,)
            ),
        )
        baseline = plan_runtime_placement(
            replace(configured, workload=scenario.workload)
        )
        result = plan_runtime_placement(configured)
        self.assertEqual(result.placement, baseline.placement)
        self.assertEqual(
            result.placement.tensor_to_component["kv_cache"], "hbm0"
        )
        self.assertNotIn("kv_cache", result.placement.tensor_bytes)

        linear = LayerSpec(
            "linear0",
            "dense",
            hidden_size=64,
            intermediate_size=128,
            attention_heads=4,
            sequence_mixer="linear_attention",
            linear_attention=LinearAttentionSpec(2, 4, 16, 16, 4),
            dtype="int8",
            quantization="w8a8",
        )
        linear_model = model_from_layer_specs("linear-state-only", (linear,))
        read_only_hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, metadata={"read_only": True})
                if component.component_id == "hbm0"
                else component
                for component in scenario.hardware.components
            ),
        )
        linear_placement = replace(
            scenario.placement,
            model_name=linear_model.name,
            kv_policy=replace(
                scenario.placement.kv_policy,
                cache_component=None,
                offload_component=None,
            ),
            parallel=replace(
                scenario.placement.parallel,
                rank_mapping=(
                    replace(
                        scenario.placement.parallel.rank_mapping[0],
                        memory_component_id="hbm1",
                    ),
                ),
                layer_to_stage={"linear0": 0},
            ),
        )
        linear_result = plan_runtime_placement(
            replace(
                scenario,
                model=linear_model,
                hardware=read_only_hardware,
                placement=linear_placement,
                workload=replace(scenario.workload, mtp=None),
            )
        )
        self.assertTrue(linear_result.fully_placed, linear_result.unplaced)
        self.assertEqual(
            linear_result.placement.tensor_to_component["linear_state"],
            "hbm1",
        )
        self.assertNotIn("linear_state", linear_result.placement.tensor_bytes)

    def test_kv_policy_dtype_does_not_materialize_workload_capacity(self):
        scenario = build_reference_scenario()
        for dtype in ("int8", "fp16", "fp32"):
            policy = replace(scenario.placement.kv_policy, dtype=dtype)
            result = plan_runtime_placement(
                replace(
                    scenario,
                    placement=replace(
                        scenario.placement,
                        kv_policy=policy,
                    ),
                )
            )
            self.assertEqual(
                result.placement.tensor_to_component["kv_cache"],
                policy.cache_component,
            )
            self.assertNotIn("kv_cache", result.placement.tensor_bytes)

    def test_lm_head_is_tied_embedding_alias_without_extra_physical_weight(self):
        scenario = build_reference_scenario()
        result = plan_runtime_placement(scenario)
        metadata = result.placement.metadata["control_plane"]["decision"]
        self.assertEqual(
            metadata["derived_logical_model_weight_bytes"],
            scenario.model.total_declared_weight_bytes,
        )
        self.assertNotIn("lm_head_weights", result.placement.tensor_to_component)
        self.assertNotIn("lm_head_weights", result.placement.tensor_bytes)
        self.assertEqual(
            metadata["logical_weight_aliases"]["lm_head_weights"],
            "embedding_weights",
        )
        lm_head = next(
            decision for decision in result.decisions if decision.item_id == "lm_head"
        )
        self.assertEqual(lm_head.tensor_id, "embedding_weights")
        self.assertIn("与 embedding_weights 共享权重", lm_head.reason)

    def test_explicit_warm_tied_runtime_copies_use_cpu_input_and_gpu_output(self):
        scenario = build_reference_scenario()
        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(
                gpu_loadable_layers=4,
                tied_weight_runtime_copies=True,
            ),
        )
        self.assertTrue(result.fully_placed, result.unplaced)
        mapped = result.apply(scenario)
        decision = mapped.placement.metadata["control_plane"]["decision"]
        details = decision["weight_tensor_details"]
        embedding_bytes = scenario.model.embedding_weight_bytes

        self.assertEqual(
            decision["logical_weight_aliases"]["lm_head_weights"],
            "embedding_weights",
        )
        self.assertEqual(
            decision["derived_logical_model_weight_bytes"],
            scenario.model.total_declared_weight_bytes,
        )
        self.assertEqual(
            mapped.placement.tensor_to_component["embedding_weights"],
            "hostmem0",
        )
        self.assertEqual(
            mapped.placement.tensor_to_component["lm_head_weights"],
            "hbm0",
        )
        self.assertEqual(
            mapped.placement.tensor_bytes["embedding_weights"],
            embedding_bytes,
        )
        self.assertEqual(
            mapped.placement.tensor_bytes["lm_head_weights"],
            embedding_bytes,
        )
        self.assertEqual(
            sum(
                entry["physical_bytes"]
                for tensor_id in ("embedding_weights", "lm_head_weights")
                for entry in decision["rank_weight_shards"][tensor_id]
            ),
            2 * embedding_bytes,
        )
        self.assertEqual(
            details["embedding_weights"]["runtime_copy_role"],
            "input_embedding",
        )
        self.assertEqual(
            details["lm_head_weights"]["runtime_copy_of"],
            "embedding_weights",
        )
        with planner_module._compilation_scope(mapped):
            head_source = planner_module._weight_source_for_tensor(
                mapped,
                "lm_head_weights",
                "gpu0",
            )
            embedding_source = planner_module._weight_source_for_tensor(
                mapped,
                "embedding_weights",
                "cpu0",
            )
        self.assertEqual(
            head_source,
            ("hbm0", "lm_head_weights", "embedding_weights"),
        )
        self.assertEqual(
            embedding_source,
            ("hostmem0", "embedding_weights", "embedding_weights"),
        )

        schedule = compile_scenario(mapped)
        embedding_accesses = [
            task
            for task in schedule.tasks
            if task.metadata.get("model_operator_kind") == "embedding"
            and task.metadata.get("event_kind") == "model_weight_access"
        ]
        embedding_transfers = [
            task
            for task in schedule.tasks
            if task.metadata.get("model_operator_kind") == "embedding"
            and task.metadata.get("event_kind") == "model_weight_read"
        ]
        lm_head_accesses = [
            task
            for task in schedule.tasks
            if task.metadata.get("weight_tensor_id") == "lm_head_weights"
            and task.metadata.get("event_kind") == "model_weight_access"
        ]
        self.assertTrue(embedding_accesses)
        self.assertFalse(embedding_transfers)
        self.assertTrue(
            all(
                task.metadata["weight_source_component"] == "hostmem0"
                and task.metadata["weight_target_component"] == "cpu0"
                and task.metadata["bytes"]
                == task.metadata["lookup_workload_bytes"]
                and task.metadata["bytes"] < embedding_bytes
                and not task.metadata["weight_source_transfer_emitted"]
                for task in embedding_accesses
            )
        )
        self.assertEqual(
            {
                task.metadata["physical_weight_row_bytes"]
                for task in embedding_accesses
            },
            {
                (
                    embedding_bytes + scenario.model.vocabulary_size - 1
                )
                // scenario.model.vocabulary_size
            },
        )
        self.assertTrue(lm_head_accesses)
        self.assertTrue(
            all(
                task.metadata["weight_source_component"] == "hbm0"
                and task.metadata["bytes"] == embedding_bytes
                and task.metadata["tensor"] == "lm_head_weights"
                and task.metadata["logical_weight_tensor"]
                == "embedding_weights"
                for task in lm_head_accesses
            )
        )
        tied_accesses = [
            access
            for access in planner_module._ordered_residency_accesses(
                schedule.tasks
            )
            if access["requested_tensor_id"]
            in {"embedding_weights", "lm_head_weights"}
        ]
        self.assertTrue(tied_accesses)
        self.assertEqual(
            {
                (
                    access["requested_tensor_id"],
                    access["physical_tensor"],
                    access["canonical_owner_id"],
                )
                for access in tied_accesses
            },
            {
                (
                    "embedding_weights",
                    "embedding_weights",
                    "embedding_weights",
                ),
                (
                    "lm_head_weights",
                    "lm_head_weights",
                    "embedding_weights",
                ),
            },
        )

        residency_metadata = dict(mapped.placement.metadata)
        residency_metadata["capacity_ledger"] = {
            "component_id": "hbm0",
            "physical_capacity_bytes": mapped.hardware.get_component(
                "hbm0"
            ).capacity_bytes,
            "unified_pool": {"enabled": True},
        }
        residency_scenario = replace(
            mapped,
            placement=replace(
                mapped.placement,
                metadata=residency_metadata,
            ),
        )
        serving_plan = serving_module.compile_serving_plan(residency_scenario)
        residency_manager = serving_module._build_allocation_residency_manager(
            serving_plan
        )
        self.assertIsNotNone(residency_manager)
        self.assertIn("lm_head_weights", residency_manager.allocations)
        self.assertNotIn("lm_head_weights", residency_manager.views)
        self.assertEqual(
            residency_manager.allocations["lm_head_weights"].size_bytes,
            embedding_bytes,
        )
        online_runtime = serving_module._OnlineRuntime(
            serving_plan,
            lambda _cohort: serving_module.BatchCost(1.0, 0.0),
        )
        online_manager = online_runtime.residency_manager
        self.assertIsNotNone(online_manager)
        migration_count = online_manager.migration_count
        residency_audit, _temporary, _kv, _state = (
            online_runtime._apply_traced_owner_accesses(
                tied_accesses,
                cohort_id="tied-runtime-copy",
            )
        )
        self.assertTrue(residency_audit)
        self.assertEqual(
            {row["target_id"] for row in residency_audit},
            {"lm_head_weights"},
        )
        self.assertEqual(online_manager.migration_count, migration_count)
        expected_available = (
            residency_scenario.hardware.get_component("hbm0").capacity_bytes
            - sum(
                int(byte_count)
                for tensor_id, byte_count in (
                    residency_scenario.placement.tensor_bytes.items()
                )
                if residency_scenario.placement.tensor_to_component.get(tensor_id)
                == "hbm0"
                and tensor_id != "kv_cache"
            )
        )
        self.assertEqual(
            serving_module._dynamic_component_capacity(
                residency_scenario,
                "hbm0",
                ("kv_cache",),
            ),
            expected_available,
        )

    def test_tied_runtime_copy_capacity_and_scope_are_fail_closed(self):
        scenario = build_reference_scenario()
        embedding_bytes = scenario.model.embedding_weight_bytes
        cramped = replace(
            scenario,
            hardware=replace(
                scenario.hardware,
                components=tuple(
                    replace(component, capacity_bytes=embedding_bytes - 1)
                    if component.component_id == "hostmem0"
                    else component
                    for component in scenario.hardware.components
                ),
            ),
        )
        cramped_result = plan_runtime_placement(
            cramped,
            PlacementPolicy(
                gpu_loadable_layers=4,
                tied_weight_runtime_copies=True,
            ),
        )
        self.assertFalse(cramped_result.fully_placed)
        self.assertEqual(
            [(item.item_id, item.required_bytes) for item in cramped_result.unplaced],
            [("embedding", embedding_bytes)],
        )

        same_device = plan_runtime_placement(
            scenario,
            PlacementPolicy(
                gpu_loadable_layers=0,
                tied_weight_runtime_copies=True,
            ),
        )
        self.assertNotIn(
            "lm_head_weights", same_device.placement.tensor_to_component
        )
        self.assertFalse(
            any(
                details.get("runtime_copy_role")
                for details in same_device.placement.metadata["control_plane"]
                ["decision"]["weight_tensor_details"].values()
            )
        )

        cold = replace(scenario, weights_resident=False)
        with self.assertRaisesRegex(
            ValueError,
            "冷态跨设备副本的初始化/复制生命周期尚未建模",
        ):
            plan_runtime_placement(
                cold,
                PlacementPolicy(
                    gpu_loadable_layers=4,
                    tied_weight_runtime_copies=True,
                ),
            )

        mtp_alias_model = replace(
            scenario.model,
            metadata={
                "runtime_cost_contract": {
                    "weight_aliases": {
                        "lm_head_weights": "mtp.aux_head.weights"
                    }
                }
            },
        )
        mtp_alias = plan_runtime_placement(
            replace(scenario, model=mtp_alias_model),
            PlacementPolicy(
                gpu_loadable_layers=4,
                tied_weight_runtime_copies=True,
            ),
        )
        self.assertTrue(mtp_alias.fully_placed, mtp_alias.unplaced)
        self.assertNotIn("lm_head_weights", mtp_alias.placement.tensor_bytes)
        self.assertEqual(
            mtp_alias.placement.metadata["control_plane"]["decision"]
            ["logical_weight_aliases"]["lm_head_weights"],
            "mtp.aux_head.weights",
        )

        untied = _with_weight_logical_bytes(
            scenario,
            {
                "embedding_weights": embedding_bytes,
                "lm_head_weights": embedding_bytes + 1,
            },
        )
        untied_result = plan_runtime_placement(
            untied,
            PlacementPolicy(
                gpu_loadable_layers=4,
                tied_weight_runtime_copies=True,
            ),
        )
        self.assertTrue(untied_result.fully_placed, untied_result.unplaced)
        untied_details = untied_result.placement.metadata["control_plane"][
            "decision"
        ]["weight_tensor_details"]
        self.assertNotIn(
            "lm_head_weights",
            untied_result.placement.metadata["control_plane"]["decision"]
            ["logical_weight_aliases"],
        )
        self.assertFalse(
            any(
                detail.get("runtime_copy_role")
                for detail in untied_details.values()
            )
        )

    def test_lm_head_positive_logical_bytes_skip_embedding_alias(self):
        embedding_bytes = 93_592_576
        output_head_bytes = 144_643_072
        scenario = _with_weight_logical_bytes(
            build_reference_scenario(),
            {
                # Observed Qwen2.5-0.5B Q4_K_M GGUF tensor allocation bytes.
                "embedding_weights": embedding_bytes,
                "lm_head_weights": output_head_bytes,
            },
        )
        result = plan_runtime_placement(scenario)
        metadata = result.placement.metadata["control_plane"]["decision"]
        lm_head = next(
            decision for decision in result.decisions if decision.item_id == "lm_head"
        )

        self.assertNotIn(
            "lm_head_weights",
            metadata["logical_weight_aliases"],
        )
        self.assertEqual(lm_head.tensor_id, "lm_head_weights")
        self.assertEqual(lm_head.tensor_bytes, output_head_bytes)
        self.assertNotIn("共享权重", lm_head.reason)
        self.assertIn("lm_head_weights", result.placement.tensor_to_component)
        self.assertEqual(
            result.placement.tensor_bytes["lm_head_weights"],
            output_head_bytes,
        )
        self.assertEqual(
            metadata["derived_logical_model_weight_bytes"],
            scenario.model.total_declared_weight_bytes,
        )

    def test_untied_lm_head_weight_uses_only_final_pp_stage(self):
        scenario = build_reference_scenario()
        scenario = _with_weight_logical_bytes(
            scenario,
            {
                "lm_head_weights": scenario.model.embedding_weight_bytes + 1,
            },
        )
        parallel = ParallelSpec(
            pp_degree=2,
            rank_mapping=(
                RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0", None),
                RankMappingSpec(1, "gpu0", 0, 1, 0, "hbm0", None),
            ),
            layer_to_stage={"dense0": 0, "moe1": 1},
        )
        result = plan_runtime_placement(
            replace(
                scenario,
                placement=replace(scenario.placement, parallel=parallel),
                cim_interconnect=None,
            )
        )
        lm_head = next(
            decision for decision in result.decisions if decision.item_id == "lm_head"
        )
        metadata = result.placement.metadata["control_plane"]["decision"]

        self.assertTrue(result.fully_placed, result.unplaced)
        self.assertNotIn("lm_head_weights", metadata["logical_weight_aliases"])
        self.assertEqual(lm_head.tensor_id, "lm_head_weights")
        self.assertEqual(
            {shard.pp_rank for shard in lm_head.rank_tensor_shards},
            {1},
        )
        self.assertEqual(
            {
                entry["pp_rank"]
                for entry in metadata["rank_weight_shards"]["lm_head_weights"]
            },
            {1},
        )

    def test_weight_alias_rejects_missing_physical_owner(self):
        scenario = build_reference_scenario()
        model = replace(
            scenario.model,
            metadata={
                "runtime_cost_contract": {
                    "weight_aliases": {
                        "lm_head_weights": "missing.output.weight"
                    }
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "missing physical owner"):
            plan_runtime_placement(replace(scenario, model=model))

    def test_weight_alias_rejects_cycles(self):
        scenario = build_reference_scenario()
        model = replace(
            scenario.model,
            metadata={
                "runtime_cost_contract": {
                    "weight_aliases": {
                        "lm_head_weights": "embedding_weights",
                        "embedding_weights": "lm_head_weights",
                    }
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "contains a cycle"):
            plan_runtime_placement(replace(scenario, model=model))

    def test_weight_alias_rejects_positive_byte_mismatch(self):
        scenario = build_reference_scenario()
        model = replace(
            scenario.model,
            metadata={
                "runtime_cost_contract": {
                    "weight_aliases": {
                        "lm_head_weights": "embedding_weights"
                    },
                    "physical_layout": {
                        "tensor_evidence": {
                            "lm_head_weights": {
                                "physical_bytes": (
                                    scenario.model.embedding_weight_bytes + 1
                                )
                            }
                        }
                    },
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "byte evidence does not match"):
            plan_runtime_placement(replace(scenario, model=model))

    def test_authored_gpu_loadable_policy_rejects_invalid_count_and_order(self):
        scenario = build_reference_scenario()

        def configured(options):
            return replace(
                scenario,
                placement=replace(
                    scenario.placement,
                    metadata={
                        "control_plane": {
                            "policy": {"options": options},
                        }
                    },
                ),
            )

        with self.assertRaisesRegex(ValueError, "超过模型可加载单元"):
            plan_runtime_placement(
                configured(
                    {
                        "gpu_loadable_layers": 10_000,
                        "gpu_loadable_order": "tail",
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "目前必须为 tail"):
            plan_runtime_placement(
                configured(
                    {
                        "gpu_loadable_layers": 1,
                        "gpu_loadable_order": "head",
                    }
                )
            )

    def test_loadable_units_group_output_and_select_each_mtp_layer_independently(self):
        scenario = build_reference_scenario()
        layer = LayerSpec(
            "target0",
            "dense",
            hidden_size=64,
            intermediate_size=128,
            attention_heads=4,
            kv_heads=2,
            dtype="int8",
            quantization="w8a8",
            weight_bytes=64 * 1024,
        )
        aux_head_bytes = 2_048
        model = model_from_layer_specs(
            "multi-mtp-load-policy",
            (layer,),
            vocabulary_size=32,
            embedding_weight_bytes=2_048,
            mtp=MTPBranchSpec(
                prediction_layers=2,
                auxiliary_head=True,
                prediction_layer_weight_bytes=4_096,
                auxiliary_head_weight_bytes=aux_head_bytes,
            ),
            metadata={
                "runtime_cost_contract": {
                    "non_executable_tensor_bytes": 128,
                    "physical_layout": {
                        "tensor_evidence": {
                            "output_norm.weight": {"physical_bytes": 128},
                        }
                    },
                    "weight_aliases": {
                        "lm_head_weights": "mtp.aux_head.weights",
                    },
                }
            },
        )

        def planned(count):
            configured = replace(
                scenario,
                model=model,
                placement=replace(
                    scenario.placement,
                    model_name=model.name,
                    op_to_component={},
                    tensor_to_component={},
                    tensor_bytes={},
                    parallel=replace(
                        scenario.placement.parallel,
                        layer_to_stage={"target0": 0},
                    ),
                    metadata={
                        "control_plane": {
                            "policy": {
                                "options": {
                                    "gpu_loadable_layers": count,
                                    "gpu_loadable_order": "tail",
                                }
                            }
                        }
                    },
                ),
            )
            result = plan_runtime_placement(configured)
            self.assertTrue(result.fully_placed, result.unplaced)
            return result, {item.item_id: item for item in result.decisions}

        zero, zero_items = planned(0)
        for item_id in (
            "lm_head",
            "mtp.aux_head",
            "output_norm.weight",
            "mtp.prediction_layer.000",
            "mtp.prediction_layer.001",
        ):
            self.assertEqual(zero_items[item_id].component_id, "cpu0")
        self.assertEqual(
            zero.placement.tensor_to_component["mtp.aux_head.weights"],
            "hostmem0",
        )
        self.assertIsNone(zero_items["lm_head"].tensor_id)
        self.assertEqual(zero_items["lm_head"].tensor_bytes, 0)
        self.assertEqual(
            sum(
                item.tensor_bytes
                for item in zero.decisions
                if item.tensor_id == "mtp.aux_head.weights"
            ),
            aux_head_bytes,
        )
        self.assertEqual(
            zero.placement.tensor_to_component["output_norm.weight"],
            "hostmem0",
        )

        one, one_items = planned(1)
        for item_id in ("lm_head", "mtp.aux_head", "output_norm.weight"):
            self.assertEqual(one_items[item_id].component_id, "gpu0")
        for item_id in (
            "mtp.prediction_layer.000",
            "mtp.prediction_layer.001",
        ):
            self.assertEqual(one_items[item_id].component_id, "cpu0")
        self.assertEqual(
            one.placement.tensor_to_component["mtp.aux_head.weights"],
            "hbm0",
        )
        self.assertIsNone(one_items["lm_head"].tensor_id)
        self.assertEqual(one_items["lm_head"].tensor_bytes, 0)
        self.assertEqual(
            sum(
                item.tensor_bytes
                for item in one.decisions
                if item.tensor_id == "mtp.aux_head.weights"
            ),
            aux_head_bytes,
        )
        self.assertEqual(
            one.placement.tensor_to_component["output_norm.weight"],
            "hbm0",
        )

        _partial, partial_items = planned(2)
        self.assertEqual(
            partial_items["mtp.prediction_layer.000"].component_id,
            "gpu0",
        )
        self.assertEqual(
            partial_items["mtp.prediction_layer.001"].component_id,
            "cpu0",
        )
        self.assertEqual(partial_items["target0.attention"].component_id, "cpu0")

    def test_fixed_tp_rank_gpu_execution_checks_gpu1_and_aggregates_cost(self):
        scenario = _two_gpu_explicit_scenario()
        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=4),
        )
        decision = next(
            item for item in result.decisions if item.item_id == "dense0.attention"
        )
        self.assertIn(decision.component_id, {"gpu0", "gpu1"})
        self.assertEqual(set(decision.execution_component_ids), {"gpu0", "gpu1"})
        self.assertIn("stage/rank", decision.reason)

        single = build_reference_scenario()
        single_decision = next(
            item
            for item in plan_runtime_placement(
                single,
                PlacementPolicy(gpu_loadable_layers=4),
            ).decisions
            if item.item_id == "dense0.attention"
        )
        self.assertGreater(decision.analytical_cost, single_decision.analytical_cost)

    def test_load_policy_cpu_operator_has_explicit_rank_cpu_targets(self):
        scenario = build_reference_scenario()
        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=0),
        )
        decision = next(
            item for item in result.decisions if item.item_id == "dense0.mlp"
        )

        self.assertTrue(result.fully_placed, result.unplaced)
        self.assertEqual(decision.component_id, "cpu0")
        self.assertEqual(
            [
                (target.rank_id, target.compute_component_id, target.component_id)
                for target in decision.rank_execution_targets
            ],
            [(0, "gpu0", "cpu0")],
        )
        self.assertEqual(
            result.placement.metadata["control_plane"]["decision"][
                "operator_execution_targets"
            ]["dense0.mlp"][0]["component_id"],
            "cpu0",
        )

    def test_tp_weights_are_capacity_charged_and_read_from_rank_local_hbm(self):
        scenario = _two_gpu_local_hbm_scenario()
        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=4),
        )
        self.assertTrue(result.fully_placed, result.unplaced)
        decision = next(
            item for item in result.decisions if item.item_id == "dense0.attention"
        )
        self.assertEqual(
            {shard.compute_component_id for shard in decision.rank_tensor_shards},
            {"gpu0", "gpu1"},
        )
        self.assertEqual(
            {shard.storage_component_id for shard in decision.rank_tensor_shards},
            {"hbm0", "hbm8"},
        )
        self.assertEqual(
            sum(shard.logical_bytes for shard in decision.rank_tensor_shards),
            decision.tensor_bytes,
        )
        details = result.placement.metadata["control_plane"]["decision"][
            "weight_tensor_details"
        ]["dense0.attention_weights"]
        self.assertEqual(details["residency"], "rank_sharded_storage")
        schedule = compile_scenario(result.apply(scenario))
        sources = {
            task.metadata.get("weight_source_component")
            for task in schedule.tasks
            if task.metadata.get("logical_weight_tensor")
            == "dense0.attention_weights"
        }
        self.assertEqual(sources, {"hbm0", "hbm8"})

    def test_typed_mtp_ops_have_independent_tp_shard_ledgers(self):
        scenario = _two_gpu_local_hbm_scenario()

        result = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=4),
        )

        mtp = tuple(
            decision
            for decision in result.decisions
            if decision.kind in {"mtp_prediction_layer", "mtp_aux_head"}
        )
        self.assertEqual(
            [decision.item_id for decision in mtp],
            ["mtp.prediction_layer.000", "mtp.aux_head"],
        )
        self.assertEqual(
            [decision.tensor_id for decision in mtp],
            [
                "mtp.prediction_layer.000.weights",
                "mtp.aux_head.weights",
            ],
        )
        self.assertTrue(
            all(len(decision.rank_tensor_shards) == 2 for decision in mtp)
        )
        self.assertTrue(
            all(
                {shard.rank for shard in decision.rank_tensor_shards}
                == {0, 1}
                for decision in mtp
            )
        )
        metadata = result.placement.metadata["control_plane"]["decision"]
        self.assertNotIn("mtp_weights", metadata["rank_weight_shards"])
        for decision in mtp:
            shards = metadata["rank_weight_shards"][decision.tensor_id]
            self.assertEqual({entry["rank_id"] for entry in shards}, {0, 1})
            self.assertEqual(
                sum(entry["logical_bytes"] for entry in shards),
                decision.tensor_bytes,
            )

    def test_tp_pp_ep_metadata_covers_exact_stage_ranks_and_shard_policy(self):
        scenario = build_reference_scenario()
        ranks = []
        rank_id = 0
        for pp_rank in range(2):
            for ep_rank in range(2):
                for tp_rank in range(2):
                    ranks.append(
                        RankMappingSpec(
                            rank_id,
                            "gpu0",
                            tp_rank,
                            pp_rank,
                            ep_rank,
                            "hbm0",
                            None,
                        )
                    )
                    rank_id += 1
        parallel = ParallelSpec(
            tp_degree=2,
            pp_degree=2,
            ep_degree=2,
            rank_mapping=tuple(ranks),
            layer_to_stage={"dense0": 0, "moe1": 1},
        )
        configured = replace(
            scenario,
            placement=replace(scenario.placement, parallel=parallel),
            cim_interconnect=None,
        )

        result = plan_runtime_placement(configured)
        metadata = result.placement.metadata["control_plane"]["decision"]
        dense_targets = metadata["operator_execution_targets"][
            "dense0.attention"
        ]
        expert_targets = metadata["operator_execution_targets"]["moe1.experts"]
        self.assertEqual(len(dense_targets), 4)
        self.assertEqual({target["pp_rank"] for target in dense_targets}, {0})
        self.assertEqual(
            {(target["tp_rank"], target["ep_rank"]) for target in dense_targets},
            {(0, 0), (1, 0), (0, 1), (1, 1)},
        )
        self.assertEqual(len(expert_targets), 4)
        self.assertEqual({target["pp_rank"] for target in expert_targets}, {1})

        dense_shards = metadata["rank_weight_shards"][
            "dense0.attention_weights"
        ]
        self.assertEqual(len(dense_shards), 4)
        self.assertEqual({item["shard_count"] for item in dense_shards}, {2})
        self.assertEqual(len({item["shard_id"] for item in dense_shards}), 2)
        expert_shards = metadata["rank_weight_shards"][
            "moe1.expert_weights"
        ]
        self.assertEqual(len(expert_shards), 4)
        self.assertEqual({item["shard_count"] for item in expert_shards}, {4})
        self.assertEqual(len({item["shard_id"] for item in expert_shards}), 4)

    def test_cim_shared_by_two_ranks_requires_every_rank_route(self):
        scenario = _two_gpu_explicit_scenario()
        disconnected = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=1)
                if component.is_storage and "cim" not in component.kind.lower()
                else component
                for component in scenario.hardware.components
            ),
            links=tuple(
                link
                for link in scenario.hardware.links
                if link.link_id != "gpu0-gpu1"
            ),
            require_connected=False,
        )
        result = plan_runtime_placement(replace(scenario, hardware=disconnected))
        attention = [
            item for item in result.unplaced if item.item_id == "dense0.attention"
        ]
        self.assertTrue(attention)
        self.assertIn("激活值/输出路由", attention[0].reason)

    def test_cim_tensor_bytes_are_padded_physical_bytes_and_capacity_is_hard(self):
        scenario = build_reference_scenario()
        tiny = LayerSpec(
            "tiny",
            "dense",
            hidden_size=64,
            intermediate_size=256,
            attention_heads=4,
            kv_heads=4,
            dtype="int8",
            quantization="w8a8",
            weight_bytes=65_536,
        )
        model = model_from_layer_specs("tiny-cim", (tiny,))
        placement = replace(
            scenario.placement,
            model_name=model.name,
            parallel=replace(
                scenario.placement.parallel,
                layer_to_stage={"tiny": 0},
            ),
        )
        roomy = replace(
            scenario,
            model=model,
            placement=placement,
            hardware=replace(
                scenario.hardware,
                components=tuple(
                    replace(component, capacity_bytes=100_000)
                    if component.component_id == "cim0"
                    else replace(component, capacity_bytes=1)
                    if component.is_storage and "cim" not in component.kind.lower()
                    else component
                    for component in scenario.hardware.components
                ),
            ),
        )
        roomy = _replace_component_profile(
            roomy,
            "cim",
            "legacy-cim",
            weight_capacity_bytes=100_000,
        )
        result = plan_runtime_placement(roomy)
        decision = next(
            item for item in result.decisions if item.item_id == "tiny.mlp"
        )
        self.assertEqual(decision.tensor_bytes, 49_152)
        self.assertEqual(decision.padded_weight_bytes, 98_304)
        self.assertEqual(result.placement.tensor_bytes["tiny.mlp_weights"], 98_304)
        metadata = result.placement.metadata["control_plane"]["decision"]
        self.assertEqual(metadata["padded_tensor_bytes"]["tiny.mlp_weights"], 98_304)
        self.assertEqual(
            metadata["cim_total_physical_bytes"]["tiny.mlp_weights"], 98_304
        )

        cramped = replace(
            roomy,
            hardware=replace(
                roomy.hardware,
                components=tuple(
                    replace(component, capacity_bytes=29_576)
                    if component.component_id == "cim0"
                    else component
                    for component in roomy.hardware.components
                ),
            ),
        )
        cramped = _replace_component_profile(
            cramped,
            "cim",
            "legacy-cim",
            weight_capacity_bytes=29_576,
        )
        cramped_result = plan_runtime_placement(cramped)
        self.assertTrue(
            any(item.item_id == "tiny.mlp" for item in cramped_result.unplaced)
        )
        self.assertFalse(cramped_result.fully_placed)

    def test_retired_lock_fields_are_rejected_even_when_empty(self):
        scenario = build_reference_scenario()
        for field_name, value in (
            ("locked_op_keys", []),
            ("locked_tensor_ids", []),
            ("locked_op_keys", ["dense0.mlp"]),
            ("locked_tensor_ids", ["dense0.mlp_weights"]),
        ):
            with self.subTest(field_name=field_name, value=value):
                configured = replace(
                    scenario,
                    placement=replace(
                        scenario.placement,
                        metadata={
                            "control_plane": {
                                "policy": {field_name: value},
                            }
                        },
                    ),
                )
                with self.assertRaisesRegex(ValueError, "retired manual"):
                    plan_runtime_placement(configured)

    def test_retired_auto_mapping_and_unknown_policy_fields_are_rejected(self):
        scenario = build_reference_scenario()
        cases = (
            ({"auto_mapping": {}}, "auto_mapping"),
            (
                {"control_plane": {"policy": {"private_mode": True}}},
                "unknown fields",
            ),
            (
                {
                    "control_plane": {
                        "policy": {"options": {"private_mode": True}}
                    }
                },
                "unknown fields",
            ),
        )
        for metadata, message in cases:
            with self.subTest(metadata=metadata):
                configured = replace(
                    scenario,
                    placement=replace(scenario.placement, metadata=metadata),
                )
                with self.assertRaisesRegex(ValueError, message):
                    plan_runtime_placement(configured)

    def test_manual_placement_maps_are_rejected_with_fingerprint_evidence(self):
        scenario = build_reference_scenario()
        metadata = {
            "control_plane": {
                "policy": {},
                "decision": {},
                "evidence": {
                    "fingerprint_algorithm": "sha256",
                    "fingerprint_schema": "runtime-control-plane-v4",
                    "input_fingerprint": "0" * 64,
                },
            }
        }
        placements = (
            replace(
                scenario.placement,
                op_to_component={"dense0.mlp": "gpu0"},
                metadata=metadata,
            ),
            replace(
                scenario.placement,
                tensor_to_component={"dense0.mlp_weights": "hbm0"},
                metadata=metadata,
            ),
            replace(
                scenario.placement,
                tensor_bytes={"dense0.mlp_weights": 1},
                metadata=metadata,
            ),
        )
        for placement in placements:
            with self.subTest(placement=placement):
                with self.assertRaisesRegex(ValueError, "manual placement maps"):
                    plan_runtime_placement(replace(scenario, placement=placement))

    def test_fabricated_generated_ids_without_fingerprint_are_rejected(self):
        scenario = build_reference_scenario()
        placement = replace(
            scenario.placement,
            op_to_component={"dense0.mlp": "gpu0"},
            tensor_to_component={"dense0.mlp_weights": "hbm0"},
            tensor_bytes={"dense0.mlp_weights": 1},
            metadata={
                "control_plane": {
                    "policy": {},
                    "decision": {
                        "generated_op_keys": ["dense0.mlp"],
                        "generated_tensor_ids": ["dense0.mlp_weights"],
                        "derived_tensor_bytes": {"dense0.mlp_weights": 1},
                    },
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "fingerprint evidence"):
            plan_runtime_placement(replace(scenario, placement=placement))

    def test_valid_materialized_placement_can_be_replanned(self):
        scenario = build_reference_scenario()
        first = plan_runtime_placement(scenario)
        replanned = plan_runtime_placement(first.apply(scenario))
        self.assertTrue(replanned.fully_placed, replanned.unplaced)
        self.assertEqual(replanned.placement, first.placement)

    def test_zero_active_memory_capacity_warning_is_explicit(self):
        scenario = build_reference_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=0)
                if component.component_id == "hbm7"
                else component
                for component in scenario.hardware.components
            ),
        )
        result = plan_runtime_placement(replace(scenario, hardware=hardware))
        self.assertTrue(
            any(
                "hbm7" in warning and "按不可用处理" in warning
                for warning in result.warnings
            )
        )

    def test_zero_capacity_non_storage_components_do_not_warn_as_unbounded(self):
        scenario = build_reference_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=0)
                if component.component_id in {"gpu0", "cpu0"}
                else component
                for component in scenario.hardware.components
            ),
        )

        result = plan_runtime_placement(replace(scenario, hardware=hardware))

        self.assertFalse(
            any(
                "capacity_bytes=0" in warning and "无上限" in warning
                for warning in result.warnings
            )
        )

    def test_zero_offload_capacity_keeps_explicit_fail_closed_warning(self):
        scenario = _two_gpu_direct_hbf_candidate_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=0)
                if component.component_id == "hbf0"
                else component
                for component in scenario.hardware.components
            ),
        )

        result = plan_runtime_placement(replace(scenario, hardware=hardware))

        self.assertTrue(
            any(
                "hbf0" in warning
                and "capacity_bytes=0" in warning
                and "不会将其作为无限容量" in warning
                for warning in result.warnings
            )
        )

    def test_builtin_proves_small_optimum(self):
        result = plan_runtime_placement(
            build_reference_scenario(),
            PlacementPolicy(
                mode="optimal", solver="builtin", time_limit_s=2.0
            ),
        )
        self.assertEqual(result.status, "optimal")
        self.assertTrue(result.optimality_proven)
        self.assertEqual(result.solver, "builtin")
        self.assertEqual(result.gap, 0.0)
        self.assertEqual(result.objective_value, result.lower_bound)

    def test_builtin_exhaustion_promotes_final_bound_to_incumbent(self):
        candidate_lists = (
            (
                _Candidate(0, "cheap-0", None, 1.0, (("memory", 6),)),
                _Candidate(0, "fit-0", None, 2.0, (("memory", 4),)),
            ),
            (
                _Candidate(1, "cheap-1", None, 1.0, (("memory", 6),)),
                _Candidate(1, "fit-1", None, 2.0, (("memory", 4),)),
            ),
        )

        solve = _solve_builtin(
            candidate_lists,
            capacities={"memory": 10},
            base_usage={},
            time_limit_s=1.0,
        )

        self.assertTrue(solve.completed)
        self.assertEqual(solve.objective_value, 3.0)
        self.assertEqual(solve.lower_bound, solve.objective_value)

    def test_cold_cim_cost_loads_nonresident_weights_to_every_target(self):
        scenario = build_reference_scenario()
        cold = replace(
            scenario,
            weights_resident=False,
        )
        requirement = next(
            item
            for item in _derive_requirements(cold)
            if item.item_id == "dense0.attention"
        )
        cim = cold.hardware.component_map()["cim0"]
        router = TopologyRouter(cold.hardware)

        with patch(
            "heterollm_sim.control_plane_planner._cim_target_ids",
            return_value=("cim0", "cim1"),
        ), patch(
            "heterollm_sim.control_plane_planner._cim_execution_pairs",
            return_value=(),
        ), patch(
            "heterollm_sim.control_plane_planner._route_cost", return_value=11.0
        ) as route_cost:
            warm_cost = _operator_cost(
                cold, "ttft", requirement, cim, cim, router
            )
            route_cost.reset_mock()
            cold_cost = _operator_cost(
                cold,
                "ttft",
                requirement,
                cim,
                cim,
                router,
                cold_cim_streaming=True,
                cold_cim_backing_component_id="hostmem0",
            )
            backing_routes = [
                call.args[1:]
                for call in route_cost.call_args_list
                if call.args[1] == "hostmem0"
            ]

        self.assertGreater(cold_cost, warm_cost)
        self.assertEqual(
            backing_routes,
            [
                ("hostmem0", "cim0", requirement.tensor_bytes),
                ("hostmem0", "cim1", requirement.tensor_bytes),
            ],
        )

    def test_dynamic_qk_pv_cost_uses_activation_precision_for_rhs(self):
        scenario = build_reference_scenario()
        base_requirement = next(
            item
            for item in _derive_requirements(scenario)
            if item.item_id == "dense0.attention.qk"
        )
        layers_by_id = {
            item.layer.layer_id: item.layer
            for item in _derive_requirements(scenario)
            if item.layer is not None
        }
        layers_by_id[base_requirement.layer.layer_id] = replace(
            base_requirement.layer, quantization="w4a16", dtype="fp16"
        )
        model = model_from_layer_specs(
            "dynamic-rhs", tuple(layers_by_id.values())
        )
        configured = replace(scenario, model=model)
        requirement = next(
            item
            for item in _derive_requirements(configured)
            if item.item_id.endswith("attention.qk")
        )
        gpu = configured.hardware.component_map()["gpu0"]
        router = TopologyRouter(configured.hardware)
        estimate = type("Estimate", (), {"service_ns": 1.0})()

        with patch(
            "heterollm_sim.control_plane_planner.estimate_gpu_gemm",
            return_value=estimate,
        ) as estimator:
            _operator_cost(
                configured,
                "ttft",
                requirement,
                gpu,
                None,
                router,
            )

        self.assertTrue(requirement.dynamic_rhs)
        self.assertIsNone(requirement.tensor_id)
        self.assertTrue(estimator.call_args_list)
        self.assertTrue(
            all(call.args[2].activation_bits == 16 for call in estimator.call_args_list)
        )
        self.assertTrue(
            all(call.args[2].weight_bits == 16 for call in estimator.call_args_list)
        )

        cpu = configured.hardware.component_map()["cpu0"]
        dynamic_rhs_bytes = sum(
            (matrix.k * matrix.n * 16 + 7) // 8
            for matrix in requirement.matrices
        )
        with patch(
            "heterollm_sim.control_plane_planner.estimate_cpu_gemm",
            return_value=estimate,
        ), patch(
            "heterollm_sim.control_plane_planner._route_cost",
            return_value=0.0,
        ) as route_cost:
            _operator_cost(
                configured,
                "ttft",
                requirement,
                cpu,
                None,
                router,
            )

        self.assertTrue(
            any(
                call.args[1:] == ("gpu0", "cpu0", dynamic_rhs_bytes)
                for call in route_cost.call_args_list
            )
        )

    def test_timeout_returns_incumbent_with_bound(self):
        scenario = build_reference_scenario()
        constrained = _replace_component_profile(
            scenario,
            "cim",
            "legacy-cim",
            weight_capacity_bytes=20 * 1024 * 1024,
        )
        result = plan_runtime_placement(
            constrained,
            PlacementPolicy(
                mode="optimal", solver="builtin", time_limit_s=1e-9
            ),
        )
        self.assertIn(result.status, {"feasible_timeout", "partial_timeout"})
        self.assertFalse(result.optimality_proven)
        self.assertEqual(result.solver, "builtin")
        self.assertIsNotNone(result.lower_bound)
        self.assertIsNotNone(result.gap)

    def test_mapping_reuses_one_parallel_plan_for_all_candidates(self):
        with patch(
            "heterollm_sim.control_plane_planner.build_parallel_plan",
            wraps=control_plane_planner_module.build_parallel_plan,
        ) as build_plan:
            result = plan_runtime_placement(build_reference_scenario())

        self.assertTrue(result.fully_placed)
        self.assertEqual(build_plan.call_count, 1)

    def test_final_validation_reuses_one_execution_view(self):
        with patch(
            "heterollm_sim.planner.model_graph_execution_view",
            wraps=planner_module.model_graph_execution_view,
        ) as execution_view:
            report = planner_module.validate_scenario(
                build_reference_scenario()
            )

        self.assertTrue(report.is_valid)
        self.assertEqual(execution_view.call_count, 1)

    def test_candidate_generation_deadline_returns_partial_timeout(self):
        checks = 0

        def expire_during_candidates(run_context):
            nonlocal checks
            checks += 1
            if checks >= 8:
                raise control_plane_planner_module._MappingDeadlineExceeded

        with patch(
            "heterollm_sim.control_plane_planner._check_mapping_deadline",
            side_effect=expire_during_candidates,
        ):
            result = plan_runtime_placement(build_reference_scenario())

        self.assertEqual(result.status, "partial_timeout")
        self.assertFalse(result.fully_placed)
        self.assertTrue(result.unplaced)
        self.assertTrue(
            any("候选生成超过" in warning for warning in result.warnings)
        )

    def test_ortools_request_falls_back_when_not_installed(self):
        with patch(
            "heterollm_sim.control_plane_planner.importlib.import_module",
            side_effect=ImportError,
        ):
            result = plan_runtime_placement(
                build_reference_scenario(),
                PlacementPolicy(
                    mode="optimal", solver="ortools", time_limit_s=1.0
                ),
            )
        self.assertEqual(result.solver, "builtin")
        self.assertTrue(
            any("OR-Tools 不可用" in warning for warning in result.warnings)
        )

    def test_parallel_and_offload_objects_are_not_rewritten(self):
        scenario = build_reference_scenario()
        parallel = replace(
            scenario.placement.parallel,
            collective_algorithm="ring",
            routing_policy="lowest_latency",
        )
        placement = replace(
            scenario.placement,
            parallel=parallel,
            metadata={"user": "keep"},
        )
        configured = replace(scenario, placement=placement)
        result = plan_runtime_placement(configured)
        self.assertEqual(result.placement.parallel, parallel)
        self.assertEqual(result.placement.kv_policy, placement.kv_policy)
        self.assertEqual(result.placement.metadata["user"], "keep")
        self.assertEqual(
            result.placement.kv_policy.offload_component,
            placement.kv_policy.offload_component,
        )


if __name__ == "__main__":
    unittest.main()
