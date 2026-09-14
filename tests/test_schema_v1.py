import json
import unittest
from dataclasses import replace

import heterollm_sim.compiler_ir as compiler_ir
from heterollm_sim.compiler_ir import (
    COMPILER_ID,
    CompilerOptions,
    CanonicalizationError,
    CompilationPhase,
    compile_canonical_scenario,
)
from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.ir import (
    ParallelSpec,
    RankMappingSpec,
    model_graph_execution_view,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.schema_v1 import (
    SCHEMA_V1_VERSION,
    CanonicalScenario,
    CanonicalValidationError,
)


def _two_rank_scenario(*, control_plane_decision=None):
    scenario = build_reference_scenario()
    gpu0 = scenario.hardware.get_component("gpu0")
    hardware = replace(
        scenario.hardware,
        components=scenario.hardware.components
        + (replace(gpu0, component_id="gpu1"),),
        require_connected=False,
    )
    parallel = ParallelSpec(
        tp_degree=2,
        pp_degree=1,
        ep_degree=1,
        rank_mapping=(
            RankMappingSpec(
                rank=0,
                component_id="gpu0",
                tp_rank=0,
                pp_rank=0,
                ep_rank=0,
                memory_component_id="hbm0",
                cim_component_id="cim0",
            ),
            RankMappingSpec(
                rank=1,
                component_id="gpu1",
                tp_rank=1,
                pp_rank=0,
                ep_rank=0,
                memory_component_id="hbm1",
                cim_component_id="cim0",
            ),
        ),
        layer_to_stage={"dense0": 0, "moe1": 0},
    )
    metadata = dict(scenario.placement.metadata)
    if control_plane_decision is not None:
        metadata["control_plane"] = {
            "policy": {},
            "decision": control_plane_decision,
            "evidence": {},
        }
    placement = replace(
        scenario.placement,
        parallel=parallel,
        metadata=metadata,
    )
    return replace(scenario, hardware=hardware, placement=placement)


class CanonicalSchemaV1Tests(unittest.TestCase):
    def test_compiler_consumes_current_control_plane_contract(self):
        scenario = build_reference_scenario()
        mapped = plan_runtime_placement(scenario).apply(scenario)

        canonical = compile_canonical_scenario(mapped)

        self.assertTrue(
            any(
                target.derivation
                == "control_plane.decision.operator_execution_targets"
                for target in canonical.placement_plan.sub_operator_targets
            )
        )
        self.assertTrue(
            any(
                plan.attributes.get("source")
                == "rank_weight_shards"
                for plan in canonical.placement_plan.tensor_plans
            )
        )
        self.assertFalse(
            any(
                item.severity == "error"
                for item in canonical.compilation.diagnostics
            )
        )

    def test_reference_compiler_builds_all_canonical_sections_without_mutation(self):
        scenario = build_reference_scenario()
        before = scenario.placement.op_to_component

        canonical = compile_canonical_scenario(scenario)

        self.assertEqual(canonical.schema_version, SCHEMA_V1_VERSION)
        self.assertEqual(canonical.attributes["source_schema_version"], "4.0.0")
        self.assertEqual(canonical.hardware.graph_id, scenario.hardware.name)
        self.assertEqual(canonical.model.graph_id, scenario.model.name)
        self.assertEqual(canonical.workload.graph_id, scenario.workload.name)
        self.assertEqual(canonical.parallel_plan.world_size, 1)
        self.assertEqual(
            {stage.layer_id for stage in canonical.parallel_plan.layer_stages},
            {
                descriptor.layer_id
                for descriptor in model_graph_execution_view(
                    scenario.model.graph,
                    schema_version=scenario.model.schema_version,
                ).layer_instances
            },
        )
        self.assertTrue(canonical.model.source_operators)
        self.assertTrue(canonical.sub_operators)
        self.assertTrue(canonical.placement_plan.sub_operator_targets)
        self.assertTrue(canonical.placement_plan.tensor_plans)
        self.assertEqual(canonical.compilation.compiler_id, COMPILER_ID)
        self.assertEqual(
            tuple(stage.stage_id for stage in canonical.compilation.stages),
            tuple(phase.value for phase in CompilationPhase),
        )
        self.assertEqual(canonical.provenance[0].source_schema_version, "4.0.0")
        self.assertIs(scenario.placement.op_to_component, before)
        profiles = canonical.attributes["profiles"]
        self.assertEqual(
            profiles["components"],
            compiler_ir.to_primitive(scenario.component_profiles),
        )
        self.assertEqual(
            profiles["component_bindings"]["gpu0"]["cost_profile_id"],
            scenario.hardware.get_component("gpu0").cost_profile_id,
        )
        self.assertEqual(
            next(
                item
                for item in canonical.hardware.nodes
                if item.node_id == "gpu0"
            ).attributes["cost_profile_id"],
            scenario.hardware.get_component("gpu0").cost_profile_id,
        )
        self.assertTrue(
            all(
                target.cost_profile_kind
                == next(
                    node
                    for node in canonical.hardware.nodes
                    if node.node_id == target.component_id
                ).attributes["cost_profile_kind"]
                and target.cost_profile_id
                == next(
                    node
                    for node in canonical.hardware.nodes
                    if node.node_id == target.component_id
                ).attributes["cost_profile_id"]
                for target in canonical.placement_plan.sub_operator_targets
            )
        )

    def test_serialization_is_stable_strict_and_round_trips(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        first = canonical.to_json(indent=None)
        second = canonical.to_json(indent=None)
        restored = CanonicalScenario.from_json(first)

        self.assertEqual(first, second)
        self.assertEqual(restored.to_json(indent=None), first)
        self.assertEqual(restored.digest, canonical.digest)
        self.assertEqual(json.loads(first)["schema_version"], "1.1")

        payload = canonical.to_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            CanonicalScenario.from_dict(payload)

    def test_removed_compiler_alias_is_not_exported(self):
        self.assertFalse(
            hasattr(compiler_ir, "canonical_scenario_from_config")
        )

    def test_runtime_primitive_registry_has_authoritative_parents_and_full_placement(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        source_ids = {
            item.operator_id for item in canonical.model.source_operators
        }
        sub_ids = [
            item.sub_operator_id for item in canonical.model.sub_operators
        ]

        self.assertEqual(len(sub_ids), len(set(sub_ids)))
        self.assertTrue(
            all(
                item.parent_operator_id in source_ids
                for item in canonical.model.sub_operators
            )
        )
        target_pairs = {
            (item.sub_operator_id, item.rank_id)
            for item in canonical.placement_plan.sub_operator_targets
        }
        expected_pairs = {
            (sub_operator.sub_operator_id, rank.rank_id)
            for sub_operator in canonical.model.sub_operators
            for rank in canonical.parallel_plan.ranks
            if rank.pp_rank == sub_operator.attributes["stage_id"]
        }
        self.assertEqual(target_pairs, expected_pairs)

    def test_explicit_authoring_parent_is_the_only_parent_fallback(self):
        scenario = build_reference_scenario()
        baseline = compile_canonical_scenario(scenario)
        primitive = next(
            item
            for item in baseline.sub_operators
            if item.sub_operator_id == "dense0.input_norm.apply"
        )
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                primitive.parent_operator_id: "cpu0",
            },
        )

        canonical = compile_canonical_scenario(
            replace(scenario, placement=placement)
        )
        target = next(
            item
            for item in canonical.placement_plan.sub_operator_targets
            if item.sub_operator_id == primitive.sub_operator_id
        )
        parent_target = next(
            item
            for item in canonical.placement_plan.operator_targets
            if item.operator_id == primitive.parent_operator_id
            and item.rank_id == target.rank_id
        )

        self.assertEqual(target.component_id, "cpu0")
        self.assertEqual(
            target.parent_fallback_operator_id,
            primitive.parent_operator_id,
        )
        self.assertEqual(target.derivation, "explicit_parent_operator_fallback")
        self.assertEqual(parent_target.component_id, "cpu0")
        self.assertEqual(target.cost_profile_kind, "cpu")
        self.assertEqual(
            target.cost_profile_id,
            scenario.hardware.get_component("cpu0").cost_profile_id,
        )
        self.assertEqual(
            (
                parent_target.cost_profile_kind,
                parent_target.cost_profile_id,
            ),
            (target.cost_profile_kind, target.cost_profile_id),
        )

    def test_partial_control_plane_sub_operator_targets_fail_closed(self):
        scenario = _two_rank_scenario(
            control_plane_decision={
                "operator_execution_targets": {
                    "dense0.mlp": [
                        {"rank_id": 0, "component_id": "gpu0"},
                        {"rank_id": 1, "component_id": "gpu1"},
                    ]
                }
            }
        )

        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(scenario)
        self.assertIn("sub_operator_target_missing", str(caught.exception))

    def test_rank_aware_operator_targets_prefer_control_plane_metadata(self):
        scenario = _two_rank_scenario()
        baseline = compile_canonical_scenario(scenario)
        direct = {}
        ranks_by_stage = {
            stage: [
                rank
                for rank in baseline.parallel_plan.ranks
                if rank.pp_rank == stage
            ]
            for stage in range(baseline.parallel_plan.pp_degree)
        }
        for sub_operator in baseline.sub_operators:
            direct[sub_operator.sub_operator_id] = [
                {
                    "rank_id": rank.rank_id,
                    "component_id": rank.compute_node_id,
                }
                for rank in ranks_by_stage[
                    sub_operator.attributes["stage_id"]
                ]
            ]
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                metadata={
                    **dict(scenario.placement.metadata),
                    "control_plane": {
                        "policy": {},
                        "decision": {
                            "operator_execution_targets": direct,
                        },
                        "evidence": {},
                    },
                },
            ),
        )

        canonical = compile_canonical_scenario(scenario)
        targets = [
            target
            for target in canonical.placement_plan.sub_operator_targets
            if target.sub_operator_id == "dense0.mlp"
        ]

        self.assertEqual(
            [(target.rank_id, target.component_id) for target in targets],
            [(0, "gpu0"), (1, "gpu1")],
        )
        self.assertTrue(
            all(
                target.derivation
                == "control_plane.decision.operator_execution_targets"
                for target in targets
            )
        )

    def test_authoring_cpu_operator_target_compiles_and_round_trips(self):
        scenario = build_reference_scenario()
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                "dense0.mlp": "cpu0",
            },
        )

        canonical = compile_canonical_scenario(
            replace(scenario, placement=placement)
        )
        targets = [
            target
            for target in canonical.placement_plan.sub_operator_targets
            if target.sub_operator_id == "dense0.mlp"
        ]

        self.assertEqual(
            [(target.rank_id, target.component_id) for target in targets],
            [(0, "cpu0")],
        )
        self.assertTrue(
            all(
                target.derivation == "authoring_op_mapping_rank_cpu"
                for target in targets
            )
        )
        restored = CanonicalScenario.from_json(canonical.to_json(indent=None))
        self.assertEqual(restored.digest, canonical.digest)

    def test_target_profile_is_resolved_from_actual_heterogeneous_component(self):
        scenario = build_reference_scenario()
        cpu0 = scenario.hardware.get_component("cpu0")
        default_profile_id = cpu0.cost_profile_id
        self.assertIsNotNone(default_profile_id)
        component_profiles = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        default_profile = component_profiles["cpu"][default_profile_id]
        alternate_profile_id = "alternate-cpu"
        component_profiles["cpu"][alternate_profile_id] = replace(
            default_profile,
            pipeline=replace(
                default_profile.pipeline,
                resource_id="cpu1.pipeline",
            ),
        )
        hardware = replace(
            scenario.hardware,
            components=scenario.hardware.components
            + (
                replace(
                    cpu0,
                    component_id="cpu1",
                    cost_profile_id=alternate_profile_id,
                ),
            ),
            require_connected=False,
        )
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                "dense0.mlp": "cpu1",
            },
        )
        configured = replace(
            scenario,
            hardware=hardware,
            placement=placement,
            component_profiles=component_profiles,
        )

        canonical = compile_canonical_scenario(configured)
        target = next(
            item
            for item in canonical.placement_plan.sub_operator_targets
            if item.sub_operator_id == "dense0.mlp"
        )

        self.assertEqual(target.component_id, "cpu1")
        self.assertEqual(target.cost_profile_kind, "cpu")
        self.assertEqual(target.cost_profile_id, alternate_profile_id)
        self.assertIs(
            configured.resolve_component_profile(target.component_id),
            component_profiles["cpu"][alternate_profile_id],
        )
        restored = CanonicalScenario.from_json(canonical.to_json(indent=None))
        self.assertEqual(restored.digest, canonical.digest)

    def test_round_trip_rejects_target_profile_binding_mismatch(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        payload = canonical.to_dict()
        payload["placement_plan"]["sub_operator_targets"][0][
            "cost_profile_id"
        ] = "wrong-profile"

        with self.assertRaises(CanonicalValidationError) as caught:
            CanonicalScenario.from_dict(payload)
        self.assertIn(
            "target_cost_profile_id_mismatch",
            {item.code for item in caught.exception.issues},
        )

    def test_rank_aware_control_plane_cpu_targets_compile_and_round_trip(self):
        scenario = build_reference_scenario()
        mapped = plan_runtime_placement(
            scenario,
            PlacementPolicy(gpu_loadable_layers=0),
        ).apply(scenario)

        canonical = compile_canonical_scenario(mapped)
        targets = [
            target
            for target in canonical.placement_plan.sub_operator_targets
            if target.sub_operator_id == "dense0.mlp"
        ]
        self.assertEqual(
            [(target.rank_id, target.component_id) for target in targets],
            [(0, "cpu0")],
        )
        restored = CanonicalScenario.from_json(canonical.to_json(indent=None))
        self.assertEqual(restored.digest, canonical.digest)

    def test_noncomputing_operator_target_is_still_rejected(self):
        scenario = _two_rank_scenario(
            control_plane_decision={
                "operator_execution_targets": {
                    "dense0.mlp": [
                        {"rank_id": 0, "component_id": "hbm0"},
                        {"rank_id": 1, "component_id": "hbm1"},
                    ]
                }
            }
        )

        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(scenario)
        self.assertIn("invalid_operator_target_component", str(caught.exception))

    def test_non_gemm_sub_operator_cannot_target_cim(self):
        scenario = build_reference_scenario()
        placement = replace(
            scenario.placement,
            op_to_component={
                **scenario.placement.op_to_component,
                "dense0.mlp.activation": "cim0",
            },
        )

        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(
                replace(scenario, placement=placement)
            )
        message = str(caught.exception)
        self.assertIn("non_gemm_cim_target_unsupported", message)
        self.assertIn("operator_id=dense0.mlp.activation", message)
        self.assertIn("operator_class=elementwise", message)
        self.assertIn("requested_target=cim0", message)
        self.assertIn("resolved_target=cim0", message)
        self.assertIn("resolution_applied=false", message)
        diagnostic = next(
            item
            for item in caught.exception.diagnostics
            if item.code == "non_gemm_cim_target_unsupported"
        )
        self.assertEqual(
            diagnostic.attributes,
            {
                "operator_id": "dense0.mlp.activation",
                "operator_class": "elementwise",
                "requested_target": "cim0",
                "resolved_target": "cim0",
                "resolution_applied": False,
            },
        )

    def test_explicit_non_gemm_cim_target_fails_with_structured_diagnostic(self):
        scenario = build_reference_scenario()
        baseline = compile_canonical_scenario(scenario)
        direct = {
            sub_operator.sub_operator_id: [
                {
                    "rank_id": rank.rank_id,
                    "component_id": rank.compute_node_id,
                }
                for rank in baseline.parallel_plan.ranks
                if rank.pp_rank == sub_operator.attributes["stage_id"]
            ]
            for sub_operator in baseline.sub_operators
        }
        direct["dense0.input_norm.reduce"] = [
            {"rank_id": 0, "component_id": "cim0"}
        ]
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                metadata={
                    **dict(scenario.placement.metadata),
                    "control_plane": {
                        "policy": {},
                        "decision": {
                            "operator_execution_targets": direct,
                        },
                        "evidence": {},
                    },
                },
            ),
        )

        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(configured)
        diagnostic = next(
            item
            for item in caught.exception.diagnostics
            if item.code == "non_gemm_cim_target_unsupported"
        )
        self.assertEqual(
            diagnostic.attributes,
            {
                "operator_id": "dense0.input_norm.reduce",
                "operator_class": "reduction",
                "requested_target": "cim0",
                "resolved_target": "cim0",
                "resolution_applied": False,
            },
        )

    def test_compiled_non_gemm_cim_target_fails_closed(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        payload = canonical.to_dict()
        target = next(
            item
            for item in payload["placement_plan"]["sub_operator_targets"]
            if item["sub_operator_id"] == "dense0.input_norm.reduce"
        )
        cim_node = next(
            item
            for item in payload["hardware"]["nodes"]
            if item["node_id"] == "cim0"
        )
        target["component_id"] = "cim0"
        target["cost_profile_kind"] = cim_node["attributes"][
            "cost_profile_kind"
        ]
        target["cost_profile_id"] = cim_node["attributes"][
            "cost_profile_id"
        ]
        target["attributes"].update(
            {
                "requested_target": "cim0",
                "resolved_target": "cim0",
                "resolution_applied": False,
            }
        )

        with self.assertRaises(CanonicalValidationError) as caught:
            CanonicalScenario.from_dict(payload)
        issue = next(
            item
            for item in caught.exception.issues
            if item.code == "non_gemm_cim_target_unsupported"
        )
        self.assertIn("operator_id=dense0.input_norm.reduce", issue.message)
        self.assertIn("operator_class=reduction", issue.message)
        self.assertIn("requested_target=cim0", issue.message)
        self.assertIn("resolved_target=cim0", issue.message)
        self.assertIn("resolution_applied=false", issue.message)

    def test_rank_weight_shards_are_consumed_without_inference(self):
        tensor_id = "dense0.mlp_weights"
        scenario = _two_rank_scenario(
            control_plane_decision={
                "rank_weight_shards": {
                    tensor_id: [
                        {
                            "rank_id": 0,
                            "component_id": "hbm0",
                            "shard_index": 0,
                            "shard_count": 2,
                            "logical_bytes": 786432,
                            "physical_bytes": 786432,
                        },
                        {
                            "rank_id": 1,
                            "component_id": "hbm1",
                            "shard_index": 1,
                            "shard_count": 2,
                            "logical_bytes": 786432,
                            "physical_bytes": 786432,
                        },
                    ]
                }
            }
        )

        canonical = compile_canonical_scenario(scenario)
        tensor_plan = next(
            plan
            for plan in canonical.placement_plan.tensor_plans
            if plan.tensor_id == tensor_id
        )

        self.assertEqual(
            [(shard.shard_index, shard.shard_count) for shard in tensor_plan.shards],
            [(0, 2), (1, 2)],
        )
        self.assertEqual(
            [(replica.rank_id, replica.component_id) for replica in tensor_plan.replicas],
            [(0, "hbm0"), (1, "hbm1")],
        )
        self.assertTrue(
            all(
                shard.derivation == "control_plane.decision.rank_weight_shards"
                for shard in tensor_plan.shards
            )
        )

    def test_weight_tensor_details_preserve_replica_semantics(self):
        tensor_id = "dense0.mlp_weights"
        scenario = build_reference_scenario()
        metadata = dict(scenario.placement.metadata)
        metadata["control_plane"] = {
            "policy": {},
            "decision": {
                "weight_tensor_details": {
                    tensor_id: {
                        "logical_bytes": 100,
                        "placement_bytes": 128,
                        "padded_bytes_per_replica": 128,
                        "replica_component_ids": ["cim0"],
                        "total_physical_bytes": 128,
                        "backing_tensor_id": None,
                        "backing_component_id": None,
                        "residency": "warm_cim_resident",
                    }
                }
            },
            "evidence": {},
        }
        scenario = replace(
            scenario,
            placement=replace(scenario.placement, metadata=metadata),
        )

        canonical = compile_canonical_scenario(scenario)
        tensor_plan = next(
            plan
            for plan in canonical.placement_plan.tensor_plans
            if plan.tensor_id == tensor_id
        )

        self.assertEqual(tensor_plan.logical_bytes, 100)
        self.assertEqual(tensor_plan.shards[0].shard_count, 1)
        self.assertEqual(tensor_plan.replicas[0].physical_bytes, 128)
        self.assertEqual(tensor_plan.replicas[0].residency, "warm_cim_resident")

    def test_current_control_plane_rank_shards_are_consumed(self):
        tensor_id = "dense0.mlp_weights"
        scenario = _two_rank_scenario(
            control_plane_decision={
                "weight_tensor_details": {
                    tensor_id: {
                        "logical_bytes": 1_572_864,
                        "residency": "rank_sharded_storage",
                    }
                },
                "rank_weight_shards": {
                    tensor_id: [
                        {
                            "rank_id": 0,
                            "tp_rank": 0,
                            "pp_rank": 0,
                            "ep_rank": 0,
                            "compute_component_id": "gpu0",
                            "component_id": "hbm0",
                            "shard_index": 0,
                            "shard_count": 2,
                            "logical_bytes": 786432,
                            "physical_bytes": 786432,
                            "shard_kind": "tp_shard",
                        },
                        {
                            "rank_id": 1,
                            "tp_rank": 1,
                            "pp_rank": 0,
                            "ep_rank": 0,
                            "compute_component_id": "gpu1",
                            "component_id": "hbm1",
                            "shard_index": 1,
                            "shard_count": 2,
                            "logical_bytes": 786432,
                            "physical_bytes": 786432,
                            "shard_kind": "tp_shard",
                        },
                    ]
                },
            }
        )

        canonical = compile_canonical_scenario(scenario)
        tensor_plan = next(
            plan
            for plan in canonical.placement_plan.tensor_plans
            if plan.tensor_id == tensor_id
        )

        self.assertEqual(
            [(shard.shard_index, shard.shard_count) for shard in tensor_plan.shards],
            [(0, 2), (1, 2)],
        )
        self.assertEqual(
            [(replica.rank_id, replica.component_id) for replica in tensor_plan.replicas],
            [(0, "hbm0"), (1, "hbm1")],
        )
        self.assertTrue(
            all(
                shard.derivation == "control_plane.decision.rank_weight_shards"
                for shard in tensor_plan.shards
            )
        )

    def test_dense_ep_rank_shards_become_replicas_of_one_tp_shard(self):
        tensor_id = "dense0.mlp_weights"
        scenario = _two_rank_scenario()
        parallel = ParallelSpec(
            tp_degree=1,
            pp_degree=1,
            ep_degree=2,
            rank_mapping=(
                RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0", "cim0"),
                RankMappingSpec(1, "gpu1", 0, 0, 1, "hbm1", "cim0"),
            ),
            layer_to_stage={"dense0": 0, "moe1": 0},
        )
        metadata = {
            "control_plane": {
                "policy": {},
                "decision": {
                    "weight_tensor_details": {
                        tensor_id: {
                            "logical_bytes": 1_572_864,
                            "residency": "rank_sharded_storage",
                        }
                    },
                    "rank_weight_shards": {
                        tensor_id: [
                            {
                                "rank_id": rank,
                                "tp_rank": 0,
                                "pp_rank": 0,
                                "ep_rank": rank,
                                "compute_component_id": "gpu{}".format(rank),
                                "component_id": "hbm{}".format(rank),
                                "shard_index": 0,
                                "shard_count": 1,
                                "logical_bytes": 1_572_864,
                                "physical_bytes": 1_572_864,
                                "shard_kind": "tp_shard_ep_replica",
                            }
                            for rank in range(2)
                        ]
                    },
                },
                "evidence": {},
            },
        }
        scenario = replace(
            scenario,
            placement=replace(
                scenario.placement,
                parallel=parallel,
                metadata=metadata,
            ),
        )

        canonical = compile_canonical_scenario(scenario)
        tensor_plan = next(
            plan
            for plan in canonical.placement_plan.tensor_plans
            if plan.tensor_id == tensor_id
        )

        self.assertEqual(len(tensor_plan.shards), 1)
        self.assertEqual(tensor_plan.shards[0].shard_count, 1)
        self.assertEqual(
            [(replica.rank_id, replica.shard_id) for replica in tensor_plan.replicas],
            [(0, tensor_plan.shards[0].shard_id), (1, tensor_plan.shards[0].shard_id)],
        )

    def test_control_plane_tp_pp_ep_contract_compiles_without_rank_inference(self):
        scenario = build_reference_scenario()
        ranks = tuple(
            RankMappingSpec(
                pp_rank * 4 + ep_rank * 2 + tp_rank,
                "gpu0",
                tp_rank,
                pp_rank,
                ep_rank,
                "hbm0",
                None,
            )
            for pp_rank in range(2)
            for ep_rank in range(2)
            for tp_rank in range(2)
        )
        parallel = ParallelSpec(
            tp_degree=2,
            pp_degree=2,
            ep_degree=2,
            rank_mapping=ranks,
            layer_to_stage={"dense0": 0, "moe1": 1},
        )
        scenario = replace(
            scenario,
            placement=replace(scenario.placement, parallel=parallel),
            cim_interconnect=None,
        )
        mapped = plan_runtime_placement(scenario).apply(scenario)

        canonical = compile_canonical_scenario(mapped)
        dense = next(
            plan
            for plan in canonical.placement_plan.tensor_plans
            if plan.tensor_id == "dense0.attention_weights"
        )
        experts = next(
            plan
            for plan in canonical.placement_plan.tensor_plans
            if plan.tensor_id == "moe1.expert_weights"
        )
        self.assertEqual((len(dense.shards), len(dense.replicas)), (2, 4))
        self.assertEqual((len(experts.shards), len(experts.replicas)), (4, 4))
        dense_targets = [
            target
            for target in canonical.placement_plan.sub_operator_targets
            if target.sub_operator_id == "dense0.attention"
        ]
        expert_targets = [
            target
            for target in canonical.placement_plan.sub_operator_targets
            if target.sub_operator_id == "moe1.experts"
        ]
        self.assertEqual({target.rank_id for target in dense_targets}, {0, 1, 2, 3})
        self.assertEqual({target.rank_id for target in expert_targets}, {4, 5, 6, 7})

    def test_removed_execution_component_ids_contract_fails_explicitly(self):
        scenario = _two_rank_scenario(
            control_plane_decision={
                "execution_component_ids": {"dense0.mlp": ["cim0"]}
            }
        )
        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(scenario)
        self.assertEqual(caught.exception.phase, CompilationPhase.PLAN_PLACEMENT)
        self.assertIn("execution_component_ids is not part of the V4 contract", str(caught.exception))

    def test_incomplete_rank_weight_shards_fail_closed(self):
        tensor_id = "dense0.mlp_weights"
        scenario = _two_rank_scenario(
            control_plane_decision={
                "rank_weight_shards": {
                    tensor_id: [
                        {
                            "rank_id": 0,
                            "component_id": "hbm0",
                            "shard_index": 0,
                            "shard_count": 2,
                            "logical_bytes": 786432,
                            "physical_bytes": 786432,
                        }
                    ]
                }
            }
        )

        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(scenario)
        self.assertEqual(caught.exception.phase, CompilationPhase.PLAN_PLACEMENT)
        self.assertIn("cover ranks [0], expected [0, 1]", str(caught.exception))

    def test_cross_graph_validation_rejects_dangling_component(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        payload = canonical.to_dict()
        payload["placement_plan"]["sub_operator_targets"][0][
            "component_id"
        ] = "missing-component"

        with self.assertRaises(CanonicalValidationError) as caught:
            CanonicalScenario.from_dict(payload)
        self.assertIn(
            "unknown_target_component",
            {issue.code for issue in caught.exception.issues},
        )

    def test_round_trip_rejects_dangling_sub_operator_parent(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        payload = canonical.to_dict()
        payload["model"]["sub_operators"][0][
            "parent_operator_id"
        ] = "missing-authoring-parent"

        with self.assertRaisesRegex(ValueError, "unknown source parent"):
            CanonicalScenario.from_dict(payload)

    def test_round_trip_rejects_duplicate_sub_operator_rank_target(self):
        canonical = compile_canonical_scenario(build_reference_scenario())
        payload = canonical.to_dict()
        payload["placement_plan"]["sub_operator_targets"].append(
            dict(payload["placement_plan"]["sub_operator_targets"][0])
        )

        with self.assertRaisesRegex(
            ValueError,
            "sub-operator targets must be unique",
        ):
            CanonicalScenario.from_dict(payload)

    def test_parallel_materialization_failure_has_compilation_phase(self):
        scenario = build_reference_scenario()
        parallel = ParallelSpec(tp_degree=2, pp_degree=1, ep_degree=1)
        scenario = replace(
            scenario,
            placement=replace(scenario.placement, parallel=parallel),
        )

        with self.assertRaises(CanonicalizationError) as caught:
            compile_canonical_scenario(scenario)
        self.assertEqual(caught.exception.phase, CompilationPhase.PLAN_PARALLELISM)

    def test_default_canonical_export_preserves_large_synthetic_template(self):
        scenario = build_reference_scenario()
        synthetic = replace(
            scenario,
            workload=replace(
                scenario.workload,
                requests=(),
                request_count=1_000_000_000,
                prompt_tokens=8,
                output_tokens=2,
                arrival_rate_rps=4.0,
            ),
        )

        canonical = compile_canonical_scenario(synthetic)
        template = canonical.workload.attributes["synthetic_template"]

        self.assertEqual(canonical.workload.requests, ())
        self.assertEqual(template["request_count"], 1_000_000_000)
        self.assertEqual(template["prompt_tokens"], 8)
        self.assertEqual(template["output_tokens"], 2)
        self.assertEqual(template["arrival_rate_rps"], 4.0)
        self.assertFalse(template["expanded"])
        self.assertEqual(canonical.to_json(indent=None), canonical.to_json(indent=None))

    def test_canonical_export_can_still_expand_small_synthetic_workload(self):
        scenario = build_reference_scenario()
        synthetic = replace(
            scenario,
            workload=replace(
                scenario.workload,
                requests=(),
                request_count=3,
                prompt_tokens=8,
                output_tokens=2,
                arrival_rate_rps=2.0,
            ),
        )

        canonical = compile_canonical_scenario(
            synthetic,
            options=CompilerOptions(expand_synthetic_requests=True),
        )

        self.assertEqual(
            [request.request_id for request in canonical.workload.requests],
            ["request-0000", "request-0001", "request-0002"],
        )
        self.assertEqual(canonical.workload.requests[2].arrival_ns, 1_000_000_000.0)
        self.assertTrue(
            canonical.workload.attributes["synthetic_template"]["expanded"]
        )


if __name__ == "__main__":
    unittest.main()
