import unittest
from dataclasses import replace
from unittest.mock import patch

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.contracts import OperatorClass, TaskCategory
from heterollm_sim.parallel import build_parallel_plan, shard_extent
from heterollm_sim.planner import (
    _TaskBuilder,
    _add_collective_tasks,
    _add_transfer_tasks,
    compile_scenario,
    validate_scenario,
)
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.ir import (
    ComponentSpec,
    HardwareSpec,
    LinkSpec,
    ParallelSpec,
    PortSpec,
    RankMappingSpec,
)
from tests.model_helpers import execution_layers, model_from_layer_specs


def _nvlink_port(port_id="nv0"):
    return PortSpec(
        port_id=port_id,
        protocol="NVLink",
        role="endpoint",
        version="4.0",
        lanes=8,
        bandwidth_gbps=400.0,
    )


def _two_gpu_nvlink_scenario(parallel):
    scenario = build_reference_scenario()
    components = scenario.hardware.component_map()
    gpu0 = components["gpu0"]
    gpu0_with_nvlink = replace(gpu0, ports=gpu0.ports + (_nvlink_port(),))
    gpu1 = ComponentSpec(
        component_id="gpu1",
        kind="gpu",
        ports=(_nvlink_port(),),
        package_id="package0",
        die_id="gpu1_die",
        capacity_bytes=gpu0.capacity_bytes,
        peak_ops_per_s=gpu0.peak_ops_per_s,
        cost_profile_id=gpu0.cost_profile_id,
    )
    hardware = HardwareSpec(
        name=scenario.hardware.name,
        components=tuple(
            gpu0_with_nvlink if component.component_id == "gpu0" else component
            for component in scenario.hardware.components
        )
        + (gpu1,),
        links=scenario.hardware.links
        + (
            LinkSpec(
                link_id="gpu0-gpu1",
                source_component="gpu0",
                source_port="nv0",
                target_component="gpu1",
                target_port="nv0",
                protocol="NVLink",
                version="4.0",
                lanes=8,
                bandwidth_gbps=400.0,
                latency_ns=25.0,
                metadata={"shared_bidirectional": True},
            ),
        ),
    )
    placement = replace(scenario.placement, parallel=parallel)
    return replace(scenario, hardware=hardware, placement=placement)


def _lower_collective(
    scenario,
    kind,
    *,
    tensor_bytes=64,
    tensor_elements=32,
    element_bits=16,
):
    plan = build_parallel_plan(scenario)
    builder = _TaskBuilder(scenario.workload.requests[0])
    _add_collective_tasks(
        builder,
        scenario,
        TopologyRouter(scenario.hardware),
        plan,
        "test_collective",
        kind,
        plan.tp_group(0),
        tensor_bytes,
        (),
        tensor_elements=tensor_elements,
        element_bits=element_bits,
    )
    return tuple(builder.tasks)


class ParallelPlanTests(unittest.TestCase):
    def test_legacy_routing_policy_aliases_are_rejected(self):
        for policy in ("shortest", "minimum_time"):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError, "routing_policy must be lowest_latency"
            ):
                ParallelSpec(routing_policy=policy)

    def test_rank_mapping_requires_typed_specs(self):
        with self.assertRaisesRegex(
            ValueError, "rank_mapping must contain RankMappingSpec values"
        ):
            ParallelSpec(rank_mapping=({"component_id": "gpu0"},))

    def test_shard_extent_records_padding(self):
        shard = shard_extent(10, 3, 1)
        self.assertEqual(shard.local_size, 4)
        self.assertEqual(shard.padded_size, 12)
        self.assertEqual(shard.padding, 2)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            shard_extent(10, 3, 1, allow_padding=False)

    def test_degree_one_plan_preserves_reference_gpu(self):
        scenario = build_reference_scenario()
        plan = build_parallel_plan(scenario)
        self.assertEqual(plan.world_size, 1)
        self.assertEqual(plan.ranks[0].component_id, "gpu0")
        self.assertEqual(set(plan.layer_to_stage.values()), {0})

    def test_component_writable_metadata_is_authoritative(self):
        hbm = ComponentSpec(
            component_id="hbm-test",
            kind="hbm",
            write_bandwidth_gbps=0.0,
        )

        self.assertTrue(hbm.is_writable)
        self.assertFalse(
            replace(hbm, metadata={"read_only": True}).is_writable
        )
        self.assertFalse(
            replace(hbm, metadata={"writable": False}).is_writable
        )

    def test_rank_mapping_rejects_read_only_active_memory(self):
        scenario = build_reference_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(
                    component,
                    metadata={**component.metadata, "read_only": True},
                )
                if component.component_id == "hbm0"
                else component
                for component in scenario.hardware.components
            ),
        )

        with self.assertRaisesRegex(
            ValueError, "writable active memory.*read-only hbm"
        ):
            build_parallel_plan(replace(scenario, hardware=hardware))

    def test_resident_aggregate_weights_reject_read_only_active_memory(self):
        scenario = build_reference_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(
                    component,
                    metadata={**component.metadata, "read_only": True},
                )
                if component.component_id == "hbm2"
                else component
                for component in scenario.hardware.components
            ),
        )
        tensor_mapping = dict(scenario.placement.tensor_to_component)
        tensor_mapping["model_weights"] = "hbm2"
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes["model_weights"] = (
            scenario.model.total_declared_weight_bytes
        )
        placement = replace(
            scenario.placement,
            tensor_to_component=tensor_mapping,
            tensor_bytes=tensor_bytes,
        )

        report = validate_scenario(
            replace(scenario, hardware=hardware, placement=placement)
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any(
                "resident aggregate model_weights component hbm2 must be "
                "writable active memory"
                in error
                for error in report.errors_en
            ),
            report.errors_en,
        )

    def test_control_plane_requires_enough_compute_components(self):
        scenario = build_reference_scenario()
        placement = replace(scenario.placement, parallel=ParallelSpec(tp_degree=2))
        with self.assertRaisesRegex(ValueError, "parallel world requires 2 compute components"):
            build_parallel_plan(replace(scenario, placement=placement))

    def test_tp2_on_two_nvlink_gpus_reports_collective_link_utilization(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))
        validation = validate_scenario(scenario)
        self.assertTrue(validation.is_valid, validation.errors)
        plan = build_parallel_plan(scenario)
        self.assertEqual(tuple(rank.component_id for rank in plan.ranks), ("gpu0", "gpu1"))

        payload = report_dict(run_scenario(scenario))

        self.assertEqual(payload["summary"]["parallel"]["tp_degree"], 2)
        self.assertEqual(payload["summary"]["parallel"]["world_size"], 2)
        self.assertIn("link.gpu0-gpu1", payload["resource_utilization"])
        self.assertGreater(payload["resource_utilization"]["link.gpu0-gpu1"], 0.0)
        self.assertGreater(payload["category_time_ns"]["communication"], 0.0)

    def test_collective_reduction_uses_gpu_hbm_launch_energy_and_typed_shape(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))

        tasks = _lower_collective(scenario, "all_reduce")
        reduction_tasks = [
            task
            for task in tasks
            if task.metadata.get("collective_phase") == "local_reduction"
        ]

        self.assertEqual(
            {task.metadata.get("phase") for task in reduction_tasks},
            {"kernel_launch", "gpu_reduction"},
        )
        self.assertEqual(
            {task.metadata.get("rank") for task in reduction_tasks},
            {0, 1},
        )
        self.assertEqual(
            {task.category for task in reduction_tasks},
            {TaskCategory.COLLECTIVE},
        )
        for rank in (0, 1):
            rank_tasks = [
                task
                for task in reduction_tasks
                if task.metadata.get("rank") == rank
            ]
            reduction = next(
                task
                for task in rank_tasks
                if task.metadata.get("phase") == "gpu_reduction"
            )
            resources = {
                demand.resource_id for demand in reduction.demands
            }
            component_id = "gpu{}".format(rank)
            self.assertIn(component_id + ".scalar", resources)
            self.assertIn(component_id + ".hbm_fabric", resources)
            self.assertEqual(
                reduction.metadata["operator_class"],
                OperatorClass.REDUCTION.value,
            )
            self.assertEqual(
                reduction.metadata["cost_model"]["operations"], 16
            )
            self.assertEqual(
                reduction.metadata["cost_model"]["read_bytes"], 64
            )
            self.assertEqual(
                reduction.metadata["cost_model"]["write_bytes"], 32
            )
            self.assertEqual(
                reduction.metadata["rank_reduction_output_elements"], 16
            )
            self.assertEqual(reduction.metadata["rank_result_elements"], 32)
            self.assertGreater(
                sum(
                    demand.energy_pj
                    for task in rank_tasks
                    for demand in task.demands
                ),
                0.0,
            )

    def test_reduce_scatter_reduction_result_is_one_rank_shard(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))

        tasks = _lower_collective(scenario, "reduce_scatter")
        reductions = [
            task
            for task in tasks
            if task.metadata.get("collective_phase") == "local_reduction"
            and task.metadata.get("phase") == "gpu_reduction"
        ]

        self.assertEqual(len(reductions), 2)
        self.assertEqual(
            {task.metadata["rank_result_elements"] for task in reductions},
            {16},
        )
        self.assertEqual(
            sum(task.metadata["cost_model"]["operations"] for task in reductions),
            32,
        )

    def test_colocated_tp_collective_skips_links_but_keeps_local_reduction(self):
        base = build_reference_scenario()
        parallel = ParallelSpec(
            tp_degree=2,
            rank_mapping=(
                RankMappingSpec(0, "gpu0", 0, 0, 0),
                RankMappingSpec(1, "gpu0", 1, 0, 0),
            ),
        )
        scenario = replace(
            base,
            placement=replace(base.placement, parallel=parallel),
        )

        tasks = _lower_collective(scenario, "all_reduce")

        self.assertFalse(
            any(
                task.metadata.get("event_kind") == "collective_transfer"
                for task in tasks
            )
        )
        self.assertEqual(
            {
                task.metadata.get("rank")
                for task in tasks
                if task.metadata.get("collective_phase") == "local_reduction"
            },
            {0, 1},
        )

    def test_collective_and_transfer_byte_counts_fail_closed(self):
        base = build_reference_scenario()
        parallel = ParallelSpec(
            tp_degree=2,
            rank_mapping=(
                RankMappingSpec(0, "gpu0", 0, 0, 0),
                RankMappingSpec(1, "gpu0", 1, 0, 0),
            ),
        )
        scenario = replace(
            base,
            placement=replace(base.placement, parallel=parallel),
        )
        plan = build_parallel_plan(scenario)
        router = TopologyRouter(scenario.hardware)

        for invalid in (-1, 1.5, True):
            with self.subTest(kind="collective", value=invalid):
                builder = _TaskBuilder(scenario.workload.requests[0])
                with self.assertRaisesRegex(
                    ValueError, "tensor_bytes must be a non-negative integer"
                ):
                    _add_collective_tasks(
                        builder,
                        scenario,
                        router,
                        plan,
                        "invalid_collective",
                        "all_reduce",
                        plan.tp_group(0),
                        invalid,
                        (),
                    )
            with self.subTest(kind="transfer", value=invalid):
                builder = _TaskBuilder(scenario.workload.requests[0])
                with self.assertRaisesRegex(
                    ValueError, "byte_count must be a non-negative integer"
                ):
                    _add_transfer_tasks(
                        builder,
                        router,
                        "gpu0",
                        "gpu0",
                        invalid,
                        (),
                        name="invalid_transfer",
                        routing_policy="lowest_latency",
                    )

    def test_collective_reduction_cost_tracks_reduction_and_hbm_profiles(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))

        def cost_metadata(current):
            task = next(
                task
                for task in _lower_collective(current, "all_reduce")
                if task.metadata.get("collective_phase")
                == "local_reduction"
                and task.metadata.get("phase") == "gpu_reduction"
            )
            return task.metadata["cost_model"]

        baseline = cost_metadata(scenario)
        gpu_profile_id = scenario.hardware.get_component("gpu0").cost_profile_id
        self.assertIsNotNone(gpu_profile_id)
        slow_reduction_profiles = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        slow_reduction_profiles["gpu"][gpu_profile_id] = replace(
            scenario.resolve_component_profile("gpu0"),
            reduction_ops_per_cycle_per_sm=0.001,
        )
        slow_reduction = cost_metadata(
            replace(
                scenario,
                component_profiles=slow_reduction_profiles,
            )
        )
        hbm_profile_id = scenario.hardware.get_component("hbm0").cost_profile_id
        self.assertIsNotNone(hbm_profile_id)
        slow_hbm_profiles = {
            kind: dict(registry)
            for kind, registry in scenario.component_profiles.items()
        }
        slow_hbm_profiles["hbm"][hbm_profile_id] = replace(
            scenario.resolve_component_profile("hbm0"),
            bandwidth_gb_s=0.1,
        )
        slow_hbm = cost_metadata(
            replace(
                scenario,
                component_profiles=slow_hbm_profiles,
            )
        )

        self.assertGreater(
            slow_reduction["compute_service_ns"],
            baseline["compute_service_ns"],
        )
        self.assertGreater(
            slow_hbm["memory_service_ns"],
            baseline["memory_service_ns"],
        )

    def test_tp2_lowers_every_typed_mtp_op_on_both_ranks(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))

        schedule = compile_scenario(scenario)

        expected = {
            "mtp.prediction_layer.000": "mtp.prediction_layer.000.weights",
            "mtp.aux_head": "mtp.aux_head.weights",
        }
        for operator_id, tensor_id in expected.items():
            with self.subTest(operator_id=operator_id):
                tasks = [
                    task
                    for task in schedule.tasks
                    if task.metadata.get("model_operator_id") == operator_id
                    and task.metadata.get("phase") is not None
                ]
                self.assertEqual(
                    {task.metadata.get("rank") for task in tasks}, {0, 1}
                )
                self.assertTrue(tasks)
                self.assertTrue(
                    all(
                        task.metadata.get("weight_tensor_id") == tensor_id
                        for task in tasks
                    )
                )
                collective_tasks = [
                    task
                    for task in schedule.tasks
                    if task.metadata.get("model_operator_id") == operator_id
                    and task.metadata.get("collective_kind")
                ]
                self.assertTrue(collective_tasks)
                first = collective_tasks[0]
                self.assertTrue(
                    any("rank000" in dependency for dependency in first.dependencies)
                )
                self.assertTrue(
                    any("rank001" in dependency for dependency in first.dependencies)
                )

    def test_pp2_emits_pipeline_activation_transfer(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(pp_degree=2))
        validation = validate_scenario(scenario)
        self.assertTrue(validation.is_valid, validation.errors)

        schedule = compile_scenario(scenario)

        self.assertTrue(
            any(
                task.metadata.get("event_kind") == "pipeline_activation"
                for task in schedule.tasks
            )
        )
        self.assertTrue(
            any(
                demand.resource_id == "link.gpu0-gpu1"
                for task in schedule.tasks
                for demand in task.demands
            )
        )

    def test_explicit_pp_mapping_rejects_reversed_decoder_order(self):
        scenario = _two_gpu_nvlink_scenario(
            ParallelSpec(
                pp_degree=2,
                layer_to_stage={"dense0": 1, "moe1": 0},
            )
        )

        with self.assertRaisesRegex(ValueError, "preserve decoder layer order"):
            build_parallel_plan(scenario)

    def test_explicit_pp_mapping_rejects_noncontiguous_stage_segments(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(pp_degree=2))
        layers = execution_layers(scenario.model)
        third = replace(layers[0], layer_id="dense2")
        model = model_from_layer_specs(
            scenario.model.name,
            layers + (third,),
            vocabulary_size=scenario.model.vocabulary_size,
            max_sequence_length=scenario.model.max_sequence_length,
            embedding_weight_bytes=scenario.model.embedding_weight_bytes,
            architecture=scenario.model.architecture,
            metadata=scenario.model.metadata,
        )
        placement = replace(
            scenario.placement,
            parallel=ParallelSpec(
                pp_degree=2,
                layer_to_stage={"dense0": 0, "moe1": 1, "dense2": 0},
            ),
        )

        with self.assertRaisesRegex(ValueError, "contiguous PP stage segments"):
            build_parallel_plan(replace(scenario, model=model, placement=placement))

    def test_stage_rank_ownership_covers_tp_ep_product_only(self):
        scenario = build_reference_scenario()
        ranks = tuple(
            RankMappingSpec(
                rank=pp_rank * 4 + ep_rank * 2 + tp_rank,
                component_id="gpu0",
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                ep_rank=ep_rank,
                memory_component_id="hbm0",
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
        plan = build_parallel_plan(
            replace(
                scenario,
                placement=replace(scenario.placement, parallel=parallel),
            )
        )

        layers = execution_layers(scenario.model)
        dense_ranks = plan.ranks_for_layer(layers[0])
        moe_ranks = plan.ranks_for_layer(layers[1])
        self.assertEqual({rank.rank for rank in dense_ranks}, {0, 1, 2, 3})
        self.assertEqual({rank.rank for rank in moe_ranks}, {4, 5, 6, 7})
        self.assertEqual(
            {(rank.tp_rank, rank.ep_rank) for rank in dense_ranks},
            {(0, 0), (1, 0), (0, 1), (1, 1)},
        )

    def test_explicit_rank_ids_must_cover_world_range(self):
        with self.assertRaisesRegex(ValueError, "cover \\[0, world_size\\)"):
            ParallelSpec(
                tp_degree=2,
                rank_mapping=(
                    RankMappingSpec(1, "gpu0", 0, 0, 0),
                    RankMappingSpec(2, "gpu1", 1, 0, 0),
                ),
            )

    def test_resident_weight_capacity_uses_physical_shards_not_one_full_copy(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))
        shard_capacity = 20 * 1024 * 1024
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=shard_capacity)
                if component.component_id in {"gpu0", "gpu1", "cim0"}
                else component
                for component in scenario.hardware.components
            ),
        )
        placement = replace(
            scenario.placement,
            tensor_to_component={
                key: value
                for key, value in scenario.placement.tensor_to_component.items()
                if "weight" not in key
            },
            tensor_bytes={
                key: value
                for key, value in scenario.placement.tensor_bytes.items()
                if "weight" not in key
            },
        )

        authoring = replace(scenario, hardware=hardware, placement=placement)
        report = validate_scenario(
            plan_runtime_placement(authoring).apply(authoring)
        )

        self.assertTrue(report.is_valid, report.errors)
        self.assertLess(shard_capacity, scenario.model.total_declared_weight_bytes)
        self.assertTrue(
            any(
                "物理 TP/PP/EP 分片" in information
                for information in report.information
            )
        )

    def test_implicit_rank_memory_uses_reachable_hbm_not_compute_capacity(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=0)
                if component.kind == "gpu"
                else component
                for component in scenario.hardware.components
            ),
        )
        placement = replace(
            scenario.placement,
            tensor_to_component={
                key: value
                for key, value in scenario.placement.tensor_to_component.items()
                if "weight" not in key
            },
            tensor_bytes={
                key: value
                for key, value in scenario.placement.tensor_bytes.items()
                if "weight" not in key
            },
        )

        report = validate_scenario(
            replace(scenario, hardware=hardware, placement=placement)
        )

        self.assertTrue(report.is_valid, report.errors)
        self.assertFalse(
            any("capacity is unknown for gpu" in error for error in report.errors)
        )

    def test_impossible_resident_weight_capacity_is_rejected(self):
        scenario = build_reference_scenario()
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=1 << 20)
                if component.component_id == "hbm0"
                else replace(component, capacity_bytes=1024)
                if component.component_id == "cim0"
                else component
                for component in scenario.hardware.components
            ),
        )
        placement = replace(
            scenario.placement,
            tensor_to_component={
                key: value
                for key, value in scenario.placement.tensor_to_component.items()
                if "weight" not in key
            },
            tensor_bytes={
                key: value
                for key, value in scenario.placement.tensor_bytes.items()
                if "weight" not in key
            },
        )

        report = validate_scenario(
            replace(scenario, hardware=hardware, placement=placement)
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("其余常驻模型权重" in error for error in report.errors)
        )

    def test_aggregate_model_weights_must_cover_declared_model(self):
        scenario = build_reference_scenario()
        mapping = dict(scenario.placement.tensor_to_component)
        mapping["model_weights"] = "hbm0"
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes["model_weights"] = (
            scenario.model.total_declared_weight_bytes - 1
        )
        placement = replace(
            scenario.placement,
            tensor_to_component=mapping,
            tensor_bytes=tensor_bytes,
        )

        report = validate_scenario(replace(scenario, placement=placement))

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("模型声明了" in error for error in report.errors)
        )

    def test_cold_weights_without_aggregate_or_detailed_backing_are_rejected(self):
        scenario = build_reference_scenario()
        placement = replace(
            scenario.placement,
            tensor_to_component={
                key: value
                for key, value in scenario.placement.tensor_to_component.items()
                if "weight" not in key
            },
            tensor_bytes={
                key: value
                for key, value in scenario.placement.tensor_bytes.items()
                if "weight" not in key
            },
        )

        report = validate_scenario(
            replace(scenario, placement=placement, weights_resident=False)
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("冷权重或流式权重需要" in error for error in report.errors)
        )

    def test_detailed_weight_reachability_checks_every_actual_tp_rank(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(tp_degree=2))
        tensor_mapping = dict(scenario.placement.tensor_to_component)
        tensor_mapping["dense0.attention_weights"] = "hbm0"
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes["dense0.attention_weights"] = 1024
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                tensor_to_component=tensor_mapping,
                tensor_bytes=tensor_bytes,
            ),
        )
        original_route = TopologyRouter.route

        def fail_second_rank(router, source, target, byte_count, **kwargs):
            if source == "hbm0" and target == "gpu1":
                raise ValueError("second-rank source route blocked")
            return original_route(
                router, source, target, byte_count, **kwargs
            )

        with patch.object(TopologyRouter, "route", new=fail_second_rank):
            report = validate_scenario(configured)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any(
                "权重张量 dense0.attention_weights 到 gpu1 的路由失败" in error
                for error in report.errors
            )
        )
        self.assertTrue(
            any(
                "second-rank source route blocked" in error
                for error in report.errors_en
            )
        )

    def test_ep2_moe_path_uses_all_to_all_collectives(self):
        scenario = _two_gpu_nvlink_scenario(ParallelSpec(ep_degree=2))
        validation = validate_scenario(scenario)
        self.assertTrue(validation.is_valid, validation.errors)

        schedule = compile_scenario(scenario)
        collective_names = {
            str(task.metadata.get("collective_name"))
            for task in schedule.tasks
            if task.metadata.get("collective_kind") == "all_to_all"
        }

        self.assertTrue(any("expert_dispatch" in name for name in collective_names))
        self.assertTrue(any("expert_combine" in name for name in collective_names))


if __name__ == "__main__":
    unittest.main()
