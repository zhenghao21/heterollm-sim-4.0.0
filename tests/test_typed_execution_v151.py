import unittest
from dataclasses import replace

from heterollm_sim.control_plane_planner import plan_runtime_placement
from heterollm_sim.config import model_from_dict
from heterollm_sim.compiler_ir import compile_canonical_scenario
from heterollm_sim.contracts import TaskCategory
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.ir import (
    ComponentSpec,
    LinkSpec,
    MTPPolicy,
    ParallelSpec,
    PortSpec,
    RankMappingSpec,
    RequestSpec,
    SchedulerSpec,
    WorkloadSpec,
    model_graph_execution_digest,
    model_graph_execution_view,
)
from heterollm_sim.control_plane_state import mapping_fingerprint_status
from heterollm_sim.metrics import summarize_metrics
from heterollm_sim.model_presets import (
    OUT_OF_DOMAIN,
    UnsupportedPresetError,
    list_model_presets,
    materialize_model_payload,
    model_preset_detail,
)
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.planner import compile_scenario, validate_scenario
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive


def _operator_map(graph):
    return {item.operator_id: item for item in graph.operators}


def _tensor_map(graph):
    return {item.tensor_id: item for item in graph.tensors}


def _graph_model_payload(payload):
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "name",
            "vocabulary_size",
            "max_sequence_length",
            "embedding_weight_bytes",
            "text_backbone_only",
            "supported_modalities",
            "excluded_subgraphs",
            "architecture",
            "metadata",
            "graph",
        )
        if key in payload
    }


def _qwen_tp2_mtp_scenario(
    preset_id="qwen3_5-0_8b", *, tp_degree=2, ep_degree=1
):
    if tp_degree * ep_degree != 2:
        raise ValueError("the two-GPU typed MTP fixture requires TP x EP = 2")
    base = build_reference_scenario()
    model = model_from_dict(
        _graph_model_payload(materialize_model_payload(preset_id))
    )
    execution_view = model_graph_execution_view(
        model.graph,
        schema_version=model.schema_version,
    )
    components = base.hardware.component_map()

    nv0 = PortSpec(
        "nv0",
        "NVLink",
        "endpoint",
        version="4.0",
        lanes=8,
        bandwidth_gbps=400.0,
    )
    nv1 = replace(nv0, port_id="nv1")
    hbm_port = next(
        port for port in components["gpu0"].ports if port.protocol == "HBM"
    )
    hbm8_port = replace(hbm_port, port_id="hbm8")

    updated_components = []
    for component in base.hardware.components:
        if component.component_id == "gpu0":
            component = replace(
                component,
                ports=component.ports + (nv0,),
                capacity_bytes=2**40,
            )
        elif component.component_id.startswith("hbm"):
            component = replace(component, capacity_bytes=2**40)
        updated_components.append(component)

    gpu0 = next(
        component
        for component in updated_components
        if component.component_id == "gpu0"
    )
    gpu1 = ComponentSpec(
        "gpu1",
        "gpu",
        cost_profile_id=gpu0.cost_profile_id,
        ports=(nv1, hbm8_port),
        package_id="package0",
        die_id="gpu1_die",
        capacity_bytes=2**40,
        peak_ops_per_s=gpu0.peak_ops_per_s,
    )
    hbm8 = replace(
        components["hbm0"],
        component_id="hbm8",
        capacity_bytes=2**40,
    )
    hbm_template = next(
        link for link in base.hardware.links if link.link_id == "gpu-hbm0"
    )
    hardware = replace(
        base.hardware,
        components=tuple(updated_components) + (gpu1, hbm8),
        links=base.hardware.links
        + (
            LinkSpec(
                "gpu0-gpu1",
                "gpu0",
                "nv0",
                "gpu1",
                "nv1",
                "NVLink",
                version="4.0",
                lanes=8,
                bandwidth_gbps=400.0,
                latency_ns=25.0,
                metadata={"shared_bidirectional": True},
            ),
            replace(
                hbm_template,
                link_id="gpu1-hbm8",
                source_component="gpu1",
                source_port="hbm8",
                target_component="hbm8",
            ),
        ),
    )
    rank_mapping = (
        RankMappingSpec(0, "gpu0", 0, 0, 0, "hbm0", None),
        RankMappingSpec(
            1,
            "gpu1",
            1 if tp_degree == 2 else 0,
            0,
            0 if tp_degree == 2 else 1,
            "hbm8",
            None,
        ),
    )
    parallel = ParallelSpec(
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        rank_mapping=rank_mapping,
        layer_to_stage={
            descriptor.layer_id: 0
            for descriptor in execution_view.layer_instances
        },
    )
    workload = WorkloadSpec(
        "typed-mtp-tiny-static",
        # Prefill commits one main token.  Three remaining outputs exercise a
        # full two-draft proposer round under the max-draft semantics.
        requests=(RequestSpec("request-0000", 0.0, 2, 4),),
        random_seed=base.workload.random_seed,
        mtp=MTPPolicy(candidate_tokens=2, acceptance_rate=1.0),
        scheduler=SchedulerSpec(mode="static", max_num_seqs=1),
    )
    placement = replace(
        base.placement,
        model_name=model.name,
        hardware_name=hardware.name,
        parallel=parallel,
        op_to_component={},
        tensor_to_component={},
        tensor_bytes={},
    )
    return replace(
        base,
        name="typed-mtp-tp{}-ep{}".format(tp_degree, ep_degree),
        model=model,
        hardware=hardware,
        placement=placement,
        workload=workload,
        cim_interconnect=None,
    )


class TypedExecutionPresetAcceptanceTests(unittest.TestCase):
    def test_all_executable_presets_have_typed_execution_view_and_ood_fail_closed(self):
        presets = list_model_presets()
        executable = [item for item in presets if item["generation_allowed"]]
        display_only = [item for item in presets if not item["generation_allowed"]]

        self.assertEqual(len(executable), 139)
        self.assertEqual(len(display_only), 8)

        for item in executable:
            with self.subTest(preset_id=item["id"]):
                payload = _graph_model_payload(
                    materialize_model_payload(item["id"])
                )
                model = model_from_dict(payload)
                view = model_graph_execution_view(
                    model.graph,
                    schema_version=model.schema_version,
                )
                operators = _operator_map(model.graph)
                tensors = _tensor_map(model.graph)
                op_kinds = {operator.op_kind for operator in view.operators}
                dense_layers = [
                    descriptor
                    for descriptor in view.layer_instances
                    if not descriptor.layer.is_moe
                ]
                moe_layers = [
                    descriptor
                    for descriptor in view.layer_instances
                    if descriptor.layer.is_moe
                ]
                shared_layers = [
                    descriptor
                    for descriptor in moe_layers
                    if descriptor.layer.has_shared_expert
                ]

                self.assertTrue(model.graph.executable)
                self.assertEqual(len(view.layer_instances), item["layer_count"])
                self.assertEqual(
                    tuple(descriptor.layer.layer_id for descriptor in view.layer_instances),
                    tuple(descriptor.layer_id for descriptor in view.layer_instances),
                )
                for operator in view.operators:
                    for tensor_id in (
                        operator.input_tensor_ids
                        + operator.output_tensor_ids
                        + operator.weight_tensor_ids
                    ):
                        self.assertIn(tensor_id, tensors)
                for tensor in view.tensors:
                    if tensor.producer_operator_id is not None:
                        self.assertIn(tensor.producer_operator_id, operators)
                    for consumer in tensor.consumer_operator_ids:
                        self.assertIn(consumer, operators)

                if dense_layers:
                    self.assertIn("dense_mlp", op_kinds)
                if moe_layers:
                    self.assertIn("moe_router", op_kinds)
                    self.assertIn("moe_experts", op_kinds)
                if shared_layers:
                    self.assertIn("shared_expert", op_kinds)

                prediction_descriptors = tuple(
                    descriptor
                    for descriptor in view.mtp_descriptors
                    if descriptor.prediction_index is not None
                )
                aux_descriptors = tuple(
                    descriptor
                    for descriptor in view.mtp_descriptors
                    if descriptor.prediction_index is None
                )
                if view.mtp_descriptors:
                    for prediction_index, descriptor in enumerate(
                        prediction_descriptors
                    ):
                        prefix = "mtp.prediction_layer.{:03d}".format(
                            prediction_index
                        )
                        self.assertEqual(descriptor.operator.operator_id, prefix)
                        self.assertEqual(descriptor.output_tensor.tensor_id, prefix + ".output")
                        self.assertEqual(descriptor.weight_tensor.tensor_id, prefix + ".weights")
                        self.assertGreater(descriptor.weight_bytes, 0)
                    self.assertLessEqual(len(aux_descriptors), 1)
                    if aux_descriptors:
                        descriptor = aux_descriptors[0]
                        self.assertEqual(descriptor.operator.operator_id, "mtp.aux_head")
                        self.assertEqual(
                            descriptor.output_tensor.tensor_id,
                            "mtp.proposal_logits",
                        )
                        self.assertEqual(
                            descriptor.weight_tensor.tensor_id,
                            "mtp.aux_head.weights",
                        )
                        self.assertGreater(descriptor.weight_bytes, 0)
                else:
                    self.assertFalse(
                        any(
                            operator.op_kind.startswith("mtp_")
                            for operator in view.operators
                        )
                    )

        for item in display_only:
            with self.subTest(preset_id=item["id"], phase="ood"):
                detail = model_preset_detail(item["id"])
                self.assertEqual(item["support_level"], OUT_OF_DOMAIN)
                self.assertIsNone(detail["model"])
                self.assertFalse(detail["graph"]["executable"])
                self.assertTrue(
                    any(
                        operator["op_kind"] == "unsupported_component_group"
                        for operator in detail["graph"]["operators"]
                    )
                )
                with self.assertRaises(UnsupportedPresetError):
                    materialize_model_payload(item["id"])
                with self.assertRaisesRegex(ValueError, "不可执行|不能投影"):
                    model_from_dict(
                        {
                            "schema_version": "4.0.0",
                            "name": item["name"],
                            "graph": detail["graph"],
                        }
                    )

    def test_qwen38_main_logits_and_mtp_proposal_are_distinct_weighted_branches(self):
        payload = _graph_model_payload(
            materialize_model_payload("qwen3_8-27b")
        )
        model = model_from_dict(payload)
        view = model_graph_execution_view(model.graph)
        operators = _operator_map(model.graph)
        tensors = _tensor_map(model.graph)

        self.assertEqual(operators["lm_head"].output_tensor_ids, ("logits",))
        self.assertEqual(operators["output"].input_tensor_ids, ("logits",))
        self.assertEqual(tensors["logits"].producer_operator_id, "lm_head")
        self.assertEqual(tensors["logits"].consumer_operator_ids, ("output",))

        branch_source = operators["final_norm"].input_tensor_ids[0]
        self.assertEqual(
            tensors[branch_source].consumer_operator_ids,
            ("final_norm", "mtp.prediction_layer.000"),
        )
        self.assertEqual(
            operators["mtp.prediction_layer.000"].input_tensor_ids,
            (branch_source,),
        )
        self.assertEqual(
            operators["mtp.aux_head"].output_tensor_ids,
            ("mtp.proposal_logits",),
        )
        self.assertEqual(
            tensors["mtp.proposal_logits"].producer_operator_id,
            "mtp.aux_head",
        )
        self.assertEqual(tensors["mtp.proposal_logits"].consumer_operator_ids, ())

        mtp_weight_ids = {
            "mtp.prediction_layer.000.weights",
            "mtp.aux_head.weights",
        }
        self.assertEqual(
            {descriptor.weight_tensor.tensor_id for descriptor in view.mtp_descriptors},
            mtp_weight_ids,
        )
        mtp_weight_bytes = sum(
            descriptor.weight_bytes for descriptor in view.mtp_descriptors
        )
        self.assertTrue(
            all(
                tensors[tensor_id].logical_bytes
                and tensors[tensor_id].logical_bytes > 0
                for tensor_id in mtp_weight_ids
            )
        )
        self.assertEqual(
            mtp_weight_bytes,
            sum(
                tensors[descriptor.weight_tensor.tensor_id].logical_bytes
                for descriptor in view.mtp_descriptors
            ),
        )

    def test_typed_mtp_graph_only_round_trip_preserves_execution_descriptors(self):
        model = model_from_dict(
            _graph_model_payload(materialize_model_payload("qwen3_8-27b"))
        )
        view = model_graph_execution_view(
            model.graph,
            schema_version=model.schema_version,
        )
        expected = tuple(
            (
                descriptor.operator.operator_id,
                descriptor.weight_tensor.tensor_id,
                descriptor.output_tensor.tensor_id,
                descriptor.prediction_index,
                descriptor.weight_bytes,
            )
            for descriptor in view.mtp_descriptors
        )

        restored = model_from_dict(
            _graph_model_payload(to_primitive(model))
        )
        restored_view = model_graph_execution_view(
            restored.graph,
            schema_version=restored.schema_version,
        )

        self.assertTrue(expected)
        self.assertEqual(
            model_graph_execution_digest(restored.graph),
            model_graph_execution_digest(model.graph),
        )
        self.assertEqual(
            tuple(
                (
                    descriptor.operator.operator_id,
                    descriptor.weight_tensor.tensor_id,
                    descriptor.output_tensor.tensor_id,
                    descriptor.prediction_index,
                    descriptor.weight_bytes,
                )
                for descriptor in restored_view.mtp_descriptors
            ),
            expected,
        )


class TypedMTPParallelPlanningAcceptanceTests(unittest.TestCase):
    def test_graph_projection_layer_ids_feed_parallel_plan(self):
        scenario = _qwen_tp2_mtp_scenario()
        view = model_graph_execution_view(
            scenario.model.graph,
            schema_version=scenario.model.schema_version,
        )
        plan = build_parallel_plan(scenario)

        self.assertEqual(
            plan.layer_to_stage,
            {descriptor.layer_id: 0 for descriptor in view.layer_instances},
        )

    def test_tp1_ep2_lowers_typed_mtp_on_both_ep_replicas_without_warning(self):
        scenario = _qwen_tp2_mtp_scenario(tp_degree=1, ep_degree=2)
        mapping = plan_runtime_placement(scenario)
        self.assertTrue(mapping.fully_placed, mapping.unplaced)
        mapped = mapping.apply(scenario)
        report = validate_scenario(mapped)
        self.assertTrue(report.is_valid, report.errors)

        view = model_graph_execution_view(
            mapped.model.graph,
            schema_version=mapped.model.schema_version,
        )
        descriptors = {
            descriptor.operator.operator_id: descriptor
            for descriptor in view.mtp_descriptors
        }
        self.assertTrue(descriptors)
        self.assertFalse(
            any(
                operator_id in warning
                and "not consumed" in warning
                for operator_id in descriptors
                for warning in report.warnings_en
            ),
            report.warnings_en,
        )

        decision_metadata = mapped.placement.metadata["control_plane"]["decision"]
        for operator_id, descriptor in descriptors.items():
            with self.subTest(operator_id=operator_id, phase="control-plane"):
                targets = decision_metadata["operator_execution_targets"][operator_id]
                self.assertEqual(
                    {
                        (
                            target["rank_id"],
                            target["tp_rank"],
                            target["ep_rank"],
                        )
                        for target in targets
                    },
                    {(0, 0, 0), (1, 0, 1)},
                )
                shards = decision_metadata["rank_weight_shards"][
                    descriptor.weight_tensor.tensor_id
                ]
                self.assertEqual({shard["rank_id"] for shard in shards}, {0, 1})
                self.assertEqual({shard["shard_count"] for shard in shards}, {1})
                self.assertEqual({shard["shard_index"] for shard in shards}, {0})
                self.assertEqual(
                    {shard["logical_bytes"] for shard in shards},
                    {descriptor.weight_bytes},
                )

        canonical = compile_canonical_scenario(mapped)
        canonical_sub_operators = {
            item.sub_operator_id: item
            for item in canonical.sub_operators
        }
        canonical_targets = {
            item.sub_operator_id: set()
            for item in canonical.sub_operators
        }
        for target in canonical.placement_plan.sub_operator_targets:
            canonical_targets[target.sub_operator_id].add(target.rank_id)
        for operator_id in descriptors:
            with self.subTest(operator_id=operator_id, phase="canonical-ir"):
                sub_operator = canonical_sub_operators[operator_id]
                self.assertEqual(
                    sub_operator.parent_operator_id,
                    operator_id,
                )
                self.assertEqual(canonical_targets[operator_id], {0, 1})

        schedule = compile_scenario(mapped)
        tasks = tuple(schedule.tasks)
        tasks_by_id = {task.task_id: task for task in tasks}
        for operator_id, descriptor in descriptors.items():
            with self.subTest(operator_id=operator_id, phase="planner"):
                rank_gemms = [
                    task
                    for task in tasks
                    if task.metadata.get("model_operator_id") == operator_id
                    and task.metadata.get("weight_tensor_id")
                    == descriptor.weight_tensor.tensor_id
                    and task.metadata.get("phase") == "gpu_gemm"
                ]
                self.assertEqual(
                    {task.metadata["rank"] for task in rank_gemms}, {0, 1}
                )
                self.assertEqual(
                    {task.metadata["mtp_ep_replica"] for task in rank_gemms},
                    {0, 1},
                )
                self.assertEqual(
                    {task.metadata["rank_weight_bytes"] for task in rank_gemms},
                    {descriptor.weight_bytes},
                )

                replica_collectives = [
                    task
                    for task in tasks
                    if task.metadata.get("model_operator_id") == operator_id
                    and task.metadata.get("weight_tensor_id")
                    == descriptor.weight_tensor.tensor_id
                    and task.metadata.get("event_kind", "").endswith(
                        "_collective"
                    )
                    and task.metadata.get("algorithm") == "local"
                ]
                self.assertEqual(
                    {task.metadata["mtp_ep_replica"] for task in replica_collectives},
                    {0, 1},
                )
                for collective in replica_collectives:
                    dependency_ep_ranks = {
                        tasks_by_id[dependency].metadata.get("ep_rank")
                        for dependency in collective.dependencies
                    }
                    self.assertEqual(
                        dependency_ep_ranks,
                        {collective.metadata["mtp_ep_replica"]},
                    )

    def test_tp2_control_plane_planning_runtime_and_workload_staleness_contract(self):
        scenario = _qwen_tp2_mtp_scenario()
        mapping = plan_runtime_placement(scenario)
        self.assertTrue(mapping.fully_placed, mapping.unplaced)
        mapped = mapping.apply(scenario)
        status = mapping_fingerprint_status(mapped)
        self.assertFalse(status["mapping_stale"])
        self.assertEqual(status["input_fingerprint"], status["current_input_fingerprint"])

        view = model_graph_execution_view(
            mapped.model.graph,
            schema_version=mapped.model.schema_version,
        )
        descriptors = {
            descriptor.operator.operator_id: descriptor
            for descriptor in view.mtp_descriptors
        }
        metadata = mapped.placement.metadata["control_plane"]["decision"]

        for operator_id, descriptor in descriptors.items():
            with self.subTest(operator_id=operator_id, phase="control-plane"):
                tensor_id = descriptor.weight_tensor.tensor_id
                targets = metadata["operator_execution_targets"][operator_id]
                self.assertEqual(
                    {(target["rank_id"], target["compute_component_id"]) for target in targets},
                    {(0, "gpu0"), (1, "gpu1")},
                )
                shards = metadata["rank_weight_shards"][tensor_id]
                self.assertEqual({shard["rank_id"] for shard in shards}, {0, 1})
                self.assertEqual({shard["shard_count"] for shard in shards}, {2})
                self.assertEqual({shard["shard_index"] for shard in shards}, {0, 1})
                self.assertEqual(
                    {shard["storage_component_id"] for shard in shards},
                    {"hbm0", "hbm8"},
                )
                self.assertEqual(
                    sum(shard["logical_bytes"] for shard in shards),
                    descriptor.weight_bytes,
                )
                self.assertTrue(all(shard["physical_bytes"] > 0 for shard in shards))
                details = metadata["weight_tensor_details"][tensor_id]
                self.assertEqual(details["logical_bytes"], descriptor.weight_bytes)
                self.assertEqual(details["residency"], "rank_sharded_storage")

        schedule = compile_scenario(mapped)
        tasks = tuple(schedule.tasks)
        for operator_id, descriptor in descriptors.items():
            with self.subTest(operator_id=operator_id, phase="planner"):
                tensor_id = descriptor.weight_tensor.tensor_id
                rank_gemms = [
                    task
                    for task in tasks
                    if task.metadata.get("model_operator_id") == operator_id
                    and task.metadata.get("weight_tensor_id") == tensor_id
                    and task.metadata.get("phase") == "gpu_gemm"
                ]
                self.assertEqual(
                    {task.metadata["rank"] for task in rank_gemms},
                    {0, 1},
                )
                collective_transfers = [
                    task
                    for task in tasks
                    if task.metadata.get("model_operator_id") == operator_id
                    and task.metadata.get("weight_tensor_id") == tensor_id
                    and task.metadata.get("event_kind", "").endswith("_collective")
                    and task.metadata.get("collective_name")
                ]
                self.assertTrue(collective_transfers)
                first_round_transfers = [
                    task
                    for task in collective_transfers
                    if ".round00." in task.name
                ]
                self.assertEqual(len(first_round_transfers), 2)
                for collective in first_round_transfers:
                    draft_step = collective.metadata["draft_step"]
                    self.assertEqual(
                        set(collective.dependencies),
                        {
                            task.task_id
                            for task in rank_gemms
                            if task.metadata["draft_step"] == draft_step
                        },
                    )

        aux_weight_reads = [
            task
            for task in tasks
            if task.metadata.get("model_operator_id") == "mtp.aux_head"
            and task.metadata.get("event_kind") == "model_weight_access"
            and task.metadata.get("rank") in {0, 1}
            and task.category == TaskCategory.COMMUNICATION
            and "model_weight_read.access" in task.task_id
        ]
        self.assertEqual({task.metadata["rank"] for task in aux_weight_reads}, {0, 1})
        self.assertTrue(
            all(
                task.metadata.get("weight_lifecycle_mode") == "preloaded_resident"
                and task.metadata.get("weight_source_is_offload_backing") is False
                and task.metadata.get("weight_backing_read_emitted") is False
                and task.metadata.get("weight_backing_read_gate")
                == "resident_active_memory_route_emitted"
                for task in aux_weight_reads
            )
        )
        self.assertTrue(
            all(
                any(
                    ".mtp.prediction_layer.000.all_reduce.complete" in dependency
                    and ".draft{:03d}.".format(
                        task.metadata["draft_step"]
                    ) in dependency
                    for dependency in task.dependencies
                )
                for task in aux_weight_reads
            )
        )

        trace = simulate_schedule(schedule)
        metrics = summarize_metrics(trace)
        communication_events = [
            task
            for task in trace.tasks
            if task.category in {TaskCategory.COMMUNICATION, TaskCategory.COLLECTIVE}
        ]
        self.assertGreater(metrics.makespan_ns, 0.0)
        self.assertGreater(metrics.category_time_ns[TaskCategory.COMMUNICATION], 0.0)
        self.assertGreater(metrics.category_time_ns[TaskCategory.COLLECTIVE], 0.0)
        self.assertTrue(communication_events)

        changed_workload = replace(
            mapped.workload,
            name="typed-mtp-workload-only-change",
            requests=(RequestSpec("request-0000", 0.0, 3, 1),),
            mtp=replace(mapped.workload.mtp, acceptance_rate=0.25),
        )
        changed = replace(mapped, workload=changed_workload)
        changed_status = mapping_fingerprint_status(changed)
        self.assertFalse(changed_status["mapping_stale"])
        self.assertEqual(
            changed_status["input_fingerprint"],
            changed_status["current_input_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
