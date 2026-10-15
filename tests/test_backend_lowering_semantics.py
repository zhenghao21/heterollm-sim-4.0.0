"""Preserve physical allocation and control-flow ownership while lowering."""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hbf_hbm_experiments import scenario
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.control_plane import _MaterializedWorkloadCounts, _runtime_profile
from heterollm_sim.planner import (
    _attach_gddr_physical_task, _summarize_dram_task_traffic, _summarize_nand_task_traffic,
)
from heterollm_sim.runtime import ControlPlaneRuntime
from heterollm_sim.runtime_lowering import build_initial_control_tasks
from heterollm_sim.runtime_state import RuntimeState


@pytest.mark.parametrize("page_cached", [False, True])
def test_host_only_weight_load_keeps_control_bytes_after_cache_transition(page_cached):
    runtime = ControlPlaneRuntime(state=RuntimeState(page_cache={"pages"} if page_cached else set()))
    result = runtime.run(None, {"request_id": "r", "required_bytes": 4096,
        "control_byte_count": 64, "transfer_byte_count": 0,
        "placement": "host", "weight_key": "weights", "page_key": "pages"})
    actions = [row.action.value for row in result.completions]
    assert ("nvme_read" in actions) is not page_cached
    assert "pcie_transfer" not in actions
    for row in result.completions:
        if row.action.value in {"batch_schedule", "operator_schedule", "vram_controller"}:
            assert row.task.demands[0].bytes_moved == 64
    assert "weights" in result.state.weight_cache


def test_per_gpu_control_payload_matches_the_profile_cost_geometry():
    counts = _MaterializedWorkloadCounts(4, 2, 30, 30, {"gpu0": 10, "gpu1": 20})
    with patch("heterollm_sim.control_plane._materialized_workload_counts", return_value=counts):
        profile = _runtime_profile(scenario(), weight_bytes=4096, decision=object())
    targets = profile["gpu_command_targets"]
    assert [row["control_byte_count"] for row in targets] == [160, 320]
    result = ControlPlaneRuntime(profile, RuntimeState(weight_cache={"weights"})).run(None,
        {"request_id": "r", "required_bytes": 4096, "control_byte_count": profile["control_byte_count"],
         "weight_key": "weights", "placement": "gpu0"})
    moved = {row.payload["gpu_component_id"]: row.task.demands[0].bytes_moved
             for row in result.completions if row.action.value == "vram_controller"}
    assert moved == {"gpu0": 160, "gpu1": 320}


@pytest.mark.parametrize("target_ids", [(), ("gpu0",), ("gpu0", "gpu1")])
def test_explicit_continuation_runs_once_after_all_gpu_controllers(target_ids):
    targets = tuple({"component_id": target, "service_ns": {"compute": 9999.},
                     "resources": {"vram_controller": (target + ".control",)}}
                    for target in target_ids)
    first = TaskSpec("body-first", "r", "first", TaskCategory.COMPUTE,
                     demands=(ResourceDemand("gpu0", 100.),))
    second = replace(first, task_id="body-last", name="last", dependencies=(first.task_id,),
                     demands=(ResourceDemand("gpu0", 200.),))
    result = ControlPlaneRuntime({"gpu_command_targets": targets}, RuntimeState(weight_cache={"weights"})).run(
        None, {"request_id": "r", "required_bytes": 4096, "control_byte_count": 64,
               "weight_key": "weights", "placement": "gpu0", "continuations": (first, second)})
    body = {row.task.task_id: row for row in result.completions if row.task.task_id.startswith("body-")}
    assert set(body) == {"body-first", "body-last"}
    assert body["body-first"].end_ns - body["body-first"].start_ns == 100.
    assert body["body-last"].end_ns - body["body-last"].start_ns == 200.
    controllers = [row for row in result.completions if row.action.value == "vram_controller"]
    assert len(controllers) == len(target_ids)
    if controllers:
        assert body["body-first"].start_ns >= max(row.end_ns for row in controllers)
    assert not any(row.action.value == "compute" for row in result.completions)
    interrupt = next(row for row in result.completions if row.action.value == "interrupt")
    assert interrupt.start_ns >= body["body-last"].end_ns


def _explicit_memory_task(**metadata):
    return TaskSpec("read", "cohort-001", "read weights", TaskCategory.COMPUTE,
        demands=(ResourceDemand("hbm0.hbm_fabric", 0., bytes_moved=64),),
        metadata={"target_component": "gpu0", "memory_accesses": ({"operation": "read",
            "buffer_id": "weight-buffer", "byte_count": 64, "offset_bytes": 0,
            "allocation_size_bytes": 4096},), **metadata})


def test_lowering_preserves_the_declared_physical_allocation_extent():
    lowered = _attach_gddr_physical_task(_explicit_memory_task(), scenario())
    assert lowered.metadata["memory_accesses"][0]["allocation_size_bytes"] == 4096
    assert lowered.metadata["physical_address_bindings"][0]["allocation_size_bytes"] == 4096


@pytest.mark.parametrize("obsolete,current", (
    ("generation", "allocation_generation"), ("size_bytes", "byte_count"),
    ("offset", "offset_bytes"), ("tensor_id", "buffer_id"),
    ("buffer_size_bytes", "allocation_size_bytes"),
))
def test_lowering_rejects_obsolete_physical_descriptor_fields(obsolete, current):
    task = _explicit_memory_task()
    access = dict(task.metadata["memory_accesses"][0])
    access[obsolete] = access.get(current, 0)
    task = replace(task, metadata={**task.metadata, "memory_accesses": (access,)})
    with pytest.raises(ValueError, match="unsupported fields"):
        _attach_gddr_physical_task(task, scenario())


@pytest.mark.parametrize("weight_generation", [None, 7])
def test_explicit_resident_weight_keeps_its_generation_across_requests(weight_generation):
    metadata = {"weight_buffer_id": "weight-buffer"}
    if weight_generation is not None:
        metadata["weight_generation"] = weight_generation
    task = _explicit_memory_task(**metadata)
    generations = []
    for request in ("plain-001", "other-001"):
        lowered = _attach_gddr_physical_task(replace(task, request_id=request), scenario())
        generations.append(lowered.metadata["memory_accesses"][0]["allocation_generation"])
    assert generations == [weight_generation or 0] * 2


def test_explicit_transient_buffer_gets_a_distinct_request_generation():
    task = _explicit_memory_task()
    requests = ("plain-001", "other-001", "\0plain-001", "请求-001")
    generations = [_attach_gddr_physical_task(replace(task, request_id=request), scenario())
                   .metadata["memory_accesses"][0]["allocation_generation"]
                   for request in requests]
    assert all(generation > 0 for generation in generations)
    assert len(set(generations)) == len(requests)


@pytest.mark.parametrize("kind,summarize", [("HBM", _summarize_dram_task_traffic),
                                          ("HBF", _summarize_nand_task_traffic)])
def test_physical_summary_does_not_round_integer_traffic_through_float(kind, summarize):
    counts = (2**53 + 1, 2)
    tasks = tuple(TaskSpec(str(i), "r", "physical read", TaskCategory.MEMORY,
        metadata={"physical_execution_by_owner": {"mem": {"kind": kind,
            "logical_bytes": count, "logical_read_bytes": count,
            "physical_bytes": count, "physical_read_bytes": count,
            "service_ns": 0.5, "queue_wait_ns": 0.25}}}) for i, count in enumerate(counts))
    summary = summarize(tasks)
    for field in ("logical_bytes", "logical_read_bytes", "physical_bytes", "physical_read_bytes"):
        assert summary[field] == sum(counts)
        assert type(summary[field]) is int
    assert summary["service_ns"] == 1.0
    assert summary["queue_wait_ns"] == 0.5


@pytest.mark.parametrize("kind,summarize,optional", [
    ("HBF", _summarize_nand_task_traffic, (
        "logical_read_bytes", "logical_write_bytes", "internal_transfer_bytes",
        "pages_read", "pages_programmed", "pages_touched", "media_waves",
        "read_operations", "program_operations", "rmw_read_operations",
        "page_access_payload_bytes", "page_access_capacity_bytes", "host_admission_samples",
        "host_queue_wait_ns", "media_queue_wait_ns")),
    ("HBM", _summarize_dram_task_traffic, (
        "read_write_switches", "refresh_wait_ns", "turnaround_wait_ns")),
])
@pytest.mark.parametrize("recording", ["missing", "zero", "value", "partial"])
def test_physical_summary_distinguishes_unknown_optional_counters_from_zero(
        kind, summarize, optional, recording):
    rows = [{"kind": kind, "logical_bytes": 64, "physical_bytes": 128}]
    if recording != "missing":
        rows[0].update(dict.fromkeys(optional, 0))
    if recording in {"value", "partial"}:
        rows[0].update({key: 0.25 if key.endswith("_ns") else 2**53 + 1 for key in optional})
    if recording == "partial":
        rows.append({"kind": kind, "logical_bytes": 64, "physical_bytes": 128})
    tasks = (TaskSpec("owners", "r", "owners", TaskCategory.MEMORY,
        metadata={"physical_execution_by_owner": {str(i): row for i, row in enumerate(rows)}}),)
    summary = summarize(tasks)
    assert summary["logical_bytes"] == 64 * len(rows)
    for key in optional:
        assert summary[key] == (rows[0][key] if recording in {"zero", "value"} else None)
        assert summary["metric_availability"][key] == {
            "status": {"missing": "not_recorded", "zero": "recorded", "value": "recorded", "partial": "partial"}[recording],
            "recorded_owner_rows": int(recording != "missing"),
            "physical_owner_rows": len(rows),
        }


@pytest.mark.parametrize("kind,summarize", [("HBM", _summarize_dram_task_traffic),
                                          ("HBF", _summarize_nand_task_traffic)])
def test_optional_summary_preserves_counters_emitted_by_the_actual_core(kind, summarize):
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    from heterollm_sim.memory_types import make_hbm_config, make_hbf_config

    config = (make_hbm_config if kind == "HBM" else make_hbf_config)(capacity_bytes=1048576)
    task = TaskSpec("read", "r", "read", TaskCategory.MEMORY,
        metadata={"physical_memory_config": config, "physical_owner": "memory",
                  "memory_accesses": ({"operation": "read", "address": 0, "byte_count": 100},)})
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    raw = resolved.metadata["physical_execution_by_owner"]["memory"]
    summary = summarize((resolved,))
    for key, availability in summary["metric_availability"].items():
        if key.startswith("nand_"):
            continue
        assert summary[key] == raw.get(key)
        assert availability["status"] == ("recorded" if raw.get(key) is not None else "not_recorded")


def test_initial_control_root_honors_the_same_action_overrides_as_later_tasks():
    root, = build_initial_control_tasks(context={"request_id": "r",
        "service_ns_by_action": {"capacity_check": 12.5},
        "resources_by_action": {"capacity_check": ("cpu.special",)}})
    assert [(d.resource_id, d.service_ns) for d in root.demands] == [("cpu.special", 12.5)]


def _ffn_scenario(*, moe, gated, fusion):
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    base = scenario(prompt=8, count=1, output=2)
    layer = LayerSpec("dense0", "moe" if moe else "dense", 256, 512, 4,
        kv_heads=2, num_experts=2 if moe else 1, experts_per_token=1,
        shared_expert_intermediate_size=256 if moe else 0,
        shared_expert_gate=moe, gated_mlp=gated, dtype="int8", quantization="w8a8")
    model = ModelSpec(name="audit-ffn", graph=build_model_graph_from_layer_specs(
        "audit-ffn", (layer,), vocabulary_size=1024, max_sequence_length=256))
    return replace(base, model=model, placement=replace(base.placement, model_name=model.name),
        fusion_policy=replace(base.fusion_policy, gemm_epilogue_activation=fusion),
        workload=replace(base.workload, scheduler=replace(base.workload.scheduler, mode="static")))


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_control_plane_ffn_weights_and_fusion_follow_the_declared_gate(moe, gated):
    from heterollm_sim.control_plane_planner import (
        PlacementPolicy, _derive_requirements, _derive_fusion_opportunities,
    )
    configured = _ffn_scenario(moe=moe, gated=gated, fusion=True)
    requirements = _derive_requirements(configured)
    rows = {row.kind: row for row in requirements}
    ffn = rows["experts" if moe else "mlp"]
    assert [(matrix.k, matrix.n) for matrix in ffn.matrices] == [
        (256, 512 * (2 if gated else 1)), (512, 256)]
    assert ("expert_activation" if moe else "mlp_activation") in rows if gated else (
        "expert_activation" not in rows and "mlp_activation" not in rows)
    if moe:
        assert [(matrix.k, matrix.n) for matrix in rows["shared_expert"].matrices] == [
            (256, 256 * (2 if gated else 1)), (256, 256)]
        assert ("shared_expert_activation" in rows) is gated
        # The independent sigmoid gate on the shared expert's final output
        # is a separate declaration and must survive gated_mlp=False.
        assert "shared_expert_gate" in rows and "shared_expert_gate_apply" in rows
    opportunities = _derive_fusion_opportunities(configured, PlacementPolicy(), requirements)
    assert any(row.group == "gemm_epilogue_activation" for row in opportunities) is gated


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("fusion", [False, True])
def test_moe_actual_lowering_matches_up_gate_geometry_and_independent_shared_gate(gated, fusion):
    from heterollm_sim.control_plane_planner import plan_runtime_placement
    from heterollm_sim.planner import compile_scenario
    configured = _ffn_scenario(moe=True, gated=gated, fusion=fusion)
    decision = plan_runtime_placement(configured)
    assert decision.fully_placed
    tasks = compile_scenario(decision.apply(configured)).tasks
    projection = "up_gate" if gated else "up"
    gemms = [task for task in tasks if task.metadata.get("phase") == "gpu_gemm"]
    routed = [task for task in gemms if str(task.metadata.get("projection_id", "")).startswith("experts.")
              and str(task.metadata["projection_id"]).endswith("." + projection)]
    shared = [task for task in gemms if task.metadata.get("projection_id") == "shared_expert." + projection]
    assert routed and shared
    assert {task.metadata["cost_model"]["weight_bytes"] for task in routed} == {256 * 512 * (2 if gated else 1)}
    assert {task.metadata["cost_model"]["weight_bytes"] for task in shared} == {256 * 256 * (2 if gated else 1)}
    activation_present = any(task.metadata.get("event_kind") == "expert_activation" or
                             task.metadata.get("ffn_op") == "shared_activation" for task in tasks)
    epilogue_present = any(task.metadata.get("cost_model", {}).get("epilogue_name") == "swiglu"
                          for task in routed + shared)
    assert (activation_present or epilogue_present) is gated
    assert any(task.metadata.get("ffn_op") == "shared_gate_projection" for task in gemms)
    assert any(task.metadata.get("ffn_op") == "shared_gate_apply" for task in tasks)
