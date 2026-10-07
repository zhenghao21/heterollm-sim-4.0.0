"""Direct operands preserve local IO, stable identities and exact byte directions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from heterollm_sim import planner as p
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import CostPhase
from heterollm_sim.model_presets import materialize_model_payload
from heterollm_sim.web import scenario_or_http_error


DIRECTORY = Path(__file__).resolve().parents[1] / "docs/frontend_native_validation_2026-10-07"


def prepared_case(hardware="local", model="qwen3-0_6b"):
    # Capture from an actual browser submission reproduces the previous
    # same-owner failure, including llama's rank/backing selection.
    path = DIRECTORY / ("ui_before_fix_physical_io_" + hardware + "_qwen3_0_6b_submission.json")
    payload = json.loads(path.read_text(encoding="utf-8"))["scenario"]
    payload["model"] = materialize_model_payload(model)
    payload["workload"]["requests"][0].update(prompt_tokens=1, output_tokens=2)
    placement = payload["placement"]
    placement["model_name"] = payload["model"]["name"]
    for key in ("op_to_component", "tensor_to_component", "tensor_bytes"):
        placement[key] = {}
    placement["metadata"].pop("control_plane", None)
    placement["parallel"]["layer_to_stage"] = {}
    assert payload["profiles"]["llama_cpp"]["context"] == 640
    # The same preparation entry point used by /api/simulations.
    return scenario_or_http_error(payload)


@pytest.mark.parametrize("hardware,model", [
    ("local", "qwen3-0_6b"), ("local", "qwen2_5-0_5b"),
    ("b200_2hbf", "qwen3-0_6b"), ("b200_3hbf", "qwen3-0_6b"),
])
def test_real_frontend_preparation_preserves_every_direct_operand(hardware, model):
    scenario = prepared_case(hardware, model)
    schedule = p.compile_scenario(scenario)
    direct_tasks = [task for task in schedule.tasks
                    if task.metadata.get("phase_metadata", {}).get("direct_memory_access")]
    assert direct_tasks
    for task in direct_tasks:
        contract = task.metadata["phase_metadata"]["direct_memory_access"]
        accesses = task.metadata["memory_accesses"]
        for operation in ("read", "write"):
            expected = contract[operation + "_bytes"] + contract["local_" + operation + "_bytes"]
            assert sum(row["byte_count"] for row in accesses if row["operation"] == operation) == expected
        assert all(row["buffer_id"] and row["address_source"] == "stable_buffer_tensor_offset"
                   for row in accesses)
        assert len(task.metadata["physical_address_bindings"]) == len(accesses)
    qkv = next(task for task in direct_tasks if task.name.endswith("qkv.gpu_gemm"))
    accesses = qkv.metadata["memory_accesses"]
    assert len([row for row in accesses if row["operation"] == "read"]) == 2
    output, = [row for row in accesses if row["operation"] == "write"]
    assert output["byte_count"] == qkv.metadata["cost_model"]["output_bytes"]
    assert sum(row["byte_count"] for row in accesses if row["operation"] == "read") == (
        qkv.metadata["cost_model"]["activation_bytes"] + qkv.metadata["cost_model"]["weight_bytes"])
    if hardware == "local":
        assert {row["physical_owner"] for row in accesses} == {"gddr0.gddr_fabric"}
    else:
        # The small model fits HBM; this reproduces the same-owner bug for HBM
        # as well as GDDR, rather than artificially forcing weights to HBF.
        assert {row["physical_owner"] for row in accesses} == {"hbm0.hbm_fabric"}


def explicit_task(scenario):
    rows = [
        {"buffer_id": "input", "offset_bytes": 0, "size_bytes": 16, "operation": "read"},
        {"buffer_id": "cache-a", "offset_bytes": 8, "size_bytes": 32, "operation": "read",
         "allocation_generation": 0, "buffer_size_bytes": 64},
        {"buffer_id": "cache-b", "offset_bytes": 0, "size_bytes": 48, "operation": "read",
         "allocation_generation": 1, "buffer_size_bytes": 96},
        {"buffer_id": "output", "offset_bytes": 0, "size_bytes": 16, "operation": "write"},
        {"buffer_id": "cache-a", "offset_bytes": 40, "size_bytes": 24, "operation": "write",
         "allocation_generation": 0, "buffer_size_bytes": 64},
    ]
    owner = "gddr0.gddr_fabric"
    direct = {"component_id": "gddr0", "physical_owner": owner, "resource_id": owner,
              "local_resource_id": owner, "read_bytes": 80, "write_bytes": 24,
              "local_read_bytes": 16, "local_write_bytes": 16,
              "rhs_buffer_accesses": rows[1:3], "output_buffer_accesses": rows[4:]}
    return TaskSpec("test", "cohort-17", "direct", TaskCategory.COMPUTE,
        demands=(ResourceDemand(owner, 123.0, bytes_moved=512),),
        metadata={"operator_id": "direct", "target_component": "gpu0",
                  "buffer_accesses": rows, "persistent_buffer_ids": ("cache-a",),
                  "phase_metadata": {"direct_memory_access": direct}, "cost_model": {}})


def test_same_owner_read_write_segments_keep_offsets_generations_and_local_io():
    scenario = prepared_case()
    bound = p._attach_gddr_physical_task(explicit_task(scenario), scenario)
    rows = bound.metadata["memory_accesses"]
    assert len(rows) == 5
    assert sum(row["byte_count"] for row in rows if row["operation"] == "read") == 96
    assert sum(row["byte_count"] for row in rows if row["operation"] == "write") == 40
    assert {(row["offset_bytes"], row["byte_count"], row["allocation_generation"])
            for row in rows if row["buffer_id"] == "cache-a"} == {(8, 32, 0), (40, 24, 0)}
    assert next(row for row in rows if row["buffer_id"] == "cache-b")["allocation_generation"] == 1
    # Neither the already-rounded demand nor physical timing previews can
    # replace the exact logical 136-byte operand contract.
    assert bound.demands[0].bytes_moved == 512
    assert bound.metadata["direct_memory_byte_conservation"]["logical_read_bytes"] == 96
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    runtime = PhysicalRuntimeContext()
    resolved = resolve_physical_task(bound, runtime, 0.0)
    execution = resolved.metadata["physical_execution"]
    assert execution["logical_read_bytes"] == 96
    assert execution["logical_write_bytes"] == 40
    assert execution["physical_read_bytes"] > 0 and execution["physical_write_bytes"] > 0
    allocator = runtime.allocators["gddr0.gddr_fabric"]
    assert len(allocator.allocations()) == 4  # cache-a read/write share one allocation
    assert allocator.get_allocation("cache-a", 0) is not None
    assert allocator.get_allocation("cache-b", 1) is not None


def test_remote_hbf_and_local_hbm_keep_independent_physical_owners():
    scenario = prepared_case("b200_2hbf")
    task = explicit_task(scenario)
    metadata = deepcopy(task.metadata)
    memory = next(item for item in scenario.hardware.components if item.normalized_kind == "hbf")
    from heterollm_sim.data_motion import resolve_service
    service = resolve_service(memory, p._resolve_component_profile(scenario, memory.component_id))
    direct = metadata["phase_metadata"]["direct_memory_access"]
    direct.update(component_id=memory.component_id, physical_owner=service.physical_owner,
                  resource_id=service.resource_id, local_resource_id="hbm0.hbm_fabric")
    task = replace(task, metadata=metadata, demands=(
        ResourceDemand("hbm0.hbm_fabric", 1.0, bytes_moved=32),
        ResourceDemand(service.resource_id, 1.0, bytes_moved=4096)))
    bound = p._attach_gddr_physical_task(task, scenario)
    rows = bound.metadata["memory_accesses"]
    assert {row["physical_owner"] for row in rows} == {"hbm0.hbm_fabric", service.physical_owner}
    assert sum(row["byte_count"] for row in rows if row["physical_owner"] == "hbm0.hbm_fabric") == 32
    assert sum(row["byte_count"] for row in rows if row["physical_owner"] == service.physical_owner) == 104
    assert set(bound.metadata["physical_memory_configs"]) == {"hbm0.hbm_fabric", service.physical_owner}
    rates = bound.metadata["physical_energy_pj_per_byte_by_owner"]
    assert rates[service.physical_owner] == p._resolve_component_profile(
        scenario, memory.component_id).energy_pj_per_byte > 0
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    resolved = resolve_physical_task(bound, PhysicalRuntimeContext(), 0.0)
    for owner, execution in resolved.metadata["physical_execution_by_owner"].items():
        # One configured endpoint coefficient prices its resolved media bytes
        # once, including read-modify-write bytes, not every internal hop.
        assert execution["energy_pj"] == pytest.approx(execution["physical_bytes"] * rates[owner])
    unpriced = replace(bound, metadata={**bound.metadata,
        "physical_energy_pj_per_byte_by_owner": dict.fromkeys(rates, 0.0)})
    baseline = resolve_physical_task(unpriced, PhysicalRuntimeContext(), 0.0)
    assert baseline.metadata["physical_completion_ns"] == resolved.metadata["physical_completion_ns"]
    assert [(d.resource_id, d.service_ns, d.bytes_moved) for d in baseline.demands] == [
        (d.resource_id, d.service_ns, d.bytes_moved) for d in resolved.demands]
    assert sum(d.energy_pj for d in resolved.demands) == pytest.approx(sum(
        row["physical_bytes"] * rates[owner]
        for owner, row in resolved.metadata["physical_execution_by_owner"].items()))


@pytest.mark.parametrize("failure", ["missing_rhs", "wrong_read_total", "wrong_local_total", "wrong_generation"])
def test_direct_operand_contract_mismatch_is_rejected(failure):
    scenario = prepared_case()
    task = explicit_task(scenario)
    metadata = deepcopy(task.metadata)
    direct = metadata["phase_metadata"]["direct_memory_access"]
    if failure == "missing_rhs":
        metadata["buffer_accesses"] = [row for row in metadata["buffer_accesses"] if row["buffer_id"] != "cache-b"]
    elif failure == "wrong_read_total":
        direct["read_bytes"] += 1
    elif failure == "wrong_local_total":
        direct["local_read_bytes"] += 1
    else:
        direct["rhs_buffer_accesses"] = deepcopy(direct["rhs_buffer_accesses"])
        direct["rhs_buffer_accesses"][1]["allocation_generation"] = 2
    with pytest.raises(ValueError, match="cover|represented|match"):
        p._attach_gddr_physical_task(replace(task, metadata=metadata), scenario)


def test_staged_memory_does_not_acquire_a_second_direct_operand():
    scenario = prepared_case()
    metadata = deepcopy(scenario.placement.metadata)
    metadata["llama_backend_memory"]["gddr0"]["access"] = "staged"
    scenario = replace(scenario, placement=replace(scenario.placement, metadata=metadata))
    rank = p._parallel_plan(scenario).ranks[0]
    phase = CostPhase("gpu_gemm", TaskCategory.COMPUTE, (ResourceDemand("gpu0.cuda", 1.0),), metadata={})
    assert p._direct_memory_phase(scenario, rank, phase, "gddr0", read_bytes=4096) is phase


@pytest.mark.parametrize("component_id", ["hostmem0", "gddr0"])
def test_already_physical_endpoint_uses_current_profile_energy_and_keeps_explicit_zero(component_id):
    from heterollm_sim.data_motion import endpoint_service, PhysicalRuntimeContext, resolve_physical_task
    scenario = prepared_case()
    component = next(item for item in scenario.hardware.components if item.component_id == component_id)
    kind = scenario.component_profile_kind(component)
    profiles = {key: dict(values) for key, values in scenario.component_profiles.items()}
    profiles[kind][component.cost_profile_id] = replace(
        p._resolve_component_profile(scenario, component.component_id), energy_pj_per_byte=7.25)
    scenario = replace(scenario, component_profiles=profiles)
    for read in (True, False):
        endpoint = endpoint_service(component, 64, read=read, name="endpoint-energy",
                                    dram_address_bytes=0, compact_preview=True)
        task = TaskSpec("endpoint-energy", "cohort-energy", endpoint.name, TaskCategory.COMPUTE,
                        demands=endpoint.demands, metadata=endpoint.metadata)
        bound = p._attach_gddr_physical_task(task, scenario)
        owner = endpoint.metadata["physical_owner"]
        assert bound.metadata["physical_energy_pj_per_byte_by_owner"] == {owner: 7.25}
        result = resolve_physical_task(bound, PhysicalRuntimeContext(), 0)
        physical = result.metadata["physical_execution"]
        assert physical["energy_pj"] == physical["physical_bytes"] * 7.25
        explicit = replace(task, metadata={**task.metadata, "physical_energy_pj_per_byte": 0.0})
        assert p._attach_gddr_physical_task(explicit, scenario).metadata["physical_energy_pj_per_byte"] == 0


def test_subtract_remote_subrange_keeps_both_local_sides():
    full = [{"buffer_id": "shared", "offset_bytes": 0, "size_bytes": 64, "operation": "read"}]
    remote = [{"buffer_id": "shared", "offset_bytes": 16, "byte_count": 32,
               "operation": "read", "allocation_generation": 0}]
    assert [(row["offset_bytes"], row["size_bytes"]) for row in p._subtract_direct_buffer_accesses(full, remote)] == [(0, 16), (48, 16)]
