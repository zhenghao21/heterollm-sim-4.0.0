"""Invalid inputs cannot produce a plausible result through a cheaper model."""

from dataclasses import asdict, fields, replace

import pytest

from heterollm_sim.config import hardware_from_dict, scheduler_from_dict, workload_from_dict
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import HostMemoryProfile
from heterollm_sim.data_motion import endpoint_service
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.physical_contract import require_physical_memory_config
from heterollm_sim.planner import _attach_gddr_physical_task, _compilation_scope
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serde import to_primitive


@pytest.mark.parametrize("kind", ("gddr", "hbm", "host_memory", "ddr", "lpddr", "hbf", "ssd"))
def test_missing_physical_config_is_rejected_at_service_boundary(kind):
    component = ComponentSpec("memory0", kind, capacity_bytes=1024, bandwidth_gbps=800)
    with pytest.raises(ValueError, match="physical_memory_config is required"):
        endpoint_service(component, 64, read=True, name="read", page_offset_bytes=0)


def test_missing_physical_config_is_rejected_before_scenario_executes():
    scenario = build_reference_scenario()
    host = scenario.hardware.get_component("hostmem0")
    metadata = dict(host.metadata)
    del metadata["physical_memory_config"]
    invalid = replace(host, metadata=metadata)
    with pytest.raises(ValueError, match="hostmem0.*physical_memory_config is required"):
        replace(scenario, hardware=replace(scenario.hardware, components=tuple(
            invalid if c.component_id == host.component_id else c for c in scenario.hardware.components
        )))


def test_partial_physical_config_cannot_be_filled_by_core_defaults_on_import():
    hardware = to_primitive(build_reference_scenario().hardware)
    memory = next(c for c in hardware["components"] if c["component_id"] == "hbm0")
    del memory["metadata"]["physical_memory_config"]["open_ns"]
    with pytest.raises(ValueError, match="missing explicit fields: open_ns"):
        hardware_from_dict(hardware)


def test_physical_capacity_disagreement_is_rejected():
    host = build_reference_scenario().hardware.get_component("hostmem0")
    with pytest.raises(ValueError, match="capacity_bytes must be positive and equal"):
        require_physical_memory_config(replace(host, capacity_bytes=host.capacity_bytes // 2))


@pytest.mark.parametrize("family", ("HBM", "LPDDR"))
def test_gpu_hbm_and_unified_lpddr_traffic_reach_shared_physical_core(family):
    scenario = build_reference_scenario()
    memory = scenario.hardware.get_component("hbm0")
    profiles = {kind: dict(values) for kind, values in scenario.component_profiles.items()}
    if family == "LPDDR":
        config = {**memory.metadata["physical_memory_config"], "kind": "LPDDR", "generation": "LPDDR5X"}
        profile = scenario.resolve_component_profile(memory)
        profile = HostMemoryProfile(**{f.name: getattr(profile, f.name) for f in fields(HostMemoryProfile) if hasattr(profile, f.name)})
        profiles["host_memory"]["gpu-lpddr"] = profile
        memory = replace(memory, kind="host_memory", cost_profile_id="gpu-lpddr", metadata={
            **memory.metadata, "physical_memory_config": config,
        })
        scenario = replace(scenario, component_profiles=profiles, hardware=replace(scenario.hardware, components=tuple(
            memory if c.component_id == memory.component_id else c for c in scenario.hardware.components
        )))
    owner = scenario.resolve_component_profile(memory).resource_id
    task = TaskSpec("read", "request", "read", TaskCategory.COMPUTE,
        demands=(ResourceDemand(owner, 999999.0, bytes_moved=192),), metadata={
            "target_component": "gpu0", "input_tensor_id": "input", "output_tensor_id": "output",
            "cost_model": {"physical_read_bytes": 128, "physical_write_bytes": 64},
        })
    with _compilation_scope(scenario):
        physical = _attach_gddr_physical_task(task, scenario)
    assert physical.metadata["physical_memory_config"]["kind"] == family
    event = UnifiedEventKernel.from_closed_graph((physical,)).step()
    execution = event.task.metadata["physical_execution"]
    assert execution["physical_read_bytes"] == 128
    assert execution["physical_write_bytes"] == 64
    assert execution["service_ns"] < 999999.0


@pytest.mark.parametrize("flag", ("stage", "memory", "launch", "request_boundary"))
def test_enabled_calibration_without_profile_is_rejected(flag):
    scenario = build_reference_scenario()
    with pytest.raises(ValueError, match="enabled native calibration requires"):
        replace(scenario, placement=replace(scenario.placement, metadata={"native_calibration_apply_" + flag: True}))


def test_empty_scheduler_or_request_shape_cannot_be_replaced_by_defaults():
    with pytest.raises(ValueError, match="scheduler missing explicit fields"):
        scheduler_from_dict({})
    with pytest.raises(ValueError, match="request missing explicit fields"):
        workload_from_dict({"name": "missing-shape", "requests": [{"request_id": "r0"}]})


def test_paged_pool_rejects_unmodeled_migration_before_changing_ownership():
    from heterollm_sim.kv_pool import DynamicKVPool, KvPoolComponent, KvPoolUnsupported

    pool = DynamicKVPool((KvPoolComponent("a", 4096), KvPoolComponent("b", 4096)), page_bytes=64)
    assert pool.allocate("request", 1, compatible_components=("a",))
    page = pool.request_pages("request")[0]
    used = dict(pool.used_bytes_by_component)
    with pytest.raises(KvPoolUnsupported, match="physical address and shared memory-event"):
        pool.migrate_page(page.logical_page_id, "b")
    assert page.owner_component == "a"
    assert pool.used_bytes_by_component == used
    assert pool.events == ()


def test_requested_invalid_kernel_calibration_does_not_return_analytical_cost():
    from heterollm_sim.cost_models import CostPhase
    from heterollm_sim.kernel_calibration import apply_kernel_calibration

    phase = CostPhase("gemm", TaskCategory.COMPUTE, (ResourceDemand("gpu.compute", 10.0),))
    with pytest.raises(ValueError, match="configured kernel calibration cannot be applied"):
        apply_kernel_calibration(phase, {}, {})


def test_direct_remote_hbm_preserves_local_activation_and_remote_weight_accesses():
    from heterollm_sim.cost_models import CostPhase
    from heterollm_sim.planner import _direct_memory_phase, _parallel_plan

    scenario = build_reference_scenario()
    scenario = replace(scenario, placement=replace(scenario.placement, metadata={
        **scenario.placement.metadata,
        "llama_backend_memory": {"hbm1": {"access": "direct", "device_id": "gpu0"}},
    }))
    rank = _parallel_plan(scenario).ranks[0]
    local = scenario.resolve_component_profile("hbm0").resource_id
    remote = scenario.resolve_component_profile("hbm1").resource_id
    cost = {"physical_read_bytes": 128, "physical_write_bytes": 64}
    phase = CostPhase("gpu_gemm", TaskCategory.COMPUTE,
        (ResourceDemand(local, 100, bytes_moved=192),), metadata={"cache": cost})
    metadata = {"rank": rank.rank, "target_component": "gpu0", "op_name": "test.gemm",
                "input_tensor_id": "input", "output_tensor_id": "output", "weight_tensor_id": "weights"}
    with _compilation_scope(scenario):
        moved = _direct_memory_phase(scenario, rank, phase, "hbm1", read_bytes=64, metadata=metadata)
        task = TaskSpec("gemm", "request", "gemm", TaskCategory.COMPUTE,
            demands=moved.demands, metadata={**metadata, "phase_metadata": moved.metadata, "cost_model": cost})
        bound = _attach_gddr_physical_task(task, scenario)
    assert set(bound.metadata["physical_memory_configs"]) == {local, remote}
    accesses = bound.metadata["memory_accesses"]
    assert sum(a["byte_count"] for a in accesses if a["physical_owner"] == local) == 128
    assert sum(a["byte_count"] for a in accesses if a["physical_owner"] == remote) == 64
    event = UnifiedEventKernel.from_closed_graph((bound,)).step()
    assert event.task.metadata["physical_execution"]["physical_read_bytes"] == 128
    assert event.task.metadata["physical_execution"]["physical_write_bytes"] == 64
    assert event.task.metadata["physical_execution"]["energy_pj"] > 0


def test_mixed_dram_nand_task_reports_only_each_physical_owners_costs():
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    from heterollm_sim.memory_types import NandConfig
    from heterollm_sim.planner import _summarize_dram_task_traffic, _summarize_nand_task_traffic

    scenario = build_reference_scenario()
    dram = scenario.hardware.get_component("hbm0")
    configs = {
        "hbm0.memory": dram.metadata["physical_memory_config"],
        "hbf0.memory": asdict(NandConfig(kind="HBF")),
    }
    task = TaskSpec("mixed", "r0", "mixed", TaskCategory.COMPUTE,
        demands=(ResourceDemand("gpu0.compute", 10.0),), metadata={
            "physical_owner": "hbm0.memory", "physical_memory_config": configs["hbm0.memory"],
            "physical_memory_configs": configs,
            "physical_energy_pj_per_byte_by_owner": {"hbm0.memory": 2.0, "hbf0.memory": 7.0},
            "memory_accesses": (
                {"physical_owner": "hbm0.memory", "operation": "write", "address": 0, "byte_count": 64},
                {"physical_owner": "hbf0.memory", "operation": "read", "address": 0, "byte_count": 256},
            ),
        })
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.0)
    dram_report = _summarize_dram_task_traffic((resolved,))
    nand_report = _summarize_nand_task_traffic((resolved,))
    assert dram_report["task_count"] == nand_report["task_count"] == 1
    assert dram_report["logical_bytes"] == 64
    assert nand_report["logical_bytes"] == 256
    assert dram_report["physical_write_bytes"] > 0
    assert nand_report["physical_read_bytes"] > 0
    assert dram_report["physical_read_bytes"] == nand_report["physical_write_bytes"] == 0
    assert dram_report["resource_totals"]
    assert nand_report["resource_totals"]
    assert all(row["owner"] == "hbm0.memory" for row in dram_report["resource_totals"].values())
    assert all(row["owner"] == "hbf0.memory" for row in nand_report["resource_totals"].values())
    for demand in resolved.demands:
        if demand.resource_id in dram_report["resource_totals"]:
            assert demand.energy_pj == demand.bytes_moved * 2.0
        elif demand.resource_id in nand_report["resource_totals"]:
            assert demand.energy_pj == demand.bytes_moved * 7.0
    by_owner = resolved.metadata["physical_execution_by_owner"]
    assert sum(row["energy_pj"] for row in by_owner.values()) == resolved.metadata["physical_execution"]["energy_pj"]
    assert nand_report["energy_pj"] == sum(row["energy_pj"] for row in nand_report["resource_totals"].values())


def test_mtp_without_operator_descriptors_cannot_use_scalar_policy_cost(monkeypatch):
    from types import SimpleNamespace
    from heterollm_sim import planner

    monkeypatch.setattr(planner, "_mtp_execution_descriptors", lambda scenario: ())
    with pytest.raises(ValueError, match="MTP execution requires explicit model operator"):
        planner._compile_parallel_mtp_proposer(
            SimpleNamespace(), build_reference_scenario(), None, None,
            next_token=0, draft_step_lanes=(1,), verifier_tokens=1,
            accepted_tokens=0, policy=SimpleNamespace(proposal_cost_scale=0.15), dependencies=(),
        )


def test_required_request_marker_calibration_cannot_silently_skip_missing_evidence():
    from heterollm_sim.calibration import NativeCalibrationProfile, request_marker_calibration_ns

    with pytest.raises(ValueError, match="request marker calibration"):
        request_marker_calibration_ns(
            NativeCalibrationProfile(), "request_begin", model_sha256=None,
            hardware_fingerprint=None, runtime_fingerprint=None,
            prompt_tokens=2, output_tokens=2, required=True,
        )
