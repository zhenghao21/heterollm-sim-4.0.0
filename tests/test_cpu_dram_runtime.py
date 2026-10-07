"""Focused runtime checks for CPU cache-miss traffic reaching physical DDR."""

from dataclasses import asdict, replace

from heterollm_sim.cost_models import MemoryWorkload, estimate_cpu_memory
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import RankMappingSpec, RequestSpec
from heterollm_sim.memory_types import DramConfig, MemoryKind, NandConfig
from heterollm_sim.planner import (
    _add_transfer_tasks,
    _TaskBuilder,
    _compilation_scope,
    _cpu_profiles,
    _physical_dram_config,
    _promote_physical_allocation_extents,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec


def _ddr_config(*, bandwidth=420.0, burst=64, rows_per_bank=256, capacity_bytes=1024 * 1024):
    return DramConfig(
        kind=MemoryKind.DDR,
        generation="DDR5",
        channels=1,
        ranks_per_channel=1,
        bank_groups_per_rank=2,
        banks_per_group=2,
        rows_per_bank=rows_per_bank,
        row_bytes=1024,
        burst_bytes=burst,
        interleave_bytes=burst,
        data_width_bits=64,
        data_rate_mt_s=3200.0,
        interface_bandwidth_gb_s=bandwidth,
        read_latency_ns=18.0,
        write_latency_ns=20.0,
        burst_interval_ns=3.0,
        capacity_bytes=capacity_bytes,
    )


def _scenario(config=None):
    scenario = build_reference_scenario()
    host = scenario.hardware.get_component("hostmem0")
    metadata = dict(host.metadata)
    if config is not None:
        metadata["physical_memory_config"] = asdict(config)
        metadata["physical_owner"] = "hostmem0.ddr"
        memory_service = dict(metadata.get("memory_service", {}))
        memory_service["physical_owner"] = "hostmem0.ddr"
        metadata["memory_service"] = memory_service
        host = replace(host, metadata=metadata)
    else:
        host = replace(host, metadata=metadata)
    components = tuple(host if item.component_id == "hostmem0" else item
                       for item in scenario.hardware.components)
    return replace(scenario, hardware=replace(scenario.hardware, components=components))


def _compiled_cpu_schedule():
    from heterollm_sim.planner import compile_scenario

    scenario = build_reference_scenario()
    config = _ddr_config(rows_per_bank=65536, capacity_bytes=256 * 1024 * 1024)
    host = scenario.hardware.get_component("hostmem0")
    metadata = dict(host.metadata)
    metadata["physical_memory_config"] = asdict(config)
    host = replace(host, metadata=metadata)
    hbm = scenario.hardware.get_component("hbm0")
    hbm_metadata = dict(hbm.metadata)
    hbm_metadata["physical_memory_config"] = asdict(DramConfig(
        kind=MemoryKind.HBM,
        generation="HBM3",
        channels=1,
        ranks_per_channel=1,
        bank_groups_per_rank=8,
        banks_per_group=4,
        rows_per_bank=65536,
        row_bytes=8192,
        burst_bytes=64,
        interleave_bytes=64,
        interface_bandwidth_gb_s=500.0,
        capacity_bytes=hbm.capacity_bytes,
    ))
    hbm = replace(hbm, metadata=hbm_metadata)
    components = tuple(
        host if item.component_id == "hostmem0" else
        hbm if item.component_id == "hbm0" else item
        for item in scenario.hardware.components
    )
    workload = replace(
        scenario.workload,
        requests=(replace(scenario.workload.requests[0], prompt_tokens=1, output_tokens=1),),
    )
    scenario = replace(
        scenario,
        hardware=replace(scenario.hardware, components=components),
        placement=replace(scenario.placement, op_to_component={
            "embedding": "cpu0", "lm_head": "cpu0",
        }),
        workload=workload,
    )
    return scenario, compile_scenario(scenario)


def _compiled_task_closure(schedule, roots):
    by_id = {task.task_id: task for task in schedule.tasks}
    closure = set()
    pending = [task.task_id for task in roots]
    while pending:
        task_id = pending.pop()
        if task_id in closure:
            continue
        closure.add(task_id)
        pending.extend(by_id[task_id].dependencies)
    return tuple(by_id[item] for item in closure)


def _cpu_memory_task(config=None):
    scenario = _scenario(config if config is not None else _ddr_config())
    cpu, memory = _cpu_profiles(scenario, "cpu0")
    estimate = estimate_cpu_memory(
        cpu,
        memory,
        MemoryWorkload(
            read_bytes=4097,
            write_bytes=1025,
            working_set_bytes=32 * 1024,
            streaming_fraction=1.0,
            name="cpu-dram-regression",
        ),
    )
    phase = next(item for item in estimate.phases if item.name == "cpu_memory")
    builder = _TaskBuilder(RequestSpec("cpu-dram-test", 0.0, 1, 1))
    with _compilation_scope(scenario):
        builder.add(
            "cpu.memory",
            phase.category,
            phase.demands,
            metadata={
                "target_component": "cpu0",
                "phase": phase.name,
                "cost_model": dict(estimate.metadata),
                "phase_metadata": dict(phase.metadata),
                "input_tensor_id": "cpu-dram-test-buffer",
                "output_tensor_id": "cpu-dram-test-buffer",
            },
        )
    return scenario, estimate, builder.tasks[0]


def test_cpu_cache_backing_traffic_is_replaced_by_ddr_and_runs_in_core():
    _scenario_config, estimate, task = _cpu_memory_task()
    cache = estimate.metadata["cache"]
    assert task.metadata["dram_backing_service"]["replaces_analytical_backing_service"]
    assert task.metadata["dram_backing_service"]["logical_read_bytes"] == cache["logical_backing_read_bytes"]
    assert task.metadata["dram_backing_service"]["logical_write_bytes"] == cache["logical_backing_write_bytes"]
    accesses = task.metadata["memory_accesses"]
    assert sum(item["byte_count"] for item in accesses if item["operation"] == "read") == cache["logical_backing_read_bytes"]
    assert sum(item["byte_count"] for item in accesses if item["operation"] == "write") == cache["logical_backing_write_bytes"]

    # The CPU pipeline and cache levels remain analytical demands; only the
    # old HostMemoryProfile backing demand is replaced by the physical owner.
    demand_ids = {item.resource_id for item in task.demands}
    assert "cpu0.pipeline" in demand_ids
    assert {"cpu0.l1d", "cpu0.l2", "cpu0.l3"} <= demand_ids
    assert "cpu0.memory" in demand_ids
    assert next(item for item in task.demands if item.resource_id == "cpu0.memory").service_ns == 0.0
    assert next(item for item in task.demands if item.resource_id == "cpu0.memory").bytes_moved == (
        cache["logical_backing_read_bytes"] + cache["logical_backing_write_bytes"]
    )
    assert all(item.bytes_moved == 0 for item in task.demands if item.resource_id == "cpu0.pipeline")
    _cpu_profile, memory_profile = _cpu_profiles(_scenario_config, "cpu0")
    phase = next(item for item in estimate.phases if item.name == "cpu_memory")
    assert tuple(item for item in task.demands if item.resource_id != memory_profile.resource_id) == tuple(
        item for item in phase.demands if item.resource_id != memory_profile.resource_id
    )

    kernel = UnifiedEventKernel.from_closed_graph((task,))
    event = kernel.step()
    assert event is not None
    execution = event.task.metadata["physical_execution"]
    assert execution["logical_read_bytes"] == cache["logical_backing_read_bytes"]
    assert execution["logical_write_bytes"] == cache["logical_backing_write_bytes"]
    assert execution["service_ns"] > 0
    assert execution["burst_count"] > 0
    assert execution["physical_bytes"] % 64 == 0
    assert execution["energy_pj"] == execution["physical_bytes"] * 12.0
    physical_ids = set(event.task.metadata["physical_demands_resource_ids"])
    assert sum(item.energy_pj for item in event.task.demands if item.resource_id in physical_ids) == execution["energy_pj"]


def test_physical_ddr_parameters_change_runtime_service_not_cpu_cache_model():
    _scenario_fast, estimate_fast, task_fast = _cpu_memory_task(_ddr_config(bandwidth=800.0))
    _scenario_slow, estimate_slow, task_slow = _cpu_memory_task(_ddr_config(bandwidth=420.0))
    assert estimate_fast.metadata["cache"] == estimate_slow.metadata["cache"]

    fast = UnifiedEventKernel.from_closed_graph((task_fast,)).step()
    slow = UnifiedEventKernel.from_closed_graph((task_slow,)).step()
    assert fast is not None and slow is not None
    assert slow.task.metadata["physical_execution"]["service_ns"] > fast.task.metadata["physical_execution"]["service_ns"]


def test_cpu_memory_without_physical_ddr_keeps_analytical_backing_semantics():
    # Rebuild without an explicit physical config to exercise the legacy path.
    scenario = _scenario(None)
    cpu, memory = _cpu_profiles(scenario, "cpu0")
    estimate = estimate_cpu_memory(cpu, memory, MemoryWorkload(
        read_bytes=4096, write_bytes=1024, working_set_bytes=32 * 1024,
        streaming_fraction=1.0, name="cpu-dram-regression"))
    phase = next(item for item in estimate.phases if item.name == "cpu_memory")
    builder = _TaskBuilder(RequestSpec("cpu-analytical-test", 0.0, 1, 1))
    with _compilation_scope(scenario):
        builder.add("cpu.memory", phase.category, phase.demands, metadata={
            "target_component": "cpu0", "phase": phase.name,
            "cost_model": dict(estimate.metadata), "phase_metadata": dict(phase.metadata),
            "input_tensor_id": "cpu-analytical-buffer",
            "output_tensor_id": "cpu-analytical-buffer",
        })
    task = builder.tasks[0]
    assert "physical_memory_config" not in task.metadata
    assert "memory_accesses" not in task.metadata
    assert any(item.resource_id == memory.resource_id and item.bytes_moved > 0 for item in task.demands)
    assert task.demands == phase.demands


def test_same_buffer_extent_promotion_is_scoped_by_physical_owner():
    config = asdict(_ddr_config())
    tasks = tuple(
        TaskSpec(
            task_id=f"owner-{owner}", request_id="owner-test", name="access",
            category=TaskCategory.MEMORY,
            demands=(ResourceDemand(owner, 1.0, bytes_moved=size),),
            metadata={
                "physical_memory_config": config,
                "physical_owner": owner,
                "memory_access": {
                    "operation": "read", "buffer_id": "same-buffer",
                    "byte_count": size, "generation": 3,
                    "physical_owner": owner, "resource_id": owner,
                },
            },
        )
        for owner, size in (("cpu-host-ddr", 128), ("gpu-vram", 256))
    )
    promoted = _promote_physical_allocation_extents(tasks)
    extents = {
        (task.metadata["physical_owner"], task.metadata["memory_accesses"][0]["allocation_size_bytes"])
        for task in promoted
    }
    assert extents == {("cpu-host-ddr", 128), ("gpu-vram", 256)}


def test_compiled_cpu_operators_and_host_pack_use_physical_ddr_hook():
    _scenario_config, schedule = _compiled_cpu_schedule()
    cpu_tasks = [task for task in schedule.tasks if task.metadata.get("target_component") == "cpu0"]
    assert any(task.metadata.get("phase") == "cpu_gemm"
               and task.metadata.get("physical_memory_config") for task in cpu_tasks)
    assert any("host_orchestration.pack.cpu_memory" in task.name
               and task.metadata.get("physical_memory_config") for task in schedule.tasks)
    assert any("cpu_token_commit.cpu_memory" in task.name
               and task.metadata.get("physical_memory_config") for task in schedule.tasks)
    assert any(task.metadata.get("phase") == "cpu_dispatch" and task.metadata.get("target_component") == "cpu0"
               and not task.metadata.get("physical_memory_config") for task in cpu_tasks)


def test_compiled_cpu_gemm_reaches_physical_ddr_at_runtime():
    _scenario_config, schedule = _compiled_cpu_schedule()
    gemm = next(task for task in schedule.tasks
                if task.metadata.get("target_component") == "cpu0"
                and task.metadata.get("phase") == "cpu_gemm"
                and task.metadata.get("physical_memory_config"))
    kernel = UnifiedEventKernel.from_closed_graph(_compiled_task_closure(schedule, (gemm,)))
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
    executed = next(event.task for event in events if event.task.task_id == gemm.task_id)
    assert executed.metadata["physical_execution"]["physical_bytes"] > 0
    assert executed.metadata["physical_execution"]["service_ns"] > 0
    assert executed.metadata["dram_backing_service"]["replaces_analytical_backing_service"]


def test_cpu_to_gpu_dma_keeps_link_path_and_physical_memory_endpoints():
    _scenario_config, schedule = _compiled_cpu_schedule()
    prefix = "prefill.host_orchestration.h2d"
    tasks_by_id = {task.task_id: task for task in schedule.tasks}
    endpoint_tasks = [task for task in schedule.tasks if task.name.startswith(prefix)]
    host_read = next(task for task in endpoint_tasks if task.name.endswith("hostmem0.read"))
    gpu_write = next(task for task in endpoint_tasks if task.name.endswith("hbm0.write"))
    links = [task for task in endpoint_tasks if ".link" in task.name]
    assert host_read.metadata.get("physical_memory_config", {}).get("kind") == "DDR"
    assert gpu_write.metadata.get("physical_memory_config") is not None
    assert [(task.metadata.get("source_component"), task.metadata.get("target_component")) for task in links] == [
        ("hostmem0", "cpu0"), ("cpu0", "gpu0"), ("gpu0", "hbm0"),
    ]

    closure_ids = set()
    pending = [host_read.task_id, gpu_write.task_id, *(task.task_id for task in links)]
    while pending:
        task_id = pending.pop()
        if task_id in closure_ids:
            continue
        closure_ids.add(task_id)
        pending.extend(tasks_by_id[task_id].dependencies)
    kernel = UnifiedEventKernel.from_closed_graph(tuple(tasks_by_id[item] for item in closure_ids))
    events = []
    while kernel.has_active_tasks:
        event = kernel.step()
        assert event is not None
        events.append(event)
    observed = {event.task.name for event in events}
    assert host_read.name in observed and gpu_write.name in observed
    assert all(task.name in observed for task in links)
    read_event = next(event for event in events if event.task.task_id == host_read.task_id)
    assert read_event.task.metadata["physical_execution"]["logical_read_bytes"] == sum(
        item["byte_count"] for item in read_event.task.metadata["memory_accesses"]
        if item["operation"] == "read"
    )


def test_physical_dram_resolver_does_not_treat_valid_nand_as_dram():
    scenario = build_reference_scenario()
    host = scenario.hardware.get_component("hostmem0")
    component = replace(host, metadata={
        **host.metadata,
        "physical_memory_config": asdict(NandConfig(kind=MemoryKind.SSD)),
    })
    assert _physical_dram_config(scenario, component) is None


def test_transfer_rank_selects_gpu_backend_and_no_rank_keeps_controller_endpoint(monkeypatch):
    from types import SimpleNamespace
    from heterollm_sim.parallel import LogicalRank
    from heterollm_sim import planner

    scenario = _scenario(_ddr_config())
    ranks = (
        LogicalRank(0, "gpu0", 0, 0, 0, memory_component_id="hbm0"),
        LogicalRank(1, "gpu0", 1, 0, 0, memory_component_id="hbm1"),
    )
    monkeypatch.setattr(planner, "_parallel_plan", lambda _scenario: SimpleNamespace(ranks=ranks))
    def compile_transfer(request_id, transfer_metadata):
        builder = _TaskBuilder(RequestSpec(request_id, 0.0, 1, 1))
        with _compilation_scope(scenario):
            router = planner._active_compilation_context(scenario).router()
            _add_transfer_tasks(
                builder, router, "cpu0", "gpu0", 64, (), name=request_id,
                routing_policy="lowest_latency", metadata=transfer_metadata,
            )
        return builder.tasks

    unresolved = compile_transfer("multi-backend-no-rank", {})
    assert any(task.metadata.get("gpu_memory_endpoint_resolution")
               == "logical_controller_multiple_backends" for task in unresolved)
    assert any(task.metadata.get("target_component") == "gpu0" for task in unresolved)

    selected = compile_transfer("multi-backend-rank-one", {"rank": 1})
    assert any(task.metadata.get("target_component") == "hbm1" for task in selected)
