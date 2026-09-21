"""R4 hardware-resource closure checks over the frozen R0 source.

This is a structural gate only.  It deliberately does not call the native
runtime, the prediction/score commands, or a microbenchmark.  A real R0 cell
is read from the frozen manifest and lowered with synthetic request records so
that the component/profile/resource/link/queue boundaries can be inspected
without making a performance claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[6]
LOOP = ROOT / "artifacts" / "development" / "native_long_grid_135_20260915" / "optimization_loop"
R0_ON = LOOP / "round_000" / "on"
R0_SOURCE = R0_ON / "source" / "src"
FREEZE = R0_ON / "freeze.json"

# Import identity is part of this gate: R4 must inspect the frozen R0 source,
# rather than the mutable checkout or a candidate source tree.
assert R0_SOURCE.is_dir(), "R0 frozen source directory is missing"
sys.path.insert(0, str(R0_SOURCE))

from heterollm_sim.communication import TopologyRouter, declared_resource_owners  # noqa: E402
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec  # noqa: E402
from heterollm_sim.event_kernel import UnifiedEventKernel  # noqa: E402
from heterollm_sim.planner import compile_scenario  # noqa: E402
from heterollm_sim.topology import validate_topology  # noqa: E402


_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "r0_native_llama_compare",
    R0_ON / "source" / "tools" / "native_llama_compare.py",
)
assert _COMPARE_SPEC and _COMPARE_SPEC.loader
_COMPARE_MODULE = importlib.util.module_from_spec(_COMPARE_SPEC)
sys.modules[_COMPARE_SPEC.name] = _COMPARE_MODULE
_COMPARE_SPEC.loader.exec_module(_COMPARE_MODULE)
build_matching_scenario = _COMPARE_MODULE.build_matching_scenario
HARDWARE_SNAPSHOT = LOOP.parent / "hardware.json"


@dataclass(frozen=True)
class ClosureEvidence:
    cell_id: str
    prompt_tokens: int
    output_tokens: int
    parallel: int
    component_ids: Tuple[str, ...]
    profile_resource_ids: Tuple[str, ...]
    demand_resource_ids: Tuple[str, ...]
    capacity_resource_ids: Tuple[str, ...]
    missing_capacity_ids: Tuple[str, ...]
    owner_map: Mapping[str, str]
    path_resource_ids: Tuple[str, ...]


def _pick_r0_multirequest_cell() -> Mapping[str, Any]:
    """Pick a deterministic, existing R0 multi-request input.

    The choice is data-driven: maximize parallelism, then minimize output and
    prompt size so this diagnostic remains bounded.  No tuning parameter is
    invented by the test.
    """

    with FREEZE.open("r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    candidates = []
    for cell in freeze["cells"]:
        static = cell.get("static_inputs") or {}
        config = static.get("config") or {}
        parallel = int(config.get("parallel", 0))
        if parallel <= 1:
            continue
        candidates.append(
            {
                "cell_id": str(cell["cell_id"]),
                "parallel": parallel,
                "prompt_tokens": int(config["expected_prompt_tokens"]),
                "output_tokens": int(config["output"]),
                "ctx": int(config["ctx"]),
                "batch": int(config["batch"]),
                "ubatch": int(config["ubatch"]),
                "threads": int(config["threads"]),
                "gpu_layers": int(config["gpu_layers"]),
            }
        )
    if not candidates:
        raise AssertionError("R0 freeze has no multi-request cell")
    return min(
        candidates,
        key=lambda item: (
            -item["parallel"],
            item["output_tokens"],
            item["prompt_tokens"],
            item["cell_id"],
        ),
    )


def _profile_resource_ids(profile: object) -> Tuple[str, ...]:
    """Collect resource IDs exposed by typed cost profiles."""

    found = []
    for name in (
        "resource_id",
        "scalar_resource_id",
        "special_function_resource_id",
        "launch_resource_id",
        "array_resource_id",
        "load_resource_id",
        "activation_resource_id",
        "noc_resource_id",
        "accumulator_resource_id",
        "peripheral_resource_id",
    ):
        value = getattr(profile, name, None)
        if isinstance(value, str) and value:
            found.append(value)
    tensor_core = getattr(profile, "tensor_core", None)
    value = getattr(tensor_core, "resource_id", None)
    if isinstance(value, str) and value:
        found.append(value)
    hierarchy = getattr(profile, "cache_hierarchy", None)
    for level in getattr(hierarchy, "levels", ()):
        value = getattr(level, "resource_id", None)
        if isinstance(value, str) and value:
            found.append(value)
    pipeline = getattr(profile, "pipeline", None)
    value = getattr(pipeline, "resource_id", None)
    if isinstance(value, str) and value:
        found.append(value)
    return tuple(dict.fromkeys(found))


@lru_cache(maxsize=1)
def _prepared() -> Tuple[object, object, ClosureEvidence]:
    cell = _pick_r0_multirequest_cell()
    with HARDWARE_SNAPSHOT.open("r", encoding="utf-8") as handle:
        hardware_snapshot = json.load(handle)
    # Use the same frozen R0 scenario constructor as the prediction path.  No
    # native runtime, prediction, score, or answer-derived parameter is used.
    scenario = build_matching_scenario(
        cell["prompt_tokens"],
        cell["output_tokens"],
        ctx=cell["ctx"],
        parallel=cell["parallel"],
        batch=cell["batch"],
        ubatch=cell["ubatch"],
        threads=cell["threads"],
        gpu_layers=cell["gpu_layers"],
        hardware_snapshot=hardware_snapshot,
    )
    schedule = compile_scenario(scenario)

    component_ids = tuple(component.component_id for component in scenario.hardware.components)
    profile_resource_ids = tuple(
        sorted(
            {
                resource_id
                for component in scenario.hardware.components
                for resource_id in _profile_resource_ids(
                    scenario.resolve_component_profile(component.component_id)
                )
            }
        )
    )
    demand_resource_ids = tuple(
        sorted({demand.resource_id for task in schedule.tasks for demand in task.demands})
    )
    capacity_resource_ids = tuple(sorted(schedule.resource_capacities))
    missing_capacity_ids = tuple(
        sorted(set(demand_resource_ids) - set(capacity_resource_ids))
    )
    router = TopologyRouter(scenario.hardware)
    # These routes are the direction-sensitive paths required by host memory
    # access and PCIe DMA.  The test records their resource identities even
    # when a lowered schedule does not request both directions.
    paths = tuple(
        hop.resource_id
        for source, target in (
            ("cpu0", "gpu0"),
            ("gpu0", "cpu0"),
            ("cpu0", "hostmem0"),
            ("hostmem0", "cpu0"),
        )
        for hop in router.route(source, target, 64)
    )
    evidence = ClosureEvidence(
        cell_id=cell["cell_id"],
        prompt_tokens=cell["prompt_tokens"],
        output_tokens=cell["output_tokens"],
        parallel=cell["parallel"],
        component_ids=component_ids,
        profile_resource_ids=profile_resource_ids,
        demand_resource_ids=demand_resource_ids,
        capacity_resource_ids=capacity_resource_ids,
        missing_capacity_ids=missing_capacity_ids,
        owner_map=dict(schedule.resource_owners),
        path_resource_ids=paths,
    )
    return scenario, schedule, evidence


def test_r0_import_components_profiles_and_paths_are_real() -> None:
    """Prove the frozen source identity and static hardware graph closure."""

    import heterollm_sim.reference as reference_module

    assert Path(reference_module.__file__).resolve().is_relative_to(R0_SOURCE)
    scenario, _schedule, evidence = _prepared()
    report = validate_topology(scenario.hardware)
    assert report.is_valid, report.format_en()

    components = scenario.hardware.component_map()
    assert set(evidence.component_ids) == set(components)
    for component in scenario.hardware.components:
        profile = scenario.resolve_component_profile(component.component_id)
        assert profile is not None
        assert component.cost_profile_id in scenario.component_profiles[
            component.kind.replace("digital_sram_cim", "cim")
        ] or component.kind == "digital_sram_cim"
        assert component.capacity_bytes > 0

    # The R0 RTX5080-local scenario intentionally exposes one active HBM
    # endpoint; the frozen constructor must not be confused with the generic
    # eight-HBM reference package.
    hbm = [item for item in scenario.hardware.components if item.kind == "hbm"]
    hbm_links = [item for item in scenario.hardware.links if item.protocol == "HBM"]
    assert len(hbm) == len(hbm_links) == 1
    assert {item.target_component for item in hbm_links} == {item.component_id for item in hbm}

    # Host memory and PCIe paths are direction-aware and preserve the declared
    # link identity, lane count, and positive bandwidth in both directions.
    for source, target in (("cpu0", "gpu0"), ("gpu0", "cpu0"), ("cpu0", "hostmem0"), ("hostmem0", "cpu0")):
        hops = TopologyRouter(scenario.hardware).route(source, target, 64)
        assert len(hops) == 1
        hop = hops[0]
        assert hop.resource_id in evidence.path_resource_ids
        assert hop.bandwidth_gbps > 0
        assert hop.latency_ns >= 0


def test_r0_multirequest_resource_demands_expose_capacity_and_owner_facts() -> None:
    """Record explicit capacity/owner facts without inferring a root cause."""

    scenario, schedule, evidence = _prepared()
    assert evidence.parallel > 1
    assert len(schedule.tasks) > evidence.parallel
    assert all(task.request_id in {request.request_id for request in scenario.workload.requests} for task in schedule.tasks)
    task_ids = {task.task_id for task in schedule.tasks}
    assert all(dependency in task_ids for task in schedule.tasks for dependency in task.dependencies)
    assert all(
        demand.service_ns >= 0
        for task in schedule.tasks
        for demand in task.demands
    )

    # Capacity is present for controller/queue resources, but the R0 schedule
    # leaves GPU execution, HBM fabric and the PCIe route on implicit default
    # lanes.  Keep that fact visible as an effective-default representation
    # detail instead of treating it as hardware evidence.
    assert evidence.missing_capacity_ids
    assert set(evidence.missing_capacity_ids) <= {
        "gpu0.frontend",
        "gpu0.l1_shared",
        "gpu0.l2",
        "gpu0.scalar",
        "gpu0.sfu",
        "gpu0.tensor_core",
        "hbm0.hbm_fabric",
        "link.cpu-gpu-pcie.cpu0->gpu0",
        "link.cpu-gpu-pcie.gpu0->cpu0",
    }
    assert schedule.resource_owners == {}
    assert declared_resource_owners(scenario.hardware) == {}

    # Profile resources are part of the demand graph.  The active HBM endpoint
    # resolves through a profile resource, while the physical owner and
    # effective concurrency remain separate evidence questions.
    assert "gpu0.hbm_fabric" in evidence.profile_resource_ids
    # Rank/resource lowering renames the profile resource to the active HBM
    # endpoint.  The name change is expected; demand conservation is checked
    # against the resolved endpoint rather than the profile namespace.
    assert "hbm0.hbm_fabric" in evidence.demand_resource_ids
    hbm_profile_ids = {
        scenario.resolve_component_profile(component.component_id).resource_id
        for component in scenario.hardware.components
        if component.kind == "hbm"
    }
    assert hbm_profile_ids == {"gpu0.hbm_fabric"}


def test_r0_queue_ledger_accepts_directional_resource_identity() -> None:
    """Exercise queue accounting using only a synthetic structural service time."""

    _scenario, schedule, evidence = _prepared()
    resource_id = evidence.missing_capacity_ids[0]
    demand = ResourceDemand(resource_id=resource_id, service_ns=1.0, work_units=1.0)
    tasks = (
        TaskSpec(
            task_id="r4.structure.queue.0",
            request_id="r4-structure",
            name="queue_probe_0",
            category=TaskCategory.COMPUTE,
            demands=(demand,),
            metadata={"direction": "host_to_device"},
        ),
        TaskSpec(
            task_id="r4.structure.queue.1",
            request_id="r4-structure",
            name="queue_probe_1",
            category=TaskCategory.COMPUTE,
            demands=(demand,),
            metadata={"direction": "host_to_device"},
        ),
    )
    implicit = UnifiedEventKernel.from_closed_graph(tasks)
    kernel = UnifiedEventKernel.from_closed_graph(
        tasks,
        resource_capacities={resource_id: 1},
    )
    def drain(target):
        events = []
        while target.has_active_tasks:
            event = target.step()
            assert event is not None
            events.append(event)
        return events
    implicit_events = drain(implicit)
    events = drain(kernel)
    assert [event.task.task_id for event in events] == [
        "r4.structure.queue.0",
        "r4.structure.queue.1",
    ]
    assert events[0].queue_wait_ns == 0.0
    assert events[1].queue_wait_ns == 1.0
    assert kernel.resource_task_count[resource_id] == 2
    assert kernel.resource_busy_ns[resource_id] == 2.0
    assert [(item.start_ns, item.end_ns) for item in implicit_events] == [
        (item.start_ns, item.end_ns) for item in events
    ]


def test_host_memory_cpu_access_and_h2d_path_are_explicit_but_owner_unknown() -> None:
    """Trace one real lowering path without inventing a DRAM owner mapping."""

    _scenario, schedule, _evidence = _prepared()
    by_id = {task.task_id: task for task in schedule.tasks}
    cpu_memory = next(
        task
        for task in schedule.tasks
        if task.metadata.get("event_kind") == "host_cohort_pack"
        and "cpu0.memory" in {d.resource_id for d in task.demands}
        and task.metadata.get("payload_bytes", 0) > 0
    )
    cpu_index = next(
        task
        for task in schedule.tasks
        if task.metadata.get("event_kind") == "iommu_translation_batch"
        and cpu_memory.task_id in task.dependencies
    )
    dma = next(
        task
        for task in schedule.tasks
        if task.metadata.get("event_kind") == "dma_controller_batch"
        and cpu_index.task_id in task.dependencies
    )
    link = next(
        task
        for task in schedule.tasks
        if task.metadata.get("event_kind") == "host_cohort_h2d"
        and dma.task_id in task.dependencies
    )
    assert cpu_memory.metadata["target_component"] == "cpu0"
    memory_demand = next(demand for demand in cpu_memory.demands if demand.resource_id == "cpu0.memory")
    assert memory_demand.bytes_moved == 1216
    assert memory_demand.service_ns > 0
    assert dma.metadata["target_component"] == "gpu0"
    dma_demand = next(demand for demand in dma.demands if demand.resource_id == "cpu0.h2d_dma")
    assert dma_demand.bytes_moved == 0
    assert dma_demand.work_units == 1
    link_demand = next(
        demand
        for demand in link.demands
        if demand.resource_id == "link.cpu-gpu-pcie.cpu0->gpu0"
    )
    assert link_demand.bytes_moved == 608
    assert link_demand.service_ns > 0
    assert cpu_memory.metadata["payload_bytes"] == dma.metadata["payload_bytes"] == link.metadata["payload_bytes"]
    assert cpu_memory.task_id in cpu_index.dependencies
    assert cpu_index.task_id in dma.dependencies
    assert dma.task_id in link.dependencies
    # The path proves ordering and resource identities, but no physical owner
    # mapping connects cpu0.memory to DMA or PCIe; H4 remains evidence-limited.
    assert schedule.resource_owners == {}
