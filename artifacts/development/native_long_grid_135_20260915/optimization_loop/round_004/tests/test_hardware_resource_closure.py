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


def test_memory_endpoint_eligibility_and_fold_metadata_are_explicit() -> None:
    """Separate endpoint capability facts from coherent-DMA fold eligibility."""

    scenario, schedule, _evidence = _prepared()
    operator_transfer = next(
        task
        for task in schedule.tasks
        if task.metadata.get("event_kind") == "operator_input_transfer"
        and task.metadata.get("operator_id") == "layer-000.input_norm.reduce"
        and task.metadata.get("bytes") == 229376
    )
    assert operator_transfer.metadata["source_component"] == "cpu0"
    assert operator_transfer.metadata["target_component"] == "gpu0"
    payload_bytes = int(operator_transfer.metadata["bytes"])
    components = scenario.hardware.component_map()
    hostmem = components["hostmem0"]
    hbm = components["hbm0"]
    assert hostmem.is_active_memory and hbm.is_active_memory
    # The measured R0 constructor carries positive bandwidth on the physical
    # DDR/HBM ports, but the endpoint service fields consumed by
    # TopologyRouter remain undeclared.  This is a representation fact, not a
    # license to invent a read/write budget.
    assert any(port.bandwidth_gbps > 0 for port in hostmem.ports)
    assert any(port.bandwidth_gbps > 0 for port in hbm.ports)
    assert hostmem.read_bandwidth_gbps == 0.0
    assert hostmem.write_bandwidth_gbps == 0.0
    assert hbm.read_bandwidth_gbps == 0.0
    assert hbm.write_bandwidth_gbps == 0.0

    phases = TopologyRouter(scenario.hardware).transfer_phases(
        "hostmem0", "hbm0", payload_bytes, name="r4.endpoint_probe"
    )
    assert phases
    assert all(phase.metadata.get("event_kind") == "transfer" for phase in phases)
    assert all(
        not demand.resource_id.startswith("component.hostmem0.")
        and not demand.resource_id.startswith("component.hbm0.")
        for phase in phases
        for demand in phase.demands
    )
    # Existing link payload labels alone do not satisfy the router's explicit
    # transfer_execution/coherent_dma_span_id contract.  Do not infer a fold.
    assert all(
        phase.metadata.get("transfer_execution") != "coherent_dma"
        for phase in phases
    )


def test_operator_transfer_has_producer_link_and_consumer_memory_coverage() -> None:
    """Bind one real transfer object to its surrounding memory services."""

    _scenario, schedule, _evidence = _prepared()
    by_id = {task.task_id: task for task in schedule.tasks}
    transfer = next(
        task
        for task in schedule.tasks
        if task.metadata.get("event_kind") == "operator_input_transfer"
        and task.metadata.get("operator_id") == "layer-000.input_norm.reduce"
        and task.metadata.get("bytes") == 229376
    )
    assert len(transfer.dependencies) == 1
    embedding_complete = by_id[transfer.dependencies[0]]
    assert embedding_complete.metadata.get("event_kind") == "embedding_complete"
    assert len(embedding_complete.dependencies) == 1
    producer = by_id[embedding_complete.dependencies[0]]
    assert producer.metadata.get("event_kind") == "embedding"
    assert producer.metadata.get("output_bytes") == 229376
    assert producer.metadata.get("input_component") == "cpu0"
    assert producer.metadata.get("weight_source_kind") == "host_memory"
    assert producer.metadata.get("weight_source_transfer_emitted") is False
    assert producer.metadata.get("weight_backing_read_gate") == (
        "source_is_cpu_attached_host_memory"
    )
    producer_memory = next(
        demand for demand in producer.demands if demand.resource_id == "cpu0.memory"
    )
    assert producer_memory.bytes_moved > 229376
    assert producer_memory.service_ns > 0

    consumer = next(
        task
        for task in schedule.tasks
        if transfer.task_id in task.dependencies
        and task.metadata.get("event_kind") == "input_norm_reduce"
    )
    assert len(consumer.dependencies) == 1
    # The kernel launch is followed by the actual reduction service task.
    reduction = next(
        task
        for task in schedule.tasks
        if consumer.task_id in task.dependencies
        and task.metadata.get("event_kind") == "input_norm_reduce"
        and any(d.resource_id == "hbm0.hbm_fabric" for d in task.demands)
    )
    hbm_demand = next(
        demand for demand in reduction.demands if demand.resource_id == "hbm0.hbm_fabric"
    )
    assert hbm_demand.bytes_moved == 229632
    assert hbm_demand.service_ns > 0
    link_demand = next(
        demand
        for demand in transfer.demands
        if demand.resource_id == "link.cpu-gpu-pcie.cpu0->gpu0"
    )
    assert link_demand.bytes_moved == 229376
    assert link_demand.service_ns > 0


def test_operator_transfer_access_roles_and_buffer_identity_boundary() -> None:
    """Account for the selected transfer's roles without inventing endpoints.

    The formal graph identifies the activation tensor and its producer, while
    the lowering currently carries only component/rank-value state across the
    CPU-to-GPU boundary.  Keep those facts separate: producer read/write,
    PCIe bulk, and consumer read/write are observable services.  The checked
    placement and task metadata do not explicitly provide a scheduler copy
    buffer identity; whether the current component/rank-value abstraction
    fully expresses the required copy semantics remains an open question.
    """

    scenario, schedule, _evidence = _prepared()
    by_id = {task.task_id: task for task in schedule.tasks}
    transfer = next(
        task
        for task in schedule.tasks
        if task.request_id == "request-0000"
        and task.metadata.get("event_kind") == "operator_input_transfer"
        and task.metadata.get("operator_id") == "layer-000.input_norm.reduce"
        and task.metadata.get("bytes") == 229376
        and ".prefill." in task.task_id
    )
    embedding_complete = by_id[transfer.dependencies[0]]
    producer = by_id[embedding_complete.dependencies[0]]
    launch = next(
        task
        for task in schedule.tasks
        if task.request_id == "request-0000"
        and task.metadata.get("event_kind") == "input_norm_reduce"
        and task.metadata.get("operator_id") == "layer-000.input_norm.reduce"
        and task.metadata.get("phase") == "kernel_launch"
        and transfer.task_id in task.dependencies
    )
    reduction = next(
        task
        for task in schedule.tasks
        if task.request_id == "request-0000"
        and task.metadata.get("event_kind") == "input_norm_reduce"
        and task.metadata.get("operator_id") == "layer-000.input_norm.reduce"
        and task.metadata.get("phase") == "gpu_reduction"
        and launch.task_id in task.dependencies
    )

    # Formal graph identity: embedding.output is produced by embedding and is
    # consumed by the first decoder block.  The placement contract maps the
    # operators and persistent weights, but has no activation-tensor residency.
    graph = scenario.model.graph
    embedding_op = next(op for op in graph.operators if op.operator_id == "embedding")
    embedding_output = next(
        tensor for tensor in graph.tensors if tensor.tensor_id == "embedding.output"
    )
    assert embedding_op.output_tensor_ids == ("embedding.output",)
    assert embedding_output.producer_operator_id == "embedding"
    assert embedding_output.consumer_operator_ids
    assert scenario.placement.op_to_component["embedding"] == "cpu0"
    assert scenario.placement.op_to_component["layer-000.input_norm.reduce"] == "gpu0"
    assert "embedding.output" not in scenario.placement.tensor_to_component
    assert scenario.placement.parallel.rank_mapping[0].memory_component_id == "hbm0"

    # The producer's typed memory profile includes both the embedding weight
    # read and activation write.  The transfer contributes only PCIe bulk, and
    # the consumer's HBM demand includes its input read and result write.
    assert producer.metadata["output_tensor_id"] == "embedding.output"
    producer_cost = producer.metadata["cost_model"]
    assert producer_cost["read_bytes"] == 272269312
    assert producer_cost["write_bytes"] == 229376
    producer_memory = next(
        demand for demand in producer.demands if demand.resource_id == "cpu0.memory"
    )
    assert producer_memory.bytes_moved == 272498688
    transfer_link = next(
        demand
        for demand in transfer.demands
        if demand.resource_id == "link.cpu-gpu-pcie.cpu0->gpu0"
    )
    assert transfer_link.bytes_moved == 229376
    reduction_cost = reduction.metadata["cost_model"]
    assert reduction_cost["read_bytes"] == 229376
    assert reduction_cost["write_bytes"] == 256
    reduction_memory = next(
        demand for demand in reduction.demands if demand.resource_id == "hbm0.hbm_fabric"
    )
    assert reduction_memory.bytes_moved == 229632

    # These are representation boundaries, not proof of a defect: ordinary
    # operator transfers carry no output tensor ID or copy-buffer ID, and the
    # reduction task has no formal output tensor ID.  The current fixture does
    # not establish how component/rank-value state constrains the scheduler's
    # source read, destination write, or physical allocation reuse.
    assert transfer.metadata.get("output_tensor_id") is None
    assert transfer.metadata.get("copy_buffer_id") is None
    assert reduction.metadata.get("output_tensor_id") is None



def _r0_static_cell_for_copy_probe(inputs):
    """Reuse the frozen worker's exact preparation prefix, never its run body."""
    import ast
    import copy
    from heterollm_sim.gguf_parity import read_gguf_metadata_cache

    source = R0_ON / "source" / "tools" / "predict_stable_native_dataset.py"
    spec = importlib.util.spec_from_file_location("r4_r0_static_worker", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.verify_import_roots()
    sidecar = inputs["gguf_metadata_sidecar_ref"]
    model_path = Path(inputs["prediction_model_ref"]["path"])
    assert Path(sidecar["path"]).is_file()
    # No fallback to a GGUF payload scan: seed the worker's normal model cache.
    gguf = read_gguf_metadata_cache(model_path, sidecar["path"], strict=False)
    cache = {module.grid.model_metadata_cache_key(model_path): (
        gguf, module.grid.build_model_from_gguf(gguf))}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    prefix = copy.deepcopy(next(node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name == "predict_cell"))
    cut = next(i for i, node in enumerate(prefix.body)
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
               and ast.unparse(node.value.func) == "grid.reporting.run_scenario")
    prefix.name = "_prepare_only"
    prefix.body = prefix.body[:cut] + [ast.Return(value=ast.Tuple(elts=[
        ast.Name(id="scenario", ctx=ast.Load()),
        ast.Name(id="placement_refresh", ctx=ast.Load())], ctx=ast.Load()))]
    calls = {ast.unparse(node.func) for node in ast.walk(prefix) if isinstance(node, ast.Call)}
    assert "replan_final_static_scenario" in calls
    assert "apply_tensor_storage_static_contract" in calls
    assert not any(name.endswith(("run_scenario", "simulate_online", "predict_cell")) for name in calls)
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])),
                 str(source), "exec"), namespace)
    scenario, refresh = namespace["_prepare_only"](inputs, model_cache=cache)
    module.verify_import_roots()
    assert refresh["normal_validation_passed"] and refresh["mapping_stale"] is False
    audit = scenario.workload.metadata["llama_cpp_tensor_storage"]
    assert audit["status"] == "enabled" and audit["qualified"] is True
    assert scenario.workload.metadata["llama_cpp_f32_hidden_storage"] is True
    print("R4_I8_PREPARED", json.dumps({"worker": str(source), "sidecar": sidecar,
        "gguf_sha256": gguf.sha256, "tensor_storage_status": audit["status"],
        "mapping_stale": refresh["mapping_stale"]}, sort_keys=True))
    return scenario


def test_source_qualified_q0_decode_generations_keep_current_embedding_copy():
    """Two retained Q0 cohorts; dependency-only replay, never a latency claim.

    Invocation caching is inapplicable to this stateless Qwen2 path.  Keep its
    real exclusion gate, compare the second same-context lowering with a fresh
    lowering, and execute only each producer/copy/first-consumer closure.
    """
    from unittest.mock import patch
    from heterollm_sim import planner
    from heterollm_sim.serving import BatchCohort, BatchItem

    assert Path(planner.__file__).resolve().is_relative_to(R0_SOURCE)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    cell_id = "qwen25_p128_o32_c4__fixed_runtime"
    inputs = next(cell["static_inputs"] for cell in freeze["cells"] if cell["cell_id"] == cell_id)
    prediction_path = R0_ON / "predictions" / (cell_id + ".prediction.json")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    retained = prediction["batch_schedule"]
    assert retained["details_truncated"] is False
    selected = retained["batches"][38:40]
    assert [row["batch_index"] for row in selected] == [38, 39]
    assert [row["items"][0]["context_tokens"] for row in selected] == [157, 158]
    assert [row["items"][0]["completion_cursor"] for row in selected] == [31, 32]
    for row in selected:
        assert row["kind"] == "decode" and row["token_count"] == 1
        assert row["request_ids"] == ["request-0003"]
        assert row["items_truncated"] is False and len(row["items"]) == 1
        assert row["cost_metadata"]["physical_batch_rows"] == 1
    print("R4_I8_Q0_GEOMETRY", json.dumps({"path": str(prediction_path), "cohorts": [
        {key: row[key] for key in ("batch_index", "cohort_id", "kind", "items", "metadata")}
        for row in selected]}, sort_keys=True))
    scenario = _r0_static_cell_for_copy_probe(inputs)
    assert scenario.workload.mtp is None
    # The Q0 diagnostic retains five item fields.  Ordinary non-MTP decode's
    # omitted append/materialized/logit counts are one in frozen serving.py's
    # BatchItem constructor (10453); this is not a fabricated task graph.
    cohorts = tuple(BatchCohort(row["cohort_id"], row["kind"], 0.0,
        tuple(BatchItem(**item, proposed_tokens=1, expected_accepted_tokens=1.0,
                        kv_append_tokens=1, kv_materialized_tokens=1, logit_tokens=1)
              for item in row["items"]), metadata=row["metadata"]) for row in selected)
    groups = [planner._serving_invocation_groups(scenario, cohort) for cohort in cohorts]
    assert all(len(items) == 1 and items[0].token_batch == 1 for items in groups)
    assert all(items[0].batching_semantics == "stateless_scheduler_batch" for items in groups)
    observations = []
    original_binding = planner._serving_invocation_segment_binding

    def record_binding(*args, **kwargs):
        result = original_binding(*args, **kwargs)
        observations.append(result is None)
        return result

    context = planner.CompilationContext(scenario, eager_full_attention_segments=False,
                                         compiled_serving_invocation_segments=True)
    with patch.object(planner, "_serving_invocation_segment_binding", side_effect=record_binding):
        with planner._compilation_scope(scenario, context):
            schedules = [planner.compile_serving_cohort_schedule(scenario, cohort) for cohort in cohorts]
    assert observations == [True, True]
    fresh_context = planner.CompilationContext(scenario, eager_full_attention_segments=False,
                                               compiled_serving_invocation_segments=True)
    with planner._compilation_scope(scenario, fresh_context):
        fresh = planner.compile_serving_cohort_schedule(scenario, cohorts[1])
    # Equality includes real demands, dependencies, earliest-start and metadata.
    assert schedules[1].tasks == fresh.tasks
    assert schedules[1].resource_capacities == fresh.resource_capacities
    assert schedules[1].resource_owners == fresh.resource_owners
    print("R4_I8_CACHE", json.dumps({"binding_none": observations,
        "path": "stateless_scheduler_batch", "invocation_cache": "not_applicable",
        "same_context_target_equals_fresh_tasks": True,
        "cohort_task_counts": [len(schedule.tasks) for schedule in schedules]}, sort_keys=True))
    identities = []
    for cohort, schedule in zip(cohorts, schedules):
        by_id = {task.task_id: task for task in schedule.tasks}

        def ancestors(ident):
            seen = set()
            pending = list(by_id[ident].dependencies)
            while pending:
                current = pending.pop()
                if current not in seen:
                    seen.add(current)
                    pending.extend(by_id[current].dependencies)
            return seen

        def one(**metadata):
            matches = [task for task in schedule.tasks
                       if all(task.metadata.get(key) == value for key, value in metadata.items())]
            assert len(matches) == 1, (metadata, [task.task_id for task in matches])
            return matches[0]

        producer = one(event_kind="embedding", phase="cpu_memory")
        produced = one(event_kind="embedding_complete")
        copy_task = one(event_kind="operator_input_transfer", operator_id="layer-000.input_norm.reduce")
        consumer = one(event_kind="input_norm_reduce", operator_id="layer-000.input_norm.reduce", phase="gpu_reduction")
        assert producer.metadata["output_tensor_id"] == "embedding.output"
        assert producer.metadata["native_get_rows_storage"]["qualified"] is True
        assert copy_task.metadata["source_component"] == "cpu0"
        assert copy_task.metadata["target_component"] == "gpu0"
        assert copy_task.metadata["bytes"] == producer.metadata["output_bytes"] == 896 * 4
        assert producer.task_id in ancestors(produced.task_id)
        assert produced.task_id in ancestors(copy_task.task_id)
        assert copy_task.task_id in ancestors(consumer.task_id)
        assert producer.task_id in ancestors(consumer.task_id)
        identities.append((producer.task_id, produced.task_id, copy_task.task_id, consumer.task_id))
        closure = ancestors(consumer.task_id) | {consumer.task_id}
        sliced = tuple(task for task in schedule.tasks if task.task_id in closure)
        assert all(dependency in closure for task in sliced for dependency in task.dependencies)
        assert len(sliced) < 100, "bounded probe must not execute a full cohort"
        kernel = UnifiedEventKernel.from_closed_graph(sliced,
            resource_capacities=schedule.resource_capacities, resource_owners=schedule.resource_owners)
        events = {}
        while kernel.has_active_tasks:
            event = kernel.step()
            assert event is not None
            events[event.task.task_id] = event
        assert len(events) == len(sliced)
        for event in events.values():
            assert all(event.start_ns >= events[dependency].end_ns
                       for dependency in event.task.dependencies)
            assert all(previous["end_ns"] <= event.start_ns
                       for previous in event.resource_predecessors.values())
        assert events[copy_task.task_id].start_ns >= events[produced.task_id].end_ns
        assert events[consumer.task_id].start_ns >= events[copy_task.task_id].end_ns
        print("R4_I8_GENERATION", json.dumps({"cohort_id": cohort.cohort_id,
            "context_tokens": cohort.items[0].context_tokens,
            "completion_cursor": cohort.items[0].completion_cursor,
            "current_value_identity": identities[-1], "copy_bytes": copy_task.metadata["bytes"],
            "source_storage": producer.metadata["native_get_rows_storage"],
            "executed_slice_tasks": len(sliced), "time_scope": "analytical_dependency_slice_not_native_latency",
            "events": [{"task_id": event.task.task_id, "dependencies": event.task.dependencies,
                "start_ns": event.start_ns, "end_ns": event.end_ns,
                "dependency_ready_ns": event.dependency_ready_ns,
                "resource_predecessors": event.resource_predecessors,
                "demands": [vars(demand) for demand in event.demands]}
                for event in events.values()]}, sort_keys=True, default=str))
    # Distinct current producers/copies/consumers are required.  A causal
    # predecessor is allowed; do not prohibit earlier-generation ancestors.
    assert all(left != right for left, right in zip(*identities))
