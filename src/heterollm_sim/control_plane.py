"""V4 dynamic placement/bootstrap integrated with the hardware event DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from .communication import TopologyRouter
from .config import ScenarioConfig, normalize_cost_profile_kind
from .control_plane_planner import (
    PlacementDecision,
    PlacementPolicy,
    plan_runtime_placement,
)
from .cost_models import CPUProfile, HostMemoryProfile, _dma_setup_service
from .event_kernel import UnifiedEventKernel
from .ir import normalize_component_kind
from .runtime import ControlPlaneRuntime, RuntimeRunResult
from .runtime_ir import RuntimeAction
from .runtime_state import RuntimeState


@dataclass(frozen=True)
class ControlPlaneBootstrap:
    """Mapped scenario plus the realized control-plane DAG and live kernel."""

    scenario: ScenarioConfig
    decision: PlacementDecision
    result: RuntimeRunResult
    kernel: UnifiedEventKernel

    @property
    def makespan_ns(self) -> float:
        return self.result.makespan_ns

    @property
    def placement_elapsed_s(self) -> float:
        """Wall-clock solver evidence; never added directly to simulated time."""

        return float(self.decision.elapsed_s)


@dataclass(frozen=True)
class _MaterializedWorkloadCounts:
    """Aggregate CPU work derived from one realized placement decision."""

    requirement_count: int
    physical_tensor_count: int
    executable_operator_count: int
    command_unit_count: int
    command_unit_count_by_gpu: Mapping[str, int]


def _materialized_workload_counts(
    scenario: ScenarioConfig,
    decision: PlacementDecision,
) -> _MaterializedWorkloadCounts:
    """Count realized placement, allocation, scheduling, and command units.

    Positive tensor allocation identities are deduplicated so logical aliases
    never allocate twice.  Rank-local shards and explicit physical replicas
    remain separate allocations, while requirements without a ``mapping_key``
    (for example physical-only GGUF artifacts) do not become executable
    commands.  Zero-byte state placement policies are allocated later from
    admitted workload bytes and therefore are not bootstrap allocations.
    """

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    if not isinstance(decision, PlacementDecision):
        raise TypeError("decision must be a PlacementDecision")
    gpu_component_ids = {
        component.component_id
        for component in scenario.hardware.components
        if normalize_component_kind(component.kind) == "gpu"
    }
    gpu_component_ids.add(
        scenario.host_orchestration_profile.gpu_component_id
    )
    physical_allocations = set()
    executable_operators = set()
    command_units = set()
    command_units_by_gpu: Dict[str, set[Tuple[object, ...]]] = {}
    for item in decision.decisions:
        tensor_id = item.tensor_id
        if tensor_id:
            if item.rank_tensor_shards:
                for shard in item.rank_tensor_shards:
                    if shard.physical_bytes <= 0:
                        continue
                    physical_allocations.add(
                        (
                            tensor_id,
                            "rank",
                            shard.rank,
                            shard.storage_component_id,
                        )
                    )
            elif (
                (item.tensor_bytes > 0 or item.padded_weight_bytes > 0)
                and item.physical_tensor_component_ids
            ):
                for component_id in item.physical_tensor_component_ids:
                    physical_allocations.add(
                        (tensor_id, "component", component_id)
                    )
            elif item.tensor_bytes > 0 or item.padded_weight_bytes > 0:
                physical_allocations.add(
                    (
                        tensor_id,
                        "component",
                        item.tensor_component_id or item.component_id,
                    )
                )

        mapping_key = item.mapping_key
        if not mapping_key:
            continue
        executable_operators.add(mapping_key)
        if item.rank_execution_targets:
            for target in item.rank_execution_targets:
                component_id = target.component_id
                if component_id not in gpu_component_ids:
                    continue
                identity = (mapping_key, "rank", target.rank_id, component_id)
                command_units.add(identity)
                command_units_by_gpu.setdefault(component_id, set()).add(identity)
            continue
        execution_component_ids = (
            tuple(item.execution_component_ids) or (item.component_id,)
        )
        for component_id in execution_component_ids:
            if component_id not in gpu_component_ids:
                continue
            identity = (mapping_key, "component", component_id)
            command_units.add(identity)
            command_units_by_gpu.setdefault(component_id, set()).add(identity)

    return _MaterializedWorkloadCounts(
        requirement_count=len(decision.decisions),
        physical_tensor_count=len(physical_allocations),
        executable_operator_count=len(executable_operators),
        command_unit_count=len(command_units),
        command_unit_count_by_gpu={
            component_id: len(units)
            for component_id, units in sorted(command_units_by_gpu.items())
        },
    )


def _gpu_weight_transfer_bytes(
    scenario: ScenarioConfig,
    decision: PlacementDecision,
) -> int:
    """Return deduplicated physical weight bytes whose storage is GPU memory."""

    gpu_memory_ids = {
        rank.memory_component_id
        for rank in scenario.placement.parallel.rank_mapping
        if rank.component_id
        in {
            component.component_id
            for component in scenario.hardware.components
            if normalize_component_kind(component.kind) == "gpu"
        }
        and rank.memory_component_id
    }
    physical_bytes: Dict[Tuple[object, ...], int] = {}
    for item in decision.decisions:
        if not item.tensor_id or item.tensor_bytes <= 0:
            continue
        if item.rank_tensor_shards:
            for shard in item.rank_tensor_shards:
                if (
                    shard.storage_component_id not in gpu_memory_ids
                    or shard.physical_bytes <= 0
                ):
                    continue
                physical_bytes[
                    (
                        item.tensor_id,
                        "rank",
                        shard.rank,
                        shard.storage_component_id,
                    )
                ] = int(shard.physical_bytes)
            continue
        component_ids = (
            tuple(item.physical_tensor_component_ids)
            or (item.tensor_component_id or item.component_id,)
        )
        for component_id in component_ids:
            if component_id not in gpu_memory_ids:
                continue
            physical_bytes[(item.tensor_id, "component", component_id)] = int(
                item.tensor_bytes
            )
    return sum(physical_bytes.values())


def _cpu_profile(scenario: ScenarioConfig) -> CPUProfile:
    profile = scenario.resolve_component_profile(
        scenario.host_orchestration_profile.cpu_component_id,
        CPUProfile,
    )
    if not isinstance(profile, CPUProfile):  # pragma: no cover - typed resolver
        raise TypeError("orchestration CPU must resolve to a CPUProfile")
    return profile


def _execution_resource_capacities(
    scenario: ScenarioConfig,
) -> Dict[str, int]:
    """Declare physical execution lanes shared by bootstrap and serving.

    Throughput and storage-accounting fields are deliberately absent.  They
    already shape analytical service time or allocation admission and are not
    independent event-kernel execution lanes.
    """

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    orchestration = scenario.host_orchestration_profile
    controller = scenario.runtime_profile
    cpu = _cpu_profile(scenario)
    cpu_id = orchestration.cpu_component_id
    capacities: Dict[str, int] = {}

    def declare(resource_id: str, capacity: int) -> None:
        existing = capacities.get(resource_id)
        if existing is not None and existing != capacity:
            raise ValueError(
                "conflicting execution resource capacity for {}: {} != {}"
                .format(resource_id, existing, capacity)
            )
        capacities[resource_id] = capacity

    declare(cpu.pipeline.resource_id, 1)
    if orchestration.scheduler_resource_id == cpu.pipeline.resource_id:
        raise ValueError(
            "host control and aggregate CPU pipeline resource ids must be distinct"
        )
    declare(
        orchestration.scheduler_resource_id,
        cpu.pipeline.core_count,
    )
    for level in cpu.cache_hierarchy.levels:
        declare(level.resource_id, 1)
    for resource_id in (
        orchestration.pack_resource_id,
        orchestration.submission_resource_id,
        "{}.control_cache".format(cpu_id),
        "host.page_cache",
        "nvme",
        "{}.iommu".format(cpu_id),
        "interrupt",
    ):
        declare(resource_id, 1)
    declare(
        orchestration.dma_resource_id,
        max(1, controller.pcie_dma_iommu.dma_engine_count),
    )

    # CPU cost lowering selects only host-memory profiles with a valid route
    # in both directions.  Keep capacity discovery on the same reachability
    # contract so disconnected authored profiles cannot become runtime lanes.
    router = TopologyRouter(scenario.hardware)
    for component in sorted(
        scenario.hardware.components,
        key=lambda item: item.component_id,
    ):
        if (
            normalize_cost_profile_kind(component.normalized_kind)
            != "host_memory"
        ):
            continue
        try:
            router.route(cpu_id, component.component_id, 1)
            router.route(component.component_id, cpu_id, 1)
            memory = scenario.resolve_component_profile(
                component,
                HostMemoryProfile,
            )
        except (KeyError, TypeError, ValueError):
            continue
        declare(memory.resource_id, 1)

    for gpu_id in sorted(controller.gpu_controllers):
        for suffix in (
            "command_processor",
            "mmu_tlb",
            "l2_controller",
            "vram_controller",
        ):
            declare("{}.{}".format(gpu_id, suffix), 1)
    return capacities


def _declared_capacity_bytes(scenario: ScenarioConfig) -> int:
    return sum(
        max(0, int(component.capacity_bytes))
        for component in scenario.hardware.components
        if component.is_storage or component.is_active_memory
    )


def _declared_weight_load_bytes(scenario: ScenarioConfig) -> int:
    """Return physical artifact bytes for CPU weight-load orchestration."""

    declared = max(0, int(scenario.model.total_declared_weight_bytes))
    metadata = scenario.model.metadata
    if not isinstance(metadata, Mapping):
        return declared
    physical_totals = set()
    for key in ("gguf", "runtime_cost_contract"):
        contract = metadata.get(key)
        if contract is None:
            continue
        if not isinstance(contract, Mapping):
            raise ValueError("model.metadata.{} must be a mapping".format(key))
        raw_total = contract.get(
            "tensor_data_bytes",
            contract.get("declared_weight_total_bytes"),
        )
        if raw_total is None:
            continue
        if (
            isinstance(raw_total, bool)
            or not isinstance(raw_total, int)
            or raw_total < 0
        ):
            raise ValueError(
                "model physical tensor-data total must be a non-negative integer"
            )
        physical_totals.add(raw_total)
    if len(physical_totals) > 1:
        raise ValueError("model physical tensor-data totals conflict")
    if not physical_totals:
        return declared
    physical = next(iter(physical_totals))
    if physical < declared:
        raise ValueError(
            "model physical tensor-data total is smaller than executable weights"
        )
    return physical


def _runtime_placement_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return placement semantics while excluding solver-only evidence."""

    normalized = dict(metadata)
    control_plane = normalized.get("control_plane")
    if isinstance(control_plane, Mapping):
        runtime_control_plane = dict(control_plane)
        runtime_control_plane.pop("evidence", None)
        normalized["control_plane"] = runtime_control_plane
    return normalized


def _runtime_placement_equal(left: object, right: object) -> bool:
    """Compare placement state that can affect lowering or execution."""

    return bool(
        type(left) is type(right)
        and getattr(left, "model_name", None) == getattr(right, "model_name", None)
        and getattr(left, "hardware_name", None)
        == getattr(right, "hardware_name", None)
        and getattr(left, "op_to_component", None)
        == getattr(right, "op_to_component", None)
        and getattr(left, "tensor_to_component", None)
        == getattr(right, "tensor_to_component", None)
        and getattr(left, "tensor_bytes", None)
        == getattr(right, "tensor_bytes", None)
        and getattr(left, "parallel", None) == getattr(right, "parallel", None)
        and getattr(left, "kv_policy", None) == getattr(right, "kv_policy", None)
        and getattr(left, "schema_version", None)
        == getattr(right, "schema_version", None)
        and _runtime_placement_metadata(getattr(left, "metadata", {}))
        == _runtime_placement_metadata(getattr(right, "metadata", {}))
    )


def _runtime_profile(
    scenario: ScenarioConfig,
    *,
    weight_bytes: int,
    gpu_transfer_bytes: Optional[int] = None,
    decision: Optional[PlacementDecision] = None,
) -> Mapping[str, object]:
    """Translate controller profiles and realized work to aggregate costs."""

    cpu = _cpu_profile(scenario)
    controller = scenario.runtime_profile
    cpu_pipeline = cpu.pipeline
    issue_width = max(
        1,
        min(
            cpu_pipeline.decode_width,
            cpu_pipeline.issue_width,
            cpu_pipeline.retire_width,
        ),
    )
    control_instructions_per_ns = max(
        1.0e-12,
        issue_width * cpu_pipeline.frequency_ghz,
    )
    authoring_operator_count = len(scenario.model.graph.operators)
    authoring_tensor_count = len(scenario.model.graph.tensors)
    if decision is None:
        workload = _MaterializedWorkloadCounts(
            requirement_count=(
                authoring_operator_count + authoring_tensor_count
            ),
            physical_tensor_count=authoring_tensor_count,
            executable_operator_count=authoring_operator_count,
            command_unit_count=authoring_operator_count,
            command_unit_count_by_gpu={
                scenario.host_orchestration_profile.gpu_component_id: (
                    authoring_operator_count
                )
            },
        )
    else:
        workload = _materialized_workload_counts(scenario, decision)
    requirement_count = workload.requirement_count
    physical_tensor_count = workload.physical_tensor_count
    executable_operator_count = workload.executable_operator_count
    command_unit_count = workload.command_unit_count
    command_unit_count_by_gpu = dict(workload.command_unit_count_by_gpu)
    component_count = len(scenario.hardware.components)
    capacity_component_count = sum(
        1
        for component in scenario.hardware.components
        if component.is_storage or component.is_active_memory
    )
    orchestration = scenario.host_orchestration_profile
    control_bytes = max(
        orchestration.descriptor_bytes_per_request,
        len(scenario.workload.requests)
        * orchestration.descriptor_bytes_per_request
        + (executable_operator_count + physical_tensor_count) * 16,
    )
    placement_instructions = (
        512 + requirement_count * max(1, component_count) * 48
    )
    # Capacity admission accumulates every physical allocation ledger and then
    # performs one final comparison per declared component.  It remains one
    # aggregate CPU/controller task; these are internal work/transaction
    # counts, not individually materialized events.
    capacity_instructions = (
        192
        + physical_tensor_count * 24
        + capacity_component_count * 32
    )
    allocator_instructions = 256 + physical_tensor_count * 24
    batch_instructions = 192 + len(scenario.workload.requests) * 48
    operator_schedule_instructions = (
        256 + executable_operator_count * 24
    )
    command_instructions = 128 + command_unit_count * 12

    transport = controller.pcie_dma_iommu
    nvme = controller.nvme_page_cache
    transfer_bytes = (
        weight_bytes
        if gpu_transfer_bytes is None
        else max(0, int(gpu_transfer_bytes))
    )
    if transfer_bytes > weight_bytes:
        raise ValueError("GPU weight transfer bytes exceed physical weight bytes")
    gpu_id = scenario.host_orchestration_profile.gpu_component_id
    gpu = controller.gpu_controllers[gpu_id]
    page_count = (
        0
        if transfer_bytes == 0
        else (
            transfer_bytes + transport.iommu_page_size_bytes - 1
        )
        // transport.iommu_page_size_bytes
    )
    iommu_waves = (
        0
        if page_count == 0
        else max(
            1,
            (
                page_count
                + transport.iommu_max_outstanding_walks
                - 1
            )
            // transport.iommu_max_outstanding_walks,
        )
    )
    dma_setup = _dma_setup_service(
        transfer_bytes,
        batch_bytes=transport.dma_batch_bytes,
        queue_depth=transport.dma_queue_depth,
        max_outstanding=transport.dma_max_outstanding,
        fixed_latency_ns=orchestration.dma_latency_ns,
        submission_ns_per_wave=orchestration.dma_queue_submission_ns,
    )
    dma_batches = dma_setup.transaction_count
    nvme_pages = max(
        1,
        (weight_bytes + nvme.page_size_bytes - 1) // nvme.page_size_bytes,
    )
    nvme_io_batches = max(
        1,
        (nvme_pages + nvme.io_batch_size - 1) // nvme.io_batch_size,
    )
    mmu = gpu.mmu_tlb
    # Resident weights do not traverse GPU controllers as a model-sized
    # bootstrap payload.  Those controllers consume the aggregate command and
    # placement descriptors; cold model bytes are charged by NVMe and PCIe.
    gpu_pages = max(
        1,
        (control_bytes + mmu.page_size_bytes - 1) // mmu.page_size_bytes,
    )
    mmu_waves = max(
        1,
        (
            gpu_pages
            + mmu.translation_batch_size
            * mmu.max_outstanding_page_walks
            - 1
        )
        // (mmu.translation_batch_size * mmu.max_outstanding_page_walks),
    )
    l2 = gpu.l2_cache
    l2_requests = max(
        1,
        (control_bytes + l2.line_size_bytes - 1) // l2.line_size_bytes,
    )
    l2_waves = max(
        1,
        (
            l2_requests
            + l2.request_batch_size * l2.max_outstanding_misses
            - 1
        )
        // (l2.request_batch_size * l2.max_outstanding_misses),
    )
    vram = gpu.vram_controller
    vram_requests = max(
        1,
        (control_bytes + l2.line_size_bytes - 1) // l2.line_size_bytes,
    )
    vram_parallel = max(
        1,
        min(
            vram.max_outstanding_requests,
            vram.controller_count
            * vram.channel_count
            * vram.lanes_per_channel,
        ),
    )
    vram_waves = max(
        1,
        (
            vram_requests
            + vram.request_batch_size * vram_parallel
            - 1
        )
        // (vram.request_batch_size * vram_parallel),
    )
    command_batches = (
        0
        if command_unit_count == 0
        else (
            command_unit_count
            + gpu.command_processor.launch_batch_size
            - 1
        )
        // gpu.command_processor.launch_batch_size
    )

    gpu_command_targets = []
    for target_gpu_id, target_command_units in sorted(
        command_unit_count_by_gpu.items()
    ):
        if target_command_units <= 0:
            continue
        target_gpu = controller.gpu_controllers.get(target_gpu_id, gpu)
        target_control_bytes = (
            control_bytes
            if len(command_unit_count_by_gpu) == 1
            else max(
                orchestration.descriptor_bytes_per_request,
                target_command_units * 16,
            )
        )
        target_command_batches = (
            target_command_units
            + target_gpu.command_processor.launch_batch_size
            - 1
        ) // target_gpu.command_processor.launch_batch_size
        target_mmu = target_gpu.mmu_tlb
        target_gpu_pages = max(
            1,
            (target_control_bytes + target_mmu.page_size_bytes - 1)
            // target_mmu.page_size_bytes,
        )
        target_mmu_waves = max(
            1,
            (
                target_gpu_pages
                + target_mmu.translation_batch_size
                * target_mmu.max_outstanding_page_walks
                - 1
            )
            // (
                target_mmu.translation_batch_size
                * target_mmu.max_outstanding_page_walks
            ),
        )
        target_l2 = target_gpu.l2_cache
        target_l2_requests = max(
            1,
            (target_control_bytes + target_l2.line_size_bytes - 1)
            // target_l2.line_size_bytes,
        )
        target_l2_waves = max(
            1,
            (
                target_l2_requests
                + target_l2.request_batch_size
                * target_l2.max_outstanding_misses
                - 1
            )
            // (
                target_l2.request_batch_size
                * target_l2.max_outstanding_misses
            ),
        )
        target_vram = target_gpu.vram_controller
        target_vram_requests = target_l2_requests
        target_vram_parallel = max(
            1,
            min(
                target_vram.max_outstanding_requests,
                target_vram.controller_count
                * target_vram.channel_count
                * target_vram.lanes_per_channel,
            ),
        )
        target_vram_waves = max(
            1,
            (
                target_vram_requests
                + target_vram.request_batch_size * target_vram_parallel
                - 1
            )
            // (target_vram.request_batch_size * target_vram_parallel),
        )
        gpu_command_targets.append(
            {
                "component_id": target_gpu_id,
                "command_unit_count": target_command_units,
                "service_ns": {
                    RuntimeAction.COMMAND_BUILD.value: (
                        128 + target_command_units * 12
                    )
                    / control_instructions_per_ns,
                    RuntimeAction.COMMAND_SUBMIT.value: (
                        orchestration.submission_ns
                    ),
                    RuntimeAction.GPU_COMMAND_PROCESS.value: (
                        target_gpu.command_processor.command_submission_latency_ns
                    ),
                    RuntimeAction.MMU_TLB_LOOKUP.value: (
                        target_mmu_waves * target_mmu.page_walk_latency_ns
                    ),
                    RuntimeAction.L2_LOOKUP.value: (
                        target_l2_waves * target_l2.hit_latency_ns
                    ),
                    RuntimeAction.VRAM_CONTROLLER.value: (
                        target_vram_waves * target_vram.access_latency_ns
                        + target_control_bytes
                        / target_vram.read_bandwidth_gb_s
                    ),
                    RuntimeAction.COMPUTE.value: 0.0,
                },
                "resources": {
                    RuntimeAction.COMMAND_BUILD.value: (
                        orchestration.scheduler_resource_id,
                    ),
                    RuntimeAction.COMMAND_SUBMIT.value: (
                        orchestration.submission_resource_id,
                    ),
                    RuntimeAction.GPU_COMMAND_PROCESS.value: (
                        "{}.command_processor".format(target_gpu_id),
                    ),
                    RuntimeAction.MMU_TLB_LOOKUP.value: (
                        "{}.mmu_tlb".format(target_gpu_id),
                    ),
                    RuntimeAction.L2_LOOKUP.value: (
                        "{}.l2_controller".format(target_gpu_id),
                    ),
                    RuntimeAction.VRAM_CONTROLLER.value: (
                        "{}.vram_controller".format(target_gpu_id),
                    ),
                    RuntimeAction.COMPUTE.value: (target_gpu_id,),
                },
                "transaction_count": {
                    RuntimeAction.COMMAND_BUILD.value: target_command_batches,
                    RuntimeAction.COMMAND_SUBMIT.value: target_command_batches,
                    RuntimeAction.GPU_COMMAND_PROCESS.value: target_command_batches,
                    RuntimeAction.MMU_TLB_LOOKUP.value: target_gpu_pages,
                    RuntimeAction.L2_LOOKUP.value: target_l2_requests,
                    RuntimeAction.VRAM_CONTROLLER.value: target_vram_requests,
                    RuntimeAction.COMPUTE.value: 1,
                },
            }
        )

    service_ns = {
        RuntimeAction.CAPACITY_CHECK.value: (
            capacity_instructions / control_instructions_per_ns
        ),
        RuntimeAction.PLACEMENT_DECISION.value: (
            placement_instructions / control_instructions_per_ns
        ),
        RuntimeAction.ALLOCATE.value: (
            allocator_instructions / control_instructions_per_ns
        ),
        RuntimeAction.WEIGHT_CACHE_LOOKUP.value: (
            controller.cpu.cache_hit_latency_ns
        ),
        RuntimeAction.PAGE_CACHE_LOOKUP.value: (
            controller.cpu.dram_latency_ns
        ),
        RuntimeAction.NVME_READ.value: (
            nvme.nvme_read_latency_ns
            + weight_bytes / nvme.nvme_read_bandwidth_gb_s
        ),
        RuntimeAction.IOMMU_TRANSLATE.value: (
            iommu_waves * transport.iommu_miss_latency_ns
        ),
        RuntimeAction.DMA_MAP.value: dma_setup.service_ns,
        RuntimeAction.PCIE_TRANSFER.value: (
            transfer_bytes / transport.aggregate_pcie_bandwidth_gb_s
        ),
        RuntimeAction.BATCH_SCHEDULE.value: (
            batch_instructions / control_instructions_per_ns
        ),
        RuntimeAction.OPERATOR_SCHEDULE.value: (
            operator_schedule_instructions / control_instructions_per_ns
        ),
        RuntimeAction.COMMAND_BUILD.value: (
            command_instructions / control_instructions_per_ns
        ),
        RuntimeAction.COMMAND_SUBMIT.value: (
            scenario.host_orchestration_profile.submission_ns
        ),
        RuntimeAction.GPU_COMMAND_PROCESS.value: (
            gpu.command_processor.command_submission_latency_ns
        ),
        RuntimeAction.MMU_TLB_LOOKUP.value: (
            mmu_waves * mmu.page_walk_latency_ns
        ),
        RuntimeAction.L2_LOOKUP.value: l2_waves * l2.hit_latency_ns,
        RuntimeAction.VRAM_CONTROLLER.value: (
            vram_waves * vram.access_latency_ns
            + control_bytes / vram.read_bandwidth_gb_s
        ),
        # The bootstrap's compute node is a causal hand-off.  Real operators
        # are appended by serving/streaming lowering on the same kernel.
        RuntimeAction.COMPUTE.value: 0.0,
        RuntimeAction.INTERRUPT.value: 250.0,
        RuntimeAction.CPU_COMPLETE.value: 64.0 / control_instructions_per_ns,
        RuntimeAction.NOOP.value: 0.0,
    }
    cpu_control_resource = orchestration.scheduler_resource_id
    cpu_id = orchestration.cpu_component_id
    gpu_id = orchestration.gpu_component_id
    resources = {
        RuntimeAction.CAPACITY_CHECK.value: (cpu_control_resource,),
        RuntimeAction.PLACEMENT_DECISION.value: (cpu_control_resource,),
        RuntimeAction.ALLOCATE.value: (cpu_control_resource,),
        RuntimeAction.WEIGHT_CACHE_LOOKUP.value: (
            "{}.control_cache".format(cpu_id),
        ),
        RuntimeAction.PAGE_CACHE_LOOKUP.value: ("host.page_cache",),
        RuntimeAction.NVME_READ.value: ("nvme",),
        RuntimeAction.IOMMU_TRANSLATE.value: (
            "{}.iommu".format(cpu_id),
        ),
        RuntimeAction.DMA_MAP.value: (orchestration.dma_resource_id,),
        RuntimeAction.PCIE_TRANSFER.value: (
            "{}.pcie_fabric".format(gpu_id),
        ),
        RuntimeAction.BATCH_SCHEDULE.value: (
            cpu_control_resource,
        ),
        RuntimeAction.OPERATOR_SCHEDULE.value: (
            cpu_control_resource,
        ),
        RuntimeAction.COMMAND_BUILD.value: (cpu_control_resource,),
        RuntimeAction.COMMAND_SUBMIT.value: (
            orchestration.submission_resource_id,
        ),
        RuntimeAction.GPU_COMMAND_PROCESS.value: (
            "{}.command_processor".format(gpu_id),
        ),
        RuntimeAction.MMU_TLB_LOOKUP.value: (
            "{}.mmu_tlb".format(gpu_id),
        ),
        RuntimeAction.L2_LOOKUP.value: (
            "{}.l2_controller".format(gpu_id),
        ),
        RuntimeAction.VRAM_CONTROLLER.value: (
            "{}.vram_controller".format(gpu_id),
        ),
        RuntimeAction.COMPUTE.value: (gpu_id,),
        RuntimeAction.INTERRUPT.value: ("interrupt",),
        RuntimeAction.CPU_COMPLETE.value: (cpu_control_resource,),
    }
    transaction_count = {
        RuntimeAction.CAPACITY_CHECK.value: max(
            1, physical_tensor_count + capacity_component_count
        ),
        RuntimeAction.PLACEMENT_DECISION.value: max(
            1,
            requirement_count * max(1, component_count),
        ),
        RuntimeAction.ALLOCATE.value: max(1, physical_tensor_count),
        RuntimeAction.WEIGHT_CACHE_LOOKUP.value: 1,
        RuntimeAction.PAGE_CACHE_LOOKUP.value: nvme_pages,
        RuntimeAction.NVME_READ.value: nvme_io_batches,
        RuntimeAction.IOMMU_TRANSLATE.value: page_count,
        RuntimeAction.DMA_MAP.value: dma_batches,
        RuntimeAction.PCIE_TRANSFER.value: dma_batches,
        RuntimeAction.BATCH_SCHEDULE.value: max(
            1, len(scenario.workload.requests)
        ),
        RuntimeAction.OPERATOR_SCHEDULE.value: max(
            1, executable_operator_count
        ),
        RuntimeAction.COMMAND_BUILD.value: command_batches,
        RuntimeAction.COMMAND_SUBMIT.value: command_batches,
        RuntimeAction.GPU_COMMAND_PROCESS.value: command_batches,
        RuntimeAction.MMU_TLB_LOOKUP.value: gpu_pages,
        RuntimeAction.L2_LOOKUP.value: l2_requests,
        RuntimeAction.VRAM_CONTROLLER.value: vram_requests,
        RuntimeAction.COMPUTE.value: 1,
        RuntimeAction.INTERRUPT.value: 1,
        RuntimeAction.CPU_COMPLETE.value: max(
            1, len(scenario.workload.requests)
        ),
        RuntimeAction.NOOP.value: 1,
    }
    # Aggregate service costs already include controller-internal parallelism.
    # Only DMA engines remain independent event-kernel lanes.
    capacities = _execution_resource_capacities(scenario)
    from .communication import declared_resource_owners
    resource_owners = declared_resource_owners(scenario.hardware)
    declared_capacity = _declared_capacity_bytes(scenario)
    profile: Dict[str, object] = {
        "service_ns": service_ns,
        "resources": resources,
        "transaction_count": transaction_count,
        "resource_capacities": capacities,
        "resource_owners": resource_owners,
        "default_required_bytes": weight_bytes,
        "control_byte_count": control_bytes,
        "gpu_weight_transfer_bytes": transfer_bytes,
        "host_weight_load_bytes": weight_bytes - transfer_bytes,
        "gpu_command_targets": tuple(gpu_command_targets),
        "workload_counts": {
            "placement_requirements": workload.requirement_count,
            "physical_tensors": workload.physical_tensor_count,
            "executable_operators": workload.executable_operator_count,
            "command_units": workload.command_unit_count,
            "capacity_components": capacity_component_count,
        },
    }
    if declared_capacity > 0:
        profile["capacity_bytes"] = {"system_memory": declared_capacity}
    return profile


def _control_plane_evidence_equal(left: object, right: object) -> bool:
    """Refresh only when the identity-bound evidence changed."""

    def evidence(placement: object) -> object:
        metadata = getattr(placement, "metadata", {})
        control_plane = (
            metadata.get("control_plane", {})
            if isinstance(metadata, Mapping)
            else {}
        )
        evidence = (
            control_plane.get("evidence", {})
            if isinstance(control_plane, Mapping)
            else {}
        )
        if not isinstance(evidence, Mapping):
            return (None, None)
        return (
            evidence.get("input_fingerprint"),
            evidence.get("fingerprint_schema"),
        )

    return evidence(left) == evidence(right)


def bootstrap_control_plane(
    scenario: ScenarioConfig,
    policy: Optional[PlacementPolicy] = None,
) -> ControlPlaneBootstrap:
    """Plan placement and realize its CPU/controller work in one live kernel."""

    if not isinstance(scenario, ScenarioConfig):
        raise TypeError("scenario must be a ScenarioConfig")
    decision = plan_runtime_placement(
        scenario,
        policy,
    )
    if not decision.fully_placed:
        reasons = tuple(
            item.reason for item in decision.unplaced if item.reason
        )
        raise ValueError(
            "runtime control plane could not place scenario: {}".format(
                "; ".join(reasons) or decision.status
            )
        )
    candidate = decision.apply(scenario)
    # A previously mapped scenario is revalidated on every run, but solver
    # evidence (for example a refreshed fingerprint) must not destroy the
    # identity-bound lowering caches used across warmup/repetitions.
    mapped = (
        scenario
        if _runtime_placement_equal(scenario.placement, candidate.placement)
        and _control_plane_evidence_equal(scenario.placement, candidate.placement)
        else candidate
    )
    weight_bytes = _declared_weight_load_bytes(mapped)
    gpu_transfer_bytes = _gpu_weight_transfer_bytes(mapped, decision)
    flow_id = "scenario-control-plane"
    weight_key = "{}.weights".format(mapped.model.name)
    page_key = "{}.pages".format(mapped.model.name)
    state = RuntimeState(
        weight_cache={weight_key} if mapped.weights_resident else set(),
        page_cache={page_key} if mapped.weights_resident else set(),
    )
    runtime = ControlPlaneRuntime(
        _runtime_profile(
            mapped,
            weight_bytes=weight_bytes,
            gpu_transfer_bytes=gpu_transfer_bytes,
            decision=decision,
        ),
        state,
    )
    runtime_profile = runtime.profile
    control_bytes = int(runtime_profile["control_byte_count"])
    result = runtime.run(
        None,
        {
            "placement_decisions": {
                flow_id: "control-plane:{}".format(
                    decision.input_fingerprint[:16]
                )
            },
            "requests": (
                {
                    "request_id": flow_id,
                    "flow_id": flow_id,
                    "required_bytes": weight_bytes,
                    "byte_count": control_bytes,
                    "control_byte_count": control_bytes,
                    "storage_byte_count": weight_bytes,
                    "host_weight_byte_count": (
                        weight_bytes - gpu_transfer_bytes
                    ),
                    "transfer_byte_count": gpu_transfer_bytes,
                    "capacity_resource": "system_memory",
                    "weight_key": weight_key,
                    "page_key": page_key,
                    "instruction_count": max(
                        1,
                        len(decision.decisions),
                    ),
                },
            ),
        },
    )
    kernel = runtime.kernel
    if kernel is None:  # pragma: no cover - run always constructs a kernel
        raise RuntimeError("control-plane runtime did not create a kernel")
    return ControlPlaneBootstrap(mapped, decision, result, kernel)


__all__ = [
    "ControlPlaneBootstrap",
    "bootstrap_control_plane",
]
