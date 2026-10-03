"""Topology-aware analytical communication planning.

The topology validator answers whether links are legal.  This module turns the
same HardwareIR graph into deterministic, contended transfer resources.  It is
transaction-level by design: links, DMA engines, and endpoint memory ports are
modelled, but packets and flits are not.
"""

from __future__ import annotations

from collections import OrderedDict
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .contracts import ResourceDemand, TaskCategory, TaskSpec
from .data_motion import LinkService, endpoint_service, resolve_service
from .ir import (
    ComponentSpec,
    HardwareSpec,
    LinkSpec,
    default_memory_resource_id,
)


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(name))
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def declared_resource_owners(hardware: HardwareSpec) -> Mapping[str, str]:
    """Read explicit service ownership without inferring shared peak rates.

    Hardware metadata may declare ``physical_resource_owners`` as a logical
    resource-id -> physical owner-id mapping. A memory component can also
    bind both endpoint directions with ``memory_service_owner`` or declare
    separate ``read_service_owner`` / ``write_service_owner`` values. A GPU
    compute memory demand shares a controller only when its existing profile
    resource ID is bound to that same owner. DMA lanes remain separate.
    """

    raw = hardware.metadata.get("physical_resource_owners", {})
    if not isinstance(raw, Mapping):
        raise ValueError("physical_resource_owners must be a mapping")
    owners: Dict[str, str] = {}

    def bind(logical: object, owner: object) -> None:
        if not isinstance(logical, str) or not logical or not isinstance(owner, str) or not owner:
            raise ValueError("resource owner ids must be non-empty strings")
        if logical in owners and owners[logical] != owner:
            raise ValueError("conflicting physical owner for " + logical)
        owners[logical] = owner

    for logical, owner in raw.items():
        bind(logical, owner)
    for component in hardware.components:
        common = (component.metadata.get("memory_service", {}).get("physical_owner")
                  or component.metadata.get("physical_owner")
                  or component.metadata.get("memory_service_owner")
                  or (default_memory_resource_id(component) if component.is_storage else None))
        for direction in ("read", "write"):
            owner = component.metadata.get(direction + "_service_owner", common)
            if owner is not None:
                bind("component.{}.{}".format(component.component_id, direction), owner)
    if any(owner in owners and owners[owner] != owner for owner in owners.values()):
        raise ValueError("resource owners must be direct physical ids, not alias chains")
    return owners


@dataclass(frozen=True)
class RouteHop:
    link_id: str
    source_component: str
    target_component: str
    resource_id: str
    bandwidth_gbps: float
    latency_ns: float
    energy_pj_per_byte: float = 0.0
    service_ref: Optional[str] = None
    queue_depth: int = 1

    def transfer_ns(self, byte_count: int) -> float:
        _non_negative_integer(byte_count, "byte_count")
        if self.bandwidth_gbps <= 0:
            raise ValueError("route hop bandwidth must be positive")
        # Gbit/s is numerically bit/ns.
        return self.latency_ns + (8.0 * byte_count) / self.bandwidth_gbps

    @property
    def service(self) -> LinkService:
        return LinkService(self.link_id, self.resource_id, self.resource_id,
                           self.bandwidth_gbps / 8.0, self.latency_ns, self.queue_depth)

    @property
    def resource_capacities(self) -> Mapping[str, int]:
        return {} if self.service_ref else self.service.resource_capacities

    def demands(self, byte_count: int) -> Tuple[ResourceDemand, ...]:
        _non_negative_integer(byte_count, "byte_count")
        if self.service_ref is not None:
            return ()
        return self.service.demands(byte_count, energy_pj_per_byte=self.energy_pj_per_byte)


@dataclass(frozen=True)
class TransferPhase:
    name: str
    demands: Tuple[ResourceDemand, ...]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class TransferPipelinePlan:
    """Opt-in, bounded block pipeline lowered to the ordinary event DAG.

    A credit reserves one whole chunk from its first producer stage until its
    consumer completes.  This conservative ownership includes in-flight bytes;
    it never promises a packet/flit model or unbounded zero-buffer overlap.
    Bounds describe an isolated replay of these exact priced chunks.
    """

    tasks: Tuple[TaskSpec, ...]
    chunk_ready_task_ids: Tuple[str, ...]
    consumer_done_task_ids: Tuple[str, ...]
    chunk_byte_counts: Tuple[int, ...]
    buffer_capacity_bytes: int
    max_inflight_chunks: int
    ideal_lower_bound_ns: float
    serialized_upper_bound_ns: float


def plan_transfer_pipeline(
    chunks: Sequence[Sequence[TransferPhase]],
    *,
    chunk_byte_counts: Sequence[int],
    buffer_capacity_bytes: int,
    max_inflight_chunks: int,
    consumers: Optional[Sequence[TransferPhase]] = None,
    request_id: str = "transfer",
    name: str = "transfer_pipeline",
    buffer_id: str = "transfer_buffer",
    resource_owners: Optional[Mapping[str, str]] = None,
    resource_capacities: Optional[Mapping[str, int]] = None,
) -> TransferPipelinePlan:
    """Add FIFO block-arrival, finite buffer and credit dependencies.

    ``chunks`` are separately costed transactions, not fractions of a whole
    transfer latency.  Every consumer waits for the last transfer stage of its
    own chunk.  Producers reserve capacity before starting and release it only
    at consumer completion; one chunk never overwrites an unconsumed chunk.
    All chunks in this plan share one buffer and credit budget. Separate
    calls do not share credits: a multi-request pipeline must lower its ordered
    chunks in one call until an online buffer-admission owner is provided.
    """

    _positive_integer(buffer_capacity_bytes, "buffer_capacity_bytes")
    _positive_integer(max_inflight_chunks, "max_inflight_chunks")
    for value, label in ((request_id, "request_id"), (name, "name"), (buffer_id, "buffer_id")):
        if not isinstance(value, str) or not value:
            raise ValueError(label + " must be non-empty text")
    priced = tuple(tuple(phases) for phases in chunks)
    sizes = tuple(chunk_byte_counts)
    if len(priced) != len(sizes):
        raise ValueError("one chunk byte count is required per chunk")
    if len(priced) > 4096:
        raise ValueError("explicit transfer pipeline exceeds 4096 chunks")
    if consumers is None:
        consumers = tuple(
            TransferPhase("consume", (), {"event_kind": "chunk_consume"})
            for _ in priced
        )
    else:
        consumers = tuple(consumers)
    if len(consumers) != len(priced):
        raise ValueError("one consumer phase is required per chunk")
    for phases, size, consumer in zip(priced, sizes, consumers):
        _positive_integer(size, "chunk bytes")
        if size > buffer_capacity_bytes:
            raise ValueError("chunk exceeds finite buffer capacity")
        if not phases or any(not isinstance(phase, TransferPhase) for phase in phases):
            raise ValueError("each chunk requires at least one TransferPhase")
        if not isinstance(consumer, TransferPhase):
            raise TypeError("consumers must be TransferPhase values")
    if priced:
        stage_resources = tuple(tuple(d.resource_id for d in phase.demands) for phase in priced[0])
        for phases in priced[1:]:
            if tuple(tuple(d.resource_id for d in phase.demands) for phase in phases) != stage_resources:
                raise ValueError("chunk pipeline stage resource paths must match")
    owners = dict(resource_owners or {})
    capacities = dict(resource_capacities or {})
    owner_capacities: Dict[str, int] = {}
    for logical, capacity in capacities.items():
        _positive_integer(capacity, "resource capacity")
        owner = owners.get(logical, logical)
        if owner in owner_capacities and owner_capacities[owner] != capacity:
            raise ValueError("conflicting physical owner capacities")
        owner_capacities[owner] = capacity
    tasks: List[TaskSpec] = []
    ready_ids: List[str] = []
    done_ids: List[str] = []
    previous_stage_ids: Tuple[str, ...] = ()
    cumulative = [0]
    serial_ns = 0.0
    longest_chunk_ns = 0.0
    owner_service: Dict[str, float] = {}
    byte_release_index = -1
    for index, (phases, size, consumer) in enumerate(zip(priced, sizes, consumers)):
        cumulative.append(cumulative[-1] + size)
        while cumulative[index + 1] - cumulative[byte_release_index + 1] > buffer_capacity_bytes:
            byte_release_index += 1
        release_index = max(byte_release_index, index - max_inflight_chunks)
        credit_dependencies = (done_ids[release_index],) if release_index >= 0 else ()
        stage_ids: List[str] = []
        chunk_ns = 0.0
        for stage_index, phase in enumerate((*phases, consumer)):
            task_id = "{}.chunk{:04d}.stage{:02d}".format(name, index, stage_index)
            dependencies = []
            if stage_ids:
                dependencies.append(stage_ids[-1])
            else:
                dependencies.extend(credit_dependencies)
            # FIFO is a declared pipeline property, even where a stage has
            # multiple physical lanes. It also makes byte-prefix release exact.
            if previous_stage_ids:
                dependencies.append(previous_stage_ids[stage_index])
            stage_ns = max((d.service_ns for d in phase.demands), default=0.0)
            chunk_ns += stage_ns
            serial_ns += stage_ns
            local_owners = [owners.get(d.resource_id, d.resource_id) for d in phase.demands]
            if len(local_owners) != len(set(local_owners)):
                raise ValueError("coalesce same-owner phase demands before pipeline lowering")
            for demand, owner in zip(phase.demands, local_owners):
                owner_service[owner] = owner_service.get(owner, 0.0) + demand.service_ns
            metadata = dict(phase.metadata)
            metadata.update({
                "transfer_execution": "finite_buffer_pipeline",
                "buffer_id": buffer_id,
                "chunk_index": index,
                "chunk_bytes": size,
                "buffer_capacity_bytes": buffer_capacity_bytes,
                "max_inflight_chunks": max_inflight_chunks,
                "buffer_reservation": "producer_start_to_consumer_end",
                "credit_dependency_ids": credit_dependencies if stage_index == 0 else (),
                "is_chunk_consumer": stage_index == len(phases),
            })
            tasks.append(TaskSpec(
                task_id=task_id, request_id=request_id,
                name="{}.{}".format(name, phase.name),
                category=(TaskCategory.COMPUTE if stage_index == len(phases) and phase.demands else TaskCategory.COMMUNICATION),
                dependencies=tuple(dict.fromkeys(dependencies)), demands=phase.demands,
                metadata=metadata,
            ))
            stage_ids.append(task_id)
        ready_ids.append(stage_ids[-2])
        done_ids.append(stage_ids[-1])
        previous_stage_ids = tuple(stage_ids)
        longest_chunk_ns = max(longest_chunk_ns, chunk_ns)
    lower_ns = max(
        longest_chunk_ns,
        max((service / owner_capacities.get(owner, 1) for owner, service in owner_service.items()), default=0.0),
    )
    return TransferPipelinePlan(
        tuple(tasks), tuple(ready_ids), tuple(done_ids), sizes,
        buffer_capacity_bytes, max_inflight_chunks, lower_ns, serial_ns,
    )


@dataclass(frozen=True)
class CollectiveTransfer:
    source_component: str
    target_component: str
    byte_count: int
    route: Tuple[RouteHop, ...]


@dataclass(frozen=True)
class CollectiveRound:
    index: int
    transfers: Tuple[CollectiveTransfer, ...]


@dataclass(frozen=True)
class CollectivePlan:
    kind: str
    algorithm: str
    participants: Tuple[str, ...]
    tensor_bytes: int
    rounds: Tuple[CollectiveRound, ...]


def _metadata_non_negative_number(
    metadata: Mapping[str, object],
    key: str,
    default: float,
    component_id: str,
) -> float:
    raw = metadata.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(
            "component {} {} must be a finite non-negative number".format(
                component_id, key
            )
        )
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "component {} {} must be a finite non-negative number".format(
                component_id, key
            )
        )
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "component {} {} must be a finite non-negative number".format(
                component_id, key
            )
        )
    return value


def _metadata_positive_integer(
    metadata: Mapping[str, object],
    key: str,
    default: int,
    component_id: str,
) -> int:
    raw = metadata.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(
            "component {} {} must be a positive integer".format(
                component_id, key
            )
        )
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "component {} {} must be a positive integer".format(
                component_id, key
            )
        )
    if value <= 0 or (isinstance(raw, float) and raw != value):
        raise ValueError(
            "component {} {} must be a positive integer".format(
                component_id, key
            )
        )
    return value


class TopologyRouter:
    """Deterministic lowest-cost routing over :class:`HardwareSpec`."""

    _ROUTE_CACHE_MAX_ENTRIES = 4096

    def __init__(
        self,
        hardware: HardwareSpec,
        *,
        coherent_dma_mode: str = "pipelined",
    ) -> None:
        mode = str(coherent_dma_mode).strip().lower().replace("-", "_")
        if mode not in {"pipelined", "strict_serialized"}:
            raise ValueError(
                "coherent_dma_mode must be 'pipelined' or 'strict_serialized'"
            )
        self.hardware = hardware
        self.resource_owners = declared_resource_owners(hardware)
        self.coherent_dma_mode = mode
        self.components = hardware.component_map()
        self._links = {link.link_id: link for link in hardware.links}
        self._route_cache: OrderedDict[
            Tuple[str, str, int, str], Tuple[RouteHop, ...]
        ] = OrderedDict()
        self._adjacency: Dict[str, List[RouteHop]] = {
            component_id: [] for component_id in self.components
        }
        for link in hardware.links:
            self._add_link_direction(link, link.source_component, link.target_component)
            if link.bidirectional:
                self._add_link_direction(link, link.target_component, link.source_component)
        self.resource_capacities = {}
        for hops in self._adjacency.values():
            for hop in hops:
                self.resource_capacities.update(hop.resource_capacities)
        for component_id in self._adjacency:
            self._adjacency[component_id].sort(
                key=lambda hop: (hop.target_component, hop.link_id, hop.resource_id)
            )

    def _add_link_direction(
        self, link: LinkSpec, source_component: str, target_component: str
    ) -> None:
        if source_component not in self.components or target_component not in self.components:
            return
        service_ref = link.metadata.get("service_ref")
        source_flag = str(link.metadata.get("bandwidth_source", "")).strip().lower()
        memory_component = None
        if service_ref or source_flag in {"component", "memory_component", "shared_component"}:
            candidates = [self.components[ident] for ident in (source_component, target_component)
                          if self.components[ident].is_storage]
            if service_ref:
                candidates = [item for item in candidates if str(service_ref) in {
                    item.component_id, item.component_id + ".access",
                    str(item.metadata.get("memory_service", {}).get("service_id", "")),
                }]
            if len(candidates) != 1:
                raise ValueError("link {} service_ref must identify one storage endpoint".format(link.link_id))
            memory_component = candidates[0]
            owner = resolve_service(memory_component).resource_id
            declared = link.metadata.get("bandwidth_resource_id")
            if declared and declared != owner:
                raise ValueError("link {} bandwidth_resource_id conflicts with service_ref {}".format(link.link_id, service_ref or memory_component.component_id))
            service_ref = memory_component.component_id + ".access"
        bandwidth = float(link.bandwidth_gbps)
        if memory_component is not None:
            direction = "read" if memory_component.component_id == source_component else "write"
            physical_service = resolve_service(memory_component)
            bandwidth = getattr(physical_service, direction + "_bandwidth_gb_s") * 8.0
        elif str(link.protocol).strip().lower() in {"hbm", "hbm_stack", "hbf", "ddr", "dram", "tsv", "lpddr5x"}:
            # An unbound local interface remains an independent topology
            # service, but its negotiated rate cannot exceed the endpoint's
            # declared direction. This is a physical link ceiling, not an
            # inferred service alias.
            candidates = [self.components[ident] for ident in (source_component, target_component)
                          if self.components[ident].is_storage]
            if candidates:
                direction = "read" if candidates[0].component_id == source_component else "write"
                endpoint_limit = candidates[0].directional_bandwidth_gbps(direction)
                if endpoint_limit > 0 and bandwidth > 0:
                    bandwidth = min(bandwidth, endpoint_limit)
        if bandwidth <= 0 and memory_component is None:
            source_port = self.hardware.get_port(source_component, (
                link.source_port if source_component == link.source_component else link.target_port
            ))
            target_port = self.hardware.get_port(target_component, (
                link.target_port if target_component == link.target_component else link.source_port
            ))
            positive = [float(value) for value in (source_port.bandwidth_gbps, target_port.bandwidth_gbps)
                        if float(value) > 0]
            bandwidth = min(positive) if positive else 0.0
        if bandwidth <= 0:
            return
        if memory_component is not None:
            resource_id = resolve_service(memory_component).resource_id
        else:
            resource_id = str(link.metadata.get("bandwidth_resource_id") or "link.{}".format(link.link_id))
            shared = bool(link.metadata.get("shared_bidirectional", not link.bidirectional))
            if not shared:
                resource_id += ".{}->{}".format(source_component, target_component)
        queue_depth = _metadata_positive_integer(link.metadata, "queue_depth", 1, "link " + link.link_id)
        self._adjacency[source_component].append(
            RouteHop(
                link_id=link.link_id,
                source_component=source_component,
                target_component=target_component,
                resource_id=resource_id,
                bandwidth_gbps=bandwidth,
                latency_ns=0.0 if service_ref else float(link.latency_ns),
                service_ref=str(service_ref) if service_ref else None,
                queue_depth=queue_depth,
                energy_pj_per_byte=_metadata_non_negative_number(
                    link.metadata,
                    "energy_pj_per_byte",
                    0.0,
                    "link {}".format(link.link_id),
                ),
            )
        )

    def route(
        self,
        source_component: str,
        target_component: str,
        byte_count: int,
        *,
        policy: str = "lowest_latency",
    ) -> Tuple[RouteHop, ...]:
        _non_negative_integer(byte_count, "byte_count")
        if source_component not in self.components:
            raise ValueError("unknown route source component {}".format(source_component))
        if target_component not in self.components:
            raise ValueError("unknown route target component {}".format(target_component))
        if source_component == target_component:
            return ()
        normalized = policy.strip().lower().replace("-", "_")
        if normalized != "lowest_latency":
            raise ValueError("unsupported routing policy {}".format(policy))

        cache_key = (
            source_component,
            target_component,
            byte_count,
            normalized,
        )
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            self._route_cache.move_to_end(cache_key)
            return cached

        # The path signature is included in the heap key so equal-cost routes
        # are stable across Python versions and input ordering.
        queue: List[Tuple[float, Tuple[str, ...], str, Tuple[RouteHop, ...]]] = [
            (0.0, (), source_component, ())
        ]
        best: Dict[str, Tuple[float, Tuple[str, ...]]] = {
            source_component: (0.0, ())
        }
        while queue:
            cost, signature, component_id, hops = heapq.heappop(queue)
            if best.get(component_id) != (cost, signature):
                continue
            if component_id == target_component:
                self._route_cache[cache_key] = hops
                self._route_cache.move_to_end(cache_key)
                if len(self._route_cache) > self._ROUTE_CACHE_MAX_ENTRIES:
                    self._route_cache.popitem(last=False)
                return hops
            for hop in self._adjacency.get(component_id, ()):
                next_cost = cost + hop.transfer_ns(byte_count)
                next_signature = signature + (hop.link_id, hop.target_component)
                previous = best.get(hop.target_component)
                candidate = (next_cost, next_signature)
                if previous is None or candidate < previous:
                    best[hop.target_component] = candidate
                    heapq.heappush(
                        queue,
                        (next_cost, next_signature, hop.target_component, hops + (hop,)),
                    )
        raise ValueError(
            "no communication route from {} to {}".format(
                source_component, target_component
            )
        )

    def transfer_phases(
        self,
        source_component: str,
        target_component: str,
        byte_count: int,
        *,
        policy: str = "lowest_latency",
        name: str = "transfer",
        coherent_dma_mode: Optional[str] = None,
        page_offset_bytes: Optional[int] = None,
        source_page_offset_bytes: Optional[int] = None,
        target_page_offset_bytes: Optional[int] = None,
    ) -> Tuple[TransferPhase, ...]:
        _non_negative_integer(byte_count, "byte_count")
        for label, value in (
            ("page_offset_bytes", page_offset_bytes),
            ("source_page_offset_bytes", source_page_offset_bytes),
            ("target_page_offset_bytes", target_page_offset_bytes),
        ):
            if value is not None:
                _non_negative_integer(value, label)
        if source_page_offset_bytes is None:
            source_page_offset_bytes = page_offset_bytes
        if target_page_offset_bytes is None:
            target_page_offset_bytes = page_offset_bytes
        if source_component not in self.components:
            raise ValueError(
                "unknown route source component {}".format(source_component)
            )
        if target_component not in self.components:
            raise ValueError(
                "unknown route target component {}".format(target_component)
            )
        if byte_count == 0 or source_component == target_component:
            return ()
        route = self.route(
            source_component, target_component, byte_count, policy=policy
        )
        if not route:
            return ()
        phases: List[TransferPhase] = []
        source = self.components[source_component]
        target = self.components[target_component]
        # A reference edge only describes connectivity. The endpoint service
        # owns its timing, payload and controller; never delete the endpoint
        # because a topology view happens to share its resource name.
        read = self._endpoint_phase(
            source,
            byte_count,
            read=True,
            name=name,
            page_offset_bytes=source_page_offset_bytes,
        )
        if read is not None:
            phases.append(read)
        source_dma = self._dma_phase(source, byte_count, name=name, direction="out")
        if source_dma is not None:
            phases.append(source_dma)
        for index, hop in enumerate(route):
            if hop.service_ref is not None:
                continue
            phases.append(
                TransferPhase(
                    name="{}.link{:02d}".format(name, index),
                    demands=hop.demands(byte_count),
                    metadata={
                        "event_kind": "transfer",
                        "source_component": hop.source_component,
                        "target_component": hop.target_component,
                        "link_id": hop.link_id,
                        "bytes": byte_count,
                        "operation_id": name,
                        "access_kind": "COPY",
                        "link_bytes": byte_count,
                        "resource_capacities": hop.resource_capacities,
                    },
                )
            )
        target_dma = self._dma_phase(target, byte_count, name=name, direction="in")
        if target_dma is not None:
            phases.append(target_dma)
        write = self._endpoint_phase(
            target,
            byte_count,
            read=False,
            name=name,
            page_offset_bytes=target_page_offset_bytes,
        )
        if write is not None:
            phases.append(write)
        mode = self.coherent_dma_mode if coherent_dma_mode is None else str(
            coherent_dma_mode
        ).strip().lower().replace("-", "_")
        if mode not in {"pipelined", "strict_serialized"}:
            raise ValueError(
                "coherent_dma_mode must be 'pipelined' or 'strict_serialized'"
            )
        # strict_serialized retains the explicit read -> DMA -> link -> DMA ->
        # write phases.  The event kernel then chains those phases as separate
        # tasks, so latency is the sum of each stage rather than the optimistic
        # max() of a folded multi-resource task.
        coherent = None if mode == "strict_serialized" else self._coherent_dma_phase(
            source,
            target,
            route,
            phases,
            byte_count=byte_count,
            name=name,
        )
        if coherent is not None:
            return (coherent,)
        return tuple(phases)

    def transfer_pipeline(
        self,
        source_component: str,
        target_component: str,
        byte_count: int,
        *,
        chunk_size_bytes: int,
        buffer_capacity_bytes: int,
        max_inflight_chunks: int,
        consumers: Optional[Sequence[TransferPhase]] = None,
        request_id: str = "transfer",
        name: str = "transfer_pipeline",
        buffer_id: str = "transfer_buffer",
        resource_owners: Optional[Mapping[str, str]] = None,
        resource_capacities: Optional[Mapping[str, int]] = None,
        source_page_offset_bytes: Optional[int] = None,
        target_page_offset_bytes: Optional[int] = None,
    ) -> TransferPipelinePlan:
        """Price blocks separately, then lower their bounded streaming DAG."""

        _non_negative_integer(byte_count, "byte_count")
        for label, value in (
            ("source_page_offset_bytes", source_page_offset_bytes),
            ("target_page_offset_bytes", target_page_offset_bytes),
        ):
            if value is not None:
                _non_negative_integer(value, label)
        _positive_integer(chunk_size_bytes, "chunk_size_bytes")
        if (byte_count + chunk_size_bytes - 1) // chunk_size_bytes > 4096:
            raise ValueError("explicit transfer pipeline exceeds 4096 chunks")
        sizes = tuple(
            min(chunk_size_bytes, byte_count - offset)
            for offset in range(0, byte_count, chunk_size_bytes)
        )
        chunks = tuple(
            self.transfer_phases(
                source_component, target_component, size, name=name,
                coherent_dma_mode="strict_serialized",
                source_page_offset_bytes=(
                    source_page_offset_bytes + offset
                    if source_page_offset_bytes is not None else None
                ),
                target_page_offset_bytes=(
                    target_page_offset_bytes + offset
                    if target_page_offset_bytes is not None else None
                ),
            )
            for offset, size in zip(range(0, byte_count, chunk_size_bytes), sizes)
        )
        return plan_transfer_pipeline(
            chunks, chunk_byte_counts=sizes,
            buffer_capacity_bytes=buffer_capacity_bytes,
            max_inflight_chunks=max_inflight_chunks, consumers=consumers,
            request_id=request_id, name=name, buffer_id=buffer_id,
            resource_owners=self.resource_owners if resource_owners is None else resource_owners,
            resource_capacities={**self.resource_capacities, **dict(resource_capacities or {})},
        )

    def _coherent_dma_phase(
        self,
        source: ComponentSpec,
        target: ComponentSpec,
        route: Sequence[RouteHop],
        phases: Sequence[TransferPhase],
        *,
        byte_count: int,
        name: str,
    ) -> Optional[TransferPhase]:
        """Fold one fully declared host-DMA span into an atomic task phase.

        This is deliberately fail-closed.  The legacy serial phases remain
        authoritative unless every routed link explicitly opts in, a
        multi-link route declares one shared span, exactly one PCIe anchor is
        present, both active-memory endpoints expose their physical service,
        and every resource is unique.  ``LinkSpec.payload`` is intentionally
        ignored: it describes carried data, not execution semantics.
        """

        if not route or not phases:
            return None
        # Preserve address-aware DRAM endpoint state by keeping explicit
        # READ/link/WRITE phases instead of folding them into one task.
        if any(
            "dram_access" in phase.metadata
            or "nand_media" in phase.metadata
            or "hbf_media" in phase.metadata
            for phase in phases
        ):
            return None
        if not source.is_active_memory or not target.is_active_memory:
            return None
        if any(
            str(self._links[hop.link_id].metadata.get("transfer_execution", ""))
            .strip()
            .lower()
            .replace("-", "_")
            != "coherent_dma"
            for hop in route
        ):
            return None

        span_ids = tuple(
            str(
                self._links[hop.link_id].metadata.get(
                    "coherent_dma_span_id", ""
                )
            ).strip()
            for hop in route
        )
        if len(route) > 1 and (
            any(not span_id for span_id in span_ids)
            or len(set(span_ids)) != 1
        ):
            return None
        span_id = span_ids[0] if span_ids and span_ids[0] else None

        def normalized_protocol(hop: RouteHop) -> str:
            return "".join(
                character
                for character in self._links[hop.link_id].protocol.lower()
                if character.isalnum()
            )

        if sum(normalized_protocol(hop) == "pcie" for hop in route) != 1:
            return None
        if any(
            self._links[hop.link_id].metadata.get("store_and_forward") is True
            or str(
                self._links[hop.link_id].metadata.get("forwarding_mode", "")
            )
            .strip()
            .lower()
            .replace("-", "_")
            == "store_and_forward"
            for hop in route
        ):
            return None

        forbidden_intermediate_kinds = {
            "bridge",
            "fabric",
            "interconnect",
            "switch",
        }
        intermediate_ids = tuple(hop.target_component for hop in route[:-1])

        def is_switch_like(component_id: str) -> bool:
            kind = self.components[component_id].normalized_kind
            return (
                kind in forbidden_intermediate_kinds
                or "switch" in kind.split("_")
            )

        if any(
            is_switch_like(component_id)
            or self.components[component_id].metadata.get("store_and_forward") is True
            for component_id in intermediate_ids
        ):
            return None

        event_kinds = tuple(
            str(phase.metadata.get("event_kind", "")) for phase in phases
        )
        if not event_kinds or event_kinds[0] != "memory_read":
            return None
        if event_kinds[-1] != "memory_write":
            return None

        demands = tuple(demand for phase in phases for demand in phase.demands)
        resource_ids = tuple(demand.resource_id for demand in demands)
        physical_owners = tuple(self.resource_owners.get(resource_id, resource_id) for resource_id in resource_ids)
        if len(physical_owners) != len(set(physical_owners)):
            # Folding same-controller read/write service with max() would
            # discard work. Retain serial phases until explicit chunking.
            return None

        resource_directions: Dict[str, str] = {}
        for phase in phases:
            event_kind = str(phase.metadata.get("event_kind", ""))
            if event_kind == "memory_read":
                direction = "read"
            elif event_kind == "memory_write":
                direction = "write"
            elif event_kind == "dma":
                direction = (
                    "read"
                    if str(phase.metadata.get("direction", "")) == "out"
                    else "write"
                )
            else:
                direction = "transfer"
            for demand in phase.demands:
                resource_directions[demand.resource_id] = direction

        route_hops = tuple(
            {
                "link_id": hop.link_id,
                "protocol": self._links[hop.link_id].protocol,
                "source_component": hop.source_component,
                "target_component": hop.target_component,
                "resource_id": hop.resource_id,
            }
            for hop in route
        )
        metadata: Dict[str, object] = {
            "event_kind": "transfer",
            "transfer_execution": "coherent_dma",
            "source_component": source.component_id,
            "target_component": target.component_id,
            "bytes": byte_count,
            "logical_transfer_bytes": byte_count,
            "logical_bytes_accounting": "once_per_transfer",
            "resource_directions": resource_directions,
            "link_ids": tuple(hop.link_id for hop in route),
            "route_hops": route_hops,
        }
        if span_id is not None:
            metadata["coherent_dma_span_id"] = span_id
        return TransferPhase(
            name="{}.coherent_dma".format(name),
            demands=demands,
            metadata=metadata,
        )

    @staticmethod
    def _endpoint_phase(
        component: ComponentSpec,
        byte_count: int,
        *,
        read: bool,
        name: str,
        page_offset_bytes: Optional[int] = None,
    ) -> Optional[TransferPhase]:
        service = endpoint_service(
            component,
            byte_count,
            read=read,
            name=name,
            page_offset_bytes=page_offset_bytes,
        )
        if service is None:
            return None
        return TransferPhase(service.name, service.demands, service.metadata)

    @staticmethod
    def _dma_phase(
        component: ComponentSpec, byte_count: int, *, name: str, direction: str
    ) -> Optional[TransferPhase]:
        _non_negative_integer(byte_count, "byte_count")
        bandwidth = _metadata_non_negative_number(
            component.metadata,
            "dma_bandwidth_gbps",
            0.0,
            component.component_id,
        )
        if bandwidth <= 0:
            configured_latency = _metadata_non_negative_number(
                component.metadata,
                "dma_latency_ns",
                0.0,
                component.component_id,
            )
            configured_energy = _metadata_non_negative_number(
                component.metadata,
                "dma_energy_pj_per_byte",
                0.0,
                component.component_id,
            )
            if configured_latency > 0.0 or configured_energy > 0.0:
                raise ValueError(
                    "component {} declares DMA latency/energy but no positive "
                    "dma_bandwidth_gbps".format(component.component_id)
                )
            return None
        latency = _metadata_non_negative_number(
            component.metadata,
            "dma_latency_ns",
            0.0,
            component.component_id,
        )
        energy = _metadata_non_negative_number(
            component.metadata,
            "dma_energy_pj_per_byte",
            0.0,
            component.component_id,
        )
        resource_id = str(
            component.metadata.get(
                "dma_resource_id", "component.{}.dma".format(component.component_id)
            )
        )
        return TransferPhase(
            name="{}.{}.dma_{}".format(name, component.component_id, direction),
            demands=(
                ResourceDemand(
                    resource_id=resource_id,
                    service_ns=latency + (8.0 * byte_count) / bandwidth,
                    bytes_moved=byte_count,
                    energy_pj=byte_count * energy,
                ),
            ),
            metadata={
                "event_kind": "dma",
                "component_id": component.component_id,
                "direction": direction,
                "bytes": byte_count,
            },
        )


def choose_collective_algorithm(
    requested: str, participant_count: int, tensor_bytes: int
) -> str:
    _positive_integer(participant_count, "participant_count")
    _non_negative_integer(tensor_bytes, "tensor_bytes")
    normalized = requested.strip().lower().replace("-", "_")
    if normalized in {"ring", "tree"}:
        return normalized
    if normalized != "auto":
        raise ValueError("unsupported collective algorithm {}".format(requested))
    # Tree minimizes startup for small collectives; ring makes better use of
    # bandwidth for larger messages and larger groups.
    return "tree" if participant_count <= 4 and tensor_bytes < 1_048_576 else "ring"


def plan_collective(
    router: TopologyRouter,
    kind: str,
    participants: Sequence[str],
    tensor_bytes: int,
    *,
    algorithm: str = "auto",
    routing_policy: str = "lowest_latency",
) -> CollectivePlan:
    normalized_kind = kind.strip().lower().replace("-", "_")
    supported = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}
    if normalized_kind not in supported:
        raise ValueError("unsupported collective kind {}".format(kind))
    _non_negative_integer(tensor_bytes, "tensor_bytes")
    ordered = tuple(str(item) for item in participants)
    if not ordered:
        raise ValueError("collective participants must not be empty")
    if len(ordered) != len(set(ordered)):
        raise ValueError("collective participants must be unique")
    if any(item not in router.components for item in ordered):
        raise ValueError("collective references an unknown component")
    requested_algorithm = algorithm.strip().lower().replace("-", "_")
    selected = choose_collective_algorithm(algorithm, len(ordered), tensor_bytes)
    # A pairwise ring is the deterministic bandwidth model for all-to-all.
    # A gather/broadcast tree creates an artificial root bottleneck, so it is
    # only used when the scenario explicitly requests ``tree``.
    if normalized_kind == "all_to_all" and requested_algorithm == "tree":
        raise ValueError(
            "tree is not a valid all_to_all algorithm; use auto or ring"
        )
    if normalized_kind == "all_to_all":
        selected = "ring"
    if len(ordered) <= 1 or tensor_bytes == 0:
        return CollectivePlan(normalized_kind, selected, ordered, tensor_bytes, ())
    if selected == "ring":
        rounds = _ring_rounds(
            router, normalized_kind, ordered, tensor_bytes, routing_policy
        )
    else:
        rounds = _tree_rounds(
            router, normalized_kind, ordered, tensor_bytes, routing_policy
        )
    return CollectivePlan(normalized_kind, selected, ordered, tensor_bytes, rounds)


def _ring_rounds(
    router: TopologyRouter,
    kind: str,
    participants: Tuple[str, ...],
    tensor_bytes: int,
    routing_policy: str,
) -> Tuple[CollectiveRound, ...]:
    count = len(participants)
    chunk = int(math.ceil(tensor_bytes / float(count)))
    round_count = {
        "all_reduce": 2 * (count - 1),
        "all_gather": count - 1,
        "reduce_scatter": count - 1,
        "all_to_all": count - 1,
    }[kind]
    rounds: List[CollectiveRound] = []
    for round_index in range(round_count):
        transfers = []
        for rank, source in enumerate(participants):
            # Ring reductions forward chunks to the next hop.  Pairwise
            # all-to-all instead rotates the destination every round so every
            # source reaches every other participant exactly once.
            distance = round_index + 1 if kind == "all_to_all" else 1
            target = participants[(rank + distance) % count]
            transfers.append(
                CollectiveTransfer(
                    source,
                    target,
                    chunk,
                    router.route(source, target, chunk, policy=routing_policy),
                )
            )
        rounds.append(CollectiveRound(round_index, tuple(transfers)))
    return tuple(rounds)


def _tree_rounds(
    router: TopologyRouter,
    kind: str,
    participants: Tuple[str, ...],
    tensor_bytes: int,
    routing_policy: str,
) -> Tuple[CollectiveRound, ...]:
    # A stable binomial tree.  Reduction-like collectives move toward rank 0;
    # gather/broadcast or shard-scatter phases reverse the same edges.
    count = len(participants)
    depth = int(math.ceil(math.log(count, 2)))
    reduce_needed = kind in {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "all_to_all",
    }
    gather_needed = kind in {"all_reduce", "all_gather", "all_to_all"}
    rounds: List[CollectiveRound] = []
    round_index = 0
    chunk = int(math.ceil(tensor_bytes / float(count)))
    padded_tensor_bytes = chunk * count
    if reduce_needed:
        # Leaves must reduce into their parent before that parent forwards the
        # accumulated value toward rank 0.  For four ranks this is
        # (1 -> 0, 3 -> 2) followed by (2 -> 0).
        for level in range(depth):
            step = 1 << level
            transfers = []
            for child in range(step, count, 2 * step):
                parent = child - step
                if parent >= count:
                    continue
                source = participants[child]
                target = participants[parent]
                if kind in {"all_reduce", "reduce_scatter"}:
                    # This is a conservative reduce-to-root tree, not a
                    # recursive-halving shortcut.  Every edge therefore
                    # carries the complete padded tensor before rank 0
                    # broadcasts the logical result or scatters disjoint
                    # result shards back down the tree.
                    byte_count = padded_tensor_bytes
                else:
                    # Gather the chunks already accumulated by the child's
                    # subtree.  Nearer the root, each transfer carries more
                    # participants' data.
                    subtree_size = min(step, count - child)
                    byte_count = min(tensor_bytes, chunk * subtree_size)
                transfers.append(
                    CollectiveTransfer(
                        source,
                        target,
                        byte_count,
                        router.route(
                            source, target, byte_count, policy=routing_policy
                        ),
                    )
                )
            if transfers:
                rounds.append(CollectiveRound(round_index, tuple(transfers)))
                round_index += 1
    if kind == "reduce_scatter":
        # Rank 0 owns the complete reduced tensor after the upward phase.
        # Send exactly the contiguous shard range owned by each child subtree;
        # reversing levels ensures a parent receives its range before it must
        # forward a descendant's shard.  This keeps arbitrary participant
        # counts correct while charging ceil-padding consistently.
        for level in reversed(range(depth)):
            step = 1 << level
            transfers = []
            for child in range(step, count, 2 * step):
                parent = child - step
                if parent >= count:
                    continue
                subtree_size = min(step, count - child)
                source = participants[parent]
                target = participants[child]
                byte_count = chunk * subtree_size
                transfers.append(
                    CollectiveTransfer(
                        source,
                        target,
                        byte_count,
                        router.route(
                            source, target, byte_count, policy=routing_policy
                        ),
                    )
                )
            if transfers:
                rounds.append(CollectiveRound(round_index, tuple(transfers)))
                round_index += 1
    if gather_needed:
        # Broadcast walks the same tree in the opposite order: parents first
        # receive the value, then fan it out to their children.
        for level in reversed(range(depth)):
            step = 1 << level
            transfers = []
            for child in range(step, count, 2 * step):
                parent = child - step
                if parent >= count:
                    continue
                source = participants[parent]
                target = participants[child]
                byte_count = tensor_bytes
                transfers.append(
                    CollectiveTransfer(
                        source,
                        target,
                        byte_count,
                        router.route(
                            source, target, byte_count, policy=routing_policy
                        ),
                    )
                )
            if transfers:
                rounds.append(CollectiveRound(round_index, tuple(transfers)))
                round_index += 1
    return tuple(rounds)


__all__ = [
    "CollectivePlan",
    "CollectiveRound",
    "CollectiveTransfer",
    "RouteHop",
    "TopologyRouter",
    "TransferPhase",
    "TransferPipelinePlan",
    "plan_transfer_pipeline",
    "choose_collective_algorithm",
    "declared_resource_owners",
    "plan_collective",
]
