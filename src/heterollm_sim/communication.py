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

from .contracts import ResourceDemand
from .ir import (
    OFFLOAD_STORAGE_COMPONENT_KINDS,
    ComponentSpec,
    HardwareSpec,
    LinkSpec,
)


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(name))
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


@dataclass(frozen=True)
class RouteHop:
    link_id: str
    source_component: str
    target_component: str
    resource_id: str
    bandwidth_gbps: float
    latency_ns: float
    energy_pj_per_byte: float = 0.0

    def transfer_ns(self, byte_count: int) -> float:
        _non_negative_integer(byte_count, "byte_count")
        if self.bandwidth_gbps <= 0:
            raise ValueError("route hop bandwidth must be positive")
        # Gbit/s is numerically bit/ns.
        return self.latency_ns + (8.0 * byte_count) / self.bandwidth_gbps


@dataclass(frozen=True)
class TransferPhase:
    name: str
    demands: Tuple[ResourceDemand, ...]
    metadata: Mapping[str, object]


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
        for component_id in self._adjacency:
            self._adjacency[component_id].sort(
                key=lambda hop: (hop.target_component, hop.link_id, hop.resource_id)
            )

    def _add_link_direction(
        self, link: LinkSpec, source_component: str, target_component: str
    ) -> None:
        if source_component not in self.components or target_component not in self.components:
            return
        bandwidth = float(link.bandwidth_gbps)
        if bandwidth <= 0:
            source_port = self.hardware.get_port(source_component, (
                link.source_port if source_component == link.source_component else link.target_port
            ))
            target_port = self.hardware.get_port(target_component, (
                link.target_port if target_component == link.target_component else link.source_port
            ))
            positive = [
                float(value)
                for value in (source_port.bandwidth_gbps, target_port.bandwidth_gbps)
                if float(value) > 0
            ]
            bandwidth = min(positive) if positive else 0.0
        if bandwidth <= 0:
            return
        shared = bool(link.metadata.get("shared_bidirectional", not link.bidirectional))
        resource_id = "link.{}".format(link.link_id)
        if not shared:
            resource_id += ".{}->{}".format(source_component, target_component)
        self._adjacency[source_component].append(
            RouteHop(
                link_id=link.link_id,
                source_component=source_component,
                target_component=target_component,
                resource_id=resource_id,
                bandwidth_gbps=bandwidth,
                latency_ns=float(link.latency_ns),
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
    ) -> Tuple[TransferPhase, ...]:
        _non_negative_integer(byte_count, "byte_count")
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
        read = self._endpoint_phase(source, byte_count, read=True, name=name)
        if read is not None:
            phases.append(read)
        source_dma = self._dma_phase(source, byte_count, name=name, direction="out")
        if source_dma is not None:
            phases.append(source_dma)
        for index, hop in enumerate(route):
            phases.append(
                TransferPhase(
                    name="{}.link{:02d}".format(name, index),
                    demands=(
                        ResourceDemand(
                            resource_id=hop.resource_id,
                            service_ns=hop.transfer_ns(byte_count),
                            bytes_moved=byte_count,
                            energy_pj=byte_count * hop.energy_pj_per_byte,
                        ),
                    ),
                    metadata={
                        "event_kind": "transfer",
                        "source_component": hop.source_component,
                        "target_component": hop.target_component,
                        "link_id": hop.link_id,
                        "bytes": byte_count,
                    },
                )
            )
        target_dma = self._dma_phase(target, byte_count, name=name, direction="in")
        if target_dma is not None:
            phases.append(target_dma)
        write = self._endpoint_phase(target, byte_count, read=False, name=name)
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
        if len(resource_ids) != len(set(resource_ids)):
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
        component: ComponentSpec, byte_count: int, *, read: bool, name: str
    ) -> Optional[TransferPhase]:
        _non_negative_integer(byte_count, "byte_count")
        bandwidth = (
            float(component.read_bandwidth_gbps)
            if read
            else float(component.write_bandwidth_gbps)
        )
        if bandwidth <= 0:
            # Offload media are active transfer endpoints.  Silently omitting
            # a missing direction used to make an HBF with unknown write
            # bandwidth accept state/KV writes at zero cost.  Active HBM/DRAM
            # interfaces may rely on their declared topology link or global
            # memory profile, but HBF/SSD media must fail closed when a real
            # transfer uses an unknown direction.
            if (
                byte_count > 0
                and component.normalized_kind
                in OFFLOAD_STORAGE_COMPONENT_KINDS
            ):
                direction = "read" if read else "write"
                raise ValueError(
                    "storage component {} requires a positive {} bandwidth "
                    "for this transfer".format(component.component_id, direction)
                )
            return None
        direction = "read" if read else "write"
        latency = _metadata_non_negative_number(
            component.metadata,
            "{}_latency_ns".format(direction),
            0.0,
            component.component_id,
        )
        granularity_raw = component.metadata.get("transfer_granularity_bytes", 0)
        if isinstance(granularity_raw, bool):
            raise ValueError(
                "component {} transfer_granularity_bytes must be a non-negative integer".format(
                    component.component_id
                )
            )
        try:
            granularity = int(granularity_raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "component {} transfer_granularity_bytes must be a non-negative integer".format(
                    component.component_id
                )
            )
        if granularity < 0 or (
            isinstance(granularity_raw, float) and granularity_raw != granularity
        ):
            raise ValueError(
                "component {} transfer_granularity_bytes must be a non-negative integer".format(
                    component.component_id
                )
            )
        transaction_count = 1 if byte_count > 0 else 0
        transferred_bytes = byte_count
        if granularity > 0 and byte_count > 0:
            transaction_count = int(math.ceil(byte_count / float(granularity)))
            transferred_bytes = transaction_count * granularity
        max_outstanding = _metadata_positive_integer(
            component.metadata,
            "max_outstanding_requests",
            1,
            component.component_id,
        )
        latency_batches = (
            int(math.ceil(transaction_count / float(max_outstanding)))
            if transaction_count
            else 0
        )
        energy_key = "{}_energy_pj_per_byte".format(direction)
        energy = _metadata_non_negative_number(
            component.metadata,
            energy_key,
            0.0,
            component.component_id,
        )
        return TransferPhase(
            name="{}.{}.{}".format(name, component.component_id, direction),
            demands=(
                ResourceDemand(
                    resource_id="component.{}.{}".format(component.component_id, direction),
                    service_ns=(
                        latency_batches * latency
                        + (8.0 * transferred_bytes) / bandwidth
                    ),
                    bytes_moved=transferred_bytes,
                    energy_pj=transferred_bytes * energy,
                ),
            ),
            metadata={
                "event_kind": "memory_{}".format(direction),
                "component_id": component.component_id,
                "bytes": byte_count,
                "transferred_bytes": transferred_bytes,
                "transfer_granularity_bytes": granularity,
                "transactions": transaction_count,
                "max_outstanding_requests": max_outstanding,
                "latency_batches": latency_batches,
                "latency_ns": latency,
            },
        )

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
    "choose_collective_algorithm",
    "plan_collective",
]
