"""Types for the lightweight physical memory transaction model.

The module deliberately stores geometry and timing, rather than cell contents
or controller state.  Times are nanoseconds, sizes are bytes and bandwidths
are decimal GB/s (so ``bytes / GB/s`` is a duration in ns).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Optional, Tuple


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


class Operation(str, Enum):
    READ = "read"
    WRITE = "write"
    ERASE = "erase"


class MemoryKind(str, Enum):
    DDR = "DDR"
    LPDDR = "LPDDR"
    HBM = "HBM"
    # GDDR is a first-class DRAM family.  The concrete generation is kept in
    # ``DramConfig.generation`` so all generations share the same DramCore.
    GDDR = "GDDR"
    SSD = "SSD"
    HBF = "HBF"


_GDDR_GENERATIONS = frozenset({"GDDR6", "GDDR6X", "GDDR7"})


def _coerce_dram_kind(value: Any, generation: Any = "") -> tuple[MemoryKind, str]:
    """Canonicalize DRAM family and GDDR generation without guessing NAND."""

    raw = str(getattr(value, "value", value)).strip().upper().replace("-", "")
    generation_text = str(generation or "").strip().upper().replace("-", "")
    # Accept generation-shaped kind values as an authoring convenience while
    # retaining the generation in the canonical config.
    if raw in _GDDR_GENERATIONS:
        if generation_text and generation_text != raw:
            raise ValueError(
                "kind {} conflicts with generation {}".format(raw, generation_text)
            )
        generation_text = generation_text or raw
        raw = "GDDR"
    if raw == "GDDR" and generation_text:
        if generation_text not in _GDDR_GENERATIONS:
            raise ValueError(
                "GDDR generation must be GDDR6, GDDR6X or GDDR7; got {}".format(
                    generation
                )
            )
    try:
        kind = MemoryKind(raw)
    except ValueError as exc:
        raise ValueError(
            "kind must be DDR, LPDDR, HBM, GDDR, SSD or HBF"
        ) from exc
    return kind, generation_text


def parse_physical_memory_config(values: Any) -> "DramConfig | NandConfig":
    """Parse one physical memory config through a single explicit dispatcher.

    Unknown kinds fail before entering the NAND branch.  Generation aliases
    such as ``GDDR7`` are canonicalized to ``kind=GDDR, generation=GDDR7``.
    """

    from dataclasses import asdict, is_dataclass

    raw = asdict(values) if is_dataclass(values) else values
    if not isinstance(raw, Mapping):
        raise ValueError("physical_memory_config must be a DRAM/NAND config mapping")
    payload = dict(raw)
    kind_value = payload.get("kind")
    kind_name = str(getattr(kind_value, "value", kind_value or "")).strip().upper().replace("-", "")
    if kind_name in _GDDR_GENERATIONS:
        return DramConfig.from_mapping(payload)
    if kind_name in {"DDR", "LPDDR", "HBM", "GDDR"}:
        return DramConfig.from_mapping(payload)
    if kind_name in {"SSD", "HBF"}:
        return NandConfig.from_mapping(payload)
    raise ValueError(
        "physical_memory_config.kind must be DDR, LPDDR, HBM, GDDR, SSD or HBF; got {}".format(
            kind_value
        )
    )


@dataclass(frozen=True)
class AccessRequest:
    """One explicit contiguous host request.

    Discrete accesses are represented as multiple requests.  ``address`` is a
    byte address in the modeled medium and ``arrival_ns`` is on the caller's
    shared time line.
    """

    request_id: str
    operation: Operation | str
    address: int
    byte_count: int
    arrival_ns: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        try:
            op = self.operation if isinstance(self.operation, Operation) else Operation(str(self.operation).lower())
        except ValueError as exc:
            raise ValueError("operation must be read, write or erase") from exc
        object.__setattr__(self, "operation", op)
        if isinstance(self.address, bool) or not isinstance(self.address, int) or self.address < 0:
            raise ValueError("address must be a non-negative integer")
        _positive_int("byte_count", self.byte_count)
        _nonnegative("arrival_ns", self.arrival_ns)


@dataclass(frozen=True)
class DramConfig:
    """Parameterized DDR/LPDDR/HBM organization consumed by one DRAM core."""

    kind: MemoryKind | str = MemoryKind.DDR
    generation: str = ""
    channels: int = 1
    subchannels_per_channel: int = 1
    pseudo_channels_per_channel: int = 1
    stacks: int = 1
    dies_per_stack: int = 1
    ranks_per_channel: int = 1
    bank_groups_per_rank: int = 1
    banks_per_group: int = 8
    rows_per_bank: int = 32768
    row_bytes: int = 8192
    burst_bytes: int = 64
    data_width_bits: int = 64
    data_rate_mt_s: float = 3200.0
    # Effective per-pin Gbit/s already includes DDR/PAM encoding effects.
    # Explicit GDDR rate mode uses it once, without changing legacy MT/s.
    effective_pin_data_rate_gbps: Optional[float] = None
    bandwidth_input_mode: str = "bandwidth"
    lane_bandwidth_gb_s: Optional[float] = None
    interface_bandwidth_gb_s: Optional[float] = None
    read_bandwidth_gb_s: Optional[float] = None
    write_bandwidth_gb_s: Optional[float] = None
    data_lanes: Optional[int] = None
    interleave_bytes: Optional[int] = None
    open_ns: float = 14.0
    close_ns: float = 14.0
    read_latency_ns: float = 14.0
    write_latency_ns: float = 12.0
    burst_interval_ns: float = 2.5
    read_recovery_ns: float = 0.0
    write_recovery_ns: float = 15.0
    read_to_write_ns: float = 0.0
    write_to_read_ns: float = 0.0
    capacity_bytes: Optional[int] = None
    max_outstanding_requests: int = 64
    max_expanded_segments: int = 1_000_000
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DramConfig":
        payload = dict(values)
        kind, generation = _coerce_dram_kind(payload.get("kind", MemoryKind.DDR), payload.get("generation", ""))
        payload["kind"] = kind
        if generation:
            payload["generation"] = generation
        return cls(**payload)

    def __post_init__(self) -> None:
        try:
            kind, generation = _coerce_dram_kind(self.kind, self.generation)
        except ValueError as exc:
            raise ValueError("kind must be DDR, LPDDR, HBM or GDDR") from exc
        if kind not in {MemoryKind.DDR, MemoryKind.LPDDR, MemoryKind.HBM, MemoryKind.GDDR}:
            raise ValueError("DramConfig.kind must be DDR, LPDDR, HBM or GDDR")
        object.__setattr__(self, "kind", kind)
        if generation:
            object.__setattr__(self, "generation", generation)
        for name in ("channels", "subchannels_per_channel", "pseudo_channels_per_channel", "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank", "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes", "data_width_bits"):
            _positive_int(name, getattr(self, name))
        _positive_int("max_outstanding_requests", self.max_outstanding_requests)
        _positive_int("max_expanded_segments", self.max_expanded_segments)
        if self.data_lanes is not None:
            _positive_int("data_lanes", self.data_lanes)
        if self.row_bytes % self.burst_bytes:
            raise ValueError("row_bytes must be a multiple of burst_bytes")
        if self.interleave_bytes is not None:
            _positive_int("interleave_bytes", self.interleave_bytes)
            if self.interleave_bytes % self.burst_bytes:
                raise ValueError("interleave_bytes must be a multiple of burst_bytes")
            bank_bytes = self.rows_per_bank * self.row_bytes
            if self.interleave_bytes > bank_bytes or bank_bytes % self.interleave_bytes:
                raise ValueError("interleave_bytes must divide the capacity of one bank")
        for name in ("data_rate_mt_s", "open_ns", "close_ns", "read_latency_ns", "write_latency_ns", "burst_interval_ns", "read_recovery_ns", "write_recovery_ns", "read_to_write_ns", "write_to_read_ns"):
            _nonnegative(name, getattr(self, name))
        for name in ("lane_bandwidth_gb_s", "interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s", "capacity_bytes"):
            value = getattr(self, name)
            if value is not None:
                _positive_int(name, value) if name == "capacity_bytes" else _nonnegative(name, value)
                if name in {"lane_bandwidth_gb_s", "interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s"} and value <= 0:
                    raise ValueError(f"{name} must be positive")
        mode = str(self.bandwidth_input_mode).strip().lower().replace("-", "_")
        if mode not in {"bandwidth", "data_rate"}:
            raise ValueError("bandwidth_input_mode must be bandwidth or data_rate")
        object.__setattr__(self, "bandwidth_input_mode", mode)
        if self.effective_pin_data_rate_gbps is not None:
            rate = _nonnegative("effective_pin_data_rate_gbps", self.effective_pin_data_rate_gbps)
            if rate <= 0:
                raise ValueError("effective_pin_data_rate_gbps must be positive")
            total_from_rate = rate * self.data_width_bits * self.lane_count / 8.0
            if self.interface_bandwidth_gb_s is not None and not math.isclose(
                float(self.interface_bandwidth_gb_s), total_from_rate, rel_tol=1e-12
            ):
                raise ValueError(
                    "interface_bandwidth_gb_s={} GB/s conflicts with effective_pin_data_rate_gbps={} Gb/s and total width {} bit ({} GB/s)".format(
                        self.interface_bandwidth_gb_s, rate,
                        self.data_width_bits * self.lane_count, total_from_rate,
                    )
                )
            if self.lane_bandwidth_gb_s is not None and not math.isclose(
                float(self.lane_bandwidth_gb_s) * self.lane_count,
                total_from_rate,
                rel_tol=1e-12,
            ):
                raise ValueError(
                    "lane_bandwidth_gb_s={} GB/s conflicts with effective_pin_data_rate_gbps={} Gb/s".format(
                        self.lane_bandwidth_gb_s, rate
                    )
                )
            if self.interface_bandwidth_gb_s is None and self.lane_bandwidth_gb_s is None:
                object.__setattr__(self, "interface_bandwidth_gb_s", total_from_rate)
        elif mode == "data_rate":
            raise ValueError("data_rate mode requires effective_pin_data_rate_gbps in Gb/s")
        if self.lane_bandwidth_gb_s is not None and self.interface_bandwidth_gb_s is not None:
            raise ValueError("choose lane_bandwidth_gb_s or interface_bandwidth_gb_s, not both")
        physical_bandwidth = (
            float(self.interface_bandwidth_gb_s)
            if self.interface_bandwidth_gb_s is not None
            else float(self.lane_bandwidth_gb_s) * self.lane_count
            if self.lane_bandwidth_gb_s is not None
            else self.data_width_bits * self.data_rate_mt_s / 8.0 / 1000.0 * self.lane_count
        )
        if physical_bandwidth <= 0 or not math.isfinite(physical_bandwidth):
            raise ValueError("physical interface bandwidth must be finite and positive")
        if any(value is not None and value > physical_bandwidth
               for value in (self.read_bandwidth_gb_s, self.write_bandwidth_gb_s)):
            raise ValueError("directional bandwidth cannot exceed the physical interface bandwidth")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if self.capacity_bytes is not None and self.capacity_bytes > self.computed_capacity_bytes:
            raise ValueError("capacity_bytes cannot exceed the declared physical geometry")
        if self.capacity_bytes is None:
            object.__setattr__(self, "capacity_bytes", self.computed_capacity_bytes)

    @property
    def bank_count(self) -> int:
        return self.ranks_per_channel * self.bank_groups_per_rank * self.banks_per_group

    @property
    def lane_count(self) -> int:
        # data_lanes is the authoritative escape hatch for non-multiplicative
        # package organizations (notably HBM pseudo-channel implementations).
        # Stacked dies add capacity and array choices; they do not imply new
        # external data wires.  Use data_lanes for an explicitly wired design.
        return self.data_lanes or (self.channels * self.subchannels_per_channel * self.pseudo_channels_per_channel)

    @property
    def effective_lane_bandwidth_gb_s(self) -> float:
        if self.interface_bandwidth_gb_s is not None:
            return float(self.interface_bandwidth_gb_s) / self.lane_count
        if self.lane_bandwidth_gb_s is not None:
            return float(self.lane_bandwidth_gb_s)
        return self.data_width_bits * self.data_rate_mt_s / 8.0 / 1000.0

    @property
    def bandwidth_gb_s(self) -> float:
        return max(self.directional_bandwidth_gb_s(Operation.READ), self.directional_bandwidth_gb_s(Operation.WRITE))

    @property
    def physical_interface_bandwidth_gb_s(self) -> float:
        """The shared physical data-interface peak, before direction caps."""

        return self.effective_lane_bandwidth_gb_s * self.lane_count

    def directional_bandwidth_gb_s(self, operation: Operation | str) -> float:
        op = operation if isinstance(operation, Operation) else Operation(str(operation).lower())
        declared = self.read_bandwidth_gb_s if op is Operation.READ else self.write_bandwidth_gb_s
        return float(declared) if declared is not None else self.effective_lane_bandwidth_gb_s * self.lane_count

    def directional_lane_bandwidth_gb_s(self, operation: Operation | str) -> float:
        op = operation if isinstance(operation, Operation) else Operation(str(operation).lower())
        return self.directional_bandwidth_gb_s(op) / self.lane_count

    @property
    def computed_capacity_bytes(self) -> int:
        return self.stacks * self.dies_per_stack * self.lane_count * self.bank_count * self.rows_per_bank * self.row_bytes

    @property
    def effective_capacity_bytes(self) -> int:
        return self.capacity_bytes or self.computed_capacity_bytes


@dataclass(frozen=True)
class NandConfig:
    """Parameterized page NAND organization for SSD or HBF."""

    kind: MemoryKind | str = MemoryKind.SSD
    generation: str = ""
    channels: int = 1
    targets_per_channel: int = 1
    dies_per_target: int = 1
    luns_per_die: int = 1
    planes_per_lun: int = 1
    planes_independent: bool = False
    parallel_units: Optional[int] = None
    blocks_per_plane: int = 2048
    pages_per_block: int = 256
    page_bytes: int = 16384
    host_granularity_bytes: int = 4096
    internal_transfer_bytes: Optional[int] = None
    host_bandwidth_gb_s: float = 4.0
    internal_bandwidth_gb_s: float = 8.0
    page_read_ns: float = 50000.0
    page_program_ns: float = 800000.0
    block_erase_ns: float = 3000000.0
    front_ns: float = 0.0
    partial_page_policy: str = "read_modify_write"
    capacity_bytes: Optional[int] = None
    max_outstanding_requests: int = 32
    max_expanded_segments: int = 1_000_000
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "NandConfig":
        return cls(**dict(values))

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, MemoryKind) else MemoryKind(str(self.kind).upper())
        except ValueError as exc:
            raise ValueError("kind must be SSD or HBF") from exc
        if kind not in {MemoryKind.SSD, MemoryKind.HBF}:
            raise ValueError("NandConfig.kind must be SSD or HBF")
        object.__setattr__(self, "kind", kind)
        for name in ("channels", "targets_per_channel", "dies_per_target", "luns_per_die", "planes_per_lun", "blocks_per_plane", "pages_per_block", "page_bytes", "host_granularity_bytes"):
            _positive_int(name, getattr(self, name))
        if self.parallel_units is not None:
            _positive_int("parallel_units", self.parallel_units)
        if self.internal_transfer_bytes is not None:
            _positive_int("internal_transfer_bytes", self.internal_transfer_bytes)
        if not isinstance(self.planes_independent, bool):
            raise ValueError("planes_independent must be a boolean")
        if self.capacity_bytes is not None:
            _positive_int("capacity_bytes", self.capacity_bytes)
        _positive_int("max_outstanding_requests", self.max_outstanding_requests)
        _positive_int("max_expanded_segments", self.max_expanded_segments)
        for name in ("host_bandwidth_gb_s", "internal_bandwidth_gb_s", "page_read_ns", "page_program_ns", "block_erase_ns", "front_ns"):
            value = _nonnegative(name, getattr(self, name))
            if name.endswith("bandwidth_gb_s") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.partial_page_policy not in {"read_modify_write", "reject"}:
            raise ValueError("partial_page_policy must be read_modify_write or reject")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if self.capacity_bytes is not None and self.capacity_bytes > self.computed_capacity_bytes:
            raise ValueError("capacity_bytes cannot exceed the declared physical geometry")
        if self.capacity_bytes is None:
            object.__setattr__(self, "capacity_bytes", self.computed_capacity_bytes)

    @property
    def array_units(self) -> int:
        if self.parallel_units is not None:
            return self.parallel_units
        base = self.channels * self.targets_per_channel * self.dies_per_target * self.luns_per_die
        return base * self.planes_per_lun if self.planes_independent else base

    @property
    def computed_capacity_bytes(self) -> int:
        return self.channels * self.targets_per_channel * self.dies_per_target * self.luns_per_die * self.planes_per_lun * self.blocks_per_plane * self.pages_per_block * self.page_bytes

    @property
    def effective_capacity_bytes(self) -> int:
        return self.capacity_bytes or self.computed_capacity_bytes

    @property
    def block_bytes(self) -> int:
        return self.pages_per_block * self.page_bytes

    @property
    def transfer_page_bytes(self) -> int:
        """Bytes carried for one complete page, rounded to the transfer quantum."""
        quantum = self.internal_transfer_bytes or self.page_bytes
        return ((self.page_bytes + quantum - 1) // quantum) * quantum


@dataclass(frozen=True)
class AddressMapping:
    address: int
    stack: int = 0
    lane: int = 0
    channel: int = 0
    subchannel: int = 0
    pseudo_channel: int = 0
    rank: int = 0
    bank_group: int = 0
    bank: int = 0
    row: int = 0
    column: int = 0
    unit: int = 0
    target: int = 0
    die: int = 0
    lun: int = 0
    plane: int = 0
    block: int = 0
    page: int = 0
    page_offset: int = 0


@dataclass(frozen=True)
class Segment:
    """A request fragment aligned to one burst or NAND page."""

    request_id: str
    operation: Operation
    address: int
    logical_bytes: int
    transfer_bytes: int
    mapping: AddressMapping
    host_transfer_bytes: int = 0
    internal_transfer_bytes: int = 0


@dataclass(frozen=True)
class StageTiming:
    name: str
    start_ns: float
    end_ns: float
    resource_id: Optional[str] = None
    bytes: int = 0


@dataclass(frozen=True)
class TransactionResult:
    request_id: str
    operation: Operation
    arrival_ns: float
    completion_ns: float
    logical_bytes: int
    transfer_bytes: int
    mapping: Tuple[AddressMapping, ...] = ()
    stages: Tuple[StageTiming, ...] = ()
    counters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def latency_ns(self) -> float:
        return max(0.0, self.completion_ns - self.arrival_ns)

    @property
    def actual_bandwidth_gb_s(self) -> float:
        return self.logical_bytes / self.latency_ns if self.latency_ns > 0 else 0.0

    @property
    def bandwidth_ceiling_gb_s(self) -> float:
        return float(self.counters.get("bandwidth_ceiling_gb_s", 0.0))

    @property
    def host_transfer_bytes(self) -> int:
        return int(self.counters.get("host_transfer_bytes", 0))

    @property
    def internal_transfer_bytes(self) -> int:
        return int(self.counters.get("internal_transfer_bytes", self.transfer_bytes))

    @property
    def physical_read_bytes(self) -> int:
        return int(self.counters.get("physical_read_bytes", 0))

    @property
    def physical_write_bytes(self) -> int:
        return int(self.counters.get("physical_write_bytes", 0))

    @property
    def queue_wait_ns(self) -> float:
        return float(self.counters.get("queue_wait_ns", 0.0))

    @property
    def pages_read(self) -> int:
        return int(self.counters.get("pages_read", 0))

    @property
    def pages_programmed(self) -> int:
        return int(self.counters.get("pages_programmed", 0))

    @property
    def erase_operations(self) -> int:
        return int(self.counters.get("erase_operations", 0))


@dataclass(frozen=True)
class BatchResult:
    """Aggregate of requests executed on one shared physical timeline."""

    requests: Tuple[TransactionResult, ...]
    first_arrival_ns: float
    last_completion_ns: float
    logical_bytes: int
    transfer_bytes: int
    counters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> float:
        return max(0.0, self.last_completion_ns - self.first_arrival_ns)

    @property
    def actual_bandwidth_gb_s(self) -> float:
        return self.logical_bytes / self.duration_ns if self.logical_bytes and self.duration_ns else 0.0

    @property
    def bandwidth_ceiling_gb_s(self) -> float:
        return float(self.counters.get("bandwidth_ceiling_gb_s", 0.0))

    @property
    def read_bandwidth_ceiling_gb_s(self) -> float:
        return float(self.counters.get("read_bandwidth_ceiling_gb_s", 0.0))

    @property
    def write_bandwidth_ceiling_gb_s(self) -> float:
        return float(self.counters.get("write_bandwidth_ceiling_gb_s", 0.0))

    @property
    def physical_read_bytes(self) -> int:
        return int(self.counters.get("physical_read_bytes", 0))

    @property
    def physical_write_bytes(self) -> int:
        return int(self.counters.get("physical_write_bytes", 0))

    @property
    def host_transfer_bytes(self) -> int:
        return int(self.counters.get("host_transfer_bytes", 0))

    @property
    def internal_transfer_bytes(self) -> int:
        return int(self.counters.get("internal_transfer_bytes", self.transfer_bytes))

    @property
    def pages_read(self) -> int:
        return int(self.counters.get("pages_read", 0))

    @property
    def pages_programmed(self) -> int:
        return int(self.counters.get("pages_programmed", 0))

    @property
    def erase_operations(self) -> int:
        return int(self.counters.get("erase_operations", 0))

    @property
    def queue_wait_ns(self) -> float:
        return float(self.counters.get("queue_wait_ns", 0.0))

    def __iter__(self):
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index):
        return self.requests[index]


def make_ddr_config(**kwargs: Any) -> DramConfig:
    kwargs["kind"] = MemoryKind.DDR
    return DramConfig(**kwargs)


def make_lpddr_config(**kwargs: Any) -> DramConfig:
    kwargs["kind"] = MemoryKind.LPDDR
    kwargs.setdefault("data_width_bits", 32)
    return DramConfig(**kwargs)


def make_hbm_config(**kwargs: Any) -> DramConfig:
    kwargs["kind"] = MemoryKind.HBM
    kwargs.setdefault("channels", 8)
    kwargs.setdefault("pseudo_channels_per_channel", 2)
    kwargs.setdefault("data_width_bits", 64)
    return DramConfig(**kwargs)


def make_gddr_config(**kwargs: Any) -> DramConfig:
    """Create a GDDR config while retaining the optional generation label."""

    kwargs["kind"] = MemoryKind.GDDR
    return DramConfig(**kwargs)


def make_ssd_config(**kwargs: Any) -> NandConfig:
    kwargs["kind"] = MemoryKind.SSD
    kwargs.setdefault("host_bandwidth_gb_s", 112.0)
    kwargs.setdefault("internal_bandwidth_gb_s", 112.0)
    return NandConfig(**kwargs)


def make_hbf_config(**kwargs: Any) -> NandConfig:
    kwargs["kind"] = MemoryKind.HBF
    kwargs.setdefault("host_bandwidth_gb_s", 488.0)
    kwargs.setdefault("internal_bandwidth_gb_s", 488.0)
    kwargs.setdefault("page_read_ns", 4_000.0)
    kwargs.setdefault("page_program_ns", 75_000.0)
    kwargs.setdefault("front_ns", 800.0)
    return NandConfig(**kwargs)


__all__ = [
    "AccessRequest", "AddressMapping", "BatchResult", "DramConfig", "MemoryKind", "NandConfig",
    "Operation", "Segment", "StageTiming", "TransactionResult",
    "make_ddr_config", "make_lpddr_config", "make_hbm_config", "make_gddr_config",
    "parse_physical_memory_config",
    "make_ssd_config", "make_hbf_config",
]
