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

from ._hbm_errors import HBMConfigError


def _hbm_config_error(config, field, *, category, rule, requirement_en,
                      requirement_zh, remedy_zh, remedy_en, label=None,
                      unit=None, related_values=None, source=None):
    return HBMConfigError(
        field=field, value=getattr(config, field), generation=config.generation,
        clock_ns=config.command_clock_ns, category=category, rule=rule,
        requirement_en=requirement_en, requirement_zh=requirement_zh,
        remedy_zh=remedy_zh, remedy_en=remedy_en, label=label, unit=unit,
        related_values=related_values, source=source,
    )


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
    SRAM = "SRAM"


_GDDR_GENERATIONS = frozenset({"GDDR6", "GDDR6X", "GDDR7"})

# Optional, authored timing constraints shared by every DRAM family. Zero
# disables a constraint; these are not inferred JEDEC/vendor timing values.
DRAM_BANK_GROUP_TIMING_FIELDS = (
    "column_same_group_ns", "column_different_group_ns",
    "activate_same_group_ns", "activate_different_group_ns",
    "four_activate_window_ns",
    "write_to_read_same_group_ns", "write_to_read_different_group_ns",
)

# Controller research choices, separate from JEDEC device requirements.
DRAM_CONTROLLER_FIELDS = (
    "address_mapping", "scheduler", "scheduler_window", "queue_scope",
)

# HBM command occupancy and array constraints are separate quantities. These
# fields are unused by the analytical DRAM model; the JEDEC HBM command
# mode requires an explicitly authored clock and command format.
HBM_CONTROL_TIMING_FIELDS = (
    "command_clock_ns", "activate_command_cycles", "precharge_command_cycles",
    "column_command_cycles", "activate_to_read_ns", "activate_to_write_ns",
    "precharge_to_activate_ns", "minimum_active_ns", "maximum_active_ns", "activate_cycle_ns",
    "read_to_precharge_ns", "write_to_precharge_ns", "column_different_sid_ns",
    "read_to_write_command_ns", "precharge_to_precharge_ns", "refresh_interval_ns", "refresh_cycle_ns",
)


def _coerce_dram_kind(value: Any, generation: Any = "") -> tuple[MemoryKind, str]:
    """Canonicalize DRAM family and GDDR generation without guessing NAND."""

    raw = str(getattr(value, "value", value)).strip().upper().replace("-", "")
    generation_text = str(generation or "").strip().upper().replace("-", "")
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


def parse_physical_memory_config(values: Any) -> "DramConfig | NandConfig | SramConfig":
    """Parse one physical memory config through a single explicit dispatcher.

    Unknown kinds fail before entering the NAND branch. GDDR generations
    are declared separately as ``kind=GDDR, generation=GDDR7``.
    This parser does not select a family preset: omitted low-level fields
    retain the dataclass's illustrative defaults. Simulation components must
    pass physical_contract's explicit-field validation before using it.
    """

    from dataclasses import asdict, is_dataclass

    raw = asdict(values) if is_dataclass(values) else values
    if not isinstance(raw, Mapping):
        raise ValueError("physical_memory_config must be a DRAM/NAND config mapping")
    payload = dict(raw)
    kind_value = payload.get("kind")
    kind_name = str(getattr(kind_value, "value", kind_value or "")).strip().upper().replace("-", "")
    if kind_name in {"DDR", "LPDDR", "HBM", "GDDR"}:
        return DramConfig.from_mapping(payload)
    if kind_name in {"SSD", "HBF"}:
        return NandConfig.from_mapping(payload)
    if kind_name == "SRAM":
        return SramConfig.from_mapping(payload)
    raise ValueError(
        "physical_memory_config.kind must be DDR, LPDDR, HBM, GDDR, SSD, HBF or SRAM; got {}".format(
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
class SramConfig:
    """Ordinary byte-addressable SRAM with explicit shared read/write ports.

    Contiguous bursts are striped over ``ports``. Each port owns an equal
    share of the aggregate interface and directional byte rates, and a read
    and write use the same port calendar. Access latency is pipelined before
    byte service, rather than a DRAM row command or a NAND page operation.
    A pooled multi-stack instance is a research organization, not a claim
    that stack-local allocation and routing have been modeled.
    """

    capacity_bytes: int
    burst_bytes: int
    interface_bandwidth_gb_s: float
    read_bandwidth_gb_s: float
    write_bandwidth_gb_s: float
    read_latency_ns: float
    write_latency_ns: float
    ports: int
    max_outstanding_requests: int
    kind: MemoryKind | str = MemoryKind.SRAM
    max_expanded_segments: int = 1024
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SramConfig":
        return cls(**dict(values))

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, MemoryKind) else MemoryKind(str(self.kind).upper())
        except ValueError as exc:
            raise ValueError("SramConfig.kind must be SRAM") from exc
        if kind is not MemoryKind.SRAM:
            raise ValueError("SramConfig.kind must be SRAM")
        object.__setattr__(self, "kind", kind)
        for name in ("capacity_bytes", "burst_bytes", "ports", "max_outstanding_requests", "max_expanded_segments"):
            _positive_int(name, getattr(self, name))
        if self.capacity_bytes % self.burst_bytes:
            raise ValueError("SRAM capacity_bytes must be a multiple of burst_bytes")
        for name in ("interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s"):
            if _nonnegative(name, getattr(self, name)) <= 0:
                raise ValueError(name + " must be positive")
        for name in ("read_bandwidth_gb_s", "write_bandwidth_gb_s"):
            if getattr(self, name) > self.interface_bandwidth_gb_s:
                raise ValueError(name + " cannot exceed SRAM interface_bandwidth_gb_s")
        for name in ("read_latency_ns", "write_latency_ns"):
            _nonnegative(name, getattr(self, name))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("SRAM metadata must be a mapping")

    @property
    def effective_capacity_bytes(self) -> int:
        return self.capacity_bytes

    @property
    def physical_interface_bandwidth_gb_s(self) -> float:
        return float(self.interface_bandwidth_gb_s)

    @property
    def bandwidth_gb_s(self) -> float:
        return float(self.interface_bandwidth_gb_s)

    def directional_bandwidth_gb_s(self, operation: Operation | str) -> float:
        operation = operation if isinstance(operation, Operation) else Operation(str(operation).lower())
        if operation is Operation.ERASE:
            raise ValueError("SRAM does not support ERASE operations")
        return float(self.read_bandwidth_gb_s if operation is Operation.READ else self.write_bandwidth_gb_s)


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
    # Explicit GDDR rate mode uses it once, independently of the MT/s input.
    effective_pin_data_rate_gbps: Optional[float] = None
    bandwidth_input_mode: str = "bandwidth"
    lane_bandwidth_gb_s: Optional[float] = None
    interface_bandwidth_gb_s: Optional[float] = None
    read_bandwidth_gb_s: Optional[float] = None
    write_bandwidth_gb_s: Optional[float] = None
    data_lanes: Optional[int] = None
    interleave_bytes: Optional[int] = None
    address_mapping: str = field(default="bank_interleaved", kw_only=True)
    scheduler: str = field(default="fcfs", kw_only=True)
    scheduler_window: int = field(default=16, kw_only=True)
    queue_scope: str = field(default="device", kw_only=True)
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
    # Group comparisons share (stack, die, lane, rank). Column intervals are
    # command-start to command-start for READ->READ or WRITE->WRITE. ACT
    # intervals are start to start; its rolling window permits at most 4 ACTs.
    # WRITE->READ intervals run from the last write data end to read command
    # start and remain separate from data-bus direction turnaround.
    column_same_group_ns: float = 0.0
    column_different_group_ns: float = 0.0
    activate_same_group_ns: float = 0.0
    activate_different_group_ns: float = 0.0
    four_activate_window_ns: float = 0.0
    write_to_read_same_group_ns: float = 0.0
    write_to_read_different_group_ns: float = 0.0
    # Addressable bank/row organization can already include all stacked dies.
    # Physical die count then remains descriptive and must not multiply it.
    # Defaults retain the original generic DRAM organization and shared bus.
    bank_geometry_scope: str = "per_die"
    stack_interface_mode: str = "shared"
    control_mode: str = "analytical"
    command_clock_ns: float = 0.0
    activate_command_cycles: float = 0.0
    precharge_command_cycles: float = 0.0
    column_command_cycles: float = 0.0
    activate_to_read_ns: float = 0.0
    activate_to_write_ns: float = 0.0
    precharge_to_activate_ns: float = 0.0
    minimum_active_ns: float = 0.0
    maximum_active_ns: float = 0.0
    activate_cycle_ns: float = 0.0
    read_to_precharge_ns: float = 0.0
    write_to_precharge_ns: float = 0.0
    column_different_sid_ns: float = 0.0
    read_to_write_command_ns: float = 0.0
    precharge_to_precharge_ns: float = 0.0
    activated_row_msb: Optional[int] = None
    refresh_policy: str = "disabled"
    refresh_interval_ns: float = 0.0
    refresh_cycle_ns: float = 0.0

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
        if kind is MemoryKind.HBM and self.control_mode == "jedec_hbm":
            # Construction, deserialization and dataclasses.replace share the
            # same dependency rules as the public configuration editor. Source
            # declarations distinguish derived values from explicit experiments.
            from ._hbm_dependencies import normalize_hbm_dependencies

            normalized = normalize_hbm_dependencies(vars(self))
            for name, value in normalized.items():
                object.__setattr__(self, name, value)
            generation = self.generation
        if self.bank_geometry_scope not in {"per_die", "per_stack"}:
            raise ValueError("bank_geometry_scope must be per_die or per_stack")
        if self.stack_interface_mode not in {"shared", "independent"}:
            raise ValueError("stack_interface_mode must be shared or independent")
        if self.control_mode not in {"analytical", "jedec_hbm"}:
            raise ValueError("control_mode must be analytical or jedec_hbm")
        if self.address_mapping not in {"bank_interleaved", "row_local"}:
            raise ValueError("address_mapping must be bank_interleaved or row_local")
        if self.address_mapping == "row_local" and self.interleave_bytes is not None:
            raise ValueError("row_local address_mapping requires interleave_bytes=None")
        if self.scheduler not in {"fcfs", "row_hit_first", "read_write_batch"}:
            raise ValueError("scheduler must be fcfs, row_hit_first or read_write_batch")
        _positive_int("scheduler_window", self.scheduler_window)
        if self.queue_scope not in {"device", "channel"}:
            raise ValueError("queue_scope must be device or channel")
        if self.refresh_policy not in {"disabled", "all_bank"}:
            raise ValueError("refresh_policy must be disabled or all_bank")
        for name in HBM_CONTROL_TIMING_FIELDS:
            _nonnegative(name, getattr(self, name))
        if self.activated_row_msb is not None:
            if (isinstance(self.activated_row_msb, bool)
                    or not isinstance(self.activated_row_msb, int)
                    or self.activated_row_msb < 0):
                if kind is MemoryKind.HBM and self.control_mode == "jedec_hbm":
                    raise _hbm_config_error(self, "activated_row_msb", category="invalid_input",
                        rule="invalid_row_address_msb", label="有效行地址最高位",
                        requirement_en="activated_row_msb must be a non-negative integer or None",
                        requirement_zh="行地址最高位必须是非负整数，或在不使用该字段时填 null。",
                        remedy_zh="使用所选代际和 rows_per_bank 对应的有效行地址最高位；建议通过公共配置入口重新生成。",
                        remedy_en="Use the effective row-address MSB for this generation and rows_per_bank, or regenerate through the public config entry point.",
                        related_values={"rows_per_bank": self.rows_per_bank})
                raise ValueError("activated_row_msb must be a non-negative integer or None")
        if self.control_mode == "jedec_hbm":
            _positive_int("rows_per_bank", self.rows_per_bank)
            if kind is not MemoryKind.HBM or generation not in {"HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4"}:
                raise ValueError("jedec_hbm control requires a supported HBM generation")
            if self.command_clock_ns <= 0:
                raise ValueError("jedec_hbm control requires a positive command_clock_ns")
            hbm2 = generation in {"HBM2", "HBM2E"}
            expected = (2.0, 1.0, 1.0) if hbm2 else (1.5, 0.5, 1.0)
            actual = (self.activate_command_cycles, self.precharge_command_cycles, self.column_command_cycles)
            if actual != expected:
                raise ValueError("jedec_hbm command occupancy conflicts with generation (ACT/PRE/RD-WR cycles)")
            latency_limits = ((3, 48), (1, 16)) if hbm2 else (((17, 90), (4, 19)) if generation == "HBM4" else ((4, 63), (4, 16)))
            for name, (minimum, maximum) in zip(("read_latency_ns", "write_latency_ns"), latency_limits):
                cycles = _nonnegative(name, getattr(self, name)) / self.command_clock_ns
                if not math.isclose(cycles, round(cycles), rel_tol=0, abs_tol=1e-9) or not minimum <= round(cycles) <= maximum:
                    raise ValueError(f"jedec_hbm {name} must encode {minimum}..{maximum} whole command clocks; actual vendor support is separate")
            if (self.pseudo_channels_per_channel != 2 or self.subchannels_per_channel != 1
                    or self.ranks_per_channel != 1 or self.data_lanes not in (None, self.channels * 2)
                    or self.burst_bytes != 32 or self.data_width_bits != (64 if hbm2 else 32)
                    or self.row_bytes != 1024 or self.bank_geometry_scope != "per_stack"
                    or self.stack_interface_mode != "independent"):
                expected_geometry = {
                    "pseudo_channels_per_channel": 2, "subchannels_per_channel": 1,
                    "ranks_per_channel": 1, "data_lanes": (None, self.channels * 2),
                    "burst_bytes": 32, "data_width_bits": 64 if hbm2 else 32,
                    "row_bytes": 1024, "bank_geometry_scope": "per_stack",
                    "stack_interface_mode": "independent",
                }
                conflicts = {name: {"actual": getattr(self, name), "required": required}
                             for name, required in expected_geometry.items()
                             if (getattr(self, name) not in required if name == "data_lanes"
                                 else getattr(self, name) != required)}
                name = next(iter(conflicts))
                required = conflicts[name]["required"]
                raise _hbm_config_error(self, name, category="model_limit",
                    rule="unsupported_pc_geometry",
                    requirement_en="jedec_hbm control requires PC-mode HBM geometry and independent stack interfaces",
                    requirement_zh=f"当前控制器仅支持伪通道模式和独立堆栈接口；{name} 需要为 {required}。",
                    remedy_zh="按所选 HBM 代际重新生成几何配置，修正列出的冲突字段；不能把等效组织标为此控制器支持的真实组织。",
                    remedy_en="Regenerate geometry for the selected HBM generation and correct the listed conflicts; an equivalent organization is not supported physical geometry.",
                    related_values={"conflicts": conflicts, "channels": self.channels})
            if self.activated_row_msb != (None if hbm2 else (self.rows_per_bank - 1).bit_length() - 1):
                expected_msb = None if hbm2 else (self.rows_per_bank - 1).bit_length() - 1
                raise _hbm_config_error(self, "activated_row_msb", category="model_limit",
                    rule="row_address_msb_conflict", label="有效行地址最高位",
                    requirement_en="jedec_hbm activated_row_msb must identify the effective row address MSB",
                    requirement_zh=f"当前控制器按有效行地址选择激活页；rows_per_bank={self.rows_per_bank} 时，此字段需要为 {'null' if expected_msb is None else expected_msb}。",
                    remedy_zh=f"将 activated_row_msb 设为 {'null' if expected_msb is None else expected_msb}，或通过公共配置入口更新行数及其依赖参数。",
                    remedy_en=f"Set activated_row_msb to {expected_msb}, or update the row count and dependencies through the public config entry point.",
                    related_values={"rows_per_bank": self.rows_per_bank, "required_value": expected_msb})
        elif self.refresh_policy != "disabled":
            raise ValueError("automatic refresh requires jedec_hbm control")
        if self.refresh_policy == "all_bank" and (self.refresh_interval_ns <= 0 or self.refresh_cycle_ns <= 0):
            raise ValueError("all_bank refresh requires positive refresh_interval_ns and refresh_cycle_ns")
        if 0 < self.maximum_active_ns < self.minimum_active_ns:
            raise ValueError("maximum_active_ns cannot be shorter than minimum_active_ns")
        for name in ("channels", "subchannels_per_channel", "pseudo_channels_per_channel", "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank", "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes", "data_width_bits"):
            _positive_int(name, getattr(self, name))
        _positive_int("max_outstanding_requests", self.max_outstanding_requests)
        _positive_int("max_expanded_segments", self.max_expanded_segments)
        if self.data_lanes is not None:
            _positive_int("data_lanes", self.data_lanes)
        if self.row_bytes % self.burst_bytes:
            raise ValueError("row_bytes must be a multiple of burst_bytes")
        if self.capacity_bytes is not None and self.capacity_bytes % self.burst_bytes:
            # Physical requests are expanded to complete bursts.  A capacity
            # ending in the middle of one burst would therefore allow the
            # last expanded transaction to cross the declared boundary.
            if kind is MemoryKind.HBM and self.control_mode == "jedec_hbm":
                raise _hbm_config_error(self, "capacity_bytes", category="model_limit",
                    rule="capacity_burst_alignment", label="可用容量预算", unit="B",
                    requirement_en="capacity_bytes must be a multiple of burst_bytes",
                    requirement_zh=f"当前模型按完整突发传输访问内存，容量预算必须是 {self.burst_bytes} 字节的整数倍。",
                    remedy_zh=f"将容量预算改为 {self.burst_bytes} 字节的整数倍，且不超过已声明几何的容量。",
                    remedy_en=f"Set capacity_bytes to a multiple of {self.burst_bytes} bytes within the declared physical capacity.",
                    related_values={"burst_bytes": self.burst_bytes})
            raise ValueError("capacity_bytes must be a multiple of burst_bytes")
        if self.interleave_bytes is not None:
            _positive_int("interleave_bytes", self.interleave_bytes)
            if self.interleave_bytes % self.burst_bytes:
                raise ValueError("interleave_bytes must be a multiple of burst_bytes")
            bank_bytes = self.rows_per_bank * self.row_bytes
            if self.interleave_bytes > bank_bytes or bank_bytes % self.interleave_bytes:
                raise ValueError("interleave_bytes must divide the capacity of one bank")
        for name in ("data_rate_mt_s", "open_ns", "close_ns", "read_latency_ns", "write_latency_ns", "burst_interval_ns", "read_recovery_ns", "write_recovery_ns", "read_to_write_ns", "write_to_read_ns"):
            _nonnegative(name, getattr(self, name))
        for name in DRAM_BANK_GROUP_TIMING_FIELDS:
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
        if self.control_mode == "jedec_hbm":
            transfers_per_ck = 2 if generation in {"HBM2", "HBM2E"} else 4
            expected_clock = transfers_per_ck * self.data_width_bits * self.lane_count / (8.0 * physical_bandwidth)
            if not math.isclose(self.command_clock_ns, expected_clock, rel_tol=1e-10):
                raise _hbm_config_error(self, "command_clock_ns", category="standard_limit",
                    rule="clock_interface_conflict",
                    requirement_en="command_clock_ns conflicts with the JEDEC CK/DQ ratio and physical interface bandwidth",
                    requirement_zh=f"本代际每个命令时钟对应 {transfers_per_ck} 次数据传输；当前接口 {physical_bandwidth:g} GB/s、{self.lane_count} 条数据通道、每通道 {self.data_width_bits} 位，要求时钟周期为 {expected_clock:g} ns。",
                    remedy_zh=f"将 command_clock_ns 设为 {expected_clock:g} ns，或通过公共配置入口一致地更新物理接口带宽和时序；读写方向限速不会改变物理时钟。",
                    remedy_en=f"Set command_clock_ns to {expected_clock:g} ns or consistently update the physical interface and dependent timings; directional caps do not change the physical clock.",
                    related_values={"required_clock_ns": expected_clock, "physical_interface_bandwidth_gb_s": physical_bandwidth,
                                    "lane_count": self.lane_count, "data_width_bits": self.data_width_bits,
                                    "transfers_per_command_clock": transfers_per_ck},
                    source="JESD235B" if hbm2 else "JESD270-4A" if generation == "HBM4" else "JESD238")
        if any(value is not None and value > physical_bandwidth
               for value in (self.read_bandwidth_gb_s, self.write_bandwidth_gb_s)):
            if kind is MemoryKind.HBM and self.control_mode == "jedec_hbm":
                name = next(name for name in ("read_bandwidth_gb_s", "write_bandwidth_gb_s")
                            if getattr(self, name) is not None and getattr(self, name) > physical_bandwidth)
                raise _hbm_config_error(self, name, category="invalid_input",
                    rule="directional_bandwidth_exceeds_interface", label="读方向限速" if name.startswith("read") else "写方向限速",
                    requirement_en="directional bandwidth cannot exceed the physical interface bandwidth",
                    requirement_zh=f"方向限速是物理接口之上的模型预算，不能超过当前物理接口的 {physical_bandwidth:g} GB/s。",
                    remedy_zh=f"将该方向限速设为不大于 {physical_bandwidth:g} GB/s；若要更换物理接口速率，请通过公共入口同时更新接口和依赖时序。",
                    remedy_en=f"Set the directional cap to at most {physical_bandwidth:g} GB/s, or update the physical interface and dependent timings through the public config entry point.",
                    related_values={"physical_interface_bandwidth_gb_s": physical_bandwidth,
                                    "read_bandwidth_gb_s": self.read_bandwidth_gb_s,
                                    "write_bandwidth_gb_s": self.write_bandwidth_gb_s})
            raise ValueError("directional bandwidth cannot exceed the physical interface bandwidth")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if self.capacity_bytes is not None and self.capacity_bytes > self.computed_capacity_bytes:
            if kind is MemoryKind.HBM and self.control_mode == "jedec_hbm":
                raise _hbm_config_error(self, "capacity_bytes", category="invalid_input",
                    rule="capacity_exceeds_geometry", label="可用容量预算", unit="B",
                    requirement_en="capacity_bytes cannot exceed the declared physical geometry",
                    requirement_zh=f"可用容量预算不能超过已声明几何可寻址的 {self.computed_capacity_bytes} 字节。",
                    remedy_zh=f"将 capacity_bytes 降至不大于 {self.computed_capacity_bytes} 字节，或通过公共入口选择足够的堆栈数与合法组织；容量预算本身不会增加物理行数。",
                    remedy_en=f"Set capacity_bytes to at most {self.computed_capacity_bytes} bytes, or select sufficient stacks and a supported organization through the public config entry point; a capacity budget does not add physical rows.",
                    related_values={"computed_capacity_bytes": self.computed_capacity_bytes,
                                    "stacks": self.stacks, "lane_count": self.lane_count,
                                    "banks_per_lane": self.bank_count, "rows_per_bank": self.rows_per_bank,
                                    "row_bytes": self.row_bytes})
            raise ValueError("capacity_bytes cannot exceed the declared physical geometry")
        if self.capacity_bytes is None:
            object.__setattr__(self, "capacity_bytes", self.computed_capacity_bytes)

    @property
    def bank_group_timing_enabled(self) -> bool:
        """Whether any explicitly authored group/rank timing gate is enabled."""
        return any(getattr(self, name) > 0 for name in DRAM_BANK_GROUP_TIMING_FIELDS)

    @property
    def bank_count(self) -> int:
        return self.ranks_per_channel * self.bank_groups_per_rank * self.banks_per_group

    @property
    def lanes_per_stack(self) -> int:
        # data_lanes is the authoritative escape hatch for non-multiplicative
        # package organizations (notably HBM pseudo-channel implementations).
        # Stacked dies add capacity and array choices; they do not imply new
        # external data wires.  Use data_lanes for an explicitly wired design.
        return self.data_lanes or (self.channels * self.subchannels_per_channel * self.pseudo_channels_per_channel)

    @property
    def lane_count(self) -> int:
        return self.lanes_per_stack * (self.stacks if self.stack_interface_mode == "independent" else 1)

    @property
    def addressable_die_count(self) -> int:
        return 1 if self.bank_geometry_scope == "per_stack" else self.dies_per_stack

    @property
    def addressable_stack_count(self) -> int:
        # Independent stack interfaces are already represented in lane_count.
        return 1 if self.stack_interface_mode == "independent" else self.stacks

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
        return self.addressable_stack_count * self.addressable_die_count * self.lane_count * self.bank_count * self.rows_per_bank * self.row_bytes

    @property
    def effective_capacity_bytes(self) -> int:
        return self.capacity_bytes or self.computed_capacity_bytes


@dataclass(frozen=True)
class NandWearConfig:
    """Explicit erase/program accounting without FTL or implicit erasures."""

    rated_pe_cycles: int = 3000
    initial_pe_cycles: int = 0
    program_latency_growth_at_rated_pe: float = 0.0
    max_program_latency_scale: float = 2.0
    assumption: str = ""

    def __post_init__(self):
        _positive_int("rated_pe_cycles", self.rated_pe_cycles)
        if isinstance(self.initial_pe_cycles, bool) or not isinstance(self.initial_pe_cycles, int) or self.initial_pe_cycles < 0:
            raise ValueError("initial_pe_cycles must be a non-negative integer")
        _nonnegative("program_latency_growth_at_rated_pe", self.program_latency_growth_at_rated_pe)
        if _nonnegative("max_program_latency_scale", self.max_program_latency_scale) < 1:
            raise ValueError("max_program_latency_scale must be at least one")
        if not isinstance(self.assumption, str) or not self.assumption.strip():
            raise ValueError("NAND wear feedback requires an explicit assumption label")


@dataclass(frozen=True)
class NandConfig:
    """Parameterized page NAND organization for SSD or HBF.

    Host bandwidths are aggregate per direction. Internal channels are
    distributed evenly across host channels; full duplex separates their
    read/write calendars. Internal bandwidth is per internal channel. Queue
    limits apply to the selected device or host-channel scope; the write
    limit is a subset of the combined limit. OCP mode covers complete-unit
    IO, not AXI packets, and derives independent banks from geometry.
    """

    kind: MemoryKind | str = MemoryKind.SSD
    # Explicit execution contracts, not automatic device-name dispatch.
    protocol_mode: str = field(default="generic_nand", kw_only=True)
    host_channels: int = field(default=1, kw_only=True)
    host_full_duplex: bool = field(default=False, kw_only=True)
    read_cache_pages_per_bank: int = field(default=0, kw_only=True)
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
    # Explicit directional ceilings are independent of the shared host link.
    host_read_bandwidth_gb_s: float = field(default=4.0, kw_only=True)
    host_write_bandwidth_gb_s: float = field(default=4.0, kw_only=True)
    page_read_ns: float = 50000.0
    page_program_ns: float = 800000.0
    block_erase_ns: float = 3000000.0
    front_ns: float = 0.0
    # Host-side write staging capacity.  Zero preserves write-through
    # behavior; positive values model finite page-merging buffering without
    # adding an FTL or garbage-collection state machine.
    write_buffer_bytes: int = field(default=0, kw_only=True)
    wear_model: Optional[NandWearConfig | Mapping[str, Any]] = field(default=None, kw_only=True)
    detailed_observability: bool = field(default=False, kw_only=True)
    partial_page_policy: str = "read_modify_write"
    capacity_bytes: Optional[int] = None
    max_outstanding_requests: int = 32
    queue_scope: str = field(default="device", kw_only=True)
    max_outstanding_write_requests: Optional[int] = field(default=None, kw_only=True)
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
        for name, cls in (("wear_model", NandWearConfig),):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, cls(**dict(value)))
            elif value is not None and not isinstance(value, cls):
                raise ValueError(name + " must be its typed config, a mapping or None")
        if not isinstance(self.detailed_observability, bool):
            raise ValueError("detailed_observability must be a boolean")
        for name in ("channels", "targets_per_channel", "dies_per_target", "luns_per_die", "planes_per_lun", "blocks_per_plane", "pages_per_block", "page_bytes", "host_granularity_bytes"):
            _positive_int(name, getattr(self, name))
        if self.parallel_units is not None:
            _positive_int("parallel_units", self.parallel_units)
        if self.internal_transfer_bytes is not None:
            _positive_int("internal_transfer_bytes", self.internal_transfer_bytes)
        if not isinstance(self.planes_independent, bool):
            raise ValueError("planes_independent must be a boolean")
        if self.protocol_mode not in {"generic_nand", "ocp_hbf_0_7"}:
            raise ValueError("protocol_mode must be generic_nand or ocp_hbf_0_7")
        if self.queue_scope not in {"device", "host_channel"}:
            raise ValueError("NAND queue_scope must be device or host_channel")
        if self.protocol_mode == "generic_nand" and self.queue_scope != "device":
            raise ValueError("generic_nand requires queue_scope=device; host-channel page-command admission is supported only by ocp_hbf_0_7")
        _positive_int("host_channels", self.host_channels)
        if self.host_channels > self.channels or self.channels % self.host_channels:
            raise ValueError("internal channels must divide evenly among host_channels")
        if not isinstance(self.host_full_duplex, bool):
            raise ValueError("host_full_duplex must be a boolean")
        if (isinstance(self.read_cache_pages_per_bank, bool)
                or not isinstance(self.read_cache_pages_per_bank, int)
                or self.read_cache_pages_per_bank < 0):
            raise ValueError("read_cache_pages_per_bank must be a non-negative integer")
        if self.protocol_mode == "ocp_hbf_0_7":
            if kind is not MemoryKind.HBF:
                raise ValueError("ocp_hbf_0_7 requires kind=HBF")
            if self.parallel_units is not None:
                raise ValueError("ocp_hbf_0_7 derives bank parallelism from geometry; parallel_units must be None")
            if not self.planes_independent:
                raise ValueError("ocp_hbf_0_7 requires independent bank resources (planes_independent=True)")
            if self.queue_scope != "host_channel":
                raise ValueError("ocp_hbf_0_7 requires queue_scope=host_channel")
            if self.page_bytes != 4096 or self.host_granularity_bytes != 64:
                raise ValueError("ocp_hbf_0_7 requires 4096-byte pages and 64-byte host granularity")
            if self.host_channels > 16 or not self.host_full_duplex:
                raise ValueError("ocp_hbf_0_7 requires at most 16 full-duplex host channels")
            if self.read_cache_pages_per_bank != 2:
                raise ValueError("ocp_hbf_0_7 requires two read-cache pages per bank")
            if self.write_buffer_bytes != 0:
                raise ValueError("ocp_hbf_0_7 ordinary writes complete on media; host-ack write buffering is unsupported")
        if self.capacity_bytes is not None:
            _positive_int("capacity_bytes", self.capacity_bytes)
            if self.capacity_bytes % self.page_bytes:
                # NAND requests are expanded to complete pages.  Accepting a
                # tail capacity that ends mid-page lets the last physical
                # page cross the declared device boundary.
                raise ValueError("capacity_bytes must be a multiple of page_bytes")
        _positive_int("max_outstanding_requests", self.max_outstanding_requests)
        if self.max_outstanding_write_requests is not None:
            _positive_int("max_outstanding_write_requests", self.max_outstanding_write_requests)
            if self.max_outstanding_write_requests > self.max_outstanding_requests:
                raise ValueError("max_outstanding_write_requests cannot exceed max_outstanding_requests")
        _positive_int("max_expanded_segments", self.max_expanded_segments)
        if (isinstance(self.write_buffer_bytes, bool)
                or not isinstance(self.write_buffer_bytes, int)
                or self.write_buffer_bytes < 0):
            raise ValueError("write_buffer_bytes must be a non-negative integer")
        for name in ("host_bandwidth_gb_s", "internal_bandwidth_gb_s", "page_read_ns", "page_program_ns", "block_erase_ns", "front_ns"):
            value = _nonnegative(name, getattr(self, name))
            if name.endswith("bandwidth_gb_s") and value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("host_read_bandwidth_gb_s", "host_write_bandwidth_gb_s"):
            value = _nonnegative(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if value > float(self.host_bandwidth_gb_s):
                raise ValueError(f"{name} cannot exceed host_bandwidth_gb_s")
        if self.partial_page_policy not in {"read_modify_write", "reject"}:
            raise ValueError("partial_page_policy must be read_modify_write or reject")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if "buffered_write_mode" in self.metadata:
            raise ValueError("metadata.buffered_write_mode is no longer supported; set write_buffer_bytes to a positive capacity or zero")
        if self.capacity_bytes is not None and self.capacity_bytes > self.computed_capacity_bytes:
            raise ValueError("capacity_bytes cannot exceed the declared physical geometry")
        if self.capacity_bytes is None:
            object.__setattr__(self, "capacity_bytes", self.computed_capacity_bytes)
        if self.write_buffer_bytes > self.effective_capacity_bytes:
            raise ValueError("write_buffer_bytes cannot exceed device capacity")

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

    def host_bandwidth_for(self, operation: Operation | str) -> float:
        """Return the authored host-link ceiling for one direction."""
        op = operation if isinstance(operation, Operation) else Operation(str(operation).lower())
        if op is Operation.READ:
            return float(self.host_read_bandwidth_gb_s)
        if op is Operation.WRITE:
            return float(self.host_write_bandwidth_gb_s)
        raise ValueError("host bandwidth requires a read or write operation")

    @property
    def buffered_write_mode(self) -> bool:
        """A positive staging capacity enables host-ack/page-merge semantics."""
        return self.write_buffer_bytes > 0


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
        return int(self.counters["internal_transfer_bytes"])

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
    def write_buffer_wait_ns(self) -> float:
        return float(self.counters.get("write_buffer_wait_ns", 0.0))

    @property
    def write_buffer_bytes(self) -> int:
        return int(self.counters.get("write_buffer_bytes", 0))

    @property
    def write_buffer_peak_bytes(self) -> int:
        return int(self.counters.get("write_buffer_peak_bytes", 0))

    @property
    def write_buffer_occupancy_bytes(self) -> int:
        return int(self.counters.get("write_buffer_occupancy_bytes", 0))

    @property
    def merge_bytes(self) -> int:
        return int(self.counters.get("merge_bytes", 0))

    @property
    def host_ack_completion_ns(self) -> Optional[float]:
        value = self.counters.get("host_ack_completion_ns")
        return None if value is None else float(value)

    @property
    def media_drain_completion_ns(self) -> Optional[float]:
        value = self.counters.get("media_drain_completion_ns")
        return None if value is None else float(value)

    @property
    def staging_read_hits(self) -> int:
        return int(self.counters.get("staging_read_hits", 0))

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
        return int(self.counters["internal_transfer_bytes"])

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

    @property
    def write_buffer_wait_ns(self) -> float:
        return float(self.counters.get("write_buffer_wait_ns", 0.0))

    @property
    def write_buffer_bytes(self) -> int:
        return int(self.counters.get("write_buffer_bytes", 0))

    @property
    def write_buffer_peak_bytes(self) -> int:
        return int(self.counters.get("write_buffer_peak_bytes", 0))

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
    """Build an illustrative low-level HBM example, not a hardware preset.

    It sets 8 channels x 2 pseudo-channels, while the parser uses generic
    DramConfig defaults. Authored simulation components
    carry complete physical configs from the hardware preset/configuration;
    they must not use these example defaults as a missing-config fallback.
    """
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
    kwargs.setdefault("host_read_bandwidth_gb_s", 112.0)
    kwargs.setdefault("host_write_bandwidth_gb_s", 112.0)
    kwargs.setdefault("internal_bandwidth_gb_s", 112.0)
    return NandConfig(**kwargs)


def make_hbf_config(**kwargs: Any) -> NandConfig:
    kwargs["kind"] = MemoryKind.HBF
    kwargs.setdefault("host_bandwidth_gb_s", 488.0)
    kwargs.setdefault("host_read_bandwidth_gb_s", 488.0)
    kwargs.setdefault("host_write_bandwidth_gb_s", 27.2)
    kwargs.setdefault("internal_bandwidth_gb_s", 488.0)
    kwargs.setdefault("page_read_ns", 4_000.0)
    kwargs.setdefault("page_program_ns", 75_000.0)
    kwargs.setdefault("front_ns", 800.0)
    kwargs.setdefault("write_buffer_bytes", 0)
    return NandConfig(**kwargs)


__all__ = [
    "AccessRequest", "AddressMapping", "BatchResult", "DramConfig", "MemoryKind", "NandConfig", "SramConfig",
    "DRAM_BANK_GROUP_TIMING_FIELDS", "DRAM_CONTROLLER_FIELDS",
    "NandWearConfig",
    "Operation", "Segment", "StageTiming", "TransactionResult",
    "make_ddr_config", "make_lpddr_config", "make_hbm_config", "make_gddr_config",
    "parse_physical_memory_config",
    "make_ssd_config", "make_hbf_config",
]
