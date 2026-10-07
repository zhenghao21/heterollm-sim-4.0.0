"""Required physical contracts at simulation and memory-service boundaries.

Incomplete components may exist in the editor. They must never execute by
substituting a bandwidth-only service or the memory core's example defaults.
"""

from dataclasses import asdict, is_dataclass
from typing import Mapping

from .memory_types import DramConfig, parse_physical_memory_config


DRAM_COMPONENT_KINDS = frozenset({
    "hbm", "hbm_stack", "gddr", "gddr_memory", "dram", "ddr", "ddr_memory",
    "lpddr", "lpddr_memory", "cxl_memory", "host_memory", "memory",
})
NAND_COMPONENT_KINDS = frozenset({"hbf", "ssd", "high_io_ssd", "nvme"})
PHYSICAL_MEMORY_COMPONENT_KINDS = DRAM_COMPONENT_KINDS | NAND_COMPONENT_KINDS

_DRAM_REQUIRED = frozenset({
    "kind", "channels", "subchannels_per_channel", "pseudo_channels_per_channel",
    "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank",
    "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes",
    "open_ns", "close_ns", "read_latency_ns", "write_latency_ns",
    "burst_interval_ns", "read_recovery_ns", "write_recovery_ns",
    "read_to_write_ns", "write_to_read_ns", "max_outstanding_requests",
})
_NAND_REQUIRED = frozenset({
    "kind", "channels", "targets_per_channel", "dies_per_target", "luns_per_die",
    "planes_per_lun", "blocks_per_plane", "pages_per_block", "page_bytes",
    "host_granularity_bytes", "host_bandwidth_gb_s", "internal_bandwidth_gb_s",
    "page_read_ns", "page_program_ns", "block_erase_ns", "front_ns",
    "planes_independent", "partial_page_policy", "max_outstanding_requests",
})


def parse_required_physical_config(component_id, kind, raw):
    """Validate authored fields before dataclass defaults can fill omissions."""
    prefix = "component {} ({}): ".format(component_id, kind)
    if kind not in PHYSICAL_MEMORY_COMPONENT_KINDS:
        raise ValueError(prefix + "no supported physical memory core; select a DRAM or NAND component")
    if is_dataclass(raw):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise ValueError(prefix + "physical_memory_config is required; reload a complete hardware preset or configure the physical memory core. Aggregate-cost fallback is disabled")
    required = _NAND_REQUIRED if kind in NAND_COMPONENT_KINDS else _DRAM_REQUIRED
    missing = sorted(name for name in required if raw.get(name) is None)
    if kind in DRAM_COMPONENT_KINDS and not (
        raw.get("interface_bandwidth_gb_s") is not None
        or raw.get("lane_bandwidth_gb_s") is not None
        or (raw.get("data_width_bits") is not None and (
            raw.get("data_rate_mt_s") is not None or raw.get("effective_pin_data_rate_gbps") is not None
        ))
    ):
        missing.append("interface bandwidth or data width and data rate")
    if missing:
        raise ValueError(prefix + "physical_memory_config missing explicit fields: " + ", ".join(missing))
    try:
        parsed = parse_physical_memory_config(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(prefix + "invalid physical_memory_config: " + str(exc)) from exc
    allowed = (
        {"HBF"} if kind == "hbf" else {"SSD"} if kind in NAND_COMPONENT_KINDS
        else {"HBM"} if kind in {"hbm", "hbm_stack"}
        else {"GDDR"} if kind in {"gddr", "gddr_memory"}
        else {"DDR"} if kind in {"ddr", "ddr_memory"}
        else {"LPDDR"} if kind in {"lpddr", "lpddr_memory"}
        else {"DDR", "LPDDR", "HBM", "GDDR"}
    )
    if parsed.kind.value not in allowed:
        raise ValueError(prefix + "physical_memory_config.kind must be " + "/".join(sorted(allowed)))
    return parsed


def require_physical_memory_config(component):
    """Return a validated core config, or identify the incomplete component."""
    prefix = "component {} ({}): ".format(component.component_id, component.kind)
    parsed = parse_required_physical_config(
        component.component_id, component.normalized_kind,
        component.metadata.get("physical_memory_config"),
    )
    capacity = component.capacity_bytes
    if capacity <= 0 or parsed.capacity_bytes != capacity:
        raise ValueError(prefix + "capacity_bytes must be positive and equal physical_memory_config capacity (component={}, physical={})".format(capacity, parsed.capacity_bytes))
    if isinstance(parsed, DramConfig):
        physical_gbps = parsed.physical_interface_bandwidth_gb_s * 8.0
    else:
        physical_gbps = parsed.host_bandwidth_gb_s * 8.0
    if component.shared_bandwidth_gbps > physical_gbps * (1.0 + 1e-12):
        raise ValueError(prefix + "component bandwidth exceeds physical memory interface bandwidth")
    return parsed
