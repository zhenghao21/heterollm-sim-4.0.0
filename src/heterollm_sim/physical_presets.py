"""Authored standard organizations and explicit analytical memory controls.

These builders live below component/architecture catalogs. They never run as
an execution fallback: a simulation component must carry the complete result
in ``metadata.physical_memory_config`` and pass ``physical_contract``.
HBM reference geometry comes from JEDEC addressing tables; product-specific
organization and array timing can still be unknown. Other analytical controls
are labelled as assumptions. Low-level ``memory_types.make_*`` helpers remain
toy defaults.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from math import ceil, isfinite
from typing import Any, Dict, Mapping

from .memory_types import DramConfig, NandConfig, DRAM_BANK_GROUP_TIMING_FIELDS, HBM_CONTROL_TIMING_FIELDS
from .hbm_geometry import make_hbm_geometry_config, normalize_hbm_generation
from .physical_contract import parse_execution_physical_config, reconfigure_physical_config


# Complete configuration sources for the research tools. Values are written
# here, rather than inherited from memory_types.make_* example defaults. A
# profile is a model assumption unless a separate published input is supplied.
_DRAM_RESEARCH_BASE = {
    "kind": "HBM", "generation": "", "channels": 8,
    "subchannels_per_channel": 1, "pseudo_channels_per_channel": 2,
    "stacks": 1, "dies_per_stack": 1, "ranks_per_channel": 1,
    "bank_geometry_scope": "per_die", "stack_interface_mode": "shared",
    "control_mode": "analytical", "activated_row_msb": None, "refresh_policy": "disabled",
    **{name: 0.0 for name in HBM_CONTROL_TIMING_FIELDS},
    "bank_groups_per_rank": 1, "banks_per_group": 8,
    "rows_per_bank": 524288, "row_bytes": 8192, "burst_bytes": 64,
    "data_width_bits": 64, "data_rate_mt_s": 3200.0,
    "effective_pin_data_rate_gbps": None, "bandwidth_input_mode": "bandwidth",
    "lane_bandwidth_gb_s": None, "interface_bandwidth_gb_s": None,
    "read_bandwidth_gb_s": None, "write_bandwidth_gb_s": None,
    "data_lanes": None, "interleave_bytes": None,
    "open_ns": 14.0, "close_ns": 14.0, "read_latency_ns": 14.0,
    "write_latency_ns": 12.0, "burst_interval_ns": 0.0,
    "read_recovery_ns": 0.0, "write_recovery_ns": 15.0,
    "read_to_write_ns": 0.0, "write_to_read_ns": 0.0,
    "capacity_bytes": 512 * 1024 ** 3, "max_outstanding_requests": 64,
    "max_expanded_segments": 1000000,
    "column_same_group_ns": 0.0, "column_different_group_ns": 0.0,
    "activate_same_group_ns": 0.0, "activate_different_group_ns": 0.0,
    "four_activate_window_ns": 0.0,
    "write_to_read_same_group_ns": 0.0, "write_to_read_different_group_ns": 0.0,
    "metadata": {
        "parameter_basis": "synthetic_research_assumptions",
        "configuration_scope": "explicit_physical_memory_research",
        "geometry_basis": "equivalent geometry; not a vendor die or bank specification",
        "timing_basis": "analytical zero bank-group timing control; other array timings are analytical assumptions",
    },
}
_NAND_RESEARCH_BASE = {
    "kind": "HBF", "generation": "", "channels": 1,
    "targets_per_channel": 1, "dies_per_target": 1, "luns_per_die": 1,
    "planes_per_lun": 1, "planes_independent": False, "parallel_units": None,
    "blocks_per_plane": 2048, "pages_per_block": 256, "page_bytes": 16384,
    "host_granularity_bytes": 4096, "internal_transfer_bytes": None,
    "host_bandwidth_gb_s": 488.0, "internal_bandwidth_gb_s": 488.0,
    "host_read_bandwidth_gb_s": 488.0, "host_write_bandwidth_gb_s": 27.2,
    "page_read_ns": 4000.0, "page_program_ns": 75000.0,
    "block_erase_ns": 3000000.0, "front_ns": 800.0,
    "write_buffer_bytes": 0, "wear_model": None, "detailed_observability": False,
    "partial_page_policy": "read_modify_write", "capacity_bytes": 8 * 1024 ** 3,
    "max_outstanding_requests": 32, "max_expanded_segments": 1000000,
    "metadata": {
        "parameter_basis": "synthetic_research_assumptions",
        "configuration_scope": "explicit_physical_memory_research",
        "geometry_basis": "equivalent channel/plane/page geometry; not a disclosed device layout",
        "timing_basis": "explicit read, program and erase assumptions; no external calibration",
    },
}
_SYNTHETIC_BANK_GROUPS = {
    "bank_groups_per_rank": 4, "banks_per_group": 2,
    "column_same_group_ns": 4.0, "column_different_group_ns": 1.0,
    "activate_same_group_ns": 8.0, "activate_different_group_ns": 2.0,
    "four_activate_window_ns": 16.0,
    "write_to_read_same_group_ns": 4.0, "write_to_read_different_group_ns": 2.0,
}


def _preset(base, **changes):
    result = deepcopy(base)
    metadata = changes.pop("metadata", {})
    result.update(changes)
    result["metadata"].update(metadata)
    return result


_PHYSICAL_CONFIG_CATALOG = {
    "hbm_analytical_zero": _preset(_DRAM_RESEARCH_BASE),
    "hbm_group_geometry_control": _preset(_DRAM_RESEARCH_BASE,
        bank_groups_per_rank=4, banks_per_group=2),
    "hbm_synthetic_bank_groups": _preset(_DRAM_RESEARCH_BASE, **_SYNTHETIC_BANK_GROUPS,
        metadata={"timing_basis": "synthetic bank-group timings in ns: column 4/1, ACT 8/2, four-ACT 16, WRITE-to-READ 4/2; not JEDEC or vendor calibration",
                  "bank_group_timing_assumptions": [
                      "Same-group commands are more constrained than different-group commands: column spacing 4/1 ns represents shared internal group resources.",
                      "ACT spacing is 8/2 ns for same/different groups, with at most four ACT commands per 16 ns window in one stack/die/lane/rank scope.",
                      "WRITE-to-READ 4/2 ns starts at the end of the write data, not at its command start; data-bus turnaround remains a separate constraint.",
                      "These modest integer intervals form a reproducible internal sensitivity assumption; they are not measured timings or a JEDEC device specification.",
                  ]}),
    "hbf_synthetic": _preset(_NAND_RESEARCH_BASE),
    "ssd_synthetic": _preset(_NAND_RESEARCH_BASE, kind="SSD",
        host_bandwidth_gb_s=112.0, internal_bandwidth_gb_s=112.0,
        host_read_bandwidth_gb_s=112.0, host_write_bandwidth_gb_s=112.0,
        page_read_ns=50000.0, page_program_ns=800000.0, front_ns=0.0),
}
_PHYSICAL_CONFIG_CATALOG["published_hbf"] = _preset(
    _NAND_RESEARCH_BASE, channels=128, parallel_units=128, capacity_bytes=512 * 1024 ** 3,
    front_ns=0.0, internal_bandwidth_gb_s=488000.0,
    metadata={"published_input_scope": "capacity, read/write bandwidth, page size and page latency are caller-selected published inputs; undisclosed geometry is an equivalent analytical mapping",
              "bandwidth_mapping": "effective shared tier on host leg; internal leg is 1000 times shared host bandwidth",
              "parallelism_mapping": "explicit effective units, otherwise ceil(read bandwidth * page read latency / page bytes); channels map one-to-one to effective units"})
_PHYSICAL_CONFIG_CATALOG["study_nvme_baseline"] = _preset(
    _PHYSICAL_CONFIG_CATALOG["ssd_synthetic"], channels=32, blocks_per_plane=8192,
    host_bandwidth_gb_s=7.0, host_read_bandwidth_gb_s=7.0, host_write_bandwidth_gb_s=4.0)

# Ideal medium controls deliberately retain their authored synthetic geometry.
# Non-ideal HBM microbenchmarks are defined with the standard references below.
_PHYSICAL_CONFIG_CATALOG.update({
    "research_hbf_sessions": _preset(_NAND_RESEARCH_BASE, channels=8, parallel_units=8,
        capacity_bytes=33554432),
    "research_hbf_media_write": _preset(_NAND_RESEARCH_BASE, channels=4, parallel_units=4,
        host_bandwidth_gb_s=64.0, host_read_bandwidth_gb_s=64.0, host_write_bandwidth_gb_s=16.0,
        internal_bandwidth_gb_s=128.0),
    "research_hbf_migration": _preset(_NAND_RESEARCH_BASE, channels=32, parallel_units=32,
        host_bandwidth_gb_s=128.0, host_read_bandwidth_gb_s=128.0, host_write_bandwidth_gb_s=32.0,
        internal_bandwidth_gb_s=128.0),
    "research_hbf_package": _preset(_NAND_RESEARCH_BASE),
    "research_hbf_bandwidth": _preset(_NAND_RESEARCH_BASE, channels=64,
        targets_per_channel=4, dies_per_target=2, luns_per_die=2, planes_per_lun=2,
        planes_independent=True, parallel_units=2048, blocks_per_plane=1,
        pages_per_block=1024, host_granularity_bytes=1, internal_bandwidth_gb_s=12000.0,
        front_ns=0.0, max_expanded_segments=1),
    "research_hbf_read_write": _preset(_NAND_RESEARCH_BASE, capacity_bytes=131072,
        channels=4, parallel_units=4, host_bandwidth_gb_s=400.0,
        host_read_bandwidth_gb_s=400.0, host_write_bandwidth_gb_s=16.0,
        internal_bandwidth_gb_s=128.0),
    "research_hbm_ideal": _preset(_DRAM_RESEARCH_BASE, channels=1,
        pseudo_channels_per_channel=1, data_lanes=1, bank_groups_per_rank=1,
        banks_per_group=1, rows_per_bank=1024, row_bytes=4096,
        interface_bandwidth_gb_s=64.0, read_bandwidth_gb_s=64.0, write_bandwidth_gb_s=64.0,
        open_ns=0.0, close_ns=0.0, read_latency_ns=0.0, write_latency_ns=0.0,
        read_recovery_ns=0.0, write_recovery_ns=0.0, capacity_bytes=4194304,
        metadata={"timing_basis": "all-zero array/group timing is an intentional ideal-memory isolation control"}),
    "research_hbf_ideal": _preset(_NAND_RESEARCH_BASE, blocks_per_plane=1024,
        pages_per_block=64, page_bytes=64, host_granularity_bytes=64,
        internal_transfer_bytes=64, host_bandwidth_gb_s=64.0,
        host_read_bandwidth_gb_s=64.0, host_write_bandwidth_gb_s=64.0,
        internal_bandwidth_gb_s=1e12, page_read_ns=0.0, page_program_ns=0.0,
        front_ns=0.0, capacity_bytes=4194304),
    "research_hbf_wear_trace": _preset(_NAND_RESEARCH_BASE, planes_per_lun=4,
        planes_independent=True, parallel_units=2, blocks_per_plane=2,
        pages_per_block=64, page_bytes=4096, host_granularity_bytes=1,
        host_bandwidth_gb_s=100.0, host_read_bandwidth_gb_s=100.0,
        host_write_bandwidth_gb_s=100.0, internal_bandwidth_gb_s=100.0,
        front_ns=0.0, detailed_observability=True, capacity_bytes=2097152,
        wear_model={"rated_pe_cycles": 3000, "initial_pe_cycles": 2968,
            "program_latency_growth_at_rated_pe": 1.0, "max_program_latency_scale": 2.0,
            "assumption": "Authored stress sensitivity, not measured HBF/NAND properties or a silicon lifetime forecast; authored program latency doubles at rated P/E and then caps"}),
})
_PHYSICAL_CONFIG_CATALOG["research_hbm_ablation"] = _preset(
    _PHYSICAL_CONFIG_CATALOG["research_hbm_ideal"], rows_per_bank=4096,
    capacity_bytes=16777216, max_outstanding_requests=4096, max_expanded_segments=128)
_PHYSICAL_CONFIG_CATALOG["research_hbf_ablation"] = _preset(
    _PHYSICAL_CONFIG_CATALOG["research_hbf_ideal"], blocks_per_plane=4096,
    parallel_units=64, capacity_bytes=16777216, max_outstanding_requests=4096,
    max_expanded_segments=128)


# One physical stack per non-ideal microbenchmark Core. Small capacity values
# remain visible budgets over a standard covering part, not invented small
# DRAM dies. Explicit overrides use the same dependency rules as every other
# HBM reference; the single-stack defaults are not a second validation policy.
_HBM_MICRO_TIMING_AND_QUEUE = {
    "max_outstanding_requests": 64, "max_expanded_segments": 1000000,
}
_HBM_MICRO_REFERENCE_DEFAULTS = {
    "research_hbm_acceptance": {"capacity_bytes": 33554432, "interface_bandwidth_gb_s": 488.0},
    "research_hbm_sessions": {"capacity_bytes": 33554432, "interface_bandwidth_gb_s": 512.0},
    "research_hbm_relay": {"capacity_bytes": 33554432, "interface_bandwidth_gb_s": 256.0},
    "research_hbm_package": {"capacity_bytes": 33554432, "interface_bandwidth_gb_s": 256.0},
    "research_hbm_path_relay": {"capacity_bytes": 16777216, "interface_bandwidth_gb_s": 128.0},
}
# These standard-organization research references do not identify a GPU part.
_HBM_REFERENCE_PROFILES = {
    "hbm3e_reference": "standard",
    "hbm3e_reference_flat": "flat",
    "hbm3e_reference_synthetic_timing": "timed",
    "hbm4_reference": "standard",
    "hbm4_reference_flat": "flat",
    "hbm4_reference_synthetic_timing": "timed",
    **{profile_id: "standard" for profile_id in _HBM_MICRO_REFERENCE_DEFAULTS},
}


def _hbm_reference(profile_id, updates, expand_geometry=False):
    if "kind" in updates and str(getattr(updates["kind"], "value", updates["kind"])).upper() != "HBM":
        raise ValueError("physical config profile kind cannot be changed by overrides")
    requested_fields = sorted(updates)
    micro_reference = profile_id in _HBM_MICRO_REFERENCE_DEFAULTS
    values = {}
    if micro_reference:
        values.update(deepcopy(_HBM_MICRO_TIMING_AND_QUEUE))
        values.update(deepcopy(_HBM_MICRO_REFERENCE_DEFAULTS[profile_id]))
        values.update(generation="HBM3E", stacks=1)
    values.update(deepcopy(dict(updates)))
    capacity = values.pop("capacity_bytes", 512 * 1024 ** 3)
    generation = normalize_hbm_generation(values.pop(
        "generation", "HBM4" if profile_id.startswith("hbm4_") else "HBM3E"))
    stacks = values.pop("stacks", None)
    if stacks is None:
        max_stack_bytes = (16 if generation in {"HBM2", "HBM2E"} else 64) * 1024 ** 3
        stacks = max(8, (capacity + max_stack_bytes - 1) // max_stack_bytes)
    interface = values.pop("interface_bandwidth_gb_s", 4000.0)
    # A rate-mode caller may clear the bandwidth input. The seed only
    # establishes a complete organization; the public contract applies the
    # explicit None and rate/clock together below.
    if interface is None:
        interface = _HBM_MICRO_REFERENCE_DEFAULTS.get(profile_id, {}).get(
            "interface_bandwidth_gb_s", 4000.0)
    values.pop("read_bandwidth_gb_s", None)
    values.pop("write_bandwidth_gb_s", None)
    channels = values.pop("channels", None)
    height = values.pop("dies_per_stack", None)
    density = values.pop("die_density_gbits", None)
    metadata = values.pop("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("physical config metadata must be a mapping")
    payload = make_hbm_geometry_config(
        generation=generation, capacity_bytes=capacity, stacks=stacks,
        dies_per_stack=height, die_density_gbits=density, channels=channels,
        read_bandwidth_gbps=interface * 8.0, write_bandwidth_gbps=interface * 8.0,
        interface_bandwidth_gbps=interface * 8.0,
        scope="research_hbm_standard_organization",
    )
    mode = _HBM_REFERENCE_PROFILES[profile_id]
    # The builder selected the initial operating point. Reapply only caller
    # inputs here: passing its default interface as an override would conflict
    # with a clock-only input, and forgetting explicit caps could retime them.
    changes = {name: updates[name] for name in (
        "interface_bandwidth_gb_s", "read_bandwidth_gb_s", "write_bandwidth_gb_s")
        if name in updates}
    if mode == "flat":
        changes.update(bank_groups_per_rank=1,
                       banks_per_group=payload["bank_groups_per_rank"] * payload["banks_per_group"])
    changes.update(values)
    changes["metadata"] = {
        **metadata,
        "physical_config_profile_id": profile_id,
        "physical_config_override_fields": requested_fields,
        "physical_config_geometry_expansion": bool(expand_geometry),
        "published_input_scope": "Only explicitly cited capacity/bandwidth/stack inputs are published; selected JEDEC organization is a research reference, not an identified vendor part",
        "timing_basis": ("synthetic bank-group timing sensitivity above known JEDEC minima; unknown vendor timings remain assumptions, not vendor calibration"
                         if mode == "timed" else payload["metadata"]["timing_basis"]),
        "geometry_control": mode,
    }
    if micro_reference:
        provenance = {**deepcopy(payload["metadata"]["parameter_provenance"]),
                      **deepcopy(metadata.get("parameter_provenance", {}))}
        for name in _HBM_MICRO_TIMING_AND_QUEUE:
            provenance[name] = {
                "status": "explicit_assumption", "source": "physical_config_override" if name in updates else "physical_presets",
                "basis": "authored microbenchmark timing or queue control, retained separately from the standard HBM organization; not vendor calibration",
            }
        changes["metadata"].update(
            configuration_scope="hbm_microbenchmark",
            reference_profile_id="hbm4_reference" if generation == "HBM4" else "hbm3e_reference"
                if generation in {"HBM3", "HBM3E"} else "JESD235B_reference",
            timing_basis=payload["metadata"]["timing_basis"],
            capacity_input_scope="microbenchmark visible capacity budget over a covering standard HBM organization",
            parameter_provenance=provenance,
        )
    result = reconfigure_physical_config(payload, overrides=changes, expand_geometry=expand_geometry)
    if mode == "timed":
        # Unknown array timings remain fixed ns assumptions. Column spacing
        # retains its max(standard minimum, analytical floor) dependency so
        # later clock changes still use the current JEDEC minimum.
        dependent = {"column_same_group_ns", "column_different_group_ns"}
        synthetic = {name: max(getattr(result, name), _SYNTHETIC_BANK_GROUPS[name])
                     for name in DRAM_BANK_GROUP_TIMING_FIELDS
                     if name not in updates and name not in dependent}
        synthetic = {name: value for name, value in synthetic.items()
                     if value != getattr(result, name)}
        if synthetic:
            result = reconfigure_physical_config(result, overrides=synthetic)
        provenance = deepcopy(result.metadata["parameter_provenance"])
        for name in dependent - updates.keys():
            provenance[name] = {
                "status": "derived_from_input", "source": "physical_presets",
                "rule": "max_standard_minimum_and_analytical_floor",
                "analytical_floor_ns": float(_SYNTHETIC_BANK_GROUPS[name]),
                "basis": "maximum of the current JEDEC column-spacing minimum and an authored synthetic ns floor; not vendor calibration",
            }
        result = reconfigure_physical_config(result,
            overrides={"metadata": {"parameter_provenance": provenance}})
    return result


for _reference_id in _HBM_REFERENCE_PROFILES:
    _PHYSICAL_CONFIG_CATALOG[_reference_id] = asdict(_hbm_reference(_reference_id, {}))


def physical_config_profile_ids():
    """Available authored physical presets (no mutable catalog is exposed)."""
    return tuple(sorted(_PHYSICAL_CONFIG_CATALOG))


def resolve_physical_config(profile_id, *, overrides=None, expand_geometry=False):
    """Resolve an authored preset plus explicit experiment variables.

    The catalog supplies every physical field. Caller overrides use native
    config names (GB/s, ns and bytes); they are recorded in metadata. Published
    HBF profiles centrally derive their equivalent parallelism and internal
    leg, keeping this physical mapping out of experiment scripts.
    """
    if profile_id not in _PHYSICAL_CONFIG_CATALOG:
        raise ValueError("unknown physical config profile: " + str(profile_id))
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ValueError("physical config overrides must be a mapping")
    if profile_id in _HBM_REFERENCE_PROFILES:
        return _hbm_reference(profile_id, overrides or {}, expand_geometry)
    authored = deepcopy(_PHYSICAL_CONFIG_CATALOG[profile_id])
    updates = deepcopy(dict(overrides or {}))
    if "kind" in updates and str(getattr(updates["kind"], "value", updates["kind"])).upper() != authored["kind"]:
        raise ValueError("physical config profile kind cannot be changed by overrides")
    requested_fields = sorted(updates)
    if profile_id == "published_hbf":
        values = {**authored, **updates}
        page_bytes = values["page_bytes"]
        if isinstance(page_bytes, bool) or not isinstance(page_bytes, int) or page_bytes <= 0:
            raise ValueError("page_bytes must be a positive integer")
        read_bw = values["host_read_bandwidth_gb_s"]
        for name, value in (("host read bandwidth", read_bw), ("page_read_ns", values["page_read_ns"]),
                            ("host_bandwidth_gb_s", values["host_bandwidth_gb_s"])):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or value < 0):
                raise ValueError(name + " must be a finite non-negative number")
        if updates.get("parallel_units") is None:
            product = read_bw * values["page_read_ns"] / page_bytes
            if not isfinite(product):
                raise ValueError("derived published HBF parallelism must be finite")
            updates["parallel_units"] = max(1, ceil(product))
        if "channels" not in updates:
            updates["channels"] = updates["parallel_units"]
        if "host_granularity_bytes" not in updates:
            updates["host_granularity_bytes"] = min(4096, page_bytes)
        if "internal_bandwidth_gb_s" not in updates:
            updates["internal_bandwidth_gb_s"] = values["host_bandwidth_gb_s"] * 1000.0
        if "blocks_per_plane" not in updates:
            # Published HBF geometry is an equivalent organization authored
            # from capacity and effective units. Construct whole blocks here;
            # later updates to an existing config need expand_geometry=True.
            layout = {**values, **updates}
            block_stripe = 1
            for name in ("channels", "targets_per_channel", "dies_per_target", "luns_per_die",
                         "planes_per_lun", "pages_per_block", "page_bytes"):
                value = layout[name]
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(name + " must be a positive integer")
                block_stripe *= value
            capacity = layout["capacity_bytes"]
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise ValueError("capacity_bytes must be a positive integer")
            updates["blocks_per_plane"] = (capacity + block_stripe - 1) // block_stripe
    metadata = updates.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("physical config metadata must be a mapping")
    updates["metadata"] = {**metadata, "physical_config_profile_id": profile_id,
        "physical_config_override_fields": requested_fields,
        "physical_config_geometry_expansion": bool(expand_geometry)}
    return reconfigure_physical_config(parse_execution_physical_config(authored),
        overrides=updates, expand_geometry=expand_geometry)


def make_analytical_dram_config(
    *, kind: str, generation: str, capacity_bytes: int,
    read_bandwidth_gbps: float, write_bandwidth_gbps: float,
    channels: int | None = None, data_width_bits: int = 64,
    interface_bandwidth_gbps: float | None = None,
    read_latency_ns: float = 35.0, write_latency_ns: float = 35.0,
    scope: str = "equivalent_memory_subsystem",
) -> Dict[str, Any]:
    """Select sourced HBM organization or an explicitly analytical DRAM array."""
    normalized_generation = generation.strip().upper().replace("-", "").replace("_", "")
    if kind.upper() == "HBM" and normalized_generation in {
        "HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4",
    }:
        max_stack_bytes = (16 if normalized_generation in {"HBM2", "HBM2E"} else 64) * 1024 ** 3
        reference_stacks = max(1, (capacity_bytes + max_stack_bytes - 1) // max_stack_bytes)
        result = make_hbm_geometry_config(
            generation=normalized_generation, capacity_bytes=capacity_bytes,
            read_bandwidth_gbps=read_bandwidth_gbps,
            write_bandwidth_gbps=write_bandwidth_gbps, channels=channels,
            interface_bandwidth_gbps=interface_bandwidth_gbps,
            stacks=reference_stacks,
            scope=scope,
        )
        result["metadata"]["unused_component_latency_ns"] = {
            "read": read_latency_ns, "write": write_latency_ns,
            "basis": "aggregate component latency is not a JEDEC RL/WL encoding and is not added to physical execution",
        }
        if reference_stacks > 1:
            result["metadata"]["parameter_provenance"]["stacks"] = {
                "status": "selected_reference",
                "basis": "minimum reference stack count covering this aggregate budget; actual product stack count unverified",
            }
            result["metadata"]["unknown_parameters"].append("vendor_stack_count")
        return asdict(reconfigure_physical_config(result))
    channels = 1 if channels is None else channels
    groups, banks, row_bytes, burst_bytes = 4, 4, 8192, 64
    bank_count = channels * groups * banks
    rows = (capacity_bytes + bank_count * row_bytes - 1) // (bank_count * row_bytes)
    interface_gb_s = (max(read_bandwidth_gbps, write_bandwidth_gbps)
                      if interface_bandwidth_gbps is None else interface_bandwidth_gbps) / 8.0
    lane_gb_s = interface_gb_s / channels
    payload = deepcopy(_DRAM_RESEARCH_BASE)
    payload.update(
        kind=kind, generation=generation, channels=channels,
        subchannels_per_channel=1, pseudo_channels_per_channel=1,
        ranks_per_channel=1, bank_groups_per_rank=groups,
        banks_per_group=banks, rows_per_bank=rows, row_bytes=row_bytes,
        burst_bytes=burst_bytes, interleave_bytes=burst_bytes,
        data_width_bits=data_width_bits,
        data_rate_mt_s=interface_gb_s * 8_000.0 / (channels * data_width_bits),
        interface_bandwidth_gb_s=interface_gb_s,
        read_bandwidth_gb_s=read_bandwidth_gbps / 8.0,
        write_bandwidth_gb_s=write_bandwidth_gbps / 8.0,
        open_ns=14.0, close_ns=14.0,
        read_latency_ns=read_latency_ns, write_latency_ns=write_latency_ns,
        burst_interval_ns=burst_bytes / lane_gb_s,
        capacity_bytes=capacity_bytes, max_outstanding_requests=64,
        metadata={
            "parameter_basis": "analytical_equivalent_assumptions",
            "configuration_scope": scope,
            "known_values": "component capacity and one-way directional bandwidth",
            "analytical_assumptions": [
                "channel count unless separately cited by the preset",
                "rank, bank, row and burst organization",
                "open/close, command-to-data and recovery timings",
                "burst interval is a payload scheduling budget, not measured tCCD",
                "bank-group column/ACT/WRITE-to-READ timings and four-ACT window are unspecified (zero disables each constraint)",
            ],
            "capacity_basis": "declared capacity with an equivalent geometry rounded upward to whole rows",
            "bandwidth_basis": "explicit component one-way bandwidth; synthetic lane width does not infer pin speed",
            "data_rate_basis": "equivalent interface rate derived from declared aggregate bandwidth and synthetic width; not a measured pin speed",
        },
    )
    if kind.upper() == "HBM":
        # No verified addressing table is selected for unnamed/unsupported HBM.
        # Preserve explicitly analytical templates without representing
        # their bank/page organization as a standard or vendor specification.
        payload["metadata"].update(
            geometry_status="explicit_assumption",
            unknown_parameters=["bank_groups_per_rank", "banks_per_group",
                                "rows_per_bank", "row_bytes", "burst_bytes",
                                "dies_per_stack", "vendor_address_mapping"],
            geometry_basis="Unverified analytical HBM organization; not a JEDEC device configuration",
        )
    return asdict(parse_execution_physical_config(payload))


def make_analytical_nand_config(
    *, kind: str, capacity_bytes: int, read_bandwidth_gbps: float,
    write_bandwidth_gbps: float, read_latency_ns: float,
    write_latency_ns: float, scope: str = "equivalent_flash_device",
) -> Dict[str, Any]:
    """Model a NAND-backed device without claiming undisclosed die layout."""
    channels, targets, dies, luns, planes = 8, 2, 2, 1, 2
    pages, page_bytes = 256, 16384
    block_bytes = channels * targets * dies * luns * planes * pages * page_bytes
    blocks = (capacity_bytes + block_bytes - 1) // block_bytes
    payload = deepcopy(_NAND_RESEARCH_BASE)
    payload.update(
        kind=kind, channels=channels, targets_per_channel=targets,
        dies_per_target=dies, luns_per_die=luns, planes_per_lun=planes,
        blocks_per_plane=blocks, pages_per_block=pages, page_bytes=page_bytes,
        host_granularity_bytes=4096,
        host_bandwidth_gb_s=max(read_bandwidth_gbps, write_bandwidth_gbps) / 8.0,
        internal_bandwidth_gb_s=max(read_bandwidth_gbps, write_bandwidth_gbps) / 8.0,
        host_read_bandwidth_gb_s=read_bandwidth_gbps / 8.0,
        host_write_bandwidth_gb_s=write_bandwidth_gbps / 8.0,
        page_read_ns=read_latency_ns, page_program_ns=write_latency_ns,
        block_erase_ns=3_000_000.0, front_ns=0.0,
        capacity_bytes=capacity_bytes, max_outstanding_requests=64,
        write_buffer_bytes=0,
        metadata={
            "parameter_basis": "analytical_equivalent_assumptions",
            "configuration_scope": scope,
            "known_values": "component capacity and sequential read/write bandwidth",
            "analytical_assumptions": [
                "channel, target, die, LUN, plane, block and page organization",
                "page read/program and block erase timings",
                "internal bandwidth and scheduling parallelism",
            ],
            "directional_bandwidth_gb_s": {
                "read": read_bandwidth_gbps / 8.0,
                "write": write_bandwidth_gbps / 8.0,
            },
            "capacity_basis": "declared capacity with equivalent array rounded upward to whole blocks",
        },
    )
    return asdict(parse_execution_physical_config(payload))
