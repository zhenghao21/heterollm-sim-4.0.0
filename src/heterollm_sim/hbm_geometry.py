"""Sourced HBM organizations, with device selection and timing kept explicit.

JEDEC channel-addressing tables describe each pseudo-channel's organization.
Its SID bits extend the bank address: stacked dies must not multiply that
organization a second time. A selected reference organization is not evidence
that a GPU or a particular vendor part uses that organization.
"""

from __future__ import annotations

from dataclasses import asdict, fields
import math
from typing import Any

from .memory_types import DRAM_BANK_GROUP_TIMING_FIELDS, DramConfig
from ._hbm_dependencies import (organization_for, refresh_cycle_ns_for, clock_ns_for,
                                reference_timings, normalize_hbm_dependencies)


HBM_GENERATIONS = frozenset({"HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4"})
HBM3_DIE_DENSITIES_GBITS = (8, 16, 24, 32)
HBM3_STACK_HEIGHTS = (4, 8, 12, 16)
HBM4_DIE_DENSITIES_GBITS = (24, 32)

JESD235B_SOURCE = "https://gitcode.com/Open-source-documentation-tutorial/a1451"
JESD238_SOURCE = "https://gitcode.com/Open-source-documentation-tutorial/69c72"
JESD270_4A_SOURCE = (
    "https://github.com/Adilohit/HBM4_verification/blob/main/resources/"
    "JEDECJESD270-4A-2025.pdf"
)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_bandwidth(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def normalize_hbm_generation(generation: str) -> str:
    """Accept conventional capitalization without guessing an HBM generation."""
    value = str(generation).strip().upper().replace("-", "").replace("_", "")
    if value not in HBM_GENERATIONS:
        raise ValueError(
            f"{generation!r} has no verified HBM organization in this catalog; "
            "supported generations are HBM2, HBM2E, HBM3, HBM3E and HBM4."
        )
    return value


def make_hbm_geometry_config(
    *, generation: str = "HBM3E", capacity_bytes: int,
    read_bandwidth_gbps: float, write_bandwidth_gbps: float,
    interface_bandwidth_gbps: float | None = None,
    stacks: int = 1, die_density_gbits: int | None = None,
    dies_per_stack: int | None = None, channels: int | None = None,
    scope: str = "hbm_memory_subsystem",
    read_latency_ns: float | None = None, write_latency_ns: float | None = None,
    read_latency_cycles: int | None = None, write_latency_cycles: int | None = None,
) -> dict[str, Any]:
    """Return a complete DramConfig mapping using a documented organization.

    Bandwidth inputs are aggregate decimal Gb/s across all stacks. Capacity is
    the visible byte budget; the selected raw organization may be larger. With
    unspecified die density/height, choose the smallest covering organization,
    preferring 8H, then 12H, 16H and 4H for equal raw capacities. Such choices
    remain ``selected_reference`` parameters, never inferred vendor facts.

    HBM2/E currently imports only the 16Gb-die base/8H organizations from
    JESD235B. For HBM2/E and HBM3/E the 4H option instantiates the no-SID base
    organization whose physical stack height is vendor-specific. JESD270-4A
    explicitly identifies its HBM4 4H organization. Speed bins and vendor
    implementations are not established by these address-organization tables.
    """
    generation = normalize_hbm_generation(generation)
    capacity_bytes = _positive_integer("capacity_bytes", capacity_bytes)
    stacks = _positive_integer("stacks", stacks)
    if capacity_bytes % 32:
        raise ValueError("capacity_bytes must be a multiple of burst_bytes (32)")
    read_bandwidth_gbps = _positive_bandwidth("read_bandwidth_gbps", read_bandwidth_gbps)
    write_bandwidth_gbps = _positive_bandwidth("write_bandwidth_gbps", write_bandwidth_gbps)
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope must be a non-empty string")

    hbm3 = generation in {"HBM3", "HBM3E"}
    hbm4 = generation == "HBM4"
    physical_channels = 32 if hbm4 else 16 if hbm3 else 8
    if channels is not None:
        _positive_integer("channels", channels)
        if channels != physical_channels:
            raise ValueError(
                f"{generation} reference organization requires {physical_channels} "
                "physical channels per stack; use stacks for multiple stacks"
            )
    densities = HBM4_DIE_DENSITIES_GBITS if hbm4 else HBM3_DIE_DENSITIES_GBITS if hbm3 else (16,)
    heights = HBM3_STACK_HEIGHTS if hbm3 or hbm4 else (4, 8)
    if die_density_gbits is not None:
        _positive_integer("die_density_gbits", die_density_gbits)
        if die_density_gbits not in densities:
            raise ValueError(f"unsupported {generation} die_density_gbits; expected one of {densities}")
    if dies_per_stack is not None:
        _positive_integer("dies_per_stack", dies_per_stack)
        if dies_per_stack not in heights:
            raise ValueError(f"unsupported {generation} dies_per_stack; expected one of {heights}")

    height_preference = {8: 0, 12: 1, 16: 2, 4: 3}
    candidates = [
        (density * height * (1 << 30) // 8, height_preference[height], density, height)
        for density in densities for height in heights
        if (die_density_gbits is None or density == die_density_gbits)
        and (dies_per_stack is None or height == dies_per_stack)
        and density * height * (1 << 30) // 8 * stacks >= capacity_bytes
    ]
    if not candidates:
        max_stack_capacity = max(densities) * max(heights) * (1 << 30) // 8
        raise ValueError(
            f"capacity_bytes={capacity_bytes} exceeds the requested {generation} "
            f"organization for {stacks} stack(s); maximum supported raw capacity "
            f"per stack is {max_stack_capacity} bytes. Increase stacks or choose "
            "a supported die density and stack height; rows are not expanded."
        )
    raw_stack_capacity, _, density, height = min(candidates)
    organization = organization_for(generation, density, height)
    sid_count = height // 4
    # 16 BA values per SID. Each PC independently addresses this bank space.
    banks_per_pc = 16 * sid_count
    banks_per_group = organization["banks_per_group"]
    groups_per_pc = organization["bank_groups_per_rank"]
    rows = organization["rows_per_bank"]
    pc_width = organization["data_width_bits"]
    lanes = stacks * physical_channels * 2
    interface_gb_s = (_positive_bandwidth("interface_bandwidth_gbps", interface_bandwidth_gbps)
                      if interface_bandwidth_gbps is not None else
                      max(read_bandwidth_gbps, write_bandwidth_gbps)) / 8.0
    lane_gb_s = interface_gb_s / lanes
    # HBM2: 64 DQ * 2 transfers/CK; HBM3/4: 32 DQ * 4 transfers/CK.
    # Every PC therefore transfers 16 bytes/CK; a 32-byte burst takes 2 CK.
    clock_ns = clock_ns_for(generation, pc_width, lanes, interface_gb_s)
    if read_latency_ns is not None and read_latency_cycles is not None:
        raise ValueError("choose read_latency_ns or read_latency_cycles, not both")
    if write_latency_ns is not None and write_latency_cycles is not None:
        raise ValueError("choose write_latency_ns or write_latency_cycles, not both")
    rl = 32 if read_latency_cycles is None else _positive_integer("read_latency_cycles", read_latency_cycles)
    wl = 8 if write_latency_cycles is None else _positive_integer("write_latency_cycles", write_latency_cycles)
    supplied_read_latency = read_latency_ns is not None or read_latency_cycles is not None
    supplied_write_latency = write_latency_ns is not None or write_latency_cycles is not None
    read_latency_ns = rl * clock_ns if read_latency_ns is None else read_latency_ns
    write_latency_ns = wl * clock_ns if write_latency_ns is None else write_latency_ns
    # These values occur in the standards' IDD measurement EXAMPLE tables.
    # They are a named reference operating condition, not vendor AC limits.
    timing_reference = reference_timings(generation, clock_ns)
    reference_ras = timing_reference["minimum_active_ns"]
    reference_rp = timing_reference["precharge_to_activate_ns"]
    refresh_ns = refresh_cycle_ns_for(generation, density, height)
    if hbm4:
        organization_source = JESD270_4A_SOURCE
        standard = "JESD270-4A"
        source_file = ".refs/jedec/JESD270-4A-2025.pdf"
        organization_basis = "JESD270-4A (2025), sections 3.1.2/3.2, Tables 4/5, printed pp. 6-7"
        geometry_derivation = (
            "Table 4/5: rows=die_density_gbits*512, banks/PC=16*(height/4), "
            "eight banks/group; 24Gb excludes RA[13:12]=11; 12H excludes SID=11"
        )
    elif hbm3:
        organization_source = JESD238_SOURCE
        standard = "JESD238"
        source_file = ".refs/jedec/JESD238-2022.pdf"
        organization_basis = "JESD238 (January 2022), sections 3.1.2/3.2, Tables 4/5, printed pp. 6-7"
        geometry_derivation = (
            "Table 4/5: rows=die_density_gbits*1024, banks/PC=16*(height/4), "
            "four banks/group; 24Gb excludes RA[14:13]=11; 12H excludes SID=11"
        )
    else:
        organization_source = JESD235B_SOURCE
        standard = "JESD235B"
        source_file = ".refs/jedec/JESD235B-2018.pdf"
        organization_basis = "JESD235B (November 2018), sections 3.2.1.2/3.2.3/3.2.4, Tables 4/5, printed pp. 6-8"
        geometry_derivation = (
            "Table 4/5: 16Gb reference uses RA[14:0] => 32768 rows; "
            "16 bank addresses/SID; base/8H => 1/2 SIDs; four banks/group"
        )
    unknown_parameters = [
        "vendor_part_number", "vendor_address_mapping", "array_timing_ns",
        "refresh_timing_ns", "controller_policy", "effective_pin_data_rate_gbps",
    ]
    if die_density_gbits is None:
        unknown_parameters.append("die_density_gbits")
    if dies_per_stack is None:
        unknown_parameters.append("dies_per_stack")

    metadata: dict[str, Any] = {
        "parameter_basis": "sourced_organization_with_analytical_timing",
        "geometry_status": "standard_organization",
        "configuration_scope": scope,
        "vendor_verified": False,
        "organization_source": organization_source,
        "organization_basis": organization_basis,
        "jedec_standard": standard,
        "jedec_reference_file": source_file,
        "source_provenance": "original JEDEC PDF obtained from a public third-party mirror; not a vendor interpretation",
        "organization_selection": "smallest covering raw capacity; equal capacity prefers 8H, 12H, 16H, 4H",
        "die_density_gbits": density,
        "physical_channels_per_stack": physical_channels,
        "physical_channel_width_bits": pc_width * 2,
        "pseudo_channel_width_bits": pc_width,
        "channel_operation_mode": "pseudo_channel",
        "channel_operation_mode_basis": (
            f"{standard} section 3.1.2 defines two pseudo-channels per channel; "
            "no alternative Legacy mode is defined for this generation"
            if hbm3 or hbm4 else
            "JESD235B section 3.2.1 defines Legacy (HBM1) and Pseudo Channel (HBM2); "
            "this catalog imports only PC-mode organizations. Mode support is fixed "
            "by device design: DEVICE_ID[17:16] is 01 for PC-only or 10 for Legacy-only; "
            "00 and 11 are illegal. This is not a runtime Legacy/PC switch."
        ),
        "stack_data_width_bits": physical_channels * pc_width * 2,
        "sid_count": sid_count,
        "banks_per_pseudo_channel": banks_per_pc,
        "bank_groups_per_pseudo_channel": groups_per_pc,
        "effective_addressed_page_bytes": 1024,
        "raw_stack_capacity_bytes": raw_stack_capacity,
        "raw_capacity_bytes": raw_stack_capacity * stacks,
        "visible_capacity_bytes": capacity_bytes,
        "unexposed_capacity_bytes": raw_stack_capacity * stacks - capacity_bytes,
        "capacity_formula": "stacks * die_density_gbits * dies_per_stack * 2**30 / 8",
        "capacity_basis": "binary DRAM address capacity; capacity_bytes is a separate visible budget",
        "bandwidth_basis": "caller aggregate directional Gb/s divided by 8; stack bandwidth is not multiplied again",
        "data_rate_basis": "equivalent per-pin rate derived from aggregate bandwidth and external data width, not measured pin speed",
        "timing_basis": "analytical assumptions, not JEDEC or vendor timing calibration",
        "unknown_parameters": unknown_parameters,
        "source_documents": [
            {"source": organization_source, "basis": organization_basis,
             "local_file": source_file, "document_author": "JEDEC",
             "access_method": "public mirror of original protocol PDF"},
        ],
        "geometry_limitations": [
            "linear mixed-radix simulation mapping is not a vendor address mapping",
            "standard organization does not establish the target vendor part or timing",
        ],
    }
    if hbm3 or hbm4:
        metadata["activated_page_bytes"] = 2048
        metadata["row_bytes_basis"] = (
            f"{standard} Table 4 note 3: row_bytes is the 1KiB effective addressed "
            "PC page; the MSB of RA selects half of an open 2KiB page"
        )
    else:
        metadata["row_bytes_basis"] = "JESD235B Table 4: effective PC page is 1KiB; physical activation organization is not asserted"
        metadata["geometry_limitations"].append(
            "catalog imports only 16Gb base/8H HBM2 organizations; other legal JEDEC organizations are not yet imported"
        )
    if height == 4 and not hbm4:
        metadata["geometry_limitations"].append(
            f"4H instantiates the no-SID base organization; {standard} Table 4 "
            "leaves base-organization physical stack height vendor-specific"
        )

    config = DramConfig(
        kind="HBM", generation=generation, channels=physical_channels,
        subchannels_per_channel=1, pseudo_channels_per_channel=2,
        stacks=stacks, dies_per_stack=height, ranks_per_channel=1,
        bank_groups_per_rank=groups_per_pc, banks_per_group=banks_per_group,
        rows_per_bank=rows, row_bytes=1024, burst_bytes=32,
        data_width_bits=pc_width,
        data_rate_mt_s=interface_gb_s * 8000.0 / (lanes * pc_width),
        bandwidth_input_mode="bandwidth", interface_bandwidth_gb_s=interface_gb_s,
        read_bandwidth_gb_s=read_bandwidth_gbps / 8.0,
        write_bandwidth_gb_s=write_bandwidth_gbps / 8.0,
        interleave_bytes=32, open_ns=14.0, close_ns=reference_rp,
        read_latency_ns=read_latency_ns, write_latency_ns=write_latency_ns,
        burst_interval_ns=32 / lane_gb_s, capacity_bytes=capacity_bytes,
        bank_geometry_scope="per_stack", stack_interface_mode="independent",
        control_mode="jedec_hbm", command_clock_ns=clock_ns,
        activate_command_cycles=1.5 if hbm3 or hbm4 else 2.0,
        precharge_command_cycles=0.5 if hbm3 or hbm4 else 1.0,
        column_command_cycles=1.0,
        activated_row_msb=(rows - 1).bit_length() - 1 if hbm3 or hbm4 else None,
        activate_to_read_ns=14.0, activate_to_write_ns=14.0,
        precharge_to_activate_ns=reference_rp,
        minimum_active_ns=reference_ras, maximum_active_ns=9 * 3900.0,
        activate_cycle_ns=reference_ras + reference_rp,
        read_to_precharge_ns=0.0, write_to_precharge_ns=15.0,
        precharge_to_precharge_ns=timing_reference["precharge_to_precharge_ns"],
        column_same_group_ns=timing_reference["column_same_group_ns"],
        column_different_group_ns=timing_reference["column_different_group_ns"],
        column_different_sid_ns=timing_reference["column_different_sid_ns"],
        refresh_policy="all_bank" if refresh_ns is not None else "disabled",
        refresh_interval_ns=3900.0 if refresh_ns is not None else 0.0,
        refresh_cycle_ns=refresh_ns or 0.0,
        metadata=metadata,
    )
    # The complete executable payload keeps provenance for defaults too, so
    # serialization cannot silently turn an unspecified timing into a fact.
    provenance = {
        item.name: {"status": "assumption", "source": "simulator",
                    "basis": "analytical execution default; not a JEDEC/vendor parameter"}
        for item in fields(DramConfig) if item.name != "metadata"
    }

    def record(names: tuple[str, ...], status: str, source: str, basis: str) -> None:
        for name in names:
            provenance[name] = {"status": status, "source": source, "basis": basis}

    record(("kind", "generation", "stacks", "capacity_bytes"), "caller_supplied", "caller",
           "requested generation, stack count and visible byte budget; not vendor verification")
    record(("channels", "pseudo_channels_per_channel", "data_width_bits", "row_bytes", "burst_bytes"),
           "standard_organization", organization_source, organization_basis)
    record(("channel_operation_mode",), "standard_organization", organization_source,
           metadata["channel_operation_mode_basis"])
    record(("rows_per_bank", "bank_groups_per_rank", "banks_per_group"),
           "derived_standard_organization", organization_source, geometry_derivation)
    record(("effective_addressed_page_bytes",), "standard_organization", organization_source,
           metadata["row_bytes_basis"])
    if hbm3 or hbm4:
        record(("activated_page_bytes",), "standard_organization", organization_source,
               metadata["row_bytes_basis"])
    for name, supplied in (("die_density_gbits", die_density_gbits), ("dies_per_stack", dies_per_stack)):
        record((name,), "caller_supplied" if supplied is not None else "selected_reference",
               "caller" if supplied is not None else organization_source,
               "explicit requested reference parameter; not vendor validation" if supplied is not None else
               "selected legal covering organization; actual target device parameter remains unknown")
    record(("subchannels_per_channel", "ranks_per_channel", "bank_geometry_scope", "stack_interface_mode"),
           "model_representation", "simulator",
           "one state space per PC across its SID banks; physical die count is descriptive; stacks have independent data interfaces")
    record(("read_bandwidth_gb_s", "write_bandwidth_gb_s", "interface_bandwidth_gb_s"),
           "derived_from_input", "caller",
           "aggregate decimal Gb/s / 8; physical interface supplied explicitly or defaults to max direction; direction caps separately pace burst starts, not DQ duration")
    record(("data_rate_mt_s", "burst_interval_ns"), "derived_from_input", "simulator",
           "aggregate bandwidth divided across stacks*channels*2 PCs; equivalent scheduling rate, not a timing speed bin")
    record(("effective_pin_data_rate_gbps",), "unknown", "unspecified",
           "no measured pin speed supplied; equivalent data_rate_mt_s is not manufacturer evidence")
    record(DRAM_BANK_GROUP_TIMING_FIELDS, "unspecified", "unspecified",
           "zero disables this timing gate; no JEDEC/vendor timing calibration was supplied")
    control_basis = ({"HBM2": "JESD235B pp.30-41,107-113", "HBM2E": "JESD235B pp.30-41,107-113",
                      "HBM3": "JESD238 pp.26,34-46,161-165", "HBM3E": "JESD238 pp.26,34-46,161-165",
                      "HBM4": "JESD270-4A pp.33,41-55,194-199"}[generation])
    record(("control_mode", "activate_command_cycles", "precharge_command_cycles",
            "column_command_cycles", "activated_row_msb"), "standard_command_semantics", organization_source, control_basis)
    record(("command_clock_ns",), "derived_from_input", organization_source,
           "CK derived from authored physical DQ operating point using the JEDEC CK/DQ ratio; not a certified speed bin")
    for name, supplied in (("read_latency_ns", supplied_read_latency), ("write_latency_ns", supplied_write_latency)):
        record((name,), "caller_supplied" if supplied else "selected_reference", organization_source,
               "legal whole-CK MR encoding; actual device support and operating speed bin remain unverified; defaults choose RL32/WL8")
        explicit_ns = read_latency_cycles is None and supplied_read_latency if name == "read_latency_ns" else write_latency_cycles is None and supplied_write_latency
        provenance[name].update(input_unit="ns" if explicit_ns else "cycles",
                                cycles=(read_latency_ns if name == "read_latency_ns" else write_latency_ns) / clock_ns)
    record(("column_same_group_ns", "column_different_group_ns", "precharge_to_precharge_ns"),
           "standard_minimum", organization_source,
           "AC table: tCCDS=2CK; tCCDL=max(4CK,2.8ns) for HBM2 or max(4CK,2.5ns) for HBM3/4; HBM2 Table33 PRE spacing=1CK, HBM3/4 tPPD=2CK")
    record(("column_different_sid_ns",), "selected_reference", organization_source,
           "4CK selected from vendor-specific tCCDR range; applies only to READs across SID, not a measured part timing")
    record(("minimum_active_ns", "precharge_to_activate_ns", "activate_cycle_ns", "close_ns"),
           "jedec_idd_example", organization_source,
           "IDD measurement EXAMPLE: tRAS33ns/tRP15ns; nRC=ceil(tRAS/CK)+ceil(tRP/CK), not normative vendor AC limits")
    record(("activate_to_read_ns", "activate_to_write_ns", "write_to_precharge_ns"),
           "assumption", "simulator", "retained analytical tRCD14ns/tWR15ns; normative AC minimum cells are unspecified")
    record(("refresh_policy",), "controller_policy", "simulator",
           "periodic REFab per PC; no postponement/pull-in or per-bank optimization; disabled where tRFC is unresolved")
    record(("refresh_interval_ns",), "standard_reference" if refresh_ns is not None else "unknown", organization_source,
           "selected 3900ns periodic refresh interval within the supported range; vendor applicability unverified; unknown tRFC disables automatic refresh")
    record(("refresh_cycle_ns",), "standard_reference" if refresh_ns is not None else "unknown", organization_source,
           "AC tRFCab table for selected density/height; HBM3 24/32Gb entries are TBD and never copied from HBM4")
    record(("maximum_active_ns",), "standard_reference", organization_source,
           "tRAS(max)=9*tREFI at the selected 3900ns reference interval; enforced independently of automatic refresh; vendor applicability unverified")
    inactive = ["activate_same_group_ns", "activate_different_group_ns", "four_activate_window_ns",
                "read_to_precharge_ns", "write_to_read_same_group_ns", "write_to_read_different_group_ns",
                "read_to_write_command_ns"]
    record(tuple(inactive), "unknown_disabled", "unspecified",
           "no vendor/PHY value supplied; zero disables this extra timing gate and does not assert a physical zero delay")
    metadata.update(
        control_status="jedec_normal_operation_subset",
        control_basis=control_basis,
        timing_basis="JEDEC command rules and stated minima; IDD example, legal MR reference selections and unknown vendor timings are separately labelled",
        timing_validation="not_a_vendor_speed_bin",
        unconfigured_timing_parameters=inactive + ([] if refresh_ns is not None else ["refresh_cycle_ns"]),
        control_limitations=["training, parity/ECC transport, auto-precharge and per-bank refresh are not implemented",
                             "FCFS open-page controller and bandwidth-cap pacing are model policies, not JEDEC scheduling requirements",
                             "electrical PHY turnaround offsets and unspecified array timing gates require vendor inputs"],
        accounting_basis="command bus occupancy, DQ busy time and dependency wait are separate; one full 32B DQ burst is charged once; WR recovery limits PRE and is not a write-completion tail",
        directional_cap_basis="independent read/write start pacing; DQ remains half-duplex at physical 2CK per burst; interleaved directions may together exceed either directional cap but never the physical interface",
        analytical_only_fields=["open_ns", "close_ns", "burst_interval_ns", "read_recovery_ns", "write_recovery_ns"],
    )
    metadata["unknown_parameters"].extend(["vendor_supported_rl_wl", *inactive])
    if refresh_ns is None:
        metadata["unknown_parameters"].append("refresh_cycle_ns")
    metadata["parameter_provenance"] = provenance
    payload = asdict(config)
    payload["metadata"] = metadata
    return normalize_hbm_dependencies(payload)
