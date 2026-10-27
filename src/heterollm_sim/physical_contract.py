"""Required physical contracts at simulation and memory-service boundaries.

Incomplete components may exist in the editor. They must never execute by
substituting a bandwidth-only service or the memory core's example defaults.
"""

from copy import deepcopy
from collections import OrderedDict
from dataclasses import asdict, fields, is_dataclass, replace
from io import BytesIO
from math import gcd
from pickle import Pickler
from threading import RLock
from typing import Mapping
from ._hbm_dependencies import normalize_hbm_dependencies
from ._hbm_errors import HBMConfigError

from .memory_types import (
    DRAM_BANK_GROUP_TIMING_FIELDS, HBM_CONTROL_TIMING_FIELDS, DRAM_CONTROLLER_FIELDS,
    DramConfig, NandConfig, SramConfig, MemoryKind, Operation, parse_physical_memory_config,
)


DRAM_COMPONENT_KINDS = frozenset({
    "hbm", "hbm_stack", "gddr", "gddr_memory", "dram", "ddr", "ddr_memory",
    "lpddr", "lpddr_memory", "cxl_memory", "host_memory", "memory",
})
NAND_COMPONENT_KINDS = frozenset({"hbf", "ssd", "high_io_ssd", "nvme"})
SRAM_COMPONENT_KINDS = frozenset({"sram", "shared_memory"})
PHYSICAL_MEMORY_COMPONENT_KINDS = DRAM_COMPONENT_KINDS | NAND_COMPONENT_KINDS | SRAM_COMPONENT_KINDS

_PARSED_CONFIGS = OrderedDict()
_PARSED_CONFIGS_LOCK = RLock()
_COMPONENT_CONTRACTS = OrderedDict()
_COMPONENT_CONTRACTS_LOCK = RLock()


class _ContractPickler(Pickler):
    """A C-level content key without executing extension serialization hooks.

    Exact builtin containers/scalars are handled by Pickler itself. Only our
    plain physical dataclasses and memory-kind enum may use its object path;
    custom containers and metadata keep the ordinary uncached validation.
    The bytes are compared directly, never persisted or unpickled. They retain
    scalar types, float bits, container order, aliases and nested metadata.
    """

    def reducer_override(self, value):
        if (type(value) in (DramConfig, NandConfig, SramConfig, MemoryKind)
                or value is DramConfig or value is NandConfig
                or value is SramConfig or value is MemoryKind):
            return NotImplemented
        raise TypeError("custom physical configuration value")


def _physical_content_key(value):
    stream = BytesIO()
    try:
        _ContractPickler(stream, protocol=5).dump(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return stream.getvalue()


def _parse_complete_config(raw):
    """Amortize dependency normalization; each caller owns its mutable metadata."""
    key = _physical_content_key(raw)
    if key is None:
        return parse_physical_memory_config(raw)
    with _PARSED_CONFIGS_LOCK:
        cached = _PARSED_CONFIGS.get(key)
        if cached is not None:
            _PARSED_CONFIGS.move_to_end(key)
    if cached is not None:
        return deepcopy(cached)
    parsed = parse_physical_memory_config(raw)
    # Never retain aliases to caller-owned dictionaries in the cache.
    stored = deepcopy(parsed)
    with _PARSED_CONFIGS_LOCK:
        _PARSED_CONFIGS[key] = stored
        if len(_PARSED_CONFIGS) > 32:
            _PARSED_CONFIGS.popitem(last=False)
    return parsed

_DRAM_REQUIRED = frozenset({
    "kind", "channels", "subchannels_per_channel", "pseudo_channels_per_channel",
    "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank",
    "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes",
    "open_ns", "close_ns", "read_latency_ns", "write_latency_ns",
    "burst_interval_ns", "read_recovery_ns", "write_recovery_ns",
    "read_to_write_ns", "write_to_read_ns", "max_outstanding_requests",
    "control_mode", "bank_geometry_scope", "stack_interface_mode",
    *DRAM_BANK_GROUP_TIMING_FIELDS,
    *DRAM_CONTROLLER_FIELDS,
})
_NAND_REQUIRED = frozenset({
    "kind", "channels", "targets_per_channel", "dies_per_target", "luns_per_die",
    "planes_per_lun", "blocks_per_plane", "pages_per_block", "page_bytes",
    "host_granularity_bytes", "host_bandwidth_gb_s", "internal_bandwidth_gb_s",
    "host_read_bandwidth_gb_s", "host_write_bandwidth_gb_s",
    "page_read_ns", "page_program_ns", "block_erase_ns", "front_ns",
    "planes_independent", "partial_page_policy", "max_outstanding_requests",
    "protocol_mode", "host_channels", "host_full_duplex", "read_cache_pages_per_bank",
    "queue_scope",
})
_SRAM_REQUIRED = frozenset({
    "kind", "capacity_bytes", "burst_bytes", "interface_bandwidth_gb_s",
    "read_bandwidth_gb_s", "write_bandwidth_gb_s", "read_latency_ns",
    "write_latency_ns", "ports", "max_outstanding_requests",
})


def _validate_standard_hbm_organization(config, prefix):
    """A retained standard label must still match its selected organization.

    This checks the limited organization tables represented by the HBM
    builder. It does not verify a vendor part, timing values, or provenance
    supplied by an external author. Analytical inputs are untouched.
    """
    if (isinstance(config, DramConfig)
            and str(config.metadata.get("control_status", "")).startswith("jedec_")
            and config.control_mode != "jedec_hbm"):
        raise ValueError(prefix + "JEDEC control status requires control_mode=jedec_hbm")
    if (not isinstance(config, DramConfig) or config.kind.value != "HBM"
            or config.metadata.get("geometry_status") != "standard_organization"):
        return
    density = config.metadata.get("die_density_gbits")
    hbm4 = config.generation == "HBM4"
    hbm3 = config.generation in {"HBM3", "HBM3E"}
    hbm2 = config.generation in {"HBM2", "HBM2E"}
    densities = (24, 32) if hbm4 else (8, 16, 24, 32) if hbm3 else (16,)
    heights = (4, 8, 12, 16) if hbm4 or hbm3 else (4, 8)
    if (not (hbm4 or hbm3 or hbm2) or isinstance(density, bool) or not isinstance(density, int)
            or density not in densities or config.dies_per_stack not in heights):
        raise ValueError(prefix + "standard HBM organization requires a supported generation, die_density_gbits and dies_per_stack")
    expected = {
        "channels": 32 if hbm4 else 16 if hbm3 else 8,
        "subchannels_per_channel": 1, "pseudo_channels_per_channel": 2,
        "ranks_per_channel": 1,
        "bank_groups_per_rank": config.dies_per_stack // 2 if hbm4 else config.dies_per_stack,
        "banks_per_group": 8 if hbm4 else 4,
        "rows_per_bank": density * 512 if hbm4 else density * 1024 if hbm3 else 32768,
        "row_bytes": 1024, "burst_bytes": 32, "data_width_bits": 64 if hbm2 else 32,
        "bank_geometry_scope": "per_stack", "stack_interface_mode": "independent",
    }
    conflicts = [name for name, value in expected.items() if getattr(config, name) != value]
    if config.data_lanes is not None and config.data_lanes != expected["channels"] * 2:
        conflicts.append("data_lanes")
    if conflicts:
        raise ValueError(prefix + "standard HBM organization conflicts with " + ", ".join(sorted(conflicts))
                         + "; custom geometry must be marked explicit_assumption")


def parse_required_physical_config(component_id, kind, raw):
    """Validate authored fields before dataclass defaults can fill omissions."""
    prefix = "component {} ({}): ".format(component_id, kind)
    if kind not in PHYSICAL_MEMORY_COMPONENT_KINDS:
        raise ValueError(prefix + "no supported physical memory core; select a DRAM, NAND or SRAM component")
    if is_dataclass(raw):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise ValueError(prefix + "physical_memory_config is required; reload a complete hardware preset or configure the physical memory core. Aggregate-cost fallback is disabled")
    required = (_NAND_REQUIRED if kind in NAND_COMPONENT_KINDS else
                _SRAM_REQUIRED if kind in SRAM_COMPONENT_KINDS else _DRAM_REQUIRED)
    missing = sorted(name for name in required if raw.get(name) is None)
    if kind in NAND_COMPONENT_KINDS and "max_outstanding_write_requests" not in raw:
        missing.append("max_outstanding_write_requests")  # Explicit None shares the combined limit.
    if raw.get("control_mode") == "jedec_hbm":
        missing += sorted(name for name in (*HBM_CONTROL_TIMING_FIELDS, "activated_row_msb", "refresh_policy")
                          if name not in raw)
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
        parsed = _parse_complete_config(raw)
    except HBMConfigError as exc:
        exc.add_context("component {} ({})".format(component_id, kind))
        raise
    except (TypeError, ValueError) as exc:
        raise ValueError(prefix + "invalid physical_memory_config: " + str(exc)) from exc
    allowed = (
        {"SRAM"} if kind in SRAM_COMPONENT_KINDS
        else {"HBF"} if kind == "hbf" else {"SSD"} if kind in NAND_COMPONENT_KINDS
        else {"HBM"} if kind in {"hbm", "hbm_stack"}
        else {"GDDR"} if kind in {"gddr", "gddr_memory"}
        else {"DDR"} if kind in {"ddr", "ddr_memory"}
        else {"LPDDR"} if kind in {"lpddr", "lpddr_memory"}
        else {"DDR", "LPDDR", "HBM", "GDDR"}
    )
    if parsed.kind.value not in allowed:
        raise ValueError(prefix + "physical_memory_config.kind must be " + "/".join(sorted(allowed)))
    _validate_standard_hbm_organization(parsed, prefix)
    return parsed


def _validated_component_contract(component):
    """Borrow a private config/caps pair only for unchanged component content.

    Component identity alone is insufficient: frozen IR still contains mutable
    metadata. Recompute the complete raw-config key on every access, including
    typed configs' nested metadata, and include every component-level guard.
    Strong ownership plus the identity check prevents recycled object ids.
    """
    metadata = component.metadata
    key = None if type(metadata) is not dict else _physical_content_key((
        component.component_id, component.kind, component.normalized_kind,
        component.capacity_bytes, component.shared_bandwidth_gbps,
        component.read_bandwidth_gbps, component.write_bandwidth_gbps,
        "memory_execution_mode" in metadata, metadata.get("physical_memory_config"),
    ))
    marker = id(component)
    if key is not None:
        with _COMPONENT_CONTRACTS_LOCK:
            cached = _COMPONENT_CONTRACTS.get(marker)
            if cached is not None and cached[0] is component and cached[1] == key:
                _COMPONENT_CONTRACTS.move_to_end(marker)
                return cached[2], cached[3], True
    prefix = "component {} ({}): ".format(component.component_id, component.kind)
    if "memory_execution_mode" in component.metadata:
        raise ValueError(prefix + "memory_execution_mode is no longer configurable; remove the obsolete setting. All memory execution requires its physical DRAM/NAND core")
    parsed = parse_required_physical_config(
        component.component_id, component.normalized_kind,
        component.metadata.get("physical_memory_config"),
    )
    capacity = component.capacity_bytes
    if capacity <= 0 or parsed.capacity_bytes != capacity:
        raise ValueError(prefix + "capacity_bytes must be positive and equal physical_memory_config capacity (component={}, physical={})".format(capacity, parsed.capacity_bytes))
    caps = physical_memory_bandwidth_caps_gbps(component, physical_config=parsed)
    if key is None:
        return parsed, caps, False
    # The parser may retain nested authored metadata on a cache miss. No
    # caller, including public require(), may ever receive that private copy.
    stored = deepcopy(parsed)
    with _COMPONENT_CONTRACTS_LOCK:
        _COMPONENT_CONTRACTS[marker] = (component, key, stored, caps)
        _COMPONENT_CONTRACTS.move_to_end(marker)
        if len(_COMPONENT_CONTRACTS) > 256:
            _COMPONENT_CONTRACTS.popitem(last=False)
    return stored, caps, True


def require_physical_memory_config(component):
    """Return an independently owned validated config, or identify bad input."""
    parsed, _caps, private = _validated_component_contract(component)
    return deepcopy(parsed) if private else parsed


def _borrow_physical_memory_config(component):
    """Read-only internal use; live cores must isolate their mutable metadata."""
    return _validated_component_contract(component)[0]


def physical_memory_bandwidth_caps_gbps(component, *, physical_config=None):
    """Return the core's shared/read/write ceilings in Gb/s.

    A prevalidated typed config may be supplied by a caller that already used
    ``require_physical_memory_config``. Component bandwidth declarations are
    validated ceilings for effective profiles; they do not retime the core or
    replace its physical interface values in reports.
    """
    if physical_config is None:
        return _validated_component_contract(component)[1]
    parsed = physical_config
    if isinstance(parsed, (DramConfig, SramConfig)):
        caps = (parsed.physical_interface_bandwidth_gb_s,
                parsed.directional_bandwidth_gb_s(Operation.READ),
                parsed.directional_bandwidth_gb_s(Operation.WRITE))
    elif isinstance(parsed, NandConfig):
        caps = (parsed.host_bandwidth_gb_s,
                parsed.host_bandwidth_for(Operation.READ),
                parsed.host_bandwidth_for(Operation.WRITE))
    else:
        raise TypeError("physical_config must be a validated DramConfig, NandConfig or SramConfig")
    caps = tuple(float(value) * 8.0 for value in caps)
    for label, declared, physical in zip(
        ("shared", "read", "write"),
        (component.shared_bandwidth_gbps, component.read_bandwidth_gbps,
         component.write_bandwidth_gbps), caps,
    ):
        if declared > physical * (1.0 + 1e-12):
            raise ValueError(
                "component {} ({}): component {} bandwidth exceeds physical memory {} bandwidth".format(
                    component.component_id, component.kind, label,
                    "interface" if label == "shared" else label))
    return caps


def parse_execution_physical_config(raw):
    """Accept a typed core config or require complete serialized execution input.

    Convenience constructors may supply analytical defaults before this
    boundary. A serialized execution request must carry those choices rather
    than silently acquiring a different geometry or timing configuration.
    """
    if isinstance(raw, (DramConfig, NandConfig, SramConfig)):
        _validate_standard_hbm_organization(raw, "execution: ")
        return raw
    if not isinstance(raw, Mapping) or raw.get("kind") is None:
        raise ValueError("physical_memory_config missing explicit fields: kind")
    value = raw["kind"]
    kind = str(getattr(value, "value", value)).strip().upper().replace("-", "")
    return parse_required_physical_config("execution", kind.lower(), raw)


def reconfigure_physical_config(config, *, overrides=None, expand_geometry=False):
    """Update a complete physical config through the execution contract.

    Overrides use the native DRAM/NAND field names and units. Metadata is
    merged without sharing caller-owned dictionaries. Capacity is never
    rounded or silently increased. With ``expand_geometry=True``, only the
    number of rows per bank or blocks per plane may grow to fit that declared
    capacity; all other geometry remains explicitly authored.
    """
    if not isinstance(expand_geometry, bool):
        raise ValueError("expand_geometry must be a boolean")
    parsed = parse_execution_physical_config(config)
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("physical config overrides must be a mapping")
    allowed = {item.name for item in fields(parsed)}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError("unknown physical config override fields: " + ", ".join(sorted(map(str, unknown))))
    payload = asdict(parsed)
    updates = deepcopy(dict(overrides))
    if isinstance(parsed, DramConfig) and updates.get("control_mode", parsed.control_mode) == "jedec_hbm":
        analytical_only = {"open_ns", "close_ns", "read_recovery_ns", "write_recovery_ns", "burst_interval_ns"}
        unsupported = analytical_only.intersection(updates)
        if unsupported:
            raise ValueError("JEDEC HBM overrides require command-anchored timing fields; analytical-only or derived fields: "
                             + ", ".join(sorted(unsupported)))
    if "metadata" in updates:
        if not isinstance(updates["metadata"], Mapping):
            raise ValueError("physical config metadata must be a mapping")
        updates["metadata"] = {**payload["metadata"], **updates["metadata"]}
    payload.update(updates)
    hbm_dependencies = isinstance(parsed, DramConfig) and payload.get("control_mode") == "jedec_hbm"
    if hbm_dependencies:
        payload = normalize_hbm_dependencies(payload, previous=asdict(parsed), explicit_fields=updates)
    rows_before_expansion = payload.get("rows_per_bank")
    if expand_geometry:
        def positive(name):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be a positive integer")
            return value

        capacity = payload.get("capacity_bytes")
        if capacity is not None:
            capacity = positive("capacity_bytes")
            if isinstance(parsed, DramConfig):
                lanes = (positive("data_lanes") if payload.get("data_lanes") is not None else
                         positive("channels") * positive("subchannels_per_channel") * positive("pseudo_channels_per_channel"))
                row_bytes = positive("row_bytes")
                addressable_dies = (1 if payload.get("bank_geometry_scope", "per_die") == "per_stack"
                                    else positive("dies_per_stack"))
                # lanes here is per stack. Independent interfaces flatten
                # stacks into execution lanes, without another capacity copy.
                stripe = (positive("stacks") * addressable_dies * lanes *
                          positive("ranks_per_channel") * positive("bank_groups_per_rank") *
                          positive("banks_per_group") * row_bytes)
                rows = max(positive("rows_per_bank"), (capacity + stripe - 1) // stripe)
                # The address mapper requires the interleave quantum to divide
                # every bank, including when it spans more than one row.
                if payload.get("interleave_bytes") is not None:
                    interleave = positive("interleave_bytes")
                    row_multiple = interleave // gcd(interleave, row_bytes)
                    rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
                payload["rows_per_bank"] = rows
            elif isinstance(parsed, NandConfig):
                stripe = 1
                for name in ("channels", "targets_per_channel", "dies_per_target", "luns_per_die",
                             "planes_per_lun", "pages_per_block", "page_bytes"):
                    stripe *= positive(name)
                payload["blocks_per_plane"] = max(positive("blocks_per_plane"), (capacity + stripe - 1) // stripe)
    if hbm_dependencies and payload.get("rows_per_bank") != rows_before_expansion:
        payload = normalize_hbm_dependencies(payload, previous=asdict(parsed),
                                             explicit_fields=set(updates) | {"rows_per_bank"})
    if isinstance(parsed, DramConfig) and not hbm_dependencies:
        geometry_fields = {
            "kind", "generation", "channels", "subchannels_per_channel",
            "pseudo_channels_per_channel", "stacks", "dies_per_stack",
            "ranks_per_channel", "bank_groups_per_rank", "banks_per_group",
            "rows_per_bank", "row_bytes", "burst_bytes", "data_width_bits",
            "data_lanes", "interleave_bytes", "bank_geometry_scope", "stack_interface_mode",
        }
        changed = {name for name in geometry_fields if payload[name] != getattr(parsed, name)}
        sourced_statuses = {"standard_organization", "explicit_assumption"}
        if changed and (parsed.metadata.get("geometry_status") in sourced_statuses
                        or payload["metadata"].get("geometry_status") in sourced_statuses):
            metadata = deepcopy(payload["metadata"])
            metadata["geometry_status"] = "explicit_assumption"
            metadata["parameter_basis"] = "explicit_organization_with_analytical_timing"
            metadata["vendor_verified"] = False
            metadata["geometry_basis"] = "explicitly modified organization; no longer a standard organization preset"
            # These keys describe the whole organization. Field-level sources
            # for unchanged facts remain useful, but cannot certify overrides.
            for name in ("geometry_source", "geometry_sources", "geometry_standard",
                         "geometry_verified", "organization_source", "organization_basis",
                         "organization_selection", "die_density_gbits", "sid_count",
                         "physical_channels_per_stack", "physical_channel_width_bits",
                         "pseudo_channel_width_bits", "stack_data_width_bits",
                         "banks_per_pseudo_channel", "bank_groups_per_pseudo_channel",
                         "capacity_formula", "jedec_standard", "jedec_reference_file",
                         "source_provenance"):
                metadata.pop(name, None)
            provenance = dict(metadata.get("parameter_provenance", {}))
            if changed & {"kind", "generation", "pseudo_channels_per_channel"}:
                for name in ("channel_operation_mode", "channel_operation_mode_basis"):
                    metadata.pop(name, None)
                    provenance.pop(name, None)
            if changed & {"kind", "generation", "row_bytes"}:
                # Effective and activated page sizes describe the original
                # organization. They no longer explain a different row size
                # or generation; source_documents retains the source itself.
                for name in ("effective_addressed_page_bytes", "activated_page_bytes",
                             "row_bytes_basis", "geometry_limitations"):
                    metadata.pop(name, None)
                    provenance.pop(name, None)
            for name in changed:
                automatic = name == "rows_per_bank" and payload[name] != updates.get(name, getattr(parsed, name))
                provenance[name] = {
                    "status": "explicit_assumption",
                    "source": "capacity_expansion" if automatic else "physical_config_override",
                }
            metadata["parameter_provenance"] = provenance
            payload["metadata"] = metadata
        if parsed.control_mode == "jedec_hbm" or payload["control_mode"] == "jedec_hbm":
            metadata = deepcopy(payload["metadata"])
            provenance = dict(metadata.get("parameter_provenance", {}))
            timing_fields = set(HBM_CONTROL_TIMING_FIELDS) | set(DRAM_BANK_GROUP_TIMING_FIELDS) | {
                "read_latency_ns", "write_latency_ns", "open_ns", "close_ns", "burst_interval_ns",
                "read_recovery_ns", "write_recovery_ns", "read_to_write_ns", "write_to_read_ns",
                "refresh_policy", "control_mode", "activated_row_msb",
            }
            for name in timing_fields.intersection(updates):
                if payload[name] != getattr(parsed, name):
                    provenance[name] = {"status": "explicit_assumption", "source": "physical_config_override",
                                        "basis": "authored timing/control experiment; not a JEDEC or vendor timing value"}
            unknown = metadata.get("unconfigured_timing_parameters", [])
            metadata["unconfigured_timing_parameters"] = [name for name in unknown if not payload.get(name)]
            if payload["control_mode"] != "jedec_hbm":
                metadata["control_status"] = "analytical_control"
                metadata["control_basis"] = "explicit analytical model selection; JEDEC command rules are not executed"
            elif changed:
                metadata["control_status"] = "jedec_command_model_on_custom_geometry"
            metadata["parameter_provenance"] = provenance
            payload["metadata"] = metadata
    if isinstance(parsed, NandConfig):
        changed = {item.name for item in fields(parsed)
                   if item.name != "metadata" and payload[item.name] != getattr(parsed, item.name)}
        if changed:
            metadata = deepcopy(payload["metadata"])
            provenance = dict(metadata.get("parameter_provenance", {}))
            for name in changed:
                provenance[name] = {
                    "value": payload[name], "status": "explicit_assumption",
                    "source": "capacity_expansion" if name == "blocks_per_plane" and name not in updates else "physical_config_override",
                }
            if changed.intersection({"channels", "targets_per_channel", "dies_per_target", "luns_per_die",
                                     "planes_per_lun", "planes_independent", "parallel_units",
                                     "blocks_per_plane", "pages_per_block", "page_bytes", "host_channels"}):
                metadata["geometry_status"] = "explicit_assumption"
                metadata["geometry_basis"] = "explicitly modified NAND organization; reference sources describe the original preset only"
            metadata["parameter_provenance"] = provenance
            payload["metadata"] = metadata
    # A complete serialized round-trip deliberately reuses the same strict
    # boundary as execution, rather than dataclass or example-helper defaults.
    result = parse_execution_physical_config(payload)
    if isinstance(result, DramConfig) and "raw_capacity_bytes" in result.metadata:
        # Visible-capacity overrides can leave geometry unchanged. Keep this
        # sourced organization's displayed capacity accounting synchronized.
        metadata = deepcopy(result.metadata)
        metadata.update(
            raw_capacity_bytes=result.computed_capacity_bytes,
            raw_stack_capacity_bytes=result.computed_capacity_bytes // result.stacks,
            visible_capacity_bytes=result.capacity_bytes,
            unexposed_capacity_bytes=result.computed_capacity_bytes - result.capacity_bytes,
        )
        result = replace(result, metadata=metadata)
    return result
