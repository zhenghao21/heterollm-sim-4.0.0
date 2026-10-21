"""Explicit physical slots for fixed-shape planner segments.

QKV and dense MLP segments have no context-shaped memory views. Their captured
physical descriptors need only namespace binding, request-owned generations,
and stable-address rebinding. Unsupported declaration families are lowered
normally; no value is inferred from samples of successive Tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Optional, Tuple


_PHYSICAL_FIELDS = frozenset({
    "physical_memory_config", "memory_accesses", "physical_memory_component_id",
    "physical_owner", "physical_address_scope", "physical_address_bindings",
    "physical_allocations", "physical_memory_configs", "hbf_host_staging_by_owner",
    "hbf_staging_replaced_link_ids", "hbf_preloaded_buffers",
    "buffer_accesses", "logical_buffer_accesses", "address_precision",
    "cache_line_identity_coverage",
})


def _copy_staging_contracts(contracts):
    """Match _attach_hbf_host_staging's exact copy/borrow boundary."""
    return {owner: {**contract, "routes": {
        direction: tuple({**hop, "resource_capacities": dict(hop["resource_capacities"])}
                         for hop in hops)
        for direction, hops in contract["routes"].items()}}
        for owner, contract in contracts.items()}


def _copy_physical_field(key, value):
    if key in {"memory_accesses", "physical_address_bindings", "buffer_accesses"}:
        rows = tuple(dict(row) for row in value)
        return list(rows) if type(value) is list else rows
    if key == "physical_memory_configs":
        return dict(value)
    if key == "hbf_host_staging_by_owner":
        return _copy_staging_contracts(value)
    # Raw configurations and typed HOST configurations are borrowed from the
    # immutable scenario by ordinary lowering too. Deep-copying them would
    # strengthen ownership and repeatedly copy hundreds of fixed fields.
    return value


_GENERATION_SCALAR_TYPES = frozenset((str, int, float, bool, bytes, type(None)))


def _has_explicit_generation(value) -> bool:
    # Scalar leaves cannot contain a declaration. Most planner metadata is
    # scalar; avoid two abstract-container checks for every such leaf.
    if type(value) in _GENERATION_SCALAR_TYPES:
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (isinstance(key, str) and key.endswith("generation")
                    and item is not None and item != 0):
                return True
            if _has_explicit_generation(item):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_has_explicit_generation(item) for item in value)
    return False


@dataclass(frozen=True)
class StaticPhysicalBinding:
    generation_slots: Tuple[Tuple[str, int, str], ...]
    address_slots: Tuple[Tuple[Tuple[Tuple[str, int], ...], int, int], ...]
    metadata_fields: Tuple[Tuple[str, object], ...] = ()
    namespace_slots: Tuple[Tuple[str, int, str], ...] = ()

    @classmethod
    def capture(cls, original, attached, *, identity_is_dynamic=None) -> Optional["StaticPhysicalBinding"]:
        if (original.metadata.get("physical_memory_config") is not None
                or original.metadata.get("stateful_l2") is not None
                or attached.metadata.get("stateful_l2") is not None
                or original.demands != attached.demands
                or _has_explicit_generation(original.metadata)):
            return None
        phase = original.metadata.get("phase_metadata", {})
        if isinstance(phase, Mapping) and phase.get("direct_memory_access") is not None:
            return None
        metadata = attached.metadata
        if metadata.get("physical_memory_config") is None:
            return cls((), ()) if original.metadata == metadata else None
        changed = {key for key, value in original.metadata.items()
                   if key not in metadata or metadata[key] != value}
        if changed - {"buffer_accesses"}:
            return None
        if "buffer_accesses" in changed and (
                original.metadata.get("buffer_view_cache_policy") != "aggregate_directional_v1"
                or metadata.get("logical_buffer_accesses") is not original.metadata["buffer_accesses"]
                or metadata.get("address_precision") != "analytical_aggregate_cache_fractional_ranges"
                or metadata.get("cache_line_identity_coverage") != "partial"):
            return None
        added_fields = tuple((key, value) for key, value in metadata.items()
                             if key not in original.metadata or key in changed)
        if any(key not in _PHYSICAL_FIELDS for key, _value in added_fields):
            return None
        if metadata.get("physical_allocations"):
            return None
        from .memory_types import DramConfig, SramConfig
        from .physical_contract import parse_execution_physical_config
        configurations = dict(metadata.get("physical_memory_configs", {}))
        owner = str(metadata.get("physical_owner", ""))
        configurations.setdefault(owner, metadata["physical_memory_config"])
        geometries = {}
        for current_owner, raw in configurations.items():
            configured = parse_execution_physical_config(raw)
            geometries[current_owner] = (
                configured.effective_capacity_bytes,
                configured.burst_bytes if isinstance(configured, (DramConfig, SramConfig)) else configured.page_bytes,
            )
        generation = int.from_bytes(b"\x01" + original.request_id.encode("utf-8"), "big")
        generation_slots, address_groups, namespace_slots = [], {}, []
        collections = ("memory_accesses", "physical_address_bindings") + (
            ("buffer_accesses",) if "buffer_accesses" in changed else ())
        for collection in collections:
            rows = metadata.get(collection, ())
            if not isinstance(rows, (tuple, list)):
                return None
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    return None
                if any(type(value) not in {str, int, float, bool, type(None)}
                       for value in row.values()):
                    return None
                if identity_is_dynamic is not None:
                    for field in ("buffer_id", "alias_of"):
                        value = row.get(field)
                        if isinstance(value, str) and identity_is_dynamic(value):
                            namespace_slots.append((collection, index, field))
                if collection == "buffer_accesses":
                    continue
                for field in ("allocation_generation", "alias_generation"):
                    value = row.get(field)
                    if value is None or value == 0:
                        continue
                    if value != generation:
                        return None
                    generation_slots.append((collection, index, field))
                source = row.get("address_source")
                if source == "stable_buffer_tensor_offset":
                    row_owner = str(row.get("physical_owner") or owner)
                    capacity, burst = geometries[row_owner]
                    if (not row.get("buffer_id")
                            or type(row.get("offset_bytes", 0)) is not int
                            or type(row.get("byte_count")) is not int
                            or row.get("allocation_size_bytes") is not None
                            and type(row["allocation_size_bytes"]) is not int):
                        return None
                    if identity_is_dynamic is None or identity_is_dynamic(row["buffer_id"]):
                        key = (row["buffer_id"], row.get("offset_bytes", 0), row["byte_count"],
                               row.get("allocation_size_bytes"), capacity, burst)
                        address_groups.setdefault(key, []).append((collection, index))
                elif source not in {"explicit_physical_address", "explicit_buffer_tensor_address"}:
                    return None
        if any(row[2] != 0 for row in metadata.get("hbf_preloaded_buffers", ())):
            return None
        if identity_is_dynamic is not None and any(
                identity_is_dynamic(row[1]) for row in metadata.get("hbf_preloaded_buffers", ())):
            return None
        return cls(tuple(generation_slots), tuple(
            (tuple(slots), key[-2], key[-1]) for key, slots in address_groups.items()),
            tuple((key, None if key == "logical_buffer_accesses" else _copy_physical_field(key, value))
                  for key, value in added_fields),
            tuple(namespace_slots))

    def bind(self, metadata, request_id: str, stable_address, bind_namespace=None) -> None:
        logical_views = metadata.get("buffer_accesses")
        for key, value in self.metadata_fields:
            metadata[key] = logical_views if key == "logical_buffer_accesses" else _copy_physical_field(key, value)
        if bind_namespace is not None:
            for collection, index, field in self.namespace_slots:
                row = metadata[collection][index]
                row[field] = bind_namespace(row[field])
        if self.generation_slots:
            generation = int.from_bytes(b"\x01" + request_id.encode("utf-8"), "big")
            for collection, index, field in self.generation_slots:
                metadata[collection][index][field] = generation
        for slots, capacity, burst in self.address_slots:
            collection, index = slots[0]
            row = metadata[collection][index]
            address = stable_address(row["buffer_id"], row.get("offset_bytes", 0),
                row["byte_count"], capacity, burst, row.get("allocation_size_bytes"))
            for collection, index in slots:
                metadata[collection][index]["address"] = address
