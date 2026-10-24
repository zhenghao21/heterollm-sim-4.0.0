"""Small, shared physical data-access service.

This module deliberately stops at service generation.  It does not schedule
tasks or choose routes; callers can lower the returned phases into the
existing :class:`ResourceDemand`/``TaskSpec`` contracts.
"""

from __future__ import annotations

import math
import copy
from contextlib import contextmanager
from contextvars import ContextVar
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple

from .contracts import EvidenceStatus, ResourceDemand, TaskCategory, TaskSpec
from .ir import ComponentSpec, LinkSpec, OFFLOAD_STORAGE_COMPONENT_KINDS, default_memory_resource_id
from .memory_service import (realtime_memory_metrics, DRAM_CONTROL_COUNT_FIELDS,
                             DRAM_CONTROL_TIME_FIELDS, DRAM_OPTIONAL_COUNTER_FIELDS)
from .nand_core import _copy_nand_read_cache


# Ordered request batching retains individual DRAM admission, accesses
# and results. Keep the switch private for strict serial equivalence tests.
_USE_COMPILED_DRAM_BATCH = True

_DEFER_ENDPOINT_CONSTRUCTION = ContextVar("defer_endpoint_construction", default=None)


@contextmanager
def deferred_endpoint_construction(enabled=True):
    """Build descriptors when the native caller owns live physical dispatch.

    This does not change standalone estimates: they still price the cold
    physical preview. Online lowering must not execute an extra cold device
    simulation whose result its existing deferred cost pass discards.
    """
    # The planner owns one immutable authored scenario during this scope.
    # Drop the identity cache at exit so a later author edit is revalidated.
    token = _DEFER_ENDPOINT_CONSTRUCTION.set(
        {} if enabled and _native_construction_callbacks() else None)
    try:
        yield
    finally:
        _DEFER_ENDPOINT_CONSTRUCTION.reset(token)

_PHYSICAL_ADDITIVE_COUNTERS = frozenset((
    "burst_count", "row_hits", "row_misses", "row_conflicts",
    "scheduler_reordered_requests", "scheduler_batches", "controller_direction_switches",
    "read_write_switches", "queue_wait_ns", "refresh_wait_ns",
    "turnaround_wait_ns", "pages_read", "pages_programmed",
    "pages_touched", "media_waves", "erase_operations",
    "read_operations", "program_operations", "rmw_read_operations",
    "page_access_payload_bytes", "page_access_capacity_bytes", "host_admission_samples",
    "host_transfer_bytes", "internal_transfer_bytes",
    "write_buffer_wait_ns", "write_buffer_bytes", "merge_bytes",
    "staging_read_hits", "staging_read_merges", "staging_write_pages",
    "read_cache_hits", "read_cache_misses",
    "merged_pages", "auto_drain_physical_programs",
) + DRAM_CONTROL_COUNT_FIELDS + DRAM_CONTROL_TIME_FIELDS)
_PHYSICAL_PEAK_COUNTERS = frozenset((
    "write_buffer_peak_bytes", "host_outstanding_at_admission_peak",
    "host_read_outstanding_at_admission_peak", "host_write_outstanding_at_admission_peak",
))


def _execution_config_cache_key(value):
    """Include value types so e.g. True cannot reuse a validated integer 1."""
    if type(value) in {str, int, float, bool, type(None)} or isinstance(value, Enum):
        return type(value), value
    if type(value) is dict:
        return dict, tuple((_execution_config_cache_key(key), _execution_config_cache_key(item))
                           for key, item in value.items())
    if type(value) in {list, tuple}:
        return type(value), tuple(_execution_config_cache_key(item) for item in value)
    raise TypeError("custom configuration values are parsed without caching")


class _OverlapCompletionIndex:
    """Online interval overlap maxima for physical access ordering.

    Each update stores a completion timestamp on the canonical segment-tree
    cover of one byte range.  A query returns the maximum value among all
    stored ranges intersecting the requested range.  The node-local value is
    included on partial traversal because every range stored there covers the
    whole node interval, hence it intersects whenever that node intersects.
    This preserves the old pairwise ordering rule while reducing each access
    from an O(previous_accesses) scan to O(log endpoints).
    """

    def __init__(self, endpoints: Sequence[int]) -> None:
        self._endpoints = tuple(endpoints)
        self._segments = max(1, len(self._endpoints) - 1)
        self._values = [0.0] * (4 * self._segments + 4)
        self._subtree = [0.0] * (4 * self._segments + 4)

    def _indices(self, address: int, byte_count: int) -> Tuple[int, int]:
        start = bisect_left(self._endpoints, address)
        end = bisect_left(self._endpoints, address + byte_count)
        # Validated accesses contribute both endpoints, so this is normally
        # an exact segment range.  Keep the bounds checked for defensive use.
        return max(0, min(self._segments - 1, start)), max(1, min(self._segments, end))

    def update(self, address: int, byte_count: int, completion_ns: float) -> None:
        left, right = self._indices(address, byte_count)
        if right <= left:
            return

        def visit(node: int, lo: int, hi: int) -> None:
            if right <= lo or hi <= left:
                return
            if left <= lo and hi <= right:
                self._values[node] = max(self._values[node], completion_ns)
            else:
                mid = (lo + hi) // 2
                visit(node * 2, lo, mid)
                visit(node * 2 + 1, mid, hi)
            child_max = 0.0
            if hi - lo > 1:
                child_max = max(self._subtree[node * 2], self._subtree[node * 2 + 1])
            self._subtree[node] = max(self._values[node], child_max)

        visit(1, 0, self._segments)

    def query(self, address: int, byte_count: int) -> float:
        left, right = self._indices(address, byte_count)
        if right <= left:
            return 0.0

        def visit(node: int, lo: int, hi: int) -> float:
            if right <= lo or hi <= left:
                return 0.0
            if left <= lo and hi <= right:
                return self._subtree[node]
            mid = (lo + hi) // 2
            return max(
                self._values[node],
                visit(node * 2, lo, mid),
                visit(node * 2 + 1, mid, hi),
            )

        return visit(1, 0, self._segments)


@dataclass
class _PhysicalRuntime:
    core: Any
    clock_ns: float = 0.0


def _copy_hbm_control_state(state: dict[str, Any]) -> dict[str, Any]:
    """Copy mutable controller state without reconstructing bank mappings.

    Every visited HBM bank holds scalar clocks and a frozen AddressMapping.
    Rebuilding those mappings via deepcopy dominates per-task snapshots on
    large stacks. Copy bank dictionaries directly, sharing only immutable
    values; copy ordinary history containers without deepcopy's per-scalar
    dispatch, retaining its behavior for extension objects.
    The memo also preserves aliases within the saved state without sharing
    mutable objects with the live controller.
    """
    from .memory_types import AddressMapping

    banks = state.get("banks")
    if type(banks) is not dict:
        return copy.deepcopy(state)
    result = {}
    bank_copies = {name: row.copy() for name, row in banks.items() if type(row) is dict}
    memo = {id(state): result, id(banks): bank_copies}
    # Register all rows first so extension fields referring to another bank
    # retain that relationship in the independent copy.
    for name, row in banks.items():
        if type(row) is dict:
            bank_copies[name] = memo.setdefault(id(row), bank_copies[name])
    immutable_types = (int, float, str, bool, type(None), AddressMapping)
    for name, row in banks.items():
        if type(row) is not dict:
            bank_copies[name] = _copy_hbm_history(row, memo)
            continue
        for field_name, value in row.items():
            if type(value) in immutable_types:
                if type(value) is AddressMapping:
                    memo[id(value)] = value
            else:
                bank_copies[name][field_name] = _copy_hbm_history(value, memo)
    result.update({name: _copy_hbm_history(value, memo) for name, value in state.items()})
    return result


def _copy_hbm_history(value, memo):
    """Copy clock histories while preserving deepcopy's aliases and cycles.

    The normal histories contain exact builtin dicts/lists, scalar clocks
    and tuple keys of integers. Sharing these immutable keys avoids walking
    every SID/group tuple through deepcopy before each physical task. Custom
    containers and extension objects still use their own deepcopy protocol.
    """
    atomic = (int, float, str, bool, type(None))
    if type(value) in atomic:
        return memo.get(id(value), value)
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if type(value) is dict:
        result = {}
        memo[identity] = result
        for key, item in value.items():
            cloned = memo.get(id(item), item) if type(item) in atomic else _copy_hbm_history(item, memo)
            if type(key) in atomic:
                key = memo.get(id(key), key)
            elif type(key) is tuple and id(key) in memo:
                key = memo[id(key)]
            elif not (type(key) is tuple and all(type(part) in atomic
                       and memo.get(id(part), part) is part for part in key)):
                key = copy.deepcopy(key, memo)
            result[key] = cloned
        return result
    if type(value) is list:
        result = []
        memo[identity] = result
        result.extend(memo.get(id(item), item) if type(item) in atomic
                      else _copy_hbm_history(item, memo) for item in value)
        return result
    return copy.deepcopy(value, memo)


@dataclass
class PhysicalRuntimeContext:
    """State for one simulation run's physical devices.

    Runtime ownership is explicit and context-local.  A context never silently
    merges two configurations under one physical owner.
    """

    config: Any = None
    physical_owner: Optional[str] = None
    runtimes: dict[str, _PhysicalRuntime] = field(default_factory=dict)
    allocators: dict[str, Any] = field(default_factory=dict)
    timeline: Any = None
    # Exact event/replay runs retain expanded stage/mapping details. Aggregate
    # cohort estimates set this false to avoid allocating those transient
    # tuples; scalar timing and traffic counters remain unchanged.
    capture_details: bool = True
    _committed_owners: set[str] = field(default_factory=set, repr=False)
    hbf_staging: dict[str, Any] = field(default_factory=dict, repr=False)
    _owner_config_cache: dict[str, Any] = field(default_factory=dict, repr=False)
    _execution_config_cache: dict[Any, Any] = field(default_factory=dict, repr=False)
    _physical_demand_cache: dict[Any, ResourceDemand] = field(default_factory=dict, repr=False)

    def parse_config(self, raw: Any) -> Any:
        """Reuse complete serialized contracts while detecting mutable edits."""
        from .physical_contract import parse_execution_physical_config
        if type(raw) is not dict:
            return parse_execution_physical_config(raw)
        try:
            key = _execution_config_cache_key(raw)
        except (TypeError, RecursionError):
            return parse_execution_physical_config(raw)
        cached = self._execution_config_cache.get(key)
        if cached is not None:
            parsed, parsed_key = cached
            try:
                if _execution_config_cache_key(vars(parsed)) == parsed_key:
                    return parsed
            except (TypeError, RecursionError):
                pass
        # A cached result must not share caller-owned mutable metadata. On a
        # later hit its saved value key detects edits to the returned object.
        parsed = copy.deepcopy(parse_execution_physical_config(raw))
        try:
            parsed_key = _execution_config_cache_key(vars(parsed))
        except (TypeError, RecursionError):
            return parsed
        if len(self._execution_config_cache) >= 16:
            self._execution_config_cache.clear()
        self._execution_config_cache[key] = (parsed, parsed_key)
        return parsed

    def release_staging_buffer(self, owner: str, buffer_id: str, generation: int,
                               *, arrival_ns: float = 0.0) -> None:
        """Discard dead logical data; request completion does not force a tail write."""
        from .hbf_staging_runtime import release
        release(self, owner, buffer_id, generation, arrival_ns)

    def __post_init__(self) -> None:
        if self.timeline is None:
            from .memory_transfer import ResourceTimeline
            self.timeline = ResourceTimeline()
        if not isinstance(self.capture_details, bool):
            raise ValueError("capture_details must be a boolean")
        if self.config is not None or self.physical_owner is not None:
            if self.config is None or not self.physical_owner:
                raise ValueError("config and physical_owner must be provided together")
            self.runtime(self.config, self.physical_owner)

    def _owner_config(self, config: Any, owner: str) -> Any:
        from .memory_types import DramConfig, SramConfig

        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("physical owner must be non-empty text")
        config = self.parse_config(config)
        cached = self._owner_config_cache.get(owner)
        if cached is not None and config == cached[0]:
            return cached[1]
        authored_config = copy.deepcopy(config)
        metadata = dict(config.metadata)
        if isinstance(config, DramConfig):
            metadata.setdefault("bank_resource_prefix", f"{owner}:dram:bank")
            metadata.setdefault("command_resource_prefix", f"{owner}:dram:command")
            metadata.setdefault("row_command_resource_prefix", f"{owner}:dram:row_command")
            metadata.setdefault("column_command_resource_prefix", f"{owner}:dram:column_command")
            metadata.setdefault("data_resource_prefix", f"{owner}:dram:data")
        elif isinstance(config, SramConfig):
            metadata.setdefault("port_resource_prefix", f"{owner}:sram:port")
        else:
            metadata.setdefault("array_resource_prefix", f"{owner}:nand:array")
            metadata.setdefault("parallel_resource_prefix", f"{owner}:nand:parallel")
            metadata.setdefault("channel_resource_prefix", f"{owner}:nand:channel")
            metadata.setdefault("host_resource_id", f"{owner}:nand:host")
        bound_config = replace(config, metadata=metadata)
        self._owner_config_cache[owner] = (authored_config, bound_config)
        return bound_config

    def runtime(self, config: Any, owner: str) -> _PhysicalRuntime:
        from .memory_types import DramConfig, SramConfig
        from .dram_core import DramCore
        from .nand_core import NandCore
        from .sram_core import SramCore
        from .memory_allocator import PhysicalAddressAllocator

        config = self._owner_config(config, owner)
        key = owner
        current = self.runtimes.get(key)
        if current is not None:
            # Compare the normalized typed contract, not dict insertion order
            # or a repr taken before owner resource prefixes were attached.
            if current.core.config != config:
                raise ValueError(
                    f"physical owner {owner} was configured with multiple physical geometries"
                )
            return current
        # Authored nested metadata must not mutate a live core's contract.
        config = replace(config, metadata=copy.deepcopy(config.metadata))
        metadata = config.metadata
        core = (
            DramCore(config, self.timeline, capture_details=self.capture_details)
            if isinstance(config, DramConfig)
            else SramCore(config, self.timeline, capture_details=self.capture_details)
            if isinstance(config, SramConfig)
            else NandCore(config, self.timeline, capture_details=self.capture_details)
        )
        if isinstance(core, DramCore):
            core._hbm_numeric_enabled = (not self.capture_details
                                         and config.control_mode == "jedec_hbm")
        elif isinstance(core, NandCore):
            core._nand_numeric_enabled = (not self.capture_details
                                          and config.protocol_mode in ("ocp_hbf_0_7", "generic_nand"))
        capacity = config.effective_capacity_bytes
        if capacity > 0:
            # The direct-backing planner uses burst alignment for DRAM and
            # page alignment for NAND.  Keep automatically materialized
            # buffers on the same physical granularity; otherwise a small
            # preceding allocation can shift a page buffer into a second
            # NAND page and create spurious read amplification.
            alignment = (
                int(config.burst_bytes)
                if isinstance(config, (DramConfig, SramConfig))
                else int(config.page_bytes)
            )
            self.allocators.setdefault(key, PhysicalAddressAllocator(capacity, alignment,
                workspace_capacity_bytes=metadata.get("workspace_capacity_bytes", 0)))
        current = _PhysicalRuntime(core=core)
        self.runtimes[key] = current
        return current

    def preview_runtime(
        self,
        config: Any,
        owner: str,
        *,
        capture_details: Optional[bool] = None,
    ) -> _PhysicalRuntime:
        from .memory_types import DramConfig, SramConfig
        config = self._owner_config(config, owner)
        from .dram_core import DramCore
        from .nand_core import NandCore
        from .sram_core import SramCore
        # Preserve detailed preview bills by default.  Summary-only planner
        # callers can disable transient stage/mapping tuples explicitly.
        details = True if capture_details is None else capture_details
        if not isinstance(details, bool):
            raise ValueError("capture_details must be a boolean")
        core = (
            DramCore(config, capture_details=details)
            if isinstance(config, DramConfig)
            else SramCore(config, capture_details=details)
            if isinstance(config, SramConfig)
            else NandCore(config, capture_details=details)
        )
        return _PhysicalRuntime(core=core)

    def snapshot(self, *, transactional: bool = False,
                 owners: Optional[Iterable[str]] = None,
                 _owned_timeline: bool = False) -> dict[str, Any]:
        """Capture a transaction-local rollback point without copying history.

        Physical tasks are submitted one at a time.  The old implementation
        deep-copied every runtime, allocator and the complete timeline before
        each task; once burst interval detail accumulated this duplicated the
        whole run for every in-flight transaction.  Core state and allocator
        entries are small mutable indexes. Only the current request's bounded
        interval recorder is copied; direct link transfers can append to it
        without a Core submission first replacing it.
        """
        from .dram_core import DramCore, _BankState
        from .nand_core import NandCore
        from .hbf_staging_runtime import snapshot as snapshot_staging

        selected_owners = None if owners is None else frozenset(owners)
        native_footprints = {}
        native_owned_hooks = (transactional and _owned_timeline and selected_owners is not None
                              and self._owners_have_native_hooks(selected_owners, _footprints=native_footprints))
        if (transactional and _owned_timeline and selected_owners is not None
                and not native_owned_hooks):
            # Unknown execution callbacks may also edit a sibling runtime.
            # They own neither a narrow resource journal nor an owner subset.
            selected_owners = None
        runtime_state = {}
        for owner, active in self.runtimes.items():
            if selected_owners is not None and owner not in selected_owners:
                continue
            core = active.core
            if not transactional and isinstance(core, DramCore):
                # Public snapshots may outlive an active transaction. Upgrade
                # its undo state before copying controller objects so callers
                # can still finish their original token after a restore.
                core._hbm_before_owned_snapshot()
            if not transactional and isinstance(core, NandCore):
                core._nand_before_reference()
            hbm_transaction = (core.begin_transaction()
                               if transactional and isinstance(core, DramCore) else None)
            nand_transaction = (core.begin_transaction()
                                if transactional and isinstance(core, NandCore) else None)
            # Core indexes (DRAM bank rows / NAND array clocks) grow with the
            # physical address stream.  Deep-copying them for every access
            # recreated the original quadratic memory behaviour.  Their
            # immutable configuration and scalar admission state are enough
            # for the normal commit path; mutable indexes remain owned by the
            # live core and are only consulted again after a successful task.
            core_state = {}
            for key, value in vars(core).items():
                if key in {"config", "timeline"}:
                    continue
                if hbm_transaction is not None and key in {
                    "_banks", "_hbm_control_state", "_hbm_transactions", "_hbm_numeric_state"
                }:
                    continue
                if nand_transaction is not None and key in {
                    "_next_program_page", "_block_wear_state", "_nand_transactions", "_read_cache",
                    "_array_ready", "_buffer_ready", "_nand_numeric_state"
                }:
                    continue
                if key == "_banks" and isinstance(value, dict):
                    # Bank rows are mutable dataclasses; copy the small bank
                    # index so a failed transaction cannot leak row state.
                    # Copy the row and both recovery/activation clocks.
                    # Construct it directly instead of invoking
                    # copy's reflection/reconstruction protocol for every
                    # bank before every physical task. Custom row objects
                    # retain their original shallow-copy behavior.
                    core_state[key] = {
                        name: (_BankState(bank.open_row, bank.ready_ns, bank.activation_ready_ns)
                               if type(bank) is _BankState else copy.copy(bank))
                        for name, bank in value.items()
                    }
                elif key in {"_array_ready", "_buffer_ready", "_next_program_page", "_host_acceptance_ns"} and isinstance(value, dict):
                    # NAND readiness values are scalars, so a shallow dict
                    # copy is sufficient and avoids copying timeline history.
                    core_state[key] = dict(value)
                elif key == "_host_inflight":
                    core_state[key] = {host: list(commands) for host, commands in value.items()}
                elif key == "_hbm_control_state":
                    core_state[key] = _copy_hbm_control_state(value)
                elif key in {"_bank_group_state", "_channel_inflight"}:
                    # Group command/activation histories contain mutable
                    # per-group clocks and activation windows. They are live
                    # execution state, so a failed task must not modify its
                    # saved retry point through a nested alias.
                    core_state[key] = copy.deepcopy(value)
                elif key in {"_inflight", "_inflight_operations", "_write_buffer_pending", "_hbm_transactions", "_nand_transactions"} and isinstance(value, list):
                    core_state[key] = list(value)
                elif key == "_read_cache":
                    core_state[key] = _copy_nand_read_cache(value)
                elif key == "_staged_pages":
                    # Buffered NAND pages are mutable admission state, not
                    # immutable timeline history. Failed transfers must not
                    # leak staged ranges, cache tags or fill/reader clocks.
                    core_state[key] = copy.deepcopy(value)
                elif key == "_block_wear_state":
                    # Feedback dictionaries contain per-plane/block scalars.
                    # A failed physical task must roll back program/erase wear.
                    core_state[key] = {name: dict(row) for name, row in value.items()}
                elif key == "_observation_events":
                    core_state[key] = list(value)
                else:
                    core_state[key] = value
            core_state["config"] = core.config
            runtime_state[owner] = {
                "active": active,
                "clock_ns": active.clock_ns,
                "core": core_state,
                "hbm_transaction": hbm_transaction,
                "nand_transaction": nand_transaction,
            }
        # A transaction copies an allocation map only if it mutates it; a
        # public snapshot may outlive transactions and keeps its own copy.
        allocator_state = {
            owner: (allocator.begin_transaction() if transactional else dict(allocator._allocations))
            for owner, allocator in self.allocators.items()
            if selected_owners is None or owner in selected_owners
        }
        timeline = self.timeline
        timeline_token = (timeline._begin_owned_transaction()
            if transactional and _owned_timeline
            and self._can_own_timeline(selected_owners, _hooks_checked=native_owned_hooks,
                                      _native_footprints=native_footprints) else None)
        if timeline_token is not None:
            for owner in selected_owners:
                core = self.runtimes[owner].core
                if type(core) is DramCore and core.config.control_mode != "jedec_hbm":
                    timeline._remember_resources(native_footprints[owner])
            if native_owned_hooks:
                for state in runtime_state.values():
                    token = state.get("nand_transaction")
                    if token is not None:
                        token.native_timeline_qualified = True
        timeline_state = ({"_owned_transaction": timeline_token,
                           "lane_available": timeline_token.before_lanes()}
                          if timeline_token is not None else {
            "ready_ns": dict(timeline.ready_ns),
            "directions": dict(timeline.directions),
            "busy_ns": dict(timeline.busy_ns),
            "bytes_moved": dict(timeline.bytes_moved),
            "last_intervals": dict(timeline.last_intervals),
            "_intervals": {rid: list(items) for rid, items in timeline._intervals.items()},
            "_interval_limit": timeline._interval_limit,
            "_interval_count": timeline._interval_count,
            "_intervals_truncated": timeline._intervals_truncated,
            "_touched": set(timeline._touched),
            "_interval_payloads": {rid: list(items) for rid, items in timeline._interval_payloads.items()},
            "lane_available": {
                resource: tuple(lanes)
                for resource, lanes in timeline.lane_available.items()
            },
            "lane_directions": {
                resource: tuple(directions)
                for resource, directions in timeline.lane_directions.items()
            },
        })
        return {
            "runtimes": dict(self.runtimes),
            "runtime_state": runtime_state,
            "allocators": dict(self.allocators),
            "allocator_state": allocator_state,
            "timeline": timeline_state,
            # Native owner qualification also governs sparse publication after
            # a successful call. Unknown callbacks can modify a foreign lane
            # without adding it to the current request's touched-resource bill.
            "_owned_native_writers": bool(native_owned_hooks),
            "committed_owners": set(self._committed_owners),
            "hbf_staging": snapshot_staging(self.hbf_staging, transactional=transactional),
        }

    def _can_own_timeline(self, owners, *, _hooks_checked=False, _native_footprints=None) -> bool:
        """Only closed native writers use sparse timeline undo."""
        from .dram_core import DramCore
        from .nand_core import NandCore
        from .sram_core import SramCore
        from .memory_transfer import ResourceTimeline, _TIMELINE_MAPS, _OWNED_TIMELINE_RESERVE
        from ._compiled_hbm import can_capture_resource_columns
        if (not owners or type(self.timeline) is not ResourceTimeline
                or self.hbf_staging or self.capture_details
                or any(type(getattr(self.timeline, name)) is not dict for name in _TIMELINE_MAPS)
                or type(self.timeline.resource_aliases) is not dict
                or type(self.timeline._touched) is not set
                or getattr(self.timeline.reserve, "__func__", None) is not _OWNED_TIMELINE_RESERVE):
            return False
        if _native_footprints is None:
            _native_footprints = {}
        if not _hooks_checked and not self._owners_have_native_hooks(owners, _footprints=_native_footprints):
            return False
        for owner in owners:
            active = self.runtimes.get(owner)
            if active is None:
                return False
            core = active.core
            if core.capture_details:
                return False
            if type(core) is DramCore:
                if core.config.control_mode != "jedec_hbm":
                    if owner not in _native_footprints:
                        return False
                else:
                    if not core._hbm_numeric_enabled:
                        return False
                    packed = core._hbm_numeric_state
                    if packed is None or not can_capture_resource_columns(self.timeline, packed.ids):
                        return False
            elif type(core) is NandCore:
                if core.config.protocol_mode != "generic_nand" or core._observe_media:
                    return False
            elif type(core) is SramCore:
                pass
            else:
                return False
        return True

    def _owners_have_native_hooks(self, owners, *, _footprints=None) -> bool:
        from .dram_core import DramCore, _OWNED_TIMELINE_DRAM_METHODS
        from .nand_core import NandCore, _OWNED_OWNER_NAND_METHODS
        from .sram_core import SramCore, _OWNED_OWNER_SRAM_METHODS
        from .memory_transfer import ResourceTimeline, _OWNED_TIMELINE_METHODS
        from ._compiled_hbm import _has_native_timeline_hooks
        from ._nand_acceleration import has_native_timeline_hooks as native_nand_hooks
        from ._nand_pack import can_own_timeline, can_own_resource_aliases
        if type(self.timeline) is not ResourceTimeline:
            return False
        from .memory_transfer import _TIMELINE_MAPS
        if (any(type(getattr(self.timeline, name)) is not dict for name in _TIMELINE_MAPS)
                or type(self.timeline.resource_aliases) is not dict
                or type(self.timeline._touched) is not set):
            return False
        if not can_own_resource_aliases(self.timeline.resource_aliases):
            return False
        for name, original in _OWNED_TIMELINE_METHODS:
            method = getattr(self.timeline, name)
            if getattr(method, "__func__", method) is not original:
                return False
        for owner in owners:
            active = self.runtimes.get(owner)
            if active is None:
                # Allocation registration can create this Core after the
                # kernel's transaction boundary. Until its class/hooks are
                # known, a callback may mutate an existing sibling owner.
                # The first dispatch keeps the original complete undo.
                return False
            core = active.core
            core_type = type(core)
            methods = (_OWNED_TIMELINE_DRAM_METHODS if core_type is DramCore else
                       _OWNED_OWNER_NAND_METHODS if core_type is NandCore else
                       _OWNED_OWNER_SRAM_METHODS if core_type is SramCore else None)
            if methods is None:
                return False
            for name, original in methods:
                method = getattr(core, name)
                if getattr(method, "__func__", method) is not original:
                    return False
            packed = core._hbm_numeric_state if core_type is DramCore else None
            if (core_type is DramCore and core.config.control_mode == "jedec_hbm"
                    and not _has_native_timeline_hooks(packed)):
                return False
            if core_type is DramCore and core.config.control_mode != "jedec_hbm":
                from ._compiled_dram import owned_resource_footprint
                resources = owned_resource_footprint(core)
                if resources is None:
                    return False
                if _footprints is not None:
                    _footprints[owner] = resources
            if core_type is NandCore and not native_nand_hooks():
                return False
            if core_type in (NandCore, SramCore):
                try:
                    if core_type is NandCore:
                        from ._nand_acceleration import owned_resource_footprint
                        if any(type(core.config.metadata.get(name, "")) is not str for name in (
                                "array_resource_prefix", "parallel_resource_prefix",
                                "channel_resource_prefix", "host_resource_id")):
                            return False
                        resources = owned_resource_footprint(core)
                    else:
                        prefix = core.config.metadata.get("port_resource_prefix", "sram:port")
                        if type(prefix) is not str or core.config.ports > 65536:
                            return False
                        resources = tuple(self.timeline.canonical_resource_id(f"{prefix}:{index}")
                                          for index in range(core.config.ports))
                except (TypeError, ValueError, RecursionError):
                    return False
                if resources is None or not can_own_timeline(self.timeline, resources):
                    return False
                if _footprints is not None:
                    _footprints[owner] = resources
        return True

    def begin_transaction(self, *, owners: Optional[Iterable[str]] = None,
                          _owned_timeline: bool = False) -> dict[str, Any]:
        """Protect live dispatch by journaling changed HBM and staging state."""
        return self.snapshot(transactional=True, owners=owners, _owned_timeline=_owned_timeline)

    def commit_transaction(self, snapshot: dict[str, Any]) -> None:
        from .hbf_staging_runtime import commit
        for state in snapshot["runtime_state"].values():
            token = state.get("hbm_transaction")
            if token is not None:
                state["active"].core.commit_transaction(token)
            nand_token = state.get("nand_transaction")
            if nand_token is not None:
                state["active"].core.commit_transaction(nand_token)
        commit(snapshot["hbf_staging"])
        for owner, save in snapshot["allocator_state"].items():
            if type(save) is not dict:
                snapshot["allocators"][owner].commit_transaction(save)
        timeline_token = snapshot["timeline"].get("_owned_transaction")
        if timeline_token is not None:
            timeline_token.finish()

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.runtimes.clear()
        self.runtimes.update(snapshot["runtimes"])
        self.allocators.clear()
        self.allocators.update(snapshot["allocators"])
        for owner, state in snapshot["runtime_state"].items():
            active = self.runtimes[owner]
            active.clock_ns = state["clock_ns"]
            journal_state = {}
            token = state.get("hbm_transaction")
            if token is not None:
                active.core.rollback_transaction(token)
                journal_state = {key: active.core.__dict__[key] for key in (
                    "_banks", "_hbm_control_state", "_hbm_transactions", "_hbm_numeric_state")}
            nand_token = state.get("nand_transaction")
            if nand_token is not None:
                active.core.rollback_transaction(nand_token)
                journal_state.update({key: active.core.__dict__[key] for key in (
                    "_next_program_page", "_block_wear_state", "_nand_transactions", "_read_cache",
                    "_array_ready", "_buffer_ready", "_nand_numeric_state")})
            active.core.__dict__.clear()
            # Restored live state must not become the rollback point itself.
            # Otherwise another submission mutates the saved bank rows,
            # staged pages or wear state and a second restore is partial.
            active.core.__dict__.update({
                key: (value if key == "config" else
                      list(value) if key in {"_hbm_transactions", "_nand_transactions"} else
                      _copy_hbm_control_state(value) if key == "_hbm_control_state" else
                      _copy_nand_read_cache(value) if key == "_read_cache" else
                      copy.deepcopy(value))
                for key, value in state["core"].items()
            })
            active.core.__dict__.update(journal_state)
            active.core.timeline = self.timeline
        for owner, allocations in snapshot["allocator_state"].items():
            allocator = self.allocators[owner]
            if type(allocations) is dict:
                allocator._restore_allocations(allocations)
            else:
                allocator.rollback_transaction(allocations)
        timeline_state = snapshot["timeline"]
        timeline = self.timeline
        timeline_token = timeline_state.get("_owned_transaction")
        if timeline_token is not None:
            timeline_token.rollback()
        else:
            for name in ("ready_ns", "directions", "busy_ns", "bytes_moved", "last_intervals"):
                target = getattr(timeline, name)
                target.clear()
                target.update(timeline_state[name])
            timeline._intervals = copy.deepcopy(timeline_state["_intervals"])
            timeline._interval_limit = timeline_state["_interval_limit"]
            timeline._interval_count = timeline_state["_interval_count"]
            timeline._intervals_truncated = timeline_state["_intervals_truncated"]
            timeline._touched = set(timeline_state["_touched"])
            timeline._interval_payloads = copy.deepcopy(timeline_state["_interval_payloads"])
            timeline.lane_available.clear()
            timeline.lane_available.update({
                resource: list(lanes)
                for resource, lanes in timeline_state["lane_available"].items()
            })
            timeline.lane_directions.clear()
            timeline.lane_directions.update({
                resource: list(directions)
                for resource, directions in timeline_state["lane_directions"].items()
            })
        self._committed_owners.clear()
        self._committed_owners.update(snapshot["committed_owners"])
        from .hbf_staging_runtime import restore
        self.hbf_staging = restore(snapshot["hbf_staging"])


def _physical_runtime(
    config: Any,
    owner: str,
    context: PhysicalRuntimeContext,
    *,
    preview: bool = False,
    capture_details: Optional[bool] = None,
) -> _PhysicalRuntime:
    return (
        context.preview_runtime(
            config, owner, capture_details=capture_details
        )
        if preview
        else context.runtime(config, owner)
    )


def _owner_physical_config(metadata: Mapping[str, Any], owner: str, primary: Any,
                           runtime: Optional[PhysicalRuntimeContext] = None) -> Any:
    """Resolve a declared physical geometry without borrowing another owner."""
    from .physical_contract import parse_execution_physical_config

    configurations = metadata.get("physical_memory_configs")
    if configurations is None:
        return primary
    if not isinstance(configurations, Mapping):
        raise ValueError("physical_memory_configs must map owners to configurations")
    raw = configurations.get(owner)
    if raw is None:
        raise ValueError("physical owner {} has no declared configuration".format(owner))
    return runtime.parse_config(raw) if runtime is not None else parse_execution_physical_config(raw)


def _memory_accesses(metadata):
    if "memory_access" in metadata:
        raise ValueError("memory_access is no longer supported; use a memory_accesses sequence")
    accesses = metadata.get("memory_accesses")
    if accesses is not None and not isinstance(accesses, (tuple, list)):
        raise ValueError("memory_accesses must be a sequence of access mappings")
    for access in accesses or ():
        if not isinstance(access, Mapping):
            raise ValueError("physical memory_accesses entries must be mappings")
        obsolete = {"generation", "size_bytes", "offset", "tensor_id", "buffer_size_bytes", "physical_resource_id"}.intersection(access)
        if obsolete:
            raise ValueError("memory_accesses uses byte_count, offset_bytes, buffer_id, allocation_generation and allocation_size_bytes; unsupported fields: " + ", ".join(sorted(obsolete)))
    return accesses


def _physical_allocations(metadata):
    declarations = metadata.get("physical_allocations", ())
    if not isinstance(declarations, (tuple, list)):
        raise ValueError("physical_allocations must be a sequence")
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise ValueError("physical allocation entries must be mappings")
        obsolete = {"allocation_generation", "buffer_size_bytes", "allocation_size_bytes", "base_address"}.intersection(declaration)
        if obsolete:
            raise ValueError("physical_allocations uses size_bytes, generation and address; unsupported fields: " + ", ".join(sorted(obsolete)))
    return declarations


def _physical_allocation_metadata(task_id: str, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Finish one task's allocation contract before its metadata is published."""
    declarations = _physical_allocations(metadata)
    raw_accesses = _memory_accesses(metadata)
    if declarations:
        return metadata  # Authored declarations already define the complete allocation boundary.
    derived_declarations = ()
    contract = metadata.get("stateful_l2")
    default_owner = str(metadata.get("physical_owner") or
                        (contract.get("memory_resource") if isinstance(contract, Mapping) else "") or "")
    if isinstance(contract, Mapping):
        # The physical preview may already be cache-trimmed. Allocate the
        # logical working set before the cache uses it, using the same
        # buffer identities and preserving explicit address/alias facts.
        previews, explicit_bases = {}, {}
        for row in raw_accesses or ():
            if not isinstance(row, Mapping) or not row.get("buffer_id"):
                continue
            key = (str(row.get("physical_owner") or default_owner), str(row["buffer_id"]),
                   row.get("allocation_generation", 0))
            previous = previews.get(key, {})
            merged = {**previous, **row}
            for fact in ("allocation_arena", "alias_of", "alias_generation", "alias_offset_bytes"):
                if fact in previous and fact in row and previous[fact] != row[fact]:
                    raise ValueError("conflicting physical allocation " + fact)
            source = str(row.get("address_source", ""))
            if row.get("address") is not None and (not source or source.startswith("explicit")):
                offset = _descriptor_non_negative_int(row.get("offset_bytes", 0), "offset_bytes")
                address = _descriptor_non_negative_int(row["address"], "address")
                base = address - offset
                if base < 0:
                    raise ValueError("explicit physical address is before buffer offset")
                if key in explicit_bases and explicit_bases[key] != base:
                    raise ValueError("conflicting physical allocation address")
                explicit_bases[key] = base
            previews[key] = merged
        cache_accesses = []
        for row in contract.get("accesses", ()):
            buffer_id = row["buffer_id"]
            if buffer_id.startswith("@tasklocal"):
                buffer_id = task_id + buffer_id[10:]
            key = (str(row.get("physical_owner") or default_owner), buffer_id,
                   row.get("allocation_generation", 0))
            preview = previews.get(key, {})
            cache_access = {**preview,
                            **{key: value for key, value in row.items()
                               if key not in {"size_bytes", "buffer_size_bytes"}},
                            "buffer_id": buffer_id, "byte_count": row["size_bytes"]}
            if cache_access.get("allocation_size_bytes") is None:
                cache_access["allocation_size_bytes"] = row.get("buffer_size_bytes")
            if key in explicit_bases:
                # A cache line has its own offset. Carry the allocation
                # base, never the last preview range's absolute address.
                offset = _descriptor_non_negative_int(row.get("offset_bytes", 0), "offset_bytes")
                cache_access.update(address=explicit_bases[key] + offset,
                                    address_source="explicit_physical_address")
            cache_accesses.append(cache_access)
        raw_accesses = tuple(cache_accesses)
    if isinstance(raw_accesses, (tuple, list)):
        derived = {}
        for access in raw_accesses:
            if not isinstance(access, Mapping):
                continue
            buffer_id = access.get("buffer_id")
            if not buffer_id:
                continue
            # Descriptor fields are part of the physical-address
            # contract.  Do not coerce floats/strings with ``int`` here:
            # ``64.9`` silently becoming ``64`` would register and later
            # access a different extent than the caller declared.
            offset = _descriptor_non_negative_int(
                access.get("offset_bytes", 0),
                "offset_bytes",
            )
            size = _descriptor_non_negative_int(
                access["byte_count"],
                "byte_count",
            )
            declared_extent = access.get("allocation_size_bytes")
            extent = (
                _descriptor_non_negative_int(declared_extent, "allocation_size_bytes")
                if declared_extent is not None
                else offset + size
            )
            if declared_extent is None and isinstance(contract, Mapping):
                line_bytes = _positive_descriptor_int(
                    contract["line_bytes"], "line_bytes"
                )
                extent = ((extent + line_bytes - 1) // line_bytes) * line_bytes
            generation = _descriptor_non_negative_int(
                access.get("allocation_generation", 0),
                "generation",
            )
            owner = str(access.get("physical_owner") or default_owner)
            key = (owner, str(buffer_id), generation)
            previous = derived.get(key)
            declaration = {"buffer_id": str(buffer_id), "size_bytes": extent,
                           "generation": generation, "physical_owner": owner,
                           "inferred": declared_extent is None}
            if access.get("allocation_arena") is not None:
                declaration["allocation_arena"] = access["allocation_arena"]
            if access.get("alias_of") is not None:
                declaration.update({
                    "alias_of": str(access["alias_of"]),
                    "alias_generation": _descriptor_non_negative_int(
                        access.get("alias_generation", generation),
                        "alias_generation",
                    ),
                    "alias_offset_bytes": _descriptor_non_negative_int(
                        access.get("alias_offset_bytes", 0),
                        "alias_offset_bytes",
                    ),
                })
            address_source = str(access.get("address_source", ""))
            if access.get("address") is not None and (not address_source or address_source.startswith("explicit")):
                explicit_address = access.get("address")
                if type(explicit_address) is not int or explicit_address < offset:
                    raise ValueError("explicit physical address is before buffer offset")
                declaration["address"] = explicit_address - offset
            if previous is not None:
                # Every range can supply facts about one allocation. A
                # later equal/smaller range must not lose its arena, alias
                # or explicit base, and a larger one cannot overwrite a
                # conflicting identity supplied by an earlier range.
                for field in ("allocation_arena", "address", "alias_of",
                              "alias_generation", "alias_offset_bytes"):
                    if field in previous:
                        if field in declaration and declaration[field] != previous[field]:
                            raise ValueError("conflicting physical allocation " + field)
                        declaration[field] = previous[field]
                declaration["size_bytes"] = max(previous["size_bytes"], extent)
                declaration["inferred"] = previous["inferred"] and declaration["inferred"]
            derived[key] = declaration
        derived_declarations = tuple(derived.values())

    return {**metadata, "physical_allocations": derived_declarations}


def lower_physical_allocations(task: TaskSpec) -> TaskSpec:
    """Compile access and cache descriptors into explicit allocation declarations."""
    metadata = _physical_allocation_metadata(task.task_id, task.metadata)
    return task if metadata is task.metadata else replace(task, metadata=metadata)


def register_physical_allocations(task: TaskSpec, runtime: PhysicalRuntimeContext, *, config=None) -> None:
    """Materialize declared buffers into the owner-scoped run allocator."""
    metadata = task.metadata
    raw_config = config if config is not None else metadata.get("physical_memory_config")
    raw_config = runtime.parse_config(raw_config)
    declarations = _physical_allocations(metadata)
    pending = sorted(
        declarations,
        key=lambda item: 1 if isinstance(item, Mapping) and item.get("alias_of") else 0,
    )
    # One parsed geometry per owner for this unchanged task metadata.
    owner_configs = {}
    while pending:
        deferred = []
        progress = 0
        for declaration in pending:
            if not isinstance(declaration, Mapping):
                raise ValueError("physical allocation entries must be mappings")
            alias_of = declaration.get("alias_of")
            owner = str(declaration.get("physical_owner") or metadata.get("physical_owner") or "")
            generation = _descriptor_non_negative_int(
                declaration.get("generation", 0),
                "generation",
            )
            alias_generation = declaration.get("alias_generation")
            if alias_of is not None and alias_generation is None:
                alias_generation = generation
            elif alias_generation is not None:
                alias_generation = _descriptor_non_negative_int(
                    alias_generation, "alias_generation"
                )
            if alias_of is not None and owner:
                existing_allocator = runtime.allocators.get(owner)
                if existing_allocator is not None and existing_allocator.get_allocation(str(alias_of), alias_generation) is None:
                    deferred.append(declaration)
                    continue
            if owner not in owner_configs:
                owner_configs[owner] = _owner_physical_config(metadata, owner, raw_config, runtime)
            try:
                _register_physical_allocation(
                    declaration, metadata, runtime, owner_configs[owner],
                )
                progress += 1
            except ValueError:
                raise
        if not deferred:
            break
        if progress == 0:
            raise ValueError("physical allocation alias target is not registered")
        pending = deferred


def _register_physical_allocation(declaration, metadata, runtime, raw_config):
    """Register one declaration after its optional alias target exists."""
    if not isinstance(declaration, Mapping):
        raise ValueError("physical allocation entries must be mappings")
    buffer_id = declaration.get("buffer_id")
    if not buffer_id:
        raise ValueError("physical allocation requires buffer_id")
    owner = str(declaration.get("physical_owner") or metadata.get("physical_owner") or "")
    if not owner:
        raise ValueError("physical allocation requires physical_owner")
    size_bytes = declaration.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ValueError("physical allocation requires positive size_bytes")
    runtime.runtime(raw_config, owner)
    allocator = runtime.allocators.get(owner)
    if allocator is None:
        raise ValueError("physical owner has no address allocator: {}".format(owner))
    generation = _descriptor_non_negative_int(
        declaration.get("generation", 0), "generation")
    external = (owner, str(buffer_id), generation) in {
        tuple(key) for key in metadata.get("physical_external_allocations", ())}
    allocation_arena = declaration.get("allocation_arena")
    if external:
        if allocation_arena not in (None, "resident"):
            raise ValueError("external page-pool allocation must use the resident arena")
        allocation_arena = "resident"
    allocator.allocate(
        str(buffer_id), size_bytes, generation,
        alias_of=declaration.get("alias_of"),
        address=declaration.get("address"),
        alias_generation=declaration.get("alias_generation"),
        alias_offset_bytes=_descriptor_non_negative_int(
            declaration.get("alias_offset_bytes", 0), "alias_offset_bytes"
        ),
        inferred=_descriptor_bool(declaration.get("inferred", False), "inferred"),
        allocation_arena=allocation_arena,
    )


def resolve_physical_task(
    task: TaskSpec,
    runtime: PhysicalRuntimeContext,
    arrival_ns: float,
    *,
    _caller_managed_transaction: bool = False,
) -> TaskSpec:
    """Commit a physical task atomically, including allocation and billing.

    The event kernel already owns a wider transaction. Standalone callers
    need the rollback point before allocation registration, and protection
    through construction of the final resource demands.
    """
    if _caller_managed_transaction:
        return _resolve_physical_task(task, runtime, arrival_ns)
    snapshot = runtime.begin_transaction()
    try:
        result = _resolve_physical_task(task, runtime, arrival_ns)
    except Exception:
        runtime.restore(snapshot)
        raise
    runtime.commit_transaction(snapshot)
    return result


def _merge_latest_resource_intervals(target, incoming):
    # Scheduled results retain input order, which need not be execution order.
    # Select by interval time so a prior row hit cannot replace the final use.
    for resource_id, interval in incoming.items():
        previous = target.get(resource_id)
        if previous is None or (interval[1], interval[0]) > (previous[1], previous[0]):
            target[resource_id] = interval


def _resolve_physical_task(
    task: TaskSpec,
    runtime: PhysicalRuntimeContext,
    arrival_ns: float,
) -> TaskSpec:
    if not task.metadata.get("hbf_host_staging_by_owner"):
        return _execute_physical_task(task, runtime, arrival_ns)
    from . import _io_execution

    token = _io_execution.begin_task(runtime)
    success = False
    try:
        result = _execute_physical_task(task, runtime, arrival_ns)
        success = True
        return result
    finally:
        _io_execution.end_task(runtime, token, success=success)


def prepare_physical_task(
    task: TaskSpec,
    runtime: PhysicalRuntimeContext,
    arrival_ns: float,
):
    """Resolve and validate one task's ordered physical instructions.

    Allocation registration and HBF preload are part of the caller's existing
    task transaction. The returned instructions are valid for this dispatch,
    and must not be cached across allocation generations or external edits.
    """
    metadata = dict(task.metadata)
    raw_config = metadata.get("physical_memory_config")
    accesses = _memory_accesses(metadata)
    if raw_config is None or not isinstance(accesses, (tuple, list)) or not accesses:
        raise ValueError("physical task requires physical_memory_config and memory_accesses")
    from .memory_types import Operation
    config = runtime.parse_config(raw_config)
    if isinstance(arrival_ns, bool) or not isinstance(arrival_ns, (int, float)) or arrival_ns < 0:
        raise ValueError("physical task arrival_ns must be non-negative")
    owners = set()
    for item in accesses:
        if not isinstance(item, Mapping):
            raise ValueError("physical memory_accesses entries must be mappings")
        owner = str(item.get("physical_owner") or metadata.get("physical_owner") or "")
        if not owner:
            raise ValueError("physical access requires physical_owner")
        owners.add(owner)
    if metadata.get("physical_memory_configs") is None and len(owners) > 1:
        raise ValueError("multi-owner physical task requires per-owner configurations")
    register_physical_allocations(task, runtime, config=config)
    staging_bindings = metadata.get("hbf_host_staging_by_owner", {})
    if staging_bindings:
        from . import hbf_staging_runtime
        for owner, binding in staging_bindings.items():
            hbf_staging_runtime.configure(runtime, owner,
                _owner_physical_config(metadata, owner, config, runtime), binding)
        for owner, buffer_id, generation in metadata.get("hbf_preloaded_buffers", ()):
            if owner in staging_bindings:
                hbf_staging_runtime.preload(runtime, owner, buffer_id, generation,
                                           staging_bindings[owner])

    # Validate every descriptor before creating a core or reserving a resource.
    # A physical address may be omitted only when the buffer has an explicit
    # allocation; this keeps L2 victims and partial line fills tied to it.
    validated = []
    for access in accesses:
        if not isinstance(access, Mapping):
            raise ValueError("physical memory_accesses entries must be mappings")
        operation = Operation(str(access.get("operation", "")).lower())
        address, byte_count = access.get("address"), access.get("byte_count")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError("physical task byte_count must be a positive integer")
        owner = str(access.get("physical_owner") or metadata.get("physical_owner") or "")
        if not owner:
            raise ValueError("physical access requires physical_owner")
        buffer_id = access.get("buffer_id")
        address_source = str(access.get("address_source", ""))
        if buffer_id:
            allocator = runtime.allocators.get(owner)
            if allocator is None:
                raise ValueError("physical owner has no address allocator: {}".format(owner))
            generation = _descriptor_non_negative_int(
                access.get("allocation_generation", 0),
                "generation",
            )
            offset = _descriptor_non_negative_int(
                access.get("offset_bytes", 0),
                "offset_bytes",
            )
            registered_address = allocator.address(str(buffer_id), offset, byte_count, generation)
            if address is not None and (not address_source or address_source.startswith("explicit")):
                if type(address) is not int or address != registered_address:
                    raise ValueError("physical access address disagrees with its buffer allocation")
            address = registered_address
        elif address is None:
            raise ValueError("physical access without address requires buffer_id")
        if isinstance(address, bool) or not isinstance(address, int) or address < 0:
            raise ValueError("physical task address must be a non-negative integer")
        access_config = _owner_physical_config(metadata, owner, config, runtime)
        if address + byte_count > access_config.capacity_bytes:
            raise ValueError(
                "physical access exceeds owner {} capacity".format(owner)
            )
        validated.append((access, operation, address, byte_count, owner, access_config))
    from ._physical_task import PreparedPhysicalTask
    return PreparedPhysicalTask(metadata, config, owners, validated, staging_bindings)


def _execute_physical_task(
    task: TaskSpec,
    runtime: PhysicalRuntimeContext,
    arrival_ns: float,
) -> TaskSpec:
    """Prepare and execute inside the existing per-task transaction."""
    return execute_prepared_physical_task(
        task, runtime, arrival_ns, prepare_physical_task(task, runtime, arrival_ns))


def execute_prepared_physical_task(task, runtime, arrival_ns, prepared) -> TaskSpec:
    """Consume validated instructions without rebuilding their descriptors.

    The caller must retain the transaction and HBF task scope around both
    preparation and execution, as ``resolve_physical_task`` does.
    """
    from .memory_types import AccessRequest, DramConfig, Operation
    from ._physical_task import PhysicalTaskLedger

    metadata, owners = prepared.metadata, prepared.owners
    validated, staging_bindings = prepared.instructions, prepared.staging_bindings
    # Previous accesses only constrain a new request when at least one side
    # writes.  The old implementation scanned every prior descriptor for
    # every descriptor (O(n**2)); a large line-expanded projection can contain
    # tens of thousands of descriptors.  Compress address endpoints once and
    # keep range-max indexes for all accesses and writes.  Each query and
    # update is O(log n), while submission order and per-access timing stay
    # unchanged.
    # A read-only task has no RAW/WAR/WAW dependencies at any address. Do not
    # construct and repeatedly update indexes that can never constrain it.
    track_overlaps = len(validated) > 1 and any(
        item[1] is Operation.WRITE for item in validated)
    endpoints_by_owner = {}
    if track_overlaps:
        for _access, _op, address, byte_count, owner, _config in validated:
            endpoints_by_owner.setdefault(owner, set()).update((address, address + byte_count))
    all_completion = {
        owner: _OverlapCompletionIndex(sorted(endpoints))
        for owner, endpoints in endpoints_by_owner.items()
    }
    write_completion = {
        owner: _OverlapCompletionIndex(sorted(endpoints))
        for owner, endpoints in endpoints_by_owner.items()
    }

    placeholder_ids = set(metadata.get("hbf_staging_replaced_link_ids", ()))
    for access, _op, _address, _bytes, _owner, _config in validated:
        for key in ("resource_id", "physical_owner"):
            value = access.get(key)
            if value:
                placeholder_ids.add(str(value))
    ledger = PhysicalTaskLedger()
    physical_results, staging_stats = ledger.records, ledger.staging_stats
    staging_sessions = {}
    resolved_accesses = []
    resource_accumulator = None
    if (len(owners) == 1 and not staging_bindings and not runtime.capture_details
            and all(isinstance(item[5], DramConfig)
                    and item[5].control_mode == "jedec_hbm"
                    and item[5].scheduler == "fcfs" for item in validated)):
        from ._hbm_pack import HBMResourceAccumulator
        resource_accumulator = HBMResourceAccumulator()
    index = 0
    while index < len(validated):
        access, operation, address, byte_count, owner, access_config = validated[index]
        if owner in staging_bindings:
            from ._io_execution import PreparedStagingIO
            session = staging_sessions.get(owner)
            if session is None:
                session = staging_sessions[owner] = PreparedStagingIO(
                    runtime, owner, staging_bindings[owner], ledger)
            access_arrival = float(arrival_ns)
            if track_overlaps:
                prior = all_completion[owner] if operation is Operation.WRITE else write_completion[owner]
                access_arrival = max(access_arrival, prior.query(address, byte_count))
            completion = session.submit(access, operation, byte_count, access_arrival)
            if track_overlaps:
                all_completion[owner].update(address, byte_count, completion)
                if operation is Operation.WRITE:
                    write_completion[owner].update(address, byte_count, completion)
            resolved_accesses.append({**dict(access), "address": address})
            index += 1
            continue
        active = runtime.runtime(access_config, owner)
        # An authored task may mix staged HBF accesses and a direct access
        # to the same HOST controller. Publish the preceding staged work
        # before the direct core reads the shared resource calendars.
        if getattr(runtime, "_hbf_host_task", None) is not None:
            from ._io_execution import flush_task
            flush_task(runtime)
        stop = index + 1
        scheduled_batch = (isinstance(active.core.config, DramConfig)
                           and active.core.config.scheduler != "fcfs")
        if scheduled_batch:
            # A controller can choose only among accesses in this task's
            # current owner batch. The core enforces arrival visibility and
            # overlapping read/write dependencies inside its finite window.
            while stop < min(len(validated), index + 1024):
                _a, next_op, _addr, _count, next_owner, next_config = validated[stop]
                if (next_owner != owner or next_config != access_config
                        or next_op not in (Operation.READ, Operation.WRITE)):
                    break
                stop += 1
        elif (_USE_COMPILED_DRAM_BATCH and not runtime.capture_details
                and isinstance(active.core.config, DramConfig)
                and operation in (Operation.READ, Operation.WRITE)):
            # Reads do not depend on each other. Writes are batched only while
            # their byte ranges are ascending and disjoint. Thus each arrival
            # depends solely on already executed groups; no intra-batch RAW,
            # WAR or WAW completion is guessed or discarded.
            previous_end = address + byte_count
            while stop < min(len(validated), index + 1024):
                _next_access, next_op, next_address, next_bytes, next_owner, _next_config = validated[stop]
                if (next_owner != owner or next_op is not operation
                        or _next_config != access_config):
                    break
                if operation is Operation.WRITE and next_address < previous_end:
                    break
                previous_end = next_address + next_bytes
                stop += 1
        requests = []
        for position in range(index, stop):
            _access, op, addr, count, _owner, _config = validated[position]
            access_arrival = float(arrival_ns)
            if track_overlaps:
                prior = all_completion[owner] if op is Operation.WRITE else write_completion[owner]
                access_arrival = max(access_arrival, prior.query(addr, count))
            requests.append(AccessRequest(f"{task.request_id or task.task_id}:{position}", op, addr, count, access_arrival))
        if resource_accumulator is not None:
            if scheduled_batch or len(requests) >= 2:
                group_results = active.core.submit_many(
                    requests, _resource_accumulator=resource_accumulator)
            else:
                group_results = tuple(active.core.submit(
                    request, _resource_accumulator=resource_accumulator) for request in requests)
        elif scheduled_batch or len(requests) >= 2:
            group_results = active.core.submit_many(requests)
        else:
            group_results = tuple(active.core.submit(request) for request in requests)
        for position, result in zip(range(index, stop), group_results):
            access, op, addr, count, _owner, _config = validated[position]
            active.clock_ns = max(active.clock_ns, result.completion_ns)
            ledger.accept(result.operation, result.completion_ns)
            physical_results.append((op, owner, access_config, result))
            if track_overlaps:
                all_completion[owner].update(addr, count, result.completion_ns)
                if op is Operation.WRITE:
                    write_completion[owner].update(addr, count, result.completion_ns)
            resolved_accesses.append({**dict(access), "address": addr})
        index = stop
    completion_ns = ledger.completion_ns
    # Single-owner tasks need only one ordered resource reduction. The owner
    # bill already performs the same additions below; copying its final maps
    # preserves independent public ownership without repeating every lookup.
    sole_physical_owner = physical_results[0][1]
    single_physical_owner = all(item[1] == sole_physical_owner for item in physical_results)
    resource_busy, resource_bytes, resource_last = {}, {}, {}
    if not single_physical_owner:
        first_counters = physical_results[0][3].counters
        resource_busy = {rid: 0.0 + value for rid, value in first_counters.get("resource_busy_ns", {}).items()}
        resource_bytes = {rid: 0 + value for rid, value in first_counters.get("resource_bytes", {}).items()}
        incoming_last = first_counters.get("resource_last_intervals", {})
        if type(incoming_last) is dict:
            resource_last = dict(incoming_last)
        else:
            _merge_latest_resource_intervals(resource_last, incoming_last)
        for _operation, owner, _config, item in physical_results[1:]:
            for resource_id, value in item.counters.get("resource_busy_ns", {}).items():
                resource_busy[resource_id] = resource_busy.get(resource_id, 0.0) + value
            for resource_id, value in item.counters.get("resource_bytes", {}).items():
                resource_bytes[resource_id] = resource_bytes.get(resource_id, 0) + value
            _merge_latest_resource_intervals(resource_last, item.counters.get("resource_last_intervals", {}))
    from ._resource_ledger import reduce_physical_results
    execution_by_owner = reduce_physical_results(
        physical_results, arrival_ns, DRAM_OPTIONAL_COUNTER_FIELDS,
        _PHYSICAL_ADDITIVE_COUNTERS, _PHYSICAL_PEAK_COUNTERS, Operation.READ)
    if resource_accumulator is not None:
        owner_bill = execution_by_owner[sole_physical_owner]
        owner_bill["resource_busy_ns"] = resource_accumulator.resource_busy_ns
        owner_bill["resource_bytes"] = resource_accumulator.resource_bytes
        owner_bill["resource_last_intervals"] = resource_accumulator.resource_last_intervals
        owner_bill["resource_owners"] = dict.fromkeys(resource_accumulator.resource_busy_ns,
                                                   sole_physical_owner)
    if single_physical_owner:
        owner_bill = execution_by_owner[sole_physical_owner]
        resource_busy = dict(owner_bill["resource_busy_ns"])
        resource_bytes = dict(owner_bill["resource_bytes"])
        resource_last = dict(owner_bill["resource_last_intervals"])
    from ._resource_ledger import physical_demands
    demands = physical_demands(
        resource_busy, resource_bytes, runtime._physical_demand_cache, ResourceDemand)
    if not demands and not ledger.all_erases:
        raise ValueError("physical memory core returned no resource timing for a positive-byte access")
    metadata.update({
        "memory_accesses": tuple(resolved_accesses),
        "physical_execution_by_owner": execution_by_owner,
        **({"hbf_host_staging": staging_stats} if staging_stats else {}),
        "physical_arrival_ns": float(arrival_ns),
        "physical_completion_ns": completion_ns,
        "physical_resource_intervals": tuple(
            item
            for _op, owner, _config, result in physical_results
            for item in result.stages
            if item.name not in {"READ_PIPELINE", "WRITE_PIPELINE"}
        ),
        "physical_resource_last_intervals": resource_last,
        "physical_demands_resource_ids": tuple(sorted(resource_busy)),
        "physical_placeholder_resource_ids": tuple(sorted(placeholder_ids)),
    })
    # Keep unrelated compute/link demands. Only preview demands for resources
    # identified by this transaction are replaced by core-owned reservations.
    memory_ids = set(metadata["physical_demands_resource_ids"]) | placeholder_ids
    canonical_memory_ids = {runtime.timeline.canonical_resource_id(rid) for rid in memory_ids}
    preserved = tuple(d for d in task.demands
                      if runtime.timeline.canonical_resource_id(d.resource_id) not in canonical_memory_ids)
    metadata["physical_nonmemory_demand_ids"] = tuple(d.resource_id for d in preserved)
    resolved = replace(task, demands=preserved + demands, metadata=metadata)
    return _recost_physical_kernel_envelope(resolved)


def _recost_physical_kernel_envelope(task: TaskSpec) -> TaskSpec:
    """Replace a native kernel's backing estimate with its actual Core span.

    The stream is an execution envelope, not another hardware service. Compute
    resource demands (including a declared calibration correction) stay intact;
    the removed memory estimate must not survive as a stream-duration floor.
    """
    metadata = task.metadata
    audit = metadata.get("phase_metadata", {}).get("kernel_model")
    contract = audit.get("physical_memory_envelope") if isinstance(audit, Mapping) else None
    if not isinstance(contract, Mapping):
        return task
    suffix = str(contract["resource_suffix"])
    stream_ids = {d.resource_id for d in task.demands if d.resource_id.endswith(suffix)}
    if not stream_ids:
        return task
    dependency_ns = contract["dependency_ns"]
    _non_negative(dependency_ns, "physical kernel dependency_ns")
    l2 = metadata.get("stateful_l2", {})
    excluded = (stream_ids | set(metadata["physical_demands_resource_ids"])
                | set(l2.get("envelope_resources", ())))
    other_span = max((d.service_ns for d in task.demands if d.resource_id not in excluded), default=0.)
    # A cache miss is submitted after its aggregate lookup. Restore that
    # prefix when expressing the completion relative to the kernel start.
    lookup_prefix = metadata.get("l2_execution", {}).get("backing_arrival_offset_ns", 0.)
    memory_span = metadata["physical_completion_ns"] - metadata["physical_arrival_ns"] + lookup_prefix
    duration = max(other_span, memory_span) + dependency_ns
    demands = tuple(replace(d, service_ns=duration) if d.resource_id in stream_ids else d
                    for d in task.demands)
    def updated(value):
        return {**value, "analytical_service_ns": duration,
                "resource_service_ns": {d.resource_id: d.service_ns for d in demands},
                "pipeline_timing_scope": "descriptor_before_physical_dispatch",
                "prediction": {**value.get("prediction", {}),
                    "prediction_ns": duration,
                    "physical_memory_recost": {
                        "memory_completion_span_ns": memory_span,
                        "other_resource_span_ns": other_span,
                        "dependency_ns": dependency_ns,
                        "scope": "Core memory plus existing compute resource model; not measured whole-kernel replay",
                    }}}
    changes = {"analytical_service_ns": duration}
    changes["phase_metadata"] = {**metadata["phase_metadata"], "kernel_model": updated(audit)}
    if isinstance(metadata.get("kernel_prediction"), Mapping):
        changes["kernel_prediction"] = updated(metadata["kernel_prediction"])
    return replace(task, demands=demands, metadata={**metadata, **changes})


def is_physical_task(task: TaskSpec) -> bool:
    metadata = task.metadata
    config = metadata.get("physical_memory_config")
    accesses = _memory_accesses(metadata)
    # Endpoint metadata always carries these optional keys.  It becomes a
    # formal physical transaction only when both values are present and the
    # descriptor is non-empty; an explicitly supplied malformed descriptor is
    # rejected instead of silently falling back to analytical pricing.
    if config is None:
        if accesses not in (None, (), []):
            raise ValueError("physical task requires physical_memory_config and memory_accesses")
        return False
    if not isinstance(accesses, (tuple, list)):
        raise ValueError("physical task requires physical_memory_config and memory_accesses")
    if not accesses:
        raise ValueError("physical task requires a non-empty memory access descriptor")
    return True


def physical_task_transaction_owners(task: TaskSpec):
    """Owners that the task's explicit memory descriptors may mutate.

    Host staging can execute additional controller work through its bindings;
    it keeps the complete transaction. Ordinary DRAM/NAND/SRAM descriptors
    name their entire memory footprint, including L2 backing allocations.
    """
    metadata = task.metadata
    if metadata.get("hbf_host_staging_by_owner"):
        return None
    default = metadata.get("physical_owner")
    owners = {str(default)} if default else set()
    for rows in (_memory_accesses(metadata), _physical_allocations(metadata)):
        for row in rows or ():
            if not isinstance(row, Mapping):
                raise ValueError("physical descriptors must be mappings")
            owner = row.get("physical_owner") or default
            if owner:
                owners.add(str(owner))
    contract = metadata.get("stateful_l2")
    if isinstance(contract, Mapping) and contract.get("memory_resource"):
        owners.add(str(contract["memory_resource"]))
    return frozenset(owners)


def physical_task_demands(task: TaskSpec) -> Tuple[ResourceDemand, ...]:
    """Return preview demands that remain externally scheduled for a physical task."""
    metadata = task.metadata
    physical_ids = set(str(item) for item in metadata.get("physical_demands_resource_ids", ()))
    physical_ids.update(str(item) for item in metadata.get("physical_placeholder_resource_ids", ()))
    physical_ids.update(str(item) for item in metadata.get("hbf_staging_replaced_link_ids", ()))
    accesses = _memory_accesses(metadata)
    for access in accesses or ():
        if isinstance(access, Mapping):
            value = access.get("resource_id")
            if value:
                physical_ids.add(str(value))
    return tuple(demand for demand in task.demands if str(demand.resource_id) not in physical_ids)


class AccessKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    COPY = "COPY"
    ERASE = "ERASE"


def _positive(value: Any, name: str) -> float:
    number = _non_negative(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite non-negative number") from None
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _descriptor_non_negative_int(value: Any, name: str) -> int:
    """Validate explicit descriptor values; producers supply omitted defaults."""
    return _non_negative_int(value, name)


def _positive_descriptor_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be a positive integer")
    return _positive_int(value, name)


def _descriptor_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str) -> int:
    value = _non_negative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _value(source: Any, name: str, *, default: Any = None) -> Any:
    """Read a field from a mapping or a profile object."""

    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def memory_service(
    *,
    read_bytes: int,
    write_bytes: int,
    bandwidth_gb_s: float,
    read_bandwidth_gb_s: Optional[float] = None,
    write_bandwidth_gb_s: Optional[float] = None,
    read_latency_ns: float = 0.0,
    write_latency_ns: float = 0.0,
    transaction_bytes: int = 256,
    max_outstanding_requests: int = 32,
    parallel_lanes: int = 1,
    service_model: str = "analytical",
) -> Mapping[str, object]:
    """Analytical bandwidth/latency envelope for one resolved payload stream.

    Reads and writes share an aggregate request window of
    ``max_outstanding_requests * parallel_lanes``.  Request-slot time divided
    by that window is an optimistic concurrency bound, not a schedule.
    A nonempty direction also cannot complete before its single-request latency.
    Bandwidth and these bounds overlap (max, not sum).  Zero latency reproduces
    the historical payload/BW model exactly, including partial transactions.

    Inputs must already reflect cache misses or other physical-flow decisions.
    No addresses, access ordering or achieved MLP are known: transaction counts
    assume coalesced per-direction streams, and payload bytes are NOT rounded up
    to transaction size.  Unaligned/strided overfetch must be supplied upstream.
    """

    _non_negative_int(read_bytes, "read_bytes")
    _non_negative_int(write_bytes, "write_bytes")
    _positive(bandwidth_gb_s, "bandwidth_gb_s")
    read_bw = bandwidth_gb_s if read_bandwidth_gb_s is None else read_bandwidth_gb_s
    write_bw = bandwidth_gb_s if write_bandwidth_gb_s is None else write_bandwidth_gb_s
    _positive(read_bw, "read_bandwidth_gb_s")
    _positive(write_bw, "write_bandwidth_gb_s")
    _non_negative(read_latency_ns, "read_latency_ns")
    _non_negative(write_latency_ns, "write_latency_ns")
    _positive_int(transaction_bytes, "transaction_bytes")
    _positive_int(max_outstanding_requests, "max_outstanding_requests")
    _positive_int(parallel_lanes, "parallel_lanes")
    logical_read_bytes, logical_write_bytes = read_bytes, write_bytes
    if service_model != "analytical":
        read_bytes = ((read_bytes + transaction_bytes - 1) // transaction_bytes) * transaction_bytes
        write_bytes = ((write_bytes + transaction_bytes - 1) // transaction_bytes) * transaction_bytes
    physical_bytes = read_bytes + write_bytes
    read_transactions = (read_bytes + transaction_bytes - 1) // transaction_bytes
    write_transactions = (write_bytes + transaction_bytes - 1) // transaction_bytes
    transaction_count = read_transactions + write_transactions
    request_window = max_outstanding_requests * parallel_lanes
    effective_outstanding = min(request_window, transaction_count)
    read_bandwidth_service_ns = read_bytes / read_bw if read_bytes else 0.0
    write_bandwidth_service_ns = write_bytes / write_bw if write_bytes else 0.0
    # One shared controller: directional durations add before rounding
    # when their rates coincide (rather than adding two rounded divisions).
    bandwidth_service_ns = (
        physical_bytes / read_bw if read_bw == write_bw else
        read_bandwidth_service_ns + write_bandwidth_service_ns
    )
    single_request_latency_ns = max(
        read_latency_ns if read_transactions else 0.0,
        write_latency_ns if write_transactions else 0.0,
    )
    request_slot_time_ns = (
        read_transactions * read_latency_ns
        + write_transactions * write_latency_ns
    )
    concurrency_service_ns = (
        request_slot_time_ns / effective_outstanding
        if effective_outstanding else 0.0
    )
    latency_service_ns = max(single_request_latency_ns, concurrency_service_ns)
    if service_model not in {"analytical", "serialized", "overlapped"}:
        raise ValueError("memory service_model must be analytical, serialized or overlapped")
    if service_model != "analytical":
        latency_service_ns = (
            math.ceil(read_transactions / request_window) * read_latency_ns
            + math.ceil(write_transactions / request_window) * write_latency_ns
        )
    service_ns = (bandwidth_service_ns + latency_service_ns
                  if service_model == "serialized" else max(bandwidth_service_ns, latency_service_ns))
    service_source = (
        "zero_traffic" if not physical_bytes else
        "latency_concurrency" if latency_service_ns > bandwidth_service_ns else
        "bandwidth_and_latency_concurrency" if latency_service_ns == bandwidth_service_ns else
        "bandwidth"
    )
    if service_source == "latency_concurrency":
        bottleneck = "latency_concurrency"
    elif service_source == "bandwidth_and_latency_concurrency":
        bottleneck = "bandwidth_and_latency_concurrency"
    elif service_source == "bandwidth":
        bottleneck = "bandwidth"
    else:
        bottleneck = "none"
    metrics = realtime_memory_metrics(
        logical_read_bytes + logical_write_bytes,
        service_ns,
        physical_bytes=physical_bytes,
        bandwidth_ceiling_gb_s=(
            physical_bytes / bandwidth_service_ns
            if bandwidth_service_ns > 0 else 0.0
        ),
        queue_wait_ns=0.0,
        request_window_utilization=(
            min(1.0, transaction_count / float(request_window))
            if transaction_count else 0.0
        ),
        bottleneck=bottleneck,
    )
    return {
        "model": "memory_bandwidth_latency_concurrency_bound_v1",
        "evidence": EvidenceStatus.ANALYTICAL.value,
        "cycle_accurate": False,
        "latency_model_enabled": read_latency_ns > 0.0 or write_latency_ns > 0.0,
        "byte_scope": "resolved_backing_payload",
        "logical_read_bytes": logical_read_bytes,
        "logical_write_bytes": logical_write_bytes,
        "logical_bytes": logical_read_bytes + logical_write_bytes,
        "physical_read_bytes": read_bytes,
        "physical_write_bytes": write_bytes,
        "physical_bytes": physical_bytes,
        "physical_traffic_basis": ("payload_only_no_transaction_padding" if service_model == "analytical"
                                   else "rounded_directional_transactions"),
        "transaction_bytes": transaction_bytes,
        "read_transactions": read_transactions,
        "write_transactions": write_transactions,
        "transaction_count": transaction_count,
        "max_outstanding_requests": max_outstanding_requests,
        "parallel_lanes": parallel_lanes,
        "request_window": request_window,
        "effective_outstanding": effective_outstanding,
        "outstanding_basis": "configured_limit_times_parallel_lanes_capped_by_transaction_count",
        "concurrency_assumption": "independent_requests_shared_read_write_window_across_aggregate_lanes",
        "read_latency_ns": read_latency_ns,
        "write_latency_ns": write_latency_ns,
        "bandwidth_gb_s": bandwidth_gb_s,
        "read_bandwidth_gb_s": read_bw,
        "write_bandwidth_gb_s": write_bw,
        "read_bandwidth_service_ns": read_bandwidth_service_ns,
        "write_bandwidth_service_ns": write_bandwidth_service_ns,
        "bandwidth_model": "shared_controller_directional_serial_payload",
        "bandwidth_service_ns": bandwidth_service_ns,
        "single_request_latency_ns": single_request_latency_ns,
        "request_slot_time_ns": request_slot_time_ns,
        "concurrency_service_ns": concurrency_service_ns,
        "latency_service_ns": latency_service_ns,
        "latency_bound": latency_service_ns > bandwidth_service_ns,
        "service_source": service_source,
        "service_ns": service_ns,
        **metrics,
        "estimated_internal_wait_ns": max(0.0, concurrency_service_ns - single_request_latency_ns),
        "unmodeled_terms": (
            "request_dependencies_and_achieved_mlp",
            "transaction_alignment_fragmentation_and_padding",
            "bank_row_buffer_refresh_and_read_write_turnaround",
        ),
    }


@dataclass(frozen=True)
class PhysicalService:
    """One authoritative data service for a physical owner."""

    service_id: str
    physical_owner: str
    resource_id: str
    read_bandwidth_gb_s: float
    write_bandwidth_gb_s: float
    read_latency_ns: float = 0.0
    write_latency_ns: float = 0.0
    transaction_granularity: int = 256
    queue_depth: int = 1
    parallel_lanes: int = 1
    latency_scope: str = "service"
    efficiency: float = 1.0
    service_model: str = "analytical"
    storage_id: str = ""
    component: Optional[ComponentSpec] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("service_id", "physical_owner", "resource_id", "latency_scope"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        _non_negative(self.read_bandwidth_gb_s, "read_bandwidth_gb_s")
        _non_negative(self.write_bandwidth_gb_s, "write_bandwidth_gb_s")
        _non_negative(self.read_latency_ns, "read_latency_ns")
        _non_negative(self.write_latency_ns, "write_latency_ns")
        _positive_int(self.transaction_granularity, "transaction_granularity")
        _positive_int(self.queue_depth, "queue_depth")
        _positive_int(self.parallel_lanes, "parallel_lanes")
        if not 0 < float(self.efficiency) <= 1:
            raise ValueError("efficiency must be in (0, 1]")

    @property
    def bandwidth_gb_s(self) -> float:
        return min(self.read_bandwidth_gb_s, self.write_bandwidth_gb_s)

    def price(
        self, kind: AccessKind, byte_count: int, *,
        page_offset_bytes: Optional[int] = None,
        runtime: Optional[PhysicalRuntimeContext] = None,
        arrival_ns: Optional[float] = None, preview: bool = False,
        summary_only: bool = False,
    ) -> Mapping[str, object]:
        """Price one access; a failed explicit submission leaves no state."""
        saved = runtime.snapshot() if runtime is not None and not preview else None
        try:
            return self._price(kind, byte_count, page_offset_bytes=page_offset_bytes,
                               runtime=runtime, arrival_ns=arrival_ns, preview=preview,
                               summary_only=summary_only)
        except Exception:
            if saved is not None:
                runtime.restore(saved)
            raise

    def _price(
        self,
        kind: AccessKind,
        byte_count: int,
        *,
        page_offset_bytes: Optional[int] = None,
        runtime: Optional[PhysicalRuntimeContext] = None,
        arrival_ns: Optional[float] = None,
        preview: bool = False,
        summary_only: bool = False,
    ) -> Mapping[str, object]:
        kind = AccessKind(kind)
        from .physical_contract import require_physical_memory_config
        if self.component is None:
            raise ValueError("physical memory service requires an explicitly configured component; aggregate-cost fallback is disabled")
        config = require_physical_memory_config(self.component)
        _non_negative_int(byte_count, "byte_count")
        if kind is AccessKind.COPY:
            raise ValueError("COPY must be expanded before billing")
        if page_offset_bytes is not None:
            _non_negative_int(page_offset_bytes, "page_offset_bytes")
        if not isinstance(summary_only, bool):
            raise TypeError("summary_only must be a boolean")
        if page_offset_bytes is None and self.component is not None:
            page_offset_bytes = self.component.metadata.get("memory_access_offset_bytes")
            if page_offset_bytes is not None:
                _non_negative_int(page_offset_bytes, "memory_access_offset_bytes")
        from .memory_types import AccessRequest, DramConfig, SramConfig, Operation
        if kind is AccessKind.ERASE and isinstance(config, (DramConfig, SramConfig)):
            raise ValueError(("SRAM" if isinstance(config, SramConfig) else "DRAM") + " does not support ERASE operations")
        if page_offset_bytes is None and self.component.metadata.get("memory_access_offset_bytes") is None:
            raise ValueError("physical_memory_config requires an explicit memory address")
        if self.component.metadata.get("host_write_staging") is not None:
            if runtime is not None and not preview:
                raise ValueError("host-staged HBF requires logical buffer descriptors and resolved host routes; use resolve_physical_task")
            if getattr(config, "protocol_mode", None) != "ocp_hbf_0_7":
                raise ValueError("host_write_staging requires strict OCP HBF")
            if kind not in {AccessKind.READ, AccessKind.WRITE}:
                raise ValueError("host-staged HBF accepts logical reads and writes only")
            # A planner has no live page-merge state or resolved HOST DRAM
            # allocations. Emit a marked placeholder only; the event kernel
            # replaces this demand with real host, link and NAND operations.
            # Do not run a generic NAND estimate or submit an illegal small
            # write to the strict physical core during graph construction.
            return {
                "logical_bytes": byte_count, "physical_bytes": byte_count,
                "physical_read_bytes": byte_count if kind is AccessKind.READ else 0,
                "physical_write_bytes": byte_count if kind is AccessKind.WRITE else 0,
                "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                "service_ns": 0.0, "actual_bandwidth_gb_s": 0.0,
                "bandwidth_ceiling_gb_s": 0.0, "pages_touched": 0,
                "operation": kind.value.lower(),
                "timing_model": "host_staging_deferred_to_physical_runtime",
                "physical_cost_deferred": True,
            }
        if byte_count == 0:
            return {
                "logical_bytes": 0, "physical_bytes": 0,
                "physical_read_bytes": 0, "physical_write_bytes": 0,
                "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
                "service_ns": 0.0, "actual_bandwidth_gb_s": 0.0,
                "bandwidth_ceiling_gb_s": 0.0,
                "operation": kind.value.lower(),
                "timing_model": "physical_transaction_v1",
                "pages_touched": 0,
            }
        is_byte_addressed = isinstance(config, (DramConfig, SramConfig))
        operation = Operation.READ if kind is AccessKind.READ else Operation.WRITE if kind is AccessKind.WRITE else Operation.ERASE
        if arrival_ns is not None:
            _non_negative(arrival_ns, "arrival_ns")
        context = PhysicalRuntimeContext() if runtime is None else runtime
        active_runtime = _physical_runtime(
            config,
            self.physical_owner,
            context,
            preview=preview or runtime is None,
            capture_details=(False if summary_only else None),
        )
        request_arrival_ns = active_runtime.clock_ns if arrival_ns is None else float(arrival_ns)
        request = AccessRequest(
            f"{self.service_id}:physical",
            operation,
            0 if page_offset_bytes is None else page_offset_bytes,
            max(1, byte_count),
            arrival_ns=request_arrival_ns,
        )
        result = active_runtime.core.submit(request)
        if not preview and runtime is not None:
            active_runtime.clock_ns = max(active_runtime.clock_ns, result.completion_ns)
        counters = dict(result.counters)
        counters.update({
            "logical_bytes": result.logical_bytes,
            "physical_bytes": result.transfer_bytes,
            "physical_read_bytes": result.physical_read_bytes,
            "physical_write_bytes": result.physical_write_bytes,
            "host_transfer_bytes": result.host_transfer_bytes,
            "internal_transfer_bytes": result.internal_transfer_bytes,
            "service_ns": result.latency_ns,
            "arrival_ns": result.arrival_ns,
            "completion_ns": result.completion_ns,
            "actual_bandwidth_gb_s": result.actual_bandwidth_gb_s,
            "bandwidth_ceiling_gb_s": result.bandwidth_ceiling_gb_s,
            "operation": operation.value,
            "timing_model": "physical_transaction_v1",
            "pages_touched": result.counters.get("page_count", 0) if not is_byte_addressed else 0,
            "write_completion": (
                result.counters.get("write_completion", "media_program_complete")
                if operation is Operation.WRITE else "n/a"
            ),
        })
        return counters

    def price_batch(
        self, requests, *, start_ns=0.0,
        runtime: Optional[PhysicalRuntimeContext] = None,
    ):
        """Price a batch atomically when an explicit runtime is supplied."""
        context = runtime
        saved = context.snapshot() if context is not None else None
        try:
            return self._price_batch(requests, start_ns=start_ns, runtime=runtime)
        except Exception:
            if saved is not None:
                context.restore(saved)
            raise

    def _price_batch(
        self,
        requests,
        *,
        start_ns=0.0,
        runtime: Optional[PhysicalRuntimeContext] = None,
    ):
        """Price an explicit DRAM/NAND request batch."""
        _non_negative(start_ns, "start_ns")
        if self.component is None:
            raise ValueError("price_batch requires a physical memory component")
        if self.component.metadata.get("host_write_staging") is not None:
            raise ValueError("host-staged HBF requires logical buffer descriptors and resolved host routes; use resolve_physical_task")
        from .physical_contract import require_physical_memory_config
        config = require_physical_memory_config(self.component)
        from .memory_types import AccessRequest, Operation
        explicit_context = runtime
        context = PhysicalRuntimeContext() if explicit_context is None else explicit_context
        active_runtime = _physical_runtime(
            config,
            self.physical_owner,
            context,
            preview=explicit_context is None,
        )
        parsed = []
        for index, item in enumerate(requests):
            if not isinstance(item, Mapping):
                raise ValueError("physical batch requests must be mappings")
            obsolete = {"page_offset_bytes", "offset_bytes", "offset", "size_bytes", "bytes"}.intersection(item)
            if obsolete:
                raise ValueError("physical batch requests use address and byte_count; unsupported fields: " + ", ".join(sorted(obsolete)))
            op = str(item.get("operation", "")).lower()
            if "address" in item:
                address = item["address"]
            else:
                raise ValueError("physical batch request requires an explicit address")
            if isinstance(address, bool) or not isinstance(address, int) or address < 0:
                raise ValueError("physical batch request address must be a non-negative integer")
            if isinstance(item.get("byte_count"), bool) or not isinstance(item.get("byte_count"), int):
                raise ValueError("physical batch request byte_count must be an integer")
            parsed.append(AccessRequest(
                str(item.get("request_id", f"request-{index}")),
                Operation(op), int(address), int(item["byte_count"]),
                item.get("arrival_ns", max(start_ns, active_runtime.clock_ns)),
            ))
        batch = active_runtime.core.run(parsed)
        if explicit_context is not None:
            active_runtime.clock_ns = max(active_runtime.clock_ns, batch.last_completion_ns)
        return {
            "model": "physical_transaction_queue_v1",
            "requests": tuple(batch.requests),
            "request_count": len(batch.requests),
            "logical_bytes": batch.logical_bytes,
            "physical_bytes": batch.transfer_bytes,
            "host_transfer_bytes": batch.host_transfer_bytes,
            "internal_transfer_bytes": batch.internal_transfer_bytes,
            "physical_read_bytes": batch.physical_read_bytes,
            "physical_write_bytes": batch.physical_write_bytes,
            "pages_read": batch.pages_read,
            "pages_programmed": batch.pages_programmed,
            "erase_operations": batch.erase_operations,
            "read_cache_hits": batch.counters.get("read_cache_hits", 0),
            "read_cache_misses": batch.counters.get("read_cache_misses", 0),
            "queue_wait_ns": batch.queue_wait_ns,
            "write_buffer_wait_ns": batch.write_buffer_wait_ns,
            "write_buffer_bytes": batch.write_buffer_bytes,
            "write_buffer_peak_bytes": batch.write_buffer_peak_bytes,
            "service_ns": batch.duration_ns,
            "end_ns": batch.last_completion_ns,
            "actual_bandwidth_gb_s": batch.actual_bandwidth_gb_s,
            "bandwidth_ceiling_gb_s": batch.bandwidth_ceiling_gb_s,
            **{key:batch.counters.get(key) for key in DRAM_OPTIONAL_COUNTER_FIELDS},
        }


@dataclass(frozen=True)
class LinkService:
    """An independent physical interconnect; a topology view has no service."""

    service_id: str
    physical_owner: str
    resource_id: str
    bandwidth_gb_s: float
    latency_ns: float = 0.0
    queue_depth: int = 1

    def __post_init__(self) -> None:
        for name in ("service_id", "physical_owner", "resource_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")
        _positive(self.bandwidth_gb_s, "bandwidth_gb_s")
        _non_negative(self.latency_ns, "latency_ns")
        _positive_int(self.queue_depth, "queue_depth")

    def demands(self, byte_count: int) -> Tuple[ResourceDemand, ...]:
        """Sending holds bandwidth; arrival holds one shared in-flight slot.

        Only the sender records bytes. A new logical resource name
        for the same owner never creates another physical request window.
        """
        _non_negative_int(byte_count, "byte_count")
        if byte_count == 0:
            return ()
        send_ns = byte_count / self.bandwidth_gb_s
        return (
            ResourceDemand(self.resource_id, send_ns, bytes_moved=byte_count),
            ResourceDemand(self.physical_owner + ".inflight",
                           send_ns + self.latency_ns),
        )

    @property
    def resource_capacities(self) -> Mapping[str, int]:
        return {self.physical_owner + ".inflight": self.queue_depth}

    def bill(self, byte_count: int) -> float:
        return max((demand.service_ns for demand in self.demands(byte_count)), default=0.0)


@dataclass(frozen=True)
class MemoryPosition:
    service_id: str
    offset_bytes: int = 0
    resource_id: Optional[str] = None
    allocation_id: str = ""

    def __post_init__(self) -> None:
        if not self.service_id:
            raise ValueError("service_id must be non-empty")
        _non_negative_int(self.offset_bytes, "offset_bytes")


@dataclass(frozen=True)
class DataAccess:
    operation_id: str
    kind: AccessKind
    byte_count: int
    source: Optional[MemoryPosition] = None
    target: Optional[MemoryPosition] = None
    dependencies: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id must be non-empty")
        kind = AccessKind(self.kind)
        object.__setattr__(self, "kind", kind)
        _non_negative_int(self.byte_count, "byte_count")
        if kind is AccessKind.READ and self.source is None:
            raise ValueError("READ requires source")
        if kind in (AccessKind.WRITE, AccessKind.ERASE) and self.target is None:
            raise ValueError("WRITE requires target")
        if kind is AccessKind.COPY and (self.source is None or self.target is None):
            raise ValueError("COPY requires source and target")
        if any(not isinstance(dep, str) or not dep for dep in self.dependencies):
            raise ValueError("dependencies must be non-empty strings")


@dataclass(frozen=True)
class MotionPhase:
    operation_id: str
    kind: AccessKind
    service_id: str
    physical_owner: str
    resource_id: str
    byte_count: int
    service_ns: float
    dependencies: Tuple[str, ...] = ()
    extra_demands: Tuple[ResourceDemand, ...] = ()
    physical_bytes: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def demand(self) -> ResourceDemand:
        return ResourceDemand(self.resource_id, self.service_ns, bytes_moved=self.byte_count if self.physical_bytes is None else self.physical_bytes)

    @property
    def demands(self) -> Tuple[ResourceDemand, ...]:
        return (self.demand,) + self.extra_demands


@dataclass(frozen=True)
class ExpandedMotion:
    operation_id: str
    phases: Tuple[MotionPhase, ...]
    resource_owners: Mapping[str, str]
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    dependencies: Tuple[str, ...] = ()

    @property
    def demands(self) -> Tuple[ResourceDemand, ...]:
        return tuple(demand for phase in self.phases for demand in phase.demands)

    def to_tasks(self, request_id: str) -> Tuple[TaskSpec, ...]:
        tasks = tuple(TaskSpec(
            phase.operation_id, request_id, phase.operation_id, TaskCategory.COMMUNICATION,
            dependencies=phase.dependencies, demands=phase.demands,
            metadata=dict(phase.metadata),
        ) for phase in self.phases)
        if not tasks or tasks[-1].task_id != self.operation_id:
            tasks += (TaskSpec(
                self.operation_id, request_id, self.operation_id, TaskCategory.COMMUNICATION,
                dependencies=(tasks[-1].task_id,) if tasks else self.dependencies,
            ),)
        return tasks


def resolve_service(
    component: ComponentSpec,
    profile: Any = None,
) -> PhysicalService:
    """Resolve one component service; calibration never replaces its physical cap.

    ScenarioConfig supplies ``memory_service`` from explicit component bindings.
    Direct ComponentSpec callers use the same component fields and explicit
    effective overrides. A directional zero remains unknown, including HBF writes.
    """
    metadata = component.metadata
    obsolete = {"service_id", "resource_id", "memory_resource_id", "physical_owner"}.intersection(metadata)
    if obsolete:
        raise ValueError("component memory service identity uses memory_service_owner; unsupported fields: " + ", ".join(sorted(obsolete)))
    from .physical_contract import physical_memory_bandwidth_caps_gbps
    core_caps = tuple(value / 8.0 for value in physical_memory_bandwidth_caps_gbps(component))
    resolved = metadata.get("memory_service", {})
    if not isinstance(resolved, Mapping):
        raise ValueError("memory_service must be a mapping")
    if "resource_id" in resolved:
        raise ValueError("memory_service uses physical_owner, not resource_id")
    service_id = component.component_id + ".access"
    if resolved.get("service_id", service_id) != service_id:
        raise ValueError("memory_service.service_id must equal " + service_id)
    # Derived service metadata is a view, never a second physical contract.
    physical = core_caps[0]
    shared_limit = min(physical, component.shared_bandwidth_gbps / 8.0) if component.shared_bandwidth_gbps > 0 else physical
    efficiency = float(_value(profile, "efficiency", default=metadata.get("efficiency", 1.0)))
    if not math.isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    measured = _value(profile, "measured_effective_bandwidth_gb_s", default=metadata.get("measured_effective_bandwidth_gb_s"))
    shared = resolved.get("bandwidth_gb_s")
    if shared is None:
        shared = measured if measured is not None else _value(profile, "effective_bandwidth_gb_s")
    if shared is None:
        shared = float(_value(profile, "bandwidth_gb_s", default=shared_limit)) * efficiency
    shared = _positive(shared, "effective bandwidth_gb_s")
    if shared > shared_limit + 1e-12:
        raise ValueError("effective bandwidth cannot exceed component physical bandwidth")

    def direction_bandwidth(direction: str) -> float:
        declared = getattr(component, direction + "_bandwidth_gbps") / 8.0
        ceiling = min(shared_limit, core_caps[1 if direction == "read" else 2])
        if declared > 0:
            ceiling = min(ceiling, declared)
        if direction == "write" and not component.is_writable:
            ceiling = 0.0
        raw = resolved.get(direction + "_bandwidth_gb_s")
        if raw is None:
            raw = _value(profile, "effective_" + direction + "_bandwidth_gb_s")
        if raw is None:
            raw = measured
        if raw is None:
            raw = _value(profile, direction + "_bandwidth_gb_s")
            raw = (float(raw) * efficiency if raw is not None else min(shared, ceiling * efficiency))
        value = _non_negative(raw, direction + "_bandwidth_gb_s")
        if value > min(physical, ceiling) + 1e-12:
            raise ValueError(f"effective {direction} bandwidth cannot exceed component physical bandwidth")
        return value

    def parameter(name: str, *, default: Any) -> Any:
        return _value(resolved, name, default=_value(metadata, name, default=_value(profile, name, default=default)))

    owner = str(resolved.get("physical_owner") or default_memory_resource_id(component))
    model = str(parameter("service_model", default="analytical" if component.is_active_memory else "serialized"))
    if model not in {"analytical", "serialized", "overlapped"}:
        raise ValueError("memory_service_model must be analytical, serialized or overlapped")
    granularity = parameter("transaction_bytes", default=256)
    _positive_int(granularity, "transaction_bytes")
    return PhysicalService(
        service_id=service_id,
        physical_owner=owner,
        resource_id=owner,
        read_bandwidth_gb_s=direction_bandwidth("read"),
        write_bandwidth_gb_s=direction_bandwidth("write"),
        read_latency_ns=_non_negative(parameter("read_latency_ns", default=0.0), "read_latency_ns"),
        write_latency_ns=_non_negative(parameter("write_latency_ns", default=0.0), "write_latency_ns"),
        transaction_granularity=granularity,
        queue_depth=parameter("max_outstanding_requests", default=32 if component.is_active_memory else 1),
        parallel_lanes=parameter("parallel_lanes", default=1),
        efficiency=efficiency,
        service_model=model,
        storage_id=str(metadata.get("physical_storage_id") or component.component_id),
        component=component,
    )


def resolve_link_service(
    link: LinkSpec,
    source: ComponentSpec,
    target: ComponentSpec,
) -> Optional[LinkService]:
    """Return a service only for a real interconnect, never a local view."""

    local_view = (
        str(link.metadata.get("bandwidth_source", "")).strip().lower()
        in {"component", "memory_component", "shared_component"}
    )
    if link.metadata.get("service_ref") or local_view:
        reference = link.metadata.get("service_ref")
        if not reference:
            raise ValueError(f"link {link.link_id} local memory view requires an explicit service_ref")
        memories = [component for component in (source, target)
                    if component.is_storage and reference == component.component_id + ".access"]
        if len(memories) != 1:
            raise ValueError(f"link {link.link_id} service_ref must identify one connected storage service")
        memory = memories[0]
        service = resolve_service(memory)
        resource = link.metadata.get("bandwidth_resource_id")
        if resource is not None and resource not in {service.resource_id, service.physical_owner}:
            raise ValueError(f"link {link.link_id} bandwidth_resource_id conflicts with referenced memory service")
        if link.bandwidth_gbps > 0 and not math.isclose(link.bandwidth_gbps, memory.shared_bandwidth_gbps):
            raise ValueError(f"link {link.link_id} bandwidth conflicts with referenced memory service")
        if link.latency_ns > 0:
            raise ValueError(f"link {link.link_id} service_ref cannot declare extra latency")
        return None
    bandwidth_gbps = float(link.bandwidth_gbps)
    if bandwidth_gbps <= 0:
        raise ValueError(f"link {link.link_id} requires positive bandwidth")
    owner = str(link.metadata.get("physical_owner", link.link_id))
    resource = str(link.metadata.get("resource_id", link.link_id))
    return LinkService(link.link_id, owner, resource, bandwidth_gbps / 8.0, float(link.latency_ns), int(link.metadata.get("queue_depth", 1)))


def expand_access(
    access: DataAccess,
    services: Mapping[str, PhysicalService],
    *,
    links: Optional[Mapping[Tuple[str, str], LinkService]] = None,
) -> ExpandedMotion:
    """Expand READ/WRITE/COPY into physical phases and explicit dependencies."""

    if (
        access.kind is AccessKind.COPY
        and access.source is not None
        and access.target is not None
        and (services[access.source.service_id].storage_id or services[access.source.service_id].service_id)
        == (services[access.target.service_id].storage_id or services[access.target.service_id].service_id)
        and access.source.offset_bytes == access.target.offset_bytes
        and access.source.allocation_id
        and access.source.allocation_id == access.target.allocation_id
    ):
        # An alias of the exact same range is already resident in the target
        # view.  Keep this a real zero-cost operation instead of charging a
        # read and write against the shared owner.
        return ExpandedMotion(access.operation_id, (), {}, dependencies=access.dependencies)

    def phase(kind: AccessKind, position: MemoryPosition, deps: Tuple[str, ...]) -> MotionPhase:
        service = services[position.service_id]
        bill = service.price(
            kind,
            access.byte_count,
            page_offset_bytes=position.offset_bytes,
        )
        metadata: dict[str, Any] = {}
        raw_config = (
            service.component.metadata.get("physical_memory_config")
            if service.component is not None
            else None
        )
        if raw_config is not None:
            metadata.update({
                "physical_memory_config": raw_config,
                "memory_accesses": ({
                    "operation": kind.value.lower(),
                    "address": position.offset_bytes,
                    "byte_count": access.byte_count,
                    "physical_owner": service.physical_owner,
                    "resource_id": service.resource_id,
                },),
            })
        return MotionPhase(access.operation_id, kind, service.service_id, service.physical_owner,
                           service.resource_id, access.byte_count, float(bill["service_ns"]), deps,
                           physical_bytes=int(bill["physical_bytes"]),
                           metadata=metadata)

    if access.kind is AccessKind.READ:
        phases = (phase(AccessKind.READ, access.source, access.dependencies),)
    elif access.kind is AccessKind.WRITE:
        phases = (phase(AccessKind.WRITE, access.target, access.dependencies),)
    elif access.kind is AccessKind.ERASE:
        phases = (phase(AccessKind.ERASE, access.target, access.dependencies),)
    else:
        read = phase(AccessKind.READ, access.source, access.dependencies)
        deps = access.dependencies + (read.operation_id + ".read",)
        read = MotionPhase(read.operation_id + ".read", read.kind, read.service_id, read.physical_owner, read.resource_id, read.byte_count, read.service_ns, read.dependencies, physical_bytes=read.physical_bytes, metadata=read.metadata)
        phases_list = [read]
        link = links.get((access.source.service_id, access.target.service_id)) if links is not None else None
        if link is not None:
            link_demands = link.demands(access.byte_count)
            phases_list.append(MotionPhase(
                access.operation_id + ".link", AccessKind.COPY, link.service_id,
                link.physical_owner, link.resource_id, access.byte_count, link_demands[0].service_ns, deps,
                extra_demands=link_demands[1:],
            ))
            deps = (access.operation_id + ".link",)
        write = phase(AccessKind.WRITE, access.target, deps)
        phases_list.append(MotionPhase(access.operation_id + ".write", write.kind, write.service_id, write.physical_owner, write.resource_id, write.byte_count, write.service_ns, deps, physical_bytes=write.physical_bytes, metadata=write.metadata))
        phases = tuple(phases_list)
    owners = {}
    for item in phases:
        previous = owners.get(item.resource_id)
        if previous is not None and previous != item.physical_owner:
            raise ValueError(
                "resource {} has conflicting physical owners {} and {}".format(
                    item.resource_id, previous, item.physical_owner
                )
            )
        owners[item.resource_id] = item.physical_owner
    capacities = {}
    for item in phases:
        for demand in item.extra_demands:
            # Inflight slots are owned by the physical link service, so two
            # logical aliases of one link share one queue window.
            owners[demand.resource_id] = item.physical_owner + ".inflight"
            capacities[demand.resource_id] = link.queue_depth
    return ExpandedMotion(access.operation_id, phases, owners, capacities, access.dependencies)


def expand_accesses(
    accesses: Iterable[DataAccess],
    services: Mapping[str, PhysicalService],
    *,
    links: Optional[Mapping[Tuple[str, str], LinkService]] = None,
) -> Tuple[ExpandedMotion, ...]:
    """Expand a batch once per operation id.

    Frontends may describe one physical operation from both an operator and a
    DMA view.  The operation id is the stable identity; the first description
    is retained only if later descriptions have the same physical semantics.
    """

    result = []
    seen = {}
    for access in accesses:
        signature = (access.kind, access.byte_count, access.source, access.target, access.dependencies)
        if access.operation_id in seen:
            if seen[access.operation_id] != signature:
                raise ValueError(f"conflicting descriptions for operation {access.operation_id}")
            continue
        seen[access.operation_id] = signature
        result.append(expand_access(access, services, links=links))
    return tuple(result)


@dataclass(frozen=True)
class EndpointService:
    name: str
    demands: Tuple[ResourceDemand, ...]
    metadata: Mapping[str, object]


_CONSTRUCTION_NATIVE_SERVICE_METHODS = tuple(
    (name, getattr(PhysicalService, name)) for name in ("price", "_price"))


def _native_construction_callbacks() -> bool:
    """Only closed native pricing may omit a discarded cold execution."""
    from .dram_core import DramCore, _OWNED_TIMELINE_DRAM_METHODS
    from .nand_core import NandCore, _OWNED_OWNER_NAND_METHODS
    from .sram_core import SramCore, _OWNED_OWNER_SRAM_METHODS
    from .memory_transfer import ResourceTimeline, _OWNED_TIMELINE_METHODS
    from ._compiled_hbm import _has_native_timeline_hooks
    from ._nand_acceleration import has_native_timeline_hooks
    return (
        all(getattr(PhysicalService, name) is original
            for name, original in _CONSTRUCTION_NATIVE_SERVICE_METHODS)
        and all(getattr(core, name) is original
                for core, methods in (
                    (DramCore, _OWNED_TIMELINE_DRAM_METHODS),
                    (NandCore, _OWNED_OWNER_NAND_METHODS),
                    (SramCore, _OWNED_OWNER_SRAM_METHODS),
                    (ResourceTimeline, _OWNED_TIMELINE_METHODS))
                for name, original in methods)
        and _has_native_timeline_hooks(None)
        and has_native_timeline_hooks()
    )


def _construction_service_bill(
    service: PhysicalService,
    kind: AccessKind,
    byte_count: int,
    *,
    page_offset_bytes: int,
    physical_config=None,
    runtime: Optional[PhysicalRuntimeContext] = None,
    arrival_ns: Optional[float] = None,
    preview: bool = False,
    summary_only: bool = True,
) -> Mapping[str, object]:
    """Preserve descriptors when live dispatch owns the physical service.

    Direct GPU backing phases and communication endpoints use the same
    construction contract. Ordinary pricing, explicit runtime submissions,
    and host-staged HBF retain their existing preview frontiers.
    """
    component = service.component
    deferred = (
        _DEFER_ENDPOINT_CONSTRUCTION.get() is not None
        and runtime is None and summary_only
        and component is not None
        and component.metadata.get("host_write_staging") is None
    )
    if not deferred:
        return service.price(
            kind, byte_count, page_offset_bytes=page_offset_bytes,
            runtime=runtime, arrival_ns=arrival_ns, preview=preview,
            summary_only=summary_only,
        )
    kind = AccessKind(kind)
    _non_negative_int(byte_count, "byte_count")
    _non_negative_int(page_offset_bytes, "memory access address")
    if kind is AccessKind.COPY:
        raise ValueError("COPY must be expanded before billing")
    if physical_config is None:
        from .physical_contract import require_physical_memory_config
        physical_config = require_physical_memory_config(component)
    from .memory_types import DramConfig, SramConfig
    if kind is AccessKind.ERASE and isinstance(physical_config, (DramConfig, SramConfig)):
        raise ValueError("byte-addressed memory does not support ERASE operations")
    if byte_count and page_offset_bytes + byte_count > physical_config.effective_capacity_bytes:
        raise ValueError("physical access exceeds memory capacity")
    return {"logical_bytes": byte_count, "physical_bytes": 0,
            "host_transfer_bytes": 0, "internal_transfer_bytes": 0,
            "service_ns": 0.0, "physical_cost_deferred": True,
            "timing_model": "native_dispatch_pending"}


def endpoint_service(
    component: ComponentSpec,
    byte_count: int,
    *,
    operation: str,
    name: str,
    page_offset_bytes: Optional[int] = None,
    runtime: Optional[PhysicalRuntimeContext] = None,
    arrival_ns: Optional[float] = None,
    preview: bool = False,
    compact_preview: bool = False,
) -> Optional[EndpointService]:
    """Lower one memory access using the canonical physical transaction model.

    ``compact_preview`` is for planner-authored physical tasks.  It removes
    expanded interval/payload traces from the analytical preview while
    preserving scalar counters; dispatch still recomputes the full trace.
    """
    _non_negative_int(byte_count, "byte_count")
    # Compute/controller endpoints have no DRAM/NAND array. Their pipelines
    # and local SRAM are accounted by the compute models.
    if not component.is_storage:
        return None
    address = page_offset_bytes
    if address is None:
        address = component.metadata.get("memory_access_offset_bytes")
    from .physical_contract import require_physical_memory_config
    cache = _DEFER_ENDPOINT_CONSTRUCTION.get() if compact_preview and runtime is None else None
    cached = cache.get(id(component)) if cache is not None else None
    if cached is None:
        physical_config = require_physical_memory_config(component)
        service = resolve_service(component)
        if cache is not None:
            cache[id(component)] = (component, physical_config, service)
    else:
        _component, physical_config, service = cached
    if address is None:
        raise ValueError("physical_memory_config requires an explicit memory address")
    if address is not None:
        _non_negative_int(address, "memory access address")
    op_name = str(operation).lower()
    if op_name not in {"read", "write", "erase"}:
        raise ValueError("operation must be read, write or erase")
    billed = _construction_service_bill(
        service,
        AccessKind.READ if op_name == "read" else AccessKind.WRITE if op_name == "write" else AccessKind.ERASE,
        byte_count, page_offset_bytes=address, physical_config=physical_config,
        runtime=runtime, arrival_ns=arrival_ns, preview=preview,
        summary_only=compact_preview,
    )
    if compact_preview and runtime is None and component.metadata.get("physical_memory_config") is not None:
        # Planner-authored endpoint tasks only need scalar preview counters.
        # The event kernel resolves the physical descriptor again at dispatch,
        # where the complete interval/payload trace is attached to the result.
        # Avoid retaining one expanded burst tuple per task in a serving
        # schedule while keeping the default endpoint API unchanged.
        heavy_keys = {
            "resource_intervals",
            "resource_interval_payloads",
            "resource_last_intervals",
            "physical_resource_intervals",
            "physical_resource_last_intervals",
            "physical_resource_lane_completions",
        }
        billed = {
            key: value for key, value in billed.items() if key not in heavy_keys
        }
        billed["details_truncated"] = True
    physical = int(billed.get("physical_bytes", 0))
    if op_name == "erase":
        physical = 0
    demand = ResourceDemand(
        resource_id=service.resource_id,
        service_ns=float(billed.get("service_ns", 0.0)),
        bytes_moved=physical,
    )
    endpoint_metadata = {
            "event_kind": f"memory_{op_name}",
            "component_id": component.component_id,
            "bytes": byte_count,
            "address": address,
            "physical_bytes": physical,
            "transferred_bytes": int(billed["host_transfer_bytes"]),
            "physical_kind": component.normalized_kind,
            "physical_service_id": service.service_id,
            "physical_owner": service.physical_owner,
            "physical_resource_id": service.resource_id,
            "access_kind": op_name.upper(),
            "timing_evidence": "ANALYTICAL",
        }
    endpoint_metadata.update({
        "physical_memory_config": component.metadata.get("physical_memory_config"),
        "physical_execution_by_owner": {service.physical_owner: {**billed, "kind": physical_config.kind.value}},
        "physical_execution_role": "construction_preview" if runtime is None else "runtime_dispatch",
    })
    if component.metadata.get("physical_memory_config") is not None:
        endpoint_metadata["memory_accesses"] = ({
            "operation": op_name,
            "address": address,
            "byte_count": byte_count,
            "physical_owner": service.physical_owner,
            "resource_id": service.resource_id,
        },)
    return EndpointService(
        name=f"{name}.{component.component_id}.{op_name}",
        demands=(demand,),
        metadata=endpoint_metadata,
    )
