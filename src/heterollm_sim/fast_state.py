"""Read-only capacity and page-boundary guards for Decode forecasting.

Nothing here advances an FTL, a device calendar, or a transaction counter.
The page totals describe monotonic logical K/V appends.  Headroom is an
explicit conservative budget derived from the observed detailed prefix,
not a claim that the unexecuted physical tail has been simulated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter, deque
import heapq
from math import ceil, gcd, lcm
from typing import Any, Mapping, TYPE_CHECKING

from .data_motion import resolve_service
from .planner import _execution_layers, _kv_components, _kv_tensor_bytes, _parallel_plan

if TYPE_CHECKING:
    from .serving import _OnlineRuntime

PAGE_BYTES = 4096


def hbf_kv_page_period(scenario) -> int:
    """Find the joint K/V flush period before executing any Decode sample."""
    period = 1
    parallel = _parallel_plan(scenario)
    for layer in _execution_layers(scenario):
        row_bytes = _kv_tensor_bytes(scenario, layer, parallel.tp_degree, 1)
        for rank in parallel.ranks_for_layer(layer):
            component_id, _, _ = _kv_components(scenario, rank, rank.component_id, layer)
            if component_id and scenario.hardware.get_component(component_id).normalized_kind == "hbf":
                period = lcm(period, PAGE_BYTES // gcd(PAGE_BYTES, row_bytes))
    return period


@dataclass(frozen=True)
class FastTailProjection:
    remaining_decode_steps: int
    current_cached_tokens: int
    final_cached_tokens: int
    logical_kv_growth_bytes: int
    required_capacity_pages: int
    capacity_pages: int
    kv_buffers: tuple[Mapping[str, Any], ...]
    hbf_devices: tuple[Mapping[str, Any], ...]
    safety_reasons: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return not self.safety_reasons

    def to_metadata(self) -> dict[str, Any]:
        return {"schema": "heterollm.fast-tail-projection/v1", **asdict(self),
                "safe": self.safe,
                "semantics": "logical_append_projection_and_observed_prefix_headroom_guard",
                "physical_state_advanced": False,
                "physical_tail_transactions_recorded": False}


def _ceil_pages(byte_count: int) -> int:
    return (byte_count + PAGE_BYTES - 1) // PAGE_BYTES


def _free_pages_by_bank(controller) -> tuple[int, ...]:
    """Count interval-backed free blocks without materializing the medium."""
    lanes = controller.config.array_units
    blocks = [0] * lanes
    for span in controller._free_blocks.ranges:
        full, tail = divmod(len(span), lanes)
        for bank in range(lanes):
            blocks[bank] += full
        for offset in range(tail):
            blocks[(span.start + offset) % lanes] += 1
    pages = [count * controller.config.pages_per_block for count in blocks]
    for block in controller._active:
        pages[block % lanes] += controller.config.pages_per_block - controller._used[block]
    return tuple(pages)


def _project_program_allocations(controller, count: int, core=None) -> dict[str, Any]:
    """Replay only the FTL's scalar allocation order, skipping full rounds.

    ``_allocate`` cannot collect while an active/free page remains. Dynamic
    buffer release invalidates owners but does not return these pages. Thus
    this projection covers every program in the supplied budget independent
    of buffer lifetimes, without copying pages or submitting physical IO.
    """
    lanes, per_block = controller._lanes, controller.config.pages_per_block
    active = deque(controller._active)
    free = controller._free_blocks.copy()
    keys = {block: per_block - controller._used[block] for block in active}
    heap = [(key, block) for block, key in keys.items()]
    heapq.heapify(heap)
    bank_width = Counter(block % lanes for block in active)
    by_bank = [0] * lanes
    page_zero_by_bank = [0] * lanes
    zero_blocks = {block for block in active if controller._used[block] == 0}
    remaining, offset, opened, erases = count, 0, 0, 0

    def program_page_zero(block):
        nonlocal erases
        if block not in zero_blocks:
            return
        zero_blocks.remove(block)
        page_zero_by_bank[block % lanes] += 1
        if core is not None:
            from .memory_mapping import map_nand_address
            mapping = map_nand_address(core.config, controller._address(block, 0))
            erases += int(bool(core._next_program_page.get(core._block_key(mapping), 0)))

    while remaining:
        while len(active) < lanes and free:
            block = free.popleft()
            active.append(block)
            keys[block] = per_block + offset
            bank_width[block % lanes] += 1
            heapq.heappush(heap, (keys[block], block))
            opened += 1
            zero_blocks.add(block)
        if not active:
            break
        while heap and keys.get(heap[0][1]) != heap[0][0]:
            heapq.heappop(heap)
        rounds = min(remaining // len(active), heap[0][0] - offset - 1)
        if rounds > 0:
            for block in tuple(zero_blocks):
                program_page_zero(block)
            for bank, width in bank_width.items():
                by_bank[bank] += rounds * width
            remaining -= rounds * len(active)
            offset += rounds
            continue
        block = active.popleft()
        program_page_zero(block)
        bank = block % lanes
        by_bank[bank] += 1
        remaining -= 1
        key = keys[block] - 1
        if key > offset:
            keys[block] = key
            active.append(block)
            heapq.heappush(heap, (key, block))
        else:
            del keys[block]
            bank_width[bank] -= 1
    return {"program_pages_by_bank": tuple(by_bank),
            "new_blocks_opened": opened,
            "page_zero_programs": sum(page_zero_by_bank),
            "page_zero_programs_by_bank": tuple(page_zero_by_bank),
            "unallocated_program_pages": remaining,
            "reused_blocks_requiring_erase": erases}


def project_fast_tail(
    runtime: _OnlineRuntime, *, detailed_decode_steps: int | None = None,
) -> FastTailProjection:
    """Describe the unexecuted single-request tail without mutating runtime.

    The caller still owns model-family eligibility and calibration accuracy.
    A clean guard says only that the declared monotonic-append assumptions
    have headroom; it never certifies the requested timing error.
    """
    if detailed_decode_steps is not None and (
        isinstance(detailed_decode_steps, bool)
        or not isinstance(detailed_decode_steps, int)
        or detailed_decode_steps < 0
    ):
        raise ValueError("detailed_decode_steps must be a non-negative integer")
    policy = runtime.plan.kv_policy
    reasons: list[str] = []
    states = runtime._state_values
    if len(states) != 1:
        return FastTailProjection(0, 0, 0, 0, 0, policy.capacity_pages,
                                  (), (), ("fast_tail_requires_single_request",))
    state = states[0]
    current = state.cached_tokens
    final = state.spec.prompt_tokens + max(0, state.spec.output_tokens - 1)
    remaining = max(0, state.spec.output_tokens - state.committed)
    if state.prefill_cursor < state.spec.prompt_tokens or state.first_token_ns is None:
        reasons.append("exact_prefill_and_first_token_required")
    if state.status.value not in {"running", "finished"}:
        reasons.append("request_not_running_or_finished")
    if policy.layout_mode != "legacy_single":
        reasons.append("fast_tail_requires_contiguous_generic_kv")
    if (policy.offload_component and policy.offload_ratio) or policy.prefetch_distance:
        reasons.append("kv_offload_or_prefetch_requires_detailed_execution")
    if runtime.plan.mtp.enabled:
        reasons.append("mtp_requires_detailed_execution")
    required_pages = runtime.ledger.pages_for_tokens(final)
    if required_pages > policy.capacity_pages:
        reasons.append("final_kv_exceeds_logical_capacity")
    scenario = runtime.plan.scenario
    parallel = _parallel_plan(scenario)
    if parallel.world_size != 1:
        reasons.append("fast_tail_requires_single_rank")
    buffers: list[Mapping[str, Any]] = []
    rows_by_owner: dict[str, list[Mapping[str, Any]]] = {}
    physical = runtime._execution_resource_kernel.physical_runtime
    for layer in _execution_layers(scenario):
        if layer.is_linear_attention:
            reasons.append("linear_attention_requires_detailed_execution")
            continue
        ranks = tuple(parallel.ranks_for_layer(layer))
        if len(ranks) != 1:
            continue
        rank = ranks[0]
        row_bytes = _kv_tensor_bytes(scenario, layer, parallel.tp_degree, 1)
        component_id, offload_id, ratio = _kv_components(scenario, rank, rank.component_id, layer)
        if not component_id or (offload_id and ratio):
            reasons.append("unsupported_kv_owner_or_offload")
            continue
        component = scenario.hardware.get_component(component_id)
        owner = resolve_service(component).physical_owner
        for part in ("k", "v"):
            buffer_id = "kv:{}:{}:rank{}:tp{}:pp{}:{}".format(
                state.spec.request_id, layer.layer_id, rank.rank,
                rank.tp_rank, rank.pp_rank, part)
            start, stop = current * row_bytes, final * row_bytes
            row = {
                "buffer_id": buffer_id, "layer_id": layer.layer_id, "part": part,
                "component_id": component_id, "physical_owner": owner,
                "bytes_per_token": row_bytes,
                "current_bytes": start, "final_bytes": stop,
                "growth_bytes": max(0, stop - start),
                "current_tail_bytes": start % PAGE_BYTES,
                "final_tail_bytes": stop % PAGE_BYTES,
                "new_logical_pages": max(0, _ceil_pages(stop) - _ceil_pages(start)),
                "completed_page_appends": max(0, stop // PAGE_BYTES - start // PAGE_BYTES),
                "tail_page_slots_upper_bound": int(row_bytes % PAGE_BYTES != 0 and remaining > 0),
            }
            buffers.append(row)
            if component.normalized_kind == "hbf":
                rows_by_owner.setdefault(owner, []).append(row)
                if row_bytes <= 0 or (PAGE_BYTES % row_bytes and row_bytes % PAGE_BYTES):
                    reasons.append("hbf_kv_rows_require_aligned_page_period:" + owner)
                allocator = physical.allocators.get(owner)
                allocation = None if allocator is None else allocator._allocations.get((buffer_id, 0))
                if allocation is None or allocation.size_bytes < stop:
                    reasons.append("hbf_full_kv_extent_not_reserved:" + buffer_id)
    devices = []
    decode_batches = tuple(batch for batch in runtime.batches if batch.kind == "decode")
    if detailed_decode_steps is not None and detailed_decode_steps != len(decode_batches):
        reasons.append("detailed_decode_step_count_does_not_match_runtime")
    for owner, rows in rows_by_owner.items():
        staging = physical.hbf_staging.get(owner)
        if staging is None:
            reasons.append("hbf_staging_state_unavailable:" + owner)
            continue
        controller = staging["controller"]
        config = controller.config
        stats = controller.stats
        for stat in ("gc_cycles", "pressure_flushes", "rmw_pages"):
            if stats.get(stat, 0):
                reasons.append("hbf_observed_" + stat + ":" + owner)
        new_pages = sum(int(row["new_logical_pages"]) for row in rows)
        page_appends = sum(int(row["completed_page_appends"]) for row in rows)
        free_logical = controller.usable_capacity_bytes - controller.occupied_bytes
        if new_pages * PAGE_BYTES > free_logical:
            reasons.append("hbf_logical_page_headroom_exhausted:" + owner)
        bank_pages = _free_pages_by_bank(controller)
        available_pages = sum(bank_pages)
        observed_max = max((int(batch.cost.metadata.get("storage_traffic", {}).get(
            "pages_programmed", 0) or 0) for batch in decode_batches), default=0)
        # Charge all observed media programs to each HBF device: this is
        # intentionally conservative for multiple-device layouts.  Add one
        # worst-case K/V boundary per append on top, even if already seen.
        kv_step_bound = sum(_ceil_pages(int(row["bytes_per_token"])) for row in rows)
        context_scale = max(1.0, final / max(1, current))
        budget_per_step = ceil(observed_max * context_scale) + kv_step_bound
        program_budget = remaining * budget_per_step
        if remaining and not decode_batches:
            reasons.append("hbf_decode_media_budget_uncalibrated:" + owner)
        if program_budget > available_pages:
            reasons.append("hbf_program_headroom_may_require_gc:" + owner)
        device = physical.runtimes.get(owner)
        core = None if device is None else device.core
        allocation_projection = _project_program_allocations(controller, program_budget, core)
        if core is None:
            reasons.append("hbf_nand_program_frontier_unavailable:" + owner)
        if allocation_projection["reused_blocks_requiring_erase"]:
            reasons.append("hbf_free_block_reuse_may_require_erase:" + owner)
        tail_slots = sum(int(row["tail_page_slots_upper_bound"]) for row in rows)
        host_slots = controller.host_capacity_bytes // PAGE_BYTES - 1
        resident_slots = len(controller._resident)
        if resident_slots + tail_slots > host_slots:
            reasons.append("hbf_host_tail_slots_may_require_pressure_flush:" + owner)
        devices.append({
            "physical_owner": owner, "bank_count": config.array_units,
            "pages_per_block": config.pages_per_block,
            "free_blocks": len(controller._free_blocks),
            "free_program_pages": available_pages,
            "free_program_pages_by_bank": bank_pages,
            "minimum_bank_program_pages": min(bank_pages, default=0),
            "program_page_guard_budget": program_budget,
            "program_allocation_guard": allocation_projection,
            # OCP page-0 writes erase even a previously unused block. This
            # is a serial service upper bound, not predicted stall time.
            "ordinary_erase_operations_guard_budget": allocation_projection["page_zero_programs"],
            "ordinary_erase_serial_time_guard_ns": allocation_projection["page_zero_programs"] * config.block_erase_ns,
            "block_erase_ns": config.block_erase_ns,
            "program_pages_per_decode_guard_budget": budget_per_step,
            "observed_max_program_pages_per_decode": observed_max,
            "kv_completed_page_appends": page_appends,
            "new_logical_kv_pages": new_pages,
            "logical_page_headroom_bytes": free_logical,
            "host_data_slots": host_slots, "observed_host_resident_slots": resident_slots,
            "additional_kv_tail_slots_upper_bound": tail_slots,
            "observed_gc_cycles": int(stats.get("gc_cycles", 0)),
            "observed_pressure_flushes": int(stats.get("pressure_flushes", 0)),
            "observed_rmw_pages": int(stats.get("rmw_pages", 0)),
            "budget_assumption": "fixed_decode_workspaces_plus_context_scaled_observed_writes_and_kv_boundary_upper_bound",
            "future_erase_or_gc_events": "ordinary_page_zero_erase_bounded_gc_excluded",
            "future_physical_state": "not_advanced",
        })
    return FastTailProjection(remaining, current, final,
        max(0, final - current) * int(policy.logical_bytes_per_token),
        required_pages, policy.capacity_pages, tuple(buffers), tuple(devices),
        tuple(dict.fromkeys(reasons)))
