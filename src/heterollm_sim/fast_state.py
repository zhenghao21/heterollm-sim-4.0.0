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


def _fixed_decode_write_geometry(runtime, kv_buffers, final_tokens):
    """Qualify constant ordinary writes using two *unexecuted* planner IRs.

    The analytic dense path has only one context-shaped operation: attention.
    Its fused output has a fixed one-token shape. Terminal SRAM admission
    proves admission throughout this interval because its tiled working set
    is nondecreasing in context. GPU kernel profiles and unfused attention
    are deliberately excluded; their scratch/score writes can change shape.
    """
    from .planner import (TopologyAwareBatchCostProvider, _compilation_scope,
                          _lower_serving_cohort)
    from .precision import canonical_dtype
    from .serving import BatchCohort, BatchItem

    unavailable = lambda reason: {"qualified": False, "reason": reason,
        "physical_state_advanced": False, "physical_io_executed": False}
    if type(runtime.lowerer) is not TopologyAwareBatchCostProvider:
        return unavailable("standard_topology_lowerer_required")
    scenario = runtime.plan.scenario
    layers = _execution_layers(scenario)
    if (len(runtime._state_values) != 1 or _parallel_plan(scenario).world_size != 1
            or any(layer.is_linear_attention or layer.is_moe or layer.kind != "dense"
                   or canonical_dtype(layer.dtype) != "bf16" or layer.quantization for layer in layers)
            or not scenario.fusion_policy.flash_attention
            or scenario.llama_cpp_config is not None):
        return unavailable("ordinary_bf16_dense_flash_attention_required")
    if any(getattr(scenario.resolve_component_profile(component), "kernel_model", None) is not None
           for component in scenario.hardware.components if component.normalized_kind == "gpu"):
        return unavailable("analytic_gpu_profile_required")
    if scenario.model.graph.attributes.get("metadata", {}).get("gguf_architecture_id"):
        return unavailable("generic_dense_lowering_required")
    state = runtime._state_values[0]
    first, last = max(state.spec.prompt_tokens, state.cached_tokens - 1), final_tokens - 1
    kv_ids = {str(row["buffer_id"]) for row in kv_buffers}
    owners = {str(row["physical_owner"]) for row in kv_buffers
              if scenario.hardware.get_component(row["component_id"]).normalized_kind == "hbf"}

    def blueprint(context):
        item = BatchItem(state.spec.request_id, "decode", 1, context,
            kv_append_tokens=1, kv_materialized_tokens=1, main_tokens=1, draft_tokens=0,
            verifier_tokens=1, committed_tokens=1, logit_tokens=1)
        cohort = BatchCohort("fast-write-geometry", "decode", 0., (item,))
        # This entry point creates IR only. It neither previews a cost
        # schedule nor binds/executes any access against the live kernel.
        lowering = _lower_serving_cohort(scenario, cohort)
        writes, declarations, fused = [], [], set()
        partials = Counter()
        rounded_pages = Counter()
        for task in lowering.schedule.tasks:
            metadata = task.metadata
            if metadata.get("fusion_group") == "flash_attention" and metadata.get("fusion_enabled") is True:
                fused.add(str(metadata.get("layer_id")))
            accesses = metadata.get("memory_accesses", ())
            if (set(metadata.get("hbf_host_staging_by_owner", ())) & owners
                    and not accesses and any(demand.bytes_moved for demand in task.demands)):
                return None, "hbf_access_descriptor_unavailable"
            for access in accesses:
                owner = str(access.get("physical_owner", ""))
                operation = getattr(access.get("operation"), "value", access.get("operation"))
                if owner not in owners or operation != "write":
                    continue
                buffer_id = str(access.get("buffer_id", ""))
                if buffer_id in kv_ids:
                    continue
                if (not buffer_id or buffer_id.startswith("kv:")
                        or not access.get("allocation_generation")
                        or access.get("allocation_arena") == "resident"):
                    return None, "non_kv_writes_require_fresh_cohort_generation"
                offset, size = access.get("offset_bytes"), access.get("byte_count")
                if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
                    return None, "non_kv_write_span_unavailable"
                if access.get("alias_of") in kv_ids or access.get("alias_generation") == 0:
                    return None, "non_kv_write_aliases_persistent_state"
                writes.append((owner, buffer_id, offset, size,
                    access.get("allocation_generation"), access.get("alias_of"),
                    access.get("alias_generation"), access.get("alias_offset_bytes", 0)))
                # An alias can shift an otherwise aligned logical span.
                # At most its two boundary pages need HOST slots, regardless
                # of that shift; charge both even when neither is partial.
                partials[owner] += 2
                rounded_pages[owner] += _ceil_pages(offset % PAGE_BYTES + size)
            for declaration in metadata.get("physical_allocations", ()):
                owner = str(declaration.get("physical_owner", ""))
                if owner not in owners or str(declaration.get("buffer_id", "")) in kv_ids:
                    continue
                declarations.append(tuple((key, declaration.get(key)) for key in (
                    "physical_owner", "buffer_id", "size_bytes", "generation", "alias_of",
                    "alias_generation", "alias_offset_bytes", "allocation_arena")))
        if fused != {str(layer.layer_id) for layer in layers}:
            return None, "terminal_flash_attention_not_admitted_for_every_layer"
        return {"signature": (tuple(writes), tuple(declarations)),
                "task_count": len(lowering.schedule.tasks), "writes": writes,
                "partial_slots": partials, "rounded_write_pages": rounded_pages}, None

    try:
        with _compilation_scope(scenario):
            start, reason = blueprint(first)
            if reason:
                return unavailable(reason)
            terminal, reason = blueprint(max(first, last))
            if reason:
                return unavailable(reason)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return unavailable("write_geometry_lowering_failed:" + type(error).__name__ + ":" + str(error))
    if start["signature"] != terminal["signature"]:
        return unavailable("non_kv_write_or_alias_geometry_changes_with_context")
    return {"qualified": True, "reason": None,
        "method": "analytic_admitted_flash_attention_fixed_non_kv_write_sequence",
        "first_context_tokens": first, "last_context_tokens": max(first, last),
        "blueprint_task_count": terminal["task_count"],
        "non_kv_write_span_count": len(terminal["writes"]),
        "non_kv_write_span_signature": terminal["signature"][0],
        "allocation_declaration_count": len(terminal["signature"][1]),
        "allocation_alias_signature": tuple(row for row in terminal["signature"][1]
            if dict(row).get("alias_of") is not None),
        "backend_kind": "standard_topology_analytic_dense",
        "interior_context_argument": "terminal_flash_sram_admission_is_monotone_and_all_other_writes_depend_only_on_single_token_and_fixed_layer_shapes",
        "partial_host_slots_by_owner": dict(terminal["partial_slots"]),
        "rounded_write_span_pages_by_owner": dict(terminal["rounded_write_pages"]),
        "physical_state_advanced": False, "physical_io_executed": False}


def _decode_program_budget(batches, rows, *, request_id, prompt_tokens,
                           current_tokens, final_tokens, remaining, write_geometry=None):
    """Separate compulsory K/V programs from cohort-local media observations.

    A monotonic K/V append programs exactly once whenever its initialized
    prefix completes a 4 KiB page: ``write`` flushes at that boundary even
    when the preceding fragments lived in HOST DRAM.  It need not program an
    incomplete final page.  We subtract only this *lower* bound from each
    cohort's own physical counter, never from commit-to-commit duration or
    from an observation that cannot be identified as an ordinary append.
    All cleanup/workspace writes remain in the observed residual.  The
    residual alone does not establish their future per-buffer page rounding:
    two 100-page buffers can each become 101 pages while ceil(200 * 1.001)
    is only 201.  Until the complete non-K/V write geometry is qualified, the
    de-duplicated result is diagnostic only unless planner IR proves fixed
    non-K/V writes. In that narrower domain their observed maximum is a
    constant; no aggregate context scaling or page rounding is needed.
    Existing pressure/RMW/GC/reuse guards stay intact.
    """
    observed = []
    counters_available = True
    deductions = []
    unavailable = None
    expected_context = prompt_tokens
    for batch in batches:
        metadata = batch.cost.metadata
        raw = metadata.get("storage_traffic", {}).get("pages_programmed")
        valid_counter = isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0
        observed.append(raw if valid_counter else 0)
        if not valid_counter:
            counters_available = False
            unavailable = unavailable or "cohort_program_counter_unavailable"
        items = getattr(batch, "items", ())
        if (metadata.get("physical_execution_scope") != "persistent_live_kernel"
                or metadata.get("cost_duration_semantics") != "live_cohort_span"):
            unavailable = unavailable or "program_counter_not_cohort_local"
        if len(items) != 1:
            unavailable = unavailable or "cohort_not_single_request_decode"
            deductions.append(0)
            continue
        item = items[0]
        append = item.token_count if item.kv_append_tokens is None else item.kv_append_tokens
        materialized = append if item.kv_materialized_tokens is None else item.kv_materialized_tokens
        if (item.request_id != request_id or item.phase != "decode"
                or item.token_count != 1 or append != 1 or materialized != 1
                or item.context_tokens != expected_context):
            unavailable = unavailable or "decode_kv_phase_not_contiguous"
            deductions.append(0)
            continue
        expected_context += 1
        compulsory = sum(((item.context_tokens + 1) * int(row["bytes_per_token"]) // PAGE_BYTES
                          - item.context_tokens * int(row["bytes_per_token"]) // PAGE_BYTES)
                         for row in rows)
        if not valid_counter or compulsory > raw:
            unavailable = unavailable or "cohort_program_count_below_kv_lower_bound"
        deductions.append(compulsory)
    if expected_context != current_tokens or len(batches) != current_tokens - prompt_tokens:
        unavailable = unavailable or "decode_kv_phase_not_contiguous"
    if not batches:
        counters_available = False
        unavailable = unavailable or "decode_program_observations_empty"
    # Aggregate traffic can contain other physical owners. Subtracting only
    # this owner's compulsory K/V pages leaves all other owners charged too,
    # so the residual remains conservative for multi-owner layouts.
    if unavailable:
        deductions = [0] * len(observed)
    residual = [total - kv for total, kv in zip(observed, deductions, strict=True)]
    observed_max = max(observed, default=0)
    residual_max = max(residual, default=0)
    scale = max(1.0, final_tokens / max(1, current_tokens))
    non_kv_per_step = ceil(residual_max * scale)
    future_kv = sum(int(row["completed_page_appends"]) for row in rows)
    legacy_per_step = ceil(observed_max * scale) + sum(
        _ceil_pages(int(row["bytes_per_token"])) for row in rows)
    candidate_budget = (remaining * legacy_per_step if unavailable
                        else remaining * non_kv_per_step + future_kv)
    lower_bound_verified = unavailable is None
    geometry_qualified = bool(write_geometry and write_geometry.get("qualified"))
    deduction_unavailable = unavailable or (None if geometry_qualified else "non_kv_write_geometry_unverified")
    qualified = deduction_unavailable is None
    # Fixed ordinary logical write/alias sequences execute the same staging
    # programs for each fresh cohort generation. Reads do not flush, release
    # discards partial HOST tails, and the separate guards exclude pressure,
    # GC, RMW and reuse. Thus every observed residual is an upper bound for
    # that sequence; using its maximum also retains extra prefix cleanup IO.
    program_budget = remaining * residual_max + future_kv if qualified else remaining * legacy_per_step
    return {"program_page_guard_budget": program_budget,
            "program_pages_per_decode_guard_budget": residual_max if qualified else legacy_per_step,
            "observed_max_program_pages_per_decode": observed_max,
            "observed_max_non_kv_program_pages_per_decode": residual_max,
            "observed_kv_completed_page_programs_deducted": sum(deductions) if qualified else 0,
            "observed_kv_completed_page_programs_verified": sum(deductions),
            "observed_program_pages_by_decode": tuple(observed),
            "observed_kv_program_lower_bound_by_decode": tuple(deductions),
            "kv_program_lower_bound_verified": lower_bound_verified,
            "kv_program_deduction_available": qualified,
            "kv_program_deduction_unavailable_reason": deduction_unavailable,
            "observed_program_counters_available": counters_available,
            "future_kv_completed_page_program_budget": future_kv,
            "context_scaled_non_kv_program_pages_per_decode": non_kv_per_step,
            "candidate_deduplicated_program_page_budget": candidate_budget,
            "candidate_deduplicated_budget_used_for_acceptance": False,
            "non_kv_write_geometry_qualified": geometry_qualified,
            "non_kv_write_geometry": write_geometry,
            "fixed_non_kv_program_pages_per_decode": residual_max if qualified else None,
            "legacy_program_page_guard_budget": remaining * legacy_per_step,
            "legacy_program_pages_per_decode_guard_budget": legacy_per_step}


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
                if allocation is not None and allocation.alias_of is not None:
                    reasons.append("hbf_kv_alias_requires_detailed_execution:" + buffer_id)
    devices = []
    decode_batches = tuple(batch for batch in runtime.batches if batch.kind == "decode")
    if detailed_decode_steps is not None and detailed_decode_steps != len(decode_batches):
        reasons.append("detailed_decode_step_count_does_not_match_runtime")
    write_geometry = (_fixed_decode_write_geometry(runtime, buffers, final)
                      if rows_by_owner and decode_batches else None)
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
        budget = _decode_program_budget(decode_batches, rows,
            request_id=state.spec.request_id, prompt_tokens=state.spec.prompt_tokens,
            current_tokens=current, final_tokens=final, remaining=remaining,
            write_geometry=write_geometry)
        program_budget = budget["program_page_guard_budget"]
        if remaining and not decode_batches:
            reasons.append("hbf_decode_media_budget_uncalibrated:" + owner)
        if remaining and not budget["observed_program_counters_available"]:
            reasons.append("hbf_decode_program_observations_unavailable:" + owner)
        if program_budget > available_pages:
            reasons.append("hbf_program_headroom_may_require_gc:" + owner)
        device = physical.runtimes.get(owner)
        core = None if device is None else device.core
        allocation_projection = _project_program_allocations(controller, program_budget, core)
        legacy_projection = (allocation_projection if program_budget == budget["legacy_program_page_guard_budget"]
            else _project_program_allocations(controller, budget["legacy_program_page_guard_budget"], core))
        if core is None:
            reasons.append("hbf_nand_program_frontier_unavailable:" + owner)
        if allocation_projection["reused_blocks_requiring_erase"]:
            reasons.append("hbf_free_block_reuse_may_require_erase:" + owner)
        tail_slots = sum(int(row["tail_page_slots_upper_bound"]) for row in rows)
        workspace_slots = (int(write_geometry.get("partial_host_slots_by_owner", {}).get(owner, 0))
                           if write_geometry and write_geometry.get("qualified") else 0)
        host_slots = controller.host_capacity_bytes // PAGE_BYTES - 1
        resident_slots = len(controller._resident)
        if resident_slots + tail_slots + workspace_slots > host_slots:
            reasons.append("hbf_host_tail_slots_may_require_pressure_flush:" + owner)
        devices.append({
            "physical_owner": owner, "bank_count": config.array_units,
            "pages_per_block": config.pages_per_block,
            "free_blocks": len(controller._free_blocks),
            "free_program_pages": available_pages,
            "free_program_pages_by_bank": bank_pages,
            "minimum_bank_program_pages": min(bank_pages, default=0),
            **budget,
            "program_allocation_guard": allocation_projection,
            # OCP page-0 writes erase even a previously unused block. This
            # is a serial service upper bound, not predicted stall time.
            "ordinary_erase_operations_guard_budget": allocation_projection["page_zero_programs"],
            "ordinary_erase_serial_time_guard_ns": allocation_projection["page_zero_programs"] * config.block_erase_ns,
            "ordinary_erase_time_upper_bound_ns": allocation_projection["page_zero_programs"] * config.block_erase_ns,
            "ordinary_erase_upper_bound_method": ("scalar_allocator_with_ir_qualified_fixed_non_kv_and_exact_kv_program_budget"
                if budget["kv_program_deduction_available"] else "legacy_observed_writes_and_per_step_kv_bound"),
            "legacy_ordinary_erase_operations_guard_budget": legacy_projection["page_zero_programs"],
            "legacy_ordinary_erase_serial_time_guard_ns": legacy_projection["page_zero_programs"] * config.block_erase_ns,
            "block_erase_ns": config.block_erase_ns,
            "kv_completed_page_appends": page_appends,
            "new_logical_kv_pages": new_pages,
            "logical_page_headroom_bytes": free_logical,
            "host_data_slots": host_slots, "observed_host_resident_slots": resident_slots,
            "additional_kv_tail_slots_upper_bound": tail_slots,
            "additional_fixed_workspace_partial_slots_upper_bound": workspace_slots,
            "observed_gc_cycles": int(stats.get("gc_cycles", 0)),
            "observed_pressure_flushes": int(stats.get("pressure_flushes", 0)),
            "observed_rmw_pages": int(stats.get("rmw_pages", 0)),
            "budget_assumption": ("ir_qualified_fixed_non_kv_write_sequence_plus_phase_exact_future_kv_page_completions"
                if budget["kv_program_deduction_available"] else "context_scaled_observed_writes_plus_per_step_kv_boundary_bound"),
            "future_erase_or_gc_events": "ordinary_page_zero_erase_bounded_gc_excluded",
            "future_physical_state": "not_advanced",
        })
    return FastTailProjection(remaining, current, final,
        max(0, final - current) * int(policy.logical_bytes_per_token),
        required_pages, policy.capacity_pages, tuple(buffers), tuple(devices),
        tuple(dict.fromkeys(reasons)))
