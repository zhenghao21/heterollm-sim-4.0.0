"""Read-only IR/preload evidence for the live-copy GC timing envelope."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

from .hbf_staging import HbfStagingController
from ._io_execution import PreparedStagingIO

PAGE = 4096

# The proof below depends on the production read/write dispatcher never
# explicitly flushing a partially initialized page. Keep authored hooks out
# of this narrower certificate without changing their physical execution.
_HOST_ONLY_METHODS = {name: getattr(HbfStagingController, name) for name in (
    "write", "read", "release", "flush", "_new_slot", "_flush_page", "_page_for_write")}
_STAGING_SUBMIT = PreparedStagingIO.submit


def temporary_read_media_bounds(writes, reads):
    """Bound temporary NAND reads from complete-page WRITE unions, IR only.

    For a fresh unaliased generation, a page whose complete byte range is
    never initialized cannot be programmed by ordinary writes. Its reads
    stay in HOST unless pressure or an explicit flush intervenes. This
    union deliberately ignores order: a later completion of a page keeps
    *all* earlier reads charged as possible NAND reads too. Unknown sources
    and aliases retain the original arbitrary-alignment upper bound.
    """
    unions, unsafe_owners = defaultdict(list), set()
    for owner, identity, offset, size, generation, alias, _, _ in writes:
        if (not identity or type(generation) is not int or generation <= 0
                or alias is not None or type(offset) is not int or offset < 0
                or type(size) is not int or size <= 0):
            unsafe_owners.add(owner)
        else:
            unions[(owner, identity, generation)].append((offset, offset + size))
    for key, ranges in unions.items():
        merged = []
        for lo, hi in sorted(ranges):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        unions[key] = merged
    result = {}
    for owner, identity, offset, size, generation, alias, _, _ in reads:
        row = result.setdefault(owner, {
            "qualified": True,
            "method": "fresh_temporary_full_page_write_union/v1",
            "rounded_read_pages_per_decode": 0,
            "potential_nand_read_pages_per_decode": 0,
            "proven_host_only_read_pages_per_decode": 0,
            "read_access_count": 0, "conservative_read_access_count": 0,
            "conservative_reasons": set(),
            "basis": "only_fully_initialized_pages_can_program_without_pressure_or_explicit_flush",
            "physical_state_advanced": False, "physical_io_executed": False})
        alignment = PAGE - 1 if alias is not None else offset % PAGE
        rounded = (alignment + size + PAGE - 1) // PAGE
        row["rounded_read_pages_per_decode"] += rounded
        row["read_access_count"] += 1
        ranges = unions.get((owner, identity, generation), ())
        reason = None
        if owner in unsafe_owners:
            reason = "owner_has_aliased_or_unknown_write_source"
        elif alias is not None:
            reason = "read_has_unknown_alias_alignment"
        elif type(generation) is not int or generation <= 0:
            reason = "read_generation_is_not_fresh"
        elif not any(lo <= offset and offset + size <= hi for lo, hi in ranges):
            reason = "read_bytes_not_covered_by_known_write_union"
        if reason:
            possible = rounded
            row["conservative_read_access_count"] += 1
            row["conservative_reasons"].add(reason)
        else:
            first, stop = offset // PAGE, (offset + size + PAGE - 1) // PAGE
            possible = sum(max(0, min(stop, hi // PAGE) - max(first, (lo + PAGE - 1) // PAGE))
                           for lo, hi in ranges)
        row["potential_nand_read_pages_per_decode"] += possible
        row["proven_host_only_read_pages_per_decode"] += rounded - possible
    for row in result.values():
        row["conservative_reasons"] = tuple(sorted(row["conservative_reasons"]))
    return result


def qualify_temporary_host_only_reads(controller, bound, *, host_slots_sufficient,
                                       read_write_dispatch_qualified):
    """Apply a WRITE-union reduction only in the no-partial-flush domain."""
    result = {**bound, "applied": False, "reason": None,
        "requirements": "fresh_fixed_ir_no_pressure_no_rmw_sufficient_host_slots_no_explicit_flush",
        "physical_state_advanced": False, "physical_io_executed": False}
    if not bound.get("qualified"):
        result["reason"] = "temporary_write_union_unavailable"
    elif (type(controller) is not HbfStagingController
            or any(name in controller.__dict__ or getattr(HbfStagingController, name) is not method
                   for name, method in _HOST_ONLY_METHODS.items())
            or PreparedStagingIO.submit is not _STAGING_SUBMIT
            or not read_write_dispatch_qualified):
        result["reason"] = "standard_nonflushing_read_write_dispatch_required"
    elif controller.stats.get("pressure_flushes", 0) or controller.stats.get("rmw_pages", 0):
        result["reason"] = "observed_pressure_or_rmw_can_program_partial_pages"
    elif not host_slots_sufficient:
        result["reason"] = "future_host_slots_can_require_partial_pressure_flush"
    else:
        result.update(applied=True, standard_dispatch_has_no_explicit_flush=True,
            future_host_slots_sufficient=True, observed_pressure_flushes=0, observed_rmw_pages=0)
    result["charged_nand_read_pages_per_decode"] = result.get(
        "potential_nand_read_pages_per_decode" if result["applied"] else "rounded_read_pages_per_decode")
    return result


def qualify_gc_read_basis(tasks, physical, owner, kv_rows, *, context_tokens,
                          token_boundary_completed):
    """Prove bounded unique KV reads and an actual immutable cache barrier.

    A barrier is one atomic physical READ task after every KV reader in the
    DAG. Its real preloaded pages must supply enough distinct tags in every
    bank to replace the complete read cache. Names and tensor byte sizes are
    not evidence. Repeated canonical immutable page accesses remain charged.
    """
    result = {"qualified": False, "reason": None,
        "method": "single_kv_prefix_read_and_actual_preload_bank_barrier/v1",
        "physical_state_advanced": False, "physical_io_executed": False}

    def reject(reason):
        result["reason"] = reason
        return result

    if not token_boundary_completed:
        return reject("gc_token_boundary_not_completed")
    if physical is None:
        return reject("gc_live_preload_state_unavailable")
    staging = physical.hbf_staging.get(owner)
    allocator = physical.allocators.get(owner)
    if staging is None or allocator is None:
        return reject("gc_live_preload_state_unavailable")
    controller = staging["controller"]
    config = controller.config
    banks, cache_slots = config.array_units, config.read_cache_pages_per_bank
    if cache_slots <= 0 or context_tokens <= 0:
        return reject("gc_read_cache_or_context_unqualified")
    expected = {str(row["buffer_id"]): int(row["bytes_per_token"]) for row in kv_rows}
    if not expected or len(expected) != len(kv_rows):
        return reject("gc_kv_read_inventory_unqualified")
    ids = {task.task_id: index for index, task in enumerate(tasks)}
    if len(ids) != len(tasks):
        return reject("gc_read_task_identity_unqualified")
    kv_counts, kv_tasks = Counter(), set()
    immutable = []
    by_task = defaultdict(list)
    task_has_kv_read = set()
    for index, task in enumerate(tasks):
        accesses = task.metadata.get("memory_accesses", ())
        for position, access in enumerate(accesses):
            if str(access.get("physical_owner", "")) != owner:
                continue
            operation = getattr(access.get("operation"), "value", access.get("operation"))
            identity = str(access.get("buffer_id", ""))
            if identity in expected:
                if operation != "read":
                    if (operation != "write" or access.get("allocation_generation") != 0
                            or access.get("alias_of") is not None
                            or access.get("offset_bytes") != (context_tokens - 1) * expected[identity]
                            or access.get("byte_count") != expected[identity]):
                        return reject("gc_kv_writes_are_not_monotonic_appends")
                    continue
                if (access.get("allocation_generation") != 0
                        or access.get("alias_of") is not None
                        or access.get("offset_bytes") != 0
                        or access.get("byte_count") != context_tokens * expected[identity]):
                    return reject("gc_kv_read_is_not_unique_unaliased_full_prefix")
                kv_counts[identity] += 1
                kv_tasks.add(index)
                task_has_kv_read.add(index)
                continue
            if (operation != "read" or access.get("allocation_generation") != 0
                    or access.get("allocation_arena") != "resident"):
                # A fresh unaliased temporary cannot populate a tag naming
                # a current KV page. Mixed GEMM tasks may read activations
                # or program logits around their immutable weight scan.
                # A live GC READ warms only its old source address; its new
                # current mapping is invalidated by the target PROGRAM.
                if (access.get("alias_of") is not None
                        or (operation == "read" and not access.get("allocation_generation"))):
                    return reject("gc_unknown_persistent_or_aliased_read_state")
                continue
            offset, size = access.get("offset_bytes"), access.get("byte_count")
            if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
                return reject("gc_immutable_read_span_unqualified")
            try:
                buffer_id, generation, canonical_offset = allocator.canonical_range(identity, offset, size, 0)
            except (ValueError, KeyError, TypeError):
                return reject("gc_immutable_read_canonical_identity_unavailable")
            preload = controller._preloads.get((buffer_id, generation))
            if preload is None or canonical_offset + size > preload.byte_count:
                return reject("gc_immutable_read_has_no_actual_preload")
            first, stop = canonical_offset // PAGE, (canonical_offset + size + PAGE - 1) // PAGE
            row = (index, position, buffer_id, generation, first, stop, preload)
            immutable.append(row)
            by_task[index].append(row)
    if set(kv_counts) != set(expected) or any(value != 1 for value in kv_counts.values()):
        return reject("gc_kv_read_multiplicity_unqualified")

    # Keep access multiplicity: embedding/head sharing or repeated aliases
    # can read a canonical page again even though their logical names differ.
    intervals = defaultdict(list)
    total_pages = 0
    for _, _, buffer_id, generation, first, stop, _ in immutable:
        intervals[(buffer_id, generation)].append((first, stop))
        total_pages += stop - first
    unique_pages = 0
    for ranges in intervals.values():
        end = -1
        for lo, hi in sorted(ranges):
            unique_pages += max(0, hi - max(lo, end))
            end = max(end, hi)
    repeated = total_pages - unique_pages

    ancestors = {}
    visiting = set()
    def ancestry(index):
        if index in ancestors:
            return ancestors[index]
        if index in visiting:
            raise ValueError("cyclic_gc_read_ir")
        visiting.add(index)
        found = 0
        for dependency in tasks[index].dependencies:
            if dependency not in ids:
                raise ValueError("external_gc_read_dependency")
            parent = ids[dependency]
            found |= 1 << parent
            found |= ancestry(parent)
        visiting.remove(index)
        ancestors[index] = found
        return found

    try:
        for index in range(len(tasks)):
            ancestry(index)
    except (ValueError, RecursionError):
        return reject("gc_barrier_dependency_proof_unavailable")
    kv_mask = sum(1 << index for index in kv_tasks)
    for candidate in reversed(tuple(by_task)):
        if candidate in task_has_kv_read:
            continue
        try:
            if ancestry(candidate) & kv_mask != kv_mask:
                continue
        except ValueError:
            return reject("gc_barrier_dependency_proof_unavailable")
        tags = [set() for _ in range(banks)]
        uncovered_banks = banks
        # Large valid barriers normally finish after k*banks pages. Bound
        # unsuccessful inspection too; an unproved bank remains a rejection.
        examined = 0
        limit = max(2 * banks * config.pages_per_block, cache_slots * banks)
        for _, _, _, _, first, stop, preload in by_task[candidate]:
            # Do not call preload.page_location: it can materialize the
            # descriptor's cached block_count property. A guard must leave
            # both Python and native ownership untouched, including caches.
            block_count = sum(len(span) for span in preload.block_ranges)
            for page_index in range(first, stop):
                group, within = divmod(page_index, controller._lanes * config.pages_per_block)
                width = min(controller._lanes, block_count - group * controller._lanes)
                page, lane = divmod(within, width)
                block = preload.block_at(group * controller._lanes + lane)
                bank = block % banks
                if len(tags[bank]) < cache_slots:
                    tags[bank].add((block, page))
                    if len(tags[bank]) == cache_slots:
                        uncovered_banks -= 1
                examined += 1
                if uncovered_banks == 0 or examined >= limit:
                    break
            if uncovered_banks == 0 or examined >= limit:
                break
        if uncovered_banks:
            continue
        result.update(qualified=True, reason=None,
            kv_reads_per_page_per_decode_upper=1,
            kv_buffer_count=len(expected), kv_reader_task_count=len(kv_tasks),
            immutable_read_access_count=len(immutable),
            immutable_read_pages_with_multiplicity=total_pages,
            immutable_distinct_canonical_read_pages=unique_pages,
            immutable_repeat_read_pages_per_decode_upper=repeated,
            initial_warm_read_pages_per_decode_upper=cache_slots * banks,
            barrier_task_id=tasks[candidate].task_id,
            barrier_distinct_tags_per_bank=tuple(len(items) for items in tags),
            barrier_all_kv_readers_are_dag_ancestors=True,
            barrier_preloads_are_canonical=True,
            barrier_mixed_temporary_accesses_cannot_warm_current_kv_tags=True,
            token_boundary_completed=True,
            gc_copy_cache_rule="source_tags_describe_old_pages_and_target_programs_invalidate_new_mapping_tags")
        return result
    return reject("gc_actual_immutable_bank_barrier_unavailable")
