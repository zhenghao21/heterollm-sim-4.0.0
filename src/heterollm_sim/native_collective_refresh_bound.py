"""Shared native REF epochs inside a closed ring collective.

This is a read-only interval proof. It neither executes a collective nor
changes a controller. Rows, directions, allocations and all physical work
remain in the original per-Task bounds. Only repeated charges for the same
absolute all-bank REF blackout are replaced by its union over a closed DAG.

Two separate bounds are used. Complete native active-service uppers bound
the elapsed horizon and hence the number of absolute REF epochs. They are
not used as a phase-error argument. The phase argument keeps each unique-
bank Task's PRE/ACT/row-to-column bridges, including every REF-free segment,
on the unchanged column/DQ/link graph. Only maintenance unavailable time
inside the same absolute [epoch, epoch+B] interval is counted once. No
counterfactual REF-disabled execution or cross-Task row-hit assumption is
used; repeated buffers continue paying their own complete row guard,
unless the separate real-allocator row-reuse proof below replaces later
rounds' guards with complete post-REF reopen and column residuals.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
import re
from types import SimpleNamespace

from .contracts import _PreparedExecutionTask
from .memory_types import DramConfig


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0.


def _remaining_phase(phase, refresh):
    # Undo a rounded positive sum conservatively. The subtraction can lose
    # more than one ULP of its much smaller result, so retain an ULP of the
    # original sum rather than comparing nearly equal rounded values.
    if not _finite(phase) or not _finite(refresh) or refresh > phase:
        raise ValueError("common_REF_Task_phase_decomposition_invalid")
    return math.nextafter(phase-refresh+math.ulp(phase), math.inf) if phase else 0.


def _epoch_source(config, owner, physical, native_runtime, *, source_proof=None):
    """Prove one absolute integer REF grid, including still-cold channels.

    clock_after uses ordinary addition when its delay is not already an
    exact half-CK multiple. An integer tREFI then stays on integer multiples
    below 2**52 ns. Lazy first-use initialization also chooses an integer
    multiple of tREFI; different initialization times cannot create phases.
    The current reference clock disables the separate idle-fold branch.
    """
    from .native_window_producer import (
        _hbm_phase_resource_ids, _native_refresh_source_qualification)

    if type(config) is not DramConfig:
        raise ValueError("common_REF_integer_epoch_recurrence_not_proven")
    interval, quantum = config.refresh_interval_ns, config.command_clock_ns/2
    if (config.control_mode != "jedec_hbm"
            or config.generation != "HBM4" or config.scheduler != "fcfs"
            or config.refresh_policy != "all_bank" or not _finite(interval)
            or not 0 < interval < 2**52 or interval != math.floor(interval)
            or interval == round(interval/quantum)*quantum
            or config.command_clock_ns*2 == math.floor(config.command_clock_ns*2)):
        raise ValueError("common_REF_integer_epoch_recurrence_not_proven")
    # A producer may pass its source reread from this same, read-only summary
    # call. Never persist this cache across physical execution or summaries.
    proof = (source_proof if source_proof is not None else
        _native_refresh_source_qualification(config, owner, physical, native_runtime))
    if proof.get("qualified") is not True:
        raise ValueError("common_REF_native_source_unqualified:"+proof.get("reason", "unknown"))
    if proof["current_physical_frontier_ns"] >= 2**52-interval:
        raise ValueError("common_REF_integer_clock_domain_exceeded")
    active = physical.runtimes.get(owner)
    actual = config if active is None else active.core.config
    prefixes = tuple(actual.metadata.get(key, f"{owner}:{default}") for key, default in (
        ("row_command_resource_prefix", "dram:row_command"),
        ("column_command_resource_prefix", "dram:column_command"),
        ("data_resource_prefix", "dram:data")))
    ports = set(_hbm_phase_resource_ids(prefixes, actual.stacks, actual.channels, actual.lane_count))
    canonical = physical.timeline.canonical_resource_id
    # A sibling native NAND descriptor can publish into its aliased host
    # port without exposing that port in the prepared Task's demands.
    if any(alias not in ports and canonical(alias) in ports
           for alias in physical.timeline.resource_aliases):
        raise ValueError("common_REF_foreign_resource_aliases_HBM_port")
    packed = vars(active.core).get("_hbm_numeric_state") if active is not None else None
    initialized = 0
    if packed is not None:
        for raw in packed.arrays[6][:, 1]:
            due = float(raw)
            if due < 0:
                continue
            multiple = round(due/interval)
            if not 0 <= due < 2**52 or due != multiple*interval:
                raise ValueError("common_REF_live_channel_has_other_epoch")
            initialized += 1
    return {"physical_owner": owner, "refresh_interval_ns": interval,
        "initialized_channel_count": initialized,
        "hbm_clock_domain_limit_ns": 2**52-interval,
        "cold_channels_use_same_absolute_epoch_grid": True,
        "native_REF_source": proof,
        "basis": "integer_tREFI_non_grid_clock_after_delay; exact_integer_addition_below_2**52; idle_fold_ineligible"}


def _epoch_reopen_bound(owners, configs, round0, rounds, extra, *, epochs, blocked,
                        read_active_upper, link_lower):
    """Charge post-REF reopens per common REF epoch, not on every round.

    Every bank is open on its only row or closed by REF. Outside REF a round
    is a hit: each read finishes within its conflict-free active upper,
    before the earliest same-round write can start after a whole link, so
    that write reuses the read's extent. A common epoch d affects at most
    three rounds: the one in progress at d (straddling IOs, a second write
    range), one that starts inside the blackout, whose reads cannot finish
    before tRFC and so delay the next round past the blackout, and the next
    read that reopens the first range. Each affected round pays one reopen
    of the read and write ranges on every owner; round 0 keeps its guards.
    """
    refresh = min(config.refresh_cycle_ns for config in configs.values())
    if not link_lower+refresh > blocked:
        return {"qualified": False, "reason": "REF_blackout_may_admit_two_round_starts"}
    if not read_active_upper < link_lower:
        return {"qualified": False, "reason": "read_may_outlive_neighbour_link_outside_REF"}
    hit = (max(row["hit_row_guard_ns"]["read"] for row in owners.values())
           +max(row["hit_row_guard_ns"]["write"] for row in owners.values()))
    reopen = (max(row["single_reopen_ns"]["read"] for row in owners.values())
              +max(row["single_reopen_ns"]["write"] for row in owners.values()))
    affected = 3*epochs
    value = math.nextafter(round0+(rounds-1)*hit+affected*reopen+extra, math.inf)
    return {"qualified": True, "row_phase_upper_ns": value, "REF_epochs": epochs,
        "affected_round_upper_per_epoch": 3, "hit_round_row_guard_ns": hit,
        "single_reopen_per_affected_round_ns": reopen, "round0_complete_guard_ns": round0,
        "read_active_upper_ns": read_active_upper, "link_service_lower_ns": link_lower,
        "basis": "closed_ring_rounds; common_absolute_REF_epochs; reads_reclaimed_before_writes_outside_REF; at_most_three_rounds_per_epoch"}


def _refined_epoch_count(owners, configs, waves, epochs, *, interval, blocked,
                         read_active_upper, link_lower):
    """Recount common REF epochs from hit rounds plus affected full rounds.

    The epoch count needs only the collective's REF-free elapsed horizon E:
    n epochs force (n-1)(tREFI-B) <= E. Outside the at most 1+3n affected
    rounds (round 0 and three per epoch, _epoch_reopen_bound) every IO hits a
    bank open on its only row: no ACT or tRCD. A same-round write also starts
    a whole link after S while its owner's only earlier IO, the read, ended
    by S+read_active; with link-read_active >= the REF-expired history upper
    that write has no live command history either. Affected rounds keep the
    complete wave. E(n) grows with n, so the largest n <= 1+floor(E(n)/(tREFI-B))
    bounds the true count; the old count remains an upper limit.
    """
    refresh = min(config.refresh_cycle_ns for config in configs.values())
    if not (link_lower+refresh > blocked and read_active_upper < link_lower):
        return {"qualified": False, "REF_epoch_count_upper": epochs,
                "reason": "epoch_round_lemmas_unproven"}
    if any("hit_active_upper_ns" not in row for row in owners.values()):
        return {"qualified": False, "REF_epoch_count_upper": epochs,
                "reason": "hit_active_upper_unavailable"}
    history = max(row["REF_expired_history_upper_ns"] for row in owners.values())
    hit_read = max(row["hit_active_upper_ns"]["read"] for row in owners.values())
    write_free = max(row["hit_active_upper_ns"]["write_history_free"] for row in owners.values())
    history_free = Fraction(link_lower) >= Fraction(read_active_upper)+Fraction(history)
    hit_write = write_free if history_free else math.nextafter(write_free+history, math.inf)
    base, extra = Fraction(0), Fraction(0)
    for wave in waves:
        hit = (Fraction(min(wave["read_wave_upper_ns"], hit_read))
               +Fraction(wave["link_wave_upper_ns"])
               +Fraction(min(wave["write_wave_upper_ns"], hit_write)))
        base += hit
        extra = max(extra, Fraction(wave["complete_serial_wave_upper_ns"])-hit)
    period = Fraction(interval)-Fraction(blocked)
    best = 1
    for count in range(1, epochs+1):
        horizon = base+min(len(waves), 1+3*count)*extra
        if count <= 1+int(horizon//period):
            best = count
    return {"qualified": True, "REF_epoch_count_upper": best, "previous_epoch_count_upper": epochs,
        "hit_read_active_upper_ns": hit_read, "hit_write_active_upper_ns": hit_write,
        "same_round_write_history_free": history_free,
        "hit_round_horizon_ns": float(base), "affected_round_extra_ns": float(extra),
        "basis": "REF_free_horizon_hit_rounds_plus_round0_and_three_affected_rounds_per_epoch; fixed_point_epoch_count; old_count_cap"}


def _row_reuse_bound(name, round_pairs, round_ios, configs, without_refresh, original_bound, *,
                     epochs, blocked, read_active_upper, link_lower,
                     rank_bound, owner_ranks, native_tasks, exact_ancestry, physical_runtime,
                     allocation_scope, active_waves=None, interval=None):
    """Replace later rounds' complete initial-row guards when rows are reused.

    Round 0 keeps every original guard. A write may first touch its second
    candidate range in any later round, once per owner; that touch also keeps
    the complete original guard. Waves and rank blocks both pay these extras.
    """
    from .native_collective_row_reuse import native_collective_row_reuse

    proof = native_collective_row_reuse(name, round_pairs, configs=configs,
        native_tasks=native_tasks, exact_ancestry=exact_ancestry,
        physical_runtime=physical_runtime, allocation_scope=allocation_scope)
    if not proof["qualified"]:
        return {**proof, "applied": False}
    owners = proof["owners"]
    guards = {task.task_id: phase for task, phase, _resources in without_refresh}
    reduced, waves, first_touch = {}, [], 0.
    for number, (reads, writes) in enumerate(round_ios):
        wave = []
        for tasks, operation in ((reads, "read"), (writes, "write")):
            values = []
            for task in tasks:
                original = guards[task.task_id]
                owner = task.metadata["memory_accesses"][0].get("physical_owner", task.metadata.get("physical_owner"))
                value = original if number == 0 else min(original, owners[owner]["reduced_row_guard_ns"][operation])
                reduced[task.task_id] = value
                values.append(value)
                if number and operation == "write" and len(owners[owner]["write_bases"]) > 1:
                    first_touch = max(first_touch, original-value)
            wave.append(max(values))
        waves.append(math.fsum(wave))
    # Each owner first touches its second write range at most once.
    second = sum(len(row["write_bases"]) > 1 for row in owners.values())
    extra = math.nextafter(min(len(round_ios)-1, second)*first_touch, math.inf) if second else 0.
    rows = [(task, reduced.get(task.task_id, phase), resources) for task, phase, resources in without_refresh]
    rank_value, rank_proof = rank_bound(rows, owner_ranks=owner_ranks)
    bound = math.nextafter(max(rank_value, math.fsum(waves))+extra, math.inf)
    refinement = (_refined_epoch_count(owners, configs, active_waves, epochs, interval=interval,
                      blocked=blocked, read_active_upper=read_active_upper, link_lower=link_lower)
                  if active_waves is not None and interval is not None
                  else {"qualified": False, "REF_epoch_count_upper": epochs, "reason": "waves_unavailable"})
    epochs = refinement["REF_epoch_count_upper"]
    epoch_bound = _epoch_reopen_bound(owners, configs, waves[0], len(round_ios), extra,
        epochs=epochs, blocked=blocked, read_active_upper=read_active_upper, link_lower=link_lower)
    if epoch_bound["qualified"]:
        bound = min(bound, epoch_bound["row_phase_upper_ns"])
    applied = bound < original_bound
    return {**proof, "applied": applied, "row_phase_upper_ns": bound if applied else original_bound,
        "reduced_wave_row_phase_ns": math.fsum(waves), "reduced_rank_row_phase_ns": rank_value,
        "second_write_range_first_touch_extra_ns": extra,
        "REF_epoch_count_refinement": refinement,
        "REF_epoch_reopen_bound": epoch_bound,
        "original_row_phase_upper_ns": original_bound, "reduced_rank_blocks": rank_proof}


def native_collective_refresh_bound(rows, bills, *, physical_runtime,
        native_runtime, owner_ranks, exact_ancestry, rank_bound, all_tasks=None,
        parsed_configs=None, source_proofs=None, allocation_scope=None, row_guard_overrides=None,
        io_active_overrides=None):
    """Return a complete replacement stage phase upper and its proof.

    Only complete native single-link ring all-reduces qualify. Every round
    has one read and one write per private HBM owner, disjoint links, and a
    full barrier. Its read/link/write wave upper is computed from the real
    descriptors; no latency is divided by the rank count. An added complete
    collective barrier preserves every external edge. Unknown graph, bus,
    source or clock semantics retain the caller's existing bound.
    """
    from .native_hbm_cold_bound import (native_hbm_initial_row_upper,
        native_hbm_refresh_service_upper, _native_refresh_source_limits)
    from .native_window_producer import _native_dag_precedes, _refresh_serial_terms
    from .physical_contract import parse_execution_physical_config
    from .data_motion import is_physical_task, physical_task_demands

    proof = {"qualified": False, "collectives": [], "rejections": [],
        "physical_state_advanced": False, "row_and_cache_phase_retained": True}
    if physical_runtime is None or native_runtime is None or exact_ancestry is None:
        return math.inf, {**proof, "reason": "common_REF_native_source_or_exact_DAG_unavailable"}
    rows = tuple(rows)
    original_tasks = tuple(all_tasks if all_tasks is not None else (row[0] for row in rows))
    if any(type(task) is not _PreparedExecutionTask for task in original_tasks):
        return math.inf, {**proof, "reason": "common_REF_unknown_prepared_Task"}
    canonical = physical_runtime.timeline.canonical_resource_id
    by_id = {task.task_id: (task, phase, resources) for task, phase, resources in rows}
    if len(by_id) != len(rows):
        return math.inf, {**proof, "reason": "common_REF_duplicate_Task_identity"}
    names = sorted({task.metadata["collective_name"] for task, _phase, _resources in rows
        if task.metadata.get("event_kind") == "collective_transfer"
        and task.metadata.get("algorithm") == "ring"
        and task.metadata.get("collective_kind") == "all_reduce"
        and isinstance(task.metadata.get("collective_name"), str)})
    groups = []
    config_cache = {} if parsed_configs is None else parsed_configs
    epoch_cache, service_cache = {}, {}
    # Ordinary resource demands can directly reserve a physical CA/DQ port
    # without a memory descriptor. Include aliases in both directions and
    # every live physical sibling. An ordered unknown port writer still
    # invalidates the native REF recurrence and must never be excused by DAG.
    physical_prefixes = set()
    for active in physical_runtime.runtimes.values():
        physical_prefixes.update(value for key, value in active.core.config.metadata.items()
            if isinstance(value, str) and (key.endswith("resource_prefix")
                or key in {"host_resource_id", "shared_command_resource"}))

    physical_aliases = {canonical(alias) for alias in physical_runtime.timeline.resource_aliases
        if any(alias == prefix or alias.startswith(prefix+":") for prefix in physical_prefixes)}

    def physical_port(identity):
        if any(identity == prefix or identity.startswith(prefix+":")
               for prefix in physical_prefixes):
            return True
        # The physical side itself may have been aliased to an ordinary
        # looking name; canonicalizing only the demand misses that case.
        return identity in physical_aliases

    direct_port_writers = {task.task_id for task in original_tasks
        if any(physical_port(canonical(d.resource_id)) for d in (
            physical_task_demands(task) if is_physical_task(task) else task.demands))}

    def before(left, right):
        return _native_dag_precedes(exact_ancestry, left, right)

    def owner_config(task):
        md = task.metadata
        accesses = tuple(md.get("memory_accesses", ()))
        if (len(accesses) != 1 or task.opaque_device_fence
                or any(md.get(key) for key in ("stateful_l2", "l2_execution",
                    "hbf_host_staging_by_owner", "hbf_host_staging"))):
            raise ValueError("common_REF_indirect_or_multiple_memory_descriptor")
        access = accesses[0]
        owner = access.get("physical_owner", md.get("physical_owner"))
        raw = md.get("physical_memory_configs", {}).get(owner, md.get("physical_memory_config"))
        if isinstance(raw, DramConfig):
            config = raw
        else:
            key = id(raw)
            if key not in config_cache:
                config_cache[key] = parse_execution_physical_config(raw)
            config = config_cache[key]
        if type(config) is not DramConfig or owner not in physical_runtime.runtimes:
            raise ValueError("common_REF_missing_live_HBM_owner")
        if not isinstance(owner, str) or len(owner_ranks.get(owner, ())) != 1:
            raise ValueError("common_REF_HBM_owner_not_private_to_rank")
        count, operation = access.get("byte_count"), access.get("operation")
        if type(count) is not int or count <= 0 or operation not in {"read", "write"}:
            raise ValueError("common_REF_invalid_positive_descriptor")
        placeholder = canonical(str(access.get("resource_id", owner)))
        if (physical_task_demands(task)
                or any(d.service_ns or canonical(d.resource_id) != placeholder for d in task.demands)):
            raise ValueError("common_REF_memory_Task_has_other_service")
        cache_key = (owner, id(config))
        if cache_key not in epoch_cache:
            source_proof = None if source_proofs is None else source_proofs.get(cache_key)
            epoch_cache[cache_key] = _epoch_source(config, owner, physical_runtime, native_runtime,
                                                 source_proof=source_proof)
        shape = (id(config), count, operation)
        if shape not in service_cache:
            # With at most one native bank sweep, each column touches a
            # different bank. REF may add a PRE/ACT bridge to the common
            # column/DQ graph, but cannot create a new same-bank successor.
            # Keep the already-proved row bridge on every REF-free segment;
            # larger multi-sweep requests retain their original envelope.
            row = native_hbm_initial_row_upper(config, count, operation,
                independent_channels=True, native_refresh_reachable=True)
            if not row.get("unique_bank_initial_row_proof", {}).get("qualified"):
                raise ValueError("common_REF_unique_bank_common_bulk_graph_unproven")
            bounds = native_hbm_refresh_service_upper(config, count, operation,
                independent_channels=True, native_refresh_reachable=True)
            if bounds.get("qualified") is not True:
                raise ValueError("common_REF_full_active_service_unavailable")
            interval, blocked, active = _refresh_serial_terms(config, count, operation,
                independent_channels=True, allow_bounded_clock=True, native_refresh_reachable=True)
            limits = _native_refresh_source_limits(config)
            clock, pcs = config.command_clock_ns, config.pseudo_channels_per_channel
            # At due=d, the native per-burst invariant places every old PRE
            # recovery by d+R, last ACT/PRE by d, and row CA by d+2CK.
            # Serial PRE/tRP/REF for both PCs plus alignment covers the last
            # tRFC end. _refresh_until runs before a new command crosses d,
            # so its next epoch preserves these inequalities inductively.
            # This is an absolute [d,d+B] blackout, not merely an IO stall.
            absolute_tail = (limits["source_precharge_tail_upper_ns"]
                +pcs*config.precharge_to_activate_ns
                +(2*pcs*config.precharge_command_cycles+pcs+2)*clock
                +config.refresh_cycle_ns)
            if blocked < max(absolute_tail, limits["source_final_tail_upper_ns"],
                             limits["source_CA_tail_upper_ns"]):
                raise ValueError("common_REF_absolute_epoch_blackout_unproven")
            service_cache[shape] = (interval, blocked, active, bounds, row)
        interval, blocked, active, bounds, row = service_cache[shape]
        # A layer row-reuse proof bounds this IO's conflict-free active work.
        active = min(active, (io_active_overrides or {}).get(task.task_id, active))
        # A layer row-reuse proof may already have lowered this Task's guard.
        required = (row_guard_overrides or {}).get(task.task_id, row["initial_state_upper_ns"])
        if _remaining_phase(by_id[task.task_id][1], bills[task.task_id]["refresh_ns"]) < min(
                required, row["initial_state_upper_ns"]):
            raise ValueError("common_REF_complete_row_phase_not_retained")
        return owner, count, operation, interval, blocked, active, bounds, epoch_cache[cache_key], config

    for name in names:
        try:
            if direct_port_writers:
                raise ValueError("common_REF_unknown_direct_physical_port_writer")
            pattern = re.compile(re.escape(name)+r"\.round(\d+)\.(?:tx(\d+)\.(.+)|(barrier))$")
            rounds = defaultdict(lambda: {"transfers": defaultdict(list), "barrier": []})
            member_ids = set()
            for task, phase, resources in rows:
                match = pattern.search(task.task_id)
                if match is None:
                    continue
                member_ids.add(task.task_id)
                number, transfer, suffix, barrier = match.groups()
                if barrier:
                    rounds[int(number)]["barrier"].append(task)
                else:
                    rounds[int(number)]["transfers"][int(transfer)].append(task)
            if not rounds or sorted(rounds) != list(range(len(rounds))):
                raise ValueError("common_REF_incomplete_round_inventory")
            count = len(rounds[0]["transfers"])
            if count < 2 or len(rounds) != 2*(count-1):
                raise ValueError("common_REF_ring_round_count_not_native")
            prior, roots, last, mapping = None, (), None, None
            owners, link_resources = set(), set()
            wave_proofs, epoch_proofs, clock_limits = [], {}, []
            blocked_values, intervals, active_waves, row_waves = [], set(), [], []
            owner_configs, round_pairs, round_ios = {}, [], []
            max_read_active, min_link = 0., math.inf
            for number in range(len(rounds)):
                data = rounds[number]
                if sorted(data["transfers"]) != list(range(count)) or len(data["barrier"]) != 1:
                    raise ValueError("common_REF_partial_transfer_round")
                barrier = data["barrier"][0]
                if (barrier.metadata.get("event_kind") != "collective_barrier"
                        or barrier.demands or barrier.opaque_device_fence
                        or barrier.metadata.get("memory_accesses")):
                    raise ValueError("common_REF_barrier_has_hidden_work")
                reads, writes, links, current_mapping = [], [], [], {}
                pair_reads, pair_writes = {}, {}
                read_active, write_active, link_active, write_ids = [], [], [], set()
                used_links = set()
                for transfer in range(count):
                    tasks = data["transfers"][transfer]
                    read = [task for task in tasks if task.metadata.get("memory_direction") == "read"]
                    write = [task for task in tasks if task.metadata.get("memory_direction") == "write"]
                    link = [task for task in tasks if not task.metadata.get("memory_accesses")]
                    if len(tasks) != 3 or len(read) != 1 or len(write) != 1 or len(link) != 1:
                        raise ValueError("common_REF_transfer_not_single_link_read_write")
                    read, write, link = read[0], write[0], link[0]
                    if (any(task.metadata.get("event_kind") != "collective_transfer" for task in tasks)
                            or any(task.metadata.get("collective_name") != name
                                or task.metadata.get("algorithm") != "ring"
                                or task.metadata.get("collective_kind") != "all_reduce" for task in (read, write))
                            or link.dependencies != (read.task_id,) or write.dependencies != (link.task_id,)
                            or link.opaque_device_fence or not link.demands
                            or any(not d.resource_id.startswith("link.") or not _finite(d.service_ns)
                                   for d in link.demands)):
                        raise ValueError("common_REF_transfer_native_edges_or_link_unproven")
                    if prior is not None and read.dependencies != (prior.task_id,):
                        raise ValueError("common_REF_round_has_external_wait")
                    if reads and read.dependencies != reads[0].dependencies:
                        raise ValueError("common_REF_round_roots_have_different_release")
                    rd, wr = owner_config(read), owner_config(write)
                    if rd[0] == wr[0] or rd[1] != wr[1] or rd[2] != "read" or wr[2] != "write":
                        raise ValueError("common_REF_transfer_descriptor_pair_invalid")
                    if rd[0] in current_mapping or wr[0] in current_mapping.values():
                        raise ValueError("common_REF_round_owner_has_multiple_IOs")
                    current_mapping[rd[0]] = wr[0]
                    pair_reads[rd[0]], pair_writes[wr[0]] = read, write
                    owner_configs[rd[0]], owner_configs[wr[0]] = rd[8], wr[8]
                    # Even zero-service demands read their resource calendar
                    # in the real scheduler; they can add an external wait.
                    ids = {canonical(d.resource_id) for d in link.demands}
                    if ids & used_links:
                        raise ValueError("common_REF_links_share_a_resource")
                    used_links.update(ids)
                    link_resources.update(ids)
                    for info in (rd, wr):
                        owners.add(info[0]); intervals.add(info[3]); blocked_values.append(info[4])
                        clock_limits.append(info[6]["hbm_clock_domain_limit_ns"])
                        epoch_proofs[info[0]] = info[7]
                    read_active.append(rd[5]); write_active.append(wr[5])
                    link_active.append(max(d.service_ns for d in link.demands))
                    max_read_active = max(max_read_active, rd[5])
                    min_link = min(min_link, link_active[-1])
                    reads.append(read); writes.append(write); links.append(link); write_ids.add(write.task_id)
                if set(current_mapping) != set(current_mapping.values()):
                    raise ValueError("common_REF_round_owner_sets_differ")
                round_pairs.append({owner: (pair_reads[owner], pair_writes[owner]) for owner in pair_reads})
                round_ios.append((tuple(reads), tuple(writes)))
                if mapping is None:
                    mapping = current_mapping
                    visited, owner = set(), next(iter(mapping))
                    while owner not in visited:
                        visited.add(owner); owner = mapping[owner]
                    if len(visited) != count:
                        raise ValueError("common_REF_transfers_are_not_one_ring")
                elif current_mapping != mapping:
                    raise ValueError("common_REF_ring_changes_between_rounds")
                if set(barrier.dependencies) != write_ids or len(barrier.dependencies) != count:
                    raise ValueError("common_REF_round_barrier_does_not_join_all_writes")
                if number == 0:
                    roots = tuple(read.task_id for read in reads)
                    if any(parent in member_ids for parent in reads[0].dependencies):
                        raise ValueError("common_REF_first_round_has_internal_incoming_edge")
                # Both native dispatch paths select the earliest ready Task.
                # Pure physical reads have no remaining preview demands, so
                # all read roots have the same key S. Every positive read's
                # link is ready strictly after S: all reads are therefore
                # submitted before any link/write, in the actual FCFS order.
                # Each owner then has only one write and each link is private.
                # This covers the real resource order; it does not assume
                # that rearranging a stateful graph with new barriers is safe.
                phase_upper = math.nextafter(math.fsum((max(read_active), max(link_active), max(write_active))), math.inf)
                active_waves.append(phase_upper)
                row_wave = math.fsum(max(_remaining_phase(by_id[task.task_id][1],
                    bills[task.task_id]["refresh_ns"]) for task in wave) for wave in (reads, writes))
                row_waves.append(math.nextafter(row_wave, math.inf))
                wave_proofs.append({"round": number, "read_wave_upper_ns": max(read_active),
                    "link_wave_upper_ns": max(link_active), "write_wave_upper_ns": max(write_active),
                    "complete_serial_wave_upper_ns": phase_upper,
                    "complete_row_guard_wave_upper_ns": row_waves[-1],
                    "native_dispatch_submits_all_same_release_reads_before_any_link": True})
                prior = last = barrier
            if len(intervals) != 1:
                raise ValueError("common_REF_owners_have_different_intervals")
            # Direct and aliased external users must be outside the entire
            # original graph interval. Merely adding an artificial barrier
            # cannot erase an actual interleaving access or foreign wait.
            for task in original_tasks:
                if task.task_id in member_ids:
                    continue
                md = task.metadata
                if any(md.get(key) for key in ("stateful_l2", "l2_execution", "hbf_host_staging_by_owner", "hbf_host_staging")):
                    raise ValueError("common_REF_external_indirect_writer_unbounded")
                # Allocation-only declarations change the owner's free list.
                touched = {access.get("physical_owner", md.get("physical_owner"))
                           for key in ("memory_accesses", "physical_allocations")
                           for access in md.get(key, ())}
                if any(owner not in physical_runtime.runtimes for owner in touched):
                    raise ValueError("common_REF_external_physical_owner_unavailable")
                resources = {canonical(d.resource_id) for d in task.demands}
                outgoing = any(parent in member_ids for parent in task.dependencies)
                if touched & owners or resources & link_resources or outgoing:
                    if not (all(before(task.task_id, root) for root in roots)
                            or before(last.task_id, task.task_id)):
                        raise ValueError("common_REF_original_subgraph_not_closed:"+task.task_id)
            # The native source qualification rejects shared CA/DQ ports.
            # Also exclude a declared link which aliases any physical bus.
            if any(physical_port(identity) for identity in link_resources):
                raise ValueError("common_REF_link_shares_physical_port")
            interval = next(iter(intervals))
            blocked = math.nextafter(max(blocked_values), math.inf)
            active_upper = math.nextafter(math.fsum(active_waves), math.inf)
            epochs = 1+int(Fraction(active_upper)//(Fraction(interval)-Fraction(blocked)))
            shared_refresh = math.nextafter(epochs*blocked, math.inf)
            original = [row for row in rows if row[0].task_id in member_ids]
            without_refresh = []
            for task, phase, resources in original:
                refresh = bills[task.task_id]["refresh_ns"]
                without_refresh.append((task, _remaining_phase(phase, refresh), resources))
            row_bound, row_proof = rank_bound(without_refresh, owner_ranks=owner_ranks)
            row_bound = max(row_bound, math.nextafter(math.fsum(row_waves), math.inf))
            reuse = _row_reuse_bound(name, round_pairs, round_ios, owner_configs, without_refresh,
                row_bound, epochs=epochs, blocked=blocked, read_active_upper=max_read_active,
                link_lower=min_link, rank_bound=rank_bound, owner_ranks=owner_ranks, native_tasks=original_tasks,
                exact_ancestry=exact_ancestry, physical_runtime=physical_runtime,
                allocation_scope=allocation_scope, active_waves=wave_proofs, interval=interval)
            if reuse.get("applied"):
                row_bound = reuse["row_phase_upper_ns"]
            refinement = reuse.get("REF_epoch_count_refinement") or {}
            if refinement.get("qualified") and refinement["REF_epoch_count_upper"] < epochs:
                # Hit rounds shorten the REF-free horizon; the blackout union
                # follows the refined epoch count.
                epochs = refinement["REF_epoch_count_upper"]
                shared_refresh = math.nextafter(epochs*blocked, math.inf)
            old_bound, _ = rank_bound(original, owner_ranks=owner_ranks)
            combined = math.nextafter(row_bound+shared_refresh, math.inf)
            if not combined < old_bound:
                raise ValueError("common_REF_closed_bound_not_tighter")
            incoming = tuple(dict.fromkeys(parent for task, _p, _r in original
                for parent in task.dependencies if parent not in member_ids))
            entry = {"collective_name": name, "task_ids": tuple(row[0].task_id for row in original),
                "incoming_dependencies": incoming, "terminal_task_id": last.task_id,
                "physical_owner_count": len(owners), "round_count": len(rounds),
                "old_collective_phase_upper_ns": old_bound,
                "row_and_cache_phase_upper_ns": row_bound,
                "common_REF_phase_upper_ns": shared_refresh,
                "collective_phase_upper_ns": combined,
                "complete_native_active_upper_ns": active_upper,
                "refresh_interval_ns": interval, "common_REF_blackout_upper_ns": blocked,
                "common_REF_epoch_count_upper": epochs,
                "hbm_clock_domain_limit_ns": min(*clock_limits,
                    *(item["hbm_clock_domain_limit_ns"] for item in epoch_proofs.values())),
                "owner_epoch_proofs": list(epoch_proofs.values()), "native_round_waves": wave_proofs,
                "row_phase_composition": row_proof,
                "closed_collective_row_reuse": reuse,
                "source_and_target_physical_state_unchanged": True,
                "basis": "closed_native_ring_round_DAG; disjoint_read_link_write_waves; common_integer_REF_epoch_blackout_union; all_row_cache_guards_retained"}
            groups.append(entry)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            proof["rejections"].append({"collective_name": name, "reason": str(error)})
    if not groups:
        return math.inf, proof
    replacements, new_rows = {}, {}
    for index, group in enumerate(groups):
        identity = "@native-common-REF-collective:"+str(index)+":"+group["collective_name"]
        for task_id in group["task_ids"]:
            if task_id in replacements:
                return math.inf, {**proof, "reason": "common_REF_overlapping_collective_groups"}
            replacements[task_id] = identity
        resources = set().union(*(by_id[task_id][2] for task_id in group["task_ids"]))
        node = SimpleNamespace(task_id=identity, dependencies=group["incoming_dependencies"], metadata={})
        new_rows[identity] = (node, group["collective_phase_upper_ns"], resources)
    # Rewrite incoming edges for collective nodes too. A second collective
    # can directly consume the first one's terminal barrier, now contracted.
    for node, _phase, _resources in new_rows.values():
        node.dependencies = tuple(dict.fromkeys(replacements.get(parent, parent)
                                                for parent in node.dependencies))
    contracted, emitted = [], set()
    for task, phase, resources in rows:
        identity = replacements.get(task.task_id)
        if identity:
            if identity not in emitted:
                contracted.append(new_rows[identity]); emitted.add(identity)
            continue
        node = SimpleNamespace(task_id=task.task_id,
            dependencies=tuple(dict.fromkeys(replacements.get(parent, parent) for parent in task.dependencies)),
            metadata=task.metadata)
        contracted.append((node, phase, resources))
    # Closed original intervals prevent external descendants from preceding
    # their new node. Check explicitly rather than trusting name ordering.
    positions = {task.task_id: index for index, (task, _p, _r) in enumerate(contracted)}
    if any(positions.get(parent, -1) >= index
           for index, (task, _p, _r) in enumerate(contracted) for parent in task.dependencies):
        return math.inf, {**proof, "reason": "common_REF_contracted_graph_not_topological"}
    bound, rank_proof = rank_bound(contracted, owner_ranks=owner_ranks)
    proof.update(qualified=True, collectives=groups, contracted_rank_blocks=rank_proof,
        hbm_clock_domain_limit_ns=min(group["hbm_clock_domain_limit_ns"] for group in groups))
    return bound, proof
