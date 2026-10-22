"""Row reuse inside a closed native ring collective, from real addresses.

This is a read-only budget proof. It replays the original first-fit owner
allocator on an address-only clone; no Core, controller, event or Token is
touched. Every guard that is not proved stays in the caller's bound.

Native lifetimes make the proof possible. A transient buffer is returned
when a dispatch starts no earlier than the completion of its last user, and
first-fit then reuses the lowest fitting extent. A ring round has one read
and one write per private HBM owner, joined by a full barrier. The next
round therefore starts from the same free list ``F0``: every read lands on
``B1 = firstfit(F0)``. A write lands on ``B1`` when its own owner's read has
already been reclaimed, otherwise on ``B2 = firstfit(F0 - B1)``; which one
happens depends on timing and is not assumed.

If every physical bank touched by the candidate ranges sees one row only,
the initial row image can differ only before the first touch of each range,
and after a REF closes every bank. The first touches keep the complete
original guard. Every later IO pays two complete post-REF reopen segments,
plus the column, direction, group/SID and write geometry differences. A
full all-bank REF lasts tRFC beyond the last PRE, longer than every native
command history term, so a reopened row has no old history to inherit.
Its blackout time remains in the separate common REF charge.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re

from .memory_types import DramConfig


_TRANSIENT_SCOPE_SCHEMA = "native-cohort-transient-allocation-scope/v1"


def _rows(task):
    """Allocation-bearing descriptor rows, as the event kernel indexes them."""
    from .data_motion import _memory_accesses, _physical_allocations

    md = task.metadata
    rows = [{**row, "allocation_generation": row.get("generation", 0), "_declaration": True}
            for row in _physical_allocations(md)]
    rows.extend(_memory_accesses(md) or ())
    contract = md.get("stateful_l2")
    if isinstance(contract, dict):
        rows.extend(item for item in contract.get("accesses", ()) if isinstance(item, dict))
    return rows


def _key(task, row):
    md = task.metadata
    contract = md.get("stateful_l2")
    owner = row.get("physical_owner") or md.get("physical_owner") or (
        contract.get("memory_resource") if isinstance(contract, dict) else None)
    buffer_id = row.get("buffer_id")
    if not owner or not buffer_id:
        return None
    buffer_id = str(buffer_id)
    if buffer_id.startswith("@tasklocal"):
        buffer_id = task.task_id+buffer_id[len("@tasklocal"):]
    return str(owner), buffer_id, int(row.get("allocation_generation", 0) or 0)


def native_allocation_scope(stages, composition, boundary, names, source_kernel):
    """Cohort-wide lifetime facts for every physical owner, read only.

    A layer may begin from the source allocator image only if every buffer
    any cohort Task touches on that owner is a native transient released by
    the kernel after its last user, and all of a buffer's users belong to one
    serially composed layer or to one separated non-layer prefix/suffix.
    Persistent, external, aliased, explicit-address, resident and pre-existing
    buffers would survive into a later layer and invalidate that image.
    """
    from .contracts import _PreparedExecutionStage

    scope = {"schema": _TRANSIENT_SCOPE_SCHEMA, "owners": {}, "physical_state_advanced": False}
    if composition.get("qualified") is not True:
        scope["reason"] = "native_serial_layer_composition_not_proven"
        return scope
    groups = {}
    for kind, ids in boundary.items():
        for identity in ids:
            groups[names.get(identity, identity)] = kind
    separated = {kind for kind in ("before_first_layer", "after_last_layer")
                 if kind not in composition.get("nonserial_non_layer_groups", ())}
    users, layers, violations = defaultdict(set), defaultdict(set), defaultdict(set)
    sizes = defaultdict(int)
    for stage in stages:
        if type(stage) is not _PreparedExecutionStage:
            scope["reason"] = "non_native_prepared_stage"
            return scope
        for task in stage.execution_tasks:
            md = task.metadata
            try:
                rows = _rows(task)
            except ValueError:
                violations["*"].add("invalid_physical_descriptor_rows")
                continue
            external = {tuple(key) for key in md.get("physical_external_allocations", ())}
            persistent = {buffer for buffers in (md.get("persistent_request_buffers") or {}).values()
                          for buffer in buffers}
            layer = md.get("layer_id")
            group = layer if layer is not None else ("@non-layer:"+groups[task.task_id]
                if groups.get(task.task_id) in separated else "@unseparated-non-layer")
            for row in rows:
                key = _key(task, row)
                if key is None:
                    continue
                owner = key[0]
                if md.get("stateful_l2") is not None or md.get("l2_execution"):
                    violations[owner].add("stateful_cache_descriptor")
                if (key in external or key[1] in persistent or row.get("alias_of")
                        or row.get("allocation_arena") == "resident" or key[2] == 0
                        or (row.get("_declaration") and row.get("address") is not None)
                        or (row.get("address") is not None
                            and str(row.get("address_source", "")).startswith("explicit"))):
                    violations[owner].add("persistent_aliased_explicit_or_resident_buffer")
                users[key].add(task.task_id)
                layers[key].add(group)
                if row.get("_declaration") and type(row.get("size_bytes")) is int:
                    sizes[key] = max(sizes[key], row["size_bytes"])
    pending = {key[0] for key in getattr(source_kernel, "_pending_physical_releases", {})}
    allocators = source_kernel.physical_runtime.allocators
    counts = defaultdict(dict)
    cumulative, non_layer = defaultdict(lambda: defaultdict(int)), set()
    for key, task_ids in users.items():
        group = next(iter(layers[key]))
        allocator = allocators.get(key[0])
        alignment = allocator.alignment_bytes if allocator is not None else 1
        # First-fit can add at most alignment-1 padding before each extent.
        cumulative[key[0]][group] += sizes[key]+alignment-1
        if group.startswith("@"):
            non_layer.add(key[0])
        owner = key[0]
        group = layers[key]
        if len(group) != 1 or "@unseparated-non-layer" in group:
            violations[owner].add("buffer_users_cross_layers_or_unseparated_non_layer")
        allocator = allocators.get(owner)
        if allocator is not None and allocator.get_allocation(key[1], key[2]) is not None:
            violations[owner].add("cohort_buffer_already_registered_in_source")
        counts[owner][key[1:]] = len(task_ids)
    for owner in set(counts) | set(violations) - {"*"}:
        reasons = sorted(violations.get(owner, set()) | violations.get("*", set()))
        if owner in pending:
            reasons.append("source_has_pending_physical_release")
        allocator = allocators.get(owner)
        highest = max((row.base_address+row.size_bytes for row in allocator.allocations()
                       if row.alias_of is None), default=0) if allocator is not None else None
        scope["owners"][owner] = {"qualified": not reasons, "reasons": reasons,
            "user_counts": counts.get(owner, {}),
            "max_group_cumulative_extent_bytes": max(cumulative[owner].values(), default=0),
            "non_layer_users": owner in non_layer, "source_highest_end_byte": highest}
    return scope


def _topological(tasks, exact_ancestry):
    # A strict predecessor has a strict subset of ancestors, hence a smaller
    # popcount. Ties are unordered pairs, which keep their enumeration order.
    indexes, ancestors = exact_ancestry
    return sorted(tasks, key=lambda task: (bin(ancestors[indexes[task.task_id]]).count("1"),
                                           indexes[task.task_id]))


def _ring_blocks(owner, tasks, before):
    """Owner-local read/write pairs of every native ring all-reduce.

    Each round must hold one read and one write on this owner, and both of
    a round's IOs must precede both IOs of the next round. That ordering is
    what returns every round to the same free list; it is checked on the
    exact DAG, not inferred from the collective's name.
    """
    found = defaultdict(lambda: defaultdict(lambda: {"read": [], "write": []}))
    for task in tasks:
        md = task.metadata
        if md.get("event_kind") != "collective_transfer" or not md.get("memory_accesses"):
            continue
        accesses = md["memory_accesses"]
        if len(accesses) != 1 or accesses[0].get("physical_owner", md.get("physical_owner")) != owner:
            continue
        match = re.search(r"\.round(\d+)\.tx\d+\.", task.task_id)
        direction = md.get("memory_direction")
        if (match is None or direction not in {"read", "write"} or md.get("algorithm") != "ring"
                or md.get("collective_kind") != "all_reduce" or not isinstance(md.get("collective_name"), str)):
            raise ValueError("collective_row_reuse_unknown_collective_IO_on_owner")
        found[md["collective_name"]][int(match.group(1))][direction].append(task)
    blocks = {}
    for name, rounds in found.items():
        if sorted(rounds) != list(range(len(rounds))):
            raise ValueError("collective_row_reuse_incomplete_owner_rounds")
        pairs = []
        for number in range(len(rounds)):
            data = rounds[number]
            if len(data["read"]) != 1 or len(data["write"]) != 1:
                raise ValueError("collective_row_reuse_round_owner_IO_not_one_read_one_write")
            pairs.append((data["read"][0], data["write"][0]))
        for previous, current in zip(pairs, pairs[1:]):
            if not all(before(left.task_id, right.task_id) for left in previous for right in current):
                raise ValueError("collective_row_reuse_rounds_not_serial_on_owner")
        blocks[name] = pairs
    return blocks


def _owner_layouts(owner, tasks, exact_ancestry, source_allocator, user_counts):
    """Replay one owner allocator through the summarized native subgraph.

    Returns, for every ring collective on this owner, its read base and its
    possible write bases, and every owner IO with its candidate address
    ranges. Any lifetime the DAG does not order is rejected.
    """
    from .memory_allocator import PhysicalAddressAllocator
    from .native_window_producer import _native_dag_precedes

    def before(left, right):
        return left != right and _native_dag_precedes(exact_ancestry, left, right)

    keys_by_task, users = {}, defaultdict(set)
    for task in tasks:
        declared, used = [], set()
        for row in _rows(task):
            key = _key(task, row)
            if key is None or key[0] != owner:
                continue
            used.add(key[1:])
            if row.get("_declaration"):
                declared.append((key[1:], row))
        if used:
            keys_by_task[task.task_id] = (task, declared, used)
            for key in used:
                users[key].add(task.task_id)
    for key, ids in users.items():
        if user_counts.get(key) != len(ids):
            raise ValueError("collective_row_reuse_buffer_user_outside_summary")
        # Exactly one earliest user registers the extent.
        firsts = [task for task in ids if not any(before(other, task) for other in ids)]
        if len(firsts) != 1 or not all(before(firsts[0], other) for other in ids - {firsts[0]}):
            raise ValueError("collective_row_reuse_first_registration_unordered")
        if not any(declared == key for declared, _row in keys_by_task[firsts[0]][1]):
            raise ValueError("collective_row_reuse_first_user_does_not_declare_extent")
    blocks = _ring_blocks(owner, tasks, before)
    block_of = {}
    for name, rounds in blocks.items():
        for read, write in rounds:
            for task in (read, write):
                block_of[task.task_id] = name
                entry = keys_by_task.get(task.task_id)
                if (entry is None or len(entry[2]) != 1 or len(entry[1]) != 1
                        or users[next(iter(entry[2]))] != {task.task_id}):
                    raise ValueError("collective_row_reuse_round_buffer_not_single_user")
    clone = PhysicalAddressAllocator.from_snapshot(source_allocator.snapshot())
    live = set()

    def reclaim(reference):
        # The kernel reclaims at every dispatch. Only the state seen by an
        # allocation matters; an ambiguous lifetime there is rejected.
        for key in tuple(live):
            ids = users[key]
            if all(before(identity, reference) for identity in ids):
                clone.release(*key)
                live.discard(key)
            elif not all(identity == reference or before(identity, reference)
                         or before(reference, identity) for identity in ids):
                raise ValueError("collective_row_reuse_unordered_buffer_lifetime:"+key[0]+"@"+reference)

    def allocate(declared, target=None):
        target = clone if target is None else target
        for (buffer_id, generation), row in sorted(declared, key=lambda item: 1 if item[1].get("alias_of") else 0):
            target.allocate(buffer_id, row["size_bytes"], generation,
                inferred=bool(row.get("inferred", False)),
                allocation_arena=row.get("allocation_arena"))
        return target

    def owner_accesses(task):
        md = task.metadata
        for index, access in enumerate(md.get("memory_accesses", ())):
            if access.get("physical_owner", md.get("physical_owner")) != owner:
                continue
            if not access.get("buffer_id"):
                raise ValueError("collective_row_reuse_owner_access_without_allocation")
            yield index, access

    layouts, done, ios = {}, set(), []
    for task in _topological([row[0] for row in keys_by_task.values()], exact_ancestry):
        name = block_of.get(task.task_id)
        if name is None:
            _task, declared, _used = keys_by_task[task.task_id]
            new = [item for item in declared if item[0] not in live
                   or clone.get_allocation(*item[0]).size_bytes < item[1]["size_bytes"]]
            if new:
                reclaim(task.task_id)
                allocate(declared)
                live.update(item[0] for item in declared)
            for index, access in owner_accesses(task):
                key = _key(task, access)[1:]
                base = clone.lookup(*key).base_address
                ios.append({"task_id": task.task_id, "access_index": index,
                    "operation": access["operation"], "byte_count": access["byte_count"],
                    "candidates": [(base+access.get("offset_bytes", 0), access["byte_count"])]})
            continue
        if name in done:
            continue
        done.add(name)
        rounds = blocks[name]
        members = {t.task_id for pair in rounds for t in pair}
        root = rounds[0][0].task_id
        # Every other owner user is outside the whole collective interval.
        for other, (_task, _declared, _used) in keys_by_task.items():
            if other in members:
                continue
            if not (all(before(other, member) for member in members)
                    or all(before(member, other) for member in members)):
                raise ValueError("collective_row_reuse_owner_user_inside_collective")
        reclaim(root)
        sizes = {(keys_by_task[read.task_id][1][0][1]["size_bytes"],
                  keys_by_task[write.task_id][1][0][1]["size_bytes"]) for read, write in rounds}
        if len(sizes) != 1:
            raise ValueError("collective_row_reuse_round_extent_changes")
        read, write = rounds[0]
        read_decl, write_decl = keys_by_task[read.task_id][1], keys_by_task[write.task_id][1]
        snapshot = clone.snapshot()
        both = allocate(write_decl, allocate(read_decl, PhysicalAddressAllocator.from_snapshot(snapshot)))
        alone = allocate(write_decl, PhysicalAddressAllocator.from_snapshot(snapshot))
        base = lambda allocator, decl: allocator.lookup(*decl[0][0]).base_address
        read_base = base(both, read_decl)
        write_bases = sorted({base(alone, write_decl), base(both, write_decl)})
        for pair in rounds:
            for io, bases in zip(pair, ((read_base,), write_bases)):
                for index, access in owner_accesses(io):
                    ios.append({"task_id": io.task_id, "access_index": index, "collective_name": name,
                        "operation": access["operation"], "byte_count": access["byte_count"],
                        "candidates": [(value+access.get("offset_bytes", 0), access["byte_count"])
                                       for value in bases]})
        layouts[name] = {"rounds": rounds, "read_base": base(both, read_decl), "read_bytes": read_decl[0][1]["size_bytes"],
            "write_bases": sorted({base(alone, write_decl), base(both, write_decl)}),
            "write_bytes": write_decl[0][1]["size_bytes"],
            "entry_live_buffer_count": len(live)}
    if set(layouts) != set(blocks):
        raise ValueError("collective_row_reuse_block_not_replayed")
    return layouts, ios


_GEOMETRY_CACHE = {}


def _range_geometry(config, base, size):
    # Every layer replays the same addresses and reparses equal configs; the
    # mapping is a pure function of the normalized geometry fields.
    from .native_hbm_cold_bound import _key as config_key

    key = (config_key(config), base, size)
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    value = _range_geometry_uncached(config, base, size)
    if len(_GEOMETRY_CACHE) >= 4096:
        _GEOMETRY_CACHE.pop(next(iter(_GEOMETRY_CACHE)))
    _GEOMETRY_CACHE[key] = value
    return value


def _range_geometry_uncached(config, base, size):
    from .memory_mapping import map_dram_address

    msb = config.activated_row_msb
    rows, banks, bursts = {}, defaultdict(set), defaultdict(int)
    first, last = base//config.burst_bytes, (base+size-1)//config.burst_bytes
    for burst in range(first, last+1):
        mapping = map_dram_address(config, burst*config.burst_bytes)
        bank = (mapping.stack, mapping.die, mapping.lane, mapping.rank, mapping.bank_group, mapping.bank)
        row = mapping.row if msb is None else mapping.row & ~(1 << msb)
        if rows.setdefault(bank, row) != row:
            raise ValueError("collective_row_reuse_bank_row_conflict")
        channel = (mapping.stack, mapping.channel)
        banks[channel].add(bank)
        bursts[channel] += 1
    return rows, max(map(len, banks.values())), dict(bursts)


def _reopen_terms(config):
    """Per-IO reopen and hit residuals; REF must expire every old history."""
    from .native_hbm_cold_bound import native_hbm_initial_row_upper

    if (type(config) is not DramConfig or config.control_mode != "jedec_hbm"
            or config.refresh_policy != "all_bank"):
        raise ValueError("collective_row_reuse_REF_geometry_unsupported")
    clock = config.command_clock_ns
    dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
    pacing = max(config.burst_bytes/config.directional_lane_bandwidth_gb_s(which)
                 for which in ("read", "write"))
    history = native_hbm_initial_row_upper(config, config.burst_bytes, "write")["initial_history_upper_ns"]
    # REF begins only after every bank's PRE; the next ACT waits a whole tRFC
    # after it. Every ACT/PRE/column/turnaround/DQ history is then settled.
    if not config.refresh_cycle_ns > history+max(config.read_latency_ns, config.write_latency_ns)+max(dq, pacing):
        raise ValueError("collective_row_reuse_REF_does_not_expire_history")
    # A single row per bank leaves no conflict, so row lookahead issues no
    # PRE. Each ACT follows the previous one by tRRD and the one four ACTs
    # earlier by tFAW, so a quarter of tFAW per bank suffices.
    act = max(max(config.activate_command_cycles*clock, config.activate_same_group_ns,
                  config.activate_different_group_ns)+3*clock,
              (config.four_activate_window_ns+3*clock)/4)
    column = max(config.column_same_group_ns, config.column_different_group_ns,
        config.column_different_sid_ns, config.read_to_write_ns, config.write_to_read_ns,
        config.read_to_write_command_ns, config.write_to_read_same_group_ns,
        config.write_to_read_different_group_ns, config.column_command_cycles*clock,
        max(0., pacing-dq))
    burst = max(config.column_same_group_ns, config.column_different_group_ns,
        config.column_different_sid_ns, config.column_command_cycles*clock, pacing, dq)+2*clock
    bridge = {operation: (config.activate_to_read_ns if operation == "read" else config.activate_to_write_ns)
        +(config.read_latency_ns if operation == "read" else config.write_latency_ns)+dq+2*clock
        for operation in ("read", "write")}
    return {"activate_chain_per_bank_ns": act, "post_REF_bridge_ns": bridge,
        "column_direction_residual_ns": column+dq+2*clock, "write_geometry_per_burst_ns": burst,
        "REF_expired_history_upper_ns": history, "refresh_cycle_ns": config.refresh_cycle_ns}


def native_collective_row_reuse(name, rounds, *, configs, native_tasks, exact_ancestry,
                                physical_runtime, allocation_scope):
    """Return per-owner reduced row guards for rounds after the first.

    ``rounds`` lists, per round, ``{owner: (read_task, write_task)}`` on the
    receiving and sending owner. The caller keeps round 0 and the first B2
    touch of each owner at their complete original guards.
    """
    if (not isinstance(allocation_scope, dict)
            or allocation_scope.get("schema") != _TRANSIENT_SCOPE_SCHEMA or exact_ancestry is None):
        return {"qualified": False, "reason": "collective_row_reuse_allocation_scope_unavailable"}
    owners = {}
    for owner in configs:
        facts = allocation_scope["owners"].get(owner)
        if facts is None or facts["qualified"] is not True:
            return {"qualified": False, "reason": "collective_row_reuse_owner_lifetime_not_closed:"+owner,
                    "owner_reasons": None if facts is None else facts["reasons"]}
        allocator = physical_runtime.allocators.get(owner)
        if allocator is None:
            return {"qualified": False, "reason": "collective_row_reuse_owner_allocator_unavailable"}
        try:
            layout = _owner_layouts(owner, native_tasks, exact_ancestry,
                                    allocator, facts["user_counts"])[0][name]
            if [pair for pair in (row[owner] for row in rounds)] != [
                    tuple(pair) for pair in layout.pop("rounds")]:
                raise ValueError("collective_row_reuse_owner_rounds_differ_from_collective_proof")
            config = configs[owner]
            reads = _range_geometry(config, layout["read_base"], layout["read_bytes"])
            writes = [_range_geometry(config, base, layout["write_bytes"]) for base in layout["write_bases"]]
            merged = dict(reads[0])
            for rows, _banks, _bursts in writes:
                for bank, row in rows.items():
                    if merged.setdefault(bank, row) != row:
                        raise ValueError("collective_row_reuse_bank_row_conflict")
            terms = _reopen_terms(config)
        except (KeyError, TypeError, ValueError) as error:
            return {"qualified": False, "reason": str(error), "physical_owner": owner}
        channels = set().union(*(geometry[2] for geometry in writes))
        delta = max(abs(writes[0][2].get(channel, 0)-writes[-1][2].get(channel, 0)) for channel in channels)
        write_banks = max(geometry[1] for geometry in writes)
        reopen = {
            "read": math.nextafter(2*(reads[1]*terms["activate_chain_per_bank_ns"]
                +terms["post_REF_bridge_ns"]["read"])+terms["column_direction_residual_ns"], math.inf),
            "write": math.nextafter(2*(write_banks*terms["activate_chain_per_bank_ns"]
                +terms["post_REF_bridge_ns"]["write"])+terms["column_direction_residual_ns"]
                +delta*terms["write_geometry_per_burst_ns"], math.inf)}
        single = {"read": reads[1]*terms["activate_chain_per_bank_ns"]+terms["post_REF_bridge_ns"]["read"],
            "write": write_banks*terms["activate_chain_per_bank_ns"]+terms["post_REF_bridge_ns"]["write"]}
        hit = {"read": terms["column_direction_residual_ns"],
            "write": terms["column_direction_residual_ns"]+delta*terms["write_geometry_per_burst_ns"]}
        # A hit issues no ACT and waits no tRCD. It keeps the history and
        # column chain of the conflict-free active upper (_repeat_io_bounds).
        # The history-free write variant applies only once every older
        # command history has provably expired before the write starts.
        clock = config.command_clock_ns
        dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
        read_bursts = max(reads[2].values(), default=0)
        write_bursts = max((max(geometry[2].values(), default=0) for geometry in writes), default=0)
        hit_active = {
            "read": math.nextafter(terms["REF_expired_history_upper_ns"]
                +read_bursts*terms["write_geometry_per_burst_ns"]
                +config.read_latency_ns+dq+2*clock, math.inf),
            "write_history_free": math.nextafter(write_bursts*terms["write_geometry_per_burst_ns"]
                +config.write_latency_ns+dq+2*clock, math.inf)}
        owners[owner] = {**layout, "read_bank_upper_per_channel": reads[1],
            "single_reopen_ns": single, "hit_row_guard_ns": hit, "hit_active_upper_ns": hit_active,
            "write_bank_upper_per_channel": write_banks, "write_channel_burst_delta": delta,
            "single_row_per_physical_bank": True, "touched_physical_bank_count": len(merged),
            "reduced_row_guard_ns": reopen, **terms}
    return {"qualified": True, "collective_name": name, "owners": owners,
        "basis": "real_first_fit_owner_allocator_replay_from_source_image; closed_native_transient_lifetimes; single_row_per_bank_over_all_candidate_ranges; two_post_REF_reopen_segments_per_IO; column_direction_group_SID_and_write_geometry_retained",
        "scope": "same_native_cohort_layers_entering_from_the_unchanged_source_allocator_image",
        "address_layout_signature": sorted((owner, row["read_base"], tuple(row["write_bases"]))
                                           for owner, row in owners.items()),
        "physical_state_advanced": False}


def _cohort_single_row_entry(owner, config, facts, physical_runtime, ios):
    """Return (the only row any bank can hold during the cohort, reason).

    Every layer and each serially separated non-layer prefix/suffix starts
    from the source allocator image (unseparated users already fail the
    owner scope). A first-fit extent never ends above the source's highest
    end plus the cumulative padded extents allocated since that image. If
    the largest such group bound stays inside one row period, all cohort
    addresses share one row. A source bank that is closed or open on that
    row then leaves only two possible states for every bank in every image.
    """
    if config.address_mapping != "bank_interleaved":
        return None, "mapping_not_bank_interleaved"
    if facts.get("source_highest_end_byte") is None:
        return None, "source_allocator_unavailable"
    from .memory_mapping import map_dram_address

    bound = facts["source_highest_end_byte"]+facts["max_group_cumulative_extent_bytes"]
    period = config.lane_count*config.bank_count*config.row_bytes
    if not 0 < bound <= period:
        return None, "cumulative_extent_bound_exceeds_one_row_period"
    msb = config.activated_row_msb
    masked = lambda value: value if msb is None or value < 0 else value & ~(1 << msb)
    first, last = map_dram_address(config, 0), map_dram_address(config, bound-1)
    # Row grows monotonically with the address inside one die; the stack
    # coordinate follows the lane in independent mode and is not a row.
    if ((first.die, masked(first.row)) != (last.die, masked(last.row))
            or any(base+size > bound for io in ios for base, size in io["candidates"])):
        return None, "cohort_address_bound_spans_rows"
    row = masked(first.row)
    active = physical_runtime.runtimes.get(owner)
    if active is None:
        return None, "source_controller_unavailable"
    core = active.core
    if any(state.open_row is not None and masked(state.open_row) != row
           for state in core.__dict__.get("_banks", {}).values()):
        return None, "source_bank_open_on_other_row"
    packed = vars(core).get("_hbm_numeric_state")
    if packed is not None and any(int(value) >= 0 and masked(int(value)) != row
                                  for value in packed.arrays[0]):
        return None, "source_bank_open_on_other_row"
    return row, None


def _repeat_io_bounds(config, terms, count, operation, geometry):
    """Row guard and conflict-free active upper for one repeat HBM IO.

    Each bank is closed or open on its only row. A REF-free segment then
    needs at most one ACT per touched bank, serialized on the channel's
    row bus, before an unchanged column/DQ chain. Segments come from the
    unchanged complete active upper, never from this smaller bound.
    """
    from .native_hbm_cold_bound import native_hbm_refresh_service_upper

    service = native_hbm_refresh_service_upper(config, count, operation,
        independent_channels=True, native_refresh_reachable=True)
    if service.get("qualified") is not True:
        return None
    epochs = 1+math.floor(service["active_service_upper_ns"]
        /(config.refresh_interval_ns-service["refresh_epoch_upper_ns"]))
    banks = max(row[1] for row in geometry)
    channels = set().union(*(row[2] for row in geometry))
    delta = max((max(row[2].get(channel, 0) for row in geometry)
                 -min(row[2].get(channel, 0) for row in geometry) for channel in channels), default=0)
    channel_bursts = max(max(row[2].values()) for row in geometry)
    clock = config.command_clock_ns
    dq = config.burst_bytes/config.effective_lane_bandwidth_gb_s
    read = operation == "read"
    guard = math.nextafter((epochs+1)*(banks*terms["activate_chain_per_bank_ns"]
        +terms["post_REF_bridge_ns"][operation])+terms["column_direction_residual_ns"]
        +delta*terms["write_geometry_per_burst_ns"], math.inf)
    active = math.nextafter(terms["REF_expired_history_upper_ns"]
        +banks*terms["activate_chain_per_bank_ns"]
        +channel_bursts*terms["write_geometry_per_burst_ns"]
        +(config.activate_to_read_ns if read else config.activate_to_write_ns)
        +(config.read_latency_ns if read else config.write_latency_ns)+dq+2*clock, math.inf)
    return {"row_guard_ns": guard, "active_upper_ns": min(active, service["active_service_upper_ns"]),
        "REF_free_segments": epochs+1, "bank_upper_per_channel": banks,
        "channel_burst_upper": channel_bursts, "candidate_channel_burst_delta": delta}


def native_layer_row_reuse(native_tasks, *, exact_ancestry, physical_runtime, allocation_scope,
                           configs, source_qualified):
    """Per-IO initial-row guards for every HBM IO of a summarized subgraph.

    The same allocator replay supplies every IO's candidate address ranges.
    If all of them keep one row per physical bank, a bank first touched by
    an IO that strictly precedes another in the native DAG is, for that
    later IO, either open on its only row or closed by REF, whatever the
    entry image was. Such a repeat IO pays one post-REF reopen per REF-free
    segment plus the column/direction and candidate-geometry residuals. Any
    IO touching a bank that no predecessor surely touched keeps its complete
    original guard; so does every IO of an owner whose proof fails.
    """
    import hashlib
    from .native_window_producer import _native_dag_precedes

    result = {"io_guards": {}, "io_active_uppers": {}, "owners": {}, "rejections": {},
        "physical_state_advanced": False,
        "basis": "real_first_fit_owner_allocator_replay; one_row_per_physical_bank_over_every_candidate_range; DAG_predecessor_sure_bank_cover; post_REF_reopen_per_REF_free_segment",
        "scope": "same_native_cohort_layers_entering_from_the_unchanged_source_allocator_image"}
    if (not isinstance(allocation_scope, dict)
            or allocation_scope.get("schema") != _TRANSIENT_SCOPE_SCHEMA or exact_ancestry is None):
        result["reason"] = "layer_row_reuse_allocation_scope_or_DAG_unavailable"
        return result

    def before(left, right):
        return left != right and _native_dag_precedes(exact_ancestry, left, right)

    for owner, config in sorted(configs.items()):
        try:
            facts = allocation_scope["owners"].get(owner)
            if facts is None or facts["qualified"] is not True:
                raise ValueError("layer_row_reuse_owner_lifetime_not_closed")
            if source_qualified.get(owner) is not True:
                raise ValueError("layer_row_reuse_native_REF_source_unqualified")
            allocator = physical_runtime.allocators.get(owner)
            if allocator is None:
                raise ValueError("layer_row_reuse_owner_allocator_unavailable")
            terms = _reopen_terms(config)
            _layouts, ios = _owner_layouts(owner, native_tasks, exact_ancestry,
                                           allocator, facts["user_counts"])
            rows = {}
            for io in ios:
                io["geometry"] = [_range_geometry(config, base, size) for base, size in io["candidates"]]
                for geometry in io["geometry"]:
                    for bank, row in geometry[0].items():
                        if rows.setdefault(bank, row) != row:
                            raise ValueError("layer_row_reuse_bank_row_conflict")
                banks = [set(geometry[0]) for geometry in io["geometry"]]
                io["sure"], io["union"] = set.intersection(*banks), set.union(*banks)
            entry_row, entry_reason = _cohort_single_row_entry(owner, config, facts, physical_runtime, ios)
            guards, actives, prior, running, kept = {}, {}, [], set(), 0
            for io in ios:
                predecessors = [other for other in prior if before(other["task_id"], io["task_id"])]
                covered = running if len(predecessors) == len(prior) else set().union(
                    *(other["sure"] for other in predecessors))
                if entry_row is not None:
                    # No image of this cohort can hold another row in any bank.
                    covered = io["union"]
                count, operation = io["byte_count"], io["operation"]
                bursts = (count+2*config.burst_bytes-2)//config.burst_bytes
                bounds = (_repeat_io_bounds(config, terms, count, operation, io["geometry"])
                          if io["union"] <= covered and bursts <= config.lane_count*config.bank_count
                          else None)
                if bounds is not None:
                    guards[(io["task_id"], io["access_index"])] = bounds["row_guard_ns"]
                    actives[(io["task_id"], io["access_index"])] = bounds["active_upper_ns"]
                else:
                    kept += 1
                prior.append(io)
                running = running | io["sure"]
        except (KeyError, TypeError, ValueError) as error:
            result["rejections"][owner] = str(error)
            continue
        result["io_guards"].update(guards)
        result["io_active_uppers"].update(actives)
        signature = hashlib.sha256(repr([(io["operation"], io["byte_count"], io["candidates"])
            for io in ios]).encode()).hexdigest()
        result["owners"][owner] = {"io_count": len(ios), "repeat_io_count": len(guards),
            "first_touch_io_count": kept, "touched_physical_bank_count": len(rows),
            "single_row_per_physical_bank": True, "address_layout_sha256": signature,
            "cohort_entry_row": entry_row, "cohort_entry_row_reason": entry_reason,
            "cohort_address_upper_byte": facts.get("source_highest_end_byte", 0)+facts.get("max_group_cumulative_extent_bytes", 0),
            "max_candidate_end_byte": max((base+size for io in ios for base, size in io["candidates"]), default=0),
            **{key: terms[key] for key in ("activate_chain_per_bank_ns", "post_REF_bridge_ns",
                "column_direction_residual_ns", "REF_expired_history_upper_ns", "refresh_cycle_ns")}}
    return result


__all__ = ["native_allocation_scope", "native_collective_row_reuse", "native_layer_row_reuse"]
