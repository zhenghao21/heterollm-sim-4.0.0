"""Boundary-conditioned windows of the existing native physical Task graph.

Each window runs complete selected layers (including their communications)
on an independent copy of the real post-ingress kernel. Omitted predecessors
are declared boundary inputs, never fabricated physical events. A sampled
kernel cannot resume the full cohort, and never commits a serving Token.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, replace
import json
import time
from typing import Mapping

from .event_kernel import CompiledGraphLayout, _validate_event_timing
from .live_execution_metrics import LiveCohortMetrics
from .native_row_bounds import first_access_ns
from .memory_allocator import PhysicalAddressAllocator

_TP_ALLOCATOR_NAMESPACE = tuple((k,v) for k,v in vars(PhysicalAddressAllocator).items() if k != "__slotnames__")


_IDENTITY_FIELDS = frozenset(("layer_id", "name", "label", "operator_id", "source_operator_id"))


class NativeLayerSamplingGuardError(ValueError):
    """A valid native graph cannot satisfy this sampling contract."""


def _layer_counts(classes, calibration, holdout):
    """Shortest head/tail runs whose steps after the first cover every class."""
    wanted, count = set(classes), len(classes)

    def covering(minimum, stepped):
        return next((n for n in range(minimum, count + 1) if set(stepped(n)) == wanted), count + 1)

    return (covering(calibration, lambda n: classes[1:n]),
            covering(holdout, lambda n: classes[count - n + 1:]))


def _semantic(value):
    if isinstance(value, Mapping):
        return {key: _semantic(item) for key, item in value.items() if key not in _IDENTITY_FIELDS}
    if isinstance(value, (tuple, list)):
        return [_semantic(item) for item in value]
    return value


def layer_inventory(scenario, specs):
    """Classify by native typed geometry/semantics, never model size alone."""
    view = scenario.model._execution_view
    by_layer = defaultdict(list)
    for task in specs:
        identity = task.metadata.get("layer_id")
        if identity is not None:
            by_layer[str(identity)].append(task)
    rows = []
    known = set()
    for index, descriptor in enumerate(view.layer_instances):
        layer = descriptor.layer
        identity = str(layer.layer_id)
        if identity in known:
            raise ValueError("native layer sampler requires unique expanded layer identities")
        known.add(identity)
        signature = _semantic(asdict(layer))
        key = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        tasks = tuple(by_layer.get(identity, ()))
        if not tasks:
            raise NativeLayerSamplingGuardError("native layer has no complete authored Task subgraph: " + identity)
        rows.append({"layer_id": identity, "layer_index": index, "class_key": key,
                     "signature": signature, "task_count": len(tasks)})
    if set(by_layer) - known:
        raise NativeLayerSamplingGuardError("native Task graph contains unknown layer identities")
    if not rows:
        raise NativeLayerSamplingGuardError("native model has no expanded layers to sample")
    return rows


def _graph_facts(specs, inventory):
    """Prove serial layer composition on the actual dependency graph."""
    by_id = {task.task_id: task for task in specs}
    if len(by_id) != len(specs):
        raise ValueError("native layer graph contains duplicate Task ids")
    layer_indices = {row["layer_id"]: row["layer_index"] for row in inventory}
    groups = {task.task_id: task.metadata.get("layer_id") for task in specs}
    children = defaultdict(list)
    indegree = {}
    for task in specs:
        if any(parent not in by_id for parent in task.dependencies):
            raise ValueError("native layer sampler requires the original closed cohort graph")
        indegree[task.task_id] = len(task.dependencies)
        for parent in task.dependencies:
            children[parent].append(task.task_id)
    queue = deque(task.task_id for task in specs if not indegree[task.task_id])
    order = []
    while queue:
        identity = queue.popleft()
        order.append(identity)
        for child in children[identity]:
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    if len(order) != len(specs):
        raise ValueError("native layer Task graph is cyclic")
    roots, terminals = defaultdict(list), defaultdict(list)
    for task in specs:
        layer = groups[task.task_id]
        if layer is None:
            continue
        if not any(groups[parent] == layer for parent in task.dependencies):
            roots[layer].append(task.task_id)
        if not any(groups[child] == layer for child in children[task.task_id]):
            terminals[layer].append(task.task_id)
    terminal_bits, required = {}, {}
    position = 0
    for row in inventory:
        mask = 0
        for identity in terminals[row["layer_id"]]:
            terminal_bits[identity] = 1 << position
            mask |= 1 << position
            position += 1
        required[row["layer_id"]] = mask
    masks, layer_before = {}, {}
    for identity in order:
        mask, before = 0, 0
        for parent in by_id[identity].dependencies:
            mask |= masks[parent] | terminal_bits.get(parent, 0)
            before |= layer_before[parent]
            parent_layer = groups[parent]
            if parent_layer is not None:
                before |= 1 << layer_indices[parent_layer]
        masks[identity], layer_before[identity] = mask, before
    layer_after = {}
    for identity in reversed(order):
        after = 0
        for child in children[identity]:
            after |= layer_after[child]
            child_layer = groups[child]
            if child_layer is not None:
                after |= 1 << layer_indices[child_layer]
        layer_after[identity] = after
    failures = []
    for previous, current in zip(inventory, inventory[1:]):
        expected = required[previous["layer_id"]]
        if not expected or any(masks[root] & expected != expected for root in roots[current["layer_id"]]):
            failures.append({"before": previous["layer_id"], "after": current["layer_id"],
                             "reason": "not_all_previous_terminals_precede_every_next_root"})
    boundary = defaultdict(list)
    for task in specs:
        if groups[task.task_id] is not None:
            continue
        before, after = layer_before[task.task_id], layer_after[task.task_id]
        kind = ("before_first_layer" if not before and after else
                "after_last_layer" if before and not after else
                "interior" if before and after else "independent")
        boundary[kind].append(task.task_id)
    # A layer may have independent rank branches, but every branch must
    # join before the next layer starts. Non-layer work can only be added
    # directly when it too is separated by the native dependency edges.
    nonserial_boundary = []
    for kind, ids in boundary.items():
        wanted = frozenset(ids)
        boundary_roots = [identity for identity in ids
                          if not any(parent in wanted for parent in by_id[identity].dependencies)]
        boundary_terminals = [identity for identity in ids
                              if not any(child in wanted for child in children[identity])]
        if kind == "before_first_layer":
            # Prove each prefix terminal is an ancestor of every first
            # layer root, with compact bitsets instead of ancestor sets.
            bits = {identity: 1 << index for index, identity in enumerate(boundary_terminals)}
            ancestry = {}
            for identity in order:
                mask = 0
                for parent in by_id[identity].dependencies:
                    mask |= ancestry[parent] | bits.get(parent, 0)
                ancestry[identity] = mask
            required_mask = (1 << len(bits))-1
            separated = all(ancestry[identity] & required_mask == required_mask
                            for identity in roots[inventory[0]["layer_id"]])
        elif kind == "after_last_layer":
            required_mask = required[inventory[-1]["layer_id"]]
            separated = all(masks[identity] & required_mask == required_mask
                            for identity in boundary_roots)
        else:
            separated = False
        if not separated:
            nonserial_boundary.append(kind)
    return {"qualified": not failures, "basis": "all_previous_layer_terminals_are_ancestors_of_all_next_layer_roots",
            "serial_layer_boundaries_proven": not failures,
            "nonserial_non_layer_groups": nonserial_boundary,
            "layer_order": [row["layer_id"] for row in inventory], "failures": failures,
            "roots": dict(roots), "terminals": dict(terminals)}, dict(boundary)


# Per-request KV records are the same rows seen through several views of a
# task's metadata. Window execution reads them only for the tasks it runs.
PER_REQUEST_RECORD_KEYS = (
    "memory_accesses", "physical_address_bindings", "buffer_accesses", "physical_allocations",
    "rhs_buffer_accesses", "kv_writeback_accesses", "persistent_request_buffers",
    "persistent_buffer_ids", "generic_kv_storage")


def drop_unsampled_request_records(metadata_rows, kept_layers):
    """Delete the per-request records of tasks in layers no window will run.

    ``metadata_rows`` are the task metadata dicts of every view that shares
    the records; the rows are only released once no view keeps them. Tasks
    outside any layer, and tasks of ``kept_layers``, are untouched.
    """
    kept = frozenset(kept_layers)
    dropped = 0
    for metadata in metadata_rows:
        layer = metadata.get("layer_id")
        if layer is None or str(layer) in kept:
            continue
        for key in PER_REQUEST_RECORD_KEYS:
            if metadata.pop(key, None) is not None:
                dropped += 1
    return dropped


def _independent_kernel(source, executor=None):
    if source.has_active_tasks:
        raise NativeLayerSamplingGuardError("native layer boundary requires no active full-cohort tasks")
    # Resource ownership is immutable; copy all mutable controllers, calendars
    # and allocations together so their internal shared references survive.
    memo = {id(source.resource_owners): source.resource_owners}
    # Buffer keys are immutable (owner, buffer id, generation) tuples and
    # holder counts are integers. Copy the mutable maps/sets once, retaining
    # their identities in the shared memo used by the kernel and PD executor.
    # Walking each scalar tuple through deepcopy twice dominates large
    # post-ingress boundaries. Custom containers keep their copy protocol.
    holders = source._request_buffer_holders
    if type(holders) is dict:
        memo[id(holders)] = holders.copy()
    requests = source._request_physical_buffers
    if type(requests) is dict:
        copied_requests = {}
        memo[id(requests)] = copied_requests
        for identity, keys in requests.items():
            if id(keys) not in memo and type(keys) is set:
                memo[id(keys)] = keys.copy()
            copied_requests[identity] = deepcopy(keys, memo)
    kernel = deepcopy(source, memo)
    if executor is None:
        return kernel, None
    # The native PD executor reserves this kernel's own physical runtime and
    # cache states. One memo rebinds the copy to the copied calendars.
    copied = deepcopy(executor, memo)
    if (copied.runtime is not kernel.physical_runtime
            or copied.cache_states is not kernel._l2_states
            or copied.cache_owners is not kernel._physical_allocation_cache_owners):
        raise NativeLayerSamplingGuardError("native PD executor is not bound to the sampled kernel")
    return kernel, copied


class _ConcurrentImports:
    """Continue real in-flight native PD imports on an independent copy.

    Each import keeps its own executor state (stream cursors, in-flight chunks
    and their phase heap, token), so the continuation submits exactly the
    phases the live runtime would. As in
    ``_advance_prefetches_before_kernel_step``, a phase due no later than the
    next ready Task is submitted first, then every frontier is republished.
    Final commit/publication of the imported context is not performed.
    """

    def __init__(self, kernel, executor, request_ids):
        self.kernel, self.executor = kernel, executor
        states = {state.get("request_id"): state for state in executor._async_plan_states.values()
                  if not state.get("done") and "destinations" in state}
        missing = sorted(set(request_ids)-set(states))
        if missing:
            raise NativeLayerSamplingGuardError("native PD import is not in flight: "+", ".join(missing))
        self.states = [states[identity] for identity in request_ids]
        self.writes, self.submitted = [], 0

    def advance_until(self, time_ns):
        from ._resource_ledger import publish_owner_frontiers, publish_timeline_frontiers
        kernel = self.kernel
        submitted = self.submitted
        while True:
            due = min(((state["completion_ns"], index) for index, state in enumerate(self.states)
                       if not state["final"]), default=None)
            if due is None or due[0] > time_ns:
                break
            state = self.states[due[1]]
            mirror = getattr(kernel, "_native_tp_mirror", None)
            if mirror is not None:
                mirror.invalidate(kernel)
            self.executor._advance_external_phase(state, state["phase_token"])
            if state["last_write"] is not None:
                self.writes.append((*state["last_write"], state["request_id"]))
            self.submitted += 1
        if self.submitted != submitted:
            mirror = getattr(kernel, "_native_tp_mirror", None)
            if mirror is not None:
                mirror.invalidate(kernel)
            # Phases reserve the physical calendar directly and never read
            # the scalar mirror, so one publication before the next Task
            # equals one after every phase.
            publish_timeline_frontiers(kernel.physical_runtime.timeline.lane_available,
                                       kernel.resource_available)
            publish_owner_frontiers(kernel.resource_owners, kernel.resource_available)

    def before_step(self, kernel):
        ready = kernel.peek_ready_key()
        if ready is not None:
            self.advance_until(float(ready[0]))

    def observation(self, start_ns, end_ns):
        inside = [row for row in self.writes if start_ns < row[0] <= end_ns]
        identities = [state["request_id"] for state in self.states]
        return {"request_ids": identities,
            "submitted_phases": self.submitted,
            "written_bytes_in_window": sum(row[1] for row in inside),
            "write_completions_ns": [row[0] for row in inside],
            "write_phases_in_window": len(inside),
            "written_bytes_by_request": {identity: sum(row[1] for row in inside if row[2] == identity)
                                         for identity in identities},
            "write_phases_by_request": {identity: sum(row[2] == identity for row in inside)
                                        for identity in identities},
            "final_reached": [state["request_id"] for state in self.states if state["final"]],
            "import_spans_ns": {state["request_id"]: (state["start_ns"], state["completion_ns"])
                                for state in self.states if state["final"]},
            "scope": "real_in_flight_native_PD_continued_on_the_sample_copy; destination_commit_not_performed"}


def remaining_import_bytes(state):
    """Logical bytes of a live import plan whose target write is not yet submitted."""
    return state["logical_bytes"]-state["submitted_write_bytes"]


class _RankRename:
    """Bijection of declared components and paired buffer descriptors."""

    def __init__(self, components):
        self.components, self.buffers, self.reverse = components, {}, {}

    def bind(self, left, right):
        if left is None or right is None:
            return left is right
        if self.buffers.get(left, right) != right or self.reverse.get(right, left) != left:
            return False
        self.buffers[left], self.reverse[right] = right, left
        return True

    def text(self, value):
        if value in self.buffers:
            return self.buffers[value]
        for left, right in self.components.items():
            if value == left or value.startswith(left + '.') or value.startswith(left + ':'):
                return right + value[len(left):]
        return value

    def copy(self, value, memo=None):
        from enum import Enum
        from operator import itemgetter
        from .memory_allocator import PhysicalAllocation
        if type(value) is PhysicalAllocation:
            return replace(value, buffer_id=self.text(value.buffer_id),
                           alias_of=None if value.alias_of is None else self.text(value.alias_of))
        memo = {} if memo is None else memo
        pending, seen = [value], set(memo)
        while pending:
            item = pending.pop()
            if id(item) in seen:
                continue
            seen.add(id(item))
            if type(item) is str:
                memo[id(item)] = self.text(item)
            elif isinstance(item, dict):
                pending.extend(item.keys())
                pending.extend(item.values())
            elif isinstance(item, (list, tuple, set, frozenset)):
                pending.extend(item)
            elif isinstance(item, itemgetter):
                pending.extend(item.__reduce__()[1])
            elif isinstance(item, Enum) or callable(item):
                continue
            elif hasattr(item, '__dict__'):
                pending.extend(vars(item).values())
            elif getattr(item, 'dtype', None) == object:
                pending.extend(item.flat)
        return deepcopy(value, memo)


def _state_bits(value):
    """Typed, exact state comparison, including IEEE bits."""
    from dataclasses import fields, is_dataclass
    from enum import Enum
    if type(value) is float:
        return (float, value.hex())
    if value is None or isinstance(value, (str, int, bool, Enum)):
        return (type(value), value)
    if isinstance(value, dict):
        return (dict, frozenset((_state_bits(k), _state_bits(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(map(_state_bits, value)))
    if isinstance(value, (set, frozenset)):
        return (type(value), frozenset(map(_state_bits, value)))
    if is_dataclass(value):
        return (type(value), tuple((f.name, _state_bits(getattr(value, f.name))) for f in fields(value)))
    raise TypeError('unsupported native TP state: ' + str(type(value)))


def _core_bits(active, rename):
    # Publish only a temporary copy: live public getters disable numeric residency.
    core = active.core
    copied = deepcopy(core, {id(core.timeline): core.timeline})
    from .dram_core import DramCore
    from .nand_core import NandCore
    if type(copied) is DramCore:
        copied._hbm_before_owned_snapshot()
    elif type(copied) is NandCore:
        copied._nand_before_reference()
    else:
        raise TypeError('unsupported native TP core')
    state = {k: v for k, v in vars(copied).items() if k != 'timeline'}
    return _state_bits((active.clock_ns, rename.copy(state)))


_TP_TIMELINE_MAPS = ('ready_ns', 'directions', 'busy_ns', 'bytes_moved', 'last_intervals',
                    'lane_available', 'lane_directions', '_intervals', '_interval_payloads')


def _owner_resources(resource, owners):
    return any(resource == owner or resource.startswith(owner + ':') or resource.startswith(owner + '.')
               for owner in owners)


def _exclusive_native_allocator(runtime, owner):
    allocator = runtime.allocators.get(owner)
    if type(allocator) is not PhysicalAddressAllocator:
        return False
    namespace = vars(PhysicalAddressAllocator)
    if (len(namespace) - ('__slotnames__' in namespace) != len(_TP_ALLOCATOR_NAMESPACE)
            or any(namespace.get(k) is not v for k,v in _TP_ALLOCATOR_NAMESPACE)
            or any(k in vars(allocator) for k,v in _TP_ALLOCATOR_NAMESPACE if callable(v))):
        return False
    metadata = runtime.runtimes[owner].core.config.metadata
    if any(type(value) is str and value and not _owner_resources(value,(owner,))
           for name,value in metadata.items() if name.endswith('resource_prefix') or name == 'host_resource_id'):
        return False
    return all(not _owner_resources(source,(owner,)) or
               _owner_resources(runtime.timeline.canonical_resource_id(target),(owner,))
               for source,target in runtime.timeline.resource_aliases.items())


def _rank_timeline(kernel, owners):
    timeline = kernel.physical_runtime.timeline
    return {name: {rid: deepcopy(value) for rid, value in getattr(timeline, name).items()
                   if _owner_resources(rid, owners)} for name in _TP_TIMELINE_MAPS}


class _TPRankMirror:
    """Reuse an equal native physical transition, preserving the full event DAG.

    Each pipeline stage and EP group uses its own TP rank zero.
    Prefixes and collectives execute normally.
    Exact reuse starts from equal native state and interns complete physical
    input tuples as an induction proof. Controllers are copied lazily before
    the next real writer; calendars and allocations remain independent.
    The explicit PD experiment instead assumes equivalent controller state,
    while still checking the graph, geometry and actual dispatch arrivals.
    """

    def __init__(self, kernel, tasks, scenario, *, approximate_imports=False,
                 allow_ep_groups=False):
        from .data_motion import is_physical_task, physical_task_transaction_owners
        self.pairs, self.representatives, self.records = {}, set(), {}
        self.tokens, self.serial, self.pending = {}, 0, {}
        self.transition_tokens, self.owner_canonical = {}, {}
        self.fast_exact = not approximate_imports
        self.stats_mode = 'approximate_PD' if approximate_imports else 'exact'
        self.stats = {'candidate_tasks': 0, 'mirrored_tasks': 0, 'fallback_tasks': 0,
                      'mode': self.stats_mode}
        parallel = scenario.placement.parallel
        if (kernel.physical_runtime.capture_details or kernel.physical_runtime.timeline._interval_limit
                or kernel.physical_runtime.hbf_staging):
            return
        if parallel.tp_degree <= 1:
            return
        # PP uses only the exact transition proof. The separate PD experiment
        # has no proof for overlapping pipeline stages.
        if parallel.pp_degree != 1 and approximate_imports:
            return
        if parallel.ep_degree != 1 and not allow_ep_groups:
            return
        mappings = sorted(getattr(parallel, 'rank_mapping', ()), key=lambda row: row.rank)
        if not mappings:
            return
        # Pipeline stages have different layers, arrivals and controller state.
        # Pair only TP siblings in the same (PP, EP) coordinate; every later
        # graph, allocation and transition-state check remains unchanged.
        world = parallel.tp_degree * parallel.pp_degree * parallel.ep_degree
        if (len(mappings) != world
                or {row.rank for row in mappings} != set(range(world))):
            return
        groups = defaultdict(list)
        for row in mappings:
            groups[(row.pp_rank, row.ep_rank)].append(row)
        if set(groups) != {(pp, ep) for pp in range(parallel.pp_degree)
                          for ep in range(parallel.ep_degree)}:
            return
        pairings = []
        for coordinate in sorted(groups):
            group = sorted(groups[coordinate], key=lambda row: row.tp_rank)
            if tuple(row.tp_rank for row in group) != tuple(range(parallel.tp_degree)):
                return
            pairings.extend((group[0], row) for row in group[1:])
        components = {row.component_id: row for row in scenario.hardware.components}
        packages = {components[m.component_id].package_id for m in mappings}
        # Each owner belongs to one TP group, so canonical transition tokens
        # and pending controller copies cannot overlap between PP/EP groups.
        if (parallel.pp_degree != 1 or parallel.ep_degree != 1) and len(packages) != len(mappings):
            return
        # A physical prefix task may initialize rank-local state before the
        # first layer. If that prefix is asymmetric, mirroring it would make
        # the later PP/TP transition proof unsound; fall back to full work.
        rank_components = {c.component_id for c in scenario.hardware.components
                           if c.package_id in packages}
        by_id = {t.task_id: t for t in tasks}
        def after_layer(task):
            pending, seen = list(task.dependencies), set()
            while pending:
                identity = pending.pop()
                if identity in seen:
                    continue
                seen.add(identity)
                parent = by_id.get(identity)
                if parent is not None:
                    if parent.metadata.get('layer_id') is not None:
                        return True
                    pending.extend(parent.dependencies)
            return False
        if self.fast_exact and any(
                t.metadata.get('layer_id') is None
                and type(t.metadata.get('rank')) is not int
                and is_physical_task(t)
                and not after_layer(t)
                and any(_owner_resources(o, rank_components)
                        for o in physical_task_transaction_owners(t) or ())
                for t in tasks):
            self.stats['asymmetric_prefix'] = True
            return
        # Unranked frontend/activation work executes normally. Its real
        # per-owner transitions advance the same exact tokens as layer work;
        # unequal prefixes therefore disable later reuse through state proof,
        # rather than rejecting an entire pipeline before dispatch.
        ranks = defaultdict(list)
        for task in tasks:
            if type(task.metadata.get('rank')) is int:
                ranks[task.metadata['rank']].append(task)
        from .communication import TopologyRouter
        router = TopologyRouter(scenario.hardware)
        for representative, mapping in pairings:
            head, tail = ranks[representative.rank], ranks[mapping.rank]
            if len(head) != len(tail):
                continue
            first = components[representative.component_id].package_id
            second = components[mapping.component_id].package_id
            left = [c for c in scenario.hardware.components if c.package_id == first]
            right = [c for c in scenario.hardware.components if c.package_id == second]
            if first == second or len(left) != len(right) or any(a.kind != b.kind for a,b in zip(left,right)):
                continue
            rename = _RankRename({a.component_id: b.component_id for a,b in zip(left,right)})
            # Bind actual negotiated hops by their typed endpoints and work,
            # rather than parsing rank/link names. Collective links are not
            # part of the physical transition being mirrored.
            for component in tuple(rename.components):
                for hop in router._adjacency.get(component, ()):
                    if hop.target_component not in rename.components:
                        continue
                    candidates = [h for h in router._adjacency.get(rename.text(component), ())
                        if h.target_component == rename.text(hop.target_component)
                        and (h.bandwidth_gbps,h.latency_ns,h.queue_depth) ==
                            (hop.bandwidth_gbps,hop.latency_ns,hop.queue_depth)]
                    if len(candidates) == 1:
                        rename.components[hop.resource_id] = candidates[0].resource_id
            ids = {a.task_id: b.task_id for a,b in zip(head,tail)}
            owners, valid = {}, True
            for owner, allocator in kernel.physical_runtime.allocators.items():
                other = rename.text(owner)
                if other == owner:
                    continue
                target = kernel.physical_runtime.allocators.get(other)
                if target is None or (allocator.capacity_bytes, allocator.alignment_bytes,
                    allocator.workspace_capacity_bytes) != (target.capacity_bytes, target.alignment_bytes,
                                                           target.workspace_capacity_bytes):
                    valid = False
                    break
                if not (_exclusive_native_allocator(kernel.physical_runtime,owner) and
                        _exclusive_native_allocator(kernel.physical_runtime,other)):
                    valid = False
                    break
                a,b = allocator._allocations, target._allocations
                aliases = bool(allocator._aliases_by_target or target._aliases_by_target)
                ai = {key:i for i,key in enumerate(a)} if aliases else {}
                bi = {key:i for i,key in enumerate(b)} if aliases else {}
                def geometry(row, positions):
                    return (row.base_address, row.size_bytes, row.generation, row.inferred,
                            row.allocation_arena, row.alias_offset_bytes,
                            None if row.alias_of is None else positions.get((row.alias_of,row.alias_generation), -1))
                if len(a) != len(b) or any(geometry(x,ai) != geometry(y,bi) for x,y in zip(a.values(),b.values())):
                    valid = False
                    break
                owners[owner] = other
            if not valid or not owners:
                continue
            for a,b in zip(head,tail):
                # Shared boundary nodes are retained; local edges must map exactly.
                if tuple(ids.get(p,p) for p in a.dependencies) != b.dependencies:
                    continue
                if (a.category,a.earliest_start_ns,a.marker,a.token_index,a.metadata.get('layer_id')) != (
                    b.category,b.earliest_start_ns,b.marker,b.token_index,b.metadata.get('layer_id')):
                    continue
                if not is_physical_task(a) or not is_physical_task(b):
                    continue
                # Core request ids are part of observations and admission
                # state. Anonymous Tasks would use different Task ids.
                if a.request_id is None or a.request_id != b.request_id:
                    continue
                # The native dispatcher tests L2 key presence, including an
                # empty/invalid contract; replay must never bypass that path.
                if any(k in a.metadata or k in b.metadata for k in ('stateful_l2','l2_execution')):
                    continue
                if any(a.metadata.get(k) or b.metadata.get(k) for k in
                       ('hbf_host_staging_by_owner','hbf_preloaded_buffers')):
                    continue
                def envelope(task):
                    return task.metadata.get('phase_metadata', {}).get('kernel_model', {}).get('physical_memory_envelope')
                if rename.copy(envelope(a)) != envelope(b):
                    continue
                source_owners = physical_task_transaction_owners(a)
                if not source_owners or not source_owners <= owners.keys() or rename.copy(a.demands) != b.demands:
                    continue
                if any(rename.text(kernel._physical_owner(x.resource_id)) != kernel._physical_owner(y.resource_id)
                       for x,y in zip(a.demands,b.demands)):
                    continue
                valid = True
                for key in ('memory_accesses','physical_allocations'):
                    aa,bb = a.metadata.get(key,()), b.metadata.get(key,())
                    if len(aa) != len(bb) or not all(rename.bind(x.get('buffer_id'),y.get('buffer_id')) for x,y in zip(aa,bb)):
                        valid = False
                        break
                    def authored(rows):
                        return [{k:v for k,v in row.items() if k != 'address' or
                                 not row.get('buffer_id') or not row.get('address_source') or
                                 str(row['address_source']).startswith('explicit')} for row in rows]
                    if rename.copy(authored(aa)) != authored(bb):
                        valid = False
                        break
                if valid and any(rename.copy(a.metadata.get(k)) != b.metadata.get(k) for k in
                                 ('physical_memory_config', 'physical_memory_configs')):
                    valid = False
                if valid:
                    self.pairs[b.task_id] = (a.task_id,rename,source_owners,frozenset(owners[o] for o in source_owners))
                    self.representatives.add(a.task_id)
        self.remaining = defaultdict(int)
        for identity, *_ in self.pairs.values():
            self.remaining[identity] += 1
        self.stats['candidate_tasks'] = len(self.pairs)
        if self.fast_exact:
            # This is the induction base. Later transitions only need equal
            # input geometry, arrival and transition tokens; no follower Core
            # is ever read. An unexpected native dispatch discards this copy.
            failed = set()
            checked = set()
            for identity, rename, source_owners, owners in self.pairs.values():
                key = tuple(sorted((o, rename.text(o)) for o in source_owners))
                if key in checked:
                    continue
                checked.add(key)
                if (not kernel.physical_runtime._owners_have_native_hooks(source_owners | owners)
                        or _state_bits(rename.copy(_rank_timeline(kernel,source_owners))) !=
                           _state_bits(_rank_timeline(kernel,owners))):
                    failed.update(owners)
                    continue
                try:
                    if any(_core_bits(kernel.physical_runtime.runtimes[o],rename) !=
                           _core_bits(kernel.physical_runtime.runtimes[rename.text(o)],_RankRename({}))
                           for o in source_owners):
                        failed.update(owners)
                except TypeError:
                    failed.update(owners)
            self.pairs = {k:p for k,p in self.pairs.items() if not (p[3] & failed)}
            self.stats['candidate_tasks'] = len(self.pairs)
            self.stats['initial_state_rejected_owners'] = len(failed)
            self.representatives = {p[0] for p in self.pairs.values()}
            for _,rename,source_owners,_ in self.pairs.values():
                for owner in source_owners:
                    self.owner_canonical[owner] = owner
                    self.owner_canonical[rename.text(owner)] = owner

    def materialize(self, kernel, owners=None):
        runtime = kernel.physical_runtime
        for owner in tuple(self.pending) if owners is None else tuple(owners):
            saved = self.pending.pop(owner, None)
            if saved is None:
                continue
            active, rename = saved
            current = runtime.runtimes[owner]
            current.core = rename.copy(active.core, {id(active.core.timeline): runtime.timeline,
                                                     id(active.core.config): current.core.config})
            current.clock_ns = active.clock_ns

    def invalidate(self, kernel):
        # PD phases mutate destination controllers outside ordinary dispatch.
        self.materialize(kernel)
        self.tokens.clear()
        if self.fast_exact:
            self.pairs.clear()
            self.representatives.clear()

    def _advance_exact_tokens(self, task, owners):
        md = task.metadata
        accesses = md.get('memory_accesses', ())
        for owner in owners:
            # A native owner is a deterministic state machine. Intern complete
            # ordered input tuples, never digests: equal pre-state + equal
            # arrivals/addresses/operations proves equal post-state. This also
            # recognizes symmetric collective endpoints that still run in full.
            key = (self.owner_canonical.get(owner, owner), self.tokens.get(owner),
                   task.request_id or task.task_id, float(md['physical_arrival_ns']).hex(),
                   tuple((i, str(a['operation']), a['address'], a['byte_count'])
                         for i, a in enumerate(accesses)
                         if a.get('physical_owner', md.get('physical_owner')) == owner))
            token = self.transition_tokens.get(key)
            if token is None:
                self.serial += 1
                token = self.transition_tokens[key] = self.serial
            self.tokens[owner] = token

    def _resolve_exact(self, kernel, task, start, demands, full):
        from .data_motion import (is_physical_task, physical_task_transaction_owners,
                                  prepare_physical_task, _recost_physical_kernel_envelope)
        runtime = kernel.physical_runtime
        owners = physical_task_transaction_owners(task) if is_physical_task(task) else ()
        pair = self.pairs.get(task.task_id)
        reference = None if pair is None else self.records.get(pair[0])
        if pair is not None:
            identity, rename, source_owners, owners = pair
            self.remaining[identity] -= 1
            if not self.remaining[identity]:
                self.records.pop(identity, None)
            matches = (reference is not None and start == reference['start'] and all(
                self.tokens.get(rename.text(o)) == reference['tokens_before'][o] for o in source_owners))
            if matches and type(kernel.resource_available) is not dict:
                matches = False
            if matches:
                # Resource names change their hashes across ranks. A source
                # bill therefore cannot prove a follower's first-publication
                # dictionary order. Let the native writer establish new keys;
                # replay then only replaces existing values, preserving that
                # order in both the physical and scheduler calendars.
                for name, rows in reference['timeline_after'].items():
                    target = getattr(runtime.timeline, name)
                    if type(target) is not dict or any(rename.text(rid) not in target for rid in rows):
                        matches = False
                        break
                if matches:
                    matches = all(not values or rename.text(rid) in kernel.resource_available
                        for rid, values in reference['timeline_after']['lane_available'].items())
            if matches:
                kernel._reclaim_physical_allocations(start)
                saves = {o:runtime.allocators[o].begin_transaction() for o in owners}
                try:
                    prepared = prepare_physical_task(task, runtime, start)
                    inputs = [(op.value,address,count,owner) for _,op,address,count,owner,_ in prepared.instructions]
                    matches = inputs == rename.copy(reference['inputs'])
                    if matches:
                        keys = ('memory_accesses','physical_execution_by_owner','physical_arrival_ns',
                                'physical_completion_ns','physical_resource_intervals','physical_resource_last_intervals',
                                'physical_demands_resource_ids','physical_placeholder_resource_ids','physical_nonmemory_demand_ids')
                        metadata = {**task.metadata, **{k:rename.copy(reference['task'].metadata[k]) for k in keys
                                                       if k in reference['task'].metadata}}
                        resolved = _recost_physical_kernel_envelope(replace(task,
                            demands=rename.copy(reference['task'].demands),metadata=metadata))
                        _validate_event_timing(resolved, start, resolved.demands)
                        new_timeline = rename.copy(reference['timeline_after'])
                        lanes = runtime.timeline.lane_available
                        before = {rid:tuple(lanes.get(rid,())) for rid in new_timeline['lane_available']}
                        for o,save in saves.items():
                            runtime.allocators[o].commit_transaction(save)
                    else:
                        for o,save in saves.items():
                            runtime.allocators[o].rollback_transaction(save)
                except BaseException:
                    for o,save in saves.items():
                        runtime.allocators[o].rollback_transaction(save)
                    raise
                if matches:
                    for name,rows in new_timeline.items():
                        target = getattr(runtime.timeline,name)
                        for rid in reference['timeline_removed'][name]:
                            target.pop(rename.text(rid),None)
                        target.update(rows)
                    for name, value in reference['recorder_after'].items():
                        setattr(runtime.timeline, name, rename.copy(value))
                    for rid,values in new_timeline['lane_available'].items():
                        if values:
                            kernel.resource_available[rid] = min(values)
                    for owner in source_owners:
                        other = rename.text(owner)
                        self.pending[other] = (reference['cores_after'][owner],rename)
                        runtime.runtimes[other].clock_ns = reference['cores_after'][owner].clock_ns
                        self.tokens[other] = reference['tokens_after'][owner]
                    self.stats['mirrored_tasks'] += 1
                    return resolved,tuple(sorted(resolved.demands,key=lambda d:d.resource_id)),True,True,{
                        'timeline': {'lane_available': before}}
            self.stats['fallback_tasks'] += 1
        # Materialize only when a real native writer needs the follower Core,
        # most often a collective endpoint. Its calendar is already current.
        self.materialize(kernel, owners)
        if not owners:
            for owner in tuple(self.tokens):
                if any(_owner_resources(d.resource_id,(owner,)) for d in task.demands):
                    self.tokens.pop(owner,None)
        before_tokens = {o:self.tokens.get(o) for o in owners or ()}
        result = full(kernel,task,start,demands)
        if owners:
            self._advance_exact_tokens(result[0],owners)
        if task.task_id in self.representatives:
            resolved = result[0]
            touched = resolved.metadata['physical_resource_last_intervals']
            self.records[task.task_id] = {
                'task':resolved,'start':start,'tokens_before':before_tokens,
                'tokens_after':{o:self.tokens[o] for o in owners},
                'inputs':[(str(a['operation']),a['address'],a['byte_count'],
                           a.get('physical_owner',task.metadata.get('physical_owner')))
                          for a in resolved.metadata['memory_accesses']],
                'cores_after':{o:deepcopy(runtime.runtimes[o],{id(runtime.timeline):runtime.timeline}) for o in owners},
                'recorder_after': {name: deepcopy(getattr(runtime.timeline, name)) for name in (
                    '_touched', '_interval_limit', '_interval_count', '_intervals_truncated')},
                'timeline_after':{name:{rid:deepcopy(getattr(runtime.timeline,name)[rid]) for rid in touched
                                       if rid in getattr(runtime.timeline,name)} for name in _TP_TIMELINE_MAPS},
                'timeline_removed':{name:tuple(rid for rid in touched if rid not in getattr(runtime.timeline,name))
                                    for name in _TP_TIMELINE_MAPS}}
        return result

    def resolve(self,kernel,task,start,demands,full):
        if self.fast_exact:
            return self._resolve_exact(kernel, task, start, demands, full)
        from .data_motion import (physical_task_transaction_owners,prepare_physical_task,
                                  _recost_physical_kernel_envelope,is_physical_task)
        runtime = kernel.physical_runtime
        pair = self.pairs.get(task.task_id)
        reference = None if pair is None else self.records.get(pair[0])
        if pair is not None:
            self.remaining[pair[0]] -= 1
        if reference is not None:
            identity,rename,source_owners,owners = pair
            if not self.remaining[identity]:
                self.records.pop(identity,None)
            matches = start == reference['start'] and runtime._owners_have_native_hooks(owners)
            if matches:
                snapshot = runtime.begin_transaction(owners=owners,_owned_timeline=True)
                try:
                    prepared = prepare_physical_task(task,runtime,start)
                    inputs = [(op.value,address,count,owner)
                              for _,op,address,count,owner,config in prepared.instructions]
                    matches = _state_bits(inputs) == _state_bits(rename.copy(reference['inputs']))
                    if matches:
                        new_timeline = rename.copy(reference['timeline_after'])
                        keys = ('memory_accesses','physical_execution_by_owner','physical_arrival_ns',
                                'physical_completion_ns','physical_resource_intervals','physical_resource_last_intervals',
                                'physical_demands_resource_ids','physical_placeholder_resource_ids','physical_nonmemory_demand_ids')
                        metadata = {**task.metadata,**{k:rename.copy(reference['task'].metadata[k]) for k in keys
                                                     if k in reference['task'].metadata}}
                        resolved = _recost_physical_kernel_envelope(replace(task,
                            demands=rename.copy(reference['task'].demands),metadata=metadata))
                        _validate_event_timing(resolved, start, resolved.demands)
                        runtime.timeline._remember_resources(new_timeline['lane_available'])
                        runtime.commit_transaction(snapshot)
                    else:
                        runtime.restore(snapshot)
                except Exception:
                    runtime.restore(snapshot)
                    raise
                if matches:
                    for owner in source_owners:
                        other = rename.text(owner)
                        self.pending[other] = (reference['cores_after'][owner], rename)
                        runtime.runtimes[other].clock_ns = reference['cores_after'][owner].clock_ns
                        self.tokens[other] = reference['tokens_after'][owner]
                    for name,rows in new_timeline.items():
                        target = getattr(runtime.timeline,name)
                        for rid in reference['timeline_removed'][name]:
                            target.pop(rename.text(rid), None)
                        target.update(rows)
                    for rid,lanes in new_timeline['lane_available'].items():
                        if lanes:
                            kernel.resource_available[rid] = min(lanes)
                    self.stats['mirrored_tasks'] += 1
                    return resolved,tuple(sorted(resolved.demands,key=lambda d:d.resource_id)),True,True,snapshot
            self.stats['fallback_tasks'] += 1
        record = None
        owners = physical_task_transaction_owners(task) if is_physical_task(task) else ()
        if not owners:
            owners = frozenset(o for o in runtime.runtimes if any(
                _owner_resources(d.resource_id, (o,)) for d in task.demands))
        self.materialize(kernel, owners)
        if task.task_id in self.representatives and owners and runtime._owners_have_native_hooks(owners):
            record = {'start':start}
        result = full(kernel,task,start,demands)
        for owner in owners or ():
            self.serial += 1
            self.tokens[owner] = self.serial
        if record is not None:
            resolved = result[0]
            touched = resolved.metadata['physical_resource_last_intervals']
            record.update(task=resolved,inputs=[(str(a['operation']),a['address'],a['byte_count'],
                a.get('physical_owner',task.metadata.get('physical_owner'))) for a in resolved.metadata['memory_accesses']],
                cores_after={o:deepcopy(runtime.runtimes[o],{id(runtime.timeline):runtime.timeline}) for o in owners},
                timeline_after={name:{rid:deepcopy(getattr(runtime.timeline,name)[rid]) for rid in touched
                                      if rid in getattr(runtime.timeline,name)} for name in _TP_TIMELINE_MAPS},
                timeline_removed={name:tuple(rid for rid in touched if rid not in getattr(runtime.timeline,name))
                                  for name in _TP_TIMELINE_MAPS},
                tokens_after={o:self.tokens[o] for o in owners})
            self.records[task.task_id] = record
        return result


def _execute_window(source, all_specs, selected_ids, *, origin_ns, control, sample_id,
                    concurrent_imports=None, row_terms=None, tp_symmetry=None, capture_prefill_terms=False):
    """Run the selected tasks on an independent copy of ``source``.

    With ``row_terms`` (task id -> term key) the record also carries
    ``first_access_ns``: per term, how far the boundary's DRAM row state
    can move it (see native_row_bounds).
    """
    started = time.perf_counter()
    kernel, executor = _independent_kernel(
        source, None if concurrent_imports is None else concurrent_imports[0])
    imports = None
    if executor is not None:
        imports = _ConcurrentImports(kernel, executor, concurrent_imports[1])
        # Reach the window boundary with every earlier PD phase submitted.
        imports.advance_until(origin_ns)
    clone_wall = time.perf_counter()-started
    wanted = frozenset(selected_ids)
    selected = []
    omitted = set()
    inputs = {}
    for task in all_specs:
        if task.task_id not in wanted:
            continue
        omitted.update(parent for parent in task.dependencies if parent not in wanted)
        for access in task.metadata.get("memory_accesses", ()):
            if access.get("operation") != "read" or not access.get("buffer_id"):
                continue
            owner = access.get("physical_owner", task.metadata.get("physical_owner"))
            generation = access.get("allocation_generation", 0)
            key = (owner, access["buffer_id"], generation)
            allocator = kernel.physical_runtime.allocators.get(owner)
            exists = allocator is not None and allocator.get_allocation(access["buffer_id"], generation) is not None
            inputs.setdefault(key, {"physical_owner": owner, "buffer_id": access["buffer_id"],
                "generation": generation, "present_at_real_boundary": exists,
                "content_semantics": "declared_boundary_input_no_prior_write_event" if not exists else "real_post_ingress_allocation"})
        selected.append(replace(task, dependencies=tuple(parent for parent in task.dependencies if parent in wanted),
                                earliest_start_ns=max(float(task.earliest_start_ns), origin_ns)))
    selected = tuple(selected)
    if not selected:
        raise ValueError("native physical layer window is empty")
    mirror = None
    if tp_symmetry is not None:
        scenario, mode = tp_symmetry
        from .execution_control import native_callback_free_control
        if mode != "off" and (imports is None or mode == "all") and native_callback_free_control(control):
            mirror = _TPRankMirror(kernel, selected, scenario,
                                   approximate_imports=imports is not None and mode == "all",
                                   allow_ep_groups=mode in ("quiet", "all"))
            if mirror.pairs:
                kernel._native_tp_mirror = mirror
    # The source is read only throughout independent window execution.
    metrics = LiveCohortMetrics(resource_owners=kernel.resource_owners, origin_ns=origin_ns)
    first, last, count, physical = None, origin_ns, 0, 0
    layer_ends = {}
    layer_starts = {}
    stage_ends = {}
    stage_starts = {}
    prefill_terms = {"prefix_tasks": [], "layer_start_ns": {}} if capture_prefill_terms else None
    facts = [] if row_terms is not None else None
    def observe(event):
        nonlocal first, last, count, physical
        metrics.observe(event)
        if facts is not None:
            md = event.task.metadata
            owner = md.get("physical_owner")
            facts.append((event.task.task_id, event.start_ns, event.end_ns, event.task.dependencies,
                          tuple(demand.resource_id for demand in event.task.demands),
                          tuple((access.get("physical_owner", owner), access["address"], access["byte_count"])
                                for access in md.get("memory_accesses", ())
                                if type(access.get("address")) is int)))
        first = event.start_ns if first is None else min(first, event.start_ns)
        last = max(last, event.end_ns)
        layer_id = event.task.metadata.get("layer_id")
        if layer_id is not None:
            layer_starts[layer_id] = min(layer_starts.get(layer_id, event.start_ns), event.start_ns)
            layer_ends[layer_id] = max(layer_ends.get(layer_id, event.end_ns), event.end_ns)
        stage_id = event.task.metadata.get("execution_stage_id")
        if stage_id is not None:
            stage_starts[stage_id] = min(stage_starts.get(stage_id, event.start_ns), event.start_ns)
            stage_ends[stage_id] = max(stage_ends.get(stage_id, event.end_ns), event.end_ns)
        if prefill_terms is not None:
            md = event.task.metadata
            if layer_id is not None:
                starts = prefill_terms["layer_start_ns"]
                starts[layer_id] = min(starts.get(layer_id, event.start_ns), event.start_ns)
            else:
                prefill_terms["prefix_tasks"].append({
                    "node_id": event.task.task_id,
                    "stage_id": md.get("execution_stage_id"),
                    "dependencies": list(event.task.dependencies),
                    "resource_ids": [d.resource_id for d in event.demands],
                    "resource_lanes": dict(event.resource_lanes),
                    "demands": [{"resource_id": d.resource_id, "service_ns": d.service_ns} for d in event.demands],
                    "start_ns": event.start_ns, "end_ns": event.end_ns,
                    "duration_ns": event.end_ns - event.start_ns,
                    "earliest_start_ns": event.task.earliest_start_ns})
        count += 1
        physical += int(bool(event.task.metadata.get("physical_execution_by_owner")))
        if count == len(selected) or count % 256 == 0:
            try:
                control.raise_if_cancelled()
                control.report("native_layer_tasks", count, len(selected), unit="schedule_tasks",
                    simulated_time_ns=event.end_ns, message="正在执行原生完整层窗口",
                    metadata={"sample_id": sample_id, "full_cohort_committed": False,
                              "visible_output_tokens": 0})
            finally:
                frame = getattr(kernel, "_token_frame", None)
                if frame is not None:
                    frame.invalidate_frontiers()
    control.raise_if_cancelled()
    before = time.perf_counter()
    layout = CompiledGraphLayout.compile(selected)
    compiled_wall = time.perf_counter()-before
    physical_started = time.perf_counter()
    kernel._drain_prevalidated_compiled(selected, layout, event_observer=observe,
        index_physical_allocations=True, owned_physical_frontiers=True,
        before_step=None if imports is None else imports.before_step)
    physical_wall = time.perf_counter()-physical_started
    metadata_started = time.perf_counter()
    metadata = metrics.metadata()
    metadata_wall = time.perf_counter()-metadata_started
    terminal = wanted - {parent for task in selected for parent in task.dependencies}
    row_state = None if facts is None else first_access_ns(
        facts, {owner: active.core.config for owner, active in kernel.physical_runtime.runtimes.items()},
        row_terms)
    if capture_prefill_terms and facts is not None:
        from .native_prefill_access_bounds import per_access_ns
        prefill_terms["per_access_row_state_ns"] = per_access_ns(
            facts, {owner: active.core.config for owner, active in kernel.physical_runtime.runtimes.items()}, row_terms)
    carry = max(0., max(source.physical_runtime.timeline.ready_ns.values(), default=origin_ns)-origin_ns)
    return {"sample_id": sample_id, "window_id": sample_id,
        **({"prefill_terms": prefill_terms} if prefill_terms is not None else {}),
        "tp_symmetry": None if mirror is None else mirror.stats,
        "concurrent_imports": None if imports is None else imports.observation(first, last),
        "layer_start_ns": layer_starts, "layer_completion_ns": layer_ends,
        "stage_start_ns": stage_starts, "stage_completion_ns": stage_ends,
        "first_access_ns": row_state,
        "start_ns": first, "first_task_start_ns": first,
        "end_ns": last, "duration_ns": last-first, "boundary_origin_ns": origin_ns,
        "elapsed_from_boundary_ns": last-origin_ns, "boundary_wait_ns": first-origin_ns,
        "wall_seconds": time.perf_counter()-started,
        "wall_breakdown_seconds": {"source_kernel_copy": clone_wall,
            "compiled_subgraph": compiled_wall, "physical_tasks": physical_wall,
            "physical_metrics": metadata_wall},
        "task_count": count, "executed_task_count": count,
        "physical_task_count": physical,
        "visible_output_tokens": 0, "full_cohort_committed": False,
        "sample_state": "independent_native_kernel_copy_not_full_cohort_prefix",
        "physical_observations": {key: metadata.get(key) for key in (
            "storage_traffic", "dram_traffic", "sram_traffic", "resource_busy_ns", "category_time_ns")},
        "input_boundary": {"qualified": True,
                           "qualification_scope": "native_logical_inputs_of_complete_selected_subgraph; controller_phase_bounded_separately",
                           "initial_carry_upper_ns": carry,
                           "omitted_dependency_task_ids": sorted(omitted),
                           "logical_buffer_inputs": list(inputs.values()),
                           "predecessor_events_fabricated": False},
        "output_boundary": {"qualified": True,
            "qualification_scope": "all_selected_native_tasks_and_terminal_events_completed; not_in_situ_full_cohort_state",
            "terminal_task_ids": sorted(terminal),
            "resource_frontiers": dict(kernel.resource_available),
            "controller_state_resumable_as_full_cohort": False}}
