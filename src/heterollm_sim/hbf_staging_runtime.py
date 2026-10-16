"""Bind logical HBF page staging to real host DRAM and fabric calendars.

The NAND core continues to receive only complete, sequential OCP page writes.
Host acceptance is explicitly a host-memory operation, never a NAND completion.
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

from .memory_types import AccessRequest, DramConfig, Operation, TransactionResult, StageTiming
from .physical_contract import parse_execution_physical_config
from ._compiled_dram import HostTaskScope


@lru_cache(maxsize=2048, typed=True)
def _link_description(link_id, resource_id, bandwidth, latency, depth, size):
    """Cache immutable demands; live aliases and lane capacities stay live."""
    from .data_motion import LinkService

    service = LinkService(link_id, resource_id, resource_id, bandwidth, latency, depth)
    return (tuple(service.resource_capacities.items()), tuple(
        (demand.resource_id, demand.service_ns, demand.bytes_moved)
        for demand in service.demands(size)))


def configure(runtime, owner, config, binding):
    from .hbf_staging import HbfStagingController

    host_config = parse_execution_physical_config(binding["host_config"])
    host_owner = binding["host_owner"]
    capacity = binding["capacity_bytes"]
    if not isinstance(host_config, DramConfig) or host_config.kind.value == "HBM":
        raise ValueError("HBF host staging requires physical HOST DRAM")
    if host_owner == owner:
        raise ValueError("HBF staging and media require distinct owners")
    if binding.get("reserve_blocks", 1) != 1:
        raise ValueError("HBF host staging reserves exactly one physical erase block")
    identity = (host_owner, host_config, capacity)
    current = runtime.hbf_staging.get(owner)
    if current is not None:
        if current["identity"] != identity or current["config"] != config:
            raise ValueError("conflicting HBF host staging configuration for " + owner)
        return current
    runtime.runtime(config, owner)
    runtime.runtime(host_config, host_owner)
    controller = HbfStagingController(config, capacity)
    allocation = runtime.allocators[host_owner].allocate(
        "@hbf-host-staging:" + owner, capacity, allocation_arena="resident")
    current = {"identity": identity, "config": config, "controller": controller,
               "host_base": allocation.base_address, "host_owner": host_owner,
               "host_config": host_config, "preloaded": set()}
    runtime.hbf_staging[owner] = current
    return current


def preload(runtime, owner, buffer_id, generation, binding):
    state = runtime.hbf_staging[owner]
    allocator = runtime.allocators[owner]
    buffer_id, generation, _ = allocator.canonical_range(buffer_id, 0, 1, generation)
    key = (buffer_id, generation)
    if key in state["preloaded"]:
        return
    allocation = allocator.lookup(buffer_id, generation)
    controller = state["controller"]
    controller.preload(buffer_id, generation, allocation.size_bytes)
    # Warm weights were loaded before the measured request. Preserve their
    # physical program frontier so later writes cannot overwrite them.
    from .memory_mapping import map_nand_address
    core = runtime.runtimes[owner].core
    for address, count in controller.preload_page_counts(buffer_id, generation):
        mapping = map_nand_address(core.config, address)
        block = core._block_key(mapping)
        core._set_program_frontier(block, max(core._next_program_page.get(block, 0), count))
    state["preloaded"].add(key)


def release(runtime, owner, buffer_id, generation, arrival_ns=0.0):
    state = runtime.hbf_staging.get(owner)
    if state is not None:
        state["controller"].release(buffer_id, generation, arrival_ns)
        state["preloaded"].discard((buffer_id, generation))


class _HostTask:
    def __init__(self, previous):
        self.previous = previous
        self.cores = {}
        self.active_core = None

    def flush(self, resources=None):
        for scope in self.cores.values():
            if scope.workspace is None:
                continue
            if (resources is None or not scope.resource_ids.isdisjoint(resources)
                    or any(id(scope.core.timeline.lane_available.get(resource))
                           in scope.calendar_slot_ids for resource in resources)):
                scope.publish()

    def submit(self, core, request):
        if core is not self.active_core:
            self.flush()
            self.active_core = core
        scope = self.cores.get(core)
        if scope is None:
            # Distinct authored owners can deliberately share resource IDs.
            # Their handoff must see every previously accepted HOST update.
            scope = self.cores[core] = HostTaskScope(core)
        return scope.submit(request)


def begin_task(runtime):
    """Enter within the physical task's existing rollback transaction."""
    previous = getattr(runtime, "_hbf_host_task", None)
    if previous is not None:
        previous.flush()
    token = None if runtime.capture_details else _HostTask(previous)
    runtime._hbf_host_task = token
    return token


def flush_task(runtime):
    """Publish before a direct core submission outside HBF staging."""
    scope = getattr(runtime, "_hbf_host_task", None)
    if scope is not None:
        scope.flush()


def end_task(runtime, token, *, success=True):
    if token is None:
        return
    try:
        if success:
            token.flush()
    finally:
        runtime._hbf_host_task = token.previous


class _PhysicalIO:
    def __init__(self, runtime, owner, state, binding, records):
        self.runtime, self.owner, self.state = runtime, owner, state
        self.binding, self.records = binding, records
        self._routes = {}

    def _memory(self, owner, config, operation, address, size, arrival):
        # configure() has already checked both owners' complete contracts.
        # Re-normalizing and validating the same geometry per 4 KiB page is
        # unnecessary; all submissions still use the real persistent core.
        active = self.runtime.runtimes[owner]
        request = AccessRequest("hbf-staging:" + str(len(self.records)), operation, address, size, arrival)
        scope = getattr(self.runtime, "_hbf_host_task", None)
        if scope is not None and not isinstance(config, DramConfig):
            self._flush_before_media(active.core, operation, request=request)
        result = scope.submit(active.core, request) if scope is not None and isinstance(config, DramConfig) else None
        if result is None:
            if scope is not None and isinstance(config, DramConfig):
                scope.flush()
            result = active.core.submit(request)
        active.clock_ns = max(active.clock_ns, result.completion_ns)
        self.records.append((operation, owner, config, result))
        return result.completion_ns

    def _flush_before_media(self, core, operation, link_resources=(), *, request=None):
        scope = getattr(self.runtime, "_hbf_host_task", None)
        if scope is None or not any(item.workspace is not None for item in scope.cores.values()):
            return
        from .nand_core import NandCore
        if (type(core) is not NandCore or core.config.protocol_mode != "ocp_hbf_0_7"
                or any(name in core.__dict__ for name in (
                    "submit", "submit_read_pages", "submit_write_pages", "_array_id",
                    "_array_resource_id", "_channel_id", "_host_id"))):
            scope.flush()
            return
        # Use the same physical resource plan as the media executor. Array
        # names are included as well as reservation slots: the core publishes
        # readiness under the raw array ID even when its slot is aliased.
        config = core.config
        base = config.channels * config.targets_per_channel * config.dies_per_target * config.luns_per_die
        banks = base * config.planes_per_lun
        if request is not None:
            from .memory_mapping import map_nand_address
            first = request.address // config.page_bytes
            stop = (request.address + request.byte_count + config.page_bytes - 1) // config.page_bytes
            resources = set(link_resources)
            # Resource coordinates repeat after one bank cycle. A scalar
            # access only resolves names it actually uses, as submit does.
            for page in range(first, min(stop, first + banks)):
                mapping = map_nand_address(config, page * config.page_bytes)
                resources.update((core._array_id(mapping), core._array_resource_id(mapping),
                                  core._channel_id(mapping), core._host_id(mapping, operation)))
            scope.flush(resources)
            return
        if banks > 65536 or config.effective_capacity_bytes < banks * config.page_bytes:
            scope.flush()
            return
        from ._compiled_nand import _geometry_plan
        units = banks if config.planes_independent else base
        plan = _geometry_plan(core, operation, banks, units, base // config.channels)
        scope.flush((*plan[2], *plan[6], *link_resources))

    def read_host(self, address, size, arrival):
        return self._memory(self.state["host_owner"], self.state["host_config"],
            Operation.READ, self.state["host_base"] + address, size, arrival)

    def write_host(self, address, size, arrival):
        return self._memory(self.state["host_owner"], self.state["host_config"],
            Operation.WRITE, self.state["host_base"] + address, size, arrival)

    def read_flash(self, address, size, arrival):
        return self._memory(self.owner, self.state["config"], Operation.READ,
                            address, size, arrival)

    def write_flash(self, address, size, arrival):
        return self._memory(self.owner, self.state["config"], Operation.WRITE,
                            address, size, arrival)

    def read_flash_pages(self, addresses, arrival):
        """Compile an unchanged NAND/link page sequence when supported."""
        return self._flash_pages(Operation.READ, addresses, arrival, "read_hbf")

    def write_flash_pages(self, addresses, arrivals):
        """Keep each initiator transfer immediately before its page program."""
        return self._flash_pages(Operation.WRITE, addresses, arrivals, "write_hbf")

    def _flash_pages(self, operation, addresses, arrivals, route):
        if self.runtime.capture_details:
            return None
        active = self.runtime.runtimes[self.owner]
        submit_pages = getattr(active.core,
            "submit_read_pages" if operation is Operation.READ else "submit_write_pages", None)
        if submit_pages is None:
            return None
        # Preparing distinct fabric lanes before submission is harmless. A
        # fabric alias of a media resource must retain ordinary interleaving,
        # including its original lane-capacity validation point.
        metadata = active.core.config.metadata
        prefixes = tuple(str(metadata[name]) for name in (
            "array_resource_prefix", "parallel_resource_prefix",
            "channel_resource_prefix", "host_resource_id"))
        timeline = self.runtime.timeline
        link_resources = {timeline.canonical_resource_id(resource)
                          for hop in self.binding["routes"][route]
                          for resource in (hop["resource_id"], hop["resource_id"] + ".inflight")}
        self._flush_before_media(active.core, operation, link_resources)
        if (any(resource.startswith(prefixes) for resource in link_resources)
                or any(resource.startswith(prefixes)
                       and timeline.canonical_resource_id(resource) in link_resources
                       for resource in timeline.resource_aliases)):
            return None
        program = self._route(route, 4096)
        result = submit_pages(addresses, arrivals, link_program=program)
        if result is None:
            return None
        completions, media, links = result
        active.clock_ns = max(active.clock_ns, media.completion_ns)
        self.records.append((operation, self.owner, self.state["config"], media))
        for owner, link in links:
            self.records.append((Operation.READ, owner, SimpleNamespace(kind="LINK"), link))
        # Keep the stream numeric for vectorized mapping/dependency updates.
        # Scalar controller paths convert once before iterating in Python.
        return completions

    def _route(self, route, size):
        timeline = self.runtime.timeline
        key = (route, size)
        cached = self._routes.get(key)
        if cached is not None:
            return cached
        compiled = []
        for hop in self.binding["routes"][route]:
            capacities, demands = _link_description(hop["link_id"], hop["resource_id"],
                hop["bandwidth_gb_s"], hop["latency_ns"], hop.get("queue_depth", 1), size)
            for resource, capacity in capacities:
                resource = timeline.canonical_resource_id(resource)
                scope = getattr(self.runtime, "_hbf_host_task", None)
                if scope is not None:
                    scope.flush((resource,))
                existing = timeline.lane_available.get(resource)
                if existing is None:
                    timeline.lane_available[resource] = [timeline.available_ns(resource)] * capacity
                elif len(existing) != capacity:
                    raise ValueError("HBF staging link queue conflicts with shared calendar")
            combined = {}
            for resource_id, duration, byte_count in demands:
                resource = timeline.canonical_resource_id(resource_id)
                row = combined.setdefault(resource, [0.0, 0])
                row[0] += duration
                row[1] += byte_count
            compiled.append((hop["resource_id"], hop.get("direction", route), tuple(
                (resource, values[0], values[1]) for resource, values in combined.items())))
        self._routes[key] = tuple(compiled)
        return self._routes[key]

    def transfer(self, route, size, arrival):
        timeline = self.runtime.timeline
        capture_details = self.runtime.capture_details
        current = arrival
        for owner, direction, combined in self._route(route, size):
            scope = getattr(self.runtime, "_hbf_host_task", None)
            if scope is not None:
                scope.flush(tuple(resource for resource, _, _ in combined))
            start = max((current, *(timeline.available_ns(r) for r, _duration, _bytes in combined)))
            busy, moved, last, intervals, payloads, stages = {}, {}, {}, {}, {}, []
            end = start
            for resource, duration, byte_count in combined:
                item = timeline.reserve(resource, start, duration,
                                        direction=direction)
                end = max(end, item.end_ns)
                timeline.bytes_moved[resource] = timeline.bytes_moved.get(resource, 0) + byte_count
                busy[resource], moved[resource] = duration, byte_count
                last[resource] = (item.start_ns, item.end_ns)
                if capture_details:
                    intervals[resource] = (last[resource],)
                    payloads[resource] = ((item.start_ns, item.end_ns, byte_count),)
                    stages.append(StageTiming("HOST_STAGING_LINK", item.start_ns, item.end_ns,
                                              resource, byte_count))
            result = TransactionResult("hbf-staging-link", Operation.READ, current, end,
                size, size, stages=tuple(stages), counters={
                    "resource_busy_ns": busy, "resource_bytes": moved,
                    "resource_intervals": intervals, "resource_interval_payloads": payloads,
                    "resource_last_intervals": last,
                    "internal_transfer_bytes": 0,
                    **({"intervals_truncated": bool(combined)} if not capture_details else {})})
            self.records.append((Operation.READ, owner, SimpleNamespace(kind="LINK"), result))
            current = end
        return current


def submit(runtime, owner, config, binding, access, request):
    state = configure(runtime, owner, config, binding)
    buffer_id = access.get("buffer_id")
    if not buffer_id:
        raise ValueError("host-staged HBF access requires a logical buffer identity")
    allocator = runtime.allocators[owner]
    buffer_id, generation, offset = allocator.canonical_range(buffer_id,
        access.get("offset_bytes", 0), request.byte_count,
        access.get("allocation_generation", 0))
    records = []
    io = _PhysicalIO(runtime, owner, state, binding, records)
    controller = state["controller"]
    before = dict(controller.stats)
    if request.operation is Operation.WRITE:
        end = controller.write(buffer_id, generation, offset, request.byte_count,
                               request.arrival_ns, io)
    elif request.operation is Operation.READ:
        end = controller.read(buffer_id, generation, offset, request.byte_count,
                              request.arrival_ns, io)
    else:
        raise ValueError("host-staged HBF exposes logical reads and writes only")
    runtime.runtimes[owner].clock_ns = max(runtime.runtimes[owner].clock_ns, end)
    stats = {key: value - before.get(key, 0) for key, value in controller.stats.items()}
    stats.update(host_occupied_bytes=controller.host_occupied_bytes,
                 host_reserved_bytes=controller.host_reserved_bytes,
                 media_occupied_bytes=controller.occupied_bytes,
                 media_reserved_bytes=controller.reserved_bytes)
    # This is a logical acceptance boundary. Only the records below count
    # physical bytes; a buffered write is not recorded as a NAND transaction.
    result = TransactionResult(request.request_id, request.operation,
        request.arrival_ns, end, request.byte_count, 0,
        counters={"internal_transfer_bytes": 0, "host_staging": stats})
    return result, records


def snapshot(states, transactional=False):
    return {owner: {**state, "preloaded": set(state["preloaded"]),
        "saved_controller": state["controller"].begin_transaction() if transactional else state["controller"].snapshot(),
        "controller_transaction": transactional} for owner, state in states.items()}


def commit(saved):
    for value in saved.values():
        if value["controller_transaction"]:
            value["controller"].commit_transaction(value["saved_controller"])


def restore(saved):
    states = {}
    for owner, value in saved.items():
        state = {key: item for key, item in value.items()
                 if key not in {"saved_controller", "controller_transaction"}}
        state["preloaded"] = set(value["preloaded"])
        if value["controller_transaction"]:
            state["controller"].rollback_transaction(value["saved_controller"])
        else:
            state["controller"].restore(value["saved_controller"])
        states[owner] = state
    return states
