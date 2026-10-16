"""Bind logical HBF page staging to real host DRAM and fabric calendars.

The NAND core continues to receive only complete, sequential OCP page writes.
Host acceptance is explicitly a host-memory operation, never a NAND completion.
"""
from __future__ import annotations

from types import SimpleNamespace

from .memory_types import AccessRequest, DramConfig, Operation, TransactionResult, StageTiming
from .physical_contract import parse_execution_physical_config


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
        core._next_program_page[block] = max(core._next_program_page.get(block, 0), count)
    state["preloaded"].add(key)


def release(runtime, owner, buffer_id, generation, arrival_ns=0.0):
    state = runtime.hbf_staging.get(owner)
    if state is not None:
        state["controller"].release(buffer_id, generation, arrival_ns)
        state["preloaded"].discard((buffer_id, generation))


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
        result = active.core.submit(AccessRequest("hbf-staging:" + str(len(self.records)),
            operation, address, size, arrival))
        active.clock_ns = max(active.clock_ns, result.completion_ns)
        self.records.append((operation, owner, config, result))
        return result.completion_ns

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
        # Convert once in C. Iterating the array would create a NumPy scalar
        # per page and retain it in controller dictionaries, making ordinary
        # Python max/comparison operations unnecessarily expensive.
        return completions.tolist()

    def _route(self, route, size):
        from .data_motion import LinkService
        timeline = self.runtime.timeline
        key = (route, size)
        cached = self._routes.get(key)
        if cached is not None:
            return cached
        compiled = []
        for hop in self.binding["routes"][route]:
            service = LinkService(hop["link_id"], hop["resource_id"], hop["resource_id"],
                hop["bandwidth_gb_s"], hop["latency_ns"], hop.get("queue_depth", 1))
            for resource, capacity in service.resource_capacities.items():
                resource = timeline.canonical_resource_id(resource)
                existing = timeline.lane_available.get(resource)
                if existing is None:
                    timeline.lane_available[resource] = [timeline.available_ns(resource)] * capacity
                elif len(existing) != capacity:
                    raise ValueError("HBF staging link queue conflicts with shared calendar")
            combined = {}
            for demand in service.demands(size):
                resource = timeline.canonical_resource_id(demand.resource_id)
                row = combined.setdefault(resource, [0.0, 0])
                row[0] += demand.service_ns
                row[1] += demand.bytes_moved
            compiled.append((hop["resource_id"], hop.get("direction", route), tuple(
                (resource, values[0], values[1]) for resource, values in combined.items())))
        self._routes[key] = tuple(compiled)
        return self._routes[key]

    def transfer(self, route, size, arrival):
        timeline = self.runtime.timeline
        current = arrival
        for owner, direction, combined in self._route(route, size):
            start = max((current, *(timeline.available_ns(r) for r, _duration, _bytes in combined)))
            busy, moved, intervals, payloads, stages = {}, {}, {}, {}, []
            end = start
            for resource, duration, byte_count in combined:
                item = timeline.reserve(resource, start, duration,
                                        direction=direction)
                end = max(end, item.end_ns)
                timeline.bytes_moved[resource] = timeline.bytes_moved.get(resource, 0) + byte_count
                busy[resource], moved[resource] = duration, byte_count
                intervals[resource] = ((item.start_ns, item.end_ns),)
                payloads[resource] = ((item.start_ns, item.end_ns, byte_count),)
                stages.append(StageTiming("HOST_STAGING_LINK", item.start_ns, item.end_ns,
                                          resource, byte_count))
            result = TransactionResult("hbf-staging-link", Operation.READ, current, end,
                size, size, stages=tuple(stages), counters={
                    "resource_busy_ns": busy, "resource_bytes": moved,
                    "resource_intervals": intervals, "resource_interval_payloads": payloads,
                    "resource_last_intervals": {r: v[-1] for r, v in intervals.items()},
                    "internal_transfer_bytes": 0})
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
