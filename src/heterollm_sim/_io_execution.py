"""Shared physical IO execution, compiled with strict floating-point rules.

One source owns HOST task residency, HBF logical submission, and each exact
media/link handoff. Public callbacks remain Python methods; their hot state
and host scopes have typed C fields when the module is compiled.
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

from .memory_types import AccessRequest, DramConfig, Operation, TransactionResult, StageTiming
from ._compiled_dram import execute_compiled, _publish_numeric
from .hbf_staging_runtime import configure

try:
    import cython
except ImportError:
    class _PythonMode:
        Py_ssize_t = int
        bint = bool
        @staticmethod
        def cclass(cls):
            return cls
        @staticmethod
        def ccall(function):
            return function
        @staticmethod
        def declare(*args, **kwargs):
            return None
    cython = _PythonMode()


@cython.cclass
class HostTaskScope:
    """Keep one HOST controller numeric inside a physical task transaction.

    The owner must publish before an intersecting external reservation and
    before returning to user code. No public state is assumed immutable
    across tasks. Reports remain complete for each individual submission.
    """

    core = cython.declare(object, visibility="public")
    workspace = cython.declare(object, visibility="public")
    ids = cython.declare(tuple, visibility="public")
    resource_ids = cython.declare(object, visibility="public")
    disabled = cython.declare(cython.bint, visibility="public")
    layout = cython.declare(object, visibility="public")
    first_data = cython.declare(cython.Py_ssize_t, visibility="public")
    calendar_slot_ids = cython.declare(object, visibility="public")
    dirty_banks = cython.declare(object, visibility="public")
    dirty_resources = cython.declare(object, visibility="public")
    loaded = cython.declare(object, visibility="public")

    def __init__(self, core):
        from .dram_core import DramCore
        self.core = core
        self.workspace = None
        self.ids = ()
        self.resource_ids = frozenset()
        self.disabled = (type(core) is not DramCore or core.config.queue_scope != "device"
            or core.config.control_mode != "analytical" or core.capture_details
            or any(name in core.__dict__ for name in ("submit", "_execute_accepted", "_submit_validated",
                "_admission_time", "_can_accelerate", "_finite_guard_required"))
            or any(type(getattr(core.timeline, field)) is not dict for field in (
                "busy_ns", "bytes_moved", "ready_ns", "lane_available", "lane_directions",
                "directions", "last_intervals")))

    @cython.ccall
    def submit(self, request):
        from .memory_mapping import validate_dram_request
        if self.disabled:
            return None
        core = self.core
        validate_dram_request(request, core.config)
        burst = core.config.burst_bytes
        count = (request.address + request.byte_count - 1) // burst - request.address // burst + 1
        if count < 4 or core._finite_guard_required(request):
            self.publish()
            return None
        effective = core._admission_time(request)
        saved = None
        if self.workspace is not None:
            saved = (self.workspace.floats.copy(), self.workspace.arrays[0].copy(),
                     self.workspace.arrays[6].copy(), self.workspace.arrays[7].copy())
        try:
            result = execute_compiled(core, request, _submission=effective, _scope=self)
        except BaseException:
            if saved is not None:
                for target, values in zip((self.workspace.floats,
                        self.workspace.arrays[0], self.workspace.arrays[6], self.workspace.arrays[7]), saved):
                    target[:] = values
            raise
        if result is None:
            if saved is not None:
                for target, values in zip((self.workspace.floats,
                        self.workspace.arrays[0], self.workspace.arrays[6], self.workspace.arrays[7]), saved):
                    target[:] = values
            self.publish()
            return None
        core._acceptance_ns = effective
        core._inflight = [end for end in core._inflight if end > effective] + [result.completion_ns]
        core._scheduler_streak = (core._scheduler_streak + 1
                                 if core._scheduler_direction == request.operation.value else 1)
        core._scheduler_direction = request.operation.value
        return result

    @cython.ccall
    def publish(self):
        if self.workspace is None:
            return
        _publish_numeric(self.core, self.workspace.arrays, self.ids, self.first_data,
                         self.dirty_banks, self.dirty_resources)
        self.workspace = None


@lru_cache(maxsize=2048, typed=True)
def _link_description(link_id, resource_id, bandwidth, latency, depth, size):
    """Cache immutable demands; live aliases and lane capacities stay live."""
    from .data_motion import LinkService

    service = LinkService(link_id, resource_id, resource_id, bandwidth, latency, depth)
    return (tuple(service.resource_capacities.items()), tuple(
        (demand.resource_id, demand.service_ns, demand.bytes_moved)
        for demand in service.demands(size)))


@cython.cclass
class _HostTask:
    previous = cython.declare(object, visibility="public")
    cores = cython.declare(dict, visibility="public")
    active_core = cython.declare(object, visibility="public")

    def __init__(self, previous):
        self.previous = previous
        self.cores = {}
        self.active_core = None

    @cython.ccall
    def flush(self, resources=None):
        scope: HostTaskScope
        for scope in self.cores.values():
            if scope.workspace is None:
                continue
            if resources is None or not scope.resource_ids.isdisjoint(resources):
                scope.publish()
                continue
            for resource in resources:
                if id(scope.core.timeline.lane_available.get(resource)) in scope.calendar_slot_ids:
                    scope.publish()
                    break

    @cython.ccall
    def submit(self, core, request):
        scope: HostTaskScope
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


@cython.cclass
class _IOExecutionState:
    runtime = cython.declare(object, visibility="public")
    owner = cython.declare(object, visibility="public")
    state = cython.declare(object, visibility="public")
    binding = cython.declare(object, visibility="public")
    records = cython.declare(list, visibility="public")
    routes = cython.declare(dict, visibility="public")
    record_base = cython.declare(cython.Py_ssize_t, visibility="public")

    def __init__(self, runtime, owner, state, binding, records):
        self.runtime, self.owner, self.state = runtime, owner, state
        self.binding, self.records = binding, records
        self.routes = {}
        self.record_base = 0


class _PhysicalIO:
    """Observable adapter; the execution state and helpers are native fields."""
    __slots__ = ("_execution",)

    def __init__(self, runtime, owner, state, binding, records):
        self._execution = _IOExecutionState(runtime, owner, state, binding, records)

    @property
    def runtime(self):
        return self._execution.runtime

    @property
    def owner(self):
        return self._execution.owner

    @property
    def state(self):
        return self._execution.state

    @state.setter
    def state(self, value):
        self._execution.state = value

    @property
    def binding(self):
        return self._execution.binding

    @property
    def records(self):
        return self._execution.records

    @property
    def _routes(self):
        return self._execution.routes

    @property
    def _record_base(self):
        return self._execution.record_base

    def begin_access(self):
        return _io_begin_access(self, self._execution)

    def _memory(self, owner, config, operation, address, size, arrival):
        return _io_memory(self, self._execution, owner, config, operation, address, size, arrival)

    def _flush_before_media(self, core, operation, link_resources=(), *, request=None):
        return _io_flush_before_media(self, self._execution, core, operation, link_resources, request=request)

    def read_host(self, address, size, arrival):
        return _io_read_host(self, self._execution, address, size, arrival)

    def write_host(self, address, size, arrival):
        return _io_write_host(self, self._execution, address, size, arrival)

    def read_flash(self, address, size, arrival):
        return _io_read_flash(self, self._execution, address, size, arrival)

    def write_flash(self, address, size, arrival):
        return _io_write_flash(self, self._execution, address, size, arrival)

    def read_flash_pages(self, addresses, arrival):
        return _io_read_flash_pages(self, self._execution, addresses, arrival)

    def write_flash_pages(self, addresses, arrivals):
        return _io_write_flash_pages(self, self._execution, addresses, arrivals)

    def _flash_pages(self, operation, addresses, arrivals, route):
        return _io_flash_pages(self, self._execution, operation, addresses, arrivals, route)

    def _route(self, route, size):
        return _io_route(self, self._execution, route, size)

    def transfer(self, route, size, arrival):
        return _io_transfer(self, self._execution, route, size, arrival)


@cython.ccall
def _io_begin_access(io, execution: _IOExecutionState):
    execution.record_base = len(execution.records)
    execution.routes.clear()

@cython.ccall
def _io_memory(io, execution: _IOExecutionState, owner, config, operation, address, size, arrival):
    active = execution.runtime.runtimes[owner]
    request = AccessRequest("hbf-staging:" + str(len(execution.records) - execution.record_base),
                            operation, address, size, arrival)
    scope = getattr(execution.runtime, "_hbf_host_task", None)
    if scope is not None and not isinstance(config, DramConfig):
        io._flush_before_media(active.core, operation, request=request)
    result = scope.submit(active.core, request) if scope is not None and isinstance(config, DramConfig) else None
    if result is None:
        if scope is not None and isinstance(config, DramConfig):
            scope.flush()
        result = active.core.submit(request)
    active.clock_ns = max(active.clock_ns, result.completion_ns)
    execution.records.append((operation, owner, config, result))
    return result.completion_ns

def _io_flush_before_media(io, execution: _IOExecutionState, core, operation, link_resources=(), *, request=None):
    scope = getattr(execution.runtime, "_hbf_host_task", None)
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
    banks = config.array_units
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
    from ._nand_execution import _geometry_plan
    units = banks if config.planes_independent else base
    plan = _geometry_plan(core, operation, banks, units, base // config.channels)
    scope.flush((*plan[2], *plan[6], *link_resources))

@cython.ccall
def _io_read_host(io, execution: _IOExecutionState, address, size, arrival):
    return io._memory(execution.state["host_owner"], execution.state["host_config"],
        Operation.READ, execution.state["host_base"] + address, size, arrival)

@cython.ccall
def _io_write_host(io, execution: _IOExecutionState, address, size, arrival):
    return io._memory(execution.state["host_owner"], execution.state["host_config"],
        Operation.WRITE, execution.state["host_base"] + address, size, arrival)

@cython.ccall
def _io_read_flash(io, execution: _IOExecutionState, address, size, arrival):
    return io._memory(execution.owner, execution.state["config"], Operation.READ,
                        address, size, arrival)

@cython.ccall
def _io_write_flash(io, execution: _IOExecutionState, address, size, arrival):
    return io._memory(execution.owner, execution.state["config"], Operation.WRITE,
                        address, size, arrival)

@cython.ccall
def _io_read_flash_pages(io, execution: _IOExecutionState, addresses, arrival):
    """Compile an unchanged NAND/link page sequence when supported."""
    return io._flash_pages(Operation.READ, addresses, arrival, "read_hbf")

@cython.ccall
def _io_write_flash_pages(io, execution: _IOExecutionState, addresses, arrivals):
    """Keep each initiator transfer immediately before its page program."""
    return io._flash_pages(Operation.WRITE, addresses, arrivals, "write_hbf")

def _io_flash_pages(io, execution: _IOExecutionState, operation, addresses, arrivals, route):
    if execution.runtime.capture_details:
        return None
    active = execution.runtime.runtimes[execution.owner]
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
    timeline = execution.runtime.timeline
    link_resources = {timeline.canonical_resource_id(resource)
                      for hop in execution.binding["routes"][route]
                      for resource in (hop["resource_id"], hop["resource_id"] + ".inflight")}
    io._flush_before_media(active.core, operation, link_resources)
    if (any(resource.startswith(prefixes) for resource in link_resources)
            or any(resource.startswith(prefixes)
                   and timeline.canonical_resource_id(resource) in link_resources
                   for resource in timeline.resource_aliases)):
        return None
    program = io._route(route, 4096)
    result = submit_pages(addresses, arrivals, link_program=program)
    if result is None:
        return None
    completions, media, links = result
    active.clock_ns = max(active.clock_ns, media.completion_ns)
    execution.records.append((operation, execution.owner, execution.state["config"], media))
    for owner, link in links:
        # Like scalar _io_transfer in aggregate mode, the stream retains
        # only the final reservation per resource, not its interval history.
        link.counters["intervals_truncated"] = bool(link.counters.get("resource_last_intervals"))
        execution.records.append((Operation.READ, owner, SimpleNamespace(kind="LINK"), link))
    # Keep the stream numeric for vectorized mapping/dependency updates.
    # Scalar controller paths convert once before iterating in Python.
    return completions

@cython.ccall
def _io_route(io, execution: _IOExecutionState, route, size):
    timeline = execution.runtime.timeline
    key = (route, size)
    cached = execution.routes.get(key)
    if cached is not None:
        return cached
    compiled = []
    for hop in execution.binding["routes"][route]:
        capacities, demands = _link_description(hop["link_id"], hop["resource_id"],
            hop["bandwidth_gb_s"], hop["latency_ns"], hop.get("queue_depth", 1), size)
        for resource, capacity in capacities:
            resource = timeline.canonical_resource_id(resource)
            scope = getattr(execution.runtime, "_hbf_host_task", None)
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
        compiled.append((hop["resource_id"], hop.get("direction", route), tuple([
            (resource, values[0], values[1]) for resource, values in combined.items()])))
    execution.routes[key] = tuple(compiled)
    return execution.routes[key]

@cython.ccall
def _io_transfer(io, execution: _IOExecutionState, route, size, arrival):
    timeline = execution.runtime.timeline
    capture_details = execution.runtime.capture_details
    current = arrival
    for owner, direction, combined in io._route(route, size):
        scope = getattr(execution.runtime, "_hbf_host_task", None)
        if scope is not None:
            scope.flush(tuple([resource for resource, _, _ in combined]))
        start = current
        for resource, _duration, _bytes in combined:
            candidate = timeline.available_ns(resource)
            if candidate > start:
                start = candidate
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
        execution.records.append((Operation.READ, owner, SimpleNamespace(kind="LINK"), result))
        current = end
    return current




@cython.cclass
class PreparedStagingIO:
    """Run a task's staged accesses directly into its physical ledger.

    HOST acceptance remains a logical completion boundary. Every physical
    Core/link result still enters the ledger in its original submission order.
    """
    runtime = cython.declare(object, visibility="public")
    owner = cython.declare(object, visibility="public")
    config = cython.declare(object, visibility="public")
    binding = cython.declare(object, visibility="public")
    ledger = cython.declare(object, visibility="public")
    io = cython.declare(object, visibility="public")

    def __init__(self, runtime, owner, binding, ledger):
        self.runtime, self.owner, self.binding, self.ledger = runtime, owner, binding, ledger
        state = runtime.hbf_staging[owner]
        self.config = state["config"]
        self.io = _PhysicalIO(runtime, owner, state, binding, ledger.records)

    @cython.ccall
    def submit(self, access, operation, byte_count, arrival_ns):
        from .memory_types import _nonnegative
        # The removed logical AccessRequest performed this check before HOST
        # staging could mutate its controller or reserve an authored route.
        _nonnegative("arrival_ns", arrival_ns)
        runtime, owner, ledger = self.runtime, self.owner, self.ledger
        # Keep authored configuration edits visible at access boundaries.
        state = configure(runtime, owner, self.config, self.binding)
        buffer_id = access.get("buffer_id")
        if not buffer_id:
            raise ValueError("host-staged HBF access requires a logical buffer identity")
        allocator = runtime.allocators[owner]
        buffer_id, generation, offset = allocator.canonical_range(buffer_id,
            access.get("offset_bytes", 0), byte_count,
            access.get("allocation_generation", 0))
        io = self.io
        io.state = state
        io.begin_access()
        controller = state["controller"]
        before = dict(controller.stats)
        if operation is Operation.WRITE:
            end = controller.write(buffer_id, generation, offset, byte_count, arrival_ns, io)
        elif operation is Operation.READ:
            end = controller.read(buffer_id, generation, offset, byte_count, arrival_ns, io)
        else:
            raise ValueError("host-staged HBF exposes logical reads and writes only")
        runtime.runtimes[owner].clock_ns = max(runtime.runtimes[owner].clock_ns, end)
        stats = ledger.staging_stats.setdefault(owner, {})
        for key, value in controller.stats.items():
            if key not in {"host_occupied_bytes", "host_reserved_bytes",
                           "media_occupied_bytes", "media_reserved_bytes"}:
                stats[key] = stats.get(key, 0) + (value - before.get(key, 0))
        stats.update(host_occupied_bytes=controller.host_occupied_bytes,
                     host_reserved_bytes=controller.host_reserved_bytes,
                     media_occupied_bytes=controller.occupied_bytes,
                     media_reserved_bytes=controller.reserved_bytes)
        ledger.accept(operation, end)
        return end
