"""Incremental credits match the pre-optimization scalar admission machine."""
from copy import deepcopy
from dataclasses import replace
from random import Random

import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_mapping import map_nand_address
from heterollm_sim.memory_types import AccessRequest, Operation
from test_generic_nand_resident import logical
from test_nand_channel_admission import settings as ocp_settings
from test_nand_generic_buffered_acceleration import PAGE, config
from test_nand_generic_request_accounting import numerical_bits


class ScalarAdmission(list):
    def __init__(self, commands, effective):
        super().__init__(commands)
        self.effective = effective
        self.reads = sum(op is Operation.READ for _, op in self)
        self.writes = sum(op is Operation.WRITE for _, op in self)
        self.population = self

    def preview(self, effective):
        return ScalarAdmission([(end, op) for end, op in self if end > effective], effective)


def scalar_admit(self, request, mapping=None):
    # Original scalar algorithm, independent of the production heaps/counts.
    host = None
    if self.config.queue_scope == 'host_channel':
        if mapping is None:
            mapping = map_nand_address(self.config, request.address)
        host = mapping.channel % self.config.host_channels
        population = self._host_inflight.get(host, ())
        acceptance = max(self._host_acceptance_ns.get(host, 0.), self._acceptance_ns)
    else:
        population, acceptance = self._inflight_operations, self._acceptance_ns
    effective = max(request.arrival_ns, acceptance)
    admitted = [(end, op) for end, op in population if end > effective]
    while True:
        blocked = []
        if len(admitted) >= self.config.max_outstanding_requests:
            blocked.append(min(end for end, _ in admitted))
        if request.operation is Operation.WRITE and self.config.max_outstanding_write_requests is not None:
            writes = [end for end, op in admitted if op is Operation.WRITE]
            if len(writes) >= self.config.max_outstanding_write_requests:
                blocked.append(min(writes))
        if not blocked:
            return effective, ScalarAdmission(admitted, effective), host
        effective = max(effective, max(blocked))
        admitted = [(end, op) for end, op in admitted if end > effective]


def scalar_record(self, host, effective, admitted, completion, operation):
    population = [*admitted, (completion, operation)]
    if host is None:
        self._acceptance_ns = effective
        self._inflight_operations = population
        self._inflight = [end for end, _ in population]
    else:
        self._host_acceptance_ns[host] = effective
        self._host_inflight[host] = population


def pair(settings, details=False):
    contexts = [PhysicalRuntimeContext(settings, 'flash', capture_details=details) for _ in range(2)]
    scalar = contexts[1].runtimes['flash'].core
    scalar._nand_numeric_enabled = False
    scalar._execute_accepted = lambda request: scalar._execute_detailed_accepted(request)
    scalar._admit_host_request = lambda request, mapping=None: scalar_admit(scalar, request, mapping)
    scalar._record_host_admission = lambda *args: scalar_record(scalar, *args)
    return contexts


def check(contexts, request):
    cores = [context.runtimes['flash'].core for context in contexts]
    actual, expected = [core.submit(request) for core in cores]
    counters = dict(actual.counters)
    counters.pop('native_fullpage_pages', None)
    assert numerical_bits(replace(actual, counters=counters)) == numerical_bits(expected)
    assert numerical_bits(logical(cores[0])) == numerical_bits(logical(cores[1]))
    return actual


@pytest.mark.parametrize('depth', [1, 3, 4096])
@pytest.mark.parametrize('host_scope', [False, True])
@pytest.mark.parametrize('write_cap', [False, True])
def test_incremental_admission_matches_original_scalar_at_full_queue(depth, host_scope, write_cap):
    settings = (ocp_settings if host_scope else config)(
        max_outstanding_requests=depth,
        max_outstanding_write_requests=max(1, depth // 2) if write_cap else None)
    contexts = pair(settings)
    rng = Random(134)
    commands = [(float(rng.randrange(1, 200)) + .125,
                 Operation.WRITE if i % 3 == 0 else Operation.READ) for i in range(depth)]
    for context in contexts:
        core = context.runtimes['flash'].core
        if host_scope:
            core._host_inflight = {host: list(commands) for host in range(settings.host_channels)}
        else:
            core._inflight_operations = list(commands)
            core._inflight = [end for end, _ in commands]
    for index in range(12):
        op = Operation.WRITE if index % 3 else Operation.READ
        # OCP requires complete units; generic exercises partial-page RMW.
        address = index % 2 * PAGE + (0 if host_scope else 17)
        check(contexts, AccessRequest(str(index), op, address, PAGE if host_scope else 99,
                                     (0., .125, 1000.)[index % 3]))


@pytest.mark.parametrize('transactional', [False, True])
@pytest.mark.parametrize('finish_inner', ['restore', 'commit_transaction'])
def test_incremental_queues_nested_transactions_public_restore_and_deepcopy(transactional, finish_inner):
    contexts = pair(config(max_outstanding_requests=3, max_outstanding_write_requests=1,
                           host_granularity_bytes=256))
    check(contexts, AccessRequest('seed', 'read', 0, 128 * PAGE, .125))
    before = [numerical_bits(logical(c.runtimes['flash'].core)) for c in contexts]
    outer = [c.begin_transaction(owners=('flash',), _owned_timeline=True) for c in contexts]
    check(contexts, AccessRequest('rmw', 'write', 17, 99, .125))
    inner = [c.begin_transaction(owners=('flash',), _owned_timeline=True) for c in contexts]
    saved = [c.snapshot(transactional=transactional) for c in contexts]
    queues = [deepcopy(c.runtimes['flash'].core._inflight_operations) for c in contexts]
    check(contexts, AccessRequest('branch', 'read', 33, 2 * PAGE, 0.))
    for c, snapshot, queue in zip(contexts, saved, queues):
        c.restore(snapshot)
        assert c.runtimes['flash'].core._inflight_operations == queue
    check(contexts, AccessRequest('after-restore', 'write', PAGE + 33, 99, .125))
    for c, token in zip(contexts, inner):
        getattr(c, finish_inner)(token)
    check(contexts, AccessRequest('tail', 'read', 0, 2 * PAGE))
    for c, token, state in zip(contexts, outer, before):
        c.restore(token)
        assert numerical_bits(logical(c.runtimes['flash'].core)) == state


def test_public_admission_lists_stay_detached_and_edits_are_observed():
    contexts = pair(config(max_outstanding_requests=3))
    check(contexts, AccessRequest('seed', 'read', 0, 2 * PAGE))
    old = [(c.runtimes['flash'].core._inflight_operations, c.runtimes['flash'].core._inflight)
           for c in contexts]
    saved = deepcopy(old)
    check(contexts, AccessRequest('next', 'write', 33, 99))
    assert old == saved
    for c in contexts:
        core = c.runtimes['flash'].core
        core._inflight_operations.append((1.e9, Operation.WRITE))
        core._inflight.append(-1.)  # This list is not an admission source.
    check(contexts, AccessRequest('edited', 'read', PAGE + 17, 99))


def test_failed_execution_does_not_retire_admitted_commands(monkeypatch):
    contexts = pair(config(max_outstanding_requests=1))
    check(contexts, AccessRequest('seed', 'write', 17, 99))
    before = [deepcopy(c.runtimes['flash'].core._inflight_operations) for c in contexts]

    def fail(request):
        raise RuntimeError('failed after admission')

    for c, state in zip(contexts, before):
        core = c.runtimes['flash'].core
        monkeypatch.setattr(core, '_execute_accepted', fail)
        with pytest.raises(RuntimeError, match='failed after admission'):
            core.submit(AccessRequest('failure', 'read', 0, PAGE))
        assert core._inflight_operations == state
        assert core._acceptance_ns == 0.


def test_deepcopy_continues_from_private_queues_and_public_snapshot_can_be_reused():
    context = PhysicalRuntimeContext(config(max_outstanding_requests=3), 'flash', capture_details=False)
    core = context.runtimes['flash'].core
    core.submit(AccessRequest('seed', 'read', 0, 128 * PAGE))
    core.submit(AccessRequest('partial', 'write', 17, 99))
    cloned = deepcopy(core)
    request = AccessRequest('continued', 'write', PAGE + 33, 99)
    assert numerical_bits(core.submit(request)) == numerical_bits(cloned.submit(request))
    assert numerical_bits(logical(core)) == numerical_bits(logical(cloned))
    saved = context.snapshot()
    original = numerical_bits(logical(core))
    for index in range(2):
        core.submit(AccessRequest(str(index), 'write', 2 * PAGE + 33, 99, 1.e5))
        context.restore(saved)
        assert numerical_bits(logical(core)) == original


@pytest.mark.parametrize('host_scope', [False, True])
def test_long_incremental_sequence_matches_scalar_without_publishing_each_request(host_scope):
    settings = (ocp_settings if host_scope else config)(
        max_outstanding_requests=4096, max_outstanding_write_requests=1024)
    contexts = pair(settings)
    fast, scalar = [c.runtimes['flash'].core for c in contexts]
    rng = Random(79)
    for index in range(4600):
        # Unequal completion times, ties, reads/writes/erases and backwards
        # arrival times exercise retirement and both independent host clocks.
        request = AccessRequest(str(index), (Operation.READ, Operation.WRITE, Operation.ERASE)[index % 3],
                                index % 2 * PAGE, PAGE, float(index % 13))
        a, b = [c._admit_host_request(request) for c in (fast, scalar)]
        assert a[0].hex() == b[0].hex()
        assert (len(a[1]), a[1].reads, a[1].writes, a[2]) == (len(b[1]), b[1].reads, b[1].writes, b[2])
        assert list(a[1]) == list(b[1])
        completion = a[0] + float(rng.randrange(1, 10) * 1.e5)
        for c, row in zip((fast, scalar), (a, b)):
            c._record_host_admission(row[2], row[0], row[1], completion, request.operation)
    assert fast._inflight_operations == scalar._inflight_operations
    assert fast._inflight == scalar._inflight
    assert fast._host_inflight == scalar._host_inflight
