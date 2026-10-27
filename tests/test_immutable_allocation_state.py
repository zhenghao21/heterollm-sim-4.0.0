"""Private allocation rollback maps preserve native extents and copy overrides."""
import copy
import math
from unittest.mock import patch

import pytest

from heterollm_sim import _timeline_history as candidate, native_layer_sampling as sampling
from heterollm_sim.memory_allocator import PhysicalAllocation
from test_native_external_kv import native_runtime, ingress_scenario
from test_native_concurrent_imports import _executor_state
from test_token_owned_history import bits, compare_physical_state

def fixture(case, large=0):
    runtime = native_runtime(ingress_scenario(case, count=2))
    runtime._arrive()
    source = runtime._execution_resource_kernel
    if large:
        allocator = max(source.physical_runtime.allocators.values(), key=lambda item: item.capacity_bytes)
        for index in range(large):
            name = f'allocation-history-native-{index:06d}'
            allocator.allocate(name, allocator.alignment_bytes, 0)
            if index % 16 == 0:
                allocator.allocate(name + ':alias', allocator.alignment_bytes, 0, alias_of=name)
    runtime._start_external_imports()
    return (runtime, source, runtime._external_executor)

def histories(executor):
    seen = set()

    def visit(snapshot):
        if id(snapshot) in seen:
            return
        seen.add(id(snapshot))
        yield snapshot
        for state in snapshot['async_plan_states'].values():
            if 'rollback_snapshot' in state:
                yield from visit(state['rollback_snapshot'])
    for state in executor._async_plan_states.values():
        if 'rollback_snapshot' in state:
            yield from visit(state['rollback_snapshot'])

def source_bits(kernel, executor):
    saved = kernel.physical_runtime.snapshot()
    return (bits(saved['timeline']), bits(saved['allocator_state']), _executor_state(executor))

def compare_history(a, b):
    left, right = (list(histories(a)), list(histories(b)))
    assert len(left) == len(right)
    for first, second in zip(left, right):
        assert bits(first['runtime']['allocator_state']) == bits(second['runtime']['allocator_state'])

@pytest.mark.parametrize('case', ('hbm_tp1', 'hbf_weights_kv_tp1'))
def test_native_allocation_history_clone_restore_and_continue(case):
    with patch.object(candidate, '_freeze_allocation_state', lambda state, allocators=None: state):
        plain, source_a, a = fixture(case)
    frozen, source_b, b = fixture(case)
    assert all((type(value) is dict for value in source_b.physical_runtime.snapshot()['allocator_state'].values()))
    assert all((type(value) is dict for value in b.snapshot()['runtime']['allocator_state'].values()))
    tables = [table for save in histories(b) for table in save['runtime']['allocator_state'].values() if type(table) is candidate._ImmutableAllocationColumn]
    assert tables
    compare_history(a, b)
    before_a, before_b = (source_bits(source_a, a), source_bits(source_b, b))
    for repeat in range(2):
        kernel_a, copy_a = sampling._independent_kernel(source_a, a)
        kernel_b, copy_b = sampling._independent_kernel(source_b, b)
        compare_physical_state(kernel_a, kernel_b)
        compare_history(copy_a, copy_b)
        for original, cloned in zip(histories(b), histories(copy_b)):
            for owner, table in original['runtime']['allocator_state'].items():
                if type(table) is candidate._ImmutableAllocationColumn:
                    assert cloned['runtime']['allocator_state'][owner] is table
                    assert cloned['runtime']['allocators'][owner] is kernel_b.physical_runtime.allocators[owner]
                    assert table is not kernel_b.physical_runtime.allocators[owner]._allocations
                    for key, allocation in table.items():
                        assert cloned['runtime']['allocator_state'][owner][key] is allocation
        continuation_a = sampling._ConcurrentImports(kernel_a, copy_a, list(plain._external_pending))
        continuation_b = sampling._ConcurrentImports(kernel_b, copy_b, list(frozen._external_pending))
        continuation_a.advance_until(math.inf)
        continuation_b.advance_until(math.inf)
        assert continuation_a.writes == continuation_b.writes
        assert _executor_state(copy_a) == _executor_state(copy_b)
        compare_physical_state(kernel_a, kernel_b)
        assert source_bits(source_a, a) == before_a and source_bits(source_b, b) == before_b
    kernel_a, copy_a = sampling._independent_kernel(source_a, a)
    kernel_b, copy_b = sampling._independent_kernel(source_b, b)
    plan_id = next(reversed(copy_b._async_plan_states))
    state_a, state_b = (copy_a._async_plan_states[plan_id], copy_b._async_plan_states[plan_id])
    save_a, save_b = (state_a['rollback_snapshot'], state_b['rollback_snapshot'])
    for executor, state in ((copy_a, state_a), (copy_b, state_b)):
        executor._rollback_external_import(state, state['phase_token'], 0)
    compare_physical_state(kernel_a, kernel_b)
    for save, executor in ((save_a, copy_a), (save_b, copy_b)):
        owner = next(iter(save['runtime']['allocator_state']))
        allocator = executor.runtime.allocators[owner]
        allocator.allocate('allocation-history-retry', allocator.alignment_bytes, 0)
        executor.restore(save)
        assert 'allocation-history-retry' not in {key[0] for key in allocator._allocations}
        assert type(allocator._allocations) is dict
        if type(save['runtime']['allocator_state'][owner]) is candidate._ImmutableAllocationColumn:
            assert allocator._allocations is not save['runtime']['allocator_state'][owner]
    compare_physical_state(kernel_a, kernel_b)
    for executor, state in ((copy_a, state_a), (copy_b, state_b)):
        executor.prepare_external_import(source_component=state['source_component'], destinations=state['destinations'], buffer_bytes=state['slot_count'] * state['chunk_bytes'], chunk_bytes=state['chunk_bytes'], request_id='allocation-history-after-rollback', arrival_ns=state['start_ns'])
    later_a = sampling._ConcurrentImports(kernel_a, copy_a, ['allocation-history-after-rollback'])
    later_b = sampling._ConcurrentImports(kernel_b, copy_b, ['allocation-history-after-rollback'])
    later_a.advance_until(math.inf)
    later_b.advance_until(math.inf)
    assert later_a.writes == later_b.writes
    compare_physical_state(kernel_a, kernel_b)
    assert _executor_state(copy_a) == _executor_state(copy_b)
    assert source_bits(source_a, a) == before_a and source_bits(source_b, b) == before_b
    runtime = source_b.physical_runtime
    token = runtime.snapshot(transactional=True, _immutable_timeline=True)
    assert all((type(value) is not candidate._ImmutableAllocationColumn for value in token['allocator_state'].values()))
    runtime.commit_transaction(token)

def test_allocation_history_native_hooks_memo_and_authored_fallback():
    key = ('allocation-history-' + str(id(object())), 12345)
    allocation = PhysicalAllocation(key[0], 0, 64, generation=key[1])
    plain = {key: allocation}
    state = {'owner': plain}
    candidate._freeze_allocation_state(state)
    frozen = state['owner']
    assert type(frozen) is candidate._ImmutableAllocationColumn
    assert copy.deepcopy(frozen, {}) is frozen and dict(frozen) == plain
    replacement = PhysicalAllocation('replacement', 128, 64)
    for source, value in ((key, ('changed', 3)), (key[0], 'changed-id'), (key[1], 9), (allocation, replacement)):
        a = copy.deepcopy(plain, {id(source): value})
        b = copy.deepcopy(frozen, {id(source): value})
        assert type(b) is dict and bits(a) == bits(b)
    events = []
    original = PhysicalAllocation.__deepcopy__

    def authored(self, memo):
        events.append(self.buffer_id)
        return self
    with patch.object(PhysicalAllocation, '__deepcopy__', authored):
        declined = {'owner': plain}
        candidate._freeze_allocation_state(declined)
        assert declined['owner'] is plain and (not events)
        a = copy.deepcopy(plain)
        expected = tuple(events)
        events.clear()
        b = copy.deepcopy(frozen)
        assert bits(a) == bits(b) and tuple(events) == expected

    def changed(self, memo):
        return ('changed-native-function-code',)
    oldcode = original.__code__
    try:
        original.__code__ = changed.__code__
        declined = {'owner': plain}
        candidate._freeze_allocation_state(declined)
        assert declined['owner'] is plain
        assert bits(copy.deepcopy(plain)) == bits(copy.deepcopy(frozen))
    finally:
        original.__code__ = oldcode

    def handler(value, memo):
        events.append('dispatch')
        return replacement
    with patch.dict(copy._deepcopy_dispatch, {PhysicalAllocation: handler}):
        declined = {'owner': plain}
        candidate._freeze_allocation_state(declined)
        assert declined['owner'] is plain
        events.clear()
        a = copy.deepcopy(plain)
        expected = tuple(events)
        events.clear()
        b = copy.deepcopy(frozen)
        assert bits(a) == bits(b) and tuple(events) == expected == ('dispatch',)
    allocation.__dict__['__deepcopy__'] = lambda memo: events.append('instance') or replacement
    try:
        events.clear()
        declined = {'owner': plain}
        candidate._freeze_allocation_state(declined)
        assert declined['owner'] is plain and (not events)
        assert next(iter(copy.deepcopy(declined['owner']).values())) is replacement
    finally:
        del allocation.__dict__['__deepcopy__']

    class UnknownKey:

        def __hash__(self):
            events.append('hash')
            return 7
    unknown = UnknownKey()
    bad = {unknown: allocation}
    events.clear()
    declined = {'owner': bad}
    candidate._freeze_allocation_state(declined)
    assert declined['owner'] is bad and (not events)

    class MemoSeed:

        def __deepcopy__(self, memo):
            events.append('earlier-hook')
            memo[id(allocation)] = replacement
            return None
    events.clear()
    a = copy.deepcopy((MemoSeed(), plain))
    expected = tuple(events)
    events.clear()
    b = copy.deepcopy((MemoSeed(), frozen))
    assert bits(a) == bits(b) and tuple(events) == expected == ('earlier-hook',)
