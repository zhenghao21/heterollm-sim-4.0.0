"""Native allocation history births reuse identity-equal tables through weak keys."""
import copy
from dataclasses import replace
import gc
from unittest.mock import patch
import weakref

from heterollm_sim import _timeline_history as candidate
from heterollm_sim import _timeline_history as history
from heterollm_sim.memory_allocator import PhysicalAddressAllocator
from test_token_owned_history import bits

def freeze(allocator, values=None):
    state = {'owner': dict(allocator._allocations) if values is None else values}
    candidate._freeze_allocation_state(state, {'owner': allocator})
    return state['owner']

def test_allocation_history_birth_cache_identity_hooks_and_weak_lifetime():
    candidate._ALLOCATION_HISTORY_CACHE.clear()
    allocator = PhysicalAddressAllocator(4096)
    a = allocator.allocate('birth-cache-native-a', 64)
    b = allocator.allocate('birth-cache-native-b', 64)
    before = bits(allocator.snapshot())
    first = freeze(allocator)
    second = freeze(allocator)
    third = freeze(allocator)
    assert first is second is third
    assert tuple.__getitem__(first, 1) is tuple.__getitem__(third, 1)
    assert bits(allocator.snapshot()) == before
    one = {'owner': dict(allocator._allocations)}
    two = {'owner': dict(allocator._allocations)}
    candidate._freeze_allocation_state(one)
    candidate._freeze_allocation_state(two)
    assert one['owner'] is not two['owner'] and bits(one) == bits(two)
    key = next(iter(allocator._allocations))
    new_key = tuple([key[0], key[1]])
    assert new_key is not key
    values = {new_key: allocator._allocations[key], **{k: v for k, v in allocator._allocations.items() if k is not key}}
    switched = freeze(allocator, values)
    assert switched is not first and next(iter(switched)) is new_key
    changed = freeze(allocator, {key: replace(a), **{k: v for k, v in allocator._allocations.items() if k is not key}})
    assert changed is not switched
    reordered = freeze(allocator, dict(reversed(tuple(allocator._allocations.items()))))
    assert list(reordered) == list(reversed(tuple(allocator._allocations)))
    assert reordered is not changed
    allocator.allocate('birth-cache-native-c', 64)
    grown = freeze(allocator)
    assert len(grown) == 3 and grown is not reordered
    events = []

    class UnknownKey:

        def __hash__(self):
            events.append('hash')
            return 9

        def __eq__(self, other):
            events.append('eq')
            return self is other
    bad = {UnknownKey(): a}
    events.clear()
    assert freeze(allocator, bad) is bad and (not events)
    assert not candidate._ALLOCATION_HISTORY_CACHE
    first = freeze(allocator)
    a.__dict__['__deepcopy__'] = lambda memo: events.append('instance-copy') or a
    try:
        events.clear()
        declined = freeze(allocator)
        assert type(declined) is dict and (not events) and (not candidate._ALLOCATION_HISTORY_CACHE)
    finally:
        del a.__dict__['__deepcopy__']
    first = freeze(allocator)
    a.__dict__['external_allocator'] = allocator
    try:
        unsafe = freeze(allocator)
        assert type(unsafe) is history._ImmutableAllocationColumn and unsafe is not first
        assert not candidate._ALLOCATION_HISTORY_CACHE
    finally:
        del a.__dict__['external_allocator']
    for name in ('__hash__', '__eq__'):

        def authored(*args):
            events.append(name)
            return 19 if name == '__hash__' else False
        candidate._ALLOCATION_HISTORY_CACHE.clear()
        with patch.object(PhysicalAddressAllocator, name, authored):
            events.clear()
            uncached = freeze(allocator)
            assert type(uncached) is history._ImmutableAllocationColumn
            assert not events and (not candidate._ALLOCATION_HISTORY_CACHE)

    class UnknownAllocators(dict):

        def get(self, *args):
            events.append('get')
            return allocator
    events.clear()
    state = {'owner': dict(allocator._allocations)}
    candidate._freeze_allocation_state(state, UnknownAllocators(owner=allocator))
    assert not events and (not candidate._ALLOCATION_HISTORY_CACHE)
    candidate._ALLOCATION_HISTORY_CACHE.clear()

    def gc_fixture():
        live = PhysicalAddressAllocator(4096)
        live.allocate('gc-native', 64)
        frozen = freeze(live)
        return (weakref.ref(live), frozen)
    weak, column = gc_fixture()
    gc.collect()
    assert weak() is None and len(candidate._ALLOCATION_HISTORY_CACHE) == 0
    assert len(column) == 1
