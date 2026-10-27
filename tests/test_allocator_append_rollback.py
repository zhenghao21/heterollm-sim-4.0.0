"""Append-only rollback preserves the original ordered allocator state."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import struct

import pytest

from heterollm_sim.memory_allocator import PhysicalAddressAllocator, PhysicalAllocation

CandidateAllocator = PhysicalAddressAllocator


class _ReferenceAllocator(PhysicalAddressAllocator):
    # Oracle below is extracted verbatim from the original source function.
    def rollback_transaction(self, save: "_AllocationSave") -> None:
        self._close(save)
        changed = False
        for key, (previous, order) in save.entries.items():
            if (self._allocations.get(key) is previous
                    and self._allocation_order.get(key) == order):
                continue
            self._before_mutation(key)
            changed = True
            if previous is None:
                self._allocations.pop(key, None)
                self._allocation_order.pop(key, None)
            else:
                self._allocations[key] = previous
                self._allocation_order[key] = order
        if changed:
            self._next_order = max(self._next_order, max(self._allocation_order.values(), default=-1) + 1)
            ordered = sorted(self._allocations, key=self._allocation_order.__getitem__)
            allocations = [(key, self._allocations[key]) for key in ordered]
            self._allocations.clear()
            self._allocations.update(allocations)
            self._rebuild_indexes()


def _ordered(value):
    if type(value) is float:
        return ('float_bits', struct.pack('!d', value))
    if is_dataclass(value):
        return (type(value).__name__, tuple((field.name, _ordered(getattr(value, field.name)))
                                           for field in fields(value)))
    if isinstance(value, Mapping):
        return (type(value).__name__, tuple((_ordered(key), _ordered(item)) for key, item in value.items()))
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(map(_ordered, value)))
    if type(value) in (set, frozenset):
        return ('set', frozenset(map(_ordered, value)))
    return value


def private_state(allocator, saves):
    identities = {id(save): index for index, save in enumerate(saves)}
    for save in saves:
        assert save.allocator is allocator
    result = []
    for name, value in vars(allocator).items():
        if name == '_saves':
            value = ('retained', frozenset(identities[id(save)] for save in value))
        elif name == '_open_saves':
            value = ('open', tuple(identities[id(save)] for save in value))
        else:
            value = _ordered(value)
        result.append((name, value))
    return tuple(result), tuple(_ordered(save.entries) for save in saves)


def assert_pair(pair, saves):
    assert private_state(pair[0], saves[0]) == private_state(pair[1], saves[1])
    for allocator in pair:
        assert tuple(allocator._allocations) == tuple(sorted(
            allocator._allocations, key=allocator._allocation_order.__getitem__))
        assert all(allocator._allocations[(row.buffer_id, row.generation)] is row
                   for row in allocator._address_rows)


def make_pair(count, *, sparse=True, native_nan=False):
    records = tuple(PhysicalAllocation(f'seed-{index:06}', index*(128 if sparse else 64), 64,
        inferred=True) for index in range(count))
    if native_nan:
        records = (PhysicalAllocation('nan-first', float('nan'), 64), *records)
    capacity = count*256+65536
    seed = PhysicalAddressAllocator.from_snapshot({'capacity_bytes': capacity, 'alignment_bytes': 64,
                                                  'allocations': records})
    pair = [cls(capacity, 64) for cls in (CandidateAllocator, _ReferenceAllocator)]
    for allocator in pair:
        # Normal native births remain eligible; public mapping imports
        # deliberately retain the conservative old reordering path.
        for key, allocation in seed._allocations.items():
            allocator._store_allocation(key, allocation)
        allocator._rebuild_indexes()
    assert all(pair[0]._allocations[key] is pair[1]._allocations[key] for key in seed._allocations)
    return pair, [[], []]


def begin(pair, saves):
    for allocator, checkpoints in zip(pair, saves):
        checkpoints.append(allocator.begin_transaction())
    return len(saves[0])-1


def append(allocator, count, prefix, *, aliases=True):
    for index in range(count):
        if aliases and index % 4 == 3:
            allocator.allocate(f'{prefix}-{index}', 16, 1,
                alias_of='seed-000000', alias_generation=0, alias_offset_bytes=16)
        else:
            allocator.allocate(f'{prefix}-{index}', 64, 1)


def rollback_pair(pair, saves, index):
    unchanged = [dict(allocator._allocations) for allocator in pair]
    for allocator, checkpoints, before in zip(pair, saves, unchanged):
        save = checkpoints[index]
        allocator.rollback_transaction(save)
        for key, (previous, order) in save.entries.items():
            if previous is None:
                assert key not in allocator._allocations
            else:
                assert allocator._allocations[key] is previous
                assert allocator._allocation_order[key] == order
        for key, allocation in before.items():
            if key not in save.entries:
                assert allocator._allocations[key] is allocation
    assert_pair(pair, saves)


def followup(pair, saves, prefix):
    results = []
    for allocator in pair:
        allocation = allocator.allocate(prefix, 31, 2)
        view = allocator.allocate(prefix+'-alias', 8, 2, alias_of=prefix, alias_offset_bytes=8)
        results.append((allocation, view))
    assert results[0] == results[1]
    assert_pair(pair, saves)


@pytest.mark.parametrize('count,new_count', [(32, 1), (257, 16), (1024, 128)])
@pytest.mark.parametrize('sparse', [False, True])
def test_append_and_new_alias_rollback_keeps_all_order_and_references(count, new_count, sparse):
    pair, saves = make_pair(count, sparse=sparse)
    public_maps = [allocator._allocations for allocator in pair]
    index = begin(pair, saves)
    for allocator in pair:
        append(allocator, new_count, 'new')
    rollback_pair(pair, saves, index)
    assert all(allocator._allocations is mapping for allocator, mapping in zip(pair, public_maps))
    followup(pair, saves, 'after')


def test_only_new_alias_and_nested_checkpoints_keep_each_before_image():
    pair, saves = make_pair(257)
    outer = begin(pair, saves)
    for allocator in pair:
        allocator.allocate('outer-new', 64, 1)
    inner = begin(pair, saves)
    for allocator in pair:
        allocator.allocate('new-view', 16, 1, alias_of='seed-000000', alias_generation=0)
        append(allocator, 16, 'inner')
    rollback_pair(pair, saves, inner)
    followup(pair, saves, 'outer-followup')
    rollback_pair(pair, saves, outer)
    followup(pair, saves, 'after-outer')


def test_commit_closes_nested_checkpoints_but_they_remain_restorable():
    pair, saves = make_pair(257)
    outer = begin(pair, saves)
    for allocator in pair:
        append(allocator, 16, 'before-inner')
    inner = begin(pair, saves)
    for allocator in pair:
        append(allocator, 16, 'inside-inner')
    for allocator, checkpoints in zip(pair, saves):
        allocator.commit_transaction(checkpoints[outer])
        append(allocator, 16, 'after-commit')
    rollback_pair(pair, saves, inner)
    followup(pair, saves, 'rebranch')
    rollback_pair(pair, saves, inner)
    rollback_pair(pair, saves, outer)
    followup(pair, saves, 'after-closed')


def test_mixed_old_replacement_keeps_original_reordering_path():
    pair, saves = make_pair(257)
    original = [allocator.lookup('seed-000000') for allocator in pair]
    index = begin(pair, saves)
    for allocator in pair:
        allocator.release('seed-000000')
        allocator.allocate('seed-000000', 64, inferred=True)
        append(allocator, 16, 'new')
        assert next(iter(allocator._allocations)) != ('seed-000000', 0)
    rollback_pair(pair, saves, index)
    assert all(allocator.lookup('seed-000000') is allocation
               for allocator, allocation in zip(pair, original))
    assert all(next(iter(allocator._allocations)) == ('seed-000000', 0) for allocator in pair)
    followup(pair, saves, 'after-old')


def test_public_restore_while_checkpoint_retained_preserves_ordinal_invariant():
    pair, saves = make_pair(64)
    stored = [dict(allocator._allocations) for allocator in pair]
    index = begin(pair, saves)
    for allocator, snapshot in zip(pair, stored):
        append(allocator, 16, 'new')
        allocator._restore_allocations(dict(reversed(tuple(snapshot.items()))))
    assert_pair(pair, saves)
    rollback_pair(pair, saves, index)
    followup(pair, saves, 'after-restore')


def test_native_nan_snapshot_keeps_stable_address_sort_and_original_objects():
    pair, saves = make_pair(8, native_nan=True)
    original = [allocator.lookup('nan-first') for allocator in pair]
    index = begin(pair, saves)
    for allocator in pair:
        allocator.allocate('new-explicit', 64, 1, address=4096)
    rollback_pair(pair, saves, index)
    assert all(allocator.lookup('nan-first') is allocation
               for allocator, allocation in zip(pair, original))


class _HashFailure(RuntimeError):
    pass


class _HashText(str):
    events = None
    fail_at = None

    def __hash__(self):
        cls = type(self)
        if cls.events is not None:
            cls.events.append(('hash', str.__str__(self)))
            count = sum(event[0] == 'hash' for event in cls.events)
            if count == cls.fail_at:
                raise _HashFailure('author hash stopped at '+str(count))
        return str.__hash__(self)


def _hash_scenario(cls, fail_at=None):
    events = []
    _HashText.events, _HashText.fail_at = events, fail_at
    allocator = None
    try:
        events.append(('phase', 'from_snapshot'))
        allocator = cls.from_snapshot({'capacity_bytes': 4096, 'alignment_bytes': 64,
            'allocations': (PhysicalAllocation(_HashText('author-key'), 0, 64),
                            PhysicalAllocation('plain', 128, 64))})
        assert allocator._native_allocation_keys is False
        events.append(('phase', 'begin_new'))
        save = allocator.begin_transaction()
        allocator.allocate('new-workspace', 64, 1)
        events.append(('phase', 'rollback'))
        allocator.rollback_transaction(save)
        outcome = None
    except _HashFailure as error:
        outcome = (type(error).__name__, str(error))
    finally:
        _HashText.events, _HashText.fail_at = None, None
    return events, outcome, allocator


def test_public_snapshot_author_key_keeps_hash_trace_and_every_error_prefix():
    expected, outcome, reference = _hash_scenario(_ReferenceAllocator)
    actual, actual_outcome, candidate = _hash_scenario(CandidateAllocator)
    assert outcome is None and actual_outcome is None and actual == expected
    assert _ordered(vars(candidate)) == _ordered(vars(reference))
    rollback_start = expected.index(('phase', 'rollback'))
    assert any(event[0] == 'hash' for event in expected[rollback_start+1:])
    count = sum(event[0] == 'hash' for event in expected)
    for fail_at in range(1, count+1):
        old_events, old_error, old_state = _hash_scenario(_ReferenceAllocator, fail_at)
        new_events, new_error, new_state = _hash_scenario(CandidateAllocator, fail_at)
        assert old_error is not None and new_error == old_error
        assert new_events == old_events, fail_at
        if old_state is not None:
            assert _ordered(vars(new_state)) == _ordered(vars(old_state)), fail_at


def test_unknown_store_marks_contract_before_first_hash_and_remains_sticky():
    allocator = CandidateAllocator(4096, 64)
    assert allocator._native_allocation_keys is True
    observations = []
    class StoreKey(str):
        def __hash__(self):
            observations.append(allocator._native_allocation_keys)
            return str.__hash__(self)
    key = StoreKey('store-author')
    allocator._store_allocation((key, 0), PhysicalAllocation(key, 0, 64))
    assert observations and all(value is False for value in observations)
    assert allocator._native_allocation_keys is False
    allocator.release('store-author')
    checkpoint = allocator.begin_transaction()
    allocator.allocate('plain-after-author', 64, 1)
    allocator.rollback_transaction(checkpoint)
    assert allocator._native_allocation_keys is False


def test_restore_disables_contract_before_mapping_author_methods():
    from collections.abc import Mapping
    allocator = CandidateAllocator(4096, 64)
    allocator.allocate('original', 64)
    observations = []
    class Imported(Mapping):
        def __iter__(self):
            observations.append(('iter', allocator._native_allocation_keys))
            return iter((('imported', 0),))
        def __len__(self):
            observations.append(('len', allocator._native_allocation_keys))
            return 1
        def __getitem__(self, key):
            observations.append(('getitem', allocator._native_allocation_keys))
            return PhysicalAllocation('imported', 128, 64)
    allocator._restore_allocations(Imported())
    assert observations and all(native is False for _, native in observations)
    assert allocator._native_allocation_keys is False


def test_native_key_contract_survives_deepcopy_and_pickle():
    import copy
    import pickle
    native = CandidateAllocator(4096, 64)
    native.allocate('native', 64)
    imported = CandidateAllocator.from_snapshot(native.snapshot())
    assert native._native_allocation_keys is True
    assert imported._native_allocation_keys is False
    for allocator in (native, imported):
        for restored in (copy.deepcopy(allocator), pickle.loads(pickle.dumps(allocator))):
            assert restored._native_allocation_keys is allocator._native_allocation_keys
            assert _ordered(restored.snapshot()) == _ordered(allocator.snapshot())


def _memo_author_scenario(cls, mode, fail_at=None):
    import copy
    source = cls(4096, 64)
    source.allocate('author-key', 64)
    events = []
    if mode == 'exact_dict_author_key':
        replacement = {(_HashText('author-key'), 0): source.lookup('author-key')}
        memo = {id(source._allocations): replacement}
    else:
        class AuthorOrder(dict):
            def __getitem__(self, key):
                events.append(('getitem', key))
                if sum(item[0] == 'getitem' for item in events) == fail_at:
                    raise _HashFailure('author map stopped')
                return super().__getitem__(key)
        memo = {id(source._allocation_order): AuthorOrder(source._allocation_order)}
    allocator = copy.deepcopy(source, memo)
    assert allocator._native_allocation_keys is False
    save = allocator.begin_transaction()
    allocator.allocate('new-workspace', 64, 1)
    _HashText.events, _HashText.fail_at = events, fail_at
    try:
        allocator.rollback_transaction(save)
        error = None
    except _HashFailure as failure:
        error = (type(failure).__name__, str(failure))
    finally:
        _HashText.events, _HashText.fail_at = None, None
    return events, error


def test_deepcopy_memo_author_key_roots_keep_original_callbacks_and_error_prefix():
    for mode in ('exact_dict_author_key', 'author_order_subclass'):
        expected, error = _memo_author_scenario(_ReferenceAllocator, mode)
        actual, actual_error = _memo_author_scenario(CandidateAllocator, mode)
        assert expected and actual == expected and error is None and actual_error is None
        for fail_at in range(1, len(expected)+1):
            old_events, old_error = _memo_author_scenario(_ReferenceAllocator, mode, fail_at)
            new_events, new_error = _memo_author_scenario(CandidateAllocator, mode, fail_at)
            assert old_error is not None and new_error == old_error
            assert new_events == old_events


def _checkpoint_memo_author_scenario(cls, fail_at=None):
    import copy
    source = cls(4096, 64)
    original = source.allocate('author-key', 64)
    checkpoint = source.begin_transaction()
    source.release('author-key')
    replacement = {(_HashText('author-key'), 0): (original, 0)}
    allocator, checkpoint = copy.deepcopy((source, checkpoint), {id(checkpoint.entries): replacement})
    assert allocator._native_allocation_keys is True
    events = []
    _HashText.events, _HashText.fail_at = events, fail_at
    try:
        events.append(('phase', 'restore-old-author-key'))
        allocator.rollback_transaction(checkpoint)
        if cls is CandidateAllocator:
            assert allocator._native_allocation_keys is False
        events.append(('phase', 'later-append-only'))
        next_checkpoint = allocator.begin_transaction()
        allocator.allocate('new-workspace', 64, 1)
        allocator.rollback_transaction(next_checkpoint)
        error = None
    except _HashFailure as failure:
        error = (type(failure).__name__, str(failure))
    finally:
        _HashText.events, _HashText.fail_at = None, None
    return events, error


def test_copied_checkpoint_author_key_stays_conservative_on_the_next_rollback():
    expected, error = _checkpoint_memo_author_scenario(_ReferenceAllocator)
    actual, actual_error = _checkpoint_memo_author_scenario(CandidateAllocator)
    assert actual == expected and error is None and actual_error is None
    boundary = expected.index(('phase', 'later-append-only'))
    assert any(event[0] == 'hash' for event in expected[boundary+1:])
    for fail_at in range(1, sum(event[0] == 'hash' for event in expected)+1):
        old_events, old_error = _checkpoint_memo_author_scenario(_ReferenceAllocator, fail_at)
        new_events, new_error = _checkpoint_memo_author_scenario(CandidateAllocator, fail_at)
        assert old_error is not None and new_error == old_error
        assert new_events == old_events


def test_copied_checkpoint_items_and_nested_unpack_deopt_before_author_code():
    import copy
    source = CandidateAllocator(4096, 64)
    source.allocate('seed', 64)
    checkpoint = source.begin_transaction()
    source.allocate('new-workspace', 64, 1)
    observed = []
    holder = []
    class BeforeImage:
        def __iter__(self):
            observed.append(('before-image-unpack', holder[0]._native_allocation_keys))
            return iter((None, None))
    class Row:
        def __iter__(self):
            observed.append(('row-unpack', holder[0]._native_allocation_keys))
            return iter((('new-workspace', 1), BeforeImage()))
    class AuthorEntries(dict):
        def items(self):
            observed.append(('items', holder[0]._native_allocation_keys))
            return [Row()]
    allocator, checkpoint = copy.deepcopy((source, checkpoint),
        {id(checkpoint.entries): AuthorEntries(checkpoint.entries)})
    holder.append(allocator)
    assert checkpoint._native_born is False and allocator._native_allocation_keys is True
    allocator.rollback_transaction(checkpoint)
    assert observed == [('items', False), ('row-unpack', False), ('before-image-unpack', False)]
    assert ('new-workspace', 1) not in allocator._allocations


def test_unknown_checkpoint_type_does_not_read_author_native_marker():
    allocator = CandidateAllocator(4096, 64)
    allocator.allocate('seed', 64)
    allocator.allocate('new-workspace', 64, 1)
    observed = []
    class AuthorSave:
        @property
        def _native_born(self):
            raise AssertionError('unknown checkpoint marker must not be read')
        @property
        def entries(self):
            observed.append(allocator._native_allocation_keys)
            return {('new-workspace', 1): (None, None)}
    allocator.rollback_transaction(AuthorSave())
    assert observed == [False]
    assert ('new-workspace', 1) not in allocator._allocations


def test_checkpoint_copy_and_pickle_are_conservative_before_close():
    import copy
    import pickle
    allocator = CandidateAllocator(4096, 64)
    allocator.allocate('seed', 64)
    checkpoint = allocator.begin_transaction()
    assert checkpoint._native_born is True
    for current, saved in (copy.deepcopy((allocator, checkpoint)),
                           pickle.loads(pickle.dumps((allocator, checkpoint)))):
        assert saved._native_born is False
        assert current._native_allocation_keys is True
        current.commit_transaction(saved)
        assert current._native_allocation_keys is False


def _ordinal_memo_scenario(cls):
    import copy
    events = []
    class AuthorOrdinal(int):
        def __lt__(self, other):
            events.append(('lt', int(self), int(other)))
            return int(self) < int(other)
        def __gt__(self, other):
            events.append(('gt', int(self), int(other)))
            return int(self) > int(other)
        def __add__(self, other):
            events.append(('add', int(self), int(other)))
            return AuthorOrdinal(int(self)+int(other))
    source = cls(4096, 64)
    source.allocate('seed-a', 64)
    source.allocate('seed-b', 64)
    allocator = copy.deepcopy(source, {id(source._next_order): AuthorOrdinal(source._next_order)})
    assert allocator._native_allocation_keys is False
    allocator.allocate('persistent-author-ordinal', 64)
    checkpoint = allocator.begin_transaction()
    allocator.allocate('new-workspace', 64, 1)
    events.clear()
    allocator.rollback_transaction(checkpoint)
    return events


def test_memo_replaced_next_ordinal_keeps_original_sort_callbacks():
    expected = _ordinal_memo_scenario(_ReferenceAllocator)
    actual = _ordinal_memo_scenario(CandidateAllocator)
    assert expected and actual == expected
