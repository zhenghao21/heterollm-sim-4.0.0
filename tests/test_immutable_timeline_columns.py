"""Immutable private rollback columns preserve native imports and copy boundaries."""
import copy
import math
from unittest.mock import patch

import pytest

from heterollm_sim import _timeline_history as history, native_layer_sampling as sampling
from test_native_external_kv import ingress_scenario, native_runtime
from test_native_concurrent_imports import _executor_state
from test_token_owned_history import bits, compare_physical_state

def fixture(case):
    runtime = native_runtime(ingress_scenario(case, count=2))
    runtime._arrive()
    runtime._start_external_imports()
    return (runtime, runtime._execution_resource_kernel, runtime._external_executor)

def source_bits(source, executor):
    saved = source.physical_runtime.snapshot()
    return (bits(saved['timeline']), _executor_state(executor))

def timeline_columns(executor):
    return [state['rollback_snapshot']['runtime']['timeline'] for state in executor._async_plan_states.values()]

def compare_columns(a, b):
    for left, right in zip(timeline_columns(a), timeline_columns(b)):
        assert bits(left) == bits(right)
        assert type(right) is dict
        assert type(right['_touched']) is set and type(right['_intervals']) is dict
        assert type(right['_interval_payloads']) is dict

@pytest.mark.parametrize('case', ('hbm_tp1', 'hbf_weights_kv_tp1'))
def test_real_import_history_multiple_clones_rollback_and_retry(case):
    with patch.object(history, '_freeze_timeline_columns', lambda state: state):
        plain, source_a, a = fixture(case)
    frozen, source_b, b = fixture(case)
    assert type(source_b.physical_runtime.snapshot()['timeline']['ready_ns']) is dict
    assert type(b.snapshot()['runtime']['timeline']['ready_ns']) is dict
    immutable = [column for state in timeline_columns(b) for name in history._COLUMNS for column in (state[name],) if type(column) is history._ImmutableTimelineColumn]
    assert len(immutable) == 14, (case, len(immutable))
    compare_columns(a, b)
    before_a, before_b = (source_bits(source_a, a), source_bits(source_b, b))
    for repeat in range(2):
        kernel_a, copy_a = sampling._independent_kernel(source_a, a)
        kernel_b, copy_b = sampling._independent_kernel(source_b, b)
        compare_physical_state(kernel_a, kernel_b)
        compare_columns(copy_a, copy_b)
        for original, copied in zip(timeline_columns(b), timeline_columns(copy_b)):
            for name in history._COLUMNS:
                assert copied[name] is original[name]
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
        executor.runtime.timeline.reserve('readonly-history-retry', 17.0, 3.0)
        executor.restore(save)
    compare_physical_state(kernel_a, kernel_b)
    destinations = state_a['destinations']
    plans = []
    for executor, state in ((copy_a, state_a), (copy_b, state_b)):
        plan = executor.prepare_external_import(source_component=state['source_component'], destinations=destinations, buffer_bytes=state['slot_count'] * state['chunk_bytes'], chunk_bytes=state['chunk_bytes'], request_id='retry-after-rollback', arrival_ns=state['start_ns'])
        plans.append(plan)
    later_a = sampling._ConcurrentImports(kernel_a, copy_a, ['retry-after-rollback'])
    later_b = sampling._ConcurrentImports(kernel_b, copy_b, ['retry-after-rollback'])
    later_a.advance_until(math.inf)
    later_b.advance_until(math.inf)
    assert later_a.writes == later_b.writes
    compare_physical_state(kernel_a, kernel_b)
    assert _executor_state(copy_a) == _executor_state(copy_b)
    assert source_bits(source_a, a) == before_a and source_bits(source_b, b) == before_b
    assert all((type(getattr(copy_b.runtime.timeline, name)) is dict for name in ('ready_ns', 'busy_ns', 'bytes_moved', 'last_intervals', 'lane_available', 'lane_directions', '_intervals', '_interval_payloads')))

def test_readonly_columns_preserve_native_mapping_and_authored_copy_boundaries():
    key = 'resource-identity-' + str(id(object()))
    interval = (1.125, 3.25)
    lane = (5.5,)
    plain = {key: interval, 'lane': lane, 'n': 9, 'none': None}
    state = {'last_intervals': plain}
    history._freeze_timeline_columns(state)
    frozen = state['last_intervals']
    assert type(frozen) is history._ImmutableTimelineColumn
    assert bits(frozen) == bits(plain)
    assert len(frozen) == len(plain) and list(frozen) == list(plain)
    assert frozen[key] is interval and frozen.get(key) is interval
    assert key in frozen and 'missing' not in frozen
    assert list(frozen.keys()) == list(plain.keys()) and list(frozen.items()) == list(plain.items())
    assert dict(frozen) == plain and frozen == plain and (plain == frozen) and (not frozen != plain)
    assert copy.deepcopy(frozen, {}) is frozen
    try:
        frozen[key] = (0.0, 0.0)
    except TypeError:
        pass
    else:
        raise AssertionError('column is writable')
    try:
        frozen.alias = {}
    except AttributeError:
        pass
    else:
        raise AssertionError('column has mutable instance attributes')
    plain['after-freeze'] = 1.0
    assert 'after-freeze' not in frozen
    del plain['after-freeze']
    for original, replacement in ((key, 'replacement-resource'), (interval, (11.0, 13.0)), (interval[0], 23.0), (lane, (29.0,)), (lane[0], 31.0), (9, 37), (None, 'null')):
        expected = copy.deepcopy(plain, {id(original): replacement})
        observed = copy.deepcopy(frozen, {id(original): replacement})
        assert type(observed) is dict and bits(expected) == bits(observed)
    events = []

    class Unknown:

        def __hash__(self):
            events.append('hash')
            return 9

        def __deepcopy__(self, memo):
            events.append('copy')
            return 'unknown-copy'
    unknown = Unknown()
    bad_values = {'ready_ns': {unknown: 1.0}, 'busy_ns': {'x': unknown}}
    events.clear()
    history._freeze_timeline_columns(bad_values)
    assert not events and all((type(value) is dict for value in bad_values.values()))
    for value in (interval, lane):
        for primitive in value:
            assert id(value) in tuple.__getitem__(frozen, 1) and id(primitive) in tuple.__getitem__(frozen, 1)

    class MutateMemo:

        def __deepcopy__(self, memo):
            events.append('earlier-copy')
            memo[id(key)] = 'edited-by-earlier-copy'
            memo[id(interval)] = (41.0, 43.0)
            return 'earlier-result'
    events.clear()
    left = copy.deepcopy((MutateMemo(), plain))
    expected_events = tuple(events)
    events.clear()
    right = copy.deepcopy((MutateMemo(), frozen))
    assert bits(left) == bits(right) and tuple(events) == expected_events
    original_handler = copy._deepcopy_dispatch[float]

    def handler(value, memo):
        events.append(('float', value.hex()))
        return original_handler(value, memo)
    with patch.dict(copy._deepcopy_dispatch, {float: handler}):
        events.clear()
        left = copy.deepcopy(plain)
        expected_events = tuple(events)
        events.clear()
        right = copy.deepcopy(frozen)
        assert bits(left) == bits(right) and tuple(events) == expected_events
        unknown_state = {'busy_ns': {'native': 3.0}}
        history._freeze_timeline_columns(unknown_state)
        assert type(unknown_state['busy_ns']) is dict
    original_keep_alive = copy._keep_alive

    def keep_alive(value, memo):
        events.append('keep-alive')
        return original_keep_alive(value, memo)
    with patch.object(copy, '_keep_alive', keep_alive):
        events.clear()
        copy.deepcopy(plain)
        expected_events = tuple(events)
        events.clear()
        copy.deepcopy(frozen)
        assert tuple(events) == expected_events == ('keep-alive',)

    class KeepAlive(list):

        def append(self, value):
            events.append('keep-alive-list')
            return super().append(value)
    events.clear()
    memo = {}
    memo[id(memo)] = KeepAlive()
    actual = copy.deepcopy(frozen, memo)
    assert type(actual) is dict and events == ['keep-alive-list']
    assert memo[id(frozen)] is actual
    memo = {unknown: 'custom-key'}
    events.clear()
    assert not all((type(key) is int for key in memo)) and (not events)
    assert type(copy.deepcopy(frozen, memo)) is dict and not events
    runtime, source, executor = fixture('hbm_tp1')
    original = source.physical_runtime.snapshot
    calls = []

    def snapshot_callback():
        calls.append('runtime-snapshot')
        return original()
    source.physical_runtime.snapshot = snapshot_callback
    result = executor.snapshot(_immutable_timeline=True)
    assert calls == ['runtime-snapshot']
    assert all((type(result['runtime']['timeline'][name]) is dict for name in history._COLUMNS))


def test_authored_executor_snapshot_callable_receives_no_private_keyword_or_attribute_probe():
    runtime, source, executor = fixture("hbm_tp1")
    saved = next(iter(executor._async_plan_states.values()))
    original = executor.snapshot
    calls = []
    class SnapshotCallback:
        def __getattribute__(self, name):
            if name == "__func__":
                calls.append("unexpected-function-probe")
            return object.__getattribute__(self, name)
        def __call__(self):
            calls.append("snapshot")
            return original()
    executor.snapshot = SnapshotCallback()
    plan = executor.prepare_external_import(
        source_component=saved["source_component"], destinations=saved["destinations"],
        buffer_bytes=saved["slot_count"] * saved["chunk_bytes"], chunk_bytes=saved["chunk_bytes"],
        request_id="authored-snapshot-retry", arrival_ns=saved["start_ns"])
    actual = next(state for state in executor._async_plan_states.values()
                  if state["request_id"] == "authored-snapshot-retry")
    assert calls == ["snapshot"]
    assert all(type(actual["rollback_snapshot"]["runtime"]["timeline"][name]) is dict
               for name in history._COLUMNS)
