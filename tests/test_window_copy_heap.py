"""A point's heap stays free of per-task cycles, and of a snapshot per in-flight import."""
import copy
import gc
import math
import random
import weakref

import pytest

from heterollm_sim import native_layer_sampling as layer
from heterollm_sim.data_motion import _OverlapCompletionIndex, _sink_plain_config
from heterollm_sim.memory_transfer import ResourceTimeline, _TimelineBefore
from test_native_external_kv import ingress_scenario, native_runtime


def _stopped_collector():
    was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()
    return was_enabled


def test_overlap_index_equals_the_pairwise_scan_and_leaves_no_cycle():
    rng = random.Random(7)
    accesses = [(rng.randrange(0, 4096), rng.randrange(1, 512)) for _ in range(300)]
    endpoints = sorted({value for address, count in accesses for value in (address, address + count)})
    was_enabled = _stopped_collector()
    try:
        index = _OverlapCompletionIndex(endpoints)
        written = []
        for address, count in accesses:
            expected = max((end for lo, hi, end in written if lo < address + count and address < hi), default=0.)
            assert index.query(address, count) == expected
            end = expected + rng.random() + 1.
            index.update(address, count, end)
            written.append((address, address + count, end))
        reference = weakref.ref(index)
        del index
        assert reference() is None
        assert gc.collect() == 0
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("case", ("hbf_weights_kv_tp1", "hbm_tp1"))
def test_in_flight_imports_keep_no_rollback_state(case):
    runtime = native_runtime(ingress_scenario(case, count=2))
    runtime._arrive()
    runtime._start_external_imports()
    assert len(runtime._external_pending) == 2
    for state in runtime._external_executor._async_plan_states.values():
        assert not any("snapshot" in key for key in state)
    for entry in runtime._external_pending.values():
        assert entry["plan"].rollback is None


@pytest.mark.parametrize("case", ("hbf_weights_kv_tp1", "hbm_tp1"))
def test_finished_window_copy_dies_by_reference_counting(case):
    runtime = native_runtime(ingress_scenario(case, count=2))
    runtime._arrive()
    runtime._start_external_imports()
    source, executor = runtime._execution_resource_kernel, runtime._external_executor
    kernel, copied = layer._independent_kernel(source, executor)
    imports = layer._ConcurrentImports(kernel, copied, list(runtime._external_pending))
    imports.advance_until(math.inf)
    timeline = weakref.ref(kernel.physical_runtime.timeline)
    was_enabled = _stopped_collector()
    try:
        del imports, copied, kernel
        assert timeline() is None
    finally:
        if was_enabled:
            gc.enable()


def test_numeric_state_follows_a_copied_core_and_leaves_no_cycle():
    runtime = native_runtime(ingress_scenario("hbm_tp1", count=2))
    runtime._arrive()
    runtime._start_external_imports()
    source, executor = runtime._execution_resource_kernel, runtime._external_executor
    kernel, copied = layer._independent_kernel(source, executor)
    layer._ConcurrentImports(kernel, copied, list(runtime._external_pending)).advance_until(math.inf)
    timeline = kernel.physical_runtime.timeline
    core = next(active.core for active in kernel.physical_runtime.runtimes.values()
                if getattr(active.core, "_hbm_numeric_state", None) is not None)
    assert core._hbm_numeric_state.core is core
    duplicate = copy.deepcopy(core, {id(timeline): timeline})
    assert duplicate._hbm_numeric_state is not core._hbm_numeric_state
    assert duplicate._hbm_numeric_state.core is duplicate
    was_enabled = _stopped_collector()
    try:
        state = weakref.ref(duplicate._hbm_numeric_state)
        del duplicate
        assert state() is None
    finally:
        if was_enabled:
            gc.enable()


def test_mirrored_nand_copy_equals_the_generic_copy():
    from heterollm_sim.nand_core import _ReadCacheSlot
    runtime = native_runtime(ingress_scenario("hbf_weights_kv_tp1", count=2))._execution_resource_kernel.physical_runtime
    owner, active = next((owner, active) for owner, active in runtime.runtimes.items()
                         if "_read_cache" in vars(active.core))
    shared = _ReadCacheSlot(page_key=(1, 2), ready_ns=3., reader_completion_ns=4., last_use=5)
    active.core.__dict__["_read_cache"].update({0: [shared, _ReadCacheSlot()], 1: [shared]})
    direct = layer._copy_owner_runtime(runtime, owner)
    generic = copy.deepcopy(active, {id(runtime.timeline): runtime.timeline})
    assert direct.core.timeline is runtime.timeline
    assert direct.core.__dict__["_read_cache"] == generic.core.__dict__["_read_cache"]
    assert direct.core.__dict__["_read_cache"][0][0] is direct.core.__dict__["_read_cache"][1][0]
    assert direct.core.__dict__["_read_cache"][0][0] is not shared


def test_before_select_equals_get_for_every_key():
    timeline = ResourceTimeline()
    timeline.lane_available.update({"changed": [1., 2.], "kept": [3.], "gone": [4.]})
    timeline.ready_ns.update(changed=2., kept=3., gone=4.)
    token = timeline._begin_owned_transaction()
    before = token.before_lanes()
    timeline._remember_resource("changed")
    timeline._remember_resource("gone")
    timeline._remember_resource("born")
    timeline.lane_available["changed"][0] = 9.
    timeline.lane_available["born"] = [5.]
    del timeline.lane_available["gone"]
    keys = ("changed", "kept", "gone", "born", "absent")
    selected = before.select(keys)
    assert type(selected) is dict
    assert selected == {key: before.get(key) for key in keys if before.get(key) is not None}
    assert selected["changed"] == (1., 2.) and selected["kept"] == (3.,) and selected["gone"] == (4.,)
    assert "born" not in selected and "absent" not in selected
    token.rollback()


def test_before_select_keeps_an_authored_reader(monkeypatch):
    timeline = ResourceTimeline()
    timeline.lane_available["x"] = [1.]
    token = timeline._begin_owned_transaction()
    before = token.before_lanes()
    monkeypatch.setattr(_TimelineBefore, "__getitem__", lambda self, key: "authored")
    assert before.select(("x", "y")) == {"x": "authored", "y": "authored"}
    monkeypatch.undo()
    token.rollback()


def test_plain_config_verdict_is_kept_per_object():
    from dataclasses import replace
    from test_hbm_compiled_control import configuration
    plain = configuration(channels=2)
    assert _sink_plain_config(plain) and _sink_plain_config(plain)
    assert _sink_plain_config(replace(plain)) and not _sink_plain_config(object())
    custom = replace(plain)
    object.__setattr__(custom, "authored", object())
    assert not _sink_plain_config(custom) and not _sink_plain_config(custom)


def test_timeline_columns_are_proven_once(monkeypatch):
    from heterollm_sim import data_motion
    calls = []
    original = data_motion._resource_ledger.can_sink_resource_columns

    def counted(timeline, names):
        calls.append(timeline)
        return original(timeline, names)

    monkeypatch.setattr(data_motion._resource_ledger, "can_sink_resource_columns", counted)
    first, second = ResourceTimeline(), ResourceTimeline()
    assert data_motion._sink_columns_proven(first) and data_motion._sink_columns_proven(first)
    assert len(calls) == 1
    second.busy_ns["x"] = float("nan")
    assert not data_motion._sink_columns_proven(second) and not data_motion._sink_columns_proven(second)
    assert len(calls) == 3
    marker = id(first)
    del first, calls[:]
    assert marker not in data_motion._SINK_PROVEN_TIMELINES


def test_worker_widens_cycle_collection_only_for_itself():
    from run_exploring_hbf_native import widen_cycle_collection
    previous = gc.get_threshold()
    try:
        gc.set_threshold(700, 10, 10)
        widen_cycle_collection()
        assert gc.get_threshold()[0] > 700
        assert gc.isenabled()
    finally:
        gc.set_threshold(*previous)
