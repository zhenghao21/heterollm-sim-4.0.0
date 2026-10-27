"""Task-scoped ordinals preserve public edits, callbacks and physical rollback."""
from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.memory_types import AccessRequest
from test_owned_nand_sram_timeline import settings
from test_token_owned_history import bits, compare_physical_state


@pytest.mark.parametrize("name", ("busy_ns", "bytes_moved"))
def test_native_scope_reuses_ordinals_and_rebuilds_in_actual_append_order(name):
    timeline = ResourceTimeline()
    values = getattr(timeline, name)
    values.update(a=1, b=2)
    timeline._owned_metric_orders = {}
    assert timeline._ordered_metric_touches(name, values, {"b", "a"}) == ["a", "b"]
    first = timeline._owned_metric_orders[name][2]
    assert first is timeline._metric_key_orders[name][1]
    values["a"] = 11
    assert timeline._ordered_metric_touches(name, values, {"a", "b"}) == ["a", "b"]
    assert timeline._owned_metric_orders[name][2] is first
    # A dense reduction may skip this helper while appending z then c.
    values.update(z=3, c=4)
    assert timeline._ordered_metric_touches(name, values, {"c", "z"}) == ["z", "c"]
    assert list(first) == ["a", "b"]
    assert list(timeline._owned_metric_orders[name][2]) == ["a", "b", "z", "c"]
    # Replacement must rebuild even when the old and new roots have equal size.
    replacement = {"c": 4, "z": 3, "b": 2, "a": 11}
    setattr(timeline, name, replacement)
    assert timeline._ordered_metric_touches(name, replacement, set(replacement)) == list(replacement)


@pytest.mark.parametrize("name", ("busy_ns", "bytes_moved"))
@pytest.mark.parametrize("edit", ("reorder", "replace_key"))
def test_public_same_length_edits_are_seen_before_and_after_native_scope(name, edit):
    timeline = ResourceTimeline()
    values = getattr(timeline, name)
    values.update(a=1, b=2, c=3)
    assert timeline._ordered_metric_touches(name, values, set(values)) == ["a", "b", "c"]
    saved_public = timeline._metric_key_orders[name]
    value = values.pop("a")
    values["a" if edit == "reorder" else "z"] = value
    expected = ["b", "c", "a" if edit == "reorder" else "z"]
    timeline._owned_metric_orders = {}
    assert timeline._ordered_metric_touches(name, values, set(values)) == expected
    assert timeline._metric_key_orders[name] is not saved_public
    assert timeline._owned_metric_orders[name][2] is timeline._metric_key_orders[name][1]
    timeline._owned_metric_orders = None
    value = values.pop("b")
    values["b"] = value
    assert timeline._ordered_metric_touches(name, values, set(values)) == expected[1:] + ["b"]


@pytest.mark.parametrize("name", ("busy_ns", "bytes_moved"))
def test_each_new_scope_borrows_valid_public_ordinals_without_rebuilding(name):
    timeline = ResourceTimeline()
    values = getattr(timeline, name)
    values.update(a=1, b=2, c=3)
    timeline._ordered_metric_touches(name, values, {"a"})
    public = timeline._metric_key_orders[name]
    assert public[2] is True
    for _ in range(3):
        timeline._owned_metric_orders = {}
        assert timeline._ordered_metric_touches(name, values, {"c", "a"}) == ["a", "c"]
        assert timeline._metric_key_orders[name] is public
        assert timeline._owned_metric_orders[name][2] is public[1]
        timeline._owned_metric_orders = None


def test_public_custom_key_disables_scope_and_records_non_native_key_qualification():
    class CustomKey(str):
        pass
    timeline = ResourceTimeline(busy_ns={"a": 1., CustomKey("b"): 2.})
    timeline._owned_metric_orders = {}
    assert timeline._ordered_metric_touches("busy_ns", timeline.busy_ns, {"a", "b"}) == ["a", "b"]
    assert timeline._metric_key_orders["busy_ns"][2] is False
    assert timeline._owned_metric_orders is None


def test_scope_preserves_sparse_delta_float_bits_and_output_order():
    timeline = ResourceTimeline()
    timeline.busy_ns.update({f"foreign:{i}": 0. for i in range(64)})
    timeline.bytes_moved.update({f"foreign:{i}": 0 for i in range(64)})
    timeline.busy_ns.update(z=0.1, a=0.2)
    timeline.bytes_moved.update(z=2**54, a=2**54 + 3)
    before = timeline.metrics_snapshot()
    timeline.busy_ns["z"] += 0.2
    timeline.busy_ns["a"] += 0.1
    timeline.bytes_moved["z"] += 7
    timeline.bytes_moved["a"] += 9
    timeline.last_intervals.update(z=(0., 1.), a=(1., 2.))
    timeline._touched = {"a", "z"}
    timeline._owned_metric_orders = {}
    result = timeline._core_metrics_delta(before)
    assert list(result["resource_busy_ns"]) == ["z", "a"]
    assert [value.hex() for value in result["resource_busy_ns"].values()] == [
        "0x1.999999999999bp-3", "0x1.999999999999cp-4"]
    assert list(result["resource_bytes"].items()) == [("z", 7), ("a", 9)]


def _physical_pair():
    config = settings("nand")
    kernels = [UnifiedEventKernel(capture_physical_details=False) for _ in range(2)]
    for kernel in kernels:
        runtime = kernel.physical_runtime
        core = runtime.runtime(config, "dev").core
        for index in range(96):
            runtime.timeline.reserve(f"foreign:{index}", float(index), .125)
        core.submit(AccessRequest("warm", "read", 0, 256 * 4096, .125))
    task = TaskSpec("writes", "r", "writes", TaskCategory.COMPUTE,
        demands=(ResourceDemand("dev", 0., bytes_moved=3 * 257),),
        metadata={"physical_owner": "dev", "physical_memory_config": config,
            "memory_accesses": tuple({"physical_owner": "dev", "resource_id": "dev",
                "operation": "write", "address": index * 4096 + 3, "byte_count": 257}
                for index in range(3))})
    return kernels, task


@pytest.mark.parametrize("boundary", ("before_step", "observer"))
def test_import_and_observer_boundaries_have_no_scope_and_reread_public_order(boundary):
    (reference, candidate), first = _physical_pair()
    second = replace(first, task_id="next", dependencies=(first.task_id,))
    tasks = (first, second)
    def edit(kernel):
        timeline = kernel.physical_runtime.timeline
        assert timeline._owned_metric_orders is None
        for name in ("busy_ns", "bytes_moved"):
            values = getattr(timeline, name)
            key = next(iter(values))
            values[key] = values.pop(key)
    reference.submit(tasks)
    expected = []
    for index in range(2):
        if boundary == "before_step" and index == 1:
            edit(reference)
        expected.append(reference.step())
        if boundary == "observer" and index == 0:
            edit(reference)
    actual, before_count = [], 0
    def before(kernel):
        nonlocal before_count
        assert kernel.physical_runtime.timeline._owned_metric_orders is None
        if boundary == "before_step" and before_count == 1:
            edit(kernel)
        before_count += 1
    def observe(event):
        assert candidate.physical_runtime.timeline._owned_metric_orders is None
        actual.append(event)
        if boundary == "observer" and len(actual) == 1:
            edit(candidate)
    candidate._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
        owned_physical_frontiers=True, before_step=before, event_observer=observe)
    assert bits(actual) == bits(expected)
    compare_physical_state(candidate, reference)


def test_physical_validation_failure_leaves_no_scope_and_rolls_back(monkeypatch):
    from heterollm_sim import _token_program
    (_, kernel), task = _physical_pair()
    before = kernel.physical_runtime.snapshot()
    def reject(*args):
        assert kernel.physical_runtime.timeline._owned_metric_orders is None
        raise ValueError("reject after native writes")
    monkeypatch.setattr(_token_program, "_validate_event_timing", reject)
    with pytest.raises(ValueError, match="reject after native writes"):
        kernel._drain_prevalidated_compiled((task,), CompiledGraphLayout.compile((task,)),
            owned_physical_frontiers=True)
    assert kernel.physical_runtime.timeline._owned_metric_orders is None
    after = kernel.physical_runtime.snapshot()
    for name in ("timeline", "allocator_state", "committed_owners", "hbf_staging"):
        assert bits(after[name]) == bits(before[name])


@pytest.mark.parametrize("hook", ("core", "dispatcher", "undo", "before_get", "before_getitem", "reducer"))
def test_authored_writer_or_undo_hooks_do_not_receive_a_native_scope(monkeypatch, hook):
    from heterollm_sim import data_motion
    from heterollm_sim import _resource_ledger
    from heterollm_sim.memory_transfer import _TimelineBefore
    (_, kernel), task = _physical_pair()
    timeline = kernel.physical_runtime.timeline
    seen = []
    owner = kernel.physical_runtime.runtimes["dev"].core
    def authored_edit():
        seen.append(timeline._owned_metric_orders)
        for name in ("busy_ns", "bytes_moved"):
            values = getattr(timeline, name)
            size = len(values)
            key = next(iter(values))
            values[key] = values.pop(key)
            assert len(values) == size
    if hook == "core":
        original = owner.submit
        def submit(*args, **kwargs):
            seen.append(timeline._owned_metric_orders)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, "submit", submit)
    elif hook == "dispatcher":
        original = data_motion.execute_prepared_physical_task
        def dispatch(*args, **kwargs):
            seen.append(timeline._owned_metric_orders)
            return original(*args, **kwargs)
        monkeypatch.setattr(data_motion, "execute_prepared_physical_task", dispatch)
    elif hook == "undo":
        outer = timeline._begin_owned_transaction()
        original = outer.remember
        def remember(*args):
            seen.append(timeline._owned_metric_orders)
            return original(*args)
        monkeypatch.setattr(outer, "remember", remember)
    elif hook in ("before_get", "before_getitem"):
        name = "get" if hook == "before_get" else "__getitem__"
        original = getattr(_TimelineBefore, name)
        def before_get(self, *args):
            authored_edit()
            return original(self, *args)
        monkeypatch.setattr(_TimelineBefore, name, before_get)
    else:
        original = _resource_ledger.core_timeline_metrics_delta
        def reducer(*args):
            authored_edit()
            return original(*args)
        monkeypatch.setattr(_resource_ledger, "core_timeline_metrics_delta", reducer)
    kernel._drain_prevalidated_compiled((task,), CompiledGraphLayout.compile((task,)),
        owned_physical_frontiers=True)
    assert seen and all(value is None for value in seen)
    assert timeline._owned_metric_orders is None
    if hook == "undo":
        outer.rollback()


@pytest.mark.parametrize("name", ("core_timeline_metrics_delta", "timeline_metrics_delta"))
def test_reducer_callback_same_length_reorder_retains_public_path(monkeypatch, name):
    from heterollm_sim import _resource_ledger
    timeline = ResourceTimeline()
    timeline.busy_ns.update({f"foreign:{i}": 0. for i in range(32)})
    timeline.bytes_moved.update({f"foreign:{i}": 0 for i in range(32)})
    timeline.busy_ns.update(a=0.1, b=0.2)
    timeline.bytes_moved.update(a=1, b=2)
    token = timeline._begin_owned_transaction()
    before = timeline._core_metrics_snapshot()
    for key in ("a", "b"):
        timeline._remember_resource(key)
        timeline.busy_ns[key] += 0.25
        timeline.bytes_moved[key] += 7
        timeline.last_intervals[key] = (0., 1.)
    timeline._touched = {"a", "b"}
    if name == "timeline_metrics_delta":
        timeline._touched.update(timeline.busy_ns)
        timeline.last_intervals.update({key: (0., 0.) for key in timeline._touched - {"a", "b"}})
    seen = []
    original = getattr(_resource_ledger, name)
    def callback(*args):
        seen.append(timeline._owned_metric_orders)
        for values in (timeline.busy_ns, timeline.bytes_moved):
            size = len(values)
            values["a"] = values.pop("a")
            assert len(values) == size
        return original(*args)
    monkeypatch.setattr(_resource_ledger, name, callback)
    assert not timeline._can_own_metric_orders()
    result = timeline._core_metrics_delta(before)
    assert seen == [None]
    assert list(result["resource_busy_ns"]) == ["b", "a"]
    assert list(result["resource_bytes"]) == ["b", "a"]
    token.rollback()


@pytest.mark.parametrize("column", ("busy_ns", "bytes_moved"))
@pytest.mark.parametrize("callback_kind", ("compare", "subtract"))
@pytest.mark.parametrize("raises", (False, True))
def test_full_reducer_foreign_numeric_callback_invalidates_later_sparse_scope(
        column, callback_kind, raises):
    from heterollm_sim.data_motion import PhysicalRuntimeContext
    context = PhysicalRuntimeContext(settings("nand"), "dev", capture_details=False)
    timeline = context.timeline
    calls = []
    def edit():
        if calls:
            return
        calls.append(timeline._owned_metric_orders)
        # 'a' has not yet been visited by the full reducer's active iterator.
        # Reinsert it after 'b', retaining exact dict identity and length.
        for values in (timeline.busy_ns, timeline.bytes_moved):
            size = len(values)
            values["a"] = values.pop("a")
            assert len(values) == size
        if raises:
            raise RuntimeError("authored foreign number")
    class ForeignNumber:
        def __eq__(self, other):
            if callback_kind == "compare":
                edit()
                return True
            return False
        def __sub__(self, other):
            edit()
            return 0.0 if column == "busy_ns" else 0
    for name in ("busy_ns", "bytes_moved"):
        values = getattr(timeline, name)
        values["foreign-number"] = ForeignNumber() if name == column else 0
        values.update({f"ordinary:{index}": 0 for index in range(32)})
        values.update(a=1, b=2)
    # Owner qualification does not inspect the unrelated foreign number.
    outer = context.begin_transaction(owners=("dev",), _owned_timeline=True)
    assert outer["_owned_native_writers"]
    assert outer["timeline"].get("_owned_transaction") is not None
    assert timeline._can_own_metric_orders()
    scope = timeline._owned_metric_orders = {}
    for name in ("busy_ns", "bytes_moved"):
        assert timeline._ordered_metric_touches(name, getattr(timeline, name), {"a", "b"}) == ["a", "b"]
    assert scope
    before = timeline._core_metrics_snapshot()
    # The full fallback must inspect foreign-number despite it not being touched.
    timeline._touched = set(timeline.busy_ns) - {"foreign-number"}
    timeline.last_intervals.update({key: (0., 0.) for key in timeline._touched})
    if raises:
        with pytest.raises(RuntimeError, match="authored foreign number"):
            timeline._core_metrics_delta(before)
    else:
        timeline._core_metrics_delta(before)
    assert calls == [None]
    assert timeline._owned_metric_orders is scope and not scope
    before = timeline._core_metrics_snapshot()
    for key in ("a", "b"):
        timeline._remember_resource(key)
        timeline.busy_ns[key] += 0.25
        timeline.bytes_moved[key] += 7
        timeline.last_intervals[key] = (0., 1.)
    timeline._touched = {"a", "b"}
    result = timeline._core_metrics_delta(before)
    assert list(result["resource_busy_ns"]) == ["b", "a"]
    assert list(result["resource_bytes"]) == ["b", "a"]
    assert [float(value).hex() for value in result["resource_busy_ns"].values()] == [
        "0x1.0000000000000p-2", "0x1.0000000000000p-2"]
    assert list(result["resource_bytes"].values()) == [7, 7]
    timeline._owned_metric_orders = None
    context.commit_transaction(outer)
