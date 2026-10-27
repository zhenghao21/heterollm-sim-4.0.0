"""Real owned Token tasks exercise the private NAND resource sink.

Source-only candidate. Do not run until the production candidate and its
compiled helpers have been installed. Positive cases fail if the sink is
not actually entered; the observer never wraps a native writer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import struct
import sys

import pytest

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "tests/test_native_metric_orders.py").is_file())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]

from heterollm_sim import data_motion, _nand_acceleration, _resource_ledger, _token_program
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.event_kernel import CompiledGraphLayout, UnifiedEventKernel
from heterollm_sim.memory_transfer import _TIMELINE_MAPS
from heterollm_sim.memory_types import AccessRequest, NandConfig
from heterollm_sim.nand_core import NandCore
from test_hbm_compiled_control import configuration as hbm_configuration
from test_nand_generic_buffered_acceleration import config as nand_configuration
from test_token_owned_history import bits, compare_physical_state

PAGE = 4096
NAND_OWNER, HBM_OWNER = "sink.flash", "sink.hbm"
RESOURCE_FIELDS = ("resource_busy_ns", "resource_bytes", "resource_last_intervals")


@contextmanager
def sink_mode(enabled):
    original = data_motion._native_multiowner_resource_sink
    if not enabled:
        # This new gate is not a dispatcher/writer. Disabling it leaves the
        # same native Token scope, submission and publication paths in use.
        data_motion._native_multiowner_resource_sink = lambda *args, **kwargs: False
    try:
        yield
    finally:
        data_motion._native_multiowner_resource_sink = original


class NativeTrace:
    """Observe Python returns without monkeypatching guarded entry points."""

    def __init__(self):
        self.gate = data_motion._native_multiowner_resource_sink.__code__
        self.task = data_motion.execute_prepared_physical_task.__code__
        self.submit = NandCore.submit.__code__
        self.bulk = _nand_acceleration.execute_buffered_full_pages.__code__
        self.scalar = NandCore._submit_resident_scalar.__code__
        self.gates, self.submits, self.bulk_calls, self.scalar_ids, self.bills = [], [], [], [], []

    def observe(self, frame, event, value):
        if event != "return":
            return
        local, code = frame.f_locals, frame.f_code
        if code is self.gate:
            runtime = local.get("runtime")
            self.gates.append((value is True, runtime is not None and
                               type(runtime.timeline._owned_metric_orders) is dict))
        elif code is self.submit and value is not None:
            self.submits.append((local["request"].request_id,
                local.get("_resource_sink") is not None,
                tuple(name for name in RESOURCE_FIELDS if name in value.counters)))
        elif code is self.bulk and value is not None:
            self.bulk_calls.append((local["request"].request_id,
                local.get("_resource_sink") is not None,
                local.get("deferred") is not None,
                tuple(name for name in RESOURCE_FIELDS if name in (local.get("_metrics") or {}))))
        elif code is self.scalar and value is not None:
            self.scalar_ids.append(local["request"].request_id)
        elif code is self.task and value is not None:
            self.bills.append((tuple(local[name] for name in
                ("resource_busy", "resource_bytes", "resource_last")),
                value.metadata["physical_execution_by_owner"]))

    def __enter__(self):
        self.previous = sys.getprofile()
        sys.setprofile(self.observe)
        return self

    def __exit__(self, *error):
        sys.setprofile(self.previous)


def make_fixture(*, count=8, pages=2, scalar=True, large=False,
                 foreign_groups=0, reorder=True):
    """Build a genuine mixed task; setup and warmup are outside measurement."""
    if large:
        from exploring_hbf_native_config import build_native_comparison_scenario
        from heterollm_sim.physical_contract import require_physical_memory_config
        scenario = build_native_comparison_scenario("llama4_maverick", "HBF", "SHORT",
                                                   batch_size=15144, tp=1, ep=8)
        component = next(row for row in scenario.hardware.components if row.component_id == "hbf0")
        nand = require_physical_memory_config(component)
        assert type(nand) is NandConfig
    else:
        nand = nand_configuration(host_granularity_bytes=256, max_outstanding_requests=32)
    hbm = hbm_configuration(channels=2)
    banks = (nand.channels * nand.targets_per_channel * nand.dies_per_target
             * nand.luns_per_die * nand.planes_per_lun)
    kernel = UnifiedEventKernel(capture_physical_details=False)
    runtime = kernel.physical_runtime
    for owner, config, warm_bytes in ((NAND_OWNER, nand, max(128, banks) * PAGE),
                                      (HBM_OWNER, hbm, 8192)):
        core = runtime.runtime(config, owner).core
        core.submit(AccessRequest("warm:" + owner, "read", 0, warm_bytes, .125))
    assert runtime.runtimes[NAND_OWNER].core._nand_numeric_state is not None
    assert runtime.runtimes[HBM_OWNER].core._hbm_numeric_state is not None
    timeline = runtime.timeline
    # Foreign plain operands exercise wide public-column qualification and
    # arbitrary-size byte totals without overflowing a device's int64 pack.
    foreign = [("foreign:huge", 2**80 + 3)]
    for group in range(foreign_groups):
        foreign.extend((f"foreign:{group}:array:{index}", None) for index in range(banks))
        foreign.extend((f"foreign:{group}:channel:{index}", 4096)
                       for index in range(nand.channels))
        foreign.append((f"foreign:{group}:host:read", 4096))
    for rid, moved in foreign:
        timeline.ready_ns[rid] = .125
        timeline.busy_ns[rid] = -0.0
        if moved is not None:
            timeline.bytes_moved[rid] = moved
        timeline.last_intervals[rid] = (0., .125)
        timeline.lane_available[rid] = [.125]
    if reorder:
        for field in ("busy_ns", "bytes_moved"):
            mapping = getattr(timeline, field)
            for key in tuple(mapping)[::3]:
                mapping[key] = mapping.pop(key)
    accesses = []
    def flash(number, single=False):
        accesses.append({"physical_owner": NAND_OWNER, "resource_id": NAND_OWNER,
            "operation": "read", "address": (number + 1) * banks * PAGE + (17 if single or pages == 2 else 0),
            "byte_count": 257 if single else pages * PAGE - (33 if pages == 2 else 0)})
    for index in range(count):
        flash(index)
    if scalar:
        flash(count, True)
        for index in range(count + 1, 2 * count + 1):
            flash(index)
    # B lies between two A groups, so owner subtotals cannot replace the
    # global input-request sequence. HBM uses its unchanged normal results.
    accesses.extend({"physical_owner": HBM_OWNER, "resource_id": HBM_OWNER,
        "operation": operation, "address": address, "byte_count": size}
        for operation, address, size in (("read", 3, 65), ("write", 512, 2048)))
    for index in range(2 * count + 1, 3 * count + 1):
        flash(index)
    task = TaskSpec("sink-task", "sink-request", "mixed private resource task", TaskCategory.COMPUTE,
        demands=tuple(ResourceDemand(owner, 0., bytes_moved=sum(row["byte_count"]
            for row in accesses if row["physical_owner"] == owner)) for owner in (NAND_OWNER, HBM_OWNER)),
        metadata={"physical_owner": NAND_OWNER, "physical_memory_config": nand,
            "physical_memory_configs": {NAND_OWNER: nand, HBM_OWNER: hbm},
            "memory_accesses": tuple(accesses)})
    return kernel, task, {"banks": banks, "pages": pages, "count": count,
        "scalar": scalar, "foreign_groups": foreign_groups,
        "busy_keys": len(timeline.busy_ns), "byte_keys": len(timeline.bytes_moved)}


def drain(kernel, task):
    tasks = (task,)
    return kernel._drain_prevalidated_compiled(tasks, CompiledGraphLayout.compile(tasks),
                                              owned_physical_frontiers=True)


def observe_run(kernel, task, enabled=True):
    with sink_mode(enabled), NativeTrace() as trace:
        events = drain(kernel, task)
    return events, trace


def assert_actual_hit(trace):
    assert trace.gates and all(accepted and scope for accepted, scope in trace.gates), trace.gates
    assert any(sink and deferred and not fields for _, sink, deferred, fields in trace.bulk_calls), trace.bulk_calls
    assert trace.submits and all(sink and not fields for _, sink, fields in trace.submits), trace.submits


def state_bits(kernel):
    """Read live physical state without publishing or exposing either Core."""
    runtime, result = kernel.physical_runtime, {}
    timeline = runtime.timeline
    result["timeline"] = tuple((name, bits(getattr(timeline, name))) for name in _TIMELINE_MAPS)
    result["recorder"] = (tuple(timeline._touched), timeline._intervals_truncated,
                           bits(timeline._intervals), bits(timeline._interval_payloads))
    cores = []
    for owner, active in runtime.runtimes.items():
        core = active.core
        packed = core.__dict__.get("_nand_numeric_state", core.__dict__.get("_hbm_numeric_state"))
        arrays = None if packed is None else tuple((array.dtype.str, array.shape, array.tobytes())
                                                   for array in packed.arrays)
        names = ("_acceptance_ns", "_cache_use", "_host_acceptance_ns", "_scheduler_direction", "_scheduler_streak")
        scalars = tuple((name, bits(core.__dict__[name])) for name in names if name in core.__dict__)
        queues = tuple((name, tuple(value)) for name in ("_inflight", "_inflight_operations")
                       if (value := core.__dict__.get(name)) is not None)
        cores.append((owner, bits(active.clock_ns), arrays, scalars, bits(queues)))
    result["cores"] = tuple(cores)
    return result


def test_real_owned_mixed_task_hits_sink_and_matches_complete_task_bits():
    reference, task, _ = make_fixture()
    candidate, candidate_task, _ = make_fixture()
    expected, baseline = observe_run(reference, task, False)
    actual, trace = observe_run(candidate, candidate_task)
    assert_actual_hit(trace)
    assert not any(accepted for accepted, _ in baseline.gates)
    assert trace.scalar_ids == ["sink-request:8"]
    assert len(trace.bulk_calls) > 16  # fast before and after the scalar, then A again.
    assert bits(actual) == bits(expected)
    assert state_bits(candidate) == state_bits(reference)
    compare_physical_state(candidate, reference)


def test_final_maps_are_independent_and_last_tuples_keep_public_identity():
    kernel, task, _ = make_fixture()
    events, trace = observe_run(kernel, task)
    assert_actual_hit(trace)
    (task_maps, owners), = trace.bills
    maps = list(task_maps)
    for owner_bill in owners.values():
        maps.extend(owner_bill[name] for name in RESOURCE_FIELDS)
    assert len({id(value) for value in maps}) == len(maps)
    last = events[0].task.metadata["physical_resource_last_intervals"]
    timeline = kernel.physical_runtime.timeline
    checked = 0
    for owner_bill in owners.values():
        for rid, interval in owner_bill["resource_last_intervals"].items():
            assert last[rid] is interval
            if bits(timeline.last_intervals[rid]) == bits(interval):
                assert timeline.last_intervals[rid] is interval
                checked += 1
    assert checked > 4
    original = dict(task_maps[0])
    owners[NAND_OWNER]["resource_busy_ns"]["test-only"] = 1.
    assert task_maps[0] == original and "test-only" not in timeline.busy_ns


def test_sink_math_retains_first_zero_big_integer_and_aba_request_order():
    task_maps, owners = ({}, {}, {}), {name: ({}, {}, {}) for name in ("A", "B")}
    last = (1., 3.)
    rows = (("A", 1e16, 2**80), ("B", -1e16, 7), ("A", 1., 2**81))
    expected = ({}, {}, {})
    for owner, value, moved in rows:
        metrics = {"resource_busy_ns": {"r": value, "zero": -0.0},
            "resource_bytes": {"r": moved}, "resource_last_intervals": {"r": last, "last-only": last}}
        _resource_ledger.consume_task_resource_sink(task_maps + owners[owner], metrics)
        for rid, amount in metrics[RESOURCE_FIELDS[0]].items():
            expected[0][rid] = expected[0].get(rid, 0.0) + amount
        for rid, amount in metrics[RESOURCE_FIELDS[1]].items():
            expected[1][rid] = expected[1].get(rid, 0) + amount
        _resource_ledger.merge_latest_intervals(expected[2], metrics[RESOURCE_FIELDS[2]])
    assert bits(task_maps) == bits(expected)
    assert task_maps[0]["r"] == 1. and task_maps[1]["r"] == 3 * 2**80 + 7
    assert struct.pack("!d", task_maps[0]["zero"]) == struct.pack("!d", 0.0)
    assert all(group[2]["r"] is last and group[2]["last-only"] is last for group in (task_maps, *owners.values()))


@pytest.mark.parametrize("name", ("_PHYSICAL_ADDITIVE_COUNTERS", "_PHYSICAL_PEAK_COUNTERS",
                                 "DRAM_OPTIONAL_COUNTER_FIELDS"))
def test_authored_counter_container_rejects_sink_and_retains_full_maps(monkeypatch, name):
    reference, task, _ = make_fixture()
    candidate, candidate_task, _ = make_fixture()
    original, calls = getattr(data_motion, name), []
    if type(original) is tuple:
        class Observed(tuple):
            def __iter__(self):
                calls.append("iterate")
                return super().__iter__()
    else:
        class Observed(frozenset):
            def __contains__(self, key):
                calls.append(key)
                return super().__contains__(key)
    monkeypatch.setattr(data_motion, name, Observed(original))
    if type(original) is frozenset:
        # The installed Cython reducer rejects builtin-container subclasses
        # before entering its body. Both paths must retain that real API
        # error and roll back, rather than inventing successful semantics.
        failures = []
        for kernel, row, enabled in ((reference, task, False),
                                     (candidate, candidate_task, True)):
            before = state_bits(kernel)
            with sink_mode(enabled), NativeTrace() as trace:
                with pytest.raises(TypeError, match="frozenset") as error:
                    drain(kernel, row)
            assert trace.gates and not any(accepted for accepted, _ in trace.gates)
            assert trace.submits and all(not sink and fields for _, sink, fields in trace.submits)
            assert state_bits(kernel) == before
            assert kernel.physical_runtime.timeline._owned_metric_orders is None
            failures.append((type(error.value), str(error.value)))
        assert failures[0] == failures[1]
        assert state_bits(candidate) == state_bits(reference)
        return
    calls.clear()
    expected, _ = observe_run(reference, task, False)
    expected_calls = list(calls)
    calls.clear()
    actual, trace = observe_run(candidate, candidate_task)
    assert trace.gates and not any(accepted for accepted, _ in trace.gates)
    assert trace.submits and all(not sink and fields for _, sink, fields in trace.submits)
    assert expected_calls and calls == expected_calls
    assert bits(actual) == bits(expected)
    assert state_bits(candidate) == state_bits(reference)


def test_authored_global_interval_callback_rejects_sink(monkeypatch):
    reference, task, _ = make_fixture()
    candidate, candidate_task, _ = make_fixture()
    original, calls = data_motion._merge_latest_resource_intervals, []
    def observed(target, incoming):
        calls.append(tuple(incoming))
        return original(target, incoming)
    monkeypatch.setattr(data_motion, "_merge_latest_resource_intervals", observed)
    expected, _ = observe_run(reference, task, False)
    previous = list(calls)
    calls.clear()
    actual, trace = observe_run(candidate, candidate_task)
    assert calls and calls == previous
    assert not any(accepted for accepted, _ in trace.gates)
    assert bits(actual) == bits(expected)


def test_post_task_exception_rolls_back_after_real_private_consumption(monkeypatch):
    reference, task, _ = make_fixture()
    candidate, candidate_task, _ = make_fixture()
    before = state_bits(candidate)
    original = _token_program._validate_event_timing
    def fail_after_task(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("after complete private physical task")
    with monkeypatch.context() as changes:
        changes.setattr(_token_program, "_validate_event_timing", fail_after_task)
        traces = []
        for kernel, row, enabled in ((reference, task, False), (candidate, candidate_task, True)):
            with sink_mode(enabled), NativeTrace() as trace:
                with pytest.raises(ValueError, match="complete private physical task"):
                    drain(kernel, row)
            traces.append(trace)
    assert_actual_hit(traces[1])
    assert state_bits(candidate) == before == state_bits(reference)
    assert candidate.physical_runtime.timeline._owned_metric_orders is None
    for kernel in (reference, candidate):
        assert "_nand_request_metric_orders" not in kernel.physical_runtime.runtimes[NAND_OWNER].core.__dict__
    # Re-enter the ordinary kernel after rollback. No task-local sink is
    # attached to either Core or timeline and no failed partial bill survives.
    resumed = []
    for kernel, enabled in ((reference, False), (candidate, True)):
        with sink_mode(enabled), NativeTrace() as trace:
            event = kernel.step()
        resumed.append(event)
        assert event is not None
    assert bits(resumed[0]) == bits(resumed[1])
    compare_physical_state(candidate, reference)


def test_real_finite_overflow_restores_the_task_after_a_private_successful_prefix():
    cases = []
    for enabled in (False, True):
        kernel, task, _ = make_fixture(count=10, scalar=False)
        runtime = kernel.physical_runtime
        core = runtime.runtimes[NAND_OWNER].core
        # Keep the already-warmed resident geometry. Only its read timing
        # changes, just as the existing deferred-calendar finite test does.
        core.config = replace(core.config, page_read_ns=1e307)
        runtime._owner_config_cache[NAND_OWNER] = tuple(
            replace(value, page_read_ns=1e307) for value in runtime._owner_config_cache[NAND_OWNER])
        configs = dict(task.metadata["physical_memory_configs"])
        configs[NAND_OWNER] = replace(configs[NAND_OWNER], page_read_ns=1e307)
        task = replace(task, metadata={**task.metadata,
            "physical_memory_config": configs[NAND_OWNER], "physical_memory_configs": configs})
        before = state_bits(kernel)
        with sink_mode(enabled), NativeTrace() as trace:
            with pytest.raises(ValueError, match="finite timing range") as error:
                drain(kernel, task)
        assert state_bits(kernel) == before
        assert runtime.timeline._owned_metric_orders is None
        assert "_nand_request_metric_orders" not in core.__dict__
        cases.append((kernel, trace, type(error.value), str(error.value)))
    assert_actual_hit(cases[1][1])
    assert len(cases[1][1].submits) >= 10
    assert cases[0][2:] == cases[1][2:]
    assert state_bits(cases[0][0]) == state_bits(cases[1][0])


def test_authored_false_accounting_keeps_original_callback_sequence_and_public_state():
    class FalseAccounting:
        def __init__(self):
            self.calls = []

        def __bool__(self):
            names = []
            frame = sys._getframe(1)
            while frame is not None and len(names) < 12:
                names.append(frame.f_code.co_name)
                frame = frame.f_back
            self.calls.append(tuple(names))
            return False

    runs = []
    for enabled in (False, True):
        kernel, task, _ = make_fixture(count=8, scalar=False)
        timeline = kernel.physical_runtime.timeline
        outer = timeline._begin_owned_transaction()
        authored = FalseAccounting()
        outer.accounting = authored
        try:
            events, trace = observe_run(kernel, task, enabled)
            state = state_bits(kernel)
            calls = tuple(authored.calls)
            assert trace.gates and not any(accepted for accepted, _ in trace.gates)
            assert trace.submits and all(not sink and fields for _, sink, fields in trace.submits)
            # The original Token scope genuinely visits this authored bool.
            # The additional predicate must reject its type without another
            # bool conversion, even though it eventually selects fallback.
            assert sum("_can_own_metric_orders" in row for row in calls) == 1
            assert not any("_native_multiowner_resource_sink" in row for row in calls)
            if enabled:
                assert all(scope for _, scope in trace.gates)
            runs.append((kernel, events, state, calls))
        finally:
            outer.accounting = False
            outer.finish()
    assert runs[0][3] == runs[1][3]
    assert bits(runs[0][1]) == bits(runs[1][1])
    assert runs[0][2] == runs[1][2]
    compare_physical_state(runs[0][0], runs[1][0])
