"""Real in-flight PD imports continue on independent sample copies."""
from copy import deepcopy
import math

import pytest

from heterollm_sim import native_layer_sampling as layer
from heterollm_sim.execution_control import ExecutionControl
from test_native_external_kv import ingress_scenario, native_runtime
from test_native_layer_sampling import native_prepared  # noqa: F401 (fixture)
from test_token_owned_history import bits


def _executor_state(executor):
    return bits({"counters": executor.counters, "next_plan": executor._next_async_plan_id,
        "next_token": executor._next_async_phase_token,
        "states": {key: {name: value for name, value in state.items()
                         if name not in {"rollback_snapshot", "source_allocation", "relays"}}
                   for key, state in executor._async_plan_states.items()}})


@pytest.mark.parametrize("case", ("hbf_weights_kv_tp1", "hbm_tp1"))
def test_copy_continuation_submits_exactly_the_live_import_phases(case):
    runtime = native_runtime(ingress_scenario(case, count=2))
    runtime._arrive()
    runtime._start_external_imports()
    pending = list(runtime._external_pending)
    assert len(pending) == 2
    source, executor = runtime._execution_resource_kernel, runtime._external_executor
    before = _executor_state(executor)
    kernel, copied = layer._independent_kernel(source, executor)
    imports = layer._ConcurrentImports(kernel, copied, pending)
    imports.advance_until(math.inf)
    copy_completion = {state["request_id"]: state["completion_ns"] for state in imports.states}
    assert all(state["final"] for state in imports.states)
    assert _executor_state(executor) == before
    written = executor.counters["write_bytes"]
    while runtime._external_pending:
        runtime.now = runtime._next_external_phase()
        runtime._complete_external_imports()
    live = {event.request_id: event.timestamp_ns for event in runtime.events
            if event.event_type == "external_kv_import_complete"}
    assert copy_completion == live
    assert sum(row[1] for row in imports.writes) == executor.counters["write_bytes"]-written
    assert copied.counters["write_bytes"] == executor.counters["write_bytes"]


def test_continuation_requires_a_real_in_flight_import():
    runtime = native_runtime(ingress_scenario("hbm_tp1", count=2))
    runtime._arrive()
    runtime._start_external_imports()
    kernel, copied = layer._independent_kernel(runtime._execution_resource_kernel,
                                               runtime._external_executor)
    with pytest.raises(layer.NativeLayerSamplingGuardError, match="not in flight: r9"):
        layer._ConcurrentImports(kernel, copied, ["r0", "r9"])
    # A separately copied executor reserves other calendars, not the sample's.
    other = deepcopy(runtime._external_executor)
    with pytest.raises(layer.NativeLayerSamplingGuardError, match="not bound"):
        layer._independent_kernel(runtime._execution_resource_kernel, other)


def test_concurrent_layer_window_runs_real_PD_phases_and_leaves_source_unchanged(native_prepared):
    runtime = native_prepared["runtime"]
    # The miniature source admits its seed before later imports start; the
    # 405B cutoff already has two in flight. Start them as serving would.
    runtime._start_external_imports()
    pending = list(runtime._external_pending)
    assert len(pending) == 2
    prepared = layer.prepare_native_layer_windows(runtime, native_prepared["cohort"],
        native_prepared["cost"], native_prepared["stages"], native_prepared["ready"], ExecutionControl())
    source, executor = prepared["source"], runtime._external_executor
    layer_id = prepared["inventory"][0]["layer_id"]
    ids = prepared["by_layer"][layer_id]
    saved = layer._independent_kernel(source)[0].physical_runtime.snapshot()
    before = (bits(saved["timeline"]), _executor_state(executor), source.makespan_ns)
    quiet = layer._execute_window(source, prepared["specs"], ids, origin_ns=prepared["origin"],
        control=ExecutionControl(), sample_id="quiet")
    busy = layer._execute_window(source, prepared["specs"], ids, origin_ns=prepared["origin"],
        control=ExecutionControl(), sample_id="busy", concurrent_imports=(executor, pending))
    alone = layer.import_progress_alone(source, executor, pending,
        start_ns=prepared["origin"], end_ns=busy["end_ns"])
    after = layer._independent_kernel(source)[0].physical_runtime.snapshot()
    assert (bits(after["timeline"]), _executor_state(executor), source.makespan_ns) == before
    assert quiet["concurrent_imports"] is None
    observed = busy["concurrent_imports"]
    assert observed["request_ids"] == pending and observed["submitted_phases"] > 0
    assert alone["submitted_phases"] > 0
    assert busy["task_count"] == quiet["task_count"]
    assert list(quiet["layer_completion_ns"]) == [layer_id]
    assert quiet["layer_completion_ns"][layer_id] == quiet["end_ns"]
    assert all(busy["start_ns"] < moment <= busy["end_ns"] for moment in observed["write_completions_ns"])
