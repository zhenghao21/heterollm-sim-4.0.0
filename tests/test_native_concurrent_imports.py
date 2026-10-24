"""Real in-flight PD imports continue on independent sample copies."""
from copy import deepcopy
import math

import pytest

from heterollm_sim import native_layer_sampling as layer
from test_native_external_kv import ingress_scenario, native_runtime
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
