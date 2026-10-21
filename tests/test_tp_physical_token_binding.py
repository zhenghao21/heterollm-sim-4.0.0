"""Native TP collectives keep exact physical slots on Decode template replay."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]

from exploring_hbf_native_config import build_native_comparison_scenario
from exploring_hbf_native_pilot import build_pilot_scenario
from heterollm_sim import planner
from heterollm_sim.data_motion import deferred_endpoint_construction
from heterollm_sim.execution_control import ExecutionControl, ExecutionCancelledError
from hbf_hbm_core_audit import CoreExecutionAudit
from heterollm_sim.reporting import run_scenario
from test_parameterized_token_binding import assert_exact, cohort
from heterollm_sim.serving import BatchCohort, BatchItem
import weakref


def scenario(tp):
    return build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", tp=tp,
        model_override=build_pilot_scenario("hbm_tp1").model,
        prompt_override=8, output_override=8, warmup_override=0, measurement_override=2)


def emit(scenario, context, compiler, name):
    with planner._compilation_scope(scenario, compiler), deferred_endpoint_construction():
        return planner._lower_serving_cohort(scenario, cohort(scenario, name, context))


def assert_task_values(actual, expected):
    # Task metadata's native contract is Mapping. The lazy compiler handoff
    # must expose the same ordered values as ordinary materialized lowering.
    assert_exact(tuple(replace(task, metadata=dict(task.metadata)) for task in actual), expected)


@pytest.mark.parametrize("tp", (2, 8))
def test_tp_decode_reuses_layer_bodies_and_rebinds_collective_addresses(tp, monkeypatch):
    configured = scenario(tp)
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True,
        _serving_invocation_lean_replay_capability=planner._SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY)
    first = emit(configured, 7, compiler, "warm")
    cache = compiler.invariant(("serving_invocation_task_segment_cache_v1",), dict)
    assert len(cache) == 1
    expected_compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=False)
    expected = emit(configured, 8, expected_compiler, "next")
    with monkeypatch.context() as patch:
        patch.setattr(planner, "_compile_parallel_layer_body",
            lambda *_args, **_kwargs: pytest.fail("a captured native TP Decode must reuse layer bodies"))
        actual = emit(configured, 8, compiler, "next")
    assert_task_values(actual.schedule.tasks, expected.schedule.tasks)
    assert_exact(actual.extra_metadata, expected.extra_metadata)
    first_addresses = [task.metadata["memory_accesses"][0]["address"]
        for task in first.schedule.tasks if task.metadata.get("physical_execution_role") == "construction_preview"]
    next_addresses = [task.metadata["memory_accesses"][0]["address"]
        for task in actual.schedule.tasks if task.metadata.get("physical_execution_role") == "construction_preview"]
    assert first_addresses and first_addresses != next_addresses


def test_priced_preview_cannot_enter_deferred_endpoint_binding():
    configured = scenario(2)
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True)
    with planner._compilation_scope(configured, compiler):
        result = planner._lower_serving_cohort(configured, cohort(configured, "priced", 7))
    endpoints = [task for task in result.schedule.tasks
                 if task.metadata.get("physical_execution_role") == "construction_preview"]
    assert endpoints
    assert all(planner._DeferredEndpointPhysicalBinding.capture(task) is None for task in endpoints)
    assert not compiler.invariant(("serving_invocation_task_segment_cache_v1",), dict)


def test_unknown_hook_after_native_capture_cannot_reuse_unpriced_endpoint_templates(monkeypatch):
    from heterollm_sim.dram_core import DramCore
    configured = scenario(2)
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True,
        _serving_invocation_lean_replay_capability=planner._SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY)
    emit(configured, 7, compiler, "warm-native")
    assert len(compiler.invariant(("serving_invocation_task_segment_cache_v1",), dict)) == 1
    calls = []
    original = DramCore.submit
    def custom(core, *args, **kwargs):
        calls.append(1)
        return original(core, *args, **kwargs)
    monkeypatch.setattr(DramCore, "submit", custom)
    actual = emit(configured, 8, compiler, "custom-hook")
    assert calls, "an authored cold callback must execute after native template capture"
    reference = planner.CompilationContext(configured, compiled_serving_invocation_segments=False)
    expected = emit(configured, 8, reference, "custom-hook")
    assert_task_values(actual.schedule.tasks, expected.schedule.tasks)
    assert_exact(actual.extra_metadata, expected.extra_metadata)


def test_endpoint_binding_rejects_non_request_generation():
    configured = scenario(2)
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True)
    result = emit(configured, 7, compiler, "initial")
    task = next(task for task in result.schedule.tasks
                if task.metadata.get("physical_execution_role") == "construction_preview")
    metadata = {**task.metadata, "memory_accesses": tuple({**row, "allocation_generation": 123}
                for row in task.metadata["memory_accesses"])}
    assert planner._DeferredEndpointPhysicalBinding.capture(replace(task, metadata=metadata)) is None


def test_tp_replacement_request_rebinds_kv_ownership_and_capacity():
    configured = scenario(2)
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True,
        _serving_invocation_lean_replay_capability=planner._SERVING_INVOCATION_LEAN_REPLAY_CAPABILITY)
    emit(configured, 7, compiler, "warm")
    request = configured.workload.requests[1]
    item = BatchItem(request.request_id, "decode", 1, 8,
        main_tokens=1, draft_tokens=0, verifier_tokens=1, committed_tokens=1, logit_tokens=1)
    next_cohort = BatchCohort("replaced", "decode", 0., (item,))
    with planner._compilation_scope(configured, compiler), deferred_endpoint_construction():
        actual = planner._lower_serving_cohort(configured, next_cohort)
    reference = planner.CompilationContext(configured, compiled_serving_invocation_segments=False)
    with planner._compilation_scope(configured, reference), deferred_endpoint_construction():
        expected = planner._lower_serving_cohort(configured, next_cohort)
    assert_task_values(actual.schedule.tasks, expected.schedule.tasks)
    assert_exact(actual.extra_metadata, expected.extra_metadata)


def test_large_tp_lowering_checks_cancellation_before_the_graph_is_finished():
    configured = scenario(8)
    calls = []
    control = ExecutionControl(cancellation_callback=lambda: bool(calls.append(1) or len(calls) >= 2))
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True,
        execution_control=control)
    with pytest.raises(ExecutionCancelledError):
        emit(configured, 7, compiler, "cancel-compilation")
    assert len(calls) == 2


def test_cached_tp_rebind_checks_cancellation():
    configured = scenario(2)
    compiler = planner.CompilationContext(configured, compiled_serving_invocation_segments=True)
    emit(configured, 7, compiler, "warm")
    compiler.execution_control = ExecutionControl(cancellation_callback=lambda: True)
    with pytest.raises(ExecutionCancelledError):
        emit(configured, 8, compiler, "cancel-rebind")


def test_native_tp_program_and_lazy_slots_preserve_live_core_sequence(monkeypatch):
    configured = scenario(2)
    with monkeypatch.context() as patch:
        patch.setattr(planner, "_parameterized_physical_decode", lambda *_args: False)
        with CoreExecutionAudit(max_call_records=10000) as reference_audit:
            reference = run_scenario(configured)
    with CoreExecutionAudit(max_call_records=10000) as actual_audit:
        actual = run_scenario(configured)
    assert_exact(actual.serving.request_metrics, reference.serving.request_metrics)
    assert_exact(tuple((row.cohort_id, row.kind, row.start_ns, row.end_ns, row.token_count)
                       for row in actual.serving.batches),
                 tuple((row.cohort_id, row.kind, row.start_ns, row.end_ns, row.token_count)
                       for row in reference.serving.batches))
    assert_exact(actual_audit.summary(), reference_audit.summary())


@pytest.mark.parametrize("defer", (False, True))
def test_live_physical_provider_does_not_retain_complete_cost_dags(defer, monkeypatch):
    configured = scenario(2)
    provider = planner.TopologyAwareBatchCostProvider(configured, defer_physical_execution=defer)
    class Cost(dict):
        pass
    calls = []
    def estimate(*_args, **_kwargs):
        calls.append(1)
        return Cost(duration_ns=1., metadata={})
    monkeypatch.setattr(planner, "_estimate_serving_cohort_cost", estimate)
    value = provider.estimate(configured, cohort(configured, "same", 7))
    retained = weakref.ref(value)
    del value
    assert (retained() is None) is defer
    provider.estimate(configured, cohort(configured, "same", 7))
    assert len(calls) == (2 if defer else 1)


def test_native_metadata_slots_bind_only_read_fields_and_preserve_aliases():
    shared = {"owner": "old.decode.buffer", "values": [1, 2]}
    source = {"constant": 3, "nested": shared, "alias": shared}
    captured, plan = planner._task_segment_capture_metadata(source,
        source_request_id="old", source_prefix="old.decode")
    arguments = dict(source_request_id="old", source_prefix="old.decode", request_id="new",
                     prefix="new.decode", counter_delta=0)
    view = planner._TaskSegmentBoundMetadata(captured, planner._task_segment_lazy_field_plans(plan),
        arguments, {}, None, None)
    assert view["constant"] == 3 and view.cache is None
    assert view["nested"] is view["alias"]
    expected = planner._task_segment_clone_metadata_with_plan(captured, plan, **arguments)
    assert_exact(dict(view), expected)
    assert source["nested"]["owner"] == "old.decode.buffer"


def test_native_metadata_slots_keep_unknown_copy_hooks_on_eager_path():
    assert planner._task_segment_lazy_field_plans(("dict", 1,
        (("custom", ("deepcopy", 2, ())),))) is None
