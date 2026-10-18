"""Whole Decode binding changes only explicitly declared context-shaped facts."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b200_hbm_hbf_scan_config import build_scan_scenario
from heterollm_sim import planner
from heterollm_sim.serving import BatchCohort, BatchItem


def assert_exact(actual, expected, path="$"):
    assert type(actual) is type(expected), path + ": type"
    if type(actual) is float:
        assert actual.hex() == expected.hex(), path
    elif is_dataclass(actual):
        for item in fields(actual):
            assert_exact(getattr(actual, item.name), getattr(expected, item.name), path + "." + item.name)
    elif isinstance(actual, Mapping):
        assert tuple(actual) == tuple(expected), path + ": key order"
        for key in actual:
            assert_exact(actual[key], expected[key], path + "." + str(key))
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected), path + ": length"
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_exact(left, right, path + "[" + str(index) + "]")
    elif isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        assert actual.tobytes() == expected.tobytes(), path
    else:
        assert actual == expected, path


def cohort(scenario, identity, context):
    item = BatchItem(scenario.workload.requests[0].request_id, "decode", 1, context - 1,
        main_tokens=1, draft_tokens=0, verifier_tokens=1, committed_tokens=1, logit_tokens=1)
    return BatchCohort(identity, "decode", 0., (item,))


def lower(scenario, context, identity, compiler):
    with planner._compilation_scope(scenario, compiler):
        return planner._lower_serving_cohort(scenario, cohort(scenario, identity, context))


def cache(compiler):
    return compiler.invariant(("serving_invocation_task_segment_cache_v1",), dict)


def check_contexts(scenario, contexts, monkeypatch):
    compiled = planner.CompilationContext(scenario, compiled_serving_invocation_segments=True)
    lower(scenario, contexts[0], "warm", compiled)
    assert len(cache(compiled)) == 1
    original = planner._compile_parallel_layer_body
    for index, context in enumerate(contexts[1:]):
        calls = []
        def counted(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)
        with monkeypatch.context() as active:
            active.setattr(planner, "_compile_parallel_layer_body", counted)
            actual = lower(scenario, context, "target-" + str(index), compiled)
        assert not calls, "a proven whole Token program must avoid rebuilding layer bodies"
        assert len(cache(compiled)) == 1
        reference = planner.CompilationContext(scenario, compiled_serving_invocation_segments=False)
        expected = lower(scenario, context, "target-" + str(index), reference)
        assert_exact(actual.schedule.tasks, expected.schedule.tasks)
        assert_exact(actual.extra_metadata, expected.extra_metadata)


@pytest.mark.parametrize("model", ["qwen2_5-0_5b", "qwen2_5-7b"])
@pytest.mark.parametrize("medium", ["HBM", "HBF"])
@pytest.mark.parametrize("boundary", ["power_of_two", "physical_page"])
def test_full_dag_rebinds_context_across_attention_and_page_boundaries(model, medium, boundary, monkeypatch):
    scenario = build_scan_scenario(model, medium, 1, 96)
    if boundary == "power_of_two":
        contexts = (31, 32, 33)
    else:
        compiler = planner.CompilationContext(scenario)
        with planner._compilation_scope(scenario, compiler):
            layer = compiler.execution_layers()[0]
            row_bytes = planner._kv_tensor_bytes(scenario, layer, 1, 1)
        tokens = 4096 // row_bytes
        assert tokens >= 3 and 4096 % row_bytes == 0
        contexts = (tokens - 1, tokens, tokens + 1)
    check_contexts(scenario, contexts, monkeypatch)


@pytest.mark.parametrize("model", ["qwen2_5-0_5b", "qwen2_5-7b"])
@pytest.mark.parametrize("medium", ["HBM", "HBF"])
def test_request_budget_remains_capacity_not_current_context(model, medium, monkeypatch):
    for budget in (64, 96):
        scenario = build_scan_scenario(model, medium, 1, budget)
        check_contexts(scenario, (31, 32, 33), monkeypatch)
        compiler = planner.CompilationContext(scenario, compiled_serving_invocation_segments=True)
        actual = lower(scenario, 33, "budget", compiler)
        contracts = [task.metadata["generic_kv_storage"] for task in actual.schedule.tasks
                     if task.metadata.get("generic_kv_storage") is not None]
        assert contracts
        assert all(owner["capacity_tokens"] == budget for contract in contracts for owner in contract["owners"])


@pytest.mark.parametrize("medium", ["HBM", "HBF"])
def test_same_attention_context_keeps_distinct_history_ranges(medium):
    scenario = build_scan_scenario("qwen2_5-0_5b", medium, 1, 64)
    def emit(compiler, history, phase):
        with planner._compilation_scope(scenario, compiler):
            plan, router = compiler.parallel_plan(), compiler.router()
            builder = planner._TaskBuilder(scenario.workload.requests[0])
            start = builder.add("input", planner.TaskCategory.COMPUTE)
            builder._rank_value_components[start] = {0: "gpu0"}
            lane = planner._ServingInvocationLane(0, scenario.workload.requests[0].request_id,
                "decode", 0, 32, history, 1, 1, True, True, False, False)
            group = planner._ServingInvocationGroup(0, "decode", (lane,), None, "serial_stateful_position")
            planner._compile_or_replay_serving_invocation(builder, scenario, plan, router, group,
                                                        phase=phase, dependencies=(start,))
            return tuple(builder.tasks)
    compiled = planner.CompilationContext(scenario, compiled_serving_invocation_segments=True)
    emit(compiled, 31, "warm")
    actual = emit(compiled, 17, "target")
    reference = planner.CompilationContext(scenario, compiled_serving_invocation_segments=False)
    expected = emit(reference, 17, "target")
    assert_exact(actual, expected)
