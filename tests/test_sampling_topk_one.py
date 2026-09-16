"""Sampling-chain semantic regressions; no native process or timing fixture."""

from dataclasses import replace

import pytest

from heterollm_sim import planner
from heterollm_sim.config import HostOutputContract, SamplingPolicy
from heterollm_sim.ir import RequestSpec
from heterollm_sim.reference import build_reference_scenario


def _policy(*, top_k=1, min_keep=0):
    return SamplingPolicy(
        mode="greedy", temperature=0.0, implementation="llama_cpp_cpu_chain",
        top_k=top_k, top_p=0.95, min_p=0.05, min_keep=min_keep,
    )


def _tail(*, vocabulary=17, rows=1, top_k=1, min_keep=0,
          committed_rows=None, policy=None):
    base = build_reference_scenario()
    scenario = replace(
        base,
        host_output_contract=HostOutputContract(
            target_component_id="hostmem0", vocabulary_size=vocabulary,
            logits_dtype="fp32", logits_element_bytes=4,
            allocation_semantics="host_visible_logits_buffer",
        ),
        sampling_policy=policy if policy is not None else _policy(
            top_k=top_k, min_keep=min_keep,
        ),
    )
    builder = planner._TaskBuilder(RequestSpec("sampling", 0.0, 1, 1))
    final = planner._add_host_visible_logits_sampling_commit(
        builder, scenario, planner._parallel_plan(scenario),
        planner._topology_router(scenario), (), phase="prefill",
        logit_rows=rows, committed_rows=rows if committed_rows is None else committed_rows,
    )
    return tuple(builder.tasks), final


def _event(tasks, kind):
    matches = [task for task in tasks if task.metadata.get("event_kind") == kind]
    assert len(matches) == 1
    return matches[0]


def _service(task):
    return sum(demand.service_ns for demand in task.demands)


@pytest.mark.parametrize("min_keep", [0, 1, 2, 128])
def test_native_nonnegative_min_keep_preserved_without_clamping(min_keep):
    assert _policy(top_k=128, min_keep=min_keep).min_keep == min_keep


@pytest.mark.parametrize("min_keep", [-1, -100, True, False, 0.0, 1.5, "0"])
def test_min_keep_rejects_negative_boolean_and_noninteger_values(min_keep):
    with pytest.raises(ValueError, match="min_keep must be a non-negative integer"):
        _policy(min_keep=min_keep)


@pytest.mark.parametrize("top_k", [0, -1, True, False, 1.0, 129])
def test_top_k_domain_is_not_broadened(top_k):
    with pytest.raises(ValueError, match="top_k"):
        _policy(top_k=top_k)


def test_min_keep_must_still_not_exceed_top_k():
    with pytest.raises(ValueError, match="min_keep must not exceed top_k"):
        _policy(top_k=1, min_keep=2)


@pytest.mark.parametrize("vocabulary", [2, 17, 129, 32001])
@pytest.mark.parametrize("rows", [1, 4])
def test_one_element_heap_scans_materialized_tail_for_each_logit_row(vocabulary, rows):
    tasks, _ = _tail(vocabulary=vocabulary, rows=rows)
    candidates = _event(tasks, "cpu_logits_candidate_materialization")
    scan = _event(tasks, "cpu_logits_sampling")
    per_row_comparisons = vocabulary - 1
    assert candidates.metadata["logical_read_bytes"] == 4 * vocabulary * rows
    assert candidates.metadata["logical_write_bytes"] == 12 * vocabulary * rows
    assert scan.metadata["sampling_model"] == "llama_cpp_cpu_chain"
    assert scan.metadata["top_k_algorithm"] == "partial_sort_one_element_heap"
    assert scan.metadata["comparison_complexity_proxy"] == "O(V)"
    assert scan.metadata["comparison_complexity_timing_use"] == "metadata_only"
    assert scan.metadata["per_row_comparison_count"] == per_row_comparisons
    assert scan.metadata["comparison_count"] == rows * per_row_comparisons
    assert scan.metadata["logical_read_bytes"] == 4 * rows * per_row_comparisons
    assert scan.metadata["logical_write_bytes"] == 0
    assert scan.metadata["candidate_record_stride_bytes"] == 12
    assert scan.metadata["logical_read_source"] == "materialized_candidate_logit_field"
    assert scan.metadata["rows_execution"] == "serial_slot_loop"
    assert scan.metadata["row_service_aggregation"] == "serial_sum"
    assert scan.metadata["min_keep"] == 0
    assert _service(scan) > 0
    assert candidates.task_id in scan.dependencies
    for task in (candidates, scan):
        assert task.metadata["physical_memory_traffic_status"] == "unknown_not_charged"
        assert task.metadata["analytical_bytes"] == 0
        assert all(demand.bytes_moved == 0 for demand in task.demands)
        assert task.metadata["timing_completeness"] == "partial"
        assert task.metadata["sampling_model_status"] == "partial"
        assert task.metadata["strict_mathematical_lower_bound"] is False


def test_rows_are_serial_for_both_materialization_and_comparison():
    one, _ = _tail(rows=1)
    four, _ = _tail(rows=4)
    for kind in ("cpu_logits_candidate_materialization", "cpu_logits_sampling"):
        single = _event(one, kind)
        multi = _event(four, kind)
        assert _service(multi) == pytest.approx(4 * _service(single))
        assert multi.metadata["cost_model"]["serial_repetitions"] == 4
        assert multi.metadata["cost_model"]["row_service_aggregation"] == "serial_sum"
        assert {demand.resource_id for demand in multi.demands} == {
            demand.resource_id for demand in single.demands
        }


@pytest.mark.parametrize("vocabulary,top_k", [(1, 1), (1, 2), (2, 2), (2, 128), (128, 128)])
def test_no_tail_scan_when_top_k_covers_vocabulary(vocabulary, top_k):
    tasks, _ = _tail(vocabulary=vocabulary, top_k=top_k)
    scan = _event(tasks, "cpu_logits_sampling")
    assert scan.demands == ()
    assert _service(scan) == 0
    assert scan.metadata.get("comparison_count", 0) == 0
    assert scan.metadata["filter_chain_service_ns"] == 0
    assert scan.metadata["sampling_model_status"] == "partial"
    assert _event(tasks, "cpu_logits_candidate_materialization").metadata["logical_write_bytes"] == 12 * vocabulary


@pytest.mark.parametrize("top_k", [2, 40, 128])
def test_existing_larger_heap_scan_keeps_comparisons_and_metadata(top_k):
    tasks, _ = _tail(vocabulary=257, rows=3, top_k=top_k, min_keep=1)
    scan = _event(tasks, "cpu_logits_sampling")
    assert scan.metadata["top_k_algorithm"] == "std_partial_sort"
    assert scan.metadata["comparison_complexity_proxy"] == "O(V log K)"
    assert scan.metadata["filter_work_model"] == "top_k_mandatory_tail_scan"
    assert scan.metadata["per_row_comparison_count"] == 257 - top_k
    assert scan.metadata["comparison_count"] == 3 * (257 - top_k)
    assert scan.metadata["logical_read_bytes"] == 12 * (257 - top_k)
    assert scan.metadata["candidate_record_stride_bytes"] == 12
    assert scan.metadata["physical_memory_traffic_status"] == "unknown_not_charged"


def test_materialization_sampling_and_output_commit_dependencies_remain_ordered():
    tasks, final = _tail(rows=4, committed_rows=2)
    candidates = _event(tasks, "cpu_logits_candidate_materialization")
    scan = _event(tasks, "cpu_logits_sampling")
    completion = _event(tasks, "logits_d2h_completion_interrupt")
    commits = [task for task in tasks if task.metadata.get("event_kind") == "cpu_token_commit"]
    assert commits
    by_id = {task.task_id: task for task in tasks}
    def ancestors(task_id):
        found = set()
        pending = list(by_id[task_id].dependencies)
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(by_id[current].dependencies)
        return found
    assert completion.task_id in ancestors(candidates.task_id)
    assert candidates.task_id in ancestors(scan.task_id)
    assert scan.task_id in ancestors(final)
    assert all(task.metadata["commit_bytes"] == 2 * 4 for task in commits)
    assert all(task.metadata["committed_rows"] == 2 for task in commits)
    assert scan.metadata["comparison_count"] == 4 * 16


def test_zero_logit_rows_do_not_materialize_or_sample():
    tasks, _ = _tail(rows=0)
    assert not any(task.metadata.get("event_kind") in {
        "cpu_logits_candidate_materialization", "cpu_logits_sampling", "cpu_token_commit",
    } for task in tasks)


def test_explicit_chain_top_k_one_is_not_legacy_greedy_reduction():
    tasks, _ = _tail()
    assert _event(tasks, "cpu_logits_sampling").metadata["sampling_model_status"] == "partial"
    legacy_tasks, _ = _tail(policy=SamplingPolicy(mode="greedy", temperature=0.0))
    legacy_sampling = [task for task in legacy_tasks if task.metadata.get("event_kind") == "cpu_logits_sampling"]
    assert legacy_sampling
    assert all(task.metadata["sampling_model"] == "greedy_argmax" for task in legacy_sampling)
    assert not any(task.metadata.get("event_kind") == "cpu_logits_candidate_materialization" for task in legacy_tasks)
