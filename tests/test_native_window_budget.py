"""Conditional native window formulas do not manufacture physical results."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.native_window_budget import evaluate_native_window, single_slot_skeleton


@dataclass(frozen=True)
class FixtureModel:
    name: str


@dataclass
class FixtureScheduler:
    max_num_seqs: int = 1
    preemption_enabled: bool = False
    mixed_phase_batching: bool = False


def fixture(*, scale=1., hbf=False):
    requests = tuple(NS(request_id=f"r{i}", prompt_tokens=1000, output_tokens=20,
        metadata={"warmup_seed": i == 0}, arrival_ns=0., priority=0, deadline_ns=None) for i in range(4))
    parallel = NS(tp_degree=8, pp_degree=1, ep_degree=1)
    scenario = NS(name="native-hbf" if hbf else "native-hbm", model=FixtureModel(name="model"),
                  placement=NS(parallel=parallel))
    runtime = NS(plan=NS(requests=requests, scheduler=FixtureScheduler(), scenario=scenario,
        kv_policy=NS(dtype="bf16", capacity_pages=10000)),
        ledger=NS(pages_for_tokens=lambda n:(n+15)//16),
        _external_ingress={"require_full_batch": True, "warmup_decode_cohorts": 19,
                           "measurement_decode_cohorts": 38},
        _external_pending={}, _external_imported_requests=1,
        _execution_resource_kernel=NS(physical_runtime=NS(runtimes={"store":NS(core=NS(config=NS(
            kind=NS(value="HBF" if hbf else "HBM"),
            protocol_mode="generic_nand" if hbf else None)))})))
    options = NS(warmup_cohorts=1, calibration_cohorts=2, validation_cohorts=2,
                 target_relative_error=.1)
    events = single_slot_skeleton(requests, 19, 38)["events"]
    samples, now = [], 0.
    for index, event in enumerate(events[:6]):
        start = now
        duration = (1000. if event["kind"] == "prefill" else 100.)*scale
        now += duration
        samples.append({"kind":event["kind"], "request_ids":[event["request_id"]],
            "cohort_index":index+1, "start_ns":start, "end_ns":now, "execution_ns":duration,
            "commit_interval_ns":duration, "visible_output_tokens":1,
            "visible_commits":[{"request_id":event["request_id"], "timestamp_ns":now-10.*scale,
                                "visible_tokens":1}],
            "context_tokens":[{"request_id":event["request_id"], "context_tokens":event["context_tokens"]}]})
    evidence = {"schema":"native-window-budget-evidence/v1",
        "capacity":{"qualified":True, "physical_workspace_and_allocator_fragmentation_covered":True},
        "steady_contract":{"qualified":True, "basis":"fixture_native_fixed_maxplus_program"},
        "constant_path_floor_ns":0., "constant_path_floor_proof":"zero_is_universal_lower_bound",
        "visible_commit_lag_ns":{"qualified":True, "lower_ns":10.*scale,
            "upper_ns":10.*scale, "basis":"fixture_fixed_cleanup_task"},
        "future_guards":{name:{"qualified":True, "basis":"fixture_closed_native_program",
            "lower_guard_ns":0., "upper_guard_ns":0.} for name in (
            "pd_service_and_interference", "cache_and_page_phase", "refresh_and_row_state",
            "allocation_reuse", "cutoff_carry_in", "prefill_query")}}
    return runtime, samples, options, evidence


def test_single_slot_window_counts_replacement_first_outputs_and_complete_fresh_requests():
    runtime, _, _, _ = fixture()
    skeleton = single_slot_skeleton(runtime.plan.requests, 19, 38)
    assert skeleton["visible_output_tokens"] == 40  # 38 Decode + two new first outputs.
    assert [row["request_id"] for row in skeleton["complete_fresh"]] == ["r1", "r2"]
    assert len(skeleton["events"]) == 60


def test_closed_fixture_passes_and_does_not_advance_native_runtime():
    runtime, samples, options, evidence = fixture()
    before = deepcopy(vars(runtime))
    result = evaluate_native_window(runtime, samples, options, budget_evidence=evidence)
    assert result["accepted"], result["reasons"]
    assert vars(runtime) == before
    assert result["metrics"]["complete_fresh_requests"] == 2
    assert result["metrics"]["measurement_visible_output_tokens"] == 40
    assert result["predicted_visible_output_tokens"] == 54
    assert result["metrics"]["tokens_per_second_per_gpu"] == result["metrics"]["cluster_tokens_per_second"]/8
    duration = result["metric_bounds"]["measurement_duration_ns"]
    throughput = result["metric_bounds"]["cluster_tokens_per_second"]
    assert throughput["lower"] == 40e9/duration["upper"]
    assert throughput["upper"] == 40e9/duration["lower"]


def test_live_prefix_without_future_proofs_rejects_instead_of_defaulting_guards_to_zero():
    runtime, samples, options, _ = fixture()
    result = evaluate_native_window(runtime, samples, options)
    assert not result["accepted"]
    assert "native_future_pd_service_and_interference_bound_unavailable" in result["reasons"]
    assert not result["metrics"]
    assert result["error_budget"]["native_evidence"]["future_guards"]["cache_and_page_phase"]["upper_guard_ns"] is None


@pytest.mark.parametrize("guard", ("pd_service_and_interference", "cache_and_page_phase",
    "refresh_and_row_state", "allocation_reuse", "cutoff_carry_in", "prefill_query"))
def test_missing_future_component_rejects(guard):
    runtime, samples, options, evidence = fixture()
    del evidence["future_guards"][guard]
    result = evaluate_native_window(runtime, samples, options, budget_evidence=evidence)
    assert not result["accepted"]
    assert "native_future_"+guard+"_bound_unavailable" in result["reasons"]


def test_legacy_or_wrong_native_reference_cannot_satisfy_paired_gate():
    runtime, samples, options, evidence = fixture()
    result = evaluate_native_window(runtime, samples, options, paired_reference=object(), budget_evidence=evidence)
    assert not result["accepted"]
    assert "native_window_reference_contract_mismatch" in result["reasons"]


def test_matched_pair_preserves_hbf_over_hbm_direction_for_tpot_and_throughput():
    baseline, samples, options, evidence = fixture()
    accepted = evaluate_native_window(baseline, samples, options, budget_evidence=evidence)
    reference = accepted["error_budget"]["paired_reference"]
    runtime, samples, options, evidence = fixture(scale=2., hbf=True)
    result = evaluate_native_window(runtime, samples, options, paired_reference=reference, budget_evidence=evidence)
    assert result["accepted"], result["reasons"]
    ratios = result["error_budget"]["paired_budget_decision"]["ratios"]
    assert ratios["mean_complete_request_tpot_ns"]["estimate"] == 2.
    assert ratios["cluster_tokens_per_second"]["estimate"] == .5
    assert all(row["direction"] == "hbf/hbm" for row in ratios.values())


def test_cleanup_commit_lag_is_not_silently_replaced_with_cohort_end():
    runtime, samples, options, evidence = fixture()
    evidence["visible_commit_lag_ns"]["qualified"] = False
    result = evaluate_native_window(runtime, samples, options, budget_evidence=evidence)
    assert not result["accepted"]
    assert "native_future_visible_commit_cleanup_bound_unavailable" in result["reasons"]


def test_multislot_flag_alone_does_not_prove_future_full_batch_after_replacement():
    runtime, samples, options, evidence = fixture()
    runtime.plan.scheduler.max_num_seqs = 8
    result = evaluate_native_window(runtime, samples, options, budget_evidence=evidence)
    assert not result["accepted"]
    assert result["reasons"] == ["native_multislot_requires_live_native_runtime"]
