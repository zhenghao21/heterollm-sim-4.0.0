"""The Decode tail projection is read-only and names what needs detailed execution."""
from dataclasses import replace

from heterollm_sim.fast_state import project_fast_tail
from test_fast_prefix_runtime import runtime


def projected_runtime():
    runner = runtime(output=35)
    runner.run_until_decode_steps(2)
    return runner


def test_projection_counts_k_and_v_separately():
    result = project_fast_tail(projected_runtime())
    assert result.remaining_decode_steps == 32
    assert result.current_cached_tokens == 9 and result.final_cached_tokens == 41
    assert len(result.kv_buffers) == 2
    for row in result.kv_buffers:
        assert row["completed_page_appends"] == row["final_bytes"] // 4096 - row["current_bytes"] // 4096
        assert row["final_tail_bytes"] == row["final_bytes"] % 4096


def test_hbf_kv_requires_detailed_execution():
    result = project_fast_tail(projected_runtime())
    assert result.safe is False
    assert "fast_tail_requires_hbm_kv:hbf0.memory" in result.safety_reasons


def test_default_offload_ratio_without_target_is_inert():
    runner = projected_runtime()
    scenario = runner.plan.scenario
    scenario = replace(scenario, placement=replace(scenario.placement,
        kv_policy=replace(scenario.placement.kv_policy,
                          offload_component=None, offload_ratio=1.)))
    runner.plan = replace(runner.plan, scenario=scenario,
        kv_policy=replace(runner.plan.kv_policy, offload_component=None, offload_ratio=1.))
    result = project_fast_tail(runner)
    assert len(result.kv_buffers) == 2
    assert not any("offload" in reason for reason in result.safety_reasons)
    assert result.to_metadata()["physical_state_advanced"] is False
    assert result.to_metadata()["physical_tail_transactions_recorded"] is False


def test_no_detailed_prefix_cannot_be_reported_as_calibrated_tail():
    result = project_fast_tail(runtime(output=35))
    assert "exact_prefill_and_first_token_required" in result.safety_reasons
    assert result.safe is False


def test_caller_cannot_label_an_unexecuted_step_as_detailed():
    result = project_fast_tail(projected_runtime(), detailed_decode_steps=3)
    assert "detailed_decode_step_count_does_not_match_runtime" in result.safety_reasons
