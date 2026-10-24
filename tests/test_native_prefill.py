"""Native Prefill pass of one request on the Prefill replica."""
from dataclasses import replace

import pytest

from heterollm_sim import native_prefill
from heterollm_sim.native_stationary import SAME_CLASS_ADDRESS_ASSUMPTION
from heterollm_sim.reporting import run_scenario
from heterollm_sim.simulation_options import (NativePrefillPassConfig, NativeWindowFastConfig, SimulationConfig,
                                              simulation_from_dict)

from native_toy import prefill_toy_scenario


def _pass(scenario, **options):
    result = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_prefill=NativePrefillPassConfig(**options))), retention_policy="aggregate")
    return result, result.approximation["prefill_pass"]


@pytest.mark.parametrize("ep", [1, 2])
def test_full_pass_window_reproduces_the_detailed_prefill_cohort(monkeypatch, ep):
    scenario = prefill_toy_scenario(8, ep=ep, full_pass=True)
    original, seen = native_prefill.sample_prefill, {}

    def sample(prepared, control, options, records, **kwargs):
        # The same copy run with the step suffix too must end where the
        # live cohort ends.
        boundary = prepared["boundary"]
        seen["run"] = native_prefill.run_window(prepared, control, [], "prefill_pass",
            [row["layer_id"] for row in prepared["inventory"]],
            (boundary.get("before_first_layer", []), boundary.get("after_last_layer", [])),
            prepared["origin"], "with-suffix")
        return original(prepared, control, options, records, **kwargs)

    monkeypatch.setattr(native_prefill, "sample_prefill", sample)
    result, prefill = _pass(scenario, full_pass=True, max_wall_seconds=None)
    batch = run_scenario(replace(scenario, simulation=SimulationConfig(mode="detailed")),
                         retention_policy="aggregate").serving.batches[0]
    assert (seen["run"]["origin_ns"], seen["run"]["end_ns"]) == pytest.approx((batch.start_ns, batch.end_ns),
                                                                            rel=0, abs=1e-6)
    assert prefill["accepted"] and [row.role for row in result.layer_window_samples] == ["prefill_pass"]
    observed = prefill["error_budget"]["observed_pass_ns"]
    # The pass ends with its last layer, before the output head.
    assert 0 < observed < batch.end_ns - batch.start_ns
    assert prefill["metric_bounds"]["prefill_pass_ns"]["lower"] < observed \
        < prefill["metric_bounds"]["prefill_pass_ns"]["upper"]
    assert prefill["error_budget"]["assumptions"] == []
    steps = prefill["error_budget"]["layer_steps_ns"]
    assert len(steps) == 7 and prefill["error_budget"]["entry_with_prefix_ns"] + sum(steps.values()) \
        == pytest.approx(observed, rel=1e-12)


@pytest.mark.parametrize("ep,layers", [(1, (3, 3)), (2, (3, 3)), (1, (2, 2))])
def test_sampled_pass_brackets_the_full_pass(ep, layers):
    scenario = prefill_toy_scenario(8, ep=ep)
    _, full = _pass(scenario, full_pass=True)
    result, sampled = _pass(scenario, calibration_layers=layers[0], holdout_layers=layers[1])
    assert sampled["accepted"] and result.approximation["effective_mode"] == "fast"
    bounds = sampled["metric_bounds"]["prefill_pass_ns"]
    assert bounds["lower"] <= full["error_budget"]["observed_pass_ns"] <= bounds["upper"]
    assert [row.role for row in result.layer_window_samples] == ["calibration", "holdout", "entry"]
    # Two layers each way step only MoE layers of the alternating stack; both
    # runs grow to three to step the dense class too.
    assert sampled["run_layers"] == {"calibration": max(layers[0], 3), "holdout": max(layers[1], 3),
                                     "entry_at_holdout_origin": 1}
    assert sampled["error_budget"]["assumptions"] == [SAME_CLASS_ADDRESS_ASSUMPTION]
    assert "exit_interval_ns" not in sampled["error_budget"]["composition"]


def test_runs_grow_until_both_step_every_layer_class():
    assert native_prefill._layer_counts(["d"] * 126, 2, 2) == (2, 2)
    assert native_prefill._layer_counts(["d", "m"] * 4, 2, 2) == (3, 3)
    # A Maverick-like period of four (dense/MoE x chunked/full).
    assert native_prefill._layer_counts(["dc", "mc", "dc", "mf"] * 12, 2, 2) == (4, 4)
    assert native_prefill._layer_counts(["d", "m"] * 4, 5, 2) == (5, 3)
    # A class only the first layer holds is never stepped.
    assert native_prefill._layer_counts(["e", "d", "d"], 2, 2)[0] == 4


def test_a_chunked_or_shared_prefill_is_not_a_pass():
    scenario = prefill_toy_scenario(4)
    chunked = replace(scenario, workload=replace(scenario.workload, scheduler=replace(
        scenario.workload.scheduler, prefill_chunk_tokens=8, max_num_batched_tokens=8, max_num_ubatch_tokens=8)))
    result, prefill = _pass(chunked)
    assert result.approximation["rejection_reasons"] == ["native_prefill_pass_requires_one_whole_prompt_forward"]
    assert not prefill["accepted"] and not result.layer_window_samples
    request = scenario.workload.requests[0]
    shared = replace(scenario, workload=replace(scenario.workload, request_count=2,
        requests=(request, replace(request, request_id="second"))))
    assert _pass(shared)[0].approximation["rejection_reasons"] == ["native_prefill_pass_requires_one_request"]


def test_native_prefill_option_is_its_own_fast_mode():
    with pytest.raises(ValueError, match="native_prefill requires fast mode"):
        SimulationConfig(mode="detailed", native_prefill=NativePrefillPassConfig())
    with pytest.raises(ValueError, match="without native_window"):
        SimulationConfig(mode="fast", native_window=NativeWindowFastConfig(),
                         native_prefill=NativePrefillPassConfig())
    parsed = simulation_from_dict({"mode": "fast", "native_prefill": {"full_pass": True, "holdout_layers": 3}})
    assert parsed.native_prefill == NativePrefillPassConfig(full_pass=True, holdout_layers=3)
    with pytest.raises(ValueError, match="unknown simulation.native_prefill fields"):
        simulation_from_dict({"mode": "fast", "native_prefill": {"layers": 3}})
    # Only a full pass may run without a wall limit.
    assert NativePrefillPassConfig(full_pass=True, max_wall_seconds=None).max_wall_seconds is None
    with pytest.raises(ValueError, match="only for a full pass"):
        NativePrefillPassConfig(max_wall_seconds=None)
