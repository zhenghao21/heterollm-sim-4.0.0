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


def test_shared_prefill_is_not_a_pass():
    scenario = prefill_toy_scenario(4)
    request = scenario.workload.requests[0]
    shared = replace(scenario, workload=replace(scenario.workload, request_count=2,
        requests=(request, replace(request, request_id="second"))))
    assert _pass(shared)[0].approximation["rejection_reasons"] == ["native_prefill_pass_requires_one_request"]


def _chunked_prefill(*, prompt=20, ep=1, model_chunks=False):
    scenario = prefill_toy_scenario(8, prompt=prompt, ep=ep)
    if model_chunks:
        from heterollm_sim.ir import build_model_graph_from_layer_specs, model_graph_execution_layers
        model = scenario.model
        layers = tuple(replace(layer,
            sequence_mixer="full_attention" if (i + 1) % 4 == 0 else "chunked_attention",
            attention_chunk_size=0 if (i + 1) % 4 == 0 else 4,
            use_rope=(i + 1) % 4 != 0)
            for i, layer in enumerate(model_graph_execution_layers(model.graph)))
        model = replace(model, graph=build_model_graph_from_layer_specs(model.name, layers,
            architecture="llama4", vocabulary_size=64, max_sequence_length=256,
            embedding_weight_bytes=32*64*2, output_weight_bytes=32*64*2, output_head_dtype="bf16"))
        from exploring_hbf_native_config import build_native_prefill_scenario
        from native_toy import _analytical_toy_hardware
        return _analytical_toy_hardware(build_native_prefill_scenario("llama4_maverick", "SHORT", tp=2,
            ep=ep, model_override=model, prompt_override=prompt), .00001)
    return replace(scenario, workload=replace(scenario.workload, scheduler=replace(
        scenario.workload.scheduler, prefill_chunk_tokens=8, max_num_batched_tokens=8, max_num_ubatch_tokens=8)))


@pytest.mark.parametrize("ep", [1, 2])
def test_chunked_full_pass_uses_real_committed_prefix_and_matches_detailed(monkeypatch, ep):
    scenario = _chunked_prefill(ep=ep)
    runtime = native_prefill._OnlineRuntime
    original, observed = runtime._schedule_execution_stages, []

    def schedule(self, cohort, *args, **kwargs):
        outcome = original(self, cohort, *args, **kwargs)
        if cohort.kind == "prefill":
            observed.append(self._live_cohort_metrics.layer_completion_ns())
        return outcome

    monkeypatch.setattr(runtime, "_schedule_execution_stages", schedule)
    result, full = _pass(scenario, full_pass=True)
    native = observed[:]
    observed.clear()
    detailed = run_scenario(replace(scenario, simulation=SimulationConfig(mode="detailed")),
                            retention_policy="aggregate").serving
    batches = [b for b in detailed.batches if b.kind == "prefill"]
    assert len(batches) == 3 and full["accepted"]
    assert native == observed  # All layers of all three chunks, including overlap.
    assert full["execution_mode"] == "complete_native_chunks"
    assert full["native_chunk_count"] == 3 and not result.layer_window_samples
    budget = full["error_budget"]
    assert budget["prompt_tokens"] == 19
    assert [(r["context_tokens"], r["prompt_tokens"]) for r in budget["chunks"]] == [(0, 8), (8, 8), (16, 3)]
    value = observed[-1]["layer-007"] - budget["pass_origin_ns"]
    assert budget["observed_pass_ns"] == value < batches[-1].end_ns - batches[0].start_ns
    assert full["metric_bounds"]["prefill_pass_ns"] == {
        "estimate": value, "lower": value, "upper": value, "relative_error_budget": 0.}
    assert budget["assumptions"] == []


@pytest.mark.parametrize("prompt", [8, 9, 10])
def test_model_fixed_chunks_keep_ring_boundaries_and_observe_the_whole_prompt(prompt):
    scenario = _chunked_prefill(prompt=prompt, model_chunks=True)
    _, full = _pass(scenario, full_pass=True)
    result, sampled = _pass(scenario)
    assert sampled["accepted"] and full["accepted"]
    bounds = sampled["metric_bounds"]["prefill_pass_ns"]
    assert bounds["lower"] <= full["error_budget"]["observed_pass_ns"] <= bounds["upper"]
    prefix = ((prompt - 2) // 4) * 4
    assert sampled["native_chunk_count"] == prefix // 4 + 1
    assert sampled["error_budget"]["prompt_tokens"] == prompt - 1
    chunks = sampled["error_budget"]["chunks"]
    assert [c["context_tokens"] for c in chunks] == list(range(0, prefix + 1, 4))
    assert chunks[-1]["prompt_tokens"] == prompt - 1 - prefix
    assert not result.layer_window_samples
    assert sampled["error_budget"]["assumptions"] == []


@pytest.mark.parametrize("tp,ep", [(2, 4), (4, 2), (8, 1)])
def test_fixed_chunk_prefill_matches_detailed_eight_gpu_layouts(monkeypatch, tp, ep):
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from exploring_hbf_native_config import build_native_prefill_scenario
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    from native_toy import _analytical_toy_hardware

    # Eight heads and KV heads keep every TP layout physically representable.
    h, layers = 64, []
    for index in range(8):
        moe, full = index % 2 == 1, (index + 1) % 4 == 0
        inter = 64 if moe else 128
        ffn = 3 * h * inter * (5 if moe else 1) + (h * 4 if moe else 0)
        layers.append(LayerSpec(f"layer-{index:03d}", "moe" if moe else "dense", h, inter, 8,
            kv_heads=8, attention_head_dim=8, dtype="bf16", weight_bytes=2 * (4 * h * h + ffn),
            sequence_mixer="full_attention" if full else "chunked_attention",
            attention_chunk_size=0 if full else 4, use_rope=not full,
            num_experts=4 if moe else 1, experts_per_token=1,
            shared_expert_intermediate_size=inter if moe else 0,
            metadata={"moe_routing": {"gating": "sigmoid", "normalize_selected_weights": False,
                "weight_scale": 1.0, "weight_application": "input", "selection_before_gating": True,
                "load_model": "deterministic_balanced_histogram"}} if moe else {}))
    name = "native-prefill-eight-gpu-fixed-chunks"
    model = ModelSpec(name=name, graph=build_model_graph_from_layer_specs(name, layers,
        architecture="llama4", vocabulary_size=128, max_sequence_length=256,
        embedding_weight_bytes=h * 128 * 2, output_weight_bytes=h * 128 * 2,
        output_head_dtype="bf16"))
    scenario = _analytical_toy_hardware(build_native_prefill_scenario("llama4_maverick", "SHORT",
        tp=tp, ep=ep, model_override=model, prompt_override=10), .00001)
    runtime = native_prefill._OnlineRuntime
    original, observed = runtime._schedule_execution_stages, []

    def schedule(self, cohort, *args, **kwargs):
        outcome = original(self, cohort, *args, **kwargs)
        if cohort.kind == "prefill":
            observed.append(self._live_cohort_metrics.layer_completion_ns())
        return outcome

    monkeypatch.setattr(runtime, "_schedule_execution_stages", schedule)
    result, prefill = _pass(scenario)
    native = observed[:]
    observed.clear()
    detailed = run_scenario(replace(scenario, simulation=SimulationConfig(mode="detailed")),
                            retention_policy="aggregate").serving
    assert prefill["accepted"] and prefill["execution_mode"] == "complete_native_chunks"
    assert not result.layer_window_samples
    assert len(native) == len(observed) == 3
    assert all(len(ends) == 8 for ends in native + observed)
    assert [{layer: end.hex() for layer, end in ends.items()} for ends in native] == [
        {layer: end.hex() for layer, end in ends.items()} for ends in observed]
    budget = prefill["error_budget"]
    assert [(chunk["context_tokens"], chunk["prompt_tokens"]) for chunk in budget["chunks"]] == [
        (0, 4), (4, 4), (8, 1)]
    last_layer_end = observed[-1]["layer-007"]
    value = last_layer_end - budget["pass_origin_ns"]
    assert budget["observed_pass_ns"] == value
    assert last_layer_end < detailed.batches[-1].end_ns
    assert prefill["metric_bounds"]["prefill_pass_ns"] == {
        "estimate": value, "lower": value, "upper": value, "relative_error_budget": 0.}
    assert budget["assumptions"] == []


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
