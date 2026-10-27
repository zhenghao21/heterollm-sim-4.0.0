"""End-to-end error experiments for explicitly enabled chunk layer sampling."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from heterollm_sim.native_prefill_chunk_inputs import (
    DECLARED_CHUNK_ADDRESS_ASSUMPTION, DECLARED_CHUNK_STATE_ASSUMPTION)
from heterollm_sim.serving import _OnlineRuntime
from test_native_prefill import _pass
from test_native_prefill_chunk_inputs import _live_boundary


def _experiment_scenario(*, tp=2, pp=1, ep=1, prompt=50, layer_count=8):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from exploring_hbf_native_config import build_native_prefill_scenario
    from heterollm_sim.ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    from native_toy import _analytical_toy_hardware

    h, layers = 64, []
    for index in range(layer_count):
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
    name = "native-prefill-chunk-sampling-experiment"
    model = ModelSpec(name=name, graph=build_model_graph_from_layer_specs(name, layers,
        architecture="llama4", vocabulary_size=128, max_sequence_length=256,
        embedding_weight_bytes=h * 128 * 2, output_weight_bytes=h * 128 * 2,
        output_head_dtype="bf16"))
    return _analytical_toy_hardware(build_native_prefill_scenario("llama4_maverick", "SHORT",
        tp=tp, pp=pp, ep=ep, model_override=model, prompt_override=prompt), .00001)


@pytest.mark.parametrize("tp,ep,prompt", [(2, 1, 50), (2, 4, 10), (4, 2, 10), (8, 1, 10)])
def test_declared_chunk_sampling_covers_full_native_reference_and_keeps_source(
        monkeypatch, record_property, tp, ep, prompt):
    scenario = _experiment_scenario(tp=tp, ep=ep, prompt=prompt)
    original = _OnlineRuntime.run_until_decode_steps
    checked_boundaries = []

    def run(self, *args, **kwargs):
        if self.plan.scenario.simulation.native_prefill.chunked_sampling:
            original_sampler = self._native_layer_sampler

            def sampler(active, cohort, *sample_args):
                before = _live_boundary(active)
                try:
                    return original_sampler(active, cohort, *sample_args)
                finally:
                    assert _live_boundary(active) == before
                    assert not active.batches and active._completed_decode_steps == 0
                    assert all(state.prefill_cursor == state.committed == 0
                               for state in active.states.values())
                    checked_boundaries.append(cohort.cohort_id)

            self._native_layer_sampler = sampler
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_OnlineRuntime, "run_until_decode_steps", run)
    result, forecast = _pass(scenario, chunked_sampling=True)
    assert checked_boundaries
    assert result.approximation["future_physical_state_advanced"] is False
    assert result.layer_window_samples
    assert forecast.get("native_chunk_count", 0) == 0
    contexts = list(range(0, prompt - 1, 4))
    assert sorted({sample.context_tokens_by_request["native-prefill-request"]
                   for sample in result.layer_window_samples}) == contexts
    budget = forecast["error_budget"]
    assert budget["native_chunk_count"] == budget["physical_full_cohorts_committed"] == 0
    assert budget["sampled_chunk_count"] == len(contexts)
    assert budget["conditional_accuracy"] is True
    assert budget["default_enabled"] is False
    assert [(chunk["context_tokens"], chunk["prompt_tokens"]) for chunk in budget["chunks"]] == [
        (context, min(4, prompt - 1 - context)) for context in contexts]
    assert sum(chunk["prompt_tokens"] for chunk in budget["chunks"]) == prompt - 1
    assumptions = budget["assumptions"]
    assert DECLARED_CHUNK_STATE_ASSUMPTION in assumptions
    assert DECLARED_CHUNK_ADDRESS_ASSUMPTION in assumptions
    _, reference = _pass(scenario, full_pass=True, max_wall_seconds=None)
    assert reference["accepted"] and reference["execution_mode"] == "complete_native_chunks"
    observed = reference["error_budget"]["observed_pass_ns"]
    bounds = forecast["metric_bounds"]["prefill_pass_ns"]
    experiment = {"tp": tp, "ep": ep, "chunk_count": len(contexts), "prompt_tokens": prompt - 1,
        "estimate_ns": bounds["estimate"], "full_native_ns": observed,
        "lower_ns": bounds["lower"], "upper_ns": bounds["upper"],
        "relative_error": abs(bounds["estimate"] - observed) / observed,
        "reference_in_interval": bounds["lower"] <= observed <= bounds["upper"],
        "conditional_assumption_count": len(assumptions)}
    record_property("chunk_sampling_experiment", experiment)
    print("chunk_sampling_experiment", experiment)
    assert experiment["reference_in_interval"], experiment
    assert experiment["relative_error"] <= .10, experiment
    assert forecast["accepted"], (experiment, result.approximation.get("rejection_reasons"))


def test_pipeline_chunk_sampling_requires_explicit_enablement():
    scenario = _experiment_scenario(pp=2, prompt=10)
    result, prefill = _pass(scenario)
    assert not prefill["accepted"] and not result.layer_window_samples
    assert result.approximation["rejection_reasons"] == ["native_prefill_chunked_sampling_not_enabled"]


def test_pipeline_chunk_sampling_matches_full_reference_and_keeps_stage_transfers():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from native_toy import _analytical_toy_hardware, toy_scenario
    from exploring_hbf_native_config import build_native_prefill_scenario

    scenario = _analytical_toy_hardware(build_native_prefill_scenario(
        "llama3_405b", "SHORT", tp=1, pp=2, ep=1,
        model_override=toy_scenario(layer_count=8).model, prompt_override=20), .00001)
    result, sampled = _pass(scenario, chunked_sampling=True,
                            calibration_layers=2, holdout_layers=2)
    _, full = _pass(scenario, full_pass=True, max_wall_seconds=None)

    assert sampled["accepted"], result.approximation
    assert sampled["execution_mode"] == "layer_samples_per_chunk"
    assert sampled["error_budget"]["physical_full_cohorts_committed"] == 0
    assert sampled["error_budget"]["conditional_accuracy"] is True
    bounds = sampled["metric_bounds"]["prefill_pass_ns"]
    observed = full["error_budget"]["observed_pass_ns"]
    assert bounds["lower"] <= observed <= bounds["upper"]
    assert abs(bounds["estimate"] - observed) / observed <= .10
    assert any(sample.physical_observations["category_time_ns"].get("communication", 0) > 0
               for sample in result.layer_window_samples)
    assert any("PP Prefill layer classes canonicalize" in assumption
               for assumption in sampled["error_budget"]["assumptions"])
