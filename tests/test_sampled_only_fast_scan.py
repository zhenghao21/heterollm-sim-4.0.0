"""Regression coverage for the sampling-only fast scan contract."""

from dataclasses import replace

from heterollm_sim.reporting import FastScenarioResult, run_scenario
from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig

from test_native_pipeline import pipeline_scenario


def test_pipeline_probe_stops_before_remaining_stage_samples(monkeypatch):
    """A failed two-cohort probe must not spend the remaining native samples."""
    import heterollm_sim.native_pipeline as pipeline

    calls = []
    original_execute = pipeline._execute_window

    def execute(*args, **kwargs):
        calls.append(kwargs.get("sample_id"))
        return original_execute(*args, **kwargs)

    def reject_after_probe(*_args, **_kwargs):
        raise ValueError("native_pipeline_independent_budget_exceeded")

    monkeypatch.setattr(pipeline, "_execute_window", execute)
    monkeypatch.setattr(pipeline, "compose_pipeline_window", reject_after_probe)
    scenario = pipeline_scenario()
    result = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=120))), retention_policy="aggregate")
    assert isinstance(result, FastScenarioResult)
    assert result.native_pipeline_forecast is None
    assert result.detailed is None
    assert len(calls) == 2
    assert result.approximation["rejection_reasons"] == [
        "native_pipeline_independent_budget_exceeded"]
    assert result.approximation["physical_complete_cohort_count"] == 0
