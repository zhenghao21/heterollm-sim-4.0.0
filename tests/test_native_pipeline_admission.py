"""PP import-pool capacity needs proof from the native admission ledger."""
from dataclasses import replace

from heterollm_sim import serving


def test_real_pp_first_cohort_rejects_a_native_pool_reservation_shortfall(monkeypatch):
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
    from native_toy import _analytical_toy_hardware, toy_scenario
    model = toy_scenario(layer_count=16).model
    from exploring_hbf_native_config import build_native_comparison_scenario
    scenario = build_native_comparison_scenario("llama3_405b", "HBM4", "SHORT", tp=1, pp=2,
        pipeline_microbatches=2, batch_size=4, model_override=model, prompt_override=20,
        output_override=3, warmup_override=2, measurement_override=4)
    scenario = _analytical_toy_hardware(scenario, .00001)
    original = serving._KVLedger.can_resize
    def limit(ledger, state, pages):
        if pages > state.kv_pages and ledger.used_pages + pages - state.kv_pages > 6 * pages:
            return False
        return original(ledger, state, pages)
    monkeypatch.setattr(serving._KVLedger, "can_resize", limit)
    result = run_scenario(replace(scenario, simulation=SimulationConfig(mode="fast",
        native_window=NativeWindowFastConfig(max_wall_seconds=120))), retention_policy="aggregate")
    diagnostics = result.approximation
    assert diagnostics["rejection_reasons"] == ["native_stationary_import_pool_not_full_at_cutoff"]
    assert diagnostics["kv_requests_held_by_group"] == [6]
    assert diagnostics["physical_complete_cohort_count"] == 0
    assert diagnostics["physical_visible_output_token_count"] == 0
    assert diagnostics["physical_ingress_accounting"]["preloaded_seed_requests"] == 4
    assert result.detailed is result.native_window_forecast is None
