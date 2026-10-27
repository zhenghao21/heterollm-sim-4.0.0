"""PP import-pool capacity needs proof from the native admission ledger."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from heterollm_sim.native_layer_sampling import NativeLayerSamplingGuardError
from heterollm_sim.native_pipeline import _first_cohort_pool_admission
from heterollm_sim import serving


def boundary(*, pool=4, occupied=2, fits=()):
    seeds = [NS(spec=NS(request_id=f"s{i}", metadata={"warmup_seed": True}, arrival_ns=0.,
                        prompt_tokens=20+i, output_tokens=6-i), ep_group=i % 2,
                status=serving.RequestStatus.RUNNING, kv_pages=2) for i in range(4)]
    fresh = [NS(spec=NS(request_id=f"f{i}", metadata={}, arrival_ns=0., prompt_tokens=20,
                        output_tokens=6), ep_group=i % 2 if i < occupied else None,
                status=serving.RequestStatus.WAITING, kv_pages=2 if i < occupied else 0)
             for i in range(6)]
    calls = []
    def can_resize(state, pages):
        calls.append((state, pages))
        return state.ep_group in fits
    states = seeds + fresh
    active = NS(plan=NS(scheduler=NS(max_num_seqs=4)), _ep_degree=2, now=10.,
        _external_ingress={"preload_warmup_seed_kv": True, "max_inflight_requests": pool},
        _external_preloaded_requests=4, _state_values=tuple(states),
        _external_initialized={state.spec.request_id for state in seeds},
        _external_pending={f"f{i}": {"completion_ns": 1e12} for i in range(occupied)},
        _external_ready=set(), states={state.spec.request_id: state for state in states},
        ledger=NS(pages_for_tokens=lambda tokens: (tokens+15)//16, can_resize=can_resize))
    cohort = NS(items=tuple(NS(request_id=state.spec.request_id) for state in seeds),
                request_ids=tuple(state.spec.request_id for state in seeds))
    return active, cohort, calls


def test_native_kv_refusal_reports_all_group_holdings_without_mutating_live_requests():
    active, cohort, calls = boundary()
    before = [(state.ep_group, state.kv_pages, dict(state.__dict__)) for state in active._state_values]
    diagnostics = {}
    with pytest.raises(NativeLayerSamplingGuardError, match="native_pipeline_import_pool_not_full"):
        _first_cohort_pool_admission(active, cohort, diagnostics)
    assert diagnostics == {"physical_admission_cutoff_ns": 10., "kv_requests_held_by_group": [3, 3]}
    assert calls and {state.ep_group for state, _pages in calls} == {0, 1}
    assert all(pages == 2 for _state, pages in calls)
    assert all(all(state is not original for original in active._state_values) for state, _pages in calls)
    assert [(state.ep_group, state.kv_pages, dict(state.__dict__)) for state in active._state_values] == before


def test_a_full_inflight_pool_is_not_capacity_limited_even_when_transfers_are_slow():
    active, cohort, calls = boundary(pool=2)
    diagnostics = {}
    _first_cohort_pool_admission(active, cohort, diagnostics)
    assert diagnostics == {} and calls == []


@pytest.mark.parametrize("fits", [(0,), (1,), (0, 1)])
def test_a_native_ledger_that_can_admit_more_does_not_prove_pool_capacity(fits):
    active, cohort, _calls = boundary(fits=fits)
    diagnostics = {}
    _first_cohort_pool_admission(active, cohort, diagnostics)
    assert diagnostics == {}


@pytest.mark.parametrize("contract", ["no_preload", "seed_count", "not_loaded", "short_cohort",
    "future_seed", "future_fresh", "too_few_fresh", "unequal_kv", "rejected_fresh", "unstarted_import"])
def test_unknown_admission_contracts_are_not_reclassified_as_capacity(contract):
    active, cohort, calls = boundary()
    if contract == "no_preload":
        active._external_ingress["preload_warmup_seed_kv"] = False
    elif contract == "seed_count":
        active.plan.scheduler.max_num_seqs = 5
    elif contract == "not_loaded":
        active._external_preloaded_requests = 3
    elif contract == "short_cohort":
        cohort.items = cohort.items[:-1]
    elif contract == "future_seed":
        active.states["s3"].spec.arrival_ns = 11.
    elif contract == "future_fresh":
        active.states["f5"].spec.arrival_ns = 11.
    elif contract == "too_few_fresh":
        active._state_values = active._state_values[:6]
    elif contract == "unequal_kv":
        active.states["f5"].spec.prompt_tokens += 1
    elif contract == "rejected_fresh":
        active.states["f5"].status = serving.RequestStatus.REJECTED
    else:
        # An incomplete pool with room in the ledger remains executable;
        # absence of a started transfer alone is not capacity evidence.
        active._external_pending.clear()
        active._external_initialized = {f"s{i}" for i in range(4)}
        active.ledger.can_resize = lambda state, pages: True
    diagnostics = {"untouched": True}
    _first_cohort_pool_admission(active, cohort, diagnostics)
    assert diagnostics == {"untouched": True}
    if contract != "unstarted_import":
        assert calls == []


def test_real_pp_first_cohort_rejects_a_native_pool_reservation_shortfall(monkeypatch):
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.simulation_options import NativeWindowFastConfig, SimulationConfig
    from native_toy import _analytical_toy_hardware, toy_scenario
    model = toy_scenario(layer_count=4).model
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
    assert diagnostics["rejection_reasons"] == ["native_pipeline_import_pool_not_full"]
    assert diagnostics["kv_requests_held_by_group"] == [6]
    assert diagnostics["physical_complete_cohort_count"] == 0
    assert diagnostics["physical_visible_output_token_count"] == 0
    assert diagnostics["physical_ingress_accounting"]["preloaded_seed_requests"] == 4
    assert result.detailed is result.native_window_forecast is None
