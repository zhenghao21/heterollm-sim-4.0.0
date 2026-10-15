"""Runtime objects and callable lowerers use one explicit current contract."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.residency import AllocationResidencyManager, MemoryAccess
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.serving import (
    BatchCohort, BatchCost, _OnlineRuntime, _declared_serving_kv_quantum, _lower_cost,
)


def test_runtime_config_exposes_only_its_current_knobs():
    config = LlamaCppRuntimeConfig(batch=64, ubatch=32, context=2048,
                                  kv_type_k="f16", kv_type_v="q8_0")
    assert (config.batch, config.ubatch, config.context) == (64, 32, 2048)
    assert config.kv_capacity_contract()["per_slot_context_limit"] == 2048
    for removed in ("n_ctx", "n_ctx_seq", "batch_size", "ubatch_size",
                    "context_length", "cache_type_k", "cache_type_v"):
        with pytest.raises(AttributeError):
            getattr(config, removed)
    assert config.to_dict()["kv_type_k"] == "f16"


def test_lowerer_receives_scenario_and_cohort_even_with_optional_arguments():
    scenario = build_reference_scenario()
    cohort = BatchCohort("batch", "decode", 0.0, ())
    observed = []

    def lower(actual_scenario, actual_cohort=None):
        observed.append((actual_scenario, actual_cohort))
        return BatchCost(12.0)

    assert _lower_cost(lower, scenario, cohort).duration_ns == 12.0
    assert observed == [(scenario, cohort)]
    with pytest.raises(TypeError):
        _lower_cost(lambda _cohort: BatchCost(12.0), scenario, cohort)


@pytest.mark.parametrize("value", [BatchCost(12.0), 12.0,
                                    {"duration_ns": 12.0, "metadata": {"custom": True}}])
def test_current_callable_cost_forms_remain_supported(value):
    scenario = build_reference_scenario()
    cohort = BatchCohort("batch", "decode", 0.0, ())
    cost = _lower_cost(lambda _scenario, _cohort: value, scenario, cohort)
    assert cost.duration_ns == 12.0


def test_provider_interface_keeps_explicit_estimate_arguments():
    scenario = build_reference_scenario()
    cohort = BatchCohort("batch", "decode", 0.0, ())

    class Provider:
        def estimate(self, actual_scenario, actual_cohort):
            assert actual_scenario is scenario
            assert actual_cohort is cohort
            return BatchCost(15.0)

    assert _lower_cost(Provider(), scenario, cohort).duration_ns == 15.0


def test_residency_lease_consumes_real_access_result_and_rejects_partial_objects():
    manager = AllocationResidencyManager("hbm", 256, page_size_bytes=64)
    manager.register("data", 128, resident_bytes=128)
    result = manager.access(MemoryAccess("data", "read", offset_bytes=64, size_bytes=32))
    runtime = object.__new__(_OnlineRuntime)
    runtime.residency_manager = manager
    runtime._residency_lease_sequence = 0
    runtime._residency_group_lease_ids = []
    runtime._residency_cohort_consumer_ids = set()
    runtime._lease_residency_result(result, ("consumer",))
    lease = next(iter(manager.leases.values()))
    assert (lease.owner_id, lease.offset_bytes, lease.size_bytes) == ("data", 64, 32)
    with pytest.raises(AttributeError, match="size_bytes"):
        runtime._lease_residency_result(SimpleNamespace(owner_id="data"), ("other",))


def test_serving_quantum_does_not_accept_old_per_slot_field():
    scenario = build_reference_scenario()
    current = replace(scenario.placement, metadata={"capacity_ledger": {
        "kv_bytes_per_token": 64, "kv_bytes_per_slot": 1024,
    }})
    plan = SimpleNamespace(scenario=replace(scenario, placement=current))
    assert _declared_serving_kv_quantum(plan) == (64, 1024)
    for current_value in ({}, {"kv_bytes_per_slot": 1024}):
        old = replace(current, metadata={"capacity_ledger": {
            **current_value, "kv_per_slot_requested_bytes": 1024,
        }})
        with pytest.raises(ValueError, match="kv_per_slot_requested_bytes"):
            _declared_serving_kv_quantum(SimpleNamespace(scenario=replace(scenario, placement=old)))
