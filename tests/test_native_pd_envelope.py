"""Default endpoint sampling for concurrent PD imports and its explicit bounds."""

from copy import deepcopy

import pytest

from heterollm_sim import native_stationary as stationary
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.simulation_options import NativeWindowFastConfig, simulation_from_dict
from test_native_ep_stationary import _evaluate_fixture, _sampling_fixture, _stub_window
from test_native_stationary import _imports


def test_pd_envelope_is_enabled_by_default_in_objects_and_json():
    assert NativeWindowFastConfig().pd_concurrency_envelope is True
    config = simulation_from_dict({"mode": "fast", "native_window": {}})
    assert config.native_window.pd_concurrency_envelope is True


@pytest.mark.parametrize("enabled", [False, True])
def test_pd_envelope_explicit_setting_round_trips_through_strict_configuration(enabled):
    config = simulation_from_dict({"mode": "fast", "native_window": {"pd_concurrency_envelope": enabled}})
    assert config.native_window.pd_concurrency_envelope is enabled


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_pd_envelope_rejects_values_that_are_not_booleans(value):
    with pytest.raises(ValueError, match="pd_concurrency_envelope"):
        NativeWindowFastConfig(pd_concurrency_envelope=value)


def _terms():
    return {
        1: {"extra_ns": (1., 2.), "span_ns": (5., 6.)},
        3: {"extra_ns": (3., 4.), "span_ns": (8., 10.)},
    }


def test_envelope_uses_endpoint_bounds_without_inventing_a_measured_sample():
    terms = _terms()
    original = deepcopy(terms)
    enveloped = stationary._monotone_import_envelope(terms, {1, 2, 3})
    assert enveloped[1] == original[1] and enveloped[3] == original[3]
    assert enveloped[2] == {"extra_ns": (1., 4.), "span_ns": (5., 10.)}
    # The original endpoint measurements can still be reported as measured.
    assert terms == original


@pytest.mark.parametrize("counts", [{1}, {1, 2}, set()])
def test_no_middle_count_means_no_synthetic_term(counts):
    terms = {1: _terms()[1], 2: _terms()[3]}
    assert stationary._monotone_import_envelope(terms, counts) == terms


@pytest.mark.parametrize("missing", [1, 3])
def test_a_missing_measured_endpoint_does_not_create_middle_terms(missing):
    terms = _terms()
    del terms[missing]
    assert stationary._monotone_import_envelope(terms, {1, 2, 3}) == terms


@pytest.mark.parametrize("field", ["extra_ns", "span_ns"])
def test_disjoint_reversed_endpoint_intervals_reject_the_monotone_assumption(field):
    terms = _terms()
    terms[1][field] = (20., 21.)
    with pytest.raises(ValueError, match="native_stationary_PD_monotonicity_inconsistent"):
        stationary._monotone_import_envelope(terms, {1, 2, 3})


def _sample(monkeypatch, counts, envelope=None):
    _stub_window(monkeypatch)
    pending = [(f"g{group}-{index}", 1000 - group * 100 - index, group)
               for group in range(2) for index in range(4)]
    runtime, prepared = _sampling_fixture(2, pending)
    monkeypatch.setattr(stationary, "_group_refill_schedule",
                        lambda *args: {"busiest_landings": {count: 1 for count in counts}})
    return stationary.sample_stationary(runtime, prepared, ExecutionControl(), (4, 2), [],
        **({} if envelope is None else {"pd_concurrency_envelope": envelope}))["pd_imports"]["samples"]


def test_direct_sampler_defaults_to_endpoint_sampling(monkeypatch):
    default = _sample(monkeypatch, [1, 2, 3, 4])
    assert [len(row["request_ids"]) for row in default] == [1, 1, 4, 4]
    assert default == _sample(monkeypatch, [1, 2, 3, 4], True)


def test_sampling_keeps_two_sets_at_each_endpoint_and_their_ep_groups(monkeypatch):
    full = _sample(monkeypatch, [1, 2, 3, 4], False)
    endpoints = _sample(monkeypatch, [1, 2, 3, 4], True)
    assert [len(row["request_ids"]) for row in full] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert [len(row["request_ids"]) for row in endpoints] == [1, 1, 4, 4]
    assert endpoints == [row for row in full if len(row["request_ids"]) in {1, 4}]
    assert [row["destination_group"] for row in endpoints] == [0, 1, 0, 1]
    for row in endpoints:
        expected_layers = 2 if len(row["request_ids"]) == 1 else 4
        assert len(row["concurrent"]["layer_ids"]) == expected_layers
        assert all(identity.startswith(f"g{row['destination_group']}-") for identity in row["request_ids"])


@pytest.mark.parametrize("counts", [[], [1], [2], [1, 2]])
def test_k_max_at_most_two_keeps_every_existing_window(monkeypatch, counts):
    assert _sample(monkeypatch, counts, True) == _sample(monkeypatch, counts, False)


def test_one_k_max_sample_cannot_supply_an_envelope():
    samples = _imports(
        ("x", 100, 100, (0., 20.), [32., 42., 52.], 52., 3),
        ("y", 100, 100, (0., 20.), [32., 42., 52.], 52., 3),
        (("a", "b", "c"), 300, (100, 100, 100),
         ((0., 40.), (0., 40.), (0., 40.)), [35., 49., 59.], 59., (3, 3, 3)))
    terms, measured = stationary._import_extra(samples)
    assert [row["concurrent_imports"] for row in measured] == [1, 1, 3]
    assert set(terms) == {1}
    assert set(stationary._monotone_import_envelope(terms, {2, 3})) == {1}


def _evaluation(monkeypatch, *, envelope, terms):
    runtime, prepared, _options, _ages = _evaluate_fixture(
        monkeypatch, degree=2, batch=26, output=6, pool=16, warmup=5, measured=10)
    monkeypatch.setattr(stationary, "_step_service", lambda prepared, samples: (1000., 1000., {}))
    monkeypatch.setattr(stationary, "_import_extra", lambda samples: (deepcopy(terms), []))
    return stationary.evaluate_stationary(runtime, prepared, {},
        NativeWindowFastConfig(pd_concurrency_envelope=envelope))


def test_evaluation_brackets_full_estimate_and_reports_the_conditional_assumption(monkeypatch):
    endpoints = _terms()
    all_terms = {**endpoints, 2: {"extra_ns": (2., 3.), "span_ns": (6., 8.)}}
    full = _evaluation(monkeypatch, envelope=False, terms=all_terms)
    bounded = _evaluation(monkeypatch, envelope=True, terms=endpoints)
    assert full["accepted"] is bounded["accepted"] is True
    assert full["reasons"] == bounded["reasons"] == []
    old = full["metric_bounds"]["mean_complete_request_tpot_ns"]
    new = bounded["metric_bounds"]["mean_complete_request_tpot_ns"]
    assert new["lower"] <= old["lower"] <= old["estimate"] <= old["upper"] <= new["upper"]
    budget = bounded["error_budget"]
    terms = {row["concurrent_imports"]: row for row in budget["pd_import_terms"]}
    assert terms[2]["extra_ns"] == {"lower": 1., "upper": 4.}
    assert terms[2]["span_ns"] == {"lower": 5., "upper": 10.}
    assert terms[2]["measured_steps"] > 0 and terms[3]["measured_steps"] > 0
    assert stationary.SAME_CLASS_ADDRESS_ASSUMPTION in budget["assumptions"]
    assert any("nondecreasing" in assumption and "E(k)" in assumption and "T(k)" in assumption
               for assumption in budget["assumptions"])
    assert "pd_concurrency_envelope" not in full["error_budget"]
    assert budget["pd_concurrency_envelope"]["sampled_endpoints"] == [1, 3]
    assert budget["pd_concurrency_envelope"]["enveloped_counts"] == [2]


def test_evaluation_still_rejects_an_unsampled_maximum(monkeypatch):
    result = _evaluation(monkeypatch, envelope=True, terms={1: _terms()[1]})
    assert result["accepted"] is False
    assert result["reasons"] == ["native_stationary_concurrent_PD_imports_not_sampled"]


def test_evaluation_reports_inconsistent_monotonicity_as_a_rejection(monkeypatch):
    terms = _terms()
    terms[1]["span_ns"] = (20., 21.)
    result = _evaluation(monkeypatch, envelope=True, terms=terms)
    assert result["accepted"] is False
    assert result["reasons"] == ["native_stationary_PD_monotonicity_inconsistent"]
