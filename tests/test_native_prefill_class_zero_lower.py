"""Zero local service bounds are legal; only a total prompt can pass its budget."""
from copy import deepcopy

import pytest

from heterollm_sim.native_prefill_chunk_adapter import _class_parts
from heterollm_sim.native_prefill_chunk_composition import compose_prefill_chunks
from heterollm_sim.native_prefill_chunk_probe import expand_class_intervals
from heterollm_sim.native_stationary import compose_layers
from heterollm_sim.native_layer_sampling import NativeLayerSamplingGuardError


def inputs(*, a=10000.0, b=1.0, width=2.0):
    ids = [f"l{i}" for i in range(8)]
    prepared = {"inventory": [{"layer_id": layer, "class_key": "a" if i % 2 == 0 else "b"}
                               for i, layer in enumerate(ids)]}

    def run(selected, start):
        time = start
        completions = {}
        for layer in selected:
            time += a if int(layer[1:]) % 2 == 0 else b
            completions[layer] = time
        return {"layer_ids": selected, "origin_ns": start, "layer_completion_ns": completions,
                "first_access_ns": {layer: width if int(layer[1:]) % 2 else 0.0 for layer in selected}}

    runs = {"calibration": run(ids[:3], 0.0), "holdout": run(ids[-3:], 100000.0),
            "entry_at_holdout_origin": run(ids[:1], 100000.0)}
    return prepared, runs


def prompt(prepared, parts):
    intervals = {row["class_key"]: row["step_ns"] for row in parts["classes"]}
    nodes = []
    for layer in prepared["inventory"]:
        nodes.append({"node_id": layer["layer_id"], "stage_id": "gpu",
                      "duration_ns": intervals[layer["class_key"]],
                      "dependencies": [nodes[-1]["node_id"]] if nodes else []})
    return compose_prefill_chunks([{"chunk_id": "prompt", "nodes": nodes,
                                     "last_layer_node_id": nodes[-1]["node_id"]}])


def test_negative_local_lower_is_clamped_with_unchanged_upper_and_can_pass_total_budget():
    prepared, runs = inputs()
    before = deepcopy(runs)
    parts = _class_parts(prepared, runs)
    b = next(row for row in parts["classes"] if row["class_key"] == "b")
    assert b["raw_lower"] == -1.0 and b["lower_clamped_to_zero"] and b["zero_lower_bound"]
    assert b["step_ns"] == {"lower": 0.0, "upper": 3.0}
    assert prompt(prepared, parts)["accepted"]
    assert runs == before
    # Decode/shared composition keeps its pre-existing guard.
    with pytest.raises(ValueError, match="drift_exhausts_lower_bound"):
        compose_layers(prepared["inventory"], runs, exit=False)


def test_zero_local_lower_does_not_force_acceptance_of_a_wide_total():
    prepared, runs = inputs(a=10.0, width=20.0)
    result = prompt(prepared, _class_parts(prepared, runs))
    assert not result["accepted"]
    assert result["reasons"] == ["native_prefill_independent_budget_exceeded"]
    assert result["metrics"] == {}


def test_total_zero_lower_still_rejects_without_manufacturing_positive_service():
    prepared, runs = inputs(a=1.0, width=2.0)
    for run in runs.values():
        run["first_access_ns"] = {layer: 2.0 for layer in run["layer_ids"]}
    result = prompt(prepared, _class_parts(prepared, runs))
    assert not result["accepted"]
    assert result["metric_bounds"]["prefill_pass_ns"]["lower"] == 0.0
    assert result["reasons"] == ["native_prefill_pass_lower_bound_not_positive"]


def test_positive_class_intervals_match_the_original_formula_exactly():
    prepared, runs = inputs(width=0.25)
    _, _, original = compose_layers(prepared["inventory"], runs, exit=False)
    parts = _class_parts(prepared, runs)
    assert [(r["class_key"], r["step_ns"]) for r in parts["classes"]] == [
        (r["class_key"], r["step_ns"]) for r in original["classes"]]
    assert all(not row["lower_clamped_to_zero"] for row in parts["classes"])


def test_optional_probe_uses_the_same_nonnegative_policy_without_changing_its_upper():
    prepared, runs = inputs(width=0.0)
    probe = {**deepcopy(runs["calibration"]), "sample_id": "history"}
    probe["first_access_ns"]["l1"] = 5.0
    result = _class_parts(prepared, runs, probe)
    b = next(row for row in result["classes"] if row["class_key"] == "b")
    assert b["cold_raw_lower"] == 1.0
    assert b["boundary_history_probe_raw_lower"] == b["raw_lower"] == -4.0
    assert b["step_ns"] == {"lower": 0.0, "upper": 6.0}
    assert b["lower_clamped_to_zero"]
    assert prompt(prepared, result)["accepted"]
    # The non-experimental probe helper is not modified or bypassed globally.
    cold = _class_parts(prepared, runs)
    with pytest.raises(NativeLayerSamplingGuardError, match="probe_drift_exhausts_lower_bound"):
        expand_class_intervals(prepared, {"runs": runs}, probe, cold)
