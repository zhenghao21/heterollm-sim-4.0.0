"""Small captured-task examples exercise the sampling/composition boundary."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from heterollm_sim import native_prefill_chunk_adapter as adapter
from heterollm_sim.native_layer_sampling import NativeLayerSamplingGuardError
from heterollm_sim.native_prefill_chunk_composition import compose_prefill_chunks


def fixture(monkeypatch, *, chunk_id="c0", fence=False, row_width=0.0):
    monkeypatch.setattr(adapter, "_class_parts", lambda *args, **kwargs: {
        "classes": [{"class_key": "first", "step_ns": {"lower": 2.0, "upper": 2.0}},
                    {"class_key": "last", "step_ns": {"lower": 3.0, "upper": 3.0}}]})
    specs = [SimpleNamespace(task_id=identity, dependencies=(), metadata={"execution_stage_id": stage})
             for identity, stage in (("a", "cpu"), ("b", "cpu"), ("l0", "gpu0"), ("l1", "gpu1"))]
    prepared = {"chunk_id": chunk_id, "origin": 0.0,
        "inventory": [{"layer_id": "layer0", "class_key": "first"},
                      {"layer_id": "layer1", "class_key": "last"}],
        "boundary": {"before_first_layer": ["a", "b"]}, "specs": specs,
        "by_layer": {"layer0": ["l0"], "layer1": ["l1"]},
        "stage_key_by_id": {"cpu": ("cpu0", 0), "gpu0": ("gpu0", 0), "gpu1": ("gpu0", 1)},
        "stage_ready_ns": {"cpu": 0.0, "gpu0": 0.0, "gpu1": 0.0},
        "stage_descriptors": [{"stage_id": stage, "participates_in_device_fence": stage != "cpu",
                               "requires_device_fence": fence}
                              for stage in ("cpu", "gpu0", "gpu1")],
        "chunk_input_boundary": {"assumptions": ["declared future state"]}}
    tasks = [
        {"node_id": "a", "stage_id": "cpu", "dependencies": [], "resource_ids": ["cpu", "dma"],
         "start_ns": 0.0, "end_ns": 10.0, "duration_ns": 10.0, "earliest_start_ns": 0.0,
         "demands": [{"resource_id": "cpu", "service_ns": 1.0},
                     {"resource_id": "dma", "service_ns": 10.0}], "resource_lanes": {"cpu": 0, "dma": 0}},
        {"node_id": "b", "stage_id": "cpu", "dependencies": [], "resource_ids": ["cpu"],
         "start_ns": 1.0, "end_ns": 2.0, "duration_ns": 1.0, "earliest_start_ns": 0.0,
         "demands": [{"resource_id": "cpu", "service_ns": 1.0}], "resource_lanes": {"cpu": 0}},
        # Capture may include a full reference's head: it must be filtered.
        {"node_id": "head", "stage_id": "gpu0", "duration_ns": 10000.0},
    ]
    calibration = {"prefill_terms": {"prefix_tasks": tasks, "layer_start_ns": {"layer0": 10.0}},
                   "layer_completion_ns": {"layer0": 12.0}, "first_access_ns": {"entry": row_width}}
    second = deepcopy(calibration)
    for row in second["prefill_terms"]["prefix_tasks"]:
        if row["node_id"] != "head":
            row["start_ns"] += 100.0
            row["end_ns"] += 100.0
            row["earliest_start_ns"] += 100.0
    second["prefill_terms"]["layer_start_ns"]["layer0"] += 100.0
    second["layer_completion_ns"]["layer0"] += 100.0
    return prepared, {"runs": {"calibration": calibration, "entry_at_holdout_origin": second}}


def test_prefix_short_resource_releases_before_long_task_end_and_head_is_ignored(monkeypatch):
    prepared, samples = fixture(monkeypatch)
    adapted = adapter.build_prefill_chunk_terms(prepared, samples, 0)
    result = compose_prefill_chunks([adapted["chunk"]], extra_edges=adapted["extra_edges"])
    bounds = result["node_bounds"]["c0"]
    assert result["accepted"] and result["metrics"]["prefill_pass_ns"] == 15
    assert bounds["prefix-start:b"]["start_ns"]["estimate"] == 1
    assert bounds["prefix-end:a"]["end_ns"]["estimate"] == 10
    assert not any("head" in identity for identity in bounds)
    assert len({row["stage_id"] for row in adapted["chunk"]["nodes"]
                if row["node_id"].startswith("layer-stage-start:")}) == 2


def test_first_layer_uses_measured_service_and_row_hull_instead_of_entry_span(monkeypatch):
    prepared, samples = fixture(monkeypatch, row_width=0.25)
    adapted = adapter.build_prefill_chunk_terms(prepared, samples, 0)
    assert adapted["diagnostics"]["first_layer_service_observations_ns"] == [2, 2]
    assert adapted["layer_intervals"]["layer0"] == {"lower": 1.75, "upper": 2.25}
    assert adapted["layer_intervals"]["layer1"] == {"lower": 3, "upper": 3}


def test_frontier_keeps_cpu_preparation_early_and_only_fences_device_stages(monkeypatch):
    prepared, samples = fixture(monkeypatch)
    first = adapter.build_prefill_chunk_terms(prepared, samples, 0)
    next_prepared, next_samples = fixture(monkeypatch, chunk_id="c1", fence=True)
    before = deepcopy(first["frontier"])
    second = adapter.build_prefill_chunk_terms(next_prepared, next_samples, 1, frontier=first["frontier"])
    assert first["frontier"] == before
    result = compose_prefill_chunks([first["chunk"], second["chunk"]],
                                    extra_edges=[*first["extra_edges"], *second["extra_edges"]])
    bounds = result["node_bounds"]["c1"]
    assert bounds["prefix-start:a"]["start_ns"]["estimate"] == 10
    assert bounds["layer-stage-start:0:0"]["start_ns"]["estimate"] == 20
    assert result["metrics"]["prefill_pass_ns"] == 25
    assert "declared future state" in second["assumptions"]


def test_lane_assignment_changes_require_a_new_boundary_model(monkeypatch):
    prepared, samples = fixture(monkeypatch)
    samples["runs"]["entry_at_holdout_origin"]["prefill_terms"]["prefix_tasks"][0]["resource_lanes"]["cpu"] = 1
    with pytest.raises(NativeLayerSamplingGuardError, match="lane_assignment_changed"):
        adapter.build_prefill_chunk_terms(prepared, samples, 0)


def test_missing_prefix_capture_is_rejected_instead_of_using_authored_cost(monkeypatch):
    prepared, samples = fixture(monkeypatch)
    samples["runs"]["calibration"]["prefill_terms"]["prefix_tasks"].pop(0)
    with pytest.raises(NativeLayerSamplingGuardError, match="prefix_tasks_not_observed"):
        adapter.build_prefill_chunk_terms(prepared, samples, 0)
