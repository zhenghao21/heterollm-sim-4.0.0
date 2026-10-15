"""Dispatch consumes current calibration identities and capacity metadata."""
from dataclasses import replace

import pytest

from heterollm_sim import kernel_level2_surfaces as surfaces
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.kernel_model import llama_blackwell_analytical_profile
from heterollm_sim.planner import compile_scenario, summarize_tensor_storage
from heterollm_sim.reference import build_reference_scenario


def test_current_calibrated_profile_uses_phase_and_kernel_specific_error():
    profile = llama_blackwell_analytical_profile("gpu", "runtime", calibrated=True)
    seen = set()
    for kernel in profile.kernels:
        if not kernel.samples:
            continue
        family = kernel.kernel_family.split("_", 2)[1]
        key = f"{kernel.phase}:{family}:{kernel.weight_formats[0]}"
        seen.add(key)
        assert kernel.validation_relative_error == surfaces.LEVEL2_VALIDATION_ERROR_BY_KEY[key]
    assert seen == set(surfaces.LEVEL2_VALIDATION_ERROR_BY_KEY)
    assert not hasattr(surfaces, "LEVEL2_VALIDATION_ERROR_BY_FORMAT")


def test_mmq_cannot_borrow_decode_validation_when_its_own_evidence_is_missing(monkeypatch):
    assert surfaces.LEVEL2_VALIDATION_ERROR_BY_KEY["decode:mmvq:q4_k"] > 0
    monkeypatch.delitem(surfaces.LEVEL2_VALIDATION_ERROR_BY_KEY, "prefill:mmq:q4_k")
    with pytest.raises(KeyError, match="prefill:mmq:q4_k"):
        llama_blackwell_analytical_profile("gpu", "runtime", calibrated=True)
    # An explicitly uncalibrated profile remains an independent current model.
    assert not any(kernel.samples for kernel in
                   llama_blackwell_analytical_profile("gpu", "runtime").kernels)


@pytest.mark.parametrize("field", ("phase", "kernel_family"))
def test_sample_rows_require_their_dispatch_identity(monkeypatch, field):
    row = next(dict(row) for row in surfaces._ROWS if row["format"] == "q4_k"
               and row["phase"] == "decode")
    del row[field]
    monkeypatch.setattr(surfaces, "_ROWS", (row,))
    with pytest.raises(KeyError, match=field):
        surfaces.level2_samples("q4_k")


def test_weight_staging_summary_counts_current_capacity_once_per_invocation():
    first = TaskSpec("copy-first", "r", "copy", TaskCategory.MEMORY, metadata={
        "weight_access_semantics": "full_weight_staging_before_row_lookup",
        "weight_read_invocation_id": "embedding-1", "rank_weight_capacity_bytes": 4096,
    })
    second = replace(first, task_id="copy-second")
    assert summarize_tensor_storage((first, second))["full_table_staging_bytes"] == 4096
    obsolete = replace(first, metadata={key: value for key, value in first.metadata.items()
                                       if key != "rank_weight_capacity_bytes"})
    obsolete = replace(obsolete, metadata={**obsolete.metadata, "rank_weight_bytes": 4096})
    assert summarize_tensor_storage((obsolete,))["full_table_staging_bytes"] == 0


def test_real_lowering_emits_only_current_rank_weight_capacity():
    scenario = build_reference_scenario()
    request = replace(scenario.workload.requests[0], prompt_tokens=1, output_tokens=1)
    scenario = replace(scenario, workload=replace(scenario.workload, requests=(request,), mtp=None))
    tasks = compile_scenario(scenario).tasks
    rows = [task.metadata for task in tasks if "rank_weight_capacity_bytes" in task.metadata]
    assert rows
    assert all("rank_weight_bytes" not in task.metadata for task in tasks)
    assert all(row["rank_weight_capacity_bytes"] > 0 for row in rows)
