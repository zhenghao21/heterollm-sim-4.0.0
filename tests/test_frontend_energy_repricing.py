from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


_SPEC = spec_from_file_location("energy_repricing", Path(__file__).parents[1] / "tools/reprice_frontend_validation_energy.py")
assert _SPEC is not None and _SPEC.loader is not None
energy = module_from_spec(_SPEC)
_SPEC.loader.exec_module(energy)


def inputs(*, corrected=False):
    scenario = {"hardware_input": {"hardware": {"components": [{
        "component_id": "hostmem0", "kind": "host_memory", "execution_profile": {
            "profile_kind": "host_memory", "profile_id": "host0",
            "parameters": {"resource_id": "hostmem0.memory", "energy_pj_per_byte": 4}}}]}}}
    physical_energy = 512 if corrected else 256
    ledger = {"task_count": 2, "physical_bytes": 128, "physical_read_bytes": 64,
              "physical_write_bytes": 64, "resource_totals": {
                  "data0": {"owner": "hostmem0.memory", "bytes_moved": 128, "energy_pj": physical_energy}}}
    batch = {"cost": {"energy_pj": 1000, "metadata": {"physical_execution_scope": "persistent_live_kernel",
              "dram_traffic": ledger, "storage_traffic": {"task_count": 0}}}}
    summary_ledger = deepcopy(ledger)
    for key in ("physical_bytes", "physical_read_bytes", "physical_write_bytes"):
        summary_ledger[key] *= 2
    for key in ("bytes_moved", "energy_pj"):
        summary_ledger["resource_totals"]["data0"][key] *= 2
    raw = {"job_id": "j1", "status": "completed", "report": {
        "summary": {"batch_count": 2, "physical_live_batch_count": 2, "total_energy_pj": 2000,
                    "dram_traffic": summary_ledger, "storage_traffic": {"task_count": 0}},
        "requests": {"r0": {"status": "finished"}}, "batch_history": [deepcopy(batch), deepcopy(batch)]}}
    compact = deepcopy(raw)
    del compact["report"]["batch_history"]
    return scenario, compact, raw


def test_saved_physical_repricing_keeps_original_files_and_other_energy():
    scenario, compact, raw = inputs()
    before = deepcopy((scenario, compact, raw))
    result = energy.reprice_record(scenario, compact, raw, prefix="ui_test", input_accounting="not_revalidated")
    assert result["status"] == "repriced"
    assert result["original_total_energy_pj"] == 2000
    assert result["unchanged_nonphysical_energy_pj"] == 1488
    assert result["repriced_total_energy_pj"] == 2512
    assert result["owners"][0]["physical_bytes"] == 256
    assert result["new_simulation_executed"] is False
    assert (scenario, compact, raw) == before


def test_corrected_v2_execution_reconstructs_total_and_each_owner():
    result = energy.reprice_record(*inputs(corrected=True), prefix="ui_test", input_accounting="physical_profile_energy_v2")
    assert result["status"] == "validated_against_v2_execution"
    assert result["v2_reconstruction_absolute_difference_pj"] == 0


@pytest.mark.parametrize("corruption", ["authored_zero", "archive_override", "missing_owner", "physical_bytes",
    "nand", "nand_bytes_without_task_count", "missing_batch", "non_live", "batch_energy", "batch_resource_energy", "raw_identity"])
def test_ambiguous_or_incomplete_physical_evidence_is_refused(corruption):
    scenario, compact, raw = inputs()
    if corruption == "authored_zero":
        scenario["workload"] = {"metadata": {"physical_energy_pj_per_byte": 0}}
    elif corruption == "archive_override":
        raw["report"]["batch_history"][0]["cost"]["metadata"]["physical_energy_pj_per_byte_by_owner"] = {"hostmem0.memory": 0}
    elif corruption == "missing_owner":
        scenario["hardware_input"]["hardware"]["components"][0]["execution_profile"]["parameters"]["resource_id"] = "other"
    elif corruption == "physical_bytes":
        for job in (compact, raw):
            job["report"]["summary"]["dram_traffic"]["physical_bytes"] += 1
    elif corruption == "nand":
        for job in (compact, raw):
            job["report"]["summary"]["storage_traffic"]["task_count"] = 1
    elif corruption == "nand_bytes_without_task_count":
        for job in (compact, raw):
            job["report"]["summary"]["storage_traffic"]["physical_bytes"] = 4096
    elif corruption == "missing_batch":
        raw["report"]["batch_history"].pop()
    elif corruption == "non_live":
        raw["report"]["batch_history"][0]["cost"]["metadata"]["physical_execution_scope"] = "preview"
    elif corruption == "batch_energy":
        raw["report"]["batch_history"][0]["cost"]["energy_pj"] += 1
    elif corruption == "batch_resource_energy":
        raw["report"]["batch_history"][0]["cost"]["metadata"]["dram_traffic"]["resource_totals"]["data0"]["energy_pj"] += 1
    else:
        raw["job_id"] = "different"
    with pytest.raises(ValueError):
        energy.reprice_record(scenario, compact, raw, prefix="ui_test", input_accounting="not_revalidated")


def test_new_energy_version_label_does_not_override_a_failed_reconstruction():
    with pytest.raises(ValueError, match="new v2 execution"):
        energy.reprice_record(*inputs(), prefix="ui_test", input_accounting="physical_profile_energy_v2")


def test_generation_audit_requires_review_for_new_task_coefficient_producer(tmp_path):
    source = tmp_path / "src/heterollm_sim"
    source.mkdir(parents=True)
    (source / "other.py").write_text('def custom_task():\n    return {"physical_energy_pj_per_byte": 0}\n')
    with pytest.raises(ValueError, match="unreviewed physical energy"):
        energy.audit_generation_paths(tmp_path)
