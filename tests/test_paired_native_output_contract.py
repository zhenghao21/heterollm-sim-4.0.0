from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from heterollm_sim.config import scenario_from_dict
from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.reporting import online_summary_dict, run_scenario
from heterollm_sim.scalable_serving import execute_cost_schedule

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/frontend_native_validation_2026-10-07"


def test_compact_output_retains_stage_owner_without_changing_event_cost():
    task = TaskSpec(task_id="d2h", request_id="r", name="host logits", category=TaskCategory.COMMUNICATION,
        demands=(ResourceDemand("pcie", 17.0, bytes_moved=64),), metadata={
            "serving_output_stage": "device_output_completion", "output_source_component_id": "gpu0",
            "output_transfer_source_component_id": "gddr0", "large_unused_details": {"unused": 3},
        })
    schedule = SimpleNamespace(tasks=(task,), resource_capacities={}, resource_owners={})
    full = execute_cost_schedule(schedule, retain_task_metadata=True)
    compact = execute_cost_schedule(schedule, retain_task_metadata=False)
    assert full.makespan_ns == compact.makespan_ns == 17.0
    assert full.resource_busy_ns == compact.resource_busy_ns
    metadata = compact.execution_records[0].metadata
    assert metadata["output_source_component_id"] == "gpu0"
    assert metadata["output_transfer_source_component_id"] == "gddr0"
    assert "large_unused_details" not in metadata


@pytest.mark.parametrize("slug", ["qwen3_0_6b_f16", "qwen3_8_27b_mixed"])
def test_paired_output_contract_survives_actual_aggregate_stage_compaction(slug, monkeypatch):
    from heterollm_sim import planner
    scenario = scenario_from_dict(json.loads((REPORT / ("scenario_" + slug + "_512_128.json")).read_text(encoding="utf-8")))
    scenario = replace(scenario, workload=replace(scenario.workload, prompt_tokens=16, output_tokens=3,
        requests=tuple(replace(request, prompt_tokens=16, output_tokens=3) for request in scenario.workload.requests)))
    append_positions = set()
    original = planner._source_f32_kv_ranges
    def capture_ranges(contract, part, *, write=False):
        if write:
            append_positions.add(contract["append_start"])
        return original(contract, part, write=write)
    monkeypatch.setattr(planner, "_source_f32_kv_ranges", capture_ranges)
    result = run_scenario(scenario, retention_policy="aggregate")
    report = online_summary_dict(result)
    assert report["summary"]["completed_requests"] == 1
    assert report["summary"]["host_output_contract_status"] == "complete"
    assert report["summary"]["logits_precision_status"] == "consistent"
    assert report["summary"]["batch_count"] == 3
    assert append_positions == {0, 16, 17}, "second decode must not replay stale cache offsets"
    for batch in result.serving.batches:
        metadata = batch.cost.metadata
        assert metadata["execution_stage_source"] != "serial_fallback"
        if slug == "qwen3_0_6b_f16" and batch.kind == "decode":
            groups = metadata["kernel_predictions"]["groups"]
            assert sum(row["count"] for row in groups if row["kernel_family"] == "cuda_mmvf_fp16") == 169


def test_output_preparation_uses_native_settings_not_latency(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = spec_from_file_location("prepare_native_output_test", ROOT / "tools/prepare_frontend_native_cases.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    scenario = scenario_from_dict(json.loads((REPORT / "scenario_qwen3_0_6b_f16_512_128.json").read_text(encoding="utf-8")))
    native = json.loads((REPORT / "native_qwen3_0_6b_f16.json").read_text(encoding="utf-8"))
    native["summary"] = {"deliberately_invalid_latency": -12345}
    path = tmp_path / "native.json"
    path.write_text(json.dumps(native), encoding="utf-8")
    prepared = module.apply_native_output_contract(scenario, path)
    assert prepared.host_output_contract.logits_element_bytes == 4
    assert prepared.sampling_policy.top_k == 40
    assert prepared.sampling_policy.min_keep == 0
    assert prepared.sampling_policy.implementation == "llama_cpp_cpu_chain"
    assert prepared.component_profiles == scenario.component_profiles
    native["native_server_props"]["default_generation_settings"]["params"]["backend_sampling"] = True
    path.write_text(json.dumps(native), encoding="utf-8")
    with pytest.raises(ValueError, match="CPU sampling"):
        module.apply_native_output_contract(scenario, path)
