import json

import pytest

from heterollm_sim.runtime_residual import (
    DEVICE_PHASES,
    PHASES,
    SCHEMA,
    load_runtime_residual_calibration,
    runtime_calibration_from_dict,
    runtime_calibration_to_dict,
    RuntimeStructureMeasurements,
    runtime_structure_measurements_from_dict,
)
from heterollm_sim.serde import to_primitive
from heterollm_sim.kernel_model import KernelModelProfile, kernel_model_from_dict


def _evidence(path):
    timing_phases = PHASES + DEVICE_PHASES
    samples = []
    for topology, multiplier in (("chain", 1), ("fork_join", 2)):
        for count, split in ((2, "train"), (4, "train"), (3, "holdout")):
            timings = {}
            for index, phase in enumerate(timing_phases, 1):
                center = multiplier * index * (100 + 20 * count)
                timings[phase] = [center - 1, center, center + 1]
            samples.append({"topology": topology, "node_count": count, "split": split, "timings_ns": timings})
    data = {
        "schema": SCHEMA,
        "source_kind": "independent_synthetic_runtime_microbenchmark",
        "target_llm_latency_used": False,
        "measurement_boundary": "host_wall_and_cuda_event_separate",
        "hardware_id": "gpu-a",
        "runtime_id": "cuda-x-driver-y",
        "architecture": "sm_120",
        "device": "gpu-a",
        "cc": "sm_120",
        "driver_version": "y",
        "runtime_version": "x",
        "cpu_id": "cpu-a",
        "os_id": "windows",
        "samples": samples,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_piecewise_costs_are_topology_and_size_bound(tmp_path):
    path = tmp_path / "microbench.json"
    _evidence(path)
    calibration = load_runtime_residual_calibration(path, hardware_id="gpu-a", cc="sm_120",
                                                    max_validation_relative_error=0.01)
    assert calibration.qualified is True
    assert calibration.validation_relative_error < 0.01
    assert calibration.costs("chain", 3)["ordinary_submit"] == pytest.approx(160)
    assert calibration.costs("fork_join", 3)["ordinary_submit"] == pytest.approx(320)
    with pytest.raises(ValueError, match="unmeasured CUDA graph topology"):
        calibration.costs("unknown", 3)
    with pytest.raises(ValueError, match="outside measured range"):
        calibration.costs("chain", 5)


def test_serialized_calibration_rechecks_machine_identity_and_gate(tmp_path):
    path = tmp_path / "microbench.json"
    _evidence(path)
    calibration = load_runtime_residual_calibration(path, max_validation_relative_error=0.01)
    record = to_primitive(calibration)
    restored = runtime_calibration_from_dict(record, cpu_id="cpu-a", driver_version="y",
                                             max_validation_relative_error=0.01)
    assert restored.costs("chain", 3) == calibration.costs("chain", 3)
    with pytest.raises(ValueError, match="identity mismatch"):
        runtime_calibration_from_dict(record, cpu_id="different-cpu", max_validation_relative_error=0.01)
    case = record["validation_cases"][0]
    case["predicted_ns"] = case["measured_ns"] * 2
    case["absolute_error_ns"] = case["measured_ns"]
    case["relative_error"] = 1.0
    record["validation_relative_error"] = 1.0
    record["validation_relative_error_by_phase"][case["phase"]] = 1.0
    record["validation_absolute_error_ns_by_phase"][case["phase"]] = case["measured_ns"]
    with pytest.raises(ValueError, match="recomputed measurement"):
        runtime_calibration_from_dict(record, max_validation_relative_error=0.1)


def test_only_requested_lifecycle_phases_need_to_qualify(tmp_path):
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    for row in data["samples"]:
        if row["split"] == "holdout":
            row["timings_ns"]["first_launch_submit"] = [5000, 5001, 5002]
    path.write_text(json.dumps(data), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    assert not calibration.qualified
    assert calibration.costs("chain", 3, phases=("replay_submit",))["replay_submit"] == pytest.approx(960)
    with pytest.raises(ValueError, match="first_launch_submit"):
        calibration.costs("chain", 3, phases=("first_launch_submit",))
    restored = runtime_calibration_from_dict(runtime_calibration_to_dict(calibration))
    assert restored.costs("chain", 3, phases=("replay_submit",)) == {"replay_submit": 960}


@pytest.mark.parametrize("tamper", ["prediction", "error", "noise", "training", "qualification", "missing_phase"])
def test_serialized_measurement_claims_are_recomputed(tmp_path, tamper):
    path = tmp_path / "microbench.json"
    _evidence(path)
    record = runtime_calibration_to_dict(load_runtime_residual_calibration(path))
    case = record["validation_cases"][0]
    if tamper == "prediction":
        case["predicted_ns"] += 1
    elif tamper == "error":
        case["relative_error"] = 0.1
    elif tamper == "noise":
        case["absolute_noise_tolerance_ns"] += 1000000
    elif tamper == "training":
        record["topology_phase_samples"][case["topology"]][case["phase"]][0][1] *= 2
    elif tamper == "qualification":
        record["qualification_by_topology_phase"]["chain"]["capture"] = False
    else:
        record["validation_cases"] = [item for item in record["validation_cases"] if item["phase"] != "capture"]
    with pytest.raises(ValueError):
        runtime_calibration_from_dict(record)


def test_noise_cannot_qualify_arbitrarily_inaccurate_measurement(tmp_path):
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    for row in data["samples"]:
        if row["split"] == "holdout":
            row["timings_ns"]["ordinary_submit"] = [1, 1000000, 100000000]
    path.write_text(json.dumps(data), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    assert calibration.qualification_by_topology_phase["chain"]["ordinary_submit"] is False
    with pytest.raises(ValueError, match="not qualified"):
        calibration.costs("chain", 3, phases=("ordinary_submit",))


def test_tighter_restored_gate_recomputes_effective_qualification(tmp_path):
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    for row in data["samples"]:
        if row["split"] == "holdout":
            row["timings_ns"]["capture"] = [value * 1.1 for value in row["timings_ns"]["capture"]]
    path.write_text(json.dumps(data), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    assert calibration.qualified
    restored = runtime_calibration_from_dict(runtime_calibration_to_dict(calibration), max_validation_relative_error=0.01)
    assert not restored.qualified
    assert restored.qualification_policy["max_relative_error"] == 0.01


def test_kernel_profile_roundtrip_restores_per_phase_calibration(tmp_path):
    path = tmp_path / "microbench.json"
    _evidence(path)
    calibration = load_runtime_residual_calibration(path)
    profile = KernelModelProfile(hardware_id="gpu-a", runtime_id="cuda-x-driver-y",
                                 architecture="sm_120", runtime_calibration=calibration,
                                 runtime_host_resource_id="cpu0.cuda_submit", graph_enabled=True)
    restored = kernel_model_from_dict(to_primitive(profile))
    assert restored.runtime_calibration.costs("chain", 3) == calibration.costs("chain", 3)


def test_outlying_interval_cannot_block_qualified_local_interval(tmp_path):
    import copy
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    for topology in ("chain", "fork_join"):
        template = next(row for row in data["samples"] if row["topology"] == topology)
        for n, split in ((8, "train"), (6, "holdout")):
            row = copy.deepcopy(template)
            row.update(node_count=n, split=split)
            if split == "holdout":
                row["timings_ns"]["ordinary_submit"] = [100000, 100001, 100002]
            data["samples"].append(row)
    path.write_text(json.dumps(data), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    assert not calibration.qualified
    assert calibration.costs("chain", 3, phases=("ordinary_submit",)) == {"ordinary_submit": 160}
    with pytest.raises(ValueError, match="not qualified"):
        calibration.costs("chain", 6, phases=("ordinary_submit",))


def test_interval_without_independent_holdout_cannot_be_used(tmp_path):
    import copy
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    for topology in ("chain", "fork_join"):
        row = copy.deepcopy(next(row for row in data["samples"] if row["topology"] == topology))
        row["node_count"] = 8
        data["samples"].append(row)
    path.write_text(json.dumps(data), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    with pytest.raises(ValueError, match="not qualified"):
        calibration.costs("chain", 6, phases=("replay_submit",))


def test_diagnostic_is_exact_measured_data_with_uncertainty_not_prediction(tmp_path):
    path = tmp_path / "microbench.json"
    _evidence(path)
    calibration = load_runtime_residual_calibration(path)
    diagnostic = calibration.measured_diagnostic("chain", 2, phases=("ordinary_submit",))
    assert diagnostic["prediction_qualified"] is False
    assert diagnostic["phase_measurements"]["ordinary_submit"] == {
        "median_ns": 140, "repeat_mad_ns": 1, "minimum_ns": 139, "maximum_ns": 141,
        "repeat_count": 3, "source_split": "train"}
    assert calibration.measured_diagnostic("chain", 3)["phase_measurements"]["capture"]["source_split"] == "holdout"
    with pytest.raises(ValueError, match="exact independently measured"):
        calibration.measured_diagnostic("chain", 5)


def _experimental():
    timings = {phase: [999.0, 1000.0, 1001.0] for phase in PHASES + DEVICE_PHASES}
    summary = {phase: {"median_ns": 1000.0, "repeat_mad_ns": 1.0, "minimum_ns": 999.0,
                      "maximum_ns": 1001.0, "repeat_count": 3} for phase in timings}
    return RuntimeStructureMeasurements(hardware_id="gpu", runtime_id="cuda", architecture="sm_120",
        device="gpu", cc="sm_120", driver_version="driver", runtime_version="runtime",
        cpu_id="cpu", os_id="os", evidence="test", limitations=("synthetic kernel parameters",),
        samples=({"topology": "chain", "node_count": 2, "source_structures": [{"source_revision": "source"}],
                  "timings_ns": timings, "phase_measurements": summary},))


def test_experimental_costs_require_explicit_mode_and_exact_structure():
    evidence = _experimental()
    with pytest.raises(ValueError, match="explicit experimental"):
        evidence.costs("chain", 2)
    assert evidence.experimental_costs("chain", 2, phases=("replay_submit",)) == {"replay_submit": 1000}
    with pytest.raises(ValueError, match="exact independently measured"):
        evidence.experimental_costs("chain", 3)
    with pytest.raises(ValueError, match="independent runtime cost profile"):
        KernelModelProfile(hardware_id="gpu", runtime_id="cuda", architecture="sm_120",
                           runtime_calibration=evidence, runtime_host_resource_id="cpu.submit")
    profile = KernelModelProfile(hardware_id="gpu", runtime_id="cuda", architecture="sm_120",
        runtime_calibration=evidence, runtime_host_resource_id="cpu.submit",
        runtime_measurement_mode="experimental_exact_structure")
    restored = kernel_model_from_dict(to_primitive(profile))
    assert restored.runtime_calibration.qualified is False
    assert restored.runtime_calibration.experimental_costs("chain", 2) == evidence.experimental_costs("chain", 2)


def test_experimental_profile_cannot_claim_qualification_or_forge_summary():
    record = to_primitive(_experimental())
    record["qualified"] = True
    with pytest.raises(ValueError, match="cannot claim"):
        runtime_structure_measurements_from_dict(record)
    record["qualified"] = False
    record["samples"][0]["phase_measurements"]["replay_submit"]["median_ns"] *= 2
    with pytest.raises(ValueError, match="recomputed measurement"):
        runtime_structure_measurements_from_dict(record)


def test_failed_update_requires_its_own_independent_measurement():
    evidence = _experimental()
    with pytest.raises(ValueError, match="no independent experimental"):
        evidence.experimental_costs("chain", 2, phases=("update_failure",))
    record = to_primitive(evidence)
    sample = record["samples"][0]
    sample["timings_ns"]["update_failure"] = [400.0, 500.0, 600.0]
    sample["phase_measurements"]["update_failure"] = {"median_ns": 500.0, "repeat_mad_ns": 100.0,
        "minimum_ns": 400.0, "maximum_ns": 600.0, "repeat_count": 3}
    restored = runtime_structure_measurements_from_dict(record)
    assert restored.experimental_costs("chain", 2, phases=("update_failure",)) == {"update_failure": 500}


def test_unqualified_data_is_never_implicitly_qualified(tmp_path):
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    for row in data["samples"]:
        if row["split"] == "holdout":
            row["timings_ns"]["ordinary_submit"] = [5000, 5001, 5002]
    path.write_text(json.dumps(data), encoding="utf-8")
    calibration = load_runtime_residual_calibration(path)
    assert calibration.qualified is False
    assert calibration.validation_relative_error > 0.2
    with pytest.raises(ValueError, match="not qualified"):
        calibration.costs("chain", 3)


def test_rejects_llm_latency_or_out_of_range_holdout(tmp_path):
    path = tmp_path / "microbench.json"
    data = _evidence(path)
    data["target_llm_latency_used"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="independent non-LLM"):
        load_runtime_residual_calibration(path)
    data["target_llm_latency_used"] = False
    data["samples"][-1]["node_count"] = 8
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the measured training range"):
        load_runtime_residual_calibration(path)
