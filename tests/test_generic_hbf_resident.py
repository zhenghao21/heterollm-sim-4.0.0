"""Warm weights on analytical NAND are pre-existing data, not timed writes."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hbf_hbm_serving_study import build_study_scenario
from heterollm_sim.kernel_model import KernelCapability, KernelModelProfile
from heterollm_sim.physical_contract import reconfigure_physical_config
from heterollm_sim.planner import ScenarioValidationError
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.serving import RequestStatus


def generic_scenario(*, native=False):
    scenario = build_study_scenario("D_hbf_weights", prompt=2, output=2, batch=1)
    assert scenario.weights_resident
    if not native:
        return scenario
    gpu = scenario.hardware.get_component("gpu0")
    profile = scenario.resolve_component_profile(gpu)
    kernels = tuple(KernelCapability(
        f"{phase}-{operator}-{dtype}-{bits}", ("int8", "fp16", "bf16", "fp32"),
        dtype, phase, "tensor", "explicit analytical test declaration",
        operator=operator, internal_dtype=dtype, output_bits=bits,
    ) for phase in ("prefill", "decode") for operator in ("gemm", "attention")
      for dtype in ("int8", "fp16", "bf16", "fp32") for bits in (8, 16, 32))
    profile = replace(profile, kernel_model=KernelModelProfile(
        "test-device", "test-runtime", "test-architecture", kernels=kernels))
    return replace(scenario, component_profiles={**scenario.component_profiles,
        "gpu": {**scenario.component_profiles["gpu"], gpu.cost_profile_id: profile}})


@pytest.mark.parametrize("native", [False, True])
def test_actual_generic_dense_and_native_runs_keep_warm_weights_and_real_nand(monkeypatch, native):
    import heterollm_sim.reporting as reporting

    scenario = generic_scenario(native=native)
    bootstrap = reporting.bootstrap_control_plane
    observed = []

    def capture(*args, **kwargs):
        result = bootstrap(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(reporting, "bootstrap_control_plane", capture)
    result = run_scenario(scenario)
    assert all(metric.status is RequestStatus.FINISHED and metric.visible_output_tokens == 2
               for metric in result.serving.request_metrics.values())
    assert result.scenario.weights_resident
    physical = observed[0].kernel.physical_runtime
    core = physical.runtimes["hbf0.memory"].core
    assert core.config.kind.value == "HBF"
    assert core.config.protocol_mode == "generic_nand"
    assert any(allocation.size_bytes > 0
               for allocation in physical.allocators["hbf0.memory"].allocations())
    shards = result.scenario.placement.metadata["control_plane"]["decision"]["rank_weight_shards"]
    placed = [row for rows in shards.values() for row in rows]
    assert placed and all(row["storage_component_id"] == "hbf0" for row in placed)
    assert sum(row["physical_bytes"] for row in placed) > 0
    traffic = report_dict(result, include_visualization=False)["storage_traffic"]
    assert traffic["logical_read_bytes"] > 0
    assert traffic["logical_write_bytes"] == 0  # pre-existing weights are not fictitious writes


def test_generic_warm_weights_still_obey_control_plane_capacity():
    scenario = generic_scenario()
    flash = scenario.hardware.get_component("hbf0")
    config = reconfigure_physical_config(flash.metadata["physical_memory_config"],
                                        overrides={"capacity_bytes": 4096})
    flash = replace(flash, capacity_bytes=4096,
                    metadata={**flash.metadata, "physical_memory_config": config})
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(
        flash if component.component_id == "hbf0" else component
        for component in scenario.hardware.components)))
    with pytest.raises(ScenarioValidationError, match="hbf0"):
        run_scenario(scenario)
