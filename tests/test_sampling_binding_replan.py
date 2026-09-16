"""Regression coverage for the R18 typed sampling control-plane binding."""

from heterollm_sim.config import SamplingPolicy
from tools.native_llama_compare import build_matching_scenario
from tools.predict_stable_native_dataset import replan_final_static_scenario


def _fixed_hardware_snapshot():
    """Small, deterministic version of the measured RTX 5080 host fixture."""
    return {
        "cpu": "AMD Ryzen 9 9950X3D 16-Core Processor",
        "cpu_topology": {"physical_cores": 16, "logical_processors": 32},
        "gpu": {
            "name": "NVIDIA GeForce RTX 5080",
            "uuid": "GPU-test",
            "memory_mib": 15_872,
            "driver": "616.64",
            "clocks": {"graphics_mhz": 2400, "memory_mhz": 1500, "sm_mhz": 2400},
            "pcie": {"gen_current": 5, "gen_max": 5, "width_current": 8, "width_max": 16},
        },
        "host_memory": {"total_bytes": 128 * 1024**3, "modules": []},
    }


def test_real_static_replan_preserves_typed_top_k_one_sampling_policy(monkeypatch):
    def forbid_simulation(*args, **kwargs):
        raise AssertionError("static control-plane replan must not run the simulator")

    monkeypatch.setattr(
        "tools.predict_stable_native_dataset.grid.reporting.run_scenario",
        forbid_simulation,
    )
    policy = SamplingPolicy(
        mode="greedy",
        temperature=0.0,
        implementation="llama_cpp_cpu_chain",
        top_k=1,
        top_p=0.95,
        min_p=0.05,
        min_keep=0,
    )
    scenario = build_matching_scenario(
        8,
        2,
        ctx=64,
        parallel=1,
        batch=8,
        ubatch=8,
        threads=16,
        gpu_layers=0,
        hardware_snapshot=_fixed_hardware_snapshot(),
        sampling_policy=policy,
    )

    replanned, evidence = replan_final_static_scenario(scenario)

    assert scenario.sampling_policy == policy
    assert replanned.sampling_policy == policy
    assert replanned.sampling_policy.implementation == "llama_cpp_cpu_chain"
    assert replanned.sampling_policy.top_k == 1
    assert replanned.sampling_policy.min_keep == 0
    assert evidence["method"] == "typed_runtime_placement_replanned_after_all_static_bindings"
    assert evidence["normal_validation_passed"] is True
