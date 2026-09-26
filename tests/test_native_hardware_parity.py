from copy import deepcopy

import pytest

from tools.native_llama_compare import build_matching_scenario, probe_hardware, _hardware_fingerprint


def _snapshot():
    return {
        "cpu": "AMD Ryzen 9 9950X3D 16-Core Processor",
        "cpu_topology": {"physical_cores": 16, "logical_processors": 32},
        "gpu": {
            "name": "NVIDIA GeForce RTX 5080", "uuid": "GPU-test",
            "memory_mib": 15_872, "driver": "616.64",
            "clocks": {"graphics_mhz": 2400, "memory_mhz": 1500, "sm_mhz": 2400},
            "pcie": {"gen_current": 5, "gen_max": 5, "width_current": 8, "width_max": 16},
        },
        "host_memory": {"total_bytes": 128 * 1024**3, "modules": []},
    }


def _snapshot_with_public_specs():
    """Return a measured snapshot with explicit vendor-published inputs.

    The measured values intentionally remain separate from the public values:
    the snapshot clocks describe this machine at capture time, while the
    public profile supplies the analytical RTX 5080/9950X3D specifications.
    """
    snapshot = _snapshot()
    snapshot["cpu_public_specs"] = {
        "model": "AMD Ryzen 9 9950X3D",
        "physical_cores": 16,
        "logical_processors": 32,
        "base_clock_mhz": 4300,
        "boost_clock_mhz": 5700,
        "memory": {
            "type": "DDR5",
            "channels": 2,
            "data_rate_mt_s": 5600,
            "theoretical_bandwidth_gb_s": 89.6,
        },
    }
    snapshot["gpu"]["public_specs"] = {
        "model": "NVIDIA GeForce RTX 5080",
        "architecture": "Blackwell",
        "sm_count": 84,
        "cuda_cores_per_sm": 128,
        "tensor_cores_per_sm": 4,
        "tensor_core_generation": 5,
        "base_clock_mhz": 2300,
        "boost_clock_mhz": 2617,
        "bf16_dense_tflops": 112.6,
        "memory": {
            "type": "GDDR7",
            "capacity_gib": 16,
            "data_rate_gbps": 30,
            "bus_width_bits": 256,
            "bandwidth_gb_s": 960.0,
        },
        "cache": {
            "l1_shared_kib_per_sm": 128,
            "l2_mib": 64,
        },
    }
    snapshot["host_memory"]["public_specs"] = {
        "type": "DDR5",
        "channels": 2,
        "data_rate_mt_s": 5600,
        "bandwidth_gb_s": 89.6,
    }
    return snapshot


def _matching_scenario(snapshot):
    return build_matching_scenario(
        8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=16,
        gpu_layers=0, hardware_snapshot=snapshot,
    )


def test_matching_scenario_binds_snapshot_physical_facts_and_resolves_auto_threads():
    scenario = build_matching_scenario(
        8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=-1,
        gpu_layers=0, hardware_snapshot=_snapshot(),
    )
    assert scenario.llama_cpp_config.threads == 32
    assert scenario.hardware.get_component("hbm0").capacity_bytes == 15_872 * 1024**2
    assert scenario.hardware.get_component("hostmem0").capacity_bytes == 128 * 1024**3
    pcie = next(link for link in scenario.hardware.links if link.link_id == "cpu-gpu-pcie")
    assert (pcie.lanes, pcie.bandwidth_gbps) == (8, pytest.approx(252.032))
    assert scenario.hardware.metadata["gpu_uuid"] == "GPU-test"
    assert scenario.hardware.metadata["analysis_input_basis"] == "legacy_fallback"
    assert "zero_gpu_weight_layers_op_offload_possible" in scenario.placement.metadata["placement_risks"]


def test_matching_scenario_uses_public_specs_for_analysis_profiles_and_links():
    snapshot = _snapshot_with_public_specs()
    scenario = _matching_scenario(snapshot)

    gpu_component = scenario.hardware.get_component("gpu0")
    assert gpu_component.peak_ops_per_s == pytest.approx(112.6e12)
    assert gpu_component.capacity_bytes == 64 * 1024**2
    gpu_profile = scenario.component_profiles["gpu"]["legacy-gpu"]
    assert gpu_profile.tensor_core.sm_count == 84
    assert gpu_profile.tensor_core.tensor_cores_per_sm == 4
    assert gpu_profile.tensor_core.frequency_ghz == pytest.approx(2.617)

    hbm_port = next(port for port in gpu_component.ports if port.port_id == "hbm0")
    assert hbm_port.bandwidth_gbps == pytest.approx(960.0 * 8.0)
    hbm = scenario.hardware.get_component("hbm0")
    assert next(port for port in hbm.ports if port.port_id == "host").bandwidth_gbps == pytest.approx(960.0 * 8.0)
    assert scenario.component_profiles["hbm"]["legacy-hbm"].bandwidth_gb_s == pytest.approx(960.0)

    hostmem = scenario.hardware.get_component("hostmem0")
    ddr_port = next(port for port in hostmem.ports if port.port_id == "ddr0")
    assert ddr_port.bandwidth_gbps == pytest.approx(89.6 * 8.0)
    assert scenario.component_profiles["host_memory"]["legacy-host-memory"].bandwidth_gb_s == pytest.approx(89.6)
    assert scenario.hardware.metadata["analysis_input_basis"] == "public_spec"


def test_public_specs_changes_are_reflected_without_fitting_native_latency():
    baseline = _snapshot_with_public_specs()
    changed = deepcopy(baseline)
    changed["gpu"]["public_specs"].update({
        "sm_count": 42,
        "tensor_cores_per_sm": 2,
        "boost_clock_mhz": 2000,
        "bf16_dense_tflops": 50.0,
    })
    changed["gpu"]["public_specs"]["memory"]["bandwidth_gb_s"] = 500.0
    changed["gpu"]["public_specs"]["cache"]["l2_mib"] = 32
    changed["host_memory"]["public_specs"]["bandwidth_gb_s"] = 64.0

    reference = _matching_scenario(baseline)
    variant = _matching_scenario(changed)
    reference_gpu = reference.hardware.get_component("gpu0")
    variant_gpu = variant.hardware.get_component("gpu0")
    assert variant_gpu.peak_ops_per_s == pytest.approx(50.0e12)
    assert variant_gpu.capacity_bytes == 32 * 1024**2
    assert variant_gpu.peak_ops_per_s != reference_gpu.peak_ops_per_s
    variant_tensor = variant.component_profiles["gpu"]["legacy-gpu"].tensor_core
    assert (variant_tensor.sm_count, variant_tensor.tensor_cores_per_sm) == (42, 2)
    assert variant_tensor.frequency_ghz == pytest.approx(2.0)
    assert variant_tensor.cycles_per_mma != reference.component_profiles["gpu"]["legacy-gpu"].tensor_core.cycles_per_mma

    variant_hbm_port = next(port for port in variant_gpu.ports if port.port_id == "hbm0")
    assert variant_hbm_port.bandwidth_gbps == pytest.approx(500.0 * 8.0)
    variant_hbm_link = next(link for link in variant.hardware.links if link.link_id == "gpu-hbm0")
    assert variant_hbm_link.bandwidth_gbps == pytest.approx(500.0 * 8.0)
    variant_ddr_link = next(link for link in variant.hardware.links if link.link_id == "cpu-hostmem-ddr")
    assert variant_ddr_link.bandwidth_gbps == pytest.approx(64.0 * 8.0)


def test_pcie_link_records_gbps_and_gb_per_second_units():
    scenario = _matching_scenario(_snapshot_with_public_specs())
    pcie = next(link for link in scenario.hardware.links if link.link_id == "cpu-gpu-pcie")
    assert pcie.bandwidth_gbps == pytest.approx(252.032)
    assert pcie.metadata["bandwidth_gbps"] == pytest.approx(252.032)
    assert pcie.metadata["bandwidth_gb_s"] == pytest.approx(31.504)


def test_matching_scenario_keeps_legacy_call_and_rejects_unknown_identity():
    legacy = build_matching_scenario(8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=16, gpu_layers=0)
    assert legacy.hardware.metadata["hardware_snapshot_source"] == "legacy-reference-default"
    bad = _snapshot()
    bad["gpu"] = {**bad["gpu"], "name": "NVIDIA GeForce RTX 4090"}
    with pytest.raises(ValueError, match="unsupported GPU"):
        build_matching_scenario(8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=16, gpu_layers=0, hardware_snapshot=bad)


def test_threads_auto_requires_measured_cpu_topology():
    snapshot = _snapshot()
    snapshot.pop("cpu_topology")
    with pytest.raises(ValueError, match="threads=-1"):
        build_matching_scenario(8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=-1, gpu_layers=0, hardware_snapshot=snapshot)


def test_legacy_auto_threads_are_rejected_without_a_snapshot():
    with pytest.raises(ValueError, match="threads=-1"):
        build_matching_scenario(8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=-1, gpu_layers=0)


def test_probe_hardware_collects_identity_memory_clocks_pcie_and_cpu_topology(monkeypatch):
    def fake_check_output(command, **kwargs):
        if command[0] == "nvidia-smi" and "pcie.link.gen.current" in command[1]:
            return "5, 5, 8, 16\n"
        if command[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 5080, GPU-test, 15872, 15000, 872, 616.64, 12.0, 2400, 1500, 2400\n"
        return '{"processors":{"Name":"AMD Ryzen 9 9950X3D 16-Core Processor","NumberOfCores":16,"NumberOfLogicalProcessors":32,"MaxClockSpeed":4500,"CurrentClockSpeed":4300},"total_memory":137438953472,"modules":[]}'

    monkeypatch.setattr("tools.native_llama_compare.subprocess.check_output", fake_check_output)
    result = probe_hardware()
    assert result["gpu"]["uuid"] == "GPU-test"
    assert result["gpu"]["clocks"]["graphics_mhz"] == 2400
    assert result["gpu"]["pcie"]["width_current"] == 8
    assert result["cpu_topology"]["logical_processors"] == 32
    assert result["host_memory"]["total_bytes"] == 137438953472


def test_hardware_fingerprint_ignores_volatile_clock_and_memory_fields():
    base = _snapshot()
    changed = {
        **base,
        "cpu_topology": {**base["cpu_topology"], "current_clock_mhz": 4300},
        "gpu": {
            **base["gpu"],
            "memory_free_mib": 100,
            "memory_used_mib": 15000,
            "clocks": {"graphics_mhz": 2760, "memory_mhz": 15000, "sm_mhz": 2760},
        },
    }
    assert _hardware_fingerprint(base) == _hardware_fingerprint(changed)


@pytest.mark.parametrize("gpu_layers", [-1, 0, 1, 12])
def test_output_layer_offload_matches_llama_ngl_semantics(gpu_layers):
    """The ``-ngl`` count includes output and selects a tail of repeating blocks.

    llama.cpp reports ``offloaded N/25 layers`` for this 24-block Qwen model:
    ``-ngl 0`` keeps output and all blocks on CPU, ``-ngl 1`` moves only the
    output layer, ``-ngl 12`` moves output plus blocks 13..23, and ``-ngl -1``
    moves output plus every repeating block.  Tied output weights are a logical
    alias of ``embedding_weights`` and therefore must not be materialised as a
    second ``lm_head_weights`` tensor in the placement contract.
    """
    scenario = build_matching_scenario(
        8,
        2,
        ctx=64,
        parallel=1,
        batch=8,
        ubatch=8,
        threads=16,
        gpu_layers=gpu_layers,
        hardware_snapshot=_snapshot(),
    )
    placement = scenario.placement
    decision = placement.metadata["control_plane"]["decision"]

    # The runtime policy preserves the llama.cpp count (with -1 expanded to
    # model blocks + output layer) and the alias remains explicit metadata.
    expected_loadable = scenario.model.num_layers + 1 if gpu_layers < 0 else gpu_layers
    assert decision["logical_weight_aliases"]["lm_head_weights"] == "embedding_weights"
    assert (
        placement.metadata["control_plane"]["policy"]["options"]["gpu_loadable_layers"]
        == expected_loadable
    )
    assert (
        placement.metadata["control_plane"]["policy"]["options"]["gpu_loadable_order"]
        == "tail"
    )
    head_component = "gpu0" if gpu_layers != 0 else "cpu0"
    assert placement.op_to_component["lm_head"] == head_component
    if gpu_layers != 0:
        # For a logically tied output, llama.cpp keeps the embedding backing
        # on host and materializes one GPU runtime copy for lm_head.
        assert placement.tensor_to_component["embedding_weights"] == "hostmem0"
        assert placement.tensor_to_component["lm_head_weights"] == "hbm0"
        assert placement.tensor_bytes["lm_head_weights"] > 0
    else:
        assert "lm_head_weights" not in placement.tensor_to_component
        assert "lm_head_weights" not in placement.tensor_bytes
        assert placement.tensor_to_component["embedding_weights"] == "hostmem0"
    if gpu_layers == 0:
        assert placement.tensor_to_component["embedding_weights"] == "hostmem0"

    repeating_gpu_count = (
        scenario.model.num_layers
        if gpu_layers < 0
        else max(0, min(gpu_layers - 1, scenario.model.num_layers))
    )
    first_gpu_block = scenario.model.num_layers - repeating_gpu_count
    for layer_id in range(scenario.model.num_layers):
        expected_op = (
            "gpu0"
            if repeating_gpu_count and layer_id >= first_gpu_block
            else "cpu0"
        )
        expected_weight = "hbm0" if expected_op == "gpu0" else "hostmem0"
        assert placement.op_to_component[f"layer-{layer_id:03d}.attention"] == expected_op
        assert placement.tensor_to_component[f"layer-{layer_id:03d}.attention_weights"] == expected_weight
        assert placement.tensor_to_component[f"layer-{layer_id:03d}.mlp_weights"] == expected_weight
