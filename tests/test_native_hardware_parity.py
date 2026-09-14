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
    assert "gpu_offload_disabled_cpu_only" in scenario.placement.metadata["placement_risks"]


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
