"""Research inputs resolve through the same explicit physical catalog."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import calibrated_hbf_profiles as profiles
import hbf_hbm_experiments as experiments
import hbf_hbm_serving_study as study
import hbf_hbm_provisioning as provisioning
from hbf_hbm_core_audit import CoreExecutionAudit
from heterollm_sim.memory_types import AccessRequest, DRAM_BANK_GROUP_TIMING_FIELDS
from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.physical_presets import resolve_physical_config


def physical(scenario, owner="hbm0"):
    return parse_execution_physical_config(scenario.hardware.get_component(owner).metadata["physical_memory_config"])


def test_serving_uses_hbm3e_reference_geometry_and_explicit_timing_assumptions():
    scenario = study.build_study_scenario("A_hbm_only", prompt=4, output=2)
    config = physical(scenario)
    assert scenario.resolve_component_profile("hbm0").parallel_lanes == config.lane_count
    assert scenario.resolve_component_profile("hbm0").transaction_bytes == config.burst_bytes
    assert config.metadata["physical_config_profile_id"] == "hbm3e_reference"
    assert (config.channels, config.pseudo_channels_per_channel, config.stacks,
            config.bank_groups_per_rank, config.banks_per_group, config.rows_per_bank) == (16, 2, 8, 16, 4, 32768)
    assert (config.data_width_bits, config.row_bytes, config.burst_bytes) == (32, 1024, 32)
    assert config.bank_geometry_scope == "per_stack"
    assert config.stack_interface_mode == "independent"
    assert config.metadata["geometry_status"] == "standard_organization"
    assert config.metadata["unknown_parameters"]
    assert config.metadata["source_device_geometry_known"] is False
    assert config.control_mode == "jedec_hbm"
    assert config.read_latency_ns == 32 * config.command_clock_ns
    assert config.write_latency_ns == 8 * config.command_clock_ns
    assert config.column_same_group_ns >= 4 * config.command_clock_ns
    assert config.column_different_group_ns == 2 * config.command_clock_ns
    assert config.close_ns == config.precharge_to_activate_ns >= 15.
    summary = profiles.hbm_protocol_summary(config)
    assert summary["control_status"] == "jedec_normal_operation_subset"
    assert summary["timing_validation"] == "not_a_vendor_speed_bin"
    assert summary["unknown_parameters"] == config.metadata["unknown_parameters"]
    assert config.capacity_bytes == 512 * 1024 ** 3
    assert config.computed_capacity_bytes == config.capacity_bytes
    assert config.read_bandwidth_gb_s == config.write_bandwidth_gb_s == 1600


def test_three_timing_profiles_keep_non_group_physical_variables_identical():
    ids = ("hbm3e_reference_flat", "hbm3e_reference", "hbm3e_reference_synthetic_timing")
    configs = [physical(study.build_study_scenario("A_hbm_only", prompt=4, output=2,
                       hbm_timing_profile=profile, hbm_parameter_profile="none")) for profile in ids]
    assert [(c.bank_groups_per_rank, c.banks_per_group) for c in configs] == [(1, 64), (16, 4), (16, 4)]
    assert all(c.bank_group_timing_enabled for c in configs)
    assert [getattr(configs[-1], key) for key in DRAM_BANK_GROUP_TIMING_FIELDS] == [
        max(getattr(configs[1], key), sensitivity) for key, sensitivity in
        zip(DRAM_BANK_GROUP_TIMING_FIELDS, (4, 1, 8, 2, 16, 4, 2))]
    excluded = {"metadata", "bank_groups_per_rank", "banks_per_group", *DRAM_BANK_GROUP_TIMING_FIELDS}
    signatures = [{k: v for k, v in asdict(c).items() if k not in excluded} for c in configs]
    assert signatures[0] == signatures[1] == signatures[2]


def test_micro_and_full_service_use_catalog_for_the_same_hbf_inputs():
    for profile in profiles._profile_specs():
        hbm, hbf, _ = profiles._profile_hardware(profile)
        if hbm is not None:
            assert hbm.config.metadata["physical_config_profile_id"] == "hbm3e_reference"
            assert hbm.config.control_mode == "jedec_hbm"
            assert hbm.config.bank_group_timing_enabled
            assert hbm.config == physical(profiles._full_backend_scenario(profile))
            if profile.hardware.get("hbm_stacks"):
                assert hbm.config.stacks == profile.hardware["hbm_stacks"]
        if hbf is not None:
            actual = physical(profiles._full_backend_scenario(profile), "hbf0")
            assert actual == hbf.config
            assert actual.metadata["physical_config_profile_id"] == "published_hbf"


def test_undisclosed_hbf_geometry_is_resolved_by_catalog():
    profile = experiments.synthetic_profile()
    profile = replace(profile, hardware={**profile.hardware, "effective_parallel_units": None},
                      granularity={}, latency={})
    _, hbf, _ = profiles._profile_hardware(profile)
    assert hbf.config.parallel_units == hbf.config.channels == 128
    assert (hbf.config.page_bytes, hbf.config.host_granularity_bytes) == (4096, 64)
    assert (hbf.config.host_channels, hbf.config.host_full_duplex,
            hbf.config.read_cache_pages_per_bank) == (16, True, 2)
    assert "never inferred" in hbf.config.metadata["parallelism_basis"]


def test_hbf_plus_comparator_uses_jedec_hbm4_reference_not_hbm3e_geometry():
    profile = profiles._profile_by_id("son_hbf_plus")
    scenario = profiles._full_backend_scenario(profile)
    config = physical(scenario)
    assert config.metadata["physical_config_profile_id"] == "hbm4_reference"
    assert config.generation == "HBM4"
    assert (config.stacks, config.channels, config.pseudo_channels_per_channel,
            config.data_width_bits, config.row_bytes, config.burst_bytes) == (8, 32, 2, 32, 1024, 32)
    assert (config.bank_groups_per_rank, config.banks_per_group, config.rows_per_bank) == (8, 8, 16384)
    assert config.capacity_bytes == 288 * profiles.GB
    assert config.computed_capacity_bytes == 512 * profiles.GIB
    assert config.metadata["geometry_status"] == "standard_organization"
    assert config.metadata["unknown_parameters"]
    assert config.metadata["source_device_geometry_known"] is False
    assert all(port.version == "4.0" for port in scenario.hardware.get_component("hbm0").ports)
    assert next(link for link in scenario.hardware.links if link.link_id == "gpu-hbm0").version == "4.0"
    row = profiles.run_profile(profile, kv_pages=2, steps=1)
    assert row["hbm_only"]["baseline_label"] == "son_hbm4_reference_not_same_profile"
    assert row["hbm_only"]["physical_config"]["generation"] == "HBM4"
    assert row["hbm_only"]["hbm_protocol"]["control_status"] == "jedec_normal_operation_subset"
    assert row["hbm_only"]["hbm_protocol"]["unknown_parameters"]


def test_study_capacity_and_nvme_use_unified_reconfiguration():
    size = 2 * 1024 ** 4
    scenario = study.build_study_scenario("F_nvme", prompt=4, output=2,
        hbm_capacity_bytes=size, hbf_capacity_bytes=size)
    hbm, nvme = physical(scenario), physical(scenario, "hbf0")
    assert hbm.capacity_bytes == nvme.capacity_bytes == size
    assert hbm.computed_capacity_bytes == size
    assert (hbm.stacks, hbm.rows_per_bank, hbm.row_bytes) == (32, 32768, 1024)
    assert nvme.computed_capacity_bytes >= size
    assert nvme.metadata["physical_config_profile_id"] == "study_nvme_baseline"
    assert (nvme.host_read_bandwidth_gb_s, nvme.host_write_bandwidth_gb_s,
            nvme.page_read_ns, nvme.page_program_ns) == (7, 4, 50000, 800000)
    dedicated = physical(experiments.scenario(kv_budget=16384), "hbm_kv")
    assert dedicated.capacity_bytes == 16384
    assert dedicated.metadata["physical_config_profile_id"] == "hbm3e_reference"
    assert dedicated.computed_capacity_bytes > dedicated.capacity_bytes


def test_package_sweep_scales_stacks_without_multiplying_channels_or_capacity_twice():
    for count in (1, 2, 4):
        scenario = provisioning.build_package_scenario(count, 4-count, "A_hbm_only",
            model_id="small", output=2)
        config = physical(scenario)
        assert (config.channels, config.pseudo_channels_per_channel, config.stacks) == (16, 2, count)
        assert config.capacity_bytes == provisioning.HBM_CAPACITY * count
        assert config.computed_capacity_bytes >= config.capacity_bytes
        assert config.read_bandwidth_gb_s == config.write_bandwidth_gb_s == 1600 * count
        assert config.metadata["geometry_status"] == "standard_organization"
        assert scenario.resolve_component_profile("hbm0").parallel_lanes == config.lane_count
        assert scenario.resolve_component_profile("hbm0").transaction_bytes == config.burst_bytes


def test_completion_matrix_requires_executed_group_twenty_rows():
    row = next(r for r in experiments.completion_matrix([]) if r["groups"] == [20])
    assert row["status"] == "本次未完成"
    row = next(r for r in experiments.completion_matrix([
        {"id": "20_bank_group_decode", "status": "passed"}]) if r["groups"] == [20])
    assert row["status"] == "本次未完成"
    assert study.POINT_RESULT_VERSION == 10


def test_script_directional_budgets_do_not_change_an_explicit_physical_interface():
    configs = [profiles._hbm_physical_config(
        capacity_bytes=1024**2, stacks=1, interface_gb_s=512.,
        read_gb_s=read, write_gb_s=write)
        for read, write in ((128., 64.), (64., 32.))]
    for config in configs:
        assert config.physical_interface_bandwidth_gb_s == 512.
        assert config.command_clock_ns == 1.
        assert config.column_different_group_ns == 2.
        assert config.read_latency_ns == 32.
        assert config.write_latency_ns == 8.
        assert config.metadata["interface_input_scope"] == "declared_physical_interface_budget"
    assert [(c.read_bandwidth_gb_s, c.write_bandwidth_gb_s) for c in configs] == [
        (128., 64.), (64., 32.)]
    inferred = profiles._hbm_physical_config(
        capacity_bytes=1024**2, stacks=1, read_gb_s=128., write_gb_s=64.)
    assert inferred.physical_interface_bandwidth_gb_s == 819.2
    assert inferred.command_clock_ns == 0.625
    assert inferred.metadata["interface_input_scope"] == "independent_reference_command_clock"


@pytest.mark.parametrize("version", [None, 7, 8])
def test_group_reuse_rejects_old_or_unknown_execution_versions_before_running(tmp_path, monkeypatch, capsys, version):
    cached = {"id": "01_ideal", "status": "passed", "rows": []}
    if version is not None:
        cached["point_result_version"] = version
    (tmp_path / "01_ideal.json").write_text(json.dumps(cached), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hbf_hbm_experiments", "--output-dir", str(tmp_path),
                                     "--groups", "1", "--reuse-groups", "1"])
    monkeypatch.setattr(experiments, "ideal_group", lambda: pytest.fail("stale reuse must be rejected before simulation"))
    with pytest.raises(SystemExit) as error:
        experiments.main()
    assert error.value.code == 2
    assert "older or unknown physical-execution version" in capsys.readouterr().err
    assert not (tmp_path / "results.json").exists()


@pytest.mark.parametrize("version", [None, 7, 8])
def test_sustained_reuse_rejects_old_or_unknown_execution_versions_before_running(monkeypatch, version):
    import hbf_hbm_sustained_study as sustained
    cached = {"id": "old-observation"}
    if version is not None:
        cached["point_result_version"] = version
    monkeypatch.setattr(sustained, "run_arrival_case", lambda *args, **kwargs:
                        pytest.fail("stale reuse must be rejected before simulation"))
    with pytest.raises(ValueError, match="older or unknown physical-execution version"):
        sustained.run_sustained_study(progress=False, reuse_rows=[cached])


def test_core_audit_retains_hbm_command_observations_separately_from_unknown_dram():
    core = profiles._hbm_core(capacity_bytes=1024**2, read_gb_s=512., write_gb_s=512., stacks=1)
    requests = tuple(AccessRequest(str(i), "read", 0, 1) for i in range(2))
    results = tuple(core.submit(request) for request in requests)
    audit = CoreExecutionAudit()
    audit._cores[id(core)] = (core, "test-hbm")
    audit._record(core, "submit_many", requests, results)
    row = audit.summary()["owners"][0]
    assert row["jedec_control_transactions"] == row["read_commands"] == 2
    assert row["activate_commands"] == 1
    assert row["row_command_bus_busy_ns"] == 1.5 * core.config.command_clock_ns
    assert row["control_metric_recorded_transactions"]["read_commands"] == row["transactions"] == 2
