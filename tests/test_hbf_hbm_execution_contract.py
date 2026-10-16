"""The study exposes its assumptions and refuses obsolete result versions."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import calibrated_hbf_profiles as profiles
import hbf_hbm_experiments as experiments
import hbf_hbm_serving_study as study
from hbf_hbm_report import render_report, uses_current_execution
from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.physical_presets import resolve_physical_config


def test_capacity_changes_stacks_and_ports_without_changing_clock_or_direction_budget():
    for gib, stacks in ((256, 4), (512, 8), (1024, 16)):
        scenario = study.build_study_scenario("A_hbm_only", prompt=4, output=2,
                                              hbm_capacity_bytes=gib * 1024**3)
        component = scenario.hardware.get_component("hbm0")
        config = parse_execution_physical_config(component.metadata["physical_memory_config"])
        profile = scenario.resolve_component_profile("hbm0")
        assert (config.stacks, config.dies_per_stack, config.metadata["die_density_gbits"]) == (stacks, 16, 32)
        assert config.command_clock_ns == 0.625
        assert config.read_bandwidth_gb_s == config.write_bandwidth_gb_s == 1600.
        physical_rate = stacks * 819.2
        assert config.physical_interface_bandwidth_gb_s == physical_rate
        assert component.bandwidth_gbps == physical_rate * 8
        assert all(port.bandwidth_gbps == physical_rate * 8 for port in component.ports)
        assert profile.bandwidth_gb_s == physical_rate
        assert profile.read_bandwidth_gb_s == profile.write_bandwidth_gb_s == 1600.
        assert config.refresh_interval_ns == 3900 and config.refresh_cycle_ns == 500
        assert config.metadata["hbm_parameter_profile"]["vendor_calibrated"] is False
        assert config.metadata["target_hardware_calibrated"] is False


def test_main_hbf_interface_and_cold_array_ceilings_are_distinct():
    _, core, _ = profiles._profile_hardware(experiments.synthetic_profile())
    summary = profiles.hbf_service_summary(core.config)
    assert summary["protocol_mode"] == "generic_nand"
    assert summary["model_abstraction"] == "analytical_page_service"
    assert (summary["page_bytes"], summary["host_granularity_bytes"]) == (4096, 64)
    assert (summary["host_channels"], summary["host_full_duplex"],
            summary["read_cache_pages_per_bank"]) == (16, True, 2)
    assert (summary["interface_read_cap_gb_s"], summary["interface_write_cap_gb_s"]) == (1600., 16.)
    assert summary["ideal_cold_array_read_ceiling_gb_s"] == 65.536
    assert summary["ideal_array_program_ceiling_gb_s"] == pytest.approx(3.4952533333333333)
    assert summary["effective_parallel_units"] == 64
    assert summary["calibrated_to_silicon"] is False


def test_hbf_summary_accepts_geometry_based_parallelism():
    config = resolve_physical_config("hbf_synthetic", overrides={"parallel_units": None})
    summary = profiles.hbf_service_summary(config)
    assert summary["effective_parallel_units"] == config.array_units
    assert summary["ideal_cold_array_read_ceiling_gb_s"] == config.array_units * config.page_bytes / config.page_read_ns


@pytest.mark.parametrize("version", [None, 3, 8])
def test_historical_report_is_labelled_and_not_reinterpreted(version):
    report = {"point_result_version": version, "groups": [{"old_schema": True}]}
    rendered = render_report(report)
    assert "历史结果：未按当前模型重跑" in rendered
    assert "当前完整服务结果" not in rendered
    assert "results.json" in rendered


def test_current_report_does_not_embed_old_figures_or_mixed_points():
    report = {"point_result_version": study.POINT_RESULT_VERSION, "groups": [],
              "figure_manifest": {"status": "complete", "point_result_version": 8,
                                  "figures": [{"old_schema": True}]}}
    assert "已忽略历史或版本未记录的图表清单" in render_report(report)
    report["groups"] = [{"point_result_version": 9,
                         "rows": [{"mode": "full_operator_serving", "point_result_version": 8}]}]
    assert not uses_current_execution(report)


def test_plot_entrypoint_rejects_historical_data_before_writing(tmp_path, monkeypatch, capsys):
    import plot_hbf_hbm_results as plots
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"point_result_version": 8, "groups": []}), encoding="utf-8")
    output = tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["plot", "--input", str(path), "--output", str(output), "--allow-partial"])
    with pytest.raises(SystemExit) as error:
        plots.main()
    assert error.value.code == 2
    assert "historical or mixed execution versions" in capsys.readouterr().err
    assert not output.exists()
