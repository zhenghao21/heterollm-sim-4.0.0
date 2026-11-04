"""End-to-end conservation checks for the media experiment output."""
import importlib.util
from pathlib import Path
from unittest.mock import patch


def test_media_experiments_conserve_payload_and_expose_real_dual_path():
    path = Path(__file__).resolve().parents[1] / "scripts" / "hbf_hbm_media_experiments.py"
    spec = importlib.util.spec_from_file_location("media_experiments_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_media_experiments()
    assert report["status"] == "passed"
    write_rows = report["groups"][0]["rows"]
    assert len(write_rows) == 720
    for row in write_rows:
        assert row["base_media_write_bytes"] == row["physical_programs"] * row["page_bytes"]
        assert row["media_write_bytes"] == row["base_media_write_bytes"] * row["gc_write_amplification"]
        assert (row["link_write_bytes"] == row["logical_write_bytes"] if row["write_buffer_capacity_bytes"]
                else row["link_write_bytes"] >= row["logical_write_bytes"])
        assert row["applied_physical_page_bytes"] == row["page_bytes"]
        assert row["write_buffer_remaining_bytes_after_drain"] == 0
        assert row["buffered_pages_remaining_after_drain"] == 0
        assert row["node_tps"] is None
        observed = row["observed_metrics"]
        assert observed["write_amplification"] == row["base_media_write_bytes"] / row["logical_write_bytes"]
        assert observed["effective_logical_write_bandwidth_gb_s"] == row["logical_write_bytes"] / row["media_drain_completion_ns"]
        assert 0 < observed["page_write_payload_utilization"] <= 1
        assert all(0 <= value <= 1 for value in observed["channel_utilization"].values())
    assert {row["page_bytes"] for row in write_rows} == {4096, 16384, 65536, 262144, 1572864}
    large_pages = [row for row in write_rows if row["page_bytes"] == 1572864]
    assert len(large_pages) == 144
    for row in large_pages:
        assert "not a vendor NAND page specification" in row["page_size_assumption"]
        assert row["logical_write_bytes"] == 128*1024
        assert row["base_media_read_bytes"] == row["base_media_write_bytes"]
        assert row["write_buffer_peak_bytes"] <= row["write_buffer_capacity_bytes"]
        effective_flush_bytes = row["write_buffer_capacity_bytes"] or row["append_bytes"]
        assert row["physical_programs"] == row["logical_write_bytes"] // effective_flush_bytes
        assert row["base_media_write_bytes"] == row["physical_programs"]*1572864
    paths = {row["path"]: row for row in report["groups"][1]["rows"]}
    assert paths["dual"]["completion_ns"] < paths["direct"]["completion_ns"]
    assert [flow["route"] for flow in paths["dual"]["flows"]] == ["relay", "direct"]
    assert {row["payload_bytes_per_flow"] for row in report["groups"][1]["block_size_scan"]} == {4096, 16384, 65536, 262144, 1572864}
    package = report["groups"][2]
    assert len(package["rows"]) == 5
    assert [row["hbm_slots"] for row in package["rows"]] == [0, 1, 2, 3, 4]
    assert len(package["accounting_checks"]["report"]["rows"]) == 3
    assert package["rows"][-1]["runnable"] is False
    assert package["slot_profiles"]["hbf"]["capacity_pages"] == 8 * package["slot_profiles"]["hbm"]["capacity_pages"]
    with patch.object(module, "_status", return_value="failed"):
        failed = module.run_group_06_path_direct_relay()
    assert failed["status"] == "failed"
    assert failed["capability"]["supported"] == []
