"""Published inputs stay distinct from explicit local media assumptions."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import exploring_hbf_paper_config as paper
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore
from heterollm_sim.physical_contract import parse_execution_physical_config


@pytest.mark.parametrize("name,stacks,capacity,read,write", [
    ("HBM4", (8, 0), 288, (12800, 0), (12800, 0)),
    ("CONV", (1, 7), 3620, (1600, 2450), (1600, 73)),
    ("CONV+", (0, 8), 4096, (0, 2800), (0, 84)),
    ("HBF", (1, 7), 3620, (1600, 11200), (1600, 112)),
    ("HBF+", (0, 8), 4096, (0, 12800), (0, 128)),
])
def test_table_i_exact_decimal_values(name, stacks, capacity, read, write):
    row = paper.hardware_by_id(name)
    assert (row.hbm_stacks, row.hbf_stacks) == stacks
    assert row.total_capacity_gb == capacity
    assert row.hbm_capacity_bytes + row.hbf_capacity_bytes == capacity * 1_000_000_000
    assert (row.hbm_read_gb_s, row.hbf_read_gb_s) == read
    assert (row.hbm_write_gb_s, row.hbf_write_gb_s) == write


def test_catalog_export_provenance_and_workload_rounding_are_explicit(tmp_path):
    data = paper.catalog_as_dict()
    assert [(row["input_tokens"], row["output_tokens"]) for row in data["workloads"]] == [
        (1660, 373), (5900, 499), (103500, 1100)]
    assert "decimal 1000" in data["workloads"][1]["length_basis"]
    assert data["units"]["GB_bytes"] == 1_000_000_000
    assert data["units"]["MB_bytes"] == 1_000_000
    assert data["units"]["read_page_bytes"] == 4096
    assert all(not row["paper_workload_reproduced"] for row in data["hardware"])
    output = tmp_path / "catalog.json"
    paper.export_catalog(output)
    assert json.loads(output.read_text(encoding="utf-8")) == data


@pytest.mark.parametrize("name,planes,read_ns", [("HBF", 25, 1000), ("HBF+", 25, 1000),
                                                    ("CONV", 16, 3000), ("CONV+", 16, 3000)])
def test_nand_builder_uses_actual_die_plane_counts_not_effective_units(name, planes, read_ns):
    hardware = paper.hardware_by_id(name)
    assumptions = paper.default_research_assumptions(name)
    config = paper.build_paper_nand_config(name, assumptions=assumptions)
    assert config.protocol_mode == "generic_nand"
    assert config.parallel_units is None
    assert (config.channels, config.targets_per_channel, config.dies_per_target,
            config.luns_per_die, config.planes_per_lun) == (hardware.hbf_stacks * 16, 1, 1, 1, planes)
    assert config.array_units == hardware.hbf_stacks * 16 * planes
    assert (config.page_bytes, config.page_read_ns, config.page_program_ns) == (4096, read_ns, 100000)
    assert config.host_read_bandwidth_gb_s == hardware.hbf_read_gb_s
    assert config.host_write_bandwidth_gb_s == hardware.hbf_write_gb_s
    assert config.capacity_bytes == hardware.hbf_capacity_bytes
    assert config.computed_capacity_bytes >= config.capacity_bytes
    assert config.computed_capacity_bytes - config.capacity_bytes < (
        config.channels * config.planes_per_lun * config.block_bytes)
    assert config.queue_scope == "device"
    assert config.metadata["physical_config_profile_id"] == "hbf_synthetic"
    assert "local choices, not parameters sourced" in config.metadata["research_assumptions"]["basis"]
    assert config.metadata["protocol_basis"].startswith("generic_nand")
    assert config.read_cache_pages_per_bank == 2
    assert config.metadata["execution_backend"] == "native_nand_core"
    assert config.metadata["native_memory_execution_contract"] == "native_nand_plane_double_buffer_v1"
    assert "native per-bank" in config.metadata["paper_read_staging_execution"]
    assert "autonomous predictive prefetch" in config.metadata["paper_read_staging_execution"]
    assert config.metadata["native_read_cache_bytes_per_stack"] == 16 * planes * 2 * 4096
    assert config.metadata["native_read_cache_bytes"] == config.array_units * 2 * 4096
    assert "3.125 MiB" in config.metadata["native_read_cache_unit_basis"]
    assert "not 3130000 decimal bytes" in config.metadata["native_read_cache_unit_basis"]
    assert "2 MiB" in config.metadata["native_read_cache_unit_basis"]
    assert "native SramCore" in config.metadata["paper_intermediate_sram_execution"]
    assert parse_execution_physical_config(asdict(config)) == config


@pytest.mark.parametrize("name,stacks,capacity,bw", [("HBM4", 8, 288, 12800),
                                                     ("HBF", 1, 36, 1600), ("CONV", 1, 36, 1600)])
def test_hbm_builder_keeps_published_budget_and_reference_geometry_separate(name, stacks, capacity, bw):
    config = paper.build_paper_hbm_config(name)
    assert config.generation == "HBM4"
    assert config.stacks == stacks
    assert config.capacity_bytes == capacity * 1_000_000_000
    assert config.computed_capacity_bytes >= config.capacity_bytes
    assert config.read_bandwidth_gb_s == config.write_bandwidth_gb_s == bw
    assert config.metadata["physical_config_profile_id"] == "hbm4_reference"
    assert config.metadata["source_device_geometry_known"] is False
    assert not config.metadata["target_hardware_calibrated"]
    assert any("undisclosed Rubin" in row for row in config.metadata["research_assumptions"])
    assert parse_execution_physical_config(asdict(config)) == config


def test_plus_keeps_zero_hbm_and_declares_native_sram_service():
    for name in ("HBF+", "CONV+"):
        row = paper.hardware_by_id(name)
        assert paper.build_paper_hbm_config(name) is None
        assert row.intermediate_sram_bytes_per_stack == 40_000_000
        assert row.as_dict()["intermediate_sram_bytes"] == 320_000_000
        assert row.as_dict()["read_staging_bytes"] == 25_040_000
        support = paper.execution_support(name)
        assert support["status"] == "available_with_research_assumptions"
        assert support["paper_serving_supported"] is True
        assert support["backend"] == "native_serving_physical"
        assert not support["missing_native_mechanisms"]
        assert any("SramCore" in mechanism for mechanism in support["mechanisms"])
        assert support["unresolved_inputs"]
    assert paper.build_paper_nand_config(
        "HBM4", assumptions=paper.default_research_assumptions("HBF")) is None


@pytest.mark.parametrize("name", ["HBM4", "CONV", "CONV+", "HBF", "HBF+"])
def test_native_support_points_to_native_integrated_execution_and_declares_limits(name):
    support = paper.execution_support(name)
    assert support["backend"] == "native_serving_physical"
    assert support["paper_serving_supported"] is True
    assert support["memory_mechanism_builder_supported"] is True
    assert support["memory_mechanism_builder_status"] == "available_with_research_assumptions"
    assert not support["missing_native_mechanisms"]
    assert any("KV ingress" in item for item in support["mechanisms"])
    assert any("measurement windows" in item for item in support["mechanisms"])
    assert any("SramCore" in item for item in support["mechanisms"]) == name.endswith("+")
    assert any("native continuous batching" in item for item in support["mechanisms"])
    assert support["model_semantics"]["status"] == "native_semantics_implemented"
    assert "fixed-chunk" in support["model_semantics"]["llama4_maverick"]
    assert "historical bypass" in support["scope"]


def test_new_catalog_scope_does_not_relabel_historical_bypass_results():
    catalog = paper.catalog_as_dict()
    assert all(row["backend"] == "native_serving_physical" for row in catalog["execution_support"])
    assert all(row["paper_serving_supported"] is True for row in catalog["execution_support"])
    assert "not yet a completed native paper experiment" in catalog["comparison_scope"]
    assert "historical bypass results retain their original identity" in catalog["comparison_scope"]


def test_nand_core_executes_controlled_pages_and_charges_real_rmw_for_small_kv_writes():
    config = paper.build_paper_nand_config("HBF+", assumptions=paper.default_research_assumptions("HBF+"))
    core = NandCore(config, capture_details=False)
    result = core.submit(AccessRequest("read-page", "read", 0, 4096, 0))
    assert result.completion_ns > config.page_read_ns
    assert result.physical_read_bytes == 4096
    assert config.partial_page_policy == "read_modify_write"
    assert config.host_granularity_bytes == 4096
    partial = NandCore(config, capture_details=False).submit(AccessRequest("small-kv", "write", 0, 256, 0))
    full = NandCore(config, capture_details=False).submit(AccessRequest("full-page", "write", 0, 4096, 0))
    assert partial.physical_read_bytes == partial.physical_write_bytes == 4096
    assert full.physical_read_bytes == 0
    assert full.physical_write_bytes == 4096
    assert partial.completion_ns > full.completion_ns
    assert "pays NandCore" in config.metadata["partial_page_basis"]


@pytest.mark.parametrize("name", ["HBF", "HBF+", "CONV", "CONV+"])
def test_native_paper_two_slot_overwrite_waits_for_the_live_reader(name):
    config = paper.build_paper_nand_config(name, assumptions=paper.default_research_assumptions(name))
    # Keep the authored hardware rates unchanged. Independent host lanes are
    # occupied by earlier traffic so both page fills precede consumption.
    timeline = ResourceTimeline(lane_available={"nand:host:read": [1e6] * 3})
    core = NandCore(config, timeline, capture_details=True)
    stride = config.array_units * config.page_bytes
    results = [core.submit(AccessRequest(f"page-{index}", "read", index * stride,
                                        config.page_bytes, 0.)) for index in range(3)]
    first, second, third = results
    transfer = lambda result: next(stage for stage in result.stages if stage.name == "INTERNAL_TRANSFER")
    assert transfer(second).end_ns < first.completion_ns
    assert transfer(third).start_ns >= first.completion_ns
    assert len(core._read_cache) == 1
    assert len(next(iter(core._read_cache.values()))) == 2
    assert all(result.pages_read == 1 for result in results)
    assert all(result.bandwidth_ceiling_gb_s == config.host_read_bandwidth_gb_s for result in results)
    assert config.host_read_bandwidth_gb_s == paper.hardware_by_id(name).hbf_read_gb_s
    assert config.host_write_bandwidth_gb_s == paper.hardware_by_id(name).hbf_write_gb_s


@pytest.mark.parametrize("name", ["HBF", "HBF+", "CONV", "CONV+"])
def test_native_read_buffers_do_not_retime_the_directional_interface(name):
    config = paper.build_paper_nand_config(name, assumptions=paper.default_research_assumptions(name))
    without_cache = replace(config, read_cache_pages_per_bank=0)
    # Separate cores avoid changing media state between read/write controls.
    for operation, expected in (("read", config.host_read_bandwidth_gb_s),
                                ("write", config.host_write_bandwidth_gb_s)):
        request = AccessRequest(operation, operation, 0, config.page_bytes, 0.)
        buffered = NandCore(config).submit(request)
        plain = NandCore(without_cache).submit(request)
        assert buffered.bandwidth_ceiling_gb_s == plain.bandwidth_ceiling_gb_s == expected
        assert buffered.host_transfer_bytes == plain.host_transfer_bytes
        assert buffered.internal_transfer_bytes == plain.internal_transfer_bytes
        assert buffered.completion_ns == plain.completion_ns


def test_reject_and_host_transfer_granularity_remain_explicit_independent_controls():
    assumptions = replace(paper.default_research_assumptions("HBF+"), partial_page_policy="reject",
                          host_granularity_bytes=64)
    config = paper.build_paper_nand_config("HBF+", assumptions=assumptions)
    assert config.page_bytes == 4096
    assert config.host_granularity_bytes == 64
    assert config.metadata["research_assumptions"]["partial_page_policy"] == "reject"
    with pytest.raises(ValueError):
        NandCore(config, capture_details=False).submit(AccessRequest("partial-write", "write", 0, 64, 0))


@pytest.mark.parametrize("field,value", [
    ("pages_per_block", 0), ("pages_per_block", True), ("blocks_per_plane", -1),
    ("block_erase_ns", float("nan")), ("block_erase_ns", float("inf")),
    ("internal_bandwidth_gb_s", float("nan")), ("internal_bandwidth_gb_s", 0),
    ("max_outstanding_requests", True), ("max_outstanding_write_requests", 4097),
    ("host_full_duplex", 1), ("host_granularity_bytes", 0), ("host_granularity_bytes", True),
    ("partial_page_policy", "free_padding"), ("basis", ""),
])
def test_invalid_research_inputs_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(paper.default_research_assumptions("HBF"), **{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True])
def test_invalid_published_numeric_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        replace(paper.hardware_by_id("HBF"), hbf_read_gb_s=value)


def test_capacity_is_not_expanded_to_hide_insufficient_assumed_blocks():
    assumptions = replace(paper.default_research_assumptions("HBF"), blocks_per_plane=1)
    with pytest.raises(ValueError, match="capacity_bytes cannot exceed"):
        paper.build_paper_nand_config("HBF", assumptions=assumptions)


def test_catalog_validation_rejects_nonfinite_nested_values_and_duplicate_configurations():
    bad = deepcopy(paper.catalog_as_dict())
    bad["serving"]["default_tpot_slo_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paper.validate_catalog(bad)
    bad = deepcopy(paper.catalog_as_dict())
    bad["hardware"][-1] = deepcopy(bad["hardware"][0])
    with pytest.raises(ValueError, match="duplicate"):
        paper.validate_catalog(bad)


def test_unknown_configs_and_implicit_assumptions_are_rejected():
    for name in ("son_hbf", "B200", "", None):
        with pytest.raises(ValueError, match="unknown paper"):
            paper.hardware_by_id(name)
    with pytest.raises(TypeError, match="explicit"):
        paper.build_paper_nand_config("HBF", assumptions={})
    with pytest.raises(ValueError, match="no NAND"):
        paper.default_research_assumptions("HBM4")
