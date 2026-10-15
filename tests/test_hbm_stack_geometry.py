"""HBM stack organization, independent interfaces and execution parity."""
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest

from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_mapping import map_dram_address
from heterollm_sim.memory_types import AccessRequest, DramConfig
from heterollm_sim.physical_contract import parse_execution_physical_config, reconfigure_physical_config


def hbm36(**overrides):
    values = dict(
        kind="HBM", generation="HBM3E", channels=16, pseudo_channels_per_channel=2,
        stacks=1, dies_per_stack=12, bank_groups_per_rank=12, banks_per_group=4,
        rows_per_bank=24576, row_bytes=1024, burst_bytes=32, data_width_bits=32,
        data_rate_mt_s=9200.0, bank_geometry_scope="per_stack",
        stack_interface_mode="independent",
    )
    values.update(overrides)
    return DramConfig(**values)


def test_hbm_12_high_counts_its_addressable_banks_once():
    config = hbm36()
    assert config.dies_per_stack == 12
    assert config.addressable_die_count == 1
    assert config.lane_count == 32
    assert config.computed_capacity_bytes == config.capacity_bytes == 36 * 1024**3
    assert config.physical_interface_bandwidth_gb_s == pytest.approx(1177.6)
    assert replace(config, dies_per_stack=8).computed_capacity_bytes == config.capacity_bytes


def test_independent_stacks_double_capacity_bandwidth_and_have_separate_lanes():
    single = hbm36()
    config = hbm36(stacks=2)
    assert config.capacity_bytes == 2 * single.capacity_bytes
    assert config.physical_interface_bandwidth_gb_s == 2 * single.physical_interface_bandwidth_gb_s
    first = map_dram_address(config, 0)
    second = map_dram_address(config, single.lane_count * config.burst_bytes)
    assert (first.stack, first.channel, first.pseudo_channel, first.die) == (0, 0, 0, 0)
    assert (second.stack, second.channel, second.pseudo_channel, second.die) == (1, 0, 0, 0)
    assert second.lane == 32
    last = map_dram_address(config, config.capacity_bytes - 1)
    assert (last.stack, last.die, last.channel, last.pseudo_channel) == (1, 0, 15, 1)
    assert (last.bank_group, last.bank, last.row, last.column) == (11, 3, 24575, 31)
    with pytest.raises(ValueError, match="outside modeled capacity"):
        map_dram_address(config, config.capacity_bytes)


@pytest.mark.parametrize("kind", ["DDR", "HBM"])
def test_per_die_geometry_and_shared_interface_require_explicit_scope(kind):
    config = DramConfig(kind=kind, channels=2, stacks=3, dies_per_stack=4,
                        banks_per_group=2, rows_per_bank=8, row_bytes=128, burst_bytes=32)
    assert config.lane_count == 2
    assert config.capacity_bytes == 3 * 4 * 2 * 2 * 8 * 128
    assert config.physical_interface_bandwidth_gb_s == pytest.approx(51.2)
    serialized = asdict(config)
    assert parse_execution_physical_config(serialized) == config
    serialized.pop("bank_geometry_scope")
    serialized.pop("stack_interface_mode")
    with pytest.raises(ValueError, match="bank_geometry_scope, stack_interface_mode"):
        parse_execution_physical_config(serialized)


@pytest.mark.parametrize("scope", ["per_die", "per_stack"])
@pytest.mark.parametrize("interface", ["shared", "independent"])
def test_all_organization_modes_have_bijective_address_coordinates(scope, interface):
    config = hbm36(channels=2, stacks=2, dies_per_stack=3, bank_groups_per_rank=1,
                   banks_per_group=2, rows_per_bank=4, row_bytes=128,
                   bank_geometry_scope=scope, stack_interface_mode=interface)
    coordinates = set()
    for address in range(0, config.capacity_bytes, config.burst_bytes):
        m = map_dram_address(config, address)
        coordinates.add((m.stack, m.die, m.lane, m.rank, m.bank_group, m.bank, m.row, m.column))
        assert m.stack < config.stacks
        assert m.die < config.addressable_die_count
        assert m.channel < config.channels
        assert m.row < config.rows_per_bank
    assert len(coordinates) == config.capacity_bytes // config.burst_bytes


@pytest.mark.parametrize("scope", ["per_die", "per_stack"])
@pytest.mark.parametrize("interface", ["shared", "independent"])
def test_detailed_folded_compiled_and_batch_paths_preserve_identical_bank_state(scope, interface):
    pytest.importorskip("numba", exc_type=ImportError)
    config = hbm36(channels=2, stacks=2, bank_groups_per_rank=1, banks_per_group=2,
                   rows_per_bank=4, data_rate_mt_s=8000.0, burst_interval_ns=1.0,
                   bank_geometry_scope=scope, stack_interface_mode=interface)
    requests = (
        AccessRequest("read_rows", "read", 0, 20 * 1024),
        AccessRequest("write_rows", "write", 16 * 1024, 20 * 1024, 3),
        AccessRequest("last", "read", config.capacity_bytes - 32, 32, 5),
    )
    reference = DramCore(config)
    expected = tuple(reference.submit(item) for item in requests)
    folded = DramCore(config, capture_details=False)
    with patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None):
        folded_results = tuple(folded.submit(item) for item in requests)
    compiled = DramCore(config, capture_details=False)
    compiled_results = tuple(compiled.submit(item) for item in requests)
    batch = DramCore(config, capture_details=False)
    batch_results = batch.submit_many(requests)
    assert compiled_results[0].counters.get("compiled_float64") is True
    assert batch_results[0].counters.get("compiled_batch_float64") is True
    for core, results in ((folded, folded_results), (compiled, compiled_results), (batch, batch_results)):
        assert core._banks == reference._banks
        assert core.timeline.bytes_moved == reference.timeline.bytes_moved
        for actual, wanted in zip(results, expected):
            assert actual.completion_ns == pytest.approx(wanted.completion_ns)
            for counter in ("row_hits", "row_misses", "row_conflicts", "burst_count"):
                assert actual.counters[counter] == wanted.counters[counter]
    assert any(key.startswith("dram:bank:1:") for key in batch._banks)


@pytest.mark.parametrize("generation,channels,groups,banks_per_group,bank_total", [
    ("HBM3E", 16, 12, 4, 12288), ("HBM4", 32, 6, 8, 24576),
])
def test_standard_multistack_arrays_use_acceleration_with_exact_state(
        generation, channels, groups, banks_per_group, bank_total):
    pytest.importorskip("numba", exc_type=ImportError)
    config = hbm36(generation=generation, channels=channels, stacks=8,
                   bank_groups_per_rank=groups, banks_per_group=banks_per_group,
                   rows_per_bank=12288 if generation == "HBM4" else 24576,
                   burst_interval_ns=1.0)
    assert config.lane_count * config.bank_count == bank_total > 8192
    row_stride = config.lane_count * config.bank_count * config.row_bytes
    requests = (
        AccessRequest("first_read", "read", 0, 4096),
        AccessRequest("same_row_write", "write", 0, 4096, 3),
        AccessRequest("next_row_read", "read", row_stride, 4096, 4),
        AccessRequest("final_banks_write", "write", config.capacity_bytes - 4096, 4096, 5),
    )
    reference = DramCore(config)
    expected = reference.submit_many(requests)
    folded = DramCore(config, capture_details=False)
    assert folded._can_accelerate()
    with patch("heterollm_sim._compiled_dram.execute_compiled", return_value=None):
        folded_results = tuple(folded.submit(item) for item in requests)
    compiled = DramCore(config, capture_details=False)
    compiled_results = tuple(compiled.submit(item) for item in requests)
    batch = DramCore(config, capture_details=False)
    batch_results = batch.submit_many(requests)
    assert all(result.counters.get("compiled_float64") for result in compiled_results)
    assert all(result.counters.get("compiled_batch_float64") for result in batch_results)
    for core, results in ((folded, folded_results), (compiled, compiled_results), (batch, batch_results)):
        assert core._banks == reference._banks
        assert core.timeline.bytes_moved == reference.timeline.bytes_moved
        for actual, wanted in zip(results, expected):
            assert actual.completion_ns == pytest.approx(wanted.completion_ns)
            for counter in ("row_hits", "row_misses", "row_conflicts", "burst_count"):
                assert actual.counters[counter] == wanted.counters[counter]
    assert any(key.startswith("dram:bank:7:") for key in batch._banks)


def test_accelerated_bank_and_batch_allocations_remain_bounded():
    from heterollm_sim._compiled_dram import execute_compiled, execute_compiled_many
    config = hbm36(generation="HBM4", channels=32, stacks=16, dies_per_stack=16,
                   bank_groups_per_rank=8, banks_per_group=8, rows_per_bank=16384)
    request = AccessRequest("bounded_read", "read", 0, 4096)
    core = DramCore(config, capture_details=False)
    assert config.lane_count * config.bank_count == 65536
    assert core._can_accelerate()
    # Reject the batch before resource IDs or NumPy buffers are allocated.
    with patch("heterollm_sim._compiled_dram._resource_ids", side_effect=AssertionError("allocated")):
        assert execute_compiled_many(core, (request,) * 128) is None
        larger = DramCore(replace(config, stacks=17), capture_details=False)
        assert not larger._can_accelerate()
        assert execute_compiled(larger, request) is None
        assert execute_compiled_many(larger, (request, request)) is None
    assert core._banks == {}
    assert core.timeline.bytes_moved == {}


def test_geometry_expansion_does_not_reapply_die_count_or_keep_standard_claims():
    base = hbm36(metadata={
        "geometry_status": "standard_organization", "geometry_source": "JEDEC table",
        "raw_capacity_bytes": 36 * 1024**3, "raw_stack_capacity_bytes": 36 * 1024**3,
        "visible_capacity_bytes": 36 * 1024**3, "unexposed_capacity_bytes": 0,
        "die_density_gbits": 24, "capacity_formula": "physical density * die count",
        "parameter_provenance": {
            "rows_per_bank": {"status": "standard", "source": "JEDEC table"},
            "row_bytes": {"status": "standard", "source": "JEDEC table"},
        }, "unknown_parameters": ["open_ns"],
    })
    stripe = base.lane_count * base.bank_count * base.row_bytes
    grown = reconfigure_physical_config(base, overrides={"capacity_bytes": base.capacity_bytes + stripe},
                                        expand_geometry=True)
    assert grown.rows_per_bank == base.rows_per_bank + 1
    assert grown.capacity_bytes == grown.computed_capacity_bytes
    assert grown.metadata["geometry_status"] == "explicit_assumption"
    assert "geometry_source" not in grown.metadata
    assert "die_density_gbits" not in grown.metadata
    assert "capacity_formula" not in grown.metadata
    assert grown.metadata["raw_capacity_bytes"] == grown.capacity_bytes
    assert grown.metadata["visible_capacity_bytes"] == grown.capacity_bytes
    assert grown.metadata["parameter_provenance"]["rows_per_bank"] == {
        "status": "explicit_assumption", "source": "capacity_expansion",
    }
    assert grown.metadata["parameter_provenance"]["row_bytes"]["status"] == "standard"
    assert grown.metadata["unknown_parameters"] == ["open_ns"]
    assert base.metadata["geometry_status"] == "standard_organization"
    limited = reconfigure_physical_config(base, overrides={"capacity_bytes": 36_000_000_000})
    assert limited.metadata["geometry_status"] == "standard_organization"
    assert limited.metadata["raw_capacity_bytes"] == base.capacity_bytes
    assert limited.metadata["visible_capacity_bytes"] == 36_000_000_000
    assert limited.metadata["unexposed_capacity_bytes"] == base.capacity_bytes - 36_000_000_000


def test_explicit_geometry_override_downgrades_only_changed_field_provenance():
    base = hbm36(metadata={"geometry_status": "standard_organization",
                           "die_density_gbits": 24,
                           "parameter_provenance": {"dies_per_stack": {"status": "vendor", "source": "vendor"}}})
    changed = reconfigure_physical_config(base, overrides={"dies_per_stack": 8})
    assert changed.metadata["geometry_status"] == "explicit_assumption"
    assert changed.metadata["parameter_provenance"]["dies_per_stack"] == {
        "status": "explicit_assumption", "source": "physical_config_override",
    }
    unchanged = reconfigure_physical_config(base, overrides={"read_latency_ns": 40.0})
    assert unchanged.metadata["geometry_status"] == "standard_organization"


@pytest.mark.parametrize("overrides", [{"row_bytes": 2048}, {"generation": "HBM4"}])
def test_changed_page_organization_drops_obsolete_standard_page_descriptions(overrides):
    source = {"source": "JEDEC original protocol PDF"}
    base = hbm36(metadata={
        "geometry_status": "standard_organization", "die_density_gbits": 24,
        "jedec_standard": "JESD238", "jedec_reference_file": "JESD238.pdf",
        "source_provenance": "original protocol PDF",
        "effective_addressed_page_bytes": 1024, "activated_page_bytes": 2048,
        "row_bytes_basis": "effective 1KiB page, half of an open 2KiB page",
        "geometry_limitations": ["row-hit model separates effective 1KiB pages"],
        "parameter_provenance": {
            "effective_addressed_page_bytes": {"status": "standard_organization"},
            "activated_page_bytes": {"status": "standard_organization"},
        },
        "source_documents": [source],
    })
    revised = reconfigure_physical_config(base, overrides=overrides)
    assert revised.metadata["geometry_status"] == "explicit_assumption"
    for name in ("jedec_standard", "jedec_reference_file", "source_provenance",
                 "effective_addressed_page_bytes", "activated_page_bytes",
                 "row_bytes_basis", "geometry_limitations"):
        assert name not in revised.metadata
        assert name not in revised.metadata["parameter_provenance"]
    assert revised.metadata["source_documents"] == [source]
    assert base.metadata["effective_addressed_page_bytes"] == 1024


@pytest.mark.parametrize("field,value", [
    ("rows_per_bank", 24577), ("bank_groups_per_rank", 16),
    ("row_bytes", 2048), ("pseudo_channels_per_channel", 3),
    ("dies_per_stack", 8), ("bank_geometry_scope", "per_die"),
    ("stack_interface_mode", "shared"), ("data_lanes", 64),
])
def test_manual_json_geometry_changes_cannot_retain_standard_label(field, value):
    base = hbm36(metadata={"geometry_status": "standard_organization", "die_density_gbits": 24})
    raw = asdict(base)
    raw[field] = value
    with pytest.raises(ValueError, match="standard HBM organization conflicts"):
        parse_execution_physical_config(raw)
    revised = reconfigure_physical_config(base, overrides={field: value})
    assert revised.metadata["geometry_status"] == "explicit_assumption"
    assert parse_execution_physical_config(asdict(revised)) == revised


@pytest.mark.parametrize("density", [24, 32])
@pytest.mark.parametrize("height", [4, 8, 12, 16])
def test_hbm4_standard_organizations_count_sid_banks_once(density, height):
    # JESD270-4A tables 4/5: 64 PCs, 4H banks per PC, eight banks per group.
    config = hbm36(generation="HBM4", channels=32, dies_per_stack=height,
                   bank_groups_per_rank=height // 2, banks_per_group=8,
                   rows_per_bank=density * 512,
                   metadata={"geometry_status": "standard_organization", "die_density_gbits": density})
    assert parse_execution_physical_config(config) is config
    assert parse_execution_physical_config(asdict(config)) == config
    assert config.computed_capacity_bytes == density * height * 1024**3 // 8
    assert config.addressable_die_count == 1
    assert config.bank_count == 4 * height
    last = map_dram_address(config, config.capacity_bytes - 1)
    assert (last.channel, last.pseudo_channel, last.bank_group, last.bank, last.row) == (
        31, 1, height // 2 - 1, 7, density * 512 - 1,
    )


@pytest.mark.parametrize("overrides", [
    {"bank_groups_per_rank": 12, "banks_per_group": 4},
    {"rows_per_bank": 24576},
])
def test_hbm4_rejects_hbm3_coordinates_with_retained_standard_label(overrides):
    base = hbm36(generation="HBM4", channels=32, bank_groups_per_rank=6,
                 banks_per_group=8, rows_per_bank=12288,
                 metadata={"geometry_status": "standard_organization", "die_density_gbits": 24})
    raw = {**asdict(base), **overrides}
    with pytest.raises(ValueError, match="standard HBM organization conflicts"):
        parse_execution_physical_config(raw)
    with pytest.raises(ValueError, match="standard HBM organization conflicts"):
        parse_execution_physical_config(replace(base, **overrides))
    revised = reconfigure_physical_config(base, overrides=overrides)
    assert revised.metadata["geometry_status"] == "explicit_assumption"
    assert parse_execution_physical_config(asdict(revised)) == revised
