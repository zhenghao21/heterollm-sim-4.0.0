from dataclasses import replace

import pytest

from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.data_motion import (
    READ,
    WRITE,
    DataAccess,
    MemoryPosition,
    endpoint_service,
    expand_access,
    resolve_service,
)
from heterollm_sim.hbf_media import nand_media_service, validate_nand_media
from heterollm_sim.ir import ComponentSpec


def _contract(**updates):
    value = {
        "version": "nand_media_v1",
        "host_transaction_bytes": 64,
        "host_max_request_bytes": 4096,
        "media_page_bytes": 4096,
        "command_queue_depth": 4,
        "media_parallelism": 4,
        "page_read_latency_ns": 100,
        "page_program_latency_ns": 200,
        "access_pattern": "contiguous_page_aligned",
        "physical_planes": 2,
        "erase_block_bytes": 262144,
        "erase_latency_ns": 1500.0,
    }
    value.update(updates)
    return value


def _ssd(kind="ssd", **metadata):
    return ComponentSpec(
        "ssd0",
        kind,
        read_bandwidth_gbps=100,
        write_bandwidth_gbps=100,
        metadata={"nand_media": _contract(**metadata)},
    )


def test_shared_nand_contract_reaches_ssd_endpoint_and_preserves_pages():
    component = _ssd(media_page_bytes=8192)
    endpoint = endpoint_service(component, 64, read=False, name="write")
    service = resolve_service(component)
    media = endpoint.metadata["nand_media"]

    assert endpoint.metadata["memory_service_model"] == "nand_media_v1"
    assert media["media_page_bytes"] == 8192
    assert media["physical_read_bytes"] == 8192
    assert media["physical_write_bytes"] == 8192
    assert media["rmw_read_operations"] == 1
    assert endpoint.demands[0].bytes_moved == 16384
    assert service.price("WRITE", 64)["physical_bytes"] == 16384
    nvme = endpoint_service(_ssd(kind="nvme"), 4096, read=True, name="read")
    assert nvme.metadata["memory_service_model"] == "nand_media_v1"


def test_shared_nand_plane_cap_and_queue_are_effective():
    wide = nand_media_service(
        _ssd(media_parallelism=16, physical_planes=16, command_queue_depth=32),
        16 * 4096,
        True,
    )
    narrow = nand_media_service(
        _ssd(media_parallelism=16, physical_planes=1, command_queue_depth=32),
        16 * 4096,
        True,
    )
    assert wide["effective_parallelism"] == 16
    assert narrow["effective_parallelism"] == 1
    assert wide["service_ns"] < narrow["service_ns"]


def test_known_data_access_offset_splits_nand_page_and_rmw():
    component = _ssd()
    service = resolve_service(component)
    aligned_read = expand_access(
        DataAccess("aligned-read", READ, 128, source=MemoryPosition("ssd0", 0)),
        {"ssd0": service},
    )
    crossing_read = expand_access(
        DataAccess("crossing-read", READ, 128, source=MemoryPosition("ssd0", 4032)),
        {"ssd0": service},
    )
    aligned_write = expand_access(
        DataAccess("aligned-write", WRITE, 128, target=MemoryPosition("ssd0", 0)),
        {"ssd0": service},
    )
    crossing_write = expand_access(
        DataAccess("crossing-write", WRITE, 128, target=MemoryPosition("ssd0", 4032)),
        {"ssd0": service},
    )

    assert aligned_read.phases[0].physical_bytes == 4096
    assert crossing_read.phases[0].physical_bytes == 8192
    assert aligned_write.phases[0].physical_bytes == 8192
    assert crossing_write.phases[0].physical_bytes == 16384
    assert crossing_read.phases[0].service_ns > aligned_read.phases[0].service_ns
    assert crossing_write.phases[0].service_ns > aligned_write.phases[0].service_ns


def test_legacy_hbf_alias_and_dram_path_remain_separate():
    hbf = ComponentSpec(
        "hbf0",
        "hbf",
        read_bandwidth_gbps=100,
        write_bandwidth_gbps=100,
        metadata={"hbf_media": {**_contract(version="cold_page_v1", command_queue_depth=256)}},
    )
    endpoint = endpoint_service(hbf, 64, read=True, name="read")
    assert endpoint.metadata["hbf_media"]["contract_field"] == "hbf_media"
    assert endpoint.metadata["memory_service_model"] == "cold_page_v1"

    profile = HBMProfile(bandwidth_gb_s=100, read_latency_ns=100)
    assert "nand_media" not in profile.memory_service(4096)


def test_shared_nand_erase_is_distinct_and_background_is_reported():
    component = _ssd(background_work={"gc": "unknown"})
    media = nand_media_service(
        component, 262144, False, operation="erase",
        background_work={"gc": "explicit"},
    )
    assert media["operation"] == "erase"
    assert media["erase_operations"] == 1
    assert media["physical_bytes"] == 262144
    assert media["background_work"] == {"gc": "explicit"}
    endpoint = endpoint_service(component, 262144, read=False, name="erase", operation="erase")
    assert endpoint.metadata["operation"] == "erase"
    assert endpoint.demands[0].bytes_moved == 262144
    legacy = __import__("heterollm_sim.hbf_media", fromlist=["hbf_media_service"]).hbf_media_service(
        component, 262144, False, operation="erase", background_work={"gc": "explicit"}
    )
    assert legacy["erase_operations"] == 1


def test_erase_does_not_require_write_bandwidth_but_shares_resolved_resource():
    component = replace(_ssd(), write_bandwidth_gbps=0)
    endpoint = endpoint_service(component, 262144, read=False, name="erase", operation="erase")
    service = resolve_service(component)
    assert endpoint.demands[0].resource_id == service.resource_id
    assert endpoint.demands[0].service_ns > 0


def test_generic_endpoint_cannot_treat_erase_as_a_write():
    component = ComponentSpec("dram-like", "hbm", read_bandwidth_gbps=100, write_bandwidth_gbps=100)
    with pytest.raises(ValueError, match="NAND media contract"):
        endpoint_service(component, 262144, read=False, name="erase", operation="erase")


def test_data_access_erase_reaches_shared_physical_service():
    service = resolve_service(_ssd())
    motion = expand_access(
        DataAccess(
            "erase", "ERASE", 262144,
            target=MemoryPosition("ssd0"),
        ),
        {"ssd0": service},
    )
    assert motion.phases[0].kind.value == "ERASE"
    assert motion.phases[0].physical_bytes == 262144


@pytest.mark.parametrize(
    "offset,count,pages,rmw",
    [(64, 64, 1, 1), (64, 4032, 1, 1), (64, 4096, 2, 2),
     (0, 4096, 1, 0), (4032, 128, 2, 2), (4096, 4096, 1, 0)],
)
def test_known_offset_prices_distinct_partial_pages(offset, count, pages, rmw):
    component = _ssd(access_pattern="unknown_alignment_conservative")
    bill = nand_media_service(component, count, False, page_offset_bytes=offset)
    assert bill["pages_touched"] == pages
    assert bill["rmw_read_operations"] == rmw
    assert bill["physical_bytes"] == (pages + rmw) * 4096
    assert bill["address_scope"] == "known_page_offset"


def test_known_erase_offset_counts_crossed_blocks_through_physical_service():
    component = _ssd(physical_planes=1)
    service = resolve_service(component)
    aligned = service.price("ERASE", 128, page_offset_bytes=0)
    crossing = service.price("ERASE", 128, page_offset_bytes=262144 - 64)
    assert aligned["erase_operations"] == 1
    assert crossing["erase_operations"] == 2
    assert crossing["physical_bytes"] == 2 * 262144
    assert crossing["media_erase_service_ns"] == 2 * aligned["media_erase_service_ns"]
    assert crossing["address_scope"] == "known_block_offset"
    assert crossing["block_offset_bytes"] == 262144 - 64


def test_parameter_evidence_is_explicit_and_missing_geometry_stays_unknown():
    legacy = nand_media_service(_ssd(), 64, True)
    assert legacy["parameter_evidence"]["media_page_bytes"]["status"] == "parameterized"
    assert legacy["parameter_evidence"]["physical_dies"]["status"] == "unknown"
    sourced = nand_media_service(_ssd(
        source="analysis protocol v1",
        parameter_evidence={"media_page_bytes": {"status": "sourced", "source": "device datasheet section 4"}},
    ), 64, True)
    assert sourced["parameter_evidence"]["media_page_bytes"] == {
        "value": 4096, "status": "sourced", "source": "device datasheet section 4",
    }
    assert sourced["parameter_evidence"]["command_queue_depth"]["status"] == "parameterized"
    assert sourced["parameter_evidence"]["command_queue_depth"]["source"] == "analysis protocol v1"
    assert sourced["source"] == "analysis protocol v1"
    assert sourced["profile_evidence"] == "analytical/no hardware validated"


@pytest.mark.parametrize("changes", [
    {"erase_block_bytes": 4095}, {"erase_block_bytes": 4097},
    {"source": ""}, {"parameter_evidence": []},
    {"parameter_evidence": {"media_page_bytes": {"status": "measured"}}},
    {"parameter_evidence": {"media_page_bytes": {"status": "verified", "source": "x"}}},
    {"parameter_evidence": {"not_a_parameter": {"status": "parameterized"}}},
])
def test_nand_geometry_and_evidence_fail_closed(changes):
    with pytest.raises(ValueError):
        _ssd(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "unknown"},
        {"host_max_request_bytes": 32},
        {"command_queue_depth": 0},
        {"media_parallelism": True},
        {"physical_planes": 0},
        {"erase_latency_ns": -1},
    ],
)
def test_shared_nand_contract_rejects_unknown_or_invalid_fields(changes):
    with pytest.raises(ValueError):
        validate_nand_media(_ssd(**changes))
