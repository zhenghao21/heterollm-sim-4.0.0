import pytest

from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.data_motion import endpoint_service, resolve_service
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


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "unknown"},
        {"host_max_request_bytes": 32},
        {"command_queue_depth": 0},
        {"media_parallelism": True},
        {"physical_planes": 0},
        {"erase_latency_ns": 1},
    ],
)
def test_shared_nand_contract_rejects_unknown_or_invalid_fields(changes):
    with pytest.raises(ValueError):
        validate_nand_media(_ssd(**changes))
