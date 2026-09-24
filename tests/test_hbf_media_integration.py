from dataclasses import replace

import pytest

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec


def contract(**updates):
    value = {
        "version": "cold_page_v1",
        "host_transaction_bytes": 64,
        "host_max_request_bytes": 4096,
        "media_page_bytes": 4096,
        "command_queue_depth": 1024,
        "media_parallelism": 96,
        "physical_planes": 96,
        "page_read_latency_ns": 4_000.0,
        "page_program_latency_ns": 75_000.0,
        "access_pattern": "contiguous_page_aligned",
    }
    value.update(updates)
    return value


def hbf(**metadata):
    return ComponentSpec(
        "hbf0", "hbf", read_bandwidth_gbps=12_288.0,
        write_bandwidth_gbps=1_024.0,
        metadata={"hbf_media": contract(**metadata), **metadata},
    )


def test_endpoint_charges_physical_page_and_preserves_host_request():
    phase = TopologyRouter._endpoint_phase(hbf(), 1, read=True, name="weight")
    assert phase.metadata["transferred_bytes"] == 64
    assert phase.metadata["physical_bytes"] == 4096
    assert phase.demands[0].bytes_moved == 4096
    assert phase.metadata["hbf_media"]["page_read_latency_ns"] == 4_000.0


def test_partial_write_charges_rmw_and_program_without_free_ack():
    phase = TopologyRouter._endpoint_phase(hbf(), 1, read=False, name="kv")
    assert phase.metadata["hbf_media"]["rmw_read_operations"] == 1
    assert phase.metadata["physical_bytes"] == 8192
    assert phase.demands[0].bytes_moved == 8192
    assert phase.demands[0].service_ns >= 75_000.0


def test_ir_rejects_invalid_hbf_media_contract_early():
    with pytest.raises(ValueError, match="command_queue_depth"):
        hbf(command_queue_depth=300)


def test_no_hbf_contract_keeps_legacy_endpoint_semantics():
    component = ComponentSpec("ssd0", "ssd", read_bandwidth_gbps=80.0, metadata={
        "read_latency_ns": 100.0, "transfer_granularity_bytes": 4096,
        "max_outstanding_requests": 1,
    })
    phase = TopologyRouter._endpoint_phase(component, 1, read=True, name="read")
    assert phase.metadata["transferred_bytes"] == 4096
    assert "hbf_media" not in phase.metadata
