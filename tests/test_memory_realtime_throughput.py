"""Regression coverage for derived, analytical memory throughput metrics."""

from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.cost_models import HBMProfile, HostMemoryProfile
from heterollm_sim.hbf_media import hbf_media_service
from heterollm_sim.ir import ComponentSpec


def _hbf_component(**updates):
    contract = {
        "version": "cold_page_v1",
        "host_transaction_bytes": 64,
        "host_max_request_bytes": 4096,
        "media_page_bytes": 4096,
        "command_queue_depth": 256,
        "media_parallelism": 4,
        "page_read_latency_ns": 100,
        "page_program_latency_ns": 200,
        "access_pattern": "contiguous_page_aligned",
    }
    contract.update(updates.pop("hbf_media", {}))
    values = {
        "read_bandwidth_gbps": 100,
        "write_bandwidth_gbps": 100,
        "metadata": {"hbf_media": contract},
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_hbm_and_ddr_report_latency_limited_realtime_rate():
    hbm = HBMProfile(
        bandwidth_gb_s=1_000.0,
        read_latency_ns=500.0,
        max_outstanding_requests=1,
    ).memory_service(256)
    ddr = HostMemoryProfile(
        bandwidth_gb_s=100.0,
        efficiency=0.5,
        read_latency_ns=500.0,
        max_outstanding_requests=1,
    ).memory_service(256)
    for service in (hbm, ddr):
        assert service["throughput_evidence"] == "ANALYTICAL_DERIVED"
        assert service["realtime_throughput_gb_s"] < service["bandwidth_ceiling_gb_s"]
        assert service["bottleneck"] == "latency_concurrency"
        assert service["queue_wait_ns"] == 0.0


def test_hbf_reports_page_overfetch_and_media_bottleneck():
    service = hbf_media_service(_hbf_component(), 64, read=True)
    assert service["physical_bytes"] == 4096
    assert service["host_bandwidth_ceiling_gb_s"] == 12.5  # 100 Gbit/s
    assert service["host_bandwidth_utilization"] <= 1.0
    assert service["media_bandwidth_ceiling_gb_s"] == 12.5  # 100 Gbit/s
    assert service["realtime_throughput_gb_s"] < service["physical_realtime_throughput_gb_s"]
    assert service["throughput_evidence"] == "ANALYTICAL_DERIVED"
    assert service["bottleneck"] == "media"
    assert service["media_bandwidth_utilization"] < 1.0


def test_ssd_and_cxl_endpoint_paths_expose_same_derived_metrics():
    for kind in ("ssd", "cxl_memory"):
        component = ComponentSpec(
            component_id=kind,
            kind=kind,
            read_bandwidth_gbps=80.0,
            metadata={
                "read_latency_ns": 100.0,
                "transfer_granularity_bytes": 4096,
                "max_outstanding_requests": 2,
            },
        )
        phase = TopologyRouter._endpoint_phase(
            component, 3 * 4096, read=True, name="read"
        )
        assert phase is not None
        metadata = phase.metadata
        assert metadata["throughput_evidence"] == "ANALYTICAL_DERIVED"
        assert metadata["transactions"] == 3
        assert metadata["request_window_utilization"] == 1.0
        assert metadata["bandwidth_ceiling_gb_s"] == 10.0  # 80 Gbit/s
        assert metadata["realtime_throughput_gb_s"] <= metadata["bandwidth_ceiling_gb_s"]
        serial = TopologyRouter._endpoint_phase(replace(component, metadata={
            **component.metadata, "memory_service_model": "serialized",
        }), 3 * 4096, read=True, name="serialized")
        assert serial.metadata["realtime_throughput_gb_s"] < metadata["bandwidth_ceiling_gb_s"]
