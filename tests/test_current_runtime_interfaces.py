"""Runtime inputs explicitly select the current ownership and cost contracts."""
from dataclasses import replace

import pytest

from heterollm_sim.batched_accelerator import BatchedGemmInput, evaluate_batched_gemm
from heterollm_sim.calibration import NativeCalibrationProfile, audit_exact_operator_coverage
from heterollm_sim.cost_models import GemmWorkload, validate_mmvq_hbm_mode
from heterollm_sim.kernel_model import KernelModelProfile, graph_launch_phases


@pytest.mark.parametrize("metadata", [
    {"bandwidth_source": "component"},
    {"bandwidth_source": "memory_component"},
    {"bandwidth_source": "shared_component"},
    {"service_ref": "hbm0"},
    {"service_ref": "old-hbm-service"},
])
def test_transfer_routing_rejects_implicit_and_aliased_memory_services(metadata):
    from heterollm_sim.communication import TopologyRouter
    from heterollm_sim.data_motion import resolve_link_service
    from heterollm_sim.reference import build_reference_scenario

    hardware = build_reference_scenario().hardware
    components = tuple(
        replace(item, metadata={**item.metadata,
            "memory_service": {**item.metadata.get("memory_service", {}),
                               "service_id": "old-hbm-service"}})
        if item.component_id == "hbm0" else item
        for item in hardware.components
    )
    link = replace(next(item for item in hardware.links if item.link_id == "gpu-hbm0"),
                   bandwidth_gbps=0., latency_ns=0., metadata=metadata)
    hardware = replace(hardware, components=components, links=(link,))
    with pytest.raises(ValueError, match="service_ref"):
        TopologyRouter(hardware)
    endpoints = hardware.component_map()
    with pytest.raises(ValueError, match="service_ref"):
        resolve_link_service(link, endpoints[link.source_component], endpoints[link.target_component])


def test_transfer_routing_preserves_explicit_memory_service_ownership():
    from heterollm_sim.communication import TopologyRouter
    from heterollm_sim.data_motion import resolve_link_service, resolve_service
    from heterollm_sim.reference import build_reference_scenario

    hardware = build_reference_scenario().hardware
    link = replace(next(item for item in hardware.links if item.link_id == "gpu-hbm0"),
                   bandwidth_gbps=0., latency_ns=0.,
                   metadata={"service_ref": "hbm0.access"})
    hardware = replace(hardware, links=(link,))
    endpoints = hardware.component_map()
    router = TopologyRouter(hardware)
    hop, = router.route(link.source_component, link.target_component, 256)
    assert hop.service_ref == "hbm0.access"
    assert hop.resource_id == resolve_service(endpoints["hbm0"]).resource_id
    assert hop.demands(256) == ()
    assert resolve_link_service(link, endpoints[link.source_component], endpoints[link.target_component]) is None


def test_graph_replay_prices_only_the_explicit_device_envelope():
    with pytest.raises(TypeError, match="graph_launch_ns"):
        KernelModelProfile("gpu", "runtime", "arch", graph_launch_ns=10)
    profile = KernelModelProfile("gpu", "runtime", "arch", launch_ns=5,
        launch_evidence="ordinary launch", graph_device_launch_ns=7,
        graph_device_launch_evidence="device replay", graph_device_launch_qualified=True)
    replay = graph_launch_phases(profile, "gpu.frontend", 4, captured=True)
    ordinary = graph_launch_phases(profile, "gpu.frontend", 4, captured=False)
    assert replay.demands[0].service_ns == 7
    assert replay.metadata["submission_count"] == 1
    assert ordinary.demands[0].service_ns == 20


def test_batched_structural_model_never_silently_uses_nominal_bandwidth():
    fields = dict(m=16, k=16, n=16, peak_tops=1, memory_bandwidth_gb_s=1)
    with pytest.raises(TypeError, match="memory_bandwidth_model"):
        BatchedGemmInput(**fields)
    with pytest.raises(ValueError, match="positive integral geometry"):
        evaluate_batched_gemm(BatchedGemmInput(**fields,
            memory_bandwidth_model="mma_output_tile_wave"), backend="numpy")
    nominal = evaluate_batched_gemm(BatchedGemmInput(**fields,
        memory_bandwidth_model="nominal_bandwidth"), backend="numpy")
    assert nominal.memory_ns[0] == nominal.gemm_io_bytes[0]


def test_packed_transform_work_requires_its_format_segments():
    with pytest.raises(ValueError, match="explicit format segments"):
        GemmWorkload(1, 32, 4, packed_weight_formats=("q4_0",),
            packed_weight_transform_operations=128)
    with pytest.raises(ValueError, match="unsupported mmvq_hbm_mode"):
        validate_mmvq_hbm_mode("legacy_mma_output_wave")


def test_calibration_uses_scoped_launches_and_canonical_operator_keys():
    with pytest.raises(TypeError, match="launch_ns_per_call"):
        NativeCalibrationProfile(launch_ns_per_call=100)
    record = dict(stage="attention", phase="decode", shape="1x16", dtype="fp16",
                  layout="contiguous", kernel_family="attention")
    key = "attention|decode|1x16|fp16|contiguous|attention"
    profile = NativeCalibrationProfile(kernel_stage_mapping={"exact_operator_keys": {
        key: {"status": "calibrated", "train_ns_per_instance": 10, "train_instances": 2}}})
    assert audit_exact_operator_coverage(profile, (record,))["exact_hits"] == 1
    record["semantic_shape"] = record.pop("shape")
    assert audit_exact_operator_coverage(profile, (record,))["malformed_records"] == 1


def test_mmvq_summary_counts_only_current_mode_decisions():
    from heterollm_sim.contracts import TaskCategory, TaskSpec
    from heterollm_sim.cost_models import MMVQ_HBM_MODE_NOMINAL
    from heterollm_sim.planner import summarize_mmvq_hbm_mode

    def task(identity, bandwidth):
        return TaskSpec(identity, "request", identity, TaskCategory.COMPUTE,
            metadata={"phase": "gpu_gemm", "cost_model": {"hbm_bandwidth": bandwidth}})

    applied = task("applied", {"requested_mmvq_hbm_mode": MMVQ_HBM_MODE_NOMINAL,
                              "mmvq_nominal_mode_applied": True})
    fallback = task("fallback", {"requested_mmvq_hbm_mode": MMVQ_HBM_MODE_NOMINAL,
                                "mmvq_nominal_mode_applied": False,
                                "mmvq_nominal_mode_reason": "not_vector_geometry"})
    summary = summarize_mmvq_hbm_mode((applied, applied, fallback, task("missing", {})), MMVQ_HBM_MODE_NOMINAL)
    assert summary["requested_tasks"] == summary["gpu_gemm_tasks"] == 3
    assert summary["applied_tasks"] == summary["fallback_tasks"] == summary["unobserved_tasks"] == 1
    assert summary["fallback_reason_counts"] == {"not_vector_geometry": 1}
    assert summary["requested_unaccounted_tasks"] == 1
    assert "legacy_tasks" not in summary
    with pytest.raises(ValueError, match="unsupported mmvq_hbm_mode"):
        summarize_mmvq_hbm_mode((task("old", {"requested_mmvq_hbm_mode": "legacy_mma_output_wave"}),),
                               MMVQ_HBM_MODE_NOMINAL)
