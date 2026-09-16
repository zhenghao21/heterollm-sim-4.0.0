"""Source-qualified MMVQ planner integration; no native or GPU execution."""

from dataclasses import replace

from heterollm_sim import planner
from heterollm_sim.conversion_work import SOURCE_SHA256 as CONVERSION_SOURCE_SHA256
from heterollm_sim.mmvq_work import SOURCE_SHA256 as MMVQ_SOURCE_SHA256
from tests.test_mmq_planner import scenario as base_scenario


def _qualified_mmvq_case(*, include_source_contract=True):
    case = base_scenario(tokens=4, weight_format="Q5_0")
    target = next(component for component in case.hardware.components if component.component_id == "gpu0")
    conversion_contract = {
        "compute_capability": 1200,
        "highest_compiled_arch": 1200,
        "warp_size": 32,
        "source_hashes": dict(CONVERSION_SOURCE_SHA256),
        "runtime_binary_sha256": "a" * 64,
        "ordinary_contiguous_2d": True,
    }
    target_metadata = dict(target.metadata)
    if include_source_contract:
        target_metadata["llama_cpp_conversion_source_contract"] = conversion_contract
    target = replace(target, metadata=target_metadata)
    flags = {
        **case.workload.metadata,
        "llama_cpp_mmq_source_work": True,
        "llama_cpp_f32_q8_1_mmvq": True,
        "llama_cpp_f32_hidden_storage": True,
        "llama_cpp_conversion_cta_costs": True,
        "llama_cpp_gpu_native_invocations": {
            "source_refs": [
                {"path": "C:/locked/" + relative, "sha256": digest}
                for relative, digest in MMVQ_SOURCE_SHA256.items()
            ]
        },
    }
    return replace(
        case,
        hardware=replace(
            case.hardware,
            components=tuple(
                target if component.component_id == "gpu0" else component
                for component in case.hardware.components
            ),
        ),
        workload=replace(case.workload, metadata=flags),
    )


def _mmvq_mains(schedule):
    return [
        task
        for task in schedule.tasks
        if task.metadata.get("phase") == "gpu_gemm"
        and task.metadata.get("mmvq_source_work", {}).get("status")
        == "source_geometry_unpriced"
    ]


def test_source_qualified_mmvq_geometry_reaches_main_execution_metadata():
    schedule = planner.compile_scenario(_qualified_mmvq_case())
    mains = _mmvq_mains(schedule)

    assert mains
    for main in mains:
        source = main.metadata["mmvq_source_work"]
        cost = main.metadata["cost_model"]
        hbm = cost["hbm_bandwidth"]
        assert source["grid"] == (source["n"] // source["rows_per_cta"], 1, 1)
        assert source["block"] == (32, source["warps_per_cta"], 1)
        assert source["source_vector_dot_calls"] > 0
        assert cost["kernel_family"] == "cuda_mmvq_vector_dp4a"
        assert cost["source_compute_category"] == (
            "integer_vector_dp4a_with_float_scale_and_warp_reduction_unpriced"
        )
        assert cost["source_geometry_priced"] is False
        assert cost["source_geometry_eligibility"] == "unpriced_no_mma_wave_claim"
        assert hbm["source_geometry_hbm_concurrency_applied"] is False
        assert "legacy analytical fallback" in hbm["source_geometry_hbm_concurrency_reason"]


def test_source_qualified_mmvq_adds_metadata_without_an_extra_main_demand():
    qualified = planner.compile_scenario(_qualified_mmvq_case())
    fallback = planner.compile_scenario(_qualified_mmvq_case(include_source_contract=False))
    qualified_mains = {task.metadata["projection_id"]: task for task in _mmvq_mains(qualified)}
    fallback_mains = {
        task.metadata["projection_id"]: task
        for task in fallback.tasks
        if task.metadata.get("phase") == "gpu_gemm"
    }

    assert qualified_mains
    for projection_id, main in qualified_mains.items():
        baseline = fallback_mains[projection_id]
        assert [
            (demand.resource_id, demand.service_ns, demand.bytes_moved)
            for demand in main.demands
        ] == [
            (demand.resource_id, demand.service_ns, demand.bytes_moved)
            for demand in baseline.demands
        ]
        assert main.metadata["cost_model"]["roofline_service_ns"] == baseline.metadata["cost_model"]["roofline_service_ns"]
        assert baseline.metadata["mmvq_source_work"]["status"] == "uncovered"
        assert "mmvq_source_work" not in baseline.metadata["cost_model"]


def test_missing_source_contract_keeps_mmvq_geometry_explicitly_uncovered():
    schedule = planner.compile_scenario(_qualified_mmvq_case(include_source_contract=False))
    mains = [task for task in schedule.tasks if task.metadata.get("phase") == "gpu_gemm"]

    assert mains
    assert any(
        task.metadata.get("mmvq_source_work", {}).get("reason")
        == "missing_mmvq_runtime_source_contract"
        for task in mains
    )
    assert all(
        task.metadata["cost_model"].get("kernel_family") != "cuda_mmvq_vector_dp4a"
        for task in mains
    )
