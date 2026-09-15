"""Source-qualified llama.cpp host GEMM offload; synthetic graphs, no native runs."""
from dataclasses import replace
from pathlib import Path

import pytest

from heterollm_sim import planner
from heterollm_sim.reporting import run_scenario
from heterollm_sim.control_plane_state import mapping_fingerprint_status
from heterollm_sim.cost_models import GemmWorkload, HostGemmOffloadCapability
from heterollm_sim.runtime_adapters import (
    LLAMA_CUDA_OP_OFFLOAD_SCHEMA, LlamaCppRuntimeConfig,
    apply_llama_cuda_op_offload, derive_llama_cuda_op_offload_contract,
)
from tests.test_physical_projection_invocations import _descriptors, _full_layer, _scenario

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source/llama.cpp-semantic"


def contract(minimum=32):
    return {
        "schema": LLAMA_CUDA_OP_OFFLOAD_SCHEMA, "backend": "CUDA",
        "source_sha256": {"fixture/ggml-cuda.cu": "a" * 64},
        "minimum_m": minimum,
        "supported_weight_formats": ("Q8_0", "Q6_K", "IQ3_S", "IQ4_XS"),
        "prediction_provenance": "conditional_development_assumption",
        "environment": {"status": "uncaptured_default_assumption"},
    }


def host_case(rows=32, *, enabled=True, available=True, evidence=None, vocabulary=256):
    base = _scenario(_full_layer(_descriptors()), tokens=rows, capability=True, vocabulary=vocabulary)
    placement = replace(base.placement,
        op_to_component={**{key: "cpu0" for key in base.placement.op_to_component}, "full0.norm": "cpu0"},
        tensor_to_component={**base.placement.tensor_to_component,
            "full0.attention_weights": "hostmem0", "full0.mlp_weights": "hostmem0",
            "embedding_weights": "hostmem0", "lm_head_weights": "hostmem0",
            "kv_cache": "hostmem0", "linear_state": "hostmem0"},
        kv_policy=replace(base.placement.kv_policy, cache_component="hostmem0"))
    base = replace(base, placement=placement)
    cfg = LlamaCppRuntimeConfig(gpu_layers=0, batch=64, ubatch=64, context=256, op_offload=enabled)
    return apply_llama_cuda_op_offload(base, cfg, source_contract=contract() if evidence is None else evidence,
                                       cuda_backend_available=available)


def graph(rows=32, **kwargs):
    return planner.compile_scenario(host_case(rows, **kwargs))


def projection_mains(schedule, name="attention.q"):
    return [t for t in schedule.tasks if t.metadata.get("projection_id") == name
            and t.metadata.get("phase") in {"cpu_gemm", "gpu_gemm"}]


def ancestors(schedule, task):
    tasks = {t.task_id: t for t in schedule.tasks}
    seen, todo = set(), list(task.dependencies)
    while todo:
        key = todo.pop()
        if key not in seen:
            seen.add(key)
            todo.extend(tasks[key].dependencies)
    return seen


@pytest.mark.parametrize("rows,gpu", [(31, False), (32, True), (33, True)])
def test_physical_gemm_m_controls_dispatch(rows, gpu):
    mains = projection_mains(graph(rows))
    assert mains
    assert all(t.metadata["host_gemm_offload_physical_m"] == rows for t in mains)
    assert all(t.metadata["host_gemm_offload_applied"] is gpu for t in mains)
    assert all(t.metadata["execution_component"] == ("gpu0" if gpu else "cpu0") for t in mains)


def test_disabled_or_unavailable_backend_clears_previously_enabled_capability():
    base = host_case()
    assert base.gpu_profile.host_gemm_offload is not None
    for enabled, available in [(False, True), (True, False)]:
        result = apply_llama_cuda_op_offload(base, replace(base.llama_cpp_config, op_offload=enabled),
                    source_contract=contract(), cuda_backend_available=available)
        assert result.gpu_profile.host_gemm_offload is None
        assert result.workload.metadata["llama_cpp_cuda_op_offload"]["status"] == "disabled"
        assert result.placement == base.placement


def test_missing_source_or_unknown_format_cannot_inherit_cuda_capability():
    assert host_case(evidence={}).gpu_profile.host_gemm_offload is None
    evidence = {**contract(), "supported_weight_formats": ("IQ3_S",)}
    mains = projection_mains(graph(evidence=evidence))
    assert mains
    assert all(t.metadata["host_gemm_offload_reason"] == "weight_format_not_source_supported" for t in mains)
    assert all(t.metadata["execution_component"] == "cpu0" for t in mains)


@pytest.mark.parametrize("fmt", ["Q6_K", "IQ3_S", "IQ4_XS"])
def test_source_declared_formats_are_supported_without_model_name(fmt):
    cap = host_case().gpu_profile.host_gemm_offload
    assert cap.supports_workload_format(GemmWorkload(32, 256, 256, packed_weight_formats=(fmt,)))
    assert not cap.supports_workload_format(GemmWorkload(32, 256, 256, packed_weight_formats=("UNKNOWN",)))
    assert not cap.supports_workload_format(GemmWorkload(32, 256, 256))


def test_output_head_dispatch_uses_selected_row_count_not_prompt_length():
    schedule = graph(64)
    heads = [t for t in schedule.tasks if t.metadata.get("event_kind") == "lm_head_projection"
             and t.metadata.get("phase") in {"cpu_gemm", "gpu_gemm"}]
    assert heads
    assert all(t.metadata["gemm_m"] == 1 for t in heads)
    assert all(t.metadata["host_gemm_offload_applied"] is False for t in heads)


def test_staging_is_per_invocation_raw_activation_once_and_kv_stays_host():
    case = host_case()
    static_placement = case.placement
    schedule = planner.compile_scenario(case)
    main = projection_mains(schedule)[0]
    assert main.metadata["placement_component"] == "cpu0"
    assert main.metadata["execution_component"] == "gpu0"
    key = main.metadata["weight_read_invocation_id"]
    transfers = [t for t in schedule.tasks if t.metadata.get("event_kind") == "model_weight_read"
                 and t.metadata.get("weight_read_invocation_id") == key]
    assert transfers
    # A route may contain several segments. Each segment transports the same
    # single payload; no duplicate transfer is introduced per segment.
    routes = [(t.metadata.get("source_component"), t.metadata.get("target_component")) for t in transfers]
    assert len(routes) == len(set(routes))
    assert all(t.metadata["bytes"] == main.metadata["weight_read_bytes"] for t in transfers)
    stages = [t for t in schedule.tasks if t.metadata.get("weight_read_invocation_id") == key
              and t.metadata.get("event_kind", "").startswith("staged_weight_")]
    assert {t.metadata["event_kind"] for t in stages} == {
        "staged_weight_register", "staged_weight_read", "staged_weight_release"}
    assert len({t.metadata["staged_weight_allocation_id"] for t in stages}) == 1
    release = next(t for t in stages if t.metadata["event_kind"] == "staged_weight_release")
    assert main.task_id in ancestors(schedule, release)
    assert all(t.metadata["release_semantics"] == "clean_discard" and not t.demands for t in stages)
    inputs = [t for t in schedule.tasks if t.metadata.get("projection_id") == "attention.q"
              and t.metadata.get("event_kind") == "operator_input_transfer"]
    assert len(inputs) == 1
    assert inputs[0].metadata["bytes"] == 4 * main.metadata["gemm_m"] * 128
    assert case.placement == static_placement
    assert case.placement.kv_policy.cache_component == "hostmem0"
    assert case.placement.tensor_to_component["full0.attention_weights"] == "hostmem0"


def test_repeated_compilation_has_new_invocation_staging_not_a_weight_cache():
    first, second = graph(), graph()
    count = lambda s: sum(t.metadata.get("event_kind") == "staged_weight_register" for t in s.tasks)
    assert count(first) > 0
    assert count(first) == count(second)
    assert sum(t.metadata.get("staged_weight_bytes", 0) for t in first.tasks if t.metadata.get("event_kind") == "staged_weight_register") == sum(
        t.metadata.get("staged_weight_bytes", 0) for t in second.tasks if t.metadata.get("event_kind") == "staged_weight_register")


@pytest.mark.skipif(not (SOURCE / "ggml/src/ggml-cuda/ggml-cuda.cu").is_file(), reason="locked native source absent")
def test_local_source_default_is_conditional_and_override_is_explicit(monkeypatch):
    monkeypatch.setenv("GGML_OP_OFFLOAD_MIN_BATCH", "999")
    default = derive_llama_cuda_op_offload_contract(SOURCE)
    assert default["minimum_m"] == 32
    assert default["environment"]["status"] == "uncaptured_default_assumption"
    assert default["prediction_provenance"] == "conditional_development_assumption"
    assert default["accuracy_validated"] is False
    assert {"IQ3_S", "IQ4_XS"}.issubset(default["supported_weight_formats"])
    assert all(len(value) == 64 for value in default["source_sha256"].values())
    captured = derive_llama_cuda_op_offload_contract(SOURCE, runtime_environment={"GGML_OP_OFFLOAD_MIN_BATCH": "64"})
    assert captured["minimum_m"] == 64
    assert captured["environment"]["status"] == "captured_override"
    absent = derive_llama_cuda_op_offload_contract(SOURCE, runtime_environment={"GGML_OP_OFFLOAD_MIN_BATCH": None})
    assert absent["environment"]["status"] == "captured_absent"
    with pytest.raises(ValueError, match="unsupported GGML"):
        derive_llama_cuda_op_offload_contract(SOURCE, runtime_environment={"GGML_OP_OFFLOAD_MIN_BATCH": "nonsense"})


def test_runtime_false_cannot_be_bypassed_by_an_authored_capability():
    case = host_case()
    case = replace(case, llama_cpp_config=replace(case.llama_cpp_config, op_offload=False))
    mains = projection_mains(planner.compile_scenario(case))
    assert mains
    assert all(t.metadata["host_gemm_offload_reason"] == "runtime_op_offload_disabled" for t in mains)


def test_gpu_layers_with_only_cpu_embedding_do_not_change_storage_or_tasks():
    base = _scenario(_full_layer(_descriptors()), tokens=32, capability=True)
    base = replace(base, placement=replace(base.placement,
        op_to_component={**base.placement.op_to_component, "embedding": "cpu0"}))
    result = apply_llama_cuda_op_offload(base, LlamaCppRuntimeConfig(),
        source_contract=contract(), cuda_backend_available=True)
    assert result.placement == base.placement
    assert result.workload.metadata.get("llama_cpp_f32_hidden_storage") == base.workload.metadata.get("llama_cpp_f32_hidden_storage")
    original, candidate = planner.compile_scenario(base), planner.compile_scenario(result)
    assert [(t.category, t.demands) for t in original.tasks] == [(t.category, t.demands) for t in candidate.tasks]


def test_declared_request_count_does_not_multiply_each_gemm_m():
    case = host_case(8)
    case = replace(case, workload=replace(case.workload, request_count=4))
    mains = projection_mains(planner.compile_scenario(case))
    assert mains
    assert all(t.metadata["host_gemm_offload_physical_m"] == 8 for t in mains)
    assert not any(t.metadata["host_gemm_offload_applied"] for t in mains)


@pytest.mark.skipif(not (SOURCE / "ggml/src/ggml-cuda/ggml-cuda.cu").is_file(), reason="locked native source absent")
def test_matching_scenario_hooks_locked_backend_and_rejects_other_binary():
    from tools.native_llama_compare import build_matching_scenario
    from tests.test_native_hardware_parity import _snapshot
    args = dict(ctx=256, parallel=1, batch=64, ubatch=64, threads=16,
                gpu_layers=0, hardware_snapshot=_snapshot())
    case = build_matching_scenario(32, 2, **args)
    assert case.gpu_profile.host_gemm_offload is not None
    assert case.workload.metadata["llama_cpp_cuda_op_offload"]["environment"]["status"] == "uncaptured_default_assumption"
    assert case.placement.kv_policy.cache_component == "hostmem0"
    unrelated = build_matching_scenario(32, 2, runtime_binary=ROOT / "unrelated-llama-server.exe", **args)
    assert unrelated.gpu_profile.host_gemm_offload is None
    disabled = build_matching_scenario(32, 2, op_offload=False, **args)
    assert disabled.gpu_profile.host_gemm_offload is None


def test_runtime_op_offload_re_materializes_control_plane_before_run_scenario():
    from tools.native_llama_compare import build_matching_scenario
    from tests.test_native_hardware_parity import _snapshot

    case = build_matching_scenario(
        8, 2, ctx=64, parallel=1, batch=8, ubatch=8, threads=16,
        gpu_layers=0, hardware_snapshot=_snapshot(),
    )
    fingerprint = mapping_fingerprint_status(case)
    assert fingerprint["mapping_stale"] is False
    result = run_scenario(case)
    assert result.scenario is case
    assert result.manifest.metadata["control_plane"]["placement_fingerprint"] == fingerprint["input_fingerprint"]
    assert result.serving.request_metrics
