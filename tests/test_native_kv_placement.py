"""Regression checks for llama.cpp parity KV placement.

The native parity helper is intentionally tested at the scenario boundary:
CPU-only runs must use host DRAM for KV pages, while CUDA offload runs use HBM.
The serving runtime must still account for decode reads and appends.
"""
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from native_llama_compare import build_matching_scenario  # noqa: E402
from heterollm_sim.gguf_parity import build_model_from_gguf, read_gguf_metadata  # noqa: E402
from heterollm_sim.serving import simulate_online  # noqa: E402
import heterollm_sim.planner as planner  # noqa: E402
from heterollm_sim.planner import _compile_parallel_request  # noqa: E402


MODEL = ROOT / "artifacts/multimodel_20260913/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _model():
    return build_model_from_gguf(read_gguf_metadata(MODEL))


def _scenario(gpu_layers: int):
    return build_matching_scenario(
        23,
        8,
        ctx=512,
        parallel=1,
        batch=64,
        ubatch=64,
        threads=16,
        gpu_layers=gpu_layers,
        model=_model(),
    )


def test_cpu_only_kv_is_host_backed_and_decode_traffic_is_materialized():
    scenario = _scenario(0)
    assert scenario.placement.kv_policy.cache_component == "hostmem0"
    assert set(scenario.placement.metadata["llama_cpp_kv_layer_components"].values()) == {"hostmem0"}
    result = simulate_online(scenario)
    assert result.kv_metrics.physical_decode_read_bytes > 0
    assert result.kv_metrics.physical_decode_append_bytes > 0


def test_cuda_offload_kv_remains_hbm_backed():
    scenario = _scenario(13)
    assert scenario.placement.kv_policy.cache_component == "hbm0"
    result = simulate_online(scenario)
    assert result.kv_metrics.physical_decode_read_bytes > 0
    assert result.kv_metrics.physical_decode_append_bytes > 0


def test_full_offload_kv_map_is_gpu_backed_for_every_layer():
    scenario = _scenario(-1)
    layer_map = scenario.placement.metadata["llama_cpp_kv_layer_components"]
    assert len(layer_map) == scenario.model.num_layers
    assert set(layer_map.values()) == {"hbm0"}


def test_partial_offload_binds_kv_owner_per_layer_and_decode_events():
    scenario = _scenario(13)
    layer_map = scenario.placement.metadata["llama_cpp_kv_layer_components"]
    assert layer_map["layer-000"] == "hostmem0"
    assert layer_map["layer-010"] == "hostmem0"
    assert layer_map["layer-011"] == "hostmem0"
    assert layer_map["layer-012"] == "hbm0"
    assert layer_map["layer-023"] == "hbm0"
    tasks = _compile_parallel_request(scenario, scenario.workload.requests[0])
    kv = [
        task for task in tasks
        if task.metadata.get("event_kind") in {"kv_read", "kv_append"}
    ]
    assert kv
    for task in kv:
        assert task.metadata["kv_cache_component"] == layer_map[task.metadata["layer_id"]]


def test_qwen38_cpu_gguf_preserves_physical_ffn_gate_up_order():
    """The real Qwen3.8 GGUF lowers gate/up as two CPU MMQ calls.

    This is a semantic guard: the physical projection split must come from
    the GGUF descriptors and the declared llama.cpp capability, rather than
    from a model-name latency adjustment.  The second call depends on the
    first, matching the observed llama.cpp graph submission order.
    """
    model_path = ROOT / "artifacts/multimodel_20260913/models/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf"
    model = build_model_from_gguf(read_gguf_metadata(model_path))
    scenario = build_matching_scenario(
        8, 4, ctx=2048, parallel=1, batch=64, ubatch=64, threads=16,
        gpu_layers=0, model=model,
    )
    schedule = planner.compile_scenario(scenario)
    layer_tasks = [
        task for task in schedule.tasks
        if task.name.startswith("prefill.layer-000")
        and task.metadata.get("phase") == "cpu_gemm"
        and task.metadata.get("projection_id") in {"mlp.gate", "mlp.up", "mlp.up_gate"}
    ]
    assert [task.metadata["projection_id"] for task in layer_tasks] == ["mlp.gate", "mlp.up"]
    assert [task.metadata.get("physical_projection") for task in layer_tasks] == ["gate", "up"]
    up_weight_read = next(
        task for task in schedule.tasks
        if task.name.startswith("prefill.layer-000.rank000.mlp_up.model_weight_read")
    )
    assert layer_tasks[0].task_id in up_weight_read.dependencies
    assert all(not task.metadata.get("fusion_enabled", False) for task in layer_tasks)
