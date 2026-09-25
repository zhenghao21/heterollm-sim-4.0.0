from dataclasses import replace

from heterollm_sim.llama_scenario import apply_llama_runtime_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.serving import compile_serving_plan, simulate_online


def test_legacy_json_keeps_single_component_contract():
    scenario = build_reference_scenario()
    plan = compile_serving_plan(scenario)
    assert plan.kv_policy.layout_mode == "legacy_single"
    assert plan.kv_layer_components == {}
    assert plan.kv_capacity_bytes_by_component == {"hbm0": plan.kv_policy.capacity_bytes}


def test_runtime_adapter_exposes_native_layer_kv_contract():
    scenario = apply_llama_runtime_config(
        build_reference_scenario(),
        LlamaCppRuntimeConfig(
            batch=32,
            ubatch=16,
            context=256,
            gpu_layers=-1,
            offload_kqv=False,
        ),
    )
    # The reference model contains MTP scratch state, which static layer KV
    # intentionally rejects until that independent capacity is modeled.
    assert scenario.placement.kv_policy.layout_mode == "llama_static_layer"
    assert scenario.placement.metadata["llama_cpp_kv_layer_components"]
    assert scenario.workload.metadata["llama_cpp_kv_capacity_contract"][
        "logical_context_tokens"
    ] == 256


def test_non_unified_pool_is_capped_by_context_times_slots():
    scenario = build_reference_scenario()
    workload = replace(
        scenario.workload,
        mtp=None,
        metadata={
            **scenario.workload.metadata,
            "llama_cpp_kv_capacity_contract": {
                "logical_context_tokens": 64,
                "n_seq_max": 2,
                "kv_unified": False,
            },
        },
        scheduler=replace(scenario.workload.scheduler, max_num_seqs=2),
    )
    policy = replace(
        scenario.placement.kv_policy,
        kv_unified=False,
        tokens_per_page=16,
        offload_component=None,
    )
    scenario = replace(
        scenario,
        workload=workload,
        placement=replace(scenario.placement, kv_policy=policy),
    )
    plan = compile_serving_plan(scenario)
    assert plan.kv_policy.logical_context_tokens == 64
    assert plan.kv_policy.n_seq_max == 2
    assert plan.kv_policy.capacity_pages <= 8
    result = simulate_online(scenario)
    assert result.kv_metrics.kv_unified is False


def test_paged_pool_uses_multiple_physical_components_without_summing_pages():
    scenario = build_reference_scenario()
    policy = replace(
        scenario.placement.kv_policy,
        layout_mode="paged_pool",
        pool_components=("hbm0", "hbm1"),
        cache_component="hbm0",
        offload_component=None,
        tokens_per_page=4,
    )
    scenario = replace(
        scenario,
        placement=replace(scenario.placement, kv_policy=policy),
    )
    plan = compile_serving_plan(scenario)
    result = simulate_online(scenario)
    assert plan.kv_policy.layout_mode == "paged_pool"
    assert set(plan.kv_policy.capacity_bytes_by_component) == {"hbm0", "hbm1"}
    assert result.kv_metrics.layout_mode == "paged_pool"
    assert result.kv_metrics.peak_used_pages > 0
