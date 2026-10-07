from dataclasses import replace
import pytest

from heterollm_sim.llama_scenario import apply_llama_runtime_config, llama_cpp_kv_layer_mapping
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


def test_llama_kv_mapping_requires_explicit_owner_without_rank_memory():
    """An arbitrary HBM cannot stand in for an unresolved rank/KV target."""

    source = build_llama_default_scenario()
    placement = replace(
        source.placement,
        parallel=replace(source.placement.parallel, rank_mapping=()),
        kv_policy=replace(source.placement.kv_policy, cache_component=None),
    )
    scenario = replace(
        source,
        placement=placement,
        llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=-1),
    )

    with pytest.raises(ValueError, match="requires rank memory or an explicit KV cache component"):
        llama_cpp_kv_layer_mapping(scenario, scenario.llama_cpp_config)


def test_llama_kv_mapping_rejects_out_of_range_main_gpu():
    scenario = build_llama_default_scenario()
    config = LlamaCppRuntimeConfig(gpu_layers=-1, split_mode="row", main_gpu=1)
    with pytest.raises(ValueError, match="main_gpu 1 has no matching GPU memory rank"):
        llama_cpp_kv_layer_mapping(scenario, config)


def test_llama_tiering_rejects_out_of_range_main_gpu_before_deferred_kv_owner():
    source = build_llama_default_scenario()
    placement = replace(
        source.placement,
        parallel=replace(source.placement.parallel, rank_mapping=()),
        kv_policy=replace(source.placement.kv_policy, cache_component=None),
    )
    config = LlamaCppRuntimeConfig(
        gpu_layers=-1, split_mode="row", main_gpu=1, device_memory_tiering=True
    )
    scenario = replace(source, placement=placement, llama_cpp_config=config)
    with pytest.raises(ValueError, match="main_gpu 1 is outside 1 available GPUs"):
        apply_llama_runtime_config(scenario, config)
