from dataclasses import replace

from heterollm_sim.llama_scenario import llama_cpp_kv_layer_mapping
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


def test_llama_kv_mapping_selects_writable_memory_without_derived_targets():
    """A fresh UI payload may clear rank/KV targets before control-plane mapping."""

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

    mapping = llama_cpp_kv_layer_mapping(scenario, scenario.llama_cpp_config)

    assert mapping["kv_layer_components"]
    assert set(mapping["kv_layer_components"].values()) == {"hbm0"}
