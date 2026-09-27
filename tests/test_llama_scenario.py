from dataclasses import replace

import pytest

from heterollm_sim.llama_scenario import (
    apply_llama_runtime_config,
    llama_cpp_gpu_layer_mapping,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.control_plane_state import mapping_input_fingerprint


def test_llama_ngl_changes_materialized_placement_and_fingerprint():
    scenario = build_reference_scenario()
    cpu = apply_llama_runtime_config(
        scenario, LlamaCppRuntimeConfig(batch=32, ubatch=16, context=256, gpu_layers=0)
    )
    gpu = apply_llama_runtime_config(
        scenario, LlamaCppRuntimeConfig(batch=32, ubatch=16, context=256, gpu_layers=-1)
    )
    assert cpu.llama_cpp_config.gpu_layers == 0
    assert gpu.llama_cpp_config.gpu_layers == -1
    assert mapping_input_fingerprint(cpu) != mapping_input_fingerprint(gpu)
    assert set(cpu.placement.op_to_component.values()) == {"cpu0"}
    assert "gpu0" in set(gpu.placement.op_to_component.values())
    assert cpu.placement.metadata["control_plane"]["evidence"]["llama_cpp_runtime"]["gpu_layers"] == 0


def test_llama_batch_and_context_are_lowered_to_scheduler():
    scenario = build_reference_scenario()
    lowered = apply_llama_runtime_config(
        scenario, LlamaCppRuntimeConfig(batch=32, ubatch=8, context=128, parallel=1)
    )
    assert lowered.workload.scheduler.max_num_batched_tokens == 32
    assert lowered.workload.scheduler.max_num_ubatch_tokens == 8
    assert lowered.workload.scheduler.max_num_seqs == 1
    assert lowered.workload.metadata["llama_cpp_runtime"]["context"] == 128


def test_no_kv_offload_moves_cache_to_host_memory():
    scenario = build_reference_scenario()
    config = LlamaCppRuntimeConfig(batch=32, ubatch=16, context=256,
                                   offload_kqv=False)
    lowered = apply_llama_runtime_config(scenario, config)
    assert lowered.placement.kv_policy.cache_component == "hostmem0"


def test_qwen_nextn_loading_units_are_mapped_only_with_exact_evidence():
    scenario = build_reference_scenario()
    metadata = {
        **scenario.model.graph.attributes.get("metadata", {}),
        "gguf_declared_block_count": scenario.model.num_layers + 1,
        "gguf_imported_executable_layers": scenario.model.num_layers,
        "gguf_mtp_layer_count": 1,
    }
    model = replace(
        scenario.model,
        graph=replace(
            scenario.model.graph,
            attributes={**scenario.model.graph.attributes, "metadata": metadata},
        ),
    )
    mapped = llama_cpp_gpu_layer_mapping(
        replace(scenario, model=model),
        LlamaCppRuntimeConfig(gpu_layers=scenario.model.num_layers + 2),
    )
    assert mapped["simulator_gpu_layers"] == scenario.model.num_layers + 1
    assert mapped["excluded_mtp_loading_units"] == 1
    assert mapped["mapping_applied"] is True

    with pytest.raises(ValueError, match="refusing to clamp"):
        llama_cpp_gpu_layer_mapping(
            scenario,
            LlamaCppRuntimeConfig(gpu_layers=scenario.model.num_layers + 2),
        )


def test_native_builder_serializes_only_single_request_prefill():
    from tools.native_llama_compare import build_matching_scenario

    single = build_matching_scenario(
        8, 1, ctx=256, parallel=1, batch=32, ubatch=16,
        threads=2, gpu_layers=-1,
    )
    batch = build_matching_scenario(
        8, 1, ctx=256, parallel=2, batch=32, ubatch=16,
        threads=2, gpu_layers=-1,
    )
    assert single.workload.metadata["llama_cpp_single_request_prefill_serialized"] is True
    assert batch.workload.metadata["llama_cpp_single_request_prefill_serialized"] is False
