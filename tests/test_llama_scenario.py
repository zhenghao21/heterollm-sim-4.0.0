from dataclasses import replace

from heterollm_sim.llama_scenario import apply_llama_runtime_config
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
