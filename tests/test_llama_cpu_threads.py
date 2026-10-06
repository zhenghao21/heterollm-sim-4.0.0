from dataclasses import replace

from heterollm_sim.planner import _cpu_profiles
from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


def _scenario(threads_batch: int):
    base = build_llama_default_scenario()
    return replace(
        base,
        llama_cpp_config=LlamaCppRuntimeConfig(
            threads=16,
            threads_batch=threads_batch,
            batch=16,
            ubatch=16,
            context=4096,
            gpu_layers=0,
        ),
    )


def test_llama_threads_batch_only_changes_multirow_graphs():
    single = _cpu_profiles(_scenario(1), graph_rows=1)[0].pipeline.core_count
    batched = _cpu_profiles(_scenario(1), graph_rows=4)[0].pipeline.core_count
    assert single == 16
    assert batched == 1

    # A one-row graph always follows n_threads, regardless of n_threads_batch.
    assert _cpu_profiles(_scenario(64), graph_rows=1)[0].pipeline.core_count == 16
    # Positive values are capped by the hardware's available cores.
    assert _cpu_profiles(_scenario(64), graph_rows=4)[0].pipeline.core_count == 16


def test_llama_threads_batch_minus_one_selects_available_cores():
    profile = _cpu_profiles(_scenario(-1), graph_rows=4)[0]
    assert profile.pipeline.core_count == 16
