from dataclasses import replace

from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig


def _scenario(prompt_tokens: int, output_tokens: int):
    base = build_llama_default_scenario()
    workload = replace(
        base.workload,
        requests=(
            replace(
                base.workload.requests[0],
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            ),
        ),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
    return replace(base, workload=workload)


def _summary(scenario, threads_batch: int):
    config = LlamaCppRuntimeConfig(
        threads=16,
        threads_batch=threads_batch,
        batch=16,
        ubatch=16,
        context=4096,
        gpu_layers=0,
    )
    return report_dict(run_scenario(replace(scenario, llama_cpp_config=config)))["summary"]


def test_cpu_prefill_uses_threads_batch():
    scenario = _scenario(prompt_tokens=4, output_tokens=1)
    one = _summary(scenario, 1)
    four = _summary(scenario, 4)
    assert four["ttft_ns"]["p50"] < one["ttft_ns"]["p50"]
    assert four["makespan_ns"] < one["makespan_ns"]


def test_single_row_graph_keeps_threads_semantics():
    scenario = _scenario(prompt_tokens=1, output_tokens=1)
    one = _summary(scenario, 1)
    four = _summary(scenario, 4)
    assert four["ttft_ns"]["p50"] == one["ttft_ns"]["p50"]
    assert four["makespan_ns"] == one["makespan_ns"]

