"""The final RMSNorm consumes initialized hidden data and its own statistics."""
from dataclasses import replace
import pytest

from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.planner import compile_scenario
from test_hbf_host_staging_integration import strict_staging_scenario


@pytest.mark.parametrize("prompt", [1, 31])
def test_final_norm_stats_do_not_alias_output_and_consumers_read_written_bytes(prompt):
    configured = strict_staging_scenario(all_hbf=True, prompt=prompt, output=2, capacity=65536)
    tasks = compile_scenario(configured).tasks
    reduce = next(task for task in tasks
                  if task.name == "prefill.rank000.final_norm_reduce.gpu_reduction")
    apply = next(task for task in tasks
                 if task.name == "prefill.rank000.final_norm_apply.gpu_elementwise")
    head = next(task for task in tasks if task.name == "prefill.rank000.lm_head.gpu_gemm")
    source = next(row for row in reduce.metadata["memory_accesses"] if row["operation"] == "read")
    statistic = next(row for row in reduce.metadata["memory_accesses"] if row["operation"] == "write")
    output = next(row for row in apply.metadata["memory_accesses"] if row["operation"] == "write")
    assert statistic["buffer_id"] != output["buffer_id"]
    assert next(row for row in head.metadata["memory_accesses"]
                if row["operation"] == "read")["buffer_id"] == output["buffer_id"]
    assert source["buffer_id"].startswith("tensor:block-group-000.output:")
    assert source["offset_bytes"] == (prompt - 1) * 256
    assert source["allocation_size_bytes"] == prompt * 256

    # Seed a real producer write, then run the physical reducer and consumer.
    # Unlike a DRAM-only identity check, strict HBF rejects unwritten bytes.
    producer = replace(reduce, metadata={**reduce.metadata, "memory_accesses": (
        {**source, "operation": "write", "offset_bytes": 0,
         "byte_count": source["allocation_size_bytes"]},)})
    runtime = PhysicalRuntimeContext()
    now = 0.0
    for task in (producer, reduce, apply, head):
        realized = resolve_physical_task(task, runtime, now)
        now = realized.metadata["physical_completion_ns"]
    assert now > 0
