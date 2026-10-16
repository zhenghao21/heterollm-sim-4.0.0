"""Qwen's separate RoPE consumes a generated table on either memory tier."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b200_hbm_hbf_scan_config import build_scan_scenario
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.planner import compile_scenario


@pytest.fixture(scope="module")
def qwen_plans():
    # Compile real Qwen geometry; only selected small physical tasks execute
    # below. This is a dataflow regression, not a shortened benchmark result.
    return {medium: compile_scenario(build_scan_scenario(
        "qwen2_5-0_5b", medium, 2048, 3)) for medium in ("HBM", "HBF")}


def rope_pair(plan, phase):
    prefix = phase + ".layer-000.rank000."
    return tuple(next(task for task in plan.tasks if task.name == prefix + suffix)
                 for suffix in ("rope_table_generate.gpu_elementwise", "rope.gpu_elementwise"))


def depends_on(plan, consumer, producer):
    tasks = {task.task_id: task for task in plan.tasks}
    pending, visited = list(consumer.dependencies), set()
    while pending:
        identity = pending.pop()
        if identity == producer.task_id:
            return True
        if identity not in visited:
            visited.add(identity)
            pending.extend(tasks[identity].dependencies)
    return False


@pytest.mark.parametrize("medium", ("HBM", "HBF"))
def test_real_qwen_prefill_generates_complete_rope_table_before_consumer(qwen_plans, medium):
    plan = qwen_plans[medium]
    producer, consumer = rope_pair(plan, "prefill")
    assert consumer.metadata["fusion_decision"] == "explicit_qkv_bias_boundary"
    assert producer.metadata["target_component"] == consumer.metadata["target_component"] == "gpu0"
    assert depends_on(plan, consumer, producer)
    writes = producer.metadata["memory_accesses"]
    assert len(writes) == 1 and writes[0]["operation"] == "write"
    table = writes[0]
    # BF16 expanded Q+K tables: 2048 tokens * (896+128) * sin/cos * 2 B.
    expected_bytes = 2048 * (896 + 128) * 2 * 2
    assert table["byte_count"] == table["allocation_size_bytes"] == expected_bytes
    assert table["offset_bytes"] == 0
    reads = [row for row in consumer.metadata["memory_accesses"]
             if row["buffer_id"] == table["buffer_id"]]
    assert len(reads) == 1 and reads[0]["operation"] == "read"
    assert reads[0]["allocation_size_bytes"] == expected_bytes
    assert reads[0]["offset_bytes"] + reads[0]["byte_count"] <= table["byte_count"]
    logical_reads = [row for row in consumer.metadata["logical_buffer_accesses"]
                     if row["buffer_id"] == table["buffer_id"]]
    assert logical_reads[0]["size_bytes"] == expected_bytes
    assert producer.metadata["cost_model"]["special_function_operations"] == expected_bytes // 2
    assert all(row[1] != table["buffer_id"] for row in producer.metadata.get("hbf_preloaded_buffers", ()))


def test_two_decode_rope_tables_require_and_consume_real_hbf_writes(qwen_plans):
    runtime = PhysicalRuntimeContext(capture_details=False)
    now, identities = 0.0, []
    for phase in ("decode0001", "decode0002"):
        producer, consumer = rope_pair(qwen_plans["HBF"], phase)
        table = producer.metadata["memory_accesses"][0]
        identities.append(table["buffer_id"])
        qkv = next(row for row in consumer.metadata["memory_accesses"]
                   if ".activation.qkv:" in row["buffer_id"])
        # Provide the actual QKV input with a physical producer write. Keep
        # the table uninitialized until its real compiled producer runs.
        seed = replace(consumer, metadata={**consumer.metadata, "memory_accesses": (
            {**qkv, "operation": "write", "offset_bytes": 0,
             "byte_count": qkv["allocation_size_bytes"]},)})
        realized = resolve_physical_task(seed, runtime, now)
        now = realized.metadata["physical_completion_ns"]
        with pytest.raises(ValueError, match="uninitialized"):
            resolve_physical_task(consumer, runtime, now)
        generated = resolve_physical_task(producer, runtime, now)
        now = generated.metadata["physical_completion_ns"]
        media = generated.metadata["physical_execution_by_owner"]["hbf0.memory"]
        assert media["pages_programmed"] == 1
        assert media["logical_write_bytes"] == table["byte_count"] == 4096
        consumed = resolve_physical_task(consumer, runtime, now)
        now = consumed.metadata["physical_completion_ns"]
        assert consumed.metadata["physical_execution_by_owner"]["hbf0.memory"]["logical_read_bytes"] == 1024
        assert (table["buffer_id"], table["allocation_generation"]) not in runtime.hbf_staging["hbf0.memory"]["preloaded"]
    assert len(set(identities)) == 2
    assert now > 0
