"""Qwen's separate RoPE consumes a generated table."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b200_hbm_hbf_scan_config import build_scan_scenario
from heterollm_sim.planner import compile_scenario


@pytest.fixture(scope="module")
def qwen_plan():
    # Compile real Qwen geometry. This is a dataflow regression, not a
    # shortened benchmark result.
    return compile_scenario(build_scan_scenario("qwen2_5-0_5b", "HBM", 2048, 3))


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


def test_real_qwen_prefill_generates_complete_rope_table_before_consumer(qwen_plan):
    plan = qwen_plan
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
