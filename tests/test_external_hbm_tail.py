"""Native HBF external-page staging charges its HBM tail mirror."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_native_config import build_native_comparison_scenario
from exploring_hbf_native_pilot import build_pilot_scenario
from heterollm_sim.engine import simulate_schedule
from heterollm_sim.planner import compile_serving_cohort_schedule, native_kv_ingress_inventory
from heterollm_sim.reporting import run_scenario
from heterollm_sim.serving import BatchCohort


def _external_hbf_scenario(*, output_override=2):
    return build_native_comparison_scenario(
        "small-functional", "HBF", "SHORT", tp=1, batch_size=1,
        prompt_override=15, output_override=output_override, warmup_override=0,
        measurement_override=0, model_override=build_pilot_scenario("hbm_tp1").model,
    )


def _hbm_accesses(task):
    return tuple(row for row in task.metadata.get("memory_accesses", ())
                 if str(row.get("physical_owner", "")).startswith("hbm0."))


def test_external_hbm_tail_inventory_reserves_two_pages_per_attention_layer():
    scenario = _external_hbf_scenario()
    request = scenario.workload.requests[0]
    inventory = native_kv_ingress_inventory(
        scenario, request.request_id, request.prompt_tokens - 1, ep_group=0,
    )
    slots = inventory["tail_slots"]
    assert len(slots) == 4  # two layers, K and V
    assert {slot["part"] for slot in slots} == {"k", "v"}
    assert all(slot["page_bytes"] == 4096 for slot in slots)
    hbm_owner = slots[0]["hbm_physical_owner"]
    assert hbm_owner in inventory["physical_memory_configs"]
    hbm_rows = [row for row in inventory["physical_allocations"]
                if row["physical_owner"] == hbm_owner
                and ":hbm_tail:" in row["buffer_id"]]
    assert len(hbm_rows) == len(slots)
    assert all(row["size_bytes"] == 4096 for row in hbm_rows)


def test_external_hbm_tail_reads_and_flush_have_native_dependencies():
    scenario = _external_hbf_scenario()
    result = run_scenario(scenario)
    ingress = result.serving.external_kv_ingress_metrics
    assert ingress["cost_ledger"]["owners"]["hbm0.hbm_fabric"]["physical_write_bytes"] > 0
    decode_batch = next(batch for batch in result.serving.batches if batch.kind == "decode")
    schedule = compile_serving_cohort_schedule(
        scenario, BatchCohort("tail", "decode", decode_batch.start_ns, decode_batch.items),
    )
    flush_reads = [task for task in schedule.tasks
                   if task.metadata.get("event_kind") == "kv_tail_flush_read"]
    assert flush_reads
    assert all(any(row.get("operation") == "read" and row.get("byte_count") == 4096
                   for row in _hbm_accesses(task)) for task in flush_reads)
    materialize_writes = [task for task in schedule.tasks
                          if task.metadata.get("event_kind") == "kv_materialize"
                          and any(row.get("operation") == "write" for row in _hbm_accesses(task))]
    assert materialize_writes
    attention = [task for task in schedule.tasks
                 if task.metadata.get("event_kind") == "fused_attention"
                 and _hbm_accesses(task)]
    assert attention
    assert all(any(row.get("operation") == "read" for row in _hbm_accesses(task))
               for task in attention)
    flush_ids = {task.task_id for task in schedule.tasks
                 if task.metadata.get("kv_tail_flush_role") == "kv_tail_flush_write"}
    assert flush_ids
    # The attention task consumes the HBM tail in parallel with the NAND read;
    # it is not ordered behind the page-program branch.
    assert all(flush_id not in task.dependencies for task in attention for flush_id in flush_ids)
    attention_ids = {candidate.task_id for candidate in attention}
    for task in flush_reads:
        assert any(parent in attention_ids for parent in task.dependencies)


def test_external_hbm_tail_waits_for_last_nand_read_or_kv_fallback():
    # Use a decode item whose append completes a NAND page.  Partial external
    # tails are intentionally read from their HBM mirror; when this small
    # scenario has no separate NAND read, the actual fused KV consumer is the
    # planner's documented fallback anchor.
    base_scenario = _external_hbf_scenario(output_override=20)
    base_result = run_scenario(base_scenario)
    decode_batch = next(
        batch for batch in base_result.serving.batches
        if batch.kind == "decode"
        and any(item.context_tokens % 16 == 15 for item in batch.items)
    )
    item = next(item for item in decode_batch.items if item.context_tokens % 16 == 15)
    schedule = compile_serving_cohort_schedule(
        base_scenario, BatchCohort("tail-after-attention", "decode",
                              decode_batch.start_ns, (item,)),
    )
    flush_reads = [task for task in schedule.tasks
                   if task.metadata.get("event_kind") == "kv_tail_flush_read"]
    assert flush_reads
    attention_flash = {task.task_id for task in schedule.tasks
                       if task.metadata.get("event_kind") == "fused_attention"
                       and ".attention_flash" in task.task_id}
    assert attention_flash
    assert all(attention_flash.intersection(task.dependencies) for task in flush_reads)
    trace = simulate_schedule(schedule)
    events = {task.task_id: task for task in trace.tasks}
    for flush in flush_reads:
        attention_end = max(events[parent].end_ns for parent in flush.dependencies
                            if parent in attention_flash)
        assert events[flush.task_id].start_ns >= attention_end
    flush_writes = [task for task in schedule.tasks
                    if task.metadata.get("kv_tail_flush_role") == "kv_tail_flush_write"]
    assert flush_writes
    assert all(any(parent in {read.task_id for read in flush_reads}
                   for parent in task.dependencies) for task in flush_writes)
