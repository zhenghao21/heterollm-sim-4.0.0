"""Actual physical feedback and queue semantics, without calibration claims."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from heterollm_sim.data_motion import PhysicalRuntimeContext, _PhysicalRuntime
from heterollm_sim.memory_types import (AccessRequest, Operation,
    NandWearConfig, make_hbf_config, parse_physical_memory_config)
from heterollm_sim.nand_core import NandCore, summarize_nand_observations
from heterollm_sim.planner import _summarize_nand_task_traffic
from heterollm_sim.reporting import _sum_batch_storage_traffic
from hbf_hbm_wear_study import run_controlled_trace, run_serving_feedback

PAGE = 4096


def config(**kwargs):
    values = dict(channels=1, targets_per_channel=1, dies_per_target=1,
        luns_per_die=1, planes_per_lun=4, planes_independent=True,
        blocks_per_plane=2, pages_per_block=256, page_bytes=PAGE,
        host_granularity_bytes=1, host_bandwidth_gb_s=100., host_read_bandwidth_gb_s=100., host_write_bandwidth_gb_s=100., internal_bandwidth_gb_s=100.,
        front_ns=0., page_read_ns=100., page_program_ns=200., block_erase_ns=300.)
    values.update(kwargs)
    return make_hbf_config(**values)


def event(result):
    return result.counters["nand_observations"]["events"][-1]


def test_disabled_feedback_and_diagnostic_only_preserve_detailed_timing():
    off = NandCore(config())
    diagnostic = NandCore(config(detailed_observability=True))
    for i, op in enumerate((Operation.READ, Operation.WRITE, Operation.ERASE, Operation.READ)):
        request = AccessRequest(str(i), op, i*PAGE, PAGE)
        a, b = off.submit(request), diagnostic.submit(request)
        assert a.completion_ns == b.completion_ns
        assert a.stages == b.stages
        assert "nand_observations" not in a.counters
        assert b.counters["nand_observations"]["events"]
    assert not off._block_wear_state


def test_physical_planes_are_not_shared_effective_slots_and_queue_is_exact():
    core = NandCore(config(detailed_observability=True, parallel_units=1))
    rows = [core.submit(AccessRequest(str(i), Operation.READ, i*PAGE, PAGE)) for i in range(2)]
    observed = summarize_nand_observations(r.counters["nand_observations"] for r in rows)
    assert len({event(r)["execution_resource"] for r in rows}) == 1
    assert len({event(r)["plane_id"] for r in rows}) == 2
    assert event(rows[0])["start_ns"] == 0
    assert event(rows[1])["start_ns"] == 100
    queue = observed["read_queue_depth"]
    assert queue["peak"] == 1
    assert queue["total_page_wait_ns"] == 100
    assert queue["mean_time_weighted"] == .5
    assert queue["depth_duration_ns"] == {0:100.,1:100.}
    assert observed["physical_plane_imbalance"]["idle_planes"] == 2
    assert observed["physical_plane_imbalance"]["max_over_mean"] == 2


def test_program_queue_begins_at_internal_data_arrival_not_host_admission():
    core = NandCore(config(detailed_observability=True, parallel_units=1))
    rows = [core.submit(AccessRequest(str(i), Operation.WRITE, i*PAGE, PAGE)) for i in range(2)]
    a, b = map(event, rows)
    assert a["eligible_ns"] == 2*PAGE/100.
    assert a["start_ns"] == a["eligible_ns"]
    assert b["eligible_ns"] == 3*PAGE/100.
    assert b["start_ns"] == a["end_ns"]
    observed = summarize_nand_observations(r.counters["nand_observations"] for r in rows)
    assert observed["program_queue_depth"]["total_page_wait_ns"] == pytest.approx(200.-PAGE/100.)


def test_erase_advances_actual_block_pe_and_later_program_latency():
    core = NandCore(config(detailed_observability=True, wear_model=NandWearConfig(
        rated_pe_cycles=2, initial_pe_cycles=0, program_latency_growth_at_rated_pe=1.,
        max_program_latency_scale=2., assumption="synthetic test")))
    first = core.submit(AccessRequest("first",Operation.WRITE,0,PAGE))
    again = core.submit(AccessRequest("again",Operation.WRITE,0,PAGE,first.completion_ns))
    assert event(again)["pe_cycles_after"] == 0
    erased = core.submit(AccessRequest("erase",Operation.ERASE,0,PAGE,again.completion_ns))
    later = core.submit(AccessRequest("later",Operation.WRITE,0,PAGE,erased.completion_ns))
    assert event(erased)["pe_cycles_after"] == 1
    assert event(later)["end_ns"]-event(later)["start_ns"] == 300.
    assert event(later)["block_programs_after"] == 3
    assert event(later)["block_erases_after"] == 1
    assert event(later)["wear_extra_ns"] == 100.


def test_runtime_rollback_restores_erase_state():
    wear = NandWearConfig(assumption="test")
    runtime = PhysicalRuntimeContext()
    core = NandCore(config(wear_model=wear),runtime.timeline)
    runtime.runtimes["hbf"] = _PhysicalRuntime(core)
    core.submit(AccessRequest("before",Operation.READ,0,PAGE))
    snapshot = runtime.snapshot()
    first = core.submit(AccessRequest("erase",Operation.ERASE,0,PAGE,100000.))
    runtime.restore(snapshot)
    second = core.submit(AccessRequest("erase",Operation.ERASE,0,PAGE,100000.))
    assert first.completion_ns == second.completion_ns
    assert event(first) == event(second)
    assert next(iter(core._block_wear_state.values()))["erases"] == 1


def test_typed_feedback_roundtrip_and_wear_validation():
    parsed = parse_physical_memory_config({"kind":"HBF",
        "wear_model":{"assumption":"test"}})
    assert isinstance(parsed.wear_model,NandWearConfig)
    for values in ({"rated_pe_cycles":True},{"initial_pe_cycles":-.1},
                   {"max_program_latency_scale":.5},{"program_latency_growth_at_rated_pe":float("inf")}):
        with pytest.raises(ValueError):
            NandWearConfig(assumption="test",**values)


def test_partial_physical_observations_remain_partial_after_batch_aggregation():
    def task(name, recorded):
        row = NandCore(config(detailed_observability=recorded)).submit(
            AccessRequest(name,Operation.READ,0,PAGE))
        return SimpleNamespace(task_id=name,demands=(),metadata={"physical_execution_by_owner":{
            "hbf0":{"kind":"HBF",**row.counters}}})
    ledger = _summarize_nand_task_traffic([task("yes",True),task("no",False)])
    result = SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=[SimpleNamespace(cost=SimpleNamespace(
        metadata={"storage_traffic":ledger}))]))
    output = _sum_batch_storage_traffic(result)
    assert output["metric_availability"]["nand_read_queue_depth"]["status"] == "partial"


def test_controlled_trace_conserves_work_and_exercises_online_feedback():
    baseline = run_controlled_trace(wear_enabled=False,cycles=2)
    feedback = run_controlled_trace(cycles=2)
    assert baseline["physical_read_bytes"] == feedback["physical_read_bytes"] == 64*PAGE
    assert baseline["physical_write_bytes"] == feedback["physical_write_bytes"] == 8*PAGE
    assert feedback["duration_ns"] > baseline["duration_ns"]
    observed = feedback["observations"]
    assert observed["actual_erase_operations"] == observed["actual_program_operations"] == 8
    assert observed["read_queue_depth"]["peak"] > 0
    assert observed["program_queue_depth"]["peak"] > 0


def test_full_serving_wear_changes_actual_request_metrics():
    a0,a1,b0,bw = run_serving_feedback()
    assert a0["node_tps"] == a1["node_tps"]
    for row in (b0,bw):
        assert row["completed_requests"] == 2
        assert row["visible_output_tokens"] == 16
        storage = row["resources"]["storage_traffic"]
        observed = storage["nand_observations"]
        assert sum(e["physical_bytes"] for e in observed["events"] if e["operation"]=="read") == storage["physical_read_bytes"]
        assert sum(e["physical_bytes"] for e in observed["events"] if e["operation"]=="program") == storage["physical_write_bytes"]
        assert observed["actual_erase_operations"] == 0
        assert storage["metric_availability"]["nand_read_queue_depth"]["status"] == "recorded"
    for row in (bw,):
        assert row["node_tps"] < b0["node_tps"]
        assert row["ttft_ns"]["p50"] > b0["ttft_ns"]["p50"]
        assert row["tpot_ns"]["p50"] > b0["tpot_ns"]["p50"]
    assert bw["memory_metrics"]["dynamic_wear"]["wear_extra_ns"] > 0
