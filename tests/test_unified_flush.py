from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
from heterollm_sim.memory_transfer import ResourceTimeline
from heterollm_sim.nand_core import NandCore


PAGE = 4096


def _logic_config():
    return NandConfig(
        kind="HBF", channels=8, planes_per_lun=2, planes_independent=True,
        page_bytes=PAGE, host_granularity_bytes=PAGE,
        host_bandwidth_gb_s=100.0, host_read_bandwidth_gb_s=100.0,
        host_write_bandwidth_gb_s=100.0, internal_bandwidth_gb_s=10.0,
        page_read_ns=1000.0, page_program_ns=100000.0, blocks_per_plane=64,
        pages_per_block=64, read_cache_pages_per_bank=2, host_channels=4,
        host_full_duplex=True, max_outstanding_requests=64,
        capacity_bytes=8 * 2 * 64 * 64 * PAGE,
        page_staging="logic_die", page_staging_bytes_per_host_channel=1 << 20,
        page_staging_latency_ns=1.0,
    )


def _core(numeric):
    core = NandCore(_logic_config(), capture_details=not numeric)
    core._nand_numeric_enabled = numeric
    return core


def _write(core, name, address, size, arrival):
    return core.submit(AccessRequest(name, Operation.WRITE, address, size, arrival))


def _nand_read_tasks(schedule, layer_id, rank):
    result = []
    for task in schedule.tasks:
        metadata = task.metadata
        if metadata.get("layer_id") != layer_id or metadata.get("rank") != rank:
            continue
        configs = metadata.get("physical_memory_configs", {})
        accesses = metadata.get("memory_accesses", ())
        for access in accesses:
            if str(access.get("operation", "")).casefold() != "read":
                continue
            owner = str(access.get("physical_owner", metadata.get("physical_owner", "")))
            if isinstance(configs.get(owner), NandConfig):
                result.append(task)
                break
    return result


@pytest.mark.parametrize("numeric", [False, True])
def test_logic_die_flush_uses_scope_and_plane_frontier(numeric):
    core = _core(numeric)
    core.set_staging_scope("dense0:rank000")
    for index in range(4):
        _write(core, str(index), 10 * PAGE + index * 1024, 1024, index * 1000.0)

    before = core.page_staging_state()
    assert before["deferred_pages"] == 1
    account = core.flush_staged_pages("dense0:rank000", 5000.0)
    assert account["programs"] == 1
    assert min(start for resource, (start, _end) in account["resource_last_intervals"].items()
               if ":array:" in resource or resource.startswith("nand:array")) >= 5000.0
    assert min(start for resource, (start, _end) in account["resource_last_intervals"].items()
               if ":channel:" in resource or resource.startswith("nand:channel")) >= 5000.0
    after = core.page_staging_state()
    assert after["deferred_pages"] == 0
    assert after["started_programs"] == 1
    assert core.flush_staged_pages("dense0:rank000", 200000.0)["programs"] == 0


def test_logic_die_flush_scope_and_counters_roll_back_with_transaction():
    core = _core(False)
    core.set_staging_scope("dense0:rank000")
    for index in range(3):
        _write(core, str(index), 10 * PAGE + index * 1024, 1024, index * 1000.0)
    before = core.page_staging_state()
    token = core.begin_transaction()
    _write(core, "last", 10 * PAGE + 3 * 1024, 1024, 4000.0)
    assert core.page_staging_state()["deferred_pages"] == 1
    core.rollback_transaction(token)
    assert core.page_staging_state() == before


def test_logic_die_without_attention_scope_programs_immediately():
    core = _core(False)
    core.set_staging_scope(None)
    rows = [_write(core, str(index), 10 * PAGE + index * 1024, 1024, index * 1000.0)
            for index in range(4)]
    assert rows[-1].pages_programmed == 1
    state = core.page_staging_state()
    assert state["deferred_pages"] == 0
    assert state["started_programs"] == 1


def test_logic_die_read_batch_reports_inherited_array_collision_wait():
    timeline = ResourceTimeline(lane_available={"nand:array:0": [50000.0]})
    core = NandCore(_logic_config(), timeline=timeline, capture_details=False)
    core._nand_numeric_enabled = True
    rows = core.submit_many((
        AccessRequest("collision", Operation.READ, 0, PAGE, 0.0),
        AccessRequest("other", Operation.READ, 2 * PAGE, PAGE, 0.0),
    ))
    assert len(rows) == 2
    assert core._last_media_queue_wait_ns == pytest.approx(50000.0)


def test_logic_die_read_batch_excludes_same_batch_array_queue_from_media_wait():
    core = NandCore(_logic_config(), timeline=ResourceTimeline(), capture_details=False)
    core._nand_numeric_enabled = True
    rows = core.submit_many((
        AccessRequest("first", Operation.READ, 0, PAGE, 0.0),
        # Sixteen pages returns to the same plane/array with a new physical page.
        AccessRequest("same-array", Operation.READ, 16 * PAGE, PAGE, 0.0),
    ))
    assert len(rows) == 2
    assert rows[1].completion_ns > rows[0].completion_ns
    assert core._last_media_queue_wait_ns == pytest.approx(0.0)


def test_scalar_read_wait_ignores_this_requests_own_use_of_a_shared_slot():
    # Two arrays share one execution slot, so the second page sees the slot
    # after the first page reserved it. Only the earlier work is inherited.
    from dataclasses import replace

    timeline = ResourceTimeline(lane_available={"nand:parallel:0": [50000.0]})
    core = NandCore(replace(_logic_config(), page_staging="none", parallel_units=1),
                    timeline=timeline, capture_details=False)
    arrival = 0.0
    result = core._execute_detailed_accepted(AccessRequest("two-arrays", Operation.READ, 0, 2 * PAGE, arrival))
    assert result.pages_read == 2
    assert result.counters["media_queue_wait_ns"] == pytest.approx(50000.0 - core.config.front_ns)


def test_logic_die_flush_task_is_background_after_last_nand_read():
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.planner import compile_serving_cohort_schedule
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.serving import BatchCohort

    scenario = build_native_comparison_scenario(
        "small-functional", "HBF+", "SHORT", tp=1, batch_size=1,
        prompt_override=15, output_override=2, warmup_override=0,
        measurement_override=0, model_override=build_pilot_scenario("hbm_tp1").model,
    )
    result = run_scenario(scenario)
    batch = next(batch for batch in result.serving.batches if batch.kind == "decode")
    schedule = compile_serving_cohort_schedule(
        scenario, BatchCohort("logic-flush", "decode", batch.start_ns, batch.items)
    )
    flushes = [task for task in schedule.tasks
               if task.metadata.get("kv_tail_flush_role") == "logic_die_flush"]
    assert flushes
    flush_ids = {task.task_id for task in flushes}
    assert all(task.metadata.get("native_background") is True for task in flushes)
    for layer_id in {task.metadata.get("layer_id") for task in flushes}:
        nand_reads = _nand_read_tasks(schedule, layer_id, 0)
        assert nand_reads
        last_nand_read = nand_reads[-1]
        layer_flushes = [task for task in flushes
                         if task.metadata.get("layer_id") == layer_id]
        assert all(last_nand_read.task_id in task.dependencies for task in layer_flushes)
        assert all(not flush_ids.intersection(task.dependencies) for task in nand_reads)


def test_pp2_flush_dependency_stays_in_invocation_group():
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.planner import compile_serving_cohort_schedule
    from heterollm_sim.reporting import run_scenario
    from heterollm_sim.serving import BatchCohort

    scenario = build_native_comparison_scenario(
        "small-functional", "HBF+", "SHORT", tp=1, pp=2, batch_size=1,
        prompt_override=15, output_override=2, warmup_override=0,
        measurement_override=0, model_override=build_pilot_scenario("hbm_tp1").model,
    )
    result = run_scenario(scenario)
    batch = next(batch for batch in result.serving.batches if batch.kind == "decode")
    schedule = compile_serving_cohort_schedule(
        scenario, BatchCohort("pp2-flush", "decode", batch.start_ns, batch.items)
    )
    by_id = {task.task_id: task for task in schedule.tasks}
    flushes = [task for task in schedule.tasks
               if task.metadata.get("kv_tail_flush_role") in {"logic_die_flush", "kv_tail_flush_read"}]
    assert flushes
    for task in flushes:
        group = task.metadata.get("operator_invocation_group_id")
        assert group is not None
        parents = [by_id[parent] for parent in task.dependencies if parent in by_id]
        assert parents
        assert all(parent.metadata.get("operator_invocation_group_id") == group for parent in parents)


def test_a_flush_reads_the_scratchpad_back_and_writes_a_page_over_the_host_link(monkeypatch):
    import heterollm_sim._nand_acceleration as acceleration

    batches, original = [], acceleration.execute_staged_flush_batch

    def counted(*args):
        batches.append(original(*args))
        return batches[-1]

    monkeypatch.setattr(acceleration, "execute_staged_flush_batch", counted)
    core = NandCore(_logic_config(), capture_details=False)
    core._nand_numeric_enabled = True
    core.set_staging_scope("dense0:rank000")
    for index in range(4):
        _write(core, str(index), 10 * PAGE + index * 1024, 1024, index * 1000.0)
    core.submit_many([
        AccessRequest("read0", Operation.READ, 1000 * PAGE, 2 * PAGE, 0.0),
        AccessRequest("read1", Operation.READ, 2000 * PAGE, 2 * PAGE, 0.0),
    ])
    account = core.flush_staged_pages("dense0:rank000", 5000.0)
    assert batches and batches[-1] is not None
    assert account["host_transfer_bytes"] == 2 * PAGE
    intervals = core.timeline.last_intervals
    back, host = intervals["nand:host:2:read"], intervals["nand:host:2:write"]
    channel = max((value for name, value in intervals.items() if "channel" in name), key=lambda value: value[1])
    assert 5000.0 <= back[0] < back[1] <= host[0] < host[1] <= channel[0]


def test_every_program_path_moves_the_bank_log():
    from dataclasses import replace

    def run(numeric):
        config = replace(_logic_config(), protocol_mode="ocp_hbf_0_7", host_granularity_bytes=64)
        core = NandCore(config, capture_details=False)
        core._nand_numeric_enabled = numeric
        core.set_staging_scope("dense0:rank000")
        for index in range(4):
            _write(core, str(index), 10 * PAGE + index * 1024, 1024, index * 1000.0)
        core.submit_many([
            AccessRequest("read0", Operation.READ, 1000 * PAGE, 2 * PAGE, 0.0),
            AccessRequest("read1", Operation.READ, 2000 * PAGE, 2 * PAGE, 0.0),
        ])
        core.flush_staged_pages("dense0:rank000", 5000.0)
        core.submit(AccessRequest("bulk", Operation.WRITE, 0, 64 * PAGE, 200000.0))
        return core._bank_log

    expected = [4] * 16
    expected[10] += 1
    assert run(False) == run(True) == expected


def test_logic_die_flush_scalar_and_owned_batch_are_bit_exact():
    def run(numeric):
        core = NandCore(_logic_config(), capture_details=False)
        core._nand_numeric_enabled = numeric
        core.set_staging_scope("dense0:rank000")
        for index in range(4):
            _write(core, str(index), 10 * PAGE + index * 1024, 1024, index * 1000.0)
        # The owned path needs a resident numeric controller, just as the
        # native attention read wave does before issuing its background flush.
        core.submit_many([
            AccessRequest("read0", Operation.READ, 1000 * PAGE, 2 * PAGE, 0.0),
            AccessRequest("read1", Operation.READ, 2000 * PAGE, 2 * PAGE, 0.0),
        ])
        account = core.flush_staged_pages("dense0:rank000", 5000.0)
        return (account, core.page_staging_state(), dict(core.timeline.ready_ns),
                dict(core.timeline.busy_ns), dict(core.timeline.bytes_moved),
                dict(core.timeline.last_intervals))

    assert run(False) == run(True)
