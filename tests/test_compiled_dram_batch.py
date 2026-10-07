from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from heterollm_sim.architecture_presets import materialize_architecture_payload
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, Operation, parse_physical_memory_config


spec = importlib.util.spec_from_file_location("capture_equivalence", Path(__file__).resolve().parents[1] /
                                            "tools/measure_physical_capture_equivalence.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


def config_for(hardware, component):
    payload = materialize_architecture_payload(hardware)
    row = next(item for item in payload["components"] if item["component_id"] == component)
    return parse_physical_memory_config(row["metadata"]["physical_memory_config"])


def compare(config, requests):
    serial, batched = [DramCore(config, capture_details=False) for _ in range(2)]
    expected = tuple(serial.submit(request) for request in requests)
    actual = batched.submit_many(requests)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.completion_ns == e.completion_ns
        assert a.logical_bytes == e.logical_bytes and a.transfer_bytes == e.transfer_bytes
        assert {key: value for key, value in a.counters.items() if key != "compiled_batch_float64"} == e.counters
    assert measure.state(serial) == measure.state(batched)
    return actual


@pytest.mark.parametrize("hardware,component", [
    (measure.HARDWARE[0], "hostmem0"), (measure.HARDWARE[0], "gddr0"),
    (measure.HARDWARE[1], "hbm0"), (measure.HARDWARE[2], "hostmem0")])
def test_batch_preserves_all_41_mixed_requests(hardware, component):
    config = config_for(hardware, component)
    actual = compare(config, measure.requests(config))
    assert all(row.counters["compiled_batch_float64"] for row in actual)


@pytest.mark.parametrize("outstanding", [1, 2, 64])
@pytest.mark.parametrize("operation", [Operation.READ, Operation.WRITE])
def test_real_transposed_kv_column_requests_keep_individual_admission(operation, outstanding):
    config = replace(config_for(measure.HARDWARE[0], "gddr0"), max_outstanding_requests=outstanding)
    requests = tuple(AccessRequest("column-" + str(column), operation,
                    column * 768 * 2 + 513 * 2, 2, 1e6) for column in range(1024))
    actual = compare(config, requests)
    assert sum(row.transfer_bytes for row in actual) == 1024 * 64
    assert max(row.queue_wait_ns for row in actual) > 0


def test_batch_preserves_direction_reversal_and_live_calendar_identity():
    config = config_for(measure.HARDWARE[0], "gddr0")
    core = DramCore(config, capture_details=False)
    slots = [17.0]
    core.timeline.lane_available["dram:data:0"] = slots
    items = (AccessRequest("read", Operation.READ, 0, 2, 0),
             AccessRequest("write", Operation.WRITE, 0, 2, 0),
             AccessRequest("read2", Operation.READ, 0, 2, 0))
    core.submit_many(items)
    assert core.timeline.lane_available["dram:data:0"] is slots
    assert core.timeline.directions["dram:data:0"] == "read"


def test_batch_detailed_capture_keeps_individual_trace_results():
    config = config_for(measure.HARDWARE[0], "gddr0")
    requests = (AccessRequest("read", Operation.READ, 0, 65, 0),
                AccessRequest("write", Operation.WRITE, 17, 2, 0))
    serial, batched = DramCore(config), DramCore(config)
    assert batched.submit_many(requests) == tuple(serial.submit(request) for request in requests)


def mixed_owner_task():
    from dataclasses import asdict
    from heterollm_sim.contracts import TaskCategory, TaskSpec
    config = config_for(measure.HARDWARE[0], "gddr0")
    accesses = []
    for owner, op, count, offset in (("a", "write", 32, 0), ("b", "read", 5, 0),
                                    ("a", "read", 32, 0), ("a", "write", 4, 0),
                                    ("a", "read", 4, 0), ("b", "write", 16, 0)):
        accesses += [{"operation": op, "address": offset + index * 1536 + 1026,
                      "byte_count": 2, "physical_owner": owner} for index in range(count)]
    # Repeated and partially overlapping writes must stay ordered, including
    # a following read. They deliberately break the disjoint-write fast run.
    accesses += [{"operation": op, "address": address, "byte_count": size, "physical_owner": "a"}
                 for op, address, size in (("write", 1026, 4), ("write", 1028, 4), ("read", 1026, 6))]
    return TaskSpec(task_id="mixed-owners", request_id="cohort-test", name="mixed-owners",
        category=TaskCategory.MEMORY, metadata={"physical_memory_config": asdict(config),
            "physical_memory_configs": {owner: asdict(config) for owner in ("a", "b")},
            "physical_energy_pj_per_byte_by_owner": {"a": 4.0, "b": 7.0},
            "memory_accesses": tuple(accesses)})


def clean_markers(value):
    if isinstance(value, dict):
        return {key: clean_markers(item) for key, item in value.items() if key != "compiled_batch_float64"}
    if isinstance(value, (list, tuple)):
        return tuple(clean_markers(item) for item in value)
    return value


def test_physical_dispatch_preserves_interleaved_owners_and_overlap_order(monkeypatch):
    from heterollm_sim import data_motion
    task = mixed_owner_task()
    contexts, results = [], []
    for enabled in (False, True):
        monkeypatch.setattr(data_motion, "_USE_COMPILED_DRAM_BATCH", enabled)
        runtime = data_motion.PhysicalRuntimeContext(capture_details=False)
        results.append(data_motion.resolve_physical_task(task, runtime, 17.0))
        contexts.append(runtime)
    assert results[0].demands == results[1].demands
    assert clean_markers(results[0].metadata) == clean_markers(results[1].metadata)
    for owner in contexts[0].runtimes:
        assert measure.state(contexts[0].runtimes[owner].core) == measure.state(contexts[1].runtimes[owner].core)


@pytest.mark.parametrize("count,byte_count,expected_batches", [(15, 2, []), (16, 2, [16]), (16, 4096, [])])
def test_dispatch_batches_only_long_sequences_of_small_requests(monkeypatch, count, byte_count, expected_batches):
    from dataclasses import asdict
    from heterollm_sim import data_motion
    from heterollm_sim.contracts import TaskCategory, TaskSpec
    config = config_for(measure.HARDWARE[0], "gddr0")
    calls = []
    original = DramCore.submit_many
    def submit_many(core, requests):
        calls.append(len(requests))
        return original(core, requests)
    monkeypatch.setattr(DramCore, "submit_many", submit_many)
    monkeypatch.setattr(data_motion, "_USE_COMPILED_DRAM_BATCH", True)
    task = TaskSpec(task_id="threshold", request_id="cohort-threshold", name="threshold",
        category=TaskCategory.MEMORY, metadata={"physical_memory_config": asdict(config),
        "memory_accesses": [{"operation": "write", "address": index * 8192,
            "byte_count": byte_count, "physical_owner": "memory"} for index in range(count)]})
    data_motion.resolve_physical_task(task, data_motion.PhysicalRuntimeContext(capture_details=False), 0.0)
    assert calls == expected_batches


def test_failed_numeric_batch_cannot_commit_partial_core_state(monkeypatch):
    from heterollm_sim import _dram_numeric
    core = DramCore(config_for(measure.HARDWARE[0], "gddr0"), capture_details=False)
    core.submit(AccessRequest("warm", Operation.READ, 0, 2, 0.0))
    before = measure.state(core)
    def fail_after_mutating_packed_state(*args):
        args[12][:] = 123.0  # bankready is an independent packed array.
        args[14][:] = 456.0  # calendar is not the live lane list.
        raise RuntimeError("injected numeric execution failure")
    monkeypatch.setattr(_dram_numeric, "run_numeric_many", fail_after_mutating_packed_state)
    with pytest.raises(RuntimeError, match="injected numeric execution failure"):
        core.submit_many(tuple(AccessRequest(str(i), Operation.WRITE, i * 1536, 2, 0.0) for i in range(16)))
    assert measure.state(core) == before


def test_outer_failed_batch_restores_all_owners(monkeypatch):
    from heterollm_sim import data_motion
    runtime = data_motion.PhysicalRuntimeContext(capture_details=False)
    original = DramCore.submit_many
    def fail_after_a_successful_batch(core, requests):
        original(core, requests)
        raise RuntimeError("injected post-batch failure")
    monkeypatch.setattr(DramCore, "submit_many", fail_after_a_successful_batch)
    monkeypatch.setattr(data_motion, "_USE_COMPILED_DRAM_BATCH", True)
    with pytest.raises(RuntimeError, match="injected post-batch failure"):
        data_motion.resolve_physical_task(mixed_owner_task(), runtime, 17.0)
    assert not runtime.runtimes


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_second_batch_failure_restores_warm_state_and_kernel_retry(monkeypatch, standalone, error_type):
    from heterollm_sim import data_motion
    from heterollm_sim.event_kernel import UnifiedEventKernel

    task = mixed_owner_task()
    seed = replace(task, task_id="warm-owner-a", metadata={
        **task.metadata, "memory_accesses": task.metadata["memory_accesses"][:1]})

    def warm_kernel():
        kernel = UnifiedEventKernel.from_closed_graph((task,), capture_physical_details=False)
        data_motion.resolve_physical_task(seed, kernel.physical_runtime, 0.0)
        return kernel

    kernel = warm_kernel()
    runtime = kernel.physical_runtime
    before = runtime.snapshot()
    original = DramCore.submit_many
    calls = []

    def fail_after_second_commit(core, requests):
        result = original(core, requests)
        assert all(item.counters.get("compiled_batch_float64") for item in result)
        calls.append(len(result))
        if len(calls) == 2:
            raise error_type("injected after second compiled batch commit")
        return result

    monkeypatch.setattr(data_motion, "_USE_COMPILED_DRAM_BATCH", True)
    monkeypatch.setattr(DramCore, "submit_many", fail_after_second_commit)
    with pytest.raises(error_type, match="second compiled batch commit"):
        data_motion.resolve_physical_task(task, runtime, 17.0) if standalone else kernel.step()
    assert calls == [32, 32]
    assert runtime.snapshot() == before
    assert set(runtime.runtimes) == {"a"}  # The failed task's new owner is removed.
    monkeypatch.setattr(DramCore, "submit_many", original)
    if not standalone:
        assert kernel.has_active_tasks
        actual = kernel.step()
        monkeypatch.setattr(data_motion, "_USE_COMPILED_DRAM_BATCH", False)
        expected = warm_kernel().step()
        assert actual.start_ns == expected.start_ns
        assert actual.end_ns == expected.end_ns
        assert actual.demands == expected.demands
        assert not kernel.has_active_tasks
