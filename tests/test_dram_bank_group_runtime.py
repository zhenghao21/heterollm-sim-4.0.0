"""Bank-group command gates survive runtime boundaries without extra billing."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest

from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import (
    AccessKind, PhysicalRuntimeContext, resolve_physical_task, resolve_service,
)
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import AccessRequest, DramConfig


def configuration(**changes):
    return replace(DramConfig(kind="HBM", channels=1, data_lanes=1,
        bank_groups_per_rank=2, banks_per_group=2, rows_per_bank=64,
        row_bytes=1024, burst_bytes=64, interface_bandwidth_gb_s=64.,
        open_ns=0., close_ns=0., read_latency_ns=3., write_latency_ns=3.,
        burst_interval_ns=2., read_recovery_ns=0., write_recovery_ns=0.,
        column_same_group_ns=10., column_different_group_ns=4.,
        capacity_bytes=262144), **changes)


def endpoint(config):
    return resolve_service(ComponentSpec("hbm", "hbm", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config": config, "memory_service_owner": "hbm"}))


# The rotated mapping alternates groups: address 128 reaches bank 1 (group 0)
# and address 64 reaches bank 2 (group 1).
def task(config, addresses=(0, 128, 64, 0), *, ident="read", operation="read", size=64):
    return TaskSpec(ident, "request", ident, TaskCategory.MEMORY,
        metadata={"physical_memory_config": config, "physical_owner": "hbm",
            "memory_accesses": tuple({"operation": operation, "address": address,
                "byte_count": size} for address in addresses)})


@pytest.mark.parametrize("details", [True, False])
def test_kernel_group_wait_changes_completion_without_duplicate_cost(details):
    config = configuration()
    baseline = replace(config, column_same_group_ns=0., column_different_group_ns=0.)

    def run(current):
        tasks = tuple(task(current, (address,), ident=str(i))
                      for i, address in enumerate((0, 128, 64, 0)))
        kernel = UnifiedEventKernel.from_closed_graph(tasks, capture_physical_details=details)
        events = [kernel.step() for _ in tasks]
        return kernel, events

    kernel, events = run(config)
    original, original_events = run(baseline)
    assert [event.end_ns for event in events] == [4., 14., 18., 24.]
    assert [event.end_ns for event in original_events] == [4., 6., 8., 10.]
    assert kernel.resource_busy_ns == original.resource_busy_ns == {
        "hbm:dram:command:0": 8., "hbm:dram:data:0": 4.}
    assert sum(kernel.physical_runtime.timeline.busy_ns.values()) == 12.
    assert sum(d.bytes_moved for e in events for d in e.demands) == 256
    assert all(len({d.resource_id for d in event.demands}) == len(event.demands)
               for event in events)


@pytest.mark.parametrize("details", [True, False])
def test_physical_service_and_resolver_observe_identical_group_history(details):
    config = configuration()
    service = endpoint(config)
    runtime = PhysicalRuntimeContext(capture_details=details)
    bills = [service.price(AccessKind.READ, 64, page_offset_bytes=address,
                           runtime=runtime, arrival_ns=0.) for address in (0, 128, 64, 0)]
    assert [bill["completion_ns"] for bill in bills] == [4., 14., 18., 24.]
    assert sum(bill["physical_bytes"] for bill in bills) == 256
    combined = resolve_physical_task(task(config), PhysicalRuntimeContext(capture_details=details), 0.)
    assert combined.metadata["physical_completion_ns"] == 24.
    assert next(iter(combined.metadata["physical_execution_by_owner"].values()))["physical_read_bytes"] == 256
    assert len({d.resource_id for d in combined.demands}) == len(combined.demands)
    assert sum(d.service_ns for d in combined.demands) == 12.


def test_snapshot_twice_restores_nested_group_clocks_and_activation_window():
    config = configuration(activate_same_group_ns=7., activate_different_group_ns=3.,
        four_activate_window_ns=50., write_to_read_same_group_ns=9.,
        write_to_read_different_group_ns=5.)
    runtime = PhysicalRuntimeContext(capture_details=False)
    resolve_physical_task(task(config, (0, 128), ident="seed", operation="write"), runtime, 0.)
    core = runtime.runtimes["hbm"].core
    saved = runtime.snapshot()
    frozen_groups = copy.deepcopy(core._bank_group_state)
    assert frozen_groups
    saved_groups = saved["runtime_state"]["hbm"]["core"]["_bank_group_state"]
    assert saved_groups is not core._bank_group_state
    outcomes = []
    for _ in range(2):
        result = resolve_physical_task(task(config, (64, 192, 4096, 0)), runtime, 0.)
        assert core._bank_group_state != frozen_groups
        assert saved_groups == frozen_groups
        outcomes.append((result.demands, next(iter(result.metadata["physical_execution_by_owner"].values()))))
        runtime.restore(saved)
        assert runtime.runtimes["hbm"].core is core
        assert core._bank_group_state == frozen_groups
        assert core._bank_group_state is not saved_groups
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("entry", ["price", "price_batch", "resolve"])
def test_post_core_failure_restores_group_history_before_retry(entry):
    config = configuration(activate_same_group_ns=7., four_activate_window_ns=50.)
    service = endpoint(config)
    runtime = PhysicalRuntimeContext(capture_details=False)
    service.price(AccessKind.WRITE, 64, page_offset_bytes=0, runtime=runtime, arrival_ns=0.)
    saved = runtime.snapshot()
    frozen_groups = copy.deepcopy(runtime.runtimes["hbm"].core._bank_group_state)

    def submit():
        if entry == "price":
            return service.price(AccessKind.READ, 64, page_offset_bytes=64,
                                 runtime=runtime, arrival_ns=0.)
        if entry == "price_batch":
            return service.price_batch([{"operation": "read", "address": 64,
                                         "byte_count": 64}], runtime=runtime)
        resolved = resolve_physical_task(task(config, (64,)), runtime, 0.)
        return resolved.demands, next(iter(resolved.metadata["physical_execution_by_owner"].values()))

    original = DramCore.submit
    def fail_after_execution(core, request):
        original(core, request)
        assert core._bank_group_state != frozen_groups
        raise ValueError("injected after group-state mutation")

    with patch.object(DramCore, "submit", fail_after_execution):
        with pytest.raises(ValueError, match="after group-state mutation"):
            submit()
    assert runtime.snapshot() == saved
    assert runtime.runtimes["hbm"].core._bank_group_state == frozen_groups
    retry = submit()
    runtime.restore(saved)
    assert submit() == retry


def test_kernel_failed_completion_rolls_back_group_history_and_keeps_ready_task():
    config = configuration()
    kernel = UnifiedEventKernel.from_closed_graph((task(config, (128,)),))
    service = endpoint(config)
    service.price(AccessKind.READ, 64, page_offset_bytes=0,
                  runtime=kernel.physical_runtime, arrival_ns=0.)
    saved = kernel.physical_runtime.snapshot()
    original = DramCore.submit
    def fail_after_execution(core, request):
        original(core, request)
        raise ValueError("injected post-core kernel failure")
    with patch.object(DramCore, "submit", fail_after_execution):
        with pytest.raises(ValueError, match="post-core kernel failure"):
            kernel.step()
    assert kernel.physical_runtime.snapshot() == saved
    event = kernel.step()
    assert event.end_ns == 14.
    assert kernel.completed_count == 1
    assert sum(d.bytes_moved for d in event.demands) == 64
    assert sum(d.service_ns for d in event.demands) == 3.


def test_preview_does_not_mutate_live_group_history_or_future_charge():
    config = configuration()
    service = endpoint(config)
    runtime = PhysicalRuntimeContext(capture_details=False)
    service.price(AccessKind.WRITE, 64, page_offset_bytes=0, runtime=runtime, arrival_ns=0.)
    saved = runtime.snapshot()
    preview = service.price(AccessKind.READ, 256, page_offset_bytes=0,
                            runtime=runtime, arrival_ns=0., preview=True)
    assert preview["physical_read_bytes"] == 256
    assert runtime.snapshot() == saved
    result = service.price(AccessKind.READ, 64, page_offset_bytes=64, runtime=runtime, arrival_ns=0.)
    runtime.restore(saved)
    assert result == service.price(AccessKind.READ, 64, page_offset_bytes=64,
                                   runtime=runtime, arrival_ns=0.)


def test_summary_batch_and_long_request_preserve_detailed_group_constraints():
    config = configuration(activate_same_group_ns=7., activate_different_group_ns=3.,
        four_activate_window_ns=50., write_to_read_same_group_ns=9.,
        write_to_read_different_group_ns=5.)
    requests = (AccessRequest("write", "write", 0, 8192),
                AccessRequest("read", "read", 8192, 4096),
                AccessRequest("reuse", "read", 0, 64))
    outcomes = []
    for mode in ("detailed", "summary", "batch"):
        core = DramCore(config, capture_details=mode == "detailed")
        results = (core.submit_many(requests) if mode == "batch"
                   else tuple(core.submit(request) for request in requests))
        assert all(not result.counters.get("compiled_float64")
                   and not result.counters.get("compiled_batch_float64") for result in results)
        outcomes.append((tuple((result.completion_ns, result.transfer_bytes,
                                result.physical_read_bytes, result.physical_write_bytes)
                               for result in results), core._banks, core._bank_group_state,
                         core.timeline.busy_ns, core.timeline.bytes_moved))
    assert outcomes[0] == outcomes[1] == outcomes[2]
