"""Physical controller accounting survives runtime and report boundaries."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.data_motion import AccessKind, PhysicalRuntimeContext, PhysicalService, resolve_physical_task
from heterollm_sim.dram_core import DramCore
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_service import DRAM_CONTROL_COUNT_FIELDS, DRAM_CONTROL_TIME_FIELDS
from heterollm_sim.memory_transfer import summarize_batch
from heterollm_sim.memory_types import AccessRequest, Operation, make_hbm_config
from heterollm_sim.physical_contract import reconfigure_physical_config
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.planner import _summarize_dram_task_traffic
from heterollm_sim.reporting import _sum_batch_dram_traffic
from heterollm_sim.streaming_des import _merge_dram_traffic


def tiny_config():
    return make_hbm_config(channels=1, pseudo_channels_per_channel=1,
        bank_groups_per_rank=1, banks_per_group=1, rows_per_bank=32, row_bytes=1024,
        burst_bytes=32, capacity_bytes=32768, interface_bandwidth_gb_s=32.,
        open_ns=0., close_ns=0., read_latency_ns=0., write_latency_ns=0.,
        burst_interval_ns=0., write_recovery_ns=0.)


def physical_task(config, accesses):
    return TaskSpec("hbm-test", "request", "HBM accounting", TaskCategory.MEMORY,
        demands=(ResourceDemand("memory", 999999., bytes_moved=999999),),
        metadata={"physical_memory_config":config, "physical_owner":"memory",
                  "memory_accesses":[{**access, "resource_id":"memory"} for access in accesses]})


def protocol_config():
    config = resolve_physical_config("hbm3e_reference", overrides={
        "capacity_bytes":1048576, "stacks":1, "interface_bandwidth_gb_s":512.,
        "read_bandwidth_gb_s":128., "write_bandwidth_gb_s":128.})
    # Array delays are explicit test assumptions. The standard organization,
    # CA occupancy, CK/DQ ratio and fixed 32-byte bursts remain physical.
    return reconfigure_physical_config(config, overrides={
        "activate_to_read_ns":4., "activate_to_write_ns":4.,
        "read_latency_ns":4., "write_latency_ns":4., "refresh_policy":"disabled"})


def test_hbm_protocol_state_rollback_keeps_nested_control_state_and_owner_isolation():
    runtime = PhysicalRuntimeContext()
    first = runtime.runtime(tiny_config(), "hbm0").core
    second = runtime.runtime(tiny_config(), "hbm1").core
    for name in ("row_command_resource_prefix", "column_command_resource_prefix"):
        assert first.config.metadata[name] != second.config.metadata[name]
    first._hbm_control_state = {"banks":{"b":{"row":1}}, "refresh":{"channel":[1., 2.]}}
    saved = runtime.snapshot()
    for _ in range(2):
        first._hbm_control_state["banks"]["b"]["row"] = 7
        first._hbm_control_state["refresh"]["channel"].append(3.)
        runtime.restore(saved)
        assert first._hbm_control_state == {"banks":{"b":{"row":1}}, "refresh":{"channel":[1.,2.]}}


def test_half_duplex_burst_bytes_and_core_time_are_not_billed_twice():
    config = tiny_config()
    task = physical_task(config, [{"operation":"read", "address":0, "byte_count":1},
                                  {"operation":"write", "address":0, "byte_count":1}])
    kernel = UnifiedEventKernel()
    kernel.submit((task,))
    event = kernel.step()
    billed = next(iter(event.task.metadata["physical_execution_by_owner"].values()))
    assert (billed["logical_read_bytes"], billed["logical_write_bytes"]) == (1, 1)
    assert (billed["physical_read_bytes"], billed["physical_write_bytes"]) == (32, 32)
    assert event.end_ns == billed["completion_ns"] == 2.
    data_resources = {k:v for k,v in billed["resource_bytes"].items() if ":dram:data:" in k}
    assert sum(data_resources.values()) == 64
    assert all(d.resource_id != "memory" for d in event.demands)


def test_profile_latency_is_only_an_estimate_and_not_added_to_physical_service():
    config = tiny_config()
    component = ComponentSpec("hbm", "hbm", capacity_bytes=config.capacity_bytes,
        metadata={"physical_memory_config":config})
    service = PhysicalService("s", "hbm", "hbm", 32., 32.,
        read_latency_ns=999999., write_latency_ns=999999., component=component)
    billed = service.price(AccessKind.READ, 1, page_offset_bytes=0)
    assert billed["physical_bytes"] == 32
    assert billed["service_ns"] == 1.
    estimate = HBMProfile(bandwidth_gb_s=32.).memory_service(32, 32)
    assert estimate["bandwidth_service_ns"] == 2.
    assert estimate["parameter_basis"] == "analytical_cost_assumptions_not_jedec_protocol"


def test_command_counts_and_bus_occupancy_are_summed_across_accesses_and_reports(monkeypatch):
    # Supplied observations isolate aggregation from command scheduling; the
    # controller tests independently check how each command is generated.
    original = DramCore.submit
    sequence = iter((1, 2))
    observations = []
    def observed(core, request):
        result = original(core, request)
        n = next(sequence)
        result = replace(result, counters={**result.counters,
            **{key:n for key in DRAM_CONTROL_COUNT_FIELDS},
            **{key:n/2 for key in DRAM_CONTROL_TIME_FIELDS}})
        observations.append(result)
        return result
    monkeypatch.setattr(DramCore, "submit", observed)
    task = physical_task(tiny_config(), [{"operation":"read", "address":0, "byte_count":1},
                                         {"operation":"read", "address":32, "byte_count":1}])
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    for row in (next(iter(resolved.metadata["physical_execution_by_owner"].values())), resolved.metadata["physical_execution_by_owner"]["memory"],
                summarize_batch(observations).counters):
        assert all(row[key] == 3 for key in DRAM_CONTROL_COUNT_FIELDS)
        assert all(row[key] == 1.5 for key in DRAM_CONTROL_TIME_FIELDS)
    ledger = _summarize_dram_task_traffic([resolved])
    result = SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={"dram_traffic":ledger}))]))
    streamed = {}
    _merge_dram_traffic(streamed, ledger)
    for row in (ledger, streamed, _sum_batch_dram_traffic(result)):
        assert all(row[key] == 3 for key in DRAM_CONTROL_COUNT_FIELDS)
        assert all(row[key] == 1.5 for key in DRAM_CONTROL_TIME_FIELDS)
        assert "may overlap" in row["service_time_scope"]


def test_unknown_or_partial_command_observations_do_not_become_zero():
    field = "refresh_commands"
    ledgers = [{"task_count":1, field:2}, {"task_count":1}]
    result = SimpleNamespace(serving=SimpleNamespace(runtime_kernel_metrics={}, batches=[
        SimpleNamespace(cost=SimpleNamespace(metadata={"dram_traffic":row})) for row in ledgers]))
    streamed = {}
    for ledger in ledgers:
        _merge_dram_traffic(streamed, ledger)
    for row in (streamed, _sum_batch_dram_traffic(result)):
        assert row[field] is None
        assert row["metric_availability"][field]["status"] == "partial"


def test_partial_access_observations_are_not_published_as_a_complete_task(monkeypatch):
    original = DramCore.submit
    sequence = iter((True, False))
    def observed(core, request):
        result = original(core, request)
        counters = dict(result.counters)
        if next(sequence):
            counters["refresh_commands"] = 1
        else:
            counters.pop("refresh_commands", None)
        return replace(result, counters=counters)
    monkeypatch.setattr(DramCore, "submit", observed)
    task = physical_task(tiny_config(), [{"operation":"read", "address":0, "byte_count":1},
                                         {"operation":"read", "address":32, "byte_count":1}])
    resolved = resolve_physical_task(task, PhysicalRuntimeContext(), 0.)
    assert next(iter(resolved.metadata["physical_execution_by_owner"].values()))["refresh_commands"] is None
    assert resolved.metadata["physical_execution_by_owner"]["memory"]["refresh_commands"] is None
    assert _summarize_dram_task_traffic([resolved])["refresh_commands"] is None


def test_jedec_kernel_uses_core_completion_and_physical_dq_occupancy_once():
    config = protocol_config()
    requests = [AccessRequest(str(i), Operation.READ, 0, 1, 0.) for i in range(3)]
    direct = DramCore(config).run(requests)
    uncapped = DramCore(replace(config, read_bandwidth_gb_s=512., write_bandwidth_gb_s=512.)).run(requests)
    kernel = UnifiedEventKernel()
    kernel.submit((physical_task(config, [
        {"operation":"read", "address":0, "byte_count":1} for _ in requests]),))
    event = kernel.step()
    billed = next(iter(event.task.metadata["physical_execution_by_owner"].values()))
    assert event.end_ns == billed["completion_ns"] == direct.last_completion_ns
    assert event.end_ns > uncapped.last_completion_ns
    assert (billed["logical_read_bytes"], billed["physical_read_bytes"]) == (3, 96)
    assert billed["activate_commands"] == 1
    assert billed["read_commands"] == 3
    assert billed["row_command_bus_busy_ns"] == 1.5
    assert billed["column_command_bus_busy_ns"] == 3.
    assert billed["resource_busy_ns"]["memory:dram:data:0"] == 6.
    assert sum(billed["resource_bytes"].values()) == 96
    assert all(d.bytes_moved == 0 for d in event.demands if "command:" in d.resource_id)
    assert kernel.resource_busy_ns == billed["resource_busy_ns"]
    ledger = _summarize_dram_task_traffic([event.task])
    assert ledger["resource_totals"]["memory:dram:data:0"]["service_ns"] == 6.
    assert ledger["service_ns"] == event.end_ns > 6.
    assert ledger["refresh_commands"] is None


def test_jedec_runtime_replay_restores_real_pc_and_command_bus_state():
    runtime, config = PhysicalRuntimeContext(), protocol_config()
    seed = physical_task(config, [{"operation":"read", "address":0, "byte_count":1}])
    resolve_physical_task(seed, runtime, 0.)
    saved = runtime.snapshot()
    core = runtime.runtime(config, "memory").core
    control = deepcopy(core._hbm_control_state)
    follow = physical_task(config, [{"operation":"read", "address":32, "byte_count":1},
                                    {"operation":"read", "address":0, "byte_count":1}])
    first = resolve_physical_task(follow, runtime, 0.)
    runtime.restore(saved)
    assert core._hbm_control_state == control
    second = resolve_physical_task(follow, runtime, 0.)
    assert next(iter(first.metadata["physical_execution_by_owner"].values())) == next(iter(second.metadata["physical_execution_by_owner"].values()))
    billed = next(iter(second.metadata["physical_execution_by_owner"].values()))
    # Adjacent 32-byte bursts select the two PCs of the same physical channel.
    assert set(k for k in billed["resource_busy_ns"] if "column_command:" in k) == {
        "memory:dram:column_command:0:0"}
    assert set(k for k in billed["resource_busy_ns"] if ":data:" in k) == {
        "memory:dram:data:0", "memory:dram:data:1"}
