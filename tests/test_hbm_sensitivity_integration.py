"""Complete timing packs and request scheduling reach live service boundaries."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrated_hbf_profiles import _hbm_physical_config
from hbf_hbm_serving_study import build_study_scenario
from heterollm_sim.contracts import TaskCategory, TaskSpec
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_contract import parse_execution_physical_config
from test_hbm_compiled_control import configuration


@pytest.mark.parametrize("capacity", [512*1024**3, 1024**4])
def test_main_factory_executes_complete_assumed_reference(capacity):
    config = _hbm_physical_config(capacity_bytes=capacity, read_gb_s=1600., write_gb_s=1600.)
    assert config.command_clock_ns == .625
    assert config.refresh_policy == "all_bank"
    assert config.refresh_cycle_ns == 500.
    for name in ("activate_same_group_ns", "activate_different_group_ns", "four_activate_window_ns",
                 "read_to_precharge_ns", "write_to_read_same_group_ns", "write_to_read_different_group_ns",
                 "read_to_write_command_ns"):
        assert getattr(config, name) > 0
    assert not config.metadata["unconfigured_timing_parameters"]
    assert config.metadata["hbm_parameter_profile"]["vendor_calibrated"] is False
    result = DramCore(config, capture_details=False).submit(AccessRequest("refresh", "read", 0, 64, 4000.))
    assert result.counters["refresh_commands"] > 0


def test_serving_inventory_preserves_parameter_and_controller_choices():
    scenario = build_study_scenario("A_hbm_only", prompt=4, output=2,
        hbm_parameter_profile="slow", hbm_controller={"address_mapping": "row_local",
            "interleave_bytes": None, "scheduler": "read_write_batch", "queue_scope": "channel",
            "max_outstanding_requests": 1})
    config = parse_execution_physical_config(scenario.hardware.get_component("hbm0").metadata["physical_memory_config"])
    assert config.metadata["hbm_parameter_profile"]["name"] == "slow"
    assert config.refresh_cycle_ns == 650.
    assert (config.address_mapping, config.scheduler, config.queue_scope) == ("row_local", "read_write_batch", "channel")


@pytest.mark.parametrize("details", [False, True])
def test_mixed_small_live_task_reaches_scheduler_and_reports_observations(monkeypatch, details):
    config = configuration(scheduler="read_write_batch", scheduler_window=8)
    calls, executed = [], []
    original = DramCore.submit_many
    def observe(core, requests):
        calls.append(tuple(r.operation.value for r in requests))
        results = original(core, requests)
        executed.extend(results)
        return results
    monkeypatch.setattr(DramCore, "submit_many", observe)
    accesses = tuple({"operation": op, "address": addr, "byte_count": 32}
        for op, addr in [("read", 0), ("write", 256), ("read", 512), ("write", 512)])
    task = TaskSpec("mixed", "request", "mixed", TaskCategory.MEMORY,
        metadata={"physical_owner": "hbm", "physical_memory_config": config, "memory_accesses": accesses})
    runtime = PhysicalRuntimeContext(capture_details=details)
    resolved = resolve_physical_task(task, runtime, 0.)
    assert calls == [("read", "write", "read", "write")]
    assert executed[3].completion_ns >= executed[2].completion_ns
    row = resolved.metadata["physical_execution_by_owner"]["hbm"]
    assert row["scheduler_batches"] == 1
    assert row["scheduler_reordered_requests"] > 0
    assert row["controller_direction_switches"] == 1
    assert row["logical_read_bytes"] == row["logical_write_bytes"] == 64


def test_channel_credit_state_is_restorable_without_mutating_saved_point():
    config = configuration(queue_scope="channel", max_outstanding_requests=1)
    runtime = PhysicalRuntimeContext(config, "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    core.submit(AccessRequest("seed", "read", 0, 32))
    saved = runtime.snapshot()
    saved_channels = deepcopy(saved["runtime_state"]["hbm"]["core"]["_channel_inflight"])
    request = AccessRequest("next", "write", 0, 64)
    first = core.submit(request)
    assert saved["runtime_state"]["hbm"]["core"]["_channel_inflight"] == saved_channels
    for _ in range(2):
        runtime.restore(saved)
        result = runtime.runtimes["hbm"].core.submit(request)
        assert result.completion_ns == first.completion_ns
        assert result.queue_wait_ns == first.queue_wait_ns


@pytest.mark.parametrize("field", ["address_mapping", "scheduler", "scheduler_window", "queue_scope"])
def test_serialized_controller_contract_rejects_omitted_fields(field):
    payload = asdict(configuration())
    del payload[field]
    with pytest.raises(ValueError, match="missing explicit fields"):
        parse_execution_physical_config(payload)
