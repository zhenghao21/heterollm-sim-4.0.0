"""REF-on source proofs reread native mutable state without publishing it."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from exploring_hbf_paper_config import build_paper_hbm_config
from heterollm_sim.contracts import _PreparedExecutionStage
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.execution_control import ExecutionControl
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.native_window_producer import (
    _native_refresh_source_qualification, _native_cold_ref_service, _phase,
    summarize_native_stages)
from heterollm_sim.planner import TopologyAwareBatchCostProvider
from heterollm_sim.serving import _OnlineRuntime
from test_native_window_producer import task


def source(operation="read", arrival=0., byte_count=32768):
    runtime = object.__new__(_OnlineRuntime)
    runtime._native_window_owns_execution_kernel = True
    runtime._execution_resource_kernel = UnifiedEventKernel(capture_physical_details=False)
    runtime.execution_control = ExecutionControl()
    runtime.lowerer = object.__new__(TopologyAwareBatchCostProvider)
    runtime.now = 0.
    config = build_paper_hbm_config("HBM4")
    physical = runtime._execution_resource_kernel.physical_runtime
    core = physical.runtime(config, "hbm").core
    core.submit(AccessRequest("actual", operation, 0, byte_count, arrival_ns=arrival))
    return runtime, config, physical, core


def qualification(data):
    runtime, config, physical, _core = data
    return _native_refresh_source_qualification(config, "hbm", physical, runtime)


def test_real_control_plane_handoff_keeps_native_origin_and_live_guards():
    from exploring_hbf_native_config import build_native_comparison_scenario
    from exploring_hbf_native_pilot import build_pilot_scenario
    from heterollm_sim.control_plane import bootstrap_control_plane, _is_native_bootstrap_kernel
    from heterollm_sim.ir import default_memory_resource_id
    from heterollm_sim.memory_types import DramConfig
    from heterollm_sim.physical_contract import parse_execution_physical_config
    from heterollm_sim.serving import compile_serving_plan

    scenario = build_native_comparison_scenario("llama3_405b", "HBF", "SHORT", tp=2,
        batch_size=1, model_override=build_pilot_scenario("hbm_tp1").model,
        prompt_override=8, output_override=4, warmup_override=0, measurement_override=2)
    bootstrap = bootstrap_control_plane(scenario, capture_physical_details=False)
    runtime = _OnlineRuntime(compile_serving_plan(bootstrap.scenario),
        TopologyAwareBatchCostProvider(bootstrap.scenario, defer_physical_execution=True),
        execution_kernel=bootstrap.kernel, runtime_origin_ns=bootstrap.makespan_ns)
    assert _is_native_bootstrap_kernel(bootstrap.kernel)
    assert runtime._native_window_owns_execution_kernel
    physical = bootstrap.kernel.physical_runtime
    for component in bootstrap.scenario.hardware.components:
        raw = component.metadata.get("physical_memory_config")
        if raw is None:
            continue
        config = parse_execution_physical_config(raw)
        if isinstance(config, DramConfig) and config.control_mode == "jedec_hbm":
            owner = default_memory_resource_id(component)
            assert _native_refresh_source_qualification(config, owner, physical, runtime)["qualified"]
            runtime.execution_control = ExecutionControl(progress_callback=lambda _progress: None)
            assert not _native_refresh_source_qualification(config, owner, physical, runtime)["qualified"]
            break
    else:
        pytest.fail("native HBF staging controller was not present")


@pytest.mark.parametrize("forged", (True, "native", object()))
def test_external_kernel_cannot_forge_native_control_plane_origin(forged):
    from heterollm_sim.control_plane import _is_native_bootstrap_kernel
    kernel = UnifiedEventKernel(capture_physical_details=False)
    assert not _is_native_bootstrap_kernel(kernel)
    kernel._native_bootstrap_origin = forged
    assert not _is_native_bootstrap_kernel(kernel)


@pytest.mark.parametrize("operation,arrival", [("read", 0.), ("write", 3899.5),
    ("read", 3900.), ("write", 1e8+.17)])
def test_real_native_source_is_read_without_pack_refresh_or_publication(operation, arrival):
    data = source(operation, arrival)
    core = data[3]
    packed = core._hbm_numeric_state
    before = tuple(value.copy() for value in packed.arrays)
    before_published = dict(vars(core)["_hbm_control_state"])
    proof = qualification(data)
    assert proof["qualified"], proof
    assert proof["state"] == "owned_numeric_REF_epoch_state"
    assert core._hbm_numeric_state is packed and core._hbm_numeric_enabled
    assert vars(core)["_hbm_control_state"] == before_published
    assert all(np.array_equal(old, new) for old, new in zip(before, packed.arrays))


@pytest.mark.parametrize("callback", ("progress_callback", "cancellation_callback"))
def test_unknown_callbacks_cannot_supply_native_source_ancestry(callback):
    data = source()
    data[0].execution_control = ExecutionControl(**{callback: lambda *args: False})
    assert not qualification(data)["qualified"]


def test_external_kernel_origin_and_unknown_kernel_lowerer_hooks_keep_coarse_bound():
    data = source()
    data[0]._native_window_owns_execution_kernel = False
    assert not qualification(data)["qualified"]
    data[0]._native_window_owns_execution_kernel = True
    data[0].lowerer.estimate = lambda *args: None
    assert not qualification(data)["qualified"]
    del data[0].lowerer.estimate
    data[0]._execution_resource_kernel.step = lambda *args: None
    assert not qualification(data)["qualified"]


@pytest.mark.parametrize("attribute", ("_banks", "_hbm_control_state"))
def test_public_bank_or_controller_exposure_prevents_future_native_ownership(attribute):
    data = source()
    getattr(data[3], attribute)
    assert not qualification(data)["qualified"]


@pytest.mark.parametrize("mutation", ("future_due", "old_ACT", "missing_PC_REF",
    "old_REF_debt", "REF_disabled", "unseen_open", "wrong_dtype", "late_column"))
def test_forged_or_restored_epoch_state_cannot_hide_maintenance_debt(mutation):
    data = source(arrival=3900.)
    core, config = data[3], data[1]
    packed = core._hbm_numeric_state
    rows, bf, _addresses, pcs, history, _seen, channels = packed.arrays[:7]
    opened = int(np.flatnonzero(rows >= 0)[0])
    lane = opened//config.bank_count
    channel = lane//2
    if mutation == "future_due":
        channels[channel, 1] += 100*config.refresh_interval_ns
    elif mutation == "old_ACT":
        bf[opened, 0] = channels[channel, 1]-2*config.refresh_interval_ns
    elif mutation == "missing_PC_REF":
        pcs[2*channel+1, 1] = 0.
    elif mutation == "old_REF_debt":
        rid = packed.ids[channel]
        data[2].timeline.lane_available[rid][0] = 1e8
    elif mutation == "REF_disabled":
        packed.g[8] = 0
    elif mutation == "unseen_open":
        channels[channel, 1] = -1.
    elif mutation == "wrong_dtype":
        packed.arrays = (rows.astype(float), *packed.arrays[1:])
    else:
        history[lane, 0, 1] = channels[channel, 1]+1.
    assert not qualification(data)["qualified"], mutation


def test_shared_lane_and_direct_sibling_bus_are_not_private_HBM_sources():
    data = source()
    timeline = data[2].timeline
    rid = data[3]._hbm_numeric_state.ids[0]
    timeline.lane_available["foreign"] = timeline.lane_available[rid]
    assert not qualification(data)["qualified"]
    del timeline.lane_available["foreign"]
    other = replace(data[1], metadata={"row_command_resource_prefix":
        data[3].config.metadata["row_command_resource_prefix"]})
    data[2].runtime(other, "other")
    assert not qualification(data)["qualified"]


def test_native_queue_is_not_a_future_state_writer():
    from queue import SimpleQueue
    data = source()
    data[0].execution_control = ExecutionControl(progress_queue=SimpleQueue())
    assert qualification(data)["qualified"]


def test_uncreated_owner_uses_actual_future_bound_ports_for_alias_and_sibling_checks():
    runtime, config, physical, _core = source()
    def check():
        return _native_refresh_source_qualification(config, "future", physical, runtime)
    assert check()["qualified"]
    physical.timeline.resource_aliases["future:dram:row_command:0:0"] = "future:dram:row_command:0:1"
    assert not check()["qualified"]
    physical.timeline.resource_aliases.clear()
    other = replace(config, metadata={"row_command_resource_prefix": "future:dram:row_command"})
    physical.runtime(other, "sibling")
    assert not check()["qualified"]


@pytest.mark.parametrize("column", range(1, 7))
def test_each_native_bank_history_column_has_its_own_physical_tail_limit(column):
    data = source(arrival=3900.)
    packed = data[3]._hbm_numeric_state
    rows, bf, _addresses, _pcs, _history, _seen, channels = packed.arrays[:7]
    opened = int(np.flatnonzero(rows >= 0)[0])
    index = opened//data[1].bank_count//2
    bf[opened, column] = channels[index, 1]+100.
    assert not qualification(data)["qualified"]


def test_public_restore_of_nonempty_python_state_does_not_recreate_numeric_ancestry():
    data = source()
    snapshot = data[2].snapshot()
    data[2].restore(snapshot)
    assert not qualification(data)["qualified"]


def test_unseen_channel_cannot_hide_an_external_port_reservation():
    data = source(byte_count=32)
    packed = data[3]._hbm_numeric_state
    unseen = int(np.flatnonzero(packed.arrays[6][:, 1] < 0.)[0])
    data[2].timeline.lane_available[packed.ids[unseen]] = [1.]
    assert not qualification(data)["qualified"]


def test_uncreated_owner_cannot_hide_a_ready_frontier_without_a_lane_list():
    runtime, config, physical, _core = source()
    physical.timeline.ready_ns["future:dram:row_command:0:0"] = 1.
    proof = _native_refresh_source_qualification(config, "future", physical, runtime)
    assert not proof["qualified"]


def test_row_REF_source_proof_requires_its_clock_even_when_cold_work_is_smaller():
    config = build_paper_hbm_config("HBM4")
    count = 2*1024**2
    proof = _native_cold_ref_service(config, count, "read", independent_channels=True,
                                      native_refresh_reachable=True)
    assert proof["native_REF_on_source_clock_retained"]
    assert not proof.get("qualified_source_required")  # Cold active work was smaller.
    from heterollm_sim.native_hbm_cold_bound import native_hbm_refresh_service_upper
    reachable = native_hbm_refresh_service_upper(config, count, "read",
        independent_channels=True, native_refresh_reachable=True)
    assert proof["hbm_clock_domain_limit_ns"] <= reachable["hbm_clock_domain_limit_ns"]
    coarse = _phase(config, 32768, "read", independent_hbm_channels=True,
                    native_refresh_reachable=True)
    guarded = _phase(config, 32768, "read", independent_hbm_channels=True,
        allow_bounded_clock=True, native_refresh_reachable=True)
    assert guarded[0] < coarse[0]  # No unclosed REF domain may shrink row state.


def test_summary_uses_small_REF_on_bound_only_with_complete_native_source_proof():
    from types import SimpleNamespace as NS
    runtime, config, physical, _core = data = source()
    item = task("actual-IR", 0, metadata={"physical_owner": "hbm",
        "physical_memory_config": config,
        "memory_accesses": ({"physical_owner": "hbm", "resource_id": "hbm",
            "operation": "read", "byte_count": 262144},)})
    stages = (_PreparedExecutionStage("body", 0, (), ("r0",), "gpu0", 0., (item,)),)
    old = summarize_native_stages(stages, NS(), physical, allow_bounded_clock=True)
    new = summarize_native_stages(stages, NS(), physical, allow_bounded_clock=True,
                                  native_runtime=runtime)
    owner = new["owner_accesses"]["hbm"]
    assert owner["native_refresh_source_proof"]["qualified"]
    assert owner["native_REF_on_service_bounds"][0]["qualified_source_required"]
    assert new["hbm_clock_domain_limit_ns"] is not None
    assert new["refresh_active_io_serial_upper_ns"] < old["refresh_active_io_serial_upper_ns"]
    runtime.execution_control = ExecutionControl(progress_callback=lambda event: None)
    refused = summarize_native_stages(stages, NS(), physical, allow_bounded_clock=True,
                                      native_runtime=runtime)
    assert not refused["owner_accesses"]["hbm"]["native_refresh_source_proof"]["qualified"]
    assert refused["owner_accesses"]["hbm"]["native_refresh_source_proof"]["reason"]
    assert any(reason.startswith("native_refresh_source_not_closed:")
               for reason in refused["unsupported"])
    assert refused["refresh_active_io_serial_upper_ns"] == old["refresh_active_io_serial_upper_ns"]
