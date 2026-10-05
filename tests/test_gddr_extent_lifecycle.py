from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.planner import _attach_gddr_physical_task

from test_gddr_planner_access_semantics import _scenario, _task


def test_event_kernel_keeps_inferred_extent_for_shorter_followup_slice():
    scenario = _scenario()
    first = _attach_gddr_physical_task(
        _task("slice-128", 128, {"read_bytes": 128, "write_bytes": 0}), scenario
    )
    second = _attach_gddr_physical_task(
        _task("slice-64", 64, {"read_bytes": 64, "write_bytes": 0}), scenario
    )
    kernel = UnifiedEventKernel.from_closed_graph(
        (first, second), resource_capacities={"gddr0.gddr_fabric": 1}
    )
    first_event = kernel.step()
    second_event = kernel.step()
    assert first_event is not None and second_event is not None
    assert first_event.task.metadata["memory_accesses"][0]["address"] == 0
    assert second_event.task.metadata["memory_accesses"][0]["address"] == 0
    assert second_event.task.metadata["physical_execution"]["physical_read_bytes"] == 64


def test_event_kernel_keeps_explicit_full_extent_contract_fixed():
    scenario = _scenario()
    first = _attach_gddr_physical_task(
        _task(
            "fixed-128", 128, {"read_bytes": 128, "write_bytes": 0},
            input_buffer_size_bytes=256,
        ),
        scenario,
    )
    second = _attach_gddr_physical_task(
        _task(
            "fixed-64", 64, {"read_bytes": 64, "write_bytes": 0},
            input_buffer_size_bytes=256,
        ),
        scenario,
    )
    kernel = UnifiedEventKernel.from_closed_graph(
        (first, second), resource_capacities={"gddr0.gddr_fabric": 1}
    )
    assert kernel.step() is not None
    assert kernel.step() is not None
