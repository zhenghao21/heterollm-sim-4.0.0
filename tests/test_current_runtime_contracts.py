"""Control capacity, command payloads and data transfers remain distinct."""
from dataclasses import replace

import pytest

from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.runtime import ControlPlaneRuntime
from heterollm_sim.runtime_lowering import (
    RuntimeDispatcher, RuntimeDispatchError, build_initial_control_tasks,
)
from heterollm_sim.runtime_state import RuntimeState


@pytest.mark.parametrize("page_cached", (False, True))
def test_storage_and_pcie_transitions_restore_the_declared_control_payload(page_cached):
    state = RuntimeState(page_cache={"pages"} if page_cached else set())
    runtime = ControlPlaneRuntime({"control_byte_count": 64}, state)
    result = runtime.run(None, {"request_id": "request", "placement": "gpu0",
        "required_bytes": 4096, "storage_byte_count": 2048, "transfer_byte_count": 1024,
        "weight_key": "weights", "page_key": "pages"})
    by_action = {completion.action.value: completion for completion in result.completions}
    assert result.state.allocations == {"request@gpu0": 4096}
    assert ("nvme_read" in by_action) is not page_cached
    if not page_cached:
        assert by_action["nvme_read"].task.demands[0].bytes_moved == 2048
    for action in ("iommu_translate", "dma_map", "pcie_transfer"):
        assert by_action[action].task.demands[0].bytes_moved == 1024
    for action in ("capacity_check", "batch_schedule", "operator_schedule", "cpu_complete"):
        assert by_action[action].task.demands[0].bytes_moved == 64
    assert all(completion.payload["required_bytes"] == 4096
               and completion.payload["control_byte_count"] == 64
               for completion in result.completions)


@pytest.mark.parametrize("with_current_field", (False, True))
def test_initial_context_rejects_the_retired_byte_count_input(with_current_field):
    context = {"request_id": "request", "byte_count": 8192}
    if with_current_field:
        context["control_byte_count"] = 64
    with pytest.raises(RuntimeDispatchError, match="request.byte_count is removed"):
        build_initial_control_tasks(context=context)


@pytest.mark.parametrize("missing", ("required_bytes", "control_byte_count"))
def test_dispatch_does_not_infer_missing_capacity_or_control_size_from_stage_bytes(missing):
    root, = build_initial_control_tasks(context={"request_id": "request",
        "required_bytes": 4096, "control_byte_count": 64})
    payload = {key: value for key, value in root.payload.items() if key != missing}
    payload["byte_count"] = 999999
    task = replace(root, payload=payload).to_task_spec()
    completion = UnifiedEventKernel.from_closed_graph((task,)).step_completion()
    with pytest.raises(RuntimeDispatchError, match="explicit " + missing):
        RuntimeDispatcher().dispatch(completion, RuntimeState(), {})


def test_unspecified_control_traffic_does_not_inherit_reserved_capacity():
    root, = build_initial_control_tasks(context={"required_bytes": 4096})
    assert root.payload["control_byte_count"] == 0
    assert root.instruction_batch.byte_count == root.controller_batch.byte_count == 0
    assert all(demand.bytes_moved == 0 for demand in root.demands)
