"""Current runtime IR names and its aggregate scheduler handoff."""
import pytest

from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.runtime import ControlPlaneRuntime
from heterollm_sim.runtime_ir import (
    ControllerKind, ControllerTransactionBatch, ExpansionResult,
    InstructionBatch, InstructionClass, RuntimeAction, RuntimePhase,
    decode_aggregate_metadata,
)
from heterollm_sim.runtime_lowering import RuntimeDispatcher, build_initial_control_tasks
from heterollm_sim.runtime_state import RuntimeState


@pytest.mark.parametrize("enum_type,removed", [
    (RuntimePhase, ("PLACEMENT", "ALLOCATOR", "NVME", "IOMMU_DMA_PCIE",
                    "GPU_COMMAND_PROCESSOR", "MMU_TLB_L2_VRAM")),
    (RuntimeAction, ("ALLOCATION", "GPU_COMMAND_PROCESSOR", "MMU_TLB", "CPU_COMPLETION")),
])
def test_runtime_enums_have_only_canonical_names(enum_type, removed):
    assert len(enum_type.__members__) == len(enum_type)
    for name in removed:
        with pytest.raises(KeyError):
            enum_type[name]
    assert RuntimePhase.ALLOCATION == 20
    assert RuntimeAction.ALLOCATE.value == "allocate"


def test_aggregate_objects_do_not_expose_duplicate_properties():
    instruction = InstructionBatch("instructions", InstructionClass.CONTROL, 7)
    controller = ControllerTransactionBatch("transactions", ControllerKind.CPU, 3)
    expansion = ExpansionResult()
    assert instruction.instruction_count == 7
    assert (controller.controller, controller.transaction_count) == (ControllerKind.CPU, 3)
    assert expansion.tasks == ()
    assert expansion.delta.allocations == {}
    for value, names in ((instruction, ("count",)),
                         (controller, ("count", "controller_kind")),
                         (expansion, ("new_tasks", "state_delta"))):
        for name in names:
            with pytest.raises(AttributeError):
                getattr(value, name)


def test_current_aggregate_root_reaches_kernel_completion_and_expansion():
    state = RuntimeState()
    context = {"request_id": "r", "required_bytes": 128, "byte_count": 16,
               "instruction_count": 7, "transaction_count": 3,
               "placement": "gpu0"}
    root, = build_initial_control_tasks(context=context)
    task = root.to_task_spec()
    instruction, controller = decode_aggregate_metadata(task)
    assert instruction == root.instruction_batch
    assert controller == root.controller_batch
    kernel = UnifiedEventKernel.from_closed_graph((task,))
    completion = kernel.step_completion()
    assert completion.instruction_batch.instruction_count == 7
    assert completion.controller_batch.transaction_count == 3
    expansion = RuntimeDispatcher().dispatch(completion, state, context)
    assert expansion.tasks[0].action is RuntimeAction.PLACEMENT_DECISION
    assert expansion.delta.capacity_reservations == {"vram": 128}
    result = ControlPlaneRuntime().run((root,), context)
    assert result.state.completed_requests == {"r"}
    assert result.completions[-1].action is RuntimeAction.CPU_COMPLETE
    assert result.state.allocations == {"r@gpu0": 128}


def test_action_configuration_uses_current_string_values_and_typed_keys():
    profile = {"service_ns": {RuntimeAction.CAPACITY_CHECK: 4.0},
               "resources": {"capacity_check": ("cpu.audit",)},
               "transaction_count": {RuntimeAction.CAPACITY_CHECK: 5}}
    root, = build_initial_control_tasks(profile=profile,
                                       context={"request_id": "r", "placement": "gpu0"})
    assert root.demands[0].resource_id == "cpu.audit"
    assert root.demands[0].service_ns == 4.0
    assert root.controller_batch.transaction_count == 5
