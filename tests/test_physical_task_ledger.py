"""Physical instructions keep task atomicity without logical HBF results."""
from dataclasses import replace

import pytest

from heterollm_sim._physical_task import PhysicalTaskLedger
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.memory_types import Operation
from test_hbf_staging_runtime import setup, task


def _two_accesses(config, binding, first="write", second="write"):
    initial = task(config, binding, first)
    access = initial.metadata["memory_accesses"][0]
    return replace(initial, metadata={**initial.metadata, "memory_accesses": (
        access, {**access, "operation": second, "offset_bytes": 256},
    )})


def test_ledger_retains_ordered_completion_and_logical_erase_status():
    ledger = PhysicalTaskLedger()
    ledger.accept(Operation.ERASE, 9.)
    ledger.accept(Operation.ERASE, 3.)
    assert ledger.completion_ns == 9.
    assert ledger.all_erases
    ledger.accept(Operation.WRITE, 12.)
    assert ledger.completion_ns == 12.
    assert not ledger.all_erases


def test_staged_task_constructs_only_physical_requests(monkeypatch):
    import heterollm_sim.memory_types as memory_types
    import heterollm_sim._io_execution as staging
    # Controller modules must bind their real request type before the test
    # rejects requests constructed by the data-motion acceptance adapter.
    import heterollm_sim.dram_core  # noqa: F401

    config, binding = setup()
    instruction = _two_accesses(config, binding)
    runtime = PhysicalRuntimeContext(capture_details=True)
    io_identities, request_ids = [], []
    original_memory = staging._PhysicalIO._memory

    def inspect_memory(io, owner, cfg, operation, address, size, arrival):
        io_identities.append(id(io))
        return original_memory(io, owner, cfg, operation, address, size, arrival)

    original_request = staging.AccessRequest

    def inspect_request(request_id, *args, **kwargs):
        request_ids.append(request_id)
        return original_request(request_id, *args, **kwargs)

    def reject_logical_request(*_args, **_kwargs):
        raise AssertionError("logical HBF acceptance constructed a physical request")

    monkeypatch.setattr(staging._PhysicalIO, "_memory", inspect_memory)
    monkeypatch.setattr(staging, "AccessRequest", inspect_request)
    monkeypatch.setattr(memory_types, "AccessRequest", reject_logical_request)
    resolved = resolve_physical_task(instruction, runtime, 0.)
    assert len(set(io_identities)) == 1
    assert request_ids == ["hbf-staging:1", "hbf-staging:1"]
    costs = resolved.metadata["physical_execution_by_owner"]
    assert costs["host"]["logical_write_bytes"] == 512
    assert costs["to_host"]["resource_bytes"]["to_host"] == 512
    assert resolved.metadata["hbf_host_staging"]["flash"]["logical_write_bytes"] == 512


def test_staged_task_validates_all_instructions_before_physical_submission(monkeypatch):
    import heterollm_sim._io_execution as staging

    config, binding = setup()
    instruction = _two_accesses(config, binding)
    first, second = instruction.metadata["memory_accesses"]
    invalid = replace(instruction, metadata={**instruction.metadata, "memory_accesses": (
        first, {**second, "byte_count": 0},
    )})
    runtime = PhysicalRuntimeContext()
    before = runtime.snapshot()

    def forbidden_memory(*_args, **_kwargs):
        raise AssertionError("hardware submitted before every descriptor was validated")

    monkeypatch.setattr(staging._PhysicalIO, "_memory", forbidden_memory)
    with pytest.raises(ValueError, match="byte_count"):
        resolve_physical_task(invalid, runtime, 0.)
    assert runtime.snapshot() == before


def test_failure_after_first_staged_access_rolls_back_whole_task():
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    before = runtime.snapshot()
    # The first write succeeds; the second read reaches uninitialized bytes.
    with pytest.raises(ValueError, match="uninitialized"):
        resolve_physical_task(_two_accesses(config, binding, second="read"), runtime, 0.)
    assert runtime.snapshot() == before


@pytest.mark.parametrize("arrival", [float("nan"), float("inf")])
def test_staged_acceptance_keeps_request_time_validation_and_rollback(arrival):
    config, binding = setup()
    runtime = PhysicalRuntimeContext()
    before = runtime.snapshot()
    with pytest.raises(ValueError, match="arrival_ns must be a finite non-negative number"):
        resolve_physical_task(task(config, binding, "write"), runtime, arrival)
    assert runtime.snapshot() == before
