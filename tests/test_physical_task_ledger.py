"""The physical task ledger keeps ordered completion and erase status."""
from heterollm_sim._physical_task import PhysicalTaskLedger
from heterollm_sim.memory_types import Operation


def test_ledger_retains_ordered_completion_and_logical_erase_status():
    ledger = PhysicalTaskLedger()
    ledger.accept(Operation.ERASE, 9.)
    ledger.accept(Operation.ERASE, 3.)
    assert ledger.completion_ns == 9.
    assert ledger.all_erases
    ledger.accept(Operation.WRITE, 12.)
    assert ledger.completion_ns == 12.
    assert not ledger.all_erases
