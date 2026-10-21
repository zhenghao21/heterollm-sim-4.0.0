"""Cold selection avoids full-bank packing without changing physical execution."""
from dataclasses import replace

import pytest

from heterollm_sim._compiled_hbm import execute_hbm_numeric
from heterollm_sim._hbm_control import _Controller
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.dram_core import DramCore
from heterollm_sim.execution_control import ExecutionCancelledError, ExecutionControl
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_presets import resolve_physical_config
from test_hbm_compiled_control import assert_core_state_equal


def eight_stack_hbm4():
    # Same native geometry and directional budgets as the paper Table-I input;
    # reference timing is not claimed to be a calibrated vendor speed grade.
    return resolve_physical_config("hbm4_reference", overrides={
        "stacks": 8, "dies_per_stack": 16, "die_density_gbits": 32,
        "capacity_bytes": 288_000_000_000,
        "interface_bandwidth_gb_s": 12800.,
        "read_bandwidth_gb_s": 12800., "write_bandwidth_gb_s": 12800.,
    })


def physical_result(result):
    return replace(result, counters={key: value for key, value in result.counters.items()
                                    if key != "compiled_hbm_float64"})


def force(core, request, numeric):
    return core._submit_validated(request, execute=(
        lambda accepted: execute_hbm_numeric(core, accepted)) if numeric else (
        lambda accepted: _Controller(core, accepted).execute()))


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("address,size,compiled", [
    (0, 4096, False), (1, 4095, False),
    (1, 4096, True), (0, 8192, True), (0, 32768, True),
])
def test_cold_selection_matches_reference_and_numeric_state(operation, address, size, compiled):
    config = eight_stack_hbm4()
    assert config.lane_count * config.bank_count == 32768
    selected, reference, numeric = (DramCore(config, capture_details=False) for _ in range(3))
    request = AccessRequest("cold", operation, address, size)
    actual = selected.submit(request)
    assert actual.counters.get("compiled_hbm_float64", False) is compiled
    assert physical_result(actual) == physical_result(force(reference, request, False))
    assert physical_result(actual) == physical_result(force(numeric, request, True))
    assert_core_state_equal(selected, reference)
    assert_core_state_equal(selected, numeric)


@pytest.mark.parametrize("first_operation", ["read", "write"])
def test_reference_prefix_then_large_numeric_preserves_every_following_state(first_operation):
    config = eight_stack_hbm4()
    selected, numeric = (DramCore(config, capture_details=False) for _ in range(2))
    for core in (selected, numeric):
        # Unrelated/self aliases are valid numeric input and remain authoritative.
        core.timeline.resource_aliases = {"kernel:port": "kernel:wire", "dram:data:0": "dram:data:0"}
    row_span = config.lane_count * config.bank_count * config.row_bytes
    requests = (
        AccessRequest("cold", first_operation, 0, 4096),
        AccessRequest("partial", "write", 17, 3000, 100.),
        AccessRequest("large", "read", 65536, 32768, 40000.),
        AccessRequest("row-conflict", "write", row_span + 1, 8192, 50000.),
        AccessRequest("after-idle", "read", 0, 4096, 1.e6),
    )
    for request in requests:
        actual = selected.submit(request)
        expected = force(numeric, request, True)
        assert physical_result(actual) == physical_result(expected)
        assert_core_state_equal(selected, numeric)


@pytest.mark.parametrize("guard", ["transaction", "resident"])
def test_small_request_keeps_transaction_and_resident_numeric_paths(guard):
    core = DramCore(eight_stack_hbm4(), capture_details=False)
    token = core.begin_transaction() if guard == "transaction" else None
    core._hbm_numeric_enabled = guard == "resident"
    result = core.submit(AccessRequest("small", "read", 0, 4096))
    assert result.counters["compiled_hbm_float64"] is True
    if token is not None:
        core.commit_transaction(token)
    else:
        packed = core._hbm_numeric_state
        assert packed is not None
        result = core.submit(AccessRequest("small-again", "write", 0, 4096))
        assert result.counters["compiled_hbm_float64"] is True
        assert core._hbm_numeric_state is packed


def test_cancellation_after_reference_prefix_rolls_back_then_retries_numeric():
    runtime = PhysicalRuntimeContext(eight_stack_hbm4(), "hbm", capture_details=False)
    core = runtime.runtimes["hbm"].core
    # A public controller view ends numeric ownership by the existing contract.
    assert core._hbm_control_state == {}
    cold = core.submit(AccessRequest("cold", "read", 0, 4096))
    assert "compiled_hbm_float64" not in cold.counters
    before = runtime.snapshot()
    transaction = runtime.begin_transaction()
    request = AccessRequest("cancelled", "write", 65536, 32768, 50000.)
    with pytest.raises(ExecutionCancelledError):
        assert core.submit(request).counters["compiled_hbm_float64"] is True
        ExecutionControl(cancellation_callback=lambda: True).raise_if_cancelled()
    runtime.restore(transaction)
    assert runtime.snapshot() == before
    transaction = runtime.begin_transaction()
    actual = core.submit(request)
    runtime.commit_transaction(transaction)
    oracle = PhysicalRuntimeContext(eight_stack_hbm4(), "hbm", capture_details=False)
    oracle.restore(before)
    expected = oracle.runtimes["hbm"].core.submit(request)
    assert actual == expected
    assert runtime.snapshot() == oracle.snapshot()
