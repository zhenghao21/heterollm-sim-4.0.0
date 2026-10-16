"""Bulk Python/NumPy conversion preserves the full B200 memory controller."""
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest
from heterollm_sim.physical_presets import resolve_physical_config
from test_hbm_compiled_control import assert_same, assert_batch_matches_scalar, assert_core_state_equal


def b200_memory():
    return resolve_physical_config("hbm3e_reference", overrides={
        "capacity_bytes": 180_000_000_000, "stacks": 8,
        "dies_per_stack": 8, "die_density_gbits": 24,
        "interface_bandwidth_gb_s": 8000.,
    })


def test_b200_column_pack_matches_burst_controller_after_full_bank_materialization(monkeypatch):
    from heterollm_sim import _hbm_numeric

    # This oracle checks packing bitwise, independently of the explicitly
    # tested diagnostic summation rounding between channel workers.
    monkeypatch.setattr(_hbm_numeric, "_CHANNEL_WORKERS", 1)
    config = b200_memory()
    reference = DramCore(config, capture_details=False)
    numeric = DramCore(config, capture_details=False)
    row_span = config.lane_count * config.bank_count * config.row_bytes
    requests = (
        AccessRequest("all-banks", "read", 0, config.lane_count * config.bank_count * config.burst_bytes),
        AccessRequest("kv-append", "write", 65536 + 3, 8192),
        AccessRequest("new-weight-row", "read", row_span + 31, 16384),
        AccessRequest("after-deadline", "read", 0, 4096, 40000.),
    )
    for request in requests:
        assert_same(reference, numeric, request)


def test_b200_large_request_batch_preserves_each_result_and_following_access():
    config = b200_memory()
    sequential = DramCore(config, capture_details=False)
    batched = DramCore(config, capture_details=False)
    requests = tuple(AccessRequest(str(index), "write" if index % 3 == 1 else "read",
                                   index * 65536 + index, 65536, float(index % 4))
                     for index in range(18))
    expected = tuple(sequential.submit(request) for request in requests)
    actual = batched.submit_many(requests)
    assert_batch_matches_scalar(actual, expected)
    assert_core_state_equal(batched, sequential)
    following = AccessRequest("following", "write", 17, 65536, 50000.)
    assert batched.submit(following) == sequential.submit(following)
    assert_core_state_equal(batched, sequential)
