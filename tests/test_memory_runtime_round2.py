"""F2 runtime ownership and preview contract tests.

These tests intentionally use a tiny synthetic DRAM geometry.  ``price``
without a runtime is a preview and must not commit row/timeline state.  A
runtime supplied by the caller owns that state, and ``arrival_ns`` is the
request's event time rather than an implicit serial clock.
"""

from heterollm_sim.data_motion import (
    AccessKind,
    PhysicalRuntimeContext,
    endpoint_service,
    resolve_service,
)
from heterollm_sim.dram_core import DramCore
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import AccessRequest, DramConfig, NandConfig


def _config() -> DramConfig:
    return DramConfig(
        channels=1,
        banks_per_group=1,
        rows_per_bank=4,
        row_bytes=128,
        burst_bytes=64,
        open_ns=10,
        read_latency_ns=5,
        burst_interval_ns=1,
        lane_bandwidth_gb_s=64,
    )


def _component(config: DramConfig, owner: str = "dram0") -> ComponentSpec:
    return ComponentSpec(
        component_id=owner,
        kind="DDR",
        bandwidth_gbps=64,
        metadata={
            "physical_memory_config": config,
            "physical_owner": owner,
            "memory_resource_id": owner,
        },
    )


def _nand_component(config: NandConfig, owner: str = "nand0") -> ComponentSpec:
    return ComponentSpec(
        component_id=owner,
        kind="SSD",
        bandwidth_gbps=64,
        metadata={
            "physical_memory_config": config,
            "physical_owner": owner,
            "memory_resource_id": owner,
        },
    )


def _context(config, owner: str):
    return PhysicalRuntimeContext(config=config, physical_owner=owner)


def _result_signature(result):
    return tuple(
        result.get(name)
        for name in (
            "service_ns",
            "arrival_ns",
            "completion_ns",
            "row_hits",
            "row_misses",
            "row_conflicts",
            "physical_bytes",
        )
    )


def test_preview_and_endpoint_preview_do_not_commit_physical_state():
    config = _config()
    service = resolve_service(_component(config))

    first = service.price(AccessKind.READ, 64, page_offset_bytes=0)
    second = service.price(AccessKind.READ, 64, page_offset_bytes=0)
    assert _result_signature(first) == _result_signature(second)
    assert first["arrival_ns"] == second["arrival_ns"] == 0.0

    endpoint_first = endpoint_service(
        _component(config), 64, read=True, name="preview", page_offset_bytes=0
    )
    endpoint_second = endpoint_service(
        _component(config), 64, read=True, name="preview", page_offset_bytes=0
    )
    assert endpoint_first is not None and endpoint_second is not None
    assert endpoint_first.demands == endpoint_second.demands
    assert endpoint_first.metadata["physical_execution"] == endpoint_second.metadata[
        "physical_execution"
    ]


def test_interleaved_contexts_are_isolated_and_arrival_is_explicit():
    config = _config()
    service = resolve_service(_component(config, owner="shared"))
    context_a = _context(config, "shared")
    context_b = _context(config, "shared")

    a1 = service.price(
        AccessKind.READ, 64, page_offset_bytes=0, runtime=context_a, arrival_ns=0
    )
    b1 = service.price(
        AccessKind.READ, 64, page_offset_bytes=0, runtime=context_b, arrival_ns=0
    )
    a2 = service.price(
        AccessKind.READ, 64, page_offset_bytes=128, runtime=context_a, arrival_ns=100
    )
    b2 = service.price(
        AccessKind.READ, 64, page_offset_bytes=128, runtime=context_b, arrival_ns=100
    )

    assert _result_signature(a1) == _result_signature(b1)
    assert _result_signature(a2) == _result_signature(b2)
    assert a1["arrival_ns"] == b1["arrival_ns"] == 0
    assert a2["arrival_ns"] == b2["arrival_ns"] == 100


def test_endpoint_main_entry_can_commit_to_an_explicit_context():
    config = _config()
    component = _component(config)
    context = _context(config, "dram0")
    first = endpoint_service(
        component,
        64,
        read=True,
        name="runtime",
        page_offset_bytes=0,
        runtime=context,
        arrival_ns=0,
    )
    second = endpoint_service(
        component,
        64,
        read=True,
        name="runtime",
        page_offset_bytes=128,
        runtime=context,
        arrival_ns=100,
    )
    assert first is not None and second is not None
    assert first.metadata["physical_execution"]["arrival_ns"] == 0
    assert second.metadata["physical_execution"]["arrival_ns"] == 100
    assert second.metadata["physical_execution"]["completion_ns"] > 100


def test_single_and_split_core_batches_agree_with_explicit_arrivals():
    config = _config()
    requests = (
        AccessRequest("r1", "read", 0, 64, arrival_ns=0),
        AccessRequest("r2", "read", 128, 64, arrival_ns=100),
    )
    single = DramCore(config).execute_batch(requests)
    split_core = DramCore(config)
    split = split_core.execute_batch(requests[:1]) + split_core.execute_batch(requests[1:])
    assert [r.completion_ns for r in single] == [r.completion_ns for r in split]
    assert [r.counters for r in single] == [r.counters for r in split]


def test_physical_resource_is_emitted_once_per_endpoint():
    endpoint = endpoint_service(
        _component(_config()), 64, read=True, name="once", page_offset_bytes=0
    )
    assert endpoint is not None
    assert len(endpoint.demands) == 1
    assert endpoint.demands[0].resource_id == "dram0"
    assert endpoint.demands[0].bytes_moved > 0


def test_nand_single_batch_and_split_batch_share_one_context_timeline():
    config = NandConfig(
        channels=1,
        luns_per_die=1,
        planes_per_lun=1,
        blocks_per_plane=2,
        pages_per_block=2,
        page_bytes=256,
        host_granularity_bytes=256,
        host_bandwidth_gb_s=1024,
        internal_bandwidth_gb_s=1024,
        page_read_ns=10,
        page_program_ns=20,
    )
    service = resolve_service(_nand_component(config))
    rows = (
        {"request_id": "a", "operation": "read", "address": 0, "byte_count": 256, "arrival_ns": 0},
        {"request_id": "b", "operation": "read", "address": 256, "byte_count": 256, "arrival_ns": 100},
    )
    batch = service.price_batch(rows, state=_context(config, "nand0"))
    context = _context(config, "nand0")
    first = service.price_batch(rows[:1], runtime=context)
    second = service.price_batch(rows[1:], runtime=context)
    assert batch["end_ns"] == second["end_ns"]
    assert [item.completion_ns for item in batch["requests"]] == [
        first["requests"][0].completion_ns,
        second["requests"][0].completion_ns,
    ]
