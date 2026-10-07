from heterollm_sim.memory_types import AccessRequest, NandConfig, Operation
from heterollm_sim.nand_core import NandCore


def test_large_hbf_read_uses_exact_cycle_acceleration():
    config = NandConfig(
        kind="HBF",
        channels=4,
        blocks_per_plane=1024,
        pages_per_block=256,
        page_bytes=4096,
        host_granularity_bytes=4096,
        host_bandwidth_gb_s=10.0,
        internal_bandwidth_gb_s=20.0,
    )
    page_count = 9000
    result = NandCore(config, capture_details=False).submit(
        AccessRequest(
            request_id="large-hbf-read",
            operation=Operation.READ,
            address=0,
            byte_count=page_count * config.page_bytes,
        )
    )

    assert result.counters["accelerated_pages"] >= page_count - 16
    assert result.counters["page_count"] == page_count
    assert result.counters["pages_read"] == page_count
    assert result.counters["physical_read_bytes"] == page_count * config.page_bytes
    assert result.mapping == ()
    assert result.stages == ()
