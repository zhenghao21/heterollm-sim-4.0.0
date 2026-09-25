from dataclasses import dataclass

import pytest

from heterollm_sim.kv_pool import (
    DynamicKVPool,
    KvPoolComponent,
    KvPoolUnsupported,
)


def components(*items):
    return {item.component_id: item for item in items}


def test_resize_assigns_real_page_owners_and_rolls_back_atomically():
    pool = DynamicKVPool(
        components(
            KvPoolComponent("hbm0", 16, kind="hbm", read_bandwidth_gbps=100, write_bandwidth_gbps=100),
            KvPoolComponent("hbm1", 32, kind="hbm", read_bandwidth_gbps=80, write_bandwidth_gbps=80),
        ),
        tokens_per_page=4,
        page_bytes=16,
    )

    assert pool.resize("req-a", 3)
    pages = pool.request_pages("req-a")
    assert len(pages) == 3
    # The first two pages fit on hbm1; the final page is placed on hbm0.
    assert [page.owner_component for page in pages] == ["hbm1", "hbm0", "hbm1"]
    assert pool.component_stats()["hbm1"]["pool_bytes"] == 32
    assert pool.component_stats()["hbm0"]["pool_bytes"] == 16

    # hbm0+hbm1 have only one page left in aggregate.  A two-page growth must
    # leave the original allocation and physical bytes untouched.
    assert not pool.resize("req-a", 5)
    assert len(pool.request_pages("req-a")) == 3
    assert pool.component_stats()["hbm1"]["pool_bytes"] == 32
    assert pool.component_stats()["hbm0"]["pool_bytes"] == 16
    assert pool.last_error and "capacity" in pool.last_error


def test_prefix_reuse_refcounts_and_pinned_page_lifetime():
    pool = DynamicKVPool(
        [KvPoolComponent("hbm0", 128, kind="hbm")],
        tokens_per_page=2,
        page_bytes=8,
    )
    assert pool.resize("producer", 2)
    source_pages = pool.request_pages("producer")
    pool.register_prefix("hello", source_pages)
    assert all(page.ref_count == 2 for page in source_pages)

    pool.release("producer")
    assert len(pool.prefix_pages("hello")) == 2
    assert all(page.ref_count == 1 for page in pool.prefix_pages("hello"))

    assert pool.resize("consumer", 2, prefix_key="hello")
    reused = pool.request_pages("consumer")
    assert [page.logical_page_id for page in reused] == [page.logical_page_id for page in source_pages]
    assert all(page.ref_count == 2 for page in reused)
    pool.pin("consumer")
    pool.release("consumer")
    # The prefix keeps the pages alive; a pin protects them even after the
    # prefix is dropped until an explicit unpin.
    pool.release_prefix("hello")
    assert len(pool.pages()) == 2
    for page in tuple(pool.pages()):
        pool.unpin_page(page.logical_page_id)
    assert len(pool.pages()) == 0


def test_external_ledger_callbacks_receive_per_component_allocations():
    calls = []
    used = {"hbm0": 0, "hbm1": 0}

    def can_adjust(component, delta):
        calls.append(("can", component, delta))
        return used[component] + delta <= 16

    def adjust(component, delta):
        calls.append(("adjust", component, delta))
        if used[component] + delta > 16:
            return False
        used[component] += delta
        return True

    pool = DynamicKVPool(
        [
            KvPoolComponent("hbm0", 16, read_bandwidth_gbps=50, write_bandwidth_gbps=50),
            KvPoolComponent("hbm1", 16, read_bandwidth_gbps=100, write_bandwidth_gbps=100),
        ],
        page_bytes=8,
        ledger_can_adjust=can_adjust,
        ledger_adjust=adjust,
    )
    assert pool.resize("r", 2)
    assert used == {"hbm0": 8, "hbm1": 8}
    assert {entry[1] for entry in calls if entry[0] == "adjust"} == {"hbm0", "hbm1"}
    pool.release("r")
    assert used == {"hbm0": 0, "hbm1": 0}


@dataclass(frozen=True)
class Hop:
    delay: float

    def transfer_ns(self, byte_count):
        return self.delay + byte_count * 2


class Topology:
    def route(self, source, target, byte_count, **kwargs):
        if {source, target} == {"hbm0", "hbm1"}:
            return (Hop(10),)
        if "ssd0" in {source, target}:
            return (Hop(30), Hop(20))
        return ()


def test_migration_offload_restore_records_topology_timing():
    pool = DynamicKVPool(
        [
            KvPoolComponent("hbm0", 64, kind="hbm", tier="hbm"),
            KvPoolComponent("hbm1", 64, kind="hbm", tier="hbm"),
            KvPoolComponent("ssd0", 64, kind="ssd", tier="ssd", active=False),
        ],
        page_bytes=8,
        topology=Topology(),
    )
    assert pool.resize("r", 1, compatible_components={"default": ["hbm0"]})
    page = pool.request_pages("r")[0]
    assert pool.migrate_page(page.logical_page_id, "hbm1")
    assert pool.offload_page(page.logical_page_id, "ssd0")
    assert page.resident_tier == "ssd"
    assert pool.restore_page(page.logical_page_id, "hbm0")
    assert page.owner_component == "hbm0"
    assert [event.kind for event in pool.events] == ["migration", "offload", "restore"]
    assert all(event.duration_ns > 0 for event in pool.events)


def test_layer_compatibility_and_cross_machine_boundary_are_explicit():
    pool = DynamicKVPool(
        [KvPoolComponent("hbm0", 16), KvPoolComponent("hbm1", 16)],
        page_bytes_by_layer={"layer0": 4},
    )
    assert pool.resize("r", 1, layer_group="layer0", compatible_components={"layer0": ["hbm1"]})
    assert pool.request_pages("r")[0].owner_component == "hbm1"

    with pytest.raises(KvPoolUnsupported, match="cross-machine"):
        DynamicKVPool(
            [
                KvPoolComponent("hbm0", 16, machine_id="host-a"),
                KvPoolComponent("hbm1", 16, machine_id="host-b"),
            ]
        )


def test_resize_accepts_logical_token_target():
    pool = DynamicKVPool([KvPoolComponent("hbm0", 64)], tokens_per_page=4, page_bytes=4)
    assert pool.resize("tokens", target_tokens=9)
    assert len(pool.request_pages("tokens")) == 3
    assert pool.request_pages("tokens")[-1].token_end == 9
