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


@pytest.mark.parametrize("owner", ["request", "prefix"])
def test_rejected_page_release_preserves_page_and_owner_state(owner):
    used = {"hbm0": 0}
    reject_release = {"value": True}

    def can_adjust(component, delta):
        return delta <= 0 or used[component] + delta <= 16

    def adjust(component, delta):
        if reject_release["value"] and delta < 0:
            return False
        if used[component] + delta > 16:
            return False
        used[component] += delta
        return True

    pool = DynamicKVPool(
        [KvPoolComponent("hbm0", 16)],
        page_bytes=8,
        ledger_can_adjust=can_adjust,
        ledger_adjust=adjust,
    )
    assert pool.resize("r", 1)
    page = pool.request_pages("r")[0]
    if owner == "prefix":
        pool.register_prefix("cached", [page])
        pool.release("r")
        release = pool.release_prefix
        owner_pages = pool.prefix_pages
        owner_key = "cached"
        bindings = set()
    else:
        release = pool.release
        owner_pages = pool.request_pages
        owner_key = "r"
        bindings = {"r"}

    with pytest.raises(RuntimeError, match="rejected page release"):
        release(owner_key)

    assert owner_pages(owner_key) == (page,)
    assert pool.pages() == (page,)
    assert page.ref_count == 1
    assert pool._bindings[page.logical_page_id] == bindings
    assert dict(pool._owned_bytes) == {"hbm0": 8}
    assert used == {"hbm0": 8}

    reject_release["value"] = False
    release(owner_key)
    assert pool.pages() == ()
    assert owner_pages(owner_key) == ()
    assert dict(pool._owned_bytes) == {}
    assert used == {"hbm0": 0}


def test_partial_release_retry_does_not_drop_shared_page_twice():
    used = {"hbm0": 0}
    reject_release = {"value": True}

    def can_adjust(component, delta):
        return delta <= 0 or used[component] + delta <= 32

    def adjust(component, delta):
        if reject_release["value"] and delta < 0:
            return False
        used[component] += delta
        return True

    pool = DynamicKVPool([KvPoolComponent("hbm0", 32)], page_bytes=8,
                         ledger_can_adjust=can_adjust, ledger_adjust=adjust)
    assert pool.resize("r", 2)
    first, second = pool.request_pages("r")
    pool.register_prefix("cached", [first])

    with pytest.raises(RuntimeError, match="rejected page release"):
        pool.release("r")

    assert pool.request_pages("r") == (second,)
    assert pool.prefix_pages("cached") == (first,)
    assert first.ref_count == 1
    assert used == {"hbm0": 16}

    reject_release["value"] = False
    pool.release("r")
    assert pool.request_pages("r") == ()
    assert pool.prefix_pages("cached") == (first,)
    assert used == {"hbm0": 8}


def test_partial_prefix_release_retry_keeps_unprocessed_pages():
    used = {"hbm0": 0}
    negative_calls = {"value": 0}
    reject_after_first = {"value": True}

    def can_adjust(component, delta):
        return delta <= 0 or used[component] + delta <= 16

    def adjust(component, delta):
        if delta < 0:
            negative_calls["value"] += 1
            if reject_after_first["value"] and negative_calls["value"] >= 2:
                return False
        used[component] += delta
        return True

    pool = DynamicKVPool([KvPoolComponent("hbm0", 16)], page_bytes=8,
                         ledger_can_adjust=can_adjust, ledger_adjust=adjust)
    assert pool.resize("r", 2)
    first, second = pool.request_pages("r")
    pool.register_prefix("cached", [first, second])
    pool.release("r")

    with pytest.raises(RuntimeError, match="rejected page release"):
        pool.release_prefix("cached")

    assert pool.pages() == (second,)
    assert pool.prefix_pages("cached") == (second,)
    assert used == {"hbm0": 8}

    reject_after_first["value"] = False
    pool.release_prefix("cached")
    assert pool.pages() == ()
    assert used == {"hbm0": 0}


def test_partial_resize_shrink_retry_keeps_shared_prefix_reference():
    used = {"hbm0": 0}
    reject_release = {"value": True}

    def can_adjust(component, delta):
        return delta <= 0 or used[component] + delta <= 16

    def adjust(component, delta):
        if reject_release["value"] and delta < 0:
            return False
        used[component] += delta
        return True

    pool = DynamicKVPool([KvPoolComponent("hbm0", 16)], page_bytes=8,
                         ledger_can_adjust=can_adjust, ledger_adjust=adjust)
    assert pool.resize("r", 2)
    first, second = pool.request_pages("r")
    pool.register_prefix("cached", [first])

    with pytest.raises(RuntimeError, match="rejected page release"):
        pool.resize("r", 0)

    assert pool.request_pages("r") == (second,)
    assert pool.prefix_pages("cached") == (first,)
    assert used == {"hbm0": 16}

    reject_release["value"] = False
    assert pool.resize("r", 0)
    assert pool.request_pages("r") == ()
    assert pool.prefix_pages("cached") == (first,)
    assert used == {"hbm0": 8}


def test_unpin_page_rejection_keeps_pin_for_retry():
    used = {"hbm0": 0}
    reject_release = {"value": True}

    def can_adjust(component, delta):
        return delta <= 0 or used[component] + delta <= 8

    def adjust(component, delta):
        if reject_release["value"] and delta < 0:
            return False
        used[component] += delta
        return True

    pool = DynamicKVPool([KvPoolComponent("hbm0", 8)], page_bytes=8,
                         ledger_can_adjust=can_adjust, ledger_adjust=adjust)
    assert pool.resize("r", 1)
    page = pool.request_pages("r")[0]
    pool.pin("r")
    pool.release("r")

    with pytest.raises(RuntimeError, match="rejected page release"):
        pool.unpin_page(page.logical_page_id)
    assert pool.page(page.logical_page_id).pinned
    assert used == {"hbm0": 8}

    reject_release["value"] = False
    assert pool.unpin_page(page.logical_page_id)
    assert pool.pages() == ()
    assert used == {"hbm0": 0}


def test_evict_skips_pages_with_multiple_prefix_references():
    pool = DynamicKVPool([KvPoolComponent("hbm0", 8)], page_bytes=8)
    assert pool.resize("r", 1)
    page = pool.request_pages("r")[0]
    pool.register_prefix("a", [page])
    pool.register_prefix("b", [page])
    pool.release("r")

    assert page.ref_count == 2
    assert pool.evict(8) == 0
    assert pool.page(page.logical_page_id) is page


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
            KvPoolComponent(
                "ssd0",
                64,
                kind="ssd",
                tier="ssd",
                active=False,
                read_bandwidth_gbps=100,
                write_bandwidth_gbps=100,
            ),
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


def test_storage_transfer_rejects_unknown_service_direction():
    pool = DynamicKVPool(
        [
            KvPoolComponent("hbm0", 64),
            KvPoolComponent("ssd0", 64, kind="ssd", active=False),
        ],
        page_bytes=8,
    )
    assert pool.resize("r", 1, compatible_components={"default": ["hbm0"]})
    page = pool.request_pages("r")[0]

    with pytest.raises(KvPoolUnsupported, match="ssd0.*write bandwidth"):
        pool.offload_page(page.logical_page_id, "ssd0")

    readable_storage = DynamicKVPool(
        [
            KvPoolComponent("hbm0", 64),
            KvPoolComponent(
                "ssd0",
                64,
                kind="ssd",
                active=False,
                write_bandwidth_gbps=100,
            ),
        ],
        page_bytes=8,
    )
    assert readable_storage.resize(
        "r", 1, compatible_components={"default": ["hbm0"]}
    )
    page = readable_storage.request_pages("r")[0]
    assert readable_storage.offload_page(page.logical_page_id, "ssd0")
    with pytest.raises(KvPoolUnsupported, match="ssd0.*read bandwidth"):
        readable_storage.restore_page(page.logical_page_id, "hbm0")


def test_transfer_without_topology_prices_source_read_and_target_write():
    """A no-topology move still models both endpoint service directions."""

    pool = DynamicKVPool(
        [
            KvPoolComponent(
                "slow-read",
                128,
                read_bandwidth_gbps=100,
                write_bandwidth_gbps=1000,
                latency_ns=5,
            ),
            KvPoolComponent(
                "slow-write",
                128,
                read_bandwidth_gbps=1000,
                write_bandwidth_gbps=10,
                latency_ns=7,
            ),
        ],
        page_bytes=100,
    )
    assert pool.resize("r", 1, compatible_components={"default": ["slow-read"]})
    page = pool.request_pages("r")[0]
    assert pool.migrate_page(page.logical_page_id, "slow-write")

    # 5 + 8*100/100 (source read) + 7 + 8*100/10 (target write).
    assert pool.events[-1].duration_ns == pytest.approx(100.0)


def test_topology_transfer_adds_endpoint_service_to_link_service():
    """A fast link cannot erase a slow source/target medium."""

    class FastLinkTopology:
        @dataclass(frozen=True)
        class FastHop:
            def transfer_ns(self, byte_count):
                return 3 + 8 * byte_count / 1000

        def route(self, source, target, byte_count, **kwargs):
            return (self.FastHop(),)

    pool = DynamicKVPool(
        [
            KvPoolComponent(
                "media-a",
                128,
                read_bandwidth_gbps=1000,
                write_bandwidth_gbps=1000,
                latency_ns=2,
            ),
            KvPoolComponent(
                "media-b",
                128,
                read_bandwidth_gbps=1000,
                write_bandwidth_gbps=100,
                latency_ns=4,
            ),
        ],
        page_bytes=100,
        topology=FastLinkTopology(),
    )
    assert pool.resize("r", 1, compatible_components={"default": ["media-a"]})
    page = pool.request_pages("r")[0]
    assert pool.migrate_page(page.logical_page_id, "media-b")

    # Endpoint service: (2 + 0.8) + (4 + 8); link service: 3 + 8*100/1000.
    assert pool.events[-1].duration_ns == pytest.approx(18.6)


def test_topology_transfer_keeps_slow_link_in_the_critical_path():
    """Fast media must not hide a slow declared route bandwidth."""

    class SlowLinkTopology:
        @dataclass(frozen=True)
        class SlowHop:
            def transfer_ns(self, byte_count):
                return 1 + 8 * byte_count / 2

        def route(self, source, target, byte_count, **kwargs):
            return (self.SlowHop(),)

    pool = DynamicKVPool(
        [
            KvPoolComponent("src", 128, read_bandwidth_gbps=10000, latency_ns=1),
            KvPoolComponent("dst", 128, write_bandwidth_gbps=10000, latency_ns=1),
        ],
        page_bytes=100,
        topology=SlowLinkTopology(),
    )
    assert pool.resize("r", 1, compatible_components={"default": ["src"]})
    page = pool.request_pages("r")[0]
    assert pool.migrate_page(page.logical_page_id, "dst")

    # Hop.transfer_ns is 1 + 8*100/2 = 401 ns; endpoints add 0.08 + 0.08
    # plus their fixed latencies.
    assert pool.events[-1].duration_ns == pytest.approx(403.16)


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

def test_page_allocator_exports_real_cache_accesses():
    pool = DynamicKVPool([KvPoolComponent('hbm0',128,kind='hbm')],tokens_per_page=4,page_bytes=16)
    assert pool.resize('request',2)
    accesses=pool.cache_accesses('request',first_token=3,token_count=3)
    assert [(owner,a.offset_bytes,a.size_bytes) for owner,a in accesses]==[('hbm0',12,4),('hbm0',0,8)]
    old_id=accesses[0][1].buffer_id
    pool.release('request')
    assert pool.resize('request',2)
    assert pool.cache_accesses('request',first_token=3,token_count=3)[0][1].buffer_id!=old_id

def test_capacity_probe_preserves_pages_prefixes_and_cache_identity():
    from copy import deepcopy
    pool = DynamicKVPool([KvPoolComponent('hbm0', 64)], tokens_per_page=4, page_bytes=16)
    assert pool.resize('r', 2)
    pool.register_prefix('prefix', pool.request_pages('r'))
    before = deepcopy(pool.__dict__)
    identities = pool.cache_accesses('r', first_token=0, token_count=8)
    for size, expected in [(0, True), (1, True), (3, True), (5, False)]:
        assert pool.can_resize('r', size) is expected
        assert pool.cache_accesses('r', first_token=0, token_count=8) == identities
        for key in ('_pages', '_requests', '_prefixes', '_bindings', '_owned_bytes',
                    '_page_counter', '_clock', '_events', '_last_error'):
            assert pool.__dict__[key] == before[key]
    assert pool.ledger.used_bytes == before['_ledger'].used_bytes


def test_capacity_probe_uses_cumulative_external_capacity_without_writes():
    writes = []
    used = {'hbm0': 0}
    def adjust(component, delta):
        writes.append(delta)
        used[component] += delta
        return True
    pool = DynamicKVPool([KvPoolComponent('hbm0', 128)], page_bytes=16,
                         ledger_can_adjust=lambda c, d: used[c] + d <= 32,
                         ledger_adjust=adjust)
    assert pool.resize('r', 1)
    assert pool.can_resize('r', 2)
    assert not pool.can_resize('r', 3)
    assert writes == [16]
    assert used == {'hbm0': 16}


def test_cache_identity_survives_aliases_but_not_round_trip_migration():
    pool = DynamicKVPool([KvPoolComponent('hbm0', 64), KvPoolComponent('hbm1', 64)],
                         tokens_per_page=4, page_bytes=16)
    assert pool.resize('r', 1)
    page = pool.request_pages('r')[0]
    owner = page.owner_component
    original = pool.cache_accesses('r', first_token=0, token_count=4)
    pool.register_prefix('prefix', [page])
    assert pool.resize('alias', 1, prefix_key='prefix')
    assert pool.cache_accesses('alias', first_token=0, token_count=4) == original
    assert pool.migrate(page.logical_page_id, 'hbm1' if owner == 'hbm0' else 'hbm0')
    assert pool.migrate(page.logical_page_id, owner)
    assert pool.cache_accesses('r', first_token=0, token_count=4) != original
    assert page.allocation_generation == 2
    assert pool.migrate(page.logical_page_id, owner)
    assert page.allocation_generation == 2


def test_independent_pools_do_not_alias_and_empty_ranges_are_validated():
    pools = [DynamicKVPool([KvPoolComponent('hbm0', 64)], tokens_per_page=4, page_bytes=16)
             for _ in range(2)]
    for pool in pools:
        assert pool.cache_accesses('r', first_token=0, token_count=0) == ()
        with pytest.raises(ValueError):
            pool.cache_accesses('r', first_token=0, token_count=1)
        assert pool.resize('r', 1)
    assert pools[0].cache_accesses('r', first_token=0, token_count=4) != pools[1].cache_accesses('r', first_token=0, token_count=4)
