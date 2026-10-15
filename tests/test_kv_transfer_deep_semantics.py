"""Logical KV transactions, chunk observations and shared fabric aliases."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from heterollm_sim.communication import RouteHop
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.kv_pool import KvPoolError, KvPoolTransferRequest
from heterollm_sim.kv_transfer import PhysicalKVTransferExecutor
from heterollm_sim.memory_types import make_hbf_config, make_hbm_config
from heterollm_sim.nand_core import summarize_nand_observations


def component(name, config):
    return ComponentSpec(name, name, capacity_bytes=config.capacity_bytes,
        read_bandwidth_gbps=800, write_bandwidth_gbps=80,
        metadata={"physical_memory_config": config})


def page(owner="hbf", size=49152):
    return SimpleNamespace(logical_page_id=0, owner_component=owner, bytes=size,
        allocation_generation=0, request_id="request", layer_group="kv")


def request(p, target):
    return KvPoolTransferRequest(p.logical_page_id, p.request_id, p.layer_group,
        p.owner_component, target, p.bytes, p.allocation_generation, 0, 1)


def executor(*, buffer_bytes=32768, reject=True, observe=False):
    cfg = make_hbf_config(capacity_bytes=1048576, page_bytes=16384,
        write_buffer_bytes=buffer_bytes,
        partial_page_policy="reject" if reject else "read_modify_write",
        detailed_observability=observe)
    return PhysicalKVTransferExecutor([component("hbf", cfg)])


@pytest.mark.parametrize("method", ["__call__", "prepare_async"])
@pytest.mark.parametrize("field,value", [
    ("logical_page_id", "0"), ("logical_page_id", 0.5),
    ("allocation_generation", "0"), ("allocation_generation", 0.5),
    ("bytes", "49152"), ("bytes", 49152.5),
])
def test_transfer_rejects_coerced_page_identity_and_extent(method, field, value):
    e = executor()
    p = page()
    e.bind_page(p)
    before = e.snapshot()
    invalid = replace(request(p, "hbf"), **{field: value})
    with pytest.raises(KvPoolError):
        getattr(e, method)(invalid)
    assert e.snapshot() == before


def test_late_rejected_chunk_reverts_entire_logical_page_write():
    e = executor()
    p = page()
    e.bind_page(p)
    before = e.snapshot()
    for _ in range(2):
        with pytest.raises(ValueError, match="partial NAND page write"):
            e.write_page(p, arrival_ns=0, size_bytes=32769)
        assert e.snapshot() == before
    result = e.write_page(p, arrival_ns=0, size_bytes=32768)
    fresh = executor()
    fresh.bind_page(p)
    assert result == fresh.write_page(p, arrival_ns=0, size_bytes=32768)


@pytest.mark.parametrize("operation", ["read_page", "write_page"])
def test_failed_page_statistics_roll_back_completed_core_work(monkeypatch, operation):
    e = executor(reject=False)
    p = page()
    e.bind_page(p)
    before = e.snapshot()

    def fail(*_args):
        assert e.runtime.timeline.bytes_moved
        raise ValueError("statistics failed")

    monkeypatch.setattr(e, "_sum_counters", fail)
    with pytest.raises(ValueError, match="statistics failed"):
        getattr(e, operation)(p, size_bytes=16384)
    assert e.snapshot() == before


def test_chunked_observations_preserve_all_events_and_one_geometry():
    e = executor(buffer_bytes=4096, reject=False, observe=True)
    p = page(size=16384)
    e.bind_page(p)
    result = e.write_page(p)
    core = e.runtime.runtimes[e.services["hbf"].physical_owner].core
    assert result["write_completion"] == "host_buffer_ack_pending_media_drain"
    assert result["completion_ns"] == result["host_ack_completion_ns"]
    assert result["media_drain_completion_ns"] is None
    assert result["host_transfer_bytes"] == p.bytes
    # Four host chunks force three drains. The final chunk is acknowledged
    # while its payload still occupies the staging buffer.
    assert result["pages_read"] == result["pages_programmed"] == 3
    assert core.pending_write_buffer_bytes == result["write_buffer_occupancy_bytes"] == 4096
    assert core.buffered_page_count == 1
    acknowledged = result["nand_observations"]
    assert len(acknowledged["events"]) == 6
    assert sum(row["physical_bytes"] for row in acknowledged["events"]) == result["physical_bytes"]

    drain = core.flush_write_buffer(now_ns=result["completion_ns"])
    assert drain["flushed_pages"] == drain["pages_read"] == drain["physical_programs"] == 1
    assert drain["write_completion"] == "media_program_complete"
    assert drain["media_drain_completion_ns"] > result["host_ack_completion_ns"]
    assert core.pending_write_buffer_bytes == core.buffered_page_count == 0
    assert core.inflight_write_buffer_bytes == 4096
    assert result["pages_read"] + drain["pages_read"] == 4
    assert result["pages_programmed"] + drain["physical_programs"] == 4
    observations = summarize_nand_observations((acknowledged, drain["nand_observations"]))
    assert len(observations["events"]) == 8
    config = e._configs["hbf"]
    plane_count = (config.channels * config.targets_per_channel * config.dies_per_target
                   * config.luns_per_die * config.planes_per_lun)
    assert sum(observations["configured_planes_by_owner"].values()) == plane_count
    assert observations["physical_plane_imbalance"]["configured_planes"] == plane_count
    assert observations["read_queue_depth"]["operations"] == 4
    assert observations["program_queue_depth"]["operations"] == 4
    assert observations["configured_planes_by_owner"] == acknowledged["configured_planes_by_owner"]
    assert sum(row["physical_bytes"] for row in observations["events"]) == (
        result["physical_bytes"] + drain["physical_read_bytes"] + drain["physical_write_bytes"])
    completed = core.flush_write_buffer(now_ns=drain["media_drain_completion_ns"])
    assert completed["flushed_pages"] == 0
    assert core.inflight_write_buffer_bytes == 0


def transfer_executor(p, *, target_reject=False, topology=None):
    hbm = make_hbm_config(capacity_bytes=1048576)
    hbf = make_hbf_config(capacity_bytes=1048576, page_bytes=16384,
        write_buffer_bytes=32768,
        partial_page_policy="reject" if target_reject else "read_modify_write")
    e = PhysicalKVTransferExecutor([component("hbm", hbm), component("hbf", hbf)],
                                   topology=topology)
    e.bind_page(p)
    return e


def test_async_phase_failure_preserves_source_and_interleaved_work():
    p = page("hbm", size=32769)
    e = transfer_executor(p, target_reject=True)
    plan = e.prepare_async(request(p, "hbf"))
    e.read_page(p, arrival_ns=plan.completion_ns, size_bytes=64)
    before_phase = e.snapshot()
    for _ in range(2):
        with pytest.raises(ValueError, match="partial NAND page write"):
            plan.advance()
        assert e.snapshot() == before_phase
    assert e.counters["read_bytes"] == p.bytes + 64
    assert e.counters["write_bytes"] == 0
    plan.cancel()
    assert e._bindings[p.logical_page_id].component_id == "hbm"
    assert not e.runtime.allocators[e.services["hbf"].physical_owner].allocations()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_transfer_link_accounting_uses_same_alias_as_calendar(asynchronous):
    p = page("hbm", size=16384)
    hop = RouteHop("link", "hbm", "hbf", "link-view", p.bytes * 8. / 100., 0.)
    topology = SimpleNamespace(route=lambda *_args, **_kwargs: (hop,))
    e = transfer_executor(p, topology=topology)
    e.runtime.timeline.resource_aliases = {"link-view": "wire"}
    e.runtime.timeline.reserve("wire", 0, 100000)
    plan = e.prepare_async(request(p, "hbf")) if asynchronous else e(request(p, "hbf"))
    while plan.advance is not None:
        plan = plan.advance()
    plan.commit()
    assert e.runtime.timeline.bytes_moved["wire"] == p.bytes
    assert "link-view" not in e.runtime.timeline.bytes_moved
    assert "link-view" not in e.runtime.timeline.lane_available
    phases = [phase for phase in e.events[-1]["phases"] if phase["kind"] == "link"]
    assert len(phases) == 1
    assert phases[0]["resource_id"] == "wire"
    assert phases[0]["start_ns"] >= 100000
