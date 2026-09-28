from dataclasses import replace

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory, TaskSpec
from heterollm_sim.cost_models import HBMProfile
from heterollm_sim.event_kernel import UnifiedEventKernel

from heterollm_sim.data_motion import (
    COPY,
    READ,
    WRITE,
    DataAccess,
    LinkService,
    MemoryPosition,
    endpoint_service,
    expand_access,
    expand_accesses,
    resolve_link_service,
    resolve_service,
)
from heterollm_sim.ir import ComponentSpec, LinkSpec, PortSpec


def _memory(component_id="hbm0", *, owner="hbm0.controller"):
    return ComponentSpec(
        component_id,
        "hbm",
        ports=(PortSpec("host", "HBM", "device", bandwidth_gbps=800.0),),
        read_bandwidth_gbps=800.0,
        write_bandwidth_gbps=400.0,
        metadata={
            "memory_service_owner": owner,
            "read_latency_ns": 20.0,
            "write_latency_ns": 30.0,
            "transfer_granularity_bytes": 256,
            "max_outstanding_requests": 4,
        },
    )


def test_read_and_write_share_one_physical_owner_but_remain_two_operations():
    component = _memory()
    service = resolve_service(component)
    reads = expand_access(
        DataAccess("op.read", READ, 1024, source=MemoryPosition("hbm0")),
        {"hbm0": service},
    )
    writes = expand_access(
        DataAccess("op.write", WRITE, 1024, target=MemoryPosition("hbm0")),
        {"hbm0": service},
    )
    assert reads.phases[0].physical_owner == writes.phases[0].physical_owner
    assert reads.phases[0].resource_id == writes.phases[0].resource_id
    assert reads.phases[0].operation_id != writes.phases[0].operation_id


def test_duplicate_operation_is_expanded_once():
    component = _memory()
    service = resolve_service(component)
    access = DataAccess("same", READ, 1024, source=MemoryPosition("hbm0"))
    motions = expand_accesses((access, replace(access, metadata={"view": "dma"})), {"hbm0": service})
    assert len(motions) == 1


def test_same_storage_alias_is_zero_cost_but_different_ranges_are_not():
    component = _memory()
    service = resolve_service(component)
    alias = expand_access(
        DataAccess(
            "alias", COPY, 4096,
            source=MemoryPosition("hbm0", 128, allocation_id="weights"),
            target=MemoryPosition("hbm0", 128, allocation_id="weights"),
        ),
        {"hbm0": service},
    )
    moved = expand_access(
        DataAccess(
            "move", COPY, 4096,
            source=MemoryPosition("hbm0", 128, allocation_id="weights"),
            target=MemoryPosition("hbm0", 512, allocation_id="weights"),
        ),
        {"hbm0": service},
    )
    assert alias.phases == ()
    assert [phase.kind for phase in moved.phases] == [READ, WRITE]


def test_copy_waits_for_read_and_link_before_target_write():
    source = resolve_service(_memory("hbm0", owner="hbm0.controller"))
    target = resolve_service(_memory("hbm1", owner="hbm1.controller"))
    motion = expand_access(
        DataAccess(
            "copy", COPY, 2048,
            source=MemoryPosition("hbm0"),
            target=MemoryPosition("hbm1"),
        ),
        {"hbm0": source, "hbm1": target},
        links={("hbm0", "hbm1"): LinkService("pcie0", "pcie0", "pcie0", 32.0)},
    )
    assert [phase.operation_id for phase in motion.phases] == [
        "copy.read", "copy.link", "copy.write"
    ]
    assert motion.phases[1].dependencies == ("copy.read",)
    assert motion.phases[2].dependencies == ("copy.link",)


def test_topology_view_does_not_create_second_local_memory_service():
    memory = _memory()
    gpu = ComponentSpec("gpu0", "gpu", ports=(PortSpec("p", "HBM", "controller"),))
    link = LinkSpec("gpu-hbm", "gpu0", "p", "hbm0", "host", "HBM", bandwidth_gbps=800.0)
    assert resolve_link_service(link, gpu, memory) is not None
    assert resolve_link_service(replace(link, metadata={"service_ref": "hbm0.access"}), gpu, memory) is None


def test_independent_link_remains_a_service():
    gpu0 = ComponentSpec("gpu0", "gpu", ports=(PortSpec("p", "PCIe", "controller"),))
    gpu1 = ComponentSpec("gpu1", "gpu", ports=(PortSpec("p", "PCIe", "controller"),))
    link = LinkSpec("pcie0", "gpu0", "p", "gpu1", "p", "PCIe", bandwidth_gbps=256.0)
    service = resolve_link_service(link, gpu0, gpu1)
    assert service is not None
    assert service.resource_id == "pcie0"


def _run(tasks, *, owners=None, capacities=None):
    kernel = UnifiedEventKernel(resource_owners=owners, resource_capacities=capacities)
    kernel.add_tasks(tasks)
    events = tuple(kernel.step() for _ in tasks)
    return kernel, {event.task.task_id: event for event in events}


@pytest.mark.parametrize("service_model", ["analytical", "serialized", "overlapped"])
@pytest.mark.parametrize("read", [True, False])
@pytest.mark.parametrize("byte_count", [0, 1, 1025, 65537])
def test_operator_endpoint_and_access_have_identical_service(read, byte_count, service_model):
    component = replace(_memory(), metadata={**_memory().metadata, "efficiency": 0.5, "memory_service_model": service_model})
    profile = HBMProfile(
        bandwidth_gb_s=100, read_bandwidth_gb_s=100, write_bandwidth_gb_s=50, service_model=service_model,
        efficiency=0.5, read_latency_ns=20, write_latency_ns=30,
        transaction_bytes=256, max_outstanding_requests=4,
    )
    service = resolve_service(component, profile)
    kind = READ if read else WRITE
    analytical = profile.memory_service(byte_count if read else 0, 0 if read else byte_count)
    endpoint = endpoint_service(component, byte_count, read=read, name="endpoint")
    access = expand_access(DataAccess(
        "op", kind, byte_count,
        source=MemoryPosition("hbm0") if read else None,
        target=None if read else MemoryPosition("hbm0"),
    ), {"hbm0": service})
    assert endpoint.demands[0].service_ns == pytest.approx(analytical["service_ns"])
    assert access.phases[0].service_ns == pytest.approx(analytical["service_ns"])
    assert endpoint.metadata["latency_ns"] == (20 if read else 30)
    assert service.read_bandwidth_gb_s == 50  # Efficiency is applied exactly once.
    assert service.write_bandwidth_gb_s == 25


def test_effective_bandwidth_cannot_override_component_physical_cap():
    with pytest.raises(ValueError, match="exceed component physical"):
        resolve_service(_memory(), HBMProfile(bandwidth_gb_s=101))


def test_kernel_operator_read_and_dma_write_share_controller_budget():
    component = _memory()
    service = resolve_service(component)
    read = expand_access(DataAccess("read", READ, 1024, source=MemoryPosition("hbm0")), {"hbm0": service})
    endpoint = endpoint_service(component, 1024, read=False, name="dma")
    write = TaskSpec("write", "request", "write", TaskCategory.COMMUNICATION, demands=endpoint.demands)
    owners = {**read.resource_owners, endpoint.demands[0].resource_id: service.physical_owner}
    kernel, events = _run(read.to_tasks("request") + (write,), owners=owners)
    assert events["write"].start_ns == events["read"].end_ns
    assert kernel.makespan_ns == pytest.approx(service.bill(READ, 1024) + service.bill(WRITE, 1024))
    assert events["write"].queue_wait_ns > 0


def test_kernel_copy_target_write_finishes_before_consumer_can_read():
    source = resolve_service(_memory("source", owner="source.controller"))
    target = resolve_service(_memory("target", owner="target.controller"))
    services = {"source": source, "target": target}
    copy = expand_access(DataAccess(
        "copy", COPY, 1024, source=MemoryPosition("source", allocation_id="weights"),
        target=MemoryPosition("target", allocation_id="staging"),
    ), services, links={("source", "target"): LinkService("link", "link", "link", 10, 100, 2)})
    read = expand_access(DataAccess(
        "consume", READ, 1024, source=MemoryPosition("target", allocation_id="staging"),
        dependencies=("copy",),
    ), services)
    _, events = _run(copy.to_tasks("request") + read.to_tasks("request"),
                     owners={**copy.resource_owners, **read.resource_owners}, capacities=copy.resource_capacities)
    assert events["copy.link"].start_ns == events["copy.read"].end_ns
    assert events["copy.write"].start_ns == events["copy.link"].end_ns
    assert events["consume"].start_ns == events["copy.write"].end_ns
    assert events["copy.write"].service_ns > 0


def test_kernel_two_inflight_requests_share_one_sender_and_arrive_at_1001_1002():
    link = LinkService("link", "controller", "link.tx", 1, 1000, 2)
    tasks = tuple(TaskSpec(op, "request", op, TaskCategory.COMMUNICATION, demands=link.demands(1))
                  for op in ("a", "b"))
    kernel, events = _run(tasks, capacities=link.resource_capacities)
    assert (events["a"].start_ns, events["a"].end_ns) == (0, 1001)
    assert (events["b"].start_ns, events["b"].end_ns) == (1, 1002)
    assert kernel.resource_busy_ns["link.tx"] == 2
    assert sum(d.bytes_moved for t in tasks for d in t.demands) == 2
    assert replace(link, resource_id="other.tx").resource_capacities == link.resource_capacities


def test_same_offset_different_allocations_and_unknown_allocations_are_not_aliases():
    service = resolve_service(_memory())
    for source_allocation, target_allocation in (("buffer-a", "buffer-b"), ("", "")):
        motion = expand_access(DataAccess(
            "copy", COPY, 1024,
            source=MemoryPosition("hbm0", allocation_id=source_allocation),
            target=MemoryPosition("hbm0", allocation_id=target_allocation),
        ), {"hbm0": service})
        assert [phase.kind for phase in motion.phases] == [READ, WRITE]


def test_same_owner_and_allocation_name_on_different_storage_are_not_aliases():
    source = resolve_service(_memory("a"))
    target = resolve_service(_memory("b"))
    motion = expand_access(DataAccess(
        "copy", COPY, 1024,
        source=MemoryPosition("a", allocation_id="weights"),
        target=MemoryPosition("b", allocation_id="weights"),
    ), {"a": source, "b": target})
    assert len(motion.phases) == 2


@pytest.mark.parametrize("change", [{"byte_count": 2048}, {"dependencies": ("earlier",)},
                                     {"source": MemoryPosition("hbm0", 1)}])
def test_conflicting_operation_identity_is_rejected(change):
    access = DataAccess("read", READ, 1024, source=MemoryPosition("hbm0"))
    with pytest.raises(ValueError, match="conflicting descriptions"):
        expand_accesses((access, replace(access, **change)), {"hbm0": resolve_service(_memory())})


def test_unknown_hbf_write_direction_is_preserved_and_rejected_on_use():
    component = ComponentSpec("flash", "hbf", read_bandwidth_gbps=80)
    service = resolve_service(component)
    assert service.write_bandwidth_gb_s == 0
    assert service.bill(READ, 1024) > 0
    with pytest.raises(ValueError, match="positive write bandwidth"):
        service.bill(WRITE, 1024)
    with pytest.raises(ValueError, match="positive write bandwidth"):
        endpoint_service(component, 1024, read=False, name="write")


def test_hbf_rmw_charges_actual_page_bytes_in_both_entry_paths():
    component = ComponentSpec("flash", "hbf", read_bandwidth_gbps=100, write_bandwidth_gbps=100,
        metadata={"read_energy_pj_per_byte": 2, "write_energy_pj_per_byte": 3, "hbf_media": {
            "version": "cold_page_v1", "host_transaction_bytes": 64,
            "host_max_request_bytes": 4096, "media_page_bytes": 4096,
            "command_queue_depth": 256, "media_parallelism": 4,
            "page_read_latency_ns": 100, "page_program_latency_ns": 200,
            "access_pattern": "contiguous_page_aligned", "physical_planes": 2,
        }})
    endpoint = endpoint_service(component, 64, read=False, name="write")
    motion = expand_access(DataAccess("write", WRITE, 64, target=MemoryPosition("flash")),
                           {"flash": resolve_service(component)})
    media = endpoint.metadata["hbf_media"]
    assert media["rmw_read_operations"] == 1
    assert media["physical_read_bytes"] == 4096
    assert media["physical_write_bytes"] == 4096
    assert motion.demands[0].bytes_moved == endpoint.demands[0].bytes_moved == 8192
    assert motion.demands[0].service_ns == endpoint.demands[0].service_ns

    assert motion.demands[0].energy_pj == endpoint.demands[0].energy_pj == 4096 * 5


@pytest.mark.parametrize("changes", [
    {"metadata": {"service_ref": "missing.access"}},
    {"metadata": {"service_ref": "hbm0.access", "bandwidth_resource_id": "other"}},
    {"metadata": {"service_ref": "hbm0.access"}, "bandwidth_gbps": 80},
    {"metadata": {"service_ref": "hbm0.access"}, "latency_ns": 10},
])
def test_reference_link_rejects_conflicting_physical_declarations(changes):
    memory = _memory()
    gpu = ComponentSpec("gpu0", "gpu", ports=(PortSpec("p", "HBM", "controller"),))
    link = LinkSpec("gpu-hbm", "gpu0", "p", "hbm0", "host", "HBM", bandwidth_gbps=800)
    with pytest.raises(ValueError):
        resolve_link_service(replace(link, **changes), gpu, memory)
