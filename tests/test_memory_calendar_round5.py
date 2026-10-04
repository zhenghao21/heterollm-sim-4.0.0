from dataclasses import asdict
from heterollm_sim.contracts import ResourceDemand, TaskSpec, TaskCategory
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.memory_types import NandConfig

def _mem(task_id, host='shared:pcie', owner='ssd'):
    config = NandConfig(page_bytes=1024, host_granularity_bytes=1024,
        host_bandwidth_gb_s=1, internal_bandwidth_gb_s=1024,
        page_read_ns=0, metadata={'host_resource_id': host})
    return TaskSpec(task_id, task_id, task_id, TaskCategory.MEMORY,
        demands=(ResourceDemand(owner, 1, bytes_moved=1024),),
        metadata={'physical_memory_config': asdict(config),
          'memory_access': {'operation':'read','address':0,'byte_count':1024,
                            'physical_owner':owner,'resource_id':owner}})

def _link(task_id, resource='shared:pcie'):
    return TaskSpec(task_id, task_id, task_id, TaskCategory.COMMUNICATION,
                    demands=(ResourceDemand(resource, 1000, bytes_moved=1000),))

def test_ordinary_then_physical_share_host_calendar():
    kernel = UnifiedEventKernel.from_closed_graph((_link('00link'), _mem('01mem')))
    first, second = kernel.step(), kernel.step()
    assert first.end_ns == 1000
    assert second.start_ns >= first.end_ns

def test_physical_then_ordinary_share_host_calendar():
    kernel = UnifiedEventKernel.from_closed_graph((_mem('00mem'), _link('01link')))
    first, second = kernel.step(), kernel.step()
    assert first.end_ns == 1025
    assert second.start_ns >= first.end_ns

def test_private_host_paths_can_overlap():
    kernel = UnifiedEventKernel.from_closed_graph((_link('00link', 'private:pcie'), _mem('01mem', 'shared:pcie')))
    first, second = kernel.step(), kernel.step()
    assert first.start_ns == second.start_ns == 0

def test_explicit_aliases_share_capacity():
    kernel = UnifiedEventKernel.from_closed_graph(
        (_link('00a', 'alias:pcie'), _link('01b', 'shared:pcie')),
        resource_owners={'alias:pcie': 'shared:pcie'},
        resource_capacities={'alias:pcie': 1, 'shared:pcie': 1})
    first, second = kernel.step(), kernel.step()
    assert second.start_ns >= first.end_ns
