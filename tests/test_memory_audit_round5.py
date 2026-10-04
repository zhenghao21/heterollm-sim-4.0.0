"""Regressions intended for the COMPLETE real repository at 6fb4375.
This file was authored but NOT executed against a complete checkout in this audit.
The separate tests/ directory records the actually executed isolated checks.
Run with the repository's normal environment and src on PYTHONPATH.
"""
from dataclasses import asdict,replace
import pytest
from heterollm_sim.contracts import ResourceDemand,TaskSpec,TaskCategory,RunManifest
from heterollm_sim.data_motion import PhysicalRuntimeContext,resolve_physical_task
from heterollm_sim.memory_types import DramConfig,NandConfig
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.engine import ScheduleIR,simulate_schedule
from heterollm_sim.schema_v4 import SIMULATION_SCHEMA_VERSION

def dram():
    return DramConfig(banks_per_group=1,lane_bandwidth_gb_s=64,open_ns=10,
                      read_latency_ns=5,write_latency_ns=5,burst_interval_ns=1,write_recovery_ns=0)

def nand(**kw):
    return NandConfig(page_bytes=1024,host_granularity_bytes=1024,host_bandwidth_gb_s=1,
                      internal_bandwidth_gb_s=1024,page_read_ns=0,**kw)

def task(config,owner='A',tid='t',preview=1,extra=(),accesses=None):
    count=config.burst_bytes if isinstance(config,DramConfig) else config.page_bytes
    if accesses is None:accesses={'operation':'read','address':0,'byte_count':count,'physical_owner':owner,'resource_id':owner}
    return TaskSpec(tid,tid,tid,TaskCategory.MEMORY,
        demands=(ResourceDemand(owner,preview,bytes_moved=count),)+tuple(extra),
        metadata={'physical_memory_config':asdict(config),'memory_access':accesses})

def trace(t):
    manifest=RunManifest(schema_version=SIMULATION_SCHEMA_VERSION,run_id='audit',random_seed=0,
            simulator_version='audit',model_name='synthetic',hardware_name='synthetic',workload_name='audit')
    return simulate_schedule(ScheduleIR(manifest,(t,)))

def test_remove_memory_preview():
    r=resolve_physical_task(task(dram(),preview=16),PhysicalRuntimeContext(),0)
    assert 'A' not in {d.resource_id for d in r.demands}

def test_warm_read_task_completion_matches_physical_completion():
    cold=task(dram(),tid='cold',preview=16)
    warm=replace(task(dram(),tid='warm',preview=16),dependencies=('cold',))
    k=UnifiedEventKernel.from_closed_graph((cold,warm))
    a=k.step();b=k.step()
    assert a.task.task_id=='cold' and b.task.task_id=='warm'
    assert b.end_ns==b.task.metadata['physical_completion_ns']

def test_independent_nand_hosts():
    ctx=PhysicalRuntimeContext()
    a=resolve_physical_task(task(nand(),'ssdA','a'),ctx,0)
    b=resolve_physical_task(task(nand(),'ssdB','b'),ctx,0)
    assert a.metadata['physical_completion_ns']==b.metadata['physical_completion_ns']

def test_normal_and_physical_host_compete():
    normal=TaskSpec('00link','r','link',TaskCategory.COMMUNICATION,
                   demands=(ResourceDemand('shared:pcie',1000,bytes_moved=1000),))
    physical=task(nand(metadata={'host_resource_id':'shared:pcie'}),tid='01mem')
    k=UnifiedEventKernel.from_closed_graph((normal,physical))
    a=k.step();b=k.step()
    assert a.task.task_id=='00link'
    host=next(s for s in b.task.metadata['physical_resource_intervals'] if s.name=='HOST_TRANSFER')
    assert host.start_ns>=a.end_ns

def test_mix_counters():
    ops=tuple({'operation':op,'address':0,'byte_count':64,'physical_owner':'A','resource_id':'A'} for op in ('read','write'))
    r=resolve_physical_task(task(dram(),accesses=ops),PhysicalRuntimeContext(),0)
    m=r.metadata['physical_execution']
    assert (m['physical_read_bytes'],m['physical_write_bytes'],m['burst_count'])==(64,64,2)

def test_error_does_not_commit_prefix():
    ops=({'operation':'read','address':0,'byte_count':64,'physical_owner':'A','resource_id':'A'},
         {'operation':'write','address':-1,'byte_count':64,'physical_owner':'A','resource_id':'A'})
    ctx=PhysicalRuntimeContext()
    with pytest.raises(ValueError):resolve_physical_task(task(dram(),accesses=ops),ctx,0)
    assert ctx.timeline.snapshot()=={}

def test_report_retains_compute_interval():
    result=trace(task(dram(),extra=(ResourceDemand('gpu.compute',1000,energy_pj=123,work_units=1234),)))
    assert any(i.resource_id=='gpu.compute' for i in result.tasks[0].resource_intervals)

def test_report_preserves_data_bytes():
    result=trace(task(dram()))
    assert sum(i.bytes_moved for i in result.tasks[0].resource_intervals if i.resource_id=='A:dram:data:0')==64
