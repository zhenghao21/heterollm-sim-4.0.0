"""Fixed-scope acceptance tests for 01df07f and the supplied candidate patch.
Authoring/syntax validation only in this audit; full-checkout execution is still required.
No simulator mocks or replaced scheduling methods are used in this file.
"""
from dataclasses import asdict
from types import SimpleNamespace
import pytest
from heterollm_sim.contracts import ResourceDemand,TaskSpec,TaskCategory
from heterollm_sim.data_motion import PhysicalRuntimeContext,resolve_physical_task
from heterollm_sim.memory_types import NandConfig
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.engine import _task_resource_intervals

def project_intervals(event):
    return tuple(_task_resource_intervals(event.task,event.demands,event.start_ns))

def cfg():
    return NandConfig(page_bytes=1024,host_granularity_bytes=1024,
        host_bandwidth_gb_s=1,internal_bandwidth_gb_s=1024,
        page_read_ns=0,block_erase_ns=1,
        metadata={'host_resource_id':'shared:pcie'})

def op(operation='read',address=0):
    return dict(operation=operation,address=address,byte_count=1024,
                physical_owner='A',resource_id='A')

def task(c,tid,ops,arrival=0):
    return TaskSpec(tid,tid,tid,TaskCategory.MEMORY,
        demands=(ResourceDemand('A',1),),earliest_start_ns=arrival,
        metadata={'physical_memory_config':c,'memory_access':ops})

def prime_and_reject(ctx,c):
    resolve_physical_task(task(c,'erase',op('erase')),ctx,0)
    before=ctx.timeline.snapshot()
    with pytest.raises(ValueError,match='capacity'):
        resolve_physical_task(task(c,'bad',(op(),op('write',c.effective_capacity_bytes))),ctx,1)
    return before

def test_failed_compound_restores_numeric_calendar():
    ctx=PhysicalRuntimeContext();c=cfg()
    before=prime_and_reject(ctx,c)
    assert ctx.timeline.snapshot()==before

def test_failed_compound_preserves_authoritative_calendar_identity():
    ctx=PhysicalRuntimeContext();c=cfg();prime_and_reject(ctx,c)
    assert ctx.runtimes['A'].core.timeline is ctx.timeline
    assert ctx.runtimes['A'].core.timeline.lane_available is ctx.timeline.lane_available

def test_post_rollback_retry_respects_ordinary_shared_link():
    c=cfg()
    normal=TaskSpec('00ordinary','r','ordinary',TaskCategory.COMMUNICATION,
        earliest_start_ns=1,demands=(ResourceDemand('shared:pcie',1000,bytes_moved=1000),))
    retry=task(c,'01retry',op(),1)
    k=UnifiedEventKernel.from_closed_graph((normal,retry))
    prime_and_reject(k.physical_runtime,c)
    a=k.step();b=k.step()
    assert a.task.task_id=='00ordinary'
    host=next(s for s in b.task.metadata['physical_resource_intervals'] if s.name=='HOST_TRANSFER')
    assert host.start_ns>=a.end_ns,(a.start_ns,a.end_ns,host.start_ns,host.end_ns)

@pytest.mark.parametrize('use_mapping',[False,True])
def test_normal_physical_then_ordinary_shared_link(use_mapping):
    c=cfg();raw=asdict(c) if use_mapping else c
    k=UnifiedEventKernel.from_closed_graph((task(raw,'physical',op()),));a=k.step()
    ordinary=TaskSpec('ordinary','r','ordinary',TaskCategory.COMMUNICATION,
        demands=(ResourceDemand('shared:pcie',1000,bytes_moved=1000),))
    k.submit((ordinary,));b=k.step()
    assert b.start_ns>=a.end_ns

def test_report_retains_nand_program_array_interval():
    out=resolve_physical_task(task(cfg(),'write',op('write')),PhysicalRuntimeContext(),0)
    actual=project_intervals(SimpleNamespace(task=out,demands=out.demands,start_ns=0))
    expected=next(s for s in out.metadata['physical_resource_intervals'] if s.name=='PAGE_PROGRAM')
    assert any((i.resource_id,i.start_ns,i.end_ns)==
               (expected.resource_id,expected.start_ns,expected.end_ns) for i in actual)


def test_report_retains_dram_bank_and_command_intervals():
    from heterollm_sim.memory_types import DramConfig
    c=DramConfig(banks_per_group=1,lane_bandwidth_gb_s=64,open_ns=10,
                 read_latency_ns=5,burst_interval_ns=1)
    access=dict(op(),byte_count=64)
    out=resolve_physical_task(task(c,'read',access),PhysicalRuntimeContext(),0)
    actual={(i.resource_id,i.start_ns,i.end_ns) for i in
            _task_resource_intervals(out,out.demands,0)}
    expected={(rid,begin,end) for rid,items in
              out.metadata['physical_execution']['resource_intervals'].items()
              for begin,end in items}
    assert expected<=actual
