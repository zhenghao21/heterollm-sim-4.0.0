"""Run against the production data_motion module in a full checkout.

Local audit used unchanged runtime class/method excerpts with the verified
five core modules. This is adapter-level testing, NOT run_scenario/HTTP E2E.
SimpleNamespace supplies only component fields that PhysicalService.price uses.
"""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from heterollm_sim.data_motion import PhysicalService,PhysicalRuntimeContext,AccessKind,reset_physical_runtimes
from heterollm_sim.memory_types import AccessRequest,DramConfig,NandConfig
from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore


def service(cfg,owner='device'):
    return PhysicalService(owner,owner,owner,1.0,1.0,
        component=SimpleNamespace(metadata={'physical_memory_config':cfg}))


@pytest.fixture(autouse=True)
def isolated_implicit_context():
    reset_physical_runtimes()
    yield
    reset_physical_runtimes()


@pytest.mark.parametrize('explicit',[False,True])
def test_preview_does_not_pin_configuration_for_future_queries(explicit):
    cfg=DramConfig(banks_per_group=1,lane_bandwidth_gb_s=64)
    context=PhysicalRuntimeContext()
    options={'runtime':context,'preview':True} if explicit else {}
    first=service(cfg).price(AccessKind.READ,64,page_offset_bytes=0,**options)
    # A new geometry is a valid independent what-if query; no operation was committed.
    second=service(replace(cfg,channels=2)).price(AccessKind.READ,64,page_offset_bytes=0,**options)
    assert first['row_misses']==second['row_misses']==1


@pytest.mark.parametrize('medium',['DRAM','NAND'])
def test_explicit_single_request_adapter_obeys_same_queue_limit_as_batch(medium):
    if medium=='DRAM':
        cfg=DramConfig(channels=2,banks_per_group=1,rows_per_bank=4,
                       row_bytes=128,burst_bytes=64,open_ns=10,read_latency_ns=10,
                       burst_interval_ns=1,lane_bandwidth_gb_s=64,max_outstanding_requests=1)
        core_type=DramCore;size=64
    else:
        cfg=NandConfig(channels=2,page_bytes=1024,host_granularity_bytes=1024,
                       internal_bandwidth_gb_s=1024,host_bandwidth_gb_s=1024,
                       page_read_ns=10,max_outstanding_requests=1)
        core_type=NandCore;size=1024
    requests=[AccessRequest(str(i),'read',i*size,size,0) for i in range(2)]
    expected=[r.completion_ns for r in core_type(cfg).execute_batch(requests)]
    context=PhysicalRuntimeContext();svc=service(cfg)
    actual=[svc.price(AccessKind.READ,size,page_offset_bytes=i*size,
                      arrival_ns=0,runtime=context)['completion_ns'] for i in range(2)]
    assert actual==pytest.approx(expected), {'adapter':actual,'batch':expected}


def test_detail_retention_limit_does_not_change_pages_touched():
    cfg=NandConfig(page_bytes=1024,host_granularity_bytes=1024,
                   max_expanded_segments=1,page_read_ns=10)
    bill=service(cfg).price(AccessKind.READ,3072,page_offset_bytes=0)
    assert bill['pages_touched']==bill['page_count']==bill['pages_read']==3, bill


def test_batch_rejects_missing_address_like_single_request():
    cfg=NandConfig(page_bytes=1024)
    with pytest.raises(ValueError,match='address'):
        service(cfg).price_batch([{'request_id':'x','operation':'read','byte_count':1024}])
