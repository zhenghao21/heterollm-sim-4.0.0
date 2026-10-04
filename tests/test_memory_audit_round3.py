"""Third audit regression suite for commit 0ddb8c276abda10504bc1adfb685b626a2c2ea92.

Run against a full checkout with the production package importable:
    python -m pytest -q test_memory_audit_round3.py

Synthetic parameters only; no network, native measurement or repository writes.
10 parameterized cases cover 7 additional issue classes. The audit also identifies
an event-kernel integration gap by source review; that is not an E2E test here.
The local audit ran five SHA-verified complete core files and original adapter
class/method excerpts. See README.md for the distinction from full-package testing.
"""
import copy
import pytest
from heterollm_sim.memory_types import AccessRequest, DramConfig, NandConfig
from heterollm_sim.memory_mapping import map_dram_address
from heterollm_sim.nand_core import NandCore


def test_rejected_streamed_nand_write_leaves_device_state_unchanged():
    cfg=NandConfig(page_bytes=1024,host_granularity_bytes=256,
                   host_bandwidth_gb_s=1024,internal_bandwidth_gb_s=1024,
                   page_program_ns=100,partial_page_policy='reject')
    core=NandCore(cfg)
    before=copy.deepcopy((core.timeline,core._array_ready,core._buffer_ready))
    with pytest.raises(ValueError,match='partial NAND page'):
        core.execute(AccessRequest('invalid','write',0,1025))
    after=(core.timeline,core._array_ready,core._buffer_ready)
    assert after==before, {'before':before,'after':after}


def test_internal_transfer_cannot_be_less_than_returned_uncompressed_data():
    # internal_transfer_bytes cannot cause a full 1024B read to cross its
    # only internal path as 64B; there is no compression model in this core.
    try:
        cfg=NandConfig(page_bytes=1024,host_granularity_bytes=64,
                       internal_transfer_bytes=64,
                       host_bandwidth_gb_s=1024,internal_bandwidth_gb_s=1,
                       page_read_ns=0)
        result=NandCore(cfg).execute(AccessRequest('full','read',0,1024))
    except ValueError:
        return
    assert result.internal_transfer_bytes>=result.logical_bytes, result


@pytest.mark.parametrize('interleave',[192,512])
def test_all_accepted_interleave_geometries_are_injective(interleave):
    # Either reject an unsupported geometry or map it injectively.
    try:
        cfg=DramConfig(channels=1,banks_per_group=2,rows_per_bank=2,
                       row_bytes=128,burst_bytes=64,interleave_bytes=interleave)
    except ValueError:
        return
    seen={}
    for address in range(0,cfg.capacity_bytes,cfg.burst_bytes):
        m=map_dram_address(cfg,address)
        key=tuple(getattr(m,k) for k in ('stack','die','lane','rank','bank_group','bank','row','column'))
        assert key not in seen, {'first':seen.get(key),'second':address,'location':key}
        seen[key]=address

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
