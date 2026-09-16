"""Tail execution affects issued arithmetic, never logical matrix storage."""
from dataclasses import replace
import pytest
from heterollm_sim.cost_models import GemmWorkload, HBMProfile, estimate_gpu_gemm
from heterollm_sim.mmq_work import derive_mmq_work
from test_cost_models import gpu_profile

def scenario(k):
    gpu=gpu_profile(kernel_launch_ns=1000)
    work=derive_mmq_work(m=64,k=k,n=896,weight_format='Q5_0',sm_count=1,shared_memory_per_block=101376)
    load=GemmWorkload(64,k,896,activation_bits=16,weight_bits=5,output_bits=32,
        packed_weight_formats=('Q5_0',),weight_storage_bytes=896*k//32*22,
        activation_storage_bytes=work.consumer_unique_bytes,output_storage_bytes=work.native_output_bytes,mmq_work=work)
    return gpu,work,load

def test_same_executed_k_different_logical_work_and_weights():
    gpu,a,x=scenario(896);_,b,y=scenario(1024);mem=HBMProfile(bandwidth_gb_s=100)
    tail,aligned=estimate_gpu_gemm(gpu,mem,x),estimate_gpu_gemm(gpu,mem,y)
    assert a.k==896 and a.k_execution==b.k_execution==1024
    assert tail.metadata['compute_service_ns']==aligned.metadata['compute_service_ns']
    assert tail.useful_ops==2*64*896*896
    assert tail.useful_ops < aligned.useful_ops
    assert x.weight_bytes==896*896//32*22
    assert x.weight_bytes < y.weight_bytes
    assert tail.metadata['logical_reduction_k']==896
    assert tail.metadata['executed_reduction_k']==1024
    assert len([p for p in tail.phases if p.name=='kernel_launch'])==1

def test_plain_gemm_never_inherits_source_padding():
    gpu,work,x=scenario(896);plain=replace(x,mmq_work=None)
    estimate=estimate_gpu_gemm(gpu,HBMProfile(bandwidth_gb_s=100),plain)
    assert 'executed_reduction_k' not in estimate.metadata
    assert 'logical_reduction_k' not in estimate.metadata

def test_source_aligned_execution_matches_generic_issued_geometry():
    gpu,work,x=scenario(1024);mem=HBMProfile(bandwidth_gb_s=100)
    # Current helper has equal int8/fp16 structural peak; compare raw issued work.
    source=estimate_gpu_gemm(gpu,mem,x);plain=estimate_gpu_gemm(gpu,mem,replace(x,mmq_work=None))
    assert source.metadata['compute_service_ns']==plain.metadata['compute_service_ns']
    assert source.metadata['issued_operations']==plain.metadata['issued_operations']

@pytest.mark.parametrize('m,qualified,expected',[(64,False,'uncovered'),(64,True,'applied'),(4,False,'mmvq_precedes_mmq')])
def test_tail_requires_extra_source_contract_but_vector_dispatch_is_unchanged(m,qualified,expected):
    from heterollm_sim import planner
    from tests.test_mmq_planner import scenario as base_scenario, CONTRACT
    base=base_scenario();target=next(c for c in base.hardware.components if c.component_id=='gpu0')
    contract={**CONTRACT}
    if qualified:contract['reduction_tail_contract']='logical-k-streamk-tail/v1'
    target=replace(target,metadata={**target.metadata,'cuda_compute_capability':1200,'llama_cpp_mmq_contract':contract})
    load=GemmWorkload(m,896,896,activation_bits=32,weight_bits=5,output_bits=32,packed_weight_formats=('Q5_0',))
    metadata={'projection_segment_count':1,'projection_segments':[{'local_k':896,'local_n':896,'format':'Q5_0','physical_tensor_name':'synthetic.weight'}]}
    result,audit,_=planner._declared_mmq_work(base,load,target,gpu_profile(),metadata,model_weight_read=True,rhs_is_activation=False)
    assert audit['status']==expected
    assert (result is not None)==(expected=='applied')

@pytest.mark.parametrize('fmt,block_bytes,extra',[('Q5_0',22,88),('Q8_0',34,136)])
def test_tail_read_is_once_at_tensor_end_without_weight_storage_expansion(fmt,block_bytes,extra):
    work=derive_mmq_work(m=64,k=896,n=896,weight_format=fmt,sm_count=1,shared_memory_per_block=101376)
    load=GemmWorkload(64,896,896,activation_bits=16,weight_bits=5 if fmt=='Q5_0' else 8,
        output_bits=32,packed_weight_formats=(fmt,),weight_storage_bytes=896*896//32*block_bytes,
        activation_storage_bytes=work.consumer_unique_bytes,mmq_work=work)
    est=estimate_gpu_gemm(gpu_profile(kernel_launch_ns=1000),HBMProfile(bandwidth_gb_s=100,resource_id='memory.hbm'),load)
    hbm=next(d for d in est.phases[-1].demands if d.resource_id=='memory.hbm')
    assert est.metadata['weight_bytes']==896*896//32*block_bytes
    assert est.metadata['source_unique_weight_tail_read_bytes']==extra
    assert hbm.bytes_moved==load.minimum_io_bytes+work.main_partial_write_bytes+extra
    assert est.metadata['minimum_io_bytes']==load.minimum_io_bytes+extra
    assert len([p for p in est.phases if p.name=='kernel_launch'])==1
    assert est.metadata['mmq_source_work']['weight_tail_allocation_proven'] is False
