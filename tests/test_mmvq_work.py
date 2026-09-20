"""Source geometry tests; no GPU or measured timing parameter is used."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
from heterollm_sim.mmvq_work import MMVQSourceContract, SOURCE_SHA256, UnsupportedMMVQ, derive_mmvq_work, verify_mmvq_source_tree


def contract(**changes):
    values=dict(compute_capability=1200,highest_compiled_arch=1200,warp_size=32,
                source_hashes=dict(SOURCE_SHA256),runtime_binary_sha256='a'*64,
                ordinary_contiguous_2d=True,force_cublas=False,mmvq_dispatch_enabled=True)
    values.update(changes)
    return MMVQSourceContract(**values)


def work(m=1,k=896,n=896,fmt='Q5_0',**kwargs):
    return derive_mmvq_work(m=m,k=k,n=n,weight_format=fmt,contract=contract(**kwargs))


@pytest.mark.parametrize('fmt,qi,bytes_per_block',[('Q5_0',4,22),('Q8_0',8,34)])
@pytest.mark.parametrize('m',[1,2,4,8])
def test_work_has_source_geometry_and_preserves_logical_io(fmt,qi,bytes_per_block,m):
    w=work(m=m,fmt=fmt)
    assert w.qi==qi and w.qk==32 and w.vdr==2
    assert w.warps_per_cta==(4 if m<=4 else 2)
    assert w.rows_per_cta==(4 if m==1 else 2)
    assert w.grid==(896//w.rows_per_cta,1,1)
    assert w.block==(32,w.warps_per_cta,1)
    assert w.logical_weight_bytes==896*(896//32)*bytes_per_block
    assert w.logical_input_f32_bytes==4*m*896
    assert w.consumer_q8_1_unique_bytes==36*m*896//32
    assert w.output_f32_bytes==4*m*896
    assert sum(w.loop_iterations_by_thread)==(896//32)*(qi//2)
    assert w.source_vector_dot_calls==(896//32)*(qi//2)*m*896
    assert w.to_metadata()['hbm_efficiency'] is None
    assert w.to_metadata()['native_dispatch_proven'] is False


@pytest.mark.parametrize('fmt,boundary',[('Q5_0',2048),('Q8_0',1024)])
def test_small_k_predicate_is_strict_and_only_single_token(fmt,boundary):
    assert work(k=boundary-32,fmt=fmt).small_k is True
    assert work(k=boundary,fmt=fmt).small_k is False
    assert work(k=boundary,fmt=fmt).rows_per_cta==1
    assert work(k=boundary+32,fmt=fmt).small_k is False
    for m in (2,4):
        assert work(m=m,k=32,fmt=fmt).small_k is False
        assert work(m=m,k=32,fmt=fmt).rows_per_cta==2


def test_q5_q8_different_k_iteration_and_padding_coverage():
    q5=work(k=4096,fmt='Q5_0');q8=work(k=4096,fmt='Q8_0')
    assert q5.blocks_per_loop_step==64 and q8.blocks_per_loop_step==32
    assert q5.maximum_thread_k_iterations==2 and q8.maximum_thread_k_iterations==4
    small=work(k=896,fmt='Q5_0')
    assert small.active_k_threads_per_cta==56
    assert small.maximum_thread_k_iterations==1
    assert small.reduction_shared_array_bytes==1536


@pytest.mark.parametrize('kwargs',[{'compute_capability':890},{'highest_compiled_arch':890},{'warp_size':64},
    {'ordinary_contiguous_2d':False},{'force_cublas':True},{'mmvq_dispatch_enabled':False},
    {'has_ids':True},{'has_fusion':True},{'channels':2},{'samples':2},{'channels':True},
    {'runtime_binary_sha256':''},{'source_hashes':{}},{'force_cublas':0}])
def test_runtime_contract_rejects_unproven_scope(kwargs):
    with pytest.raises(UnsupportedMMVQ):contract(**kwargs)


@pytest.mark.parametrize('kwargs',[{'m':9},{'m':True},{'k':897},{'n':3},{'fmt':'Q4_K'},{'fmt':'q5_0'}])
def test_shape_format_tail_uncovered_is_not_silently_derived(kwargs):
    with pytest.raises(UnsupportedMMVQ):work(**kwargs)


def test_source_geometry_contract_has_no_empirical_rate_or_model_keys():
    metadata=work().to_metadata()
    assert metadata['cost_model_applied'] is False
    assert metadata['binary_source_equivalence_proven'] is False
    assert 'model_name' not in metadata and 'prompt_fingerprint' not in metadata
    assert not any(k.endswith('_ns') for k in metadata)
    root=Path(__file__).resolve().parents[1]
    planner_source=(root/'src/heterollm_sim/planner.py').read_text(encoding='utf-8-sig')
    cost_source=(root/'src/heterollm_sim/cost_models.py').read_text(encoding='utf-8-sig')
    assert 'from .mmvq_work import' in planner_source
    assert 'from .mmvq_work import' in cost_source
    assert 'source_geometry_unpriced' in planner_source
    assert 'source_geometry_priced' in cost_source
    assert 'unpriced_no_mma_wave_claim' in cost_source


def test_contract_hashes_immutable():
    c=contract()
    with pytest.raises(TypeError):c.source_hashes['ggml-cuda/mmvq.cu']='b'*64


def test_locked_source_hashes_if_local_source_available():
    root=Path(__file__).resolve().parents[1]/'source/llama.cpp-semantic/ggml/src'
    if not (root/'ggml-cuda/mmvq.cu').is_file():pytest.skip('optional local locked CUDA source unavailable')
    assert dict(verify_mmvq_source_tree(root))==dict(SOURCE_SHA256)


def test_completed_trace_geometry_only_crosscheck():
    root=Path(__file__).resolve().parents[1]
    path=root/'tests/fixtures/baseline_evidence/kernel_launch_geometry.json'
    doc=json.loads(path.read_text(encoding='utf-8'))['mmvq_case'];c=doc['config'];w=work(m=c['M'],n=c['N'],k=c['K'],fmt=c['quant'])
    count=0
    for call in doc['calls']:
        for pair in call['kernel_launch_pairs']:
            k=pair['kernel']
            if k['short_name_text']!='mul_mat_vec_q':continue
            assert tuple(k[name] for name in ('gridX','gridY','gridZ'))==w.grid
            assert tuple(k[name] for name in ('blockX','blockY','blockZ'))==w.block
            assert k['dynamicSharedMemory']==0
            assert '(bool)1, (bool)0>' in k['demangled_name_text']
            count+=1
    assert count==36
