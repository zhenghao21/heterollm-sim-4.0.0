"""Synthetic CUPTI/NVTX evidence tests; no native or GPU execution."""
import copy
import pytest
import native_trace as nt
from contract import CONFIG

def fixture():
    pid=12;tid=(1<<48)|(pid<<24)|2;globalpid=tid&nt.PROCESS_MASK
    markers=[];apis=[];kernels=[];calls=[]
    for ordinal,(phase,i) in enumerate(nt.expected_calls()):
        start=ordinal*1000+(1_000_001_000 if ordinal else 0);end=start+900
        label=f"graph_submit/{CONFIG['id']}/{phase}/{i}"
        markers.append({'start':start,'end':end,'globalTid':tid,'resolved_text':label})
        calls.append({'label':label,'phase':phase,'index':i})
        apis.extend([{'start':start+1,'end':start+10,'globalTid':tid,'name_text':'cudaGraphLaunch','correlationId':ordinal,'returnValue':0}, {'start':start+810,'end':start+890,'globalTid':tid,'name_text':'cudaStreamSynchronize','correlationId':1000+ordinal,'returnValue':0}])
        for node in range(CONFIG['nodes']):
            kernels.append({'evidence_rowid':ordinal*8+node,'start':start+20+node*90,'end':start+50+node*90,'globalPid':globalpid,'correlationId':ordinal,'deviceId':0,'streamId':1,'short_name_text':'scale_f32','gridX':1024,'gridY':1,'gridZ':1,'blockX':256,'blockY':1,'blockZ':1,'dynamicSharedMemory':0})
    markers.append({'start':950,'end':1_000_001_000,'globalTid':tid,'resolved_text':f"graph_submit/{CONFIG['id']}/settle/1000ms"})
    extra={**kernels[0],'evidence_rowid':99999,'start':2000,'end':3000,'correlationId':99999};kernels.append(extra)
    trace={'tables':{'NVTX_EVENTS':markers,'CUPTI_ACTIVITY_KIND_RUNTIME':apis,'CUPTI_ACTIVITY_KIND_KERNEL':kernels}}
    doc={'header':{'pid':pid},'setup':{'config':CONFIG['id'],'scheduling':{'caller_thread_id':2}},'footer':{'graph_calls':36},'calls':calls,'settle_metadata':{'condition_settle_ms':1000,'iterations':1,'qpc_frequency':1000,'qpc_begin':18,'qpc_end':1018}}
    return trace,doc

def test_settle_work_is_retained_but_not_mapped_or_estimated(monkeypatch):
    trace,doc=fixture();monkeypatch.setattr(nt,'read_trace',lambda _:trace)
    result=nt.analyze_trace('mock',doc,CONFIG)
    assert result['all36_chain_complete'] and result['formal_30_complete'] and result['actual_kernels_mapped']==288
    assert result['formal_kernel_union']['count']==30 and result['formal_kernel_union']['median_ns']==240
    assert result['unmapped_kernel_rowids']==[99999]
    assert result['unmapped_kernels_verbatim'][0]['evidence_rowid']==99999
    assert result['settle_work']['observed_kernel_count']==1 and result['settle_work']['estimator_time_used'] is False

@pytest.mark.parametrize('mutation',['missing','early','short','wrong_thread','no_work'])
def test_settle_trace_rejects_bad_evidence(monkeypatch,mutation):
    trace,doc=fixture();marker=trace['tables']['NVTX_EVENTS'][-1]
    if mutation=='missing':trace['tables']['NVTX_EVENTS'].pop()
    elif mutation=='early':marker['start']=850
    elif mutation=='short':marker['end']=999_999
    elif mutation=='wrong_thread':marker['globalTid']+=1
    else:trace['tables']['CUPTI_ACTIVITY_KIND_KERNEL'].pop()
    monkeypatch.setattr(nt,'read_trace',lambda _:trace)
    with pytest.raises(ValueError,match='settle'):nt.analyze_trace('mock',doc,CONFIG)

def test_strict_native_pid_binding_still_rejects_other_process(monkeypatch):
    trace,doc=fixture();doc['header']['pid']=99;monkeypatch.setattr(nt,'read_trace',lambda _:trace)
    with pytest.raises(ValueError,match='actual native header'):nt.analyze_trace('mock',doc,CONFIG)
