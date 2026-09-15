"""Synthetic timestamps only: no device API, executable launch or clock measurement."""
from pathlib import Path
import copy,importlib.util,json
import pytest
spec=importlib.util.spec_from_file_location('launch_probe_offline_analyzer',Path(__file__).with_name('analyze_results.py'))
analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
PROTOCOL=json.loads(Path(__file__).with_name('protocol.json').read_text(encoding='utf-8'))

def data():
    raw=[{'type':'run_identity','qpc_frequency_hz':1000000000}]
    raw += [{'type':'qpc_control','ordinal':i,'begin':100*i,'end':100*i+10} for i in range(256)]
    checksums={(k,b):analyzer.expected_checksum(PROTOCOL,k,b) for k in PROTOCOL['kernels'] for b in PROTOCOL['burst_lengths']}
    for ordinal,(phase,repeat,kernel,mode,n) in enumerate(analyzer.expected_order(PROTOCOL)):
        t=1000000*ordinal;begin=t;start=[t,t+100];t+=110;enq=[];sync=[]
        for _ in range(n):
            enq.append([t,t+1000]);t+=1010
            if mode=='stream_sync_each':sync.append({'begin':t,'end':t+500,'cuda_status':0});t+=510
        stop=[t,t+100];t+=110;sync.append({'begin':t,'end':t+500,'cuda_status':0});end=t+500
        raw.append({'type':'sample','phase':phase,'repeat':repeat,'ordinal':ordinal,'kernel':kernel,'supply_mode':mode,'burst_length':n,
            'wall_qpc':[begin,end],'start_event_record_qpc':start,'end_event_record_qpc':stop,'enqueue_qpc':enq,'synchronize_qpc':sync,
            'host_enqueue_call_total_ns':1000*n,'host_enqueue_envelope_ns':enq[-1][1]-enq[0][0],
            'host_sync_call_total_ns':500*len(sync),'total_wall_ns':end-begin,'cuda_event_span_ms':.02,'cuda_event_span_ns':20000,
            'cuda_status':{'start_event_record':0,'end_event_record':0,'last_launch_error':0,'start_event_query_after_sync':0,'end_event_query_after_sync':0,'event_elapsed_time':0},
            'output_validation':{'passed':True,'checked_words':2048,'mismatches':0,'checksum_fnv1a64':checksums[(kernel,n)]}})
    raw.append({'type':'run_complete','status':'complete','sample_count':640})
    return raw

def test_complete_synthetic_protocol_and_independent_output_checks():
    summary=analyzer.summarize(data(),PROTOCOL)
    assert summary['status']=='valid_raw'
    assert summary['raw_warmup_count']==160 and summary['raw_formal_count']==480
    assert len(summary['configurations'])==16
    assert all(r['status']=='stable_diagnostic' for r in summary['configurations'])
    assert summary['coefficient_candidate'] is None and summary['model_profile_updated'] is False

def test_missing_or_reordered_raw_samples_rejected():
    rows=data();rows[258],rows[259]=rows[259],rows[258]
    with pytest.raises(AssertionError,match='predeclared order'):analyzer.summarize(rows,PROTOCOL)

def test_fake_success_flag_does_not_hide_wrong_output():
    rows=data();next(r for r in rows if r['type']=='sample')['output_validation']['checksum_fnv1a64']='0'*16
    summary=analyzer.summarize(rows,PROTOCOL)
    assert summary['status']=='invalid_raw' and summary['invalid_samples']

def test_qpc_overlap_or_derived_time_mismatch_rejected():
    rows=data();first=next(r for r in rows if r['type']=='sample');first['enqueue_qpc'][0][0]=first['start_event_record_qpc'][0]
    with pytest.raises(AssertionError,match='overlap'):analyzer.summarize(rows,PROTOCOL)
    rows=data();next(r for r in rows if r['type']=='sample')['host_enqueue_call_total_ns']+=100
    with pytest.raises(AssertionError,match='derived host duration'):analyzer.summarize(rows,PROTOCOL)

def test_instability_and_resolution_limits_do_not_write_coefficients():
    rows=data()
    for r in rows:
        if r['type']=='sample' and r['phase']=='formal' and r['kernel']=='empty' and r['burst_length']==1:
            value=100 if r['repeat']<15 else 10000
            r['cuda_event_span_ms']=value/1e6;r['cuda_event_span_ns']=value
    summary=analyzer.summarize(rows,PROTOCOL)
    assert any(r['reasons'] for r in summary['configurations'])
    assert summary['coefficient_candidate'] is None
    rows=data()
    for r in rows:
        if r['type']=='sample':r['cuda_event_span_ns']=500;r['cuda_event_span_ms']=.0005
    summary=analyzer.summarize(rows,PROTOCOL)
    assert all(r['metrics']['cuda_event_span_ns']['gate_status']=='resolution_limited' for r in summary['configurations'])
