"""Strict effective-argument comparison; no latency data or broad normalization."""
from copy import deepcopy
import json
import math
import struct
from pathlib import Path

TRUTH_FLAGS=('geometry_observed','supported_api','decoded','exit_seen','extended_api','attribute_list_observed','attributes_source_qualified')
FALSE_FLAGS=('api_name_mismatch','missing_config','missing_attributes','truncated_attributes','unknown_attributes')
POINTER_FIELDS=({'x','vy'},{'vx','vy','dst'})
HARDWARE={'uuid':'GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b','major':12,'minor':0,'SM_count':84,'warp_size':32}
CONFIG={'format':'Q5_0','type_id':6,'M':1,'K':4096,'N':3072,'seed':20260916,'ids':False,'fusion':False,'graph_replay':False}
SYMBOLS=('_Z13quantize_q8_1PKfPvxxxxxj5uint3','_Z13mul_mat_vec_qIL9ggml_type6ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj')

def same(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
def need(condition,reason):
    if not condition:raise ValueError(reason)
def natural(value,name,positive=False):
    need(type(value) is int and value>=(1 if positive else 0),'missing/invalid integer '+name)
    return value

def validate(raw,role):
    need(raw.get('status')=='qualified_runtime_pair','unqualified '+role+' raw record')
    for key in ('timed','performance_parameter_admitted','target_LLM_latency_used','device_pointer_dereferenced_in_callback','overflow','malformed_callback'):
        need(raw.get(key) is False,'invalid scope/status '+role+':'+key)
    need(same(raw.get('configuration'),CONFIG) and same(raw.get('hardware'),HARDWARE),'configuration/hardware differs')
    if role=='wrapper':
        need(raw.get('origin')=='standalone_source_wrapper' and raw.get('gpu_launch_pair_executed') is True and raw.get('gpu_graph_executed') is False,'wrapper execution origin differs')
    else:need(raw.get('gpu_graph_executed') is True,'target graph not executed')
    need(natural(raw.get('launch_count'),'launch_count')==2 and len(raw.get('launches',[]))==2,'exact two kernels required')
    need(natural(raw.get('memory_api_count'),'memory_api_count')==0,'memory copy/set inside pair')
    aux=raw.get('runtime_auxiliary_calls')
    need(isinstance(aux,list) and natural(raw.get('runtime_auxiliary_count'),'runtime_auxiliary_count')==len(aux),'auxiliary count differs')
    need(len(aux)==1 and aux[0].get('api_name')=='cudaStreamSynchronize' and aux[0].get('classification')=='synchronization' and aux[0].get('exit_seen') is True and type(aux[0].get('return_code')) is int and aux[0]['return_code']==0,'target pair must finish with one successful synchronization')
    if role=='wrapper':
        need(raw.get('new_shim_numerical_qualified') is True and raw.get('launch_pair_qualified') is True,'new shim numerical/path qualification missing')
        numeric=raw.get('numerical_check',{})
        need(numeric.get('same_new_shim_used') is True and numeric.get('separate_post_capture_pass') is True and numeric.get('reference_fixed_before_GPU') is True and numeric.get('capture_includes_numerical_pass') is False,'numerical reference boundary differs')
        need(numeric.get('q8_bytes_compared')==4608 and numeric.get('q8_byte_errors')==0 and numeric.get('outputs_compared')==3072 and numeric.get('output_failures')==0,'new shim numerical result failed')
    events=raw['launches']
    for i,e in enumerate(events):
        need(natural(e.get('index'),'index')==i and e.get('symbol')==SYMBOLS[i],'symbol/order mismatch')
        need(e.get('phase')==('q8_1_conversion','mmvq_main')[i],'phase mismatch')
        need(natural(e.get('api_id'),'api_id')==430 and e.get('api_name')=='cudaLaunchKernelExC','launch API/PDL path differs')
        need(all(e.get(k) is True for k in TRUTH_FLAGS) and all(e.get(k) is False for k in FALSE_FLAGS),'unobserved/unqualified launch data')
        need(natural(e.get('return_code'),'return_code')==0,'launch error')
        for name in ('correlation','context','function'):natural(e.get(name),name,True)
        natural(e.get('stream'),'stream');natural(e.get('shared'),'shared')
        need(same(e.get('grid'),([16,1,1],[3072,1,1])[i]) and same(e.get('block'),([256,1,1],[32,4,1])[i]) and e['shared']==0,'launch geometry mismatch')
        attrs=e.get('attributes')
        need(natural(e.get('reported_attribute_count'),'attrs')==natural(e.get('captured_attribute_count'),'attrs')==1 and isinstance(attrs,list) and len(attrs)==1,'attribute count differs')
        a=attrs[0]
        need(natural(a.get('index'),'attribute index')==0 and natural(a.get('id'),'attribute id')==6 and a.get('is_source_PDL') is True,'unknown PDL attribute')
        need(type(a.get('programmaticStreamSerializationAllowed')) is int and a['programmaticStreamSerializationAllowed']==1,'PDL value differs')
        need(a.get('inactive_union_bytes_are_semantic') is False and natural(a.get('interpreted_prefix_bytes'),'interpreted bytes')==4,'attribute interpretation differs')
        opaque=bytes.fromhex(a.get('opaque_value_bytes_hex',''))
        need(len(opaque)==64 and int.from_bytes(opaque[:4],'little',signed=True)==1,'PDL raw active bytes differ')
    c,m=events;ca,ma=c['arguments'],m['arguments'];t=raw.get('graph_tensors',{})
    need(c['context']==m['context'] and c['stream']==m['stream'],'pair context/stream relationship differs')
    need(c['function']!=m['function'] and c['correlation']!=m['correlation'],'ambiguous kernel identity')
    for field in ('weights','input','output'):natural(t.get(field),field,True)
    for value in [ca.get('x'),ca.get('vy'),ma.get('vx'),ma.get('vy'),ma.get('dst')]:natural(value,'pointer',True)
    need(ca['x']==t['input'] and ma['vx']==t['weights'] and ma['dst']==t['output'] and ca['vy']==ma['vy'],'graph/producer/consumer pointer relation differs')
    need(len({ca['x'],ca['vy'],ma['vx'],ma['dst']})==4,'unexpected pointer alias')
    expected_c={'ne00':4096,'s01':4096,'s02':4096,'s03':4096,'ne0':4096,'ne1':1,'ne2_fastdiv':[1,0,1]}
    expected_m={'ids':0,'fusion':{'x_bias':0,'gate':0,'gate_bias':0,'x_scale':0,'gate_scale':0,'glu_op':0,'glu_limit':0},'ncols_x':4096,'nchannels_y_fastdiv':[0,0,0],
        'stride_row_x':128,'stride_col_y':128,'stride_col_dst':3072,'channel_ratio_fastdiv':[1,0,1],
        'stride_channel_x':393216,'stride_channel_y':128,'stride_channel_dst':3072,'sample_ratio_fastdiv':[1,0,1],
        'stride_sample_x':393216,'stride_sample_y':128,'stride_sample_dst':3072,'ids_stride':0}
    projected=[]
    for e,pointers,expected in zip(events,POINTER_FIELDS,(expected_c,expected_m)):
        args={k:v for k,v in e['arguments'].items() if k not in pointers}
        need(same(args,expected),'complete kernel scalar/layout mismatch (including six strides)')
        projected.append({'api_id':e['api_id'],'api_name':e['api_name'],'symbol':e['symbol'],'phase':e['phase'],
            'grid':e['grid'],'block':e['block'],'dynamic_shared_bytes':e['shared'],'arguments':args,
            'attributes':[{'id':6,'programmaticStreamSerializationAllowed':1}]})
    return {'hardware':raw['hardware'],'configuration':raw['configuration'],'launches':projected,
            'relations':{'same_pair_context':True,'same_pair_stream':True,'distinct_kernel_functions':True,'distinct_correlations':True,
                'input_to_conversion':True,'conversion_to_main':True,'weights_to_main':True,'main_to_output':True,'four_non_aliasing_buffers':True}}

def compare(target,wrapper):
    a=validate(target,'target');b=validate(wrapper,'wrapper')
    need(same(a,b),'effective target/wrapper arguments differ')
    return {'schema':'heterollm.mmvq-wrapper-dynamic-match/v1','status':'effective_launch_pair_matches',
        'scope':'fixed Q5_0 M1 K4096 N3072, no ids/fusion, exact six layout strides, ExC PDL=1',
        'canonical_observations':a,'ignored_cross_process_values':['absolute buffer/function/stream addresses','numeric context/correlation IDs after relations checked','inactive launch-attribute union bytes'],
        'numerical_correctness_verified_by_this_comparison':False,'timing_equivalence_verified':False,'performance_parameter_admitted':False}


def verify_numerical_files(raw,check_ref):
    numeric=raw['numerical_check'];refs=numeric['raw_refs']
    need(set(refs)=={'actual_q8','expected_q8','actual_output','reference_output','bounds'},'numerical evidence membership differs')
    for ref in refs.values():check_ref(ref)
    data={k:Path(r['path']).read_bytes() for k,r in refs.items()}
    need(len(data['actual_q8'])==4608 and data['actual_q8']==data['expected_q8'],'converted bytes differ')
    need(len(data['actual_output'])==3072*4 and len(data['reference_output'])==len(data['bounds'])==3072*8,'numerical evidence dimensions differ')
    actual=struct.unpack('<3072f',data['actual_output']);expected=struct.unpack('<3072d',data['reference_output']);bounds=struct.unpack('<3072d',data['bounds'])
    errors=[];ratios=[]
    for a,b,tolerance in zip(actual,expected,bounds):
        need(all(math.isfinite(v) for v in (a,b,tolerance)) and tolerance>0,'invalid numerical value/bound')
        error=abs(a-b);need(error<=tolerance,'new shim output exceeds predetermined bound');errors.append(error);ratios.append(error/tolerance)
    need(math.isclose(max(errors),numeric['max_absolute_error'],rel_tol=1e-12,abs_tol=1e-15),'reported absolute error differs')
    need(math.isclose(max(ratios),numeric['max_bound_ratio'],rel_tol=1e-12,abs_tol=1e-15),'reported bound ratio differs')
    return {'converted_bytes':4608,'output_values':3072,'all_predetermined_bounds_passed':True,'max_absolute_error':max(errors),'max_bound_ratio':max(ratios)}
