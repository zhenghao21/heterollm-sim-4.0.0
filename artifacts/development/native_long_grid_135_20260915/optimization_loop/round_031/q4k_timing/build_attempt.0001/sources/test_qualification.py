"""Host-only R31 qualification and timing admission regressions; never launches CUDA."""
from pathlib import Path
import importlib.util,json,struct
import pytest
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r31_host_qualification',P/'qualification.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
spec=importlib.util.spec_from_file_location('r31_host_timing',P/'runner.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def captured():
    protocol=json.loads((P/'protocol.json').read_text())
    raw={'schema':'r31-q4k-wrapper-launch-capture/v1','status':'path_observed_pending_independent_numeric',
        'timed':False,'performance_parameter_admitted':False,'target_LLM_latency_used':False,
        'protocol_sha256':q.ref(P/'protocol.json')['sha256'],'configuration':protocol['shape'],
        'hardware':{'uuid':protocol['GPU_uuid'],'SM_count':84},'compile_CUPTI_API_version':26,'runtime_CUPTI_API_version':130401,
        'CUPTI_path':protocol['CUPTI']['path'],'loaded_modules_before':protocol['target_modules'],'loaded_modules_after':protocol['target_modules'],
        'overflow':False,'malformed_callback':False,'memory_api_count':0,'launch_count':2,
        'runtime_auxiliary_count':0,'runtime_auxiliary_calls':[],
        'pointer_associations':{'verified':True},'graph_tensors':{'input':1,'weights':2,'output':3},'launches':[]}
    args=[{'x':1,'vy':4,'ne00':2048,'ne0':2048,'s01':2048,'s02':2048,'s03':2048,'ne1':1,'ne2_fastdiv':[1,0,1]},
      {'vx':2,'vy':4,'dst':3,'ncols_x':2048,'stride_row_x':8,'stride_col_y':64,'stride_col_dst':2048,
       'stride_channel_x':16384,'stride_sample_x':16384,'stride_channel_y':64,'stride_sample_y':64,
       'stride_channel_dst':2048,'stride_sample_dst':2048,'ids_stride':0,'ids':0,'fusion':{'glu_op':0},
       'channel_ratio_fastdiv':[1,0,1],'sample_ratio_fastdiv':[1,0,1],'nchannels_y_fastdiv':[0,0,0]}]
    for i,key in enumerate(('expected_conversion','expected_main')):
        expected=protocol[key]
        raw['launches'].append({'symbol':expected['symbol'],'api_id':430,'exit_seen':True,'return_code':0,
            'grid':expected['grid'],'block':expected['block'],'shared':0,'geometry_observed':True,
            'attributes_source_qualified':True,'extended_api':True,'reported_attribute_count':1,'captured_attribute_count':1,
            'attributes':[{'id':6,'programmaticStreamSerializationAllowed':1}],
            'context':1,'stream':10,'correlation':10+i,'function':100+i,'arguments':args[i]})
    return raw,protocol


def test_type12_fixed_shape_and_exact_strides():
    raw,p=captured();assert q.validate_capture(raw,p)
    assert p['development']['format']=='Q4_K' and p['development']['K']==p['development']['N']==2048
    assert p['actual_key']['stride_row_x']==8 and p['actual_key']['stride_col_y']==64
    assert p['actual_key']['stride_channel_x']==16384 and p['qualification_budget']['captured_pairs']==1
    assert p['sampling']['maximum_formal_processes']==100 and p['sampling']['maximum_extensions']==0


@pytest.mark.parametrize('mutate',['type','shape','stride','PDL','allocation','early_sync'])
def test_Q4_wrapper_path_mutations_rejected(mutate):
    raw,p=captured()
    if mutate=='type':raw['launches'][1]['symbol']=raw['launches'][1]['symbol'].replace('type12E','type6E')
    elif mutate=='shape':raw['configuration']={**raw['configuration'],'M':4}
    elif mutate=='stride':raw['launches'][1]['arguments']['stride_row_x']=64
    elif mutate=='PDL':raw['launches'][1]['attributes'][0]['programmaticStreamSerializationAllowed']=0
    else:
        raw['runtime_auxiliary_count']=1;raw['runtime_auxiliary_calls']=[{'api_name':'cudaMallocAsync' if mutate=='allocation' else 'cudaStreamSynchronize',
            'classification':'allocation' if mutate=='allocation' else 'synchronization','after_launch_pair':mutate!='early_sync','exit_seen':True,'return_code':0}]
    with pytest.raises(ValueError):q.validate_capture(raw,p)


@pytest.mark.parametrize('name',['q4k-target.0003.exe','q4k-timing.exe','q4k-wrapper-qualification.exe'])
def test_direct_or_orphan_target_is_excluded(name):
    assert q.local_process_conflicts([{'pid':100,'ppid':0,'name':name,'executable_path':None}],current_pid=999)


def test_new_wrapper_gate_cannot_borrow_R29_success(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'P',tmp_path)
    (tmp_path/'protocol.json').write_text('{}')
    monkeypatch.setattr(q.b,'verify_build',lambda:{})
    with pytest.raises(FileNotFoundError):q.verify_qualified()


def test_timing_extension_is_rejected_before_any_process(tmp_path):
    with pytest.raises(ValueError,match='no extension'):r.campaign(tmp_path/'run.0001','extend')


def test_new_wrapper_independent_numeric_rejects_one_changed_Q8_byte(tmp_path):
    p=q.load(P/'protocol.json');files=p['fixture_files'];values=struct.unpack('<2048d',Path(files['reference.f64.bin']['path']).read_bytes())
    (tmp_path/'actual.f32.bin').write_bytes(struct.pack('<2048f',*values))
    data=bytearray(Path(files['expected.q8_1.bin']['path']).read_bytes());data[4]^=1
    (tmp_path/'actual.q8_1.bin').write_bytes(data)
    with pytest.raises(ValueError,match='numerical'):q.validate_numeric(tmp_path,p)
    result=q.load(tmp_path/'numerical.json');assert result['Q8_byte_mismatch_count']==1 and not result['passed']


def test_timing_rejects_substituted_reference_even_with_matching_local_hash(tmp_path):
    mapping={'packed_q4':'weights.q4_k.bin','input_f32':'input.f32.bin','expected_q8':'expected.q8_1.bin','reference_f64':'reference.f64.bin','bound_f64':'bounds.f64.bin'}
    p=q.load(P/'protocol.json');refs={key:p['fixture_files'][name] for key,name in mapping.items()}
    fake=tmp_path/'fake.bin';fake.write_bytes(b'not the frozen weights');refs['packed_q4']=q.ref(fake)
    with pytest.raises(ValueError,match='fixture changed'):r.validate_numeric({'references':refs})
