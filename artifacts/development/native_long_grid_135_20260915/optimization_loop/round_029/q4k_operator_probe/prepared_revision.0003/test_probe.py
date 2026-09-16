"""Host only: packed arithmetic, callback fixtures and future-runner failures."""
from pathlib import Path
import copy,json,math,struct
import pytest
import reference as r
import build,run_probe as run


def test_scale_min_all_six_bit_values_and_high_bits():
    for base in range(64):
        scales=[(base+j*7)%64 for j in range(8)];mins=[(base+j*13)%64 for j in range(8)]
        packed=r.pack_scales(scales,mins)
        assert [r.decode_scales(packed,j) for j in range(8)]==list(zip(scales,mins))


def test_Q8_nonzero_dyadic_reference_exact():
    x=r.make_input();q8=r.expected_q8(x)
    assert len(q8)==2304
    for i in range(64):
        d,s=struct.unpack_from('<ee',q8,i*36);q=struct.unpack_from('<32b',q8,i*36+4)
        assert max(q)==127 and min(q)<0 and s!=0 and s==d*sum(q)


def test_grouped_CPU_reference_matches_independent_elementwise_interpretation():
    packed=r.make_weights(3);x=r.make_input();q8=r.expected_q8(x)
    expected=r.reference(packed,q8,3)
    assert expected['values']==r.direct_reference(packed,x,3)
    assert all(v>0 for v in expected['bounds'])
    assert all(c>0 for c in expected['min_correction'])


def test_missing_min_correction_fails_derived_bound():
    packed=r.make_weights(3);q=r.expected_q8(r.make_input())
    good=r.reference(packed,q,3);bad=r.reference(packed,q,3,ignore_min=True)
    assert all(abs(a-b)>bound for a,b,bound in zip(good['values'],bad['values'],good['bounds']))


def test_fp32_sequential_elementwise_within_conservative_bound():
    packed=r.make_weights(2);x=r.make_input();q=r.expected_q8(x);expected=r.reference(packed,q,2)
    f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
    values=struct.unpack('<2048f',x)
    for row in range(2):
        total=0.0
        for i in range(2048):
            block,rem=divmod(i,256);g,j=divmod(rem,32);off=(row*8+block)*144
            d,dm=struct.unpack_from('<ee',packed,off);scale,minimum=r.decode_scales(packed[off+4:off+16],g)
            nibble=(packed[off+16+(g//2)*32+j]>>(4*(g%2)))&15
            total=f32(total+f32(f32(f32(d*scale)*nibble-f32(dm*minimum))*values[i]))
        assert abs(total-expected['values'][row])<=expected['bounds'][row]


def captured():
    protocol=json.loads((build.HERE/'protocol.json').read_text())
    raw={'schema':'r29-q4k-target-launch-capture/v1','status':'path_observed_pending_independent_numeric',
        'timed':False,'performance_parameter_admitted':False,'target_LLM_latency_used':False,
        'protocol_sha256':build.ref(build.HERE/'protocol.json')['sha256'],'configuration':protocol['shape'],
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


def test_synthetic_capture_passes_but_is_not_runtime_claim():
    raw,p=captured();assert run.validate_capture(raw,p)


@pytest.mark.parametrize('mutation',['shape','symbol','grid','PDL','pointer','stride','cost','protocol','alias','fastdiv'])
def test_capture_changes_rejected(mutation):
    raw,p=captured();raw=copy.deepcopy(raw)
    if mutation=='shape':raw['configuration']['M']=4
    elif mutation=='symbol':raw['launches'][1]['symbol']='wrong'
    elif mutation=='grid':raw['launches'][1]['block']=[32,2,1]
    elif mutation=='PDL':raw['launches'][1]['attributes'][0]['programmaticStreamSerializationAllowed']=0
    elif mutation=='pointer':raw['launches'][1]['arguments']['vy']=8
    elif mutation=='stride':raw['launches'][1]['arguments']['stride_row_x']=64
    elif mutation=='cost':raw['timed']=True
    elif mutation=='protocol':raw['protocol_sha256']='0'*64
    elif mutation=='alias':raw['launches'][1]['arguments']['vx']=1
    else:raw['launches'][1]['arguments']['sample_ratio_fastdiv']=[0,0,0]
    with pytest.raises(ValueError):run.validate_capture(raw,p)


def test_numerical_full_output_and_Q8_mismatches_retained(tmp_path,monkeypatch):
    p=json.loads((build.HERE/'protocol.json').read_text());files=p['fixture_files']
    expected=Path(files['reference.f64.bin']['path']).read_bytes();values=struct.unpack('<2048d',expected)
    (tmp_path/'actual.f32.bin').write_bytes(struct.pack('<2048f',*values))
    q8=bytearray(Path(files['expected.q8_1.bin']['path']).read_bytes());q8[4]^=1
    (tmp_path/'actual.q8_1.bin').write_bytes(q8)
    with pytest.raises(ValueError,match='numerical'):run.validate_numeric(tmp_path,p)
    evidence=json.loads((tmp_path/'numerical.json').read_text())
    assert evidence['Q8_byte_mismatch_count']==1 and not evidence['passed'] and len(evidence['full_outputs'])==2048


def test_untimed_runner_launch_error_retained_no_kill(tmp_path,monkeypatch):
    protocol={'fixture_manifest':{}};(tmp_path/'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(run,'HERE',tmp_path)
    frozen={'target_executable':{'path':'never-run'}}
    monkeypatch.setattr(run,'snapshot',lambda *a:frozen)
    monkeypatch.setattr(run,'idle',lambda *a:{'conflicts':[]})
    monkeypatch.setattr(run,'environment',lambda *a:{})
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(OSError('synthetic launch failure')))
    with pytest.raises(ValueError,match='rejected'):run.execute(tmp_path/'manifest')
    finish=json.loads((tmp_path/'runs/capture.0001/finish.json').read_text())
    assert finish['status']=='rejected' and finish['returncode'] is None and not finish['termination_requested']
    with pytest.raises(FileExistsError):run.execute(tmp_path/'manifest')


def test_source_has_no_timing_or_LLama_measurement_entry():
    source=(build.HERE/'run_probe.py').read_text()
    assert 'process.kill(' not in source
    assert not any(api in source for api in ('cudaEventCreate','cudaEventRecord','cudaEventElapsedTime'))
    assert 'create_thread' not in source



def test_final_missing_raw_file_cannot_qualify(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text(json.dumps({'fixture_manifest':{}}))
    monkeypatch.setattr(run,'HERE',tmp_path)
    monkeypatch.setattr(run,'snapshot',lambda *a:{'target_executable':{'path':'never-run'}})
    monkeypatch.setattr(run,'idle',lambda *a:{'conflicts':[]})
    monkeypatch.setattr(run,'environment',lambda *a:{})
    folder=tmp_path/'runs/capture.0001'
    class Child:
        pid=1
        def wait(self):
            (folder/'raw.json').write_text('{}')
            (folder/'actual.q8_1.bin').write_bytes(b'')
            (folder/'actual.f32.bin').write_bytes(b'')
            return 0
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k:Child())
    monkeypatch.setattr(run,'validate_capture',lambda *a:True)
    def numeric(*args):
        (folder/'raw.json').unlink();return {'fixture':'synthetic'}
    monkeypatch.setattr(run,'validate_numeric',numeric)
    with pytest.raises(ValueError,match='rejected'):run.execute(tmp_path/'manifest')
    evidence=json.loads((folder/'finish.json').read_text())
    assert evidence['path_qualified'] and evidence['numeric_qualified'] and evidence['status']=='rejected'


@pytest.mark.parametrize('name,executable,parent',[
    ('q4k-target.0003.exe',None,0),
    ('Q4K-TARGET.0002.EXE',None,987654),
    ('q4k-target.0001.exe','C:/other/q4k-target.0001.exe',0),
    ('renamed.exe',str(build.HERE/'q4k-target.0003.exe'),0),
    ('direct.exe',str(build.HERE/'direct.exe'),0),
    ('nested.exe',str(build.HERE/'subdirectory/nested.exe'),456),
])
def test_local_target_or_orphan_is_blocked_without_parent(name,executable,parent):
    row={'pid':100,'ppid':parent,'name':name,'executable_path':executable,'cmdline':None}
    bad=run.local_process_conflicts([row],current_pid=999)
    assert len(bad)==1 and bad[0]['pid']==100 and bad[0]['reason'].startswith('blocked_r29_')


@pytest.mark.parametrize('name,executable',[
    ('q4k-host-tests.0003.exe',str(build.HERE/'q4k-host-tests.0003.exe')),
    ('unrelated.exe','C:/elsewhere/unrelated.exe'),
    ('q4k-target.0003.exe.txt',None),
    ('q4k-target.0003.exe.backup',None),
])
def test_local_guard_keeps_host_tests_and_unrelated_processes(name,executable):
    assert not run.local_process_conflicts([{'pid':100,'ppid':0,'name':name,'executable_path':executable}],current_pid=999)


def test_idle_combines_frozen_guard_and_local_executable_inventory(monkeypatch):
    protocol={'execution_guard_refs':[{'path':'frozen-guard','sha256':'fixed'}]}
    observed=[]
    class Guard:
        def process_conflicts(self,rows):
            observed.extend(rows)
            return [{'pid':101,'reason':'blocked_project'}]
    monkeypatch.setattr(run,'load_guard',lambda _:Guard())
    class Child:
        pid=444;returncode=0
        def communicate(self):
            return json.dumps([
                {'ProcessId':100,'ParentProcessId':0,'Name':'q4k-target.0003.exe','CommandLine':None,'ExecutablePath':None},
                {'ProcessId':101,'ParentProcessId':0,'Name':'python.exe','CommandLine':'run_candidate.py','ExecutablePath':'C:/python.exe'}]),''
    def popen(command,**kwargs):
        assert 'ExecutablePath' in command[-1]
        return Child()
    monkeypatch.setattr(run.subprocess,'Popen',popen)
    value=run.process_snapshot(protocol)
    assert {r['pid'] for r in value['conflicts']}=={100,101}
    assert observed[0]['executable_path'] is None and value['frozen_base_guard_ref']==protocol['execution_guard_refs'][0]
    assert value['local_checker_ref']['sha256']==build.ref(run.__file__)['sha256']
    with pytest.raises(ValueError,match='not idle'):run.idle(protocol)


@pytest.mark.parametrize('name,classification,after,exit_seen,code',[
    ('cudaMalloc','allocation',True,True,0),
    ('cudaMallocAsync','allocation',True,True,0),
    ('cudaFreeAsync','allocation',True,True,0),
    ('cudaHostAlloc','allocation',True,True,0),
    ('cudaMemcpyAsync','memory_copy_or_set',True,True,0),
    ('cudaFreeAsync','synchronization',True,True,0),
    ('cudaStreamSynchronize','synchronization',False,True,0),
    ('cudaStreamSynchronize','synchronization',True,False,0),
    ('cudaStreamSynchronize','synchronization',True,True,7),
])
def test_auxiliary_allocation_free_and_nonterminal_sync_rejected(name,classification,after,exit_seen,code):
    raw,p=captured()
    raw['runtime_auxiliary_count']=1
    raw['runtime_auxiliary_calls']=[{'api_name':name,'classification':classification,
        'after_launch_pair':after,'exit_seen':exit_seen,'return_code':code}]
    with pytest.raises(ValueError):run.validate_capture(raw,p)


@pytest.mark.parametrize('name',['cudaStreamSynchronize','cudaStreamSynchronize_ptsz','cudaDeviceSynchronize','cudaEventSynchronize'])
def test_successful_terminal_sync_allowed(name):
    raw,p=captured();raw['runtime_auxiliary_count']=1
    raw['runtime_auxiliary_calls']=[{'api_name':name,'classification':'synchronization',
        'after_launch_pair':True,'exit_seen':True,'return_code':0}]
    assert run.validate_capture(raw,p)


def test_auxiliary_coverage_mismatch_rejected():
    raw,p=captured();raw['runtime_auxiliary_count']=1
    with pytest.raises(ValueError,match='coverage'):run.validate_capture(raw,p)
