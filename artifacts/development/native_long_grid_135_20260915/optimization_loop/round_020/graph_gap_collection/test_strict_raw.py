import copy,json,hashlib,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import strict_raw
from native_trace import expected_calls
def ref(p):
 b=Path(p).read_bytes();return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def fixture(tmp_path,arm='control'):
 mod=tmp_path/'m.dll';mod.write_bytes(b'm'); manifest={'protocol':{'sha256':'a'*64},'files':[ref(mod)]}; protocol={'runtime':{'environment':{'X':'1'},'extra_clear_environment':['Y']}}
 stage={'arm':arm,'pair_id':'p','config':{'id':'c','nodes':2,'elements':4},'app_argv':['exe','--arm',arm]}; rows=[]; h={'record':'header','schema':'graph-gap-probe/v1','protocol_sha256':'a'*64,'pid':12,'qpc_start':1,'qpc_frequency':10,'argv':stage['app_argv']}; rows+=[h,{'record':'setup','config':'c','arm':arm,'pair_id':'p','scheduling':{'caller_thread_id':2},'environment':{'X':'1','Y':None},'requested_nodes':2,'actual_ggml_nodes':2,'elements':4,'tensor_payload_bytes':16,'allocated_payload_bytes':48,'logical_read_bytes':32,'logical_write_bytes':32,'allocated_buffer_bytes':48,'graph_nodes':[{'index':0,'name':'scale_1','src0':'graph_input','operator':'SCALE','dtype':'F32','scale':.5,'ne':[4,1,1,1]},{'index':1,'name':'scale_2','src0':'scale_1','operator':'SCALE','dtype':'F32','scale':.5,'ne':[4,1,1,1]}],'loaded_modules_before':[ref(mod)]},{'record':'arm_metadata','arm':arm,'config':'c','actual_argv':stage['app_argv'],'freeze':{'protocol_sha256':'a'*64}}]
 if arm=='buffered': rows.append({'record':'buffered_started','arm':'buffered'})
 q=2
 for phase,i in expected_calls():
  times=list(range(q,q+16));q+=16;rows.append({'record':'graph_call','label':f'graph_submit/c/{phase}/{i}','phase':phase,'index':i,'arm':arm,'graph_status':0,'observed_device_kernel_count':None,'observed_cuda_graph_launch_count':None,**dict(zip(['qpc_outer_push_start','qpc_outer_push_end','qpc_submit_push_start','qpc_submit_push_end','qpc_submit_start','qpc_submit_end','qpc_submit_pop_start','qpc_submit_pop_end','qpc_sync_push_start','qpc_sync_push_end','qpc_sync_start','qpc_sync_end','qpc_sync_pop_start','qpc_sync_pop_end','qpc_outer_pop_start','qpc_outer_pop_end'],times))})
 blocks=[]
 if arm=='control':
  for o in range(36):
   blocks.append({'record':'numeric_block','arm':arm,'name':'per_call_final','after_ordinal':o,'stages':[{'stage':2,'result':{'checked':4,'mismatches':0,'nonfinite':0,'raw_fnv1a64':'0'*16,'bitwise_pass':True}}]})
   if o in (0,35):blocks.append({'record':'numeric_block','arm':arm,'name':'first_or_postformal_intermediate_stages','after_ordinal':o,'stages':[{'stage':1,'result':{'checked':4,'mismatches':0,'nonfinite':0,'raw_fnv1a64':'0'*16,'bitwise_pass':True}}]})
 else:
  for name,o in [('first',0),('post_warmup',5),('post_formal',35)]:blocks.append({'record':'numeric_block','arm':arm,'name':name,'after_ordinal':o,'stages':[{'stage':1,'result':{'checked':4,'mismatches':0,'nonfinite':0,'raw_fnv1a64':'0'*16,'bitwise_pass':True}},{'stage':2,'result':{'checked':4,'mismatches':0,'nonfinite':0,'raw_fnv1a64':'0'*16,'bitwise_pass':True}}]})
 rows+=blocks; checked=sum(x['result']['checked'] for b in blocks for x in b['stages']);rows.append({'record':'footer','status':'complete','arm':arm,'graph_calls':36,'numeric_blocks':len(blocks),'checked_values':checked,'mismatches':0,'nonfinite':0,'graph_status_pass':True,'math_pass':True,'loaded_modules_after':[ref(mod)],'modules_stable':True,'qpc_end':q})
 raw=tmp_path/f'{arm}.jsonl';raw.write_text('\n'.join(json.dumps(x) for x in rows)+'\n'); side={'raw_sha256':ref(raw)['sha256'],'raw_bytes':raw.stat().st_size,'raw_path':str(raw.resolve()),'protocol_sha256':'a'*64,'arm':arm,'pair_id':'p','config':'c'};(tmp_path/f'{arm}.jsonl.receipt.json').write_text(json.dumps(side));return raw,stage,manifest,protocol,rows
def test_control_and_buffered_pass(tmp_path):
 for arm in ('control','buffered'):
  raw,stage,m,p,_=fixture(tmp_path,arm); assert strict_raw.audit(raw,stage,m,p,12)['math_pass']
@pytest.mark.parametrize('kind,msg',[('pair','config/arm/pair'),('argv','exact actual argv'),('pid','direct PID'),('qpc','QPC ordering'),('stage','numeric cadence'),('footer','footer count'),('math','footer status'),('module','module lifetime'),('sha','closed raw')])
def test_rejects_critical_tamper(tmp_path,kind,msg):
 raw,stage,m,p,rows=fixture(tmp_path); 
 if kind=='pair':rows[1]['pair_id']='x'
 elif kind=='argv':rows[0]['argv']=['x']
 elif kind=='pid':rows[0]['pid']=99
 elif kind=='qpc':rows[3]['qpc_submit_end']=0
 elif kind=='stage':rows[39]['stages'][0]['stage']=1
 elif kind=='footer':rows[-1]['graph_calls']=35
 elif kind=='math':rows[-1]['math_pass']=False
 elif kind=='module':rows[1]['loaded_modules_before'][0]['sha256']='f'*64
 elif kind=='sha': pass
 raw.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
 side=json.loads((Path(str(raw)+'.receipt.json')).read_text())
 if kind=='sha': side['raw_sha256']='0'*64
 else: side.update(raw_sha256=ref(raw)['sha256'],raw_bytes=raw.stat().st_size)
 Path(str(raw)+'.receipt.json').write_text(json.dumps(side))
 with pytest.raises(ValueError,match=msg):strict_raw.audit(raw,stage,m,p,12)
def test_buffered_requires_start_and_per_call_not_claimed(tmp_path):
 raw,stage,m,p,rows=fixture(tmp_path,'buffered');rows=[r for r in rows if r.get('record')!='buffered_started'];raw.write_text('\n'.join(json.dumps(x) for x in rows)+'\n');side=json.loads(Path(str(raw)+'.receipt.json').read_text());side.update(raw_sha256=ref(raw)['sha256'],raw_bytes=raw.stat().st_size);Path(str(raw)+'.receipt.json').write_text(json.dumps(side));
 with pytest.raises(ValueError,match='buffered_started'):strict_raw.audit(raw,stage,m,p,12)
