from test_source_reference import *
OLD=P.parents[2]/'round_008/stream_event_probe_v2/runs/20260915T225012796Z'
results=[]
for k in (896,1024):
 n=4096;m=4;state=20260914;vals=np.empty(n*k+m*k,np.float32)
 for i in range(len(vals)):
  state^=(state<<13)&0xffffffff;state^=state>>17;state^=(state<<5)&0xffffffff;vals[i]=((state&65535)-32768)/65536.
 w=vals[:n*k].reshape(n,k//32,32);x=vals[n*k:].reshape(m,k)
 for quant in ('Q5_0','Q8_0'):
  if quant=='Q5_0':
   idx=np.argmax(np.abs(w),axis=2);maximum=np.take_along_axis(w,idx[:,:,None],axis=2)[:,:,0];d=maximum/np.float32(-16);inv=np.divide(np.float32(1),d,out=np.zeros_like(d),where=d!=0);q=np.minimum(31,np.trunc(w*inv[:,:,None]+np.float32(16.5))).astype(np.uint8);hi=np.sum(((q>>4)&1).astype(np.uint32)*(np.uint32(1)<<np.arange(32,dtype=np.uint32)),axis=2,dtype=np.uint32);packed=np.empty((n,k//32,22),np.uint8);packed[:,:,2:6]=hi.astype('<u4').view(np.uint8).reshape(n,k//32,4);packed[:,:,6:]=(q[:,:,:16]&15)|((q[:,:,16:]&15)<<4)
  else:
   d=np.max(np.abs(w),axis=2)/np.float32(127);inv=np.divide(np.float32(1),d,out=np.zeros_like(d),where=d!=0);scaled=w*inv[:,:,None];q=(np.sign(scaled)*np.floor(np.abs(scaled)+np.float32(.5))).astype(np.int8);packed=np.empty((n,k//32,34),np.uint8);packed[:,:,2:]=q.view(np.uint8)
  packed[:,:,:2]=d.astype('<f2').view(np.uint8).reshape(n,k//32,2)
  for m in (1,2,4):
   path=OLD/f'{"dev" if k==896 else "validation"}_{quant}_m{m}_k{k}.event.json';raw=json.loads(path.read_text());rows=raw['runs'][0]['correctness']['samples'];pairs=[(r['n_index'],r['m_index']) for r in rows];host=call(packed,x,k,quant,'MMVQ',pairs);native=np.array([r['actual'] for r in rows]);delta=np.abs(host-native)
   weight_sha=hashlib.sha256(packed.tobytes()).hexdigest();input_sha=hashlib.sha256(x[:m].tobytes()).hexdigest();assert weight_sha==raw['quantization']['packed_weight_sha256'];assert input_sha==raw['quantization']['input_sha256'];failures=int(np.count_nonzero(delta>1e-4+1e-5*np.abs(host)));assert failures==0,(quant,k,m,failures)
   results.append({'config':path.stem,'raw':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'samples':len(rows),'packed_weight_sha_matches':True,'input_sha_matches':True,'source_path_max_abs_error':float(delta.max()),'path_tolerance_failures':failures,'original_evidence_untouched':True})
(P/'old_raw_reference_validation.json').write_text(json.dumps({'status':'numeric_source_reference_regression_passed','gpu_access':False,'timing_fields_used':False,'prior_failures_reclassified':False,'results':results,'limitation':'MMVQ historical numeric samples only; no MMQ native evidence or new runtime pass claim.'},indent=2)+'\n')
print(json.dumps({'configs':len(results),'samples':sum(r['samples'] for r in results),'max_path_error':max(r['source_path_max_abs_error'] for r in results),'gpu_access':False}))
