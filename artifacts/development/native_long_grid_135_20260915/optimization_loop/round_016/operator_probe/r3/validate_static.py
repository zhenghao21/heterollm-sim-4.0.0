from pathlib import Path
import json,hashlib,datetime,py_compile
P=Path(__file__).resolve().parent
source=next(Path(x['path']).parent for x in json.loads((P/'identity_lock.json').read_text())['files'] if Path(x['path']).name=='quantize.cu')
anchors={
'quantize.cu':[('MMVQ activation half scale and original sum','const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);'),('MMVQ half2 store','y[ib].ds = make_half2(d, sum);'),('MMQ F32 inverse','const float d_inv = 127.0f / amax;'),('MMQ reciprocal','const float d = 1.0f / d_inv;'),('MMQ F32 D4 store','y[ib].d4[iqs/32]  = d;')],
'mmq.cuh':[('Q5 D4 dispatch','case GGML_TYPE_Q5_0:'),('Q8 D4 dispatch','case GGML_TYPE_Q8_0:')],
'mmq-load-tiles.cuh':[('Q5 signed dot weight transform','qs0     = __vsubss4(qs0, 0x10101010);')],
'vecdotq.cuh':[('Q5 MMVQ original sum correction','return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);'),('Q8 signed dot','return d8_0*d8_1 * ((T) sumi);')]
}
evidence=[]
for name,entries in anchors.items():
 p=source/name;s=p.read_text();lines=s.splitlines()
 for purpose,anchor in entries:
  matching=[i+1 for i,line in enumerate(lines) if anchor in line]
  if not matching:raise RuntimeError('Missing source anchor '+anchor)
  evidence.append({'purpose':purpose,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matching_lines':matching,'anchor':anchor})
(P/'source_semantics_evidence.json').write_text(json.dumps({'source_commit':json.loads((P/'identity_lock.json').read_text())['source_commit'],'source_runtime_equivalence_proven':False,'anchors':evidence,'gpu_access':False},indent=2)+'\n')
for p in P.glob('*.py'):py_compile.compile(str(p),doraise=True)
s=(P/'stream_event_probe.cpp').read_text()
assert 'o.batch!=1' in s and 'call<o.batch' not in s
assert s.count('ggml_backend_graph_compute_async(r.backend,g)')==1
assert s.index('v.evict_start=qpc()')<s.index('nvtxRangePushA(v.label.c_str())')<s.index('ggml_backend_graph_compute_async(r.backend,g)')<s.index('nvtxRangePop()')<s.index('v.validation_start=qpc()')
assert 'Q8_0 original math gate' not in s
(P/'static_validation.json').write_text(json.dumps({'status':'passed','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'single_graph_call':True,'cache_sweep_before_NVTX':True,'validation_outside_NVTX_and_timing':True,'both_quant_source_path_gates':True,'python_syntax_files':len(list(P.glob('*.py'))),'gpu_access':False},indent=2)+'\n')
print('source semantic and static host validation passed')
