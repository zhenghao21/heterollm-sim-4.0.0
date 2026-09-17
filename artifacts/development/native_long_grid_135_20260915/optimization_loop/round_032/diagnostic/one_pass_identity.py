"""One diagnostic read only; cannot accept or resume the rejected R32 campaign."""
import hashlib,importlib.util,json,os,sys,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
R32=P.parent
HELPER=R32.parent/'round_027/identity_diagnostic/dual_hash.py'
EXPECTED='decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33'
MODEL=R32.parents[4]/'artifacts/multimodel_20260913/models/smollm2-1.7b-instruct-q4_k_m.gguf'
spec=importlib.util.spec_from_file_location('dual_diagnostic',HELPER);dual=importlib.util.module_from_spec(spec);spec.loader.exec_module(dual)
ref=lambda p:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
selftest=dual.self_test(True)
assert not (P/'one_pass_result.json').exists()
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
stat_before=dual.stat_record(MODEL.stat());chunks=[];length=0;openssl=hashlib.sha256()
with dual.CNGSHA256() as cng, MODEL.open('rb') as stream:
 before=dual.stat_record(os.fstat(stream.fileno()))
 for block in iter(lambda:stream.read(4*1024*1024),b''):
  openssl.update(block)
  for pos in range(0,len(block),dual.MAX_CHUNK):cng.update(block[pos:pos+dual.MAX_CHUNK])
  chunks.append({'offset':length,'bytes':len(block),'openssl_chunk_sha256':hashlib.sha256(block).hexdigest()});length+=len(block)
 after=dual.stat_record(os.fstat(stream.fileno()));cng_digest=cng.hexdigest()
stat_after=dual.stat_record(MODEL.stat());digest=openssl.hexdigest()
fields=('st_dev','st_ino','st_size','st_mtime_ns')
fingerprint=lambda v:tuple(v[k] for k in fields)
result={'schema':'r32-rejected-freeze-single-read-diagnostic/v1','started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_ref':ref(Path(__file__)),'helper_ref':ref(HELPER),'self_test':selftest,'path':str(MODEL.resolve()),'expected_sha256':EXPECTED,'openssl_sha256':digest,'cng_sha256':cng_digest,'bytes_read':length,'chunks':chunks,'path_before':stat_before,'handle_before':before,'handle_after':after,'path_after':stat_after,'original_fingerprint_equal':fingerprint(before)==fingerprint(after)==fingerprint(stat_after),'both_digests_equal_expected':digest==cng_digest==EXPECTED,'read_count':1,'block_bytes':4*1024*1024,'campaign_retry':False,'retroactive_acceptance':False,'native_run':False,'scope':'one later read; original rejection actual digest/stat unavailable; cannot establish earlier cause'}
with (P/'one_pass_result.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['chunks','self_test']},indent=2))
