"""Host-only contract tests; fabricated data, never evidence of GPU accuracy."""
import copy,json,unittest,tempfile,hashlib
from pathlib import Path
from full_raw_audit import HERE,load,audit_run
from assess import assess
from summarize_matrix import raw_digest_issues,verify_manifest

PROTOCOL=load(HERE/'protocol.json')
CFG=PROTOCOL['configs'][0]

def fixture(mode='event'):
    samples=[{'m_index':0,'n_index':i,'actual':1.0,'reference':1.0,'math_reference':1.0,'math_absolute_error':0.,'math_pass':True,'path_reference':1.,'path_absolute_error':0.,'path_pass':True,'pass':True} for i in range(4096)]
    check={'passed':True,'finite_all_outputs':True,'sample_count':4096,'samples':samples}
    rows=[]
    for phase,index in [('first_call',0)]+[('warmup',i) for i in range(5)]+[('formal',i) for i in range(30)]:
        row={'phase':phase,'index':index,'graph_computations':64,'qpc_start':0,'qpc_record_begin_start':1,'qpc_record_begin_end':2,'qpc_submit_start':3,'qpc_submit_end':70,'qpc_record_end_start':71,'qpc_record_end_end':72,'qpc_wait_start':73,'qpc_wait_end':100,'qpc_end':100,'host_wall_ns':100000,'host_per_graph_ns':1562.5,'correctness':check,'event_envelope_ms':0.09 if mode=='event' else None,'cuda_query_before_wait':600}
        row.update({key:0 for key in ['ggml_status','cuda_submit_status','cuda_wait_status','cuda_begin_record_status','cuda_end_record_status','cuda_query_after_wait','cuda_elapsed_status']});rows.append(row)
    modules=[{'path':'F:\\fixture\\stream-event-probe.exe','sha256':'a'*64,'bytes':1234}]
    doc={'schema':'backend-stream-event-probe/v3','M':1,'N':4096,'K':896,'seed':20260914,'group':'dev','weight_format':'Q5_0','input_dtype':'F32','output_dtype':'F32','layout':'ordinary_contiguous_2d','device':'cuda','threads':1,'cuda_index':0,'status':'measured','modules_stable':True,'supported':True,'wait_api':'ggml_backend_synchronize','cache_policy':'same_buffers_repeated_hot_cache_no_flush','graph_computations_per_batch':64,'graph_compute_calls':2304,'warmup_requested':5,'formal_repeats_requested':30,'control_mode':mode=='control','qpc_frequency':1000000,'driver_version':13040,'runtime_version':12080,'compute_capability_major':12,'compute_capability_minor':0,'timing_contract':{'id':'actual-backend-stream-batch-envelope/v2','host_device_times_additive':False},'environment':{'GGML_CUDA_DISABLE_GRAPHS':'1',**{k:None for k in ['GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','CUDA_VISIBLE_DEVICES','LLAMA_TRACE_ANNOTATIONS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_CUBLAS_COMPUTE_TYPE']}},'quantization':{'packed_weight_sha256':'b'*64,'input_sha256':'c'*64,'bytes':1234},'loaded_modules_before':modules,'loaded_modules_after':modules,'correctness_contract':{'absolute_tolerance':0.05,'relative_tolerance':0.03,'path_absolute_tolerance':.0001,'path_relative_tolerance':.00001,'reference_mode':'dual_math_and_source_path','source_runtime_equivalence_proven':False},'first_call_correctness':check,'final_correctness':check,'runs':rows}
    doc.update({k:'fixture' for k in ['backend_name','device_name','device_description','gpu_name','pci_bus_id']})
    return doc

class Contracts(unittest.TestCase):
    def test_complete_fixture(self):
        self.assertEqual(audit_run(fixture(),CFG,'event',PROTOCOL),[])
        self.assertTrue(assess(fixture(),fixture('control'),CFG,PROTOCOL,None)['accepted'])
    def test_missing_both_identities_rejected(self):
        a,b=fixture(),fixture('control');a.pop('pci_bus_id');b.pop('pci_bus_id')
        self.assertFalse(assess(a,b,CFG,PROTOCOL,None)['accepted'])
    def test_partial_submission_rejected(self):
        d=fixture();d['runs'][0]['graph_computations']=63
        self.assertTrue(any('submission' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_warmup_numeric_failure_rejected(self):
        d=fixture();d['runs'][1]['correctness']={'passed':False}
        self.assertTrue(any('numerical failure' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_forged_pass_with_bad_numeric_rejected(self):
        d=fixture();d['runs'][0]['correctness']['samples'][0]['actual']=2
        self.assertTrue(any('numerical sample rejected' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_bad_reference_index_rejected(self):
        d=fixture();d['runs'][0]['correctness']['samples'][0]['n_index']=1
        self.assertTrue(any('index mismatch' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_timing_derivation_rejected(self):
        d=fixture();d['runs'][0]['host_per_graph_ns']=100000
        self.assertTrue(any('derivation' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_wait_method_mismatch_rejected(self):
        d=fixture();d['wait_api']='cudaEventSynchronize'
        self.assertTrue(audit_run(d,CFG,'event',PROTOCOL))
    def test_loaded_executable_mismatch_rejected(self):
        d=fixture();manifest={'files':[],'executable':{'path':d['loaded_modules_before'][0]['path'],'sha256':'d'*64}}
        self.assertTrue(any('loaded identity mismatch' in x for x in audit_run(d,CFG,'event',PROTOCOL,manifest)))
    def test_grid_and_gates(self):
        self.assertEqual(len(PROTOCOL['configs']),12)
        self.assertEqual({(c['quant'],c['M'],c['K']) for c in PROTOCOL['configs']},{(q,m,k) for q in ('Q5_0','Q8_0') for m in (1,2,4) for k in (896,1024)})
        self.assertEqual(PROTOCOL['correctness']['atol'],.05);self.assertEqual(PROTOCOL['correctness']['rtol'],.03)
        self.assertEqual(PROTOCOL['quality_gates']['dispersion_p90_p10_max'],1.5)
    def test_impossible_event_duration_rejected(self):
        d=fixture();d['runs'][0]['event_envelope_ms']=1e9
        self.assertTrue(any('exceeds enclosing host wall' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_post_audit_raw_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);rows=[]
            for cfg in PROTOCOL['configs']:
                for mode in ('event','control'):
                    path=directory/f"{cfg['id']}.{mode}.json";path.write_text('{}')
                    rows.append({'config':cfg['id'],'mode':mode,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            self.assertEqual(raw_digest_issues(directory,{'rows':rows},PROTOCOL),[])
            (directory/f"{CFG['id']}.event.json").write_text('{"changed":true}')
            self.assertTrue(any('changed after audit' in x for x in raw_digest_issues(directory,{'rows':rows},PROTOCOL)))
    def test_empty_manifest_rejected(self):
        self.assertTrue(verify_manifest({'files':[]}))
    def test_q5_math_failure_preserved_without_path_failure(self):
        d=fixture();v=d['runs'][0]['correctness']['samples'][0];v.update(reference=0.,math_reference=0.,math_absolute_error=1.,math_pass=False)
        self.assertEqual(audit_run(d,CFG,'event',PROTOCOL),[])
    def test_tight_path_failure_rejected(self):
        d=fixture();v=d['runs'][0]['correctness']['samples'][0];v.update(path_reference=1.001,path_absolute_error=.001,path_pass=False)
        self.assertTrue(any('numerical sample rejected' in x for x in audit_run(d,CFG,'event',PROTOCOL)))
    def test_missing_dual_evidence_rejected(self):
        d=fixture();d['runs'][0]['correctness']['samples'][0].pop('math_reference')
        self.assertTrue(audit_run(d,CFG,'event',PROTOCOL))
    def test_source_uniform_wait_and_batch(self):
        source=(HERE/'stream_event_probe.cpp').read_text()
        self.assertNotIn('cudaEventSynchronize(',source)
        self.assertEqual(source.count('ggml_backend_synchronize(r.backend);v.wait_end=qpc();'),2)
        self.assertIn('for(int call=0;call<o.batch;++call)',source)
        self.assertIn('GGML_TYPE_Q5_0',source)

if __name__=='__main__':unittest.main(verbosity=2)
