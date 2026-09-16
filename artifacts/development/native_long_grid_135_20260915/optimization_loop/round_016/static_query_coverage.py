"""Read only frozen inputs, retained batch geometry and bounded GGUF directory prefixes."""
from __future__ import annotations
import ast
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,struct
from pathlib import Path

P=Path(__file__).resolve().parent
CANDIDATE=P.parent/'round_014/tail_candidate_r2'
SOURCE=CANDIDATE/'source/src/heterollm_sim'


def file_ref(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read_directory(model_ref):
    # Reuse only the frozen metadata scalar readers, never read_gguf_metadata's full-file hash.
    source=(SOURCE/'gguf_parity.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    wanted={'GGUFError','_read_string','_read_value'}
    selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in wanted]
    namespace={'struct':struct}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])),str(SOURCE/'gguf_parity.py'),'exec'),namespace)
    class BoundedReader:
        def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.bytes=0
        def read(self,count):
            if count<0 or self.bytes+count>32*1024*1024:raise ValueError('GGUF metadata prefix bound exceeded')
            raw=self.stream.read(count);self.digest.update(raw);self.bytes+=len(raw);return raw
    path=Path(model_ref['path']);before=path.stat()
    if before.st_size!=model_ref['bytes']:raise ValueError('model size differs from frozen reference')
    with path.open('rb') as stream:
        reader=BoundedReader(stream);head=reader.read(24)
        if len(head)!=24 or head[:4]!=b'GGUF':raise ValueError('GGUF header')
        version,tensors,entries=struct.unpack('<IQQ',head[4:]);metadata={}
        for _ in range(entries):
            name=namespace['_read_string'](reader);typ=struct.unpack('<I',reader.read(4))[0]
            metadata[name]=namespace['_read_value'](reader,typ)
        directory=[]
        for _ in range(tensors):
            name=namespace['_read_string'](reader);rank=struct.unpack('<I',reader.read(4))[0]
            if rank>8:raise ValueError('tensor rank bound')
            shape=struct.unpack('<'+'Q'*rank,reader.read(8*rank));typ=struct.unpack('<I',reader.read(4))[0];offset=struct.unpack('<Q',reader.read(8))[0]
            directory.append({'name':name,'shape':list(shape),'type_id':typ,'offset':offset})
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('model changed during prefix read')
    return {'directory':directory,'metadata':{k:v for k,v in metadata.items() if k=='general.architecture' or k.endswith(('.block_count','.nextn_predict_layers','.embedding_length'))},
        'input_identity':{'frozen_model_ref':model_ref,'full_file_sha256_reverified':False,'tensor_payload_read':False,
            'prefix_bytes_read':reader.bytes,'prefix_sha256':reader.digest.hexdigest(),'file_size_matches_frozen':True,'file_stat_stable_during_read':True}}


def classify_projection(name):
    parts=('attn_q','attn_k','attn_v','attn_qkv','attn_output','ssm_out','attn_gate','ssm_alpha','ssm_beta','ffn_gate','ffn_up','ffn_down')
    for part in parts:
        if name.endswith('.'+part+'.weight'):return part
    return None


def main():
    output=P/'static_query_coverage.json'
    if output.exists():raise ValueError('refusing overwrite')
    freeze=json.loads((CANDIDATE/'freeze.json').read_text());entries={e['cell_id']:e for e in freeze['cells']}
    protocol_path=P/'operator_probe/r3/protocol.json';protocol=json.loads(protocol_path.read_text())
    targets={(c['M'],c['N'],c['K'],c['quant']):c for c in protocol['configs']}
    type_names={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',29:'IQ1_M',30:'BF16'}
    models={};queries=Counter();row_by_query=defaultdict(Counter);source_refs=[];cell_rows=[]
    for pred_path in sorted((CANDIDATE/'predictions').glob('*.prediction.json')):
        source_refs.append(file_ref(pred_path));pred=json.loads(pred_path.read_text());entry=entries[pred['cell_id']];static=entry['static_inputs'];model_key=entry['model_key']
        if model_key not in models:models[model_key]=read_directory(static['prediction_model_ref'])
        model=models[model_key];md=model['metadata'];arch=md['general.architecture'];layers=md[arch+'.block_count']-md.get(arch+'.nextn_predict_layers',0)
        tensors=[]
        for t in model['directory']:
            match=re.match(r'blk\.(\d+)\.',t['name']);projection=classify_projection(t['name'])
            if match and int(match[1])<layers and projection and len(t['shape'])==2:
                tensors.append({**t,'projection':projection,'format':type_names.get(t['type_id'],'UNKNOWN_'+str(t['type_id']))})
        model['selected_layer_matrix_count']=len(tensors)
        batches=pred['batch_schedule']['batches'];counts=Counter()
        for batch in batches:
            m=batch['cost_metadata'].get('physical_batch_rows')
            if type(m) is not int or m<1:raise ValueError('physical batch rows unavailable')
            for t in tensors:
                k,n=t['shape'];key=(model_key,m,n,k,t['format'],t['projection'])
                queries[key]+=1;row_by_query[key][pred['cell_id']]+=1;counts['static_tensor_batch_pairs']+=1
                if (m,n,k,t['format']) in targets:counts['exact_MNK_format_target_pairs']+=1
        cell_rows.append({'cell_id':pred['cell_id'],'model_key':model_key,'retained_batches':len(batches),'declared_layer_matrices':len(tensors),**counts,
            'cached_hardware_sm_clock_samples_mhz':static['config'].get('gpu_sm_clock_samples_mhz'),
            'gpu_layers':static['config'].get('gpu_layers'),'basis':'Static physical layer matrix x retained physical batch M; not an exact invocation-count replay.'})
    rows=[];totals=Counter();per_model=defaultdict(Counter)
    for key,count in sorted(queries.items()):
        model,m,n,k,fmt,projection=key;target=targets.get((m,n,k,fmt));reasons=[]
        if fmt not in {'Q5_0','Q8_0'}:reasons.append('format_outside_frozen26')
        if m not in {c['M'] for c in protocol['configs']}:reasons.append('M_outside_frozen26')
        if (n,k) not in {(c['N'],c['K']) for c in protocol['configs']}:reasons.append('NK_outside_frozen26')
        if target is None:reasons.append('no_exact_MNK_format_key')
        # Do not infer actual dispatch, cache or frequency match from expected shapes.
        reasons+=['cache_domain_not_proven','native_dispatch_not_observed','frequency_domain_not_proven']
        if m==1 and projection in ('ffn_gate','ffn_up'):reasons.append('possible_fused_gate_up_requires_invocation_ledger')
        row={'model_key':model,'M':m,'N':n,'K':k,'weight_format':fmt,'physical_projection':projection,'tensor_batch_pair_count':count,
            'distinct_retained_cells':len(row_by_query[key]),'shape_matches_config_id':target['id'] if target else None,'matching_group':target['group'] if target else None,
            'exact_shape_format_match':bool(target),'shape_only_expected_family':('MMVQ' if m<=8 else 'MMQ') if fmt in ('Q5_0','Q8_0') else None,
            'runtime_observed_family':None,'cache_state':'unknown','full_domain_match':False,'fallback_reasons':reasons}
        rows.append(row);totals['tensor_batch_pairs']+=count;per_model[model]['tensor_batch_pairs']+=count
        totals['unique_static_keys']+=1;per_model[model]['unique_static_keys']+=1
        if target:
            totals['shape_format_matching_pairs']+=count;totals['matching_unique_static_keys']+=1;per_model[model]['shape_format_matching_pairs']+=count
        for reason in reasons:totals['reason:'+reason]+=count
    result={'schema':'r14-static-query-coverage/v1','utc':datetime.now(timezone.utc).isoformat(),'scope':'20 existing predictions; metadata-only model-directory export, no run_scenario/planner/estimator/native invoked',
        'query_basis':'Cartesian expansion of recognized unique physical layer matrices and retained batch physical M. This is an auditable static potential-query distribution, not measured or reconstructed exact invocation counts.',
        'sources':{'freeze':file_ref(CANDIDATE/'freeze.json'),'protocol':file_ref(protocol_path),'frozen_directory_reader':file_ref(SOURCE/'gguf_parity.py'),'analytical_script_reviewed':file_ref(P/'analytical_predictions.py'),'predictions':source_refs,'exporter':file_ref(Path(__file__))},
        'totals':dict(totals),'per_model':dict(per_model),'models':models,'cells':cell_rows,'query_rows':rows,
        'limits':['No tensor payload read or full-file model SHA256 replay; previous full digest is inherited, current metadata prefix independently hashed.',
                  'Not a task replay: aliases deduplicated by physical tensor name; M=1 gate/up fusion and output-row selection can change actual call multiplicity.',
                  'Output head, activation RHS attention products, SSM non-matrix operations and MTP are outside this static layer-matrix expansion.',
                  'GPU placement/fusion/role/cache/current clock/runtime dispatch need independent qualification before cost lookup.',
                  'All full-domain matches remain false even where exact shape+format matches. No fitting or LLM actual read.'],
        'native_actuals_read':False,'full_GGUF_payload_hash_performed':False,'simulation_executed':False,'fit_performed':False}
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'totals':dict(totals),'per_model':dict(per_model),'model_prefix_bytes':{k:v['input_identity']['prefix_bytes_read'] for k,v in models.items()},'output':str(output)},ensure_ascii=False))

if __name__=='__main__':main()
