"""Static-only SM120a MMVQ device-code comparison.

Reads ELF/cubin and nvdisasm JSON artifacts only. It never loads or executes a CUDA module.
Normalization removes a function's ELF start offset and control-flow target addresses only;
opcodes, predicates, registers, immediate operands, instruction lengths and all non-control-flow
operands remain part of the comparison.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
HERE=Path(__file__).resolve().parent
EX=HERE/'extract_all'
TARGET_DLL=ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/bin/ggml-cuda.dll'
WRAPPER_EXE=ROOT/'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_024/mmvq_device_probe/r4_source_slice/mmvq-source-slice-link-probe.exe'
SOURCE=ROOT/'source/llama.cpp-semantic/ggml/src/ggml-cuda/mmvq.cu'
PAIRS={
  'mmvq_main': ('ggml-cuda.38.sm_120a.cubin','mmvq-source-slice-link-probe.1.sm_120a.cubin'),
  'q8_1_conversion': ('ggml-cuda.48.sm_120a.cubin','mmvq-source-slice-link-probe.2.sm_120a.cubin'),
}
TYPE_NAMES={6:'Q5_0',8:'Q8_0',12:'Q4_K',14:'Q6_K'}

def sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path:Path): return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
def load_functions(cubin: str):
    # nvdisasm JSON is a two-item list: ELF metadata then device function records.
    data=json.loads((EX/(Path(cubin).stem+'.sass.json')).read_text(encoding='utf8'))
    assert data[0]['SM']['version']=={'major':12,'minor':0}
    return {f['function-name']: f for f in data[1]}

def normalize_instruction(ins:dict):
    value={key:ins[key] for key in ('predicate','opcode','operands','other-attributes') if key in ins}
    # Branch target positions can differ under ELF layout changes.  Replace only hexadecimal
    # target tokens in instructions explicitly marked as control-flow, never ordinary immediates.
    attrs=value.get('other-attributes',{})
    if attrs.get('control-flow')=='True' and 'operands' in value and value['opcode'] in {'BRA','BRX','BSSY','BSYNC','JMX','CALL'}:
        value['operands']=re.sub(r'(?:(?<=^)|(?<=,))\s*-?0x[0-9a-fA-F]+', ' <CF_ADDR>', value['operands'])
    return value

def normalize_function(fun:dict):
    return {
      'function-name':fun['function-name'],
      # start intentionally excluded: it is an ELF address. length is retained.
      'length':fun['length'],
      'other-attributes':fun.get('other-attributes',[]),
      'sass-instructions':[normalize_instruction(i) for i in fun['sass-instructions']],
    }

def resource_map(cubin: str):
    text=(EX/(Path(cubin).stem+'.resource_usage.txt')).read_text(encoding='utf8')
    result={}
    # Keep the full resource record: some functions have CONSTANT[2] in addition to CONSTANT[0].
    for match in re.finditer(r'^ Function (.+?):\r?\n\s+(.+)$',text,re.M):
        result[match.group(1)] = match.group(2).strip()
    return result

def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def selected_m1(functions, type_id):
    prefix=f'_Z13mul_mat_vec_qIL9ggml_type{type_id}ELi1'
    return {name:functions[name] for name in sorted(functions) if name.startswith(prefix)}

source_lines=SOURCE.read_text(encoding='utf8').splitlines()
source_launch={
 'source_sha256':sha(SOURCE),
 'template_line':584,
 'template_declaration_line':583,
 'template_declaration':source_lines[582],
 'launch_bounds_expression':source_lines[583],
 'interpretation':'The bound is a source expression over type, ncols_dst, runtime device table and specialization booleans. This static comparison proves its compiled cubin metadata is identical, but does not convert it into a target runtime launch configuration.'
}
result={
 'schema':'heterollm.r4-sm120a-device-code-compare/v1',
 'static_only':True,
 'gpu_execution_performed':False,
 'native_execution_performed':False,
 'target_llm_latency_used':False,
 'toolchain':{'cuobjdump':str(Path(r'E:\cuda\bin\cuobjdump.exe')),'nvdisasm':str(Path(r'E:\cuda\bin\nvdisasm.exe'))},
 'target_dll':ref(TARGET_DLL),'wrapper_executable':ref(WRAPPER_EXE),
 'architecture':'sm_120a','launch_bounds_source':source_launch,
 'pairs':{},
 'limitations':[
  'This is device-code qualification only. It does not prove target host dispatch, runtime arguments, grid/block, stream, cache residency, launch/synchronization timing, or deployment equivalence.',
  'No performance coefficient, calibration value or timing conclusion is authorized by a byte-identical cubin.',
  'A shape is not encoded in these function symbols beyond type and ncols_dst specialization. Target-binary CUPTI remains necessary to connect an observed shape to a selected device function.',
  'The Q4_K K=896 diagnostic remains source-ineligible for source-qualified MMVQ timing; byte equality of a generic Q4_K M=1 template does not override that eligibility gate.'
 ]
}
for label,(target,wrapper) in PAIRS.items():
    target_path, wrapper_path=EX/target, EX/wrapper
    tf,wf=load_functions(target),load_functions(wrapper)
    tr,wr=resource_map(target),resource_map(wrapper)
    common=sorted(set(tf)&set(wf))
    exact_json = (EX/(Path(target).stem+'.sass.json')).read_bytes()==(EX/(Path(wrapper).stem+'.sass.json')).read_bytes()
    item={'target_cubin':ref(target_path),'wrapper_cubin':ref(wrapper_path),'cubin_bytes_equal':target_path.read_bytes()==wrapper_path.read_bytes(),'nvdisasm_json_bytes_equal':exact_json,'target_function_count':len(tf),'wrapper_function_count':len(wf),'common_function_count':len(common),'functions':{}}
    if label=='mmvq_main':
        for type_id,type_name in TYPE_NAMES.items():
            a,b=selected_m1(tf,type_id),selected_m1(wf,type_id)
            rows=[]
            for name in sorted(set(a)|set(b)):
                left,right=a.get(name),b.get(name)
                normal_left=normalize_function(left) if left else None
                normal_right=normalize_function(right) if right else None
                rows.append({'symbol':name,'target_present':left is not None,'wrapper_present':right is not None,'target_length':left.get('length') if left else None,'wrapper_length':right.get('length') if right else None,'instruction_count':len(left['sass-instructions']) if left else None,'normalized_sass_sha256_target':json_hash(normal_left) if normal_left else None,'normalized_sass_sha256_wrapper':json_hash(normal_right) if normal_right else None,'normalized_sass_equal':normal_left==normal_right,'resource_target':tr.get(name),'resource_wrapper':wr.get(name),'resource_equal':tr.get(name)==wr.get(name)})
            item['functions'][type_name]={'type_id':type_id,'ncols_dst':1,'template_count_target':len(a),'template_count_wrapper':len(b),'all_present_and_normalized_equal':bool(rows) and all(r['target_present'] and r['wrapper_present'] and r['normalized_sass_equal'] and r['resource_equal'] for r in rows),'variants':rows}
    else:
        names=[n for n in common if n.startswith('_Z13quantize_q8_1')]
        rows=[]
        for name in names:
            left,right=tf[name],wf[name]; nl,nr=normalize_function(left),normalize_function(right)
            rows.append({'symbol':name,'target_length':left['length'],'wrapper_length':right['length'],'instruction_count':len(left['sass-instructions']),'normalized_sass_sha256_target':json_hash(nl),'normalized_sass_sha256_wrapper':json_hash(nr),'normalized_sass_equal':nl==nr,'resource_target':tr.get(name),'resource_wrapper':wr.get(name),'resource_equal':tr.get(name)==wr.get(name)})
        item['functions']['Q8_1_conversion']={'symbols':rows,'all_present_and_normalized_equal':bool(rows) and all(r['normalized_sass_equal'] and r['resource_equal'] for r in rows)}
    result['pairs'][label]=item

(HERE/'device_code_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
summary=[]
for label,item in result['pairs'].items():
    summary.append(f"{label}: cubin_equal={item['cubin_bytes_equal']} functions={item['target_function_count']}/{item['wrapper_function_count']} common={item['common_function_count']}")
print('\n'.join(summary))
