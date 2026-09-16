import json, re, subprocess, hashlib
from pathlib import Path
ROOT=Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
HERE=Path(__file__).resolve().parent
CC=ROOT/'source/llama.cpp-semantic/build-semantic-direct/compile_commands.json'
SRC=ROOT/'source/llama.cpp-semantic/ggml/src/ggml-cuda'
MMVQ=SRC/'mmvq.cu'; QUANT=SRC/'quantize.cu'
SLICE_END_LINE=1404
COMP=json.loads(CC.read_text(encoding='utf8'))

def command_for(source):
    for e in COMP:
        if Path(e['file']).resolve()==Path(source).resolve(): return e['command']
    raise RuntimeError('compile command missing '+str(source))

def rewrite(command, old, new, obj):
    command=command.replace(str(old).replace('\\','/'),str(new).replace('\\','/'))
    command=command.replace(str(old),str(new))
    command=re.sub(r' -o\s+\S+', lambda _: ' -o '+str(obj), command)
    command=re.sub(r' -Xcompiler=-Fd[^\s,]+(?:,[^\s]+)?', '', command)
    command=re.sub(r' -DGGML_BACKEND_SHARED\b', '', command)
    command=re.sub(r' -Dggml_cuda_EXPORTS\b', '', command)
    return command

def run_logged(command, log):
    vcvars=r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    proc=subprocess.run('call "{}" >nul && '.format(vcvars)+command,cwd=ROOT,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    Path(log).write_bytes((proc.stdout or b'')+b'\n'+(proc.stderr or b''))
    return proc

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Mechanically derive source prefix.  No MMVQ helper or kernel body is edited;
# the excluded suffix begins at the complete-backend entry ggml_cuda_mul_mat_vec_q.
source_raw=MMVQ.read_bytes()
source_lines=source_raw.splitlines(keepends=True)
assert source_lines[SLICE_END_LINE].decode('utf8').startswith('void ggml_cuda_mul_mat_vec_q(')
prefix=b''.join(source_lines[:SLICE_END_LINE])
shims=r'''
#include "mmvq_probe_abi.h"
// Generated C ABI shims. The preceding 1..1403 lines are byte-derived from the locked mmvq.cu source.
extern "C" void heterollm_mmvq_probe_convert_q8_1(
    const float * src, void * q8_1, int ggml_type_id, int k, int m, int padded_k, cudaStream_t stream) {
    quantize_row_q8_1_cuda(src, nullptr, q8_1, static_cast<ggml_type>(ggml_type_id),
        k, k, k*m, k*m, padded_k, m, 1, 1, stream);
}
extern "C" void heterollm_mmvq_probe_main(
    const void * packed_weight, int ggml_type_id, const void * q8_1, float * output,
    int k, int n, int m, int q8_1_stride_blocks, cudaStream_t stream) {
    const ggml_type type = static_cast<ggml_type>(ggml_type_id);
    ggml_cuda_mm_fusion_args_device fusion{};
    const int stride_row_x = k / ggml_blck_size(type);
    mul_mat_vec_q_switch_type(packed_weight, type, q8_1, nullptr, fusion, output,
        k, n, m, stride_row_x, q8_1_stride_blocks, n,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);
}
'''
SLICE=HERE/'mmvq_source_slice.cu'
SLICE.write_bytes(prefix + shims.encode('utf8'))
source_proof={'path':str(MMVQ),'sha256':sha(MMVQ),'included_line_start':1,'included_line_end':SLICE_END_LINE,'first_excluded_line':SLICE_END_LINE+1,'first_excluded_text':source_lines[SLICE_END_LINE].decode('utf8').strip(),'included_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'derived_slice_sha256':sha(SLICE)}

specs=[
    (MMVQ, SLICE, 'mmvq_source_slice.obj'),
    (QUANT, QUANT, 'quantize_same_source.obj'),
    (QUANT, HERE/'mmvq_runtime_support.cu', 'mmvq_runtime_support.obj'),
]
records=[]
for original, replacement, objname in specs:
    cmd=rewrite(command_for(original),original,replacement,HERE/objname)
    # Generated slice resides outside ggml-cuda; add only the original source directory
    # so its unchanged local #include directives resolve to their locked source files.
    if replacement == SLICE:
        cmd += ' -I'+str(SRC)
    proc=run_logged(cmd,HERE/(objname+'.compile.log'))
    records.append({'compile_command_provenance':str(original),'source':str(replacement),'object':str(HERE/objname),'command':cmd,'returncode':proc.returncode,'log':str(HERE/(objname+'.compile.log'))})
    if proc.returncode: break

link_record={'attempted':False}
if len(records)==len(specs) and all(x['returncode']==0 for x in records):
    cl=r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\link.exe'
    compiler=r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe'
    vcvars=r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    exe=HERE/'mmvq-source-slice-link-probe.exe'
    objects=[HERE/'link_driver.obj',HERE/'mmvq_source_slice.obj',HERE/'quantize_same_source.obj',HERE/'mmvq_runtime_support.obj']
    link_cmd='"{}" /nologo /OPT:REF /OUT:"{}" {} /LIBPATH:"E:\\cuda\\lib\\x64" /LIBPATH:"{}" cudart.lib cuda.lib ggml-base.lib ggml-cpu.lib ggml.lib'.format(cl,exe,' '.join('"{}"'.format(p) for p in objects),ROOT/'source/llama.cpp-semantic/build-semantic-direct/ggml/src')
    cmd='call "{}" >nul && "{}" /nologo /std:c++17 /EHsc /MD /I"E:\\cuda\\include" /c "{}" /Fo"{}" && {}'.format(vcvars,compiler,HERE/'link_driver.cpp',HERE/'link_driver.obj',link_cmd)
    proc=run_logged(cmd,HERE/'link_attempt.log')
    link_record={'attempted':True,'returncode':proc.returncode,'command':cmd,'log':str(HERE/'link_attempt.log'),'executable':str(exe),'executed':False,'opt_ref_enabled':True}

manifest={
 'schema':'heterollm.mmvq-shared-abi-link/v1',
 'status':'compiled_and_linked_not_executed' if link_record.get('returncode')==0 else 'link_gap_or_compile_gap',
 'gpu_execution_performed':False,'timed_runs':0,'target_llm_latency_used':False,
 'source_slice_proof':source_proof,
 'shared_abi': {'path': str(HERE/'mmvq_probe_abi.h'), 'sha256':sha(HERE/'mmvq_probe_abi.h')},
 'driver': {'path':str(HERE/'link_driver.cpp'), 'sha256':sha(HERE/'link_driver.cpp')},
 'source_files':{'quantize.cu':{'path':str(QUANT),'sha256':sha(QUANT)}},
 'target_dll':{'path':str(ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/bin/ggml-cuda.dll'),'sha256':sha(ROOT/'source/llama.cpp-native-thread-control/build-native-thread-control/bin/ggml-cuda.dll')},
 'compile_records':records,'link_record':link_record,
 'semantic_boundary':'The generated slice contains every mmvq.cu line through the closing brace of private mul_mat_vec_q_switch_type (line 1403). The excluded suffix begins with complete-backend host entry ggml_cuda_mul_mat_vec_q at line 1405 and is unrelated to the direct main-shim call.',
 'compatibility':'The kernel/helper source in the slice is exact source text, but this remains a standalone construction with removed complete-backend export flags and a probe-only device-state adapter. It cannot claim binary-level identity with the locked DLL. GPU execution remains prohibited pending target-path identity/CUPTI qualification.'
}
(HERE/'source_slice_receipt.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':manifest['status'],'compile_returncodes':[x['returncode'] for x in records],'link_returncode':link_record.get('returncode'),'gpu_execution_performed':False},ensure_ascii=False))
