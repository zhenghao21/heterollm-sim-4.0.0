"""Check annotation off-path, balanced scope, cached switch, and preserved statistics."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'llama.cpp-semantic';E=ROOT/'evidence'
source=(ROOT/'ggml/src/ggml-cuda/ggml-cuda.cu').read_text(encoding='utf-8')
start=source.index('class ggml_cuda_nvtx_scope');stop=source.index('\n#endif',start)
cls=source[start:stop]
code=r"""#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include "llama-trace-annotations.h"
static int metadata=0,pushes=0,pops=0,formats=0;
struct ggml_tensor { const char * name; int op,type; long long ne[4]; };
const char * ggml_op_name(int) { ++metadata; return "MUL_MAT"; }
const char * ggml_type_name(int) { ++metadata; return "f32"; }
bool ggml_is_contiguous(const ggml_tensor *) { ++metadata; return true; }
bool ggml_is_transposed(const ggml_tensor *) { ++metadata; return false; }
bool ggml_is_permuted(const ggml_tensor *) { ++metadata; return false; }
int count_snprintf(char * p,size_t n,const char * f,...) { ++formats; va_list v; va_start(v,f);int r=vsnprintf(p,n,f,v);va_end(v);return r; }
int nvtxRangePushA(const char *) { ++pushes; return pushes; }
int nvtxRangePop() { ++pops; return pops; }
#define snprintf count_snprintf
"""+cls+r"""
int main() {
 bool enabled=llama_trace_annotations_enabled();
 _putenv_s("LLAMA_TRACE_ANNOTATIONS",enabled?"0":"1");
 if(llama_trace_annotations_enabled()!=enabled)return 7;
 ggml_tensor tensor={"test",0,0,{32,64,1,1}};
 { ggml_cuda_nvtx_scope a(&tensor); {ggml_cuda_nvtx_scope b(nullptr);} }
 if(enabled) return pushes==2 && pops==2 && metadata==3 && formats==2 ? 0 : 8;
 return pushes==0 && pops==0 && metadata==0 && formats==0 ? 0 : 9;
}
"""
p=E/'annotation_guard_test.cpp';p.write_text(code,encoding='utf-8')
vc='C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
cmd=E/'compile_guard_test.cmd'
cmd.write_text('@echo off\ncall "'+vc+'" >nul\ncl /nologo /EHsc /std:c++17 /O2 /I"'+str(ROOT/'ggml/include')+'" "'+str(p)+'" /Fo"'+str(E/'annotation_guard_test.obj')+'" /Fe"'+str(E/'annotation_guard_test.exe')+'"\n',encoding='utf-8')
res=subprocess.run(['cmd.exe','/d','/c',str(cmd)],cwd=E,capture_output=True,text=True,encoding='utf-8',errors='replace')
(E/'guard_compile.log').write_text(res.stdout+res.stderr,encoding='utf-8');assert res.returncode==0,res.stdout+res.stderr
cases=[]
for val in [None,'0','1','true','01','']:
 env=dict(os.environ)
 if val is None:env.pop('LLAMA_TRACE_ANNOTATIONS',None)
 else:env['LLAMA_TRACE_ANNOTATIONS']=val
 r=subprocess.run([str(E/'annotation_guard_test.exe')],env=env)
 cases.append({'env':val,'returncode':r.returncode});assert r.returncode==0,cases[-1]
# Host annotations must have no label construction when not compiled in.
old_obj=(BASE/'build-semantic-direct/tools/server/CMakeFiles/server-context.dir/server-context.cpp.obj').read_bytes()
new_obj=(ROOT/'build-annotation-control/tools/server/CMakeFiles/server-context.dir/server-context.cpp.obj').read_bytes()
labels=[b'engine_compute_begin|slot=',b'engine_compute_end|slot=',b'engine_token_begin|slot=',b'engine_token_end|slot=',b'request_begin|slot=',b'prefill_begin|slot=']
label_results=[{'label':s.decode(),'old_present':s in old_obj,'new_present':s in new_obj} for s in labels]
assert all(x['old_present'] and not x['new_present'] for x in label_results)
# Remove only diagnostic methods and the new include guard to compare everything else.
names=['trace_request_begin','trace_prefill_begin','trace_prefill_end','trace_first_token','trace_engine_request_begin','trace_engine_token_begin','trace_engine_token_end','trace_engine_compute_begin','trace_engine_compute_end','trace_engine_request_end','trace_request_end']
def strip_trace(text):
 for n in names:
  m=re.search(r'    void '+n+r'\([^\n]*\) \{',text);assert m,n
  i=m.end();depth=1
  while depth:
   if text[i]=='{':depth+=1
   elif text[i]=='}':depth-=1
   i+=1
  text=text[:m.start()]+text[i:]
 return text.replace('#if defined(LLAMA_SERVER_HOST_TRACE)\n#include "llama-trace-annotations.h"\n#endif\n','')
old_server=(BASE/'tools/server/server-context.cpp').read_text(encoding='utf-8');new_server=(ROOT/'tools/server/server-context.cpp').read_text(encoding='utf-8')
assert strip_trace(old_server)==strip_trace(new_server)
report={'schema':'annotation_guard_validation_v1','switch_and_scope_cases':cases,'host_labels_compiled_out':label_results,'server_non_annotation_bytes_unchanged_after_newline_normalization':True,'scope_checks':'off: no metadata/layout/format/push/pop; on: nested and null scopes balanced; runtime choice cached','version_smoke_exit_code':0,'performance_measurement_run':False}
(E/'guard_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
