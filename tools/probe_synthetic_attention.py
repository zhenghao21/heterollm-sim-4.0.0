"""Synthetic GGML FlashAttention probe; no model or LLM timing inputs."""
import argparse,ctypes as C,hashlib,json,os,time
from pathlib import Path
import numpy as np


def loaded_modules(required=('ggml-base.dll', 'ggml-cuda.dll')):
    """Record actual loaded GGML/CUDA modules, rather than requested paths."""
    from ctypes import wintypes as W
    kernel = C.WinDLL('kernel32', use_last_error=True)
    psapi = C.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    process = kernel.GetCurrentProcess()
    enum = psapi.EnumProcessModules
    enum.argtypes = [W.HANDLE, C.POINTER(W.HMODULE), W.DWORD, C.POINTER(W.DWORD)]
    enum.restype = W.BOOL
    filename = psapi.GetModuleFileNameExW
    filename.argtypes = [W.HANDLE, W.HMODULE, W.LPWSTR, W.DWORD]
    filename.restype = W.DWORD
    modules = (W.HMODULE * 4096)(); needed = W.DWORD()
    if not enum(process, modules, C.sizeof(modules), C.byref(needed)):
        raise C.WinError(C.get_last_error())
    if needed.value > C.sizeof(modules):
        raise RuntimeError('module enumeration overflow')
    result = {}
    for module in modules[:needed.value // C.sizeof(W.HMODULE)]:
        path = C.create_unicode_buffer(32768)
        length = filename(process, module, path, len(path))
        if not length or length >= len(path):
            raise RuntimeError('module path unavailable or truncated')
        file = Path(path.value)
        if file.name.lower().startswith(('ggml', 'cublas', 'cudart', 'nvcuda')):
            result[file.name.lower()] = {'path': str(file), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    if not set(required) <= result.keys():
        raise RuntimeError('required GGML modules absent from process')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--queries',type=int,default=1);p.add_argument('--context',type=int,default=1024);p.add_argument('--causal',action='store_true');a=p.parse_args()
    if not 1<=a.queries<=128 or a.context not in (256,1024,1536,2048,4096):raise ValueError('outside frozen probe domain')
    root=Path(__file__).resolve().parents[1];folder=root/'source/llama.cpp-semantic/build-semantic-hosttrace/bin'
    handles=[os.add_dll_directory(str(folder)),os.add_dll_directory('E:/cuda/bin')]
    base=C.CDLL(str(folder/'ggml-base.dll'));cuda=C.CDLL(str(folder/'ggml-cuda.dll'))
    def fn(lib,name,ret,args):
        f=getattr(lib,name);f.restype=ret;f.argtypes=args;return f
    ptr=C.c_void_p;i64=C.c_int64;size=C.c_size_t
    class Init(C.Structure):_fields_=[('mem_size',size),('mem_buffer',ptr),('no_alloc',C.c_bool)]
    init=fn(base,'ggml_init',ptr,[Init]);tensor=fn(base,'ggml_new_tensor_4d',ptr,[ptr,C.c_int,i64,i64,i64,i64])
    backend=fn(cuda,'ggml_backend_cuda_init',ptr,[C.c_int])(0)
    ctx=init(Init(16*1024*1024,None,True))
    if not backend or not ctx:raise RuntimeError('backend/context initialization failed')
    buffer=None
    try:
        D,H,HK=64,4,2;M,L=a.queries,a.context
        q=tensor(ctx,0,D,M,H,1);k=tensor(ctx,1,D,L,HK,1);v=tensor(ctx,1,D,L,HK,1)
        flash=fn(base,'ggml_flash_attn_ext',ptr,[ptr,ptr,ptr,ptr,ptr,C.c_float,C.c_float,C.c_float])
        mask=tensor(ctx,1,L,((M+63)//64)*64,1,1) if a.causal else None
        output=flash(ctx,q,k,v,mask,D**-.5,0.,0.)
        graph=fn(base,'ggml_new_graph_custom',ptr,[ptr,size,C.c_bool])(ctx,128,False)
        fn(base,'ggml_build_forward_expand',None,[ptr,ptr])(graph,output)
        buffer=fn(base,'ggml_backend_alloc_ctx_tensors',ptr,[ptr,ptr])(ctx,backend)
        if not buffer:raise RuntimeError('allocation failed')
        rng=np.random.default_rng(20260929)
        qa=(rng.standard_normal((H,M,D))*.1).astype(np.float32)
        ka=(rng.standard_normal((HK,L,D))*.1).astype(np.float16)
        va=(rng.standard_normal((HK,L,D))*.1).astype(np.float16)
        set_tensor=fn(base,'ggml_backend_tensor_set',None,[ptr,ptr,size,size])
        for t,x in ((q,qa),(k,ka),(v,va)):set_tensor(t,x.ctypes.data,0,x.nbytes)
        if mask:
            mask_data=np.full((((M+63)//64)*64,L),-np.inf,dtype=np.float16)
            for row in range(M):mask_data[row,:L-M+row+1]=0
            set_tensor(mask,mask_data.ctypes.data,0,mask_data.nbytes)
        if not fn(base,'ggml_backend_supports_op',C.c_bool,[ptr,ptr])(backend,output):
            raise RuntimeError('loaded CUDA backend does not support requested FlashAttention')
        compute=fn(base,'ggml_backend_graph_compute',C.c_int,[ptr,ptr]);runs=[]
        modules_before=loaded_modules()
        for index in range(24):
            start=time.perf_counter_ns();status=compute(backend,graph);end=time.perf_counter_ns()
            if status:raise RuntimeError('graph compute failed '+str(status))
            runs.append({'phase':'warmup' if index<4 else 'formal','elapsed_ns':end-start})
        actual=np.empty((M,H,D),np.float32)
        fn(base,'ggml_backend_tensor_get',None,[ptr,ptr,size,size])(output,actual.ctypes.data,0,actual.nbytes)
        expected=np.empty_like(actual,dtype=np.float64)
        for head in range(H):
            kh=head//(H//HK);score=qa[head].astype(np.float64)@ka[kh].astype(np.float64).T/D**.5
            if a.causal:
                for row in range(M):score[row,L-M+row+1:]=-np.inf
            score-=score.max(axis=1,keepdims=True);weights=np.exp(score);weights/=weights.sum(axis=1,keepdims=True)
            expected[:,head,:]=weights@va[kh].astype(np.float64)
        error=np.abs(actual-expected);passed=bool(np.isfinite(actual).all() and np.all(error<=1e-3+1e-2*np.abs(expected)))
        modules_after=loaded_modules()
        if modules_before != modules_after:raise RuntimeError('module identity changed during measurement')
        result={'loaded_modules':modules_after,'modules_stable':True,'schema':'synthetic-flash-attention/v1','shape':{'queries':M,'context':L,'head_dim':D,'query_heads':H,'kv_heads':HK},'mask':'causal' if a.causal else 'none','cache':'hot_same_buffers','correctness':{'passed':passed,'max_abs':float(error.max()),'atol':1e-3,'rtol':1e-2},'runs':runs,'timing_scope':'host_graph_wall_not_kernel','seed':20260929,'module_sha256':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('ggml-base.dll','ggml-cuda.dll')},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['correctness']),flush=True)
        if not passed:raise RuntimeError('attention correctness failed')
    finally:
        if buffer:fn(base,'ggml_backend_buffer_free',None,[ptr])(buffer)
        fn(base,'ggml_free',None,[ptr])(ctx);fn(base,'ggml_backend_free',None,[ptr])(backend)
if __name__=='__main__':main()
