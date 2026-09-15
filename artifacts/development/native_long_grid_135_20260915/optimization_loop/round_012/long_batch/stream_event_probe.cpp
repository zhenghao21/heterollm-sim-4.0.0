// Standalone GGML GEMM evidence. Never reads GGUF or LLM latency results.
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <bcrypt.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "identity_lock.h"
#include "source_reference.h"
#include <cstddef>
#include <type_traits>
#include <intrin.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static const auto process_start = Clock::now();
static long long now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-process_start).count(); }
static std::string quote(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c:s) { switch(c) {
        case '"':o<<"\\\"";break;case '\\':o<<"\\\\";break;case '\n':o<<"\\n";break;
        case '\r':o<<"\\r";break;case '\t':o<<"\\t";break;
        default:if(c<32) o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;
    }} o << '"';return o.str();
}
static std::string utf8(const wchar_t *p) {
    int n=WideCharToMultiByte(CP_UTF8,0,p,-1,nullptr,0,nullptr,nullptr);
    if(n<=0) throw std::runtime_error("UTF8 path conversion failed");
    std::string out(size_t(n),'\0');WideCharToMultiByte(CP_UTF8,0,p,-1,out.data(),n,nullptr,nullptr);out.pop_back();return out;
}
static std::string sha256_bytes(const void *data,size_t bytes) {
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0) throw std::runtime_error("SHA256 algorithm unavailable");
    ULONG object_bytes=0,returned=0;BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_bytes),sizeof(object_bytes),&returned,0);
    std::vector<unsigned char> object(object_bytes);unsigned char digest[32];
    if(BCryptCreateHash(alg,&hash,object.data(),object_bytes,nullptr,0,0)<0) {BCryptCloseAlgorithmProvider(alg,0);throw std::runtime_error("SHA256 create failed");}
    const auto *p=static_cast<const unsigned char*>(data);
    while(bytes) {ULONG chunk=static_cast<ULONG>(std::min<size_t>(bytes,1u<<26));if(BCryptHashData(hash,const_cast<PUCHAR>(p),chunk,0)<0) {BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);throw std::runtime_error("SHA256 update failed");}p+=chunk;bytes-=chunk;}
    if(BCryptFinishHash(hash,digest,32,0)<0) {BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);throw std::runtime_error("SHA256 finish failed");}
    BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
    std::ostringstream out;for(auto b:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);return out.str();
}
// Module paths are captured in-process; hashes can be compared with the build manifest.
static std::string modules_json() {
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE|TH32CS_SNAPMODULE32,GetCurrentProcessId());
    if(snapshot==INVALID_HANDLE_VALUE)throw std::runtime_error("module enumeration failed");
    MODULEENTRY32W entry{};entry.dwSize=sizeof(entry);std::ostringstream o;o<<'[';bool first=true;
    if(Module32FirstW(snapshot,&entry))do {
        std::string name=utf8(entry.szModule),path=utf8(entry.szExePath);
        std::string lowered=name;std::transform(lowered.begin(),lowered.end(),lowered.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(lowered.find("ggml")!=std::string::npos||lowered.find("cuda")!=std::string::npos||lowered.find("cublas")!=std::string::npos||lowered=="stream-event-probe.exe") {
            std::ifstream f(entry.szExePath,std::ios::binary);if(!f){CloseHandle(snapshot);throw std::runtime_error("loaded module unreadable");}
            std::vector<char> bytes((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
            if(!first)o<<',';first=false;o<<"{\"name\":"<<quote(name)<<",\"path\":"<<quote(path)<<",\"sha256\":"<<quote(sha256_bytes(bytes.data(),bytes.size()))<<",\"bytes\":"<<bytes.size()<<'}';
        }
    }while(Module32NextW(snapshot,&entry));CloseHandle(snapshot);o<<']';return o.str();
}
struct Options {
    std::string device="cuda",format="Q4_K",output; bool control=false;
    int64_t m=0,n=0,k=0;int batch=1024,threads=1,cuda_index=0,warmup=5,repeats=30,samples=4096;
    uint32_t seed=20260914;double atol=0.05,rtol=0.03;bool run=false,nvtx=false;int evict_mib=0;
};
static int64_t integer(const std::string &s) {size_t used=0;auto n=std::stoll(s,&used);if(used!=s.size())throw std::runtime_error("invalid integer: "+s);return n;}
static Options parse(int argc,char **argv) {
    Options o;for(int i=1;i<argc;++i) {
        std::string a=argv[i];if(a=="--control"){o.control=true;continue;}if(a=="--run"){o.run=true;continue;}if(a=="--nvtx"){o.nvtx=true;continue;}if(a=="--check-only"){o.run=false;continue;}
        if(a=="--help"){std::cout<<"stream-event-probe --device cuda --m 1|2|4 --n 4096 --k 896|1024 --quant Q5_0|Q8_0 --batch 1024 --threads N [--cuda-index N] [--check-only|--run] [--nvtx] [--evict-mib 0|128] [--warmup 3 --repeats 20 --samples 32 --seed 20260914 --atol .05 --rtol .03 --output file.json]\nUse invoke.ps1 without -Run for identity-only checks without device access. Binary check-only performs GPU backend support checks.\n";std::exit(0);}
        if(i+1==argc)throw std::runtime_error("option needs value: "+a);std::string v=argv[++i];
        if(a=="--device")o.device=v;else if(a=="--quant")o.format=v;else if(a=="--output")o.output=v;
        else if(a=="--m")o.m=integer(v);else if(a=="--n")o.n=integer(v);else if(a=="--k")o.k=integer(v);
        else if(a=="--batch")o.batch=int(integer(v));else if(a=="--threads")o.threads=int(integer(v));else if(a=="--cuda-index")o.cuda_index=int(integer(v));
        else if(a=="--evict-mib")o.evict_mib=int(integer(v));else if(a=="--warmup")o.warmup=int(integer(v));else if(a=="--repeats")o.repeats=int(integer(v));else if(a=="--samples")o.samples=int(integer(v));
        else if(a=="--seed")o.seed=uint32_t(integer(v));else if(a=="--atol")o.atol=std::stod(v);else if(a=="--rtol")o.rtol=std::stod(v);else throw std::runtime_error("unknown option: "+a);
    }
    if(o.m<=0||o.n<=0||o.k<=0||o.threads<=0||o.cuda_index<0||o.warmup<0||o.repeats<1||o.samples<1)throw std::runtime_error("positive M/N/K/threads/repeats/samples required");
    if(o.evict_mib<0||o.evict_mib>1024)throw std::runtime_error("eviction bound 0..1024 MiB");
    if(o.warmup>10000||o.repeats>100000||o.samples>4096||o.threads>1024)throw std::runtime_error("iteration/thread/sample bound exceeded");
    if(!std::isfinite(o.atol)||!std::isfinite(o.rtol)||o.atol<0||o.rtol<0)throw std::runtime_error("finite nonnegative correctness tolerance required");
    if(o.device!="cpu"&&o.device!="cuda")throw std::runtime_error("device must be cpu or cuda; no fallback");
    if(o.device!="cuda" || (o.format!="Q5_0"&&o.format!="Q8_0") || (o.m!=1&&o.m!=2&&o.m!=4) || o.n!=4096 || (o.k!=896&&o.k!=1024)) throw std::runtime_error("configuration outside frozen synthetic protocol");
    if(o.batch!=1024||o.threads!=1||o.warmup!=5||o.repeats!=30||o.samples!=4096||o.seed!=20260914||o.atol!=0.05||o.rtol!=0.03||o.evict_mib||o.nvtx||o.cuda_index!=0) throw std::runtime_error("protocol override forbidden");
    return o;
}
static ggml_type quant_type(const std::string &q) {
    static const std::map<std::string,ggml_type> types={{"F16",GGML_TYPE_F16},{"Q4_0",GGML_TYPE_Q4_0},{"Q5_0",GGML_TYPE_Q5_0},{"Q4_K",GGML_TYPE_Q4_K},{"Q6_K",GGML_TYPE_Q6_K},{"Q8_0",GGML_TYPE_Q8_0},{"IQ3_S",GGML_TYPE_IQ3_S},{"IQ4_XS",GGML_TYPE_IQ4_XS}};
    auto it=types.find(q);if(it==types.end())throw std::runtime_error("unsupported format; no substitution: "+q);return it->second;
}
static size_t product(int64_t a,int64_t b) {
    constexpr size_t maximum_elements=size_t(1)<<29;
    if(a<=0||b<=0||uint64_t(a)>maximum_elements/uint64_t(b))throw std::runtime_error("matrix allocation bound exceeded (2 GiB float tensor maximum)");
    return size_t(a)*size_t(b);
}
static float random_value(uint32_t &s) {s^=s<<13;s^=s>>17;s^=s<<5;return float(int32_t(s&65535)-32768)/65536.0f;}
struct Resources {
    ggml_context *ctx=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t buffer=nullptr;
    ~Resources(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);if(backend)ggml_backend_free(backend);ggml_quantize_free();}
};

static_assert(sizeof(void*)==8,"locked Windows x64 ABI only");
static_assert(std::is_standard_layout<ggml_backend_event>::value,"event must be standard layout");
static_assert(sizeof(ggml_backend_event)==16 && alignof(ggml_backend_event)==8,"event layout changed");
static_assert(offsetof(ggml_backend_event,device)==0 && offsetof(ggml_backend_event,context)==8,"event offsets changed");
static long long qpc() { LARGE_INTEGER v; if(!QueryPerformanceCounter(&v)) throw std::runtime_error("QPC failure");return v.QuadPart; }
static long long qpf() { LARGE_INTEGER v; if(!QueryPerformanceFrequency(&v)||v.QuadPart<=0)throw std::runtime_error("QPC frequency failure");return v.QuadPart; }
static void cuda_ok(cudaError_t s,const char *what) {if(s!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+std::to_string(int(s))+" "+cudaGetErrorString(s));}
static std::string file_hash(const char *path) {std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error(std::string("cannot read frozen file: ")+path);std::vector<char>b((std::istreambuf_iterator<char>(f)),{});return sha256_bytes(b.data(),b.size());}
static void verify_files() {for(auto &x:frozen_files)if(file_hash(x.path)!=x.hash)throw std::runtime_error(std::string("frozen identity mismatch: ")+x.path);}
static void verify_loaded_modules() {
 for(auto &x:frozen_modules){std::string p=x.path;auto name=p.substr(p.find_last_of("/\\")+1);HMODULE m=GetModuleHandleA(name.c_str());if(!m)throw std::runtime_error("expected native DLL not loaded: "+name);char buf[32768];DWORD n=GetModuleFileNameA(m,buf,sizeof(buf));if(!n||n>=sizeof(buf))throw std::runtime_error("module path unavailable");if(_stricmp(buf,x.path)!=0||file_hash(buf)!=x.hash)throw std::runtime_error("loaded DLL identity mismatch: "+name);}
}
struct Events { cudaEvent_t begin=nullptr,end=nullptr;
 void create(){cuda_ok(cudaEventCreateWithFlags(&begin,cudaEventDefault),"create begin");cuda_ok(cudaEventCreateWithFlags(&end,cudaEventDefault),"create end");}
 void close(){if(end){cuda_ok(cudaEventDestroy(end),"destroy end");end=nullptr;}if(begin){cuda_ok(cudaEventDestroy(begin),"destroy begin");begin=nullptr;}}
 ~Events(){if(end)cudaEventDestroy(end);if(begin)cudaEventDestroy(begin);}
};
struct Run {std::string phase;int index;long long start=0,end=0,record_begin_start=0,record_begin_end=0,submit_start=0,submit_end=0,record_end_start=0,record_end_end=0,wait_start=0,wait_end=0;int graph_count=0,status=0,begin_record_status=0,end_record_status=0,submit_cuda_status=0,query_before=-1,wait_status=0,query_after=-1,elapsed_status=-1;float event_ms=0;bool correct=false;std::string correctness="null";};
struct Check {bool finite=false,passed=false;double max_abs=0,max_rel=0,rmse=0,path_max_abs=0,path_sq=0;bool math_passed=true;int count=0;std::string rows="[]";};
static Check validate(ggml_tensor *out,const std::vector<uint8_t>&packed,const std::vector<float>&input,const Options&o,ggml_type type) {
    std::vector<float> actual(product(o.m,o.n));ggml_backend_tensor_get(out,actual.data(),0,actual.size()*sizeof(float));
    Check c;c.finite=true;c.passed=true;for(float value:actual)if(!std::isfinite(value))c.finite=false;
    auto *traits=ggml_get_type_traits(type);if(!traits||!traits->to_float)throw std::runtime_error("quantized to_float reference unavailable");
    c.count=int(std::min<size_t>(size_t(o.samples),actual.size()));std::vector<float> weight(size_t(o.k));std::ostringstream rows;rows<<'[';double sq=0;
    for(int j=0;j<c.count;++j) {
        size_t i=c.count==1?0:size_t(j)*(actual.size()-1)/size_t(c.count-1),ni=i%size_t(o.n),mi=i/size_t(o.n);
        traits->to_float(packed.data()+ni*ggml_row_size(type,o.k),weight.data(),o.k);
        double expected=0;for(int64_t k=0;k<o.k;++k)expected+=double(weight[size_t(k)])*double(input[mi*size_t(o.k)+size_t(k)]);
        double delta=std::abs(double(actual[i])-expected),tol=o.atol+o.rtol*std::abs(expected),rel=delta/std::max(std::abs(expected),1e-12);
        bool math_okay=std::isfinite(actual[i])&&std::isfinite(expected)&&delta<=tol;c.math_passed=c.math_passed&&math_okay;
        double path_ref=type==GGML_TYPE_Q5_0?q5_source_reference(packed.data()+ni*ggml_row_size(type,o.k),input.data()+mi*size_t(o.k),int(o.k)):expected;
        double path_error=std::abs(double(actual[i])-path_ref);
        bool okay=type==GGML_TYPE_Q5_0?(std::isfinite(actual[i])&&std::isfinite(path_ref)&&path_error<=1e-4+1e-5*std::abs(path_ref)):math_okay;
        c.path_max_abs=std::max(c.path_max_abs,path_error);c.path_sq+=path_error*path_error;c.passed=c.passed&&okay;
        if(std::isfinite(delta)){c.max_abs=std::max(c.max_abs,delta);c.max_rel=std::max(c.max_rel,rel);sq+=delta*delta;}
        if(j)rows<<',';rows<<"{\"m_index\":"<<mi<<",\"n_index\":"<<ni<<",\"reference\":"<<std::setprecision(17)<<expected<<",\"actual\":";
        if(std::isfinite(actual[i]))rows<<actual[i];else rows<<"null";rows<<",\"math_reference\":"<<expected<<",\"math_absolute_error\":"<<delta<<",\"math_pass\":"<<(math_okay?"true":"false")<<",\"path_reference\":"<<path_ref<<",\"path_absolute_error\":"<<path_error<<",\"path_pass\":"<<(okay?"true":"false")<<",\"pass\":"<<(okay?"true":"false")<<'}';
    }
    rows<<']';c.rows=rows.str();c.rmse=std::sqrt(sq/double(c.count));c.passed=c.passed&&c.finite;return c;
}
static std::string check_json(const Check&c) {
    std::ostringstream s;s<<std::setprecision(17)<<"{\"finite_all_outputs\":"<<(c.finite?"true":"false")<<",\"passed\":"<<(c.passed?"true":"false")<<",\"sample_count\":"<<c.count<<",\"max_absolute_error\":"<<c.max_abs<<",\"max_relative_error\":"<<c.max_rel<<",\"rmse\":"<<c.rmse<<",\"math_passed\":"<<(c.math_passed?"true":"false")<<",\"path_max_absolute_error\":"<<c.path_max_abs<<",\"path_rmse\":"<<std::sqrt(c.path_sq/std::max(c.count,1))<<",\"samples\":"<<c.rows<<'}';return s.str();
}
static std::string environment_json() {
    const char *names[]={"GGML_OP_OFFLOAD_MIN_BATCH","GGML_CUDA_DISABLE_GRAPHS","GGML_CUDA_FORCE_MMQ","GGML_CUDA_FORCE_CUBLAS","CUDA_VISIBLE_DEVICES","GGML_SCHED_DEBUG","OMP_NUM_THREADS","GGML_NO_IQ_PANEL","LLAMA_TRACE_ANNOTATIONS","GGML_CPU_DISABLE_FUSION","GGML_CUDA_DISABLE_FUSION","GGML_CUDA_CUBLAS_COMPUTE_TYPE"};
    std::ostringstream s;s<<'{';bool first=true;for(auto name:names){if(!first)s<<',';first=false;const char*v=std::getenv(name);s<<quote(name)<<':'<<(v?quote(v):"null");}s<<'}';return s.str();
}
static int run(const Options &o,std::string &json) {
    verify_files();verify_loaded_modules();
    const char *graphs=std::getenv("GGML_CUDA_DISABLE_GRAPHS");if(!graphs||std::string(graphs)!="1")throw std::runtime_error("GGML_CUDA_DISABLE_GRAPHS=1 required");
    cuda_ok(cudaSetDevice(o.cuda_index),"set device");int active_device=-1;cuda_ok(cudaGetDevice(&active_device),"get device");if(active_device!=o.cuda_index)throw std::runtime_error("device mismatch");
    int driver=0,runtime=0;char pci_bus[64]{};cuda_ok(cudaDeviceGetPCIBusId(pci_bus,sizeof(pci_bus),o.cuda_index),"PCI bus identity");cudaDeviceProp prop{};cuda_ok(cudaDriverGetVersion(&driver),"driver version");cuda_ok(cudaRuntimeGetVersion(&runtime),"runtime version");cuda_ok(cudaGetDeviceProperties(&prop,o.cuda_index),"device properties");
    ggml_time_init();auto type=quant_type(o.format);if(o.k%ggml_blck_size(type))throw std::runtime_error("K must be divisible by quantization block size; no silent padding");
    auto nw=product(o.k,o.n),nb=product(o.k,o.m);product(o.m,o.n);
    Resources r;ggml_init_params ip{ggml_tensor_overhead()*16+ggml_graph_overhead_custom(16,false)*2,nullptr,true};r.ctx=ggml_init(ip);if(!r.ctx)throw std::runtime_error("ggml context allocation failed");
    auto *a=ggml_new_tensor_2d(r.ctx,type,o.k,o.n);auto *b=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,o.k,o.m);
    ggml_set_name(a,"synthetic_weight");ggml_set_name(b,"synthetic_input");auto *out=ggml_mul_mat(r.ctx,a,b);ggml_set_name(out,"generic_gemm_output");
    auto *g=ggml_new_graph_custom(r.ctx,16,false);ggml_build_forward_expand(g,out);
    if(o.device=="cpu"){r.backend=ggml_backend_cpu_init();if(r.backend)ggml_backend_cpu_set_n_threads(r.backend,o.threads);}else r.backend=ggml_backend_cuda_init(o.cuda_index);
    if(!r.backend)throw std::runtime_error("requested backend unavailable; fallback forbidden");
    if(!ggml_backend_is_cuda(r.backend))throw std::runtime_error("not a CUDA backend");
    bool supported=ggml_backend_supports_op(r.backend,out);auto dev=ggml_backend_get_device(r.backend);
    int l2_bytes=0;size_t evict_bytes=0;ggml_tensor *evict_input=nullptr;ggml_cgraph *evict_graph=nullptr;
    if(o.device=="cuda") {
        if(cudaDeviceGetAttribute(&l2_bytes,cudaDevAttrL2CacheSize,o.cuda_index)!=cudaSuccess)throw std::runtime_error("cannot read GPU L2 size");
        if(o.evict_mib) {
            evict_bytes=std::max<size_t>(size_t(o.evict_mib)<<20,size_t(l2_bytes)*4);
            if(evict_bytes>(size_t(1)<<30))throw std::runtime_error("GPU eviction buffer exceeds bound");
            evict_input=ggml_new_tensor_1d(r.ctx,GGML_TYPE_F32,int64_t(evict_bytes/sizeof(float)));
            ggml_set_name(evict_input,"untimed_cache_eviction_input");
            auto *evict_out=ggml_scale(r.ctx,evict_input,0.5f);ggml_set_name(evict_out,"untimed_cache_eviction_output");
            evict_graph=ggml_new_graph_custom(r.ctx,16,false);ggml_build_forward_expand(evict_graph,evict_out);
            if(!ggml_backend_supports_op(r.backend,evict_out))throw std::runtime_error("eviction op unsupported");
        }
    }

    std::ostringstream header;header<<"\"schema\":\"backend-stream-event-probe/v3\",\"M\":"<<o.m<<",\"N\":"<<o.n<<",\"K\":"<<o.k<<",\"weight_format\":"<<quote(o.format)<<",\"input_dtype\":\"F32\",\"output_dtype\":\"F32\",\"layout\":\"ordinary_contiguous_2d\",\"device\":"<<quote(o.device)<<",\"backend_name\":"<<quote(ggml_backend_name(r.backend))<<",\"device_name\":"<<quote(ggml_backend_dev_name(dev))<<",\"device_description\":"<<quote(ggml_backend_dev_description(dev))<<",\"threads\":"<<o.threads<<",\"cuda_index\":"<<o.cuda_index<<",\"seed\":"<<o.seed<<",\"supported\":"<<(supported?"true":"false")<<",\"environment\":"<<environment_json();
    header<<",\"timing_contract\":{\"id\":\"actual-backend-stream-batch-envelope/v2\",\"event_scope\":\"1024 repeated graph computations; conversion + GEMM + fixups + internal memset + stream idle gaps; not pure kernel sum\",\"stream_source\":\"ggml_backend_event_record delegates to cuda_ctx->stream()\",\"timing_event_flags\":0,\"host_device_times_additive\":false},\"control_mode\":"<<(o.control?"true":"false")<<",\"qpc_frequency\":"<<qpf()<<",\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<",\"gpu_name\":"<<quote(prop.name)<<",\"pci_bus_id\":"<<quote(pci_bus)<<",\"compute_capability_major\":"<<prop.major<<",\"compute_capability_minor\":"<<prop.minor<<",\"group\":"<<quote(o.k==896?"dev":"validation");
    header<<",\"cache_policy\":"<<quote(o.evict_mib?(o.device=="cuda"?"untimed_read_write_sweep_at_least_4x_device_L2":"clflush_weight_and_input_before_each_call"):"same_buffers_repeated_hot_cache_no_flush")<<",\"gpu_l2_bytes\":"<<l2_bytes<<",\"cache_eviction_bytes\":"<<evict_bytes<<",\"nvtx_enabled\":"<<(o.nvtx?"true":"false")<<",\"hbm_bandwidth_validation\":false,\"latency_fitting_uses_llm_answers\":false,\"warmup_requested\":"<<o.warmup<<",\"formal_repeats_requested\":"<<o.repeats<<",\"graph_computations_per_batch\":"<<o.batch<<",\"wait_api\":\"ggml_backend_synchronize\",\"correctness_scope\":\"final graph output after every batch; intermediate overwritten outputs not individually validated\"";
    if(!supported||!o.run){json="{"+header.str()+",\"status\":"+quote(supported?"supported_only_not_executed":"unsupported")+",\"graph_compute_calls\":0,\"runs\":[],\"loaded_modules\":"+modules_json()+"}";return supported?0:3;}
    // Data generation and true quantization happen outside every measured interval.
    std::vector<float> weight(nw),input(nb),imatrix(size_t(o.k),1.0f);uint32_t state=o.seed?o.seed:1;
    for(auto &x:weight)x=random_value(state);for(auto&x:input)x=random_value(state);
    std::vector<uint8_t> packed(ggml_nbytes(a));bool importance=ggml_quantize_requires_imatrix(type);
    size_t quantized=ggml_quantize_chunk(type,weight.data(),packed.data(),0,o.n,o.k,importance?imatrix.data():nullptr);
    if(quantized!=packed.size())throw std::runtime_error("quantized byte count mismatch");
    r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);if(!r.buffer)throw std::runtime_error("requested device tensor allocation failed");
    ggml_backend_tensor_set(a,packed.data(),0,packed.size());ggml_backend_tensor_set(b,input.data(),0,input.size()*sizeof(float));
    // Snapshot/hash modules before warmup, never between the final warmup and formal measurements.
    if(evict_input){std::vector<float> trash(evict_bytes/sizeof(float),0.5f);ggml_backend_tensor_set(evict_input,trash.data(),0,evict_bytes);}
    verify_loaded_modules();auto modules_before=modules_json();std::vector<Run> runs;Events events;if(!o.control)events.create();
    ggml_backend_event begin_event{dev,reinterpret_cast<void*>(events.begin)},end_event{dev,reinterpret_cast<void*>(events.end)};
    auto compute=[&](const std::string &phase,int index){
        ggml_backend_synchronize(r.backend);cuda_ok(cudaGetLastError(),"before measurement");
        Run v;v.phase=phase;v.index=index;v.start=qpc();
        if(!o.control){v.record_begin_start=qpc();ggml_backend_event_record(&begin_event,r.backend);v.record_begin_end=qpc();v.begin_record_status=int(cudaGetLastError());cuda_ok(cudaError_t(v.begin_record_status),"record begin");}
        v.submit_start=qpc();
        for(int call=0;call<o.batch;++call){
            v.status=int(ggml_backend_graph_compute_async(r.backend,g));
            if(v.status!=GGML_STATUS_SUCCESS)throw std::runtime_error("async graph status failure at batch offset "+std::to_string(call));
            ++v.graph_count;
        }
        v.submit_end=qpc();v.submit_cuda_status=int(cudaGetLastError());cuda_ok(cudaError_t(v.submit_cuda_status),"async submission");
        if(v.status!=GGML_STATUS_SUCCESS)throw std::runtime_error("async graph compute failed: "+std::to_string(v.status));
        if(!o.control){
            v.record_end_start=qpc();ggml_backend_event_record(&end_event,r.backend);v.record_end_end=qpc();v.end_record_status=int(cudaGetLastError());cuda_ok(cudaError_t(v.end_record_status),"record end");
            v.query_before=int(cudaEventQuery(events.end));if(v.query_before!=cudaSuccess&&v.query_before!=cudaErrorNotReady)cuda_ok(cudaError_t(v.query_before),"query before wait");
            v.wait_start=qpc();ggml_backend_synchronize(r.backend);v.wait_end=qpc();v.end=v.wait_end;v.wait_status=int(cudaGetLastError());cuda_ok(cudaError_t(v.wait_status),"backend synchronize");
            v.query_after=int(cudaEventQuery(events.end));cuda_ok(cudaError_t(v.query_after),"query after wait");v.elapsed_status=int(cudaEventElapsedTime(&v.event_ms,events.begin,events.end));cuda_ok(cudaError_t(v.elapsed_status),"event elapsed");
            if(!std::isfinite(v.event_ms)||v.event_ms<=0)throw std::runtime_error("invalid event envelope");
        }else{v.wait_start=qpc();ggml_backend_synchronize(r.backend);v.wait_end=qpc();v.end=v.wait_end;v.wait_status=int(cudaGetLastError());cuda_ok(cudaError_t(v.wait_status),"backend synchronize");}
        if(!(v.start<=v.submit_start&&v.submit_start<=v.submit_end&&v.submit_end<=v.wait_start&&v.wait_start<=v.wait_end))throw std::runtime_error("QPC ordering failed");
        if(!o.control && !(v.start<=v.record_begin_start&&v.record_begin_start<=v.record_begin_end&&v.record_begin_end<=v.submit_start&&v.submit_end<=v.record_end_start&&v.record_end_start<=v.record_end_end&&v.record_end_end<=v.wait_start))throw std::runtime_error("event QPC ordering failed");
        // Same readback/reference policy in both modes; excluded from all timing intervals.
        auto c=validate(out,packed,input,o,type);v.correct=c.passed;v.correctness=check_json(c);runs.push_back(v);return c.passed;
    };
    bool okay=compute("first_call",0);Check first_check,final_check;
    if(okay){first_check=validate(out,packed,input,o,type);okay=first_check.passed;}
    for(int i=0;okay&&i<o.warmup;++i)okay=compute("warmup",i);
    for(int i=0;okay&&i<o.repeats;++i)okay=compute("formal",i);
    if(okay){final_check=validate(out,packed,input,o,type);okay=final_check.passed;}
    ggml_backend_synchronize(r.backend);cuda_ok(cudaGetLastError(),"final synchronize");events.close();verify_loaded_modules();
    auto modules_after=modules_json();okay=okay&&(modules_before==modules_after);std::ostringstream all;all<<'{'+header.str()<<",\"status\":"<<quote(okay?"measured":"failed_compute_or_correctness")<<",\"graph_compute_calls\":"<<(runs.size()*size_t(o.batch))<<",\"quantization\":{\"method\":\"ggml_quantize_chunk\",\"bytes\":"<<quantized<<",\"importance_matrix\":"<<quote(importance?"synthetic_ones_declared_no_LLM_data":"not_required")<<",\"packed_weight_sha256\":"<<quote(sha256_bytes(packed.data(),packed.size()))<<",\"input_sha256\":"<<quote(sha256_bytes(input.data(),input.size()*sizeof(float)))<<"}";
    all<<",\"correctness_contract\":{\"reference\":\"ggml_to_float_quantized_weight_double_dot_F32_input\",\"sample_policy\":\"uniform_output_flat_indices\",\"absolute_tolerance\":"<<o.atol<<",\"relative_tolerance\":"<<o.rtol<<",\"path_absolute_tolerance\":0.0001,\"path_relative_tolerance\":0.00001,\"reference_mode\":\"dual_math_and_source_path\",\"source_runtime_equivalence_proven\":false,\"criterion\":\"Q5_0 source-path <=1e-4+1e-5*abs(path);Q8_0 original math gate; retain math diagnostics\"},\"first_call_correctness\":"<<check_json(first_check)<<",\"final_correctness\":"<<check_json(final_check)<<",\"runs\":[";
    for(size_t i=0;i<runs.size();++i){auto&v=runs[i];if(i)all<<',';
    all<<"{\"phase\":"<<quote(v.phase)<<",\"index\":"<<v.index<<",\"qpc_start\":"<<v.start<<",\"qpc_end\":"<<v.end<<",\"graph_computations\":"<<v.graph_count<<",\"host_wall_ns\":"<<double(v.end-v.start)*1e9/double(qpf())<<",\"host_per_graph_ns\":"<<double(v.end-v.start)*1e9/double(qpf())/o.batch<<",\"qpc_record_begin_start\":"<<v.record_begin_start<<",\"qpc_record_begin_end\":"<<v.record_begin_end<<",\"qpc_submit_start\":"<<v.submit_start<<",\"qpc_submit_end\":"<<v.submit_end<<",\"qpc_record_end_start\":"<<v.record_end_start<<",\"qpc_record_end_end\":"<<v.record_end_end<<",\"qpc_wait_start\":"<<v.wait_start<<",\"qpc_wait_end\":"<<v.wait_end<<",\"ggml_status\":"<<v.status<<",\"cuda_begin_record_status\":"<<v.begin_record_status<<",\"cuda_end_record_status\":"<<v.end_record_status<<",\"cuda_submit_status\":"<<v.submit_cuda_status<<",\"cuda_query_before_wait\":"<<v.query_before<<",\"cuda_wait_status\":"<<v.wait_status<<",\"cuda_query_after_wait\":"<<v.query_after<<",\"cuda_elapsed_status\":"<<v.elapsed_status<<",\"event_envelope_ms\":";if(o.control)all<<"null";else all<<std::setprecision(9)<<v.event_ms;all<<",\"correctness\":"<<v.correctness<<'}';}
    all<<"],\"loaded_modules_before\":"<<modules_before<<",\"loaded_modules_after\":"<<modules_after<<",\"modules_stable\":"<<(modules_before==modules_after?"true":"false")<<'}';json=all.str();return okay?0:4;
}
int main(int argc,char**argv) {
    Options options;std::string result;int status=2;
    try{options=parse(argc,argv);status=run(options,result);}catch(const std::exception&e){result="{\"schema\":\"backend-stream-event-probe/v3\",\"status\":\"error\",\"error\":"+quote(e.what())+"}";}
    std::cout<<result<<'\n';
    if(!options.output.empty()){std::ofstream f(options.output,std::ios::binary);if(!f){std::cerr<<"cannot open output file\n";return 5;}f<<result<<'\n';if(!f)return 5;}
    return status;
}
