// Standalone GGML GEMM evidence. Never reads GGUF or LLM latency results.
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <bcrypt.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
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
        if(lowered.find("ggml")!=std::string::npos||lowered.find("cuda")!=std::string::npos||lowered.find("cublas")!=std::string::npos||lowered=="generic-gemm-microbench.exe") {
            std::ifstream f(entry.szExePath,std::ios::binary);if(!f){CloseHandle(snapshot);throw std::runtime_error("loaded module unreadable");}
            std::vector<char> bytes((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
            if(!first)o<<',';first=false;o<<"{\"name\":"<<quote(name)<<",\"path\":"<<quote(path)<<",\"sha256\":"<<quote(sha256_bytes(bytes.data(),bytes.size()))<<",\"bytes\":"<<bytes.size()<<'}';
        }
    }while(Module32NextW(snapshot,&entry));CloseHandle(snapshot);o<<']';return o.str();
}
struct Options {
    std::string device="cpu",format="F16",output;
    int64_t m=0,n=0,k=0;int threads=1,cuda_index=0,warmup=3,repeats=20,samples=32;
    uint32_t seed=20260914;double atol=0.05,rtol=0.03;bool run=false,nvtx=false;int evict_mib=0;
};
static int64_t integer(const std::string &s) {size_t used=0;auto n=std::stoll(s,&used);if(used!=s.size())throw std::runtime_error("invalid integer: "+s);return n;}
static Options parse(int argc,char **argv) {
    Options o;for(int i=1;i<argc;++i) {
        std::string a=argv[i];if(a=="--run"){o.run=true;continue;}if(a=="--nvtx"){o.nvtx=true;continue;}if(a=="--check-only"){o.run=false;continue;}
        if(a=="--help"){std::cout<<"generic-gemm-microbench --device cpu|cuda --m M --n N --k K --quant F16|Q4_K|Q6_K|Q8_0|IQ3_S|IQ4_XS --threads N [--cuda-index N] [--check-only|--run] [--nvtx] [--evict-mib 0|128] [--warmup 3 --repeats 20 --samples 32 --seed 20260914 --atol .05 --rtol .03 --output file.json]\nDefault check-only performs backend/API support checks, never graph compute.\n";std::exit(0);}
        if(i+1==argc)throw std::runtime_error("option needs value: "+a);std::string v=argv[++i];
        if(a=="--device")o.device=v;else if(a=="--quant")o.format=v;else if(a=="--output")o.output=v;
        else if(a=="--m")o.m=integer(v);else if(a=="--n")o.n=integer(v);else if(a=="--k")o.k=integer(v);
        else if(a=="--threads")o.threads=int(integer(v));else if(a=="--cuda-index")o.cuda_index=int(integer(v));
        else if(a=="--evict-mib")o.evict_mib=int(integer(v));else if(a=="--warmup")o.warmup=int(integer(v));else if(a=="--repeats")o.repeats=int(integer(v));else if(a=="--samples")o.samples=int(integer(v));
        else if(a=="--seed")o.seed=uint32_t(integer(v));else if(a=="--atol")o.atol=std::stod(v);else if(a=="--rtol")o.rtol=std::stod(v);else throw std::runtime_error("unknown option: "+a);
    }
    if(o.m<=0||o.n<=0||o.k<=0||o.threads<=0||o.cuda_index<0||o.warmup<0||o.repeats<1||o.samples<1)throw std::runtime_error("positive M/N/K/threads/repeats/samples required");
    if(o.evict_mib<0||o.evict_mib>1024)throw std::runtime_error("eviction bound 0..1024 MiB");
    if(o.warmup>10000||o.repeats>100000||o.samples>4096||o.threads>1024)throw std::runtime_error("iteration/thread/sample bound exceeded");
    if(!std::isfinite(o.atol)||!std::isfinite(o.rtol)||o.atol<0||o.rtol<0)throw std::runtime_error("finite nonnegative correctness tolerance required");
    if(o.device!="cpu"&&o.device!="cuda")throw std::runtime_error("device must be cpu or cuda; no fallback");
    return o;
}
static ggml_type quant_type(const std::string &q) {
    static const std::map<std::string,ggml_type> types={{"F16",GGML_TYPE_F16},{"Q4_0",GGML_TYPE_Q4_0},{"Q4_K",GGML_TYPE_Q4_K},{"Q6_K",GGML_TYPE_Q6_K},{"Q8_0",GGML_TYPE_Q8_0},{"IQ3_S",GGML_TYPE_IQ3_S},{"IQ4_XS",GGML_TYPE_IQ4_XS}};
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
struct Run {std::string phase;int index;long long start,end;int status;};
struct Check {bool finite=false,passed=false;double max_abs=0,max_rel=0,rmse=0;int count=0;std::string rows="[]";};
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
        bool okay=std::isfinite(actual[i])&&std::isfinite(expected)&&delta<=tol;c.passed=c.passed&&okay;
        if(std::isfinite(delta)){c.max_abs=std::max(c.max_abs,delta);c.max_rel=std::max(c.max_rel,rel);sq+=delta*delta;}
        if(j)rows<<',';rows<<"{\"m_index\":"<<mi<<",\"n_index\":"<<ni<<",\"reference\":"<<std::setprecision(17)<<expected<<",\"actual\":";
        if(std::isfinite(actual[i]))rows<<actual[i];else rows<<"null";rows<<",\"pass\":"<<(okay?"true":"false")<<'}';
    }
    rows<<']';c.rows=rows.str();c.rmse=std::sqrt(sq/double(c.count));c.passed=c.passed&&c.finite;return c;
}
static std::string check_json(const Check&c) {
    std::ostringstream s;s<<std::setprecision(17)<<"{\"finite_all_outputs\":"<<(c.finite?"true":"false")<<",\"passed\":"<<(c.passed?"true":"false")<<",\"sample_count\":"<<c.count<<",\"max_absolute_error\":"<<c.max_abs<<",\"max_relative_error\":"<<c.max_rel<<",\"rmse\":"<<c.rmse<<",\"samples\":"<<c.rows<<'}';return s.str();
}
static std::string environment_json() {
    const char *names[]={"GGML_OP_OFFLOAD_MIN_BATCH","GGML_CUDA_DISABLE_GRAPHS","GGML_CUDA_FORCE_MMQ","GGML_CUDA_FORCE_CUBLAS","CUDA_VISIBLE_DEVICES","GGML_SCHED_DEBUG","OMP_NUM_THREADS","GGML_NO_IQ_PANEL","LLAMA_TRACE_ANNOTATIONS","GGML_CPU_DISABLE_FUSION","GGML_CUDA_DISABLE_FUSION","GGML_CUDA_CUBLAS_COMPUTE_TYPE"};
    std::ostringstream s;s<<'{';bool first=true;for(auto name:names){if(!first)s<<',';first=false;const char*v=std::getenv(name);s<<quote(name)<<':'<<(v?quote(v):"null");}s<<'}';return s.str();
}
static int run(const Options &o,std::string &json) {
    ggml_time_init();auto type=quant_type(o.format);if(o.k%ggml_blck_size(type))throw std::runtime_error("K must be divisible by quantization block size; no silent padding");
    auto nw=product(o.k,o.n),nb=product(o.k,o.m);product(o.m,o.n);
    Resources r;ggml_init_params ip{ggml_tensor_overhead()*16+ggml_graph_overhead_custom(16,false)*2,nullptr,true};r.ctx=ggml_init(ip);if(!r.ctx)throw std::runtime_error("ggml context allocation failed");
    auto *a=ggml_new_tensor_2d(r.ctx,type,o.k,o.n);auto *b=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,o.k,o.m);
    ggml_set_name(a,"synthetic_weight");ggml_set_name(b,"synthetic_input");auto *out=ggml_mul_mat(r.ctx,a,b);ggml_set_name(out,"generic_gemm_output");
    auto *g=ggml_new_graph_custom(r.ctx,16,false);ggml_build_forward_expand(g,out);
    if(o.device=="cpu"){r.backend=ggml_backend_cpu_init();if(r.backend)ggml_backend_cpu_set_n_threads(r.backend,o.threads);}else r.backend=ggml_backend_cuda_init(o.cuda_index);
    if(!r.backend)throw std::runtime_error("requested backend unavailable; fallback forbidden");
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

    std::ostringstream header;header<<"\"schema\":\"generic-ggml-gemm-microbench/v2\",\"M\":"<<o.m<<",\"N\":"<<o.n<<",\"K\":"<<o.k<<",\"weight_format\":"<<quote(o.format)<<",\"input_dtype\":\"F32\",\"output_dtype\":\"F32\",\"layout\":\"ordinary_contiguous_2d\",\"device\":"<<quote(o.device)<<",\"backend_name\":"<<quote(ggml_backend_name(r.backend))<<",\"device_name\":"<<quote(ggml_backend_dev_name(dev))<<",\"device_description\":"<<quote(ggml_backend_dev_description(dev))<<",\"threads\":"<<o.threads<<",\"cuda_index\":"<<o.cuda_index<<",\"seed\":"<<o.seed<<",\"supported\":"<<(supported?"true":"false")<<",\"environment\":"<<environment_json();
    header<<",\"timing_contract\":{\"id\":\"ggml-single-device-compute-sync/v1\",\"clock\":\"steady_clock_ns_since_process_start\",\"scope\":\"ggml_backend_graph_compute_includes_internal_synchronize\",\"includes\":[\"backend_dispatch\",\"activation_conversion_if_selected\",\"all_GEMM_kernels\",\"internal_synchronization\"],\"excludes\":[\"host_data_generation\",\"weight_quantization\",\"tensor_upload\",\"reference_check\",\"external_wait_before_start\"],\"add_separate_launch_or_sync_cost\":false}";
    header<<",\"cache_policy\":"<<quote(o.evict_mib?(o.device=="cuda"?"untimed_read_write_sweep_at_least_4x_device_L2":"clflush_weight_and_input_before_each_call"):"same_buffers_repeated_hot_cache_no_flush")<<",\"gpu_l2_bytes\":"<<l2_bytes<<",\"cache_eviction_bytes\":"<<evict_bytes<<",\"nvtx_enabled\":"<<(o.nvtx?"true":"false")<<",\"hbm_bandwidth_validation\":false,\"latency_fitting_uses_llm_answers\":false,\"warmup_requested\":"<<o.warmup<<",\"formal_repeats_requested\":"<<o.repeats;
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
    auto modules_before=modules_json();std::vector<Run> runs;
    auto compute=[&](const std::string &phase,int index){
        ggml_backend_synchronize(r.backend);
        if(o.evict_mib){
            if(evict_graph){if(ggml_backend_graph_compute(r.backend,evict_graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("eviction graph failed");}
            else {for(size_t i=0;i<ggml_nbytes(a);i+=64)_mm_clflush(static_cast<char*>(a->data)+i);for(size_t i=0;i<ggml_nbytes(b);i+=64)_mm_clflush(static_cast<char*>(b->data)+i);_mm_mfence();}
            ggml_backend_synchronize(r.backend);
        }
        std::string marker="GENERIC_GEMM|phase="+phase+"|index="+std::to_string(index)+"|format="+o.format+"|M="+std::to_string(o.m)+"|N="+std::to_string(o.n)+"|K="+std::to_string(o.k);
        if(o.nvtx)nvtxRangePushA(marker.c_str());
        long long start=now_ns();auto status=ggml_backend_graph_compute(r.backend,g);long long end=now_ns();
        if(o.nvtx)nvtxRangePop();
        runs.push_back({phase,index,start,end,int(status)});return status==GGML_STATUS_SUCCESS;
    };
    bool okay=compute("first_call",0);Check first_check,final_check;
    if(okay){first_check=validate(out,packed,input,o,type);okay=first_check.passed;}
    for(int i=0;okay&&i<o.warmup;++i)okay=compute("warmup",i);
    for(int i=0;okay&&i<o.repeats;++i)okay=compute("formal",i);
    if(okay){final_check=validate(out,packed,input,o,type);okay=final_check.passed;}
    auto modules_after=modules_json();std::ostringstream all;all<<'{'+header.str()<<",\"status\":"<<quote(okay?"measured":"failed_compute_or_correctness")<<",\"graph_compute_calls\":"<<runs.size()<<",\"quantization\":{\"method\":\"ggml_quantize_chunk\",\"bytes\":"<<quantized<<",\"importance_matrix\":"<<quote(importance?"synthetic_ones_declared_no_LLM_data":"not_required")<<",\"packed_weight_sha256\":"<<quote(sha256_bytes(packed.data(),packed.size()))<<",\"input_sha256\":"<<quote(sha256_bytes(input.data(),input.size()*sizeof(float)))<<"}";
    all<<",\"correctness_contract\":{\"reference\":\"ggml_to_float_quantized_weight_double_dot_F32_input\",\"sample_policy\":\"uniform_output_flat_indices\",\"absolute_tolerance\":"<<o.atol<<",\"relative_tolerance\":"<<o.rtol<<",\"criterion\":\"abs(actual-reference)<=atol+rtol*abs(reference);all_outputs_finite\"},\"first_call_correctness\":"<<check_json(first_check)<<",\"final_correctness\":"<<check_json(final_check)<<",\"runs\":[";
    for(size_t i=0;i<runs.size();++i){auto&v=runs[i];if(i)all<<',';all<<"{\"phase\":"<<quote(v.phase)<<",\"index\":"<<v.index<<",\"start_ns\":"<<v.start<<",\"end_ns\":"<<v.end<<",\"elapsed_ns\":"<<v.end-v.start<<",\"ggml_status\":"<<v.status<<'}';}
    all<<"],\"loaded_modules_before\":"<<modules_before<<",\"loaded_modules_after\":"<<modules_after<<",\"modules_stable\":"<<(modules_before==modules_after?"true":"false")<<'}';json=all.str();return okay?0:4;
}
int main(int argc,char**argv) {
    Options options;std::string result;int status=2;
    try{options=parse(argc,argv);status=run(options,result);}catch(const std::exception&e){result="{\"schema\":\"generic-ggml-gemm-microbench/v2\",\"status\":\"error\",\"error\":"+quote(e.what())+"}";}
    std::cout<<result<<'\n';
    if(!options.output.empty()){std::ofstream f(options.output,std::ios::binary);if(!f){std::cerr<<"cannot open output file\n";return 5;}f<<result<<'\n';if(!f)return 5;}
    return status;
}
