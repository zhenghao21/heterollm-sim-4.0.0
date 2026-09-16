// Synthetic CUDA graph controller evidence; never opens GGUF or LLM latency data.
#define NOMINMAX
#include <windows.h>
#include <tlhelp32.h>
#include <bcrypt.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include "prepared_identity.h" // generated and then frozen by build_probe.py
#include "frozen_module_guard.h"
#include "math_reference.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <utility>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(sizeof(void*)==8, "frozen Windows x64 ABI only");
static long long qpc() { LARGE_INTEGER v{}; if(!QueryPerformanceCounter(&v)) throw std::runtime_error("QPC unavailable"); return v.QuadPart; }
static long long qpf() { LARGE_INTEGER v{}; if(!QueryPerformanceFrequency(&v)||v.QuadPart<=0) throw std::runtime_error("QPC frequency unavailable"); return v.QuadPart; }
static std::string quote(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) switch(c) {
    case '"':o<<"\\\"";break; case '\\':o<<"\\\\";break; case '\n':o<<"\\n";break;
    case '\r':o<<"\\r";break; case '\t':o<<"\\t";break;
    default: if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;
    }
    o << '"'; return o.str();
}
static std::wstring wide(const std::string &s) {
    int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),int(s.size()),nullptr,0);
    if(n<=0) throw std::runtime_error("invalid UTF8 path");
    std::wstring r(n,0); MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),int(s.size()),r.data(),n); return r;
}
static std::string utf8(const wchar_t *s) {
    int n=WideCharToMultiByte(CP_UTF8,0,s,-1,nullptr,0,nullptr,nullptr);
    if(n<=0)throw std::runtime_error("module path conversion failed");
    std::string r(n,0);WideCharToMultiByte(CP_UTF8,0,s,-1,r.data(),n,nullptr,nullptr);r.pop_back();return r;
}
static std::string utc() {
    SYSTEMTIME s{};GetSystemTime(&s);char b[64];
    sprintf_s(b,"%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",s.wYear,s.wMonth,s.wDay,s.wHour,s.wMinute,s.wSecond,s.wMilliseconds);return b;
}
static std::string file_hash(const std::string &path) {
    std::ifstream f(std::filesystem::path(wide(path)),std::ios::binary); if(!f) throw std::runtime_error("unreadable identity file: "+path);
    BCRYPT_ALG_HANDLE a=nullptr; BCRYPT_HASH_HANDLE h=nullptr;
    if(BCryptOpenAlgorithmProvider(&a,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("SHA256 provider failure");
    ULONG size=0,got=0; if(BCryptGetProperty(a,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&size),sizeof(size),&got,0)<0) {BCryptCloseAlgorithmProvider(a,0);throw std::runtime_error("SHA256 properties failure");}
    std::vector<unsigned char> object(size),buffer(1<<20); unsigned char digest[32];
    if(BCryptCreateHash(a,&h,object.data(),size,nullptr,0,0)<0){BCryptCloseAlgorithmProvider(a,0);throw std::runtime_error("SHA256 create failure");}
    while(f) {f.read(reinterpret_cast<char*>(buffer.data()),buffer.size()); const auto n=f.gcount(); if(n && BCryptHashData(h,buffer.data(),ULONG(n),0)<0){BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);throw std::runtime_error("SHA256 update failure");}}
    const bool okay=f.eof() && BCryptFinishHash(h,digest,sizeof(digest),0)>=0;
    BCryptDestroyHash(h);BCryptCloseAlgorithmProvider(a,0);if(!okay)throw std::runtime_error("SHA256 finish/read failure");
    std::ostringstream o; for(auto b:digest)o<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b); return o.str();
}
static void verify_files() { for(const auto &f:frozen_files) if(file_hash(f.path)!=f.sha256)throw std::runtime_error(std::string("frozen source/runtime changed: ")+f.path); }
struct NativeModuleApi {
    using Handle=HMODULE;
    void require_absolute_path(const char *p) const {std::string s(p);if(s.size()<3||!std::isalpha(static_cast<unsigned char>(s[0]))||s[1]!=':'||(s[2]!='\\'&&s[2]!='/'))throw std::runtime_error("DLL path not absolute");}
    HMODULE load_absolute(const char *p) const {auto h=LoadLibraryExW(wide(p).c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);if(!h)throw std::runtime_error(std::string("absolute DLL preload failed: ")+p);return h;}
    void verify_handle(HMODULE h,const char *path,const char *hash) const {wchar_t b[32768];DWORD n=GetModuleFileNameW(h,b,32768);if(!n||n>=32768||_stricmp(utf8(b).c_str(),path)!=0||file_hash(utf8(b))!=hash)throw std::runtime_error(std::string("loaded DLL identity mismatch: ")+path);}
    void release(HMODULE h) const noexcept {FreeLibrary(h);}
};
static void verify_loaded_modules() {
    NativeModuleApi api;
    for(const auto &f:frozen_modules) {std::string p=f.path;std::string n=p.substr(p.find_last_of("/\\")+1);HMODULE h=GetModuleHandleW(wide(n).c_str());if(!h)throw std::runtime_error("missing frozen module: "+n);api.verify_handle(h,f.path,f.hash);}
}
static std::string modules_json() {
    HANDLE s=CreateToolhelp32Snapshot(TH32CS_SNAPMODULE|TH32CS_SNAPMODULE32,GetCurrentProcessId());if(s==INVALID_HANDLE_VALUE)throw std::runtime_error("module snapshot unavailable");
    MODULEENTRY32W e{};e.dwSize=sizeof(e);std::vector<std::string> entries;
    try {if(!Module32FirstW(s,&e))throw std::runtime_error("empty module snapshot");do {
        std::string name=utf8(e.szModule),path=utf8(e.szExePath),lower=name;
        std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(lower.find("ggml")!=std::string::npos||lower.find("cuda")!=std::string::npos||lower.find("cublas")!=std::string::npos||lower.find("nvtx")!=std::string::npos||lower=="graph-submit-probe.exe")
            entries.push_back("{\"name\":"+quote(name)+",\"path\":"+quote(path)+",\"sha256\":"+quote(file_hash(path))+"}");
    }while(Module32NextW(s,&e));}catch(...){CloseHandle(s);throw;}
    CloseHandle(s);std::sort(entries.begin(),entries.end());std::string out="[";for(size_t i=0;i<entries.size();++i){if(i)out+=',';out+=entries[i];}return out+"]";
}
static std::string environment_json(bool check) {
    std::string o="{";bool first=true;
    for(const auto &e:frozen_environment) {const char *v=std::getenv(e.name);if(check&&((e.value==nullptr)!=(v==nullptr)||(e.value&&v&&std::string(e.value)!=v)))throw std::runtime_error(std::string("runtime environment mismatch: ")+e.name);if(!first)o+=',';first=false;o+=quote(e.name)+":"+(v?quote(v):"null");}return o+"}";
}
class Jsonl {
    HANDLE h=INVALID_HANDLE_VALUE;
public:
    explicit Jsonl(const std::string &path){h=CreateFileW(wide(path).c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("raw output must be a new writable path; refusing overwrite");}
    ~Jsonl(){if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}
    void line(std::string s){s+='\n';size_t p=0;while(p<s.size()){DWORD n=0;DWORD count=DWORD(std::min<size_t>(s.size()-p,1<<20));if(!WriteFile(h,s.data()+p,count,&n,nullptr)||n!=count)throw std::runtime_error("raw output write failed");p+=n;}}
    void durable(){if(!FlushFileBuffers(h))throw std::runtime_error("raw output flush failed");}
};
struct Resources {
    ggml_backend_t backend=nullptr;ggml_context *ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
    ~Resources(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);if(backend)ggml_backend_free(backend);}
};
static void cuda_ok(cudaError_t s,const char *op){if(s!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));}
static std::string hardware_json() {
    cudaDeviceProp p{};int driver=0,runtime=0;cuda_ok(cudaGetDeviceProperties(&p,0),"device properties");cuda_ok(cudaDriverGetVersion(&driver),"driver version");cuda_ok(cudaRuntimeGetVersion(&runtime),"runtime version");
    char pci[64];cuda_ok(cudaDeviceGetPCIBusId(pci,sizeof(pci),0),"PCI id");
    std::ostringstream uuid;uuid<<"GPU-";for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<int(static_cast<unsigned char>(p.uuid.bytes[i]));}
    if(uuid.str()!=expected_gpu_uuid||std::string(p.name)!=expected_gpu_name||p.major!=expected_cc_major||p.minor!=expected_cc_minor)throw std::runtime_error("actual device does not match frozen native device");
    std::ostringstream o;o<<"{\"name\":"<<quote(p.name)<<",\"uuid\":"<<quote(uuid.str())<<",\"cc_major\":"<<p.major<<",\"cc_minor\":"<<p.minor<<",\"SMs\":"<<p.multiProcessorCount<<",\"total_memory_bytes\":"<<p.totalGlobalMem<<",\"L2_bytes\":"<<p.l2CacheSize<<",\"PCI_bus_id\":"<<quote(pci)<<",\"cuda_driver_api_version\":"<<driver<<",\"cuda_runtime_version\":"<<runtime<<"}";return o.str();
}
static std::string scheduling_json(){DWORD_PTR process=0,system=0;if(!GetProcessAffinityMask(GetCurrentProcess(),&process,&system))throw std::runtime_error("affinity unavailable");DWORD priority=GetPriorityClass(GetCurrentProcess());int thread=GetThreadPriority(GetCurrentThread());if(priority!=NORMAL_PRIORITY_CLASS||thread!=THREAD_PRIORITY_NORMAL)throw std::runtime_error("unexpected process/thread priority");std::ostringstream o;o<<"{\"process_affinity\":"<<process<<",\"system_affinity\":"<<system<<",\"process_priority_class\":"<<priority<<",\"caller_thread_priority\":"<<thread<<",\"caller_thread_id\":"<<GetCurrentThreadId()<<",\"CPU_worker_pool_created\":false}";return o.str();}
struct Check {size_t checked=0,bad=0,nonfinite=0;long long first_bad=-1;double max_abs=0;};
static Check validate(ggml_tensor *tensor,int stage,std::vector<float> &host) {
    ggml_backend_tensor_get(tensor,host.data(),0,host.size()*sizeof(float)); // timed and classified outside graph, after final sync
    Check c;c.checked=host.size();for(size_t i=0;i<host.size();++i){float ref=reference_at(i,stage);if(!std::isfinite(host[i]))++c.nonfinite;if(!std::isfinite(host[i])||float_bits(host[i])!=float_bits(ref)){++c.bad;if(c.first_bad<0)c.first_bad=static_cast<long long>(i);}if(std::isfinite(host[i]))c.max_abs=std::max(c.max_abs,std::abs(double(host[i])-ref));}return c;
}
static std::string check_json(const Check &c){std::ostringstream o;o<<"{\"checked\":"<<c.checked<<",\"mismatches\":"<<c.bad<<",\"nonfinite\":"<<c.nonfinite<<",\"first_bad_index\":"<<c.first_bad<<",\"max_abs_error\":"<<std::setprecision(17)<<c.max_abs<<",\"bitwise_pass\":"<<(c.bad==0?"true":"false")<<"}";return o.str();}
struct Range {long long push_start=0,push_end=0,pop_start=0,pop_end=0;bool open=false;explicit Range(const std::string &s){push_start=qpc();nvtxRangePushA(s.c_str());push_end=qpc();open=true;}void close(){if(open){pop_start=qpc();nvtxRangePop();pop_end=qpc();open=false;}}~Range(){if(open)nvtxRangePop();}};
struct Call {long long submit_start=0,submit_end=0,sync_start=0,sync_end=0;int status=0;};
static Call compute_once(Resources &r,ggml_cgraph *graph,const std::string &label,Jsonl &raw) {
    Range outer(label);Call c;
    Range submit(label+"/host_submit");c.submit_start=qpc();
    c.status=int(ggml_backend_graph_compute_async(r.backend,graph));
    c.submit_end=qpc();submit.close();
    Range sync(label+"/final_sync");c.sync_start=qpc();
    ggml_backend_synchronize(r.backend); // Exactly one explicit final synchronization per whole graph.
    c.sync_end=qpc();sync.close();outer.close();
    std::ostringstream o;o<<"{\"record\":\"graph_call\",\"label\":"<<quote(label)<<",\"graph_status\":"<<c.status<<",\"observed_device_kernel_count\":null,\"observed_cuda_graph_launch_count\":null,\"qpc_submit_start\":"<<c.submit_start<<",\"qpc_submit_end\":"<<c.submit_end<<",\"qpc_sync_start\":"<<c.sync_start<<",\"qpc_sync_end\":"<<c.sync_end;
    for(const auto &entry:std::vector<std::pair<std::string,const Range*>>{{"outer",&outer},{"submit",&submit},{"sync",&sync}})o<<",\"qpc_"<<entry.first<<"_push_start\":"<<entry.second->push_start<<",\"qpc_"<<entry.first<<"_push_end\":"<<entry.second->push_end<<",\"qpc_"<<entry.first<<"_pop_start\":"<<entry.second->pop_start<<",\"qpc_"<<entry.first<<"_pop_end\":"<<entry.second->pop_end;
    o<<"}";raw.line(o.str());return c;
}
struct Options {std::string config,pair,output;bool run=false;};
static Options parse(int argc,char **argv){Options o;for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--run"){o.run=true;continue;}if(a=="--check-only")continue;if(i+1==argc)throw std::runtime_error("missing argument");std::string v=argv[++i];if(a=="--config")o.config=v;else if(a=="--pair-id")o.pair=v;else if(a=="--output")o.output=v;else throw std::runtime_error("unknown option "+a);}if(o.run&&(o.output.empty()||o.config.empty()||o.pair.empty()))throw std::runtime_error("run requires config, pair-id, new output");return o;}
static int run(const Options &o,Jsonl &raw) {
    const ProbeConfig *config=nullptr;for(const auto &c:frozen_configs)if(o.config==c.id)config=&c;if(!config)throw std::runtime_error("configuration outside six frozen combinations");
    const std::string env=environment_json(true),scheduling=scheduling_json();
    verify_files();NativeModuleApi api;PinnedFrozenModules<NativeModuleApi> pinned(api);pinned.preload(frozen_modules);verify_loaded_modules();
    Resources r;const auto init_start=qpc();r.backend=ggml_backend_cuda_init(0);if(!r.backend)throw std::runtime_error("CUDA backend initialization failed");const auto init_end=qpc();
    nvtxMarkA("graph_submit/process_setup");
    const std::string hardware=hardware_json(),before=modules_json();
    const auto host_start=qpc();std::vector<float> input(size_t(config->elements)),host(size_t(config->elements));for(size_t i=0;i<input.size();++i)input[i]=reference_at(i,0);const auto host_end=qpc();
    const auto build_start=qpc();const int capacity=128;ggml_init_params params{ggml_tensor_overhead()*size_t(capacity)+ggml_graph_overhead_custom(capacity,false),nullptr,true};r.ctx=ggml_init(params);if(!r.ctx)throw std::runtime_error("GGML context allocation failed");
    auto *input_tensor=ggml_new_tensor_1d(r.ctx,GGML_TYPE_F32,config->elements);ggml_set_name(input_tensor,"graph_input");std::vector<ggml_tensor*> stages;ggml_tensor *previous=input_tensor;
    for(int j=0;j<config->nodes;++j){auto *node=ggml_scale(r.ctx,previous,0.5f);std::string name="scale_"+std::to_string(j+1);ggml_set_name(node,name.c_str());if(node==previous||node->src[0]!=previous||node->op!=GGML_OP_SCALE||node->view_src!=nullptr||!ggml_is_contiguous(node)||!ggml_backend_supports_op(r.backend,node))throw std::runtime_error("graph is not a supported distinct SCALE dependency chain");stages.push_back(node);previous=node;}
    auto *graph=ggml_new_graph_custom(r.ctx,capacity,false);ggml_build_forward_expand(graph,stages.back());
    if(ggml_graph_n_nodes(graph)!=config->nodes)throw std::runtime_error("actual GGML graph node count mismatch");for(int j=0;j<config->nodes;++j)if(ggml_graph_node(graph,j)!=stages[size_t(j)])throw std::runtime_error("actual GGML graph dependency order mismatch");const auto build_end=qpc();
    const auto alloc_start=qpc();r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,r.backend);if(!r.buffer)throw std::runtime_error("device buffer allocation failed");const auto alloc_end=qpc();
    for(size_t i=0;i<stages.size();++i){if(stages[i]->data==input_tensor->data)throw std::runtime_error("unexpected input/output alias");for(size_t j=0;j<i;++j)if(stages[i]->data==stages[j]->data)throw std::runtime_error("unexpected intermediate alias");}
    const auto upload_start=qpc();ggml_backend_tensor_set(input_tensor,input.data(),0,input.size()*sizeof(float));const auto upload_end=qpc();
    std::ostringstream setup;setup<<"{\"record\":\"setup\",\"config\":"<<quote(o.config)<<",\"pair_id\":"<<quote(o.pair)<<",\"elements\":"<<config->elements<<",\"requested_nodes\":"<<config->nodes<<",\"actual_ggml_nodes\":"<<ggml_graph_n_nodes(graph)<<",\"observed_device_kernel_count\":null,\"allocated_buffer_bytes\":"<<ggml_backend_buffer_get_size(r.buffer)<<",\"tensor_payload_bytes\":"<<input.size()*sizeof(float)<<",\"allocated_payload_bytes\":"<<(config->nodes+1)*input.size()*sizeof(float)<<",\"logical_read_bytes\":"<<config->nodes*input.size()*sizeof(float)<<",\"logical_write_bytes\":"<<config->nodes*input.size()*sizeof(float)<<",\"environment\":"<<env<<",\"scheduling\":"<<scheduling<<",\"hardware_actual\":"<<hardware<<",\"loaded_modules_before\":"<<before<<",\"qpc_init_start\":"<<init_start<<",\"qpc_init_end\":"<<init_end<<",\"qpc_host_reference_start\":"<<host_start<<",\"qpc_host_reference_end\":"<<host_end<<",\"qpc_graph_build_start\":"<<build_start<<",\"qpc_graph_build_end\":"<<build_end<<",\"qpc_allocation_start\":"<<alloc_start<<",\"qpc_allocation_end\":"<<alloc_end<<",\"qpc_upload_start\":"<<upload_start<<",\"qpc_upload_end\":"<<upload_end<<",\"graph_nodes\":[";
    for(size_t j=0;j<stages.size();++j){auto *t=stages[j];if(j)setup<<',';setup<<"{\"index\":"<<j<<",\"name\":"<<quote(ggml_get_name(t))<<",\"src0\":"<<quote(ggml_get_name(t->src[0]))<<",\"operator\":\"SCALE\",\"dtype\":\"F32\",\"scale\":0.5,\"bias\":0,\"ne\":["<<t->ne[0]<<','<<t->ne[1]<<','<<t->ne[2]<<','<<t->ne[3]<<"],\"nb\":["<<t->nb[0]<<','<<t->nb[1]<<','<<t->nb[2]<<','<<t->nb[3]<<"]}";}setup<<"]}";raw.line(setup.str());raw.durable();
    bool okay=true;int count=0;
    for(int ordinal=0;ordinal<36;++ordinal){const std::string phase=ordinal==0?"first":ordinal<=5?"warmup":"formal";const int index=ordinal==0?0:ordinal<=5?ordinal-1:ordinal-6;const std::string label="graph_submit/"+o.config+"/"+phase+"/"+std::to_string(index);
        const auto call=compute_once(r,graph,label,raw);++count;
        if(call.status!=GGML_STATUS_SUCCESS)throw std::runtime_error("GGML graph execution returned failure");
        const auto validation_start=qpc();Check final=validate(stages.back(),config->nodes,host);okay=okay&&final.bad==0;const auto validation_end=qpc();
        std::ostringstream check;check<<"{\"record\":\"validation\",\"label\":"<<quote(label)<<",\"phase\":"<<quote(phase)<<",\"index\":"<<index<<",\"stage\":"<<config->nodes<<",\"qpc_validation_start\":"<<validation_start<<",\"qpc_validation_end\":"<<validation_end<<",\"result\":"<<check_json(final)<<"}";raw.line(check.str());
        if(ordinal==0||ordinal==35){for(int j=1;j<config->nodes;++j){const auto start=qpc();Check v=validate(stages[size_t(j-1)],j,host);const auto end=qpc();okay=okay&&v.bad==0;std::ostringstream row;row<<"{\"record\":\"intermediate_validation\",\"label\":"<<quote(label)<<",\"stage\":"<<j<<",\"qpc_start\":"<<start<<",\"qpc_end\":"<<end<<",\"result\":"<<check_json(v)<<"}";raw.line(row.str());}}
    }
    verify_files();verify_loaded_modules();const std::string after=modules_json();const bool stable=before==after;const bool math_pass=okay;okay=okay&&stable;
    raw.line("{\"record\":\"footer\",\"status\":"+quote(okay?"complete":"quality_failed")+",\"math_pass\":"+(math_pass?"true":"false")+",\"graph_calls\":"+std::to_string(count)+",\"loaded_modules_after\":"+after+",\"modules_stable\":"+(stable?"true":"false")+",\"calibration_ready\":false,\"trace_confirmation_required\":true,\"qpc_end\":"+std::to_string(qpc())+",\"utc_end\":"+quote(utc())+"}");raw.durable();return okay?0:4;
}
int main(int argc,char **argv) {
    std::unique_ptr<Jsonl> raw;
    try {const Options o=parse(argc,argv);if(!o.run){verify_files();std::cout<<"{\"status\":\"identity_verified_no_device_access\",\"protocol_sha256\":"<<quote(protocol_sha256)<<"}\n";return 0;}
        raw=std::make_unique<Jsonl>(o.output);std::ostringstream argv_json;argv_json<<'[';for(int i=0;i<argc;++i){if(i)argv_json<<',';argv_json<<quote(argv[i]);}argv_json<<']';
        raw->line("{\"record\":\"header\",\"schema\":\"graph-submit-probe/v1\",\"utc_start\":"+quote(utc())+",\"qpc_start\":"+std::to_string(qpc())+",\"qpc_frequency\":"+std::to_string(qpf())+",\"pid\":"+std::to_string(GetCurrentProcessId())+",\"protocol_sha256\":"+quote(protocol_sha256)+",\"argv\":"+argv_json.str()+",\"probe_cuda_events\":false,\"timing_clock\":\"absolute_QPC\",\"trace_clock_subtraction_allowed\":false}");raw->durable();return run(o,*raw);
    }catch(const std::exception &e){const std::string error="{\"record\":\"error\",\"status\":\"failed\",\"message\":"+quote(e.what())+",\"utc\":"+quote(utc())+",\"calibration_ready\":false}";if(raw)try{raw->line(error);raw->durable();}catch(...){}std::cerr<<error<<'\n';return 2;}
}
