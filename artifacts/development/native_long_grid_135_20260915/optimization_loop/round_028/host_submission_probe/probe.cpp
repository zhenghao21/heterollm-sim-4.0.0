// External GGML target-DLL probe. This program requires explicit future GPU authorization.
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "launch_decode.h"
#include "identity_support.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <vector>
#include <iostream>

static void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void cuda_ok(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
struct Stamp{int64_t qpc=0;uint64_t user=0,kernel=0,cycles=0;};
static uint64_t ticks(FILETIME x){return (uint64_t(x.dwHighDateTime)<<32)|x.dwLowDateTime;}
static Stamp stamp(){
    FILETIME created{},ended{},kernel{},user{};LARGE_INTEGER q{};ULONG64 cycles{};
    require(QueryPerformanceCounter(&q)!=0,"QPC unavailable");
    require(GetThreadTimes(GetCurrentThread(),&created,&ended,&kernel,&user)!=0,"thread CPU clock unavailable");
    require(QueryThreadCycleTime(GetCurrentThread(),&cycles)!=0,"thread cycle counter unavailable");
    return {q.QuadPart,ticks(user),ticks(kernel),cycles};
}
struct Delta{double wall_ns=0;uint64_t user_ticks=0,kernel_ticks=0,cycles=0;};
static Delta delta(Stamp a,Stamp b){
    LARGE_INTEGER frequency{};require(QueryPerformanceFrequency(&frequency)!=0,"QPC frequency unavailable");
    require(b.qpc>=a.qpc&&b.user>=a.user&&b.kernel>=a.kernel&&b.cycles>=a.cycles,"counter reversed");
    return {double(b.qpc-a.qpc)*1e9/double(frequency.QuadPart),b.user-a.user,b.kernel-a.kernel,b.cycles-a.cycles};
}
static std::string timing_json(const Delta& d){
    std::ostringstream out;out<<std::setprecision(17)<<"{\"wall_ns\":"<<d.wall_ns<<",\"thread_user_100ns_ticks\":"<<d.user_ticks
      <<",\"thread_kernel_100ns_ticks\":"<<d.kernel_ticks<<",\"thread_cycles\":"<<d.cycles
      <<",\"thread_CPU_service_inferred\":false,\"thread_CPU_counter_threshold_met\":"<<((d.user_ticks+d.kernel_ticks)>=100?"true":"false")<<"}";return out.str();
}
struct Graph{
    ggml_context* ctx=nullptr;ggml_cgraph* graph=nullptr;ggml_tensor* input=nullptr;
    ggml_backend_buffer_t buffer=nullptr;std::vector<ggml_tensor*> nodes;
    Graph(int count,bool chain){
        ggml_init_params params{ggml_tensor_overhead()*64+ggml_graph_overhead_custom(64,false),nullptr,true};
        ctx=ggml_init(params);require(ctx!=nullptr,"GGML metadata allocation failed");
        input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,probe::cols,1);ggml_set_name(input,"probe.input");
        graph=ggml_new_graph_custom(ctx,64,false);
        for(int i=0;i<count;++i){auto* tensor=ggml_rms_norm(ctx,(chain&&i)?nodes.back():input,probe::epsilon);
            std::string name="probe.norm."+std::to_string(i);ggml_set_name(tensor,name.c_str());ggml_set_output(tensor);nodes.push_back(tensor);
            if(!chain)ggml_build_forward_expand(graph,tensor);}
        if(chain)ggml_build_forward_expand(graph,nodes.back());
        require(ggml_graph_n_nodes(graph)==count,"unexpected executable node count");
        for(int i=0;i<count;++i)require(ggml_graph_node(graph,i)==nodes[i]&&nodes[i]->op==GGML_OP_RMS_NORM,"unexpected graph order/operator");
    }
    void allocate(ggml_backend_t backend){buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer!=nullptr,"GGML target allocation failed");}
    void initialize(){std::vector<float> x(probe::cols);for(int i=0;i<probe::cols;++i)x[i]=float((i%97)-48)/64.0f;
        ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));}
    void submit(ggml_backend_t backend){require(ggml_backend_graph_compute_async(backend,graph)==GGML_STATUS_SUCCESS,"GGML async graph failed");}
    double verify(bool chain){
        std::vector<double> ref(probe::cols);for(int i=0;i<probe::cols;++i)ref[i]=double(float((i%97)-48)/64.0f);
        const auto origin=ref;double maximum=0;
        for(auto* node:nodes){if(!chain)ref=origin;double sum=0;for(double x:ref)sum+=x*x;
            const double scale=1.0/std::sqrt(sum/probe::cols+double(probe::epsilon));for(double& x:ref)x*=scale;
            std::vector<float> observed(probe::cols);ggml_backend_tensor_get(node,observed.data(),0,observed.size()*sizeof(float));
            for(int i=0;i<probe::cols;++i){double err=std::abs(double(observed[i])-ref[i]);maximum=std::max(maximum,err);
                require(std::isfinite(observed[i])&&err<=2e-5+2e-5*std::abs(ref[i]),"target numerical mismatch");}}
        return maximum;
    }
    ~Graph(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
struct Cupti{
    HMODULE lib=nullptr;CUpti_SubscriberHandle subscriber{};bool active=false;
    decltype(&cuptiSubscribe) subscribe=nullptr;decltype(&cuptiEnableDomain) domain=nullptr;decltype(&cuptiUnsubscribe) unsubscribe=nullptr;
    template<class T>T symbol(const char* name){auto x=GetProcAddress(lib,name);require(x!=nullptr,"CUPTI export absent");return reinterpret_cast<T>(x);}
    explicit Cupti(probe::Recorder& recorder){
        require(hash_file(locked::cupti_path)==locked::cupti_sha,"CUPTI SHA differs");
        lib=LoadLibraryExA(locked::cupti_path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);require(lib!=nullptr,"CUPTI load failed");
        char actual[32768]{};require(GetModuleFileNameA(lib,actual,sizeof(actual))!=0&&!_stricmp(actual,locked::cupti_path),"CUPTI path differs");
        auto version=symbol<decltype(&cuptiGetVersion)>("cuptiGetVersion");uint32_t v=0;require(version(&v)==CUPTI_SUCCESS&&v==130401,"CUPTI version differs");
        subscribe=symbol<decltype(subscribe)>("cuptiSubscribe");domain=symbol<decltype(domain)>("cuptiEnableDomain");unsubscribe=symbol<decltype(unsubscribe)>("cuptiUnsubscribe");
        require(subscribe(&subscriber,probe::callback,&recorder)==CUPTI_SUCCESS,"CUPTI subscribe failed");active=true;
        require(domain(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API)==CUPTI_SUCCESS,"CUPTI enable failed");
    }
    ~Cupti(){if(active){domain(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API);unsubscribe(subscriber);}if(lib)FreeLibrary(lib);}
};
static void forbid_profiler(){
    for(const char* name:{"CUDA_INJECTION64_PATH","CUDA_PROFILE","NSYS_INJECTION_LIBRARY_PATH","NVTX_INJECTION64_PATH"})
        require(std::getenv(name)==nullptr,"profiling environment forbidden in timing mode");
    HMODULE modules[1024]{};DWORD bytes=0;require(EnumProcessModules(GetCurrentProcess(),modules,sizeof(modules),&bytes)&&bytes<=sizeof(modules),"module inventory failed");
    for(unsigned i=0;i<bytes/sizeof(HMODULE);++i){char path[32768]{};require(GetModuleFileNameA(modules[i],path,sizeof(path))!=0,"module name failed");
        std::string name=path;std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return char(std::tolower(c));});
        require(name.find("cupti")==std::string::npos&&name.find("nsys")==std::string::npos&&name.find("nvperf")==std::string::npos,"profiling module loaded in timing process");}
}
static std::string capture_json(const probe::Recorder& r){
    std::ostringstream out;out<<'[';for(unsigned i=0;i<r.count&&i<r.launches.size();++i){if(i)out<<',';auto& x=r.launches[i];
      auto geometry=[](dim3 v){return std::string("[")+std::to_string(v.x)+","+std::to_string(v.y)+","+std::to_string(v.z)+"]";};
      out<<"{\"index\":"<<i<<",\"symbol\":"<<quote(x.actual_symbol)<<",\"api_name\":"<<quote(x.api_name)<<",\"api_id\":"<<x.api
      <<",\"stream\":"<<(x.observed?std::to_string(x.stream):"null")<<",\"function\":"<<(x.observed?std::to_string(x.function):"null")
      <<",\"geometry_observed\":"<<(x.observed?"true":"false")<<",\"grid\":"<<(x.observed?geometry(x.grid):"null")
      <<",\"block\":"<<(x.observed?geometry(x.block):"null")<<",\"shared_bytes\":"<<(x.observed?std::to_string(x.shared):"null")
      <<",\"input\":"<<(x.decoded?std::to_string(x.input):"null")<<",\"output\":"<<(x.decoded?std::to_string(x.output):"null")
      <<",\"ncols\":"<<(x.decoded?std::to_string(x.ncols):"null")<<",\"attributes_source_qualified\":"<<(x.attributes_ok?"true":"false")
      <<",\"exit_seen\":"<<(x.exited?"true":"false")<<",\"return_code\":"<<(x.exited?std::to_string(x.result):"null")<<'}';}
    out<<']';return out.str();
}
int main(int argc,char** argv){
    if(argc!=5||!std::getenv("HOST_PROBE_AUTHORIZED_GPU_RUN")||std::strcmp(std::getenv("HOST_PROBE_AUTHORIZED_GPU_RUN"),"1")){
        std::cerr<<"Explicit future GPU authorization required; mode topology output protocol-sha\n";return 2;}
    const std::string mode=argv[1],topology=argv[2];const bool path_mode=mode=="path",chain=topology=="chain";
    if((!path_mode&&mode!="timing")||(!chain&&topology!="fanout"))return 2;
    HANDLE output=CreateFileA(argv[3],GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(output==INVALID_HANDLE_VALUE)return 3;
    std::ostringstream cases;std::string reason,modules,uuid;int exitcode=0;ggml_backend_t backend=nullptr;unsigned completed=0;
    try{
        modules=verify_modules();if(!path_mode)forbid_profiler();
        require(SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1))!=0,"thread affinity unavailable");
        cudaDeviceProp properties{};cuda_ok(cudaGetDeviceProperties(&properties,0));uuid=uuid_string(properties.uuid);
        require(uuid==locked::uuid&&properties.major==12&&properties.minor==0&&properties.multiProcessorCount==84,"GPU hardware identity differs");
        backend=ggml_backend_cuda_init(0);require(backend!=nullptr,"target CUDA backend unavailable");
        cases<<'[';
        for(int count:{1,4,16}){
            if(completed)cases<<',';
            {Graph warm(count,chain);warm.allocate(backend);warm.initialize();ggml_backend_synchronize(backend);
             for(int i=0;i<16;++i){warm.submit(backend);ggml_backend_synchronize(backend);}warm.verify(chain);}
            if(path_mode){
                Graph g(count,chain);g.allocate(backend);g.initialize();ggml_backend_synchronize(backend);
                probe::Recorder recorder;std::array<uintptr_t,16> ins{},outs{};
                for(int i=0;i<count;++i){ins[i]=reinterpret_cast<uintptr_t>(g.nodes[i]->src[0]->data);outs[i]=reinterpret_cast<uintptr_t>(g.nodes[i]->data);}
                {Cupti cupti(recorder);recorder.enabled=true;g.submit(backend);recorder.enabled=false;ggml_backend_synchronize(backend);}
                const bool qualified=probe::capture_ok(recorder,count,ins,outs);double error=g.verify(chain);
                cases<<"{\"nodes\":"<<count<<",\"executed_graph_nodes\":"<<ggml_graph_n_nodes(g.graph)<<",\"captured_launch_count\":"<<recorder.count
                     <<",\"qualified\":"<<(qualified?"true":"false")<<",\"max_abs_error\":"<<std::setprecision(17)<<error<<",\"launches\":"<<capture_json(recorder)<<'}';
                require(qualified,"runtime graph path does not match N unfused RMS_NORM kernels");
            }else{
                forbid_profiler();cases<<"{\"nodes\":"<<count<<",\"samples\":[";
                for(int sample=0;sample<31;++sample){if(sample)cases<<',';
                    auto a=stamp();auto g=std::make_unique<Graph>(count,chain);auto b=stamp();auto construction=delta(a,b);
                    a=stamp();g->allocate(backend);b=stamp();auto allocation=delta(a,b);
                    g->initialize();ggml_backend_synchronize(backend);
                    cudaStream_t envelope_stream=nullptr;cudaEvent_t first=nullptr,last=nullptr;
                    cuda_ok(cudaStreamCreateWithFlags(&envelope_stream,cudaStreamNonBlocking));cuda_ok(cudaEventCreate(&first));cuda_ok(cudaEventCreate(&last));
                    cuda_ok(cudaEventRecord(first,envelope_stream));cuda_ok(cudaEventSynchronize(first));
                    a=stamp();for(int replay=0;replay<128;++replay)g->submit(backend);b=stamp();auto submission=delta(a,b);
                    a=stamp();ggml_backend_synchronize(backend);b=stamp();auto synchronization=delta(a,b);
                    cuda_ok(cudaEventRecord(last,envelope_stream));cuda_ok(cudaEventSynchronize(last));float envelope_ms=0;cuda_ok(cudaEventElapsedTime(&envelope_ms,first,last));
                    cuda_ok(cudaEventDestroy(first));cuda_ok(cudaEventDestroy(last));cuda_ok(cudaStreamDestroy(envelope_stream));
                    double error=g->verify(chain);
                    cases<<"{\"sample\":"<<sample<<",\"graph_replays\":128,\"construction\":"<<timing_json(construction)<<",\"backend_allocation\":"<<timing_json(allocation)
                         <<",\"submit_batch\":"<<timing_json(submission)<<",\"synchronize\":"<<timing_json(synchronization)
                         <<",\"GPU_event_envelope_ms_includes_host_sync_gaps\":"<<std::setprecision(17)<<envelope_ms<<",\"pure_GPU_service_ms\":null,\"max_abs_error\":"<<error<<'}';
                }cases<<"]}";
            }++completed;
        }cases<<']';require(verify_modules()==modules,"target modules changed during run");
    }catch(const std::exception& error){reason=error.what();exitcode=1;}
    if(backend){ggml_backend_synchronize(backend);ggml_backend_free(backend);}
    std::ostringstream result;result<<"{\"schema\":\"host-submission-target-probe/v1\",\"mode\":"<<quote(mode)<<",\"topology\":"<<quote(topology)
      <<",\"status\":"<<quote(exitcode?"failed":path_mode?"path_qualified":"timing_recorded_unadmitted")<<",\"reason\":"<<quote(reason)
      <<",\"protocol_sha256\":"<<quote(argv[4])<<",\"GPU_uuid\":"<<quote(uuid)<<",\"pid\":"<<GetCurrentProcessId()<<",\"thread_id\":"<<GetCurrentThreadId()
      <<",\"calling_thread_CPU_only\":true,\"LLM_graph_reuse_internal_observed\":false,\"cost_model_admitted\":false,\"profiling\":"<<(path_mode?"true":"false")
      <<",\"completed_cases\":"<<completed<<",\"target_modules\":"<<(modules.empty()?"[]":modules)<<",\"cases\":"<<(exitcode?"[]":cases.str())
      <<",\"partial_case_records\":"<<(exitcode?quote(cases.str()):"null")<<"}\n";
    auto text=result.str();DWORD bytes=0;bool wrote=WriteFile(output,text.data(),DWORD(text.size()),&bytes,nullptr)!=0;CloseHandle(output);
    return wrote&&bytes==text.size()?exitcode:5;
}
