// R28 bounded exact-shape timing. No default GPU operation. All output is exclusive.
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_runtime_cbid.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_026/mmvq_wrapper_correctness/cpu_reference.h"
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_024/mmvq_device_probe/r6_shared_abi/mmvq_probe_abi.h"
#include "json.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
using J=nlohmann::ordered_json;
namespace fs=std::filesystem;
static constexpr const char *protocol_sha="eadb23997980f5b7b13c3c794ffb7d603125c51f25b1a9f065cf5d7cef6253e3";
static_assert(CUPTI_API_VERSION==26,"ActivityKernel9 ABI locked to matched SDK26");
static_assert(sizeof(ggml_backend_event)==16 && offsetof(ggml_backend_event,context)==8,"locked event ABI changed");
static constexpr const char *conv_symbol="_Z13quantize_q8_1PKfPvxxxxxj5uint3";
static constexpr const char *main_symbol="_Z13mul_mat_vec_qIL9ggml_type6ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
static long long tick(){LARGE_INTEGER v;QueryPerformanceCounter(&v);return v.QuadPart;}
static long long frequency(){LARGE_INTEGER v;QueryPerformanceFrequency(&v);return v.QuadPart;}
static double ns(long long a,long long b){return double(b-a)*1e9/double(frequency());}
static void need(bool b,const std::string&m){if(!b)throw std::runtime_error(m);}
static void cuda_ok(cudaError_t c){need(c==cudaSuccess,"CUDA failure "+std::to_string(c));}
static void driver_ok(CUresult c){need(c==CUDA_SUCCESS,"driver failure "+std::to_string(c));}
static std::string hash_file(const std::string&path){
 BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;std::ifstream in(path,std::ios::binary);need(bool(in),"cannot read identity file "+path);
 need(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"SHA provider");
 need(BCryptCreateHash(alg,&hash,nullptr,0,nullptr,0,0)>=0,"SHA create");std::array<unsigned char,65536>b{};std::array<unsigned char,32>d{};
 while(in){in.read(reinterpret_cast<char*>(b.data()),b.size());if(in.gcount())need(BCryptHashData(hash,b.data(),ULONG(in.gcount()),0)>=0,"SHA update");}
 need(in.eof(),"SHA read");need(BCryptFinishHash(hash,d.data(),d.size(),0)>=0,"SHA finish");BCryptDestroyHash(hash);BCryptCloseAlgorithmProvider(alg,0);
 std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto x:d)out<<std::setw(2)<<unsigned(x);return out.str();
}
static J ref(const fs::path&p){return J{{"path",fs::absolute(p).string()},{"sha256",hash_file(p.string())},{"bytes",fs::file_size(p)}};}
static J read_json(const fs::path&p){std::ifstream in(p);need(bool(in),"JSON missing "+p.string());J j;in>>j;return j;}
static void verify_ref(const J&r){need(r.is_object()&&r.size()==3&&r==ref(r.at("path").get<std::string>()),"reference identity differs");}
static void write_bytes(const fs::path&p,const void*data,size_t size){
 HANDLE h=CreateFileA(p.string().c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);need(h!=INVALID_HANDLE_VALUE,"output exists/unwritable "+p.string());
 DWORD written=0;BOOL ok=WriteFile(h,data,DWORD(size),&written,nullptr);CloseHandle(h);need(ok&&written==size,"short exclusive write");
}
static void write_json(const fs::path&p,const J&j){const auto s=j.dump(2)+"\n";write_bytes(p,s.data(),s.size());}
static std::string module_path(HMODULE m){char b[32768]{};need(GetModuleFileNameA(m,b,sizeof(b))>0,"module path missing");return b;}
static void verify_loaded(const J&r,bool required){
 const auto path=r.at("path").get<std::string>();const auto name=fs::path(path).filename().string();HMODULE h=GetModuleHandleA(name.c_str());if(!h){need(!required,"required module not loaded: "+name);return;}
 const auto actual=module_path(h);need(_stricmp(actual.c_str(),path.c_str())==0&&hash_file(actual)==r.at("sha256"),"loaded module mismatch: "+name);
}
static J modules(const J&p,bool traced){
 J rows=J::array();for(const auto&m:p.at("identities").at("native_modules")){verify_loaded(m.at("ref"),m.at("required"));if(auto h=GetModuleHandleA(m.at("name").get<std::string>().c_str()))rows.push_back(ref(module_path(h)));}
 for(auto key:{"CUPTI","nvperf_host","nvperf_target"}){const auto&r=p.at("identities").at(key);verify_loaded(r,traced&&std::string(key)=="CUPTI");if(auto h=GetModuleHandleA(fs::path(r.at("path").get<std::string>()).filename().string().c_str())){need(traced,"profiler DLL loaded in unobserved control");rows.push_back(ref(module_path(h)));}}
 need(!GetModuleHandleA("cupti64_134.dll"),"different CUPTI runtime loaded");return rows;
}
static std::string uuid_string(const CUuuid&u){std::ostringstream s;s<<"GPU-"<<std::hex<<std::setfill('0');for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)s<<'-';s<<std::setw(2)<<unsigned(static_cast<unsigned char>(u.bytes[i]));}return s.str();}
static J driver_hardware(){
 CUdevice d;driver_ok(cuDeviceGet(&d,0));CUuuid u;driver_ok(cuDeviceGetUuid(&u,d));J j{{"uuid",uuid_string(u)}};
 for(auto v:std::initializer_list<std::pair<const char*,CUdevice_attribute>>{{"major",CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},{"minor",CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},{"SM_count",CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},{"warp_size",CU_DEVICE_ATTRIBUTE_WARP_SIZE},{"L2_bytes",CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE}}){int n=0;driver_ok(cuDeviceGetAttribute(&n,v.second,d));j[v.first]=n;}
 int version=0;driver_ok(cuDriverGetVersion(&version));j["driver_version"]=version;return j;
}
static J context_state(){CUcontext c=nullptr;driver_ok(cuCtxGetCurrent(&c));unsigned flags=0;int active=0;driver_ok(cuDevicePrimaryCtxGetState(0,&flags,&active));return J{{"current_context",uintptr_t(c)},{"primary_active",active},{"primary_flags",flags}};}
struct Trace;
static Trace*live=nullptr;
struct Trace{
 HMODULE dll=nullptr;CUpti_SubscriberHandle subscriber{};bool subscribed=false,enabled=false;uint32_t version=0;std::string phase="setup";
 #define API(name) decltype(&name) name##_p=nullptr
 API(cuptiFinalize);API(cuptiGetVersion);API(cuptiSubscribe);API(cuptiUnsubscribe);API(cuptiEnableDomain);API(cuptiActivityEnableHWTrace);API(cuptiActivityEnableLatencyTimestamps);API(cuptiActivityRegisterCallbacks);API(cuptiActivityEnable);API(cuptiActivityDisable);API(cuptiActivityEnableRuntimeApi);API(cuptiActivityGetNextRecord);API(cuptiActivityGetNumDroppedRecords);API(cuptiActivityFlushAll);API(cuptiActivityPushExternalCorrelationId);API(cuptiActivityPopExternalCorrelationId);
 #undef API
 struct Buffer{unsigned char*data=nullptr;size_t valid=0;CUcontext context=nullptr;uint32_t stream=0;bool returned=false;};std::array<Buffer,8>buffers{};std::atomic<size_t>requested{0};std::atomic<bool>overflow{false};std::mutex mutex;J states=J::array(),calls=J::array();size_t dropped=0;
 template<class T>void sym(T&f,const char*n){f=reinterpret_cast<T>(GetProcAddress(dll,n));need(f!=nullptr,std::string("CUPTI export missing ")+n);}
 static void CUPTIAPI state_callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void*data){if(!live||domain!=CUPTI_CB_DOMAIN_STATE)return;try{const auto*s=static_cast<const CUpti_StateData*>(data);std::lock_guard<std::mutex>lock(live->mutex);live->states.push_back(J{{"id",id},{"result",int(s->notification.result)},{"message",s->notification.message?s->notification.message:""},{"phase",live->phase},{"qpc",tick()}});}catch(...){live->overflow=true;}}
 static void CUPTIAPI request(uint8_t**buffer,size_t*size,size_t*records){const size_t i=live->requested.fetch_add(1);if(i>=live->buffers.size()){live->overflow=true;*buffer=nullptr;*size=0;*records=0;return;}*buffer=live->buffers[i].data;*size=1048576;*records=0;}
 static void CUPTIAPI complete(CUcontext ctx,uint32_t stream,uint8_t*buffer,size_t,size_t valid){try{std::lock_guard<std::mutex>lock(live->mutex);for(auto&b:live->buffers)if(b.data==buffer){need(!b.returned,"CUPTI buffer returned twice");b.valid=valid;b.context=ctx;b.stream=stream;b.returned=true;return;}live->overflow=true;}catch(...){live->overflow=true;}}
 void record(const char*name,CUptiResult status){calls.push_back(J{{"name",name},{"returncode",int(status)},{"phase",phase},{"qpc",tick()}});}
 void checked(const char*name,CUptiResult status){record(name,status);need(status==CUPTI_SUCCESS,std::string(name)+" failed "+std::to_string(status));}
 void load(const J&p){const auto&r=p.at("identities").at("CUPTI");verify_ref(r);dll=LoadLibraryExA(r.at("path").get<std::string>().c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);need(dll!=nullptr,"matched CUPTI failed loading");verify_loaded(r,true);live=this;
 #define LOAD(name) sym(name##_p,#name)
 LOAD(cuptiFinalize);LOAD(cuptiGetVersion);LOAD(cuptiSubscribe);LOAD(cuptiUnsubscribe);LOAD(cuptiEnableDomain);LOAD(cuptiActivityEnableHWTrace);LOAD(cuptiActivityEnableLatencyTimestamps);LOAD(cuptiActivityRegisterCallbacks);LOAD(cuptiActivityEnable);LOAD(cuptiActivityDisable);LOAD(cuptiActivityEnableRuntimeApi);LOAD(cuptiActivityGetNextRecord);LOAD(cuptiActivityGetNumDroppedRecords);LOAD(cuptiActivityFlushAll);LOAD(cuptiActivityPushExternalCorrelationId);LOAD(cuptiActivityPopExternalCorrelationId);
 #undef LOAD
 checked("cuptiGetVersion",cuptiGetVersion_p(&version));need(version==26,"runtime/header activity ABI mismatch");checked("cuptiSubscribe",cuptiSubscribe_p(&subscriber,state_callback,nullptr));subscribed=true;checked("enable_STATE",cuptiEnableDomain_p(1,subscriber,CUPTI_CB_DOMAIN_STATE));}
 void enable_hes(){phase="enable_HES_before_context";checked("cuptiActivityEnableHWTrace(1)",cuptiActivityEnableHWTrace_p(1));}
 void begin(){phase="formal";for(auto&b:buffers){b.data=static_cast<unsigned char*>(_aligned_malloc(1048576,8));need(b.data!=nullptr,"activity pool allocation failed");}
 checked("register_activity_buffers",cuptiActivityRegisterCallbacks_p(request,complete));checked("enable_CONCURRENT_KERNEL",cuptiActivityEnable_p(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));enabled=true;checked("enable_EXTERNAL_CORRELATION",cuptiActivityEnable_p(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));checked("enable_only_ExC430",cuptiActivityEnableRuntimeApi_p(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,1));}
 void push(uint64_t id){need(cuptiActivityPushExternalCorrelationId_p(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,id)==CUPTI_SUCCESS,"push correlation");}
 void pop(uint64_t expected){uint64_t id=0;need(cuptiActivityPopExternalCorrelationId_p(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&id)==CUPTI_SUCCESS&&id==expected,"pop correlation");}
 J finish(const fs::path&out){phase="flush_after_terminal_sync";checked("disable_ExC430",cuptiActivityEnableRuntimeApi_p(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,0));checked("disable_CONCURRENT_KERNEL",cuptiActivityDisable_p(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));checked("disable_EXTERNAL_CORRELATION",cuptiActivityDisable_p(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));enabled=false;checked("flush_all_completed",cuptiActivityFlushAll_p(0));
 J raw=J::array(),kernels=J::array(),apis=J::array(),links=J::array(),unknown=J::array();size_t index=0;
 for(auto&b:buffers){if(!b.returned)continue;need(b.valid<=1048576,"invalid activity buffer length");const fs::path path=out/("activity_buffer."+std::to_string(index++)+".bin");write_bytes(path,b.data,b.valid);raw.push_back(J{{"ref",ref(path)},{"context",uintptr_t(b.context)},{"stream",b.stream},{"valid_bytes",b.valid}});
 size_t lost=0;checked("get_dropped_records",cuptiActivityGetNumDroppedRecords_p(b.context,b.stream,&lost));dropped+=lost;
 CUpti_Activity*a=nullptr;while(true){auto s=cuptiActivityGetNextRecord_p(b.data,b.valid,&a);if(s==CUPTI_ERROR_MAX_LIMIT_REACHED)break;need(s==CUPTI_SUCCESS&&a,"activity record decode error");
 if(a->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){const auto*k=reinterpret_cast<const CUpti_ActivityKernel9*>(a);kernels.push_back(J{{"kind",int(a->kind)},{"name",k->name?k->name:""},{"start",k->start},{"end",k->end},{"completed",k->completed},{"queued",k->queued},{"submitted",k->submitted},{"correlation",k->correlationId},{"context",k->contextId},{"stream",k->streamId},{"device",k->deviceId},{"grid",{k->gridX,k->gridY,k->gridZ}},{"block",{k->blockX,k->blockY,k->blockZ}},{"dynamic_shared",k->dynamicSharedMemory},{"static_shared",k->staticSharedMemory},{"registers_per_thread",k->registersPerThread},{"graph_id",k->graphId},{"graph_node_id",k->graphNodeId}});}
 else if(a->kind==CUPTI_ACTIVITY_KIND_RUNTIME){const auto*x=reinterpret_cast<const CUpti_ActivityAPI*>(a);apis.push_back(J{{"cbid",x->cbid},{"start",x->start},{"end",x->end},{"correlation",x->correlationId},{"return_value",x->returnValue}});}
 else if(a->kind==CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION){const auto*x=reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(a);links.push_back(J{{"kind",int(x->externalKind)},{"external_id",x->externalId},{"correlation",x->correlationId}});}
 else unknown.push_back(int(a->kind));}}
 return J{{"kernel_record_ABI","CUpti_ActivityKernel9"},{"raw_buffers",raw},{"kernels",kernels},{"runtime_APIs",apis},{"external_links",links},{"unknown_kinds",unknown},{"dropped_records",dropped},{"buffer_overflow",bool(overflow)}};
 }
 J evidence(){std::lock_guard<std::mutex>lock(mutex);return J{{"compile_API_version",CUPTI_API_VERSION},{"runtime_API_version",version},{"calls",calls},{"STATE",states},{"callback_overflow",bool(overflow)}};}
 ~Trace(){if(enabled){cuptiActivityEnableRuntimeApi_p(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060,0);cuptiActivityDisable_p(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);cuptiActivityDisable_p(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);cuptiActivityFlushAll_p(0);}if(subscribed)cuptiUnsubscribe_p(subscriber);if(dll)cuptiFinalize_p();for(auto&b:buffers)if(b.data)_aligned_free(b.data);if(dll)FreeLibrary(dll);if(live==this)live=nullptr;}
};
struct Slot{ggml_context*ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_tensor*w=nullptr,*x=nullptr,*y=nullptr;ggml_cgraph*graph=nullptr;void*dw=nullptr,*dx=nullptr,*dq=nullptr,*dy=nullptr;};
struct Work{
 std::string role;bool pair=false;ggml_backend_t backend=nullptr;cudaStream_t stream=nullptr;std::vector<Slot>slots;std::vector<float>weights,input;std::vector<uint8_t>packed,q8;check::Reference reference;
 Work(const std::string&role_,size_t count):role(role_),pair(role_=="pair"),weights(check::make_weights()),input(check::make_input()),packed(check::quantize_weight(weights)),q8(check::expected_q8(input)),reference(check::reference(packed,q8)){
  cuda_ok(cudaSetDevice(0));if(pair){backend=ggml_backend_cuda_init(0);need(backend&&ggml_backend_is_cuda(backend),"no native CUDA backend");}else cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));slots.resize(count);
  for(auto&s:slots){if(pair){ggml_init_params init{ggml_tensor_overhead()*8+ggml_graph_overhead_custom(8,false),nullptr,true};s.ctx=ggml_init(init);need(s.ctx,"ggml context missing");s.w=ggml_new_tensor_2d(s.ctx,GGML_TYPE_Q5_0,check::K,check::N);s.x=ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,check::K,1);s.y=ggml_mul_mat(s.ctx,s.w,s.x);s.graph=ggml_new_graph_custom(s.ctx,8,false);ggml_build_forward_expand(s.graph,s.y);need(ggml_backend_supports_op(backend,s.y),"unsupported target op");s.buffer=ggml_backend_alloc_ctx_tensors(s.ctx,backend);need(s.buffer&&ggml_backend_buffer_get_usage(s.buffer)!=GGML_BACKEND_BUFFER_USAGE_COMPUTE,"target buffer usage differs");ggml_backend_tensor_set(s.w,packed.data(),0,packed.size());ggml_backend_tensor_set(s.x,input.data(),0,input.size()*4);}
   else {cuda_ok(cudaMalloc(&s.dw,packed.size()));cuda_ok(cudaMalloc(&s.dx,input.size()*4));cuda_ok(cudaMalloc(&s.dq,q8.size()));cuda_ok(cudaMalloc(&s.dy,check::N*4));cuda_ok(cudaMemcpyAsync(s.dw,packed.data(),packed.size(),cudaMemcpyHostToDevice,stream));cuda_ok(cudaMemcpyAsync(s.dx,input.data(),input.size()*4,cudaMemcpyHostToDevice,stream));cuda_ok(cudaMemcpyAsync(s.dq,q8.data(),q8.size(),cudaMemcpyHostToDevice,stream));}}
  sync();
 }
 void invoke(size_t i){auto&s=slots[i];if(pair)need(ggml_backend_graph_compute_async(backend,s.graph)==GGML_STATUS_SUCCESS,"native graph failed");else if(role=="main")heterollm_mmvq_probe_main(s.dw,GGML_TYPE_Q5_0,s.dq,static_cast<float*>(s.dy),check::K,check::N,1,check::K/32,stream);else heterollm_mmvq_probe_convert_q8_1(static_cast<const float*>(s.dx),s.dq,GGML_TYPE_Q5_0,check::K,1,check::K,stream);}
 void sync(){if(pair){ggml_backend_synchronize(backend);cuda_ok(cudaGetLastError());}else cuda_ok(cudaStreamSynchronize(stream));}
 void event(cudaEvent_t e){if(pair){ggml_backend_event ev{ggml_backend_get_device(backend),reinterpret_cast<void*>(e)};ggml_backend_event_record(&ev,backend);cuda_ok(cudaGetLastError());}else cuda_ok(cudaEventRecord(e,stream));}
 J validate(const fs::path&out,const std::string&phase){J rows=J::array();bool all=true;std::vector<float>actual(check::N);std::vector<uint8_t>converted(q8.size());
  for(size_t i=0;i<slots.size();++i){auto&s=slots[i];if(role=="convert"){cuda_ok(cudaMemcpyAsync(converted.data(),s.dq,converted.size(),cudaMemcpyDeviceToHost,stream));sync();size_t bad=check::byte_mismatches(converted,q8);all&=bad==0;const auto path=out/(phase+"."+std::to_string(i)+".q8.bin");write_bytes(path,converted.data(),converted.size());rows.push_back(J{{"slot",i},{"byte_mismatches",bad},{"actual_ref",ref(path)}});}
   else {if(pair)ggml_backend_tensor_get(s.y,actual.data(),0,actual.size()*4);else{cuda_ok(cudaMemcpyAsync(actual.data(),s.dy,actual.size()*4,cudaMemcpyDeviceToHost,stream));sync();}auto v=check::compare_output(actual,reference);all&=v.failed==0;const auto path=out/(phase+"."+std::to_string(i)+".f32.bin");write_bytes(path,actual.data(),actual.size()*4);rows.push_back(J{{"slot",i},{"values",v.tested},{"failed",v.failed},{"max_absolute_error",std::isfinite(v.max_absolute_error)?J(v.max_absolute_error):J(nullptr)},{"max_bound_ratio",std::isfinite(v.max_bound_ratio)?J(v.max_bound_ratio):J(nullptr)},{"actual_ref",ref(path)}});}}
  return J{{"passed",all},{"rows",rows}};
 }
 J save_reference(const fs::path&out){J j;for(const auto&v:std::initializer_list<std::tuple<std::string,const void*,size_t>>{{"packed_q5",packed.data(),packed.size()},{"input_f32",input.data(),input.size()*4},{"expected_q8",q8.data(),q8.size()},{"reference_f64",reference.dot.data(),reference.dot.size()*8},{"bound_f64",reference.bound.data(),reference.bound.size()*8}}){auto path=out/(std::get<0>(v)+".bin");write_bytes(path,std::get<1>(v),std::get<2>(v));j[std::get<0>(v)]=ref(path);}return j;}
 ~Work(){for(auto&s:slots){if(s.buffer)ggml_backend_buffer_free(s.buffer);if(s.ctx)ggml_free(s.ctx);if(s.dw)cudaFree(s.dw);if(s.dx)cudaFree(s.dx);if(s.dq)cudaFree(s.dq);if(s.dy)cudaFree(s.dy);}if(backend)ggml_backend_free(backend);if(stream)cudaStreamDestroy(stream);}
};
static void verify_capabilities(const fs::path&path,const J&p){const auto j=read_json(path);need(j.at("schema")=="r28-hes-counterfactual-proof/v1","counterfactual proof missing");J docs;for(auto key:{"without_hes","after_hes"}){const auto&r=j.at(key);verify_ref(r);docs[key]=read_json(r.at("path").get<std::string>());need(docs[key].at("protocol_ref")==ref(p.at("_protocol_path").get<std::string>()),"capability protocol differs");need(docs[key].at("status")=="capability_observed"&&docs[key].at("contexts_before").at("primary_active")==0&&docs[key].at("contexts_before").at("current_context")==0,"capability phase invalid");}
 need(docs["without_hes"].at("latency_api_returncode")==int(CUPTI_SUCCESS)&&docs["after_hes"].at("latency_api_returncode")==int(CUPTI_ERROR_NOT_SUPPORTED)&&docs["after_hes"].at("HES_enable_returncode")==int(CUPTI_SUCCESS),"HES actual-mode counterfactual unproven");}
int main(int argc,char**argv){
 std::map<std::string,std::string>a;for(int i=1;i+1<argc;i+=2)a[argv[i]]=argv[i+1];
 if(!std::getenv("GPU_OPERATOR_TIMING_AUTHORIZED")||std::string(std::getenv("GPU_OPERATOR_TIMING_AUTHORIZED"))!="1"||!a.count("--protocol")||!a.count("--output")||!a.count("--kind")){std::cerr<<"No GPU executed. Explicit root runner required.\n";return 2;}
 const fs::path out=a["--output"];if(!fs::is_directory(out)||fs::exists(out/"raw.json")){std::cerr<<"New prepared output directory required\n";return 2;}
 J result{{"schema","r28-gpu-operator-raw/v1"},{"status","rejected"},{"kind",a["--kind"]},{"timed",false},{"performance_parameter_admitted",false},{"QPC_frequency",frequency()},{"expected_CUPTI_API",26}};
 Trace trace;bool traced=false;int exitcode=4;
 try{need(hash_file(a["--protocol"])==protocol_sha,"protocol identity differs");J p=read_json(a["--protocol"]);p["_protocol_path"]=a["--protocol"];result["protocol_ref"]=ref(a["--protocol"]);
 need(std::getenv("GGML_CUDA_DISABLE_GRAPHS")&&std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))=="1","graphs must be disabled");need(std::getenv("GGML_CUDA_PDL")&&std::string(std::getenv("GGML_CUDA_PDL"))=="1","PDL must be explicit1");
 const bool capability=a["--kind"].rfind("capability_",0)==0;traced=capability||(a.count("--mode")&&(a["--mode"]=="B"||a["--mode"]=="AB"));
 if(traced)trace.load(p);const auto before_driver=tick();const auto init=cuInit(0);result["driver_init"]={{"qpc_begin",before_driver},{"qpc_end",tick()},{"returncode",int(init)}};driver_ok(init);
 result["contexts_before"]=context_state();need(result["contexts_before"]["primary_active"]==0&&result["contexts_before"]["current_context"]==0,"context already active before HES selection");result["hardware"]=driver_hardware();for(auto it=p["hardware"].begin();it!=p["hardware"].end();++it)need(result["hardware"].at(it.key())==it.value(),"wrong physical GPU");
 if(capability){need(a["--kind"]=="capability_without_hes"||a["--kind"]=="capability_after_hes","unknown capability");result["HES_enable_returncode"]=nullptr;if(a["--kind"]=="capability_after_hes"){trace.enable_hes();result["HES_enable_returncode"]=0;}
 trace.phase="counterfactual_latency_api_only";const auto s=trace.cuptiActivityEnableLatencyTimestamps_p(1);trace.record("counterfactual_latency_enable(1)",s);result["latency_api_returncode"]=int(s);if(s==CUPTI_SUCCESS)trace.checked("restore_latency_disabled",trace.cuptiActivityEnableLatencyTimestamps_p(0));result["contexts_after"]=context_state();need(result["contexts_after"]["current_context"]==0&&result["contexts_after"]["primary_active"]==0,"capability unexpectedly created context");result["status"]="capability_observed";result["GPU_kernel_executed"]=false;exitcode=0;
 }else{
 need(a["--kind"]=="formal"&&a.count("--state")&&a.count("--mode")&&a.count("--proof"),"formal arguments missing");const auto mode=a["--mode"];need(mode=="U"||mode=="A"||mode=="B"||mode=="AB","unknown mode");verify_capabilities(a["--proof"],p);result["mode_proof_ref"]=ref(a["--proof"]);if(traced)trace.enable_hes();
 result["contexts_after_HES_before_work"]=context_state();need(result["contexts_after_HES_before_work"]["primary_active"]==0&&result["contexts_after_HES_before_work"]["current_context"]==0,"HES timing boundary wrong");
 J state;for(auto&s:p["states"])if(s["id"]==a["--state"])state=s;need(!state.is_null(),"unknown/held-out state forbidden");result["state"]=state;result["mode"]=mode;result["modules_before"]=modules(p,traced);
 const size_t weight_bytes=size_t(check::K)*check::N/32*22;const size_t L2=result["hardware"]["L2_bytes"].get<size_t>();need(L2>0,"L2 unavailable");const size_t count=state["cache"]=="rotation"?std::max<size_t>(2,(4*L2+weight_bytes-1)/weight_bytes+1):1;
 need(count<=128&&count*weight_bytes<=1073741824,"rotation exceeds frozen allocation bound");result["rotation"]={{"slots",count},{"weight_bytes_per_slot",weight_bytes},{"total_weight_bytes",count*weight_bytes},{"L2_bytes",L2},{"policy",state["cache"]}};
 Work w(state["role"],count);result["references"]=w.save_reference(out);for(size_t i=0;i<count;++i){w.invoke(i);w.sync();}result["correctness_before"]=w.validate(out,"before");need(result["correctness_before"]["passed"],"pre-timing correctness failed");
 const auto warm_start=tick();size_t warm=0;do{w.invoke(warm%count);w.sync();++warm;need(ns(warm_start,tick())<=5e9,"warmup maximum exceeded");}while(warm<32||ns(warm_start,tick())<5e8);result["warmup"]={{"calls",warm},{"begin_qpc",warm_start},{"end_qpc",tick()}};
 cudaEvent_t begin=nullptr,end=nullptr;const bool events=(mode=="A"||mode=="AB");if(events){cuda_ok(cudaEventCreate(&begin));cuda_ok(cudaEventCreate(&end));}if(traced){need(trace.states.empty(),"STATE notification before formal");trace.begin();}
 const auto prime_begin=tick();for(int i=0;i<32;++i){if(traced)trace.push(1000000+uint64_t(i));if(events)w.event(begin);w.invoke((warm+size_t(i))%count);if(events)w.event(end);w.sync();if(traced)trace.pop(1000000+uint64_t(i));}result["observer_priming"]={{"calls",32},{"begin_qpc",prime_begin},{"end_qpc",tick()},{"external_base",1000000}};const size_t base_slot=(warm+32)%count;result["formal_base_slot"]=base_slot;J samples=J::array();result["formal_begin_qpc"]=tick();result["GPU_kernel_executed"]=true;result["timed"]=true;
 for(int i=0;i<64;++i){const uint64_t id=uint64_t(i)+1;if(traced)trace.push(id);const auto start=tick();if(events)w.event(begin);const auto submit_begin=tick();w.invoke((base_slot+size_t(i))%count);const auto submit_end=tick();if(events)w.event(end);const auto wait_begin=tick();w.sync();const auto stop=tick();if(traced)trace.pop(id);
 J s{{"sample",i},{"external_id",id},{"slot",(base_slot+size_t(i))%count},{"begin_qpc",start},{"end_qpc",stop},{"wall_ns",ns(start,stop)},{"submit_begin_qpc",submit_begin},{"submit_end_qpc",submit_end},{"wait_begin_qpc",wait_begin},{"event_ns",nullptr}};
 if(events){float ms=0;cuda_ok(cudaEventElapsedTime(&ms,begin,end));need(std::isfinite(ms)&&ms>0,"event envelope nonpositive");s["event_ns"]=double(ms)*1e6;}samples.push_back(s);}
 result["formal_end_qpc"]=tick();result["samples"]=samples;if(traced)result["activity"]=trace.finish(out);else result["activity"]=nullptr;
 if(begin)cuda_ok(cudaEventDestroy(begin));if(end)cuda_ok(cudaEventDestroy(end));result["correctness_after"]=w.validate(out,"after");need(result["correctness_after"]["passed"],"post-timing correctness failed");result["modules_after"]=modules(p,traced);if(traced)need(trace.states.empty()&&!trace.overflow,"STATE/callback failure observed");result["status"]="formal_observed_pending_independent_validation";result["actual_mode_evidence"]={{"kind",traced?"HWTrace_API_success_bound_to_counterfactual_and_clean_STATE":"unobserved_control"},{"direct_mode_readback_available",false},{"latency_timestamps_requested_in_formal",false},{"software_fallback_allowed",false}};exitcode=0;
 }
 }catch(const std::exception&e){result["error"]=e.what();}
 if(traced)result["CUPTI_evidence"]=trace.evidence();else result["CUPTI_evidence"]=nullptr;
 try{write_json(out/"raw.json",result);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 5;}return exitcode;
}
