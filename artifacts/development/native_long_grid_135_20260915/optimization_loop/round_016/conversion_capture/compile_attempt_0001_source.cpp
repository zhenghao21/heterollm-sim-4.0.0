// Recorder only: no device pointer dereference and no CUDA calls in callback.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include <generated_cuda_runtime_api_meta.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
struct CuptiApi {
 HMODULE library=nullptr;
 decltype(&cuptiGetVersion) get_version=nullptr;
 decltype(&cuptiSubscribe) subscribe=nullptr;
 decltype(&cuptiEnableCallback) enable_callback=nullptr;
 decltype(&cuptiUnsubscribe) unsubscribe=nullptr;
 uint32_t runtime_api_version=0;std::string actual_path;
 template<typename T> T symbol(const char *name){auto address=GetProcAddress(library,name);if(!address)throw std::runtime_error(std::string("CUPTI export unavailable: ")+name);return reinterpret_cast<T>(address);}
 void load(){
  const char *path=std::getenv("CAPTURE_CUPTI_DLL");if(!path||!path[0])throw std::runtime_error("absolute CUPTI DLL selection required");
  if(!(std::strlen(path)>3&&path[1]==':'&&(path[2]=='\\'||path[2]=='/')))throw std::runtime_error("CUPTI DLL path must be absolute");
  library=LoadLibraryExA(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if(!library)throw std::runtime_error("CUPTI DLL loading failed: "+std::to_string(GetLastError()));
  char loaded[32768]{};if(!GetModuleFileNameA(library,loaded,sizeof(loaded)))throw std::runtime_error("CUPTI loaded identity unavailable");actual_path=loaded;
  std::string expected=path;std::replace(expected.begin(),expected.end(),'/','\\');
  if(_stricmp(expected.c_str(),actual_path.c_str())!=0)throw std::runtime_error("CUPTI DLL resolved to a different path");
  get_version=symbol<decltype(get_version)>("cuptiGetVersion");subscribe=symbol<decltype(subscribe)>("cuptiSubscribe");
  enable_callback=symbol<decltype(enable_callback)>("cuptiEnableCallback");unsubscribe=symbol<decltype(unsubscribe)>("cuptiUnsubscribe");
  auto status=get_version(&runtime_api_version);if(status!=CUPTI_SUCCESS)throw std::runtime_error("cuptiGetVersion failed");
  const char *expected_version=std::getenv("CAPTURE_CUPTI_API_VERSION");
  if(!expected_version||std::to_string(runtime_api_version)!=expected_version)throw std::runtime_error("selected CUPTI version differs from verified entry");
 }
 ~CuptiApi(){if(library)FreeLibrary(library);}
};

struct ConversionArgs {
 uintptr_t x=0,ids=0,vy=0;
 int64_t ne00=0,s01=0,s02=0,s03=0,ne0=0;
 int ne1=0,ne2=0,n_expert_used=0;
};
struct Launch {
 char symbol[1024]{};uint32_t correlation=0,context=0;
 uintptr_t function=0,stream=0;dim3 grid{},block{};size_t shared=0;
 bool conversion=false;bool signature_valid=false;ConversionArgs arguments{};
};
static std::array<Launch,64> records;
static std::atomic<unsigned> count{0};static std::atomic<bool> overflow{false};
static std::atomic<bool> malformed_callback{false};
static constexpr const char *conversion_mangled="_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0ELb0EEvPKfPKiPvxxxxxiii";
static constexpr const char *conversion_demangled="void quantize_mmq_q8_1<(mmq_q8_1_ds_layout)0, (bool)0>(const float *, const int *, void *, long long, long long, long long, long long, long long, int, int, int)";
template<typename T> T arg(void **args,int index){T value{};std::memcpy(&value,args[index],sizeof(value));return value;}
static ConversionArgs conversion_args(void **args){
 ConversionArgs a;a.x=arg<uintptr_t>(args,0);a.ids=arg<uintptr_t>(args,1);a.vy=arg<uintptr_t>(args,2);
 a.ne00=arg<int64_t>(args,3);a.s01=arg<int64_t>(args,4);a.s02=arg<int64_t>(args,5);a.s03=arg<int64_t>(args,6);a.ne0=arg<int64_t>(args,7);
 a.ne1=arg<int>(args,8);a.ne2=arg<int>(args,9);a.n_expert_used=arg<int>(args,10);return a;
}
static void CUPTIAPI callback(void *,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void *data){
 if(domain!=CUPTI_CB_DOMAIN_RUNTIME_API||id!=CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)return;
 if(!data){malformed_callback=true;return;}
 const auto *info=static_cast<const CUpti_CallbackData*>(data);if(info->callbackSite!=CUPTI_API_ENTER)return;
 if(!info->functionParams){malformed_callback=true;return;}
 unsigned index=count.fetch_add(1);if(index>=records.size()){overflow=true;return;}
 auto &out=records[index];const auto *launch=static_cast<const cudaLaunchKernel_v7000_params*>(info->functionParams);
 if(info->symbolName)std::strncpy(out.symbol,info->symbolName,sizeof(out.symbol)-1);
 out.correlation=info->correlationId;out.context=info->contextUid;out.function=reinterpret_cast<uintptr_t>(launch->func);
 out.stream=reinterpret_cast<uintptr_t>(launch->stream);out.grid=launch->gridDim;out.block=launch->blockDim;out.shared=launch->sharedMem;
 // Exact symbol was observed in the existing frozen Q8_0 MMQ SQLite trace; other symbols remain undecoded.
 out.conversion=std::strcmp(out.symbol,conversion_mangled)==0||std::strcmp(out.symbol,conversion_demangled)==0;
 if(out.conversion){
  if(!launch->args){malformed_callback=true;return;}
  for(int i=0;i<11;++i)if(!launch->args[i]){malformed_callback=true;return;}
  out.arguments=conversion_args(launch->args);auto &a=out.arguments;
  out.signature_valid=a.x!=0&&a.vy!=0&&a.ids==0&&a.ne00==1024&&a.ne0==1024&&a.s01==1024&&a.ne1==64&&a.ne2==1&&a.n_expert_used==0;
 }

}
static float random_value(uint32_t &s){s^=s<<13;s^=s>>17;s^=s<<5;return float(int32_t(s&65535)-32768)/65536.0f;}
static void check(CUptiResult code){if(code!=CUPTI_SUCCESS)throw std::runtime_error("CUPTI status "+std::to_string(int(code)));}
static std::string quote(const char *s){std::string out="\"";for(;*s;++s){if(*s=='\\'||*s=='\"')out+='\\';if(*s>=32)out+=*s;}return out+'\"';}
static int host_test(){
 uintptr_t x=1,ids=0,y=2;int64_t a=1024,b=1024,c=65536,d=65536,e=1024;int f=64,g=1,h=0;
 void *args[]={&x,&ids,&y,&a,&b,&c,&d,&e,&f,&g,&h};auto copied=conversion_args(args);
 if(copied.x!=1||copied.vy!=2||copied.ne00!=1024||copied.ne1!=64||copied.n_expert_used!=0)return 9;
 static_assert(sizeof(uintptr_t)==8);static_assert(sizeof(int64_t)==8);static_assert(sizeof(int)==4);
 cudaLaunchKernel_v7000_params launch{};launch.args=args;launch.gridDim=dim3(64,2,1);launch.blockDim=dim3(128,1,1);launch.stream=reinterpret_cast<cudaStream_t>(uintptr_t(3));
 CUpti_CallbackData data{};data.callbackSite=CUPTI_API_ENTER;data.functionParams=&launch;data.symbolName=conversion_mangled;data.correlationId=77;data.contextUid=1;
 callback(nullptr,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);
 if(count!=1||!records[0].conversion||!records[0].signature_valid||records[0].arguments.ne00!=1024||records[0].correlation!=77||malformed_callback)return 10;
 data.symbolName="unknown_quantize_mmq_q8_1_variant";launch.args=nullptr;
 callback(nullptr,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);
 if(count!=2||records[1].conversion||malformed_callback)return 11;
 CuptiApi api;api.load();
 std::cout<<"{\"host_api_struct_test\":true,\"synthetic_callback_test\":true,\"unknown_symbol_not_decoded\":true,\"GPU_context_created\":false,\"CUPTI_subscription_created\":false,\"compile_header_api_version\":"<<CUPTI_API_VERSION<<",\"runtime_api_version\":"<<api.runtime_api_version<<",\"header_runtime_versions_equal\":"<<(CUPTI_API_VERSION==api.runtime_api_version?"true":"false")<<",\"runtime_library\":"<<quote(api.actual_path.c_str())<<",\"actual_GPU_callback_compatibility_validated\":false}\n";
 return 0;
}

int main(int argc,char **argv){
 if(argc==2&&std::string(argv[1])=="--host-api-test"){try{return host_test();}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 12;}}
 if(argc!=3||std::string(argv[1])!="--run-recorder-only"){std::cerr<<"Explicit --run-recorder-only NEW_OUTPUT required. No GPU executed.\n";return 2;}
 HANDLE output=CreateFileA(argv[2],GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
 if(output==INVALID_HANDLE_VALUE){std::cerr<<"Refuse overwrite/unwritable output\n";return 2;}
 CuptiApi cupti;CUpti_SubscriberHandle subscriber{};bool subscribed=false;ggml_context *ctx=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t buffer=nullptr;int exitcode=0;
 std::string result;
 try{
  if(!std::getenv("GGML_CUDA_DISABLE_GRAPHS")||std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))!="1")throw std::runtime_error("graphs must be disabled");
  const char *native=std::getenv("CAPTURE_LOCKED_NATIVE_BIN");if(!native)throw std::runtime_error("locked native directory missing");
  for(const char *name:{"ggml-base.dll","ggml-cuda.dll"}){HMODULE mod=GetModuleHandleA(name);char actual[MAX_PATH]{};if(!mod||!GetModuleFileNameA(mod,actual,MAX_PATH))throw std::runtime_error("loaded module missing");std::string expected=std::string(native)+"\\"+name;if(_stricmp(actual,expected.c_str())!=0)throw std::runtime_error("loaded native DLL path mismatch");}
  cupti.load();check(cupti.subscribe(&subscriber,callback,nullptr));subscribed=true;
  constexpr int M=64,N=896,K=1024;
  ggml_init_params parameters{ggml_tensor_overhead()*8+ggml_graph_overhead_custom(8,false),nullptr,true};ctx=ggml_init(parameters);
  if(!ctx)throw std::runtime_error("context");
  auto *weights=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,K,N);auto *input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,M);
  auto *out=ggml_mul_mat(ctx,weights,input);auto *graph=ggml_new_graph_custom(ctx,8,false);ggml_build_forward_expand(graph,out);
  backend=ggml_backend_cuda_init(0);if(!backend||!ggml_backend_is_cuda(backend))throw std::runtime_error("CUDA backend unavailable");
  if(!ggml_backend_supports_op(backend,out))throw std::runtime_error("unsupported graph");
  std::vector<float>w(N*K),x(M*K);uint32_t seed=20260914;for(auto &v:w)v=random_value(seed);for(auto &v:x)v=random_value(seed);
  std::vector<uint8_t>packed(ggml_row_size(GGML_TYPE_Q8_0,K)*N);auto bytes=ggml_quantize_chunk(GGML_TYPE_Q8_0,w.data(),packed.data(),0,N,K,nullptr);
  if(bytes!=packed.size())throw std::runtime_error("packed size mismatch");
  buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)throw std::runtime_error("buffer");
  ggml_backend_tensor_set(weights,packed.data(),0,packed.size());ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));ggml_backend_synchronize(backend);
  check(cupti.enable_callback(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
  auto status=ggml_backend_graph_compute_async(backend,graph);ggml_backend_synchronize(backend);
  check(cupti.enable_callback(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));
  if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("graph failed");
  unsigned conversion_count=0;for(unsigned i=0;i<std::min<unsigned>(count.load(),records.size());++i)if(records[i].conversion&&records[i].signature_valid)++conversion_count;
  if(count==0||overflow||malformed_callback||conversion_count!=1)throw std::runtime_error("missing, ambiguous or invalid exact conversion callback; capture not accepted");
  result="{\"compile_header_api_version\":"+std::to_string(CUPTI_API_VERSION)+",\"runtime_api_version\":"+std::to_string(cupti.runtime_api_version)+",\"runtime_library\":"+quote(cupti.actual_path.c_str())+",\"schema\":\"original-dll-launch-recorder/v1\",\"timed\":false,\"device_buffer_copied\":false,\"quantized_code_observed\":false,\"overflow\":"+std::string(overflow?"true":"false")+",\"launches\":[";
  unsigned size=std::min<unsigned>(count.load(),records.size());
  for(unsigned i=0;i<size;++i){auto &r=records[i];if(i)result+=',';result+="{\"symbol\":"+quote(r.symbol)+",\"correlation\":"+std::to_string(r.correlation)+",\"context\":"+std::to_string(r.context)+",\"stream\":"+std::to_string(r.stream)+",\"grid\":["+std::to_string(r.grid.x)+","+std::to_string(r.grid.y)+","+std::to_string(r.grid.z)+"],\"block\":["+std::to_string(r.block.x)+","+std::to_string(r.block.y)+","+std::to_string(r.block.z)+"],\"shared\":"+std::to_string(r.shared)+",\"conversion_signature_decoded\":"+(r.conversion?"true":"false");
   if(r.conversion){auto&a=r.arguments;result+=",\"arguments\":{\"x\":"+std::to_string(a.x)+",\"ids\":"+std::to_string(a.ids)+",\"vy\":"+std::to_string(a.vy)+",\"ne00\":"+std::to_string(a.ne00)+",\"s01\":"+std::to_string(a.s01)+",\"s02\":"+std::to_string(a.s02)+",\"s03\":"+std::to_string(a.s03)+",\"ne0\":"+std::to_string(a.ne0)+",\"ne1\":"+std::to_string(a.ne1)+",\"ne2\":"+std::to_string(a.ne2)+",\"n_expert_used\":"+std::to_string(a.n_expert_used)+"}";}result+='}';}
  result+="]}";
 }catch(const std::exception&e){exitcode=4;result="{\"status\":\"failed\",\"error\":"+quote(e.what())+",\"quantized_code_observed\":false}";}
 if(subscribed)cupti.unsubscribe(subscriber);if(buffer)ggml_backend_buffer_free(buffer);if(backend)ggml_backend_free(backend);if(ctx)ggml_free(ctx);
 DWORD wrote=0;WriteFile(output,result.data(),DWORD(result.size()),&wrote,nullptr);CloseHandle(output);return wrote==result.size()?exitcode:5;
}
