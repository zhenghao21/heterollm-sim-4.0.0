// Capture the ORIGINAL DLL runtime func, then replay it on owned buffers.
// No original pool read, no CUDA calls in callback, no rewritten device kernel.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <utility>
#include <sstream>
#include <limits>
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
 uintptr_t function=0,stream=0,context_handle=0;dim3 grid{},block{};size_t shared=0;
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
 out.correlation=info->correlationId;out.context=info->contextUid;out.context_handle=reinterpret_cast<uintptr_t>(info->context);out.function=reinterpret_cast<uintptr_t>(launch->func);
 out.stream=reinterpret_cast<uintptr_t>(launch->stream);out.grid=launch->gridDim;out.block=launch->blockDim;out.shared=launch->sharedMem;
 // Exact symbol was observed in the existing frozen Q8_0 MMQ SQLite trace; other symbols remain undecoded.
 out.conversion=std::strcmp(out.symbol,conversion_mangled)==0||std::strcmp(out.symbol,conversion_demangled)==0;
 if(out.conversion){
  if(!launch->args){malformed_callback=true;return;}
  for(int i=0;i<11;++i)if(!launch->args[i]){malformed_callback=true;return;}
  out.arguments=conversion_args(launch->args);auto &a=out.arguments;
  out.signature_valid=a.x!=0&&a.vy!=0&&a.ids==0&&a.ne00==1024&&a.ne0==1024&&a.s01==1024&&a.s02==65536&&a.s03==65536&&a.ne1==64&&a.ne2==1&&a.n_expert_used==0;
 }

}

constexpr int M=64,N=896,K=1024,QK=128;
struct BlockQ8MMQD4 { float d4[4]; int8_t qs[QK]; };
static_assert(sizeof(BlockQ8MMQD4)==144,"D4 block must match source block_q8_1_mmq");
static_assert(offsetof(BlockQ8MMQD4,d4)==0&&offsetof(BlockQ8MMQD4,qs)==16,"D4 offsets");
static_assert(sizeof(uintptr_t)==8&&sizeof(int64_t)==8&&sizeof(int)==4&&sizeof(float)==4,"captured ABI");
static_assert(sizeof(ggml_fp16_t)==2,"original Q8_0 scale layout");
constexpr size_t input_bytes=size_t(M)*K*sizeof(float);
constexpr size_t output_bytes=sizeof(BlockQ8MMQD4)*M*(K/QK);
constexpr size_t graph_bytes=size_t(M)*N*sizeof(float);

static float random_value(uint32_t &s){s^=s<<13;s^=s>>17;s^=s<<5;return float(int32_t(s&65535)-32768)/65536.0f;}
static void check(CUptiResult c){if(c!=CUPTI_SUCCESS)throw std::runtime_error("CUPTI status "+std::to_string(int(c)));}
static void check_cuda(cudaError_t c,const char *operation){if(c!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+std::to_string(int(c))+" "+cudaGetErrorString(c));}
static std::string quote(const char *s){std::string v="\"";for(;*s;++s){const unsigned char c=static_cast<unsigned char>(*s);if(c=='\\'||c=='\"'){v+='\\';v+=char(c);}else if(c=='\n')v+="\\n";else if(c=='\r')v+="\\r";else if(c=='\t')v+="\\t";else if(c>=32)v+=char(c);}return v+'"';}
static std::string hex(uintptr_t p){std::ostringstream out;out<<"0x"<<std::hex<<p;return quote(out.str().c_str());}
static std::string real_json(float value){if(!std::isfinite(value))return "null";std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();}
static std::string dimensions(dim3 d){return "["+std::to_string(d.x)+","+std::to_string(d.y)+","+std::to_string(d.z)+"]";}
static std::string args_json(const ConversionArgs &a){std::ostringstream o;o<<"{\"x\":"<<a.x<<",\"ids\":"<<a.ids<<",\"vy\":"<<a.vy<<",\"ne00\":"<<a.ne00<<",\"s01\":"<<a.s01<<",\"s02\":"<<a.s02<<",\"s03\":"<<a.s03<<",\"ne0\":"<<a.ne0<<",\"ne1\":"<<a.ne1<<",\"ne2\":"<<a.ne2<<",\"n_expert_used\":"<<a.n_expert_used<<"}";return o.str();}
static bool same_scalars(const ConversionArgs&a,const ConversionArgs&b){return a.ids==b.ids&&a.ne00==b.ne00&&a.s01==b.s01&&a.s02==b.s02&&a.s03==b.s03&&a.ne0==b.ne0&&a.ne1==b.ne1&&a.ne2==b.ne2&&a.n_expert_used==b.n_expert_used;}
static void validate_capture(const Launch &r){
 const auto&a=r.arguments;
 if(!r.function||!r.conversion||!r.signature_valid||!a.x||!a.vy||a.ids||a.ne00!=K||a.s01!=K||a.s02!=M*K||a.s03!=M*K||a.ne0!=K||a.ne1!=M||a.ne2!=1||a.n_expert_used!=0)throw std::runtime_error("capture does not match bounded D4 non-scatter signature");
 if(r.grid.x!=64||r.grid.y!=2||r.grid.z!=1||r.block.x!=128||r.block.y!=1||r.block.z!=1||r.shared!=0)throw std::runtime_error("captured geometry differs from original frozen observation");
}
struct ReplayArgs {
 const float *x;const int32_t *ids=nullptr;void *vy;
 int64_t ne00,s01,s02,s03,ne0;int ne1,ne2,n_expert_used;
 std::array<void*,11> slots{};
 ReplayArgs(const ConversionArgs &a,const float *input,void *output):x(input),vy(output),ne00(a.ne00),s01(a.s01),s02(a.s02),s03(a.s03),ne0(a.ne0),ne1(a.ne1),ne2(a.ne2),n_expert_used(a.n_expert_used){
  if(a.ids)throw std::runtime_error("only captured nullptr ids accepted");
  slots={&x,&ids,&vy,&ne00,&s01,&s02,&s03,&ne0,&ne1,&ne2,&n_expert_used};
 }
 ReplayArgs(const ReplayArgs&)=delete;ReplayArgs&operator=(const ReplayArgs&)=delete;
 ConversionArgs decoded(){return conversion_args(slots.data());}
};
static size_t block_index(int row,int k){if(row<0||row>=M||k<0||k>=K)throw std::runtime_error("D4 decode index out of bounds");return size_t(k/QK)*M+row;}
static size_t q_offset(int row,int k){return block_index(row,k)*sizeof(BlockQ8MMQD4)+offsetof(BlockQ8MMQD4,qs)+(k%QK);}
static size_t d_offset(int row,int k){return block_index(row,k)*sizeof(BlockQ8MMQD4)+offsetof(BlockQ8MMQD4,d4)+(k%QK/32)*sizeof(float);}
static std::string launches_json(){std::string out="[";for(unsigned i=0;i<std::min<unsigned>(count.load(),records.size());++i){const auto&r=records[i];if(i)out+=',';out+="{\"symbol\":"+quote(r.symbol)+",\"correlation\":"+std::to_string(r.correlation)+",\"context_uid\":"+std::to_string(r.context)+",\"context_handle\":"+hex(r.context_handle)+",\"runtime_func\":"+hex(r.function)+",\"runtime_func_decimal\":"+std::to_string(r.function)+",\"stream\":"+hex(r.stream)+",\"grid\":"+dimensions(r.grid)+",\"block\":"+dimensions(r.block)+",\"shared\":"+std::to_string(r.shared)+",\"conversion_signature_decoded\":"+(r.conversion?"true":"false")+",\"signature_valid\":"+(r.signature_valid?"true":"false");if(r.conversion)out+=",\"arguments\":"+args_json(r.arguments);out+='}';}return out+"]";}
static std::string module_path(HMODULE mod){char path[32768]{};if(!mod||!GetModuleFileNameA(mod,path,sizeof(path)))throw std::runtime_error("loaded module path unavailable");return path;}
static std::string verify_native_modules(){const char*native=std::getenv("CAPTURE_LOCKED_NATIVE_BIN");if(!native)throw std::runtime_error("locked native directory missing");std::string out="[";for(const char*name:{"ggml-base.dll","ggml-cuda.dll"}){auto actual=module_path(GetModuleHandleA(name));std::string expected=std::string(native)+"\\"+name;std::replace(expected.begin(),expected.end(),'/','\\');if(_stricmp(actual.c_str(),expected.c_str())!=0)throw std::runtime_error("loaded native DLL path mismatch");if(out!="[")out+=',';out+="{\"name\":"+quote(name)+",\"actual_path\":"+quote(actual.c_str())+"}";}return out+"]";}
static std::string host_module_json(uintptr_t function){HMODULE mod=nullptr;const bool ok=GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCSTR>(function),&mod)!=0;std::string out="{\"api\":\"GetModuleHandleExA(FROM_ADDRESS)\",\"host_address_only_not_device_module_resolution\":true,\"resolved\":"+std::string(ok?"true":"false");if(ok)out+=",\"path\":"+quote(module_path(mod).c_str());else out+=",\"win32_error\":"+std::to_string(GetLastError());return out+"}";}
static std::string attributes_json(const cudaFuncAttributes&a){std::ostringstream o;o<<"{\"api\":\"cudaFuncGetAttributes\",\"binaryVersion\":"<<a.binaryVersion<<",\"ptxVersion\":"<<a.ptxVersion<<",\"numRegs\":"<<a.numRegs<<",\"sharedSizeBytes\":"<<a.sharedSizeBytes<<",\"constSizeBytes\":"<<a.constSizeBytes<<",\"localSizeBytes\":"<<a.localSizeBytes<<",\"maxThreadsPerBlock\":"<<a.maxThreadsPerBlock<<",\"cacheModeCA\":"<<a.cacheModeCA<<",\"maxDynamicSharedSizeBytes\":"<<a.maxDynamicSharedSizeBytes<<",\"preferredShmemCarveout\":"<<a.preferredShmemCarveout<<",\"clusterDimMustBeSet\":"<<a.clusterDimMustBeSet<<",\"requiredClusterWidth\":"<<a.requiredClusterWidth<<",\"requiredClusterHeight\":"<<a.requiredClusterHeight<<",\"requiredClusterDepth\":"<<a.requiredClusterDepth<<"}";return o.str();}
struct NewFile {
 HANDLE handle=INVALID_HANDLE_VALUE;std::string path;
 explicit NewFile(std::string p):path(std::move(p)){handle=CreateFileA(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);if(handle==INVALID_HANDLE_VALUE)throw std::runtime_error("refuse existing/unwritable output: "+path);}
 void write(const void *data,size_t size){if(size>std::numeric_limits<DWORD>::max())throw std::runtime_error("bounded output too large");DWORD wrote=0;if(!WriteFile(handle,data,DWORD(size),&wrote,nullptr)||wrote!=size||!FlushFileBuffers(handle))throw std::runtime_error("output write failed: "+path);}
 ~NewFile(){if(handle!=INVALID_HANDLE_VALUE)CloseHandle(handle);}
 NewFile(const NewFile&)=delete;NewFile&operator=(const NewFile&)=delete;
};
struct OwnedDeviceBuffer {
 void *data=nullptr;
 explicit OwnedDeviceBuffer(size_t n){check_cuda(cudaMalloc(&data,n),"cudaMalloc owned buffer");}
 void release(){void *old=data;data=nullptr;if(old)check_cuda(cudaFree(old),"cudaFree owned buffer");}
 ~OwnedDeviceBuffer(){if(data)cudaFree(data);}
 OwnedDeviceBuffer(const OwnedDeviceBuffer&)=delete;OwnedDeviceBuffer&operator=(const OwnedDeviceBuffer&)=delete;
};
static void require(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
static int host_test(){
 ConversionArgs original;original.x=1;original.vy=2;original.ne00=K;original.s01=K;original.s02=M*K;original.s03=M*K;original.ne0=K;original.ne1=M;original.ne2=1;
 ReplayArgs launch_args(original,reinterpret_cast<const float*>(uintptr_t(1)),reinterpret_cast<void*>(uintptr_t(2)));
 cudaLaunchKernel_v7000_params launch{};launch.func=reinterpret_cast<const void*>(uintptr_t(0x12345));launch.args=launch_args.slots.data();launch.gridDim=dim3(64,2,1);launch.blockDim=dim3(128,1,1);launch.stream=reinterpret_cast<cudaStream_t>(uintptr_t(3));
 CUpti_CallbackData data{};data.callbackSite=CUPTI_API_ENTER;data.functionParams=&launch;data.symbolName=conversion_mangled;data.correlationId=77;data.contextUid=1;
 callback(nullptr,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);
 require(count==1&&!malformed_callback&&records[0].function==uintptr_t(0x12345)&&records[0].stream==3,"runtime func callback copy");validate_capture(records[0]);
 ReplayArgs substituted(records[0].arguments,reinterpret_cast<const float*>(uintptr_t(0xabc)),reinterpret_cast<void*>(uintptr_t(0xdef)));auto actual=substituted.decoded();
 require(actual.x==0xabc&&actual.vy==0xdef&&same_scalars(actual,original)&&original.x==1&&original.vy==2,"replace only x/vy with preserved scalar storage");
 Launch invalid=records[0];invalid.grid.y=3;bool rejected=false;try{validate_capture(invalid);}catch(const std::exception&){rejected=true;}require(rejected,"unexpected geometry must fail");
 invalid=records[0];invalid.arguments.s02++;rejected=false;try{validate_capture(invalid);}catch(const std::exception&){rejected=true;}require(rejected,"unexpected strides must fail");
 data.symbolName="unknown_quantize_mmq_q8_1_variant";launch.args=nullptr;callback(nullptr,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,&data);
 require(count==2&&!records[1].conversion&&!malformed_callback,"unknown symbol must not decode");
 std::vector<BlockQ8MMQD4> synthetic(M*(K/QK));
 for(int row=0;row<M;++row)for(int k=0;k<K;++k){auto&b=synthetic[block_index(row,k)];b.qs[k%QK]=int8_t((row+k)%255-127);if(k%32==0)b.d4[k%QK/32]=float(row*32+k/32+1)/2048.0f;}
 const auto *raw=reinterpret_cast<const uint8_t*>(synthetic.data());
 for(int row=0;row<M;++row)for(int k=0;k<K;++k){int8_t q;float d;std::memcpy(&q,raw+q_offset(row,k),1);std::memcpy(&d,raw+d_offset(row,k),4);require(q==int8_t((row+k)%255-127)&&d==float(row*32+k/32+1)/2048.0f,"D4 transposed layout decode");}
 require(q_offset(11,33)==1633&&d_offset(11,33)==1588&&output_bytes==73728&&input_bytes==262144,"target offset and size derivation");
 CuptiApi api;api.load();
 std::cout<<"{\"host_api_struct_test\":true,\"synthetic_callback_test\":true,\"runtime_func_is_from_cudaLaunchKernel_params\":true,\"only_x_vy_substitution_test\":true,\"unexpected_signature_geometry_rejected\":true,\"unknown_symbol_not_decoded\":true,\"D4_all_65536_cells_layout_test\":true,\"block_size\":"<<sizeof(BlockQ8MMQD4)<<",\"q_offset_row11_k33\":"<<q_offset(11,33)<<",\"d4_offset_row11_k33\":"<<d_offset(11,33)<<",\"input_bytes\":"<<input_bytes<<",\"output_bytes\":"<<output_bytes<<",\"GPU_context_created\":false,\"CUPTI_subscription_created\":false,\"compile_header_api_version\":"<<CUPTI_API_VERSION<<",\"runtime_api_version\":"<<api.runtime_api_version<<",\"runtime_library\":"<<quote(api.actual_path.c_str())<<",\"actual_GPU_replay_executed\":false}\n";
 return 0;
}
static int run(const char *output_path){
 NewFile result_file(output_path); // Reserve every artifact before any GPU API.
 CuptiApi cupti;CUpti_SubscriberHandle subscriber{};bool subscribed=false,callback_enabled=false;
 ggml_context *ctx=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t buffer=nullptr;
 int exitcode=0;std::string stage="reserve_outputs",evidence,result;
 try{
  const std::string base=output_path;
  NewFile q_file(base+".q8_1_d4.bin"),x_file(base+".input_f32.bin"),w_file(base+".weights_q8_0.bin"),g_file(base+".graph_f32.bin");
  stage="identity_and_subscription";
  require(std::getenv("GGML_CUDA_DISABLE_GRAPHS")&&std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))=="1","graphs must be disabled");
  evidence="\"schema\":\"original-dll-owned-conversion-replay/v1\",\"timed\":false,\"source_built_kernel\":false,\"original_DLL_rebuilt\":false,\"original_conversion_pool_read\":false,\"runtime_function_pointer_not_driver_CUfunction\":true,\"loaded_native_modules\":"+verify_native_modules();
  cupti.load();evidence+=",\"compile_header_api_version\":"+std::to_string(CUPTI_API_VERSION)+",\"runtime_api_version\":"+std::to_string(cupti.runtime_api_version)+",\"runtime_library\":"+quote(cupti.actual_path.c_str());
  check(cupti.subscribe(&subscriber,callback,nullptr));subscribed=true;
  stage="prepare_original_graph";
  ggml_init_params parameters{ggml_tensor_overhead()*8+ggml_graph_overhead_custom(8,false),nullptr,true};ctx=ggml_init(parameters);require(ctx!=nullptr,"context");
  auto *weights=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,K,N);auto *input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,M);
  auto *out=ggml_mul_mat(ctx,weights,input);auto *graph=ggml_new_graph_custom(ctx,8,false);ggml_build_forward_expand(graph,out);
  backend=ggml_backend_cuda_init(0);require(backend&&ggml_backend_is_cuda(backend),"CUDA backend unavailable");require(ggml_backend_supports_op(backend,out),"unsupported graph");
  std::vector<float>w(N*K),x(M*K),graph_output(M*N);uint32_t seed=20260914;for(auto &v:w)v=random_value(seed);for(auto &v:x)v=random_value(seed);
  std::vector<uint8_t>packed(ggml_row_size(GGML_TYPE_Q8_0,K)*N);auto bytes=ggml_quantize_chunk(GGML_TYPE_Q8_0,w.data(),packed.data(),0,N,K,nullptr);require(bytes==packed.size()&&bytes==size_t(N)*(K/32)*34,"packed Q8_0 size mismatch");
  buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer!=nullptr,"graph buffer");
  ggml_backend_tensor_set(weights,packed.data(),0,packed.size());ggml_backend_tensor_set(input,x.data(),0,input_bytes);ggml_backend_synchronize(backend);
  stage="capture_original_graph";
  check(cupti.enable_callback(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));callback_enabled=true;
  auto status=ggml_backend_graph_compute_async(backend,graph);ggml_backend_synchronize(backend);
  check(cupti.enable_callback(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));callback_enabled=false;
  check(cupti.unsubscribe(subscriber));subscribed=false;
  evidence+=",\"original_graph_status\":"+std::to_string(int(status))+",\"launches\":"+launches_json()+",\"callbacks_disabled_and_unsubscribed_before_replay\":true";
  require(status==GGML_STATUS_SUCCESS,"original graph failed");
  const unsigned callback_count=count.load();unsigned conversions=0;Launch captured{};
  for(unsigned i=0;i<std::min<unsigned>(callback_count,records.size());++i)if(records[i].conversion){captured=records[i];++conversions;}
  require(callback_count>0&&!overflow&&!malformed_callback&&conversions==1,"missing, ambiguous or malformed conversion capture");validate_capture(captured);
  stage="copy_persistent_graph_output";
  // This is the graph's still-owned F32 output tensor, never the conversion pool allocation.
  ggml_backend_tensor_get(out,graph_output.data(),0,graph_bytes);ggml_backend_synchronize(backend);
  x_file.write(x.data(),input_bytes);w_file.write(packed.data(),packed.size());g_file.write(graph_output.data(),graph_bytes);
  stage="query_captured_runtime_function";
  const void *runtime_func=reinterpret_cast<const void*>(captured.function);const cudaStream_t same_stream=reinterpret_cast<cudaStream_t>(captured.stream);
  int device=-1,runtime_version=0,driver_version=0;check_cuda(cudaGetDevice(&device),"cudaGetDevice");require(device==0,"original CUDA device changed");
  check_cuda(cudaRuntimeGetVersion(&runtime_version),"cudaRuntimeGetVersion");check_cuda(cudaDriverGetVersion(&driver_version),"cudaDriverGetVersion");
  evidence+=",\"selected_conversion_func\":"+hex(captured.function)+",\"runtime_func_host_module\":"+host_module_json(captured.function)+",\"cuda_device\":"+std::to_string(device)+",\"cuda_runtime_version\":"+std::to_string(runtime_version)+",\"cuda_driver_version\":"+std::to_string(driver_version);
  cudaFuncAttributes attrs{};check_cuda(cudaFuncGetAttributes(&attrs,runtime_func),"cudaFuncGetAttributes captured runtime func");evidence+=",\"function_attributes\":"+attributes_json(attrs);
  stage="allocate_owned_buffers";
  OwnedDeviceBuffer owned_x(input_bytes),owned_q(output_bytes);
  ReplayArgs replay(captured.arguments,static_cast<const float*>(owned_x.data),owned_q.data);auto replay_args=replay.decoded();
  require(same_scalars(captured.arguments,replay_args)&&replay_args.ids==0,"replay changed captured scalar arguments");
  evidence+=",\"replay\":{\"same_process_runtime_func\":"+hex(captured.function)+",\"same_stream\":"+hex(captured.stream)+",\"same_grid\":"+dimensions(captured.grid)+",\"same_block\":"+dimensions(captured.block)+",\"same_shared\":"+std::to_string(captured.shared)+",\"argument_slots_substituted\":[0,2],\"arguments\":"+args_json(replay_args)+",\"all_captured_scalars_preserved\":true,\"owned_input_bytes\":"+std::to_string(input_bytes)+",\"owned_output_bytes\":"+std::to_string(output_bytes)+",\"output_init_byte\":165}";
  check_cuda(cudaMemcpyAsync(owned_x.data,x.data(),input_bytes,cudaMemcpyHostToDevice,same_stream),"copy owned input");
  check_cuda(cudaMemsetAsync(owned_q.data,0xa5,output_bytes,same_stream),"initialize owned output");
  check_cuda(cudaStreamSynchronize(same_stream),"synchronize input and output initialization");
  stage="replay_same_original_runtime_func";
  // runtime_func is p.func from cudaLaunchKernel_v7000_params, not a driver handle.
  check_cuda(cudaLaunchKernel(runtime_func,captured.grid,captured.block,replay.slots.data(),captured.shared,same_stream),"cudaLaunchKernel same original runtime func");
  check_cuda(cudaStreamSynchronize(same_stream),"synchronize original-function replay");
  stage="copy_full_owned_output";
  std::vector<BlockQ8MMQD4> host_output(M*(K/QK));
  check_cuda(cudaMemcpyAsync(host_output.data(),owned_q.data,output_bytes,cudaMemcpyDeviceToHost,same_stream),"copy full owned D4 output");
  check_cuda(cudaStreamSynchronize(same_stream),"synchronize owned output copy");
  require(count.load()==callback_count&&!overflow&&!malformed_callback,"callback record count changed after disabled/unsubscribed replay");
  q_file.write(host_output.data(),output_bytes);
  const auto &target=host_output[block_index(11,33)];uint32_t scale_bits=0;std::memcpy(&scale_bits,&target.d4[1],4);
  std::ostringstream target_json;target_json<<std::setprecision(17)<<"{\"row\":11,\"k\":33,\"block_index\":"<<block_index(11,33)<<",\"q_byte_offset\":"<<q_offset(11,33)<<",\"scale_byte_offset\":"<<d_offset(11,33)<<",\"q\":"<<int(target.qs[33])<<",\"d4\":"<<real_json(target.d4[1])<<",\"d4_bits\":"<<scale_bits<<"}";
  evidence+=",\"target_row11_k33\":"+target_json.str()+",\"callback_count_before_replay\":"+std::to_string(callback_count)+",\"callback_count_after_replay\":"+std::to_string(count.load())+",\"layout\":{\"block_bytes\":144,\"d4_offset\":0,\"qs_offset\":16,\"blocks\":512,\"dimensions_M_N_K\":[64,896,1024],\"block_index_formula\":\"(k / 128) * M + row; z = 0\"},\"full_output_bytes_copied\":"+std::to_string(output_bytes);
  stage="release_owned_buffers";owned_q.release();owned_x.release();
  result="{"+evidence+",\"status\":\"replay_complete\",\"quantized_code_observed\":true,\"original_pool_quantized_code_observed\":false,\"owned_buffers_freed_before_backend_teardown\":true}";
 }catch(const std::exception &e){exitcode=4;result="{"+(evidence.empty()?"":evidence+",")+"\"status\":\"failed\",\"failed_stage\":"+quote(stage.c_str())+",\"error\":"+quote(e.what())+",\"quantized_code_observed\":false}";}
 if(subscribed){if(callback_enabled)cupti.enable_callback(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);cupti.unsubscribe(subscriber);}
 if(buffer)ggml_backend_buffer_free(buffer);if(backend)ggml_backend_free(backend);if(ctx)ggml_free(ctx);
 result_file.write(result.data(),result.size());return exitcode;
}
int main(int argc,char **argv){try{
 if(argc==2&&std::string(argv[1])=="--host-api-test")return host_test();
 if(argc!=3||std::string(argv[1])!="--run-replay-only"){std::cerr<<"Explicit --run-replay-only NEW_OUTPUT required. No GPU executed.\n";return 2;}
 return run(argv[2]);
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 12;}}
