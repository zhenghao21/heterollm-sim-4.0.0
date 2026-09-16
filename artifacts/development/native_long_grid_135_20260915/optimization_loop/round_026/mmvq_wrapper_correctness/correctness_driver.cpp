// This driver contains a single synthetic correctness case, never a timing loop.
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include "nlohmann/json.hpp"
#include "mmvq_probe_abi.h"       // Unchanged R6 shared declaration, no local redeclaration.
#include "cpu_reference.h"
#include "build_identity.h"

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
using ConvertABI=void (*)(const float*,void*,int,int,int,int,cudaStream_t);
using MainABI=void (*)(const void*,int,const void*,float*,int,int,int,int,cudaStream_t);
static_assert(std::is_same<decltype(&heterollm_mmvq_probe_convert_q8_1),ConvertABI>::value,"convert ABI changed");
static_assert(std::is_same<decltype(&heterollm_mmvq_probe_main),MainABI>::value,"main ABI changed");

std::string sha256(const std::uint8_t* data,std::size_t length) {
    BCRYPT_ALG_HANDLE alg=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    check::require(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"BCryptOpenAlgorithmProvider failed");
    auto cleanup=[&](){ if(hash)BCryptDestroyHash(hash); if(alg)BCryptCloseAlgorithmProvider(alg,0); };
    try {
        DWORD object_size=0,got=0;
        check::require(BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&got,0)>=0,"BCryptGetProperty failed");
        std::vector<std::uint8_t> object(object_size);
        check::require(BCryptCreateHash(alg,&hash,object.data(),object_size,nullptr,0,0)>=0,"BCryptCreateHash failed");
        for(std::size_t offset=0;offset<length;) {
            const auto chunk=static_cast<ULONG>(std::min<std::size_t>(length-offset,1U<<24));
            check::require(BCryptHashData(hash,const_cast<PUCHAR>(data+offset),chunk,0)>=0,"BCryptHashData failed"); offset+=chunk;
        }
        std::array<std::uint8_t,32> digest{};
        check::require(BCryptFinishHash(hash,digest.data(),32,0)>=0,"BCryptFinishHash failed");
        std::ostringstream text; text<<std::hex<<std::setfill('0');for(auto c:digest)text<<std::setw(2)<<int(c);
        cleanup();return text.str();
    } catch(...) { cleanup();throw; }
}
std::vector<std::uint8_t> file_bytes(const fs::path& p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);check::require(bool(f),"cannot read "+p.u8string());
    const auto n=f.tellg();check::require(n>=0,"negative file size");std::vector<std::uint8_t> b(static_cast<std::size_t>(n));f.seekg(0);
    if(!b.empty())f.read(reinterpret_cast<char*>(b.data()),n);check::require(bool(f),"short read "+p.u8string());return b;
}
std::string file_sha(const fs::path& p) { const auto b=file_bytes(p);return sha256(b.data(),b.size()); }
json file_ref(const fs::path& p) { return { {"path",p.u8string()},{"sha256",file_sha(p)},{"bytes",fs::file_size(p)} }; }
std::string utc_now() {
    SYSTEMTIME s{};GetSystemTime(&s);char text[64];
    std::snprintf(text,sizeof(text),"%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",s.wYear,s.wMonth,s.wDay,s.wHour,s.wMinute,s.wSecond,s.wMilliseconds);return text;
}
fs::path module_path(HMODULE module) {
    std::vector<wchar_t> name(32768);const auto n=GetModuleFileNameW(module,name.data(),DWORD(name.size()));
    check::require(n>0 && n<name.size(),"GetModuleFileNameW failed");return fs::path(std::wstring(name.data(),n));
}
class ExclusiveFile {
    HANDLE h=INVALID_HANDLE_VALUE;
public:
    explicit ExclusiveFile(const fs::path& p) {
        check::require(p.is_absolute(),"output must be an absolute path");
        h=CreateFileW(p.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        check::require(h!=INVALID_HANDLE_VALUE,"exclusive output creation failed: "+p.u8string()+" (Windows "+std::to_string(GetLastError())+")");
    }
    ExclusiveFile(const ExclusiveFile&)=delete;
    ~ExclusiveFile(){if(h!=INVALID_HANDLE_VALUE)CloseHandle(h);}
    void write(const void* source,std::size_t bytes) {
        LARGE_INTEGER zero{};check::require(SetFilePointerEx(h,zero,nullptr,FILE_BEGIN),"seek result failed");
        const auto* p=static_cast<const std::uint8_t*>(source);
        for(std::size_t done=0;done<bytes;){const DWORD chunk=DWORD(std::min<std::size_t>(bytes-done,1U<<24));DWORD written=0;
            check::require(WriteFile(h,p+done,chunk,&written,nullptr) && written==chunk,"write result failed");done+=written;}
        check::require(SetEndOfFile(h) && FlushFileBuffers(h),"flush result failed");
    }
    void record(const json& result) {const std::string s=result.dump(2)+"\n";write(s.data(),s.size());}
};
json save_raw(const fs::path& p,const void* data,std::size_t n) {ExclusiveFile f(p);f.write(data,n);return file_ref(p);}
json verify_identity() {
    const fs::path manifest=fs::u8path(BUILD_INPUTS_PATH);
    check::require(file_sha(manifest)==BUILD_INPUTS_SHA256,"build input manifest changed");
    const auto bytes=file_bytes(manifest);const auto inputs=json::parse(bytes.begin(),bytes.end());
    for(const auto& item:inputs.at("input_refs")) {
        const auto p=fs::u8path(item.at("path").get<std::string>());
        check::require(fs::file_size(p)==item.at("bytes").get<std::uintmax_t>() && file_sha(p)==item.at("sha256").get<std::string>(),"build/source input drift: "+p.u8string());
    }
    json identity={{"build_inputs",file_ref(manifest)},{"executable",file_ref(module_path(nullptr))},{"process_id",GetCurrentProcessId()},{"loaded_modules",json::array()}};
    for(auto it=inputs.at("expected_runtime_dlls").begin();it!=inputs.at("expected_runtime_dlls").end();++it) {
        const auto module=GetModuleHandleW(fs::u8path(it.key()).c_str());
        if(!module)continue;
        const auto p=module_path(module);const auto observed=file_ref(p);
        check::require(observed.at("sha256")==it.value().at("sha256"),"loaded runtime module differs from locked build: "+it.key());
        identity["loaded_modules"].push_back(observed);
    }
    const auto base=GetModuleHandleW(L"ggml-base.dll");check::require(base!=nullptr,"locked ggml-base was not loaded");
    check::require(file_sha(module_path(base))==inputs.at("expected_runtime_dlls").at("ggml-base.dll").at("sha256").get<std::string>(),"ggml_quantize_chunk provider identity mismatch");
    identity["ggml_quantize_chunk_provider"]=file_ref(module_path(base));
    identity["target_cuda_dll_is_not_this_wrapper"]=inputs.at("locked_target_cuda");
    return identity;
}
json comparison_json(const check::Compare& c) {
    return {{"tested",c.tested},{"failed",c.failed},{"max_absolute_error",std::isfinite(c.max_absolute_error)?json(c.max_absolute_error):json(nullptr)},
            {"max_bound_ratio",std::isfinite(c.max_bound_ratio)?json(c.max_bound_ratio):json(nullptr)},{"worst_index",c.worst_index}};
}
json cpu_self_test(const std::vector<float>& weights,const std::vector<float>& input,const std::vector<std::uint8_t>& q5,const std::vector<std::uint8_t>& q8,const check::Reference& ref) {
    check::require(std::all_of(weights.begin(),weights.end(),[](float x){return std::isnormal(x);}),"weights not normal/nonzero");
    check::require(std::all_of(input.begin(),input.end(),[](float x){return std::isnormal(x);}),"input not normal/nonzero");
    std::vector<float> expected(check::N);
    for(int i=0;i<check::N;++i)expected[i]=static_cast<float>(ref.dot[i]);
    const auto good=check::compare_output(expected,ref);check::require(good.failed==0,"rounded CPU reference failed preset bound");
    auto bad=expected;bad[31]=static_cast<float>(ref.dot[31]+2*ref.bound[31]+1);
    check::require(check::compare_output(bad,ref).failed==1,"out-of-bound corruption not rejected");
    bad=expected;bad[47]=std::numeric_limits<float>::quiet_NaN();check::require(check::compare_output(bad,ref).failed==1,"NaN output not rejected");
    auto qbad=q8;qbad[0]^=1;check::require(check::byte_mismatches(qbad,q8)==1,"Q8_1 scale corruption not rejected");
    qbad=q8;qbad[2]^=1;check::require(check::byte_mismatches(qbad,q8)==1,"Q8_1 sum corruption not rejected");
    qbad=q8;qbad[4]^=1;check::require(check::byte_mismatches(qbad,q8)==1,"Q8_1 quant corruption not rejected");
    const auto q5_second=check::quantize_weight(weights);check::require(q5==q5_second,"locked CPU quantizer not deterministic");
    const auto q8_second=check::expected_q8(input);check::require(q8==q8_second,"Q8_1 CPU reference not deterministic");
    return {{"status","passed"},{"all_input_elements_nonzero",true},{"all_weight_elements_nonzero",true},{"packed_weight_bytes",q5.size()},
        {"expected_q8_1_bytes",q8.size()},{"locked_quantizer_deterministic",true},{"output_bound_positive_fixture",comparison_json(good)},
        {"negative_checks",{"output_exceeds_derived_bound","nan_output","q8_scale_byte","q8_sum_byte","q8_quant_byte"}},
        {"reference_scope","Q5_0 M1 K4096 N3072 exact dyadic inputs; no rounding ties or padded columns"}};
}
#ifndef MMVQ_CPU_SELF_TEST_ONLY
void cuda_checked(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorName(status)+" / "+cudaGetErrorString(status));
}
#define CUDA_CHECK(expr) cuda_checked((expr),#expr)
struct DeviceBuffer {std::uint8_t* allocation=nullptr; std::size_t bytes=0; static constexpr std::size_t GUARD=256;
    void allocate(std::size_t n,cudaStream_t stream){bytes=n;CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&allocation),n+2*GUARD));CUDA_CHECK(cudaMemsetAsync(allocation,0xa5,n+2*GUARD,stream));}
    void* data(){return allocation+GUARD;}
    void check_guard(cudaStream_t stream){std::array<std::uint8_t,2*GUARD> b{};
        CUDA_CHECK(cudaMemcpyAsync(b.data(),allocation,GUARD,cudaMemcpyDeviceToHost,stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data()+GUARD,allocation+GUARD+bytes,GUARD,cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        check::require(std::all_of(b.begin(),b.end(),[](auto v){return v==0xa5;}),"CUDA allocation guard was overwritten");}
};
void gpu_correctness(json& result,ExclusiveFile& journal,const fs::path& output,const std::vector<float>& input,const std::vector<std::uint8_t>& q5,const std::vector<std::uint8_t>& expected_q8,const check::Reference& preflight_ref) {
    cudaStream_t stream=nullptr;DeviceBuffer dw,dx,dq,dy;
    std::vector<std::string> failures;
    bool main_entered=false;
    try {
        int runtime=0,driver=0,count=0;
        result["cuda_api_calls_started"]=true;journal.record(result);
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));CUDA_CHECK(cudaDriverGetVersion(&driver));CUDA_CHECK(cudaGetDeviceCount(&count));
        check::require(count>0,"no CUDA device");CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));char bus[64]{};CUDA_CHECK(cudaDeviceGetPCIBusId(bus,sizeof(bus),0));
        std::ostringstream uuid;uuid<<std::hex<<std::setfill('0');for(auto c:p.uuid.bytes)uuid<<std::setw(2)<<int(static_cast<unsigned char>(c));
        result["device"]={{"index",0},{"name",p.name},{"uuid_hex",uuid.str()},{"pci_bus_id",bus},{"compute_major",p.major},{"compute_minor",p.minor},{"sm_count",p.multiProcessorCount},
            {"warp_size",p.warpSize},{"total_global_memory",p.totalGlobalMem},{"l2_cache_bytes",p.l2CacheSize},{"memory_bus_width_bits",p.memoryBusWidth},{"core_clock_khz",p.clockRate},{"memory_clock_khz",p.memoryClockRate},
            {"cuda_runtime_version",runtime},{"cuda_driver_version",driver},{"runtime_header_version",CUDART_VERSION}};
        if(auto nvcuda=GetModuleHandleW(L"nvcuda.dll"))result["cuda_driver_module"]=file_ref(module_path(nvcuda));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        result["stream"]={{"kind","explicit_nonblocking"},{"host_handle_hex",[&](){std::ostringstream s;s<<std::hex<<reinterpret_cast<std::uintptr_t>(stream);return s.str();}()}};
        dw.allocate(q5.size(),stream);dx.allocate(input.size()*sizeof(float),stream);dq.allocate(expected_q8.size(),stream);dy.allocate(check::N*sizeof(float),stream);
        CUDA_CHECK(cudaMemcpyAsync(dw.data(),q5.data(),q5.size(),cudaMemcpyHostToDevice,stream));CUDA_CHECK(cudaMemcpyAsync(dx.data(),input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        std::vector<float> actual(check::N,std::numeric_limits<float>::quiet_NaN());
        CUDA_CHECK(cudaMemcpyAsync(dy.data(),actual.data(),actual.size()*sizeof(float),cudaMemcpyHostToDevice,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        result["stage"]="conversion_started";result["gpu_execution_performed"]=true;journal.record(result);
        heterollm_mmvq_probe_convert_q8_1(static_cast<const float*>(dx.data()),dq.data(),int(GGML_TYPE_Q5_0),check::K,check::M,check::PADDED_K,stream);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint8_t> actual_q8(expected_q8.size());CUDA_CHECK(cudaMemcpyAsync(actual_q8.data(),dq.data(),actual_q8.size(),cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        result["raw_artifacts"]["actual_q8_1"]=save_raw(fs::path(output.wstring()+L".actual.q8_1.bin"),actual_q8.data(),actual_q8.size());
        const auto bad=check::byte_mismatches(actual_q8,expected_q8);result["conversion"]={{"bytes_compared",expected_q8.size()},{"byte_mismatches",bad},{"acceptance","all bytes exactly equal to independently constructed CPU Q8_1 layout"}};
        check::require(bad==0,"Q8_1 conversion/layout mismatch");dq.check_guard(stream);dx.check_guard(stream);
        // Reference is recomputed from the returned Q8_1 bytes, after strict conversion validation.
        const auto r=check::reference(q5,actual_q8);check::require(r.dot==preflight_ref.dot && r.bound==preflight_ref.bound,"post-conversion reference changed");
        result["stage"]="main_started";journal.record(result);main_entered=true;
        heterollm_mmvq_probe_main(dw.data(),int(GGML_TYPE_Q5_0),dq.data(),static_cast<float*>(dy.data()),check::K,check::N,check::M,check::PADDED_K/check::BLOCK,stream);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(stream));CUDA_CHECK(cudaMemcpyAsync(actual.data(),dy.data(),actual.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        result["raw_artifacts"]["actual_output_f32"]=save_raw(fs::path(output.wstring()+L".actual.output.f32.bin"),actual.data(),actual.size()*sizeof(float));
        const auto check_result=check::compare_output(actual,r);result["main_output"]=comparison_json(check_result);
        check::require(check_result.failed==0,"MMVQ output exceeded predeclared forward-error bound");
        dw.check_guard(stream);dx.check_guard(stream);dq.check_guard(stream);dy.check_guard(stream);result["allocation_guards_intact"]=true;
    } catch(const std::exception& e) {failures.push_back(e.what());}
    for(auto* b:{&dy,&dq,&dx,&dw})if(b->allocation){const auto status=cudaFree(b->allocation);if(status!=cudaSuccess)failures.push_back(std::string("cudaFree: ")+cudaGetErrorString(status));b->allocation=nullptr;}
    if(stream){const auto status=cudaStreamDestroy(stream);if(status!=cudaSuccess)failures.push_back(std::string("cudaStreamDestroy: ")+cudaGetErrorString(status));}
    result["main_shim_called"]=main_entered;result["cuda_errors_or_validation_failures"]=failures;
    check::require(failures.empty(),failures.empty()?"":failures.front());
    result["stage"]="conversion_and_main_correctness_passed";
}
#endif
int wmain(int argc,wchar_t** argv) {
    std::unique_ptr<ExclusiveFile> journal;json result;bool complete=false;
    try {
        check::require(argc==4 && std::wstring(argv[2])==L"--output","usage: executable (--self-test | --run-correctness) --output ABSOLUTE_NEW_JSON");
#ifdef MMVQ_CPU_SELF_TEST_ONLY
        check::require(std::wstring(argv[1])==L"--self-test","CPU executable supports only --self-test");const bool cpu_only=true;
#else
        check::require(std::wstring(argv[1])==L"--run-correctness","GPU executable requires explicit --run-correctness; not executed during build/test");const bool cpu_only=false;
#endif
        const fs::path output=fs::path(argv[3]);journal=std::make_unique<ExclusiveFile>(output);
        result={{"schema","heterollm.mmvq-wrapper-correctness/v1"},{"status","started_not_passed"},{"started_at_utc",utc_now()},{"stage","identity_check"},{"cpu_only",cpu_only},
                {"gpu_execution_performed",false},{"cuda_api_calls_started",false},{"timed_runs",0},{"target_llm_latency_used",false},{"performance_parameters_admitted",0},
                {"fixture",{{"weight_type","Q5_0"},{"ggml_type_id",int(GGML_TYPE_Q5_0)},{"K",check::K},{"N",check::N},{"M",check::M},{"padded_K",check::PADDED_K},{"q8_1_stride_blocks",check::PADDED_K/check::BLOCK},{"seed",check::SEED}}}};
        journal->record(result);
        if(cpu_only) {
            check::require(!GetModuleHandleW(L"cudart64_12.dll") && !GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"ggml-cuda.dll"),"CPU-only binary unexpectedly loaded a CUDA module");
            result["cpu_self_test_cuda_modules_absent"]=true;
        }
        result["identity"]=verify_identity();
        const auto weights=check::make_weights();const auto input=check::make_input();
        const auto q5=check::quantize_weight(weights);const auto q8=check::expected_q8(input);const auto ref=check::reference(q5,q8);
        result["cpu_self_test"]=cpu_self_test(weights,input,q5,q8,ref);
        result["tolerance"]={{"rule","gamma(2*(K/32)+3,2^-24)*unsigned_amplitude/(1-gamma64) + gamma64*sum_abs_products/(1-gamma64); rounded upward"},
            {"gamma64_rule","gamma(4*K+64,2^-53)"},{"fp32_rounding_steps",check::FP32_ROUNDING_STEPS},{"gamma32",check::gamma(check::FP32_ROUNDING_STEPS,std::ldexp(1.0,-24))},{"gamma64",check::gamma(4*check::K+64,std::ldexp(1.0,-53))},
            {"min_bound",*std::min_element(ref.bound.begin(),ref.bound.end())},{"max_bound",*std::max_element(ref.bound.begin(),ref.bound.end())},
            {"selected_before_observing_gpu_output",true},{"quantization_error_against_original_float_weights_not_an_acceptance_metric",true}};
        result["fixture_hashes"]={{"weights_f32",sha256(reinterpret_cast<const std::uint8_t*>(weights.data()),weights.size()*sizeof(float))},
            {"input_f32",sha256(reinterpret_cast<const std::uint8_t*>(input.data()),input.size()*sizeof(float))},{"packed_q5_0",sha256(q5.data(),q5.size())},{"expected_q8_1",sha256(q8.data(),q8.size())},
            {"reference_f64",sha256(reinterpret_cast<const std::uint8_t*>(ref.dot.data()),ref.dot.size()*sizeof(double))},{"bounds_f64",sha256(reinterpret_cast<const std::uint8_t*>(ref.bound.data()),ref.bound.size()*sizeof(double))}};
        if(!cpu_only) {
            result["raw_artifacts"]={{"packed_weight_q5_0",save_raw(fs::path(output.wstring()+L".weights.q5_0.bin"),q5.data(),q5.size())},
                {"input_f32",save_raw(fs::path(output.wstring()+L".input.f32.bin"),input.data(),input.size()*sizeof(float))},
                {"expected_q8_1",save_raw(fs::path(output.wstring()+L".expected.q8_1.bin"),q8.data(),q8.size())},
                {"reference_f64",save_raw(fs::path(output.wstring()+L".reference.f64.bin"),ref.dot.data(),ref.dot.size()*sizeof(double))},
                {"bounds_f64",save_raw(fs::path(output.wstring()+L".bounds.f64.bin"),ref.bound.data(),ref.bound.size()*sizeof(double))}};
#ifndef MMVQ_CPU_SELF_TEST_ONLY
            gpu_correctness(result,*journal,output,input,q5,q8,ref);
#endif
        }
        result["status"]=cpu_only?"cpu_reference_self_test_passed_gpu_unverified":"synthetic_correctness_passed_runtime_equivalence_unverified";
        result["runtime_equivalence_verified"]=false;complete=true;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";result["status"]="failed";result["failure"]=e.what();}
    result["finished_at_utc"]=utc_now();
    if(journal){try{journal->record(result);}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 3;}}
    return complete?0:2;
}
