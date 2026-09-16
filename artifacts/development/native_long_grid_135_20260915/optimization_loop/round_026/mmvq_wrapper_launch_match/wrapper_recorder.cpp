// Adapted from R16 conversion_capture/launch_recorder.cpp.
// Identity recorder only: no timing, no device-pointer dereference in callbacks.
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_026/mmvq_target_capture_ex/launch_decode.h"
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_026/mmvq_target_capture_ex/launch_metadata_json.h"
#include "wrapper_identity.h"
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_026/mmvq_wrapper_correctness/cpu_reference.h"
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/artifacts/development/native_long_grid_135_20260915/optimization_loop/round_024/mmvq_device_probe/r6_shared_abi/mmvq_probe_abi.h"

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace mmvq_capture;
struct CuptiApi {
    HMODULE library = nullptr;
    decltype(&cuptiGetVersion) get_version = nullptr;
    decltype(&cuptiSubscribe) subscribe = nullptr;
    decltype(&cuptiEnableDomain) enable_domain = nullptr;
    decltype(&cuptiUnsubscribe) unsubscribe = nullptr;
    uint32_t runtime_api_version = 0;
    std::string actual_path;
    template<class T> T symbol(const char * name) {
        auto address = GetProcAddress(library, name);
        if (!address) throw std::runtime_error(std::string("CUPTI export missing: ") + name);
        return reinterpret_cast<T>(address);
    }
    void load() {
        const char * path = std::getenv("CAPTURE_CUPTI_DLL");
        if (!path || std::strlen(path) < 4 || path[1] != ':' || (path[2] != '\\' && path[2] != '/'))
            throw std::runtime_error("absolute pinned CUPTI DLL required");
        library = LoadLibraryExA(path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!library) throw std::runtime_error("CUPTI DLL loading failed");
        char loaded[32768]{};
        if (!GetModuleFileNameA(library, loaded, sizeof(loaded))) throw std::runtime_error("CUPTI loaded path missing");
        actual_path = loaded; std::string expected = path; std::replace(expected.begin(), expected.end(), '/', '\\');
        if (_stricmp(expected.c_str(), actual_path.c_str())) throw std::runtime_error("CUPTI resolved path differs");
        get_version = symbol<decltype(get_version)>("cuptiGetVersion");
        subscribe = symbol<decltype(subscribe)>("cuptiSubscribe");
        enable_domain = symbol<decltype(enable_domain)>("cuptiEnableDomain");
        unsubscribe = symbol<decltype(unsubscribe)>("cuptiUnsubscribe");
        if (get_version(&runtime_api_version) != CUPTI_SUCCESS || runtime_api_version != 130401)
            throw std::runtime_error("selected CUPTI API version mismatch");
    }
    ~CuptiApi() { if (library) FreeLibrary(library); }
};
static std::string quote(const std::string & value) {
    std::ostringstream s; s << '"';
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') s << '\\' << char(c);
        else if (c < 32) s << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else s << char(c);
    }
    s << '"'; return s.str();
}
static std::string hash_file(const std::string & path) {
    BCRYPT_ALG_HANDLE algorithm{}; BCRYPT_HASH_HANDLE hash{};
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("identity file unavailable");
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        throw std::runtime_error("SHA256 provider unavailable");
    try {
        if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) throw std::runtime_error("SHA256 creation failed");
        std::array<unsigned char, 65536> buffer{}; std::array<unsigned char, 32> digest{};
        while (input) {
            input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
            if (input.gcount() && BCryptHashData(hash, buffer.data(), ULONG(input.gcount()), 0) < 0)
                throw std::runtime_error("SHA256 update failed");
        }
        if (!input.eof()) throw std::runtime_error("identity read failed");
        if (BCryptFinishHash(hash, digest.data(), digest.size(), 0) < 0) throw std::runtime_error("SHA256 finish failed");
        BCryptDestroyHash(hash); hash = nullptr; BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr;
        std::ostringstream s; s << std::hex << std::setfill('0'); for (auto x : digest) s << std::setw(2) << unsigned(x);
        return s.str();
    } catch (...) { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); throw; }
}
static std::string verify_modules() {
    if(GetModuleHandleA("ggml-cuda.dll"))throw std::runtime_error("original backend DLL must not be loaded by standalone wrapper");
    std::ostringstream out; out << '['; bool first = true;
    for (const auto & item : locked_identity::modules) {
        HMODULE mod = GetModuleHandleA(item.name);
        char loaded[32768]{};
        if(!mod&&!item.required)continue;
        if (!mod || !GetModuleFileNameA(mod, loaded, sizeof(loaded))) throw std::runtime_error("locked module not loaded");
        if (_stricmp(loaded, item.path)) throw std::runtime_error(std::string("loaded module path differs: ") + item.name);
        const std::string digest = hash_file(loaded);
        if (digest != item.sha256) throw std::runtime_error(std::string("loaded module full SHA differs: ") + item.name);
        if (!first) out << ','; first=false;
        out << "{\"name\":" << quote(item.name) << ",\"path\":" << quote(loaded) << ",\"sha256\":" << quote(digest) << '}';
    }
    out << ']'; return out.str();
}
static void cupti_checked(CUptiResult status) { if (status != CUPTI_SUCCESS) throw std::runtime_error("CUPTI failure " + std::to_string(int(status))); }
static float random_value(uint32_t & state) { state ^= state<<13; state ^= state>>17; state ^= state<<5; return float(int32_t(state&65535)-32768)/65536.0f; }
static std::string vector_json(uint3 v) { return '[' + std::to_string(v.x) + ',' + std::to_string(v.y) + ',' + std::to_string(v.z) + ']'; }
static std::string vector_json(dim3 v) { return '[' + std::to_string(v.x) + ',' + std::to_string(v.y) + ',' + std::to_string(v.z) + ']'; }
static std::string uuid_string(const cudaUUID_t & uuid) {
    std::ostringstream s; s << "GPU-" << std::hex << std::setfill('0');
    for (unsigned i=0; i<16; ++i) {
        if(i==4 || i==6 || i==8 || i==10) s << '-';
        s << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return s.str();
}
static std::string finite_float(float value) {
    if(!std::isfinite(value)) return "null";
    std::ostringstream s; s << std::setprecision(9) << value; return s.str();
}
static std::string launches_json(const Recorder & rec) {
    std::ostringstream s; s << '[';
    const unsigned count = (std::min)(rec.count.load(), unsigned(rec.records.size()));
    for (unsigned i=0; i<count; ++i) {
        if(i) s<<','; const auto & r=rec.records[i];
        s << "{\"index\":" << i << ",\"api_id\":" << r.api_id << ",\"api_name\":" << quote(r.api_name)
          << ",\"symbol\":" << quote(r.symbol) << ",\"correlation\":" << r.correlation << ",\"context\":" << r.context
          << ',' << launch_metadata_json(r)
          << ",\"supported_api\":" << (r.supported_api?"true":"false") << ",\"decoded\":" << (r.decoded?"true":"false")
          << ",\"exit_seen\":" << (r.exit_seen?"true":"false") << ",\"return_code\":" << r.return_code;
        if(r.is_conversion && r.decoded) {
            const auto & a=r.conversion;
            s << ",\"phase\":\"q8_1_conversion\",\"arguments\":{\"x\":" << a.x << ",\"vy\":" << a.vy
              << ",\"ne00\":" << a.ne00 << ",\"s01\":" << a.s01 << ",\"s02\":" << a.s02 << ",\"s03\":" << a.s03
              << ",\"ne0\":" << a.ne0 << ",\"ne1\":" << a.ne1 << ",\"ne2_fastdiv\":" << vector_json(a.ne2) << '}';
        } else if(r.is_main && r.decoded) {
            const auto & a=r.main; const auto & f=a.fusion;
            s << ",\"phase\":\"mmvq_main\",\"arguments\":{\"vx\":" << a.vx << ",\"vy\":" << a.vy << ",\"ids\":" << a.ids
              << ",\"fusion\":{\"x_bias\":" << uintptr_t(f.x_bias) << ",\"gate\":" << uintptr_t(f.gate)
              << ",\"gate_bias\":" << uintptr_t(f.gate_bias) << ",\"x_scale\":" << uintptr_t(f.x_scale)
              << ",\"gate_scale\":" << uintptr_t(f.gate_scale) << ",\"glu_op\":" << int(f.glu_op)
              << ",\"glu_limit\":" << finite_float(f.glu_limit) << "},\"dst\":" << a.dst << ",\"ncols_x\":" << a.ncols_x
              << ",\"nchannels_y_fastdiv\":" << vector_json(a.nchannels_y) << ",\"stride_row_x\":" << a.stride_row_x
              << ",\"stride_col_y\":" << a.stride_col_y << ",\"stride_col_dst\":" << a.stride_col_dst
              << ",\"channel_ratio_fastdiv\":" << vector_json(a.channel_ratio) << ",\"stride_channel_x\":" << a.stride_channel_x
              << ",\"stride_channel_y\":" << a.stride_channel_y << ",\"stride_channel_dst\":" << a.stride_channel_dst
              << ",\"sample_ratio_fastdiv\":" << vector_json(a.sample_ratio) << ",\"stride_sample_x\":" << a.stride_sample_x
              << ",\"stride_sample_y\":" << a.stride_sample_y << ",\"stride_sample_dst\":" << a.stride_sample_dst
              << ",\"ids_stride\":" << a.ids_stride << '}';
        }
        s << '}';
    }
    s << ']'; return s.str();
}
static std::string finite_double(double value) {
    if(!std::isfinite(value))return "null";
    std::ostringstream s;s<<std::setprecision(17)<<value;return s.str();
}
static std::string save_binary(const std::string & path,const void * data,size_t size) {
    HANDLE f=CreateFileA(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(f==INVALID_HANDLE_VALUE)throw std::runtime_error("numerical output exists/unwritable");
    DWORD wrote=0;const bool ok=WriteFile(f,data,DWORD(size),&wrote,nullptr)!=0;CloseHandle(f);
    if(!ok||wrote!=size)throw std::runtime_error("numerical output short write");
    return "{\"path\":"+quote(path)+",\"bytes\":"+std::to_string(size)+",\"sha256\":"+quote(hash_file(path))+"}";
}
static std::string runtime_json(const Recorder & rec) {
    std::ostringstream s; s << '[';
    const unsigned count=(std::min)(rec.runtime_count.load(),unsigned(rec.runtime_calls.size()));
    for(unsigned i=0;i<count;++i) {
        if(i)s<<',';const auto &r=rec.runtime_calls[i];
        const char *kind=r.classification==RuntimeClass::MemoryCopyOrSet?"memory_copy_or_set":
                         r.classification==RuntimeClass::Allocation?"allocation":"synchronization";
        s<<"{\"api_name\":"<<quote(r.api_name)<<",\"classification\":"<<quote(kind)
         <<",\"api_id\":"<<r.api_id<<",\"correlation\":"<<r.correlation<<",\"context\":"<<r.context
         <<",\"exit_seen\":"<<(r.exit_seen?"true":"false")<<",\"return_code\":"<<r.return_code<<'}';
    }
    s<<']';return s.str();
}
int main(int argc, char ** argv) {
    // There is deliberately no implicit run, benchmark mode or host test here.
    // Host tests use a separate executable with zero GPU/CUPTI DLL imports.
    if(argc!=3 || std::string(argv[1])!="--run-recorder-only" ||
       !std::getenv("CAPTURE_RUNTIME_AUTHORIZED") || std::string(std::getenv("CAPTURE_RUNTIME_AUTHORIZED"))!="1") {
        std::cerr<<"No GPU executed. Explicit authorized --run-recorder-only NEW_OUTPUT required.\n";return 2;
    }
    HANDLE output=CreateFileA(argv[2],GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(output==INVALID_HANDLE_VALUE) {std::cerr<<"Refuse existing/unwritable output\n";return 2;}
    Recorder rec; CuptiApi api; CUpti_SubscriberHandle subscriber{}; bool subscribed=false, enabled=false;
    cudaStream_t stream=nullptr;void *dw=nullptr,*dx=nullptr,*dq=nullptr,*dy=nullptr;
    TensorPointers tensors{}; bool executed=false, valid=false, path_qualified=false, numeric_qualified=false; int exitcode=4;
    std::string reason="not_run", loaded_before="[]", loaded_after="[]", hardware="null", numerical="null";
    try {
        if(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") || std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))!="1")
            throw std::runtime_error("CUDA graphs must be disabled");
        loaded_before=verify_modules(); api.load();
        if(hash_file(api.actual_path)!=locked_identity::cupti_sha256) throw std::runtime_error("CUPTI full SHA differs");
        cupti_checked(api.subscribe(&subscriber,callback,&rec)); subscribed=true;
        // Predeclare independent dyadic numerical fixture and error bound before
        // either GPU pass. The captured path still uses the original target RNG.
        const auto num_weights=check::make_weights();const auto num_input=check::make_input();
        const auto num_q5=check::quantize_weight(num_weights);const auto num_q8=check::expected_q8(num_input);
        const auto num_reference=check::reference(num_q5,num_q8);
        cudaDeviceProp prop{};
        if(cudaGetDeviceProperties(&prop,0)!=cudaSuccess || prop.major!=12 || prop.minor!=0 || prop.warpSize!=32 ||
           prop.multiProcessorCount!=84 || uuid_string(prop.uuid)!="GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b")
            throw std::runtime_error("source-qualified GPU UUID/sm_120/84-SM/warp32 identity required");
        hardware="{\"uuid\":"+quote(uuid_string(prop.uuid))+",\"major\":"+std::to_string(prop.major)+
                 ",\"minor\":"+std::to_string(prop.minor)+",\"SM_count\":"+std::to_string(prop.multiProcessorCount)+
                 ",\"warp_size\":"+std::to_string(prop.warpSize)+"}";
        const auto cuda_ok=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error("CUDA status "+std::to_string(int(status)));};
        std::vector<float> weights(size_t(kK)*kN), input(size_t(kK)*kM); uint32_t seed=20260916;
        for(auto &v:weights)v=random_value(seed);for(auto &v:input)v=random_value(seed);
        std::vector<unsigned char> packed(ggml_row_size(GGML_TYPE_Q5_0,kK)*kN);
        if(ggml_quantize_chunk(GGML_TYPE_Q5_0,weights.data(),packed.data(),0,kN,kK,nullptr)!=packed.size())throw std::runtime_error("quantized size mismatch");
        cuda_ok(cudaSetDevice(0));cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cuda_ok(cudaMalloc(&dw,packed.size()));cuda_ok(cudaMalloc(&dx,input.size()*sizeof(float)));
        cuda_ok(cudaMalloc(&dq,ggml_row_size(GGML_TYPE_Q8_1,kK)*kM));cuda_ok(cudaMalloc(&dy,size_t(kN)*kM*sizeof(float)));
        cuda_ok(cudaMemcpyAsync(dw,packed.data(),packed.size(),cudaMemcpyHostToDevice,stream));
        cuda_ok(cudaMemcpyAsync(dx,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        cuda_ok(cudaStreamSynchronize(stream));tensors={uintptr_t(dw),uintptr_t(dx),uintptr_t(dy)};
        cupti_checked(api.enable_domain(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API));enabled=true;executed=true;
        // Single capture pass: adjacent original-source conversion and main.
        // No event, memory copy, host correctness check or synchronization between.
        heterollm_mmvq_probe_convert_q8_1(static_cast<const float*>(dx),dq,int(GGML_TYPE_Q5_0),kK,kM,kK,stream);
        heterollm_mmvq_probe_main(dw,int(GGML_TYPE_Q5_0),dq,static_cast<float*>(dy),kK,kN,kM,kK/32,stream);
        cuda_ok(cudaStreamSynchronize(stream));
        cupti_checked(api.enable_domain(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API));enabled=false;
        path_qualified=validate(rec,tensors,reason);
        if(!path_qualified)throw std::runtime_error("captured pair refused: "+reason);
        // A separate adjacent pass requalifies the changed host shim numerically.
        // All input transfers precede it; all observation copies follow it. The
        // CUPTI capture is already closed, and no times are collected.
        cuda_ok(cudaMemcpyAsync(dw,num_q5.data(),num_q5.size(),cudaMemcpyHostToDevice,stream));
        cuda_ok(cudaMemcpyAsync(dx,num_input.data(),num_input.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        std::vector<float> numeric_actual(kN,std::numeric_limits<float>::quiet_NaN());
        cuda_ok(cudaMemcpyAsync(dy,numeric_actual.data(),numeric_actual.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
        cuda_ok(cudaStreamSynchronize(stream));
        heterollm_mmvq_probe_convert_q8_1(static_cast<const float*>(dx),dq,int(GGML_TYPE_Q5_0),kK,kM,kK,stream);
        heterollm_mmvq_probe_main(dw,int(GGML_TYPE_Q5_0),dq,static_cast<float*>(dy),kK,kN,kM,kK/32,stream);
        cuda_ok(cudaStreamSynchronize(stream));
        std::vector<std::uint8_t> numeric_q8(num_q8.size());
        cuda_ok(cudaMemcpyAsync(numeric_q8.data(),dq,numeric_q8.size(),cudaMemcpyDeviceToHost,stream));
        cuda_ok(cudaMemcpyAsync(numeric_actual.data(),dy,numeric_actual.size()*sizeof(float),cudaMemcpyDeviceToHost,stream));
        cuda_ok(cudaStreamSynchronize(stream));
        const auto byte_errors=check::byte_mismatches(numeric_q8,num_q8);
        const auto output_check=check::compare_output(numeric_actual,num_reference);
        std::ostringstream numeric;numeric<<std::setprecision(17);
        numeric<<"{\"separate_post_capture_pass\":true,\"same_new_shim_used\":true,\"capture_includes_numerical_pass\":false,"
               <<"\"fixture\":\"nonzero_dyadic_Q5_0_M1_K4096_N3072\",\"reference_fixed_before_GPU\":true,"
               <<"\"q8_bytes_compared\":"<<num_q8.size()<<",\"q8_byte_errors\":"<<byte_errors
               <<",\"outputs_compared\":"<<output_check.tested<<",\"output_failures\":"<<output_check.failed
               <<",\"max_absolute_error\":"<<finite_double(output_check.max_absolute_error)<<",\"max_bound_ratio\":"<<finite_double(output_check.max_bound_ratio)
               <<",\"raw_refs\":{\"actual_q8\":"<<save_binary(std::string(argv[2])+".actual.q8.bin",numeric_q8.data(),numeric_q8.size())
               <<",\"expected_q8\":"<<save_binary(std::string(argv[2])+".expected.q8.bin",num_q8.data(),num_q8.size())
               <<",\"actual_output\":"<<save_binary(std::string(argv[2])+".actual.f32.bin",numeric_actual.data(),numeric_actual.size()*sizeof(float))
               <<",\"reference_output\":"<<save_binary(std::string(argv[2])+".reference.f64.bin",num_reference.dot.data(),num_reference.dot.size()*sizeof(double))
               <<",\"bounds\":"<<save_binary(std::string(argv[2])+".bounds.f64.bin",num_reference.bound.data(),num_reference.bound.size()*sizeof(double))<<"}}";
        numerical=numeric.str();numeric_qualified=byte_errors==0&&output_check.failed==0;
        if(!numeric_qualified)throw std::runtime_error("new shim numerical qualification failed");
        loaded_after=verify_modules();
        if(hash_file(api.actual_path)!=locked_identity::cupti_sha256) throw std::runtime_error("CUPTI changed during capture");
        valid=path_qualified&&numeric_qualified;exitcode=valid?0:4;
    } catch(const std::exception & e) {reason=e.what();}
    if(enabled) api.enable_domain(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API);
    if(subscribed) api.unsubscribe(subscriber);
    if(dw)cudaFree(dw);if(dx)cudaFree(dx);if(dq)cudaFree(dq);if(dy)cudaFree(dy);if(stream)cudaStreamDestroy(stream);
    std::ostringstream result;
    result<<"{\"schema\":\"heterollm.mmvq-wrapper-launch-capture/v1\",\"status\":"<<quote(valid?"qualified_runtime_pair":"rejected")
          <<",\"reason\":"<<quote(reason)<<",\"timed\":false,\"performance_parameter_admitted\":false,\"target_LLM_latency_used\":false"
          <<",\"origin\":\"standalone_source_wrapper\",\"gpu_graph_executed\":false,\"gpu_launch_pair_executed\":"<<(executed?"true":"false")<<",\"device_pointer_dereferenced_in_callback\":false"
          <<",\"compile_CUPTI_API_version\":"<<CUPTI_API_VERSION<<",\"runtime_CUPTI_API_version\":"<<api.runtime_api_version
          <<",\"CUPTI_path\":"<<quote(api.actual_path)<<",\"loaded_modules_before\":"<<loaded_before<<",\"loaded_modules_after\":"<<loaded_after
          <<",\"numerical_check\":"<<numerical<<",\"new_shim_numerical_qualified\":"<<(numeric_qualified?"true":"false")
          <<",\"launch_pair_qualified\":"<<(path_qualified?"true":"false")<<",\"hardware\":"<<hardware<<",\"configuration\":{\"format\":\"Q5_0\",\"type_id\":6,\"M\":1,\"K\":4096,\"N\":3072,\"seed\":20260916,\"ids\":false,\"fusion\":false,\"graph_replay\":false}"
          <<",\"graph_tensors\":{\"weights\":"<<tensors.weights<<",\"input\":"<<tensors.input<<",\"output\":"<<tensors.output<<'}'
          <<",\"overflow\":"<<(rec.overflow?"true":"false")<<",\"malformed_callback\":"<<(rec.malformed_callback?"true":"false")
          <<",\"memory_api_count\":"<<rec.memory_api_count<<",\"runtime_auxiliary_count\":"<<rec.runtime_count
          <<",\"runtime_auxiliary_calls\":"<<runtime_json(rec)<<",\"launch_count\":"<<rec.count<<",\"launches\":"<<launches_json(rec)
          <<",\"pointer_associations\":{\"verified\":"<<(valid?"true":"false")
          <<",\"conversion_input_equals_graph_input\":"<<(rec.records[0].conversion.x==tensors.input && tensors.input?"true":"false")
          <<",\"conversion_output_equals_main_input\":"<<(rec.records[0].conversion.vy && rec.records[0].conversion.vy==rec.records[1].main.vy?"true":"false")
          <<",\"main_weights_equals_graph_weights\":"<<(rec.records[1].main.vx==tensors.weights && tensors.weights?"true":"false")
          <<",\"main_output_equals_graph_output\":"<<(rec.records[1].main.dst==tensors.output && tensors.output?"true":"false")<<"}}\n";
    const std::string text=result.str();DWORD wrote=0;WriteFile(output,text.data(),DWORD(text.size()),&wrote,nullptr);CloseHandle(output);
    return wrote==text.size()?exitcode:5;
}
