// Adapted from R16 conversion_capture/launch_recorder.cpp.
// Identity recorder only: no timing, no device-pointer dereference in callbacks.
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include "launch_decode.h"
#include "launch_metadata_json.h"
#include "locked_identity.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include "fixture_lock.h"
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
        const auto ds=BCryptDestroyHash(hash);hash=nullptr;const auto cs=BCryptCloseAlgorithmProvider(algorithm,0);algorithm=nullptr;
        if(ds<0||cs<0)throw std::runtime_error("SHA cleanup failed");
        std::ostringstream s; s << std::hex << std::setfill('0'); for (auto x : digest) s << std::setw(2) << unsigned(x);
        return s.str();
    } catch (...) { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); throw; }
}
static std::string verify_modules() {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto & item : locked_identity::modules) {
        HMODULE mod = GetModuleHandleA(item.name);
        char loaded[32768]{};
        if (!mod || !GetModuleFileNameA(mod, loaded, sizeof(loaded))) throw std::runtime_error("locked module not loaded");
        if (_stricmp(loaded, item.path)) throw std::runtime_error(std::string("loaded module path differs: ") + item.name);
        const std::string digest = hash_file(loaded);
        if (digest != item.sha256) throw std::runtime_error(std::string("loaded module full SHA differs: ") + item.name);
        if (!first) out << ','; first=false;
        out << "{\"name\":" << quote(item.name) << ",\"path\":" << quote(loaded) << ",\"sha256\":" << quote(digest) << '}';
    }
    out << ']'; return out.str();
}
static void check(CUptiResult status) { if (status != CUPTI_SUCCESS) throw std::runtime_error("CUPTI failure " + std::to_string(int(status))); }
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
static std::vector<unsigned char> read_fixture(const char* name,size_t size) {
    const auto path=(std::filesystem::path(fixture_lock::directory)/name).string();
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("fixture file missing");
    std::vector<unsigned char> bytes(size);f.read(reinterpret_cast<char*>(bytes.data()),size);
    if(size_t(f.gcount())!=size||f.peek()!=EOF)throw std::runtime_error("fixture size differs");return bytes;
}
static void save_binary(const std::filesystem::path& path,const void*data,size_t size) {
    HANDLE h=CreateFileA(path.string().c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("refuse output overwrite");DWORD wrote=0;
    const bool ok=WriteFile(h,data,DWORD(size),&wrote,nullptr)&&wrote==size&&FlushFileBuffers(h);CloseHandle(h);
    if(!ok)throw std::runtime_error("binary write failed");
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
    ggml_context *ctx=nullptr; ggml_backend_t backend=nullptr; ggml_backend_buffer_t buffer=nullptr;
    TensorPointers tensors{}; bool executed=false, valid=false; int exitcode=4;
    std::string reason="not_run", loaded_before="[]", loaded_after="[]", hardware="null";
    try {
        if(!std::getenv("GGML_CUDA_DISABLE_GRAPHS") || std::string(std::getenv("GGML_CUDA_DISABLE_GRAPHS"))!="1")
            throw std::runtime_error("CUDA graphs must be disabled");
        loaded_before=verify_modules(); api.load();
        if(hash_file(api.actual_path)!=locked_identity::cupti_sha256) throw std::runtime_error("CUPTI full SHA differs");
        check(api.subscribe(&subscriber,callback,&rec)); subscribed=true;
        cudaDeviceProp prop{};
        if(cudaGetDeviceProperties(&prop,0)!=cudaSuccess || prop.major!=12 || prop.minor!=0 || prop.warpSize!=32 ||
           prop.multiProcessorCount!=84 || uuid_string(prop.uuid)!="GPU-83b80720-113c-3f3d-c624-f1dc642b3f8b")
            throw std::runtime_error("source-qualified GPU UUID/sm_120/84-SM/warp32 identity required");
        hardware="{\"uuid\":"+quote(uuid_string(prop.uuid))+",\"major\":"+std::to_string(prop.major)+
                 ",\"minor\":"+std::to_string(prop.minor)+",\"SM_count\":"+std::to_string(prop.multiProcessorCount)+
                 ",\"warp_size\":"+std::to_string(prop.warpSize)+"}";
        ggml_init_params params{ggml_tensor_overhead()*8+ggml_graph_overhead_custom(8,false),nullptr,true}; ctx=ggml_init(params);
        if(!ctx) throw std::runtime_error("GGML context unavailable");
        auto *w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,kK,kN); auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,kK,kM);
        auto *y=ggml_mul_mat(ctx,w,x); auto *graph=ggml_new_graph_custom(ctx,8,false); ggml_build_forward_expand(graph,y);
        backend=ggml_backend_cuda_init(0);
        if(!backend || !ggml_backend_is_cuda(backend) || !ggml_backend_supports_op(backend,y)) throw std::runtime_error("unsupported CUDA graph");
        for(const auto &r:fixture_lock::files)if(hash_file(r.path)!=r.sha)throw std::runtime_error("frozen fixture identity differs");
        auto packed=read_fixture("weights.q4_k.bin",size_t(kK/256)*144*kN);
        auto raw_input=read_fixture("input.f32.bin",size_t(kK)*4);std::vector<float> input(kK);
        std::memcpy(input.data(),raw_input.data(),raw_input.size());
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); if(!buffer) throw std::runtime_error("device buffer unavailable");
        // Synthetic constant graph. ANY and WEIGHTS both bypass the source0 COMPUTE padding memset.
        if(ggml_backend_buffer_get_usage(buffer)==GGML_BACKEND_BUFFER_USAGE_COMPUTE) throw std::runtime_error("unexpected compute-buffer usage");
        ggml_backend_tensor_set(w,packed.data(),0,packed.size()); ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        ggml_backend_synchronize(backend); tensors={uintptr_t(w->data),uintptr_t(x->data),uintptr_t(y->data)};
        // One untimed allocation/dispatch warmup; captured replay below is separate.
        if(ggml_backend_graph_compute_async(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("warmup graph failed");
        ggml_backend_synchronize(backend);rec.reset();
        check(api.enable_domain(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API));enabled=true;
        executed=true; const auto status=ggml_backend_graph_compute_async(backend,graph);ggml_backend_synchronize(backend);
        check(api.enable_domain(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API));enabled=false;
        if(status!=GGML_STATUS_SUCCESS) throw std::runtime_error("GGML graph compute failed");
        loaded_after=verify_modules();
        if(hash_file(api.actual_path)!=locked_identity::cupti_sha256) throw std::runtime_error("CUPTI changed during capture");
        valid=validate(rec,tensors,reason);
        if(valid)for(unsigned i=0;i<2;++i)if(rec.records[i].api_id!=430||!rec.records[i].extended_api){valid=false;reason="target requires observed ExC430";}
        if(valid){
            // Device reads happen only after callback capture and sync, never inside callbacks.
            const auto dir=std::filesystem::path(argv[2]).parent_path();
            std::vector<unsigned char> actual_q8(size_t(kK/32)*36);std::vector<float> actual(kN);
            if(cudaMemcpy(actual_q8.data(),reinterpret_cast<void*>(rec.records[0].conversion.vy),actual_q8.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("Q8 readback failed");
            ggml_backend_tensor_get(y,actual.data(),0,actual.size()*sizeof(float));
            save_binary(dir/"actual.q8_1.bin",actual_q8.data(),actual_q8.size());
            save_binary(dir/"actual.f32.bin",actual.data(),actual.size()*sizeof(float));
            for(const auto &r:fixture_lock::files)if(hash_file(r.path)!=r.sha)throw std::runtime_error("fixture changed after execution");
        }
        loaded_after=verify_modules();
        exitcode=valid?0:4;
    } catch(const std::exception & e) {reason=e.what();valid=false;exitcode=4;}
    if(enabled && api.enable_domain(0,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API)!=CUPTI_SUCCESS){valid=false;exitcode=4;reason="disable capture failed";}
    if(subscribed && api.unsubscribe(subscriber)!=CUPTI_SUCCESS){valid=false;exitcode=4;reason="unsubscribe failed";}
    if(buffer) ggml_backend_buffer_free(buffer); if(backend) ggml_backend_free(backend); if(ctx) ggml_free(ctx);
    std::ostringstream result;
    result<<"{\"schema\":\"r29-q4k-target-launch-capture/v1\",\"status\":"<<quote(valid?"path_observed_pending_independent_numeric":"rejected")
          <<",\"reason\":"<<quote(reason)<<",\"timed\":false,\"performance_parameter_admitted\":false,\"target_LLM_latency_used\":false"
          <<",\"gpu_graph_executed\":"<<(executed?"true":"false")<<",\"device_pointer_dereferenced_in_callback\":false"
          <<",\"protocol_sha256\":"<<quote(fixture_lock::protocol_sha)
          <<",\"compile_CUPTI_API_version\":"<<CUPTI_API_VERSION<<",\"runtime_CUPTI_API_version\":"<<api.runtime_api_version
          <<",\"CUPTI_path\":"<<quote(api.actual_path)<<",\"loaded_modules_before\":"<<loaded_before<<",\"loaded_modules_after\":"<<loaded_after
          <<",\"hardware\":"<<hardware<<",\"configuration\":{\"format\":\"Q4_K\",\"type_id\":12,\"M\":1,\"K\":2048,\"N\":2048,\"fixture\":\"packed_dyadic_v1\",\"ids\":false,\"fusion\":false,\"graph_replay\":false}"
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
