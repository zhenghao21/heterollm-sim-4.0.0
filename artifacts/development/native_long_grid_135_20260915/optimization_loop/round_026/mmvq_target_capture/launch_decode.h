#pragma once
// Host-only CUPTI argument decoding. No CUDA calls or device-pointer dereferences.
#include <cuda_runtime_api.h>
#include <cupti.h>
#include "ggml.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

namespace mmvq_capture {
inline constexpr int kM = 1, kK = 4096, kN = 3072;
inline constexpr const char * kConversionSymbol = "_Z13quantize_q8_1PKfPvxxxxxj5uint3";
inline constexpr const char * kMainSymbol = "_Z13mul_mat_vec_qIL9ggml_type6ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj";
// Exact host ABI mirror of common.cuh:1578..1586. build.py pins its source text,
// field declarations and headers. This is a decoder record, never a kernel ABI.
struct FusionArgs {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    const void * x_scale = nullptr;
    const void * gate_scale = nullptr;
    ggml_glu_op glu_op{};
    float glu_limit = 0.0f;
};
static_assert(sizeof(void *) == 8 && sizeof(uintptr_t) == 8 && sizeof(int64_t) == 8);
static_assert(sizeof(uint32_t) == 4 && sizeof(uint3) == 12 && alignof(uint3) == 4);
static_assert(sizeof(cudaError_t) == 4);
static_assert(sizeof(ggml_glu_op) == 4 && sizeof(float) == 4);
static_assert(std::is_standard_layout<FusionArgs>::value && std::is_trivially_copyable<FusionArgs>::value);
static_assert(sizeof(FusionArgs) == 48 && alignof(FusionArgs) == 8);
static_assert(offsetof(FusionArgs, x_bias) == 0 && offsetof(FusionArgs, gate) == 8);
static_assert(offsetof(FusionArgs, gate_bias) == 16 && offsetof(FusionArgs, x_scale) == 24);
static_assert(offsetof(FusionArgs, gate_scale) == 32 && offsetof(FusionArgs, glu_op) == 40);
static_assert(offsetof(FusionArgs, glu_limit) == 44);

struct ConversionArgs {
    uintptr_t x = 0, vy = 0;
    int64_t ne00 = 0, s01 = 0, s02 = 0, s03 = 0, ne0 = 0;
    uint32_t ne1 = 0;
    uint3 ne2{};
};
struct MainArgs {
    uintptr_t vx = 0, vy = 0, ids = 0;
    FusionArgs fusion{};
    uintptr_t dst = 0;
    uint32_t ncols_x = 0;
    uint3 nchannels_y{};
    uint32_t stride_row_x = 0, stride_col_y = 0, stride_col_dst = 0;
    uint3 channel_ratio{};
    uint32_t stride_channel_x = 0, stride_channel_y = 0, stride_channel_dst = 0;
    uint3 sample_ratio{};
    uint32_t stride_sample_x = 0, stride_sample_y = 0, stride_sample_dst = 0;
    uint32_t ids_stride = 0;
};
struct Launch {
    char symbol[1024]{};
    char api_name[128]{};
    uint32_t api_id = 0, correlation = 0, context = 0;
    uintptr_t function = 0, stream = 0;
    dim3 grid{}, block{};
    size_t shared = 0;
    bool supported_api = false, is_conversion = false, is_main = false;
    bool decoded = false, missing_symbol = false, truncated_symbol = false;
    bool malformed_arguments = false, exit_seen = false;
    int return_code = -1;
    ConversionArgs conversion{};
    MainArgs main{};
};
enum class RuntimeClass { MemoryCopyOrSet, Allocation, Synchronization };
struct RuntimeCall {
    char api_name[128]{};
    uint32_t api_id = 0, correlation = 0, context = 0;
    RuntimeClass classification = RuntimeClass::MemoryCopyOrSet;
    bool exit_seen = false;
    int return_code = -1;
};
struct Recorder {
    std::array<Launch, 64> records{};
    std::array<RuntimeCall, 64> runtime_calls{};
    std::atomic<unsigned> count{0}, runtime_count{0}, memory_api_count{0};
    std::atomic<bool> overflow{false}, malformed_callback{false};
    void reset() {
        for (auto & value : records) value = Launch{};
        for (auto & value : runtime_calls) value = RuntimeCall{};
        count = 0; runtime_count = 0; memory_api_count = 0; overflow = false; malformed_callback = false;
    }
};
inline bool eq(uint3 value, uint32_t x, uint32_t y, uint32_t z) {
    return value.x == x && value.y == y && value.z == z;
}
inline bool eq(dim3 value, uint32_t x, uint32_t y, uint32_t z) {
    return value.x == x && value.y == y && value.z == z;
}
template<class T> inline T argument(void ** args, unsigned index) {
    static_assert(std::is_trivially_copyable<T>::value);
    T value{};
    std::memcpy(&value, args[index], sizeof(T));
    return value;
}
inline bool has_arguments(void ** args, unsigned size) {
    if (!args) return false;
    for (unsigned i = 0; i < size; ++i) if (!args[i]) return false;
    return true;
}
inline bool decode_conversion(void ** args, ConversionArgs & a) {
    // quantize.cu:54: 9 parameters, NOT the old 11-parameter MMQ signature.
    if (!has_arguments(args, 9)) return false;
    a.x = argument<uintptr_t>(args, 0); a.vy = argument<uintptr_t>(args, 1);
    a.ne00 = argument<int64_t>(args, 2); a.s01 = argument<int64_t>(args, 3);
    a.s02 = argument<int64_t>(args, 4); a.s03 = argument<int64_t>(args, 5);
    a.ne0 = argument<int64_t>(args, 6); a.ne1 = argument<uint32_t>(args, 7);
    a.ne2 = argument<uint3>(args, 8);
    return true;
}
inline bool decode_main(void ** args, MainArgs & a) {
    // mmvq.cu:585: exactly 19 parameters, including the 48-byte by-value fusion.
    if (!has_arguments(args, 19)) return false;
    a.vx = argument<uintptr_t>(args, 0); a.vy = argument<uintptr_t>(args, 1);
    a.ids = argument<uintptr_t>(args, 2); a.fusion = argument<FusionArgs>(args, 3);
    a.dst = argument<uintptr_t>(args, 4); a.ncols_x = argument<uint32_t>(args, 5);
    a.nchannels_y = argument<uint3>(args, 6); a.stride_row_x = argument<uint32_t>(args, 7);
    a.stride_col_y = argument<uint32_t>(args, 8); a.stride_col_dst = argument<uint32_t>(args, 9);
    a.channel_ratio = argument<uint3>(args, 10); a.stride_channel_x = argument<uint32_t>(args, 11);
    a.stride_channel_y = argument<uint32_t>(args, 12); a.stride_channel_dst = argument<uint32_t>(args, 13);
    a.sample_ratio = argument<uint3>(args, 14); a.stride_sample_x = argument<uint32_t>(args, 15);
    a.stride_sample_y = argument<uint32_t>(args, 16); a.stride_sample_dst = argument<uint32_t>(args, 17);
    a.ids_stride = argument<uint32_t>(args, 18);
    return true;
}
inline bool launch_api(const char * name) {
    // Observe all runtime launch APIs, then reject unsupported forms. Enabling
    // only the legacy API could silently miss an additional graph/PDL launch.
    return name && (std::strncmp(name, "cudaLaunch", 10) == 0 ||
                    std::strncmp(name, "cudaGraphLaunch", 15) == 0);
}
inline bool runtime_class(const char * name, RuntimeClass & classification) {
    if (!name) return false;
    if (std::strncmp(name, "cudaMemcpy", 10) == 0 || std::strncmp(name, "cudaMemset", 10) == 0) {
        classification = RuntimeClass::MemoryCopyOrSet; return true;
    }
    if (std::strncmp(name, "cudaMalloc", 10) == 0 || std::strncmp(name, "cudaFree", 8) == 0) {
        classification = RuntimeClass::Allocation; return true;
    }
    if (std::strncmp(name, "cudaStreamSynchronize", 21) == 0 ||
        std::strcmp(name, "cudaDeviceSynchronize") == 0 || std::strcmp(name, "cudaEventSynchronize") == 0) {
        classification = RuntimeClass::Synchronization; return true;
    }
    return false;
}
inline void record_runtime(Recorder & recorder, CUpti_CallbackId id, const CUpti_CallbackData & info, RuntimeClass classification) {
    if (info.callbackSite == CUPTI_API_ENTER) {
        const unsigned i = recorder.runtime_count.fetch_add(1);
        if (classification == RuntimeClass::MemoryCopyOrSet) ++recorder.memory_api_count;
        if (i >= recorder.runtime_calls.size()) { recorder.overflow = true; return; }
        auto & out = recorder.runtime_calls[i]; out.classification = classification;
        out.api_id = id; out.correlation = info.correlationId; out.context = info.contextUid;
        if (std::strlen(info.functionName) >= sizeof(out.api_name)) recorder.malformed_callback = true;
        std::strncpy(out.api_name, info.functionName, sizeof(out.api_name)-1);
        // Only function name/identity/return code are needed: memory calls are
        // rejected, and their many different functionParams ABIs are never cast.
    } else if (info.callbackSite == CUPTI_API_EXIT) {
        RuntimeCall * found = nullptr;
        const unsigned n = (std::min)(recorder.runtime_count.load(), unsigned(recorder.runtime_calls.size()));
        for (unsigned i=0; i<n; ++i) {
            auto & candidate = recorder.runtime_calls[i];
            if (candidate.api_id == id && candidate.correlation == info.correlationId) {
                if (found) { recorder.malformed_callback = true; return; }
                found = &candidate;
            }
        }
        if (!found || found->exit_seen || !info.functionReturnValue) {recorder.malformed_callback=true;return;}
        std::memcpy(&found->return_code, info.functionReturnValue, sizeof(int)); found->exit_seen=true;
    } else recorder.malformed_callback=true;
}
inline void CUPTIAPI callback(void * user, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void * payload) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || !user) return;
    auto & recorder = *static_cast<Recorder *>(user);
    if (!payload) { recorder.malformed_callback = true; return; }
    const auto & info = *static_cast<const CUpti_CallbackData *>(payload);
    RuntimeClass classification{};
    if (runtime_class(info.functionName, classification)) { record_runtime(recorder,id,info,classification); return; }
    const bool supported = id == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000;
    if (!supported && !launch_api(info.functionName)) return;
    if (info.callbackSite == CUPTI_API_EXIT) {
        const unsigned n = (std::min)(recorder.count.load(), unsigned(recorder.records.size()));
        Launch * found = nullptr;
        for (unsigned i = 0; i < n; ++i) {
            auto & launch = recorder.records[i];
            if (launch.correlation == info.correlationId && launch.api_id == id) {
                if (found) { recorder.malformed_callback = true; return; }
                found = &launch;
            }
        }
        if (!found || found->exit_seen || !info.functionReturnValue) { recorder.malformed_callback = true; return; }
        found->return_code = 0;
        std::memcpy(&found->return_code, info.functionReturnValue, sizeof(int));
        found->exit_seen = true;
        return;
    }
    if (info.callbackSite != CUPTI_API_ENTER) { recorder.malformed_callback = true; return; }
    const unsigned index = recorder.count.fetch_add(1);
    if (index >= recorder.records.size()) { recorder.overflow = true; return; }
    auto & out = recorder.records[index];
    out.api_id = id; out.correlation = info.correlationId; out.context = info.contextUid;
    out.supported_api = supported;
    if (info.functionName) std::strncpy(out.api_name, info.functionName, sizeof(out.api_name) - 1);
    out.missing_symbol = !info.symbolName || !info.symbolName[0];
    if (!out.missing_symbol) {
        const size_t length = std::strlen(info.symbolName);
        out.truncated_symbol = length >= sizeof(out.symbol);
        std::strncpy(out.symbol, info.symbolName, sizeof(out.symbol) - 1);
    }
    if (!supported) return; // Unknown functionParams ABI is never reinterpreted.
    if (!info.functionParams) { recorder.malformed_callback = true; return; }
    const auto & launch = *static_cast<const cudaLaunchKernel_v7000_params *>(info.functionParams);
    out.function = reinterpret_cast<uintptr_t>(launch.func);
    out.stream = reinterpret_cast<uintptr_t>(launch.stream);
    out.grid = launch.gridDim; out.block = launch.blockDim; out.shared = launch.sharedMem;
    if (out.truncated_symbol || out.missing_symbol) return;
    out.is_conversion = std::strcmp(out.symbol, kConversionSymbol) == 0;
    out.is_main = std::strcmp(out.symbol, kMainSymbol) == 0;
    if (out.is_conversion) out.decoded = decode_conversion(launch.args, out.conversion);
    else if (out.is_main) out.decoded = decode_main(launch.args, out.main);
    // Unknown symbols have zero decoded fields and can never pass validation.
    if ((out.is_conversion || out.is_main) && !out.decoded) out.malformed_arguments = true;
}
struct TensorPointers { uintptr_t weights = 0, input = 0, output = 0; };
inline bool validate(const Recorder & recorder, const TensorPointers & tensors, std::string & reason) {
    const auto reject = [&](const char * text) { reason = text; return false; };
    if (recorder.overflow || recorder.malformed_callback) return reject("callback_overflow_or_malformed");
    if (recorder.memory_api_count != 0) return reject("unexpected_graph_memory_api");
    const unsigned runtime_count = (std::min)(recorder.runtime_count.load(), unsigned(recorder.runtime_calls.size()));
    for (unsigned i=0;i<runtime_count;++i) {
        const auto & api=recorder.runtime_calls[i];
        if (!api.exit_seen || api.return_code != 0) return reject("missing_or_failed_runtime_return");
    }
    if (recorder.count != 2) return reject("exactly_two_launches_required");
    const auto & c = recorder.records[0]; const auto & m = recorder.records[1];
    for (const auto * launch : {&c, &m}) {
        if (!launch->supported_api) return reject("unsupported_launch_api");
        if (launch->missing_symbol || launch->truncated_symbol) return reject("missing_or_truncated_symbol");
        if (!launch->decoded || launch->malformed_arguments) return reject("unknown_symbol_or_arguments");
        if (!launch->exit_seen || launch->return_code != 0) return reject("missing_or_failed_launch_return");
        if (!launch->function || !launch->correlation || !launch->context) return reject("missing_launch_identity");
    }
    if (!c.is_conversion || !m.is_main) return reject("conversion_main_order_mismatch");
    if (c.function == m.function || c.correlation == m.correlation) return reject("ambiguous_launch_identity");
    if (c.context != m.context || c.stream != m.stream) return reject("context_or_stream_mismatch");
    if (!eq(c.grid, 16, 1, 1) || !eq(c.block, 256, 1, 1) || c.shared) return reject("conversion_geometry_mismatch");
    if (!eq(m.grid, 3072, 1, 1) || !eq(m.block, 32, 4, 1) || m.shared) return reject("main_geometry_mismatch");
    const auto & a = c.conversion; const auto & b = m.main;
    if (!tensors.weights || !tensors.input || !tensors.output ||
        a.x != tensors.input || b.vx != tensors.weights || b.dst != tensors.output) return reject("graph_tensor_pointer_mismatch");
    if (!a.vy || a.vy != b.vy) return reject("conversion_to_main_pointer_mismatch");
    if (a.vy == a.x || a.vy == b.vx || a.vy == b.dst ||
        b.vx == b.dst || a.x == b.vx || a.x == b.dst) return reject("unexpected_tensor_aliasing");
    if (a.ne00 != kK || a.s01 != kK || a.s02 != kK || a.s03 != kK ||
        a.ne0 != kK || a.ne1 != kM || !eq(a.ne2, 1, 0, 1)) return reject("conversion_shape_or_stride_mismatch");
    const auto & f = b.fusion;
    if (b.ids || f.x_bias || f.gate || f.gate_bias || f.x_scale || f.gate_scale ||
        static_cast<int>(f.glu_op) != 0 || f.glu_limit != 0.0f) return reject("unsupported_ids_or_fusion");
    if (b.ncols_x != kK || !eq(b.nchannels_y, 0, 0, 0) ||
        b.stride_row_x != 128 || b.stride_col_y != 128 || b.stride_col_dst != kN ||
        !eq(b.channel_ratio, 1, 0, 1) || b.stride_channel_x != 393216 ||
        b.stride_channel_y != 128 || b.stride_channel_dst != kN ||
        !eq(b.sample_ratio, 1, 0, 1) || b.stride_sample_x != 393216 ||
        b.stride_sample_y != 128 || b.stride_sample_dst != kN || b.ids_stride != 0)
        return reject("main_shape_or_stride_mismatch");
    reason = "source_qualified_q5_0_m1_k4096_n3072_pair";
    return true;
}
} // namespace mmvq_capture
