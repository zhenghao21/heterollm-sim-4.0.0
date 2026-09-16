// Development-only MMVQ wrapper support.  It implements only the CUDA-device
// state that mmvq.cu's selected main path queries.  Runtime initialization is
// intentionally lazy and derives values from CUDA, never from LLM latency data.
#include "F:/codex_project/37_LLMsim/heterollm-sim-4.0.0/source/llama.cpp-semantic/ggml/src/ggml-cuda/common.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {
std::once_flag init_once;
ggml_cuda_device_info info{};
thread_local int current_device = 0;

void init_info() {
    int count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&count);
    if (count_status != cudaSuccess || count <= 0) {
        // Link-only preparation may run without a GPU.  A later GPU execution
        // must observe this condition and stop before launching any kernel.
        info.device_count = 0;
        info.physical_device_count = 0;
        return;
    }
    info.device_count = count > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : count;
    info.physical_device_count = count;
    for (int i = 0; i < info.device_count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            continue;
        }
        auto & d = info.devices[i];
        d.cc = prop.major * 100 + prop.minor * 10;
        d.nsm = prop.multiProcessorCount;
        d.smpb = prop.sharedMemPerBlock;
        d.smpbo = prop.sharedMemPerBlockOptin;
        d.integrated = prop.integrated != 0;
        d.total_vram = prop.totalGlobalMem;
        d.warp_size = prop.warpSize;
        d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
        d.physical_device = i;
        d.physical_share_count = 1;
        d.virtual_index = 0;
        info.default_tensor_split[i] = 1.0f / float(info.device_count);
    }
}
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "MMVQ probe CUDA error: %s (%s) at %s:%d in %s\n", stmt, msg ? msg : "", file, line, func);
    std::abort();
}

const ggml_cuda_device_info & ggml_cuda_info() {
    std::call_once(init_once, init_info);
    return info;
}

void ggml_cuda_set_device(int device) {
    current_device = device;
    (void) cudaSetDevice(device);
}

int ggml_cuda_get_device() {
    int device = current_device;
    if (cudaGetDevice(&device) == cudaSuccess) {
        current_device = device;
    }
    return current_device;
}
