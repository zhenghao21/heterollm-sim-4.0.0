// Independent launch-gap measurements. No model, weights, or target latency.
// This does NOT measure an additional copy of the GPU kernel body.
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using U64 = unsigned long long;
#define CHECK(call) do { auto error = (call); if (error != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error)); } while (0)
__device__ __forceinline__ U64 timer_ns() {
    U64 value; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)); return value;
}

template<bool Programmatic>
__global__ void timed_body(U64 *times, unsigned int *sink, int node, int blocks, U64 body_ns) {
    if constexpr (Programmatic) cudaGridDependencySynchronize();
    const int index = node * blocks + blockIdx.x;
    if (threadIdx.x == 0) {
        U64 begin = timer_ns();
        times[2 * index] = begin;
        U64 now = begin;
        while (now - begin < body_ns) now = timer_ns();
        sink[index] += static_cast<unsigned int>(now - begin + 1);
        // The timer probe's small exit/entry cost remains in the observed gap.
        // Keep this fact explicit instead of claiming pure scheduler latency.
        times[2 * index + 1] = timer_ns();
    }
    __syncthreads();
    if constexpr (Programmatic) cudaTriggerProgrammaticLaunchCompletion();
}

struct Gate { std::atomic<bool> released{false}, expired{false}; };
static void CUDART_CB host_gate(void *data) {
    auto &gate = *static_cast<Gate *>(data);
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (!gate.released.load(std::memory_order_acquire)) {
        if (Clock::now() >= deadline) { gate.expired.store(true); return; }
        std::this_thread::yield();
    }
}
__global__ void mapped_gpu_gate(volatile unsigned int *flags) {
    if (blockIdx.x || threadIdx.x) return;
    const auto begin = timer_ns();
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ready(*const_cast<unsigned int *>(flags + 64));
    ready.store(1, cuda::memory_order_release);
    unsigned int released = 0;
    do {
        // System-scope acquire is required here. Volatile / ld.global.cv on
        // this WDDM system did not observe the CPU release and were rejected.
        cuda::atomic_ref<unsigned int, cuda::thread_scope_system> release_flag(*const_cast<unsigned int *>(flags));
        released = release_flag.load(cuda::memory_order_acquire);
        if (timer_ns() - begin > 500000000ULL) { flags[32] = 1; __threadfence_system(); return; }
    } while (!released);
}
static void release_gpu_gate(volatile unsigned int *flags) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> release(*const_cast<unsigned int *>(flags));
    release.store(1, cuda::memory_order_release);
}
static bool gpu_gate_ready(volatile unsigned int *flags) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ready(*const_cast<unsigned int *>(flags + 64));
    return ready.load(cuda::memory_order_acquire) != 0;
}

struct Config {
    int nodes = 128, blocks = 1, threads = 32, repetitions = 31;
    U64 body_ns = 0;
    bool programmatic = false;
    bool flush_query = true;
    bool gpu_gate = true;
};
static void enqueue(const Config &cfg, cudaStream_t stream, U64 *times, unsigned int *sink) {
    for (int index = 0; index < cfg.nodes; ++index) {
        if (cfg.programmatic && index > 0) {
            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t launch{};
            launch.gridDim = dim3(cfg.blocks); launch.blockDim = dim3(cfg.threads);
            launch.stream = stream; launch.attrs = &attr; launch.numAttrs = 1;
            CHECK(cudaLaunchKernelEx(&launch, timed_body<true>, times, sink, index, cfg.blocks, cfg.body_ns));
        } else {
            timed_body<false><<<cfg.blocks, cfg.threads, 0, stream>>>(times, sink, index, cfg.blocks, cfg.body_ns);
            CHECK(cudaPeekAtLastError());
        }
    }
}

struct Sample {
    std::string mode;
    bool gated;
    double host_submit_ns;
    std::vector<U64> begin, end;
};
static Sample measure(const Config &cfg, cudaStream_t stream, cudaGraph_t graph,
                      cudaGraphExec_t steady, U64 *times, unsigned int *sink,
                      volatile unsigned int *host_flags, volatile unsigned int *device_flags,
                      const std::string &mode, bool gated) {
    cudaGraphExec_t first{};
    if (mode == "first_launch" || mode == "uploaded_first_launch") {
        CHECK(cudaGraphInstantiate(&first, graph, nullptr, nullptr, 0));
        if (mode == "uploaded_first_launch") {
            CHECK(cudaGraphUpload(first, stream)); CHECK(cudaStreamSynchronize(stream));
        }
    }
    cudaEvent_t tail{};
    if (gated && cfg.flush_query) CHECK(cudaEventCreateWithFlags(&tail, cudaEventDisableTiming));
    Gate gate;
    CHECK(cudaMemsetAsync(times, 0, sizeof(U64) * cfg.nodes * cfg.blocks * 2, stream));
    CHECK(cudaStreamSynchronize(stream));
    host_flags[0] = host_flags[32] = host_flags[64] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const auto before_gate = Clock::now();
    if (gated) {
        if (cfg.gpu_gate) {
            mapped_gpu_gate<<<1, 1, 0, stream>>>(device_flags); CHECK(cudaPeekAtLastError());
            // Query pushes the gate itself to the driver. Seeing its ready
            // flag establishes that the gate is executing on the GPU before
            // the measured chain is enqueued, unlike a host callback gate.
            const auto status = cudaStreamQuery(stream);
            if (status != cudaSuccess && status != cudaErrorNotReady) CHECK(status);
            const auto deadline = Clock::now() + std::chrono::milliseconds(400);
            while (!gpu_gate_ready(host_flags) && Clock::now() < deadline) std::this_thread::yield();
            if (!gpu_gate_ready(host_flags)) {
                release_gpu_gate(host_flags);
                CHECK(cudaStreamSynchronize(stream));
                throw std::runtime_error("GPU gate did not become device-resident before chain submission");
            }
        } else CHECK(cudaLaunchHostFunc(stream, host_gate, &gate));
    }
    const auto start = Clock::now();
    try {
        if (mode == "ordinary") enqueue(cfg, stream, times, sink);
        else CHECK(cudaGraphLaunch(first ? first : steady, stream));
    } catch (...) {
        gate.released.store(true, std::memory_order_release);
        release_gpu_gate(host_flags);
        cudaStreamSynchronize(stream);
        if (tail) cudaEventDestroy(tail);
        if (first) cudaGraphExecDestroy(first);
        throw;
    }
    const auto finish = Clock::now();
    // A returned CUDA launch API need not mean WDDM driver command buffers
    // reached the device. The query is nonblocking and explicitly requests
    // tail-event progress before releasing the host gate. It is reported as
    // an experimental queue-flush condition, never assumed to prove residency.
    try {
        if (tail) {
            CHECK(cudaEventRecord(tail, stream));
            const auto status = cudaEventQuery(tail);
            if (status != cudaSuccess && status != cudaErrorNotReady) CHECK(status);
        }
    } catch (...) {
        gate.released.store(true, std::memory_order_release);
        release_gpu_gate(host_flags);
        cudaStreamSynchronize(stream);
        if (tail) cudaEventDestroy(tail);
        if (first) cudaGraphExecDestroy(first);
        throw;
    }
    gate.released.store(true, std::memory_order_release);
    release_gpu_gate(host_flags);
    CHECK(cudaStreamSynchronize(stream));
    if (tail) CHECK(cudaEventDestroy(tail));
    if (first) CHECK(cudaGraphExecDestroy(first));
    if (gate.expired.load() || host_flags[32]) throw std::runtime_error(
        "submission gate expired: mode=" + mode + " submit_ns=" +
        std::to_string(std::chrono::duration<double, std::nano>(finish - start).count()) +
        " gate_ready_wait_ns=" + std::to_string(std::chrono::duration<double, std::nano>(start - before_gate).count()) +
        " release_to_complete_ns=" + std::to_string(std::chrono::duration<double, std::nano>(Clock::now() - finish).count()) +
        "; queue was not fully preloaded; sample is invalid");
    std::vector<U64> raw(cfg.nodes * cfg.blocks * 2);
    CHECK(cudaMemcpy(raw.data(), times, raw.size() * sizeof(U64), cudaMemcpyDeviceToHost));
    Sample sample{mode, gated, std::chrono::duration<double, std::nano>(finish - start).count(), {}, {}};
    for (int node = 0; node < cfg.nodes; ++node) {
        U64 begin = ~U64(0), end = 0;
        for (int block = 0; block < cfg.blocks; ++block) {
            const auto offset = 2 * (node * cfg.blocks + block);
            if (!raw[offset] || raw[offset + 1] < raw[offset]) throw std::runtime_error("invalid GPU globaltimer probe");
            begin = std::min(begin, raw[offset]); end = std::max(end, raw[offset + 1]);
        }
        sample.begin.push_back(begin); sample.end.push_back(end);
    }
    const auto origin = sample.begin[0];
    for (auto &value : sample.begin) value -= origin;
    for (auto &value : sample.end) value -= origin;
    return sample;
}
static void array(const std::vector<U64> &values) {
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
int main(int argc, char **argv) {
    try {
        Config cfg;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("option value missing");
            std::string value = argv[++i];
            if (arg == "--nodes") cfg.nodes = std::stoi(value);
            else if (arg == "--blocks") cfg.blocks = std::stoi(value);
            else if (arg == "--threads") cfg.threads = std::stoi(value);
            else if (arg == "--body-ns") cfg.body_ns = std::stoull(value);
            else if (arg == "--repetitions") cfg.repetitions = std::stoi(value);
            else if (arg == "--flush-query" && (value == "0" || value == "1")) cfg.flush_query = value == "1";
            else if (arg == "--gate" && (value == "host" || value == "gpu")) cfg.gpu_gate = value == "gpu";
            else if (arg == "--edge" && (value == "default" || value == "programmatic")) cfg.programmatic = value == "programmatic";
            else throw std::runtime_error("unknown option or edge value: " + arg);
        }
        if (cfg.nodes < 2 || cfg.nodes > 4096 || cfg.blocks < 1 || cfg.blocks > 1024 ||
            cfg.threads < 1 || cfg.threads > 1024 || cfg.body_ns > 100000 || cfg.repetitions < 3)
            throw std::runtime_error("configuration is outside bounded independent probe domain");
        int device{}; cudaDeviceProp prop{}; int driver{}, runtime{};
        CHECK(cudaGetDevice(&device)); CHECK(cudaGetDeviceProperties(&prop, device));
        CHECK(cudaDriverGetVersion(&driver)); CHECK(cudaRuntimeGetVersion(&runtime));
        cudaStream_t stream{}; cudaGraph_t graph{}; cudaGraphExec_t steady{};
        U64 *times{}; unsigned int *sink{};
        unsigned int *host_flags{}, *device_flags{};
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cudaMalloc(&times, sizeof(U64) * cfg.nodes * cfg.blocks * 2));
        CHECK(cudaMalloc(&sink, sizeof(unsigned int) * cfg.nodes * cfg.blocks));
        CHECK(cudaHostAlloc(&host_flags, sizeof(unsigned int) * 96, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        CHECK(cudaHostGetDevicePointer(&device_flags, host_flags, 0));
        CHECK(cudaMemset(sink, 0, sizeof(unsigned int) * cfg.nodes * cfg.blocks));
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        enqueue(cfg, stream, times, sink); CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&steady, graph, nullptr, nullptr, 0));
        CHECK(cudaGraphLaunch(steady, stream)); CHECK(cudaStreamSynchronize(stream));
        std::mt19937 random(1729);
        std::vector<Sample> samples;
        std::vector<std::pair<std::string, bool>> order;
        for (const auto &mode : {"ordinary", "first_launch", "uploaded_first_launch", "replay"})
            for (bool gated : {false, true}) order.emplace_back(mode, gated);
        for (int repetition = -2; repetition < cfg.repetitions; ++repetition) {
            std::shuffle(order.begin(), order.end(), random);
            for (const auto &entry : order) {
                auto result = measure(cfg, stream, graph, steady, times, sink, host_flags, device_flags, entry.first, entry.second);
                if (repetition >= 0) samples.push_back(std::move(result));
            }
        }
        std::cout << "{\"schema\":\"heterollm.cuda-launch-gap/v1\",\"target_llm_latency_used\":false,"
                  << "\"hardware_id\":\"" << prop.name << "\",\"architecture\":\"sm_" << prop.major << prop.minor
                  << "\",\"driver_version\":\"" << driver << "\",\"runtime_version\":\"" << runtime
                  << "\",\"node_count\":" << cfg.nodes << ",\"blocks\":" << cfg.blocks << ",\"threads\":" << cfg.threads
                  << ",\"requested_body_ns\":" << cfg.body_ns << ",\"edge\":\"" << (cfg.programmatic ? "programmatic" : "default")
                  << "\",\"gate_flush\":\"" << (cfg.flush_query ? "tail_event_query" : "none")
                  << "\",\"gate_kind\":\"" << (cfg.gpu_gate ? "device_resident_mapped_flag" : "host_callback") << "\",\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto &sample = samples[i]; if (i) std::cout << ',';
            std::cout << "{\"mode\":\"" << sample.mode << "\",\"gated\":" << (sample.gated ? "true" : "false")
                      << ",\"host_submit_ns\":" << sample.host_submit_ns << ",\"node_begin_ns\":";
            array(sample.begin); std::cout << ",\"node_end_ns\":"; array(sample.end); std::cout << '}';
        }
        std::cout << "]}\n";
        CHECK(cudaGraphExecDestroy(steady)); CHECK(cudaGraphDestroy(graph)); CHECK(cudaFree(times));
        CHECK(cudaFree(sink)); CHECK(cudaStreamDestroy(stream));
        CHECK(cudaFreeHost(host_flags));
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}
