// Independent command-supply/consumer queue probe with per-call host timing.
#define main launch_gap_reference_main
#include "cuda_graph_launch_gap_microbench.cu"
#undef main
#include <sstream>

template<int Bytes> struct Payload { unsigned int words[Bytes / sizeof(unsigned int)]; };
template<int Bytes>
__global__ void payload_body(U64 *times, unsigned int *sink, int node, int blocks,
                             U64 body_ns, Payload<Bytes> payload) {
    const int index = node * blocks + blockIdx.x;
    if (threadIdx.x == 0) {
        const auto begin = timer_ns(); times[2 * index] = begin;
        auto now = begin; while (now - begin < body_ns) now = timer_ns();
        sink[index] += static_cast<unsigned int>(now - begin + 1) + payload.words[0] + payload.words[Bytes / 4 - 1];
        times[2 * index + 1] = timer_ns();
    }
    __syncthreads();
}
template<int Bytes>
static void launch_payload(cudaStream_t stream, U64 *times, unsigned int *sink, int node,
                           int blocks, int threads, U64 body) {
    Payload<Bytes> payload{}; payload.words[0] = payload.words[Bytes / 4 - 1] = 1;
    payload_body<Bytes><<<blocks, threads, 0, stream>>>(times, sink, node, blocks, body, payload);
}
static U64 host_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
static std::vector<U64> parse_bodies(const std::string &value, int nodes) {
    std::stringstream parser(value); std::string field; std::vector<U64> bodies;
    while (std::getline(parser, field, ',')) bodies.push_back(std::stoull(field));
    if (bodies.size() == 1) bodies.resize(nodes, bodies[0]);
    if (bodies.size() != static_cast<size_t>(nodes)) throw std::runtime_error("one uniform body or every node body is required");
    for (auto body : bodies) if (body > 100000) throw std::runtime_error("body exceeds independent probe domain");
    return bodies;
}
struct QueueSample {
    double event_ns{};
    U64 host_release{};
    std::vector<U64> begin, end, host_begin, host_end;
};
static void submit_nodes(const Config &cfg, const std::vector<U64> &bodies, int payload,
                         cudaStream_t stream, U64 *times, unsigned int *sink,
                         QueueSample *sample, bool observe) {
    for (int index = 0; index < cfg.nodes; ++index) {
        U64 begin{}; if (observe) begin = host_ns();
        switch (payload) {
        case 0: timed_body<false><<<cfg.blocks, cfg.threads, 0, stream>>>(times, sink, index, cfg.blocks, bodies[index]); break;
        case 64: launch_payload<64>(stream, times, sink, index, cfg.blocks, cfg.threads, bodies[index]); break;
        case 256: launch_payload<256>(stream, times, sink, index, cfg.blocks, cfg.threads, bodies[index]); break;
        case 1024: launch_payload<1024>(stream, times, sink, index, cfg.blocks, cfg.threads, bodies[index]); break;
        default: throw std::runtime_error("unsupported payload footprint");
        }
        U64 end{}; if (observe) end = host_ns();
        CHECK(cudaPeekAtLastError());
        if (observe) { sample->host_begin.push_back(begin); sample->host_end.push_back(end); }
    }
}
int main(int argc, char **argv) {
    try {
        Config cfg; int payload = 0; bool observe = true; std::string body_arg = "0";
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i]; if (i + 1 >= argc) throw std::runtime_error("argument value missing");
            std::string value = argv[++i];
            if (arg == "--nodes") cfg.nodes = std::stoi(value);
            else if (arg == "--blocks") cfg.blocks = std::stoi(value);
            else if (arg == "--threads") cfg.threads = std::stoi(value);
            else if (arg == "--repetitions") cfg.repetitions = std::stoi(value);
            else if (arg == "--payload-bytes") payload = std::stoi(value);
            else if (arg == "--bodies-ns") body_arg = value;
            else if (arg == "--observe-host" && (value == "0" || value == "1")) observe = value == "1";
            else throw std::runtime_error("unsupported queue probe option");
        }
        if (cfg.nodes < 2 || cfg.nodes > 4096 || cfg.repetitions < 3 || cfg.repetitions > 999 || cfg.blocks < 1 || cfg.blocks > 1024 || cfg.threads < 1 || cfg.threads > 1024)
            throw std::runtime_error("queue probe configuration outside bounded domain");
        auto bodies = parse_bodies(body_arg, cfg.nodes);
        int device{}, driver{}, runtime{}; cudaDeviceProp prop{};
        CHECK(cudaGetDevice(&device)); CHECK(cudaGetDeviceProperties(&prop, device));
        CHECK(cudaDriverGetVersion(&driver)); CHECK(cudaRuntimeGetVersion(&runtime));
        const void *function{};
        switch (payload) {
        case 0: function = reinterpret_cast<const void *>(timed_body<false>); break;
        case 64: function = reinterpret_cast<const void *>(payload_body<64>); break;
        case 256: function = reinterpret_cast<const void *>(payload_body<256>); break;
        case 1024: function = reinterpret_cast<const void *>(payload_body<1024>); break;
        default: throw std::runtime_error("unsupported parameter payload");
        }
        size_t offset{}, last_size{};
        CHECK(cudaFuncGetParamInfo(function, payload ? 5 : 4, &offset, &last_size));
        const size_t parameter_extent = offset + last_size;
        cudaStream_t stream{}; cudaEvent_t ev_begin{}, ev_end{}; U64 *times{};
        unsigned int *sink{}, *flags{}, *device_flags{};
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cudaEventCreate(&ev_begin)); CHECK(cudaEventCreate(&ev_end));
        CHECK(cudaMalloc(&times, sizeof(U64) * cfg.nodes * cfg.blocks * 2));
        CHECK(cudaMalloc(&sink, sizeof(unsigned int) * cfg.nodes * cfg.blocks));
        CHECK(cudaMemset(sink, 0, sizeof(unsigned int) * cfg.nodes * cfg.blocks));
        CHECK(cudaHostAlloc(&flags, sizeof(unsigned int) * 96, cudaHostAllocMapped | cudaHostAllocWriteCombined));
        CHECK(cudaHostGetDevicePointer(&device_flags, flags, 0));
        std::vector<QueueSample> samples;
        for (int repetition = -2; repetition < cfg.repetitions; ++repetition) {
            CHECK(cudaStreamSynchronize(stream));
            flags[0] = flags[32] = flags[64] = 0; std::atomic_thread_fence(std::memory_order_seq_cst);
            mapped_gpu_gate<<<1, 1, 0, stream>>>(device_flags); CHECK(cudaPeekAtLastError());
            auto status = cudaStreamQuery(stream); if (status != cudaSuccess && status != cudaErrorNotReady) CHECK(status);
            auto deadline = Clock::now() + std::chrono::milliseconds(400);
            while (!gpu_gate_ready(flags) && Clock::now() < deadline) std::this_thread::yield();
            if (!gpu_gate_ready(flags)) { release_gpu_gate(flags); cudaStreamSynchronize(stream); throw std::runtime_error("queue GPU gate not resident"); }
            QueueSample sample;
            try {
                CHECK(cudaEventRecord(ev_begin, stream));
                submit_nodes(cfg, bodies, payload, stream, times, sink, &sample, observe);
                CHECK(cudaEventRecord(ev_end, stream));
                status = cudaEventQuery(ev_end); if (status != cudaSuccess && status != cudaErrorNotReady) CHECK(status);
            } catch (...) { release_gpu_gate(flags); cudaStreamSynchronize(stream); throw; }
            sample.host_release = host_ns(); release_gpu_gate(flags); CHECK(cudaStreamSynchronize(stream));
            if (flags[32]) throw std::runtime_error("queue gate timed out before all submissions");
            float event{}; CHECK(cudaEventElapsedTime(&event, ev_begin, ev_end)); sample.event_ns = event * 1.0e6;
            std::vector<U64> raw(cfg.nodes * cfg.blocks * 2);
            CHECK(cudaMemcpy(raw.data(), times, raw.size() * sizeof(U64), cudaMemcpyDeviceToHost));
            for (int node = 0; node < cfg.nodes; ++node) {
                U64 begin = ~U64(0), end = 0;
                for (int block = 0; block < cfg.blocks; ++block) {
                    int i = 2 * (node * cfg.blocks + block);
                    if (!raw[i] || raw[i + 1] < raw[i]) throw std::runtime_error("invalid queue GPU body timestamp");
                    begin = std::min(begin, raw[i]); end = std::max(end, raw[i + 1]);
                }
                sample.begin.push_back(begin); sample.end.push_back(end);
            }
            const U64 origin = sample.begin[0];
            for (auto &x : sample.begin) x -= origin;
            for (auto &x : sample.end) x -= origin;
            if (observe) {
                const U64 origin_host = sample.host_begin[0];
                for (auto &x : sample.host_begin) x -= origin_host;
                for (auto &x : sample.host_end) x -= origin_host;
                sample.host_release -= origin_host;
            }
            if (repetition >= 0) samples.push_back(std::move(sample));
        }
        std::cout << "{\"schema\":\"heterollm.cuda-dispatch-queue/v1\",\"target_llm_latency_used\":false,\"hardware_id\":\"" << prop.name
                  << "\",\"architecture\":\"sm_" << prop.major << prop.minor << "\",\"driver_version\":\"" << driver
                  << "\",\"runtime_version\":\"" << runtime << "\",\"node_count\":" << cfg.nodes << ",\"blocks\":" << cfg.blocks
                  << ",\"threads\":" << cfg.threads << ",\"extra_parameter_bytes\":" << payload << ",\"host_observation\":" << (observe ? "true" : "false")
                  << ",\"cuda_parameter_extent_bytes\":" << parameter_extent << ",\"requested_bodies_ns\":"; array(bodies); std::cout << ",\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i) std::cout << ','; const auto &s = samples[i];
            std::cout << "{\"device_event_ns\":" << s.event_ns << ",\"node_begin_ns\":"; array(s.begin);
            std::cout << ",\"node_end_ns\":"; array(s.end);
            std::cout << ",\"host_enqueue_begin_ns\":"; array(s.host_begin);
            std::cout << ",\"host_enqueue_end_ns\":"; array(s.host_end);
            std::cout << ",\"host_release_ns\":" << (observe ? s.host_release : 0) << '}';
        }
        std::cout << "]}\n";
        CHECK(cudaFree(times)); CHECK(cudaFree(sink)); CHECK(cudaFreeHost(flags));
        CHECK(cudaEventDestroy(ev_begin)); CHECK(cudaEventDestroy(ev_end)); CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}
